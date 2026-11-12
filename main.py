#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10 for make.fun (roblox-up-and-coming-forecast-01).

Reads the chart history at $MAKEFUN_INPUT_DIR (else /input) and writes one
forecast to $MAKEFUN_OUTPUT_DIR (else /output): for each of the top 10 places a
probability distribution over games, plus an ordered top 10. The single
forecast is scored by log score against every hourly chart read in the 72
hours after the history ends, so this program estimates the *time-averaged*
place distribution over that window.

Model, fitted from the history itself:
  * Latent strength per game: a recency-weighted mean of exp(-(rank-1)/12)
    over every read (absent from the chart counts as 0), half-life 22 h.
  * Momentum: a shrunk, clamped per-hour slope of that value over time.
  * Ageing out: when the history's details carry a game's `created` date, the
    game leaves the chart 28 days after publication; it is excluded from the
    simulated hours past that point.
  * Monte-Carlo: for each forecast hour h = 1..72, strengths drift, Gaussian
    noise with standard deviation growing like sqrt(h) is added (scale set
    from the churn measured in the history), games are ranked, and the place
    counts are averaged over hours and draws. A per-place share is left
    unnamed on purpose: it is the probability that the place is held by a
    game outside the modelled field (new publications enter this chart
    constantly), and the scoring spreads it over every unnamed chart game.

Deterministic: a fixed seed, no clock, no network, standard library only.
"""

import json
import math
import os
import random
import sys
from datetime import datetime, timezone

SEED = 20261113
RANK_SCALE = 12.0          # exp(-(rank-1)/RANK_SCALE): rank 1 -> 1, rank 10 -> 0.47
HALF_LIFE_H = 22.0         # recency weight half-life for strength
DRIFT_CAP = 0.008          # max |strength drift| per hour
DRIFT_SHRINK = 6.0         # slope shrinkage: n/(n+DRIFT_SHRINK)
SIGMA_24 = 0.27            # ranking noise (strength units) at a 24 h horizon
SIGMA_MIN_FRAC = 0.35      # noise floor as a fraction of the 72 h noise
DRAWS = 160                # Monte-Carlo rankings drawn per forecast hour
MIN_ROW_P = 0.003          # smallest probability worth naming a row for
MAX_ROWS_PER_PLACE = 100
AGE_LIMIT_DAYS = 28.0
# Share of each place deliberately left unnamed (the field outside the model).
# Backtests on the practice history put the optimum near a flat quarter: the
# winner of a place is often a game the model rates low, and the scoring
# spreads this share over every unnamed game on the read's chart.
UNNAMED = {k: 0.25 for k in range(1, 11)}


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def rank_value(rank):
    return math.exp(-(rank - 1) / RANK_SCALE)


def load_history(input_dir):
    reads = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chart = rec.get("up_and_coming") or []
            rows = []
            for g in chart:
                if g.get("universe_id") is None or g.get("rank") is None:
                    continue
                rows.append((int(g["rank"]), int(g["universe_id"])))
            if rows:
                rows.sort()
                reads.append({"t": parse_time(rec["at"]), "rows": rows,
                              "details": rec.get("details")})
    reads.sort(key=lambda r: r["t"])
    return reads


def created_epochs(reads):
    """Latest known publication time per game, from any read's details."""
    created = {}
    for rec in reads:
        det = rec["details"]
        if not det:
            continue
        items = det.items() if isinstance(det, dict) else (
            (d.get("universe_id") or d.get("id"), d) for d in det if isinstance(d, dict))
        for key, d in items:
            if not isinstance(d, dict) or not d.get("created"):
                continue
            try:
                uid = int(d.get("universe_id") or key)
            except (TypeError, ValueError):
                continue
            try:
                created[uid] = parse_time(d["created"])
            except (TypeError, ValueError):
                pass
    return created


def churn_rate(reads):
    """Mean share of top-10 places changing occupant per sqrt(hour of gap)."""
    num = 0.0
    den = 0.0
    for a, b in zip(reads, reads[1:]):
        gap_h = max((b["t"] - a["t"]) / 3600.0, 0.25)
        top_a = [u for _, u in a["rows"][:10]]
        top_b = [u for _, u in b["rows"][:10]]
        changed = sum(1 for x, y in zip(top_a, top_b) if x != y) / 10.0
        num += changed
        den += math.sqrt(gap_h)
    if den == 0:
        return 0.24
    return num / den


def strengths(reads, t_end):
    """Per-game strength and drift from the reads, as of t_end."""
    series = {}   # uid -> list of (t, value)
    for rec in reads:
        seen = {}
        for rank, uid in rec["rows"]:
            seen[uid] = rank_value(rank)
        for uid, val in seen.items():
            series.setdefault(uid, []).append((rec["t"], val))
        for uid in series:
            if uid not in seen:
                series[uid].append((rec["t"], 0.0))
    theta, drift = {}, {}
    for uid, pts in series.items():
        pts.sort()
        wsum = 0.0
        vsum = 0.0
        tw = 0.0
        vw = 0.0
        tww = 0.0
        for t, v in pts:
            age_h = max((t_end - t) / 3600.0, 0.0)
            w = 0.5 ** (age_h / HALF_LIFE_H)
            wsum += w
            vsum += w * v
            tw += w * t
            vw += w * v * (t - t_end) / 3600.0
            tww += w * ((t - t_end) / 3600.0) ** 2
        if wsum == 0:
            continue
        theta[uid] = vsum / wsum
        # Weighted slope of value per hour, from the same moments:
        # slope = cov(t,v)/var(t) using weighted means.
        mean_t = tw / wsum
        mean_v = vsum / wsum
        cov = 0.0
        var = 0.0
        for t, v in pts:
            age_h = max((t_end - t) / 3600.0, 0.0)
            w = 0.5 ** (age_h / HALF_LIFE_H)
            dt = (t - mean_t) / 3600.0
            cov += w * dt * (v - mean_v)
            var += w * dt * dt
        slope = cov / var if var > 1.0 else 0.0
        n = len(pts)
        slope *= n / (n + DRIFT_SHRINK)
        drift[uid] = max(-DRIFT_CAP, min(DRIFT_CAP, slope))
    return theta, drift


def build_forecast(reads, task):
    places = int(task.get("places", 10))
    window = task.get("window") or {}
    hours = int(window.get("slots", 72))
    t_end = reads[-1]["t"]
    if task.get("history_until"):
        t_end = max(t_end, parse_time(task["history_until"]))
    theta, drift = strengths(reads, t_end)
    created = created_epochs(reads)

    # Candidate field: strongest 170 games -- anyone with a realistic path
    # into the top 10 during the window, risers from deep included.
    field = sorted(theta, key=lambda u: -theta[u])[:170]
    base = [theta[u] for u in field]
    dr = [drift[u] for u in field]
    # Hour (from t_end) at which each game ages out, or None.
    age_out = []
    for u in field:
        c = created.get(u)
        if c is None:
            age_out.append(None)
        else:
            age_out.append((c + AGE_LIMIT_DAYS * 86400.0 - t_end) / 3600.0)

    churn = churn_rate(reads)
    # Archive-era churn is ~0.24 per sqrt(hour); scale the noise with what
    # this history actually shows, clamped so odd inputs stay sane.
    sigma_scale = max(0.7, min(1.5, churn / 0.24))
    rng = random.Random(SEED)
    counts = [[0.0] * places for _ in field]
    total_draws = 0
    for h in range(1, hours + 1):
        frac = h / hours
        sigma = SIGMA_24 * math.sqrt(1.0 + 2.0 * frac) * sigma_scale
        sigma = max(sigma, SIGMA_24 * SIGMA_MIN_FRAC)
        alive = [i for i in range(len(field))
                 if age_out[i] is None or age_out[i] > h]
        if not alive:
            break
        means = [base[i] + dr[i] * h for i in alive]
        for _ in range(DRAWS):
            scored = [(means[j] + rng.gauss(0.0, sigma), alive[j])
                      for j in range(len(alive))]
            scored.sort(key=lambda x: -x[0])
            for place0 in range(min(places, len(scored))):
                counts[scored[place0][1]][place0] += 1.0
        total_draws += DRAWS

    probabilities = []
    game_score = {}
    for place in range(1, places + 1):
        rows = []
        for i, uid in enumerate(field):
            p = counts[i][place - 1] / total_draws if total_draws else 0.0
            if p >= MIN_ROW_P:
                rows.append((uid, p))
        rows.sort(key=lambda x: -x[1])
        rows = rows[:MAX_ROWS_PER_PLACE]
        cap = 1.0 - UNNAMED.get(place, 0.15)
        tot = sum(p for _, p in rows)
        if tot > cap and tot > 0:
            rows = [(u, p * cap / tot) for u, p in rows]
        for uid, p in rows:
            p = round(p, 6)
            if p <= 0:
                continue
            probabilities.append({"universe_id": uid, "place": place, "p": p})
            game_score[uid] = game_score.get(uid, 0.0) + p * (places + 1 - place)
    # Guarantee each place's rounded sum stays <= 1: trim the largest row.
    by_place = {}
    for row in probabilities:
        by_place.setdefault(row["place"], []).append(row)
    for place, rows in by_place.items():
        s = sum(r["p"] for r in rows)
        if s > 1.0:
            big = max(rows, key=lambda r: r["p"])
            big["p"] = round(big["p"] - (s - 1.0) - 1e-6, 6)
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    top10 = [u for u, _ in sorted(game_score.items(), key=lambda x: -x[1])[:places]]
    if len(top10) < places:  # degenerate history: fall back to the last chart
        for _, uid in reads[-1]["rows"]:
            if uid not in top10:
                top10.append(uid)
            if len(top10) == places:
                break
    return {"probabilities": probabilities, "top10": top10[:places]}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as fh:
        task = json.load(fh)
    reads = load_history(input_dir)
    if not reads:
        print("no usable reads in history", file=sys.stderr)
        return 1
    forecast = build_forecast(reads, task)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(forecast) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
