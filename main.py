#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10 (roblox-up-and-coming-forecast-01).

Reads the chart history at $MAKEFUN_INPUT_DIR (else /input) and writes one
forecast to $MAKEFUN_OUTPUT_DIR (else /output): for each of the top 10
places a probability distribution over games, plus an ordered top 10. The
single forecast is scored by log score against every hourly chart read in
the 72 hours after the history ends, so the quantity that pays is the
*time-averaged* probability that each game holds each place over the
window -- a marginal distribution, not a trajectory.

Model: the empirical place-occupancy table of the history itself.
  * Each read contributes its top 10 to a per-(game, place) count, weighted
    by recency with a 6-hour half-life, so the table describes the chart's
    current regime rather than its whole past.
  * Counts are shrunk toward the game's overall top-10 presence share
    (strength 10): a place a game held rarely still gets the probability
    its general presence earns, which keeps the table honest when the
    recency-weighted sample is a handful of reads.
  * A mild temperature (0.95) is applied and each place is renormalised,
    then 15% of every place is left unnamed: the share for games outside
    the history's top 10 (new publications enter this chart constantly),
    which the scoring spreads over the read's unnamed chart games.
  * The top 10 list is a greedy assignment on the marginal matrix: the
    largest remaining (game, place) probability is fixed first, which is
    the order that maximises expected agreement.

Deterministic: no randomness, no clock, no network, standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime

HALF_LIFE_H = 6.0        # recency half-life of a read's weight, in hours
SHRINKAGE = 10.0         # pseudo-count strength of the presence prior
TEMPERATURE = 0.95       # p -> p ** (1 / T), renormalised per place
UNNAMED = 0.15           # share of each place left for games outside the table
MIN_ROW_P = 0.001        # smallest probability worth naming a row for
MAX_ROWS_PER_PLACE = 100


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def load_history(input_dir):
    reads = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chart = rec.get("up_and_coming") or []
            # The list is the chart in its order, so a game's place is its
            # position; dedupe defensively (a game listed twice in one read
            # would otherwise be counted at two places).
            rows = []
            seen = set()
            for g in chart:
                if g.get("universe_id") is None:
                    continue
                uid = int(g["universe_id"])
                if uid in seen:
                    continue
                seen.add(uid)
                rows.append((len(rows) + 1, uid))
            if rows:
                reads.append({"t": parse_time(rec["at"]), "rows": rows})
    reads.sort(key=lambda r: r["t"])
    return reads


def build_forecast(reads, task):
    places = int(task.get("places", 10))
    t_end = reads[-1]["t"]

    counts = {}     # (universe_id, place) -> recency-weighted reads there
    presence = {}   # universe_id -> recency-weighted reads in any top place
    tot_place = [0.0] * (places + 1)
    tot_presence = 0.0
    for rec in reads:
        age_h = max((t_end - rec["t"]) / 3600.0, 0.0)
        w = 0.5 ** (age_h / HALF_LIFE_H)
        for rank, uid in rec["rows"][:places]:
            counts[(uid, rank)] = counts.get((uid, rank), 0.0) + w
            presence[uid] = presence.get(uid, 0.0) + w
            tot_place[rank] += w
            tot_presence += w
    if tot_presence == 0:
        raise ValueError("history holds no top-10 rows")

    # Prior for every place: the game's share of all top-10 slots.
    prior = {u: c / tot_presence for u, c in presence.items()}

    dist = {}   # place -> {universe_id: probability}, before the unnamed cut
    for k in range(1, places + 1):
        denom = tot_place[k] + SHRINKAGE
        d = {u: (counts.get((u, k), 0.0) + SHRINKAGE * prior[u]) / denom
             for u in presence}
        if TEMPERATURE != 1.0:
            d = {u: p ** (1.0 / TEMPERATURE) for u, p in d.items()}
            s = sum(d.values())
            d = {u: p / s for u, p in d.items()}
        dist[k] = d

    probabilities = []
    game_score = {}
    for k in range(1, places + 1):
        rows = sorted(dist[k].items(), key=lambda x: -x[1])[:MAX_ROWS_PER_PLACE]
        tot = sum(p for _, p in rows)
        if tot > 0:
            rows = [(u, p * (1.0 - UNNAMED) / tot) for u, p in rows]
        for uid, p in rows:
            p = round(p, 6)
            if p < MIN_ROW_P or p <= 0:
                continue
            probabilities.append({"universe_id": uid, "place": k, "p": p})
            game_score[uid] = game_score.get(uid, 0.0) + p * (places + 1 - k)
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

    # Top 10 by greedy assignment on the marginal matrix.
    cells = sorted(
        ((p, u, k) for k in range(1, places + 1) for u, p in dist[k].items()),
        key=lambda x: -x[0],
    )
    used_games, used_places, assign = set(), set(), {}
    for _p, uid, k in cells:
        if uid in used_games or k in used_places:
            continue
        assign[k] = uid
        used_games.add(uid)
        used_places.add(k)
        if len(assign) == places:
            break
    top10 = [assign[k] for k in range(1, places + 1) if k in assign]
    if len(top10) < places:  # degenerate table: fill by expected agreement
        for uid, _s in sorted(game_score.items(), key=lambda x: -x[1]):
            if uid not in top10:
                top10.append(uid)
            if len(top10) == places:
                break
    if len(top10) < places:  # last resort: the closing chart as it stood
        for _rank, uid in reads[-1]["rows"]:
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
