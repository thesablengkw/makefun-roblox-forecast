# Roblox Up-and-Coming top-10 forecast

Entry for make.fun's `roblox-up-and-coming-forecast-01`. `main.py` (Python
3.13, standard library only, no network) reads the chart history at
`/input` and writes `/output/forecast.json`: for each of places 1-10 a
probability distribution over games, plus an ordered top 10. The one forecast
is scored by log score against every hourly chart read in the 72 hours after
the history ends.

## Model

- Each game gets a latent strength: a recency-weighted mean (half-life 22 h)
  of `exp(-(rank-1)/12)` over all reads, absent counting as 0, plus a shrunk,
  clamped per-hour drift fitted from its rank trajectory.
- When the history's game details carry a `created` date, a game leaves the
  simulation once it passes 28 days since publication, as the real chart
  drops it.
- Monte-Carlo: for each forecast hour 1-72 the strengths drift, Gaussian
  ranking noise (sd growing with the horizon, scaled by the churn measured
  in the history) is added, games are ranked, and place counts are averaged
  over hours and 160 draws per hour.
- A flat quarter of every place is deliberately left unnamed: the scoring
  spreads it over every unnamed game on a read's chart, which insures
  against the new and low-rated games that so often take a place.

Fixed seed, no clock: two runs on the same history are byte-identical.

## Backtest (practice history, exact RULES log score)

| cut | this model | house starter | hard persistence |
| --- | --- | --- | --- |
| fit ≤ 2026-11-03, scored 10 reads | -41.11 | -50.89 | -81.97 |
| fit ≤ 2026-11-04T23:00, scored 12 reads | -42.13 | -51.77 | -79.06 |

Parameters were tuned on the first cut and reported unchanged on the second.
