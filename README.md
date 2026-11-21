[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/soedirman566-stack/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/soedirman566-stack/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecaster

A forecasting program for the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
for every PlayStation 5 trophy of Grand Theft Auto VI, it predicts five
quantiles (5%, 25%, 50%, 75%, 95%) of the trophy's earn rate on Exophase at
the case's target time — in the final, the share of tracked players holding
each trophy one week after launch.

## Method

Everything is computed per trophy on the logit scale, the scale the
competition's pinball loss uses.

- **Page states, not reads.** Exophase only changes its counts at
  refreshes, so consecutive identical reads are collapsed into one state,
  weighted by how long it lasted with a mild recency tilt.
- **Saturating trend.** A weighted least-squares line of logit(rate)
  against ln(1 + hours since release), fitted over the last 48 hours of
  states. The ln-time axis makes progression decelerate in clock time, and
  the recent window leaves the launch-night regime (a few hundred tracked
  players) behind.
- **Shrunk slopes.** Each trophy's slope is shrunk toward the cross-trophy
  median slope, empirical-Bayes style, so a noisy or flat series borrows
  the field's typical movement while a clean strong trend keeps its own.
  The forecast re-anchors at the last observed rate and applies the shrunk
  slope over the remaining ln-time distance, capped at ±3.2 logit.
- **Honest spread.** The band combines the fit's residual scale, the slope
  uncertainty over the extrapolation distance, and a horizon term growing
  like sqrt(days), targeting ~90% coverage.
- **Edge cases.** Trophies never observed earned keep a tiny median and a
  wide band; rates are clamped to the printable range [0.01, 99.99].

An earners/players decomposition model was built and rejected during
development — its log-earner extrapolation compounds launch-week growth
and scored far worse on backtests. See `NOTES.md` for the design record
and validation numbers.

## Run it

```sh
MAKEFUN_INPUT_DIR=/path/to/cases MAKEFUN_OUTPUT_DIR=/path/to/out python3 main.py
```

The program reads every `*.in` case file in the input directory (default
`/input`) and writes one `<id>.out` JSON file per case to the output
directory (default `/output`). Python 3 standard library only; no network;
deterministic — two runs on the same case produce byte-identical output,
in well under a second per case.

`backtest.py` holds the local validation harness (format checks,
determinism, and a synthetic cohort-process backtest scored with the
competition's exact pinball loss against both house baselines).

## License

MIT — see `LICENSE`.
