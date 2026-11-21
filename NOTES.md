# Design and validation notes

## The problem

Forecast, for each of GTA VI's 55 PS5 trophies, the 5/25/50/75/95%
quantiles of its Exophase earn rate at 2026-11-26T00:00Z, from the house's
reads of the trophy page up to the deadline (2026-11-21T11:30Z). Scoring is
mean pinball loss on the logit scale, so a miss from 1% to 2% costs about
what a miss from 33% to 50% does — rare trophies count as much as common
ones, and the spread is scored as strictly as the median.

To pay out, the final loss must beat the best of three house entries: a
"stays the same" baseline (spread 0.5·sqrt(days) on the logit scale), a
curve baseline (logit rate linear in ln(1 + hours since release), fitted
on the last 24 h of reads, unshrunk), and a house AI model's forecast.

## What the live page showed (sanity check, 2026-11-21 ~02:40 UTC)

One look at https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/
two days after launch, used only to sanity-check priors — nothing from the
page is hardcoded in `main.py`, which forecasts from the case history
alone. Observations:

- The field is extremely skewed: one trophy at 100.00% (the first-action
  trophy), Chapter 1 at 18.02%, Chapter 2 at 3.61%, a handful of easy
  side trophies between 1% and 17%, and the great majority of trophies at
  0.00–0.13%. The platinum prints 0.00%.
- Against launch night (638 tracked players; Chapter 1 at 1.25%, Chapter 3
  at 1.10%), early trophies have RISEN sharply (progression outpacing
  dilution) while late-story trophies have FALLEN (dilution dominating:
  almost nobody has reached them, and the tracked base exploded). Any
  useful model must therefore fit each trophy's own direction rather than
  assume a common drift — and must not let the launch-night regime, cut
  from a few hundred pre-release players, anchor the fit.

## Design (as shipped in main.py)

1. Collapse the read history to distinct page states (Exophase changes
   only at refreshes); weight states by duration × a recency factor
   (36 h half-life).
2. Per trophy, fit weighted least squares of logit(rate) against
   ln(1 + hours since 2026-11-19T00:00Z) over the states of the last
   48 hours (falling back to all states when fewer than 4 are in window).
   The slope's standard error is inflated by the logit-scale rounding
   noise implied by rates being printed to 2 decimals — material for
   sub-0.1% trophies.
3. Shrink each slope toward the cross-trophy median slope with a normal
   prior (sd 1.4): noisy/flat trophies borrow the field's movement, clean
   trends keep theirs.
4. Median = last observed logit + shrunk slope × remaining ln-time
   distance, blended 12% back toward the last value, total move capped at
   ±3.2 logit.
5. Spread s = sqrt(resid² + (slope_se × distance)² + (0.37·sqrt(days))² +
   0.15²), floored at 0.35, capped at 2.0. Quantiles are the normal
   quantiles at the five levels, clamped to [0.01, 99.99], rounded to 4
   decimals.
6. Trophies never observed earned (0.00% in every read) get median 0.02%
   with s = 1.25; trophies with no observations at all get a wide band
   around 0.05%.

## What was tried and rejected

An earners/players decomposition (forecast ln(earners) and ln(players)
separately, divide at the target, blend with the trend by fit quality)
was implemented first. On the cohort backtest below it scored 1.04–1.43
on its own — far worse than the plain trend's ~0.19. The log-earner line
in ln-time compounds launch-week growth (the tracked base and the
cohort's progression multiply), so it extrapolates earners far past any
plausible count; a quadratic players fit did not rescue it. The shipped
program is the trend model alone, with the decomposition removed.

Window choice: a 24 h trend window scored a hair better in simulation
(0.189 vs 0.194); the 48 h window was kept because the real series is
stepped by Exophase refreshes (fewer, noisier states) and spans a regime
change into the first weekend — more states buy robustness for ~2% of
simulated loss.

## Validation

Practice cases (kit `input/p00.in`, `p01.in`): the kit ships no answer
reads, so practice validation is format + behaviour, not a score:

- Output validates exactly: `{"trophies":[{"id":n,"q":[5 quantiles]}]}`,
  55 rows, ids matching the case, quantiles monotone, within [0, 100],
  2,646-byte files (limit 65,536).
- Deterministic: two runs byte-identical. Wall time ~0.07 s for both
  cases locally (limit 8 s per case).
- Behaviour on the flat launch-night histories: medians stay at the last
  rates (e.g. trophy 4 at 1.10%) with a wide band (trophy 4:
  [0.0414, 0.2878, 1.1, 4.1098, 22.9862]) — the honest response to a
  history that has not yet seen a refresh, where the house baseline's
  narrow band is what produced its ~1.51 practice loss at ~0 coverage.

Synthetic backtest (`backtest.py`, exact competition pinball loss): a
cohort generating process — players join on a polynomial growth curve and
earn trophies by play-age with per-trophy caps and time constants,
producing the mixed rise/fall pattern of the real page — checkpointed at
the final's shape (120 reads, target 4.52 days out). Mean over 4 seeds:

| forecaster        | loss   |
|-------------------|--------|
| this entry        | 0.1942 |
| house curve base  | 0.2281 |
| house stays-same  | 0.3609 |

90% coverage of this entry's bands: 0.93 (target ~0.90; deliberately a
touch conservative, since the real series carries refresh-step risk the
simulation only approximates).

Edge cases exercised: a single-read history, null rates sprinkled through
reads, a trophy absent from every read, an empty history, rates pinned at
0.00 and 100.00 — all produce valid, monotone, in-range output.
