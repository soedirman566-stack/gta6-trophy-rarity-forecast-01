#!/usr/bin/env python3
"""Forecast GTA VI PS5 trophy earn rates on Exophase at a case's target time.

Reads every *.in case file in $MAKEFUN_INPUT_DIR (else /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output). For each trophy the output
holds five quantiles (5/25/50/75/95%) of its earn rate, in percent, at the
case's target. Standard library only, deterministic, no network.

Method, per trophy, on the logit scale (the scale the pinball loss uses):

  * The read history is first collapsed to distinct page states: Exophase
    only changes its counts at refreshes, so consecutive identical reads
    carry no new information. Each state is weighted by how long it lasted,
    with a mild recency tilt.
  * A weighted least-squares line of logit(rate) against ln(1 + hours since
    release) is fitted over the recent states (the last TREND_WINDOW_H
    hours; the launch-night regime does not extrapolate). The ln-time axis
    makes the trend saturate in clock time, which is how trophy progression
    behaves. The slope is shrunk toward the cross-trophy median slope
    (empirical-Bayes style, prior sd TAU_SLOPE), so a trophy whose own
    series is noisy or flat borrows the field's typical movement instead of
    extrapolating noise, and a trophy with a clean strong trend keeps it.
  * The median is that trend, blended a little toward the last observed
    rate, with the total move away from the last rate capped. (A second
    model -- forecasting earners and players_tracked separately and
    dividing -- was built and rejected: on cohort-process backtests its
    log-earner extrapolation compounds launch-week growth and it scored
    far worse than the plain trend. See NOTES.md.)
  * The spread combines the fit residual scale, the slope uncertainty over
    the extrapolation distance, and a horizon term that grows like
    sqrt(days), so the 90% band is honestly wide for a 4.5-day horizon and
    narrow for a 90-minute one.
"""

import json
import math
import os
from datetime import datetime, timezone

EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
TAU_SLOPE = 1.4          # prior sd of a trophy's slope around the field median
RECENCY_HALF_LIFE_H = 36.0
TREND_WINDOW_H = 48.0    # the rate-trend fit uses this much recent history
MAX_LOGIT_MOVE = 3.2     # cap on |median logit - last logit|
SPREAD_FLOOR = 0.35
SPREAD_CAP = 2.0
HORIZON_SPREAD = 0.37    # logit sd per sqrt(horizon day)


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_epoch(moment):
    return (moment - EPOCH).total_seconds() / 3600.0


def clamp_rate(rate):
    return min(99.99, max(0.01, rate))


def logit_pct(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(value):
    if value >= 0.0:
        p = 1.0 / (1.0 + math.exp(-value))
    else:
        e = math.exp(value)
        p = e / (1.0 + e)
    return 100.0 * p


def rounding_noise_sd(rate):
    """Logit-scale sd implied by rates being printed to 2 decimals."""
    r = clamp_rate(rate)
    deriv = 100.0 / (r * (100.0 - r))     # d logit / d rate (rate in percent)
    return min(1.0, 0.29 * 0.01 * deriv)  # uniform rounding, sd = width/sqrt(12)


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.

    Returns (a, b, resid_sd, slope_se). Degenerate inputs collapse to the
    weighted mean with slope 0 and a large slope_se.
    """
    n = len(xs)
    sw = sum(ws)
    if n == 0 or sw <= 0.0:
        return 0.0, 0.0, 1.0, 10.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if n < 2 or sxx <= 1e-12:
        return my, 0.0, 1.0, 10.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1.0, sw - 2.0 * (sw / n))
    var = sum(w * r * r for r, w in zip(resid, ws)) / dof
    resid_sd = math.sqrt(max(var, 1e-9))
    slope_se = resid_sd / math.sqrt(sxx)
    return a, b, resid_sd, slope_se


def shrink_slope(slope, slope_se, field_median):
    """Posterior-mean slope under a normal prior centred on the field median."""
    var_i = slope_se * slope_se
    var_p = TAU_SLOPE * TAU_SLOPE
    return (slope / var_i + field_median / var_p) / (1.0 / var_i + 1.0 / var_p)


def median_of(values):
    vals = sorted(values)
    m = len(vals)
    if m == 0:
        return 0.0
    if m % 2:
        return vals[m // 2]
    return 0.5 * (vals[m // 2 - 1] + vals[m // 2])


def build_states(history):
    """Collapse consecutive identical reads into page states.

    Returns a list of dicts with keys t (hours since epoch), players, rates,
    weight (duration in hours, mildly recency-tilted).
    """
    states = []
    for read in history:
        players = read.get("players_tracked")
        rates = read.get("rates") or []
        if players is None:
            continue
        t = hours_since_epoch(parse_time(read["at"]))
        if states and states[-1]["players"] == players \
                and states[-1]["rates"] == list(rates):
            continue
        states.append({"t": t, "players": float(players),
                       "rates": list(rates), "weight": 0.0})
    if not states:
        return states
    durations = []
    for i, st in enumerate(states):
        if i + 1 < len(states):
            dur = max(0.05, states[i + 1]["t"] - st["t"])
        else:
            dur = median_of(durations) if durations else 1.0
        durations.append(dur)
        st["weight"] = dur
    t_last = states[-1]["t"]
    for st in states:
        age = max(0.0, t_last - st["t"])
        st["weight"] *= 0.5 ** (age / RECENCY_HALF_LIFE_H)
    return states


def fit_series(states, values):
    """Fit values (already on their working scale) against ln(1 + hours)."""
    xs, ys, ws = [], [], []
    for st, v in zip(states, values):
        if v is None:
            continue
        xs.append(math.log(1.0 + max(0.0, st["t"])))
        ys.append(v)
        ws.append(st["weight"])
    return xs, ys, ws


def forecast_case(case):
    trophies = case["trophies"]
    n_tr = len(trophies)
    states = build_states(case.get("history") or [])
    t_target = hours_since_epoch(parse_time(case["target"]))
    x_target = math.log(1.0 + max(0.0, t_target))
    horizon_days = max(0.0, (parse_time(case["target"])
                             - parse_time(case["at"])).total_seconds()) / 86400.0

    if not states:
        return [{"id": tr["id"],
                 "q": [round(inv_logit_pct(logit_pct(1.0) + z * 1.5), 4)
                       for z in Z]} for tr in trophies]

    # ---- per-trophy raw fits, first pass (for the field-median slope) ----
    # The rate trend is fitted on the recent window only: the launch-night
    # regime (a few hundred tracked players, pre-refresh rates) does not
    # extrapolate, while the last two days' dynamics do.
    t_ckpt = hours_since_epoch(parse_time(case["at"]))
    raw = []
    for j in range(n_tr):
        rates = [st["rates"][j] if j < len(st["rates"]) else None
                 for st in states]
        obs = [(st, r) for st, r in zip(states, rates) if r is not None]
        if not obs:
            raw.append(None)
            continue
        recent = [(st, r) for st, r in obs
                  if st["t"] >= t_ckpt - TREND_WINDOW_H]
        fit_obs = recent if len(recent) >= 4 else obs
        xs, ys, ws = fit_series([s for s, _ in fit_obs],
                                [logit_pct(r) for _, r in fit_obs])
        a, b, resid, se = wls_line(xs, ys, ws)
        noise = median_of([rounding_noise_sd(r) for _, r in obs])
        se = math.sqrt(se * se + (noise * noise) / max(1.0, sum(ws)))
        raw.append({
            "obs": obs, "b": b, "resid": resid, "se": se,
            "n_fit": len(fit_obs),
            "last_rate": obs[-1][1], "last_logit": logit_pct(obs[-1][1]),
            "max_rate": max(r for _, r in obs),
            "x_last": xs[-1] if xs else 0.0,
        })

    rate_slopes = [d["b"] for d in raw if d is not None and d["n_fit"] >= 3]
    field_rate_slope = median_of(rate_slopes) if rate_slopes else 0.0

    rows = []
    for j, tr in enumerate(trophies):
        d = raw[j]
        if d is None:
            med, spread = logit_pct(0.05), 1.4
        elif d["max_rate"] <= 0.005:
            # Never observed earned (prints 0.00 in every read): a handful of
            # earners may appear by the target; keep the median tiny and the
            # band wide.
            med, spread = logit_pct(0.02), 1.25
        else:
            b = shrink_slope(d["b"], d["se"], field_rate_slope)
            # Re-anchor: keep the observed level at the last state, apply
            # the shrunk slope over the remaining ln-time distance only.
            trend = d["last_logit"] + b * (x_target - d["x_last"])

            w_last = 0.12 if d["n_fit"] >= 4 else 0.3
            med = (1.0 - w_last) * trend + w_last * d["last_logit"]
            move = med - d["last_logit"]
            if abs(move) > MAX_LOGIT_MOVE:
                med = d["last_logit"] + math.copysign(MAX_LOGIT_MOVE, move)

            var = (d["resid"] ** 2
                   + (d["se"] * (x_target - d["x_last"])) ** 2
                   + (HORIZON_SPREAD * math.sqrt(horizon_days)) ** 2
                   + 0.15 ** 2)
            spread = min(SPREAD_CAP, max(SPREAD_FLOOR, math.sqrt(var)))
        q = []
        for z in Z:
            v = inv_logit_pct(med + z * spread)
            q.append(round(min(99.99, max(0.01, v)), 4))
        for k in range(1, len(q)):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": tr["id"], "q": q})
    return rows


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        payload = json.dumps({"trophies": rows}, separators=(",", ":"))
        out_path = os.path.join(out_dir, str(case["id"]) + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(payload)


if __name__ == "__main__":
    main()
