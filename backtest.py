#!/usr/bin/env python3
"""Local validation for the trophy forecaster (not part of the entry).

1. Runs main.py on the kit practice cases and validates the output format.
2. Checks determinism (two runs byte-identical) and wall time.
3. Builds a synthetic final-like case (refresh-stepped page states, growing
   players_tracked, saturating per-trophy earner curves), checkpoints it at
   2026-11-21T11:30Z, and scores this forecaster against the two house
   baselines with the competition's exact pinball-on-logit loss.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
KIT_INPUT = os.path.expanduser(
    "~/workspace/makefun/trophy-kit/gta6-trophy-rarity-forecast-01-kit/input")
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)


def logit(x):
    p = min(99.99, max(0.01, x)) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(v):
    if v >= 0:
        p = 1.0 / (1.0 + math.exp(-v))
    else:
        e = math.exp(v)
        p = e / (1.0 + e)
    return 100.0 * p


def pinball_loss(rows, truth):
    """rows: {id: [q05..q95]}, truth: {id: rate}. Exact competition loss."""
    total, count = 0.0, 0
    for tid, y in truth.items():
        ly = logit(y)
        for tau, q in zip(LEVELS, rows[tid]):
            d = ly - logit(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            count += 1
    return total / count


def validate_output(path, case):
    raw = open(path, "rb").read()
    assert len(raw) <= 65536, f"{path}: {len(raw)} bytes > 65536"
    doc = json.loads(raw.decode("utf-8"))
    assert set(doc.keys()) == {"trophies"}, doc.keys()
    rows = doc["trophies"]
    want = [t["id"] for t in case["trophies"]]
    got = [r["id"] for r in rows]
    assert sorted(got) == sorted(want) and len(set(got)) == len(got), "id set"
    for r in rows:
        assert set(r.keys()) == {"id", "q"}, r.keys()
        q = r["q"]
        assert len(q) == 5 and all(isinstance(v, (int, float)) for v in q)
        assert all(0.0 <= v <= 100.0 for v in q), q
        assert all(q[i] <= q[i + 1] for i in range(4)), f"not monotone: {q}"
    return {r["id"]: r["q"] for r in rows}


def run_program(in_dir, out_dir, script="main.py"):
    env = dict(os.environ, MAKEFUN_INPUT_DIR=in_dir, MAKEFUN_OUTPUT_DIR=out_dir)
    t0 = time.time()
    subprocess.run([sys.executable, os.path.join(HERE, script)],
                   env=env, check=True)
    return time.time() - t0


# ---------------------------------------------------------------- baselines

def baseline_stays_same(case):
    last = case["history"][-1]
    days = case["horizon_minutes"] / 1440.0
    s = 0.5 * math.sqrt(days)
    rows = {}
    for j, tr in enumerate(case["trophies"]):
        r = last["rates"][j]
        r = 1.0 if r is None else r
        rows[tr["id"]] = [round(inv_logit(logit(r) + z * s), 4) for z in Z]
    return rows


def _wls(xs, ys, ws):
    sw = sum(ws)
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum(w * (y - (a + b * x)) ** 2
                          for x, y, w in zip(xs, ys, ws)) / max(1.0, sw - 2.0))
    return a, b, resid


def baseline_curve(case):
    """House curve baseline per RULES.txt: logit ~ ln(1+h), last 24h of reads."""
    days = case["horizon_minutes"] / 1440.0
    ckpt = datetime.fromisoformat(case["at"].replace("Z", "+00:00"))
    target = datetime.fromisoformat(case["target"].replace("Z", "+00:00"))
    x_tgt = math.log(1.0 + (target - EPOCH).total_seconds() / 3600.0)
    rows = {}
    for j, tr in enumerate(case["trophies"]):
        xs, ys = [], []
        for read in case["history"]:
            t = datetime.fromisoformat(read["at"].replace("Z", "+00:00"))
            if (ckpt - t).total_seconds() > 24 * 3600:
                continue
            r = read["rates"][j]
            if r is None:
                continue
            xs.append(math.log(1.0 + max(0.0, (t - EPOCH).total_seconds() / 3600.0)))
            ys.append(logit(r))
        last_r = case["history"][-1]["rates"][j]
        last_r = 1.0 if last_r is None else last_r
        if len(xs) < 3:
            med, resid = logit(last_r), 0.0
        else:
            a, b, resid = _wls(xs, ys, [1.0] * len(xs))
            med = a + b * x_tgt
        s = math.sqrt((0.5 * math.sqrt(days)) ** 2 + resid ** 2)
        rows[tr["id"]] = [round(inv_logit(med + z * s), 4) for z in Z]
    return rows


# ---------------------------------------------------------------- synthetic

def synthetic_case(seed=7):
    """A final-shaped case from a cohort generating process.

    Players join over time (players_tracked grows polynomially). A player
    who joined at time s has earned trophy i by time h with probability
    cap_i * (1 - exp(-((h-s)/tau_i)^0.9)) -- progression in play-age. New
    joiners therefore dilute late trophies while early trophies track their
    cap quickly: the mixed rise/fall pattern the real page shows.
    """
    import random
    rng = random.Random(seed)
    trophies = []
    curves = []  # (kind, cap, tau) per trophy
    for i in range(55):
        tier = ("platinum" if i == 0 else "gold" if i < 3 else
                "silver" if i < 9 else "bronze")
        trophies.append({"id": i + 1, "name": f"T{i+1}", "description": "",
                         "tier": tier,
                         "points": 300 if tier == "platinum" else
                         90 if tier == "gold" else 30 if tier == "silver" else 15,
                         "secret": False})
        if i == 1:
            curves.append(("full", 1.0, 0.1))      # earned on first playing
        elif i in (0, 13, 14):
            curves.append(("zero", 0.0, 1e9))      # never earned in week one
        else:
            cap = rng.choice([0.003, 0.01, 0.04, 0.15, 0.45, 0.85])
            tau = rng.uniform(2, 300)
            curves.append(("grow", cap, tau))

    def players(h):
        return int(600 * (1 + max(0.0, h) / 14.0) ** 2.0) + 38

    # Cohorts: one per refresh boundary, sized by the players growth.
    refresh_hours = []
    h = -5.17
    while h < 170:
        refresh_hours.append(h)
        h += rng.uniform(3.5, 7.0)
    cohorts = []  # (join_time, size)
    prev_p = 0
    for k, rh in enumerate(refresh_hours):
        p = players(rh)
        cohorts.append((rh, max(0, p - prev_p)))
        prev_p = p

    def earners(i, h):
        kind, cap, tau = curves[i]
        total = 0.0
        for s, size in cohorts:
            if s > h:
                break
            age = h - s
            if kind == "full":
                total += size
            elif kind == "zero":
                continue
            else:
                total += size * cap * (1 - math.exp(-(age / tau) ** 0.9))
        return total

    def page_rates(h):
        p = players(h)
        return [round(100.0 * earners(i, h) / p, 2) for i in range(55)], p

    history = []
    t = datetime(2026, 11, 18, 23, 50, tzinfo=timezone.utc)
    ckpt = datetime(2026, 11, 21, 11, 30, tzinfo=timezone.utc)
    while t < ckpt:
        hh = (t - EPOCH).total_seconds() / 3600.0
        state_h = max(r for r in refresh_hours if r <= hh)
        rates, p = page_rates(state_h)
        history.append({"slot": t.isoformat().replace("+00:00", "Z"),
                        "at": t.isoformat().replace("+00:00", "Z"),
                        "players_tracked": p,
                        "trophies_earned": int(sum(rates) * p / 100),
                        "rates": rates})
        t += timedelta(minutes=30)
    target = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
    t_h = (target - EPOCH).total_seconds() / 3600.0
    truth_rates, _ = page_rates(max(r for r in refresh_hours if r <= t_h))
    case = {"id": "f00", "schema": "makefun/gta6-trophies-case/v1",
            "at": "2026-11-21T11:30:00Z", "target": "2026-11-26T00:00:00Z",
            "horizon_minutes": 6510, "levels": list(LEVELS),
            "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                     "release_date": "2026-11-19", "source": "synthetic"},
            "trophies": trophies, "history": history,
            "notes": "synthetic backtest case"}
    truth = {i + 1: truth_rates[i] for i in range(55)}
    return case, truth


def main():
    kit = {n[:-3]: json.load(open(os.path.join(KIT_INPUT, n)))
           for n in os.listdir(KIT_INPUT) if n.endswith(".in")}

    with tempfile.TemporaryDirectory() as td:
        out1, out2 = os.path.join(td, "o1"), os.path.join(td, "o2")
        dt = run_program(KIT_INPUT, out1)
        run_program(KIT_INPUT, out2)
        for cid, case in sorted(kit.items()):
            rows = validate_output(os.path.join(out1, cid + ".out"), case)
            same = open(os.path.join(out1, cid + ".out"), "rb").read() == \
                open(os.path.join(out2, cid + ".out"), "rb").read()
            print(f"practice {cid}: format OK, deterministic={same}, "
                  f"median[id3]={rows[3][2]}, median[id1]={rows[1][2]}")
        print(f"practice wall time (2 cases): {dt:.2f}s")

    import importlib.util
    spec = importlib.util.spec_from_file_location("entry_main",
                                                   os.path.join(HERE, "main.py"))
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    agg = {"mine": 0.0, "stay": 0.0, "curve": 0.0, "cov": 0.0}
    seeds = (7, 11, 23, 42)
    for seed in seeds:
        case, truth = synthetic_case(seed)
        mine = {r["id"]: r["q"] for r in entry.forecast_case(case)}
        lm = pinball_loss(mine, truth)
        ls = pinball_loss(baseline_stays_same(case), truth)
        lc = pinball_loss(baseline_curve(case), truth)
        cov = sum(1 for tid, y in truth.items()
                  if mine[tid][0] <= y <= mine[tid][4]) / len(truth)
        agg["mine"] += lm
        agg["stay"] += ls
        agg["curve"] += lc
        agg["cov"] += cov
        print(f"seed {seed}: mine={lm:.4f} stays={ls:.4f} curve={lc:.4f} "
              f"coverage={cov:.2f}")
    n = len(seeds)
    print(f"MEAN over {n} seeds: mine={agg['mine']/n:.4f} "
          f"stays={agg['stay']/n:.4f} curve={agg['curve']/n:.4f} "
          f"coverage={agg['cov']/n:.2f}")


if __name__ == "__main__":
    main()
