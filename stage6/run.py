"""
Stage 6: the lob-sim market makers on real L2 data.

    python stage6/run.py --symbol SOLUSDT                  # full grid, 2 workers
    python stage6/run.py --symbol BTCUSDT --quick          # small grid
    python stage6/run.py --symbol SOLUSDT --latency 50     # slower connection

Same discipline as the simulator work:
  - days are split by date: the earlier half trains, the later half is the holdout.
    nothing from a holdout day is seen before the configs are fixed.
  - on training days each family picks its config by mean daily gross pnl
    (before fees, so the choice is about edge, not about trading less):
      naive  half spread
      AS     half spread and gamma
      flow   the AS choice plus a flow coefficient
  - on holdout days the three fixed configs are compared paired by day,
    with a Holm correction across the comparisons.

Reported per bot: gross pnl, edge in basis points of traded notional (the
number to hold against the exchange fee in bp), the split into spread and
inventory pnl and net pnl at each fee tier. Every fill is post only, so the
maker fee applies to all of them and net = gross - maker_rate * traded value.

Units: hs and the skews are in ticks, order size in base units, position cap
in multiples of the order size, one quoting interval is --interval-ms.
"""
import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

FEE_TIERS = {"VIP rebate -0.5bp": -0.00005, "zero": 0.0, "maker 1bp": 0.0001, "retail 2bp": 0.0002}
TAKER = 0.0005


def run_one(args):
    npz, cfg, tick, lot, latency_ms, interval_ms, qty, cap = args
    from stage6.bots import NCOL, R_EQUITY, R_MID, R_POS, R_FILLS, R_TVALUE, run_bot
    from stage6.common import make_backtest
    hbt = make_backtest(npz, tick, lot, latency_ms=latency_ms)
    out = np.zeros((1_000_000, NCOL))
    n = run_bot(hbt, out, int(interval_ms * 1e6), cfg["hs"], qty, cap,
                cfg.get("gamma", 0.0), 50.0, 15.0, cfg.get("flow", 0.0), 20 if cfg.get("flow") else 0, 200)
    hbt.close()
    r = out[:n]
    if n < 2:
        return npz, cfg, None
    mid, pos, eq = r[:, R_MID], r[:, R_POS], r[:, R_EQUITY]
    gross = eq[-1]
    inv = float(np.sum(pos[:-1] * np.diff(mid)))
    tval = r[-1, R_TVALUE]
    liq_cost = abs(pos[-1]) * (tick + TAKER * mid[-1])
    return npz, cfg, {
        "gross": gross, "inventory": inv, "spread": gross - inv, "tvalue": tval,
        "fills": int(r[-1, R_FILLS]), "edge_bp": 1e4 * gross / tval if tval > 0 else float("nan"),
        "max_pos": float(np.abs(pos).max() / qty), "liq_cost": liq_cost,
    }


def grid(quick):
    hs = [1, 2, 3] if quick else [1, 2, 3, 5, 8]
    gammas = [0.001, 0.005] if quick else [0.0005, 0.001, 0.002, 0.005, 0.01]
    coefs = [2, 6] if quick else [1, 2, 4, 6, 8]
    return hs, gammas, coefs


def key(cfg):
    return tuple(sorted(cfg.items()))


def mean_t(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2 or xs.std(ddof=1) == 0:
        return float(xs.mean()) if len(xs) else float("nan"), float("nan")
    return float(xs.mean()), float(xs.mean() / (xs.std(ddof=1) / math.sqrt(len(xs))))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def t_p(t, n):
    """Two sided p for a paired t with n-1 degrees of freedom. Real data comes
    one free day a month, so n is small and the normal approximation would be
    badly wrong: with 2 days, t = 23 is p = 0.03, not 1e-119."""
    from scipy.stats import t as student_t
    if n < 2 or not math.isfinite(t):
        return 1.0
    return float(2 * student_t.sf(abs(t), df=n - 1))


def main():
    from stage6.common import LOT, TICK, days, infer_tick
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--npzdir", default="data/npz")
    ap.add_argument("--tick", type=float)
    ap.add_argument("--lot", type=float)
    ap.add_argument("--qty", type=float, help="order size in base units, default 1 lot")
    ap.add_argument("--cap", type=float, default=30, help="position cap in order-size units")
    ap.add_argument("--latency", type=float, default=10.0, help="order entry and response latency, ms")
    ap.add_argument("--interval-ms", type=float, default=100.0)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=2,
                    help="parallel backtests; each holds a whole day in memory (default 2)")
    a = ap.parse_args()

    ds = days(a.npzdir, a.symbol)
    if len(ds) < 2:
        raise SystemExit(f"need at least 2 converted days in {a.npzdir}; run download.py and convert.py")
    tick = a.tick or TICK.get(a.symbol) or infer_tick(ds[0])
    lot = a.lot or LOT.get(a.symbol) or tick
    qty = a.qty or lot
    k = max(1, int(round(len(ds) * a.train_frac)))
    train, test = ds[:k], ds[k:]
    if not test:
        raise SystemExit("no holdout days left; download more days or lower --train-frac")
    print(f"{a.symbol}: tick {tick}, lot {lot}, order {qty}, cap {a.cap} orders, latency {a.latency}ms,"
          f" interval {a.interval_ms}ms")
    print(f"train days: {[os.path.basename(d)[-12:-4] for d in train]}")
    print(f"holdout days: {[os.path.basename(d)[-12:-4] for d in test]}")

    hs_list, gammas, coefs = grid(a.quick)
    common = (tick, lot, a.latency, a.interval_ms, qty, a.cap)

    def evaluate(cfgs, day_list):
        jobs = [(d, c) + common for c in cfgs for d in day_list]
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(run_one, jobs))
        by = {}
        for d, c, r in res:
            if r is not None:
                by.setdefault(key(c), {})[d] = r
        return by

    def best(by, cfgs):
        scored = []
        for c in cfgs:
            rs = by.get(key(c), {})
            if rs:
                scored.append((np.mean([r["gross"] for r in rs.values()]), c))
        return max(scored, key=lambda x: x[0])

    # training
    naive_cfgs = [{"hs": h} for h in hs_list]
    as_cfgs = [{"hs": h, "gamma": g} for h in hs_list for g in gammas]
    print(f"\ntraining: {len(naive_cfgs) + len(as_cfgs)} configs x {len(train)} days ...")
    tr = evaluate(naive_cfgs + as_cfgs, train)
    s_n, best_n = best(tr, naive_cfgs)
    s_a, best_a = best(tr, as_cfgs)
    flow_cfgs = [dict(best_a, flow=c) for c in coefs]
    tr.update(evaluate(flow_cfgs, train))
    s_f, best_f = best(tr, flow_cfgs)
    print(f"  chosen naive {best_n}  (train mean gross {s_n:,.2f})")
    print(f"  chosen AS    {best_a}  (train mean gross {s_a:,.2f})")
    print(f"  chosen flow  {best_f}  (train mean gross {s_f:,.2f})")

    # holdout
    chosen = {"naive": best_n, "AS": best_a, "flow": best_f}
    ho = evaluate(list(chosen.values()), test)
    print(f"\n== holdout, {len(test)} days, per day means ==")
    print(f"{'bot':6} | {'gross':>10} | {'edge bp':>8} | {'spread':>10} {'inventory':>10} | {'fills':>7} "
          f"| {'max pos':>7} | " + " | ".join(f"{t:>17}" for t in FEE_TIERS))
    rows = {}
    for name, c in chosen.items():
        rs = [ho[key(c)][d] for d in test if d in ho.get(key(c), {})]
        rows[name] = rs
        g = np.mean([r["gross"] for r in rs])
        tv = np.mean([r["tvalue"] for r in rs])
        nets = [np.mean([r["gross"] - fee * r["tvalue"] - r["liq_cost"] for r in rs]) for fee in FEE_TIERS.values()]
        print(f"{name:6} | {g:>10,.2f} | {1e4 * g / tv if tv else float('nan'):>8.3f} | "
              f"{np.mean([r['spread'] for r in rs]):>10,.2f} {np.mean([r['inventory'] for r in rs]):>10,.2f} | "
              f"{np.mean([r['fills'] for r in rs]):>7,.0f} | {np.mean([r['max_pos'] for r in rs]):>7.1f} | "
              + " | ".join(f"{x:>17,.2f}" for x in nets))
    print("(net columns subtract the maker fee on all traded value and a liquidation cost of"
          " one tick plus the taker fee on the end of day position)")

    print("\n== paired by holdout day, gross pnl ==")
    tests, ps = [], []
    for b, a_ in (("AS", "naive"), ("flow", "AS")):
        da = {d: r for d, r in zip(test, rows[a_])}
        db = {d: r for d, r in zip(test, rows[b])}
        common_days = [d for d in test if d in da and d in db]
        diff = [db[d]["gross"] - da[d]["gross"] for d in common_days]
        dinv = [db[d]["inventory"] - da[d]["inventory"] for d in common_days]
        m, t = mean_t(diff)
        mi, _ = mean_t(dinv)
        wins = sum(1 for x in diff if x > 0)
        tests.append((f"{b} minus {a_}", m, t, wins, len(diff), mi))
        ps.append(t_p(t, len(diff)))
    for (name, m, t, w, n, mi), p, pa in zip(tests, ps, holm(ps)):
        part = f"inventory part {mi:,.2f}" if math.isfinite(mi) else ""
        print(f"  {name:14} mean {m:>10,.2f}  t {t:>6.2f}  better {w}/{n} days  p {p:.3g}  holm p {pa:.3g}  {part}")
    if len(test) < 10:
        print(f"  note: {len(test)} holdout days is few; treat t values as indicative, not conclusive")


if __name__ == "__main__":
    main()
