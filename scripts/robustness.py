"""
Robustness of the headline results, on holdout seeds the parameters were
never tuned on (gamma and the naive half spread were chosen on seeds 100-119).

Every comparison is paired: same tape, naive vs AS. Each row is one
hypothesis test. The table ends with a Holm correction across all of them,
since with 30 odd tests a few will look good by chance.

python scripts/robustness.py          # 30 holdout seeds, uses all cores, ~10 min
python scripts/robustness.py 10       # quicker
"""
import math
import sys
from concurrent.futures import ProcessPoolExecutor

from lobsim.analytics import evaluate, paired_compare
from lobsim.strategy import InventoryMaker, NaiveMaker

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
HOLDOUT = list(range(1000, 1000 + N))
METRICS = ("total_pnl", "liq_pnl", "max_drawdown", "pos_std")


def naive(hs, **kw):
    return lambda: NaiveMaker(half_spread=hs, requote_always=False, max_position=30, **kw)


def as_(hs, g=0.002, **kw):
    return lambda: InventoryMaker(half_spread=hs, gamma=g, spread_mode="fixed",
                                  requote_always=False, max_position=30, **kw)


TAPES = {
    "flat": {},
    "trend +0.4": {"drift": 0.4},
    "trend -0.4": {"drift": -0.4},
    "trend +0.2": {"drift": 0.2},
    "trend -0.2": {"drift": -0.2},
    "flat, fat tails (t, df 4)": {"tail_df": 4},
    "flat, vol regimes (3x, p=0.01)": {"vol_regime": (3, 0.01)},
    "flat, fat tails + vol regimes": {"tail_df": 4, "vol_regime": (3, 0.01)},
    "trend +0.4, fat tails + vol regimes": {"drift": 0.4, "tail_df": 4, "vol_regime": (3, 0.01)},
    "flat, latency 1": {"bot_latency": 1},
    "flat, latency 3": {"bot_latency": 3},
    "trend +0.4, latency 3": {"drift": 0.4, "bot_latency": 3},
}

# fees in basis points of notional. one tick is 1bp of price at 10,000.
# binance usd-m futures base tier: maker 2bp, taker 5bp.
FEES = {"rebate -0.5bp / 5bp": (-0.00005, 0.0005), "zero": (0.0, 0.0),
        "maker 1bp / taker 5bp": (0.0001, 0.0005), "retail 2bp / 5bp": (0.0002, 0.0005)}


def paired_job(args):
    tape, hs, kw = args
    return tape, hs, paired_compare(naive(hs), as_(hs), HOLDOUT, metrics=METRICS, **kw)


def fee_job(args):
    fname, sname, hs, kind = args
    mf, tf = FEES[fname]
    mk = naive(hs, maker_fee=mf, taker_fee=tf) if kind == "naive" else as_(hs, maker_fee=mf, taker_fee=tf)
    r = evaluate(mk, HOLDOUT[:20])
    return fname, sname, r


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, adj, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def t_to_p(t, n):
    # two sided p from a t statistic, normal approximation is fine at n >= 20
    # for the purpose of ranking; it is slightly anti-conservative at n = 10.
    if math.isinf(t):
        return 0.0
    return math.erfc(abs(t) / math.sqrt(2))


if __name__ == "__main__":
    jobs = [(tape, hs, kw) for tape, kw in TAPES.items() for hs in (3, 8)]
    with ProcessPoolExecutor() as ex:
        paired = list(ex.map(paired_job, jobs))
        fee_jobs = [(f, s, hs, kind) for f in FEES for s, hs, kind in
                    (("naive hs3", 3, "naive"), ("naive hs8", 8, "naive"),
                     ("AS hs3", 3, "as"), ("AS hs8", 8, "as"))]
        fees = list(ex.map(fee_job, fee_jobs))

    print(f"== AS g=0.002 minus naive, same half spread, {N} holdout seeds 1000+ ==")
    print(f"{'tape':40} {'hs':>3} | {'total diff':>10} {'t':>6} {'b/w':>6} | {'liq diff':>9} {'t':>6} "
          f"| {'dd t':>6} | {'posstd t':>8}")
    rows, ps = [], []
    for tape, hs, r in paired:
        tp, lq, dd, ps_ = r["total_pnl"], r["liq_pnl"], r["max_drawdown"], r["pos_std"]
        rows.append((tape, hs, r))
        ps.append(t_to_p(lq["t"], lq["n"]))
        print(f"{tape:40} {hs:>3} | {tp['mean_diff']:>10,} {tp['t']:>6} {tp['better']:>2}/{tp['worse']:<2} "
              f"| {lq['mean_diff']:>9,} {lq['t']:>6} | {dd['t']:>6} | {ps_['t']:>8}")

    adj = holm(ps)
    print(f"\n== Holm-adjusted p for the liquidation pnl difference ({len(ps)} tests) ==")
    for (tape, hs, r), p, pa in zip(rows, ps, adj):
        verdict = "AS better" if pa < 0.05 and r["liq_pnl"]["mean_diff"] > 0 else \
                  "naive better" if pa < 0.05 else "no significant difference"
        print(f"{tape:40} hs={hs:<2} raw p={p:.2g}  holm p={pa:.2g}  -> {verdict}")

    print(f"\n== fees, flat, 20 holdout seeds, pnl after liquidation ==")
    by = {}
    for f, s, r in fees:
        by.setdefault(f, {})[s] = r
    names = ("naive hs3", "naive hs8", "AS hs3", "AS hs8")
    print(f"{'fees':24} | " + " | ".join(f"{n:>18}" for n in names))
    for f in FEES:
        print(f"{f:24} | " + " | ".join(
            f"{by[f][n]['liq_pnl'][0]:>9,.0f} ±{by[f][n]['liq_pnl'][1]:>6,.0f}" for n in names))
