"""
Stage 5: adverse selection defences.

    python scripts/stage5.py ablate     # training seeds 100-109, each defence and combinations
    python scripts/stage5.py            # holdout seeds 1000-1029, the locked config, Holm, fees

The defended config was chosen on the training seeds with a rule fixed before
looking at the holdout: maximise the worst t across flat, flat with latency 3
and trend. That picked trade flow skew alone, coefficient 6, window 20.
Everything in holdout mode is that one config, untouched.
"""
import sys
from concurrent.futures import ProcessPoolExecutor

from lobsim.analytics import evaluate, paired_compare
from lobsim.strategy import DefensiveMaker, InventoryMaker, NaiveMaker

BASE = dict(gamma=0.002, spread_mode="fixed", requote_always=False, max_position=30)
LOCKED = dict(flow_coef=6, flow_window=20)

TRAIN = range(100, 110)
HOLDOUT = range(1000, 1030)

ABLATE_TAPES = {"flat": {}, "flat lat3": {"bot_latency": 3}, "trend": {"drift": 0.4}}
ABLATE = {
    "requote tol 1": {"requote_tol": 1}, "requote tol 2": {"requote_tol": 2},
    "imbalance 1": {"imb_coef": 1}, "imbalance 4": {"imb_coef": 4},
    "flow 2": {"flow_coef": 2}, "flow 4": {"flow_coef": 4}, "flow 6": {"flow_coef": 6},
    "flow 8": {"flow_coef": 8},
    "toxicity 1": {"tox_gain": 1}, "toxicity 2": {"tox_gain": 2},
    "flow 4 + tol 1": {"flow_coef": 4, "requote_tol": 1},
    "flow 4 + toxicity 1": {"flow_coef": 4, "tox_gain": 1},
    "all four": {"flow_coef": 4, "tox_gain": 1, "requote_tol": 1, "imb_coef": 1},
}

HOLD_TAPES = {
    "flat": {},
    "trend +0.4": {"drift": 0.4},
    "trend -0.4": {"drift": -0.4},
    "trend +0.2": {"drift": 0.2},
    "trend -0.2": {"drift": -0.2},
    "flat, fat tails": {"tail_df": 4},
    "flat, vol regimes": {"vol_regime": (3, 0.01)},
    "flat, fat tails + vol regimes": {"tail_df": 4, "vol_regime": (3, 0.01)},
    "trend +0.4, fat tails + vol regimes": {"drift": 0.4, "tail_df": 4, "vol_regime": (3, 0.01)},
    "flat, latency 1": {"bot_latency": 1},
    "flat, latency 3": {"bot_latency": 3},
    "trend +0.4, latency 3": {"drift": 0.4, "bot_latency": 3},
}
METRICS = ("liq_pnl", "fair_edge", "max_drawdown", "pos_std")

FEES = {"zero": (0.0, 0.0), "maker 1bp / taker 5bp": (0.0001, 0.0005),
        "retail 2bp / 5bp": (0.0002, 0.0005)}


def ablate_job(a):
    name, tape = a
    return name, tape, paired_compare(
        lambda: InventoryMaker(**BASE), lambda: DefensiveMaker(**BASE, **ABLATE[name]),
        TRAIN, metrics=METRICS, **ABLATE_TAPES[tape])


def hold_job(tape):
    return tape, paired_compare(
        lambda: InventoryMaker(**BASE), lambda: DefensiveMaker(**BASE, **LOCKED),
        HOLDOUT, metrics=METRICS, **HOLD_TAPES[tape])


def fee_job(a):
    fee, name = a
    mf, tf = FEES[fee]
    f = dict(maker_fee=mf, taker_fee=tf)
    make = {
        "naive hs3": lambda: NaiveMaker(requote_always=False, max_position=30, **f),
        "naive hs8": lambda: NaiveMaker(half_spread=8, requote_always=False, max_position=30, **f),
        "AS hs3": lambda: InventoryMaker(**BASE, **f),
        "defended hs3": lambda: DefensiveMaker(**BASE, **LOCKED, **f),
        "defended hs8": lambda: DefensiveMaker(**{**BASE, "half_spread": 8}, **LOCKED, **f),
    }[name]
    return fee, name, evaluate(make, list(HOLDOUT)[:20])


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def p_two_sided(t):
    import math
    return 0.0 if math.isinf(t) else math.erfc(abs(t) / math.sqrt(2))


def row(r):
    l, e, d, p = r["liq_pnl"], r["fair_edge"], r["max_drawdown"], r["pos_std"]
    return (f"{l['mean_diff']:>9,} {l['t']:>6} {l['better']:>2}/{l['n']:<2} | {e['mean_diff']:>9,} "
            f"{e['t']:>6} | {d['t']:>6} | {p['mean_diff']:>6} {p['t']:>6}")


HDR = f"{'pnl diff':>9} {'t':>6} {'b/n':>5} | {'edge diff':>9} {'t':>6} | {'dd t':>6} | {'pos sd':>6} {'t':>6}"

if __name__ == "__main__":
    if "ablate" in sys.argv:
        jobs = [(n, t) for n in ABLATE for t in ABLATE_TAPES]
        with ProcessPoolExecutor() as ex:
            out = list(ex.map(ablate_job, jobs))
        print(f"== training seeds {TRAIN.start}-{TRAIN.stop - 1}: defended minus AS ==")
        print(f"{'config':20} {'tape':10} | {HDR}")
        for n, t, r in out:
            print(f"{n:20} {t:10} | {row(r)}")
        sys.exit()

    with ProcessPoolExecutor() as ex:
        held = list(ex.map(hold_job, HOLD_TAPES))
        fees = list(ex.map(fee_job, [(f, n) for f in FEES for n in
                                     ("naive hs3", "naive hs8", "AS hs3", "defended hs3", "defended hs8")]))

    print(f"== holdout seeds {HOLDOUT.start}-{HOLDOUT.stop - 1}: defended (flow 6) minus AS g=0.002, hs3 ==")
    print(f"{'tape':38} | {HDR}")
    for t, r in held:
        print(f"{t:38} | {row(r)}")

    ps = [p_two_sided(r["liq_pnl"]["t"]) for _, r in held]
    adj = holm(ps)
    print(f"\n== Holm-adjusted p, pnl after liquidation ({len(ps)} tests) ==")
    for (t, r), p, a in zip(held, ps, adj):
        d = r["liq_pnl"]["mean_diff"]
        v = "defended better" if a < 0.05 and d > 0 else "AS better" if a < 0.05 else "no significant difference"
        print(f"{t:38} raw p={p:.2g}  holm p={a:.2g}  -> {v}")

    print("\n== fees, flat, 20 holdout seeds, pnl after liquidation ==")
    names = ("naive hs3", "naive hs8", "AS hs3", "defended hs3", "defended hs8")
    by = {}
    for f, n, r in fees:
        by.setdefault(f, {})[n] = r
    print(f"{'fees':22} | " + " | ".join(f"{n:>17}" for n in names))
    for f in FEES:
        print(f"{f:22} | " + " | ".join(
            f"{by[f][n]['liq_pnl'][0]:>8,.0f} ±{by[f][n]['liq_pnl'][1]:>6,.0f}" for n in names))
