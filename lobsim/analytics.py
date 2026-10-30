"""
Post-trade analytics. Everything a strategy run gets judged on lives here so
the numbers are computed the same way for every bot.
"""

import math
import statistics

from .market import INFORMED_PID, Market
from .orderbook import Side


def mid_series(history):
    """step -> last known mid, forward filled over the odd empty-side step."""
    out, last = {}, None
    for t, _, mid, _, _ in history:
        if mid is not None:
            last = mid
        out[t] = last
    return out


def markouts(fills, history, horizons=(1, 5, 20, 50, 100)):
    """
    Average pnl per lot h steps after each fill, split by counterparty.

    Positive: the market moved in your favour after you got filled.
    Negative: adverse selection, whoever hit you knew something.
    """
    mids = mid_series(history)
    out = {"informed": [], "noise": [], "all": []}
    for h in horizons:
        num = {"informed": 0.0, "noise": 0.0, "all": 0.0}
        den = {"informed": 0, "noise": 0, "all": 0}
        for t, side, px, qty, pid, _ in fills:
            m = mids.get(t + h)
            if m is None:
                continue
            v = (m - px) if side is Side.BUY else (px - m)
            for k in ("all", "informed" if pid == INFORMED_PID else "noise"):
                num[k] += v * qty
                den[k] += qty
        for k in out:
            out[k].append(num[k] / den[k] if den[k] else float("nan"))
    return list(horizons), out


def fair_edge(fills, history):
    """
    Edge at fill measured against the simulator's true price, summed over
    fills, lot weighted. The strategy never sees this; it is a post-trade
    check that only a simulator can do. Spread pnl against the mid can look
    large when the mid lags fair value or when the bot's own quotes set the
    mid. This is the number that says whether the fills were actually good.
    """
    true_at = {t: tr for t, tr, *_ in history}
    out = 0.0
    for t, side, px, qty, _, _ in fills:
        tp = true_at.get(t)
        if tp is None:
            continue
        out += (tp - px) * qty if side is Side.BUY else (px - tp) * qty
    return out


def max_drawdown(pnl):
    peak, worst = 0.0, 0.0
    for x in pnl:
        peak = max(peak, x)
        worst = min(worst, x - peak)
    return worst


def sharpe(pnl, block=100):
    """
    PnL is a running total. Chunk it into blocks of `block` steps, take the
    per block increments and report mean/std. NaN when the increments have
    zero variance (no fills, or a perfectly steady curve); evaluate() drops
    NaN runs from the average, so read n alongside it. Annualising is meaningless in
    simulated time so this is just a signal to noise ratio.
    """
    inc = [pnl[i] - pnl[i - block] for i in range(block, len(pnl), block)]
    if len(inc) < 2:
        return float("nan")
    sd = statistics.pstdev(inc)
    return statistics.mean(inc) / sd if sd > 0 else float("nan")


def risk_summary(blotter, cap=None):
    pnl = [c[2] for c in blotter.curve]
    pos = [c[1] for c in blotter.curve]
    s = blotter.summary()
    if cap is not None and cap > 0:
        s["time_at_cap"] = round(sum(1 for p in pos if abs(p) >= cap) / max(1, len(pos)), 3)
    s.update({
        "max_drawdown": round(max_drawdown(pnl), 1),
        "sharpe_100": round(sharpe(pnl), 2),
        "pnl_per_fill": round(pnl[-1] / len(blotter.fills), 2) if blotter.fills else 0.0,
        "pos_std": round(statistics.pstdev(pos), 1) if pos else 0.0,
    })
    return s


def evaluate(make_bot, seeds, steps=4000, **market_kw):
    """
    Run the same strategy over many seeds and report mean and std of every
    metric. make_bot is a zero arg callable so each
    run gets a fresh bot.
    """
    rows = []
    for seed in seeds:
        bot = make_bot()
        m = Market(seed=seed, bot=bot, **market_kw).run(steps)
        bot.blotter.liquidate(m.book)
        row = risk_summary(bot.blotter, cap=getattr(bot, "max_position", None))
        row["fair_edge"] = round(fair_edge(bot.blotter.fills, m.history), 1)
        rows.append(row)
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if not (isinstance(r[k], float) and math.isnan(r[k]))]
        if not vals:
            continue
        mean = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        nd = 3 if abs(mean) < 10 and k != "n_seeds" else 1
        out[k] = (round(mean, nd), round(sd, nd))
    out["n_seeds"] = (len(rows), 0)
    return out


def fmt_eval(res):
    def f(x):
        return f"{x:,.3f}" if abs(x) < 10 else f"{x:,.1f}"
    return "\n".join(f"{k:>20}: {f(m):>12}  ±{f(sd)}" for k, (m, sd) in res.items())


# +1: higher is better for the second strategy. -1: lower is better.
# metrics without a direction (fills, final_position) are refused.
DIRECTION = {
    "total_pnl": +1, "liq_pnl": +1, "spread_pnl": +1, "fair_edge": +1, "inventory_pnl": +1, "pnl_per_fill": +1,
    "sharpe_100": +1, "max_drawdown": +1,
    "pos_std": -1, "max_abs_position": -1, "avg_abs_position": -1,
    "time_at_cap": -1, "fees_paid": -1,
}


def paired_compare(make_a, make_b, seeds, steps=4000, metrics=("total_pnl", "max_drawdown", "pos_std"),
                   **market_kw):
    """
    Same seeds, same tape, two strategies. For each metric: mean and std of
    (b - a), a paired t statistic and how many seeds b is better, worse or
    tied, using the direction in DIRECTION. Seeds where either side is NaN
    are dropped and n says how many were used. Mean ± std across seeds ignores
    that the two runs share the tape, the paired difference doesn't.
    """
    for m in metrics:
        if m not in DIRECTION:
            raise ValueError(f"no better/worse direction defined for {m!r}")
    diffs = {m: [] for m in metrics}
    for seed in seeds:
        a, b = make_a(), make_b()
        ma = Market(seed=seed, bot=a, **market_kw).run(steps)
        mb = Market(seed=seed, bot=b, **market_kw).run(steps)
        a.blotter.liquidate(ma.book)
        b.blotter.liquidate(mb.book)
        ra = risk_summary(a.blotter, cap=getattr(a, "max_position", None))
        rb = risk_summary(b.blotter, cap=getattr(b, "max_position", None))
        ra["fair_edge"] = fair_edge(a.blotter.fills, ma.history)
        rb["fair_edge"] = fair_edge(b.blotter.fills, mb.history)
        for m in metrics:
            if m not in ra or m not in rb:
                raise KeyError(f"{m!r} missing, e.g. time_at_cap needs both bots to have a cap")
            x, y = ra[m], rb[m]
            if isinstance(x, float) and math.isnan(x) or isinstance(y, float) and math.isnan(y):
                continue
            diffs[m].append(y - x)
    out = {}
    for m, d in diffs.items():
        n = len(d)
        if n == 0:
            out[m] = {"n": 0}
            continue
        mean = statistics.mean(d)
        sd = statistics.stdev(d) if n > 1 else 0.0
        if sd > 0:
            t = mean / (sd / math.sqrt(n))
        else:
            t = 0.0 if mean == 0 else math.copysign(math.inf, mean)
        sgn = DIRECTION[m]
        better = sum(1 for x in d if x * sgn > 0)
        worse = sum(1 for x in d if x * sgn < 0)
        out[m] = {"n": n, "mean_diff": round(mean, 3 if abs(mean) < 10 else 1),
                  "sd_diff": round(sd, 3 if abs(sd) < 10 else 1), "t": round(t, 2),
                  "better": better, "worse": worse, "tied": n - better - worse,
                  "wins": f"{better}/{n}"}
    return out
