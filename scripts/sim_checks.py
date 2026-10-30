"""
Checks on the simulator itself, the things a market making backtest can get
wrong without any bug in the strategy code:

  1. stylized facts of the simulated mid: fat tails, bid-ask bounce,
     volatility clustering, against what real markets show
  2. spread pnl measured against the mid vs against the true price. the sim
     knows the truth, so we can check the attribution is not flattering
  3. self impact: how often the bot's own quote sets the touch (so also the
     mid that 70% of noise traders anchor on
  4. fills on orders whose cancel was already in flight (only exist with
     latency; a backtest that ignores cancel latency deletes them)
  5. what marking leftover inventory at mid hides

python scripts/sim_checks.py
"""
import statistics

from lobsim.market import BOT_PID, Market
from lobsim.orderbook import Side
from lobsim.strategy import InventoryMaker, NaiveMaker

SEEDS = range(1000, 1010)
STEPS = 4000


def acf(xs, lag):
    mu = statistics.mean(xs)
    v = sum((x - mu) ** 2 for x in xs)
    return sum((xs[i] - mu) * (xs[i - lag] - mu) for i in range(lag, len(xs))) / v


def kurt(xs):
    mu = statistics.mean(xs)
    sd = statistics.pstdev(xs)
    return sum((x - mu) ** 4 for x in xs) / len(xs) / sd ** 4


def mid_returns(m):
    mids, last = [], None
    for _, _, mid, _, _ in m.history:
        last = mid if mid is not None else last
        mids.append(last)
    return [b - a for a, b in zip(mids, mids[1:]) if a is not None and b is not None]


print("== 1. stylized facts of the simulated mid (seed-averaged, 10 seeds) ==")
print("   real markets: kurtosis >> 3, negative lag-1 return autocorrelation (bid-ask bounce),")
print("   |return| autocorrelation positive and slowly decaying (volatility clustering)\n")
for name, kw in (("default (gaussian + jumps)", {}),
                 ("no jumps", {"jump_prob": 0}),
                 ("fat tails df 4, no jumps", {"jump_prob": 0, "tail_df": 4}),
                 ("vol regimes, no jumps", {"jump_prob": 0, "vol_regime": (3, 0.01)}),
                 ("fat tails + vol regimes + jumps", {"tail_df": 4, "vol_regime": (3, 0.01)})):
    k, a1, aa1, aa20 = [], [], [], []
    for s in SEEDS:
        r = mid_returns(Market(seed=s, **kw).run(STEPS))
        ar = [abs(x) for x in r]
        k.append(kurt(r)); a1.append(acf(r, 1)); aa1.append(acf(ar, 1)); aa20.append(acf(ar, 20))
    print(f"   {name:34} kurtosis {statistics.mean(k):6.1f}   acf r(1) {statistics.mean(a1):+.3f}"
          f"   acf |r|(1) {statistics.mean(aa1):+.3f}   acf |r|(20) {statistics.mean(aa20):+.3f}")


def run_bot(make, **kw):
    out = []
    for s in SEEDS:
        b = make()
        m = Market(seed=s, bot=b, **kw)
        own_touch = 0
        for _ in range(STEPS):
            m.step()
            bk = m.book
            bb = bk.bids.peekitem(0)[1].head if bk.bids else None
            ba = bk.asks.peekitem(0)[1].head if bk.asks else None
            if (bb is not None and m.pids.get(bb.order_id) == BOT_PID) or \
               (ba is not None and m.pids.get(ba.order_id) == BOT_PID):
                own_touch += 1
        true_at = {t: tr for t, tr, *_ in m.history}
        vs_mid = sum((mid - px) * q if sd is Side.BUY else (px - mid) * q
                     for _, sd, px, q, _, mid in b.blotter.fills)
        vs_true = sum((true_at[t] - px) * q if sd is Side.BUY else (px - true_at[t]) * q
                      for t, sd, px, q, _, _ in b.blotter.fills)
        tot = b.blotter.curve[-1][2]
        liq = b.blotter.liquidate(m.book)
        out.append({"vs_mid": vs_mid, "vs_true": vs_true, "own_touch": own_touch / STEPS,
                    "total": tot, "liq": liq, "end_pos": abs(b.blotter.position),
                    "inf_share": b.blotter.n_informed / max(1, len(b.blotter.fills))})
    return {k: statistics.mean(r[k] for r in out) for k in out[0]}


print("\n== 2, 3, 5. attribution vs truth, self impact, leftover inventory (10 seeds) ==")
for name, make, kw in (
        ("naive hs3, flat", lambda: NaiveMaker(requote_always=False, max_position=30), {}),
        ("naive hs8, flat", lambda: NaiveMaker(half_spread=8, requote_always=False, max_position=30), {}),
        ("AS g=0.002 hs3, flat", lambda: InventoryMaker(gamma=0.002, spread_mode="fixed",
                                                        requote_always=False, max_position=30), {}),
        ("naive hs3, trend", lambda: NaiveMaker(requote_always=False, max_position=30), {"drift": 0.4}),
        ("AS g=0.002 hs3, trend", lambda: InventoryMaker(gamma=0.002, spread_mode="fixed",
                                                         requote_always=False, max_position=30), {"drift": 0.4})):
    r = run_bot(make, **kw)
    print(f"   {name:24} spread pnl vs mid {r['vs_mid']:>8,.0f}  vs true price {r['vs_true']:>8,.0f}"
          f" | own quote at touch {r['own_touch']:.0%} | end |pos| {r['end_pos']:4.1f}"
          f"  mark-to-mid {r['total']:>8,.0f}  after liquidation {r['liq']:>8,.0f}")

print("\n== 4. fills while a cancel was in flight (latency 3, flat, 10 seeds) ==")
tot_fills, in_flight = 0, 0
for s in SEEDS:
    b = NaiveMaker(requote_always=False, max_position=30)
    m = Market(seed=s, bot=b, bot_latency=3)
    cancel_sent = {}
    orig_cancel = m.cancel

    def tracking_cancel(oid, _m=m, _c=cancel_sent, _orig=orig_cancel):
        _c.setdefault(oid, _m.t)
        _orig(oid)
    m.gw.cancel = tracking_cancel        # the bot cancels through its gateway
    m.run(STEPS)
    for tr in m.book.trades:
        if m.pids.get(tr.maker_order_id) == BOT_PID:
            tot_fills += 1
            if tr.maker_order_id in cancel_sent and tr.timestamp >= cancel_sent[tr.maker_order_id]:
                in_flight += 1
print(f"   {in_flight} of {tot_fills} bot fills ({in_flight / max(1, tot_fills):.1%}) landed after the bot had"
      f" already sent the cancel")

print("\n== 6. order flow realism (10 seeds, flat) ==")
print("   real markets: trade signs have long memory (positive autocorrelation decaying as a")
print("   power law, still clearly positive after hundreds of trades, Lillo and Farmer)\n")
a1, a10, a100, pred = [], [], [], []
for s in SEEDS:
    m = Market(seed=s).run(STEPS)
    signs = [1 if tr.aggressor_side is Side.BUY else -1 for tr in m.book.trades]
    a1.append(acf(signs, 1)); a10.append(acf(signs, 10)); a100.append(acf(signs, 100))
    mids = {t: mid for t, _, mid, _, _ in m.history}
    flow = {}
    for tr in m.book.trades:
        flow[tr.timestamp] = flow.get(tr.timestamp, 0) + (tr.qty if tr.aggressor_side is Side.BUY else -tr.qty)
    xs, ys = [], []
    for t in range(40, STEPS - 40, 20):
        a, b = mids.get(t), mids.get(t + 20)
        if a is not None and b is not None:
            xs.append(sum(flow.get(u, 0) for u in range(t - 20, t)))
            ys.append(b - a)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    pred.append(sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                / (statistics.pstdev(xs) * statistics.pstdev(ys) * len(xs)))
print(f"   trade sign autocorrelation: lag 1 {statistics.mean(a1):+.3f}   lag 10 {statistics.mean(a10):+.3f}"
      f"   lag 100 {statistics.mean(a100):+.3f}")
print(f"   corr(net flow over last 20 steps, mid change over next 20): {statistics.mean(pred):+.3f}")
