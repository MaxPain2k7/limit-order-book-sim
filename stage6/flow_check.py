"""
Does order flow predict price on a real exchange?

Stage 5's best defence leans quotes with net aggressive flow on the tape. In
the simulator that works because informed traders hit the stale side. This
measures the same thing on real data, per day, with days as independent
replications:

  1. trade sign autocorrelation at lags 1, 10, 100, 1000 trades
     (real markets: long memory, Lillo and Farmer)
  2. net flow over the last W seconds vs price change over the next H seconds,
     sampled on non-overlapping forward windows
  3. the same-window (contemporaneous) relation, the one Cont, Kukanov and
     Stoikov find strong, as a reference point

    python stage6/flow_check.py --symbol SOLUSDT                 # from data/raw trades files
    python stage6/flow_check.py --symbol SOLUSDT --npz           # from converted days, exact mid

Trades-only mode uses the VWAP of each 1 second bucket as the price. That
overstates predictability: trades in the next second tend to continue on the
same side (trade signs are autocorrelated) and execute on that side of the
spread, so the "price change" partly measures trade sign persistence rather
than a move in the mid. Tested on simulated days where both are available:
trades-only gave +0.11 for W10/H1 against +0.04 from the true mid. Treat
trades-only as a cheap screen; --npz on converted days is the real answer.
"""
import argparse
import glob
import gzip
import math
import os
import re

import numpy as np

LAGS = (1, 10, 100, 1000)
WINDOWS = ((10, 1), (10, 10), (60, 10), (60, 60))     # (lookback W, horizon H) in seconds


def load_trades(path):
    try:
        import pandas as pd
        df = pd.read_csv(path, usecols=["timestamp", "side", "price", "amount"])
        df = df[df["side"].isin(["buy", "sell"])]
        return (df["timestamp"].to_numpy(np.int64), np.where(df["side"].to_numpy() == "buy", 1, -1),
                df["price"].to_numpy(float), df["amount"].to_numpy(float))
    except ImportError:
        import csv
        ts, sg, px, q = [], [], [], []
        with gzip.open(path, "rt") as f:
            for r in csv.DictReader(f):
                if r["side"] not in ("buy", "sell"):
                    continue
                ts.append(int(r["timestamp"])); sg.append(1 if r["side"] == "buy" else -1)
                px.append(float(r["price"])); q.append(float(r["amount"]))
        return np.array(ts), np.array(sg), np.array(px), np.array(q)


def acf(x, lag):
    x = x - x.mean()
    v = (x * x).sum()
    return float((x[lag:] * x[:-lag]).sum() / v) if v > 0 and len(x) > lag else float("nan")


def _ffill(x):
    ok = np.isfinite(x)
    idx = np.where(ok, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    return x[idx]


def buckets_from_trades(ts, sg, px, q, sec=1.0, price="midproxy"):
    """
    Per-second aggressive buy volume, sell volume and a price, from trades only.

    price="midproxy": average of the last buyer-initiated trade price (which
      sits at the ask) and the last seller-initiated one (at the bid), each
      carried forward. A stand-in for the mid that removes most of the bid-ask
      bounce and trade-sign persistence that contaminate trade prices.
    price="vwap": each second's trade VWAP. Overstates predictability, kept
      only to show how much.
    """
    b = ((ts - ts[0]) // int(sec * 1_000_000)).astype(np.int64)
    n = int(b[-1]) + 1
    buy = np.bincount(b, weights=q * (sg > 0), minlength=n)
    sell = np.bincount(b, weights=q * (sg < 0), minlength=n)
    if price == "vwap":
        vol = buy + sell
        pv = np.bincount(b, weights=px * q, minlength=n)
        p = np.full(n, np.nan)
        has = vol > 0
        p[has] = pv[has] / vol[has]
        return buy, sell, _ffill(p)
    last_buy = np.full(n, np.nan)
    last_sell = np.full(n, np.nan)
    # trades are time ordered, so assigning in order leaves the last one per bucket
    mb, ms = sg > 0, sg < 0
    last_buy[b[mb]] = px[mb]
    last_sell[b[ms]] = px[ms]
    lb, ls = _ffill(last_buy), _ffill(last_sell)
    return buy, sell, (lb + ls) / 2.0


def buckets_from_npz(npz, tick, lot, sec=1.0):
    from stage6.bots import observe
    from stage6.common import make_backtest
    hbt = make_backtest(npz, tick, lot)
    out = np.zeros((200_000, 7))
    n = observe(hbt, out, int(sec * 1e9))
    hbt.close()
    o = out[:n]
    mid = o[:, 1]
    ok = np.isfinite(mid)
    idx = np.where(ok, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    return o[:, 4], o[:, 5], mid[idx]


def predictive(buy, sell, price, W, H):
    """corr(normalised net flow over [t-W, t), price change over [t, t+H)),
    t stepping by H so forward windows do not overlap."""
    cb, cs = np.concatenate([[0], np.cumsum(buy)]), np.concatenate([[0], np.cumsum(sell)])
    xs, ys, cx, cy = [], [], [], []
    for t in range(W, len(price) - H, H):
        b, s = cb[t] - cb[t - W], cs[t] - cs[t - W]
        if b + s <= 0 or not (np.isfinite(price[t - 1]) and np.isfinite(price[t + H - 1])):
            continue
        xs.append((b - s) / (b + s))
        ys.append(price[t + H - 1] - price[t - 1])
        # contemporaneous: flow and price change over the same window
        bc, sc = cb[t + H] - cb[t], cs[t + H] - cs[t]
        if bc + sc > 0:
            cx.append((bc - sc) / (bc + sc))
            cy.append(price[t + H - 1] - price[t - 1])
    def corr(a, b):
        a, b = np.asarray(a), np.asarray(b)
        if len(a) < 30 or a.std() == 0 or b.std() == 0:
            return float("nan"), 0
        return float(np.corrcoef(a, b)[0, 1]), len(a)
    return corr(xs, ys), corr(cx, cy)


def summarise(name, vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    if len(v) == 0:
        return f"{name}: no data"
    t = v.mean() / (v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 and v.std(ddof=1) > 0 else float("nan")
    return f"{name}: mean {v.mean():+.3f}  across {len(v)} days  t {t:+.1f}  (min {v.min():+.3f}, max {v.max():+.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--npzdir", default="data/npz")
    ap.add_argument("--npz", action="store_true", help="use converted days and the true mid")
    ap.add_argument("--price", choices=["midproxy", "vwap"], default="midproxy",
                    help="trades-only price: midproxy (default) or vwap (biased, for comparison)")
    ap.add_argument("--tick", type=float)
    ap.add_argument("--lot", type=float)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.raw, f"{a.symbol}_*_trades.csv.gz")))
    if not files:
        raise SystemExit(f"no trades files for {a.symbol} in {a.raw}; run stage6/download.py first")

    sign_acf = {l: [] for l in LAGS}
    pred = {w: [] for w in WINDOWS}
    cont = {w: [] for w in WINDOWS}
    for f in files:
        day = re.search(r"_(\d{8})_trades", f).group(1)
        ts, sg, px, q = load_trades(f)
        for l in LAGS:
            sign_acf[l].append(acf(sg.astype(float), l))
        if a.npz:
            from stage6.common import LOT, TICK
            npz = os.path.join(a.npzdir, f"{a.symbol}_{day}.npz")
            if not os.path.exists(npz):
                print(f"{day}: no {npz}, skipping")
                continue
            buy, sell, price = buckets_from_npz(npz, a.tick or TICK.get(a.symbol), a.lot or LOT.get(a.symbol))
        else:
            buy, sell, price = buckets_from_trades(ts, sg, px, q, price=a.price)
        line = [f"{day}: {len(ts):,} trades, sign acf(1) {sign_acf[1][-1]:+.3f}"]
        for w in WINDOWS:
            (c, n), (cc, _) = predictive(buy, sell, price, *w)
            pred[w].append(c)
            cont[w].append(cc)
            line.append(f"W{w[0]}/H{w[1]} pred {c:+.3f} cont {cc:+.3f}")
        print("  ".join(line))

    print(f"\n== {a.symbol}, {len(files)} days, price from {'true mid' if a.npz else ('mid proxy from last buy/sell trades' if a.price == 'midproxy' else 'trade vwap (biased up, comparison only)')} ==")
    print("trade sign autocorrelation (real markets: long memory, positive at long lags)")
    for l in LAGS:
        print("  " + summarise(f"lag {l:>4}", sign_acf[l]))
    print("net flow over the last W seconds vs price change over the next H seconds (predictive)")
    for w in WINDOWS:
        print("  " + summarise(f"W {w[0]:>2}s H {w[1]:>2}s", pred[w]))
    print("net flow vs price change over the same window (contemporaneous, Cont et al. reference)")
    seen = set()
    for w in WINDOWS:
        if w[1] in seen:
            continue
        seen.add(w[1])
        print("  " + summarise(f"   H {w[1]:>2}s", cont[w]))
    print("\nlob-sim for comparison: sign acf lag 1 +0.44, lag 10 +0.03, lag 100 -0.01;"
          " flow 20 steps -> next 20 steps +0.10")


if __name__ == "__main__":
    main()
