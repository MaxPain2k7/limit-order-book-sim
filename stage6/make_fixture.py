"""
Write a lob-sim run out as Tardis.dev CSV files, so the real data pipeline can
be tested end to end without network access.

The schemas follow https://docs.tardis.dev/downloadable-csv-files/data-types:

  trades:              exchange,symbol,timestamp,local_timestamp,id,side,price,amount
  incremental_book_L2: exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount

timestamps are microseconds since epoch, trade side is the aggressor, book
amount is the new total at that level (0 removes it). The file starts with
an is_snapshot=true block like Tardis's start-of-day snapshot.

One simulator step is 100ms. Prices are ticks * TICK, sizes are lots * LOT.

    python stage6/make_fixture.py data/fixture 4000 0.0
"""
import csv
import gzip
import os
import sys

from lobsim.market import Market
from lobsim.orderbook import EventType, Side

TICK = 0.1
LOT = 0.001
STEP_US = 100_000
T0_US = 1_735_689_600_000_000      # 2025-01-01 00:00:00 UTC
FEED_LAT_US = 3_000                # exchange -> local
EXCHANGE, SYMBOL = "binance-futures", "SIMUSDT"


def fmt_px(ticks):
    return f"{ticks * TICK:.1f}"


def fmt_qty(lots):
    return f"{lots * LOT:.3f}"


def make(out_dir, steps=4000, drift=0.0, seed=1000, **market_kw):
    os.makedirs(out_dir, exist_ok=True)
    m = Market(seed=seed, drift=drift, **market_kw)
    book = m.book

    levels = {}            # (side, price) -> lots, as last written
    book_rows = []
    trade_rows = []
    counter = {"n": 0}

    def ts():
        # every event gets a unique, increasing exchange timestamp inside its step
        counter["n"] += 1
        return T0_US + m.t * STEP_US + counter["n"]

    def emit_level(side, price, t):
        lvl = (book.bids if side is Side.BUY else book.asks).get(price)
        now = lvl.total_qty if lvl is not None else 0
        key = (side, price)
        if levels.get(key, 0) != now:
            levels[key] = now
            book_rows.append([EXCHANGE, SYMBOL, t, t + FEED_LAT_US, "false",
                              "bid" if side is Side.BUY else "ask", fmt_px(price), fmt_qty(now)])
            if now == 0:
                del levels[key]

    # start of day snapshot of the seeded book
    t = T0_US
    for side, sbook in ((Side.BUY, book.bids), (Side.SELL, book.asks)):
        for price, lvl in sbook.items():
            levels[(side, price)] = lvl.total_qty
            book_rows.append([EXCHANGE, SYMBOL, t, t + FEED_LAT_US, "true",
                              "bid" if side is Side.BUY else "ask", fmt_px(price), fmt_qty(lvl.total_qty)])

    orig = book._on_event

    def hook(ev):
        orig(ev)
        if ev.type in (EventType.REJECT, EventType.TRIGGER):
            return
        t = ts()
        if ev.type is EventType.TRADE:
            tr = ev.trade
            trade_rows.append([EXCHANGE, SYMBOL, t, t + FEED_LAT_US, str(tr.trade_id),
                               "buy" if tr.aggressor_side is Side.BUY else "sell",
                               fmt_px(tr.price), fmt_qty(tr.qty)])
            maker_side = Side.SELL if tr.aggressor_side is Side.BUY else Side.BUY
            emit_level(maker_side, tr.price, t)
        elif ev.price is not None and ev.side is not None:
            emit_level(ev.side, ev.price, t)

    book._on_event = hook
    m.run(steps)

    tf = os.path.join(out_dir, f"{SYMBOL}_trades.csv.gz")
    bf = os.path.join(out_dir, f"{SYMBOL}_book.csv.gz")
    with gzip.open(tf, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exchange", "symbol", "timestamp", "local_timestamp", "id", "side", "price", "amount"])
        w.writerows(trade_rows)
    with gzip.open(bf, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exchange", "symbol", "timestamp", "local_timestamp", "is_snapshot", "side", "price", "amount"])
        w.writerows(book_rows)
    return tf, bf, m


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/fixture"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    drift = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    tf, bf, m = make(out, steps, drift)
    print(f"wrote {tf} and {bf}: {len(m.book.trades)} trades, {m.book._seq} book events")


def make_days(raw_dir, n_days=4, steps=20_000, drift=0.0, first_seed=2000):
    """Write n_days fixture days in the layout download.py produces:
    raw_dir/SIMUSDT_2025MM01_trades.csv.gz and _book.csv.gz"""
    os.makedirs(raw_dir, exist_ok=True)
    out = []
    for i in range(n_days):
        tmp = os.path.join(raw_dir, f"_tmp{i}")
        tf, bf, _ = make(tmp, steps, drift, seed=first_seed + i)
        day = f"2025{i + 1:02d}01"
        nt = os.path.join(raw_dir, f"{SYMBOL}_{day}_trades.csv.gz")
        nb = os.path.join(raw_dir, f"{SYMBOL}_{day}_book.csv.gz")
        os.replace(tf, nt)
        os.replace(bf, nb)
        os.rmdir(tmp)
        out.append(day)
    return out
