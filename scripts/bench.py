"""Throughput benchmark. Run: python scripts/bench.py"""
import random
import time

from lobsim.orderbook import Order, OrderBook, OrderType, Side

N = 300_000
rng = random.Random(1)
book = OrderBook()

ops = []
for i in range(1, N + 1):
    r = rng.random()
    side = rng.choice([Side.BUY, Side.SELL])
    if r < 0.55:
        ops.append(("L", Order(i, side, rng.randint(9_950, 10_050), rng.randint(1, 50), i)))
    elif r < 0.70:
        ops.append(("M", Order(i, side, None, rng.randint(1, 80), i, OrderType.MARKET)))
    elif r < 0.80:
        ops.append(("I", Order(i, side, rng.randint(9_950, 10_050), rng.randint(1, 50), i, OrderType.IOC)))
    else:
        ops.append(("C", i))

live = []
t0 = time.perf_counter()
for kind, payload in ops:
    if kind == "C":
        if live:
            victim = live.pop(rng.randrange(len(live)))
            if book.get_order(victim) is not None:
                book.cancel_order(victim, timestamp=payload)
        continue
    book.submit(payload)
    if kind == "L" and book.get_order(payload.order_id) is not None:
        live.append(payload.order_id)
dt = time.perf_counter() - t0

print(f"{N:,} ops in {dt:.2f}s  ->  {N / dt:,.0f} ops/sec")
print(f"trades: {len(book.trades):,}   events: {len(book.events):,}   resting: {len(book):,}")
print(f"levels: {len(book.bids)} bid / {len(book.asks)} ask   spread: {book.spread}")
print(book)
