"""
Multi seed evaluation of the naive maker under different conditions.

python scripts/evaluate.py            # 20 seeds, a few minutes
python scripts/evaluate.py 5          # quicker
"""
import sys
import time

from lobsim.analytics import evaluate, fmt_eval
from lobsim.strategy import InventoryMaker, NaiveMaker

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEEDS = range(100, 100 + N)
STEPS = 4000

# binance futures style: small maker rebate, bigger taker fee
FEES = dict(maker_fee=-0.00005, taker_fee=0.0007)

configs = [
    ("requote every step, no cap, flat",
     lambda: NaiveMaker(), {}),
    ("requote on change only, no cap, flat",
     lambda: NaiveMaker(requote_always=False), {}),
    ("requote on change, cap 30, flat",
     lambda: NaiveMaker(requote_always=False, max_position=30), {}),
    ("requote on change, cap 30, flat, latency 3",
     lambda: NaiveMaker(requote_always=False, max_position=30), {"bot_latency": 3}),
    ("requote on change, cap 30, flat, with fees",
     lambda: NaiveMaker(requote_always=False, max_position=30, **FEES), {}),
    ("requote on change, cap 30, trend 0.4",
     lambda: NaiveMaker(requote_always=False, max_position=30), {"drift": 0.4}),
    ("requote on change, cap 30, toxic flow x2",
     lambda: NaiveMaker(requote_always=False, max_position=30), {"informed_rate": 0.6}),
    ("AS g=0.002, fixed spread, cap 30, flat",
     lambda: InventoryMaker(gamma=0.002, spread_mode="fixed", requote_always=False, max_position=30), {}),
    ("AS g=0.002, fixed spread, cap 30, flat, latency 3",
     lambda: InventoryMaker(gamma=0.002, spread_mode="fixed", requote_always=False, max_position=30), {"bot_latency": 3}),
    ("AS g=0.002, fixed spread, cap 30, trend 0.4",
     lambda: InventoryMaker(gamma=0.002, spread_mode="fixed", requote_always=False, max_position=30), {"drift": 0.4}),
    ("GLFT g=0.05, cap 30, flat",
     lambda: InventoryMaker(gamma=0.05, spread_mode="glft", requote_always=False, max_position=30), {}),
]

t0 = time.time()
for name, make, mkw in configs:
    res = evaluate(make, SEEDS, steps=STEPS, **mkw)
    print(f"\n== {name} ==")
    print(fmt_eval(res))
print(f"\n{time.time() - t0:.0f}s")
