"""
Paired comparison on identical tapes: naive vs Avellaneda-Stoikov, at the
3 tick half spread used in the sweep and at the naive bot's best half spread.

python scripts/paired.py        # 20 seeds, a few minutes
"""
import sys

from lobsim.analytics import paired_compare
from lobsim.strategy import InventoryMaker, NaiveMaker

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEEDS = range(100, 100 + N)
METRICS = ("total_pnl", "max_drawdown", "pos_std", "time_at_cap")


def naive(hs):
    return lambda: NaiveMaker(half_spread=hs, requote_always=False, max_position=30)


def as_(g, hs=3):
    return lambda: InventoryMaker(half_spread=hs, gamma=g, spread_mode="fixed",
                                  requote_always=False, max_position=30)


def show(r):
    for m, x in r.items():
        print(f"{m:>14}: mean diff {x['mean_diff']:>9,}  sd {x['sd_diff']:>9,}  t = {x['t']:>7}"
              f"  AS better {x['better']:>2}, worse {x['worse']:>2}, tied {x['tied']:>2}")


for hs in (3, 8):
    for tape, kw in (("flat", {}), ("trend", {"drift": 0.4})):
        print(f"\n== {tape}, half spread {hs}: AS g=0.002 minus naive, {N} seeds ==")
        show(paired_compare(naive(hs), as_(0.002, hs), SEEDS, metrics=METRICS, **kw))

for tape, kw in (("flat", {}), ("trend", {"drift": 0.4})):
    print(f"\n== {tape}, half spread 3: ordering inside the gamma band ==")
    for lo, hi in ((0.001, 0.002), (0.002, 0.005)):
        x = paired_compare(as_(lo), as_(hi), SEEDS, metrics=("total_pnl",), **kw)["total_pnl"]
        print(f"  g={hi} - g={lo}: mean diff {x['mean_diff']:>8,}  t = {x['t']:>6}"
              f"  higher g better {x['better']}/{x['n']}")
