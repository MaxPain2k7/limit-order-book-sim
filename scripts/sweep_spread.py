"""
How wide should the naive maker quote? Half spread sweep, 20 seeds, flat and
trending tapes. The answer for a flat tape is "wider than 3". The answer for a
trend is that no width helps, which is the case for inventory skew.

python scripts/sweep_spread.py      # 20 seeds, a few minutes
"""
import sys

from lobsim.analytics import evaluate
from lobsim.strategy import NaiveMaker

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEEDS = range(100, 100 + N)

print(f"{'half':>5} | {'flat total':>16} {'fills':>6} | {'trend total':>16} {'fills':>6}")
for hs in (1, 2, 3, 5, 8, 12, 20, 50):
    row = []
    for kw in ({}, {"drift": 0.4}):
        r = evaluate(lambda: NaiveMaker(half_spread=hs, requote_always=False, max_position=30), SEEDS, **kw)
        row.append(f"{r['total_pnl'][0]:>8,.0f} ±{r['total_pnl'][1]:>6,.0f} {r['fills'][0]:>6,.0f}")
    print(f"{hs:>5} | {row[0]} | {row[1]}")
