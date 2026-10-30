"""
Gamma sweep for the Avellaneda-Stoikov maker (fixed 3 tick half spread) and
GLFT, against the naive maker. More risk aversion buys less inventory variance
and lower drawdown. Whether it also buys more total pnl depends on the tape:
see scripts/paired.py for the paired tests.

python scripts/sweep_gamma.py        # 20 seeds, a few minutes
python scripts/sweep_gamma.py 5
"""
import sys

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
DOCS = Path(__file__).resolve().parents[1] / "docs"
DOCS.mkdir(exist_ok=True)
import matplotlib.pyplot as plt

from lobsim.analytics import evaluate
from lobsim.strategy import InventoryMaker, NaiveMaker

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEEDS = range(100, 100 + N)
GAMMAS = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
GLFT_GAMMAS = [0.02, 0.05, 0.1, 0.5, 1.0]
TAPES = {"flat": {}, "trend": {"drift": 0.4}}

rows = {}
for tape, mkw in TAPES.items():
    naive = evaluate(lambda: NaiveMaker(requote_always=False, max_position=30), SEEDS, **mkw)
    rows[tape] = {"naive": naive, "as": [], "glft": []}
    print(f"\n== {tape} ==")
    print(f"{'naive':>8}  total={naive['total_pnl'][0]:>8,.0f} ±{naive['total_pnl'][1]:>6,.0f}"
          f"  spread={naive['spread_pnl'][0]:>7,.0f}  inv={naive['inventory_pnl'][0]:>8,.0f}"
          f"  posstd={naive['pos_std'][0]:>5}  dd={naive['max_drawdown'][0]:>8,.0f}")
    for g in GAMMAS:
        r = evaluate(lambda: InventoryMaker(gamma=g, spread_mode="fixed",
                                            requote_always=False, max_position=30),
                     SEEDS, **mkw)
        rows[tape]["as"].append(r)
        print(f"as g={g:<6}  total={r['total_pnl'][0]:>8,.0f} ±{r['total_pnl'][1]:>6,.0f}"
              f"  spread={r['spread_pnl'][0]:>7,.0f}  inv={r['inventory_pnl'][0]:>8,.0f}"
              f"  posstd={r['pos_std'][0]:>5}  dd={r['max_drawdown'][0]:>8,.0f}")
    for g in GLFT_GAMMAS:
        r = evaluate(lambda: InventoryMaker(gamma=g, spread_mode="glft",
                                            requote_always=False, max_position=30),
                     SEEDS, **mkw)
        rows[tape]["glft"].append(r)
        print(f"glft g={g:<4}  total={r['total_pnl'][0]:>8,.0f} ±{r['total_pnl'][1]:>6,.0f}"
              f"  spread={r['spread_pnl'][0]:>7,.0f}  inv={r['inventory_pnl'][0]:>8,.0f}"
              f"  posstd={r['pos_std'][0]:>5}  dd={r['max_drawdown'][0]:>8,.0f}  fills={r['fills'][0]:>5,.0f}")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for tape, color in (("flat", "tab:blue"), ("trend", "tab:red")):
    rs = rows[tape]["as"]
    nv = rows[tape]["naive"]
    tot = [r["total_pnl"][0] for r in rs]
    err = [r["total_pnl"][1] for r in rs]
    spr = [r["spread_pnl"][0] for r in rs]
    pst = [r["pos_std"][0] for r in rs]
    dd = [r["max_drawdown"][0] for r in rs]

    ax[0].errorbar(GAMMAS, tot, yerr=err, marker="o", color=color, label=f"AS, {tape}", capsize=3)
    ax[0].axhline(nv["total_pnl"][0], color=color, ls="--", lw=1, label=f"naive, {tape}")

    ax[1].plot(pst, spr, marker="o", color=color, label=f"AS fixed spread, {tape}")
    gl = rows[tape]["glft"]
    ax[1].plot([r["pos_std"][0] for r in gl], [r["spread_pnl"][0] for r in gl], marker="^",
               color=color, ls=":", label=f"GLFT, {tape}")
    ax[1].plot([nv["pos_std"][0]], [nv["spread_pnl"][0]], marker="s", ms=9, color=color,
               ls="none", label=f"naive, {tape}")
    for g, x, y in zip(GAMMAS, pst, spr):
        ax[1].annotate(str(g), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    for g, r in zip(GLFT_GAMMAS, gl):
        ax[1].annotate(f"g{g}", (r["pos_std"][0], r["spread_pnl"][0]), textcoords="offset points",
                       xytext=(4, -10), fontsize=7, color=color, alpha=0.8)

    ax[2].plot(GAMMAS, dd, marker="o", color=color, label=f"AS, {tape}")
    ax[2].axhline(nv["max_drawdown"][0], color=color, ls="--", lw=1)

ax[0].set_xscale("log"); ax[0].set_xlabel("gamma"); ax[0].set_ylabel("total pnl")
ax[0].set_title("total pnl vs gamma (AS, 3 tick half spread), bars = ±1 sd over seeds"); ax[0].axhline(0, color="grey", lw=0.5); ax[0].legend(fontsize=8)
ax[1].set_xlabel("position std (lots)"); ax[1].set_ylabel("spread pnl")
ax[1].set_title("spread captured vs inventory carried (labels: gamma)"); ax[1].legend(fontsize=8)
ax[2].set_xscale("log"); ax[2].set_xlabel("gamma"); ax[2].set_ylabel("max drawdown")
ax[2].set_title("drawdown vs risk aversion (dashed = naive)"); ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(str(DOCS / "gamma_sweep.png"), dpi=120)
print("\nsaved docs/gamma_sweep.png")
