"""Naive vs AS vs defended (AS + trade flow skew) on one tape.

python scripts/run_defended.py          # flat
python scripts/run_defended.py trend
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lobsim.analytics import fair_edge, markouts, risk_summary
from lobsim.market import Market
from lobsim.strategy import DefensiveMaker, InventoryMaker, NaiveMaker

DOCS = Path(__file__).resolve().parents[1] / "docs"
DOCS.mkdir(exist_ok=True)

drift = 0.4 if "trend" in sys.argv else 0.0
SEED, STEPS = 1000, 4000          # a holdout seed, never used for tuning
BASE = dict(gamma=0.002, spread_mode="fixed", requote_always=False, max_position=30)


class Rec(DefensiveMaker):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.trace = []

    def on_tick(self, book, t):
        super().on_tick(book, t)
        self.trace.append((t, self.last_flow))


bots = {
    "naive": NaiveMaker(requote_always=False, max_position=30),
    "AS g=0.002": InventoryMaker(**BASE),
    "defended (AS + flow 6)": Rec(**BASE, flow_coef=6, flow_window=20),
}
markets = {n: Market(seed=SEED, drift=drift, bot=b).run(STEPS) for n, b in bots.items()}

for n, b in bots.items():
    m = markets[n]
    b.blotter.liquidate(m.book)
    s = risk_summary(b.blotter, cap=30)
    hz, mo = markouts(b.blotter.fills, m.history, horizons=(1, 5, 20, 50))
    print(f"\n== {n} ==")
    print(f"  pnl after liquidation {s['liq_pnl']:>9,.0f}   edge vs true price "
          f"{fair_edge(b.blotter.fills, m.history):>8,.0f}   drawdown {s['max_drawdown']:>8,.0f}"
          f"   pos std {s['pos_std']:>5}   fills {s['fills']}")
    print("  markout vs informed: " + "  ".join(f"h={h} {v:+.1f}" for h, v in zip(hz, mo["informed"])))

colors = {"naive": "tab:grey", "AS g=0.002": "tab:blue", "defended (AS + flow 6)": "tab:green"}
fig, ax = plt.subplots(2, 2, figsize=(14, 8))
for n, b in bots.items():
    c = b.blotter.curve
    t = [x[0] for x in c]
    ax[0, 0].plot(t, [x[2] for x in c], color=colors[n], label=n)
    ax[1, 0].plot(t, [x[1] for x in c], color=colors[n], label=n, lw=0.7)
ax[0, 0].set_title("total pnl (zero fees)"); ax[0, 0].axhline(0, color="grey", lw=0.5); ax[0, 0].legend()
ax[1, 0].set_title("position (lots)"); ax[1, 0].axhline(0, color="grey", lw=0.5); ax[1, 0].set_xlabel("step")

d = bots["defended (AS + flow 6)"]
tr = d.trace
ax[0, 1].plot([x[0] for x in tr], [x[1] for x in tr], color="tab:green", lw=0.6)
ax[0, 1].axhline(0, color="grey", lw=0.5)
ax[0, 1].set_title("net aggressive flow on the public tape, last 20 steps\n(+1 all buying, -1 all selling)")

names = list(bots)
for i, n in enumerate(names):
    m = markets[n]
    hz, mo = markouts(bots[n].blotter.fills, m.history, horizons=(1, 5, 20, 50))
    ax[1, 1].plot(hz, mo["informed"], marker="o", color=colors[n], label=n)
ax[1, 1].axhline(0, color="grey", lw=0.5)
ax[1, 1].set_title("markout of fills against informed traders (ticks per lot)")
ax[1, 1].set_xlabel("steps after fill"); ax[1, 1].legend()

plt.suptitle(f"stage 5, {'trending' if drift else 'flat'} tape, holdout seed {SEED}")
plt.tight_layout()
out = DOCS / ("defended_trend.png" if drift else "defended_flat.png")
plt.savefig(out, dpi=120)
print("\nsaved", out)
