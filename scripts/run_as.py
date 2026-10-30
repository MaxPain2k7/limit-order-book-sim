"""Naive vs Avellaneda-Stoikov on one tape, plus what the online estimators did.

python scripts/run_as.py          # flat
python scripts/run_as.py trend
"""
import sys

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
DOCS = Path(__file__).resolve().parents[1] / "docs"
DOCS.mkdir(exist_ok=True)
import matplotlib.pyplot as plt

from lobsim.analytics import markouts, risk_summary
from lobsim.market import Market
from lobsim.strategy import InventoryMaker, NaiveMaker

drift = 0.4 if "trend" in sys.argv else 0.0
SEED, STEPS = 42, 4000


class Recording(InventoryMaker):
    """Same bot, keeps a trace of its internals for the chart."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.trace = []

    def on_tick(self, book, t):
        super().on_tick(book, t)
        if self.sigma is not None:
            self.trace.append((t, self.sigma, self.kappa, self.skew, self.half))


naive = NaiveMaker(requote_always=False, max_position=30)
asb = Recording(gamma=0.002, spread_mode="fixed", requote_always=False, max_position=30)
m1 = Market(seed=SEED, drift=drift, bot=naive).run(STEPS)
m2 = Market(seed=SEED, drift=drift, bot=asb).run(STEPS)
glft = Recording(gamma=0.1, spread_mode="glft", requote_always=False, max_position=30)
Market(seed=SEED, drift=drift, bot=glft).run(STEPS)     # same tape, for the kappa panel

for name, bot, m in (("naive", naive, m1), ("AS g=0.002", asb, m2)):
    print(f"\n== {name} ==")
    for k, v in risk_summary(bot.blotter, cap=30).items():
        print(f"{k:>20}: {v}")
    hz, mo = markouts(bot.blotter.fills, m.history)
    print(f"{'markout informed':>20}: " + "  ".join(f"{h}:{v:+.1f}" for h, v in zip(hz, mo["informed"])))
    print(f"{'markout noise':>20}: " + "  ".join(f"{h}:{v:+.1f}" for h, v in zip(hz, mo["noise"])))

fig, ax = plt.subplots(3, 2, figsize=(14, 10))
for bot, color, label in ((naive, "tab:grey", "naive"), (asb, "tab:blue", "AS g=0.002")):
    c = bot.blotter.curve
    t = [x[0] for x in c]
    ax[0, 0].plot(t, [x[2] for x in c], color=color, label=label)
    ax[1, 0].plot(t, [x[1] for x in c], color=color, label=label, lw=0.8)
    ax[0, 1].plot(t, [x[4] for x in c], color=color, label=label)
ax[0, 0].set_title("total pnl"); ax[0, 0].axhline(0, color="grey", lw=0.5); ax[0, 0].legend()
ax[1, 0].set_title("position (lots)"); ax[1, 0].axhline(0, color="grey", lw=0.5)
ax[1, 0].axhline(30, color="red", lw=0.5, ls="--"); ax[1, 0].axhline(-30, color="red", lw=0.5, ls="--")
ax[0, 1].set_title("inventory pnl"); ax[0, 1].axhline(0, color="grey", lw=0.5); ax[0, 1].legend()

tr = asb.trace
tt = [x[0] for x in tr]
ax[1, 1].plot(tt, [x[3] for x in tr], color="tab:purple", lw=0.8)
ax[1, 1].axhline(0, color="grey", lw=0.5)
ax[1, 1].set_title("AS skew applied (ticks), positive = leaning to sell")
ax[2, 0].plot(tt, [x[1] for x in tr], color="tab:orange")
ax[2, 0].set_title("sigma estimate (ticks/step, robust)"); ax[2, 0].set_xlabel("step")
gt = [x for x in glft.trace if glft.A is not None]
ax[2, 1].plot([x[0] for x in glft.trace], [x[2] for x in glft.trace], color="tab:green")
ax[2, 1].set_title("kappa fitted from the tape by the GLFT bot, same tape\n(step 100, then every 500; fixed-spread AS does not use it)"); ax[2, 1].set_xlabel("step")

plt.suptitle(f"naive vs Avellaneda-Stoikov, {'trending' if drift else 'flat'} tape, seed {SEED}")
plt.tight_layout()
name = str(DOCS / "as_trend.png") if drift else str(DOCS / "as_flat.png")
plt.savefig(name, dpi=120)
print("\nsaved", name)
