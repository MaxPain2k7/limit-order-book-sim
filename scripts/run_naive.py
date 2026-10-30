"""Run the naive maker on a flat and a trending tape and chart what happens.

python scripts/run_naive.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
DOCS = Path(__file__).resolve().parents[1] / "docs"
DOCS.mkdir(exist_ok=True)
import matplotlib.pyplot as plt

from lobsim.analytics import markouts, risk_summary
from lobsim.market import Market
from lobsim.strategy import NaiveMaker

STEPS = 4000
SEED = 42


def run(drift, max_position=None):
    bot = NaiveMaker(half_spread=3, size=5, max_position=max_position, requote_always=False)
    m = Market(seed=SEED, drift=drift, bot=bot).run(STEPS)
    return m, bot


runs = {
    "flat": run(0.0),
    "trend": run(0.4),
    "trend, pos limit 30": run(0.4, max_position=30),
}

for name, (m, bot) in runs.items():
    print(f"\n== {name} ==")
    for k, v in risk_summary(bot.blotter).items():
        print(f"{k:>20}: {v}")
    hz, mo = markouts(bot.blotter.fills, m.history)
    print(f"{'markout vs informed':>20}: " + "  ".join(f"h={h} {v:+.1f}" for h, v in zip(hz, mo['informed'])))
    print(f"{'markout vs noise':>20}: " + "  ".join(f"h={h} {v:+.1f}" for h, v in zip(hz, mo['noise'])))

fig, ax = plt.subplots(3, 2, figsize=(14, 10))
for col, name in enumerate(["flat", "trend"]):
    m, bot = runs[name]
    c = bot.blotter.curve
    t = [x[0] for x in c]
    ax[0, col].plot(t, [x[2] for x in c], color="black", label="total")
    ax[0, col].plot(t, [x[3] for x in c], color="tab:green", label="spread")
    ax[0, col].plot(t, [x[4] for x in c], color="tab:red", label="inventory")
    ax[0, col].axhline(0, color="grey", lw=0.5)
    ax[0, col].set_title(f"naive maker, {name} market: pnl")
    ax[0, col].legend()

    ax[1, col].plot(t, [x[1] for x in c], color="tab:blue")
    ax[1, col].axhline(0, color="grey", lw=0.5)
    ax[1, col].set_title("position (lots)")

    hz, mo = markouts(bot.blotter.fills, m.history)
    ax[2, col].plot(hz, mo["informed"], marker="o", color="tab:red", label="fills vs informed")
    ax[2, col].plot(hz, mo["noise"], marker="o", color="tab:green", label="fills vs noise")
    ax[2, col].axhline(0, color="grey", lw=0.5)
    ax[2, col].set_title("markout: pnl per lot h steps after fill")
    ax[2, col].set_xlabel("h")
    ax[2, col].legend()

plt.tight_layout()
plt.savefig(str(DOCS / "naive_maker.png"), dpi=120)
print("\nsaved docs/naive_maker.png")
