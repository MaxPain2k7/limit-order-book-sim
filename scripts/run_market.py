"""Run the simulator with no bot and plot what the market looks like.

python scripts/run_market.py            # flat market
python scripts/run_market.py trend      # upward drift
"""
import sys

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
DOCS = Path(__file__).resolve().parents[1] / "docs"
DOCS.mkdir(exist_ok=True)
import matplotlib.pyplot as plt

from lobsim.market import Market

drift = 0.4 if "trend" in sys.argv else 0.0
m = Market(seed=42, drift=drift).run(4000)

for k, v in m.summary().items():
    print(f"{k:>20}: {v}")

t = [h[0] for h in m.history]
true = [h[1] for h in m.history]
def ffill(col):
    # one side of the book is empty for a step now and then, carry the last value
    out, last = [], None
    for h in m.history:
        if h[col] is not None:
            last = h[col]
        out.append(last)
    return out

mid, bid, ask = ffill(2), ffill(3), ffill(4)

fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                       gridspec_kw={"height_ratios": [3, 1]})
ax[0].plot(t, true, lw=1, label="true price", color="black")
ax[0].plot(t, mid, lw=1, label="mid", color="tab:blue")
ax[0].fill_between(t, bid, ask, alpha=0.25, color="tab:blue", label="bid/ask")
ax[0].legend(loc="upper left")
ax[0].set_ylabel("ticks")
ax[0].set_title("simulated market" + (" with drift" if drift else ""))

err = [mm - tt for mm, tt in zip(mid, true)]
ax[1].plot(t, err, lw=0.8, color="tab:red")
ax[1].axhline(0, color="black", lw=0.5)
ax[1].set_ylabel("mid - true")
ax[1].set_xlabel("step")

name = str(DOCS / "market_trend.png") if drift else str(DOCS / "market_flat.png")
plt.tight_layout()
plt.savefig(name, dpi=120)
print("saved", name)
