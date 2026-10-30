"""
One command for everything. Works on Windows, macOS and Linux.

    python run_all.py              # setup, tests, stage 6 quick look on real trades (small download)
    python run_all.py --full       # also downloads full order books and runs the bots (large download)
    python run_all.py --sim-only   # setup, tests and the simulator results, no internet needed

Needs Python 3.11 or newer for stage 6 (hftbacktest). 3.10 runs everything else.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def step(title, cmd):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)
    r = subprocess.run([sys.executable] + cmd, cwd=HERE)
    if r.returncode != 0:
        print(f"\n!! step failed: {title} (exit {r.returncode}). Fix this before going further.")
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="download L2 books and run the bots (gigabytes)")
    ap.add_argument("--sim-only", action="store_true", help="skip everything that needs the internet")
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--months", nargs=2, default=["2025-01", "2025-12"])
    a = ap.parse_args()

    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}")
    if v < (3, 10):
        sys.exit("Python 3.10 or newer is needed. Install it from python.org and run again.")
    real = v >= (3, 11) and not a.sim_only
    if not real and not a.sim_only:
        print("Python is older than 3.11, so stage 6 (real data) will be skipped. Everything else runs.")

    extras = "dev,real" if real else "dev"
    step("1. install lob-sim and its dependencies", ["-m", "pip", "install", "-e", f".[{extras}]"])
    step("2. run the test suite", ["-m", "pytest", "-q"])
    step("3. engine throughput", ["scripts/bench.py"])
    step("4. simulator charts into docs/", ["scripts/run_market.py"])
    step("   naive market maker", ["scripts/run_naive.py"])
    step("   Avellaneda-Stoikov vs naive", ["scripts/run_as.py", "trend"])
    step("   stage 5 defended bot", ["scripts/run_defended.py", "trend"])

    if not real:
        print("\nDone. Open README.md for the write-up and docs/ for the charts.")
        return

    sym, (m0, m1) = a.symbol, a.months
    step(f"5. download {sym} trades, first day of each month {m0} to {m1} (small)",
         ["stage6/download.py", "--symbol", sym, "--months", m0, m1, "--trades-only"])
    step("6. order flow check on real trades (quick screen)",
         ["stage6/flow_check.py", "--symbol", sym])

    if not a.full:
        print("\nDone with the quick look.")
        print("For the full stage 6 (order books + bots, gigabytes): python run_all.py --full")
        return

    step(f"7. download {sym} full order books (large, can take a while)",
         ["stage6/download.py", "--symbol", sym, "--months", m0, m1])
    step("8. convert to hftbacktest format", ["stage6/convert.py", "--symbol", sym])
    step("9. order flow check with the true mid", ["stage6/flow_check.py", "--symbol", sym, "--npz"])
    step("10. market makers on real data, train then holdout", ["stage6/run.py", "--symbol", sym, "--quick"])
    print("\nDone. Steps 9 and 10 are the real data results.")


if __name__ == "__main__":
    main()
