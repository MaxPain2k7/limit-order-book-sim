# Start here

Everything for lob-sim is in this folder: the code, the tests, the charts
and the full write-up in README.md.

## Run it

You need Python 3.11 or newer (python.org). Open a terminal in this folder,
then:

```
python run_all.py
```

That installs everything, runs all the tests, makes the charts in `docs/`,
downloads one month-start day of real SOLUSDT trades per month for 2025 (small),
and runs the order flow check on them.

Other options:

```
python run_all.py --sim-only     # no internet: setup, tests, simulator results only
python run_all.py --full         # also full order books + the bots on real data (gigabytes)
python run_all.py --symbol ETHUSDT --months 2025-01 2025-06
```

On Windows, if `python` is not found, use `py run_all.py`.

## Put it on GitHub

```
git init
git add .
git commit -m "lob-sim: order book engine and market making research"
git branch -M main
git remote add origin https://github.com/MaxPain2k7/lob-sim.git
git push -u origin main
```

Create the empty `lob-sim` repo on github.com first. Downloaded data in
`data/` is ignored by git, so it will not be pushed.

## What is where

```
README.md        the project write-up, results and every check
lobsim/          engine, simulator, strategies, analytics
stage6/          real data: download, convert, flow check, bots on hftbacktest
scripts/         experiments that produce the README numbers and charts
tests/           153 tests
docs/            charts
run_all.py       everything in one command
```
