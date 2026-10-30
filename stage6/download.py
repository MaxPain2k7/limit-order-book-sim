"""
Download free Tardis.dev days for stage 6. Run this on your own machine.

Tardis makes the first day of every month downloadable without an API key:
https://docs.tardis.dev/downloadable-csv-files/overview

    python stage6/download.py --symbol SOLUSDT --months 2025-01 2025-12
    python stage6/download.py --symbol BTCUSDT --months 2025-01 2025-03 --trades-only

Files land in data/raw/SYMBOL_YYYYMMDD_trades.csv.gz and ..._book.csv.gz.

Size warning: the incremental_book_L2 file for one BTCUSDT futures day is
large (gigabytes). Start with --trades-only for the order flow check, which
needs only trades (tens of MB a day), or with a less liquid symbol.
"""
import argparse
import os
import sys
import time
import urllib.request

BASE = "https://datasets.tardis.dev/v1/{exchange}/{dtype}/{y}/{m}/01/{symbol}.csv.gz"


def months(start, end):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def fetch(url, path, retries=3):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"  have {path}")
        return True
    tmp = path + ".part"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done, t0 = 0, time.time()
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        sys.stdout.write(f"\r  {os.path.basename(path)} {done / 1e6:,.0f} / {total / 1e6:,.0f} MB"
                                         f"  {done / 1e6 / max(time.time() - t0, 1e-9):.1f} MB/s")
                        sys.stdout.flush()
            os.replace(tmp, path)
            print()
            return True
        except Exception as e:                      # noqa: BLE001  network errors of every kind
            print(f"\n  attempt {attempt} failed: {e}")
            time.sleep(2 * attempt)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--exchange", default="binance-futures")
    ap.add_argument("--months", nargs=2, metavar=("START", "END"), default=["2025-01", "2025-06"],
                    help="inclusive, YYYY-MM YYYY-MM; the first day of each month is fetched")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--trades-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    kinds = [("trades", "trades")] + ([] if a.trades_only else [("incremental_book_L2", "book")])
    ok = 0
    for y, m in months(*a.months):
        print(f"{a.symbol} {y}-{m:02d}-01")
        for dtype, tag in kinds:
            url = BASE.format(exchange=a.exchange, dtype=dtype, y=y, m=f"{m:02d}", symbol=a.symbol)
            path = os.path.join(a.out, f"{a.symbol}_{y}{m:02d}01_{tag}.csv.gz")
            ok += fetch(url, path)
    print(f"done, {ok} files")
    if ok == 0:
        sys.exit("nothing was downloaded. check your internet connection and that the symbol "
                 f"({a.symbol}) and months exist on Tardis for {a.exchange}.")


if __name__ == "__main__":
    main()
