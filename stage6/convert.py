"""
Convert downloaded Tardis day files to hftbacktest .npz, one file per day.

    python stage6/convert.py --symbol SOLUSDT

The converter preallocates a row buffer (64 bytes a row). A fixed 100 million
row buffer is 6.4 GB of RAM, enough to crash a laptop, so the buffer is sized
from the actual line count of each day's files plus 10%. --buffer overrides.

Reads data/raw/SYMBOL_YYYYMMDD_trades.csv.gz + _book.csv.gz,
writes data/npz/SYMBOL_YYYYMMDD.npz. Trades go first in the input list, as the
hftbacktest docs require, so a depth update caused by a trade does not reduce
the estimated queue position twice.
"""
import argparse
import glob
import gzip
import os
import re

from hftbacktest.data.utils import tardis


def count_rows(path):
    n = 0
    with gzip.open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 24)
            if not chunk:
                return n
            n += chunk.count(b"\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/npz")
    ap.add_argument("--buffer", type=int, help="row buffer; default: sized from the files")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    books = sorted(glob.glob(os.path.join(a.raw, f"{a.symbol}_*_book.csv.gz")))
    if not books:
        raise SystemExit(f"no book files for {a.symbol} in {a.raw}; run stage6/download.py without --trades-only")
    for bf in books:
        day = re.search(r"_(\d{8})_book", bf).group(1)
        tf = bf.replace("_book.csv.gz", "_trades.csv.gz")
        out = os.path.join(a.out, f"{a.symbol}_{day}.npz")
        if os.path.exists(out):
            print(f"have {out}")
            continue
        if not os.path.exists(tf):
            print(f"skip {day}: no trades file")
            continue
        rows = a.buffer or int(1.1 * (count_rows(tf) + count_rows(bf))) + 10_000
        print(f"converting {day}: buffer {rows:,} rows, about {rows * 64 / 1e9:.1f} GB of RAM")
        tardis.convert([tf, bf], output_filename=out, buffer_size=rows)


if __name__ == "__main__":
    main()
