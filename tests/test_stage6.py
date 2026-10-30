"""Stage 6 pipeline tests. Skipped unless the 'real' extras are installed:
pip install -e ".[dev,real]"
"""
import csv
import gzip
import math

import numpy as np
import pytest

pytest.importorskip("hftbacktest")
pytest.importorskip("scipy")

from stage6 import make_fixture  # noqa: E402


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    d = tmp_path_factory.mktemp("fx")
    tf, bf, m = make_fixture.make(str(d), steps=1500, seed=3000)
    return d, tf, bf, m


@pytest.fixture(scope="module")
def npz(fx):
    from hftbacktest.data.utils import tardis
    d, tf, bf, _ = fx
    out = str(d / "SIMUSDT_20250101.npz")
    tardis.convert([tf, bf], output_filename=out)
    return out


def test_fixture_follows_tardis_schema(fx):
    _, tf, bf, _ = fx
    with gzip.open(tf, "rt") as f:
        assert f.readline().strip() == "exchange,symbol,timestamp,local_timestamp,id,side,price,amount"
    with gzip.open(bf, "rt") as f:
        assert f.readline().strip() == "exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount"
        first = next(csv.reader(f))
        assert first[4] == "true"                       # starts with a snapshot block
    with gzip.open(tf, "rt") as f:
        sides = {r["side"] for r in csv.DictReader(f)}
    assert sides <= {"buy", "sell"}


def test_fixture_rebuilds_the_simulator_book(fx):
    _, tf, bf, m = fx
    T, L = make_fixture.TICK, make_fixture.LOT
    bids, asks = {}, {}
    with gzip.open(bf, "rt") as f:
        for r in csv.DictReader(f):
            d = bids if r["side"] == "bid" else asks
            px, q = round(float(r["price"]) / T), round(float(r["amount"]) / L)
            if q == 0:
                d.pop(px, None)
            else:
                d[px] = q
    assert bids == {p: l.total_qty for p, l in m.book.bids.items()}
    assert asks == {p: l.total_qty for p, l in m.book.asks.items()}
    with gzip.open(tf, "rt") as f:
        n = sum(1 for _ in csv.DictReader(f))
    assert n == len(m.book.trades)


def test_converter_accepts_fixture(npz):
    data = np.load(npz)["data"]
    assert len(data) > 1000
    assert data["local_ts"].min() > 0
    assert np.all(data["local_ts"] >= data["exch_ts"])       # feed latency is never negative


def test_bots_run_and_respect_cap(npz):
    from stage6.bots import NCOL, R_POS, run_bot
    from stage6.common import make_backtest
    qty, cap = 0.002, 5.0
    for gamma, flow in ((0.0, 0.0), (0.002, 0.0), (0.002, 6.0)):
        hbt = make_backtest(npz, 0.1, 0.001)
        out = np.zeros((5000, NCOL))
        n = run_bot(hbt, out, 100_000_000, 3.0, qty, cap, gamma, 50.0, 15.0, flow, 20, 200)
        hbt.close()
        assert n > 1000
        assert np.abs(out[:n, R_POS]).max() <= cap * qty + 1e-9


def test_observe_records_mid_and_flow(npz):
    from stage6.bots import observe
    from stage6.common import make_backtest
    hbt = make_backtest(npz, 0.1, 0.001)
    out = np.zeros((5000, 7))
    n = observe(hbt, out, 1_000_000_000)
    hbt.close()
    o = out[:n]
    assert n > 100
    assert np.nanmin(o[:, 1]) > 0
    assert o[:, 4].sum() > 0 and o[:, 5].sum() > 0


def test_flow_predictive_detects_planted_signal():
    from stage6.flow_check import predictive
    rng = np.random.default_rng(0)
    n = 20_000
    buy = rng.exponential(1, n)
    sell = rng.exponential(1, n)
    net = buy - sell
    price = np.cumsum(0.5 * np.concatenate([[0], net[:-1]]) + rng.normal(0, 1, n))
    (c, k), _ = predictive(buy, sell, price, 1, 1)
    assert c > 0.1 and k > 1000
    (c0, _), _ = predictive(buy, sell, np.cumsum(rng.normal(0, 1, n)), 1, 1)
    assert abs(c0) < 0.05


def test_t_p_uses_student_t():
    from stage6.run import t_p
    assert t_p(23.22, 2) == pytest.approx(0.0274, abs=1e-3)
    assert t_p(2.0, 1000) == pytest.approx(math.erfc(2 / math.sqrt(2)), abs=1e-3)
    assert t_p(float("nan"), 5) == 1.0


def test_download_month_range_and_url():
    from stage6.download import BASE, months
    assert list(months("2024-11", "2025-02")) == [(2024, 11), (2024, 12), (2025, 1), (2025, 2)]
    url = BASE.format(exchange="binance-futures", dtype="trades", y=2025, m="01", symbol="SOLUSDT")
    assert url == "https://datasets.tardis.dev/v1/binance-futures/trades/2025/01/01/SOLUSDT.csv.gz"


def test_midproxy_removes_bounce():
    # constant bid 99.9 / ask 100.1, trades alternate sides: the mid never moves.
    # vwap sees a bouncing price, the mid proxy must see a flat one.
    from stage6.flow_check import buckets_from_trades
    n = 4000
    ts = np.arange(n, dtype=np.int64) * 250_000           # 4 trades a second
    rng = np.random.default_rng(1)
    sg = np.where(rng.random(n) < 0.5, 1, -1)
    px = np.where(sg > 0, 100.1, 99.9)
    q = np.ones(n)
    _, _, mid = buckets_from_trades(ts, sg, px, q, price="midproxy")
    _, _, vw = buckets_from_trades(ts, sg, px, q, price="vwap")
    assert np.nanstd(mid[5:]) < 1e-9
    assert np.nanstd(vw[5:]) > 0.01
