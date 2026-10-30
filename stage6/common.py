"""Shared helpers for stage 6: building backtests from converted .npz days."""
import glob
import os

import numpy as np

from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest

# defaults for Binance USD-M futures. check the exchange's instrument info for
# the symbol you download; run.py and flow_check.py take --tick and --lot.
TICK = {"BTCUSDT": 0.1, "ETHUSDT": 0.01, "SOLUSDT": 0.01, "SIMUSDT": 0.1}
LOT = {"BTCUSDT": 0.001, "ETHUSDT": 0.001, "SOLUSDT": 1.0, "SIMUSDT": 0.001}


def days(data_dir, symbol):
    """Sorted list of converted day files: data_dir/SYMBOL_YYYYMMDD.npz"""
    return sorted(glob.glob(os.path.join(data_dir, f"{symbol}_*.npz")))


def infer_tick(npz):
    px = np.unique(np.load(npz)["data"]["px"])
    px = px[np.isfinite(px) & (px > 0)]
    d = np.diff(px)
    d = d[d > 1e-12]
    return float(np.round(d.min(), 10)) if len(d) else None


def make_backtest(npz, tick, lot, latency_ms=10.0, maker_fee=0.0, taker_fee=0.0,
                  queue="power2"):
    lat = int(latency_ms * 1_000_000)
    a = (BacktestAsset().data([npz]).linear_asset(1.0)
         .constant_order_latency(lat, lat)
         .no_partial_fill_exchange()
         .trading_value_fee_model(maker_fee, taker_fee)
         .tick_size(tick).lot_size(lot)
         .last_trades_capacity(1_000_000))
    a = a.power_prob_queue_model(2.0) if queue == "power2" else a.risk_adverse_queue_model()
    return HashMapMarketDepthBacktest([a])
