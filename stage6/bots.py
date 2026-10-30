"""
The lob-sim market makers ported to hftbacktest, for real L2 data.

One numba function runs all three, so they share every line of order
handling and differ only in how they price:

    naive   quotes fair +/- half_spread ticks, fair = mid
    AS      fair = mid - gamma * q * sigma^2 * horizon      (q in quote-size units,
                                                             sigma in ticks per interval)
    flow    fair += flow_coef * net aggressive flow over the last flow_window
            intervals, (buy qty - sell qty) / total, from the public trade tape

Same conventions as lob-sim: sigma is a winsorised rolling std of mid
changes, the skew is clamped, quotes are post only (GTX) so they never cross,
and a hard cap counts resting orders as exposure.

hftbacktest models what lob-sim gets for free from its matching engine:
queue position (estimated from depth changes and trades, since L2 data does
not show individual orders), order and feed latency and fees.
"""
import numpy as np
from numba import njit

from hftbacktest import BUY, GTX, LIMIT, SELL, BUY_EVENT

# columns of the per-interval record
R_TS, R_MID, R_POS, R_EQUITY, R_FEE, R_FILLS, R_FLOW, R_SKEW, R_TVALUE = range(9)
NCOL = 9


@njit
def run_bot(hbt, out, interval_ns, half_spread, order_qty, max_pos_units,
            gamma, horizon, max_skew, flow_coef, flow_window, sigma_window):
    """
    out: float64 array (n, NCOL), filled one row per interval. Returns rows used.
    max_pos_units: position cap in quote-size units.
    """
    tick = hbt.depth(0).tick_size
    mids = np.zeros(sigma_window)
    fw = max(flow_window, 1)
    buys = np.zeros(fw)
    sells = np.zeros(fw)
    n_mid = 0
    row = 0
    next_id = 1
    sigma = 0.0
    while hbt.elapse(interval_ns) == 0:
        if row >= out.shape[0]:
            break
        hbt.clear_inactive_orders(0)
        depth = hbt.depth(0)
        bb, ba = depth.best_bid, depth.best_ask
        if not (np.isfinite(bb) and np.isfinite(ba)) or ba <= bb:
            continue
        mid = (bb + ba) / 2.0

        # public tape since last interval: net aggressive volume
        buy_q = 0.0
        sell_q = 0.0
        trades = hbt.last_trades(0)
        for i in range(len(trades)):
            if trades[i].ev & BUY_EVENT == BUY_EVENT:
                buy_q += trades[i].qty
            else:
                sell_q += trades[i].qty
        hbt.clear_last_trades(0)
        buys[row % fw] = buy_q
        sells[row % fw] = sell_q
        # rolling, winsorised sigma of mid changes in ticks
        mids[n_mid % sigma_window] = mid
        n_mid += 1
        if n_mid > 20:
            k = min(n_mid, sigma_window)
            d = np.empty(k - 1)
            for i in range(k - 1):
                a = mids[(n_mid - k + i) % sigma_window]
                b = mids[(n_mid - k + i + 1) % sigma_window]
                d[i] = (b - a) / tick
            mad = np.median(np.abs(d))
            lim = max(5.0 * 1.4826 * mad, 3.0)
            for i in range(len(d)):
                if d[i] > lim:
                    d[i] = lim
                elif d[i] < -lim:
                    d[i] = -lim
            sigma = max(np.std(d), 0.25)

        pos = hbt.position(0)
        q = pos / order_qty
        fair = mid
        skew = 0.0
        if gamma > 0.0 and n_mid > 20:
            skew = gamma * q * sigma * sigma * horizon
            skew = min(max_skew, max(-max_skew, skew))
            fair -= skew * tick
        sb = buys.sum()
        ss = sells.sum()
        flow = (sb - ss) / (sb + ss) if sb + ss > 0 else 0.0
        if flow_window > 0 and flow_coef != 0.0:
            fair += flow_coef * flow * tick

        bid = np.floor((fair - half_spread * tick) / tick + 0.5) * tick
        ask = np.floor((fair + half_spread * tick) / tick + 0.5) * tick
        bid = min(bid, ba - tick)
        ask = max(ask, bb + tick)

        # exposure cap: position plus what is resting on that side
        rest_buy = 0.0
        rest_sell = 0.0
        orders = hbt.orders(0)
        vals = orders.values()
        while vals.has_next():
            o = vals.get()
            if o.side == BUY:
                rest_buy += o.leaves_qty
            else:
                rest_sell += o.leaves_qty
        can_buy = (pos + rest_buy) / order_qty < max_pos_units
        can_sell = (-pos + rest_sell) / order_qty < max_pos_units

        # keep an order already at the target price (queue priority),
        # cancel everything else
        have_bid = False
        have_ask = False
        vals = orders.values()
        while vals.has_next():
            o = vals.get()
            if not o.cancellable:
                continue
            target = bid if o.side == BUY else ask
            if abs(o.price - target) < tick * 0.5:
                if o.side == BUY:
                    have_bid = True
                else:
                    have_ask = True
            else:
                hbt.cancel(0, o.order_id, False)
        if not have_bid and can_buy:
            hbt.submit_buy_order(0, next_id, bid, order_qty, GTX, LIMIT, False)
            next_id += 1
        if not have_ask and can_sell:
            hbt.submit_sell_order(0, next_id, ask, order_qty, GTX, LIMIT, False)
            next_id += 1

        sv = hbt.state_values(0)
        out[row, R_TS] = hbt.current_timestamp
        out[row, R_MID] = mid
        out[row, R_POS] = pos
        out[row, R_EQUITY] = sv.balance + pos * mid - sv.fee
        out[row, R_FEE] = sv.fee
        out[row, R_FILLS] = sv.num_trades
        out[row, R_FLOW] = flow
        out[row, R_SKEW] = skew
        out[row, R_TVALUE] = sv.trading_value
        row += 1
    return row


@njit
def observe(hbt, out, interval_ns):
    """
    Trade nothing; record (timestamp, mid, best bid, best ask, aggressive buy
    qty, aggressive sell qty, trade count) every interval. This is the cheap
    first look at real data: does order flow predict the next mid move?
    """
    row = 0
    while hbt.elapse(interval_ns) == 0:
        if row >= out.shape[0]:
            break
        depth = hbt.depth(0)
        bb, ba = depth.best_bid, depth.best_ask
        buy_q = 0.0
        sell_q = 0.0
        trades = hbt.last_trades(0)
        for i in range(len(trades)):
            if trades[i].ev & BUY_EVENT == BUY_EVENT:
                buy_q += trades[i].qty
            else:
                sell_q += trades[i].qty
        n_tr = len(trades)
        hbt.clear_last_trades(0)
        out[row, 0] = hbt.current_timestamp
        out[row, 1] = (bb + ba) / 2.0 if np.isfinite(bb) and np.isfinite(ba) else np.nan
        out[row, 2] = bb
        out[row, 3] = ba
        out[row, 4] = buy_q
        out[row, 5] = sell_q
        out[row, 6] = n_tr
        row += 1
    return row
