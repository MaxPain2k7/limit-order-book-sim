"""
Stage 3: a market maker that quotes a fixed spread around mid and nothing else.

At a tight spread it loses money. It has no idea what inventory is and does
not care who is hitting it. With requote_always=True it also requotes every
step whether the market moved or not.
Everything in stage 4 and 5 is measured against this.

PnL is split the way a desk would split it:
  spread pnl     what you earn by buying on the bid and selling on the ask
  inventory pnl  what your open position gains or loses as mid moves
Total = cash + position * mid. Marked to mid every step.
"""

import math

from .orderbook import OrderType, Side
from .market import BOT_PID, INFORMED_PID


class Blotter:
    """Position and PnL. Fed one fill at a time, marked on demand.

    spread_pnl is the edge captured at each fill measured against the mid at
    that moment: buy below mid, sell above mid. Everything else is inventory
    pnl, meaning what the open position did as mid moved. That is where
    adverse selection shows up: a fill that looked like +3 ticks of edge is
    followed by mid moving 10 ticks against you.
    """

    def __init__(self, maker_fee=0.0, taker_fee=0.0):
        # fees as a fraction of notional. negative maker fee = rebate.
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.fees_paid = 0.0
        self.position = 0
        self.cash = 0.0
        self.spread_pnl = 0.0
        self.liq_pnl = None
        self.last_mid = None
        self.fills = []                 # (t, side, price, qty, counterparty_pid, mid_at_fill)
        self.n_informed = 0
        self.n_noise = 0
        self.curve = []                 # (t, position, total_pnl, spread_pnl, inv_pnl)

    def on_fill(self, t, side, price, qty, counterparty, mid, as_maker=True):
        fee = qty * price * (self.maker_fee if as_maker else self.taker_fee)
        self.fees_paid += fee
        self.cash -= fee
        if counterparty == INFORMED_PID:
            self.n_informed += 1
        else:
            self.n_noise += 1
        self.fills.append((t, side, price, qty, counterparty, mid))
        if side is Side.BUY:
            self.position += qty
            self.cash -= qty * price
            self.spread_pnl += qty * (mid - price)
        else:
            self.position -= qty
            self.cash += qty * price
            self.spread_pnl += qty * (price - mid)

    def mark(self, t, mid):
        if mid is None:
            mid = self.last_mid if self.last_mid is not None else (self.fills[-1][5] if self.fills else 0)
        self.last_mid = mid
        total = self.cash + self.position * mid
        self.curve.append((t, self.position, total, self.spread_pnl,
                           total - self.spread_pnl + self.fees_paid))
        return total

    def liquidate(self, book, penalty_ticks=10):
        """
        PnL if the open position were flattened now by crossing the book,
        paying the taker fee. Marking leftover inventory at mid assumes it can
        be closed for free, which flatters any bot that ends a run carrying
        size.

        Whatever the visible book cannot absorb (including a side that is
        empty at the final step) is assumed to clear `penalty_ticks` beyond
        the worse of the last swept level and the last valid mark. The
        reference is never an undefined mid: a one-sided book at the end of a
        run once made this function book +287k on a 28 lot short.
        """
        q = self.position
        ref = book.mid if book.mid is not None else self.last_mid
        if q == 0 or ref is None:
            self.liq_pnl = self.cash + q * (ref or 0)
            return self.liq_pnl
        selling = q > 0
        need = abs(q)
        opp = book.bids if selling else book.asks
        got, cost, last = 0, 0.0, None
        for p, lvl in opp.items():
            take = min(need - got, lvl.total_qty)
            cost += take * p
            got += take
            last = p
            if got == need:
                break
        if got < need:
            if selling:
                px = min(last if last is not None else ref, ref) - penalty_ticks
                px = max(px, 1)
            else:
                px = max(last if last is not None else ref, ref) + penalty_ticks
            cost += (need - got) * px
        proceeds = cost if selling else -cost
        fee = cost * self.taker_fee
        self.liq_pnl = self.cash + proceeds - fee
        return self.liq_pnl

    def summary(self):
        tot = self.curve[-1][2] if self.curve else 0
        pos = [c[1] for c in self.curve]
        return {
            "total_pnl": round(tot, 1),
            **({"liq_pnl": round(self.liq_pnl, 1)} if self.liq_pnl is not None else {}),
            "spread_pnl": round(self.spread_pnl, 1),
            "inventory_pnl": round(tot - self.spread_pnl + self.fees_paid, 1),
            "fees_paid": round(self.fees_paid, 1),
            "fills": len(self.fills),
            "fills_vs_informed": self.n_informed,
            "fills_vs_noise": self.n_noise,
            "final_position": self.position,
            "max_abs_position": max((abs(p) for p in pos), default=0),
            "avg_abs_position": round(sum(abs(p) for p in pos) / max(1, len(pos)), 1),
        }


class NaiveMaker:
    """
    half_spread      ticks either side of mid
    size             lots per quote
    max_position     hard cap. quote size is cut to the room left, so the
                     cap is never exceeded. None means no cap.
    requote_always   True: cancel and resend every step, which throws away
                     queue position. False: leave a quote alone if its price
                     is still right. Real makers do the second thing.
    """

    def __init__(self, half_spread=3, size=5, max_position=None,
                 requote_always=True, maker_fee=0.0, taker_fee=0.0, requote_tol=0):
        self.half_spread = half_spread
        self.size = size
        self.max_position = max_position
        self.requote_always = requote_always
        self.requote_tol = requote_tol
        self.blotter = Blotter(maker_fee, taker_fee)
        self.bid_id = self.ask_id = None
        self.bid_px = self.ask_px = None
        self.open = {}                   # oid -> (side, qty, t_sent) for every order we own

    def attach(self, gw):
        self.gw = gw

    def _exposure(self, side):
        """Lots that could still fill on this side. Resting orders count their
        remaining qty. Orders in flight, or whose cancel is in flight, count
        what we sent. Under latency there can be several of these at once,
        so the position alone understates the risk."""
        book, now, lat = self.gw.book, self.gw.book_time, self.gw.latency
        total = 0
        for oid, (sd, qty, t0) in list(self.open.items()):
            o = book.get_order(oid)
            if o is not None:
                if sd is side:
                    total += o.qty
            elif now - t0 <= lat:
                if sd is side:
                    total += qty
            else:
                del self.open[oid]       # landed and gone, or never will be seen again
        return total

    def _purge_open(self):
        """Forget own orders that are neither resting nor in flight."""
        book, now, lat = self.gw.book, self.gw.book_time, self.gw.latency
        for oid in [o for o, (_, _, t0) in self.open.items()
                    if now - t0 > lat and book.get_order(o) is None]:
            del self.open[oid]

    def _room(self, side):
        if self.max_position is None:
            return self.size
        pos = self.blotter.position
        room = self.max_position - pos if side is Side.BUY else self.max_position + pos
        room -= self._exposure(side)
        return max(0, min(self.size, room))

    def _quote(self, side, px):
        is_bid = side is Side.BUY
        oid = self.bid_id if is_bid else self.ask_id
        cur_px = self.bid_px if is_bid else self.ask_px
        resting = oid is not None and self.gw.book.get_order(oid) is not None
        in_flight = (oid is not None and oid in self.open
                     and self.gw.book_time - self.open[oid][2] < self.gw.latency)
        live = resting or in_flight
        qty = self._room(side)

        self._purge_open()
        # keep the resting quote if it is at the target, or behind it by at
        # most requote_tol ticks (safer, keeps queue priority). never
        # keep a quote that is now more aggressive than the target: that is
        # the stale quote informed flow picks off.
        if cur_px is None:
            safe = False
        elif is_bid:
            safe = px - self.requote_tol <= cur_px <= px
        else:
            safe = px <= cur_px <= px + self.requote_tol
        keep = (not self.requote_always) and live and safe
        if keep:
            return
        if oid is not None:
            self.gw.cancel(oid)
        oid = px_out = None
        if qty > 0:
            oid = self.gw.send(side, px, qty, OrderType.POST_ONLY)
            px_out = px
        if oid is not None:
            self.open[oid] = (side, qty, self.gw.book_time)
        if is_bid:
            self.bid_id, self.bid_px = oid, px_out
        else:
            self.ask_id, self.ask_px = oid, px_out

    def on_tick(self, book, t):
        self.blotter.mark(t, book.mid)
        if book.mid is None:
            return
        mid = round(book.mid)
        self._quote(Side.BUY, max(1, mid - self.half_spread))
        self._quote(Side.SELL, mid + self.half_spread)

    def on_trade(self, tr, maker_pid, taker_pid, pre_mid=None):
        mid = pre_mid if pre_mid is not None else (self.gw.book.mid or tr.price)
        # we only post, so normally we are the maker. the taker leg is here
        # for completeness: if a quote ever crossed one of our own resting
        # orders both legs must be booked or the position goes wrong.
        if maker_pid == BOT_PID:
            side = Side.SELL if tr.aggressor_side is Side.BUY else Side.BUY
            self.blotter.on_fill(tr.timestamp, side, tr.price, tr.qty, taker_pid, mid, as_maker=True)
        if taker_pid == BOT_PID:
            self.blotter.on_fill(tr.timestamp, tr.aggressor_side, tr.price, tr.qty, maker_pid, mid, as_maker=False)


class InventoryMaker(NaiveMaker):
    """
    Avellaneda-Stoikov market maker. Finite horizon formulas with the time
    to horizon held constant (no session end in the sim, so T - t = horizon).

        reservation  r = mid - q * gamma * sigma^2 * horizon
        half spread  d = gamma sigma^2 horizon / 2 + (1/gamma) ln(1 + gamma/kappa)
        bid = r - d,  ask = r + d

    q is the position. If we are long, r sits below mid, so the ask is closer
    to the market and the bid is further away: we lean on the side that
    reduces the position. gamma sets how hard we lean.

    sigma   std dev of mid changes per step, rolling window on the book
    kappa   how fast fill intensity decays with distance from mid, fitted
            to the public trade tape: lambda(d) = A * exp(-kappa * d)
    Both come from the book only. The strategy never sees the true price.

    The naive maker's hard cap and in-flight exposure logic are inherited as
    a backstop, but the point is that the skew should keep the cap from ever
    being hit.
    """

    def __init__(self, gamma=0.002, horizon=50, kappa=None, sigma_window=200,
                 kappa_window=2000, min_half_spread=1, spread_mode="as",
                 max_skew=15, max_half_spread=25, **kw):
        """spread_mode
        "fixed"  half_spread from NaiveMaker, skew from AS. Only the skew
                 differs from the naive bot so the comparison is like for like.
        "as"     full AS half spread, g s^2 h / 2 + (1/g) ln(1 + g/k).
        "glft"   Gueant-Lehalle-Fernandez-Tapia. Both half spread and skew
                 come from fitted A and k. This is the form hftbacktest's
                 own examples use, so stage 6 ports directly.
        """
        super().__init__(**kw)
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if kappa is not None and kappa <= 0:
            raise ValueError("kappa must be positive")
        if sigma_window < 20:
            raise ValueError("sigma_window must be at least 20")
        if spread_mode not in ("fixed", "as", "glft"):
            raise ValueError(f"unknown spread_mode {spread_mode!r}")
        self.gamma = gamma
        self.spread_mode = spread_mode
        self.max_half_spread = max_half_spread
        self.max_skew = max_skew         # ticks. a jump in sigma must not send quotes to the moon
        self.skew = 0.0
        self.horizon = horizon
        self.kappa_fixed = kappa
        self.sigma_window = sigma_window
        self.kappa_window = kappa_window
        self.min_half_spread = min_half_spread
        self.mids = []                   # (t, mid)
        self.sigma = None
        self.kappa = kappa if kappa is not None else 0.5
        self.A = None                    # fill intensity at the mid, trades per step
        self.reservation = None
        self.half = None
        self._kappa_fit_at = 0
        self._trades_seen = 0

    # estimators

    def _update_sigma(self):
        """Winsorised std. A plain std blows up on the one-step dislocations
        you get when a side of the book empties. An inflated sigma pins
        the skew at its clamp. A pure MAD is robust but quantises on half-tick
        mid changes. So: clip at 5 MADs (at least 3 ticks), then take the std."""
        w = self.sigma_window
        if len(self.mids) < 20:
            return
        xs = [m for _, m in self.mids[-w:]]
        d = [b - a for a, b in zip(xs, xs[1:])]
        mad = sorted(abs(x) for x in d)[len(d) // 2]
        lim = max(5 * 1.4826 * mad, 3.0)
        d = [max(-lim, min(lim, x)) for x in d]
        mean = sum(d) / len(d)
        var = sum((x - mean) ** 2 for x in d) / len(d)
        self.sigma = max(var ** 0.5, 0.25)          # floor so skew never dies

    def _update_kappa(self, book, t):
        """Fit kappa to the market's own trades: bucket trades by how far
        their price was from the mid at the time, then least squares on
        log counts. First fit at step 100, then every 500 steps."""
        first = self.A is None and t >= 100
        if not first and t - self._kappa_fit_at < 500:
            return
        self._kappa_fit_at = t
        t0 = t - self.kappa_window
        mid_at = dict(self.mids[-self.kappa_window:])
        counts = {}
        for tr in reversed(book.trades):
            if tr.timestamp <= t0:
                break
            m = mid_at.get(tr.timestamp - 1)        # mid the taker saw, not the one it left
            if m is None:
                continue
            d = round(abs(tr.price - m))
            counts[d] = counts.get(d, 0) + 1
        pts = [(d, math.log(c)) for d, c in counts.items() if 1 <= d <= 12 and c >= 3]
        if len(pts) < 3:
            return
        n = len(pts)
        sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
        sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
        if self.kappa_fixed is not None:
            slope = -self.kappa_fixed            # hold kappa, fit only the level
        else:
            slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        if slope < 0:
            if self.kappa_fixed is None:
                self.kappa = min(max(-slope, 0.05), 5.0)
            intercept = (sy - slope * sx) / n
            window_steps = min(t, self.kappa_window)
            # counts are both sides of the tape, GLFT's A is one side's arrival rate
            self.A = math.exp(intercept) / window_steps / 2

    # hooks for subclasses ------------------------------------------------- #

    def _fair(self, book, t):
        """Reference price the quotes are centred on. Plain AS uses the mid."""
        return book.mid

    def _widen(self, book, t):
        """Extra ticks on (bid, ask). Plain AS adds none."""
        return 0.0, 0.0

    @staticmethod
    def _glft(gamma, k, A, sigma, delta=1.0):
        """
        GLFT closed form, as simplified in the hftbacktest tutorial.
            c1 = (1/(xi d)) ln(1 + xi d / k)
            c2 = sqrt( gamma / (2 A d k) * (1 + xi d / k)^(k/(xi d) + 1) )
            half spread = c1 + d/2 c2 sigma
            skew per unit inventory = c2 sigma
        xi = gamma, d = 1 unit of inventory (one quote size).
        sigma is per step. A is trades per step at the mid.
        """
        xi = gamma
        xd = xi * delta
        c1 = math.log(1 + xd / k) / xd
        c2 = math.sqrt(gamma / (2 * A * delta * k) * (1 + xd / k) ** (k / xd + 1))
        half = c1 + delta / 2 * c2 * sigma
        skew = c2 * sigma
        return half, skew


    def on_tick(self, book, t):
        self.blotter.mark(t, book.mid)
        if book.mid is None:
            return
        self.mids.append((t, book.mid))
        if len(self.mids) > 2 * self.kappa_window:
            del self.mids[: self.kappa_window]
        self._update_sigma()
        self._update_kappa(book, t)
        if self.sigma is None or (self.spread_mode == "glft" and self.A is None):
            # warm up: quote like the naive bot until the estimators exist.
            # GLFT gammas are 10-100x the AS ones, so running the AS skew with
            # a GLFT gamma would pin the skew at the clamp.
            mid = round(book.mid)
            self._quote(Side.BUY, max(1, mid - self.half_spread))
            self._quote(Side.SELL, mid + self.half_spread)
            return

        q = self.blotter.position
        g, k = self.gamma, self.kappa
        if self.spread_mode == "glft":
            half, skew_per_lot = self._glft(g, k, self.A, self.sigma)
            skew = skew_per_lot * q / self.size
            self.half = min(self.max_half_spread, max(self.min_half_spread, half))
        else:
            skew = q * g * self.sigma ** 2 * self.horizon
            if self.spread_mode == "fixed":
                self.half = self.half_spread
            else:
                # full AS half spread with constant time-to-horizon
                self.half = min(self.max_half_spread, max(
                    self.min_half_spread,
                    g * self.sigma ** 2 * self.horizon / 2 + math.log1p(g / k) / g))
        self.skew = max(-self.max_skew, min(self.max_skew, skew))
        self.reservation = self._fair(book, t) - self.skew
        wb, wa = self._widen(book, t)

        # round half away from zero on both so the quoted spread averages
        # 2*half, same as the naive bot. floor/ceil would add a tick.
        bid = int(math.floor(self.reservation - self.half - wb + 0.5))
        ask = int(math.floor(self.reservation + self.half + wa + 0.5))

        # post only: never cross, never quote a locked market
        if book.best_ask is not None:
            bid = min(bid, book.best_ask - 1)
        if book.best_bid is not None:
            ask = max(ask, book.best_bid + 1)
        if ask <= bid:
            ask = bid + 1
        bid = max(bid, 1)

        self._quote(Side.BUY, bid)
        self._quote(Side.SELL, ask)


class DefensiveMaker(InventoryMaker):
    """
    Stage 5: Avellaneda-Stoikov plus defences against adverse selection.
    Everything here uses information a real market maker has: the book, the
    public trade tape (price, size, aggressor side) and its own fills. It
    never uses counterparty ids, which no real venue reveals.

    requote_tol   keep a resting quote up to this many ticks behind target
                  instead of cancelling it (inherited from NaiveMaker). Fewer
                  cancels in flight, so fewer stale quotes to be picked off.

    imb_coef      ticks the fair value moves per unit of book imbalance over
                  the top imb_levels levels, with the bot's own quotes taken
                  out first so it does not chase its own reflection.

    flow_coef     ticks the fair value moves per unit of net aggressive flow
                  on the public tape over the last flow_window steps,
                  (buy volume - sell volume) / total, in [-1, 1]. Informed
                  traders arrive in bursts on one side; this leans with them.

    tox_*         online toxicity. Each own fill is marked out tox_h steps
                  later against the mid the bot sees then. A per side EWMA of
                  those markouts (ticks per lot) widens that side by
                  tox_gain ticks per tick of negative markout, capped at
                  tox_max. Buys that keep losing -> bid backs off.
    """

    def __init__(self, imb_coef=0.0, imb_levels=3, flow_coef=0.0, flow_window=20,
                 tox_h=10, tox_alpha=0.1, tox_gain=0.0, tox_max=6.0, **kw):
        super().__init__(**kw)
        if not 0 < tox_alpha <= 1:
            raise ValueError("tox_alpha must be in (0, 1]")
        self.imb_coef = imb_coef
        self.imb_levels = imb_levels
        self.flow_coef = flow_coef
        self.flow_window = flow_window
        self.tox_h = tox_h
        self.tox_alpha = tox_alpha
        self.tox_gain = tox_gain
        self.tox_max = tox_max
        self.tox = {Side.BUY: 0.0, Side.SELL: 0.0}   # EWMA markout per side, ticks per lot
        self._pending = []                             # own fills awaiting markout
        self.last_imb = 0.0
        self.last_flow = 0.0

    # signals ---------------------------------------------------------------- #

    def _own_qty_at(self, price, side):
        q = 0
        for oid in self.open:
            o = self.gw.book.get_order(oid)
            if o is not None and o.price == price and o.side is side:
                q += o.qty
        return q

    def _imbalance_ex_own(self, book):
        bids, asks = book.depth(self.imb_levels)
        bq = sum(max(0, q - self._own_qty_at(p, Side.BUY)) for p, q in bids)
        aq = sum(max(0, q - self._own_qty_at(p, Side.SELL)) for p, q in asks)
        return (bq - aq) / (bq + aq) if bq + aq > 0 else 0.0

    def _flow(self, book, t):
        t0 = t - self.flow_window
        buy = sell = 0
        for tr in reversed(book.trades):
            if tr.timestamp <= t0:
                break
            if tr.aggressor_side is Side.BUY:
                buy += tr.qty
            else:
                sell += tr.qty
        return (buy - sell) / (buy + sell) if buy + sell > 0 else 0.0

    def _update_tox(self, book, t):
        mid = book.mid
        if mid is None:
            return
        keep = []
        a = self.tox_alpha
        for ft, side, px, qty in self._pending:
            if t - ft < self.tox_h:
                keep.append((ft, side, px, qty))
                continue
            mo = (mid - px) if side is Side.BUY else (px - mid)
            self.tox[side] = (1 - a) * self.tox[side] + a * mo
        self._pending = keep

    # hooks ------------------------------------------------------------------ #

    def _fair(self, book, t):
        f = book.mid
        if self.imb_coef:
            self.last_imb = self._imbalance_ex_own(book)
            f += self.imb_coef * self.last_imb
        if self.flow_coef:
            self.last_flow = self._flow(book, t)
            f += self.flow_coef * self.last_flow
        return f

    def _widen(self, book, t):
        if not self.tox_gain:
            return 0.0, 0.0
        wb = min(self.tox_max, max(0.0, -self.tox[Side.BUY]) * self.tox_gain)
        wa = min(self.tox_max, max(0.0, -self.tox[Side.SELL]) * self.tox_gain)
        return wb, wa

    def on_tick(self, book, t):
        self._update_tox(book, t)
        super().on_tick(book, t)

    def on_trade(self, tr, maker_pid, taker_pid, pre_mid=None):
        super().on_trade(tr, maker_pid, taker_pid, pre_mid)
        if maker_pid == BOT_PID:
            side = Side.SELL if tr.aggressor_side is Side.BUY else Side.BUY
            self._pending.append((tr.timestamp, side, tr.price, tr.qty))
