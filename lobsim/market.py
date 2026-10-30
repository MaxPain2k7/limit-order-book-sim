"""
Stage 2: simulated market on top of the order book.

Three kinds of traders send orders:

  noise      random limit orders, market orders and cancels. They don't know
             anything, they just provide liquidity and churn.
  informed   see where the true price is heading and hit the side that's
             about to be wrong. They are the reason market making is hard.
  bot slot   any object with on_tick(book, t) and on_trade(trade, maker_pid,
             taker_pid, pre_mid), plus an optional attach(gateway) called once.
             The gateway exposes the book, send() and cancel(), nothing else,
             so a bot cannot reach the true price even by accident.

The "true" price is a hidden random walk with drift and occasional jumps.
Most noise traders quote around the mid they can see, a minority around a
noisy guess of true price. Informed traders see the true price and sweep
whichever side of the book has gone stale, which is what makes market
making hard: by the time they hit you, you were already wrong.

Everything is seeded. Same seed, same run, so strategies can be compared
on identical tapes.
"""

import math
import random

from .orderbook import Order, OrderBook, OrderType, Side


NOISE_PID = 1
INFORMED_PID = 2
BOT_PID = 99


class Gateway:
    """What a bot is given: the book for reading, send/cancel for acting, the
    order latency and a clock. No public attribute leads to the market. This
    is a convention, not a sandbox: Python introspection (closures, the
    book's event callback) can still reach it. Strategies in this repo do not."""

    __slots__ = ("book", "send", "cancel", "latency", "_clock")

    def __init__(self, market):
        self.book = market.book
        self.latency = market.bot_latency
        # wrapped so there is no public attribute path (gw.send.__self__) to
        # the market. still reachable through __closure__ if someone digs.
        send, cancel = market.send, market.cancel
        self.send = lambda side, px, qty, ot=OrderType.LIMIT: send(side, px, qty, ot)
        self.cancel = lambda oid: cancel(oid)
        self._clock = lambda: market.t

    @property
    def book_time(self):
        return self._clock()


class Market:
    def __init__(self,
                 seed=0,
                 start_price=10_000,      # ticks
                 vol=2.0,                 # true price std dev per step, in ticks
                 jump_prob=0.002,         # chance per step of a jump
                 jump_size=25,            # jump std dev in ticks
                 tail_df=None,            # Student-t degrees of freedom for price steps, None = Gaussian
                 vol_regime=None,         # (high_vol_multiplier, switch_prob): two-state Markov volatility
                 drift=0.0,               # ticks per step, set nonzero for a trend
                 noise_rate=4.0,          # noise orders per step
                 noise_depth=8,           # noise limits land within this many ticks of mid
                 noise_size=(1, 10),
                 noise_mkt_frac=0.25,     # fraction of noise orders that are market orders
                 noise_collar=16,         # noise "market" orders are IOC limits this far through mid
                 noise_fair_frac=0.3,     # fraction of noise limits that anchor on a noisy true price
                 noise_fair_err=10,       # std dev of that noise, in ticks
                 cancel_rate=1.5,         # noise cancels per step
                 informed_rate=0.3,       # informed orders per step
                 informed_size=(5, 20),
                 informed_edge=3,         # informed only trade when stale by this many ticks
                 bot=None,
                 bot_latency=0,           # steps before a bot order/cancel reaches the book
                 stp_mode=None):
        self.rng = random.Random(seed)
        self.true_price = float(start_price)
        self.vol = vol
        self.jump_prob = jump_prob
        self.jump_size = jump_size
        if tail_df is not None and tail_df <= 2:
            raise ValueError("tail_df must be > 2 so the variance exists")
        self.tail_df = tail_df
        self.vol_regime = vol_regime
        self.high_vol = False             # current regime when vol_regime is set
        self.drift = drift
        self.noise_rate = noise_rate
        self.noise_depth = noise_depth
        self.noise_size = noise_size
        self.noise_mkt_frac = noise_mkt_frac
        self.noise_collar = noise_collar
        self.noise_fair_frac = noise_fair_frac
        self.noise_fair_err = noise_fair_err
        self.cancel_rate = cancel_rate
        self.informed_rate = informed_rate
        self.informed_size = informed_size
        self.informed_edge = informed_edge
        self.bot = bot
        self.bot_latency = bot_latency
        self._bot_queue = []             # (arrive_at, kind, payload)
        self._fills = []                 # fills waiting to be handed to the bot
        self._pre_mid = None             # mid just before the order being matched arrived
        self.self_trades = 0             # bot on both sides. should stay zero

        kw = {"stp_mode": stp_mode} if stp_mode else {}
        self.book = OrderBook(on_event=self._on_event, **kw)
        self.t = 0
        self.next_id = 1
        self.noise_orders = []           # ids we may cancel later
        self.pids = {}                   # order id -> participant, outlives the order
        self.history = []                # (t, true, mid, best_bid, best_ask)
        self.informed_trades = 0
        self.noise_trades = 0
        self.bot_trades = 0

        self._seed_book(start_price)
        self.gw = Gateway(self)
        self._bot_wants_pre_mid = False
        if self.bot is not None:
            # decide the on_trade signature once. catching TypeError at call
            # time would also swallow genuine TypeErrors raised inside the bot
            # and could book a fill twice.
            import inspect
            fn = getattr(self.bot, "on_trade", None)
            if fn is not None:
                try:
                    inspect.signature(fn).bind(None, None, None, None)
                    self._bot_wants_pre_mid = True
                except TypeError:
                    self._bot_wants_pre_mid = False
            if hasattr(self.bot, "attach"):
                self.bot.attach(self.gw)


    def run(self, steps):
        for _ in range(steps):
            self.step()
        return self

    def step(self):
        self.t += 1
        self._move_true_price()

        self._deliver_bot_queue()

        for _ in range(self._poisson(self.noise_rate)):
            self._noise_order()
        for _ in range(self._poisson(self.cancel_rate)):
            self._noise_cancel()
        for _ in range(self._poisson(self.informed_rate)):
            self._informed_order()

        if self.bot is not None:
            self._flush_fills()
            self.bot.on_tick(self.book, self.t)
            self._flush_fills()

        b = self.book
        self.history.append((self.t, self.true_price, b.mid, b.best_bid, b.best_ask))

    # price process

    def _shock(self):
        """Unit variance shock: Gaussian, or Student-t rescaled to variance 1
        so tail_df changes the shape of the distribution, not its scale."""
        z = self.rng.gauss(0, 1)
        if self.tail_df is None:
            return z
        nu = self.tail_df
        chi2 = sum(self.rng.gauss(0, 1) ** 2 for _ in range(int(nu))) if float(nu).is_integer() \
            else self.rng.gammavariate(nu / 2, 2)
        return z / math.sqrt(chi2 / nu) * math.sqrt((nu - 2) / nu)

    def _move_true_price(self):
        vol = self.vol
        if self.vol_regime is not None:
            mult, p_switch = self.vol_regime
            if self.rng.random() < p_switch:
                self.high_vol = not self.high_vol
            # scale so the long run average variance equals vol^2
            lo = math.sqrt(2 / (1 + mult ** 2))
            vol = vol * (lo * mult if self.high_vol else lo)
        self.true_price += self.drift + vol * self._shock()
        if self.rng.random() < self.jump_prob:
            self.true_price += self.rng.gauss(0, self.jump_size)

    # traders

    def _seed_book(self, px):
        if px < 12:
            raise ValueError("start_price must be at least 12 ticks so the seed book fits above zero")
        for i in range(1, 11):
            self._send(Side.BUY, px - i, self.rng.randint(5, 20), NOISE_PID)
            self._send(Side.SELL, px + i, self.rng.randint(5, 20), NOISE_PID)

    def _noise_order(self):
        side = self.rng.choice((Side.BUY, Side.SELL))
        qty = self.rng.randint(*self.noise_size)

        if self.rng.random() < self.noise_mkt_frac:
            # a price-protected market order, as every real venue enforces.
            # without the collar a thin book lets noise sweep hundreds of
            # ticks into any far stub quote. Then a 500 tick wide "market
            # maker" becomes the best strategy in the sim.
            b = self.book
            ref = b.mid if b.mid is not None else (b.last_trade_price or self.true_price)
            c = self.noise_collar
            px = round(ref) + c if side is Side.BUY else round(ref) - c
            self._send(side, px, qty, NOISE_PID, OrderType.IOC)
            return

        ref = self._reference()
        # most noise rests a few ticks off the reference, sometimes right at touch
        off = int(abs(self.rng.gauss(0, self.noise_depth / 2))) + 1
        px = round(ref) - off if side is Side.BUY else round(ref) + off
        self._send(side, px, qty, NOISE_PID)

    def _noise_cancel(self):
        while self.noise_orders:
            oid = self.noise_orders.pop(self.rng.randrange(len(self.noise_orders)))
            o = self.book.get_order(oid)
            if o is not None and o.participant_id == NOISE_PID:
                self.book.cancel_order(oid, timestamp=self.t)
                return

    def _informed_order(self):
        b = self.book
        tp = round(self.true_price)
        qty = self.rng.randint(*self.informed_size)
        edge = self.informed_edge

        # take: ask is stale and cheap -> lift it. bid stale and rich -> hit it.
        # the limit sits at fair value less an edge so it sweeps whatever is
        # mispriced and rests if the far side is empty.
        if b.best_ask is not None and tp - b.best_ask >= edge:
            self._send(Side.BUY, tp - edge, qty, INFORMED_PID)
            return
        if b.best_bid is not None and b.best_bid - tp >= edge:
            self._send(Side.SELL, tp + edge, qty, INFORMED_PID)
            return

        # make: if a side is empty or far from fair, post there at fair +/- edge.
        # this is what competing liquidity providers would do. It also stops the
        # mid collapsing onto stale orders when one side gets eaten in a trend.
        if b.best_ask is None or b.best_ask - tp > 2 * edge:
            self._send(Side.SELL, tp + edge, qty, INFORMED_PID)
        elif b.best_bid is None or tp - b.best_bid > 2 * edge:
            self._send(Side.BUY, tp - edge, qty, INFORMED_PID)

    def _reference(self):
        # what a noise trader thinks the price is. most look at the screen,
        # some have a rough idea of fair value. without the second group the
        # book never catches up in a trend: noise refills cheap asks around a
        # stale mid and informed just keeps eating them.
        if self.rng.random() < self.noise_fair_frac:
            return self.true_price + self.rng.gauss(0, self.noise_fair_err)
        b = self.book
        if b.mid is not None:
            return b.mid
        if b.last_trade_price is not None:
            return b.last_trade_price
        return self.true_price

    # plumbing

    def _send(self, side, px, qty, pid, ot=OrderType.LIMIT):
        if px is not None and px < 1:
            if side is Side.SELL:
                px = 1               # a sell can always be priced at the floor
            else:
                return None          # a buy below 1 is meaningless, drop it
        oid = self.next_id
        self.next_id += 1
        o = Order(oid, side, px, qty, self.t, ot, participant_id=pid)
        self.pids[oid] = pid
        self._pre_mid = self.book.mid
        self.book.submit(o)
        if ot is OrderType.LIMIT and pid == NOISE_PID and self.book.get_order(oid) is not None:
            self.noise_orders.append(oid)
        return oid

    def send(self, side, px, qty, ot=OrderType.LIMIT):
        """For the bot. Returns the order id. With latency > 0 the order is
        queued and hits the book that many steps later."""
        oid = self.next_id
        self.next_id += 1
        self.pids[oid] = BOT_PID
        if self.bot_latency == 0:
            self._pre_mid = self.book.mid
            self.book.submit(Order(oid, side, px, qty, self.t, ot, participant_id=BOT_PID))
        else:
            self._bot_queue.append((self.t + self.bot_latency, "send", (oid, side, px, qty, ot)))
        return oid

    def cancel(self, oid):
        if self.bot_latency == 0:
            self._cancel_now(oid)
        else:
            self._bot_queue.append((self.t + self.bot_latency, "cancel", oid))

    def _cancel_now(self, oid):
        if self.book.get_order(oid) is not None:
            self.book.cancel_order(oid, timestamp=self.t)

    def _deliver_bot_queue(self):
        due = [x for x in self._bot_queue if x[0] <= self.t]
        self._bot_queue = [x for x in self._bot_queue if x[0] > self.t]
        for _, kind, payload in due:
            if kind == "send":
                oid, side, px, qty, ot = payload
                self._pre_mid = self.book.mid
                self.book.submit(Order(oid, side, px, qty, self.t, ot, participant_id=BOT_PID))
            else:
                self._cancel_now(payload)

    def _flush_fills(self):
        # the bot only hears about fills between book operations, never during
        # one. calling send() from inside on_trade would otherwise mutate a
        # price level while the matcher is still walking it.
        fills, self._fills = self._fills, []
        if not hasattr(self.bot, "on_trade"):
            return
        for tr, maker, taker, pre_mid in fills:
            if self._bot_wants_pre_mid:
                self.bot.on_trade(tr, maker, taker, pre_mid)
            else:
                self.bot.on_trade(tr, maker, taker)

    def _on_event(self, ev):
        if ev.trade is None:
            return
        tr = ev.trade
        maker = self.pids.get(tr.maker_order_id)
        taker = self.pids.get(tr.taker_order_id)
        if taker == INFORMED_PID:
            self.informed_trades += 1
        elif taker == BOT_PID:
            self.bot_trades += 1
        else:
            self.noise_trades += 1
        if maker == BOT_PID and taker == BOT_PID:
            self.self_trades += 1
        if self.bot is not None and (maker == BOT_PID or taker == BOT_PID):
            # pre_mid: the mid the taker saw when it sent the order. that is the
            # reference a maker's edge should be measured against, not the mid
            # after the trade has already moved the touch.
            self._fills.append((tr, maker, taker, self._pre_mid))


    def _poisson(self, lam):
        if lam <= 0:
            return 0
        if lam > 50:
            # Knuth's method underflows exp(-lam) near 745. normal approximation
            # is accurate to well under 1% at this size.
            return max(0, int(round(self.rng.gauss(lam, math.sqrt(lam)))))
        L = math.exp(-lam)
        k, p = 0, 1.0
        while True:
            p *= self.rng.random()
            if p < L:
                return k
            k += 1

    def save_history(self, path):
        """Dump the tape as csv: step, true price, mid, bid, ask."""
        with open(path, "w") as f:
            f.write("t,true,mid,bid,ask\n")
            for t, true, mid, bid, ask in self.history:
                f.write(f"{t},{true:.3f},{'' if mid is None else mid},"
                        f"{'' if bid is None else bid},{'' if ask is None else ask}\n")

    def save_trades(self, path):
        """Dump every trade with who was on each side."""
        with open(path, "w") as f:
            f.write("trade_id,t,price,qty,aggressor,maker_pid,taker_pid\n")
            for tr in self.book.trades:
                f.write(f"{tr.trade_id},{tr.timestamp},{tr.price},{tr.qty},{tr.aggressor_side.value},"
                        f"{self.pids.get(tr.maker_order_id)},{self.pids.get(tr.taker_order_id)}\n")

    def summary(self):
        b = self.book
        mids = [h[2] for h in self.history if h[2] is not None]
        err = [abs(h[2] - h[1]) for h in self.history if h[2] is not None]
        return {
            "steps": self.t,
            "trades": len(b.trades),
            "informed_trades": self.informed_trades,
            "noise_trades": self.noise_trades,
            "bot_trades": self.bot_trades,
            "final_true": round(self.true_price, 1),
            "final_mid": b.mid,
            "avg_spread": round(sum(h[4] - h[3] for h in self.history if h[3] and h[4]) / max(1, len(mids)), 2),
            "avg_abs_mid_error": round(sum(err) / max(1, len(err)), 2),
            "resting_orders": len(b),
        }
