import pytest
from lobsim.market import BOT_PID, INFORMED_PID, NOISE_PID, Market
from lobsim.orderbook import Side


def test_same_seed_same_result():
    a = Market(seed=3).run(500).summary()
    b = Market(seed=3).run(500).summary()
    assert a == b


def test_different_seed_different_result():
    a = Market(seed=3).run(500).summary()
    b = Market(seed=4).run(500).summary()
    assert a != b


def test_book_never_crosses():
    m = Market(seed=5)
    for _ in range(2000):
        m.step()
        b = m.book
        if b.best_bid is not None and b.best_ask is not None:
            assert b.best_bid < b.best_ask


def test_mid_follows_true_price():
    m = Market(seed=6).run(3000)
    s = m.summary()
    assert abs(s["final_mid"] - s["final_true"]) < 60
    # with no informed traders the book should drift away from truth
    m2 = Market(seed=6, informed_rate=0).run(3000)
    assert m2.summary()["avg_abs_mid_error"] > s["avg_abs_mid_error"]


def test_informed_traders_actually_trade():
    m = Market(seed=7).run(2000)
    assert m.informed_trades > 0
    assert m.noise_trades > m.informed_trades


def test_no_informed_when_rate_zero():
    m = Market(seed=7, informed_rate=0).run(2000)
    assert m.informed_trades == 0


def test_drift_moves_price():
    up = Market(seed=8, drift=1.0).run(2000).summary()["final_true"]
    down = Market(seed=8, drift=-1.0).run(2000).summary()["final_true"]
    assert up > 10_000 > down


def test_bot_hooks_are_called():
    class Spy:
        ticks = 0
        fills = []

        def on_tick(self, book, t):
            self.ticks += 1
            if t == 1:
                # rest a bid right at the touch so it gets hit
                self.oid = market.send(Side.BUY, book.best_bid, 5)

        def on_trade(self, trade, maker_pid, taker_pid, pre_mid=None):
            if maker_pid == BOT_PID or taker_pid == BOT_PID:
                self.fills.append(trade)

    spy = Spy()
    market = Market(seed=9, bot=spy)
    market.run(500)
    assert spy.ticks == 500
    assert len(spy.fills) > 0
    assert all(t.maker_order_id == spy.oid for t in spy.fills)


def test_cancel_only_touches_noise_orders():
    m = Market(seed=10)
    oid = m.send(Side.BUY, 9_990, 1)
    m.run(300)
    # our order may be filled, but it must never be cancelled by noise logic
    cancels = [e for e in m.book.events if e.order_id == oid and e.type.value == "CANCEL"]
    assert cancels == []


def test_bot_fills_delivered_outside_matching():
    # sending from inside on_trade must be safe now that fills are buffered
    class Bot:
        def __init__(self):
            self.n = 0

        def on_tick(self, book, t):
            if t == 1:
                self.oid = m.send(Side.BUY, book.best_bid, 50)

        def on_trade(self, tr, maker_pid, taker_pid, pre_mid=None):
            self.n += 1
            m.send(Side.SELL, tr.price + 2, 1)

    bot = Bot()
    m = Market(seed=11, bot=bot)
    m.run(300)
    assert bot.n > 0
    b = m.book
    if b.best_bid is not None and b.best_ask is not None:
        assert b.best_bid < b.best_ask


def test_bot_only_hears_its_own_fills():
    seen = []

    class Bot:
        def on_tick(self, book, t):
            if t == 1:
                m.send(Side.BUY, book.best_bid - 1, 3)

        def on_trade(self, tr, maker_pid, taker_pid, pre_mid=None):
            seen.append((maker_pid, taker_pid))

    m = Market(seed=12, bot=Bot())
    m.run(400)
    assert seen
    assert all(BOT_PID in pair for pair in seen)


def test_bot_latency_delays_arrival():
    class Bot:
        def on_tick(self, book, t):
            if t == 5:
                self.oid = m.send(Side.BUY, 9_000, 1)   # deep, won't fill

    bot = Bot()
    m = Market(seed=13, bot=bot, bot_latency=3)
    m.run(5)
    assert m.book.get_order(bot.oid) is None            # not there yet
    m.run(2)
    assert m.book.get_order(bot.oid) is None            # t=7, still in flight
    m.run(1)
    assert m.book.get_order(bot.oid) is not None        # t=8, landed
    m.cancel(bot.oid)
    m.run(2)
    assert m.book.get_order(bot.oid) is not None        # cancel in flight
    m.run(1)
    assert m.book.get_order(bot.oid) is None


def test_bot_cannot_see_true_price_through_book():
    # the book exposes nothing about the hidden process
    m = Market(seed=14).run(10)
    assert not hasattr(m.book, "true_price")


def test_save_history_and_trades(tmp_path):
    m = Market(seed=15).run(200)
    h, t = tmp_path / "h.csv", tmp_path / "t.csv"
    m.save_history(h)
    m.save_trades(t)
    assert len(h.read_text().splitlines()) == 201
    assert len(t.read_text().splitlines()) == len(m.book.trades) + 1
    assert "taker_pid" in t.read_text().splitlines()[0]


def test_gateway_hides_market():
    from lobsim.market import Gateway

    class Bot:
        def attach(self, gw):
            self.gw = gw

        def on_tick(self, book, t):
            pass

        def on_trade(self, *a):
            pass

    bot = Bot()
    m = Market(seed=16, bot=bot)
    assert isinstance(bot.gw, Gateway)
    assert bot.gw.book is m.book
    assert not hasattr(bot.gw, "true_price")
    assert not hasattr(bot.gw, "market")
    with pytest.raises(AttributeError):
        bot.gw.anything = 1          # slots, nothing can be bolted on
    oid = bot.gw.send(Side.BUY, 9_000, 1)
    assert m.book.get_order(oid) is not None


def test_gateway_has_no_handle_on_market():
    from lobsim.market import Gateway
    m = Market(seed=17)
    gw = m.gw
    for name in dir(gw):
        if name.startswith("__"):
            continue
        v = getattr(gw, name)
        assert not isinstance(v, Market), name
    assert gw.book_time == m.t
    m.run(3)
    assert gw.book_time == 3


def test_pre_mid_is_mid_before_the_fill():
    seen = []

    class Bot:
        def on_tick(self, book, t):
            if t == 1:
                self.px = book.best_bid
                m.send(Side.BUY, self.px, 1000)        # huge, will be the best bid for a while

        def on_trade(self, tr, maker_pid, taker_pid, pre_mid=None):
            seen.append((tr, pre_mid))

    m = Market(seed=70, bot=Bot())
    m.run(200)
    assert seen
    for tr, pre_mid in seen:
        # our bid was the best bid when hit, so pre_mid must sit above it
        assert pre_mid is not None and pre_mid > tr.price


def test_gateway_has_no_public_path_to_market():
    # a convention, not a sandbox: no public attribute or __self__ leads to
    # the market. closures can still be dug into, which the README says.
    m = Market(seed=18)
    gw = m.gw
    for name in ("book", "send", "cancel", "latency", "book_time"):
        v = getattr(gw, name)
        assert not isinstance(v, Market)
        assert not isinstance(getattr(v, "__self__", None), Market)


def test_genuine_typeerror_in_bot_is_not_swallowed():
    import pytest
    from lobsim.strategy import NaiveMaker

    class Buggy(NaiveMaker):
        def on_trade(self, tr, maker_pid, taker_pid, pre_mid=None):
            super().on_trade(tr, maker_pid, taker_pid, pre_mid)
            self._log(tr, pre_mid)          # wrong arity

        def _log(self, tr):
            pass

    b = Buggy(requote_always=False, max_position=30)
    with pytest.raises(TypeError, match="_log"):
        Market(seed=5, bot=b).run(300)



# simulator edge cases (extreme params, thin books)

def test_noise_market_orders_are_price_protected():
    # noise "market" orders are IOC limits at mid +/- collar, so a resting
    # quote 200 ticks away can never be swept by noise flow
    m = Market(seed=1)
    far = m.send(Side.SELL, round(m.book.mid) + 200, 5)
    m.run(1500)
    assert all(m.pids[tr.taker_order_id] != NOISE_PID
               for tr in m.book.trades if tr.maker_order_id == far)
    noise_types = {e.order_type for e in m.book.events if m.pids.get(e.order_id) == NOISE_PID}
    from lobsim.orderbook import OrderType
    assert OrderType.MARKET not in noise_types


def test_wide_stub_is_not_a_money_machine():
    from lobsim.analytics import evaluate
    from lobsim.strategy import NaiveMaker
    stub = evaluate(lambda: NaiveMaker(half_spread=200, max_position=30, requote_always=False),
                    range(100, 106), steps=3000)
    tight = evaluate(lambda: NaiveMaker(half_spread=8, max_position=30, requote_always=False),
                     range(100, 106), steps=3000)
    assert stub["total_pnl"][0] < tight["total_pnl"][0]
    assert stub["fills"][0] < 40


def test_low_start_price_does_not_crash():
    for sp in (12, 50, 100, 200):
        for seed in range(5):
            Market(seed=seed, start_price=sp).run(3000)


def test_start_price_too_low_is_a_clear_error():
    with pytest.raises(ValueError, match="start_price"):
        Market(seed=0, start_price=5)


def test_poisson_large_rate_returns():
    m = Market(seed=0)
    xs = [m._poisson(1000) for _ in range(200)]
    assert 950 < sum(xs) / len(xs) < 1050
