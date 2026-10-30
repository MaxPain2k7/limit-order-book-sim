import pytest
from lobsim.market import INFORMED_PID, NOISE_PID, Market
from lobsim.orderbook import Side
from lobsim.strategy import Blotter, NaiveMaker
from lobsim.market import Market


def test_blotter_spread_and_inventory_split():
    b = Blotter()
    b.on_fill(1, Side.BUY, 97, 10, NOISE_PID, mid=100)      # 3 ticks edge
    b.on_fill(2, Side.SELL, 103, 10, NOISE_PID, mid=100)    # 3 ticks edge
    assert b.position == 0
    assert b.spread_pnl == 60
    assert b.mark(3, 100) == 60                               # cash only
    assert b.curve[-1][4] == 0                                # no inventory pnl


def test_blotter_inventory_pnl_moves_with_mid():
    b = Blotter()
    b.on_fill(1, Side.BUY, 99, 10, INFORMED_PID, mid=100)     # +10 spread
    b.mark(2, 100)
    assert b.curve[-1][2] == 10
    b.mark(3, 90)                                              # mid falls, we are long
    assert b.curve[-1][2] == -90
    assert b.curve[-1][3] == 10                                # spread untouched
    assert b.curve[-1][4] == -100                              # inventory ate it
    assert b.n_informed == 1 and b.n_noise == 0


def test_blotter_flip_through_zero():
    b = Blotter()
    b.on_fill(1, Side.BUY, 100, 5, NOISE_PID, mid=100)
    b.on_fill(2, Side.SELL, 100, 8, NOISE_PID, mid=100)
    assert b.position == -3
    assert b.cash == 300
    assert b.mark(3, 100) == 0


def test_naive_maker_quotes_both_sides_each_tick():
    bot = NaiveMaker(half_spread=3, size=5)
    m = Market(seed=1, bot=bot)
    m.run(50)
    assert bot.bid_id is not None and bot.ask_id is not None
    bid, ask = m.book.get_order(bot.bid_id), m.book.get_order(bot.ask_id)
    # one may have been filled already, any that rest must be at mid +/- 3
    mid = m.history[-1][2]
    if bid is not None:
        assert bid.price == round(mid) - 3
    if ask is not None:
        assert ask.price == round(mid) + 3


def test_naive_maker_only_ever_makes():
    bot = NaiveMaker()
    m = Market(seed=2, bot=bot).run(1000)
    assert m.bot_trades == 0                       # never the aggressor
    assert bot.blotter.fills


def test_naive_maker_bleeds_in_a_trend():
    flat = NaiveMaker(); Market(seed=3, bot=flat).run(2000)
    trend = NaiveMaker(); Market(seed=3, drift=0.5, bot=trend).run(2000)
    assert trend.blotter.summary()["total_pnl"] < flat.blotter.summary()["total_pnl"]
    assert trend.blotter.position < -100           # short into a rally


def test_position_limit_is_respected():
    bot = NaiveMaker(size=5, max_position=20)
    Market(seed=4, drift=0.5, bot=bot).run(2000)
    assert bot.blotter.summary()["max_abs_position"] < 20 + 5


def test_informed_fills_mark_out_worse_than_noise():
    bot = NaiveMaker()
    m = Market(seed=5, bot=bot).run(3000)
    mids = {}
    last = None
    for t, _, mid, _, _ in m.history:
        last = mid if mid is not None else last
        mids[t] = last

    def markout(filter_pid, h=20):
        vals = []
        for t, side, px, qty, pid, _ in bot.blotter.fills:
            if (pid == INFORMED_PID) != filter_pid:
                continue
            mm = mids.get(t + h)
            if mm is None:
                continue
            vals.append((mm - px) if side is Side.BUY else (px - mm))
        return sum(vals) / len(vals)

    assert markout(True) < markout(False)
    assert markout(True) < 0


def test_position_cap_holds_under_latency():
    for seed in range(20, 26):
        bot = NaiveMaker(size=5, max_position=20, requote_always=False)
        Market(seed=seed, drift=0.5, bot=bot, bot_latency=3).run(2000)
        assert bot.blotter.summary()["max_abs_position"] <= 20, seed


def test_position_cap_exact_without_latency():
    bot = NaiveMaker(size=5, max_position=20)
    Market(seed=4, drift=0.5, bot=bot).run(2000)
    assert bot.blotter.summary()["max_abs_position"] <= 20


def test_fees_and_rebates():
    b = Blotter(maker_fee=-0.01, taker_fee=0.02)
    b.on_fill(1, Side.BUY, 100, 10, NOISE_PID, mid=100, as_maker=True)
    assert b.fees_paid == -10                   # rebate
    b.on_fill(2, Side.SELL, 100, 10, NOISE_PID, mid=100, as_maker=False)
    assert b.fees_paid == 10                    # 20 taker fee minus 10 rebate
    assert b.mark(3, 100) == -10
    s = b.summary()
    assert s["spread_pnl"] == 0 and s["inventory_pnl"] == 0 and s["fees_paid"] == 10


def test_requote_on_change_keeps_queue_position():
    bot = NaiveMaker(requote_always=False)
    m = Market(seed=6, bot=bot)
    m.run(1)
    first_bid = bot.bid_id
    # keep stepping while the mid stays put; the order id must not change
    for _ in range(50):
        mid_before = m.book.mid
        m.step()
        if m.book.mid == mid_before and m.book.get_order(first_bid) is not None:
            assert bot.bid_id == first_bid
        else:
            break


# avellaneda-Stoikov maker

from lobsim.strategy import InventoryMaker


def _warm(bot, steps=300, seed=30, **kw):
    m = Market(seed=seed, bot=bot, **kw)
    m.run(steps)
    return m


def test_as_skews_against_inventory():
    bot = InventoryMaker(gamma=0.1, spread_mode="fixed", requote_always=True)
    m = _warm(bot)
    mid = m.book.mid
    # force a long position and requote
    bot.blotter.position = 25
    bot.on_tick(m.book, m.t + 1)
    assert bot.reservation < mid
    assert bot.skew > 0
    long_bid, long_ask = bot.bid_px, bot.ask_px
    bot.blotter.position = -25
    bot.on_tick(m.book, m.t + 2)
    assert bot.reservation > mid
    assert bot.bid_px > long_bid and bot.ask_px > long_ask


def test_as_flat_inventory_quotes_symmetric():
    bot = InventoryMaker(gamma=0.1, spread_mode="fixed", half_spread=3, requote_always=True)
    m = _warm(bot)
    bot.blotter.position = 0
    mid = m.book.mid                     # before our own quotes can move the touch
    bot.on_tick(m.book, m.t + 1)
    assert bot.skew == 0
    assert bot.ask_px - bot.bid_px == 6
    # integer prices around a half tick mid can be off by at most one tick
    assert abs((bot.ask_px - mid) - (mid - bot.bid_px)) <= 1


def test_as_skew_is_clamped():
    bot = InventoryMaker(gamma=10.0, spread_mode="fixed", max_skew=7, requote_always=True)
    m = _warm(bot)
    bot.blotter.position = 30
    bot.on_tick(m.book, m.t + 1)
    assert bot.skew == 7
    assert bot.bid_px >= 1


def test_as_never_crosses_the_book():
    bot = InventoryMaker(gamma=0.5, spread_mode="fixed", half_spread=1, requote_always=True)
    m = Market(seed=31, drift=0.6, bot=bot)
    m.run(1500)
    rejects = [e for e in m.book.events
               if e.type.value == "REJECT" and m.pids.get(e.order_id) == 99]
    assert rejects == []
    assert m.bot_trades == 0


def test_sigma_estimator_tracks_volatility():
    lo = InventoryMaker(spread_mode="fixed"); _warm(lo, 800, vol=1.0)
    hi = InventoryMaker(spread_mode="fixed"); _warm(hi, 800, vol=4.0)
    assert hi.sigma > lo.sigma * 1.5


def test_kappa_estimator_fits_from_tape():
    bot = InventoryMaker()
    _warm(bot, 2000)
    assert 0.05 < bot.kappa < 5.0
    assert bot.kappa != 0.5 or bot.kappa_fixed is not None   # it moved off the default
    assert bot.half >= 1


def test_as_beats_naive_on_inventory_control():
    seeds = range(40, 46)
    for drift in (0.0, 0.4):
        kw = {"drift": drift} if drift else {}
        nv = [NaiveMaker(requote_always=False, max_position=30) for _ in seeds]
        asb = [InventoryMaker(gamma=0.002, spread_mode="fixed", requote_always=False, max_position=30) for _ in seeds]
        for s, a, b in zip(seeds, nv, asb):
            Market(seed=s, bot=a, **kw).run(3000)
            Market(seed=s, bot=b, **kw).run(3000)
        from lobsim.analytics import risk_summary
        nv_s = [risk_summary(b.blotter) for b in nv]
        as_s = [risk_summary(b.blotter) for b in asb]
        mean = lambda rows, k: sum(r[k] for r in rows) / len(rows)
        assert mean(as_s, "pos_std") < mean(nv_s, "pos_std") * 0.6
        assert mean(as_s, "max_drawdown") > mean(nv_s, "max_drawdown")   # less negative
        assert mean(as_s, "total_pnl") > mean(nv_s, "total_pnl")


def test_as_cap_holds_under_latency():
    for seed in range(50, 54):
        bot = InventoryMaker(gamma=0.002, spread_mode="fixed", size=5, max_position=20, requote_always=False)
        Market(seed=seed, drift=0.5, bot=bot, bot_latency=3).run(1500)
        assert bot.blotter.summary()["max_abs_position"] <= 20, seed


def test_fixed_mode_quotes_same_spread_as_naive():
    import statistics
    bot = InventoryMaker(spread_mode="fixed", gamma=1e-6, half_spread=3, requote_always=True)
    m = Market(seed=60, bot=bot)
    sp = []
    for _ in range(1500):
        m.step()
        if bot.bid_px and bot.ask_px:
            sp.append(bot.ask_px - bot.bid_px)
    assert statistics.mean(sp) == 6


def test_sigma_is_robust_to_dislocations():
    bot = InventoryMaker(spread_mode="fixed")
    Market(seed=4, drift=0.5, bot=bot).run(2000)
    assert bot.sigma < 3          # true per step vol is 2, a plain std gave 19 here


def test_glft_fits_A_and_quotes_sensibly():
    bot = InventoryMaker(spread_mode="glft", gamma=0.1)
    Market(seed=61, bot=bot).run(3000)
    assert bot.A is not None and bot.A > 0
    half, skew = bot._glft(bot.gamma, bot.kappa, bot.A, bot.sigma)
    assert half > 0 and skew > 0
    assert bot.half >= bot.min_half_spread
    # more risk aversion -> wider half spread and more skew
    h2, s2 = bot._glft(bot.gamma * 10, bot.kappa, bot.A, bot.sigma)
    assert h2 > half and s2 > skew


def test_glft_controls_inventory_vs_naive():
    nv = NaiveMaker(requote_always=False, max_position=30)
    gl = InventoryMaker(spread_mode="glft", gamma=0.05, requote_always=False, max_position=30)
    Market(seed=62, drift=0.4, bot=nv).run(3000)
    Market(seed=62, drift=0.4, bot=gl).run(3000)
    from lobsim.analytics import risk_summary
    assert risk_summary(gl.blotter)["pos_std"] < risk_summary(nv.blotter)["pos_std"]
    assert risk_summary(gl.blotter)["total_pnl"] > risk_summary(nv.blotter)["total_pnl"]


def test_time_at_cap_metric():
    from lobsim.analytics import risk_summary
    nv = NaiveMaker(requote_always=False, max_position=30)
    Market(seed=5, drift=0.4, bot=nv).run(3000)
    s = risk_summary(nv.blotter, cap=30)
    assert 0 < s["time_at_cap"] < 1
    assert "time_at_cap" not in risk_summary(nv.blotter)


def test_requote_on_change_keeps_orders_under_latency():
    # the keep test must treat an in-flight order as live, otherwise every
    # quote is cancelled before it even lands
    bot = NaiveMaker(size=5, max_position=30, requote_always=False)
    m = Market(seed=4, bot=bot, bot_latency=3)
    m.run(1000)
    sent = sum(1 for e in m.book.events if e.type.value == "ADD" and m.pids.get(e.order_id) == 99)
    bot2 = NaiveMaker(size=5, max_position=30, requote_always=True)
    m2 = Market(seed=4, bot=bot2, bot_latency=3)
    m2.run(1000)
    sent2 = sum(1 for e in m2.book.events if e.type.value == "ADD" and m2.pids.get(e.order_id) == 99)
    assert sent < 0.7 * sent2


def test_glft_c2_matches_reference_formula():
    import math
    g, k, A, sig = 0.05, 0.529, 1.201, 1.099
    half, skew = InventoryMaker._glft(g, k, A, sig)
    xd = g
    c1 = math.log(1 + xd / k) / xd
    c2 = math.sqrt(g / (2 * A * k) * (1 + xd / k) ** (k / xd + 1))
    assert half == pytest.approx(c1 + 0.5 * c2 * sig)
    assert skew == pytest.approx(c2 * sig)


def test_A_fit_uses_time_window_not_count():
    hi = InventoryMaker(spread_mode="glft"); Market(seed=63, noise_rate=12, bot=hi).run(2500)
    lo = InventoryMaker(spread_mode="glft"); Market(seed=63, noise_rate=4, bot=lo).run(2500)
    assert hi.A > lo.A * 1.5
    assert lo.A < 5           # a few trades per step at the mid, not tens


def test_three_arg_on_trade_still_works():
    seen = []

    class OldBot:
        def on_tick(self, book, t):
            if t == 1:
                m.send(Side.BUY, book.best_bid, 5)

        def on_trade(self, tr, maker_pid, taker_pid):
            seen.append(tr)

    m = Market(seed=64, bot=OldBot())
    m.run(300)
    assert seen


def test_paired_compare_shape_and_sign():
    from lobsim.analytics import paired_compare
    naive = lambda: NaiveMaker(requote_always=False, max_position=30)
    asb = lambda: InventoryMaker(gamma=0.002, spread_mode="fixed", requote_always=False, max_position=30)
    r = paired_compare(naive, asb, range(100, 104), steps=2000, drift=0.4)
    assert set(r) == {"total_pnl", "max_drawdown", "pos_std"}
    assert r["total_pnl"]["mean_diff"] > 0 and r["total_pnl"]["wins"] == "4/4"
    assert r["pos_std"]["mean_diff"] < 0 and r["pos_std"]["wins"] == "4/4"
    assert r["max_drawdown"]["t"] > 0


def test_paired_compare_identical_bots_is_zero():
    from lobsim.analytics import paired_compare
    mk = lambda: NaiveMaker(requote_always=False, max_position=30)
    r = paired_compare(mk, mk, range(1, 4), steps=500)
    for v in r.values():
        assert v["mean_diff"] == 0 and v["sd_diff"] == 0
        assert v["t"] == 0 and v["tied"] == 3 and v["better"] == 0


def test_glft_never_runs_as_skew_during_warmup():
    bot = InventoryMaker(spread_mode="glft", gamma=0.5, max_skew=15, requote_always=False, max_position=30)
    m = Market(seed=100, bot=bot)
    pinned = 0
    for _ in range(600):
        m.step()
        if bot.A is None:
            assert bot.skew == 0.0            # naive warm-up quotes, no skew
        elif abs(bot.skew) >= 15:
            pinned += 1
    assert bot.A is not None and m.t >= 100
    assert pinned < 30


def test_glft_fits_A_early():
    bot = InventoryMaker(spread_mode="glft")
    m = Market(seed=101, bot=bot)
    m.run(150)
    assert bot.A is not None


def test_as_mode_includes_inventory_term_in_spread():
    import math
    bot = InventoryMaker(spread_mode="as", gamma=0.01, kappa=0.5, horizon=50)
    Market(seed=102, bot=bot).run(400)
    expect = 0.01 * bot.sigma ** 2 * 50 / 2 + math.log(1 + 0.01 / 0.5) / 0.01
    assert bot.half == pytest.approx(max(bot.min_half_spread, expect))



# strategy and analytics edge cases

def test_naive_never_sends_nonpositive_price():
    Market(seed=0, start_price=15, bot=NaiveMaker(half_spread=15)).run(500)
    Market(seed=0, start_price=15, bot=InventoryMaker(spread_mode="glft", gamma=0.1, half_spread=15)).run(500)


def test_glft_with_fixed_kappa_still_fits_A():
    bot = InventoryMaker(spread_mode="glft", gamma=0.1, kappa=0.5, requote_always=False, max_position=30)
    Market(seed=1, bot=bot).run(1000)
    assert bot.A is not None and bot.A > 0
    assert bot.kappa == 0.5
    assert bot.half is not None


def test_open_orders_map_is_bounded_without_cap():
    bot = NaiveMaker()
    Market(seed=0, bot=bot).run(5000)
    assert len(bot.open) < 20


def test_parameter_validation():
    for kw in ({"gamma": 0}, {"gamma": -0.1}, {"kappa": 0}, {"kappa": -1},
               {"sigma_window": 1}, {"spread_mode": "nope"}):
        with pytest.raises(ValueError):
            InventoryMaker(**kw)


def test_half_spread_is_clamped():
    bot = InventoryMaker(spread_mode="as", gamma=0.06, max_half_spread=25)
    Market(seed=3, vol=10, bot=bot).run(800)
    assert bot.half <= 25


def test_markouts_are_lot_weighted():
    from lobsim.analytics import markouts
    from lobsim.market import NOISE_PID
    hist = [(1, 0, 100.0, 99, 101), (2, 0, 101.0, 100, 102)]
    fills = [(1, Side.BUY, 100, 1, NOISE_PID, 100.0), (1, Side.BUY, 101, 9, NOISE_PID, 100.0)]
    _, mo = markouts(fills, hist, horizons=(1,))
    assert mo["all"][0] == pytest.approx((1 * 1 + 9 * 0) / 10)


def test_paired_compare_directions_and_nan():
    from lobsim.analytics import paired_compare
    nv = lambda: NaiveMaker(requote_always=False, max_position=30)
    asb = lambda: InventoryMaker(gamma=0.002, spread_mode="fixed", requote_always=False, max_position=30)
    r = paired_compare(nv, asb, range(100, 103), steps=2000,
                       metrics=("time_at_cap", "avg_abs_position"), drift=0.4)
    assert r["time_at_cap"]["better"] == 3
    assert r["avg_abs_position"]["better"] == 3
    with pytest.raises(ValueError):
        paired_compare(nv, asb, range(1), steps=200, metrics=("fills",))
    quiet = lambda: NaiveMaker(half_spread=50, max_position=30)
    r = paired_compare(quiet, quiet, range(3), steps=300, metrics=("sharpe_100",),
                       noise_mkt_frac=0, informed_rate=0)
    assert r["sharpe_100"]["n"] <= 3
    with pytest.raises(KeyError):
        paired_compare(lambda: NaiveMaker(), asb, range(1), steps=200, metrics=("time_at_cap",))


# backtest pitfalls

def test_liquidation_crosses_the_book_and_pays_taker_fee():
    from lobsim.orderbook import OrderBook, Order
    book = OrderBook()
    book.submit(Order(1, Side.BUY, 99, 3, 0))
    book.submit(Order(2, Side.BUY, 97, 10, 0))
    book.submit(Order(3, Side.SELL, 101, 5, 0))
    b = Blotter(taker_fee=0.001)
    b.on_fill(1, Side.BUY, 100, 5, NOISE_PID, mid=100)
    b.mark(2, book.mid)
    liq = b.liquidate(book)
    # sell 5: 3 @ 99 + 2 @ 97 = 491, fee 0.491, cash -500
    assert liq == pytest.approx(-500 + 491 - 0.491)
    assert liq < b.curve[-1][2]            # worse than mark to mid
    assert "liq_pnl" in b.summary()


def test_liquidation_flat_position_equals_mark():
    b = Blotter()
    b.on_fill(1, Side.BUY, 100, 5, NOISE_PID, mid=100)
    b.on_fill(2, Side.SELL, 102, 5, NOISE_PID, mid=101)
    b.mark(3, 101)
    from lobsim.orderbook import OrderBook
    assert b.liquidate(OrderBook()) == b.curve[-1][2]


def test_liquidation_thin_book_does_not_crash():
    from lobsim.orderbook import OrderBook, Order
    book = OrderBook()
    book.submit(Order(1, Side.BUY, 99, 1, 0))
    book.submit(Order(2, Side.SELL, 101, 1, 0))
    b = Blotter()
    b.on_fill(1, Side.BUY, 100, 50, NOISE_PID, mid=100)
    b.mark(2, 100)
    assert b.liquidate(book) < b.curve[-1][2]


def test_fair_edge_uses_true_price():
    from lobsim.analytics import fair_edge
    hist = [(1, 105.0, 100.0, 99, 101)]
    fills = [(1, Side.BUY, 99, 2, NOISE_PID, 100.0)]
    assert fair_edge(fills, hist) == 12.0       # bought 6 below fair, 2 lots


def test_fat_tails_and_vol_regimes():
    import statistics
    def steps(**kw):
        m = Market(seed=1, jump_prob=0, **kw)
        xs = []
        for _ in range(20000):
            a = m.true_price
            m._move_true_price()
            xs.append(m.true_price - a)
        return xs
    def kurt(xs):
        mu, sd = statistics.mean(xs), statistics.pstdev(xs)
        return sum((x - mu) ** 4 for x in xs) / len(xs) / sd ** 4
    g, t, r = steps(), steps(tail_df=4), steps(vol_regime=(3, 0.01))
    assert 2.7 < kurt(g) < 3.3
    assert kurt(t) > 6
    for xs in (g, t, r):
        assert 1.7 < statistics.pstdev(xs) < 2.4      # same overall scale
    ar = [abs(x) for x in r]
    mu = statistics.mean(ar)
    ac = sum((ar[i] - mu) * (ar[i - 1] - mu) for i in range(1, len(ar))) / sum((x - mu) ** 2 for x in ar)
    assert ac > 0.1                                    # volatility clustering
    with pytest.raises(ValueError):
        Market(seed=0, tail_df=2)


def test_evaluate_reports_liq_and_fair_edge():
    from lobsim.analytics import evaluate
    r = evaluate(lambda: NaiveMaker(requote_always=False, max_position=30), range(2), steps=800)
    assert "liq_pnl" in r and "fair_edge" in r


def test_liquidation_with_empty_far_side_is_bounded():
    # regression: a one-sided book at the end made the old fallback buy back
    # a short at ~2 ticks and book +287k
    from lobsim.orderbook import OrderBook, Order
    b = Blotter()
    b.on_fill(1, Side.SELL, 10_600, 28, NOISE_PID, mid=10_600)
    b.mark(2, 10_600)
    book = OrderBook()
    book.submit(Order(1, Side.BUY, 10_594, 5, 0))       # bids only, asks empty
    b.mark(3, book.mid)                                  # mid is None, keeps last
    liq = b.liquidate(book, penalty_ticks=10)
    assert liq == pytest.approx(28 * 10_600 - 28 * 10_610)
    assert liq < b.curve[-1][2]


def test_liquidation_never_far_from_mark_in_sim():
    for s in (1021, 1000, 1005):
        for d in (0.2, -0.4):
            bot = NaiveMaker(half_spread=8, requote_always=False, max_position=30)
            m = Market(seed=s, drift=d, bot=bot).run(4000)
            mark = bot.blotter.curve[-1][2]
            liq = bot.blotter.liquidate(m.book)
            assert liq <= mark + 1e-9
            assert mark - liq < 30 * 40                  # at most ~40 ticks a lot


# stage 5, DefensiveMaker

from lobsim.strategy import DefensiveMaker


def test_defensive_with_defaults_equals_as():
    kw = dict(gamma=0.002, spread_mode="fixed", requote_always=False, max_position=30)
    a, d = InventoryMaker(**kw), DefensiveMaker(**kw)
    Market(seed=110, drift=0.4, bot=a).run(1500)
    Market(seed=110, drift=0.4, bot=d).run(1500)
    assert a.blotter.curve == d.blotter.curve


def test_defensive_uses_no_counterparty_ids():
    import inspect
    src = inspect.getsource(DefensiveMaker)
    body = src.split("def on_trade")[0]           # decision code, before fill bookkeeping
    assert "INFORMED_PID" not in src and "NOISE_PID" not in src
    assert "taker_pid" not in body and "maker_pid" not in body


def test_requote_tol_keeps_only_safer_quotes():
    bot = NaiveMaker(requote_always=False, requote_tol=2)
    m = Market(seed=111, bot=bot)
    m.run(50)
    oid, px = bot.bid_id, bot.bid_px
    if m.book.get_order(oid) is not None:
        bot._quote(Side.BUY, px + 1)               # target moved up: ours is 1 behind, keep
        assert bot.bid_id == oid
        bot._quote(Side.BUY, px - 1)               # target moved down: ours now too aggressive
        assert bot.bid_id != oid


def test_requote_tol_reduces_cancels_under_latency():
    def cancels(tol):
        bot = DefensiveMaker(gamma=0.002, spread_mode="fixed", requote_always=False,
                             max_position=30, requote_tol=tol)
        m = Market(seed=112, bot=bot, bot_latency=3)
        m.run(1500)
        return sum(1 for e in m.book.events if e.type.value == "CANCEL" and m.pids.get(e.order_id) == 99)
    assert cancels(2) < 0.8 * cancels(0)


def test_imbalance_excludes_own_quotes():
    from lobsim.orderbook import Order
    bot = DefensiveMaker(imb_coef=1.0, imb_levels=1, spread_mode="fixed")
    m = Market(seed=113, bot=bot)
    m.run(5)
    b = m.book
    bb = b.best_bid
    big = m.send(Side.BUY, bb, 500)               # our own huge bid at the touch
    bot.open[big] = (Side.BUY, 500, m.t)
    with_own = b.imbalance(1)
    ex_own = bot._imbalance_ex_own(b)
    assert with_own > 0.5
    assert ex_own < with_own - 0.3


def test_flow_signal_sign():
    bot = DefensiveMaker(flow_coef=1.0, flow_window=5)
    m = Market(seed=114, bot=bot)
    m.run(10)
    from lobsim.orderbook import OrderType
    px = round(m.book.mid) + 20
    for _ in range(5):
        m._send(Side.BUY, px, 50, 2, ot=OrderType.IOC)
    assert bot._flow(m.book, m.t) > 0.3


def test_toxicity_widens_the_losing_side():
    bot = DefensiveMaker(tox_gain=1.0, tox_h=2, tox_alpha=1.0, spread_mode="fixed")
    m = Market(seed=115, bot=bot)
    m.run(400)
    bot.tox = {Side.BUY: -4.0, Side.SELL: +1.0}
    wb, wa = bot._widen(m.book, m.t)
    assert wb == 4.0 and wa == 0.0
    bot.tox = {Side.BUY: -50.0, Side.SELL: 0.0}
    assert bot._widen(m.book, m.t)[0] == bot.tox_max


def test_toxicity_ewma_tracks_markouts():
    bot = DefensiveMaker(tox_gain=1.0, tox_h=5, tox_alpha=0.5, spread_mode="fixed")
    m = Market(seed=116, drift=1.0, bot=bot)       # strong uptrend: sells mark out badly
    m.run(1500)
    assert bot.tox[Side.SELL] < bot.tox[Side.BUY]


def test_defensive_validation():
    with pytest.raises(ValueError):
        DefensiveMaker(tox_alpha=0)
