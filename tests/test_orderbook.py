import pytest

from lobsim.orderbook import (Event, EventType, Order, OrderBook, OrderType,
                       RejectReason, Side, STPMode)


def lim(oid, side, price, qty, ts=0):
    return Order(order_id=oid, side=side, price=price, qty=qty, timestamp=ts)


def mkt(oid, side, qty, ts=0):
    return Order(order_id=oid, side=side, price=None, qty=qty,
                 timestamp=ts, order_type=OrderType.MARKET)


@pytest.fixture
def book():
    return OrderBook()


# resting and queries

def test_empty_book(book):
    assert book.best_bid is None
    assert book.best_ask is None
    assert book.mid is None
    assert book.spread is None
    assert len(book) == 0


def test_rest_non_crossing_orders(book):
    book.add_limit_order(lim(1, Side.BUY, 99, 10))
    book.add_limit_order(lim(2, Side.SELL, 101, 5))
    assert book.best_bid == 99
    assert book.best_ask == 101
    assert book.mid == 100
    assert book.spread == 2
    assert len(book) == 2


def test_best_prices_sorted_correctly(book):
    book.add_limit_order(lim(1, Side.BUY, 98, 1))
    book.add_limit_order(lim(2, Side.BUY, 99, 1))
    book.add_limit_order(lim(3, Side.BUY, 97, 1))
    book.add_limit_order(lim(4, Side.SELL, 103, 1))
    book.add_limit_order(lim(5, Side.SELL, 101, 1))
    book.add_limit_order(lim(6, Side.SELL, 102, 1))
    assert book.best_bid == 99
    assert book.best_ask == 101
    bids, asks = book.depth(3)
    assert [p for p, _ in bids] == [99, 98, 97]
    assert [p for p, _ in asks] == [101, 102, 103]


def test_depth_aggregates_qty_per_level(book):
    book.add_limit_order(lim(1, Side.BUY, 99, 10))
    book.add_limit_order(lim(2, Side.BUY, 99, 5))
    bids, _ = book.depth()
    assert bids == [(99, 15)]


def test_duplicate_id_rejected(book):
    book.add_limit_order(lim(1, Side.BUY, 99, 1))
    with pytest.raises(ValueError):
        book.add_limit_order(lim(1, Side.BUY, 98, 1))


def test_invalid_orders():
    with pytest.raises(ValueError):
        Order(order_id=1, side=Side.BUY, price=10, qty=0, timestamp=0)
    with pytest.raises(ValueError):
        Order(order_id=1, side=Side.BUY, price=None, qty=1, timestamp=0)


# matching

def test_full_fill_at_resting_price(book):
    book.add_limit_order(lim(1, Side.SELL, 101, 10))
    trades = book.add_limit_order(lim(2, Side.BUY, 105, 10))   # aggressive
    assert len(trades) == 1
    t = trades[0]
    assert t.price == 101            # not 105, taker pays maker's price
    assert t.qty == 10
    assert t.maker_order_id == 1
    assert t.taker_order_id == 2
    assert t.aggressor_side is Side.BUY
    assert len(book) == 0


def test_partial_fill_rests_remainder(book):
    book.add_limit_order(lim(1, Side.SELL, 101, 4))
    trades = book.add_limit_order(lim(2, Side.BUY, 101, 10))
    assert len(trades) == 1 and trades[0].qty == 4
    assert book.best_ask is None
    assert book.best_bid == 101
    assert book.get_order(2).qty == 6
    assert book.get_order(2).filled_qty == 4


def test_partial_fill_of_resting_order(book):
    book.add_limit_order(lim(1, Side.SELL, 101, 10))
    book.add_limit_order(lim(2, Side.BUY, 101, 3))
    assert book.get_order(1).qty == 7
    assert book.asks[101].total_qty == 7


def test_sweeps_multiple_levels(book):
    book.add_limit_order(lim(1, Side.SELL, 101, 5))
    book.add_limit_order(lim(2, Side.SELL, 102, 5))
    book.add_limit_order(lim(3, Side.SELL, 103, 5))
    trades = book.add_limit_order(lim(4, Side.BUY, 102, 12))
    assert [t.price for t in trades] == [101, 102]
    assert [t.qty for t in trades] == [5, 5]
    assert book.best_ask == 103
    assert book.best_bid == 102
    assert book.get_order(4).qty == 2


def test_does_not_match_non_crossing(book):
    book.add_limit_order(lim(1, Side.SELL, 101, 5))
    trades = book.add_limit_order(lim(2, Side.BUY, 100, 5))
    assert trades == []
    assert len(book) == 2


def test_time_priority_within_level(book):
    book.add_limit_order(lim(1, Side.SELL, 101, 5, ts=1))
    book.add_limit_order(lim(2, Side.SELL, 101, 5, ts=2))
    trades = book.add_limit_order(lim(3, Side.BUY, 101, 7))
    assert [(t.maker_order_id, t.qty) for t in trades] == [(1, 5), (2, 2)]
    assert book.get_order(1) is None
    assert book.get_order(2).qty == 3


def test_price_priority_beats_time_priority(book):
    book.add_limit_order(lim(1, Side.SELL, 102, 5, ts=1))   # earlier but worse
    book.add_limit_order(lim(2, Side.SELL, 101, 5, ts=2))
    trades = book.add_limit_order(lim(3, Side.BUY, 105, 5))
    assert trades[0].maker_order_id == 2


def test_sell_side_matching(book):
    book.add_limit_order(lim(1, Side.BUY, 100, 5))
    book.add_limit_order(lim(2, Side.BUY, 99, 5))
    trades = book.add_limit_order(lim(3, Side.SELL, 99, 8))
    assert [t.price for t in trades] == [100, 99]
    assert [t.qty for t in trades] == [5, 3]
    assert book.best_bid == 99
    assert book.bids[99].total_qty == 2


# market orders

def test_market_order_fills_and_never_rests(book):
    book.add_limit_order(lim(1, Side.SELL, 101, 5))
    book.add_limit_order(lim(2, Side.SELL, 103, 5))
    trades = book.add_market_order(mkt(3, Side.BUY, 8))
    assert [(t.price, t.qty) for t in trades] == [(101, 5), (103, 3)]
    assert book.get_order(3) is None
    assert book.asks[103].total_qty == 2


def test_market_order_on_empty_side(book):
    trades = book.add_market_order(mkt(1, Side.BUY, 5))
    assert trades == []
    assert len(book) == 0


def test_market_order_unfilled_remainder_dropped(book):
    book.add_limit_order(lim(1, Side.SELL, 101, 3))
    trades = book.add_market_order(mkt(2, Side.BUY, 10))
    assert sum(t.qty for t in trades) == 3
    assert book.best_ask is None
    assert book.get_order(2) is None


def test_submit_handles_every_type(book):
    book.submit(lim(1, Side.SELL, 101, 5))
    book.add_limit_order(lim(2, Side.BUY, 99, 5))
    book.add_market_order(mkt(3, Side.BUY, 1))
    assert len(book.trades) == 1 and len(book) == 2


# cancel and modify

def test_cancel_removes_order_and_empty_level(book):
    book.add_limit_order(lim(1, Side.BUY, 99, 5))
    book.add_limit_order(lim(2, Side.BUY, 98, 5))
    book.cancel_order(1)
    assert book.best_bid == 98
    assert 99 not in book.bids
    assert book.get_order(1) is None


def test_cancel_keeps_level_if_others_remain(book):
    book.add_limit_order(lim(1, Side.BUY, 99, 5))
    book.add_limit_order(lim(2, Side.BUY, 99, 7))
    book.cancel_order(1)
    assert book.best_bid == 99
    assert book.bids[99].total_qty == 7


def test_cancel_unknown_raises(book):
    with pytest.raises(KeyError):
        book.cancel_order(42)


def test_cancel_filled_order_raises(book):
    book.add_limit_order(lim(1, Side.SELL, 101, 5))
    book.add_limit_order(lim(2, Side.BUY, 101, 5))
    with pytest.raises(KeyError):
        book.cancel_order(1)


def test_modify_reduce_qty_keeps_queue_position(book):
    book.add_limit_order(lim(1, Side.SELL, 101, 5))
    book.add_limit_order(lim(2, Side.SELL, 101, 5))
    book.modify_order(2, new_price=101, new_qty=2, timestamp=5)
    assert book.queue_position(2) == 5         # still behind order 1
    assert book.asks[101].total_qty == 7


def test_modify_increase_qty_loses_queue_position(book):
    book.add_limit_order(lim(1, Side.SELL, 101, 5))
    book.add_limit_order(lim(2, Side.SELL, 101, 5))
    book.modify_order(1, new_price=101, new_qty=9, timestamp=5)
    assert book.queue_position(1) == 5         # now behind order 2
    assert book.queue_position(2) == 0


def test_modify_price_moves_order(book):
    book.add_limit_order(lim(1, Side.SELL, 103, 5))
    book.modify_order(1, new_price=101, new_qty=5, timestamp=5)
    assert book.best_ask == 101
    assert 103 not in book.asks


def test_modify_into_crossing_price_fills(book):
    book.add_limit_order(lim(1, Side.BUY, 99, 5))
    book.add_limit_order(lim(2, Side.SELL, 105, 5))
    trades = book.modify_order(2, new_price=99, new_qty=5, timestamp=5)
    assert len(trades) == 1 and trades[0].price == 99
    assert len(book) == 0


# signals

def test_microprice_and_imbalance(book):
    book.add_limit_order(lim(1, Side.BUY, 99, 30))
    book.add_limit_order(lim(2, Side.SELL, 101, 10))
    # heavy bid -> microprice leans toward ask
    assert book.microprice() == pytest.approx(100.5)
    assert book.imbalance() == pytest.approx(0.5)


def test_event_callback_and_log(book):
    seen = []
    book = OrderBook(on_event=seen.append)
    book.add_limit_order(lim(1, Side.SELL, 101, 5))
    book.add_limit_order(lim(2, Side.BUY, 101, 5))
    assert [e.type for e in seen] == [EventType.ADD, EventType.TRADE]
    assert book.events == seen
    assert seen[1].trade.trade_id == 1
    assert book.last_trade_price == 101


# invariants under random load

def test_random_fuzz_invariants():
    import random
    rng = random.Random(42)
    book = OrderBook()
    live = set()
    oid = 0
    for ts in range(5000):
        r = rng.random()
        if r < 0.6:
            oid += 1
            side = rng.choice([Side.BUY, Side.SELL])
            price = rng.randint(90, 110)
            book.add_limit_order(lim(oid, side, price, rng.randint(1, 20), ts))
            if book.get_order(oid) is not None:
                live.add(oid)
        elif r < 0.75:
            oid += 1
            book.add_market_order(mkt(oid, rng.choice([Side.BUY, Side.SELL]),
                                      rng.randint(1, 30), ts))
        elif live:
            victim = rng.choice(sorted(live))
            if book.get_order(victim) is not None:
                book.cancel_order(victim)
            live.discard(victim)

        # drop ids that got filled by someone else
        live = {i for i in live if book.get_order(i) is not None}

        # invariants
        if book.best_bid is not None and book.best_ask is not None:
            assert book.best_bid < book.best_ask, "book is crossed"
        for p, lvl in book.bids.items():
            assert lvl.total_qty == sum(o.qty for o in lvl)
            assert all(o.side is Side.BUY for o in lvl)
        for p, lvl in book.asks.items():
            assert lvl.total_qty == sum(o.qty for o in lvl)
            assert all(o.side is Side.SELL for o in lvl)
        assert len(book) == sum(len(l) for l in book.bids.values()) + \
                            sum(len(l) for l in book.asks.values())


# advanced order types

def typed(oid, side, price, qty, ot, ts=0, stop=None, pid=None):
    return Order(order_id=oid, side=side, price=price, qty=qty, timestamp=ts,
                 order_type=ot, stop_price=stop, participant_id=pid)


def test_ioc_fills_then_cancels_remainder(book):
    book.submit(lim(1, Side.SELL, 101, 4))
    trades = book.submit(typed(2, Side.BUY, 101, 10, OrderType.IOC))
    assert sum(t.qty for t in trades) == 4
    assert book.get_order(2) is None
    assert book.best_bid is None
    assert book.events[-1].type is EventType.CANCEL


def test_ioc_respects_limit_price(book):
    book.submit(lim(1, Side.SELL, 101, 5))
    book.submit(lim(2, Side.SELL, 103, 5))
    trades = book.submit(typed(3, Side.BUY, 102, 10, OrderType.IOC))
    assert [t.price for t in trades] == [101]
    assert book.asks[103].total_qty == 5


def test_fok_rejects_when_insufficient(book):
    book.submit(lim(1, Side.SELL, 101, 5))
    trades = book.submit(typed(2, Side.BUY, 101, 6, OrderType.FOK))
    assert trades == []
    assert book.asks[101].total_qty == 5           # book untouched
    assert book.events[-1].type is EventType.REJECT
    assert book.events[-1].reason is RejectReason.FOK_UNFILLABLE


def test_fok_rejects_when_liquidity_only_beyond_limit(book):
    book.submit(lim(1, Side.SELL, 101, 3))
    book.submit(lim(2, Side.SELL, 105, 10))
    trades = book.submit(typed(3, Side.BUY, 102, 5, OrderType.FOK))
    assert trades == []
    assert len(book) == 2


def test_fok_fills_fully_across_levels(book):
    book.submit(lim(1, Side.SELL, 101, 3))
    book.submit(lim(2, Side.SELL, 102, 3))
    trades = book.submit(typed(3, Side.BUY, 102, 6, OrderType.FOK))
    assert sum(t.qty for t in trades) == 6
    assert len(book) == 0


def test_post_only_rests_when_passive(book):
    book.submit(lim(1, Side.SELL, 101, 5))
    book.submit(typed(2, Side.BUY, 100, 5, OrderType.POST_ONLY))
    assert book.best_bid == 100


def test_post_only_rejected_when_crossing(book):
    book.submit(lim(1, Side.SELL, 101, 5))
    trades = book.submit(typed(2, Side.BUY, 101, 5, OrderType.POST_ONLY))
    assert trades == []
    assert book.get_order(2) is None
    assert book.events[-1].reason is RejectReason.POST_ONLY_WOULD_CROSS
    assert book.asks[101].total_qty == 5


def test_market_order_no_liquidity_emits_reject(book):
    book.submit(mkt(1, Side.BUY, 5))
    assert book.events[-1].type is EventType.REJECT
    assert book.events[-1].reason is RejectReason.NO_LIQUIDITY


# stop orders

def test_stop_waits_until_triggered(book):
    book.submit(lim(1, Side.SELL, 105, 5))
    book.submit(lim(2, Side.SELL, 103, 5))
    book.submit(lim(3, Side.BUY, 100, 5))
    # buy stop at 104, becomes market once a trade prints >= 104
    book.submit(typed(10, Side.BUY, None, 3, OrderType.STOP, stop=104))
    assert book.pending_stops == 1
    assert book.get_order(10) is not None
    # trade at 103 does not trigger
    book.submit(mkt(4, Side.BUY, 1))
    assert book.pending_stops == 1
    # sweep through 103 into 105 triggers it, stop buys the remaining 105s
    trades = book.submit(typed(5, Side.BUY, 105, 5, OrderType.LIMIT))
    assert book.pending_stops == 0
    assert book.asks[105].total_qty == 1       # 5 - 1 (order 5) - 3 (stop)
    assert any(t.taker_order_id == 10 for t in trades)
    assert book.events[[e.type for e in book.events].index(EventType.TRIGGER)].order_id == 10


def test_sell_stop_limit_rests_at_limit(book):
    book.submit(lim(1, Side.BUY, 100, 2))
    book.submit(lim(2, Side.BUY, 98, 5))
    # sell stop-limit: trigger at 99, limit 98
    book.submit(typed(10, Side.SELL, 98, 10, OrderType.STOP_LIMIT, stop=99))
    book.submit(mkt(3, Side.SELL, 2))          # trades at 100, no trigger
    assert book.pending_stops == 1
    book.submit(lim(4, Side.BUY, 99, 1))
    book.submit(mkt(5, Side.SELL, 1))          # trades at 99, triggers
    assert book.pending_stops == 0
    # stop limit sells 5 at 98 then rests the remaining 5 at 98
    assert book.get_order(10).qty == 5
    assert book.best_ask == 98


def test_stop_cascade(book):
    book.submit(lim(1, Side.SELL, 101, 1))
    book.submit(lim(2, Side.SELL, 102, 1))
    book.submit(lim(3, Side.SELL, 103, 1))
    book.submit(typed(10, Side.BUY, None, 1, OrderType.STOP, stop=101))
    book.submit(typed(11, Side.BUY, None, 1, OrderType.STOP, stop=102))
    book.submit(mkt(4, Side.BUY, 1))           # prints 101 -> stop 10 -> prints 102 -> stop 11
    assert book.pending_stops == 0
    assert [t.price for t in book.trades] == [101, 102, 103]


def test_cancel_pending_stop(book):
    book.submit(typed(10, Side.BUY, None, 1, OrderType.STOP, stop=101))
    book.cancel_order(10)
    assert book.pending_stops == 0
    assert book.get_order(10) is None


def test_invalid_stop_without_stop_price():
    with pytest.raises(ValueError):
        Order(order_id=1, side=Side.BUY, price=None, qty=1, timestamp=0,
              order_type=OrderType.STOP)


# self trade prevention

def test_stp_none_allows_self_trade():
    book = OrderBook(stp_mode=STPMode.NONE)
    book.submit(typed(1, Side.SELL, 101, 5, OrderType.LIMIT, pid=7))
    trades = book.submit(typed(2, Side.BUY, 101, 5, OrderType.LIMIT, pid=7))
    assert len(trades) == 1


def test_stp_cancel_newest():
    book = OrderBook(stp_mode=STPMode.CANCEL_NEWEST)
    book.submit(typed(1, Side.SELL, 101, 5, OrderType.LIMIT, pid=7))
    trades = book.submit(typed(2, Side.BUY, 101, 5, OrderType.LIMIT, pid=7))
    assert trades == []
    assert book.asks[101].total_qty == 5
    assert book.get_order(2) is None
    assert book.events[-1].reason is RejectReason.STP


def test_stp_cancel_oldest_then_continues():
    book = OrderBook(stp_mode=STPMode.CANCEL_OLDEST)
    book.submit(typed(1, Side.SELL, 101, 5, OrderType.LIMIT, pid=7))
    book.submit(typed(2, Side.SELL, 101, 5, OrderType.LIMIT, pid=8))
    trades = book.submit(typed(3, Side.BUY, 101, 5, OrderType.LIMIT, pid=7))
    assert len(trades) == 1 and trades[0].maker_order_id == 2
    assert book.get_order(1) is None
    assert len(book) == 0


def test_stp_cancel_both():
    book = OrderBook(stp_mode=STPMode.CANCEL_BOTH)
    book.submit(typed(1, Side.SELL, 101, 5, OrderType.LIMIT, pid=7))
    trades = book.submit(typed(2, Side.BUY, 101, 5, OrderType.LIMIT, pid=7))
    assert trades == []
    assert len(book) == 0


def test_stp_ignores_orders_without_participant():
    book = OrderBook(stp_mode=STPMode.CANCEL_BOTH)
    book.submit(lim(1, Side.SELL, 101, 5))
    trades = book.submit(lim(2, Side.BUY, 101, 5))
    assert len(trades) == 1


# analytics and replay

def test_volume_at_or_better_and_vwap(book):
    book.submit(lim(1, Side.SELL, 101, 5))
    book.submit(lim(2, Side.SELL, 102, 5))
    book.submit(lim(3, Side.SELL, 104, 5))
    assert book.volume_at_or_better(Side.BUY, 102) == 10
    assert book.volume_at_or_better(Side.BUY, 100) == 0
    assert book.vwap_to_fill(Side.BUY, 8) == pytest.approx((5 * 101 + 3 * 102) / 8)
    assert book.vwap_to_fill(Side.BUY, 100) is None


def test_multi_level_imbalance(book):
    book.submit(lim(1, Side.BUY, 99, 10))
    book.submit(lim(2, Side.BUY, 98, 30))
    book.submit(lim(3, Side.SELL, 101, 10))
    book.submit(lim(4, Side.SELL, 102, 10))
    assert book.imbalance(1) == 0
    assert book.imbalance(2) == pytest.approx((40 - 20) / 60)


def test_snapshot(book):
    book.submit(lim(1, Side.BUY, 99, 10))
    book.submit(lim(2, Side.SELL, 101, 5))
    book.submit(mkt(3, Side.BUY, 1))
    snap = book.snapshot()
    assert snap == {"bids": [(99, 10)], "asks": [(101, 4)], "last": 101}


def test_replay_reconstructs_state():
    import random
    rng = random.Random(7)
    book = OrderBook()
    oid = 0
    for ts in range(2000):
        r = rng.random()
        oid += 1
        if r < 0.5:
            book.submit(lim(oid, rng.choice([Side.BUY, Side.SELL]),
                            rng.randint(95, 105), rng.randint(1, 10), ts))
        elif r < 0.65:
            book.submit(mkt(oid, rng.choice([Side.BUY, Side.SELL]), rng.randint(1, 15), ts))
        elif r < 0.8:
            book.submit(typed(oid, rng.choice([Side.BUY, Side.SELL]),
                              rng.randint(95, 105), rng.randint(1, 10), OrderType.IOC, ts))
        else:
            ids = list(book._orders)
            if ids:
                victim = rng.choice(ids)
                if rng.random() < 0.5:
                    book.cancel_order(victim, timestamp=ts)
                else:
                    o = book.get_order(victim)
                    book.modify_order(victim, o.price, max(1, o.qty - 1), ts)
    rebuilt = OrderBook.replay(book.events)
    assert rebuilt.snapshot() == book.snapshot()
    assert len(rebuilt) == len(book)


def test_cancel_is_o1_unlink(book):
    # middle of a long queue, make sure neighbours relink correctly
    for i in range(1, 6):
        book.submit(lim(i, Side.SELL, 101, i))
    book.cancel_order(3)
    assert [o.order_id for o in book.asks[101]] == [1, 2, 4, 5]
    assert book.asks[101].total_qty == 1 + 2 + 4 + 5
    book.cancel_order(1)
    book.cancel_order(5)
    assert [o.order_id for o in book.asks[101]] == [2, 4]
    assert book.asks[101].head.order_id == 2
    assert book.asks[101].tail.order_id == 4


# bugs found in stage 1, keep them fixed

def test_partial_market_order_remainder_is_cancel_not_reject(book):
    book.submit(lim(1, Side.SELL, 100, 3))
    book.submit(mkt(2, Side.BUY, 10))
    e = book.events[-1]
    assert e.type is EventType.CANCEL
    assert e.reason is RejectReason.NO_LIQUIDITY
    assert e.qty == 7


def test_replay_ignores_pending_stops(book):
    book.submit(lim(1, Side.BUY, 100, 5))
    book.submit(typed(2, Side.SELL, 90, 5, OrderType.STOP_LIMIT, stop=95))
    book.submit(typed(3, Side.BUY, None, 5, OrderType.STOP, stop=110))
    r = OrderBook.replay(book.events)
    assert r.snapshot() == book.snapshot()
    assert r.best_ask is None


def test_fok_is_all_or_nothing_under_stp():
    book = OrderBook(stp_mode=STPMode.CANCEL_NEWEST)
    book.submit(typed(1, Side.SELL, 100, 2, OrderType.LIMIT, pid=9))
    book.submit(typed(2, Side.SELL, 100, 2, OrderType.LIMIT, pid=7))
    book.submit(typed(3, Side.SELL, 100, 2, OrderType.LIMIT, pid=9))
    t = book.submit(typed(4, Side.BUY, 100, 6, OrderType.FOK, pid=7))
    assert t == []
    assert book.asks[100].total_qty == 6
    # under CANCEL_OLDEST the own order is skipped, so 4 lots are reachable
    book2 = OrderBook(stp_mode=STPMode.CANCEL_OLDEST)
    book2.submit(typed(1, Side.SELL, 100, 2, OrderType.LIMIT, pid=9))
    book2.submit(typed(2, Side.SELL, 100, 2, OrderType.LIMIT, pid=7))
    book2.submit(typed(3, Side.SELL, 100, 2, OrderType.LIMIT, pid=9))
    t = book2.submit(typed(4, Side.BUY, 100, 4, OrderType.FOK, pid=7))
    assert sum(x.qty for x in t) == 4
    assert len(book2) == 0


def test_modify_pending_stop_rejected(book):
    book.submit(typed(1, Side.BUY, None, 1, OrderType.STOP, stop=105))
    with pytest.raises(ValueError):
        book.modify_order(1, 100, 1, 1)


def test_negative_price_rejected():
    with pytest.raises(ValueError):
        Order(order_id=1, side=Side.BUY, price=-5, qty=1, timestamp=0)
    with pytest.raises(ValueError):
        Order(order_id=1, side=Side.BUY, price=None, qty=1, timestamp=0,
              order_type=OrderType.STOP, stop_price=0)


# property based: any sequence of ops keeps the invariants and replays exactly

from hypothesis import given, settings
from hypothesis import strategies as st

_side = st.sampled_from([Side.BUY, Side.SELL])
_price = st.integers(95, 105)
_qty = st.integers(1, 10)
_op = st.one_of(
    st.tuples(st.just("L"), _side, _price, _qty),
    st.tuples(st.just("M"), _side, _qty),
    st.tuples(st.just("I"), _side, _price, _qty),
    st.tuples(st.just("F"), _side, _price, _qty),
    st.tuples(st.just("P"), _side, _price, _qty),
    st.tuples(st.just("C"), st.integers(0, 50)),
    st.tuples(st.just("X"), st.integers(0, 50), _price, _qty),
)


def _check_invariants(book):
    if book.best_bid is not None and book.best_ask is not None:
        assert book.best_bid < book.best_ask
    n = 0
    for side_book, side in ((book.bids, Side.BUY), (book.asks, Side.SELL)):
        for p, lvl in side_book.items():
            assert len(lvl) > 0
            assert lvl.total_qty == sum(o.qty for o in lvl)
            assert all(o.side is side and o.price == p and o.qty > 0 for o in lvl)
            n += len(lvl)
    assert n == len(book)


@settings(max_examples=300, deadline=None)
@given(st.lists(_op, min_size=1, max_size=60))
def test_property_invariants_and_replay(ops):
    book = OrderBook()
    live = []
    oid = 0
    for i, op in enumerate(ops):
        oid += 1
        k = op[0]
        if k == "L":
            book.submit(lim(oid, op[1], op[2], op[3], i)); live.append(oid)
        elif k == "M":
            book.submit(mkt(oid, op[1], op[2], i))
        elif k in ("I", "F", "P"):
            ot = {"I": OrderType.IOC, "F": OrderType.FOK, "P": OrderType.POST_ONLY}[k]
            book.submit(typed(oid, op[1], op[2], op[3], ot, ts=i))
            if k == "P":
                live.append(oid)
        elif k == "C" and live:
            v = live[op[1] % len(live)]
            if book.get_order(v) is not None:
                book.cancel_order(v, timestamp=i)
        elif k == "X" and live:
            v = live[op[1] % len(live)]
            if book.get_order(v) is not None:
                book.modify_order(v, op[2], op[3], i)
        _check_invariants(book)

    r = OrderBook.replay(book.events)
    assert r.snapshot() == book.snapshot()
    assert len(r) == len(book)


_stp = st.sampled_from(list(STPMode))
_pid = st.integers(1, 3)
_op2 = st.one_of(
    st.tuples(st.just("L"), _side, _price, _qty, _pid),
    st.tuples(st.just("M"), _side, _qty, _pid),
    st.tuples(st.just("I"), _side, _price, _qty, _pid),
    st.tuples(st.just("F"), _side, _price, _qty, _pid),
    st.tuples(st.just("P"), _side, _price, _qty, _pid),
    st.tuples(st.just("S"), _side, _price, _qty, _pid),          # stop (market on trigger)
    st.tuples(st.just("SL"), _side, _price, _qty, _pid, _price), # stop limit
    st.tuples(st.just("C"), st.integers(0, 50)),
    st.tuples(st.just("X"), st.integers(0, 50), _price, _qty),
)


@settings(max_examples=300, deadline=None)
@given(_stp, st.lists(_op2, min_size=1, max_size=60))
def test_property_with_stops_and_stp(stp, ops):
    book = OrderBook(stp_mode=stp)
    live = []
    pids = {}
    oid = 0
    for i, op in enumerate(ops):
        oid += 1
        k = op[0]
        if k not in ("C", "X"):
            pids[oid] = op[3] if k == "M" else op[4]
        if k == "L":
            book.submit(typed(oid, op[1], op[2], op[3], OrderType.LIMIT, ts=i, pid=op[4])); live.append(oid)
        elif k == "M":
            book.submit(typed(oid, op[1], None, op[2], OrderType.MARKET, ts=i, pid=op[3]))
        elif k in ("I", "F", "P"):
            ot = {"I": OrderType.IOC, "F": OrderType.FOK, "P": OrderType.POST_ONLY}[k]
            book.submit(typed(oid, op[1], op[2], op[3], ot, ts=i, pid=op[4]))
            if k == "P":
                live.append(oid)
        elif k == "S":
            book.submit(typed(oid, op[1], None, op[3], OrderType.STOP, ts=i, stop=op[2], pid=op[4]))
            live.append(oid)
        elif k == "SL":
            book.submit(typed(oid, op[1], op[2], op[3], OrderType.STOP_LIMIT, ts=i, stop=op[5], pid=op[4]))
            live.append(oid)
        elif k == "C" and live:
            v = live[op[1] % len(live)]
            if book.get_order(v) is not None:
                book.cancel_order(v, timestamp=i)
        elif k == "X" and live:
            v = live[op[1] % len(live)]
            if v in book._orders:
                book.modify_order(v, op[2], op[3], i)
        _check_invariants(book)
        # a triggered stop must never still be pending
        last = book.last_trade_price
        if last is not None:
            for o in book._stops.values():
                if o.side is Side.BUY:
                    assert last < o.stop_price
                else:
                    assert last > o.stop_price
    if stp is not STPMode.NONE:
        for tr in book.trades:
            assert pids[tr.maker_order_id] != pids[tr.taker_order_id]
    r = OrderBook.replay(book.events, stp_mode=stp)
    assert r.snapshot() == book.snapshot()
    assert len(r) == len(book)
    # every event sequence number is unique and increasing
    seqs = [e.seq for e in book.events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


# more regressions

def test_stp_cancelled_taker_is_not_marked_filled():
    book = OrderBook(stp_mode=STPMode.CANCEL_NEWEST)
    book.submit(typed(1, Side.SELL, 100, 5, OrderType.LIMIT, pid=7))
    o = typed(2, Side.BUY, 100, 5, OrderType.LIMIT, pid=7)
    t = book.submit(o)
    assert t == []
    assert not o.is_filled and o.filled_qty == 0 and o.qty == 5
    assert o.stp_cancelled
    assert book.get_order(2) is None               # not resting either


def test_order_ids_cannot_be_reused_after_fill_or_cancel(book):
    book.submit(lim(1, Side.SELL, 100, 5))
    book.submit(lim(2, Side.BUY, 100, 5))          # fills 1
    with pytest.raises(ValueError):
        book.submit(lim(1, Side.SELL, 100, 5))
    book.submit(lim(3, Side.SELL, 100, 5))
    book.cancel_order(3)
    with pytest.raises(ValueError):
        book.submit(lim(3, Side.SELL, 100, 5))


def test_modify_keeps_order_type_post_only_rejected_if_crossing(book):
    book.submit(lim(1, Side.SELL, 101, 5))
    book.submit(typed(2, Side.BUY, 99, 5, OrderType.POST_ONLY))
    trades = book.modify_order(2, new_price=101, new_qty=5, timestamp=1)
    assert trades == []
    assert book.get_order(2) is None
    assert book.events[-1].type is EventType.REJECT
    assert book.events[-1].reason is RejectReason.POST_ONLY_WOULD_CROSS
    assert book.asks[101].total_qty == 5
    # a plain limit modified to cross does trade
    book.submit(lim(3, Side.BUY, 99, 5))
    assert book.modify_order(3, new_price=101, new_qty=5, timestamp=2)[0].price == 101
