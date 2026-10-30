"""
Limit order book with price-time priority matching.

Design
------
- Each side is a sorted dict of price -> PriceLevel. Bids sorted descending,
  asks ascending, so index 0 is always the best price.
- Each PriceLevel is an intrusive doubly linked list of resting orders.
  Earliest order at a price gets filled first (time priority). Because every
  Order knows its prev/next neighbours, unlinking on cancel is O(1) instead of
  O(n). Removing a level that empties is O(log levels) in the sorted map.
- Orders are looked up by id in O(1) for cancels and modifies.
- Every state change emits an Event (ADD, CANCEL, MODIFY, TRADE, REJECT).
  The book keeps the event log so any state can be replayed deterministically,
  and the caller can also pass a callback to react live.

Order types
-----------
- LIMIT          match what crosses, rest the remainder
- MARKET         match against the book, never rest, drop the remainder
- IOC            like LIMIT but the remainder is cancelled instead of resting
- FOK            fill the whole qty at or better than the limit, or do nothing
- POST_ONLY      rest only. If it would cross it is rejected, so you are
                 guaranteed maker fees (and a known queue position)
- STOP           dormant until the last trade price touches the stop price,
                 then it becomes a MARKET order
- STOP_LIMIT     same trigger, becomes a LIMIT order at the limit price

Self-trade prevention
---------------------
Orders carry an optional participant_id. When an incoming order would match
a resting order from the same participant, the book applies one of:
- CANCEL_NEWEST   the incoming order is cancelled at that point
- CANCEL_OLDEST   the resting order is cancelled and matching continues
- CANCEL_BOTH     both are cancelled
Real exchanges (CME, Coinbase, Binance) all expose variants of this.

Prices and quantities are ints. Price is in ticks, qty is in lots. This
avoids float comparison bugs and matches how real matching engines work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from sortedcontainers import SortedDict


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"

    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    IOC = "IOC"
    FOK = "FOK"
    POST_ONLY = "POST_ONLY"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class STPMode(Enum):
    NONE = "NONE"
    CANCEL_NEWEST = "CANCEL_NEWEST"
    CANCEL_OLDEST = "CANCEL_OLDEST"
    CANCEL_BOTH = "CANCEL_BOTH"


class EventType(Enum):
    ADD = "ADD"            # order rested on the book
    TRADE = "TRADE"
    CANCEL = "CANCEL"      # resting order removed (user cancel, IOC remainder, STP)
    MODIFY = "MODIFY"
    REJECT = "REJECT"      # order never touched the book (FOK fail, POST_ONLY cross)
    TRIGGER = "TRIGGER"    # stop order activated


class RejectReason(Enum):
    FOK_UNFILLABLE = "FOK_UNFILLABLE"
    POST_ONLY_WOULD_CROSS = "POST_ONLY_WOULD_CROSS"
    STP = "STP"
    NO_LIQUIDITY = "NO_LIQUIDITY"


_STOP_TYPES = {OrderType.STOP, OrderType.STOP_LIMIT}
_NEEDS_PRICE = {OrderType.LIMIT, OrderType.IOC, OrderType.FOK,
                OrderType.POST_ONLY, OrderType.STOP_LIMIT}


@dataclass(eq=False)
class Order:
    order_id: int
    side: Side
    price: Optional[int]          # limit price. None for MARKET / STOP
    qty: int                      # remaining quantity
    timestamp: int
    order_type: OrderType = OrderType.LIMIT
    stop_price: Optional[int] = None
    participant_id: Optional[int] = None
    original_qty: int = field(init=False)

    stp_cancelled: bool = field(default=False, init=False)

    # intrusive linked list pointers, managed by PriceLevel only
    _prev: Optional["Order"] = field(default=None, init=False, repr=False)
    _next: Optional["Order"] = field(default=None, init=False, repr=False)
    _level: Optional["PriceLevel"] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("qty must be positive")
        if self.order_type in _NEEDS_PRICE and self.price is None:
            raise ValueError(f"{self.order_type.value} order needs a price")
        if self.order_type in _STOP_TYPES and self.stop_price is None:
            raise ValueError("stop order needs a stop_price")
        for p in (self.price, self.stop_price):
            if p is not None and p <= 0:
                raise ValueError("prices must be positive")
        self.original_qty = self.qty

    @property
    def filled_qty(self) -> int:
        return self.original_qty - self.qty

    @property
    def is_filled(self) -> bool:
        return self.qty == 0

    @property
    def is_resting(self) -> bool:
        return self._level is not None


@dataclass(frozen=True)
class Trade:
    trade_id: int
    price: int
    qty: int
    timestamp: int
    maker_order_id: int
    taker_order_id: int
    aggressor_side: Side


@dataclass(frozen=True)
class Event:
    seq: int
    timestamp: int
    type: EventType
    order_id: int
    side: Optional[Side] = None
    price: Optional[int] = None
    qty: Optional[int] = None
    trade: Optional[Trade] = None
    reason: Optional[RejectReason] = None
    order_type: Optional[OrderType] = None


class PriceLevel:
    """Intrusive doubly linked FIFO of orders resting at one price."""

    __slots__ = ("price", "head", "tail", "total_qty", "count")

    def __init__(self, price: int) -> None:
        self.price = price
        self.head: Optional[Order] = None
        self.tail: Optional[Order] = None
        self.total_qty = 0
        self.count = 0

    def append(self, order: Order) -> None:
        order._level = self
        order._prev = self.tail
        order._next = None
        if self.tail is None:
            self.head = order
        else:
            self.tail._next = order
        self.tail = order
        self.total_qty += order.qty
        self.count += 1

    def remove(self, order: Order) -> None:
        """O(1) unlink."""
        if order._level is not self:
            raise KeyError(order.order_id)
        p, n = order._prev, order._next
        if p is None:
            self.head = n
        else:
            p._next = n
        if n is None:
            self.tail = p
        else:
            n._prev = p
        order._prev = order._next = None
        order._level = None
        self.total_qty -= order.qty
        self.count -= 1

    def __iter__(self) -> Iterator[Order]:
        o = self.head
        while o is not None:
            nxt = o._next          # capture first, caller may unlink o
            yield o
            o = nxt

    def __len__(self) -> int:
        return self.count

    def queue_position(self, order_id: int) -> int:
        """Quantity resting ahead of this order."""
        ahead = 0
        for o in self:
            if o.order_id == order_id:
                return ahead
            ahead += o.qty
        raise KeyError(order_id)


class OrderBook:
    def __init__(self,
                 on_event: Optional[Callable[[Event], None]] = None,
                 stp_mode: STPMode = STPMode.NONE) -> None:
        self.bids: SortedDict = SortedDict(lambda p: -p)   # best first
        self.asks: SortedDict = SortedDict()
        self._orders: Dict[int, Order] = {}
        self._stops: Dict[int, Order] = {}
        self.trades: List[Trade] = []
        self.events: List[Event] = []
        self.last_trade_price: Optional[int] = None
        self.stp_mode = stp_mode
        self._on_event = on_event
        self._next_trade_id = 1
        self._seq = 0
        self._seen_ids: set = set()

    # public API

    def submit(self, order: Order) -> List[Trade]:
        """Single entry point for every order type."""
        if order.order_id in self._seen_ids:
            raise ValueError(f"duplicate order id {order.order_id}")
        self._seen_ids.add(order.order_id)

        if order.order_type in _STOP_TYPES:
            self._stops[order.order_id] = order
            self._emit(EventType.ADD, order)
            # it may already be triggerable
            return self._check_stops(order.timestamp)

        trades = self._execute(order)
        trades += self._check_stops(order.timestamp)
        return trades

    # thin aliases kept for readability in strategy code
    def add_limit_order(self, order: Order) -> List[Trade]:
        return self.submit(order)

    def add_market_order(self, order: Order) -> List[Trade]:
        return self.submit(order)

    def cancel_order(self, order_id: int, timestamp: Optional[int] = None) -> Order:
        if order_id in self._stops:
            order = self._stops.pop(order_id)
            self._emit(EventType.CANCEL, order, ts=timestamp)
            return order
        order = self._orders[order_id]
        self._unlink(order)
        self._emit(EventType.CANCEL, order, ts=timestamp)
        return order

    def modify_order(self, order_id: int, new_price: int, new_qty: int,
                     timestamp: int) -> List[Trade]:
        """
        Reducing qty at the same price keeps queue position and emits MODIFY.
        Changing price, or increasing qty, loses it and is emitted as CANCEL
        followed by whatever the re-add produces. The order type is kept, so a
        POST_ONLY moved to a crossing price is rejected rather than aggressing.
        """
        if order_id in self._stops:
            raise ValueError("cancel and resubmit to change a pending stop")
        order = self._orders[order_id]
        if new_qty <= 0:
            raise ValueError("new_qty must be positive")

        if new_price == order.price and new_qty <= order.qty:
            level = order._level
            level.total_qty -= order.qty - new_qty
            order.original_qty -= order.qty - new_qty     # a reduction is not a fill
            order.qty = new_qty
            self._emit(EventType.MODIFY, order, ts=timestamp)
            return []

        self._unlink(order)
        self._emit(EventType.CANCEL, order, ts=timestamp)
        replacement = Order(order_id=order_id, side=order.side, price=new_price,
                            qty=new_qty, timestamp=timestamp,
                            order_type=order.order_type,
                            participant_id=order.participant_id)
        self._seen_ids.discard(order_id)      # same id, deliberate re-add
        return self.submit(replacement)

    # queries

    @property
    def best_bid(self) -> Optional[int]:
        return self.bids.peekitem(0)[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[int]:
        return self.asks.peekitem(0)[0] if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> Optional[int]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def microprice(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        bq = self.bids[self.best_bid].total_qty
        aq = self.asks[self.best_ask].total_qty
        return (self.best_bid * aq + self.best_ask * bq) / (bq + aq)

    def imbalance(self, levels: int = 1) -> Optional[float]:
        """(bid_qty - ask_qty) / (bid_qty + ask_qty) over the top N levels."""
        bids, asks = self.depth(levels)
        if not bids or not asks:
            return None
        bq = sum(q for _, q in bids)
        aq = sum(q for _, q in asks)
        return (bq - aq) / (bq + aq)

    def depth(self, levels: int = 5) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        b = [(p, lvl.total_qty) for p, lvl in self.bids.items()[:levels]]
        a = [(p, lvl.total_qty) for p, lvl in self.asks.items()[:levels]]
        return b, a

    def volume_at_or_better(self, side: Side, price: int) -> int:
        """Liquidity an aggressive `side` order at `price` could hit."""
        book = self._book_for(side.opposite())
        vol = 0
        for p, lvl in book.items():
            if (side is Side.BUY and p > price) or (side is Side.SELL and p < price):
                break
            vol += lvl.total_qty
        return vol

    def vwap_to_fill(self, side: Side, qty: int) -> Optional[float]:
        """Average price to fill `qty` by sweeping the book. None if not enough."""
        book = self._book_for(side.opposite())
        remaining, cost = qty, 0
        for p, lvl in book.items():
            take = min(remaining, lvl.total_qty)
            cost += take * p
            remaining -= take
            if remaining == 0:
                return cost / qty
        return None

    def get_order(self, order_id: int) -> Optional[Order]:
        return self._orders.get(order_id) or self._stops.get(order_id)

    def queue_position(self, order_id: int) -> int:
        order = self._orders[order_id]
        return order._level.queue_position(order_id)

    def snapshot(self, levels: Optional[int] = None) -> dict:
        n = levels if levels is not None else max(len(self.bids), len(self.asks))
        bids, asks = self.depth(n)
        return {"bids": bids, "asks": asks, "last": self.last_trade_price}

    def __len__(self) -> int:
        return len(self._orders)

    @property
    def pending_stops(self) -> int:
        return len(self._stops)

    # replay

    @classmethod
    def replay(cls, events: List[Event], stp_mode: STPMode = STPMode.NONE) -> "OrderBook":
        """
        Rebuild a book from an event log. Only ADD and CANCEL/MODIFY of
        resting orders matter for final state; trades are implied.
        Useful for debugging a backtest at any point in time.
        """
        book = cls(stp_mode=stp_mode)
        for e in events:
            if e.type is EventType.ADD and e.order_id not in book._orders:
                # ADD is emitted after matching, so this is the resting qty.
                # stops are dormant, they don't belong on the book.
                if e.price is None or e.order_type in _STOP_TYPES:
                    continue
                book._rest(Order(order_id=e.order_id, side=e.side, price=e.price,
                                 qty=e.qty, timestamp=e.timestamp))
            elif e.type is EventType.CANCEL and e.order_id in book._orders:
                book._unlink(book._orders[e.order_id])
            elif e.type is EventType.MODIFY and e.order_id in book._orders:
                o = book._orders[e.order_id]
                o._level.total_qty -= o.qty - e.qty
                o.qty = e.qty
            elif e.type is EventType.TRADE:
                m = book._orders.get(e.trade.maker_order_id)
                if m is not None:
                    m._level.total_qty -= e.trade.qty
                    m.qty -= e.trade.qty
                    if m.qty == 0:
                        book._unlink(m)
                book.last_trade_price = e.trade.price
        return book

    # internals

    def _execute(self, order: Order) -> List[Trade]:
        ot = order.order_type

        if ot is OrderType.POST_ONLY:
            if self._would_cross(order):
                self._emit(EventType.REJECT, order, reason=RejectReason.POST_ONLY_WOULD_CROSS)
                return []
            self._rest(order)
            self._emit(EventType.ADD, order)
            return []

        if ot is OrderType.FOK:
            if not self._fok_fillable(order):
                self._emit(EventType.REJECT, order, reason=RejectReason.FOK_UNFILLABLE)
                return []
            return self._match(order)

        trades = self._match(order)

        if order.qty > 0 and not order.stp_cancelled:
            if ot is OrderType.LIMIT:
                self._rest(order)
                self._emit(EventType.ADD, order)
            elif ot is OrderType.IOC:
                self._emit(EventType.CANCEL, order)
            elif ot is OrderType.MARKET:
                # nothing filled -> reject. something filled -> remainder cancelled
                et = EventType.REJECT if not trades else EventType.CANCEL
                self._emit(et, order, reason=RejectReason.NO_LIQUIDITY)
        return trades

    def _match(self, taker: Order) -> List[Trade]:
        trades: List[Trade] = []
        opp_book = self._book_for(taker.side.opposite())

        while taker.qty > 0 and opp_book and not taker.stp_cancelled:
            best_price, level = opp_book.peekitem(0)
            if not self._crosses(taker, best_price):
                break

            maker = level.head
            while taker.qty > 0 and maker is not None:
                nxt = maker._next

                # self trade prevention
                if (self.stp_mode is not STPMode.NONE
                        and taker.participant_id is not None
                        and taker.participant_id == maker.participant_id):
                    if self.stp_mode is STPMode.CANCEL_NEWEST:
                        self._emit(EventType.CANCEL, taker, reason=RejectReason.STP)
                        taker.stp_cancelled = True
                        break
                    if self.stp_mode is STPMode.CANCEL_OLDEST:
                        self._unlink(maker)
                        self._emit(EventType.CANCEL, maker, reason=RejectReason.STP)
                        maker = nxt
                        continue
                    # CANCEL_BOTH
                    self._unlink(maker)
                    self._emit(EventType.CANCEL, maker, reason=RejectReason.STP)
                    self._emit(EventType.CANCEL, taker, reason=RejectReason.STP)
                    taker.stp_cancelled = True
                    break

                fill_qty = min(taker.qty, maker.qty)
                maker.qty -= fill_qty
                taker.qty -= fill_qty
                level.total_qty -= fill_qty

                trade = Trade(trade_id=self._next_trade_id, price=best_price,
                              qty=fill_qty, timestamp=taker.timestamp,
                              maker_order_id=maker.order_id,
                              taker_order_id=taker.order_id,
                              aggressor_side=taker.side)
                self._next_trade_id += 1
                self.last_trade_price = best_price
                trades.append(trade)
                self.trades.append(trade)
                self._emit(EventType.TRADE, taker, trade=trade)

                if maker.is_filled:
                    self._unlink(maker)
                maker = nxt

            if len(level) == 0 and best_price in opp_book:
                del opp_book[best_price]

        return trades

    def _check_stops(self, timestamp: int) -> List[Trade]:
        """Trigger any stops touched by the last trade. Cascades."""
        trades: List[Trade] = []
        last = self.last_trade_price
        if last is None or not self._stops:
            return trades

        triggered = [o for o in self._stops.values()
                     if (o.side is Side.BUY and last >= o.stop_price)
                     or (o.side is Side.SELL and last <= o.stop_price)]
        triggered.sort(key=lambda o: o.timestamp)
        for o in triggered:
            del self._stops[o.order_id]
            self._emit(EventType.TRIGGER, o, ts=timestamp)
            live_type = OrderType.MARKET if o.order_type is OrderType.STOP else OrderType.LIMIT
            live = Order(order_id=o.order_id, side=o.side, price=o.price, qty=o.qty,
                         timestamp=timestamp, order_type=live_type,
                         participant_id=o.participant_id)
            trades += self._execute(live)
        if trades:
            trades += self._check_stops(timestamp)     # a trigger can trigger more
        return trades

    def _fok_fillable(self, order: Order) -> bool:
        """Walk the sweep range. Enough qty and no self-match that STP would
        turn into a partial fill."""
        book = self._book_for(order.side.opposite())
        need = order.qty
        for p, lvl in book.items():
            if not self._crosses(order, p):
                break
            for o in lvl:
                if (self.stp_mode is not STPMode.NONE
                        and order.participant_id is not None
                        and o.participant_id == order.participant_id):
                    if self.stp_mode is STPMode.CANCEL_OLDEST:
                        continue          # would be cancelled and skipped
                    return False          # NEWEST / BOTH would cut the fill short
                need -= o.qty
                if need <= 0:
                    return True
        return False

    def _book_for(self, side: Side) -> SortedDict:
        return self.bids if side is Side.BUY else self.asks

    def _crosses(self, order: Order, resting_price: int) -> bool:
        if order.order_type is OrderType.MARKET:
            return True
        if order.side is Side.BUY:
            return order.price >= resting_price
        return order.price <= resting_price

    def _would_cross(self, order: Order) -> bool:
        opp = self.best_ask if order.side is Side.BUY else self.best_bid
        return opp is not None and self._crosses(order, opp)

    def _rest(self, order: Order) -> None:
        book = self._book_for(order.side)
        level = book.get(order.price)
        if level is None:
            level = PriceLevel(order.price)
            book[order.price] = level
        level.append(order)
        self._orders[order.order_id] = order

    def _unlink(self, order: Order) -> None:
        level = order._level
        level.remove(order)
        if len(level) == 0:
            book = self._book_for(order.side)
            if book.get(order.price) is level:
                del book[order.price]
        del self._orders[order.order_id]

    def _emit(self, etype: EventType, order: Order, *, trade: Optional[Trade] = None,
              reason: Optional[RejectReason] = None, ts: Optional[int] = None) -> None:
        self._seq += 1
        ev = Event(seq=self._seq, timestamp=ts if ts is not None else order.timestamp,
                   type=etype, order_id=order.order_id, side=order.side,
                   price=order.price, qty=order.qty, trade=trade, reason=reason,
                   order_type=order.order_type)
        self.events.append(ev)
        if self._on_event:
            self._on_event(ev)

    def __repr__(self) -> str:
        bids, asks = self.depth(5)
        lines = ["         BID  |  ASK"]
        for i in range(5):
            b = f"{bids[i][1]:>5} @ {bids[i][0]:<5}" if i < len(bids) else " " * 13
            a = f"{asks[i][1]:>5} @ {asks[i][0]:<5}" if i < len(asks) else ""
            lines.append(f"{b} | {a}")
        if self.last_trade_price is not None:
            lines.append(f"last: {self.last_trade_price}")
        return "\n".join(lines)
