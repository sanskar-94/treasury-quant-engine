"""Broker abstraction: the vocabulary every execution venue is expressed in.

The rest of the stack (OMS, risk gate, live runner) never talks to a venue
directly - it talks to the :class:`Broker` protocol defined here.  That is what
lets the identical order-generation code run against a deterministic
:class:`~tqe.execution.paper.PaperBroker` in a backtest and against a real REST
adapter in production without a single branch on "am I live?".

Design notes that matter for a trading system
---------------------------------------------
*Identity*.  Every :class:`Order` carries a client-side ``id`` (uuid4 hex)
generated at construction, **not** assigned by the venue.  A client order id is
what makes a submission idempotent: if the process dies between "send" and
"acknowledge", the recovery path can ask the venue about *this* id instead of
guessing whether the order exists.  ``created_at``/``updated_at`` are timezone
aware UTC for the same reason - an audit trail with naive local timestamps is
worthless the first time the clocks change.

*Position accounting*.  :meth:`Position.apply_fill` implements the single piece
of arithmetic that execution systems get wrong most often: a trade that carries
a position **through zero**.  Selling 120 against a long 100 is not "one trade";
it is a close of 100 (which realises P&L against the *old* average cost) plus an
opening short of 20 (which starts a *new* average cost at the trade price).
Blending the trade price into the old average, or realising the whole 120
against the old average, both silently corrupt realised P&L.  The logic lives
here, once, so every broker implementation inherits the correct version.

*Serialisation*.  ``to_dict``/``from_dict`` on :class:`Order`, :class:`Fill` and
:class:`Position` are exact round-trips through JSON so a paper book can be
persisted across restarts and an audit log can be replayed.

Causality
---------
Nothing in this module reads a time series, so there is no look-ahead surface.
The only temporal state is the wall-clock stamps on orders and fills, which are
written at the moment the event happens and never back-dated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

__all__ = [
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Order",
    "Fill",
    "Position",
    "AccountState",
    "Broker",
    "utcnow",
    "new_order_id",
]

# Quantities below this are floating-point dust, not a position.  Snapping to
# zero stops a residual of 1e-16 units from being reported as an open position
# (and, worse, from re-triggering a "close me" order every single day).
QTY_EPS: float = 1e-9


def utcnow() -> datetime:
    """Timezone-aware current UTC time.

    Returns
    -------
    datetime
        ``datetime.now(timezone.utc)``.  Every timestamp in the execution stack
        goes through here so that audit records are directly comparable across
        machines and across daylight-saving boundaries.
    """
    return datetime.now(timezone.utc)


def new_order_id() -> str:
    """Fresh client order id (32-char uuid4 hex, no dashes)."""
    return uuid4().hex


def _iso(dt: datetime | None) -> str | None:
    """Serialise a datetime to ISO-8601, coercing naive stamps to UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _parse_dt(value: Any) -> datetime:
    """Parse an ISO-8601 string (or pass through a datetime) as aware UTC."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if value is None:
        return utcnow()
    dt = datetime.fromisoformat(str(value))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class OrderSide(str, Enum):
    """Direction of an order.  ``str`` mixin so it serialises as ``"buy"``."""

    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        """+1 for a buy, -1 for a sell - the signed multiplier on quantity."""
        return 1 if self is OrderSide.BUY else -1

    @classmethod
    def from_signed(cls, quantity: float) -> OrderSide:
        """Side implied by a signed quantity (``>= 0`` is a buy)."""
        return cls.BUY if quantity >= 0 else cls.SELL


class OrderType(str, Enum):
    """Execution instruction.  Only the two types a rates book actually needs."""

    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    """Lifecycle state.

    ``NEW`` is client-side only (constructed, not yet sent).  ``SUBMITTED`` means
    the venue has acknowledged and the order is resting.  ``PARTIAL`` means some
    quantity has traded and the remainder is still working.
    """

    NEW = "new"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        """``True`` once the order can never trade again."""
        return self in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)

    @property
    def is_open(self) -> bool:
        """``True`` while the order can still receive a fill."""
        return not self.is_terminal


# --------------------------------------------------------------------------- #
# Order
# --------------------------------------------------------------------------- #
@dataclass
class Order:
    """A single instruction to trade one instrument.

    Field order follows ``docs/ARCHITECTURE.md`` §8 exactly.  ``id`` leads and is
    auto-generated, so orders are normally built with keywords::

        Order(symbol="IEF", side=OrderSide.BUY, quantity=1_000)

    Parameters
    ----------
    id:
        Client order id.  Stable for the life of the order and used as the
        idempotency key on resubmission after a crash.
    symbol:
        Instrument identifier (an ETF ticker via ``cfg.execution.instrument_map``,
        or a CMT tenor label when trading the synthetic cash curve).
    side:
        :class:`OrderSide`.  ``quantity`` is always **positive**; direction lives
        in the side, which is how every real venue models it.
    quantity:
        Absolute size in instrument units (shares, or face/100 for cash bonds).
    order_type, limit_price:
        A ``LIMIT`` order must carry a price; a ``MARKET`` order must not need one.
    status, filled_quantity, avg_fill_price:
        Execution state, mutated only through :meth:`record_fill` and the broker.
    created_at, updated_at:
        Aware UTC stamps.  ``updated_at`` advances on every state transition.
    tag:
        Free-form provenance string (e.g. ``"oms:2026-08-28"``) that ties an
        order back to the run that produced it.
    reject_reason:
        Populated by the risk gate or the venue when ``status == REJECTED``.

    Raises
    ------
    ValueError
        If the symbol is blank, the quantity is not strictly positive and finite,
        or a limit order carries a non-positive limit price.
    """

    id: str = field(default_factory=new_order_id)
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    order_type: OrderType = OrderType.LIMIT
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    tag: str = ""
    reject_reason: str = ""

    def __post_init__(self) -> None:
        # Coerce loosely-typed input (JSON, YAML, a CLI string) into the enums so
        # downstream `is`-comparisons are safe.
        self.side = OrderSide(self.side)
        self.order_type = OrderType(self.order_type)
        self.status = OrderStatus(self.status)
        self.quantity = float(self.quantity)
        self.filled_quantity = float(self.filled_quantity)
        self.avg_fill_price = float(self.avg_fill_price)
        if self.limit_price is not None:
            self.limit_price = float(self.limit_price)
        self.created_at = _parse_dt(self.created_at)
        self.updated_at = _parse_dt(self.updated_at)

        if not self.symbol:
            raise ValueError("Order.symbol is required")
        # A zero or NaN quantity is never a legitimate instruction; catching it at
        # construction keeps garbage out of the risk gate and the audit log.
        if not (self.quantity > 0.0) or self.quantity != self.quantity:
            raise ValueError(f"Order.quantity must be a positive finite number, got {self.quantity!r}")
        if self.order_type is OrderType.LIMIT and self.limit_price is not None and self.limit_price <= 0:
            raise ValueError(f"Limit price must be positive, got {self.limit_price!r}")

    # ------------------------------------------------------------------ #
    @classmethod
    def create(
        cls,
        symbol: str,
        side: OrderSide | str,
        quantity: float,
        *,
        order_type: OrderType | str = OrderType.LIMIT,
        limit_price: float | None = None,
        tag: str = "",
    ) -> Order:
        """Convenience constructor with the trade-describing fields first."""
        return cls(
            symbol=symbol,
            side=OrderSide(side),
            quantity=quantity,
            order_type=OrderType(order_type),
            limit_price=limit_price,
            tag=tag,
        )

    @classmethod
    def from_signed_quantity(
        cls,
        symbol: str,
        signed_quantity: float,
        *,
        order_type: OrderType | str = OrderType.LIMIT,
        limit_price: float | None = None,
        tag: str = "",
    ) -> Order:
        """Build from a signed delta (negative -> SELL), the OMS's natural form."""
        return cls.create(
            symbol,
            OrderSide.from_signed(signed_quantity),
            abs(signed_quantity),
            order_type=order_type,
            limit_price=limit_price,
            tag=tag,
        )

    # ------------------------------------------------------------------ #
    @property
    def signed_quantity(self) -> float:
        """Order size with direction folded in (negative for a sell)."""
        return self.side.sign * self.quantity

    @property
    def signed_filled_quantity(self) -> float:
        """Filled size with direction folded in."""
        return self.side.sign * self.filled_quantity

    @property
    def remaining_quantity(self) -> float:
        """Unfilled size, floored at zero."""
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_open(self) -> bool:
        """``True`` while the order may still trade."""
        return self.status.is_open

    def notional(self, price: float | None = None) -> float:
        """Absolute notional at ``price`` (defaults to the limit price).

        Returns ``nan`` when no price is available, which the risk gate treats as
        an un-checkable order and therefore blocks - failing closed, not open.
        """
        px = price if price is not None else self.limit_price
        if px is None:
            return float("nan")
        return abs(self.quantity * float(px))

    def touch(self, when: datetime | None = None) -> None:
        """Advance ``updated_at``."""
        self.updated_at = when or utcnow()

    def record_fill(self, quantity: float, price: float, when: datetime | None = None) -> None:
        """Fold an execution into the order's own state.

        Maintains ``avg_fill_price`` as the quantity-weighted mean over all
        partial executions of this order, and promotes the status to ``PARTIAL``
        or ``FILLED``.

        Parameters
        ----------
        quantity:
            Absolute quantity executed (never signed - the side is on the order).
        price:
            Execution price per unit.
        when:
            Event time; defaults to now.
        """
        if quantity <= 0:
            return
        filled = self.filled_quantity + quantity
        # Weighted average across partials, not a simple mean of prices.
        self.avg_fill_price = (self.avg_fill_price * self.filled_quantity + price * quantity) / filled
        self.filled_quantity = filled
        self.status = OrderStatus.FILLED if filled >= self.quantity - QTY_EPS else OrderStatus.PARTIAL
        self.touch(when)

    def reject(self, reason: str, when: datetime | None = None) -> Order:
        """Mark the order rejected with a human-readable reason."""
        self.status = OrderStatus.REJECTED
        self.reject_reason = reason
        self.touch(when)
        return self

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """JSON-safe mapping (enums -> values, datetimes -> ISO-8601)."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "tag": self.tag,
            "reject_reason": self.reject_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Order:
        """Inverse of :meth:`to_dict`; tolerates missing optional keys."""
        return cls(
            id=d.get("id") or new_order_id(),
            symbol=d["symbol"],
            side=OrderSide(d["side"]),
            quantity=float(d["quantity"]),
            order_type=OrderType(d.get("order_type", OrderType.LIMIT)),
            limit_price=d.get("limit_price"),
            status=OrderStatus(d.get("status", OrderStatus.NEW)),
            filled_quantity=float(d.get("filled_quantity", 0.0)),
            avg_fill_price=float(d.get("avg_fill_price", 0.0)),
            created_at=_parse_dt(d.get("created_at")),
            updated_at=_parse_dt(d.get("updated_at")),
            tag=d.get("tag", ""),
            reject_reason=d.get("reject_reason", ""),
        )


# --------------------------------------------------------------------------- #
# Fill
# --------------------------------------------------------------------------- #
@dataclass
class Fill:
    """One execution against one order.

    Parameters
    ----------
    order_id:
        The :attr:`Order.id` this execution belongs to.
    symbol, side, quantity, price:
        What traded.  ``quantity`` is absolute; ``side`` carries the direction.
    timestamp:
        Aware UTC execution time.
    commission:
        Dollar commission charged on this execution.
    slippage:
        Adverse deviation **per unit** from the prevailing mid, i.e.
        ``(fill_price - mid) * side.sign``.  Positive means the fill was worse
        than mid, which is the normal case since a taker crosses the spread.
        Summing ``quantity * slippage`` over fills gives realised implementation
        shortfall versus the arrival mid, the number an execution desk is judged on.
    fill_id:
        Unique id for the execution itself, so a replayed audit log can be
        de-duplicated.
    """

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime = field(default_factory=utcnow)
    commission: float = 0.0
    slippage: float = 0.0
    fill_id: str = field(default_factory=new_order_id)

    def __post_init__(self) -> None:
        self.side = OrderSide(self.side)
        self.quantity = float(self.quantity)
        self.price = float(self.price)
        self.timestamp = _parse_dt(self.timestamp)
        self.commission = float(self.commission)
        self.slippage = float(self.slippage)

    @property
    def signed_quantity(self) -> float:
        """Direction-aware quantity, the form position accounting consumes."""
        return self.side.sign * self.quantity

    @property
    def notional(self) -> float:
        """Absolute traded notional, commission excluded."""
        return abs(self.quantity * self.price)

    @property
    def cash_flow(self) -> float:
        """Signed cash impact including commission (negative when buying)."""
        return -self.signed_quantity * self.price - self.commission

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe mapping."""
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": _iso(self.timestamp),
            "commission": self.commission,
            "slippage": self.slippage,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Fill:
        """Inverse of :meth:`to_dict`."""
        return cls(
            order_id=d["order_id"],
            symbol=d["symbol"],
            side=OrderSide(d["side"]),
            quantity=float(d["quantity"]),
            price=float(d["price"]),
            timestamp=_parse_dt(d.get("timestamp")),
            commission=float(d.get("commission", 0.0)),
            slippage=float(d.get("slippage", 0.0)),
            fill_id=d.get("fill_id") or new_order_id(),
        )


# --------------------------------------------------------------------------- #
# Position
# --------------------------------------------------------------------------- #
@dataclass
class Position:
    """Holding in one instrument, with weighted-average cost accounting.

    Parameters
    ----------
    symbol:
        Instrument identifier.
    quantity:
        **Signed** holding: positive long, negative short.
    avg_price:
        Weighted-average cost of the *currently open* quantity.  It is a property
        of the open lot only - closing trades never change it, they realise
        against it.
    market_price:
        Last mark.  ``0.0`` means "never marked".
    unrealized_pnl, realized_pnl:
        Mark-to-market on the open lot, and cumulative realised P&L since
        inception (it survives the position going flat, which is why flat
        positions are kept in the book rather than deleted).
    """

    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    market_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def __post_init__(self) -> None:
        self.quantity = float(self.quantity)
        self.avg_price = float(self.avg_price)
        self.market_price = float(self.market_price)
        self.unrealized_pnl = float(self.unrealized_pnl)
        self.realized_pnl = float(self.realized_pnl)

    # ------------------------------------------------------------------ #
    @property
    def is_flat(self) -> bool:
        """``True`` when the open quantity is (numerically) zero."""
        return abs(self.quantity) < QTY_EPS

    @property
    def is_long(self) -> bool:
        return self.quantity > QTY_EPS

    @property
    def is_short(self) -> bool:
        return self.quantity < -QTY_EPS

    @property
    def market_value(self) -> float:
        """Signed mark-to-market value of the holding."""
        return self.quantity * self.market_price

    @property
    def notional(self) -> float:
        """Absolute market value - what position limits are expressed against."""
        return abs(self.market_value)

    @property
    def cost_basis(self) -> float:
        """Signed cost of the open lot."""
        return self.quantity * self.avg_price

    @property
    def total_pnl(self) -> float:
        """Realised plus unrealised."""
        return self.realized_pnl + self.unrealized_pnl

    # ------------------------------------------------------------------ #
    def mark(self, price: float) -> float:
        """Re-mark the position and refresh unrealised P&L.

        Parameters
        ----------
        price:
            New mark (mid, normally).

        Returns
        -------
        float
            The updated unrealised P&L.  Signed quantity makes this correct for
            shorts automatically: a short (``quantity < 0``) gains when the mark
            falls below the average sale price.
        """
        self.market_price = float(price)
        self.unrealized_pnl = (self.market_price - self.avg_price) * self.quantity if not self.is_flat else 0.0
        return self.unrealized_pnl

    def apply_fill(self, signed_quantity: float, price: float) -> float:
        """Apply an execution and return the **realised P&L of this fill**.

        This is the one function that must not be wrong.  Three regimes:

        1. **Opening or adding** (fill has the same sign as the position, or the
           position is flat).  Nothing is realised; the average cost becomes the
           quantity-weighted blend of the old lot and the new one.
        2. **Reducing** (opposite sign, ``|fill| <= |position|``).  The closed
           quantity realises ``(price - avg_price) * closed * sign(position)``.
           The average cost of the surviving lot is **unchanged** - the remaining
           bonds were bought at the old price and nothing about them changed.
        3. **Flipping through zero** (opposite sign, ``|fill| > |position|``).
           Split the fill: the first ``|position|`` units close the old lot and
           realise against the *old* average; the residual
           ``|fill| - |position|`` units *open a brand-new position on the other
           side* whose average cost is the trade price.  Blending the trade price
           into the old average here - the classic bug - corrupts both realised
           and unrealised P&L from that moment on, and the error never washes out
           because it is baked into the cost basis.

        Parameters
        ----------
        signed_quantity:
            Executed quantity, negative for a sell.
        price:
            Execution price per unit.

        Returns
        -------
        float
            Realised P&L attributable to this fill (0.0 when only opening).
        """
        q = float(signed_quantity)
        px = float(price)
        if q == 0.0:
            return 0.0

        q0 = self.quantity
        realized = 0.0

        if abs(q0) < QTY_EPS or (q0 > 0) == (q > 0):
            # --- regime 1: open / add ---------------------------------- #
            new_qty = q0 + q
            # Weighted by absolute size so the blend is correct for shorts too.
            self.avg_price = (abs(q0) * self.avg_price + abs(q) * px) / abs(new_qty)
            self.quantity = new_qty
        else:
            # --- regimes 2 and 3: reduce / close / flip ----------------- #
            closing = min(abs(q), abs(q0))
            direction = 1.0 if q0 > 0 else -1.0
            realized = (px - self.avg_price) * closing * direction
            new_qty = q0 + q
            if abs(new_qty) < QTY_EPS:
                # Fully closed: snap to exactly flat and forget the cost basis.
                self.quantity = 0.0
                self.avg_price = 0.0
            elif (new_qty > 0) != (q0 > 0):
                # Flipped through zero: the residual is a NEW lot at this price.
                self.quantity = new_qty
                self.avg_price = px
            else:
                # Partially reduced: surviving lot keeps its original cost.
                self.quantity = new_qty

        self.realized_pnl += realized
        # Re-mark at the trade price so equity is consistent immediately after a
        # fill even if the next mark-to-market has not arrived yet.
        self.mark(px if self.market_price == 0.0 else self.market_price)
        return realized

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """JSON-safe mapping."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_price": self.avg_price,
            "market_price": self.market_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Position:
        """Inverse of :meth:`to_dict`."""
        return cls(
            symbol=d["symbol"],
            quantity=float(d.get("quantity", 0.0)),
            avg_price=float(d.get("avg_price", 0.0)),
            market_price=float(d.get("market_price", 0.0)),
            unrealized_pnl=float(d.get("unrealized_pnl", 0.0)),
            realized_pnl=float(d.get("realized_pnl", 0.0)),
        )


# --------------------------------------------------------------------------- #
# Account
# --------------------------------------------------------------------------- #
@dataclass
class AccountState:
    """Snapshot of the trading account at a point in time.

    Parameters
    ----------
    cash:
        Settled cash.  Buying reduces it, selling (including selling short)
        increases it, commission always reduces it.
    equity:
        ``cash + sum(position.market_value)`` - net liquidation value.
    buying_power:
        Notional still available to deploy under the leverage cap.
    positions:
        Open positions keyed by symbol.
    timestamp:
        Aware UTC time of the snapshot.  The risk gate keys its daily loss
        baseline off this date, so it must be the real event time.
    """

    cash: float
    equity: float
    buying_power: float
    positions: dict[str, Position] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        self.cash = float(self.cash)
        self.equity = float(self.equity)
        self.buying_power = float(self.buying_power)
        self.timestamp = _parse_dt(self.timestamp)

    @property
    def gross_exposure(self) -> float:
        """Sum of absolute position values - the leverage numerator."""
        return sum(p.notional for p in self.positions.values())

    @property
    def net_exposure(self) -> float:
        """Signed sum of position values - directional exposure."""
        return sum(p.market_value for p in self.positions.values())

    @property
    def leverage(self) -> float:
        """Gross exposure divided by equity (0.0 when equity is non-positive)."""
        return self.gross_exposure / self.equity if self.equity > 0 else 0.0

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values())

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe mapping, positions included."""
        return {
            "cash": self.cash,
            "equity": self.equity,
            "buying_power": self.buying_power,
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "timestamp": _iso(self.timestamp),
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AccountState:
        """Inverse of :meth:`to_dict` (derived fields are ignored)."""
        return cls(
            cash=float(d["cash"]),
            equity=float(d["equity"]),
            buying_power=float(d.get("buying_power", 0.0)),
            positions={s: Position.from_dict(p) for s, p in (d.get("positions") or {}).items()},
            timestamp=_parse_dt(d.get("timestamp")),
        )


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #
@runtime_checkable
class Broker(Protocol):
    """Structural interface every venue adapter satisfies.

    A ``Protocol`` rather than an ABC on purpose: adapters (paper, Alpaca, a
    future FIX gateway) need not inherit from anything, and test doubles are just
    objects with the right methods.  ``runtime_checkable`` allows a cheap
    ``isinstance`` smoke test at wiring time - it verifies method *presence*, not
    signatures, so it is a guard rail, not a proof.
    """

    def submit_order(self, order: Order) -> Order:
        """Send an order; return it with venue state applied (status, fills)."""
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a working order.  ``False`` if unknown or already terminal."""
        ...

    def get_order(self, order_id: str) -> Order | None:
        """Look up an order by client id."""
        ...

    def list_orders(self, open_only: bool = False) -> list[Order]:
        """All known orders, or only those still working."""
        ...

    def get_account(self) -> AccountState:
        """Current cash/equity/buying-power snapshot."""
        ...

    def get_positions(self) -> dict[str, Position]:
        """Open (non-flat) positions keyed by symbol."""
        ...

    def get_quote(self, symbol: str) -> tuple[float, float]:
        """Current ``(bid, ask)`` for ``symbol``."""
        ...

    def is_market_open(self) -> bool:
        """Whether the venue is currently accepting trades."""
        ...
