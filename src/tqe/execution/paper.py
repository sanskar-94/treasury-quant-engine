"""Deterministic paper broker: a simulated venue with real accounting.

The point of this module is that the *only* thing separating a backtest from
production should be which object is bound to the ``Broker`` protocol.  So the
paper broker is not a stub - it keeps a real book: weighted-average cost,
realised versus unrealised P&L, cash, commission, partial fills and resting
limit orders that only trade when the market comes to them.

Microstructure model
--------------------
Each symbol carries a **mid** and a **half-spread**, so the touch is
``bid = mid - h``, ``ask = mid + h``.  From there:

* a **market** order crosses the spread - a buy lifts the ask, a sell hits the
  bid - and additionally pays a random adverse slippage drawn from a half-normal
  (``|N(0, slippage_bp)|`` in basis points of price).  Slippage is *always*
  adverse: a simulator that lets you earn positive slippage half the time will
  flatter every strategy you ever test on it.
* a **limit** order fills only when its price is *through the touch* (buy limit
  ``>=`` ask, sell limit ``<=`` bid).  It then fills **at the touch**, not at the
  limit, because you never pay more than the market requires.  A limit that is
  not marketable rests and is re-evaluated on every quote update - which is what
  makes "my order never filled" a testable outcome rather than an assumption.

Determinism
-----------
All randomness comes from a single ``numpy.random.Generator`` seeded in the
constructor, and its bit-generator state is persisted alongside the book.  Two
runs with the same seed and the same order sequence produce byte-identical
fills, and a restored broker continues the *same* random stream rather than
restarting it - otherwise a crash mid-day would silently change the simulation.

Causality
---------
The broker only ever sees quotes that have been explicitly pushed to it and
orders that have already been submitted.  It cannot look at a future price
because it has no access to the price series at all; the caller (backtest engine
or live runner) advances the clock by calling :meth:`PaperBroker.set_quotes`
with day *t*'s marks before submitting day *t*'s orders.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..data.calendar import is_business_day
from ..logging_utils import audit, get_logger
from .broker import (
    QTY_EPS,
    AccountState,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    utcnow,
)

__all__ = ["PaperBroker", "Quote"]

_LOG = get_logger("execution.paper")

STATE_VERSION = 1


class Quote:
    """Immutable two-sided quote derived from a mid and a half-spread.

    Parameters
    ----------
    symbol:
        Instrument.
    mid:
        Mid price per unit.
    half_spread:
        Half the bid-ask spread **in price units** (not basis points).
    """

    __slots__ = ("symbol", "mid", "half_spread")

    def __init__(self, symbol: str, mid: float, half_spread: float) -> None:
        self.symbol = symbol
        self.mid = float(mid)
        self.half_spread = float(half_spread)

    @property
    def bid(self) -> float:
        return self.mid - self.half_spread

    @property
    def ask(self) -> float:
        return self.mid + self.half_spread

    def as_tuple(self) -> tuple[float, float]:
        """``(bid, ask)`` - the shape the ``Broker`` protocol returns."""
        return (self.bid, self.ask)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Quote({self.symbol!r}, bid={self.bid:.6f}, mid={self.mid:.6f}, ask={self.ask:.6f})"


class PaperBroker:
    """Simulated venue implementing the :class:`~tqe.execution.broker.Broker` protocol.

    Parameters
    ----------
    cfg:
        Optional :class:`~tqe.config.Config`.  When supplied it provides the
        defaults for ``initial_cash`` (``portfolio.capital``),
        ``commission_per_million`` (``costs.commission_per_million``),
        ``leverage`` (``portfolio.max_leverage``), ``state_dir``
        (``execution.state_dir``) and ``seed``.  Explicit keyword arguments
        always win over the config.
    initial_cash:
        Starting settled cash.  Also the denominator for return reporting.
    half_spread_bp:
        Default half-spread in basis points **of price**.  For a bond quoted per
        100, 1bp of price is 0.01 points, i.e. about a third of a 32nd - a
        realistic on-the-run 10y touch is ~1.5bp of price.
    slippage_bp:
        Standard deviation of the half-normal adverse slippage applied to
        marketable *market* orders, in basis points of price.  Set to 0 to make
        fills perfectly reproducible at the touch.
    fill_ratio:
        Fraction of the **original** order size that can trade per fill attempt.
        ``1.0`` fills marketable orders in one go; ``0.25`` produces four
        partials, which is how you find out whether downstream code handles
        ``PARTIAL`` correctly.
    commission_per_million:
        Dollars charged per $1mm of traded notional, per fill.
    min_commission:
        Floor applied to every fill's commission.
    leverage:
        Multiple of equity that may be deployed gross; drives ``buying_power``.
    seed:
        Seed for the slippage generator.
    state_dir, state_file:
        Where :meth:`persist` writes and :meth:`restore` reads.
    market_open:
        ``None`` (default) derives market status from the bond-market calendar;
        ``True``/``False`` pins it, which tests need because they run at weekends.
    clock:
        Injectable time source, so a replay can stamp events with historical times.
    reject_when_closed:
        If ``True`` the venue itself rejects orders outside trading days.  Default
        ``False`` - enforcing session times is the risk gate's job
        (``cfg.risk.require_market_open``), and doubling up makes backtests that
        stamp synthetic timestamps unreasonably hard to write.
    """

    def __init__(
        self,
        cfg: Config | None = None,
        *,
        initial_cash: float | None = None,
        half_spread_bp: float = 1.0,
        slippage_bp: float = 0.5,
        fill_ratio: float = 1.0,
        commission_per_million: float | None = None,
        min_commission: float = 0.0,
        leverage: float | None = None,
        seed: int | None = None,
        state_dir: str | Path | None = None,
        state_file: str = "paper_broker.json",
        market_open: bool | None = None,
        clock: Callable[[], datetime] = utcnow,
        reject_when_closed: bool = False,
    ) -> None:
        if not (0.0 < fill_ratio <= 1.0):
            raise ValueError(f"fill_ratio must be in (0, 1], got {fill_ratio!r}")
        if half_spread_bp < 0 or slippage_bp < 0:
            raise ValueError("half_spread_bp and slippage_bp must be non-negative")

        self.cfg = cfg
        self.initial_cash = float(
            initial_cash if initial_cash is not None else (cfg.portfolio.capital if cfg else 10_000_000.0)
        )
        self.half_spread_bp = float(half_spread_bp)
        self.slippage_bp = float(slippage_bp)
        self.fill_ratio = float(fill_ratio)
        self.commission_per_million = float(
            commission_per_million
            if commission_per_million is not None
            else (cfg.costs.commission_per_million if cfg else 12.5)
        )
        self.min_commission = float(min_commission)
        self.leverage = float(leverage if leverage is not None else (cfg.portfolio.max_leverage if cfg else 1.0))
        self.seed = int(seed if seed is not None else (cfg.seed if cfg else 42))
        self._rng = np.random.default_rng(self.seed)
        self.clock = clock
        self._market_open = market_open
        self.reject_when_closed = bool(reject_when_closed)

        base_dir = state_dir if state_dir is not None else (cfg.execution.state_dir if cfg else None)
        self.state_dir: Path | None = Path(base_dir) if base_dir is not None else None
        self.state_file = state_file

        # ---- the book ------------------------------------------------- #
        self._cash: float = self.initial_cash
        self._commission_paid: float = 0.0
        self._realized_pnl: float = 0.0
        # Flat positions are RETAINED: they still carry realised P&L, and
        # deleting them would silently reset the strategy's track record.
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self._quotes: dict[str, Quote] = {}

    # ------------------------------------------------------------------ #
    # Quotes
    # ------------------------------------------------------------------ #
    def set_quote(
        self,
        symbol: str,
        mid: float,
        half_spread: float | None = None,
        *,
        fill_open: bool = True,
    ) -> Quote:
        """Publish a new mid for ``symbol`` and re-mark the position.

        Parameters
        ----------
        symbol:
            Instrument.
        mid:
            New mid price per unit.
        half_spread:
            Absolute half-spread in price units; defaults to
            ``mid * half_spread_bp / 1e4``.
        fill_open:
            Whether resting limit orders should be re-evaluated against the new
            touch.  This is how a limit order that was away from the market gets
            filled when the market trades through it.

        Returns
        -------
        Quote
            The published quote.
        """
        mid = float(mid)
        if not np.isfinite(mid) or mid <= 0:
            raise ValueError(f"Quote mid for {symbol!r} must be positive and finite, got {mid!r}")
        hs = float(half_spread) if half_spread is not None else mid * self.half_spread_bp / 1e4
        quote = Quote(symbol, mid, hs)
        self._quotes[symbol] = quote
        pos = self._positions.get(symbol)
        if pos is not None:
            pos.mark(mid)
        if fill_open:
            self._sweep_open_orders(symbol)
        return quote

    def set_quotes(self, mids: Mapping[str, float], *, fill_open: bool = True) -> None:
        """Publish many mids at once (one trading day's marks, typically)."""
        for symbol, mid in mids.items():
            self.set_quote(symbol, mid, fill_open=fill_open)

    def has_quote(self, symbol: str) -> bool:
        """Whether a mid has been published for ``symbol``."""
        return symbol in self._quotes

    def mid(self, symbol: str) -> float:
        """Current mid for ``symbol``.

        Raises
        ------
        KeyError
            If no quote has been published.  Failing loudly is deliberate: a
            silent default price would mis-mark the whole book.
        """
        try:
            return self._quotes[symbol].mid
        except KeyError:
            raise KeyError(f"No quote published for {symbol!r}") from None

    def get_quote(self, symbol: str) -> tuple[float, float]:
        """``(bid, ask)`` for ``symbol``; raises ``KeyError`` if unquoted."""
        try:
            return self._quotes[symbol].as_tuple()
        except KeyError:
            raise KeyError(f"No quote published for {symbol!r}") from None

    def mark_to_market(self, mids: Mapping[str, float] | None = None) -> float:
        """Re-mark every position (optionally publishing new mids first).

        Returns
        -------
        float
            Account equity after marking.
        """
        if mids:
            self.set_quotes(mids)
        for symbol, pos in self._positions.items():
            q = self._quotes.get(symbol)
            if q is not None:
                pos.mark(q.mid)
        return self.equity

    # ------------------------------------------------------------------ #
    # Account
    # ------------------------------------------------------------------ #
    @property
    def cash(self) -> float:
        """Settled cash."""
        return self._cash

    @property
    def equity(self) -> float:
        """Net liquidation value: cash plus the marked value of every position."""
        return self._cash + sum(p.market_value for p in self._positions.values())

    @property
    def realized_pnl(self) -> float:
        """Cumulative realised trading P&L, **before** commission."""
        return self._realized_pnl

    @property
    def unrealized_pnl(self) -> float:
        """Mark-to-market P&L on open lots."""
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def commission_paid(self) -> float:
        """Cumulative commission."""
        return self._commission_paid

    @property
    def gross_exposure(self) -> float:
        return sum(p.notional for p in self._positions.values())

    @property
    def fills(self) -> list[Fill]:
        """Chronological execution blotter."""
        return list(self._fills)

    def get_account(self) -> AccountState:
        """Snapshot of cash, equity, buying power and open positions."""
        equity = self.equity
        # Buying power under a gross-leverage constraint: how much more absolute
        # exposure the account could take on before breaching `leverage * equity`.
        buying_power = max(0.0, self.leverage * equity - self.gross_exposure)
        return AccountState(
            cash=self._cash,
            equity=equity,
            buying_power=buying_power,
            positions=self.get_positions(),
            timestamp=self.clock(),
        )

    def get_positions(self) -> dict[str, Position]:
        """Open (non-flat) positions, keyed by symbol.

        Flat symbols are hidden here - the way a real broker reports - but are
        retained internally so their realised P&L is not lost.  Use
        :meth:`all_positions` for the full book.
        """
        return {s: p for s, p in self._positions.items() if not p.is_flat}

    def all_positions(self) -> dict[str, Position]:
        """Every symbol ever traded, flat ones included (realised P&L intact)."""
        return dict(self._positions)

    def position(self, symbol: str) -> Position:
        """Position for ``symbol``, creating a flat one on first touch."""
        pos = self._positions.get(symbol)
        if pos is None:
            q = self._quotes.get(symbol)
            pos = Position(symbol=symbol, market_price=q.mid if q else 0.0)
            self._positions[symbol] = pos
        return pos

    def is_market_open(self) -> bool:
        """Session status: pinned override if given, else the bond calendar."""
        if self._market_open is not None:
            return bool(self._market_open)
        return is_business_day(self.clock().date())

    def set_market_open(self, is_open: bool | None) -> None:
        """Pin (or un-pin with ``None``) the simulated session status."""
        self._market_open = is_open

    # ------------------------------------------------------------------ #
    # Orders
    # ------------------------------------------------------------------ #
    def submit_order(self, order: Order) -> Order:
        """Accept an order and attempt to fill it immediately.

        A marketable order trades on submission; a resting limit order is
        acknowledged (``SUBMITTED``) and waits for a quote update.  Orders for
        unquoted symbols are rejected - the simulator refuses to invent a price.

        Parameters
        ----------
        order:
            The order to send.  It is mutated in place and returned, matching how
            REST adapters behave.

        Returns
        -------
        Order
            The same object with venue state applied.
        """
        self._orders[order.id] = order

        if order.status.is_terminal:
            return order
        if self.reject_when_closed and not self.is_market_open():
            return self._reject(order, "market_closed")
        if not self.has_quote(order.symbol):
            return self._reject(order, f"no_quote:{order.symbol}")
        if order.order_type is OrderType.LIMIT and order.limit_price is None:
            return self._reject(order, "limit_order_without_price")

        order.status = OrderStatus.SUBMITTED
        order.touch(self.clock())
        audit(
            _LOG,
            "order_submitted",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            order_type=order.order_type.value,
            limit_price=order.limit_price,
            tag=order.tag,
        )
        self._try_fill(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a working order.

        Returns
        -------
        bool
            ``True`` if the order existed and was working (a partially filled
            order can be cancelled; the filled part stands).
        """
        order = self._orders.get(order_id)
        if order is None or order.status.is_terminal:
            return False
        order.status = OrderStatus.CANCELLED
        order.touch(self.clock())
        audit(_LOG, "order_cancelled", order_id=order_id, symbol=order.symbol,
              filled_quantity=order.filled_quantity)
        return True

    def get_order(self, order_id: str) -> Order | None:
        """Look up an order by client id."""
        return self._orders.get(order_id)

    def list_orders(self, open_only: bool = False) -> list[Order]:
        """Orders in submission order; optionally only those still working."""
        orders = list(self._orders.values())
        if open_only:
            orders = [o for o in orders if o.is_open]
        return orders

    # ------------------------------------------------------------------ #
    # Fill engine
    # ------------------------------------------------------------------ #
    def _sweep_open_orders(self, symbol: str) -> None:
        """Re-evaluate resting orders in ``symbol`` against the new touch."""
        for order in list(self._orders.values()):
            if order.symbol == symbol and order.is_open and order.status is not OrderStatus.NEW:
                self._try_fill(order)

    def _executable_price(self, order: Order) -> float | None:
        """Price this order can trade at right now, or ``None`` if it must rest.

        Market orders cross the spread and pay adverse slippage.  Limit orders
        trade only when they are through the touch, and then *at* the touch - a
        buy limit of 101 against an ask of 100 pays 100, never 101.
        """
        quote = self._quotes[order.symbol]
        if order.order_type is OrderType.MARKET:
            touch = quote.ask if order.side is OrderSide.BUY else quote.bid
            return touch + self._slippage(touch, order.side)
        limit = order.limit_price
        if limit is None:
            return None
        if order.side is OrderSide.BUY:
            return quote.ask if limit >= quote.ask else None
        return quote.bid if limit <= quote.bid else None

    def _slippage(self, price: float, side: OrderSide) -> float:
        """Adverse price concession for a liquidity-taking order.

        Drawn as ``|N(0, slippage_bp)|`` basis points of price and signed against
        the trader: buys pay up, sells get hit down.  The half-normal (rather
        than a symmetric normal) encodes the reality that a taker's execution is
        never better than the touch it crossed.
        """
        if self.slippage_bp <= 0.0:
            return 0.0
        bp = abs(self._rng.normal(0.0, self.slippage_bp))
        return side.sign * price * bp / 1e4

    def _try_fill(self, order: Order) -> Fill | None:
        """Attempt one execution slice against the current market."""
        if not order.is_open or not self.has_quote(order.symbol):
            return None
        price = self._executable_price(order)
        if price is None:
            return None  # resting away from the market

        # Each attempt trades at most `fill_ratio` of the ORIGINAL size, so a
        # partially filled order completes in a finite number of quote updates
        # instead of asymptotically approaching completion.
        slice_qty = min(order.remaining_quantity, order.quantity * self.fill_ratio)
        if slice_qty <= QTY_EPS:
            return None
        return self._execute(order, slice_qty, price)

    def _execute(self, order: Order, quantity: float, price: float) -> Fill:
        """Book one execution: position, cash, commission, blotter, audit."""
        mid = self._quotes[order.symbol].mid
        notional = abs(quantity * price)
        commission = max(self.min_commission, notional / 1e6 * self.commission_per_million)
        # Slippage recorded per unit versus the arrival mid, signed so that
        # positive always means "worse than mid".
        slippage = (price - mid) * order.side.sign

        pos = self.position(order.symbol)
        realized = pos.apply_fill(order.side.sign * quantity, price)
        self._realized_pnl += realized
        # Cash: buying spends, selling (including short selling) receives; the
        # commission is a pure debit either way.
        self._cash -= order.side.sign * quantity * price
        self._cash -= commission
        self._commission_paid += commission
        pos.mark(mid)

        when = self.clock()
        order.record_fill(quantity, price, when)
        fill = Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            timestamp=when,
            commission=commission,
            slippage=slippage,
        )
        self._fills.append(fill)
        audit(
            _LOG,
            "order_filled",
            order_id=order.id,
            fill_id=fill.fill_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=quantity,
            price=price,
            mid=mid,
            commission=commission,
            slippage=slippage,
            realized_pnl=realized,
            status=order.status.value,
            position=pos.quantity,
            avg_price=pos.avg_price,
            cash=self._cash,
        )
        return fill

    def _reject(self, order: Order, reason: str) -> Order:
        """Reject an order at the venue and audit it."""
        order.reject(reason, self.clock())
        audit(_LOG, "order_rejected_by_venue", order_id=order.id, symbol=order.symbol, reason=reason)
        return order

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _resolve_path(self, path: str | Path | None) -> Path:
        """Resolve the state file location from an argument or ``state_dir``."""
        if path is not None:
            return Path(path)
        if self.state_dir is None:
            raise ValueError("PaperBroker has no state_dir; pass an explicit path to persist/restore")
        return Path(self.state_dir) / self.state_file

    def state_dict(self) -> dict[str, Any]:
        """Complete, JSON-serialisable snapshot of the book.

        Includes the RNG bit-generator state so a restored broker continues the
        *same* random stream: restarting the stream would make a crash change the
        simulated fills, and a simulator whose results depend on uptime is not a
        simulator.
        """
        return {
            "version": STATE_VERSION,
            "saved_at": utcnow().isoformat(),
            "params": {
                "initial_cash": self.initial_cash,
                "half_spread_bp": self.half_spread_bp,
                "slippage_bp": self.slippage_bp,
                "fill_ratio": self.fill_ratio,
                "commission_per_million": self.commission_per_million,
                "min_commission": self.min_commission,
                "leverage": self.leverage,
                "seed": self.seed,
            },
            "cash": self._cash,
            "commission_paid": self._commission_paid,
            "realized_pnl": self._realized_pnl,
            # Flat positions included - they carry realised P&L history.
            "positions": {s: p.to_dict() for s, p in self._positions.items()},
            "orders": [o.to_dict() for o in self._orders.values()],
            "fills": [f.to_dict() for f in self._fills],
            "quotes": {s: {"mid": q.mid, "half_spread": q.half_spread} for s, q in self._quotes.items()},
            "rng_state": self._rng.bit_generator.state,
        }

    def persist(self, path: str | Path | None = None) -> Path:
        """Atomically write the book to JSON.

        Writes to a temporary file and renames, so a crash mid-write cannot leave
        a truncated state file that would be silently loaded on restart.

        Returns
        -------
        pathlib.Path
            The file written.
        """
        target = self._resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state_dict(), indent=2, default=str))
        tmp.replace(target)
        audit(_LOG, "broker_persisted", path=str(target), cash=self._cash, equity=self.equity,
              n_positions=len(self.get_positions()), n_orders=len(self._orders))
        return target

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Replace the in-memory book with a persisted snapshot."""
        version = int(state.get("version", 0))
        if version != STATE_VERSION:
            raise ValueError(f"Unsupported paper-broker state version {version} (expected {STATE_VERSION})")
        params = state.get("params", {})
        self.initial_cash = float(params.get("initial_cash", self.initial_cash))
        self.half_spread_bp = float(params.get("half_spread_bp", self.half_spread_bp))
        self.slippage_bp = float(params.get("slippage_bp", self.slippage_bp))
        self.fill_ratio = float(params.get("fill_ratio", self.fill_ratio))
        self.commission_per_million = float(params.get("commission_per_million", self.commission_per_million))
        self.min_commission = float(params.get("min_commission", self.min_commission))
        self.leverage = float(params.get("leverage", self.leverage))
        self.seed = int(params.get("seed", self.seed))

        self._cash = float(state["cash"])
        self._commission_paid = float(state.get("commission_paid", 0.0))
        self._realized_pnl = float(state.get("realized_pnl", 0.0))
        self._positions = {s: Position.from_dict(p) for s, p in (state.get("positions") or {}).items()}
        self._orders = {}
        for raw in state.get("orders") or []:
            order = Order.from_dict(raw)
            self._orders[order.id] = order
        self._fills = [Fill.from_dict(f) for f in (state.get("fills") or [])]
        self._quotes = {
            s: Quote(s, q["mid"], q["half_spread"]) for s, q in (state.get("quotes") or {}).items()
        }
        rng_state = state.get("rng_state")
        if rng_state:
            self._rng.bit_generator.state = _coerce_rng_state(rng_state)

    def restore(self, path: str | Path | None = None, *, missing_ok: bool = True) -> bool:
        """Load a persisted book from JSON.

        Parameters
        ----------
        path:
            State file; defaults to ``state_dir / state_file``.
        missing_ok:
            Return ``False`` instead of raising when the file does not exist -
            the normal "first ever run" path.

        Returns
        -------
        bool
            ``True`` if state was loaded.
        """
        target = self._resolve_path(path)
        if not target.exists():
            if missing_ok:
                return False
            raise FileNotFoundError(target)
        self.load_state(json.loads(target.read_text()))
        audit(_LOG, "broker_restored", path=str(target), cash=self._cash, equity=self.equity,
              n_positions=len(self.get_positions()))
        return True

    @classmethod
    def from_state(cls, path: str | Path, **kwargs: Any) -> PaperBroker:
        """Construct a broker and immediately restore ``path`` into it."""
        broker = cls(**kwargs)
        broker.restore(path, missing_ok=False)
        return broker

    # ------------------------------------------------------------------ #
    def reset(self, *, keep_quotes: bool = True) -> None:
        """Wipe the book back to inception (used between backtest runs)."""
        self._cash = self.initial_cash
        self._commission_paid = 0.0
        self._realized_pnl = 0.0
        self._positions = {}
        self._orders = {}
        self._fills = []
        if not keep_quotes:
            self._quotes = {}
        self._rng = np.random.default_rng(self.seed)

    def blotter(self) -> list[dict[str, Any]]:
        """Execution blotter as plain dicts (ready for a DataFrame)."""
        return [f.to_dict() for f in self._fills]

    def summary(self) -> dict[str, float | int]:
        """One-line health check of the book."""
        return {
            "cash": self._cash,
            "equity": self.equity,
            "realized_pnl": self._realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "commission_paid": self._commission_paid,
            "gross_exposure": self.gross_exposure,
            "n_positions": len(self.get_positions()),
            "n_orders": len(self._orders),
            "n_fills": len(self._fills),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"PaperBroker(cash={self._cash:,.2f}, equity={self.equity:,.2f}, "
            f"positions={len(self.get_positions())}, fills={len(self._fills)})"
        )


def _coerce_rng_state(state: Any) -> dict[str, Any]:
    """Restore a numpy bit-generator state that has round-tripped through JSON.

    JSON has no integer/float distinction on the way back in, and numpy insists
    the PCG64 ``state``/``inc`` words are Python ints, so they are re-cast here.
    """
    if not isinstance(state, dict):
        raise TypeError(f"Unusable RNG state of type {type(state)!r}")
    out = dict(state)
    inner = out.get("state")
    if isinstance(inner, dict):
        out["state"] = {k: int(v) if isinstance(v, (int, float, str)) else v for k, v in inner.items()}
    for key in ("has_uint32", "uinteger"):
        if key in out:
            out[key] = int(out[key])
    return out

