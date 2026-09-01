"""Order management: target book in, executed orders out, nothing lost.

The OMS is the seam between "what the strategy wants to own" and "what the
account actually owns".  It does four things, in this order:

1. **Diff.**  Targets are *dollar notionals per symbol*.  The OMS values the
   current book at the live mid and trades only the difference, so a strategy
   that repeats yesterday's view generates no orders at all.  Anything smaller
   than ``min_trade_notional`` is dropped: a $900 rebalance in a $10mm book pays
   the spread for an exposure change nobody can measure.
2. **Gate.**  Every order goes through the :class:`~tqe.execution.risk_gate.RiskGate`.
   Rejected orders are stamped, recorded and audited - never silently dropped.
   A rejection you cannot see is indistinguishable from an order that vanished.
3. **Submit.**  Only when ``dry_run=False``.  In a dry run ``submit_order`` is
   never called, which the tests assert against a broker that raises on contact.
4. **Reconcile.**  Internal expected positions versus the broker's truth, with
   every discrepancy reported rather than papered over.

Idempotency
-----------
:meth:`OMS.daily_run` is safe to invoke twice for the same date.  There are two
independent defences, and both are exercised in the module's tests:

* a **processed-date set** persisted to ``state_dir``, so the second call on
  ``2026-08-28`` short-circuits before generating anything - and, because it is
  on disk rather than in memory, it survives the process being restarted by a
  supervisor mid-morning, which is exactly when double-trading happens;
* the **diff itself**, which is naturally idempotent: once the fills have landed,
  the current book equals the target and the delta is zero.  ``force=True``
  bypasses the date guard and demonstrates this second line of defence.

Causality
---------
The OMS never reads a time series.  It consumes a target dictionary that the
caller has already computed from information available at *t-1*, and prices
everything at the broker's current quote.  There is no path by which a future
price can influence an order.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..logging_utils import audit, get_logger
from .broker import QTY_EPS, Broker, Order, OrderSide, OrderStatus, OrderType, Position
from .risk_gate import RiskCheck, RiskGate

__all__ = ["OMS", "OMS_STATE_VERSION"]

_LOG = get_logger("execution.oms")

OMS_STATE_VERSION = 1


def schedule_order(
    order,
    n_slices: int = 1,
    strategy: str = "twap",
    **kwargs,
) -> list:
    """Split one parent order into child orders on an execution schedule.

    The OMS otherwise sends a parent order as a single print, which is what the
    backtest's impact model assumes and what a desk would not do for size. This
    turns a parent into children on a TWAP, VWAP or Almgren-Chriss trajectory
    (see :mod:`tqe.execution.scheduling`), preserving the parent's id as a tag so
    the fills reconcile back.

    ``n_slices=1`` returns the parent untouched, which is the default everywhere
    - slicing is a decision the caller makes deliberately.
    """
    from dataclasses import replace

    from .scheduling import almgren_chriss_schedule, twap_schedule, vwap_schedule

    if n_slices <= 1:
        return [order]

    qty = float(order.quantity)
    if strategy == "vwap":
        sched = vwap_schedule(qty, kwargs.pop("volume_profile", [1.0] * n_slices))
    elif strategy in ("ac", "almgren_chriss"):
        sched = almgren_chriss_schedule(qty, n_slices=n_slices, **kwargs)
    else:
        sched = twap_schedule(qty, n_slices)

    children = []
    for i, q in enumerate(sched.quantities):
        if abs(q) < 1e-9:
            continue
        child = replace(order, quantity=abs(float(q)))
        child.id = f"{order.id}-{i:02d}"
        child.tag = f"{order.tag}|parent:{order.id}|slice:{i}"
        children.append(child)
    return children


class OMS:
    """Order lifecycle, risk routing, persistence and reconciliation.

    Parameters
    ----------
    broker:
        Any object satisfying the :class:`~tqe.execution.broker.Broker` protocol.
    risk_gate:
        Pre-trade gate every order must clear.
    cfg:
        Full :class:`~tqe.config.Config`.  Supplies ``execution.order_type``,
        ``execution.limit_offset_bp``, ``execution.state_dir`` and
        ``portfolio.capital``.  Optional; sensible defaults apply without it.
    state_dir:
        Directory for ``oms_state.json``.  Defaults to ``cfg.execution.state_dir``.
    min_trade_notional:
        Orders below this dollar size are skipped.  Defaults to 25bp of
        ``portfolio.capital`` ($25k on a $10mm book) - roughly the point at which
        the spread paid stops being worth the tracking-error reduction.
    dv01_map:
        Symbol -> DV01 dollars per +1bp per unit, forwarded to the risk gate so
        the DV01 caps are live.
    lot_size:
        Round order quantities to a multiple of this (e.g. ``1.0`` for whole ETF
        shares).  ``None`` leaves quantities continuous, which is right for cash
        bonds quoted per 100 face.
    load_state:
        Restore ``oms_state.json`` at construction.  On by default - that is what
        makes the idempotency guarantee survive a restart.

    Attributes
    ----------
    processed_dates : set[str]
        ISO dates already run.
    """

    def __init__(
        self,
        broker: Broker,
        risk_gate: RiskGate,
        cfg: Config | None = None,
        *,
        state_dir: str | Path | None = None,
        min_trade_notional: float | None = None,
        dv01_map: Mapping[str, float] | None = None,
        lot_size: float | None = None,
        quantity_decimals: int = 6,
        state_file: str = "oms_state.json",
        load_state: bool = True,
    ) -> None:
        self.broker = broker
        self.gate = risk_gate
        self.cfg = cfg

        capital = cfg.portfolio.capital if cfg else 10_000_000.0
        self.min_trade_notional = float(
            min_trade_notional if min_trade_notional is not None else 0.0025 * capital
        )
        self.order_type = OrderType(cfg.execution.order_type if cfg else OrderType.LIMIT)
        self.limit_offset_bp = float(cfg.execution.limit_offset_bp if cfg else 1.0)
        self.dv01_map: dict[str, float] = dict(dv01_map or {})
        self.lot_size = float(lot_size) if lot_size else None
        self.quantity_decimals = int(quantity_decimals)

        base_dir = state_dir if state_dir is not None else (cfg.execution.state_dir if cfg else None)
        self.state_dir: Path | None = Path(base_dir) if base_dir is not None else None
        self.state_file = state_file

        # ---- internal book ------------------------------------------- #
        self._processed: dict[str, dict[str, Any]] = {}
        self._positions: dict[str, float] = {}          # expected quantity by symbol
        self._orders: dict[str, dict[str, Any]] = {}    # id -> serialised order
        self._filled: dict[str, float] = {}             # id -> last-seen filled qty
        self._rejected: list[dict[str, Any]] = []

        if load_state and self.state_dir is not None:
            self.restore(missing_ok=True)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def processed_dates(self) -> set[str]:
        """ISO dates for which :meth:`daily_run` has already executed."""
        return set(self._processed)

    @property
    def rejected_orders(self) -> list[dict[str, Any]]:
        """Every order the risk gate (or venue) refused, with its reason."""
        return list(self._rejected)

    @property
    def internal_positions(self) -> dict[str, float]:
        """Expected quantity per symbol, from this OMS's own fills."""
        return {s: q for s, q in self._positions.items() if abs(q) > QTY_EPS}

    def orders(self, open_only: bool = False) -> list[dict[str, Any]]:
        """Serialised order log, newest last."""
        records = list(self._orders.values())
        if open_only:
            records = [r for r in records if not OrderStatus(r["status"]).is_terminal]
        return records

    def set_dv01_map(self, dv01_map: Mapping[str, float]) -> None:
        """Replace the DV01 map handed to the risk gate."""
        self.dv01_map = dict(dv01_map)

    # ------------------------------------------------------------------ #
    # Pricing helpers
    # ------------------------------------------------------------------ #
    def _quote(self, symbol: str) -> tuple[float, float] | None:
        """``(bid, ask)`` for ``symbol``, or ``None`` when the venue has no market."""
        try:
            bid, ask = self.broker.get_quote(symbol)
        except (KeyError, ValueError):
            return None
        if not (np.isfinite(bid) and np.isfinite(ask)) or bid <= 0 or ask <= 0:
            return None
        return (float(bid), float(ask))

    def _limit_price(self, side: OrderSide, bid: float, ask: float) -> float | None:
        """Marketable limit price: through the touch by ``limit_offset_bp``.

        A daily rebalancer wants certainty of execution far more than it wants
        the last basis point, so the limit is placed *aggressively* - a buy
        priced above the ask, a sell below the bid.  The limit still exists to
        cap the damage if the market gaps between decision and arrival, which is
        the entire reason not to send a naked market order.
        """
        if self.order_type is OrderType.MARKET:
            return None
        offset = self.limit_offset_bp / 1e4
        return ask * (1.0 + offset) if side is OrderSide.BUY else bid * (1.0 - offset)

    def _round_quantity(self, quantity: float) -> float:
        """Apply lot rounding / decimal rounding to a raw quantity."""
        if self.lot_size:
            lots = np.round(quantity / self.lot_size)
            return float(lots * self.lot_size)
        return float(np.round(quantity, self.quantity_decimals))

    # ------------------------------------------------------------------ #
    # 1. Diff
    # ------------------------------------------------------------------ #
    def generate_orders(self, targets: Mapping[str, float], *, tag: str = "") -> list[Order]:
        """Diff target notionals against the live book and emit only the deltas.

        Parameters
        ----------
        targets:
            Symbol -> **signed target dollar notional**.  Negative is short.  A
            symbol that is currently held but absent from ``targets`` is treated
            as a target of zero, i.e. it gets closed - forgetting this is how
            stale positions quietly survive a strategy change.
        tag:
            Provenance string stamped on every order (the OMS uses
            ``"oms:<date>"``).

        Returns
        -------
        list[Order]
            Orders in symbol order, all with positive quantity and a side.

        Notes
        -----
        Current exposure is valued at the **mid**, not at cost: the question
        "how much do I need to trade?" is about market exposure, and pricing the
        existing book at its entry price would make the diff drift with P&L.
        """
        positions = self.broker.get_positions()
        symbols = sorted(set(targets) | set(positions))
        orders: list[Order] = []
        skipped: list[dict[str, Any]] = []

        for symbol in symbols:
            target_notional = float(targets.get(symbol, 0.0))
            quote = self._quote(symbol)
            if quote is None:
                # No market -> no order.  Recorded, not swallowed: a missing quote
                # on a symbol we are trying to exit is an operational incident.
                self._record_rejection(symbol, target_notional, "no_quote", tag)
                continue
            bid, ask = quote
            mid = 0.5 * (bid + ask)

            pos = positions.get(symbol)
            current_qty = pos.quantity if pos else 0.0
            current_notional = current_qty * mid
            delta_notional = target_notional - current_notional

            if abs(delta_notional) < self.min_trade_notional:
                skipped.append(
                    {"symbol": symbol, "delta_notional": delta_notional, "reason": "below_min_trade"}
                )
                continue

            quantity = self._round_quantity(delta_notional / mid)
            if abs(quantity) <= QTY_EPS or abs(quantity * mid) < self.min_trade_notional:
                # Lot rounding can push a marginal trade back under the floor.
                skipped.append(
                    {"symbol": symbol, "delta_notional": delta_notional, "reason": "rounded_to_zero"}
                )
                continue

            side = OrderSide.from_signed(quantity)
            order = Order.from_signed_quantity(
                symbol,
                quantity,
                order_type=self.order_type,
                limit_price=self._limit_price(side, bid, ask),
                tag=tag,
            )
            orders.append(order)
            audit(
                _LOG, "order_generated", order_id=order.id, symbol=symbol, side=side.value,
                quantity=order.quantity, limit_price=order.limit_price, mid=mid,
                target_notional=target_notional, current_notional=current_notional,
                delta_notional=delta_notional, tag=tag,
            )

        if skipped:
            audit(_LOG, "orders_skipped", n=len(skipped), min_trade_notional=self.min_trade_notional,
                  detail=skipped, tag=tag)
        self._last_skipped = skipped
        return orders

    # ------------------------------------------------------------------ #
    # 2/3. Gate and submit
    # ------------------------------------------------------------------ #
    def execute(
        self,
        orders: list[Order],
        dry_run: bool = True,
        *,
        dv01_map: Mapping[str, float] | None = None,
    ) -> list[Order]:
        """Risk-check and (unless ``dry_run``) submit each order.

        Parameters
        ----------
        orders:
            Orders from :meth:`generate_orders` (or anywhere else - they are all
            checked identically).
        dry_run:
            When ``True``, ``broker.submit_order`` is **never** called.  Orders
            that pass the gate come back with status ``NEW`` and a tag suffix of
            ``|dry_run``, so a rehearsal can never be mistaken for a real day in
            the audit log.
        dv01_map:
            Overrides the instance map for this batch.

        Returns
        -------
        list[Order]
            The same order objects, each either submitted, rejected, or left as
            ``NEW`` (dry run).
        """
        dmap = dict(dv01_map) if dv01_map is not None else self.dv01_map
        results: list[Order] = []
        market_open = self.broker.is_market_open()

        for order in orders:
            account = self.broker.get_account()
            positions = self.broker.get_positions()
            quote = self._quote(order.symbol)
            mid = 0.5 * (quote[0] + quote[1]) if quote else None

            check: RiskCheck = self.gate.check_order(
                order,
                account,
                positions,
                dmap,
                reference_price=mid,
                market_open=market_open,
            )
            if not check.passed:
                order.reject(check.reason)
                self._orders[order.id] = order.to_dict()
                self._rejected.append(
                    {
                        "order_id": order.id,
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "quantity": order.quantity,
                        "reason": check.reason,
                        "failed": check.failed_checks,
                        "dry_run": dry_run,
                        "timestamp": order.updated_at.isoformat(),
                    }
                )
                audit(_LOG, "oms_order_rejected", order_id=order.id, symbol=order.symbol,
                      reason=check.reason, failed=check.failed_checks, dry_run=dry_run)
                results.append(order)
                continue

            if dry_run:
                order.tag = f"{order.tag}|dry_run" if order.tag else "dry_run"
                self._orders[order.id] = order.to_dict()
                audit(_LOG, "oms_order_dry_run", order_id=order.id, symbol=order.symbol,
                      side=order.side.value, quantity=order.quantity, limit_price=order.limit_price)
                results.append(order)
                continue

            # ---- live path -------------------------------------------- #
            self._filled[order.id] = 0.0
            submitted = self.broker.submit_order(order)
            self.gate.record_submission(submitted)
            self._orders[submitted.id] = submitted.to_dict()
            audit(_LOG, "oms_order_submitted", order_id=submitted.id, symbol=submitted.symbol,
                  side=submitted.side.value, quantity=submitted.quantity,
                  status=submitted.status.value, filled=submitted.filled_quantity,
                  avg_fill_price=submitted.avg_fill_price)
            results.append(submitted)

        if not dry_run:
            # Fold whatever traded (immediately or on a later quote) into the
            # internal book, so reconcile() compares like with like.
            self.poll_orders()
        return results

    def _record_rejection(self, symbol: str, notional: float, reason: str, tag: str) -> None:
        """Record a pre-order failure (e.g. no market) as a first-class rejection."""
        record = {
            "order_id": None,
            "symbol": symbol,
            "side": None,
            "quantity": None,
            "target_notional": notional,
            "reason": reason,
            "failed": [reason],
            "dry_run": None,
            "timestamp": datetime.now().astimezone().isoformat(),
            "tag": tag,
        }
        self._rejected.append(record)
        audit(_LOG, "oms_order_not_generated", **record)
        _LOG.warning("No order generated for %s: %s", symbol, reason)

    # ------------------------------------------------------------------ #
    # Fill tracking
    # ------------------------------------------------------------------ #
    def poll_orders(self) -> int:
        """Fold newly-filled quantity on OMS orders into the internal book.

        Real venues fill asynchronously: an order acknowledged at 09:30 may only
        complete at 09:34.  Rather than assume, the OMS remembers how much of
        each of *its* orders it has already accounted for and applies the delta.
        This is what keeps :meth:`reconcile` meaningful - it means a difference
        against the broker is a genuine break, not just a fill we had not
        collected yet.

        Returns
        -------
        int
            Number of orders whose fill state advanced.
        """
        updated = 0
        for order_id, seen in list(self._filled.items()):
            live = self.broker.get_order(order_id)
            if live is None:
                continue
            delta = live.filled_quantity - seen
            if abs(delta) > QTY_EPS:
                self._positions[live.symbol] = (
                    self._positions.get(live.symbol, 0.0) + delta * live.side.sign
                )
                self._filled[order_id] = live.filled_quantity
                self._orders[order_id] = live.to_dict()
                updated += 1
            if live.status.is_terminal and abs(live.remaining_quantity) <= QTY_EPS:
                # Fully done: stop polling it, but keep the order record.
                self._filled.pop(order_id, None)
                self._orders[order_id] = live.to_dict()
        return updated

    # ------------------------------------------------------------------ #
    # 4. Reconciliation
    # ------------------------------------------------------------------ #
    def reconcile(self, tolerance: float = 1e-6) -> dict[str, Any]:
        """Compare the OMS's expected book against the broker's actual book.

        Discrepancy kinds reported:

        ``quantity_mismatch``
            Both sides hold the symbol but disagree on size - typically a fill
            the OMS never saw, or a manual trade.
        ``missing_at_broker``
            The OMS thinks it holds something the broker does not - the dangerous
            direction, because the strategy believes it has exposure it does not.
        ``untracked_at_broker``
            The broker holds something the OMS did not create: a manual trade, a
            leftover from a previous strategy, or another process on the account.
        ``open_order_mismatch``
            Working-order counts disagree.

        Parameters
        ----------
        tolerance:
            Absolute quantity difference treated as noise.

        Returns
        -------
        dict
            ``in_sync``, ``n_discrepancies``, ``discrepancies``, both position
            maps, cash/equity and open-order counts.
        """
        self.poll_orders()
        broker_positions: dict[str, Position] = self.broker.get_positions()
        broker_qty = {s: p.quantity for s, p in broker_positions.items()}
        internal_qty = self.internal_positions

        discrepancies: list[dict[str, Any]] = []
        for symbol in sorted(set(broker_qty) | set(internal_qty)):
            mine = internal_qty.get(symbol, 0.0)
            theirs = broker_qty.get(symbol, 0.0)
            diff = mine - theirs
            if abs(diff) <= tolerance:
                continue
            if symbol not in broker_qty:
                kind = "missing_at_broker"
            elif symbol not in internal_qty:
                kind = "untracked_at_broker"
            else:
                kind = "quantity_mismatch"
            discrepancies.append(
                {
                    "symbol": symbol,
                    "kind": kind,
                    "internal_quantity": mine,
                    "broker_quantity": theirs,
                    "difference": diff,
                }
            )

        open_internal = sum(1 for r in self.orders(open_only=True))
        open_broker = len(self.broker.list_orders(open_only=True))
        if open_internal != open_broker:
            discrepancies.append(
                {
                    "symbol": None,
                    "kind": "open_order_mismatch",
                    "internal_quantity": float(open_internal),
                    "broker_quantity": float(open_broker),
                    "difference": float(open_internal - open_broker),
                }
            )

        account = self.broker.get_account()
        report = {
            "as_of": account.timestamp.isoformat(),
            "in_sync": not discrepancies,
            "n_discrepancies": len(discrepancies),
            "discrepancies": discrepancies,
            "internal_positions": internal_qty,
            "broker_positions": broker_qty,
            "cash": account.cash,
            "equity": account.equity,
            "open_orders_internal": open_internal,
            "open_orders_broker": open_broker,
        }
        audit(_LOG, "oms_reconcile", in_sync=report["in_sync"],
              n_discrepancies=len(discrepancies), discrepancies=discrepancies)
        if discrepancies:
            _LOG.error("Reconciliation found %d discrepancies: %s", len(discrepancies), discrepancies)
        return report

    def sync_from_broker(self) -> dict[str, float]:
        """Adopt the broker's positions as the internal truth.

        The manual break-fix path.  Deliberately *not* automatic: silently
        adopting the venue's book would make :meth:`reconcile` incapable of ever
        reporting a problem, which defeats its purpose.
        """
        self._positions = {s: p.quantity for s, p in self.broker.get_positions().items()}
        audit(_LOG, "oms_sync_from_broker", positions=self._positions)
        return dict(self._positions)

    # ------------------------------------------------------------------ #
    # Daily driver
    # ------------------------------------------------------------------ #
    def daily_run(
        self,
        targets: Mapping[str, float],
        dry_run: bool = True,
        *,
        as_of: date | str | None = None,
        force: bool = False,
        dv01_map: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run one trading day end to end, exactly once.

        Parameters
        ----------
        targets:
            Symbol -> signed target dollar notional.
        dry_run:
            When ``True`` nothing is submitted (see :meth:`execute`).
        as_of:
            The trading date this run represents; defaults to today.  This is the
            idempotency key.
        force:
            Re-run a date that has already been processed.  The date guard is
            skipped, but the position diff still applies - so a forced re-run of
            an already-executed day generates zero orders because the book is
            already at the target.
        dv01_map:
            Overrides the instance DV01 map for this run.

        Returns
        -------
        dict
            Run summary: ``date``, ``status`` (``executed`` | ``skipped``),
            ``generated``/``submitted``/``rejected`` counts, the serialised
            orders, the rejections and a reconciliation report.

        Notes
        -----
        A dry run also claims the date.  That is the conservative choice: if a
        rehearsal did not claim it, an operator who reran with ``dry_run=False``
        would trade a day the audit log already shows as processed.  Use
        ``force=True`` to intentionally re-run.
        """
        run_date = _as_date(as_of)
        key = run_date.isoformat()

        if key in self._processed and not force:
            prior = self._processed[key]
            audit(_LOG, "oms_daily_run_skipped", date=key, reason="already_processed",
                  prior_run=prior.get("ran_at"), prior_submitted=prior.get("submitted"))
            _LOG.info("daily_run(%s) skipped - already processed at %s", key, prior.get("ran_at"))
            return {
                "date": key,
                "status": "skipped",
                "reason": "already_processed",
                "dry_run": dry_run,
                "generated": 0,
                "submitted": 0,
                "rejected": 0,
                "orders": [],
                "rejections": [],
                "prior_run": prior,
            }

        account = self.broker.get_account()
        if self.gate.current_day != run_date:
            # Fresh risk day: re-baseline the daily-loss limit and the order counter.
            self.gate.start_day(account.equity, run_date)

        tag = f"oms:{key}"
        orders = self.generate_orders(targets, tag=tag)
        executed = self.execute(orders, dry_run=dry_run, dv01_map=dv01_map)

        submitted = [o for o in executed if o.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL, OrderStatus.FILLED)]
        rejected = [o for o in executed if o.status is OrderStatus.REJECTED]

        post_account = self.broker.get_account()
        portfolio_check = self.gate.check_portfolio(
            self.broker.get_positions(),
            dict(dv01_map) if dv01_map is not None else self.dv01_map,
            post_account.equity,
            self.gate.peak_equity,
        )
        reconciliation = self.reconcile()

        record = {
            "date": key,
            "status": "executed",
            "dry_run": dry_run,
            "forced": force,
            "ran_at": datetime.now().astimezone().isoformat(),
            "generated": len(orders),
            "submitted": len(submitted),
            "rejected": len(rejected),
            "skipped_below_min": len(getattr(self, "_last_skipped", [])),
            "targets": dict(targets),
            "order_ids": [o.id for o in executed],
            "equity": post_account.equity,
            "cash": post_account.cash,
        }
        self._processed[key] = record
        self.persist()

        audit(_LOG, "oms_daily_run", **record, in_sync=reconciliation["in_sync"],
              portfolio_ok=portfolio_check.passed)

        return {
            **record,
            "orders": [o.to_dict() for o in executed],
            "rejections": [
                r for r in self._rejected if r.get("order_id") in {o.id for o in rejected} or r.get("tag") == tag
            ],
            "reconciliation": reconciliation,
            "portfolio_check": portfolio_check.as_dict(),
            "risk_gate": self.gate.state_dict(),
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _resolve_path(self, path: str | Path | None) -> Path:
        if path is not None:
            return Path(path)
        if self.state_dir is None:
            raise ValueError("OMS has no state_dir; pass an explicit path to persist/restore")
        return Path(self.state_dir) / self.state_file

    def state_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot: processed dates, positions, orders, rejections."""
        return {
            "version": OMS_STATE_VERSION,
            "saved_at": datetime.now().astimezone().isoformat(),
            "processed": self._processed,
            "positions": self._positions,
            "orders": self._orders,
            "filled": self._filled,
            "rejected": self._rejected[-1000:],  # bounded: the audit log is the full record
            "risk_gate": self.gate.state_dict(),
        }

    def persist(self, path: str | Path | None = None) -> Path | None:
        """Atomically write OMS state.

        Returns ``None`` when no ``state_dir`` is configured - an in-memory OMS is
        legitimate for a backtest, but it forfeits cross-restart idempotency, and
        that is logged rather than assumed.
        """
        if path is None and self.state_dir is None:
            _LOG.debug("OMS has no state_dir - idempotency is in-memory only for this process")
            return None
        target = self._resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state_dict(), indent=2, default=str))
        tmp.replace(target)
        return target

    def restore(self, path: str | Path | None = None, *, missing_ok: bool = True) -> bool:
        """Load OMS state, including the processed-date set and the gate latch."""
        target = self._resolve_path(path)
        if not target.exists():
            if missing_ok:
                return False
            raise FileNotFoundError(target)
        state = json.loads(target.read_text())
        version = int(state.get("version", 0))
        if version != OMS_STATE_VERSION:
            raise ValueError(f"Unsupported OMS state version {version} (expected {OMS_STATE_VERSION})")
        self._processed = dict(state.get("processed") or {})
        self._positions = {s: float(q) for s, q in (state.get("positions") or {}).items()}
        self._orders = dict(state.get("orders") or {})
        self._filled = {k: float(v) for k, v in (state.get("filled") or {}).items()}
        self._rejected = list(state.get("rejected") or [])
        gate_state = state.get("risk_gate")
        if gate_state:
            # A trip must survive a restart, otherwise restarting the process is
            # an accidental risk-limit override.
            self.gate.load_state(gate_state)
        audit(_LOG, "oms_restored", path=str(target), processed_dates=len(self._processed),
              positions=len(self._positions), tripped=self.gate.is_tripped)
        return True

    def summary(self) -> dict[str, Any]:
        """Compact status line for the CLI / API."""
        return {
            "processed_dates": len(self._processed),
            "last_date": max(self._processed) if self._processed else None,
            "orders": len(self._orders),
            "open_orders": len(self.orders(open_only=True)),
            "rejected": len(self._rejected),
            "positions": self.internal_positions,
            "risk_gate_tripped": self.gate.is_tripped,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"OMS(processed={len(self._processed)}, orders={len(self._orders)}, "
            f"rejected={len(self._rejected)}, gate={'TRIPPED' if self.gate.is_tripped else 'armed'})"
        )


def _as_date(value: date | str | datetime | None) -> date:
    """Coerce ``as_of`` into a plain ``date`` (the idempotency key)."""
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _iter_symbols(*maps: Iterable[str]) -> list[str]:  # pragma: no cover - helper
    """Sorted union of several symbol collections."""
    out: set[str] = set()
    for m in maps:
        out.update(m)
    return sorted(out)
