"""Pre-trade risk gate: the last thing between a model and the market.

Every order in this system passes through :meth:`RiskGate.check_order` before it
reaches a broker.  The gate is deliberately dumb and deliberately hard: it does
not reason about whether a trade is *good*, only about whether it is *allowed*.
That separation is the whole point - alpha logic is complicated and changes
weekly, whereas "never send an order larger than $2mm" must hold even when the
alpha logic is broken, mis-configured, or fed corrupt data.

Design principles
-----------------
**Fail closed.**  A check that cannot be evaluated fails.  If no price is
available for an order, its notional is unknown, so the notional limit is
recorded as *failed* rather than skipped.  An un-priceable order is exactly the
kind of thing that shows up when a data feed breaks, which is precisely when you
least want to be sending orders.

**Every rule reports.**  :class:`RiskCheck` carries a per-rule boolean dict, not
just a verdict.  When an order is blocked at 07:58 you need to know which of
eleven limits bit, without re-running anything.

**The latch is sticky.**  Portfolio-level breaches (daily loss, drawdown, DV01
caps, kill switch) call :meth:`RiskGate.trip`, and once tripped the gate blocks
*everything* until a human calls :meth:`RiskGate.reset`.  Auto-resetting risk
limits is how a bad day becomes a catastrophic one: the same signal that
breached the loss limit is usually still telling you to trade.

**Everything is audited.**  Every rejection goes through
:func:`tqe.logging_utils.audit`, so the JSON audit log alone answers "why did we
not trade that day?".

Causality
---------
The gate is a pure function of the *current* order, account and positions plus
its own accumulated state (orders sent today, day-start equity, peak equity).
It never reads a price series, so it cannot look ahead; the day-start equity
baseline is captured from the first account snapshot seen on a given date and is
never revised with later information.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np

from ..config import Config, PortfolioConfig, RiskConfig
from ..data.calendar import is_business_day
from ..logging_utils import audit, get_logger
from .broker import AccountState, Order, OrderType, Position

__all__ = ["RiskCheck", "RiskGate"]

_LOG = get_logger("execution.risk_gate")


@dataclass
class RiskCheck:
    """Verdict of a risk evaluation.

    Parameters
    ----------
    passed:
        ``True`` only if every rule passed.
    reason:
        Empty when passed; otherwise a compact, machine-greppable description of
        every rule that failed (e.g. ``"order_notional: 3,000,000 > 2,000,000"``).
    checks:
        Per-rule pass/fail, keyed by stable rule names so dashboards and alerts
        can be built on them.
    """

    passed: bool
    reason: str = ""
    checks: dict[str, bool] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Allow ``if gate.check_order(...):`` to read naturally."""
        return self.passed

    @property
    def failed_checks(self) -> list[str]:
        """Names of the rules that failed, in evaluation order."""
        return [name for name, ok in self.checks.items() if not ok]

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe mapping for the audit log."""
        return {"passed": self.passed, "reason": self.reason, "checks": dict(self.checks)}


class RiskGate:
    """Hard pre-trade limits, evaluated per order and per portfolio.

    Parameters
    ----------
    cfg:
        :class:`~tqe.config.RiskConfig` (a full :class:`~tqe.config.Config` is
        also accepted and unpacked, since that is what the CLI has to hand).
    portfolio_cfg:
        :class:`~tqe.config.PortfolioConfig` supplying the DV01 and leverage
        caps.  Optional only when a full ``Config`` was passed as ``cfg``.

    Notes
    -----
    ``dv01_map`` throughout maps *symbol -> DV01 in dollars per +1bp per unit of
    quantity*.  Multiplying by the signed position gives signed portfolio DV01,
    so a long bond contributes positive DV01 and a short contributes negative -
    and ``max_net_dv01`` is checked against the absolute value, because a $15mm
    DV01 short is exactly as dangerous as the long.
    """

    def __init__(
        self,
        cfg: RiskConfig | Config,
        portfolio_cfg: PortfolioConfig | None = None,
    ) -> None:
        if isinstance(cfg, Config):
            portfolio_cfg = portfolio_cfg or cfg.portfolio
            cfg = cfg.risk
        if portfolio_cfg is None:
            raise ValueError("portfolio_cfg is required when cfg is a RiskConfig")
        self.cfg: RiskConfig = cfg
        self.portfolio_cfg: PortfolioConfig = portfolio_cfg

        self._tripped: bool = False
        self._trip_reason: str = ""
        self._tripped_at: datetime | None = None
        self._orders_today: int = 0
        self._current_day: date | None = None
        self._day_start_equity: float | None = None
        self._peak_equity: float = 0.0
        self._rejections: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Latch
    # ------------------------------------------------------------------ #
    @property
    def is_tripped(self) -> bool:
        """``True`` while the gate is latched shut."""
        return self._tripped

    @property
    def trip_reason(self) -> str:
        """Why the gate tripped (empty when it has not)."""
        return self._trip_reason

    def trip(self, reason: str) -> None:
        """Latch the gate shut.

        Idempotent: re-tripping keeps the *original* reason, because the first
        breach is the one that explains the day.
        """
        if self._tripped:
            return
        self._tripped = True
        self._trip_reason = reason
        self._tripped_at = datetime.now().astimezone()
        audit(_LOG, "risk_gate_tripped", reason=reason)
        _LOG.critical("RISK GATE TRIPPED: %s - all further orders blocked until reset()", reason)

    def reset(self) -> None:
        """Release the latch.  Must be an explicit human (or operator) action."""
        was = self._trip_reason
        self._tripped = False
        self._trip_reason = ""
        self._tripped_at = None
        audit(_LOG, "risk_gate_reset", previous_reason=was)
        _LOG.warning("Risk gate reset (was: %s)", was or "not tripped")

    # ------------------------------------------------------------------ #
    # Daily bookkeeping
    # ------------------------------------------------------------------ #
    @property
    def orders_today(self) -> int:
        """Orders recorded as submitted on the current risk day."""
        return self._orders_today

    @property
    def current_day(self) -> date | None:
        """The risk date the daily counters currently refer to."""
        return self._current_day

    @property
    def day_start_equity(self) -> float | None:
        """Equity baseline the daily-loss limit is measured against."""
        return self._day_start_equity

    @property
    def peak_equity(self) -> float:
        """High-water mark of equity, for the drawdown limit."""
        return self._peak_equity

    def start_day(self, equity: float, day: date | None = None) -> None:
        """Open a new risk day: reset the order counter and set the P&L baseline.

        Parameters
        ----------
        equity:
            Account equity at the open.
        day:
            The risk date; defaults to today.
        """
        self._current_day = day or date.today()
        self._orders_today = 0
        self._day_start_equity = float(equity)
        self._peak_equity = max(self._peak_equity, float(equity))

    def _roll_day(self, when: date, equity: float) -> None:
        """Roll the daily counters when the date changes, lazily."""
        if self._current_day != when:
            self.start_day(equity, when)

    def record_submission(self, order: Order) -> int:
        """Count an order that has actually been sent.

        Kept separate from :meth:`check_order` on purpose: checking is a pure,
        repeatable query (the OMS may check the same order twice, e.g. in a dry
        run and then live), whereas *sending* is the event the per-day limit is
        supposed to count.
        """
        self._roll_day(order.created_at.date(), self._day_start_equity or 0.0)
        self._orders_today += 1
        return self._orders_today

    @property
    def rejections(self) -> list[dict[str, Any]]:
        """Audit-shaped record of every rejection this gate has issued."""
        return list(self._rejections)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _reference_price(
        order: Order,
        positions: Mapping[str, Position],
        reference_price: float | None,
    ) -> float:
        """Best available price for valuing ``order``.

        Preference order: an explicitly supplied mark (the OMS passes the live
        mid), then the order's own limit price, then the last mark on the
        existing position.  Returns ``nan`` when nothing is available, which
        makes the notional checks fail closed.
        """
        for candidate in (
            reference_price,
            order.limit_price,
            positions[order.symbol].market_price if order.symbol in positions else None,
        ):
            if candidate is not None and np.isfinite(candidate) and candidate > 0:
                return float(candidate)
        return float("nan")

    @staticmethod
    def _book_dv01(
        positions: Mapping[str, Position],
        dv01_map: Mapping[str, float] | None,
        overrides: Mapping[str, float] | None = None,
    ) -> tuple[float, float]:
        """``(gross, net)`` DV01 of a book, applying post-trade quantity overrides.

        ``overrides`` lets the caller substitute the *post-trade* quantity for
        one symbol, so the limit is tested against the book the order would
        create rather than the book that already exists.
        """
        if not dv01_map:
            return (0.0, 0.0)
        quantities: dict[str, float] = {s: p.quantity for s, p in positions.items()}
        if overrides:
            quantities.update(overrides)
        gross = 0.0
        net = 0.0
        for symbol, qty in quantities.items():
            unit = dv01_map.get(symbol)
            if unit is None:
                continue
            contribution = qty * float(unit)
            gross += abs(contribution)
            net += contribution
        return (gross, net)

    def _fail(
        self,
        checks: dict[str, bool],
        reasons: list[str],
        name: str,
        ok: bool,
        message: str = "",
    ) -> None:
        """Record one rule outcome."""
        checks[name] = bool(ok)
        if not ok:
            reasons.append(message or name)

    # ------------------------------------------------------------------ #
    # Order-level check
    # ------------------------------------------------------------------ #
    def check_order(
        self,
        order: Order,
        account: AccountState,
        positions: Mapping[str, Position],
        dv01_map: Mapping[str, float] | None = None,
        *,
        reference_price: float | None = None,
        market_open: bool | None = None,
    ) -> RiskCheck:
        """Evaluate every hard limit against a single proposed order.

        Parameters
        ----------
        order:
            The proposed order.  Not mutated - the caller decides what to do with
            a rejection.
        account:
            Current account snapshot (cash, equity, timestamp).
        positions:
            Current open positions keyed by symbol.
        dv01_map:
            Symbol -> DV01 dollars per +1bp per unit.  When omitted the DV01 caps
            are recorded as passing, since there is nothing to measure - a rates
            book should always supply it.
        reference_price:
            Mark used to value the order (the OMS passes the live mid).
        market_open:
            Session status.  ``None`` falls back to the bond-market calendar for
            the order's creation date.

        Returns
        -------
        RiskCheck
            Verdict plus the per-rule breakdown.

        Notes
        -----
        Rules evaluated: ``kill_switch``, ``not_tripped``, ``market_open``,
        ``order_valid``, ``order_notional``, ``position_notional``,
        ``orders_per_day``, ``daily_loss``, ``drawdown``, ``gross_dv01``,
        ``net_dv01``.

        A *portfolio* breach (kill switch, daily loss, drawdown) latches the gate;
        a bad individual order is merely rejected.  A single fat-finger order is
        not a reason to stop trading, but a 3% daily loss is.
        """
        checks: dict[str, bool] = {}
        reasons: list[str] = []
        cfg = self.cfg

        equity = float(account.equity)
        self._roll_day(account.timestamp.date(), equity)
        self._peak_equity = max(self._peak_equity, equity)

        # ---- 1. kill switch and latch --------------------------------- #
        self._fail(checks, reasons, "kill_switch", not cfg.kill_switch, "kill_switch: enabled in config")
        self._fail(checks, reasons, "not_tripped", not self._tripped, f"gate_tripped: {self._trip_reason}")

        # ---- 2. session ----------------------------------------------- #
        if cfg.require_market_open:
            is_open = market_open if market_open is not None else is_business_day(order.created_at.date())
            self._fail(checks, reasons, "market_open", is_open, "market_open: market is closed")
        else:
            checks["market_open"] = True

        # ---- 3. order sanity ------------------------------------------ #
        valid = order.quantity > 0 and np.isfinite(order.quantity)
        if order.order_type is OrderType.LIMIT and (
            order.limit_price is None or not np.isfinite(order.limit_price) or order.limit_price <= 0
        ):
            valid = False
        self._fail(checks, reasons, "order_valid", valid, "order_valid: malformed order")

        # ---- 4. notional limits --------------------------------------- #
        price = self._reference_price(order, positions, reference_price)
        order_notional = order.notional(price)
        if np.isnan(order_notional):
            # Fail closed: an order we cannot value is an order we cannot risk.
            self._fail(checks, reasons, "order_notional", False, "order_notional: no reference price")
            self._fail(checks, reasons, "position_notional", False, "position_notional: no reference price")
        else:
            ok = order_notional <= cfg.max_order_notional
            self._fail(
                checks, reasons, "order_notional", ok,
                f"order_notional: {order_notional:,.0f} > {cfg.max_order_notional:,.0f}",
            )
            current_qty = positions[order.symbol].quantity if order.symbol in positions else 0.0
            post_qty = current_qty + order.signed_quantity
            post_notional = abs(post_qty * price)
            ok = post_notional <= cfg.max_position_notional
            self._fail(
                checks, reasons, "position_notional", ok,
                f"position_notional: {order.symbol} {post_notional:,.0f} > {cfg.max_position_notional:,.0f}",
            )

        # ---- 5. order rate -------------------------------------------- #
        ok = self._orders_today < cfg.max_orders_per_day
        self._fail(
            checks, reasons, "orders_per_day", ok,
            f"orders_per_day: {self._orders_today} >= {cfg.max_orders_per_day}",
        )

        # ---- 6. loss limits ------------------------------------------- #
        daily_ok, daily_msg = self._check_daily_loss(equity)
        self._fail(checks, reasons, "daily_loss", daily_ok, daily_msg)
        dd_ok, dd_msg = self._check_drawdown(equity, self._peak_equity)
        self._fail(checks, reasons, "drawdown", dd_ok, dd_msg)

        # ---- 7. DV01 caps on the POST-TRADE book ----------------------- #
        if dv01_map:
            current_qty = positions[order.symbol].quantity if order.symbol in positions else 0.0
            post = {order.symbol: current_qty + order.signed_quantity}
            gross, net = self._book_dv01(positions, dv01_map, post)
            ok = gross <= self.portfolio_cfg.max_gross_dv01
            self._fail(
                checks, reasons, "gross_dv01", ok,
                f"gross_dv01: {gross:,.0f} > {self.portfolio_cfg.max_gross_dv01:,.0f}",
            )
            ok = abs(net) <= self.portfolio_cfg.max_net_dv01
            self._fail(
                checks, reasons, "net_dv01", ok,
                f"net_dv01: |{net:,.0f}| > {self.portfolio_cfg.max_net_dv01:,.0f}",
            )
        else:
            checks["gross_dv01"] = True
            checks["net_dv01"] = True

        passed = all(checks.values())
        result = RiskCheck(passed=passed, reason="; ".join(reasons), checks=checks)

        if not passed:
            self._record_rejection(order, result, order_notional, equity)
            # Account-level breaches latch the gate; a single bad order does not.
            if not checks["kill_switch"]:
                self.trip("kill_switch enabled in config")
            elif not checks["daily_loss"]:
                self.trip(daily_msg)
            elif not checks["drawdown"]:
                self.trip(dd_msg)
        return result

    def _record_rejection(
        self, order: Order, result: RiskCheck, notional: float, equity: float
    ) -> None:
        """Persist and audit a rejection - never drop one silently."""
        record = {
            "order_id": order.id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "notional": None if np.isnan(notional) else notional,
            "equity": equity,
            "reason": result.reason,
            "failed": result.failed_checks,
            "timestamp": order.created_at.isoformat(),
        }
        self._rejections.append(record)
        audit(_LOG, "risk_rejected", **record, checks=result.checks)
        _LOG.warning("Order %s (%s %s %s) REJECTED: %s", order.id[:8], order.side.value,
                     order.quantity, order.symbol, result.reason)

    # ------------------------------------------------------------------ #
    # Loss limits
    # ------------------------------------------------------------------ #
    def _check_daily_loss(self, equity: float) -> tuple[bool, str]:
        """Compare today's P&L against ``max_daily_loss_pct`` of the day's open."""
        base = self._day_start_equity
        if base is None or base <= 0:
            return (True, "")
        pnl_pct = (equity - base) / base
        if pnl_pct < -self.cfg.max_daily_loss_pct:
            return (False, f"daily_loss: {pnl_pct:.2%} worse than {-self.cfg.max_daily_loss_pct:.2%}")
        return (True, "")

    def _check_drawdown(self, equity: float, peak: float) -> tuple[bool, str]:
        """Compare equity against the high-water mark."""
        if peak <= 0:
            return (True, "")
        dd = equity / peak - 1.0
        if dd < -self.cfg.max_drawdown_pct:
            return (False, f"drawdown: {dd:.2%} worse than {-self.cfg.max_drawdown_pct:.2%}")
        return (True, "")

    # ------------------------------------------------------------------ #
    # Portfolio-level check
    # ------------------------------------------------------------------ #
    def check_portfolio(
        self,
        positions: Mapping[str, Position],
        dv01_map: Mapping[str, float] | None,
        equity: float,
        peak_equity: float,
    ) -> RiskCheck:
        """Evaluate book-level limits, independent of any particular order.

        Called after every fill batch and at the end of the day.  Unlike
        :meth:`check_order`, **any** failure here latches the gate: a book that is
        over its DV01 cap or through its drawdown limit must stop trading
        immediately, whatever the next order happens to be.

        Parameters
        ----------
        positions:
            Current positions.
        dv01_map:
            Symbol -> DV01 per unit.
        equity:
            Current net liquidation value.
        peak_equity:
            High-water mark to measure drawdown against; the gate's own internal
            peak is used if this is smaller.

        Returns
        -------
        RiskCheck
            Verdict with rules ``kill_switch``, ``not_tripped``, ``gross_dv01``,
            ``net_dv01``, ``position_notional``, ``leverage``, ``daily_loss``,
            ``drawdown``.
        """
        checks: dict[str, bool] = {}
        reasons: list[str] = []
        equity = float(equity)
        self._peak_equity = max(self._peak_equity, equity, float(peak_equity))

        self._fail(checks, reasons, "kill_switch", not self.cfg.kill_switch, "kill_switch: enabled in config")
        self._fail(checks, reasons, "not_tripped", not self._tripped, f"gate_tripped: {self._trip_reason}")

        gross, net = self._book_dv01(positions, dv01_map)
        if dv01_map:
            ok = gross <= self.portfolio_cfg.max_gross_dv01
            self._fail(checks, reasons, "gross_dv01", ok,
                       f"gross_dv01: {gross:,.0f} > {self.portfolio_cfg.max_gross_dv01:,.0f}")
            ok = abs(net) <= self.portfolio_cfg.max_net_dv01
            self._fail(checks, reasons, "net_dv01", ok,
                       f"net_dv01: |{net:,.0f}| > {self.portfolio_cfg.max_net_dv01:,.0f}")
        else:
            checks["gross_dv01"] = True
            checks["net_dv01"] = True

        worst = max((p.notional for p in positions.values()), default=0.0)
        self._fail(checks, reasons, "position_notional", worst <= self.cfg.max_position_notional,
                   f"position_notional: {worst:,.0f} > {self.cfg.max_position_notional:,.0f}")

        gross_exposure = sum(p.notional for p in positions.values())
        lev = gross_exposure / equity if equity > 0 else 0.0
        self._fail(checks, reasons, "leverage", lev <= self.portfolio_cfg.max_leverage,
                   f"leverage: {lev:.2f}x > {self.portfolio_cfg.max_leverage:.2f}x")

        daily_ok, daily_msg = self._check_daily_loss(equity)
        self._fail(checks, reasons, "daily_loss", daily_ok, daily_msg)
        dd_ok, dd_msg = self._check_drawdown(equity, self._peak_equity)
        self._fail(checks, reasons, "drawdown", dd_ok, dd_msg)

        passed = all(checks.values())
        result = RiskCheck(passed=passed, reason="; ".join(reasons), checks=checks)
        if not passed:
            audit(_LOG, "risk_portfolio_breach", equity=equity, gross_dv01=gross, net_dv01=net,
                  leverage=lev, reason=result.reason, checks=checks)
            # Book-level breaches are always fatal for the session.
            self.trip(result.reason)
        return result

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot of the mutable gate state.

        Persisted by the OMS so that a restart cannot launder away a trip or
        reset the orders-per-day counter.
        """
        return {
            "tripped": self._tripped,
            "trip_reason": self._trip_reason,
            "tripped_at": self._tripped_at.isoformat() if self._tripped_at else None,
            "orders_today": self._orders_today,
            "current_day": self._current_day.isoformat() if self._current_day else None,
            "day_start_equity": self._day_start_equity,
            "peak_equity": self._peak_equity,
            "n_rejections": len(self._rejections),
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore the mutable state written by :meth:`state_dict`."""
        self._tripped = bool(state.get("tripped", False))
        self._trip_reason = str(state.get("trip_reason", ""))
        stamp = state.get("tripped_at")
        self._tripped_at = datetime.fromisoformat(stamp) if stamp else None
        self._orders_today = int(state.get("orders_today", 0))
        day = state.get("current_day")
        self._current_day = date.fromisoformat(day) if day else None
        dse = state.get("day_start_equity")
        self._day_start_equity = float(dse) if dse is not None else None
        self._peak_equity = float(state.get("peak_equity", 0.0))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = f"TRIPPED({self._trip_reason})" if self._tripped else "armed"
        return f"RiskGate({state}, orders_today={self._orders_today}, peak_equity={self._peak_equity:,.0f})"
