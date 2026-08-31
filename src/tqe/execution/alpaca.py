"""Alpaca broker adapter.

Implements the :class:`~tqe.execution.broker.Broker` protocol against Alpaca's
REST API, so the same OMS, risk gate and live runner that drive the paper broker
can drive a real account with no changes above this layer.

Two deliberate design choices:

**Paper endpoint by default.** ``ALPACA_BASE_URL`` defaults to
``paper-api.alpaca.markets``. Pointing at the live endpoint has to be an explicit
act, and the adapter refuses to start against a live URL unless
``allow_live=True`` is passed as well - a single mistyped environment variable
should not be able to send real orders.

**No dependency at import time.** ``requests`` is the only requirement and the
module imports cleanly without credentials, so ``tqe`` remains installable and
testable on a machine that has never heard of Alpaca.

Cash Treasuries are not tradable through Alpaca, so
``cfg.execution.instrument_map`` maps each CMT tenor onto the closest liquid
Treasury ETF (SHY / IEI / IEF / TLT). The durations are approximate: a strategy
run this way expresses the same *direction* as the research book, not the same
DV01. That is a real limitation and it is stated in the README rather than
hidden here.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from ..config import Config
from ..logging_utils import audit, get_logger
from .broker import (
    AccountState,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

log = get_logger("execution.alpaca")

__all__ = ["AlpacaBroker", "AlpacaError"]

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"

_STATUS_MAP = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.SUBMITTED,
    "accepted_for_bidding": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIAL,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.CANCELLED,
    "canceled": OrderStatus.CANCELLED,
    "expired": OrderStatus.CANCELLED,
    "replaced": OrderStatus.CANCELLED,
    "pending_cancel": OrderStatus.SUBMITTED,
    "pending_replace": OrderStatus.SUBMITTED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
    "stopped": OrderStatus.CANCELLED,
}


class AlpacaError(RuntimeError):
    """Raised when the Alpaca API rejects a request or is unreachable."""


class AlpacaBroker:
    """Live/paper broker backed by Alpaca.

    Parameters
    ----------
    cfg:
        Engine configuration; ``cfg.execution`` supplies order type and
        time-in-force defaults.
    api_key, api_secret:
        Credentials. Read from ``ALPACA_API_KEY_ID`` / ``ALPACA_API_SECRET_KEY``
        when omitted.
    base_url:
        API root. Defaults to ``ALPACA_BASE_URL`` or the paper endpoint.
    allow_live:
        Required to point at the live trading endpoint. Guards against a stray
        environment variable turning a paper session into a real one.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        cfg: Config | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str | None = None,
        allow_live: bool = False,
        timeout: float = 20.0,
    ) -> None:
        self.cfg = cfg or Config()
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY_ID", "")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET_KEY", "")
        self.base_url = (base_url or os.environ.get("ALPACA_BASE_URL") or PAPER_URL).rstrip("/")
        self.timeout = float(timeout)

        if not self.api_key or not self.api_secret:
            raise AlpacaError(
                "Alpaca credentials missing. Set ALPACA_API_KEY_ID and "
                "ALPACA_API_SECRET_KEY (see .env.example)."
            )
        if self.base_url.rstrip("/") == LIVE_URL and not allow_live:
            raise AlpacaError(
                "Refusing to connect to the LIVE Alpaca endpoint without allow_live=True. "
                "This would place real orders with real money."
            )

        self._is_live = self.base_url.rstrip("/") == LIVE_URL
        self._orders: dict[str, Order] = {}
        self._broker_ids: dict[str, str] = {}  # our id -> alpaca id

        import requests

        self._session = requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        })
        log.info("Alpaca broker connected to %s (%s)",
                 self.base_url, "LIVE" if self._is_live else "paper")

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #
    def _request(self, method: str, path: str, *, root: str | None = None, **kw: Any) -> Any:
        url = f"{root or self.base_url}{path}"
        try:
            resp = self._session.request(method, url, timeout=self.timeout, **kw)
        except Exception as exc:  # noqa: BLE001
            raise AlpacaError(f"{method} {path} failed: {exc}") from exc
        if resp.status_code >= 400:
            raise AlpacaError(f"{method} {path} -> {resp.status_code}: {resp.text[:300]}")
        if not resp.content:
            return None
        return resp.json()

    # ------------------------------------------------------------------ #
    # Broker protocol
    # ------------------------------------------------------------------ #
    def submit_order(self, order: Order) -> Order:
        """Send an order and update it in place with the broker's response."""
        payload: dict[str, Any] = {
            "symbol": order.symbol,
            "qty": str(abs(round(order.quantity, 6))),
            "side": "buy" if order.side is OrderSide.BUY else "sell",
            "type": "limit" if order.order_type is OrderType.LIMIT else "market",
            "time_in_force": self.cfg.execution.time_in_force,
            "client_order_id": order.id,
        }
        if order.order_type is OrderType.LIMIT:
            if order.limit_price is None:
                raise AlpacaError(f"Limit order {order.id} has no limit price")
            payload["limit_price"] = str(round(float(order.limit_price), 4))

        try:
            data = self._request("POST", "/v2/orders", json=payload)
        except AlpacaError as exc:
            order.status = OrderStatus.REJECTED
            order.reject_reason = str(exc)[:300]
            order.updated_at = datetime.now(timezone.utc)
            audit(log, "order_rejected", order_id=order.id, symbol=order.symbol,
                  reason=order.reject_reason)
            return order

        self._apply(order, data)
        self._orders[order.id] = order
        self._broker_ids[order.id] = data.get("id", "")
        audit(log, "order_submitted", order_id=order.id, symbol=order.symbol,
              side=order.side.value, quantity=order.quantity, venue="alpaca",
              live=self._is_live)
        return order

    def cancel_order(self, order_id: str) -> bool:
        broker_id = self._broker_ids.get(order_id, order_id)
        try:
            self._request("DELETE", f"/v2/orders/{broker_id}")
        except AlpacaError as exc:
            log.warning("cancel failed for %s: %s", order_id, exc)
            return False
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED
            self._orders[order_id].updated_at = datetime.now(timezone.utc)
        audit(log, "order_cancelled", order_id=order_id)
        return True

    def get_order(self, order_id: str) -> Order | None:
        """Refresh one order from the broker - the broker is the source of truth."""
        broker_id = self._broker_ids.get(order_id, order_id)
        try:
            data = self._request("GET", f"/v2/orders/{broker_id}")
        except AlpacaError:
            return self._orders.get(order_id)
        order = self._orders.get(order_id) or self._from_payload(data)
        self._apply(order, data)
        self._orders[order.id] = order
        return order

    def list_orders(self, open_only: bool = False) -> list[Order]:
        params = {"status": "open" if open_only else "all", "limit": 500}
        try:
            rows = self._request("GET", "/v2/orders", params=params) or []
        except AlpacaError as exc:
            log.warning("list_orders failed: %s", exc)
            return list(self._orders.values())
        out = []
        for row in rows:
            o = self._from_payload(row)
            self._orders[o.id] = o
            out.append(o)
        return out

    def get_account(self) -> AccountState:
        data = self._request("GET", "/v2/account")
        return AccountState(
            cash=float(data.get("cash", 0.0)),
            equity=float(data.get("equity", 0.0)),
            buying_power=float(data.get("buying_power", 0.0)),
            positions=self.get_positions(),
            timestamp=datetime.now(timezone.utc),
        )

    def get_positions(self) -> dict[str, Position]:
        try:
            rows = self._request("GET", "/v2/positions") or []
        except AlpacaError as exc:
            log.warning("get_positions failed: %s", exc)
            return {}
        out: dict[str, Position] = {}
        for row in rows:
            symbol = row["symbol"]
            out[symbol] = Position(
                symbol=symbol,
                quantity=float(row.get("qty", 0.0)),
                avg_price=float(row.get("avg_entry_price", 0.0)),
                market_price=float(row.get("current_price") or 0.0),
                unrealized_pnl=float(row.get("unrealized_pl") or 0.0),
                realized_pnl=0.0,  # Alpaca does not report realised P&L per position
            )
        return out

    def get_quote(self, symbol: str) -> tuple[float, float]:
        """Latest NBBO. Falls back to the last trade when the book is empty."""
        try:
            data = self._request("GET", f"/v2/stocks/{symbol}/quotes/latest", root=DATA_URL)
            q = data.get("quote", {})
            bid, ask = float(q.get("bp", 0.0)), float(q.get("ap", 0.0))
            if bid > 0 and ask > 0:
                return bid, ask
        except AlpacaError as exc:
            log.debug("quote lookup failed for %s: %s", symbol, exc)

        data = self._request("GET", f"/v2/stocks/{symbol}/trades/latest", root=DATA_URL)
        price = float(data.get("trade", {}).get("p", 0.0))
        if price <= 0:
            raise AlpacaError(f"No usable quote for {symbol}")
        # Synthesise a token spread so downstream cost logic still works.
        half = price * 1e-4
        return price - half, price + half

    def is_market_open(self) -> bool:
        try:
            return bool(self._request("GET", "/v2/clock").get("is_open", False))
        except AlpacaError as exc:
            log.warning("clock lookup failed (%s); assuming closed", exc)
            return False

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply(order: Order, data: dict[str, Any]) -> None:
        """Overlay an Alpaca order payload onto our Order object."""
        order.status = _STATUS_MAP.get(str(data.get("status", "")).lower(), OrderStatus.SUBMITTED)
        order.filled_quantity = float(data.get("filled_qty") or 0.0)
        filled_avg = data.get("filled_avg_price")
        order.avg_fill_price = float(filled_avg) if filled_avg else 0.0
        order.updated_at = datetime.now(timezone.utc)
        if order.status is OrderStatus.REJECTED:
            order.reject_reason = str(data.get("status", "rejected"))

    def _from_payload(self, data: dict[str, Any]) -> Order:
        order = Order(
            id=data.get("client_order_id") or data.get("id", ""),
            symbol=data.get("symbol", ""),
            side=OrderSide.BUY if data.get("side") == "buy" else OrderSide.SELL,
            quantity=float(data.get("qty") or 0.0),
            order_type=OrderType.LIMIT if data.get("type") == "limit" else OrderType.MARKET,
            limit_price=float(data["limit_price"]) if data.get("limit_price") else None,
        )
        self._apply(order, data)
        if data.get("id"):
            self._broker_ids[order.id] = data["id"]
        return order

    def __repr__(self) -> str:
        mode = "LIVE" if self._is_live else "paper"
        return f"AlpacaBroker({mode}, {self.base_url})"
