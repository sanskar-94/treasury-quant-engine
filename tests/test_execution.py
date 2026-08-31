"""Tests for the execution stack.

The position-flip test is the one that matters. A book that goes long, then
sells through zero into a short, must realise P&L on the closed portion at the
*old* average price and restart the average on the residual. Getting this wrong
is the most common bug in a home-grown broker, and it is invisible until the
P&L is reconciled against a real statement.
"""

from __future__ import annotations

import datetime as dt

import pytest

from tqe.config import Config
from tqe.execution.broker import Order, OrderSide, OrderStatus, OrderType
from tqe.execution.oms import OMS
from tqe.execution.paper import PaperBroker
from tqe.execution.risk_gate import RiskGate


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def broker(cfg, tmp_path) -> PaperBroker:
    """A frictionless broker, so P&L assertions are exact."""
    pb = PaperBroker(
        cfg, initial_cash=10_000_000.0, half_spread_bp=0.0, slippage_bp=0.0,
        commission_per_million=0.0, seed=42, state_dir=str(tmp_path),
    )
    pb.set_quote("X", 100.0)
    return pb


def _market(symbol: str, side: OrderSide, qty: float) -> Order:
    return Order(symbol=symbol, side=side, quantity=qty, order_type=OrderType.MARKET)


class TestPaperBrokerAccounting:
    def test_buy_sets_average_price(self, broker):
        broker.submit_order(_market("X", OrderSide.BUY, 100))
        pos = broker.position("X")
        assert pos.quantity == pytest.approx(100)
        assert pos.avg_price == pytest.approx(100.0)
        assert broker.realized_pnl == pytest.approx(0.0)

    def test_averaging_up(self, broker):
        broker.submit_order(_market("X", OrderSide.BUY, 100))
        broker.set_quote("X", 110.0)
        broker.submit_order(_market("X", OrderSide.BUY, 100))
        pos = broker.position("X")
        assert pos.quantity == pytest.approx(200)
        assert pos.avg_price == pytest.approx(105.0)
        assert broker.realized_pnl == pytest.approx(0.0), "adding to a position realises nothing"

    def test_partial_close_realises_proportionally(self, broker):
        broker.submit_order(_market("X", OrderSide.BUY, 100))
        broker.set_quote("X", 110.0)
        broker.submit_order(_market("X", OrderSide.SELL, 60))
        pos = broker.position("X")
        assert pos.quantity == pytest.approx(40)
        assert pos.avg_price == pytest.approx(100.0), "the average must not move on a close"
        assert broker.realized_pnl == pytest.approx(600.0)

    def test_position_flip_through_zero(self, broker):
        """THE test: close 40 at the old average, open a short 20 at the new price."""
        broker.submit_order(_market("X", OrderSide.BUY, 100))
        broker.set_quote("X", 110.0)
        broker.submit_order(_market("X", OrderSide.SELL, 60))     # realise 60 * 10 = 600
        broker.set_quote("X", 120.0)
        broker.submit_order(_market("X", OrderSide.SELL, 60))     # realise 40 * 20 = 800

        pos = broker.position("X")
        assert pos.quantity == pytest.approx(-20)
        assert pos.avg_price == pytest.approx(120.0), "the short must start at the fill price"
        assert broker.realized_pnl == pytest.approx(1400.0)

    def test_closing_a_short_profits_when_price_falls(self, broker):
        broker.submit_order(_market("X", OrderSide.SELL, 100))
        broker.set_quote("X", 90.0)
        broker.submit_order(_market("X", OrderSide.BUY, 100))
        assert broker.position("X").quantity == pytest.approx(0)
        assert broker.realized_pnl == pytest.approx(1000.0)

    def test_full_round_trip_equity(self, broker):
        start = broker.equity
        broker.submit_order(_market("X", OrderSide.BUY, 100))
        broker.set_quote("X", 110.0)
        broker.submit_order(_market("X", OrderSide.SELL, 100))
        assert broker.equity == pytest.approx(start + 1000.0)

    def test_unrealised_marks_to_market(self, broker):
        broker.submit_order(_market("X", OrderSide.BUY, 100))
        broker.set_quote("X", 105.0)
        broker.mark_to_market()
        assert broker.unrealized_pnl == pytest.approx(500.0)


class TestPaperBrokerFills:
    def test_market_orders_cross_the_spread(self, cfg, tmp_path):
        pb = PaperBroker(cfg, initial_cash=1e7, half_spread_bp=10.0, slippage_bp=0.0,
                         commission_per_million=0.0, seed=1, state_dir=str(tmp_path))
        pb.set_quote("Y", 100.0)
        bid, ask = pb.get_quote("Y")
        assert bid < 100.0 < ask
        buy = pb.submit_order(_market("Y", OrderSide.BUY, 10))
        sell = pb.submit_order(_market("Y", OrderSide.SELL, 10))
        assert buy.avg_fill_price == pytest.approx(ask)
        assert sell.avg_fill_price == pytest.approx(bid)

    def test_limit_order_does_not_fill_through_the_touch(self, cfg, tmp_path):
        pb = PaperBroker(cfg, initial_cash=1e7, half_spread_bp=10.0, slippage_bp=0.0,
                         commission_per_million=0.0, seed=1, state_dir=str(tmp_path))
        pb.set_quote("Y", 100.0)
        o = pb.submit_order(Order(symbol="Y", side=OrderSide.BUY, quantity=10,
                                  order_type=OrderType.LIMIT, limit_price=99.0))
        assert o.status is not OrderStatus.FILLED
        assert o.filled_quantity == 0

    def test_marketable_limit_fills(self, cfg, tmp_path):
        pb = PaperBroker(cfg, initial_cash=1e7, half_spread_bp=10.0, slippage_bp=0.0,
                         commission_per_million=0.0, seed=1, state_dir=str(tmp_path))
        pb.set_quote("Y", 100.0)
        o = pb.submit_order(Order(symbol="Y", side=OrderSide.BUY, quantity=10,
                                  order_type=OrderType.LIMIT, limit_price=101.0))
        assert o.status is OrderStatus.FILLED
        assert o.filled_quantity == pytest.approx(10)

    def test_deterministic_under_a_fixed_seed(self, cfg, tmp_path):
        def run():
            pb = PaperBroker(cfg, initial_cash=1e7, half_spread_bp=5.0, slippage_bp=2.0,
                             seed=7, state_dir=str(tmp_path))
            pb.set_quote("Z", 100.0)
            return pb.submit_order(_market("Z", OrderSide.BUY, 100)).avg_fill_price

        assert run() == pytest.approx(run())


class TestPersistence:
    def test_persist_and_restore(self, broker, cfg, tmp_path):
        broker.submit_order(_market("X", OrderSide.BUY, 100))
        broker.set_quote("X", 110.0)
        broker.submit_order(_market("X", OrderSide.SELL, 60))
        broker.persist()

        restored = PaperBroker(cfg, state_dir=str(tmp_path))
        restored.restore()
        assert restored.cash == pytest.approx(broker.cash)
        assert restored.realized_pnl == pytest.approx(broker.realized_pnl)
        assert restored.position("X").quantity == pytest.approx(40)


class TestRiskGate:
    @pytest.fixture
    def gate(self, cfg) -> RiskGate:
        return RiskGate(cfg.risk, cfg.portfolio)

    def _check(self, gate, broker, order):
        return gate.check_order(order, broker.get_account(), broker.get_positions(),
                                reference_price=100.0, market_open=True)

    def test_small_order_passes(self, gate, broker):
        assert self._check(gate, broker, _market("X", OrderSide.BUY, 1_000)).passed

    def test_oversized_order_blocked(self, gate, broker, cfg):
        qty = cfg.risk.max_order_notional / 100.0 * 2
        res = self._check(gate, broker, _market("X", OrderSide.BUY, qty))
        assert not res.passed
        assert "order_notional" in res.reason

    def test_trip_blocks_everything_until_reset(self, gate, broker):
        gate.trip("manual")
        assert gate.is_tripped
        assert not self._check(gate, broker, _market("X", OrderSide.BUY, 10)).passed
        gate.reset()
        assert not gate.is_tripped
        assert self._check(gate, broker, _market("X", OrderSide.BUY, 10)).passed

    def test_config_kill_switch_blocks(self, cfg, broker):
        cfg.risk.kill_switch = True
        gate = RiskGate(cfg.risk, cfg.portfolio)
        assert not self._check(gate, broker, _market("X", OrderSide.BUY, 10)).passed

    def test_checks_dict_reports_every_rule(self, gate, broker):
        res = self._check(gate, broker, _market("X", OrderSide.BUY, 10))
        for rule in ("kill_switch", "not_tripped", "order_notional", "daily_loss", "drawdown"):
            assert rule in res.checks

    def test_gross_dv01_limit_enforced(self, cfg, broker):
        gate = RiskGate(cfg.risk, cfg.portfolio)
        huge_dv01 = {"X": cfg.portfolio.max_gross_dv01}  # 1 unit blows the whole budget
        order = _market("X", OrderSide.BUY, 1_000)
        res = gate.check_order(order, broker.get_account(), broker.get_positions(),
                               huge_dv01, reference_price=100.0, market_open=True)
        assert not res.passed


class TestOMS:
    @pytest.fixture
    def oms(self, cfg, tmp_path):
        pb = PaperBroker(cfg, initial_cash=10_000_000.0, half_spread_bp=1.0,
                         slippage_bp=0.0, seed=3, state_dir=str(tmp_path))
        for s in ("IEF", "TLT"):
            pb.set_quote(s, 95.0)
        gate = RiskGate(cfg.risk, cfg.portfolio)
        return OMS(broker=pb, risk_gate=gate, cfg=cfg, state_dir=str(tmp_path)), pb

    def test_generates_orders_for_new_targets(self, oms):
        o, _ = oms
        orders = o.generate_orders({"IEF": 500_000.0, "TLT": -300_000.0})
        assert len(orders) == 2
        sides = {x.symbol: x.side for x in orders}
        assert sides["IEF"] is OrderSide.BUY
        assert sides["TLT"] is OrderSide.SELL

    def test_dry_run_sends_nothing(self, oms):
        o, pb = oms
        o.daily_run({"IEF": 500_000.0}, dry_run=True, as_of=dt.date(2026, 1, 5))
        assert not pb.get_positions()

    def test_idempotent_on_repeat(self, oms):
        """Re-running a session must never double-trade."""
        o, _ = oms
        day = dt.date(2026, 1, 5)
        targets = {"IEF": 500_000.0, "TLT": -300_000.0}
        first = o.daily_run(targets, dry_run=False, as_of=day)
        second = o.daily_run(targets, dry_run=False, as_of=day)
        assert first["submitted"] == 2
        assert second["generated"] == 0
        assert second["status"] == "skipped"

    def test_reconciles_with_the_broker(self, oms):
        o, _ = oms
        out = o.daily_run({"IEF": 500_000.0}, dry_run=False, as_of=dt.date(2026, 1, 6))
        assert out["reconciliation"]["in_sync"]
        assert out["reconciliation"]["n_discrepancies"] == 0

    def test_only_trades_the_delta(self, oms):
        o, _ = oms
        o.daily_run({"IEF": 500_000.0}, dry_run=False, as_of=dt.date(2026, 1, 7))
        orders = o.generate_orders({"IEF": 600_000.0})
        assert len(orders) == 1
        # ~100k of extra notional at ~95, not a fresh 600k
        assert orders[0].quantity == pytest.approx(100_000 / 95.0, rel=0.05)

    def test_skips_targets_below_the_minimum(self, oms):
        o, _ = oms
        o.daily_run({"IEF": 500_000.0}, dry_run=False, as_of=dt.date(2026, 1, 8))
        assert o.generate_orders({"IEF": 500_000.01}) == []

    def test_rejections_are_recorded_not_dropped(self, oms, cfg):
        o, _ = oms
        o.gate.trip("test")
        out = o.daily_run({"IEF": 500_000.0}, dry_run=False, as_of=dt.date(2026, 1, 9))
        assert out["submitted"] == 0
        assert out["rejected"] >= 1
        assert out["rejections"]
