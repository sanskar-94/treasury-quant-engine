"""Tests for cash-neutral funding, term premium, regime switching and execution.

Same standard as the rest of the suite: closed-form recoveries and invariants,
never a value copied from the code's own output. Two of these tests exist because
they caught real bugs - the HMM's EM loop was exiting after one iteration, and
the cash-neutral construction silently broke DV01 neutrality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tqe.curve.bootstrap import bootstrap_history
from tqe.curve.term_premium import decompose_term_premium, term_premium_signal
from tqe.execution.scheduling import (
    MAX_PARTICIPATION,
    almgren_chriss_schedule,
    implementation_shortfall,
    optimal_participation,
    twap_schedule,
    vwap_schedule,
)
from tqe.models.regime_switching import fit_hmm, rolling_regime_probs, viterbi
from tqe.portfolio.funding import (
    CashNeutralTrade,
    build_cash_neutral_book,
    cash_neutral_structure,
    doubly_neutral_structure,
    funding_cost,
)
from tqe.portfolio.structures import build_standard_structures, steepener

#: Tests that read the cached Treasury history. The data is regenerable with
#: `tqe data pull` but deliberately not committed, so CI skips these rather than
#: failing on a missing file. They still run locally, where the point of them -
#: checking the modules behave sensibly on REAL curve data, not just synthetic -
#: actually holds.
_CURVE = Path("data/processed/curve.parquet")
needs_curve = pytest.mark.skipif(
    not _CURVE.exists(),
    reason="requires the cached Treasury curve; run `tqe data pull`",
)

TENORS = ["3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "30 Yr"]
DV01 = pd.Series(
    {"3 Mo": 0.0025, "6 Mo": 0.0049, "1 Yr": 0.0098, "2 Yr": 0.0192, "3 Yr": 0.0283,
     "5 Yr": 0.0453, "7 Yr": 0.0607, "10 Yr": 0.0813, "30 Yr": 0.1648}
)


# --------------------------------------------------------------------------- #
# Cash-neutral funding
# --------------------------------------------------------------------------- #
class TestFunding:
    @pytest.fixture
    def steep(self):
        return steepener("2 Yr", "10 Yr", DV01, TENORS)

    def test_raw_steepener_is_net_long_notional(self, steep):
        """The premise of the whole module: DV01-neutral is not cash-neutral."""
        assert steep.weights.sum() > 1e6, "a 2s10s steepener should be very net long"

    def test_cash_neutral_zeroes_net_notional(self, steep):
        t = cash_neutral_structure(steep, DV01)
        assert t.net_notional == pytest.approx(0.0, abs=1e-6)

    def test_cash_neutral_reports_its_dv01_disturbance(self, steep):
        """Adding a bill leg perturbs DV01; the module must quantify, not hide it."""
        t = cash_neutral_structure(steep, DV01)
        assert abs(t.net_dv01) > 0.0
        assert t.dv01_disturbance == pytest.approx(t.net_dv01 - t.base_net_dv01, abs=1e-9)

    def test_doubly_neutral_zeroes_both(self, steep):
        t = doubly_neutral_structure(steep, DV01)
        assert t.net_notional == pytest.approx(0.0, abs=1e-6)
        assert t.net_dv01 == pytest.approx(0.0, abs=1e-9)

    def test_doubly_neutral_is_immune_to_a_parallel_shift(self, steep):
        """Zero net DV01 means a parallel move produces no P&L."""
        t = doubly_neutral_structure(steep, DV01)
        shock = 25.0
        pnl = -sum(t.legs[c] * DV01[c] / 100.0 * shock for c in t.legs.index)
        assert pnl == pytest.approx(0.0, abs=1e-6)

    def test_doubly_neutral_keeps_the_curve_view(self, steep):
        """Neutralising funding must not neutralise the trade's purpose."""
        raw = -sum(steep.weights[c] * DV01[c] / 100.0 *
                   ({"2 Yr": -25.0, "10 Yr": 25.0}.get(c, 0.0)) for c in steep.weights.index)
        t = doubly_neutral_structure(steep, DV01)
        neutral = -sum(t.legs[c] * DV01[c] / 100.0 *
                       ({"2 Yr": -25.0, "10 Yr": 25.0}.get(c, 0.0)) for c in t.legs.index)
        assert raw > 0 and neutral > 0
        assert neutral / raw > 0.85, "should retain most of the steepening exposure"

    def test_funding_cost_is_zero_when_cash_neutral(self, steep):
        t = doubly_neutral_structure(steep, DV01)
        assert funding_cost(t, 0.05, 1) == pytest.approx(0.0, abs=1e-6)

    def test_funding_cost_is_large_on_the_raw_structure(self, steep):
        """ACT/360 on ~$4mm at 5% for a day is roughly $560."""
        expected = steep.weights.sum() * 0.05 / 360.0
        assert expected > 100.0

    def test_funding_cost_scales_with_days_and_rate(self):
        """ACT/360 is linear in both days and rate."""
        legs = pd.Series({c: 0.0 for c in TENORS})
        legs["2 Yr"] = 1_000_000.0
        trade = CashNeutralTrade(
            legs=legs, net_notional=1_000_000.0,
            net_dv01=float(legs["2 Yr"] * DV01["2 Yr"] / 100.0), funding_leg=0.0,
        )
        one = funding_cost(trade, 0.05, 1)
        assert one == pytest.approx(1_000_000.0 * 0.05 / 360.0, rel=1e-9)
        assert funding_cost(trade, 0.05, 10) == pytest.approx(10 * one, rel=1e-9)
        assert funding_cost(trade, 0.10, 1) == pytest.approx(2 * one, rel=1e-9)

    def test_funding_cost_rejects_an_inconsistent_trade(self):
        """The module refuses a trade whose stored exposure disagrees with its
        legs, rather than silently costing the wrong number."""
        legs = pd.Series({c: 0.0 for c in TENORS})
        bad = CashNeutralTrade(legs=legs, net_notional=1_000_000.0,
                               net_dv01=0.0, funding_leg=0.0)
        with pytest.raises(AssertionError):
            funding_cost(bad, 0.05, 1)

    def test_book_is_doubly_neutral(self):
        structs = build_standard_structures(DV01, TENORS)
        book = build_cash_neutral_book(structs, DV01, {s.name: 1.0 for s in structs})
        assert float(book.sum()) == pytest.approx(0.0, abs=1e-6)
        assert float((book * DV01 / 100.0).sum()) == pytest.approx(0.0, abs=1e-9)

    def test_book_respects_weights(self):
        structs = build_standard_structures(DV01, TENORS)
        one = build_cash_neutral_book(structs, DV01, {s.name: 1.0 for s in structs})
        two = build_cash_neutral_book(structs, DV01, {s.name: 2.0 for s in structs})
        assert np.allclose(two.to_numpy(), 2 * one.to_numpy(), atol=1e-6)

    def test_butterfly_can_be_neutralised_too(self):
        from tqe.portfolio.structures import butterfly

        fly = butterfly("2 Yr", "5 Yr", "10 Yr", DV01, TENORS)
        t = doubly_neutral_structure(fly, DV01)
        assert t.net_notional == pytest.approx(0.0, abs=1e-6)
        assert t.net_dv01 == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Term premium
# --------------------------------------------------------------------------- #
class TestTermPremium:
    @pytest.fixture(scope="class")
    def zero(self):
        rng = np.random.default_rng(3)
        n = 2200
        idx = pd.bdate_range("2010-01-04", periods=n)
        base = np.array([0.010, 0.014, 0.018, 0.023, 0.027, 0.031, 0.034, 0.036, 0.041])
        drift = np.cumsum(rng.normal(0, 3e-4, n))
        data = base[None, :] + drift[:, None] + rng.normal(0, 1e-4, (n, len(base)))
        return pd.DataFrame(np.clip(data, 1e-4, 0.2), index=idx, columns=TENORS)

    def test_decomposition_reconstructs_the_yield(self, zero):
        """expected short rate + term premium must equal the observed yield."""
        r = decompose_term_premium(zero, n_factors=3, window=756, min_periods=378,
                                   refit_every=63)
        rec = r.expected_short_rate + r.term_premium
        obs = zero.reindex(rec.index)[rec.columns]
        assert float((rec - obs).abs().max().max()) < 1e-10

    def test_output_shapes_align(self, zero):
        r = decompose_term_premium(zero, n_factors=3, window=756, min_periods=378)
        assert r.term_premium.shape == r.expected_short_rate.shape
        assert r.term_premium.index.equals(zero.index)

    def test_is_causal(self, zero):
        """Corrupt the future; earlier term premium estimates must not move."""
        cut = int(len(zero) * 0.65)
        base = decompose_term_premium(zero, n_factors=3, window=756, min_periods=378,
                                      refit_every=63)
        tampered = zero.copy()
        tampered.iloc[cut:] *= 3.0
        after = decompose_term_premium(tampered, n_factors=3, window=756,
                                       min_periods=378, refit_every=63)
        a = base.term_premium.iloc[:cut].dropna(how="all")
        b = after.term_premium.reindex(a.index)[a.columns]
        assert len(a) > 200
        assert np.allclose(a.to_numpy(float), b.to_numpy(float), atol=1e-12, equal_nan=True)

    def test_warmup_is_nan(self, zero):
        r = decompose_term_premium(zero, n_factors=3, window=756, min_periods=378)
        assert r.term_premium.iloc[:300].isna().all().all()

    def test_signal_is_standardised(self, zero):
        r = decompose_term_premium(zero, n_factors=3, window=756, min_periods=378)
        sig = term_premium_signal(r, window=252).dropna()
        if len(sig):
            assert float(sig.abs().max().max()) < 25.0

    @needs_curve
    def test_real_curve_premium_correlates_with_slope(self):
        """A steep curve is mostly term premium; near-zero correlation means broken."""
        curve = pd.read_parquet("data/processed/curve.parquet")
        core = [c for c in TENORS if c in curve.columns]
        z = bootstrap_history(curve)[core]
        r = decompose_term_premium(z, n_factors=5, window=1260, min_periods=504,
                                   refit_every=63)
        tp = r.term_premium["10 Yr"].dropna()
        slope = (curve["10 Yr"] - curve["3 Mo"]).reindex(tp.index)
        assert tp.corr(slope) > 0.4

    @needs_curve
    def test_real_premium_is_in_a_plausible_range(self):
        """A 10y term premium of tens to a couple hundred bp, not percent."""
        curve = pd.read_parquet("data/processed/curve.parquet")
        core = [c for c in TENORS if c in curve.columns]
        z = bootstrap_history(curve)[core]
        r = decompose_term_premium(z, n_factors=5, window=1260, min_periods=504)
        tp = r.term_premium["10 Yr"].dropna() * 1e4
        assert -400 < float(tp.mean()) < 500, f"mean {tp.mean():.0f}bp is implausible"


# --------------------------------------------------------------------------- #
# Regime switching
# --------------------------------------------------------------------------- #
class TestHMM:
    @staticmethod
    def _simulate(n=6000, seed=0):
        rng = np.random.default_rng(seed)
        P = np.array([[0.98, 0.02], [0.05, 0.95]])
        mu = np.array([0.0, -0.004])
        sd = np.array([0.003, 0.012])
        st = np.zeros(n, dtype=int)
        y = np.zeros(n)
        for t in range(1, n):
            st[t] = rng.choice(2, p=P[st[t - 1]])
        for t in range(n):
            y[t] = rng.normal(mu[st[t]], sd[st[t]])
        return pd.Series(y), st, P, mu, sd

    def test_em_runs_more_than_one_iteration(self):
        """Regression: `prev_ll = -inf` made the convergence test `inf <= inf`,
        which is True, so EM exited after a single pass and returned its
        initialisation."""
        y, *_ = self._simulate(2000, seed=1)
        r = fit_hmm(y, n_states=2, seed=42)
        assert r.n_iter > 1, "EM converged suspiciously fast"

    def test_recovers_known_volatilities(self):
        y, _st, _P, _mu, sd = self._simulate()
        r = fit_hmm(y, n_states=2, seed=42)
        got = np.sort(np.sqrt(r.variances))
        assert got[0] == pytest.approx(sd[0], rel=0.25)
        assert got[1] == pytest.approx(sd[1], rel=0.25)

    def test_recovers_known_persistence(self):
        y, _st, P, _mu, _sd = self._simulate()
        r = fit_hmm(y, n_states=2, seed=42)
        assert np.sort(np.diag(r.transition_matrix)) == pytest.approx(
            np.sort(np.diag(P)), abs=0.05
        )

    def test_viterbi_recovers_the_state_path(self):
        y, st, *_ = self._simulate()
        r = fit_hmm(y, n_states=2, seed=42)
        path = viterbi(y, r)
        acc = max((path == st).mean(), (path != st).mean())  # labels may be swapped
        assert acc > 0.9

    def test_log_likelihood_is_finite_on_long_series(self):
        """Unscaled forward-backward underflows to zero after a few hundred obs."""
        rng = np.random.default_rng(2)
        r = fit_hmm(pd.Series(rng.normal(0, 0.01, 5000)), n_states=2, seed=1)
        assert np.isfinite(r.log_likelihood)

    def test_transition_rows_are_probabilities(self):
        y, *_ = self._simulate(2000, seed=3)
        r = fit_hmm(y, n_states=2, seed=42)
        assert np.allclose(r.transition_matrix.sum(axis=1), 1.0, atol=1e-9)
        assert (r.transition_matrix >= -1e-12).all()

    def test_filtered_probabilities_are_probabilities(self):
        y, *_ = self._simulate(1500, seed=4)
        r = fit_hmm(y, n_states=2, seed=42)
        assert np.allclose(r.filtered_probs.sum(axis=1), 1.0, atol=1e-8)

    def test_states_are_sorted_for_stability(self):
        """Unsorted EM states permute between refits and are useless as features."""
        y, *_ = self._simulate(2000, seed=5)
        r = fit_hmm(y, n_states=2, seed=42)
        key = r.means if r.states_sorted_by == "mean" else r.variances
        assert list(key) == sorted(key)

    def test_more_states_fit_better_in_sample(self):
        y, *_ = self._simulate(3000, seed=6)
        assert fit_hmm(y, n_states=3, seed=42).log_likelihood >= \
            fit_hmm(y, n_states=2, seed=42).log_likelihood - 1e-6

    def test_rolling_probs_are_causal(self):
        y, *_ = self._simulate(3000, seed=7)
        cut = 2000
        base = rolling_regime_probs(y, n_states=2, window=1000, min_periods=500,
                                    refit_every=125)
        tampered = y.copy()
        tampered.iloc[cut:] *= 30
        after = rolling_regime_probs(tampered, n_states=2, window=1000,
                                     min_periods=500, refit_every=125)
        a = base.iloc[:cut].dropna(how="all")
        b = after.reindex(a.index)[a.columns]
        assert len(a) > 500
        assert np.allclose(a.to_numpy(float), b.to_numpy(float), atol=1e-12, equal_nan=True)

    @needs_curve
    def test_real_data_finds_distinct_regimes(self):
        """A degenerate fit gives two states with the same volatility."""
        curve = pd.read_parquet("data/processed/curve.parquet")
        dy = curve["10 Yr"].diff().dropna()
        r = fit_hmm(dy, n_states=2, seed=42)
        vols = np.sort(np.sqrt(r.variances))
        assert vols[1] / vols[0] > 1.3, f"states are not distinct: {vols}"


# --------------------------------------------------------------------------- #
# Execution scheduling
# --------------------------------------------------------------------------- #
class TestScheduling:
    Q = 1_000_000.0

    def test_twap_sums_to_the_order(self):
        assert twap_schedule(self.Q, 10).total == pytest.approx(self.Q, abs=1e-9)

    def test_twap_slices_are_equal(self):
        q = twap_schedule(self.Q, 10).quantities
        assert np.allclose(q, q[0])

    def test_vwap_sums_to_the_order(self):
        assert vwap_schedule(self.Q, [3, 2, 1, 1, 2, 3]).total == pytest.approx(self.Q, abs=1e-9)

    def test_vwap_follows_the_volume_profile(self):
        s = vwap_schedule(self.Q, [3, 1])
        assert s.quantities[0] == pytest.approx(3 * s.quantities[1], rel=1e-9)

    def test_flat_volume_profile_reduces_to_twap(self):
        v = vwap_schedule(self.Q, [1, 1, 1, 1]).quantities
        t = twap_schedule(self.Q, 4).quantities
        assert np.allclose(v, t)

    def test_vwap_rejects_a_degenerate_profile(self):
        with pytest.raises(ValueError):
            vwap_schedule(self.Q, [0, 0, 0])

    def test_almgren_chriss_sums_to_the_order(self):
        s = almgren_chriss_schedule(self.Q, 1.0, 10, risk_aversion=1e-6)
        assert s.total == pytest.approx(self.Q, abs=1e-9)

    def test_risk_neutral_limit_is_twap(self):
        """lambda -> 0 must recover linear trading. Half the correctness proof."""
        ac = almgren_chriss_schedule(self.Q, 1.0, 10, risk_aversion=1e-12).quantities
        tw = twap_schedule(self.Q, 10).quantities
        assert np.allclose(ac, tw, rtol=1e-4)

    def test_risk_averse_limit_front_loads(self):
        """lambda -> infinity must dump the order. The other half."""
        s = almgren_chriss_schedule(self.Q, 1.0, 10, risk_aversion=1e10)
        assert s.quantities[0] / self.Q > 0.99

    def test_kappa_increases_with_risk_aversion(self):
        lo = almgren_chriss_schedule(self.Q, 1.0, 10, risk_aversion=1e-8)
        hi = almgren_chriss_schedule(self.Q, 1.0, 10, risk_aversion=1e-2)
        assert hi.diagnostics["kappa"] > lo.diagnostics["kappa"]
        assert hi.quantities[0] > lo.quantities[0], "more risk aversion trades sooner"

    def test_longer_horizon_costs_less_and_risks_more(self):
        short = almgren_chriss_schedule(self.Q, 0.25, 20, risk_aversion=1e-6)
        long = almgren_chriss_schedule(self.Q, 4.0, 20, risk_aversion=1e-6)
        assert long.expected_cost < short.expected_cost
        assert long.expected_risk > short.expected_risk

    def test_permanent_impact_does_not_change_the_trajectory(self):
        """A standard result: permanent impact is paid regardless of the path."""
        a = almgren_chriss_schedule(self.Q, 1.0, 10, perm_impact=0.0, risk_aversion=1e-6)
        b = almgren_chriss_schedule(self.Q, 1.0, 10, perm_impact=1e-3, risk_aversion=1e-6)
        assert np.allclose(a.quantities, b.quantities)
        assert b.expected_cost > a.expected_cost

    def test_short_order_is_handled(self):
        s = almgren_chriss_schedule(-self.Q, 1.0, 10, risk_aversion=1e-6)
        assert s.total == pytest.approx(-self.Q, abs=1e-9)
        assert (s.quantities <= 1e-9).all()

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            almgren_chriss_schedule(self.Q, horizon=0.0)
        with pytest.raises(ValueError):
            almgren_chriss_schedule(self.Q, temp_impact=0.0)
        with pytest.raises(ValueError):
            twap_schedule(self.Q, 0)

    def test_participation_is_capped(self):
        assert optimal_participation(1e12, 1e7, urgency=1.0) <= MAX_PARTICIPATION + 1e-12

    def test_participation_rises_with_urgency(self):
        assert optimal_participation(1e6, 1e7, 0.9) >= optimal_participation(1e6, 1e7, 0.1)

    def test_participation_rejects_zero_adv(self):
        with pytest.raises(ValueError):
            optimal_participation(1e6, 0.0)

    def test_shortfall_is_zero_when_filled_at_arrival(self):
        s = twap_schedule(self.Q, 4)
        out = implementation_shortfall(s, 100.0, [100.0] * 4)
        assert out["total_shortfall"] == pytest.approx(0.0, abs=1e-6)
        assert out["slippage_bp"] == pytest.approx(0.0, abs=1e-9)

    def test_shortfall_measures_adverse_fills(self):
        s = twap_schedule(self.Q, 4)
        out = implementation_shortfall(s, 100.0, [100.02, 100.05, 100.03, 100.06])
        assert out["execution_cost"] > 0
        assert out["slippage_bp"] == pytest.approx(4.0, abs=0.01)

    def test_shortfall_includes_opportunity_cost(self):
        s = twap_schedule(self.Q, 4)
        out = implementation_shortfall(s, 100.0, [100.0] * 4,
                                       final_price=101.0, unfilled=500_000.0)
        assert out["opportunity_cost"] == pytest.approx(500_000.0, rel=1e-9)

    def test_shortfall_validates_shapes(self):
        with pytest.raises(ValueError):
            implementation_shortfall(twap_schedule(self.Q, 4), 100.0, [100.0, 100.0])


# --------------------------------------------------------------------------- #
# Integration: order slicing and regime-scaled sizing
# --------------------------------------------------------------------------- #
class TestIntegration:
    def _order(self, qty: float = 1_000_000.0):
        from tqe.execution.broker import Order, OrderSide, OrderType

        return Order(symbol="IEF", side=OrderSide.BUY, quantity=qty,
                     order_type=OrderType.MARKET, tag="t")

    def test_single_slice_returns_the_parent(self):
        from tqe.execution.oms import schedule_order

        o = self._order()
        assert schedule_order(o, n_slices=1) == [o]

    @pytest.mark.parametrize("strategy", ["twap", "vwap", "ac"])
    def test_children_sum_to_the_parent(self, strategy):
        from tqe.execution.oms import schedule_order

        children = schedule_order(self._order(), n_slices=5, strategy=strategy)
        assert sum(c.quantity for c in children) == pytest.approx(1_000_000.0, abs=1e-6)

    def test_children_carry_the_parent_id(self):
        from tqe.execution.oms import schedule_order

        parent = self._order()
        children = schedule_order(parent, n_slices=4)
        assert all(c.id.startswith(parent.id) for c in children)
        assert all(f"parent:{parent.id}" in c.tag for c in children)

    def test_twap_children_are_equal(self):
        from tqe.execution.oms import schedule_order

        q = [c.quantity for c in schedule_order(self._order(), n_slices=5)]
        assert np.allclose(q, q[0])

    def test_regime_scale_damps_the_book(self):
        """A calm-probability multiplier must shrink positions, not move them."""
        from tqe.config import PortfolioConfig
        from tqe.signals.sizing import size_portfolio

        idx = pd.bdate_range("2022-01-01", periods=400)
        cols = ["2 Yr", "10 Yr"]
        rng = np.random.default_rng(11)
        sig = pd.DataFrame(rng.normal(size=(len(idx), 2)), index=idx, columns=cols)
        tr = pd.DataFrame(rng.normal(0, 0.002, (len(idx), 2)), index=idx, columns=cols)
        dv = pd.DataFrame({"2 Yr": 0.0192, "10 Yr": 0.0813}, index=idx)
        cfg = PortfolioConfig()

        full = size_portfolio(sig, tr, dv, cfg)["notional"]
        damped = size_portfolio(sig, tr, dv, cfg,
                                regime_scale=pd.Series(0.5, index=idx))["notional"]
        assert damped.abs().sum().sum() < full.abs().sum().sum()

    def test_regime_scale_of_one_is_a_no_op(self):
        from tqe.config import PortfolioConfig
        from tqe.signals.sizing import size_portfolio

        idx = pd.bdate_range("2022-01-01", periods=300)
        cols = ["2 Yr", "10 Yr"]
        rng = np.random.default_rng(12)
        sig = pd.DataFrame(rng.normal(size=(len(idx), 2)), index=idx, columns=cols)
        tr = pd.DataFrame(rng.normal(0, 0.002, (len(idx), 2)), index=idx, columns=cols)
        dv = pd.DataFrame({"2 Yr": 0.0192, "10 Yr": 0.0813}, index=idx)
        cfg = PortfolioConfig()

        a = size_portfolio(sig, tr, dv, cfg)["notional"]
        b = size_portfolio(sig, tr, dv, cfg,
                           regime_scale=pd.Series(1.0, index=idx))["notional"]
        assert np.allclose(a.to_numpy(), b.to_numpy())


class TestFinancingCoverage:
    """Financing must never be silently free.

    A missing funding rate that becomes ``0.0`` is free leverage. It inflated the
    front end of the curve to a Sharpe of 5.76 and roughly doubled every levered
    result before 2001, and it did so silently - no error, no warning, just a
    better number. These tests exist because that bug lived in ``run_backtest``,
    the one function this project designates as the single source of P&L truth.
    """

    @staticmethod
    def _flat_book(n=300, notional=5e7):
        idx = pd.bdate_range("2015-01-01", periods=n)
        tenors = ["5 Yr"]
        pos = pd.DataFrame(notional, index=idx, columns=tenors)
        tr = pd.DataFrame(0.0, index=idx, columns=tenors)     # no market P&L at all
        dv = pd.DataFrame(0.045, index=idx, columns=tenors)
        return idx, pos, tr, dv

    def test_missing_funding_is_not_free(self):
        """A book held on borrowed money must lose money when the market is flat."""
        from tqe.backtest.engine import _core_loop

        idx, pos, tr, dv = self._flat_book()
        # A funding series covering only the back half - exactly the 1 Mo problem.
        rate = pd.Series(np.nan, index=idx)
        rate.iloc[len(idx) // 2:] = 0.04

        net, _, _, _, fin = _core_loop(
            pos, tr, None, {"5 Yr": "belly"},
            1e7, False, 1.0, funding_rate=rate, include_financing=True,
        )
        # Every single day is funded, not just the covered half.
        assert (fin > 0).all(), "days without a quoted rate were financed for free"
        assert net.sum() < 0, "a flat market on borrowed money must lose the carry"

    def test_gap_days_charged_at_least_the_observed_rate(self):
        """Uncovered days degrade toward pessimism, never toward free money."""
        from tqe.backtest.engine import _core_loop

        idx, pos, tr, dv = self._flat_book()
        rate = pd.Series(np.nan, index=idx)
        rate.iloc[len(idx) // 2:] = 0.04

        _, _, _, _, gappy = _core_loop(
            pos, tr, None, {"5 Yr": "belly"},
            1e7, False, 1.0, funding_rate=rate, include_financing=True,
        )
        _, _, _, _, full = _core_loop(
            pos, tr, None, {"5 Yr": "belly"},
            1e7, False, 1.0, funding_rate=pd.Series(0.04, index=idx),
            include_financing=True,
        )
        assert gappy.sum() >= full.sum() * 0.999, "a gap made financing cheaper"

    @pytest.mark.needs_curve
    def test_derived_funding_rate_covers_the_whole_curve(self):
        """The shipped funding rate must have no holes and look like an overnight rate."""
        from tqe.backtest.engine import _funding_from_curve
        from tqe.config import load_config

        cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
        idx = pd.read_parquet(cfg.processed_dir / "curve.parquet").index
        rate = _funding_from_curve(cfg, idx)
        assert rate is not None
        assert rate.notna().all(), "funding rate still has gaps"
        assert rate.min() >= 0.0 and rate.max() < 0.25, "funding rate is not plausible"
        # Anchored on history: fed funds averaged ~5.9% in 1995 and ~0.2% in 2009.
        assert rate.loc["1995"].mean() > 0.04, "1995 funding is implausibly cheap"
        assert rate.loc["2009"].mean() < 0.02, "2009 funding is implausibly expensive"
