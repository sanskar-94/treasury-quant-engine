"""Tests for the advanced modelling layer.

Dynamic Nelson-Siegel, P&L attribution, relative-value structures, hedging and
nested tuning. As elsewhere, the assertions are invariants and closed-form
recoveries rather than values copied from the code's own output - a VAR that
recovers a known coefficient matrix, an attribution that sums exactly to the
total, a steepener whose DV01 is genuinely zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tqe.backtest.attribution import attribute_by_factor, attribute_by_source
from tqe.config import Config
from tqe.curve.dynamic import (
    VARModel,
    beta_to_yields,
    dns_forecast_history,
    fit_var,
)
from tqe.curve.nelson_siegel import DIEBOLD_LI_TAU1, SVENSSON_FIXED_TAU2
from tqe.portfolio.hedging import (
    dv01_hedge,
    hedge_effectiveness,
    krd_hedge,
    minimum_variance_hedge,
)
from tqe.portfolio.structures import (
    build_standard_structures,
    butterfly,
    cash_and_duration_neutral,
    steepener,
    structure_returns,
)

TENORS = ["3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "30 Yr"]
DV01 = pd.Series(
    {"3 Mo": 0.0025, "6 Mo": 0.0049, "1 Yr": 0.0098, "2 Yr": 0.0192, "3 Yr": 0.0283,
     "5 Yr": 0.0453, "7 Yr": 0.0607, "10 Yr": 0.0813, "30 Yr": 0.1648}
)


# --------------------------------------------------------------------------- #
# VAR / Dynamic Nelson-Siegel
# --------------------------------------------------------------------------- #
class TestVAR:
    @staticmethod
    def _simulate(A, c, n=6000, seed=0, sd=0.01):
        rng = np.random.default_rng(seed)
        k = len(c)
        y = np.zeros((n, k))
        for t in range(1, n):
            y[t] = c + A @ y[t - 1] + rng.normal(0, sd, k)
        return pd.DataFrame(y, columns=[f"x{i}" for i in range(k)])

    def test_recovers_a_known_coefficient_matrix(self):
        A = np.array([[0.90, 0.05], [0.00, 0.85]])
        c = np.array([0.01, -0.02])
        m = fit_var(self._simulate(A, c), lags=1)
        assert np.abs(m.coefs[0] - A).max() < 0.05
        assert np.abs(m.intercept - c).max() < 0.05

    def test_recovers_a_diagonal_process(self):
        A = np.diag([0.7, 0.5])
        m = fit_var(self._simulate(A, np.zeros(2)), lags=1)
        assert m.coefs[0][0, 0] == pytest.approx(0.7, abs=0.05)
        # Off-diagonals should be near zero for an independent process.
        assert abs(m.coefs[0][0, 1]) < 0.05

    def test_stability_check(self):
        stable = VARModel(coefs=np.array([[[0.5, 0.0], [0.0, 0.5]]]),
                          intercept=np.zeros(2), lags=1)
        explosive = VARModel(coefs=np.array([[[1.5, 0.0], [0.0, 0.9]]]),
                             intercept=np.zeros(2), lags=1)
        assert stable.is_stable()
        assert not explosive.is_stable()

    def test_forecast_of_a_random_walk_is_the_last_value(self):
        m = VARModel(coefs=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
                     intercept=np.zeros(2), lags=1)
        hist = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert np.allclose(m.forecast(hist, steps=5)[-1], [3.0, 4.0])

    def test_forecast_converges_to_the_unconditional_mean(self):
        """A stable VAR must mean-revert, not drift."""
        A = np.array([[0.5, 0.0], [0.0, 0.5]])
        c = np.array([1.0, 2.0])
        m = VARModel(coefs=np.array([A]), intercept=c, lags=1)
        expected = np.linalg.solve(np.eye(2) - A, c)   # (I-A)^-1 c
        assert np.allclose(m.forecast(np.zeros((1, 2)), steps=100)[-1], expected, atol=1e-6)

    def test_multi_lag_fit(self):
        rng = np.random.default_rng(1)
        n = 3000
        y = np.zeros((n, 2))
        for t in range(2, n):
            y[t] = 0.5 * y[t - 1] + 0.3 * y[t - 2] + rng.normal(0, 0.01, 2)
        m = fit_var(pd.DataFrame(y), lags=2)
        assert m.lags == 2
        assert m.coefs.shape == (2, 2, 2)

    def test_rejects_too_few_observations(self):
        with pytest.raises(ValueError):
            fit_var(np.zeros((2, 3)), lags=2)


class TestDNS:
    def test_level_loading_is_flat(self):
        t = np.array([0.25, 2.0, 10.0, 30.0])
        y = beta_to_yields(np.array([[1.0, 0.0, 0.0, 0.0]]), t,
                           DIEBOLD_LI_TAU1, SVENSSON_FIXED_TAU2)[0]
        assert np.allclose(y, 1.0)

    def test_slope_loading_decays_to_zero(self):
        t = np.array([0.083, 1.0, 10.0, 30.0])
        y = beta_to_yields(np.array([[0.0, 1.0, 0.0, 0.0]]), t,
                           DIEBOLD_LI_TAU1, SVENSSON_FIXED_TAU2)[0]
        assert y[0] > 0.9 and y[-1] < 0.1
        assert all(a > b for a, b in zip(y, y[1:])), "slope loading must be monotone"

    def test_curvature_loading_is_humped(self):
        t = np.array([0.083, 0.5, 2.0, 10.0, 30.0])
        y = beta_to_yields(np.array([[0.0, 0.0, 1.0, 0.0]]), t,
                           DIEBOLD_LI_TAU1, SVENSSON_FIXED_TAU2)[0]
        assert y.argmax() not in (0, len(y) - 1), "curvature must peak in the middle"

    def test_short_rate_is_level_plus_slope(self):
        """As t -> 0 the model's yield approaches beta0 + beta1."""
        b = np.array([[0.04, -0.02, 0.01, 0.0]])
        y = beta_to_yields(b, np.array([1e-8]), DIEBOLD_LI_TAU1, SVENSSON_FIXED_TAU2)[0]
        assert y[0] == pytest.approx(0.04 - 0.02, abs=1e-4)

    def test_forecast_history_is_causal(self):
        """Corrupt the future; earlier forecasts must be untouched."""
        rng = np.random.default_rng(4)
        n = 1600
        idx = pd.bdate_range("2015-01-01", periods=n)
        betas = pd.DataFrame(
            {
                "beta0": 0.04 + np.cumsum(rng.normal(0, 2e-4, n)),
                "beta1": -0.01 + np.cumsum(rng.normal(0, 2e-4, n)),
                "beta2": np.cumsum(rng.normal(0, 2e-4, n)),
                "beta3": np.cumsum(rng.normal(0, 2e-4, n)),
                "tau1": DIEBOLD_LI_TAU1, "tau2": SVENSSON_FIXED_TAU2,
            },
            index=idx,
        )
        tmap = {"2 Yr": 2.0, "10 Yr": 10.0}
        cut = 1200
        base = dns_forecast_history(betas, tmap, DIEBOLD_LI_TAU1, SVENSSON_FIXED_TAU2,
                                    window=504, min_periods=252, refit_every=63)
        tampered = betas.copy()
        for c in ("beta0", "beta1", "beta2", "beta3"):
            tampered.iloc[cut:, tampered.columns.get_loc(c)] *= 20.0
        after = dns_forecast_history(tampered, tmap, DIEBOLD_LI_TAU1, SVENSSON_FIXED_TAU2,
                                     window=504, min_periods=252, refit_every=63)
        a = base.iloc[:cut].dropna(how="all")
        b = after.reindex(a.index)[a.columns]
        assert len(a) > 500
        assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                           atol=1e-12, equal_nan=True)

    def test_forecast_history_emits_yield_changes(self):
        rng = np.random.default_rng(5)
        n = 900
        betas = pd.DataFrame(
            {"beta0": 0.04 + np.cumsum(rng.normal(0, 1e-4, n)),
             "beta1": -0.01 + np.cumsum(rng.normal(0, 1e-4, n)),
             "beta2": np.cumsum(rng.normal(0, 1e-4, n)),
             "beta3": np.cumsum(rng.normal(0, 1e-4, n))},
            index=pd.bdate_range("2018-01-01", periods=n),
        )
        out = dns_forecast_history(betas, {"10 Yr": 10.0}, DIEBOLD_LI_TAU1,
                                   SVENSSON_FIXED_TAU2, window=252, min_periods=252)
        assert "f_10 Yr" in out.columns and "dy_10 Yr" in out.columns
        assert out["dy_10 Yr"].notna().sum() > 100


# --------------------------------------------------------------------------- #
# Structures
# --------------------------------------------------------------------------- #
class TestStructures:
    def test_steepener_is_dv01_neutral(self):
        s = steepener("2 Yr", "10 Yr", DV01, TENORS)
        assert s.dv01_of(DV01) == pytest.approx(0.0, abs=1e-9)

    def test_steepener_is_long_the_short_leg(self):
        s = steepener("2 Yr", "10 Yr", DV01, TENORS)
        assert s.weights["2 Yr"] > 0 and s.weights["10 Yr"] < 0

    def test_steepener_leg_dv01_matches_target(self):
        s = steepener("2 Yr", "10 Yr", DV01, TENORS, target_dv01=1000.0)
        assert s.weights["2 Yr"] * DV01["2 Yr"] / 100.0 == pytest.approx(1000.0)

    def test_butterfly_is_dv01_neutral(self):
        b = butterfly("2 Yr", "5 Yr", "10 Yr", DV01, TENORS)
        assert b.dv01_of(DV01) == pytest.approx(0.0, abs=1e-9)

    def test_butterfly_is_long_the_belly(self):
        b = butterfly("2 Yr", "5 Yr", "10 Yr", DV01, TENORS)
        assert b.weights["5 Yr"] > 0
        assert b.weights["2 Yr"] < 0 and b.weights["10 Yr"] < 0

    def test_fifty_fifty_wings_split_dv01_equally(self):
        """Equal wing DV01 is what makes the fly neutral to a linear twist."""
        b = butterfly("2 Yr", "5 Yr", "10 Yr", DV01, TENORS, fifty_fifty=True)
        short_dv01 = b.weights["2 Yr"] * DV01["2 Yr"] / 100.0
        long_dv01 = b.weights["10 Yr"] * DV01["10 Yr"] / 100.0
        assert short_dv01 == pytest.approx(long_dv01, rel=1e-9)

    def test_zero_dv01_leg_rejected(self):
        bad = DV01.copy()
        bad["10 Yr"] = 0.0
        with pytest.raises(ValueError):
            steepener("2 Yr", "10 Yr", bad, TENORS)

    def test_double_neutral_projection(self):
        w = pd.Series(1e6, index=TENORS)
        p = cash_and_duration_neutral(w, DV01)
        assert float(p.sum()) == pytest.approx(0.0, abs=1e-6)
        assert float((p * DV01 / 100.0).sum()) == pytest.approx(0.0, abs=1e-9)

    def test_projection_is_orthogonal(self):
        """The removed component must be spanned by the constraint vectors."""
        rng = np.random.default_rng(3)
        w = pd.Series(rng.normal(0, 1e6, len(TENORS)), index=TENORS)
        p = cash_and_duration_neutral(w, DV01)
        removed = (w - p).to_numpy()
        A = np.vstack([np.ones(len(TENORS)), DV01[TENORS].to_numpy()])
        # residual of regressing `removed` on the constraint rows must vanish
        coef, *_ = np.linalg.lstsq(A.T, removed, rcond=None)
        assert np.allclose(A.T @ coef, removed, atol=1e-6)

    def test_all_standard_structures_are_neutral(self):
        built = build_standard_structures(DV01, TENORS)
        assert len(built) >= 5
        for s in built:
            assert s.dv01_of(DV01) == pytest.approx(0.0, abs=1e-8)

    def test_structure_returns_shape(self):
        idx = pd.bdate_range("2024-01-01", periods=50)
        rets = pd.DataFrame(0.001, index=idx, columns=TENORS)
        r = structure_returns(steepener("2 Yr", "10 Yr", DV01, TENORS), rets)
        assert len(r) == 50


# --------------------------------------------------------------------------- #
# Hedging
# --------------------------------------------------------------------------- #
class TestHedging:
    def test_dv01_hedge_flattens(self):
        n = dv01_hedge(5000.0, DV01["10 Yr"])
        assert 5000.0 + n * DV01["10 Yr"] / 100.0 == pytest.approx(0.0, abs=1e-9)

    def test_dv01_hedge_rejects_zero(self):
        with pytest.raises(ValueError):
            dv01_hedge(1000.0, 0.0)

    def test_krd_hedge_neutralises_when_exactly_determined(self):
        """Two key rates, two instruments: the hedge should be exact."""
        keys = ["2 Yr", "10 Yr"]
        pk = pd.Series({"2 Yr": -1500.0, "10 Yr": 5000.0})
        hk = pd.DataFrame({t: [DV01[k] if k == t else 0.0 for k in keys] for t in keys},
                          index=keys)
        h = krd_hedge(pk, hk, size_penalty=0.0)
        assert h.reduction > 0.999
        assert abs(h.residual_dv01) < 1e-6

    def test_krd_hedge_penalty_shrinks_the_hedge(self):
        keys = ["2 Yr", "10 Yr"]
        pk = pd.Series({"2 Yr": -1500.0, "10 Yr": 5000.0})
        hk = pd.DataFrame({t: [DV01[k] if k == t else 0.0 for k in keys] for t in keys},
                          index=keys)
        small = krd_hedge(pk, hk, size_penalty=1e-6)
        large = krd_hedge(pk, hk, size_penalty=1.0)
        assert large.hedge_cost_notional < small.hedge_cost_notional
        assert large.reduction < small.reduction

    def test_krd_hedge_respects_a_notional_cap(self):
        keys = ["2 Yr", "10 Yr"]
        pk = pd.Series({"2 Yr": -1500.0, "10 Yr": 5000.0})
        hk = pd.DataFrame({t: [DV01[k] if k == t else 0.0 for k in keys] for t in keys},
                          index=keys)
        capped = krd_hedge(pk, hk, size_penalty=0.0, max_notional=1e6)
        assert float(capped.notionals.abs().max()) <= 1e6 + 1e-6

    def test_krd_hedge_requires_shared_key_rates(self):
        pk = pd.Series({"2 Yr": 100.0})
        hk = pd.DataFrame({"x": [1.0]}, index=["30 Yr"])
        with pytest.raises(ValueError):
            krd_hedge(pk, hk)

    def test_minimum_variance_hedge_reduces_variance(self):
        rng = np.random.default_rng(6)
        n = 500
        idx = pd.bdate_range("2022-01-01", periods=n)
        factor = pd.Series(rng.normal(0, 0.01, n), index=idx)
        port = factor * 2.0 + pd.Series(rng.normal(0, 0.002, n), index=idx)
        hedges = pd.DataFrame({"f": factor})
        h = minimum_variance_hedge(port, hedges)
        assert h.reduction > 0.8
        assert h.notionals["f"] == pytest.approx(-2.0, abs=0.15)

    def test_hedge_effectiveness_reports_reduction(self):
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2022-01-01", periods=300)
        u = pd.Series(rng.normal(0, 0.01, 300), index=idx)
        eff = hedge_effectiveness(u, u * 0.2)
        assert eff["variance_reduction"] == pytest.approx(1 - 0.04, abs=0.01)


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #
class TestAttribution:
    @staticmethod
    def _setup(n=400, seed=8):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2022-01-01", periods=n)
        cols = ["2 Yr", "5 Yr", "10 Yr", "30 Yr"]
        dy = pd.DataFrame(rng.normal(0, 0.0005, (n, 4)), index=idx, columns=cols)
        dv = pd.DataFrame({c: DV01[c] for c in cols}, index=idx)
        pos = pd.DataFrame(rng.normal(0, 1e6, (n, 4)), index=idx, columns=cols)
        return pos, dy, dv

    def test_contributions_sum_to_total(self):
        """The decomposition is exact or it is worthless."""
        pos, dy, dv = self._setup()
        a = attribute_by_factor(pos, dy, dv)
        assert np.allclose(a.contributions.sum(axis=1).to_numpy(),
                           a.total.to_numpy(), atol=1e-8)

    def test_total_matches_direct_pnl(self):
        pos, dy, dv = self._setup()
        a = attribute_by_factor(pos, dy, dv)
        expected = -((pos * dv / 100.0) * dy * 1e4).sum(axis=1)
        assert np.allclose(a.total.to_numpy(), expected.to_numpy(), atol=1e-8)

    def test_three_factors_explain_most_variance(self):
        pos, dy, dv = self._setup()
        a = attribute_by_factor(pos, dy, dv)
        assert a.explained_ratio > 0.5

    def test_factor_shares_sum_to_one(self):
        pos, dy, dv = self._setup()
        a = attribute_by_factor(pos, dy, dv)
        assert sum(a.factor_share.values()) == pytest.approx(1.0, abs=1e-9)

    def test_level_only_book_attributes_to_level(self):
        """A book with uniform DV01 across tenors is a pure level bet."""
        rng = np.random.default_rng(9)
        n = 600
        idx = pd.bdate_range("2022-01-01", periods=n)
        cols = ["2 Yr", "5 Yr", "10 Yr", "30 Yr"]
        # Parallel-only yield moves: every tenor shifts together.
        shift = rng.normal(0, 0.0005, n)
        dy = pd.DataFrame({c: shift for c in cols}, index=idx)
        dv = pd.DataFrame({c: DV01[c] for c in cols}, index=idx)
        pos = pd.DataFrame({c: 1e6 / DV01[c] for c in cols}, index=idx)
        a = attribute_by_factor(pos, dy, dv)
        assert a.factor_share["level"] > 0.9, a.factor_share

    def test_rejects_disjoint_inputs(self):
        pos, dy, dv = self._setup()
        with pytest.raises(ValueError):
            attribute_by_factor(pos.rename(columns={"2 Yr": "GBP"})[["GBP"]], dy, dv)

    def test_source_attribution_reconciles(self):
        from tqe.backtest.engine import run_backtest

        rng = np.random.default_rng(10)
        n = 300
        idx = pd.bdate_range("2022-01-01", periods=n)
        cols = ["2 Yr", "10 Yr"]
        rets = pd.DataFrame(rng.normal(0, 0.002, (n, 2)), index=idx, columns=cols)
        dv = pd.DataFrame({c: DV01[c] for c in cols}, index=idx)
        sig = pd.DataFrame(rng.normal(size=(n, 2)), index=idx, columns=cols)
        r = run_backtest(sig, rets, dv, Config(), run_canary=False,
                         funding_rate=pd.Series(0.03, index=idx))
        src = attribute_by_source(r)
        cap = Config().backtest.initial_capital
        assert np.allclose(src["net"].to_numpy(),
                           (r.returns * cap).to_numpy(), atol=1e-6)
