"""Tests for splits, metrics, models and the backtest engine.

The tests that matter here are the negative controls: a deliberately leaky split
that ``validate_splits`` must reject, and a deflated Sharpe that must fall as the
number of searched configurations rises. A test suite that only checks the happy
path cannot catch the failure mode this project exists to avoid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tqe.config import Config
from tqe.models import build_ensemble, create_model
from tqe.models.linear import ZeroModel
from tqe.training.metrics import (
    deflated_sharpe_ratio,
    drawdown_series,
    information_coefficient,
    performance_metrics,
    probabilistic_sharpe_ratio,
    rank_information_coefficient,
    regression_metrics,
)
from tqe.training.splits import (
    Split,
    purged_kfold_splits,
    validate_splits,
    walk_forward_splits,
)

IDX = pd.bdate_range("2000-01-03", periods=3000)


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
class TestSplits:
    def test_walk_forward_produces_folds(self):
        s = walk_forward_splits(IDX, n_splits=5, test_size=252, min_train_size=756)
        assert 1 <= len(s) <= 5

    def test_folds_are_forward_chaining(self):
        for f in walk_forward_splits(IDX, n_splits=5, test_size=252, min_train_size=756):
            assert f.train_idx.max() < f.test_idx.min(), "training must precede testing"

    def test_test_blocks_do_not_overlap(self):
        s = walk_forward_splits(IDX, n_splits=5, test_size=252, min_train_size=756)
        seen: set[int] = set()
        for f in s:
            assert not seen & set(f.test_idx.tolist())
            seen |= set(f.test_idx.tolist())

    def test_expanding_window_grows(self):
        s = walk_forward_splits(IDX, n_splits=4, test_size=252, min_train_size=756, expanding=True)
        sizes = [len(f.train_idx) for f in s]
        assert sizes == sorted(sizes) and sizes[0] < sizes[-1]

    def test_rolling_window_stays_bounded(self):
        s = walk_forward_splits(IDX, n_splits=4, test_size=252, min_train_size=756, expanding=False)
        assert all(len(f.train_idx) <= 756 + 1 for f in s)

    @pytest.mark.parametrize("horizon", [1, 5, 21])
    def test_purge_gap_equals_horizon(self, horizon):
        s = walk_forward_splits(IDX, n_splits=3, test_size=252, min_train_size=756, horizon=horizon)
        for f in s:
            gap = int(f.test_idx[0] - f.train_idx[-1] - 1)
            assert gap == horizon - 0, f"expected a {horizon}-row purge, saw {gap}"

    @pytest.mark.parametrize("horizon", [1, 5, 21])
    def test_audit_passes_on_valid_splits(self, horizon):
        s = walk_forward_splits(IDX, n_splits=3, test_size=252, min_train_size=756, horizon=horizon)
        assert validate_splits(s, horizon=horizon)["ok"]

    def test_audit_rejects_a_leaky_split(self):
        """Negative control - the audit is worthless if it cannot fail."""
        bad = Split(
            train_idx=np.arange(0, 1000), test_idx=np.arange(995, 1100),
            train_start=IDX[0], train_end=IDX[999],
            test_start=IDX[995], test_end=IDX[1099], fold=0,
        )
        res = validate_splits([bad], horizon=1)
        assert not res["ok"]
        assert not res["disjoint"]
        assert res["violations"]

    def test_audit_rejects_insufficient_purging(self):
        """A train block butted against the test block leaks even at horizon 1.

        Training row 999's label is the return over day 1000 - the first day of
        the test block - so it must be purged. Correctly purged splits end the
        training block at ``test_start - horizon``.
        """
        butted = Split(
            train_idx=np.arange(0, 1000), test_idx=np.arange(1000, 1100),
            train_start=IDX[0], train_end=IDX[999],
            test_start=IDX[1000], test_end=IDX[1099], fold=0,
        )
        assert not validate_splits([butted], horizon=1)["purged"]
        assert not validate_splits([butted], horizon=10)["purged"]

        properly_purged = Split(
            train_idx=np.arange(0, 999), test_idx=np.arange(1000, 1100),
            train_start=IDX[0], train_end=IDX[998],
            test_start=IDX[1000], test_end=IDX[1099], fold=0,
        )
        assert validate_splits([properly_purged], horizon=1)["ok"]
        # ... but the same split is under-purged for a 10-day target horizon.
        assert not validate_splits([properly_purged], horizon=10)["purged"]

    def test_empty_index(self):
        assert walk_forward_splits(pd.DatetimeIndex([]), n_splits=3) == []

    def test_insufficient_history_yields_nothing(self):
        assert walk_forward_splits(IDX[:100], n_splits=5, test_size=252, min_train_size=1260) == []

    def test_purged_kfold_audit(self):
        s = purged_kfold_splits(IDX, n_splits=5, embargo=5, horizon=1)
        assert len(s) == 5
        assert validate_splits(s, horizon=1)["ok"]

    def test_unsorted_index_rejected(self):
        with pytest.raises(ValueError):
            walk_forward_splits(IDX[::-1], n_splits=3)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
class TestMetrics:
    def test_sharpe_recovers_known_value(self):
        rng = np.random.default_rng(3)
        m, s, n = 0.0005, 0.008, 300_000
        r = pd.Series(rng.normal(m, s, n))
        expected = m / s * np.sqrt(252)
        assert performance_metrics(r)["sharpe"] == pytest.approx(expected, rel=0.05)

    def test_max_drawdown_on_a_known_path(self):
        eq = pd.Series([100, 110, 120, 90, 95, 130])
        assert drawdown_series(eq).min() == pytest.approx(90 / 120 - 1)

    def test_drawdown_is_never_positive(self):
        rng = np.random.default_rng(1)
        eq = (1 + pd.Series(rng.normal(0.0004, 0.01, 2000))).cumprod()
        assert drawdown_series(eq).max() <= 1e-12

    def test_vol_scales_with_sqrt_time(self):
        rng = np.random.default_rng(5)
        r = pd.Series(rng.normal(0, 0.01, 100_000))
        assert performance_metrics(r)["ann_vol"] == pytest.approx(0.01 * np.sqrt(252), rel=0.05)

    def test_all_contract_keys_present(self):
        rng = np.random.default_rng(2)
        m = performance_metrics(pd.Series(rng.normal(0.0003, 0.01, 1000)))
        for k in ("total_return", "cagr", "ann_vol", "sharpe", "sortino", "calmar",
                  "max_drawdown", "max_dd_duration_days", "hit_rate", "profit_factor",
                  "skew", "kurtosis", "var_95", "cvar_95", "best_day", "worst_day"):
            assert k in m

    @pytest.mark.parametrize("series", [
        pd.Series(np.zeros(100)), pd.Series([0.01]), pd.Series(dtype=float),
    ])
    def test_degenerate_input_does_not_raise(self, series):
        performance_metrics(series)

    def test_ic_is_one_for_a_perfect_forecast(self):
        rng = np.random.default_rng(4)
        a = pd.Series(rng.normal(size=2000))
        assert information_coefficient(a, a * 2 + 1) == pytest.approx(1.0)

    def test_ic_is_zero_for_independent_series(self):
        rng = np.random.default_rng(4)
        a, b = rng.normal(size=20_000), rng.normal(size=20_000)
        assert abs(information_coefficient(pd.Series(a), pd.Series(b))) < 0.05

    def test_rank_ic_survives_a_monotone_transform(self):
        rng = np.random.default_rng(6)
        a = pd.Series(rng.normal(size=2000))
        assert rank_information_coefficient(a, np.exp(a)) == pytest.approx(1.0, abs=1e-9)

    def test_deflated_sharpe_falls_with_more_trials(self):
        """The whole point of deflation - searching harder must cost you."""
        vals = [deflated_sharpe_ratio(1.5, n, 2520) for n in (1, 10, 100, 1000, 10_000)]
        assert all(a > b for a, b in zip(vals, vals[1:]))

    def test_deflated_sharpe_is_a_probability(self):
        for n in (1, 50, 5000):
            assert 0.0 <= deflated_sharpe_ratio(1.2, n, 1000) <= 1.0

    def test_deflated_sharpe_low_for_a_weak_result(self):
        assert deflated_sharpe_ratio(0.1, 500, 1000) < 0.5

    def test_psr_rises_with_the_sharpe(self):
        a = probabilistic_sharpe_ratio(0.5, 0.0, 1000, 0.0, 3.0)
        b = probabilistic_sharpe_ratio(1.5, 0.0, 1000, 0.0, 3.0)
        assert b > a

    def test_psr_penalises_negative_skew(self):
        good = probabilistic_sharpe_ratio(1.0, 0.0, 1000, 0.5, 3.0)
        bad = probabilistic_sharpe_ratio(1.0, 0.0, 1000, -0.5, 3.0)
        assert good > bad

    def test_regression_metrics_keys(self):
        rng = np.random.default_rng(9)
        y = rng.normal(size=500)
        m = regression_metrics(y, y * 0.5 + rng.normal(size=500) * 0.3)
        for k in ("rmse", "mae", "r2", "directional_accuracy", "ic", "rank_ic"):
            assert k in m

    def test_rmse_is_zero_for_a_perfect_fit(self):
        y = np.array([1.0, 2.0, 3.0])
        assert regression_metrics(y, y)["rmse"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 10)), columns=[f"f{i}" for i in range(10)])
    y = pd.DataFrame({
        "a": X.f0 * 0.4 + rng.normal(size=400) * 0.5,
        "b": X.f1 * 0.3 + rng.normal(size=400) * 0.5,
    })
    return X, y


class TestModels:
    @pytest.mark.parametrize("name", ["ridge", "lasso", "ols", "random_forest",
                                      "extra_trees", "gbm", "ar_baseline", "zero"])
    def test_fit_predict_shape(self, name, toy):
        X, y = toy
        m = create_model(name).fit(X, y)
        assert m.predict(X).shape == (len(X), y.shape[1])

    def test_predict_before_fit_raises(self, toy):
        X, _ = toy
        with pytest.raises(RuntimeError):
            create_model("ridge").predict(X)

    def test_missing_feature_at_predict_raises(self, toy):
        X, y = toy
        m = create_model("ridge").fit(X, y)
        with pytest.raises(ValueError):
            m.predict(X.drop(columns=["f0"]))

    def test_non_finite_input_rejected(self, toy):
        X, y = toy
        X2 = X.copy()
        X2.iloc[0, 0] = np.nan
        with pytest.raises(ValueError):
            create_model("ridge").fit(X2, y)

    def test_zero_model_predicts_zero(self, toy):
        X, y = toy
        assert np.allclose(ZeroModel().fit(X, y).predict(X), 0.0)

    def test_ridge_beats_zero_on_a_learnable_target(self, toy):
        X, y = toy
        ridge = create_model("ridge").fit(X, y).predict(X)
        zero = ZeroModel().fit(X, y).predict(X)
        assert np.mean((y.to_numpy() - ridge) ** 2) < np.mean((y.to_numpy() - zero) ** 2)

    def test_feature_importance_uses_real_names(self, toy):
        X, y = toy
        imp = create_model("ridge").fit(X, y).feature_importance
        assert imp is not None and set(imp.index) <= set(X.columns)

    def test_ensemble_importance_uses_real_names(self, toy):
        """Base models are fitted on arrays inside the stack; names must survive."""
        X, y = toy
        cfg = Config().model
        cfg.learners = ["ridge", "random_forest"]
        e = build_ensemble(cfg).fit(X, y)
        imp = e.feature_importance
        assert imp is not None and set(imp.index) <= set(X.columns)

    def test_stacked_weights_are_non_negative(self, toy):
        X, y = toy
        cfg = Config().model
        cfg.learners = ["ridge", "ar_baseline", "zero"]
        e = build_ensemble(cfg).fit(X, y)
        assert (e.weights_frame >= -1e-12).all()

    def test_single_learner_is_not_wrapped(self):
        cfg = Config().model
        cfg.learners = ["ridge"]
        assert build_ensemble(cfg).name == "ridge"

    def test_save_load_roundtrip(self, toy, tmp_path):
        X, y = toy
        m = create_model("ridge").fit(X, y)
        p = m.save(tmp_path / "m.joblib")
        from tqe.models.base import BaseModel

        assert np.allclose(BaseModel.load(p).predict(X), m.predict(X))

    def test_bundle_roundtrip(self, toy, tmp_path):
        from tqe.models.registry import load_bundle, save_bundle

        X, y = toy
        m = create_model("ridge").fit(X, y)
        save_bundle(m, None, {"note": "test"}, tmp_path / "bundle")
        m2, _, meta = load_bundle(tmp_path / "bundle")
        assert meta["note"] == "test"
        assert np.allclose(m2.predict(X), m.predict(X))

    def test_unknown_model_name_raises(self):
        with pytest.raises(KeyError):
            create_model("does_not_exist")


# --------------------------------------------------------------------------- #
# Backtest engine
# --------------------------------------------------------------------------- #
class TestBacktest:
    @staticmethod
    def _panels(n=600, seed=3):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2020-01-01", periods=n)
        tenors = ["2 Yr", "5 Yr", "10 Yr"]
        rets = pd.DataFrame(rng.normal(0, 0.003, (n, 3)), index=idx, columns=tenors)
        dv01 = pd.DataFrame({"2 Yr": 0.019, "5 Yr": 0.045, "10 Yr": 0.081},
                            index=idx, columns=tenors)
        return idx, tenors, rets, dv01

    def test_zero_signal_produces_flat_equity(self):
        from tqe.backtest.engine import run_backtest

        idx, tenors, rets, dv01 = self._panels()
        sig = pd.DataFrame(0.0, index=idx, columns=tenors)
        r = run_backtest(sig, rets, dv01, Config(), run_canary=False)
        assert r.equity.std() == pytest.approx(0.0, abs=1e-6)
        assert r.costs.sum() == pytest.approx(0.0)

    def test_costs_reduce_returns(self):
        from tqe.backtest.engine import run_backtest

        idx, tenors, rets, dv01 = self._panels()
        rng = np.random.default_rng(11)
        sig = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=tenors)
        cfg = Config()
        r = run_backtest(sig, rets, dv01, cfg, run_canary=False)
        assert r.metrics["sharpe"] <= r.metrics["sharpe_gross"] + 1e-9
        assert r.costs.sum() > 0

    def test_no_costs_flag(self):
        from tqe.backtest.engine import run_backtest

        idx, tenors, rets, dv01 = self._panels()
        rng = np.random.default_rng(11)
        sig = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=tenors)
        cfg = Config()
        cfg.backtest.include_costs = False
        r = run_backtest(sig, rets, dv01, cfg, run_canary=False)
        assert r.costs.sum() == pytest.approx(0.0)

    def test_perfect_foresight_canary_beats_a_random_signal(self):
        """The canary must represent an unreachable ceiling.

        Compared on a cash-neutral basis: both the canary and the strategy are
        stripped of their funding position, so this measures forecasting skill
        rather than who happened to be on the right side of the repo rate.
        """
        from tqe.backtest.engine import run_backtest

        idx, tenors, rets, dv01 = self._panels()
        rng = np.random.default_rng(11)
        sig = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=tenors)
        neutral = sig.sub(sig.mean(axis=1), axis=0)
        r = run_backtest(neutral, rets, dv01, Config(), run_canary=True)
        assert r.metrics["lookahead_canary_sharpe"] > abs(r.metrics["sharpe"]), (
            f"canary {r.metrics['lookahead_canary_sharpe']:.2f} should exceed "
            f"|honest| {abs(r.metrics['sharpe']):.2f}"
        )

    def test_leverage_cap_is_respected(self):
        """The bug that made this project's first backtest meaningless."""
        from tqe.backtest.engine import run_backtest

        idx, tenors, rets, dv01 = self._panels()
        rng = np.random.default_rng(11)
        sig = pd.DataFrame(rng.normal(size=(len(idx), 3)) * 5, index=idx, columns=tenors)
        cfg = Config()
        r = run_backtest(sig, rets, dv01, cfg, run_canary=False)
        gross = r.positions.abs().sum(axis=1)
        cap = cfg.portfolio.capital * cfg.portfolio.max_leverage
        assert gross.max() <= cap * 1.001, f"gross {gross.max():,.0f} exceeded cap {cap:,.0f}"

    def test_positions_never_see_the_same_day_return(self):
        """Changing a single day's return must not change that day's position."""
        from tqe.backtest.engine import run_backtest

        idx, tenors, rets, dv01 = self._panels()
        rng = np.random.default_rng(11)
        sig = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=tenors)
        base = run_backtest(sig, rets, dv01, Config(), run_canary=False)

        tampered = rets.copy()
        tampered.iloc[400] *= 50.0
        after = run_backtest(sig, tampered, dv01, Config(), run_canary=False)
        assert np.allclose(base.positions.iloc[400].to_numpy(),
                           after.positions.iloc[400].to_numpy())

    def test_mismatched_tenors_raise(self):
        from tqe.backtest.engine import run_backtest

        idx, tenors, rets, dv01 = self._panels()
        sig = pd.DataFrame(0.0, index=idx, columns=["GBP", "EUR"])
        with pytest.raises(ValueError):
            run_backtest(sig, rets, dv01, Config())


# --------------------------------------------------------------------------- #
# Financing
# --------------------------------------------------------------------------- #
class TestFinancing:
    """Regression tests for the most expensive bug this project had.

    The backtest originally computed a *total* return and called it a strategy
    return, charging nothing for the money used to hold the positions. Because a
    three-month bill is nearly riskless, an unfunded backtest scored holding
    cash at a Sharpe above 12, and any strategy with a net long bias inherited a
    large fictitious edge. These tests exist so that cannot come back.
    """

    @staticmethod
    def _panels(n=500, seed=7):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2020-01-01", periods=n)
        tenors = ["2 Yr", "10 Yr"]
        rets = pd.DataFrame(rng.normal(0.00002, 0.002, (n, 2)), index=idx, columns=tenors)
        dv01 = pd.DataFrame({"2 Yr": 0.019, "10 Yr": 0.081}, index=idx, columns=tenors)
        return idx, tenors, rets, dv01

    def test_long_book_pays_financing(self):
        from tqe.backtest.engine import run_backtest

        idx, tenors, rets, dv01 = self._panels()
        pos = pd.DataFrame(1_000_000.0, index=idx, columns=tenors)
        sig = pd.DataFrame(1.0, index=idx, columns=tenors)
        funding = pd.Series(0.05, index=idx)
        cfg = Config()
        r = run_backtest(sig, rets, dv01, cfg, positions=pos,
                         funding_rate=funding, run_canary=False)
        assert r.financing.sum() > 0, "a net long book must pay funding"
        # 2mm notional at 5% for ~2 years
        assert r.metrics["total_financing"] == pytest.approx(
            2_000_000 * 0.05 * (idx[-1] - idx[0]).days / 360.0, rel=0.05
        )

    def test_short_book_receives_financing(self):
        from tqe.backtest.engine import run_backtest

        idx, tenors, rets, dv01 = self._panels()
        pos = pd.DataFrame(-1_000_000.0, index=idx, columns=tenors)
        sig = pd.DataFrame(-1.0, index=idx, columns=tenors)
        r = run_backtest(sig, rets, dv01, Config(), positions=pos,
                         funding_rate=pd.Series(0.05, index=idx), run_canary=False)
        assert r.financing.sum() < 0, "a net short book receives funding"

    def test_cash_neutral_book_pays_no_GC_but_still_pays_the_spread(self):
        """A cash-neutral book escapes GC. It does not escape the balance sheet.

        This test previously asserted ``abs(financing) < 1e-6`` - that a
        cash-neutral book is financed for free. That is what the engine did, not
        what a desk pays, and asserting it made the test a rubber stamp on the
        code's own convention (rule 5). GC accrues on net cash borrowed and the
        repo bid/offer accrues on gross: long the bond you finance at GC + s,
        short it you lend cash at GC - s, so the desk pays s either way.
        """
        from tqe.backtest.engine import run_backtest

        idx, tenors, rets, dv01 = self._panels()
        pos = pd.DataFrame({"2 Yr": 1_000_000.0, "10 Yr": -1_000_000.0}, index=idx)
        sig = pd.DataFrame({"2 Yr": 1.0, "10 Yr": -1.0}, index=idx)
        cfg = Config()
        r = run_backtest(sig, rets, dv01, cfg, positions=pos,
                         funding_rate=pd.Series(0.05, index=idx), run_canary=False)

        spread = cfg.costs.repo_spread_bp / 1e4
        gross = float(pos.abs().sum(axis=1).iloc[0])
        days = np.empty(len(idx))
        days[0] = 1.0
        days[1:] = np.diff(idx.to_numpy().astype("datetime64[D]").astype(float))
        expected = gross * spread * days.sum() / 360.0

        assert r.financing.sum() == pytest.approx(expected, rel=1e-9), (
            "a cash-neutral book must pay exactly gross * repo_spread, "
            "no GC and no free lunch"
        )
        assert r.financing.sum() > 0.0

    def test_net_short_book_is_charged_the_spread_not_credited(self):
        """The sign error: net * spread pays the strategy when it is net short."""
        from tqe.backtest.engine import run_backtest

        idx, tenors, rets, dv01 = self._panels()
        pos = pd.DataFrame({"2 Yr": -1_000_000.0, "10 Yr": -2_000_000.0}, index=idx)
        sig = pd.DataFrame({"2 Yr": -1.0, "10 Yr": -1.0}, index=idx)
        cfg = Config()
        r = run_backtest(sig, rets, dv01, cfg, positions=pos,
                         funding_rate=pd.Series(0.0, index=idx), run_canary=False)
        # GC is zero, so whatever remains is the spread leg alone. It must be a
        # cost to a short book, never a credit.
        assert r.financing.sum() > 0.0, "a net-short book was paid the repo spread"

    def test_financing_is_monotone_in_gross_at_fixed_net(self):
        """More balance sheet must cost more, even when the net is unchanged.

        The invariant that catches a spread charged on net: it makes financing
        completely blind to gross, so a $50mm book and a $200mm book with the
        same net cost exactly the same.
        """
        from tqe.backtest.engine import run_backtest

        idx, tenors, rets, dv01 = self._panels()
        cfg = Config()
        totals = []
        for extra in (0.0, 5_000_000.0, 20_000_000.0):
            pos = pd.DataFrame({"2 Yr": 10_000_000.0 + extra, "10 Yr": -extra},
                               index=idx)
            sig = pd.DataFrame({"2 Yr": 1.0, "10 Yr": -1.0}, index=idx)
            r = run_backtest(sig, rets, dv01, cfg, positions=pos,
                             funding_rate=pd.Series(0.04, index=idx), run_canary=False)
            totals.append(float(r.financing.sum()))
        assert totals[0] < totals[1] < totals[2], (
            f"financing ignored gross notional: {totals}"
        )

    def test_financing_reduces_a_long_book_return(self):
        from tqe.backtest.engine import run_backtest

        idx, tenors, rets, dv01 = self._panels()
        pos = pd.DataFrame(1_000_000.0, index=idx, columns=tenors)
        sig = pd.DataFrame(1.0, index=idx, columns=tenors)
        funding = pd.Series(0.05, index=idx)
        cfg_on, cfg_off = Config(), Config()
        cfg_off.backtest.include_financing = False
        on = run_backtest(sig, rets, dv01, cfg_on, positions=pos,
                          funding_rate=funding, run_canary=False)
        off = run_backtest(sig, rets, dv01, cfg_off, positions=pos,
                           funding_rate=funding, run_canary=False)
        assert on.metrics["ann_return"] < off.metrics["ann_return"]

    def test_riskless_carry_is_not_alpha(self):
        """The headline regression.

        A book that is only ever long a near-riskless instrument yielding the
        funding rate must earn approximately zero once funded. Before the fix
        this scored a Sharpe above 12.
        """
        from tqe.backtest.engine import run_backtest

        n = 500
        idx = pd.bdate_range("2020-01-01", periods=n)
        rate = 0.05
        # A bill returning the funding rate daily with negligible volatility.
        daily = rate / 252.0
        rng = np.random.default_rng(3)
        rets = pd.DataFrame({"3 Mo": daily + rng.normal(0, 1e-6, n)}, index=idx)
        dv01 = pd.DataFrame({"3 Mo": 0.0025}, index=idx)
        pos = pd.DataFrame({"3 Mo": 10_000_000.0}, index=idx)
        sig = pd.DataFrame({"3 Mo": 1.0}, index=idx)

        unfunded = Config()
        unfunded.backtest.include_financing = False
        u = run_backtest(sig, rets, dv01, unfunded, positions=pos,
                         funding_rate=pd.Series(rate, index=idx), run_canary=False)
        f = run_backtest(sig, rets, dv01, Config(), positions=pos,
                         funding_rate=pd.Series(rate, index=idx), run_canary=False)

        assert u.metrics["sharpe"] > 5, "the unfunded bug should look spectacular"
        assert abs(f.metrics["ann_return"]) < 0.01, (
            f"funded carry should be ~0, got {f.metrics['ann_return']:.4%}"
        )
