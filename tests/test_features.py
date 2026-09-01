"""Tests for feature engineering.

This is the layer where look-ahead bias enters a system, so most of these are
causality tests rather than correctness tests. The technique used throughout is
the same: corrupt the future, then assert the past is unchanged. A feature that
peeks forward will move; one that does not, cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tqe.config import Config
from tqe.features.builder import build_features, make_targets
from tqe.features.macro import PUBLICATION_LAG_DAYS, apply_publication_lag, macro_features
from tqe.features.regime import regime_features, rolling_regime_labels
from tqe.features.technical import (
    curve_residual_features,
    curve_shape_features,
    mean_reversion_features,
    momentum_features,
    reversal_features,
    volatility_features,
    zscore_features,
)

TENORS = ["3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "30 Yr"]
YEARS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0])


@pytest.fixture
def curve() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2012-01-02", periods=1500, name="date")
    level = 0.03 + np.cumsum(rng.normal(0, 0.0004, len(idx)))
    slope = 0.012 * (np.log(YEARS) - np.log(YEARS).mean()) / np.log(YEARS).std()
    data = level[:, None] + slope[None, :] + rng.normal(0, 0.0002, (len(idx), len(YEARS)))
    return pd.DataFrame(np.clip(data, 0.0005, 0.2), index=idx, columns=TENORS)


def _past_is_unchanged(fn, frame: pd.DataFrame, cut: int = 900, factor: float = 30.0) -> bool:
    """Corrupt everything after ``cut`` and check the earlier output is identical."""
    base = fn(frame)
    tampered = frame.copy()
    tampered.iloc[cut:] *= factor
    after = fn(tampered)
    a = base.iloc[:cut].dropna(how="all")
    b = after.reindex(a.index)[a.columns]
    return np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                       atol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------- #
# Technical blocks
# --------------------------------------------------------------------------- #
class TestTechnical:
    def test_momentum_is_a_trailing_difference(self, curve):
        m = momentum_features(curve[["10 Yr"]], windows=(5,))
        expected = curve["10 Yr"].diff(5)
        assert m["mom5_10 Yr"].equals(expected)

    def test_momentum_is_causal(self, curve):
        assert _past_is_unchanged(lambda d: momentum_features(d, (1, 5, 21)), curve)

    def test_volatility_is_causal(self, curve):
        ch = curve.diff()
        assert _past_is_unchanged(lambda d: volatility_features(d, (10, 21)), ch)

    def test_zscore_is_causal(self, curve):
        assert _past_is_unchanged(lambda d: zscore_features(d, (21, 63)), curve)

    def test_mean_reversion_is_causal(self, curve):
        assert _past_is_unchanged(lambda d: mean_reversion_features(d, (21, 63)), curve)

    def test_zscore_has_roughly_unit_scale(self, curve):
        z = zscore_features(curve[["10 Yr"]], (252,)).dropna()
        assert 0.3 < float(z.std().iloc[0]) < 3.0

    def test_percentile_feature_is_bounded(self, curve):
        mr = mean_reversion_features(curve[["10 Yr"]], (63,)).dropna()
        pct = mr["pct63_10 Yr"]
        assert pct.min() >= -1e-9 and pct.max() <= 1 + 1e-9

    def test_volatility_ratio_present(self, curve):
        v = volatility_features(curve.diff(), (10, 63))
        assert "volratio_10 Yr" in v.columns

    def test_zero_variance_input_does_not_divide_by_zero(self):
        flat = pd.DataFrame({"a": np.full(300, 0.04)},
                            index=pd.bdate_range("2020-01-01", periods=300))
        z = zscore_features(flat, (21,))
        assert not np.isinf(z.to_numpy(dtype=float)).any()


class TestCurveShape:
    def test_slopes_and_flies_computed(self, curve):
        f = curve_shape_features(curve)
        for col in ("slope_2s10s", "slope_3m10y", "fly_2_5_10", "level_10y"):
            assert col in f.columns

    def test_slope_matches_the_definition(self, curve):
        f = curve_shape_features(curve)
        assert np.allclose(f["slope_2s10s"], curve["10 Yr"] - curve["2 Yr"])

    def test_butterfly_matches_the_definition(self, curve):
        f = curve_shape_features(curve)
        expected = 2 * curve["5 Yr"] - curve["2 Yr"] - curve["10 Yr"]
        assert np.allclose(f["fly_2_5_10"], expected)

    def test_missing_leg_is_skipped_not_guessed(self, curve):
        partial = curve.drop(columns=["30 Yr"])
        f = curve_shape_features(partial)
        assert "slope_5s30s" not in f.columns
        assert "slope_2s10s" in f.columns

    def test_inversion_flag_and_duration(self):
        idx = pd.bdate_range("2020-01-01", periods=10)
        df = pd.DataFrame({"2 Yr": [0.02] * 10,
                           "10 Yr": [0.03, 0.03, 0.01, 0.01, 0.01, 0.03, 0.03, 0.01, 0.03, 0.03]},
                          index=idx)
        f = curve_shape_features(df)
        assert list(f["inverted_2s10s"]) == [0, 0, 1, 1, 1, 0, 0, 1, 0, 0]
        assert list(f["inversion_days"]) == [0, 0, 1, 2, 3, 0, 0, 1, 0, 0]


# --------------------------------------------------------------------------- #
# Macro publication lags
# --------------------------------------------------------------------------- #
class TestMacroLags:
    def test_cpi_is_not_visible_before_release(self):
        """The single most valuable test in this file.

        January CPI is stamped 1 January but published mid-February. A model
        must not see it on 2 January.
        """
        macro = pd.DataFrame(
            {"cpi_yoy": [100.0, 101.0, 102.0]},
            index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
        )
        idx = pd.bdate_range("2020-01-02", "2020-03-31")
        out = macro_features(macro, idx, transforms=False)
        col = out["macro_cpi_yoy"]

        assert col.loc["2020-01-02"] != 100.0 or pd.isna(col.loc["2020-01-02"])
        # 45-day lag: the January print becomes visible on/after 15 February.
        assert col.loc["2020-02-20"] == pytest.approx(100.0)

    def test_lag_shifts_forward_not_backward(self):
        macro = pd.DataFrame({"unemployment": [5.0]}, index=pd.to_datetime(["2020-01-01"]))
        lagged = apply_publication_lag(macro)
        assert lagged.index[0] > macro.index[0]

    def test_every_lag_is_positive(self):
        assert all(v > 0 for v in PUBLICATION_LAG_DAYS.values())

    def test_recession_flag_has_a_long_lag(self):
        """NBER dates are announced with roughly a year's delay."""
        assert PUBLICATION_LAG_DAYS["recession"] >= 365

    def test_monthly_series_shifted_by_days_not_rows(self):
        """A row-based shift would move a monthly series by years."""
        macro = pd.DataFrame(
            {"cpi_yoy": [1.0, 2.0, 3.0]},
            index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
        )
        lagged = apply_publication_lag(macro)
        shift_days = (lagged.index[0] - macro.index[0]).days
        assert shift_days == PUBLICATION_LAG_DAYS["cpi_yoy"]

    def test_empty_macro_returns_an_indexed_frame(self):
        idx = pd.bdate_range("2020-01-01", periods=10)
        out = macro_features(pd.DataFrame(), idx)
        assert out.empty and out.index.equals(idx)

    def test_macro_block_is_causal(self):
        idx = pd.bdate_range("2015-01-01", periods=800)
        rng = np.random.default_rng(2)
        macro = pd.DataFrame({"vix": rng.uniform(10, 40, 800)}, index=idx)
        base = macro_features(macro, idx)
        tampered = macro.copy()
        tampered.iloc[500:] *= 50
        after = macro_features(tampered, idx)
        a, b = base.iloc[:490], after.iloc[:490]
        assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)


# --------------------------------------------------------------------------- #
# Regimes
# --------------------------------------------------------------------------- #
class TestRegime:
    def test_regime_block_builds(self, curve):
        f = regime_features(curve, n_states=3)
        assert not f.empty
        assert any(c.startswith("reg_") for c in f.columns)

    def test_volatility_regime_takes_three_values(self, curve):
        f = regime_features(curve, n_states=3)
        vals = set(f["reg_vol_state"].dropna().unique())
        assert vals <= {0.0, 1.0, 2.0}

    def test_rolling_labels_are_causal(self, curve):
        basis = pd.DataFrame(
            {
                "vol": curve["10 Yr"].diff().rolling(21).std(),
                "trend": curve["10 Yr"].diff(63),
                "slope": curve["10 Yr"] - curve["2 Yr"],
            }
        ).dropna()
        base = rolling_regime_labels(basis, n_states=3, window=252, refit_every=63)
        tampered = basis.copy()
        tampered.iloc[800:] *= 40
        after = rolling_regime_labels(tampered, n_states=3, window=252, refit_every=63)
        a = base.iloc[:790].dropna(how="all")
        if len(a) < 50:
            pytest.skip("not enough warmed-up rows to compare")
        b = after.reindex(a.index)
        assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                           atol=1e-10, equal_nan=True)


# --------------------------------------------------------------------------- #
# Targets and assembly
# --------------------------------------------------------------------------- #
class TestTargets:
    @staticmethod
    def _returns(curve):
        from tqe.data.universe import constant_maturity_total_return

        return constant_maturity_total_return(curve, TENORS)

    def test_target_is_the_next_days_return(self, curve):
        rets = self._returns(curve)
        y = make_targets(rets, "price_return", horizon=1, tenors=["10 Yr"])
        realised = rets["10 Yr"]["price_return"]
        # row t of y must equal the return realised on t+1
        assert y["10 Yr"].iloc[10] == pytest.approx(realised.iloc[11])

    def test_last_row_target_is_nan(self, curve):
        y = make_targets(self._returns(curve), "price_return", 1, ["10 Yr"])
        assert pd.isna(y["10 Yr"].iloc[-1]), "there is no tomorrow for the final row"

    def test_multi_day_horizon_compounds(self, curve):
        rets = self._returns(curve)
        y = make_targets(rets, "total_return", horizon=5, tenors=["10 Yr"])
        r = rets["10 Yr"]["total_return"]
        expected = float((1 + r.iloc[11:16]).prod() - 1)
        assert y["10 Yr"].iloc[10] == pytest.approx(expected, abs=1e-12)

    def test_direction_target_is_binary(self, curve):
        y = make_targets(self._returns(curve), "direction", 1, ["10 Yr"]).dropna()
        assert set(y["10 Yr"].unique()) <= {0.0, 1.0}

    def test_unknown_target_raises(self, curve):
        with pytest.raises(KeyError):
            make_targets(self._returns(curve), "not_a_column", 1, ["10 Yr"])


class TestBuildFeatures:
    @pytest.fixture
    def built(self, curve):
        from tqe.data.universe import constant_maturity_total_return

        cfg = Config()
        cfg.data.core_tenors = TENORS
        cfg.features.include_macro = False
        cfg.features.momentum_windows = [1, 5, 21]
        cfg.features.vol_windows = [10, 21]
        cfg.features.zscore_windows = [21, 63]
        cfg.features.min_feature_coverage = 0.5
        rets = constant_maturity_total_return(curve, TENORS)
        return build_features(curve, pd.DataFrame(), cfg, returns=rets), cfg

    def test_output_is_clean_and_aligned(self, built):
        fs, _ = built
        assert len(fs) > 200
        assert not fs.X.isna().any().any()
        assert not fs.y.isna().any().any()
        assert fs.X.index.equals(fs.y.index)

    def test_no_feature_correlates_suspiciously_with_the_target(self, built):
        """The leak canary. A single feature above ~0.35 means a bug."""
        fs, _ = built
        worst = float(fs.X.corrwith(fs.y.mean(axis=1)).abs().max())
        assert worst < 0.35, f"feature correlating {worst:.3f} with next-day return"

    def test_feature_lag_is_applied(self, curve):
        """Raising the lag must shift the features, not merely reorder them."""
        from tqe.data.universe import constant_maturity_total_return

        rets = constant_maturity_total_return(curve, TENORS)

        def build(lag):
            cfg = Config()
            cfg.data.core_tenors = TENORS
            cfg.features.include_macro = False
            cfg.features.include_regime = False
            cfg.features.momentum_windows = [5]
            cfg.features.vol_windows = [21]
            cfg.features.zscore_windows = [63]
            cfg.features.min_feature_coverage = 0.5
            cfg.features.feature_lag = lag
            return build_features(curve, pd.DataFrame(), cfg, returns=rets)

        a, b = build(1), build(3)
        common = a.X.index.intersection(b.X.index)
        col = a.X.columns[0]
        assert not np.allclose(a.X.loc[common, col], b.X.loc[common, col])

    def test_metadata_records_the_contract(self, built):
        fs, cfg = built
        assert fs.metadata["feature_lag"] == cfg.features.feature_lag
        assert fs.metadata["horizon"] == cfg.model.horizon
        assert fs.metadata["n_rows"] == len(fs)

    def test_requires_returns(self, curve):
        with pytest.raises(ValueError):
            build_features(curve, pd.DataFrame(), Config(), returns=None)

    def test_slice_keeps_x_and_y_aligned(self, built):
        fs, _ = built
        mid = fs.index[len(fs) // 2]
        sub = fs.slice(start=mid)
        assert sub.X.index.equals(sub.y.index)
        assert sub.index.min() >= mid


# --------------------------------------------------------------------------- #
# Mean-reversion blocks
# --------------------------------------------------------------------------- #
class TestMeanReversion:
    """Added after measuring that the momentum features anti-predict beyond a day."""

    @pytest.fixture
    def nss(self, curve):
        from tqe.curve.nelson_siegel import fit_nss_history_fixed

        return fit_nss_history_fixed(curve)

    def test_rich_cheap_matches_the_scalar_api(self, curve, nss):
        """The vectorised residual must equal the scalar NSS evaluation.

        The block computes the NSS loadings directly because the library helper
        validates tau as a scalar while this path has a per-date vector. Same
        formula, so the two must agree exactly.
        """
        from tqe.curve.nelson_siegel import NSSParams
        from tqe.data.sources import TENOR_YEARS

        rc = curve_residual_features(curve, nss)
        d = nss.dropna(subset=["beta0"]).index[-1]
        r = nss.loc[d]
        p = NSSParams(r.beta0, r.beta1, r.beta2, r.beta3, r.tau1, r.tau2)
        for tenor in ("3 Mo", "2 Yr", "10 Yr", "30 Yr"):
            expected = (curve.loc[d, tenor] - float(p.zero_rate(TENOR_YEARS[tenor]))) * 1e4
            assert rc.loc[d, f"rc_{tenor}"] == pytest.approx(expected, abs=1e-9)

    def test_rich_cheap_residuals_are_small_and_centred(self, curve, nss):
        """A good curve fit leaves residuals of a few basis points, not percent."""
        rc = curve_residual_features(curve, nss)
        cols = [c for c in rc.columns if c.startswith("rc_") and c not in
                ("rc_dispersion", "rc_absmean", "rc_fit_rmse")]
        vals = rc[cols].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        assert abs(float(np.mean(vals))) < 15.0, "residuals should be roughly centred"
        assert float(np.percentile(np.abs(vals), 95)) < 100.0, "under 100bp at the 95th pct"

    def test_rich_cheap_is_causal(self, curve, nss):
        """The NSS fit is cross-sectional per date, so the past cannot move."""
        base = curve_residual_features(curve, nss)
        tampered = curve.copy()
        tampered.iloc[900:] *= 3.0
        after = curve_residual_features(tampered, nss)
        a = base.iloc[:900].dropna(how="all")
        b = after.reindex(a.index)[a.columns]
        assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                           atol=1e-12, equal_nan=True)

    def test_rich_cheap_handles_missing_nss(self, curve):
        out = curve_residual_features(curve, pd.DataFrame())
        assert out.empty and out.index.equals(curve.index)

    def test_reversal_is_the_negated_change(self, curve):
        rev = reversal_features(curve[["10 Yr"]], windows=(5,))
        assert np.allclose(rev["rev5_10 Yr"].dropna(),
                           -curve["10 Yr"].diff(5).dropna())

    def test_reversal_opposes_momentum(self, curve):
        """The whole point: reversal must be the mirror of momentum."""
        mom = momentum_features(curve[["10 Yr"]], windows=(21,))
        rev = reversal_features(curve[["10 Yr"]], windows=(21,))
        joined = pd.concat([mom["mom21_10 Yr"], rev["rev21_10 Yr"]], axis=1).dropna()
        assert joined.corr().iloc[0, 1] == pytest.approx(-1.0, abs=1e-9)

    def test_extension_is_volatility_scaled(self, curve):
        """`ext` must be unit-free, so its scale cannot depend on the regime."""
        rev = reversal_features(curve[["10 Yr"]], windows=(21,))
        ext = rev["ext21_10 Yr"].dropna()
        assert ext.abs().median() < 5.0

    def test_reversal_is_causal(self, curve):
        assert _past_is_unchanged(lambda d: reversal_features(d, (5, 21)), curve)

    def test_blocks_appear_when_enabled(self, curve):
        """They ship default-off because they dilute the daily signal, so the
        test has to turn them on explicitly - which is also the documented way
        to use them, alongside a longer horizon."""
        from tqe.curve.nelson_siegel import fit_nss_history_fixed
        from tqe.data.universe import constant_maturity_total_return

        cfg = Config()
        cfg.data.core_tenors = TENORS
        cfg.features.include_macro = False
        cfg.features.min_feature_coverage = 0.5
        cfg.features.include_reversal = True
        cfg.features.include_rich_cheap = True
        rets = constant_maturity_total_return(curve, TENORS)
        fs = build_features(curve, pd.DataFrame(), cfg, returns=rets,
                            nss=fit_nss_history_fixed(curve))
        assert any(c.startswith("rc_") for c in fs.X.columns)
        assert any(c.startswith("rev") for c in fs.X.columns)
        assert fs.metadata["blocks"]["reversal"]
        assert fs.metadata["blocks"]["rich_cheap"]

    def test_blocks_are_off_by_default(self, curve):
        """The shipped default at h=1, justified in the config comment."""
        cfg = Config()
        assert not cfg.features.include_reversal
        assert not cfg.features.include_rich_cheap
