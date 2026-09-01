"""Assembly of the model design matrix.

This module owns the single most important line in the project::

    X = X.shift(cfg.features.feature_lag)

Every feature block is written to be causal-as-of-its-own-close. That still is
not enough to predict: to forecast day ``t``'s return you may only use
information from day ``t-1``'s close. That one shift is where the boundary is
drawn, and it is drawn exactly once so it can be audited in one place.

The alignment contract produced by :func:`build_features` is::

    row t of X   = features observable at the close of day t-1
    row t of y   = the return realised over day t  (or t..t+horizon-1)

so a model fitted on ``(X[t], y[t])`` is genuinely predicting forward.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from .macro import macro_features
from .regime import regime_features
from .technical import (
    carry_rolldown_features,
    cross_tenor_features,
    curve_residual_features,
    curve_shape_features,
    mean_reversion_features,
    momentum_features,
    reversal_features,
    volatility_features,
    zscore_features,
)

log = get_logger("features.builder")

__all__ = ["FeatureSet", "build_features", "make_targets", "feature_report"]


@dataclass
class FeatureSet:
    """A fully aligned, leakage-checked design matrix and its targets."""

    X: pd.DataFrame
    y: pd.DataFrame
    feature_names: list[str] = field(default_factory=list)
    target_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature_names:
            self.feature_names = list(self.X.columns)
        if not self.target_names:
            self.target_names = list(self.y.columns)

    def __len__(self) -> int:
        return len(self.X)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.X.index

    def slice(self, start=None, end=None) -> FeatureSet:
        """Restrict to a date window, keeping X and y aligned."""
        Xs = self.X.loc[start:end]
        return FeatureSet(Xs, self.y.loc[Xs.index], self.feature_names, self.target_names, dict(self.metadata))

    def summary(self) -> str:
        return (
            f"FeatureSet({len(self.X)} rows x {self.X.shape[1]} features, "
            f"{self.y.shape[1]} targets, {self.index.min().date()}..{self.index.max().date()})"
        )


def make_targets(
    returns: dict[str, pd.DataFrame],
    target: str = "price_return",
    horizon: int = 1,
    tenors: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build forward-looking targets aligned so row *t* holds the FUTURE return.

    ``shift(-horizon)`` moves tomorrow's realised return onto today's row. For
    ``horizon > 1`` the target is the compounded return over the whole window,
    not just the single day at the end, because that is what a position actually
    earns if held.

    ``target="direction"`` emits ``{0, 1}`` for a classification setup.

    **Relative targets.** ``target="relative_return"`` subtracts the
    cross-sectional mean, so the model forecasts which tenor *outperforms the
    curve* rather than where rates go. This matters because a directional
    forecast has to beat the funding rate to be worth anything, whereas a
    relative-value book is cash- and duration-neutral by construction and pays
    almost no financing. On this dataset the directional target produced a
    strategy whose entire positive return came from a funding position; the
    relative target removes that channel entirely, so whatever it earns is
    forecast quality and nothing else.

    ``target="excess_return"`` subtracts the shortest tenor's return instead,
    which is the closest thing here to a return over cash.
    """
    tenors = list(tenors) if tenors else list(returns)
    cols: dict[str, pd.Series] = {}

    # Relative targets are built from an underlying return series, then
    # demeaned across the cross-section AFTER the forward shift - demeaning
    # before would mix information across dates.
    relative = target in ("relative_return", "excess_return")
    base_field = "total_return" if relative else target

    for tenor in tenors:
        frame = returns.get(tenor)
        if frame is None:
            continue
        if target == "direction":
            base = frame["total_return"]
            fwd = (1.0 + base).rolling(horizon).apply(np.prod, raw=True) - 1.0 if horizon > 1 else base
            cols[tenor] = (fwd.shift(-horizon) > 0).astype(float).where(fwd.shift(-horizon).notna())
            continue

        if base_field not in frame.columns:
            raise KeyError(f"Target {target!r} not in the analytics frame for {tenor}")
        base = frame[base_field]

        if horizon == 1:
            cols[tenor] = base.shift(-1)
        elif base_field in ("price_return", "total_return"):
            compounded = (1.0 + base).rolling(horizon).apply(np.prod, raw=True) - 1.0
            cols[tenor] = compounded.shift(-horizon)
        else:  # yield_change and other additive quantities
            cols[tenor] = base.rolling(horizon).sum().shift(-horizon)

    out = pd.DataFrame(cols)

    if target == "relative_return":
        # Row-wise, so it uses only that date's own cross-section.
        out = out.sub(out.mean(axis=1), axis=0)
    elif target == "excess_return":
        from ..data.sources import TENOR_YEARS

        known = [c for c in out.columns if c in TENOR_YEARS]
        if known:
            short = min(known, key=lambda c: TENOR_YEARS[c])
            out = out.sub(out[short], axis=0)
    out.index.name = "date"
    return out


def build_features(
    curve: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    cfg: Config | None = None,
    returns: dict[str, pd.DataFrame] | None = None,
    nss: pd.DataFrame | None = None,
    pca_factors: pd.DataFrame | None = None,
    zero: pd.DataFrame | None = None,
    tenors: Sequence[str] | None = None,
    dropna: bool = True,
) -> FeatureSet:
    """Assemble every feature block, enforce the lag, and align to targets.

    Parameters
    ----------
    curve:
        Par-yield history.
    macro:
        FRED bundle, or ``None``/empty to skip the macro block.
    cfg:
        Configuration; a default :class:`~tqe.config.Config` is used if omitted.
    returns:
        Output of :func:`tqe.data.universe.constant_maturity_total_return`.
        Required - the targets come from it.
    nss:
        Fitted Nelson-Siegel-Svensson betas per date. Use the **fixed-tau**
        variant here; free-tau betas are not identifiable and behave as noise.
    pca_factors:
        Causal rolling PCA scores.
    zero:
        Bootstrapped zero curve, used for forward-rate features.
    tenors:
        Which tenors to build targets for. Defaults to ``cfg.data.core_tenors``.
    dropna:
        Drop rows with any missing feature or target. Leave on for training.

    Returns
    -------
    FeatureSet
    """
    cfg = cfg or Config()
    fc = cfg.features
    tenors = list(tenors) if tenors else [t for t in cfg.data.core_tenors if t in curve.columns]
    if returns is None:
        raise ValueError("build_features needs `returns` to construct targets")

    curve = curve.sort_index()
    sub = curve[[c for c in tenors if c in curve.columns]]
    changes = sub.diff()

    blocks: list[pd.DataFrame] = []

    # ---- yield level / momentum / vol / stretch ---------------------------- #
    blocks.append(momentum_features(sub, fc.momentum_windows, prefix="ymom", diff=True))
    blocks.append(volatility_features(changes, fc.vol_windows, prefix="yvol"))
    blocks.append(zscore_features(sub, fc.zscore_windows, prefix="yz"))
    blocks.append(mean_reversion_features(sub, fc.zscore_windows, prefix="ymr"))
    blocks.append(cross_tenor_features(changes))

    # ---- mean reversion --------------------------------------------------- #
    # Added in response to a measured defect: the momentum blocks give an IC of
    # +0.025 at one day, -0.072 at five and -0.174 at twenty-one, i.e. they
    # extrapolate a move that has already turned. These point the other way.
    if fc.include_reversal:
        blocks.append(reversal_features(sub, fc.reversal_windows))

    # ---- realised total-return momentum ------------------------------------ #
    tr = pd.DataFrame(
        {t: returns[t]["total_return"] for t in tenors if t in returns}, index=curve.index
    )
    if not tr.empty:
        blocks.append(momentum_features(tr.fillna(0.0).cumsum(), fc.momentum_windows, prefix="rmom", diff=True))
        blocks.append(volatility_features(tr, fc.vol_windows, prefix="rvol"))

    # ---- curve shape -------------------------------------------------------- #
    if fc.include_curve_shape:
        shape = curve_shape_features(curve)
        blocks.append(shape)
        # Momentum and stretch of the shape variables matter more than the levels.
        slope_cols = [c for c in shape.columns if c.startswith(("slope_", "fly_"))]
        if slope_cols:
            blocks.append(momentum_features(shape[slope_cols], (5, 21, 63), prefix="smom", diff=True))
            blocks.append(zscore_features(shape[slope_cols], (63, 252), prefix="sz"))

    # ---- carry and roll-down ------------------------------------------------ #
    if fc.include_carry_rolldown:
        blocks.append(carry_rolldown_features(curve, returns))

    # ---- rich/cheap versus the fitted curve --------------------------------- #
    if fc.include_rich_cheap and nss is not None and not nss.empty:
        blocks.append(curve_residual_features(curve, nss, windows=fc.zscore_windows))

    # ---- parametric curve factors ------------------------------------------ #
    if nss is not None and not nss.empty:
        keep = [c for c in ("beta0", "beta1", "beta2", "beta3", "rmse") if c in nss.columns]
        nb = nss[keep].add_prefix("nss_")
        blocks.append(nb)
        blocks.append(momentum_features(nb, (5, 21, 63), prefix="nssmom", diff=True))

    # ---- PCA factors -------------------------------------------------------- #
    if fc.include_pca and pca_factors is not None and not pca_factors.empty:
        pf = pca_factors.add_prefix("pca_")
        blocks.append(pf)
        # Cumulative factor scores are the economically meaningful state; the raw
        # daily score is a change.
        blocks.append(momentum_features(pf.fillna(0.0).cumsum(), (5, 21, 63), prefix="pcamom", diff=True))
        blocks.append(zscore_features(pf, (63, 252), prefix="pcaz"))

    # ---- forward rates from the zero curve ---------------------------------- #
    if zero is not None and not zero.empty:
        fwd = {}
        pairs = [("2 Yr", "5 Yr"), ("5 Yr", "10 Yr"), ("1 Yr", "3 Yr"), ("10 Yr", "30 Yr")]
        for a, b in pairs:
            if a in zero.columns and b in zero.columns:
                fwd[f"fwdspread_{a}_{b}".replace(" ", "")] = zero[b] - zero[a]
        if fwd:
            blocks.append(pd.DataFrame(fwd, index=zero.index))

    # ---- regimes ------------------------------------------------------------ #
    if fc.include_regime:
        blocks.append(regime_features(curve, returns, n_states=fc.regime_states))

    # ---- macro -------------------------------------------------------------- #
    if fc.include_macro and macro is not None and not macro.empty:
        blocks.append(macro_features(macro, curve.index))

    # ---- combine ------------------------------------------------------------ #
    X = pd.concat([b.reindex(curve.index) for b in blocks if b is not None and not b.empty], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.replace([np.inf, -np.inf], np.nan)

    # ====================================================================== #
    # THE CAUSALITY BOUNDARY.  Row t now holds information from t-1's close.
    # ====================================================================== #
    X = X.shift(fc.feature_lag)

    y = make_targets(returns, cfg.model.target, cfg.model.horizon, tenors)
    y = y.reindex(X.index)

    # Drop features that are constant or almost entirely missing - they cost
    # compute and can destabilise a linear model without adding information.
    min_cov = float(getattr(fc, "min_feature_coverage", 0.30))
    coverage = X.notna().mean()
    sparse = coverage[coverage < min_cov].index.tolist()
    if sparse:
        log.info(
            "dropping %d features below %.0f%% coverage (e.g. %s)",
            len(sparse), min_cov * 100, ", ".join(sparse[:4]),
        )
        X = X.drop(columns=sparse)
    nunique = X.nunique(dropna=True)
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        log.info("dropping %d constant features", len(constant))
        X = X.drop(columns=constant)

    if dropna:
        valid = X.notna().all(axis=1) & y.notna().all(axis=1)
        X, y = X.loc[valid], y.loc[valid]

    meta = {
        "target": cfg.model.target,
        "horizon": cfg.model.horizon,
        "feature_lag": fc.feature_lag,
        "tenors": tenors,
        "min_feature_coverage": min_cov,
        "dropped_sparse": len(sparse),
        "n_features": X.shape[1],
        "n_rows": len(X),
        "start": str(X.index.min().date()) if len(X) else None,
        "end": str(X.index.max().date()) if len(X) else None,
        "blocks": {
            "macro": bool(fc.include_macro and macro is not None and not macro.empty),
            "pca": bool(fc.include_pca and pca_factors is not None),
            "nss": bool(nss is not None and not nss.empty),
            "regime": bool(fc.include_regime),
            "reversal": bool(fc.include_reversal),
            "rich_cheap": bool(fc.include_rich_cheap and nss is not None and not nss.empty),
        },
    }
    fs = FeatureSet(X=X, y=y, metadata=meta)
    log.info("%s", fs.summary())
    return fs


def feature_report(fs: FeatureSet, top: int = 25) -> pd.DataFrame:
    """Per-feature diagnostics: coverage, dispersion and correlation to target.

    The correlation column is the quickest way to spot a leak: a single feature
    correlating above ~0.2 with a next-day bond return is not a discovery, it is
    a bug.
    """
    y = fs.y.mean(axis=1)
    rows = []
    for col in fs.X.columns:
        s = fs.X[col]
        rows.append(
            {
                "feature": col,
                "coverage": float(s.notna().mean()),
                "std": float(s.std()),
                "corr_target": float(s.corr(y)) if s.std() > 0 else np.nan,
            }
        )
    rep = pd.DataFrame(rows).set_index("feature")
    rep["abs_corr"] = rep["corr_target"].abs()
    return rep.sort_values("abs_corr", ascending=False).head(top)
