"""Price- and yield-based feature blocks.

Every function here returns a ``DataFrame`` aligned to the input index whose
values at row *t* are computable from data **up to and including** row *t*.
The single global lag that turns them into genuine predictors is applied once,
centrally, in :func:`tqe.features.builder.build_features` - doing it here as
well would double-lag and quietly destroy signal.

That split is deliberate: it means every rolling statistic in this module can be
written the obvious way (``.rolling(w).mean()``, which in pandas is
right-aligned and therefore already causal), and there is exactly one place in
the codebase where the prediction-time boundary is enforced.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger("features.technical")

__all__ = [
    "curve_residual_features",
    "reversal_features",
    "momentum_features",
    "volatility_features",
    "zscore_features",
    "mean_reversion_features",
    "curve_shape_features",
    "carry_rolldown_features",
    "cross_tenor_features",
]

EPS = 1e-12


def _safe_z(x: pd.DataFrame | pd.Series, mu, sd) -> pd.DataFrame | pd.Series:
    """Z-score guarding against a zero rolling standard deviation."""
    return (x - mu) / sd.where(sd.abs() > EPS)


def momentum_features(
    series: pd.DataFrame,
    windows: Sequence[int] = (1, 5, 10, 21, 63, 126, 252),
    prefix: str = "mom",
    diff: bool = True,
) -> pd.DataFrame:
    """Trailing changes over several horizons.

    For yields, ``diff=True`` gives the change in decimal rate (the natural
    unit - a 25bp move is 0.0025 whatever the level).  For prices or an equity
    curve, ``diff=False`` gives a percentage return instead.

    Momentum in rates is genuinely predictive at short horizons and mean-reverts
    at long ones, which is why several windows are kept rather than one.
    """
    out: dict[str, pd.Series] = {}
    for col in series.columns:
        s = series[col]
        for w in windows:
            key = f"{prefix}{w}_{col}"
            out[key] = s.diff(w) if diff else s.pct_change(w)
    return pd.DataFrame(out, index=series.index)


def volatility_features(
    changes: pd.DataFrame,
    windows: Sequence[int] = (10, 21, 63, 252),
    prefix: str = "vol",
    annualize: bool = True,
) -> pd.DataFrame:
    """Trailing realised volatility, plus the ratio of fast to slow vol.

    The ratio is the useful part: it says whether the market is *currently*
    agitated relative to its own recent norm, which conditions how much risk a
    signal deserves.  A raw vol level mostly encodes the decade.
    """
    scale = np.sqrt(252.0) if annualize else 1.0
    out: dict[str, pd.Series] = {}
    for col in changes.columns:
        s = changes[col]
        vols = {}
        for w in windows:
            v = s.rolling(w, min_periods=max(2, w // 2)).std() * scale
            vols[w] = v
            out[f"{prefix}{w}_{col}"] = v
        if len(windows) >= 2:
            fast, slow = min(windows), max(windows)
            out[f"{prefix}ratio_{col}"] = vols[fast] / vols[slow].where(vols[slow].abs() > EPS)
    return pd.DataFrame(out, index=changes.index)


def zscore_features(
    series: pd.DataFrame,
    windows: Sequence[int] = (21, 63, 252),
    prefix: str = "z",
) -> pd.DataFrame:
    """Level standardised by its own trailing window.

    A trailing z-score is the honest way to say "this yield is high": it uses
    only what was knowable at the time.  Standardising by full-sample statistics
    is a classic leak, because the mean and standard deviation embed the future.
    """
    out: dict[str, pd.Series] = {}
    for col in series.columns:
        s = series[col]
        for w in windows:
            mp = max(2, w // 2)
            mu = s.rolling(w, min_periods=mp).mean()
            sd = s.rolling(w, min_periods=mp).std()
            out[f"{prefix}{w}_{col}"] = _safe_z(s, mu, sd)
    return pd.DataFrame(out, index=series.index)


def mean_reversion_features(
    series: pd.DataFrame,
    windows: Sequence[int] = (21, 63, 252),
    prefix: str = "mr",
) -> pd.DataFrame:
    """Distance from a trailing mean, and position within a trailing range.

    ``mr{w}`` is the gap to the moving average; ``pct{w}`` is where the current
    level sits inside its trailing min-max band, in ``[0, 1]``.  Both capture
    stretch, which in rates tends to revert over weeks.
    """
    out: dict[str, pd.Series] = {}
    for col in series.columns:
        s = series[col]
        for w in windows:
            mp = max(2, w // 2)
            out[f"{prefix}{w}_{col}"] = s - s.rolling(w, min_periods=mp).mean()
            lo = s.rolling(w, min_periods=mp).min()
            hi = s.rolling(w, min_periods=mp).max()
            rng = (hi - lo).where((hi - lo).abs() > EPS)
            out[f"pct{w}_{col}"] = (s - lo) / rng
    return pd.DataFrame(out, index=series.index)


def curve_shape_features(curve: pd.DataFrame) -> pd.DataFrame:
    """Slopes, butterflies and level - the vocabulary rates traders actually use.

    Only pairs whose two legs are both present are emitted, so the ragged
    coverage of the Treasury file (no 20y before 1993, no 30y 2002-2006) produces
    NaNs rather than silently wrong spreads.

    A butterfly ``2 * belly - short - long`` is the classic curvature trade: it
    is positive when the belly is cheap relative to the wings.
    """
    out: dict[str, pd.Series] = {}
    have = set(curve.columns)

    slopes = [
        ("2s10s", "2 Yr", "10 Yr"),
        ("3m10y", "3 Mo", "10 Yr"),
        ("2s5s", "2 Yr", "5 Yr"),
        ("5s10s", "5 Yr", "10 Yr"),
        ("5s30s", "5 Yr", "30 Yr"),
        ("10s30s", "10 Yr", "30 Yr"),
        ("3m2y", "3 Mo", "2 Yr"),
        ("1s3s", "1 Yr", "3 Yr"),
    ]
    for name, short, long in slopes:
        if short in have and long in have:
            out[f"slope_{name}"] = curve[long] - curve[short]

    flies = [
        ("2_5_10", "2 Yr", "5 Yr", "10 Yr"),
        ("5_10_30", "5 Yr", "10 Yr", "30 Yr"),
        ("3m_2y_10y", "3 Mo", "2 Yr", "10 Yr"),
        ("1_3_7", "1 Yr", "3 Yr", "7 Yr"),
    ]
    for name, s_, b_, l_ in flies:
        if all(c in have for c in (s_, b_, l_)):
            out[f"fly_{name}"] = 2.0 * curve[b_] - curve[s_] - curve[l_]

    # Level: the average of whatever is quoted, and the classic 10y anchor.
    out["level_mean"] = curve.mean(axis=1)
    if "10 Yr" in have:
        out["level_10y"] = curve["10 Yr"]
    if "3 Mo" in have:
        out["level_3m"] = curve["3 Mo"]

    # Inversion is a regime marker with real predictive content for the economy.
    if "2 Yr" in have and "10 Yr" in have:
        inv = (curve["10 Yr"] - curve["2 Yr"]) < 0
        out["inverted_2s10s"] = inv.astype(float)
        # How long the inversion has persisted, in days - a deeper signal than
        # the binary flag. cumcount within each contiguous inverted block.
        grp = (inv != inv.shift()).cumsum()
        out["inversion_days"] = inv.groupby(grp).cumsum().where(inv, 0.0).astype(float)

    # Steepness normalised by level: a 50bp slope means something different at
    # 1% than at 8%.
    if "3 Mo" in have and "10 Yr" in have:
        denom = curve["10 Yr"].abs().clip(lower=1e-4)
        out["slope_ratio"] = (curve["10 Yr"] - curve["3 Mo"]) / denom

    return pd.DataFrame(out, index=curve.index)


def carry_rolldown_features(
    curve: pd.DataFrame,
    returns: dict[str, pd.DataFrame] | None = None,
    tenor_years: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Carry and roll-down per tenor - the return earned if nothing moves.

    Roll-down is approximated on the quoted grid: a bond currently at tenor
    ``T`` will, in three months, be priced off the curve a little further down.
    The pickup is ``(y_T - y_{T-}) * duration``, where ``y_{T-}`` is the yield at
    the next shorter quoted tenor, scaled for the distance rolled.

    This is a genuine expected-return component, not a technical indicator; a
    directional forecast has to beat carry + roll to be worth trading.
    """
    if tenor_years is None:
        from ..data.sources import TENOR_YEARS

        tenor_years = TENOR_YEARS

    cols = [c for c in curve.columns if c in tenor_years]
    cols = sorted(cols, key=lambda c: tenor_years[c])
    out: dict[str, pd.Series] = {}

    for i, col in enumerate(cols):
        y = curve[col]
        out[f"carry_{col}"] = y  # the yield itself is the running carry
        if i == 0:
            continue
        prev = cols[i - 1]
        dt = tenor_years[col] - tenor_years[prev]
        if dt <= 0:
            continue
        # Yield pickup per year of roll.
        slope_per_year = (y - curve[prev]) / dt
        dur = None
        if returns is not None and col in returns:
            dur = returns[col].get("duration")
        if dur is None:
            dur = pd.Series(tenor_years[col] * 0.8, index=curve.index)  # crude fallback
        # Rolling down a quarter of a year.
        out[f"roll_{col}"] = slope_per_year * 0.25 * dur.reindex(curve.index)
        out[f"carryroll_{col}"] = y * 0.25 + out[f"roll_{col}"]

    return pd.DataFrame(out, index=curve.index)


def cross_tenor_features(changes: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Dispersion and co-movement across the curve.

    ``dispersion`` is the cross-sectional standard deviation of today's yield
    changes: high when the curve is twisting, low on a clean parallel shift.
    ``avg_corr`` is the mean pairwise trailing correlation, a summary of how
    one-dimensional the market currently is.
    """
    out: dict[str, pd.Series] = {}
    out["dispersion"] = changes.std(axis=1)
    out["cross_mean"] = changes.mean(axis=1)
    out["dispersion_z"] = _safe_z(
        out["dispersion"],
        out["dispersion"].rolling(window, min_periods=window // 2).mean(),
        out["dispersion"].rolling(window, min_periods=window // 2).std(),
    )

    # Mean pairwise correlation over the trailing window.  Computed from the
    # rolling correlation of each column with the cross-sectional mean, which is
    # a cheap and stable proxy for the full pairwise average.
    ref = changes.mean(axis=1)
    corrs = [
        changes[c].rolling(window, min_periods=window // 2).corr(ref) for c in changes.columns
    ]
    out["avg_corr"] = pd.concat(corrs, axis=1).mean(axis=1)
    return pd.DataFrame(out, index=changes.index)


def curve_residual_features(
    curve: pd.DataFrame,
    nss: pd.DataFrame,
    tenor_years: dict[str, float] | None = None,
    windows: Sequence[int] = (21, 63, 252),
) -> pd.DataFrame:
    """Rich/cheap: how far each tenor trades from the fitted curve.

    This is the canonical relative-value signal in rates and it is the direct
    answer to a specific defect measured in this project. The information
    coefficient of the momentum-driven feature set is +0.025 at one day,
    -0.072 at five and -0.174 at twenty-one: it extrapolates a move that has
    already turned. Rich/cheap points the other way by construction - a tenor
    trading cheap to the curve everyone else is on tends to richen back to it.

    The residual is ``market yield - NSS fitted yield`` in basis points. The NSS
    fit is a smooth four-factor description of the whole curve, so a large
    residual means that one sector has dislocated from the shape the rest of the
    curve implies, which is a mean-reverting condition rather than a trend.

    Both the level and its trailing z-score are emitted; the z-score is what
    makes "cheap" comparable across a decade in which the average dislocation
    changed size.

    Causality: the NSS parameters for date ``t`` are fitted on date ``t``'s own
    quotes only - a cross-sectional fit, using no history - so the residual is
    known at that day's close like any other feature block.
    """
    if tenor_years is None:
        from ..data.sources import TENOR_YEARS

        tenor_years = TENOR_YEARS
    if nss is None or nss.empty:
        return pd.DataFrame(index=curve.index)

    need = ["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"]
    if any(c not in nss.columns for c in need):
        return pd.DataFrame(index=curve.index)

    params = nss[need].reindex(curve.index)
    cols = [c for c in curve.columns if c in tenor_years]
    b0, b1, b2, b3, t1, t2 = (params[c].to_numpy(dtype=float) for c in need)

    def _loading(x: np.ndarray) -> np.ndarray:
        """(1 - exp(-x)) / x, with the x -> 0 limit of 1 handled explicitly."""
        return np.where(np.abs(x) < 1e-8, 1.0 - x / 2.0, (1.0 - np.exp(-x)) / np.where(x == 0, 1.0, x))

    out: dict[str, pd.Series] = {}
    resid_frame = {}
    for col in cols:
        # The scalar helper in curve.nelson_siegel validates tau as a scalar;
        # here every parameter is a per-date vector, so the loadings are built
        # directly. Same formula, evaluated elementwise across dates.
        t = float(tenor_years[col])
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            x1 = np.where(t1 > 0, t / np.where(t1 > 0, t1, 1.0), np.nan)
            x2 = np.where(t2 > 0, t / np.where(t2 > 0, t2, 1.0), np.nan)
            f1, f2 = _loading(x1), _loading(x2)
            fitted = b0 + b1 * f1 + b2 * (f1 - np.exp(-x1)) + b3 * (f2 - np.exp(-x2))
        resid = (curve[col].to_numpy(dtype=float) - np.asarray(fitted, dtype=float)) * 1e4  # bp
        s_res = pd.Series(resid, index=curve.index)
        resid_frame[col] = s_res
        out[f"rc_{col}"] = s_res
        for w in windows:
            mp = max(2, w // 2)
            mu = s_res.rolling(w, min_periods=mp).mean()
            sd = s_res.rolling(w, min_periods=mp).std()
            out[f"rcz{w}_{col}"] = _safe_z(s_res, mu, sd)

    if resid_frame:
        rf = pd.DataFrame(resid_frame)
        # Cross-sectional dispersion of dislocation: high when the curve is
        # locally kinked, which is when RV opportunities exist at all.
        out["rc_dispersion"] = rf.std(axis=1)
        out["rc_absmean"] = rf.abs().mean(axis=1)
    if "rmse" in nss.columns:
        out["rc_fit_rmse"] = nss["rmse"].reindex(curve.index) * 1e4

    return pd.DataFrame(out, index=curve.index)


def reversal_features(
    series: pd.DataFrame,
    windows: Sequence[int] = (5, 21, 63),
    prefix: str = "rev",
) -> pd.DataFrame:
    """Explicit short-horizon reversal signals.

    The mirror image of :func:`momentum_features`, and included because the
    measured IC degradation says this feature set needs it. Three constructions:

    * ``rev{w}``  - the negated trailing change, so a large recent rally is a
      negative (short) signal rather than a positive one,
    * ``acc{w}``  - acceleration, the change in the trailing change: a move that
      is decelerating is closer to exhausting than one that is not,
    * ``ext{w}``  - extension, the trailing change divided by its own trailing
      volatility, which is how far the move has run in units of normal noise.

    ``ext`` is the useful one. A 20bp move is ordinary in 2022 and extreme in
    2019, and only the volatility-scaled version says which regime you are in.
    """
    out: dict[str, pd.Series] = {}
    for col in series.columns:
        s = series[col]
        for w in windows:
            chg = s.diff(w)
            out[f"{prefix}{w}_{col}"] = -chg
            out[f"acc{w}_{col}"] = chg - chg.shift(w)
            vol = s.diff().rolling(w, min_periods=max(2, w // 2)).std() * np.sqrt(w)
            out[f"ext{w}_{col}"] = -chg / vol.where(vol.abs() > EPS)
    return pd.DataFrame(out, index=series.index)
