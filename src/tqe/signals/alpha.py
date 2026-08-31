"""Turning raw forecasts into tradable signals.

A model's output is a predicted return in decimal units - roughly 1e-4 in
magnitude, with a scale that drifts as the model is refitted and as market
volatility changes. That is not something you can size a position from directly.

The signal layer solves three problems:

1. **Scale.** Standardise the forecast against its own recent history so that
   "+1" means the same thing in 1994 and in 2021.
2. **Outliers.** Clip. An unclipped forecast on a day like 2020-03-09 will
   demand a position far outside any sane risk budget.
3. **Noise.** Apply a deadband so a forecast indistinguishable from zero does
   not generate a trade, and optionally smooth so the book does not churn.

Every statistic used is **trailing**. Standardising a signal by its full-sample
mean and standard deviation is a leak, and a particularly seductive one because
it looks like innocuous preprocessing rather than like cheating.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger("signals.alpha")

__all__ = [
    "scale_to_return_units",
    "predictions_to_signal",
    "blend_signals",
    "signal_decay",
    "apply_deadband",
    "cross_sectional_rank",
    "signal_diagnostics",
]

EPS = 1e-12


def predictions_to_signal(
    preds: pd.DataFrame,
    method: str = "zscore",
    window: int = 252,
    clip: float = 3.0,
    min_abs: float = 0.0,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Standardise raw forecasts into bounded, comparable signals.

    Parameters
    ----------
    preds:
        ``date x tenor`` frame of predicted returns.
    method:
        ``"zscore"``
            ``(x - trailing_mean) / trailing_std``. The default. Removes any
            slow drift in the model's bias as well as rescaling.
        ``"vol_scale"``
            ``x / trailing_std``, keeping the sign and level of the raw
            forecast. Preferred when you believe the model's zero point is
            meaningful and should not be recentred.
        ``"rank"``
            Cross-sectional rank across tenors, mapped to ``[-1, 1]``. Purely
            relative - use for curve trades where the level view is hedged out.
        ``"raw"``
            Pass through unchanged (then clipped).
    window:
        Trailing window for the standardising statistics.
    clip:
        Absolute bound applied after standardising.
    min_abs:
        Deadband - signals smaller than this in absolute value become zero.
    min_periods:
        Minimum observations before a signal is emitted. Defaults to
        ``window // 4``, so the series starts sooner at the cost of a noisier
        early scale estimate.

    Returns
    -------
    pd.DataFrame
        Same shape as ``preds``; NaN during the warm-up window.
    """
    if preds.empty:
        return preds.copy()
    mp = int(min_periods or max(20, window // 4))

    if method == "rank":
        sig = cross_sectional_rank(preds)
    elif method == "raw":
        sig = preds.copy()
    else:
        sd = preds.rolling(window, min_periods=mp).std()
        sd = sd.where(sd.abs() > EPS)
        if method == "zscore":
            mu = preds.rolling(window, min_periods=mp).mean()
            sig = (preds - mu) / sd
        elif method == "vol_scale":
            sig = preds / sd
        else:
            raise ValueError(f"Unknown signal method {method!r}")

    if clip and clip > 0:
        sig = sig.clip(-abs(clip), abs(clip))
    if min_abs and min_abs > 0:
        sig = apply_deadband(sig, min_abs)
    return sig


def cross_sectional_rank(preds: pd.DataFrame) -> pd.DataFrame:
    """Rank forecasts across tenors each day, scaled to ``[-1, 1]``.

    Row-wise and therefore inherently causal - it uses only that day's own
    cross-section, no history at all.
    """
    ranked = preds.rank(axis=1, pct=True)
    return (ranked - 0.5) * 2.0


def apply_deadband(signal: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Zero out signals too small to be worth their transaction cost.

    A deadband is not cosmetic: with a daily rebalance, forecasts that hover
    around zero generate constant small trades whose costs swamp their edge.
    """
    return signal.where(signal.abs() >= threshold, 0.0)


def signal_decay(signal: pd.DataFrame, halflife: float = 3.0) -> pd.DataFrame:
    """Exponentially smooth the signal to damp turnover.

    ``ewm`` is causal by construction. The trade-off is explicit: a longer
    half-life cuts costs but delays the response to genuine new information.
    """
    if halflife <= 0:
        return signal
    return signal.ewm(halflife=halflife, min_periods=1).mean()


def blend_signals(
    signals: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float] | Sequence[float] | None = None,
) -> pd.DataFrame:
    """Combine several signal families into one.

    Missing values are treated as "no view" (zero) rather than propagating NaN,
    so a signal that has not warmed up yet simply abstains instead of blanking
    the whole book. Weights are renormalised over the signals that are actually
    present on each date.
    """
    if not signals:
        raise ValueError("No signals to blend")
    names = list(signals)
    if weights is None:
        w = {n: 1.0 / len(names) for n in names}
    elif isinstance(weights, Mapping):
        total = sum(weights.get(n, 0.0) for n in names) or 1.0
        w = {n: weights.get(n, 0.0) / total for n in names}
    else:
        arr = np.asarray(list(weights), dtype=float)
        if len(arr) != len(names):
            raise ValueError("weights length must match the number of signals")
        total = arr.sum() or 1.0
        w = dict(zip(names, arr / total))

    index = signals[names[0]].index
    columns = signals[names[0]].columns
    for n in names[1:]:
        index = index.union(signals[n].index)
        columns = columns.union(signals[n].columns)

    numer = pd.DataFrame(0.0, index=index, columns=columns)
    denom = pd.DataFrame(0.0, index=index, columns=columns)
    for n in names:
        s = signals[n].reindex(index=index, columns=columns)
        present = s.notna()
        numer = numer.add(s.fillna(0.0) * w[n], fill_value=0.0)
        denom = denom.add(present.astype(float) * w[n], fill_value=0.0)

    out = numer / denom.where(denom > EPS)
    return out


def signal_diagnostics(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Per-tenor signal quality: IC, hit rate, autocorrelation and turnover.

    ``autocorr`` and ``turnover`` matter as much as ``ic`` here. A signal with a
    respectable IC but daily autocorrelation near zero flips sign constantly and
    will not survive transaction costs, which is invisible if you only look at
    predictive accuracy.
    """
    from scipy import stats

    rows = []
    common = signal.index.intersection(forward_returns.index)
    for col in signal.columns:
        if col not in forward_returns.columns:
            continue
        s = signal.loc[common, col]
        r = forward_returns.loc[common, col]
        both = pd.concat([s, r], axis=1).dropna()
        if len(both) < 30:
            continue
        sv, rv = both.iloc[:, 0], both.iloc[:, 1]
        ic = float(sv.corr(rv)) if sv.std() > 0 else np.nan
        rank_ic = float(stats.spearmanr(sv, rv).statistic) if sv.std() > 0 else np.nan
        active = sv.abs() > EPS
        rows.append(
            {
                "tenor": col,
                "n": len(both),
                "ic": ic,
                "rank_ic": rank_ic,
                "hit_rate": float((np.sign(sv) == np.sign(rv))[active].mean()) if active.any() else np.nan,
                "autocorr_1d": float(sv.autocorr(1)) if sv.std() > 0 else np.nan,
                "turnover": float(sv.diff().abs().mean()),
                "pct_active": float(active.mean()),
                "mean_abs": float(sv.abs().mean()),
            }
        )
    return pd.DataFrame(rows).set_index("tenor") if rows else pd.DataFrame()


def scale_to_return_units(
    preds: pd.DataFrame,
    realised_returns: pd.DataFrame,
    ic: float = 0.05,
    window: int = 252,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Rescale raw forecasts onto the scale of actual returns.

    A z-scored signal is fine for sizing, because sizing only cares about the
    *shape* of the view. A mean-variance optimiser is different: it compares
    ``mu`` against ``Sigma`` and against transaction costs, so ``mu`` has to be
    an honest expected return in the same units as the covariance. Feed it a
    z-score and the risk aversion is meaningless; feed it a raw ensemble output
    and you inherit whatever arbitrary scale the learner happened to produce.

    That scale really is arbitrary. A stacked ensemble fitted by non-negative
    least squares on a very noisy target shrinks hard - on this dataset the
    fitted combination weights sum to roughly 0.003, so the ensemble's raw
    output is about 0.3% of the magnitude of a real daily return. Optimising
    against it produces an empty book, correctly but uselessly.

    The rescaling uses the standard result that an optimal forecast of a return
    has standard deviation ``IC * sigma_r``, where ``IC`` is the correlation
    between forecast and outcome. So the forecast is renormalised to unit
    trailing variance and multiplied by ``ic * trailing_return_vol``.

    Parameters
    ----------
    preds:
        Raw model output, ``date x tenor``.
    realised_returns:
        Historical returns for the same tenors, used only through a trailing
        window so the transformation stays causal.
    ic:
        Assumed information coefficient. 0.05 is a realistic daily figure for
        this problem; setting it higher scales positions up proportionally, so
        it functions as an explicit conviction dial rather than a hidden one.
    window:
        Trailing window for both volatility estimates.
    min_periods:
        Minimum observations before output is produced.

    Returns
    -------
    pd.DataFrame
        Forecasts in return units, directly usable as ``mu``.
    """
    if preds.empty:
        return preds.copy()
    mp = int(min_periods or max(20, window // 4))

    pred_sd = preds.rolling(window, min_periods=mp).std()
    ret_sd = realised_returns.reindex_like(preds).rolling(window, min_periods=mp).std()

    normalised = preds / pred_sd.where(pred_sd.abs() > EPS)
    scaled = normalised * (abs(ic) * ret_sd)
    return scaled.replace([np.inf, -np.inf], np.nan)
