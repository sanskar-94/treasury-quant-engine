"""Market-regime features.

Rates behave differently depending on the state of the world: a momentum signal
that works in a hiking cycle can be exactly wrong in a flight-to-quality. Giving
the model an explicit regime label lets it learn state-dependent behaviour
instead of averaging incompatible regimes together.

Every regime here is computed from a **trailing** window and is available at the
close of the day it labels, so it can be lagged once, centrally, like every
other feature block. In particular the Gaussian-mixture regime model is refitted
on a rolling basis with the fit window ending strictly before the day being
labelled - fitting one mixture on the whole sample and labelling history with it
would leak the future into every row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger("features.regime")

__all__ = [
    "volatility_regime",
    "trend_regime",
    "level_regime",
    "fit_regime_model",
    "rolling_regime_labels",
    "regime_features",
]


def volatility_regime(
    changes: pd.Series,
    short: int = 21,
    long: int = 252,
    n_states: int = 3,
) -> pd.DataFrame:
    """Classify realised volatility into calm / normal / stressed.

    Thresholds are the trailing *quantiles* of the long window rather than fixed
    numbers, so the classification adapts across decades: 8bp daily vol was calm
    in 1994 and alarming in 2015.
    """
    vol = changes.rolling(short, min_periods=max(2, short // 2)).std()
    lo = vol.rolling(long, min_periods=long // 2).quantile(1.0 / n_states)
    hi = vol.rolling(long, min_periods=long // 2).quantile(1.0 - 1.0 / n_states)

    state = pd.Series(np.nan, index=changes.index, dtype=float)
    state = state.mask(vol <= lo, 0.0)
    state = state.mask((vol > lo) & (vol < hi), 1.0)
    state = state.mask(vol >= hi, 2.0)

    rank = vol.rolling(long, min_periods=long // 2).rank(pct=True)
    return pd.DataFrame(
        {"vol_state": state, "vol_percentile": rank, "vol_level": vol},
        index=changes.index,
    )


def trend_regime(level: pd.Series, short: int = 21, long: int = 252) -> pd.DataFrame:
    """Bull (falling yields) / bear (rising yields) / range-bound.

    Encoded as ``-1 / +1 / 0`` on the *yield*, so ``+1`` means yields trending up
    - a bear market for bond prices.
    """
    ma_s = level.rolling(short, min_periods=max(2, short // 2)).mean()
    ma_l = level.rolling(long, min_periods=long // 2).mean()
    sd_l = level.rolling(long, min_periods=long // 2).std()

    gap = (ma_s - ma_l) / sd_l.where(sd_l.abs() > 1e-12)
    state = pd.Series(0.0, index=level.index)
    state = state.mask(gap > 0.5, 1.0)
    state = state.mask(gap < -0.5, -1.0)
    state = state.where(gap.notna(), np.nan)
    return pd.DataFrame({"trend_state": state, "trend_gap": gap}, index=level.index)


def level_regime(level: pd.Series, window: int = 1260) -> pd.DataFrame:
    """Where the yield level sits within its own multi-year history.

    Uses a five-year trailing window: long enough to span a policy cycle, short
    enough that the 1990s do not permanently define "high" for 2026.
    """
    rank = level.rolling(window, min_periods=window // 4).rank(pct=True)
    return pd.DataFrame({"level_percentile": rank}, index=level.index)


def fit_regime_model(features: pd.DataFrame, n_states: int = 3, random_state: int = 42):
    """Fit a Gaussian mixture over standardised feature rows.

    A full hidden Markov model would be the textbook choice, but a mixture on
    trailing statistics captures most of the benefit here without adding a
    dependency, and - crucially - it is cheap enough to refit on a rolling basis,
    which is what makes the labels causal.

    Returns ``(model, scaler)``; ``None`` if there is not enough clean data.
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    clean = features.dropna()
    if len(clean) < max(50, n_states * 20):
        return None
    scaler = StandardScaler().fit(clean.to_numpy(dtype=float))
    X = scaler.transform(clean.to_numpy(dtype=float))
    model = GaussianMixture(
        n_components=n_states,
        covariance_type="full",
        random_state=random_state,
        n_init=3,
        reg_covar=1e-5,
    ).fit(X)
    return model, scaler


def rolling_regime_labels(
    features: pd.DataFrame,
    n_states: int = 3,
    window: int = 1260,
    refit_every: int = 63,
    random_state: int = 42,
) -> pd.DataFrame:
    """Causal regime probabilities from a rolling Gaussian mixture.

    The model labelling day ``t`` is fitted on ``features.iloc[start:t]`` - the
    slice bound is exclusive, so day ``t`` itself is out of sample, as is
    everything after it.

    Mixture component indices are arbitrary and can permute between refits, which
    would make the raw label useless as a feature. Components are therefore
    **sorted by their mean first-feature value** at every refit, so state 0 is
    always the lowest-volatility cluster and the labels stay comparable through
    time.
    """
    clean = features.dropna()
    if clean.empty:
        return pd.DataFrame(index=features.index)

    cols = [f"regime_p{i}" for i in range(n_states)]
    out = np.full((len(clean), n_states + 1), np.nan)
    model = None
    scaler = None
    order = np.arange(n_states)
    last_fit = -10**9

    values = clean.to_numpy(dtype=float)
    for t in range(len(clean)):
        if t < max(window // 4, n_states * 20):
            continue
        if model is None or (t - last_fit) >= refit_every:
            start = max(0, t - window)
            fitted = fit_regime_model(clean.iloc[start:t], n_states, random_state)
            if fitted is None:
                continue
            model, scaler = fitted
            # Stabilise component identity across refits.
            order = np.argsort(model.means_[:, 0])
            last_fit = t
        if model is None or scaler is None:
            continue
        x = scaler.transform(values[t : t + 1])
        proba = model.predict_proba(x)[0][order]
        out[t, :n_states] = proba
        out[t, n_states] = float(np.argmax(proba))

    frame = pd.DataFrame(out, index=clean.index, columns=[*cols, "regime_state"])
    return frame.reindex(features.index)


def regime_features(
    curve: pd.DataFrame,
    returns: dict[str, pd.DataFrame] | None = None,
    n_states: int = 3,
    window: int = 252,
    anchor: str = "10 Yr",
    use_mixture: bool = True,
) -> pd.DataFrame:
    """Assemble the full regime block.

    Combines the cheap rule-based regimes (volatility, trend, level percentile,
    curve inversion) with the rolling mixture-model probabilities.
    """
    if anchor not in curve.columns:
        anchor = curve.columns[-1]
    level = curve[anchor]
    changes = level.diff()

    blocks = [
        volatility_regime(changes, short=21, long=max(window, 252), n_states=n_states),
        trend_regime(level, short=21, long=max(window, 252)),
        level_regime(level, window=1260),
    ]

    if "2 Yr" in curve.columns and "10 Yr" in curve.columns:
        slope = curve["10 Yr"] - curve["2 Yr"]
        blocks.append(
            pd.DataFrame(
                {
                    "slope_percentile": slope.rolling(1260, min_periods=252).rank(pct=True),
                    "slope_regime": np.sign(slope),
                },
                index=curve.index,
            )
        )

    out = pd.concat(blocks, axis=1)

    if use_mixture:
        basis = pd.DataFrame(
            {
                "vol": changes.rolling(21, min_periods=10).std(),
                "trend": level.diff(63),
                "slope": (curve["10 Yr"] - curve["2 Yr"])
                if "2 Yr" in curve.columns and "10 Yr" in curve.columns
                else level * 0.0,
            },
            index=curve.index,
        )
        try:
            out = pd.concat([out, rolling_regime_labels(basis, n_states=n_states)], axis=1)
        except Exception as exc:  # noqa: BLE001 - regime block is optional
            log.warning("rolling regime model unavailable (%s); continuing without it", exc)

    out = out.add_prefix("reg_")
    out.index.name = curve.index.name or "date"
    return out
