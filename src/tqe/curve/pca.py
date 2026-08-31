"""Principal-component decomposition of the yield curve.

Three factors explain essentially all of the variation in the Treasury curve,
and they have stable economic interpretations:

===== =========== ======================= ====================================
PC    Name        Loading shape           Economic reading
===== =========== ======================= ====================================
1     Level       flat, same sign         parallel shift - the whole curve moves
2     Slope       monotone in maturity    steepening / flattening
3     Curvature   hump, ends vs belly     butterfly - belly rich or cheap
===== =========== ======================= ====================================

Two implementation choices matter and are easy to get wrong:

**Fit on changes, not levels.**  Yield *levels* are non-stationary; a PCA of
levels mostly recovers the sample mean and the slow secular decline in rates,
and the "factors" it produces are not tradable.  A PCA of daily *changes* is a
decomposition of risk, which is what a hedger or a relative-value trader
actually needs.

**Sign is arbitrary.**  An eigenvector and its negation are equally valid, and
which one LAPACK returns can flip between runs or between samples.  Left alone,
that silently flips the sign of a feature halfway through a backtest.
:func:`fit_curve_pca` therefore pins the orientation with an explicit
convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger("curve.pca")

__all__ = [
    "CurvePCA",
    "fit_curve_pca",
    "rolling_pca_factors",
    "FACTOR_NAMES",
    "reconstruction_error",
]

FACTOR_NAMES = ("level", "slope", "curvature")


def _factor_name(i: int) -> str:
    return FACTOR_NAMES[i] if i < len(FACTOR_NAMES) else f"pc{i + 1}"


@dataclass
class CurvePCA:
    """A fitted curve PCA.

    Attributes
    ----------
    components_:
        ``(n_factors, n_tenors)`` eigenvectors, sign-normalised.
    explained_variance_:
        Eigenvalues, descending.
    explained_variance_ratio_:
        Share of total variance per factor.
    mean_:
        Column means removed before the decomposition.
    tenors:
        Column labels, in the order the loadings refer to.
    """

    components_: np.ndarray
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray
    mean_: np.ndarray
    tenors: list[str] = field(default_factory=list)

    @property
    def n_factors(self) -> int:
        return int(self.components_.shape[0])

    @property
    def factor_names(self) -> list[str]:
        return [_factor_name(i) for i in range(self.n_factors)]

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Project observations onto the factors."""
        arr = X[self.tenors].to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, float)
        return (arr - self.mean_) @ self.components_.T

    def inverse_transform(self, F: np.ndarray) -> np.ndarray:
        """Rebuild curve changes from factor scores."""
        return np.asarray(F, dtype=float) @ self.components_ + self.mean_

    def factor_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Factor scores as a named DataFrame aligned to ``X``'s index."""
        scores = self.transform(X)
        return pd.DataFrame(scores, index=X.index, columns=self.factor_names)

    def loadings_frame(self) -> pd.DataFrame:
        """Loadings as ``tenor x factor``, for plotting and inspection."""
        return pd.DataFrame(self.components_.T, index=self.tenors, columns=self.factor_names)

    def summary(self) -> str:
        parts = [
            f"{n}={r:.1%}" for n, r in zip(self.factor_names, self.explained_variance_ratio_)
        ]
        return f"CurvePCA({', '.join(parts)}; cumulative={self.explained_variance_ratio_.sum():.2%})"


def _normalise_signs(components: np.ndarray, tenor_years: np.ndarray | None) -> np.ndarray:
    """Pin the arbitrary eigenvector sign to a fixed economic convention.

    * PC1 (level): loads positively on every tenor, so a positive score is a
      sell-off.
    * PC2 (slope): increasing in maturity, so a positive score is a steepening.
    * PC3 (curvature): the belly moves opposite to the wings, and the convention
      taken here is a positive score = belly *underperforms* (loads negatively
      in the middle, positively at the ends).
    * Any further PC: sign fixed by its largest-magnitude loading being positive,
      which is arbitrary but at least reproducible.
    """
    comps = components.copy()
    n_tenors = comps.shape[1]
    idx = np.arange(n_tenors, dtype=float) if tenor_years is None else np.asarray(tenor_years, float)

    for i in range(comps.shape[0]):
        v = comps[i]
        if i == 0:
            flip = v.sum() < 0
        elif i == 1:
            # Correlate the loading with maturity: positive => steepener.
            flip = np.corrcoef(idx, v)[0, 1] < 0 if np.std(v) > 0 else False
        elif i == 2:
            # Belly (middle third) versus the wings.
            lo, hi = n_tenors // 3, 2 * n_tenors // 3
            belly = v[lo:hi].mean() if hi > lo else v[n_tenors // 2]
            wings = np.concatenate([v[:lo], v[hi:]]).mean() if hi > lo else -belly
            flip = (wings - belly) < 0
        else:
            flip = v[int(np.argmax(np.abs(v)))] < 0
        if flip:
            comps[i] = -v
    return comps


def fit_curve_pca(
    changes: pd.DataFrame,
    n_factors: int = 3,
    sign_convention: bool = True,
    demean: bool = True,
) -> CurvePCA:
    """Fit a PCA to daily yield **changes**.

    Parameters
    ----------
    changes:
        ``date x tenor`` frame of yield changes (decimals).  Rows containing any
        NaN are dropped, so pass a set of tenors with common coverage - mixing
        the 1-month (from 2001) with the 30-year (gap 2002-2006) would silently
        discard most of the sample.
    n_factors:
        Number of components to keep.
    sign_convention:
        Apply the orientation rules described in :func:`_normalise_signs`.
        Leave this on; see the module docstring for why.
    demean:
        Remove column means first.  Daily yield changes have a mean near zero, so
        this matters little, but it makes the decomposition a true covariance PCA.

    Returns
    -------
    CurvePCA

    Notes
    -----
    Computed by SVD of the centred data matrix rather than by forming the
    covariance matrix and calling ``eigh``.  Both are algebraically identical;
    the SVD route is numerically better conditioned, because forming ``X'X``
    squares the condition number.
    """
    if n_factors < 1:
        raise ValueError("n_factors must be >= 1")
    clean = changes.dropna(how="any")
    if clean.empty:
        raise ValueError("No complete rows in the supplied changes frame")
    n_factors = min(n_factors, clean.shape[1])

    X = clean.to_numpy(dtype=float)
    mean = X.mean(axis=0) if demean else np.zeros(X.shape[1])
    Xc = X - mean

    # SVD: Xc = U S Vt.  Eigenvalues of the covariance are S^2 / (n-1).
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    variances = (S**2) / max(len(Xc) - 1, 1)
    total = variances.sum()
    comps = Vt[:n_factors]

    if sign_convention:
        from ..data.sources import TENOR_YEARS

        years = np.array([TENOR_YEARS.get(c, np.nan) for c in clean.columns], dtype=float)
        if not np.all(np.isfinite(years)):
            years = None
        comps = _normalise_signs(comps, years)

    return CurvePCA(
        components_=comps,
        explained_variance_=variances[:n_factors],
        explained_variance_ratio_=variances[:n_factors] / total if total > 0 else variances[:n_factors],
        mean_=mean,
        tenors=list(clean.columns),
    )


def reconstruction_error(pca: CurvePCA, changes: pd.DataFrame) -> pd.Series:
    """Per-date RMS error of rebuilding the curve change from the kept factors.

    A useful rich/cheap signal in its own right: a day the three factors cannot
    reproduce is a day some sector moved idiosyncratically.
    """
    clean = changes[pca.tenors].dropna(how="any")
    recon = pca.inverse_transform(pca.transform(clean))
    resid = clean.to_numpy(dtype=float) - recon
    return pd.Series(np.sqrt((resid**2).mean(axis=1)), index=clean.index, name="pca_resid")


def rolling_pca_factors(
    changes: pd.DataFrame,
    window: int = 252,
    n_factors: int = 3,
    min_periods: int | None = None,
    refit_every: int = 21,
    expanding: bool = False,
) -> pd.DataFrame:
    """Causal factor scores - loadings for day *t* use only data through *t-1*.

    This is the look-ahead-safe counterpart to :func:`fit_curve_pca`.  Fitting a
    single PCA on the whole sample and then using those scores as features is one
    of the most common (and most flattering) leaks in curve modelling: the
    loadings encode which tenors moved together over the *entire* history,
    including the future.

    Here the loadings are refitted every ``refit_every`` observations on a
    trailing window that **ends at t-1**, and are then applied to days ``t``
    onward until the next refit.  A day's score therefore never depends on its
    own observation or on anything after it.

    Parameters
    ----------
    changes:
        ``date x tenor`` yield changes.
    window:
        Trailing window length for each refit.  Ignored when ``expanding=True``.
    n_factors:
        Components to keep.
    min_periods:
        Minimum observations before the first score is produced.  Defaults to
        ``window``.
    refit_every:
        Refit cadence in observations.  Refitting daily is ~20x slower and
        changes the factors very little, since loadings are highly persistent.
    expanding:
        Use all history up to ``t-1`` instead of a fixed trailing window.

    Returns
    -------
    pd.DataFrame
        Columns ``level``, ``slope``, ``curvature``, ... indexed like ``changes``.
        Leading rows before ``min_periods`` are NaN.
    """
    clean = changes.dropna(how="any")
    if clean.empty:
        raise ValueError("No complete rows in the supplied changes frame")
    min_periods = int(min_periods or window)
    n_factors = min(n_factors, clean.shape[1])
    names = [_factor_name(i) for i in range(n_factors)]

    values = clean.to_numpy(dtype=float)
    n = len(values)
    out = np.full((n, n_factors), np.nan)

    fitted: CurvePCA | None = None
    last_fit = -10**9

    for t in range(n):
        if t < min_periods:
            continue
        if fitted is None or (t - last_fit) >= refit_every:
            start = 0 if expanding else max(0, t - window)
            # `t` is EXCLUSIVE here - that single slice bound is what makes the
            # whole function causal.
            train = clean.iloc[start:t]
            if len(train) < n_factors + 2:
                continue
            fitted = fit_curve_pca(train, n_factors=n_factors)
            last_fit = t
        out[t] = (values[t] - fitted.mean_) @ fitted.components_.T

    result = pd.DataFrame(out, index=clean.index, columns=names)
    return result.reindex(changes.index)
