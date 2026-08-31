"""Portfolio risk measurement for a fixed-income book.

Three questions, in the order a risk manager actually asks them:

1. **How do the tenors co-move?**  :func:`covariance` estimates the covariance
   of daily changes.  A rates book is not a basket of independent bets - the
   first principal component of the Treasury curve explains roughly 90% of
   daily variance - so a diagonal risk model would understate a directional
   book by a factor of ~3 and overstate a DV01-neutral curve trade by about as
   much.  Getting the correlation structure right *is* the risk model.

2. **How much can we lose on a normal bad day?**  :func:`parametric_var`,
   :func:`historical_var`, :func:`expected_shortfall`.  Parametric VaR is fast
   and smooth (an optimiser can differentiate it) but assumes Gaussian returns;
   historical VaR keeps the fat tails that Treasuries demonstrably have.
   Reporting both, and the ratio between them, is the cheapest tail diagnostic
   in existence: when historical/parametric drifts above ~1.3 the Gaussian
   model is lying to you.

3. **How much can we lose on an abnormal day?**  :func:`stress_scenarios` and
   :func:`apply_stress`.  VaR is structurally silent about 1994, 2008 and 2022
   because those moves are 10-20 sigma under a Gaussian fitted to a calm
   sample.  The historical scenarios here are the *realised* curve shocks of
   those episodes, measured tenor-by-tenor from the Treasury CMT history in
   ``data/processed/curve.parquet`` over the exact windows named in
   :data:`SCENARIO_INFO`, so the book is marked against what the market did
   rather than against somebody's intuition.

Units
-----
Everything except the stress block is unit-agnostic and the caller owns
consistency.  If ``returns`` are fractional P&L, VaR comes back as a fraction
of capital.  If they are dollar P&L per unit of DV01 (the natural rates-desk
choice: ``-delta_yield_bp`` per tenor, weighted by DV01), VaR comes back in
dollars.  The stress block is *always* in **basis points per tenor**.

Sign conventions
----------------
* VaR and Expected Shortfall are **positive numbers denoting a loss**.
* DV01 is **positive for a long** and equals the loss per **+1bp**, matching
  :func:`tqe.pricing.analytics.dv01`.  A stress P&L is therefore

  .. math:: \\text{PnL} = -\\sum_t \\text{DV01}_t \\times \\text{shock}_t^{bp}

  so a long book (positive DV01) *loses* when yields rise (positive shock).
  :func:`apply_stress` returns a **signed** P&L: negative is a loss.

No look-ahead
-------------
Every function here is a pure function of the sample it is handed - nothing is
shifted, reindexed, resampled or forward-filled, and no function reads a global
dataset.  Causality is therefore the *caller's* contract: inside a backtest the
covariance used to size day ``t`` must be estimated on returns up to and
including ``t-1`` only.  Fitting one covariance on the full sample and reusing
it every day is the classic in-sample risk-model leak - it makes 2008 look
foreseeable from 1998 and flatters every risk-adjusted metric downstream.
:func:`covariance` stamps ``attrs["sample_end"]`` on its output precisely so a
backtest can assert that the matrix it is about to use never saw the day being
sized.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..data.sources import TENOR_YEARS
from ..logging_utils import get_logger

log = get_logger("portfolio.risk")

__all__ = [
    "covariance",
    "parametric_var",
    "historical_var",
    "expected_shortfall",
    "stress_scenarios",
    "apply_stress",
    "stress_table",
    "risk_report",
    "SCENARIO_INFO",
    "COVARIANCE_METHODS",
]

COVARIANCE_METHODS: tuple[str, ...] = ("ewma", "sample", "ledoit_wolf")

#: Tenor grid the stress scenarios are quoted on.  It is the always-liquid CMT
#: set; :func:`apply_stress` interpolates in log-maturity for any book tenor not
#: listed here (e.g. "1 Mo"), so a scenario is a *curve shape*, not a lookup.
_STRESS_TENORS: tuple[str, ...] = (
    "3 Mo",
    "6 Mo",
    "1 Yr",
    "2 Yr",
    "3 Yr",
    "5 Yr",
    "7 Yr",
    "10 Yr",
    "20 Yr",
    "30 Yr",
)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _as_frame(returns: pd.DataFrame | pd.Series | np.ndarray) -> pd.DataFrame:
    """Coerce a returns panel to a DataFrame with usable column labels."""
    if isinstance(returns, pd.DataFrame):
        return returns
    if isinstance(returns, pd.Series):
        return returns.to_frame()
    arr = np.asarray(returns, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"returns must be 2-D, got shape {arr.shape}")
    return pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])


def _clean_1d(x: pd.Series | np.ndarray | Sequence[float]) -> np.ndarray:
    """Flatten to a finite 1-D float array, dropping NaN/inf."""
    arr = np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _align_weights(
    weights: pd.Series | Mapping[str, float] | np.ndarray | Sequence[float],
    cov: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Line ``weights`` up with ``cov`` and return ``(w, S, labels)``.

    Alignment is by label whenever both objects carry labels.  A silent
    positional zip of a Series against a DataFrame is one of the easiest ways to
    ship a risk number for the wrong book, so a missing label is a hard error
    rather than an implicit zero.
    """
    if isinstance(cov, pd.DataFrame):
        labels = [str(c) for c in cov.columns]
        S = cov.to_numpy(dtype=float)
        if list(cov.index.astype(str)) != labels:
            raise ValueError("cov must be square with matching index and columns")
    else:
        S = np.asarray(cov, dtype=float)
        labels = [f"x{i}" for i in range(S.shape[0])]

    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"cov must be square, got shape {S.shape}")

    if isinstance(weights, Mapping) and not isinstance(weights, pd.Series):
        weights = pd.Series(weights, dtype=float)

    if isinstance(weights, pd.Series):
        idx = weights.index.astype(str)
        missing = [lab for lab in labels if lab not in set(idx)]
        if missing:
            raise KeyError(f"weights are missing covariance assets: {missing}")
        w = weights.set_axis(idx).reindex(labels).to_numpy(dtype=float)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != S.shape[0]:
            raise ValueError(f"weights length {w.size} != cov dimension {S.shape[0]}")

    if not np.all(np.isfinite(w)):
        raise ValueError("weights contain NaN/inf")
    return w, S, labels


def _project_psd(S: np.ndarray, eig_floor_rel: float = 1e-10) -> tuple[np.ndarray, int]:
    """Symmetrise ``S`` and clip its spectrum onto the PSD cone.

    A covariance matrix must be positive semi-definite or ``w'Sw`` can come out
    negative and the VaR of a real book becomes the square root of a negative
    number.  Estimation error, EWMA weighting with a short half-life and any
    pairwise-complete NaN handling can all produce tiny negative eigenvalues.

    The fix is the standard eigenvalue projection: ``S = V diag(max(l, floor))
    V'``.  The floor is set *relative to the largest eigenvalue* rather than to
    zero so that the reconstruction's own floating-point error (order
    ``eps * ||S||``) cannot push the smallest eigenvalue back below zero - a
    literal zero floor routinely returns matrices whose minimum eigenvalue is
    ``-1e-22`` and fails a naive ``>= 0`` assertion.

    Returns
    -------
    tuple[numpy.ndarray, int]
        The projected matrix and the number of eigenvalues that were clipped.
    """
    S = 0.5 * (S + S.T)
    eigvals, eigvecs = np.linalg.eigh(S)
    max_eig = float(eigvals[-1])
    if max_eig <= 0.0:
        # Degenerate (all-zero or all-negative) input: the only PSD matrix
        # consistent with it is the zero matrix.
        return np.zeros_like(S), int(np.sum(eigvals < 0.0))
    floor = eig_floor_rel * max_eig
    n_clipped = int(np.sum(eigvals < floor))
    if n_clipped == 0:
        return S, 0
    fixed = (eigvecs * np.clip(eigvals, floor, None)) @ eigvecs.T
    return 0.5 * (fixed + fixed.T), n_clipped


def _ewma_weights(n: int, halflife: float) -> np.ndarray:
    """Normalised exponential weights, newest observation last.

    ``lambda = 0.5 ** (1/halflife)`` so an observation ``halflife`` days old
    carries exactly half the weight of today's.  RiskMetrics' classic
    ``lambda = 0.94`` corresponds to a half-life of ~11 days; the 63-day default
    here (one quarter) is the usual compromise for a daily-rebalanced rates book
    between reacting to a vol regime change and not chasing noise.
    """
    if halflife <= 0:
        raise ValueError(f"halflife must be positive, got {halflife}")
    lam = 0.5 ** (1.0 / float(halflife))
    age = np.arange(n - 1, -1, -1, dtype=float)  # oldest row gets age n-1
    w = lam**age
    return w / w.sum()


def _slope_shock(bp_2y: float, bp_10y: float, tau: float = 3.0) -> dict[str, float]:
    """Build a synthetic curve shock from a level plus a Nelson-Siegel slope.

    The shock is ``a + b * f(t)`` where ``f(t) = (1 - exp(-t/tau)) / (t/tau)`` is
    the Nelson-Siegel *slope* loading: monotonically decaying from 1 at the front
    of the curve to ~0.1 at 30 years.  ``a`` and ``b`` are solved so the shock
    hits ``bp_2y`` at 2 years and ``bp_10y`` at 10 years.

    Why this and not a straight line in maturity: the empirical second principal
    component of daily Treasury yield changes has exactly this decaying shape.
    Anchoring on 2s10s and extrapolating linearly in maturity (or in log
    maturity) produces absurd wings - a "bear steepener" in which the 3-month
    bill *rallies* 40bp.  With the NS loading, a 2s10s +50bp bear steepener comes
    out as roughly ``3M +3, 2Y +25, 10Y +75, 30Y +95``, which is what a bear
    steepener looks like on a screen.
    """
    years = np.array([TENOR_YEARS[t] for t in _STRESS_TENORS], dtype=float)
    load = (1.0 - np.exp(-years / tau)) / (years / tau)
    f2 = float((1.0 - np.exp(-2.0 / tau)) / (2.0 / tau))
    f10 = float((1.0 - np.exp(-10.0 / tau)) / (10.0 / tau))
    b = (bp_2y - bp_10y) / (f2 - f10)
    a = bp_10y - b * f10
    return {t: round(float(a + b * v), 1) for t, v in zip(_STRESS_TENORS, load)}


def _flat_shock(bp: float) -> dict[str, float]:
    """A parallel shift of ``bp`` basis points across every stress tenor."""
    return {t: float(bp) for t in _STRESS_TENORS}


# --------------------------------------------------------------------------- #
# Covariance
# --------------------------------------------------------------------------- #
def covariance(
    returns: pd.DataFrame | np.ndarray,
    method: str = "ewma",
    halflife: float = 63,
    shrinkage: float = 0.1,
    *,
    min_obs: int = 30,
    eig_floor_rel: float = 1e-10,
) -> pd.DataFrame:
    """Estimate the covariance matrix of a returns panel.

    Parameters
    ----------
    returns : pandas.DataFrame or numpy.ndarray
        Observations in rows, assets in columns.  Daily yield changes, daily
        total returns and daily P&L per unit DV01 are all valid inputs; the
        estimator is unit-agnostic.
    method : {"ewma", "sample", "ledoit_wolf"}, default "ewma"
        ``"ewma"``
            Exponentially weighted, half-life ``halflife`` days.  The right
            default for trading: rates volatility is strongly clustered, so a
            2015 observation should not carry the same weight as yesterday's
            when sizing today's book.
        ``"sample"``
            Equal-weighted sample covariance (``ddof=1``).  Slow to react but
            unbiased and the natural benchmark.
        ``"ledoit_wolf"``
            Sample covariance with the analytically optimal shrinkage intensity
            of Ledoit & Wolf (2004) toward a scaled identity.  The intensity is
            estimated from the data, so the ``shrinkage`` argument is ignored
            and the realised intensity is reported in ``attrs["shrinkage"]``.
    halflife : float, default 63
        EWMA half-life in observations (~one quarter of trading days).  Ignored
        by the other methods.
    shrinkage : float, default 0.1
        Intensity ``a`` in ``S_shrunk = (1 - a) * S + a * diag(S)``, applied to
        ``"ewma"`` and ``"sample"``.  The diagonal target leaves every asset's
        own variance **exactly** unchanged and shrinks only the off-diagonals,
        i.e. it damps correlations toward zero by a factor ``(1 - a)``.  That is
        the right bias to accept in a rates book: single-tenor volatilities are
        estimated from thousands of observations and are reliable, whereas the
        45 pairwise correlations of a 10-tenor curve are the noisy part, and a
        correlation that is over-estimated at 0.99 makes a DV01-neutral curve
        trade look riskless to the optimiser.
    min_obs : int, keyword-only, default 30
        Minimum number of complete rows required after NaN handling.
    eig_floor_rel : float, keyword-only, default 1e-10
        Smallest permitted eigenvalue as a fraction of the largest; see
        :func:`_project_psd`.

    Returns
    -------
    pandas.DataFrame
        Square, symmetric, positive semi-definite covariance matrix indexed and
        columned by asset.  ``DataFrame.attrs`` carries ``method``, ``halflife``,
        ``shrinkage`` (the intensity actually applied), ``n_obs``, ``n_dropped``,
        ``n_eigenvalues_clipped``, ``min_eigenvalue``, ``condition_number`` and
        ``sample_start`` / ``sample_end`` timestamps.

    Raises
    ------
    ValueError
        Unknown ``method``, ``shrinkage`` outside ``[0, 1]``, or fewer than
        ``min_obs`` complete observations.

    Notes
    -----
    **NaN handling.**  Rows containing any NaN are dropped wholesale (listwise
    deletion).  The tempting alternative - pairwise-complete estimation, where
    each entry uses whatever overlap that pair happens to have - is what
    ``pandas.DataFrame.cov`` does by default, and it routinely produces
    matrices that are *not* positive semi-definite because the entries come from
    inconsistent samples.  Forward-filling instead would be worse still: the
    Treasury 30-year has a genuine publication gap from 2002-02 to 2006-02, and
    filling it manufactures four years of exactly-zero yield changes, which
    collapses the estimated 30-year volatility and makes a long-bond position
    look nearly riskless.  A warning is logged when more than 5% of rows are
    dropped, because that usually means a short-history tenor (``1 Mo`` starts
    2001, ``2 Mo`` 2018, ``4 Mo`` 2022) has silently truncated the sample.

    **Guaranteed PSD.**  The result is symmetrised and its spectrum is clipped
    at ``eig_floor_rel * max_eigenvalue``, so ``w'Sw >= 0`` for every ``w``.
    Shrinkage toward a non-negative diagonal cannot break PSD, so in practice
    the projection is a no-op and ``attrs["n_eigenvalues_clipped"]`` is 0; it
    exists to keep the guarantee unconditional when ``n_obs <= n_assets`` or the
    EWMA half-life is short enough that the effective sample size is below the
    asset count.

    **Causality.**  A pure function of the rows it is given: no shifting, no
    reindexing, no interpolation.  Inside a walk-forward backtest, slice
    ``returns.loc[:t - 1 day]`` before calling, and cross-check
    ``attrs["sample_end"]`` against the day being sized.
    """
    if method not in COVARIANCE_METHODS:
        raise ValueError(f"method must be one of {COVARIANCE_METHODS}, got {method!r}")
    if not 0.0 <= float(shrinkage) <= 1.0:
        raise ValueError(f"shrinkage must lie in [0, 1], got {shrinkage}")

    frame = _as_frame(returns)
    n_raw = len(frame)
    clean = frame.dropna(axis=0, how="any")
    n_obs, n_assets = clean.shape
    n_dropped = n_raw - n_obs

    if n_assets == 0:
        raise ValueError("returns has no columns")
    if n_obs < min_obs:
        raise ValueError(
            f"only {n_obs} complete observations after dropping NaN rows "
            f"(need >= {min_obs}); check for a short-history tenor in "
            f"{list(frame.columns)}"
        )
    if n_dropped > 0.05 * max(n_raw, 1):
        log.warning(
            "covariance dropped %d/%d rows (%.1f%%) to NaN - a ragged tenor is "
            "truncating the sample",
            n_dropped,
            n_raw,
            100.0 * n_dropped / max(n_raw, 1),
        )
    if n_obs <= n_assets:
        log.warning(
            "covariance has %d observations for %d assets - the sample matrix is "
            "singular; shrinkage/PSD projection is doing the heavy lifting",
            n_obs,
            n_assets,
        )

    X = clean.to_numpy(dtype=float)
    applied_shrinkage = float(shrinkage)

    if method == "ewma":
        w = _ewma_weights(n_obs, halflife)
        mu = w @ X
        Z = X - mu
        # Reliability-weight bias correction: dividing by (1 - sum w^2) is the
        # weighted analogue of Bessel's n-1, needed because the mean subtracted
        # above was estimated from the same weighted sample.
        S = (Z * w[:, None]).T @ Z / (1.0 - float(np.sum(w**2)))
    elif method == "sample":
        S = np.cov(X, rowvar=False, ddof=1)
        S = np.atleast_2d(S)
    else:  # ledoit_wolf
        # Imported lazily so this module stays importable in a slim environment.
        from sklearn.covariance import ledoit_wolf

        S, lw_intensity = ledoit_wolf(X)
        applied_shrinkage = float(lw_intensity)

    if method in ("ewma", "sample") and applied_shrinkage > 0.0:
        target = np.diag(np.diag(S))
        S = (1.0 - applied_shrinkage) * S + applied_shrinkage * target

    S, n_clipped = _project_psd(S, eig_floor_rel=eig_floor_rel)
    eigvals = np.linalg.eigvalsh(S)
    max_eig = float(eigvals[-1])
    min_eig = float(eigvals[0])

    out = pd.DataFrame(S, index=clean.columns, columns=clean.columns)
    index = clean.index
    out.attrs.update(
        {
            "method": method,
            "halflife": float(halflife) if method == "ewma" else None,
            "shrinkage": applied_shrinkage,
            "n_obs": int(n_obs),
            "n_dropped": int(n_dropped),
            "n_eigenvalues_clipped": int(n_clipped),
            "min_eigenvalue": min_eig,
            "condition_number": float(max_eig / min_eig) if min_eig > 0 else float("inf"),
            "sample_start": index[0] if isinstance(index, pd.DatetimeIndex) else None,
            "sample_end": index[-1] if isinstance(index, pd.DatetimeIndex) else None,
        }
    )
    return out


# --------------------------------------------------------------------------- #
# Value at Risk / Expected Shortfall
# --------------------------------------------------------------------------- #
def parametric_var(
    weights: pd.Series | Mapping[str, float] | np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1,
) -> float:
    """Gaussian (variance-covariance) Value at Risk.

    .. math:: \\text{VaR} = z_{c}\\,\\sqrt{w' S w}\\,\\sqrt{h}

    Parameters
    ----------
    weights : pandas.Series, mapping or array
        Position weights in the same units as the returns used to build ``cov``.
        Aligned to ``cov`` **by label** when both are labelled; a label present
        in ``cov`` but absent from ``weights`` raises rather than defaulting to
        zero.  Negative entries (shorts) are fully supported - the quadratic form
        handles them, which is exactly why VaR is computed from the covariance
        rather than from a sum of per-asset VaRs.
    cov : pandas.DataFrame or array
        Covariance matrix of one-period returns, e.g. from :func:`covariance`.
    confidence : float, default 0.99
        Confidence level in ``(0, 1)``.  0.99 is the Basel market-risk standard;
        0.95 is the usual internal number.
    horizon : int, default 1
        Holding period in the same periods as ``cov``.  Scaled by
        ``sqrt(horizon)``.

    Returns
    -------
    float
        A **positive** number: the loss that is exceeded with probability
        ``1 - confidence``.  Zero for a flat book.

    Notes
    -----
    Two assumptions are baked in and both are worth stating out loud rather than
    discovering in a drawdown.

    *Zero drift.*  The expected return over the horizon is set to 0.  Over a
    day or ten this is well inside the noise, and assuming zero is the
    conservative choice - a positive drift would shrink the reported loss.

    *Square-root-of-time.*  Scaling by ``sqrt(horizon)`` is exact only for
    i.i.d. returns.  Treasury yield changes have mild negative autocorrelation
    at the daily frequency (which makes it slightly conservative) but strong
    volatility clustering (which makes it *optimistic* precisely in the
    stressed periods where a 10-day number is used).  For horizons beyond a
    couple of weeks, prefer historical simulation of overlapping windows.

    The Gaussian assumption is the third: Treasury daily changes are leptokurtic,
    so this number is systematically below :func:`historical_var` at 99%.  Report
    the pair, not either alone.

    **Causality.**  A pure function of the covariance handed in; it inherits
    whatever sample the caller used.  See the module docstring.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    w, S, _ = _align_weights(weights, cov)
    variance = float(w @ S @ w)
    # S is PSD by construction, so this can only be negative by round-off.
    variance = max(variance, 0.0)
    z = float(stats.norm.ppf(confidence))
    return float(z * np.sqrt(variance) * np.sqrt(float(horizon)))


def historical_var(
    portfolio_returns: pd.Series | np.ndarray, confidence: float = 0.99
) -> float:
    """Historical-simulation Value at Risk.

    The empirical ``1 - confidence`` quantile of the realised P&L distribution,
    sign-flipped so a loss is positive.  No distributional assumption at all:
    if the sample contains 2008, the number contains 2008.

    Parameters
    ----------
    portfolio_returns : pandas.Series or array
        One-period portfolio P&L / returns.  NaN and inf are dropped.
    confidence : float, default 0.99
        Confidence level in ``(0, 1)``.

    Returns
    -------
    float
        Positive loss at the given confidence.  It can legitimately come out
        **negative** for a book that made money even in the worst 1% of days;
        that is information, so it is not clamped at zero.

    Notes
    -----
    Precision at 99% is limited by sample size: with 250 observations the
    estimate rests on the 2nd- and 3rd-worst days, and the standard error of the
    quantile is large.  Prefer several years of history, and read
    :func:`expected_shortfall` alongside it - the mean of the tail is a far more
    stable statistic than a single order statistic.

    **Causality.**  Pure function of the sample given; it makes no attempt to
    reindex or align, so a backtest passing ``returns.loc[:t-1]`` is safe by
    construction.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    r = _clean_1d(portfolio_returns)
    if r.size == 0:
        raise ValueError("portfolio_returns contains no finite observations")
    return float(-np.quantile(r, 1.0 - confidence))


def expected_shortfall(
    portfolio_returns: pd.Series | np.ndarray, confidence: float = 0.99
) -> float:
    """Expected Shortfall (CVaR, conditional VaR, expected tail loss).

    The mean loss *given* that the VaR threshold was breached: the average of
    every observation at or below the ``1 - confidence`` quantile, sign-flipped.

    Parameters
    ----------
    portfolio_returns : pandas.Series or array
        One-period portfolio P&L / returns.  NaN and inf are dropped.
    confidence : float, default 0.99
        Confidence level in ``(0, 1)``; must match the VaR it is reported with.

    Returns
    -------
    float
        Positive expected loss in the tail.

    Notes
    -----
    ``ES >= VaR`` always, by construction: every observation averaged in is at
    or below the quantile, so its mean is too.  The module's test-suite asserts
    this and it is worth understanding *why* it is an identity rather than an
    empirical finding.

    ES is the reason the Basel Committee moved off VaR in FRTB.  VaR says
    nothing about the shape beyond the threshold, so two books with identical
    99% VaR can have wildly different catastrophic exposure; and VaR is not
    sub-additive, meaning the VaR of a merged book can *exceed* the sum of the
    parts, which makes it useless for allocating risk limits.  ES is coherent
    and does neither.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    r = _clean_1d(portfolio_returns)
    if r.size == 0:
        raise ValueError("portfolio_returns contains no finite observations")
    threshold = np.quantile(r, 1.0 - confidence)
    tail = r[r <= threshold]
    if tail.size == 0:  # unreachable: the quantile is never below the minimum
        return float(-threshold)
    return float(-tail.mean())


# --------------------------------------------------------------------------- #
# Stress testing
# --------------------------------------------------------------------------- #
#: Provenance for every scenario.  Historical entries name the exact window the
#: shock was measured over in ``data/processed/curve.parquet`` (US Treasury
#: Daily Par Yield Curve Rates, the "CMT" curve); the values in
#: :func:`stress_scenarios` are the realised tenor-by-tenor changes over that
#: window, rounded to the nearest basis point.
SCENARIO_INFO: dict[str, str] = {
    "bond_massacre_1994": (
        "Historical: 1993-12-31 -> 1994-11-30. Greenspan's unsignalled tightening "
        "cycle took Fed Funds from 3.00% to 5.50% in eleven months. A violent bear "
        "flattener: the 1Y rose 328bp against 208bp on the 10Y and 164bp on the 30Y. "
        "The episode that killed Orange County and Askin Capital, and the reason "
        "central banks now telegraph moves."
    ),
    "flight_to_quality_2008": (
        "Historical: 2008-10-31 -> 2008-11-28. Post-Lehman panic. The front end was "
        "already pinned at the zero bound (3M 0.46% -> 0.01%, so only -45bp of room), "
        "while the duration bid drove the 10Y down 108bp. The canonical bull "
        "flattener - and the scenario that hurts a short-duration or steepener book."
    ),
    "taper_tantrum_2013": (
        "Historical: 2013-05-01 -> 2013-09-05. Bernanke's 22 May 'step down the pace "
        "of purchases' remark. A textbook bear steepener: the front end was anchored "
        "by forward guidance (3M -4bp, 1Y +5bp) while the belly and 10Y sold off 120-138bp. "
        "The scenario that punishes carry-and-rolldown books hardest."
    ),
    "covid_flight_2020": (
        "Historical: 2020-02-19 -> 2020-03-09. Equity peak to the 10Y all-time low of "
        "0.54%. A near-parallel collapse of ~100bp as the Fed cut 150bp inside two weeks; "
        "the 3M fell 125bp and the 30Y 102bp."
    ),
    "covid_reversal_2020": (
        "Historical: 2020-03-09 -> 2020-03-18. The violent reversal nine days later: a "
        "dash-for-cash forced liquidation of Treasuries (relative-value basis trades "
        "unwinding) sent the 10Y back up 64bp and the 30Y 78bp while bills kept rallying. "
        "Paired with covid_flight_2020 this is the real lesson of March 2020 - the "
        "round trip, not the direction."
    ),
    "hiking_cycle_2022": (
        "Historical: 2021-12-31 -> 2022-12-30. The fastest tightening cycle since 1980: "
        "425bp of hikes. A massive bear flattener into deep inversion - 6M +457bp, "
        "2Y +368bp, 10Y +236bp, 30Y +207bp - which drove the worst calendar year for "
        "US Treasury total return in the history of the series."
    ),
    "parallel_up_100": "Synthetic: +100bp parallel shift. The standard first-order duration shock.",
    "parallel_down_100": "Synthetic: -100bp parallel shift.",
    "bear_steepener": (
        "Synthetic: yields up, long end more (2s10s +50bp). Anchored at 2Y +25bp / "
        "10Y +75bp and shaped by the Nelson-Siegel slope loading (tau=3y). Term-premium "
        "repricing / fiscal supply shock."
    ),
    "bull_steepener": (
        "Synthetic: yields down, front end more (2s10s +50bp). Anchored at 2Y -75bp / "
        "10Y -25bp. The easing-cycle shape - a recession scare that prices cuts."
    ),
    "bear_flattener": (
        "Synthetic: yields up, front end more (2s10s -50bp). Anchored at 2Y +75bp / "
        "10Y +25bp. The hiking-cycle shape - an inflation surprise that prices hikes."
    ),
    "bull_flattener": (
        "Synthetic: yields down, long end more (2s10s -50bp). Anchored at 2Y -25bp / "
        "10Y -75bp. The duration-grab shape - growth downgrade with policy on hold."
    ),
}

# Historical shocks in basis points, measured tenor-by-tenor over the windows
# documented in SCENARIO_INFO.  Held as a module constant so the scenario set is
# deterministic and offline - it must not depend on whatever data happens to be
# on disk when the risk report runs.
_HISTORICAL_SHOCKS: dict[str, dict[str, float]] = {
    "bond_massacre_1994": {
        "3 Mo": 265.0, "6 Mo": 292.0, "1 Yr": 328.0, "2 Yr": 315.0, "3 Yr": 304.0,
        "5 Yr": 258.0, "7 Yr": 231.0, "10 Yr": 208.0, "20 Yr": 162.0, "30 Yr": 164.0,
    },
    "flight_to_quality_2008": {
        "3 Mo": -45.0, "6 Mo": -50.0, "1 Yr": -44.0, "2 Yr": -56.0, "3 Yr": -53.0,
        "5 Yr": -87.0, "7 Yr": -94.0, "10 Yr": -108.0, "20 Yr": -103.0, "30 Yr": -90.0,
    },
    "taper_tantrum_2013": {
        "3 Mo": -4.0, "6 Mo": -2.0, "1 Yr": 5.0, "2 Yr": 32.0, "3 Yr": 67.0,
        "5 Yr": 120.0, "7 Yr": 138.0, "10 Yr": 132.0, "20 Yr": 120.0, "30 Yr": 105.0,
    },
    "covid_flight_2020": {
        "3 Mo": -125.0, "6 Mo": -129.0, "1 Yr": -116.0, "2 Yr": -104.0, "3 Yr": -99.0,
        "5 Yr": -95.0, "7 Yr": -94.0, "10 Yr": -102.0, "20 Yr": -99.0, "30 Yr": -102.0,
    },
    "covid_reversal_2020": {
        "3 Mo": -31.0, "6 Mo": -19.0, "1 Yr": -10.0, "2 Yr": 16.0, "3 Yr": 26.0,
        "5 Yr": 33.0, "7 Yr": 52.0, "10 Yr": 64.0, "20 Yr": 73.0, "30 Yr": 78.0,
    },
    "hiking_cycle_2022": {
        "3 Mo": 436.0, "6 Mo": 457.0, "1 Yr": 434.0, "2 Yr": 368.0, "3 Yr": 325.0,
        "5 Yr": 273.0, "7 Yr": 252.0, "10 Yr": 236.0, "20 Yr": 220.0, "30 Yr": 207.0,
    },
}

_SYNTHETIC_SHOCKS: dict[str, dict[str, float]] = {
    "parallel_up_100": _flat_shock(100.0),
    "parallel_down_100": _flat_shock(-100.0),
    "bear_steepener": _slope_shock(bp_2y=25.0, bp_10y=75.0),
    "bull_steepener": _slope_shock(bp_2y=-75.0, bp_10y=-25.0),
    "bear_flattener": _slope_shock(bp_2y=75.0, bp_10y=25.0),
    "bull_flattener": _slope_shock(bp_2y=-25.0, bp_10y=-75.0),
}

_SCENARIOS: dict[str, dict[str, float]] = {**_HISTORICAL_SHOCKS, **_SYNTHETIC_SHOCKS}


def stress_scenarios() -> dict[str, dict[str, float]]:
    """The named curve shocks, in **basis points per tenor**.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{scenario_name: {tenor_label: shock_bp}}``.  A fresh copy is returned
        on every call, so a caller that mutates the result (to bump a shock, or
        to add a bespoke scenario) cannot corrupt the module constant.

    Notes
    -----
    Six historical and six synthetic scenarios.  The historical shocks are the
    realised tenor-by-tenor changes in the US Treasury CMT par-yield curve over
    the exact windows recorded in :data:`SCENARIO_INFO`, taken from
    ``data/processed/curve.parquet``, rounded to the nearest basis point:

    ==========================  ======  ======  ======  ======
    Scenario                      3 Mo    2 Yr   10 Yr   30 Yr
    ==========================  ======  ======  ======  ======
    bond_massacre_1994            +265    +315    +208    +164
    flight_to_quality_2008         -45     -56    -108     -90
    taper_tantrum_2013              -4     +32    +132    +105
    covid_flight_2020             -125    -104    -102    -102
    covid_reversal_2020            -31     +16     +64     +78
    hiking_cycle_2022             +436    +368    +236    +207
    ==========================  ======  ======  ======  ======

    They are deliberately *not* recomputed from disk at call time: a risk limit
    must be reproducible, and a scenario set that silently changes when the data
    is refreshed is a scenario set nobody can sign off.  The magnitudes are
    reproduced by the module's verification script against the parquet.

    Why both kinds.  Historical scenarios are defensible - "this happened" ends
    the argument - but they are a sample of six from a distribution, and the
    next crisis will not be one of them.  Synthetic level/slope shocks fill the
    space between: they are constructed from the Nelson-Siegel level and slope
    loadings, which are empirically the first two principal components of daily
    Treasury yield changes, so together they span the moves that actually
    dominate a rates book's variance.

    Note that the four synthetic twists are all *large* single-day-equivalent
    moves; the historical ones span weeks to a year and are therefore not
    comparable as daily shocks.  They answer different questions - "what if the
    curve twists hard tomorrow" versus "what if 1994 happens again" - and both
    belong on the same page.
    """
    return {name: dict(shock) for name, shock in _SCENARIOS.items()}


def apply_stress(
    positions_dv01: Mapping[str, float],
    scenario: Mapping[str, float],
    *,
    interpolate: bool = True,
) -> float:
    """Mark a book to a curve shock and return the P&L.

    .. math:: \\text{PnL} = -\\sum_t \\text{DV01}_t \\times \\text{shock}_t^{bp}

    Parameters
    ----------
    positions_dv01 : mapping
        ``{tenor_label: signed DV01}`` in currency per basis point.  **Positive
        means long duration** (you own the bond and lose when yields rise),
        matching :func:`tqe.pricing.analytics.dv01`; negative means short.
    scenario : mapping
        ``{tenor_label: shock_bp}``, e.g. one entry of
        :func:`stress_scenarios`.
    interpolate : bool, keyword-only, default True
        When the book holds a tenor the scenario does not quote, interpolate the
        shock linearly in **log-maturity** from the scenario's own points
        (flat-extrapolated at the ends).  A scenario is a curve *shape*, so a
        1-month bill should inherit the front-end shock rather than silently
        receive zero - defaulting a missing tenor to zero understates risk,
        which is the one direction a risk system must never err in.  Set
        ``False`` to require exact labels.

    Returns
    -------
    float
        **Signed** P&L in the currency of ``positions_dv01``.  Negative is a
        loss.

    Raises
    ------
    KeyError
        A book tenor is absent from the scenario and either ``interpolate`` is
        ``False`` or the label is not in
        :data:`tqe.data.sources.TENOR_YEARS` so its maturity is unknown.

    Examples
    --------
    A book long $10,000 of DV01 in the 10-year under the 2022 hiking cycle
    (10Y +236bp) loses ``-10_000 * 236 = -$2,360,000``.  A first-order estimate:
    it ignores convexity, which for a long position is a *positive* correction,
    so the linear number is conservative for a long and optimistic for a short.

    Notes
    -----
    **Causality.**  Instantaneous mark of a stated book against a stated shock;
    no time series is touched.  The only causality requirement is on the caller:
    the DV01s must be the ones held going into the day, not the ones chosen
    afterwards with hindsight.
    """
    if not positions_dv01:
        return 0.0

    scen_labels = [t for t in scenario if t in TENOR_YEARS]
    if interpolate and scen_labels:
        order = np.argsort([TENOR_YEARS[t] for t in scen_labels])
        scen_years = np.array([TENOR_YEARS[scen_labels[i]] for i in order], dtype=float)
        scen_bp = np.array([float(scenario[scen_labels[i]]) for i in order], dtype=float)
        log_scen_years = np.log(scen_years)
    else:
        log_scen_years = scen_bp = None  # type: ignore[assignment]

    pnl = 0.0
    for tenor, dv01_amount in positions_dv01.items():
        if tenor in scenario:
            shock = float(scenario[tenor])
        elif interpolate and log_scen_years is not None and tenor in TENOR_YEARS:
            # np.interp flat-extrapolates outside the range, which is the right
            # behaviour here: a 1-month bill takes the 3-month shock rather than
            # an extrapolated one that could flip sign.
            shock = float(np.interp(np.log(TENOR_YEARS[tenor]), log_scen_years, scen_bp))
        else:
            raise KeyError(
                f"tenor {tenor!r} is not in the scenario and cannot be interpolated "
                f"(interpolate={interpolate}, known tenor={tenor in TENOR_YEARS})"
            )
        pnl -= float(dv01_amount) * shock
    return float(pnl)


def stress_table(
    positions_dv01: Mapping[str, float],
    scenarios: Mapping[str, Mapping[str, float]] | None = None,
    *,
    interpolate: bool = True,
) -> pd.DataFrame:
    """Run every scenario over one book and return a sorted P&L table.

    Parameters
    ----------
    positions_dv01 : mapping
        ``{tenor_label: signed DV01}``; see :func:`apply_stress`.
    scenarios : mapping, optional
        Defaults to :func:`stress_scenarios`.
    interpolate : bool, keyword-only, default True
        Passed through to :func:`apply_stress`.

    Returns
    -------
    pandas.DataFrame
        Indexed by scenario name, worst P&L first, with columns ``pnl``,
        ``kind`` (``"historical"`` / ``"synthetic"``) and ``description``.
        Sorting worst-first is deliberate: a stress report is read from the top
        and the first line should be the one that hurts.
    """
    scenarios = stress_scenarios() if scenarios is None else scenarios
    rows = [
        {
            "scenario": name,
            "pnl": apply_stress(positions_dv01, shock, interpolate=interpolate),
            "kind": "historical" if name in _HISTORICAL_SHOCKS else "synthetic",
            "description": SCENARIO_INFO.get(name, ""),
        }
        for name, shock in scenarios.items()
    ]
    table = pd.DataFrame(rows).set_index("scenario")
    return table.sort_values("pnl", ascending=True)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def risk_report(
    weights: pd.Series | Mapping[str, float] | np.ndarray,
    returns: pd.DataFrame,
    cov: pd.DataFrame,
    positions_dv01: Mapping[str, float] | None = None,
    *,
    confidences: Sequence[float] = (0.95, 0.99),
    horizons: Sequence[int] = (1, 10),
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Assemble the full risk picture for one book into a single dict.

    Parameters
    ----------
    weights : pandas.Series, mapping or array
        Position weights, aligned to ``cov`` by label.  Units must match
        ``returns`` (fractional weights against fractional returns, or DV01
        against P&L-per-unit-DV01).
    returns : pandas.DataFrame
        Historical per-asset returns used for the historical-simulation block.
        Rows with any NaN are dropped, consistently with :func:`covariance`.
    cov : pandas.DataFrame
        Covariance matrix, typically from :func:`covariance`.
    positions_dv01 : mapping, optional
        ``{tenor_label: signed DV01}``.  When given, the report gains a DV01
        summary and the full stress table.
    confidences : sequence of float, keyword-only, default (0.95, 0.99)
        VaR/ES confidence levels to report.
    horizons : sequence of int, keyword-only, default (1, 10)
        Parametric VaR horizons in periods.  10 days is the regulatory holding
        period for market risk.
    periods_per_year : float, keyword-only, default 252
        Annualisation factor for volatility.  Use
        :func:`tqe.data.calendar.annualization_factor` on the return index for a
        data-derived value.

    Returns
    -------
    dict
        JSON-serialisable, with these blocks:

        ``exposure``
            gross, net, per-asset weights and a Herfindahl concentration index.
        ``volatility``
            per-period and annualised portfolio volatility from ``cov``, plus
            the realised volatility of the fixed-weight historical series and
            the diversification ratio.
        ``var``
            ``parametric_{c}_{h}d``, ``historical_{c}``, ``expected_shortfall_{c}``
            and ``tail_ratio_{c}`` (historical / parametric at 1 day) - the
            single most useful number in the block, because it says how far the
            Gaussian model is from the realised tail.
        ``risk_contribution``
            Euler decomposition ``CTR_i = w_i (Sw)_i / sigma``, which sums
            **exactly** to the portfolio volatility, plus percentages.
        ``dv01`` and ``stress``
            Only when ``positions_dv01`` is supplied.

    Notes
    -----
    **What the historical block actually asks.**  It applies *today's* weights
    to every historical day and reads the tail of the resulting counterfactual
    P&L series: "what would the book I am holding right now have lost on each
    day since 1990?"  That is historical simulation as a desk runs it, and it is
    causal - it uses only past returns and present positions.  It is emphatically
    not the strategy's realised P&L, which would require the weights actually
    held on each of those days.

    **Risk contributions can be negative.**  For a long-short book a hedging leg
    has ``w_i (Sw)_i < 0`` and genuinely *removes* risk.  The contributions
    still sum to the total volatility by Euler's theorem on the homogeneous
    degree-1 function ``sigma(w)``; percentages outside ``[0, 100]`` are correct,
    not a bug.

    **Causality.**  Pure function of ``weights``, ``returns`` and ``cov``.  The
    covariance's own sample window is echoed back in the output
    (``covariance.sample_end``) so a backtest can assert it predates the day
    being sized.
    """
    w, S, labels = _align_weights(weights, cov)

    # ---- exposure -------------------------------------------------------- #
    gross = float(np.abs(w).sum())
    net = float(w.sum())
    hhi = float(np.sum(w**2) / gross**2) if gross > 0 else 0.0

    # ---- volatility ------------------------------------------------------ #
    variance = max(float(w @ S @ w), 0.0)
    sigma = float(np.sqrt(variance))
    asset_sigma = np.sqrt(np.clip(np.diag(S), 0.0, None))
    undiversified = float(np.abs(w) @ asset_sigma)
    div_ratio = float(undiversified / sigma) if sigma > 0 else float("nan")

    # ---- historical simulation on the current book ----------------------- #
    panel = _as_frame(returns)
    missing = [lab for lab in labels if lab not in {str(c) for c in panel.columns}]
    if missing:
        raise KeyError(f"returns is missing covariance assets: {missing}")
    aligned = panel.set_axis(panel.columns.astype(str), axis=1)[labels].dropna(how="any")
    port_returns = pd.Series(aligned.to_numpy(dtype=float) @ w, index=aligned.index)

    var_block: dict[str, float] = {}
    for c in confidences:
        tag = f"{int(round(c * 100))}"
        for h in horizons:
            var_block[f"parametric_{tag}_{h}d"] = parametric_var(w, S, confidence=c, horizon=h)
        hist = historical_var(port_returns, confidence=c)
        es = expected_shortfall(port_returns, confidence=c)
        var_block[f"historical_{tag}"] = hist
        var_block[f"expected_shortfall_{tag}"] = es
        param_1d = var_block.get(f"parametric_{tag}_{min(horizons)}d")
        if param_1d and param_1d > 0:
            var_block[f"tail_ratio_{tag}"] = float(hist / param_1d)

    # ---- Euler risk decomposition ---------------------------------------- #
    if sigma > 0:
        marginal = (S @ w) / sigma          # d sigma / d w_i
        contribution = w * marginal         # sums to sigma exactly
        contribution_pct = 100.0 * contribution / sigma
    else:
        marginal = np.zeros_like(w)
        contribution = np.zeros_like(w)
        contribution_pct = np.zeros_like(w)

    report: dict[str, Any] = {
        "assets": labels,
        "n_assets": len(labels),
        "n_obs": int(len(port_returns)),
        "covariance": {
            "method": cov.attrs.get("method") if isinstance(cov, pd.DataFrame) else None,
            "shrinkage": cov.attrs.get("shrinkage") if isinstance(cov, pd.DataFrame) else None,
            "n_obs": cov.attrs.get("n_obs") if isinstance(cov, pd.DataFrame) else None,
            "min_eigenvalue": cov.attrs.get("min_eigenvalue") if isinstance(cov, pd.DataFrame) else None,
            "condition_number": cov.attrs.get("condition_number") if isinstance(cov, pd.DataFrame) else None,
            "sample_start": str(cov.attrs.get("sample_start")) if isinstance(cov, pd.DataFrame) else None,
            "sample_end": str(cov.attrs.get("sample_end")) if isinstance(cov, pd.DataFrame) else None,
        },
        "exposure": {
            "weights": {lab: float(v) for lab, v in zip(labels, w)},
            "gross": gross,
            "net": net,
            "concentration_hhi": hhi,
        },
        "volatility": {
            "per_period": sigma,
            "annualised": float(sigma * np.sqrt(periods_per_year)),
            "realised_per_period": float(port_returns.std(ddof=1)) if len(port_returns) > 1 else float("nan"),
            "undiversified": undiversified,
            "diversification_ratio": div_ratio,
        },
        "var": var_block,
        "risk_contribution": {
            "marginal": {lab: float(v) for lab, v in zip(labels, marginal)},
            "component": {lab: float(v) for lab, v in zip(labels, contribution)},
            "percent": {lab: float(v) for lab, v in zip(labels, contribution_pct)},
        },
        "historical": {
            "start": str(port_returns.index[0]) if len(port_returns) else None,
            "end": str(port_returns.index[-1]) if len(port_returns) else None,
            "worst": float(port_returns.min()) if len(port_returns) else float("nan"),
            "best": float(port_returns.max()) if len(port_returns) else float("nan"),
            "mean": float(port_returns.mean()) if len(port_returns) else float("nan"),
            "skew": float(port_returns.skew()) if len(port_returns) > 2 else float("nan"),
            "excess_kurtosis": float(port_returns.kurt()) if len(port_returns) > 3 else float("nan"),
        },
    }

    if positions_dv01:
        table = stress_table(positions_dv01)
        dv01_values = np.array([float(v) for v in positions_dv01.values()], dtype=float)
        report["dv01"] = {
            "by_tenor": {str(k): float(v) for k, v in positions_dv01.items()},
            "gross": float(np.abs(dv01_values).sum()),
            "net": float(dv01_values.sum()),
        }
        report["stress"] = {
            "pnl": {str(k): float(v) for k, v in table["pnl"].items()},
            "worst_scenario": str(table.index[0]),
            "worst_pnl": float(table["pnl"].iloc[0]),
            "best_scenario": str(table.index[-1]),
            "best_pnl": float(table["pnl"].iloc[-1]),
        }

    return report
