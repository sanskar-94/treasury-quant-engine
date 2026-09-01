"""Term-premium decomposition - separating an expectation from a payment.

A ten-year yield is not a view on ten-year bonds. It is two economically
different things added together::

    y_t^(10)  =  average short rate the market expects over the next ten years
              +  term premium for bearing ten years of duration risk

Only the second term is *compensation*. The first is an arithmetic average of
expected policy rates, and owning duration because the curve is steep for
expectational reasons is not a trade - it is a bet against the market's own
forecast of the Fed, which is a much harder thing to be right about.

This project established that its return model has no directional skill: the
factor attribution earns on slope and curvature and loses on level, while
carrying 42% of its gross risk in level. "Should I own duration at all" is the
question that failed, and the answer is not another daily return forecast. It is
a decomposition: how much of today's long yield is payment for risk rather than
an average of expected short rates. That number is a *level*, it is persistent,
and it is the natural input to a duration-timing signal.

The estimator
-------------
Adrian, Crump & Moench (2013) price the curve with a handful of PCA factors, a
VAR for their physical dynamics, and a market price of risk estimated from
excess-return regressions. Implemented here is the tractable core of that:

1. PCA the zero-curve **levels** into ``n_factors`` pricing factors ``X_t``.
2. Fit a VAR(``lags``) to ``X_t`` - the physical (P-measure) dynamics.
3. Iterate the VAR forward, mapping the state back to the short rate at every
   step, and average. That average is the *risk-neutral* (expectations) yield:
   what the yield would be if investors demanded nothing for duration risk. ACM
   compute exactly this object by setting the price of risk to zero, at which
   point the Q dynamics collapse onto the P dynamics.
4. ``term_premium = observed_yield - expected_average_short_rate``.

What is deliberately left out, and what it costs
------------------------------------------------
* **The price-of-risk regression.** Full ACM estimates ``lambda_0, lambda_1``
  from excess holding-period returns and prices yields with the affine
  recursion, so the term premium comes out of a no-arbitrage model. Here the
  observed yield is taken as given, which means any cross-sectional pricing
  error lands in the term premium rather than in a residual. The size of that
  error is reported: :attr:`TermPremiumResult.r2` is the fit of the factor
  reconstruction to the observed curve, and it is 0.9999 on real data, so the
  channel exists but is worth a fraction of a basis point.
* **The convexity (Jensen) adjustment.** The average of expected short rates is
  not exactly the yield of the expectations-hypothesis bond; for a ten-year
  point the difference is of order 10bp and grows with the square of maturity.
  Omitting it biases the long-end premium *up* by that amount, uniformly enough
  that it does not move the time-series shape at all.
* **Compounding.** Observed zeros are semi-annually compounded; the expectation
  is an arithmetic average of annualised short rates. Same order of error as the
  convexity term, and in the same direction.

**The estimation window is the single most consequential choice here, so it is
stated rather than buried.** Long-horizon expectations are dominated by where
the model thinks the short rate ends up, and with the PCA demeaning the curve by
its own estimation-window mean, that anchor *is* the average curve over the
window. A five-year rolling window in 2015 anchors the ten-year expectation to
five years of zero rates and therefore reports a large positive premium; an
expanding window anchors it to the 1990-2015 average and reports a compressed
one. Both are causal and neither is wrong - they are different statements about
what an investor in 2015 believed the long run looked like. Measured on the real
curve, the difference in the 2012-2021 mean 10y premium is roughly 200bp
(``expanding=True`` is the ACM-like configuration).

Causality
---------
Every parameter used at date ``t`` - PCA loadings, the window mean, the VAR
coefficients, the short-rate map - is estimated on a slice ``[start:t]`` whose
upper bound is **exclusive**, the same discipline as
:func:`tqe.curve.dynamic.dns_forecast_history` and
:func:`tqe.curve.pca.rolling_pca_factors`. The curve observed *at* ``t`` does
enter, through the state ``X_t``: a term premium is a decomposition of that
day's yield, not a forecast of it, so the day's own quote is an input by
definition. Nothing dated ``t`` or later touches a parameter. The test suite
corrupts the last third of the sample and asserts that earlier estimates are
bit-identical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .dynamic import VARModel, fit_var
from .pca import CurvePCA, fit_curve_pca

log = get_logger("curve.term_premium")

__all__ = [
    "TermPremiumResult",
    "decompose_term_premium",
    "term_premium_signal",
]

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TermPremiumResult:
    """The decomposition, per date and per tenor.

    Attributes
    ----------
    expected_short_rate:
        ``date x tenor``. The average short rate the fitted VAR expects over
        each tenor's horizon, in decimals. This is the expectations (risk
        neutral) component of the yield.
    term_premium:
        ``date x tenor``. ``observed_yield - expected_short_rate``, in decimals.
        Positive means duration is being paid for.
    fitted_yield:
        ``date x tenor``. The curve rebuilt from the ``n_factors`` pricing
        factors using causal loadings. Not used in the decomposition - it is the
        diagnostic that says whether the factors actually span the curve.
    r2:
        Pooled :math:`R^2` of ``fitted_yield`` against the observed curve over
        every date and tenor where both exist. A low value means the factor
        space is too small and the premium is absorbing pricing error.
    n_factors:
        Number of pricing factors actually used (capped at the tenor count).
    """

    expected_short_rate: pd.DataFrame
    term_premium: pd.DataFrame
    fitted_yield: pd.DataFrame
    r2: float
    n_factors: int

    @property
    def observed_yield(self) -> pd.DataFrame:
        """Reconstruct the input curve from the two components.

        The decomposition is exact by construction - the premium is defined as a
        residual - so this returns the observed zeros to machine precision. It
        exists so callers can assert that rather than assume it.
        """
        return self.expected_short_rate + self.term_premium

    def annual_means_bp(self, tenors: Sequence[str] | str | None = None) -> pd.DataFrame:
        """Calendar-year mean term premium in basis points.

        The natural resolution for looking at this series: a term premium is a
        slow-moving compensation level, and daily prints of it are noise around
        a story told in years.
        """
        frame = self.term_premium
        if tenors is not None:
            cols = [tenors] if isinstance(tenors, str) else list(tenors)
            frame = frame[cols]
        return frame.groupby(frame.index.year).mean() * 1e4

    def summary(self) -> str:
        tp = self.term_premium.dropna(how="all")
        if tp.empty:
            return f"TermPremiumResult(empty, n_factors={self.n_factors})"
        span = f"{tp.index[0]:%Y-%m-%d}..{tp.index[-1]:%Y-%m-%d}"
        means = ", ".join(f"{c}={v * 1e4:+.0f}bp" for c, v in tp.mean().items())
        return (
            f"TermPremiumResult({span}, n={len(tp)}, factors={self.n_factors}, "
            f"R2={self.r2:.6f}; mean premium {means})"
        )


@dataclass(frozen=True)
class _FactorState:
    """Everything fitted on one training window, held together deliberately.

    The VAR coefficients are only meaningful in the coordinate system of the
    loadings they were fitted through, so the PCA, the VAR and the short-rate
    map are refitted and carried as one object. Keeping them in separate
    variables is how a stale-loadings bug gets written.
    """

    pca: CurvePCA
    var: VARModel
    delta0: float
    delta1: np.ndarray
    gain: np.ndarray        # (n_tenors, k * lags) - state -> expected average short rate
    offset: np.ndarray      # (n_tenors,)
    rho: float              # largest companion eigenvalue BEFORE any shrinkage


# --------------------------------------------------------------------------- #
# VAR algebra
# --------------------------------------------------------------------------- #
def _companion(model: VARModel) -> tuple[np.ndarray, np.ndarray]:
    """Companion form ``Z_{t+1} = c + A Z_t`` of a VAR(p).

    Stacking the lags into one first-order system is what makes the horizon
    algebra below a geometric series instead of a recursion special-cased on
    ``p``.
    """
    k, p = model.k, model.lags
    A = np.zeros((k * p, k * p))
    A[:k] = np.hstack([model.coefs[i] for i in range(p)])
    if p > 1:
        A[k:, :-k] = np.eye(k * (p - 1))
    c = np.zeros(k * p)
    c[:k] = model.intercept
    return A, c


def _shrink_roots(model: VARModel, max_eigenvalue: float) -> tuple[VARModel, float]:
    """Impose stationarity by scaling the autoregressive roots.

    Curve levels are near unit-root processes, and five years of daily data
    cannot tell 0.9990 from 1.0005. OLS routinely lands on the wrong side: a
    largest root of 1.0005 iterated over a ten-year horizon (2,520 steps)
    multiplies the state by ``e^{1.26}``, and the "expected short rate" comes out
    at several hundred percent. Stationarity therefore has to be imposed rather
    than hoped for.

    Replacing ``A_l`` with ``kappa^l A_l`` scales every root of the
    characteristic polynomial by exactly ``kappa`` - the standard root-shrinkage
    device, and exact for any lag order rather than approximate. The intercept is
    left alone: the factors are demeaned by their estimation window (the PCA
    subtracts that window's mean curve), so the model's unconditional mean is
    already the window average and shrinking the roots does not move it.

    Note the direction of the bias this introduces. Bauer, Rudebusch & Wu (2012)
    show OLS *understates* persistence in exactly this setting, so a cap pushes
    further the way the small-sample bias already points: expectations revert to
    the window mean faster than they should, which makes the estimated premium
    more variable, not less. That is a cost paid knowingly to keep the iteration
    finite.

    Returns
    -------
    (model, rho):
        The possibly-shrunk model and the largest root **before** shrinkage.
    """
    A, _ = _companion(model)
    try:
        rho = float(np.max(np.abs(np.linalg.eigvals(A)))) if A.size else 0.0
    except np.linalg.LinAlgError:
        rho = float("inf")
    if not np.isfinite(rho):
        raise np.linalg.LinAlgError("VAR companion eigenvalues did not converge")
    if rho <= max_eigenvalue or rho <= EPS:
        return model, rho
    kappa = max_eigenvalue / rho
    scaled = np.stack([model.coefs[i] * kappa ** (i + 1) for i in range(model.lags)])
    return replace(model, coefs=scaled), rho


def _horizon_average_map(
    model: VARModel,
    delta0: float,
    delta1: np.ndarray,
    horizons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Affine map from today's state to the expected average short rate.

    With the short rate affine in the factors, ``r_t = delta0 + delta1' X_t``,
    the expected average over the next ``N`` steps is itself affine in the
    current state::

        (1/N) sum_{i=0}^{N-1} E_t[r_{t+i}] = offset_N + gain_N . Z_t

    Iterating the VAR ``N`` times per date per tenor would be 9,000 x 8 x 2,520
    matrix products. It is unnecessary: ``sum_{i<N} A^i`` is a matrix geometric
    series with the closed form ``(I - A)^{-1}(I - A^N)``, which needs
    ``log2(N)`` products via binary exponentiation. ``(I - A)`` is invertible
    because :func:`_shrink_roots` has already guaranteed a spectral radius below
    one. The step-by-step iteration is kept in the test suite as the reference
    the closed form is checked against.

    The average runs from ``i = 0``, so it includes today's short rate - the
    yield of a bond bought today covers ``[t, t+T)``, not ``(t, t+T]``.

    Parameters
    ----------
    model:
        A stationary VAR in the factor space.
    delta0, delta1:
        Intercept and loadings of the short rate on the factors.
    horizons:
        Number of VAR steps per tenor, ``(n_tenors,)``.

    Returns
    -------
    (gain, offset):
        ``(n_tenors, k*lags)`` and ``(n_tenors,)``.
    """
    A, c = _companion(model)
    kp = A.shape[0]
    k = model.k
    eye = np.eye(kp)
    resolvent = eye - A

    d1 = np.zeros(kp)
    d1[:k] = np.asarray(delta1, dtype=float)

    gain = np.zeros((len(horizons), kp))
    offset = np.zeros(len(horizons))
    # Horizons repeat across tenors only rarely, but the solve is the expensive
    # part and caching it costs one dict.
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for j, n_steps in enumerate(horizons):
        n = int(n_steps)
        if n not in cache:
            # S = sum_{i=0}^{N-1} A^i ; B = sum_{i=0}^{N-1} (I - A)^{-1}(I - A^i) c
            s_mat = np.linalg.solve(resolvent, eye - np.linalg.matrix_power(A, n))
            b_vec = np.linalg.solve(resolvent, (n * eye - s_mat) @ c)
            cache[n] = (s_mat, b_vec)
        s_mat, b_vec = cache[n]
        gain[j] = (d1 @ s_mat) / n
        offset[j] = delta0 + float(d1 @ b_vec) / n
    return gain, offset


def _short_rate_map(factors: np.ndarray, short_rate: np.ndarray) -> tuple[float, np.ndarray]:
    """OLS projection of the short rate onto the pricing factors.

    ACM's ``r_t = delta_0 + delta_1' X_t``. With the short end of the curve in
    the PCA input this is nearly an identity - the factor loadings already
    reconstruct that tenor - but estimating it rather than reading it off the
    loadings keeps the map honest when the short rate is a tenor the factors
    reproduce imperfectly, and costs one least-squares solve per refit.
    """
    design = np.column_stack([np.ones(len(factors)), factors])
    coef, *_ = np.linalg.lstsq(design, np.asarray(short_rate, dtype=float), rcond=None)
    return float(coef[0]), coef[1:]


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #
def _fit_state(
    train: pd.DataFrame,
    n_factors: int,
    lags: int,
    short_idx: int,
    horizons: np.ndarray,
    max_eigenvalue: float,
) -> _FactorState | None:
    """Fit one window. Returns ``None`` if the window is unusable."""
    if len(train) < max(n_factors + 2, lags + 5):
        return None
    try:
        pca = fit_curve_pca(train, n_factors=n_factors)
    except ValueError:
        return None

    # PCA subtracts this window's mean curve, so the factor series it produces is
    # exactly demeaned and the VAR's long-run anchor is the window average.
    factors = pca.transform(train)
    try:
        var = fit_var(factors, lags=lags)
    except (ValueError, np.linalg.LinAlgError):
        return None

    var, rho = _shrink_roots(var, max_eigenvalue)
    delta0, delta1 = _short_rate_map(factors, train.to_numpy(dtype=float)[:, short_idx])
    gain, offset = _horizon_average_map(var, delta0, delta1, horizons)
    return _FactorState(pca=pca, var=var, delta0=delta0, delta1=delta1,
                        gain=gain, offset=offset, rho=rho)


def decompose_term_premium(
    zero_curve: pd.DataFrame,
    n_factors: int = 5,
    lags: int = 1,
    window: int = 1260,
    min_periods: int = 504,
    refit_every: int = 63,
    *,
    tenor_years: Mapping[str, float] | None = None,
    short_rate_tenor: str | None = None,
    steps_per_year: int = 252,
    max_eigenvalue: float = 0.999,
    expanding: bool = False,
) -> TermPremiumResult:
    """Decompose a zero curve into expected short rates and a term premium.

    Parameters
    ----------
    zero_curve:
        ``date x tenor`` **zero** (spot) rates in decimals, e.g. the output of
        :func:`tqe.curve.bootstrap.bootstrap_history`. Par yields would work
        mechanically but are the wrong object: a par yield is a coupon, not the
        return on a single cashflow, so it cannot be compared with an average of
        expected short rates. Rows with any missing tenor are dropped, so pass a
        set of tenors with common coverage - mixing the 1-month (starts 2001)
        with the 30-year (gap 2002-2006) discards most of the sample.
    n_factors:
        Pricing factors extracted from the curve levels. ACM use five; three
        captures level/slope/curvature and five picks up the pieces of the short
        end that matter for the expectations component. Capped at the number of
        tenors supplied.
    lags:
        VAR order on the factors.
    window:
        Trailing observations per refit. Read the module docstring before
        changing this - it sets the long-run anchor and therefore the level of
        the estimated premium.
    min_periods:
        Observations required before the first estimate.
    refit_every:
        Refit cadence in observations. Loadings and VAR coefficients on series
        this persistent barely move between refits; a quarterly cadence is 63x
        cheaper than daily and changes the estimate in the third decimal of a
        basis point.
    tenor_years:
        Tenor label to maturity in years. Defaults to
        :data:`tqe.data.sources.TENOR_YEARS`.
    short_rate_tenor:
        Which tenor stands in for the instantaneous short rate. Defaults to the
        shortest column supplied. The 3-month bill is the usual empirical proxy;
        no observable instantaneous rate exists, and the choice is worth a few
        basis points at the long end.
    steps_per_year:
        VAR steps per year, i.e. the observation frequency of ``zero_curve``.
        252 for daily data.
    max_eigenvalue:
        Stationarity cap on the largest autoregressive root. See
        :func:`_shrink_roots`; 0.999 daily is a half-life of 2.8 years for the
        most persistent factor.
    expanding:
        Use all history before ``t`` instead of a trailing ``window``. This is
        the ACM-like configuration: the long-run anchor becomes the whole
        available past rather than the last five years.

    Returns
    -------
    TermPremiumResult

    Notes
    -----
    **Causality.** For each block of ``refit_every`` dates starting at ``t``, the
    PCA, the VAR and the short-rate map are estimated on ``clean.iloc[start:t]``
    - the slice bound is exclusive, and that one line is what makes the whole
    function causal. Those parameters are then applied to dates ``t`` onward
    until the next refit, so a parameter used on date ``d >= t`` was estimated
    strictly before ``t <= d``. The state ``X_d`` itself is the curve observed on
    ``d``, which is not look-ahead: the term premium is a decomposition of that
    day's yield rather than a forecast of it.
    """
    if n_factors < 1:
        raise ValueError("n_factors must be >= 1")
    if lags < 1:
        raise ValueError("lags must be >= 1")
    if min_periods <= lags:
        raise ValueError("min_periods must exceed lags")
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")
    if steps_per_year < 1:
        raise ValueError("steps_per_year must be >= 1")
    if not 0.0 < max_eigenvalue < 1.0:
        raise ValueError("max_eigenvalue must lie strictly inside the unit circle")

    if tenor_years is None:
        from ..data.sources import TENOR_YEARS

        tenor_years = TENOR_YEARS

    cols = [c for c in zero_curve.columns if c in tenor_years]
    if len(cols) < 2:
        raise ValueError("Need at least two recognised tenor columns to decompose a curve")
    cols.sort(key=lambda c: float(tenor_years[c]))
    years = np.array([float(tenor_years[c]) for c in cols], dtype=float)

    clean = zero_curve[cols].dropna(how="any")
    if clean.empty:
        raise ValueError("No complete rows in the supplied zero curve")
    dropped = 1.0 - len(clean) / max(len(zero_curve), 1)
    if dropped > 0.10:
        log.warning(
            "term premium: %.0f%% of rows dropped for missing tenors; "
            "consider passing a narrower, better-covered tenor set", dropped * 100
        )

    if short_rate_tenor is None:
        short_idx = 0
    elif short_rate_tenor in cols:
        short_idx = cols.index(short_rate_tenor)
    else:
        raise KeyError(f"short_rate_tenor {short_rate_tenor!r} is not among {cols}")

    n_factors = int(min(n_factors, len(cols)))
    # A tenor's horizon in VAR steps. `max(1, ...)` keeps a sub-daily tenor from
    # asking for a zero-step average.
    horizons = np.maximum(1, np.rint(years * steps_per_year)).astype(int)

    values = clean.to_numpy(dtype=float)
    n, m = values.shape
    expected = np.full((n, m), np.nan)
    fitted = np.full((n, m), np.nan)

    state: _FactorState | None = None
    n_shrunk = 0
    n_blocks = 0
    t = int(min_periods)
    while t < n:
        block_end = min(t + refit_every, n)
        start = 0 if expanding else max(0, t - window)
        # `t` is EXCLUSIVE - the causality bound for every parameter below.
        candidate = _fit_state(clean.iloc[start:t], n_factors, lags, short_idx,
                               horizons, max_eigenvalue)
        if candidate is not None:
            state = candidate
            n_blocks += 1
            n_shrunk += int(candidate.rho > max_eigenvalue)
        if state is None:
            t = block_end
            continue

        # Only rows this block needs are transformed: the lag history it must
        # stack, and the block itself. Nothing dated after `block_end` is touched.
        first = max(0, t - lags + 1)
        rows = values[first:block_end]
        block_factors = (rows - state.pca.mean_) @ state.pca.components_.T
        pos = np.arange(t - first, block_end - first)
        stacked = np.hstack([block_factors[pos - j] for j in range(lags)])

        expected[t:block_end] = stacked @ state.gain.T + state.offset
        fitted[t:block_end] = block_factors[pos] @ state.pca.components_ + state.pca.mean_
        t = block_end

    if n_blocks:
        log.info(
            "term premium: %d refits, %d (%.0f%%) required root shrinkage to stay stationary",
            n_blocks, n_shrunk, 100.0 * n_shrunk / n_blocks,
        )

    mask = np.isfinite(fitted) & np.isfinite(values)
    if mask.any():
        resid = values[mask] - fitted[mask]
        obs = values[mask]
        sst = float(((obs - obs.mean()) ** 2).sum())
        r2 = float(1.0 - (resid**2).sum() / sst) if sst > EPS else float("nan")
    else:
        r2 = float("nan")

    index = zero_curve.index
    exp_frame = pd.DataFrame(expected, index=clean.index, columns=cols).reindex(index)
    fit_frame = pd.DataFrame(fitted, index=clean.index, columns=cols).reindex(index)
    obs_frame = pd.DataFrame(values, index=clean.index, columns=cols).reindex(index)
    for frame in (exp_frame, fit_frame, obs_frame):
        frame.index.name = index.name or "date"

    return TermPremiumResult(
        expected_short_rate=exp_frame,
        term_premium=obs_frame - exp_frame,
        fitted_yield=fit_frame,
        r2=r2,
        n_factors=n_factors,
    )


def term_premium_signal(
    tp: TermPremiumResult | pd.DataFrame,
    window: int = 252,
    *,
    min_periods: int | None = None,
    clip: float | None = None,
) -> pd.DataFrame:
    """Trailing z-score of the term premium - is duration well paid *right now*?

    A term premium of 80bp means nothing on its own; the answer depends on
    whether 80bp is generous or stingy by the standard of the recent past. The
    z-score is the standard way to ask that, and it is the form the sizing layer
    already expects (:mod:`tqe.signals.sizing`). Positive means duration is
    better compensated than it has lately been, which is the case for owning it.

    Note that demeaning is the *right* transform here, unlike for a return
    forecast where it destroys the meaningful zero point (see
    :func:`tqe.signals.alpha.predictions_to_signal`). A term premium has no
    natural zero to preserve - the level of compensation the market demands
    drifts across decades with inflation risk and the supply of duration - so
    what is tradable is the deviation from its own recent level.

    Parameters
    ----------
    tp:
        A :class:`TermPremiumResult` or a ``date x tenor`` premium frame.
    window:
        Trailing window for the mean and standard deviation.
    min_periods:
        Minimum observations before a value is emitted. Defaults to
        ``max(20, window // 4)``.
    clip:
        Optional absolute bound. Left off by default so this returns a plain
        z-score; bound it in the sizing layer where the risk budget lives.

    Returns
    -------
    pd.DataFrame
        Same shape and index as the input premium; NaN through the warm-up.

    Notes
    -----
    **Causality.** ``rolling`` looks strictly backwards, so the statistics at
    ``t`` use ``[t-window+1, t]``. Including ``t`` itself is legitimate here and
    would not be for a forecast: the term premium at ``t`` is an estimate of a
    state observable at ``t``, so standardising it against its own history uses
    nothing that was not published by that day's close.
    """
    frame = tp.term_premium if isinstance(tp, TermPremiumResult) else tp
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("tp must be a TermPremiumResult or a DataFrame")
    if window < 2:
        raise ValueError("window must be >= 2")
    if frame.empty:
        return frame.copy()

    mp = int(min_periods or max(20, window // 4))
    mu = frame.rolling(window, min_periods=mp).mean()
    sd = frame.rolling(window, min_periods=mp).std()
    sd = sd.where(sd.abs() > EPS)
    signal = (frame - mu) / sd
    if clip:
        signal = signal.clip(-abs(clip), abs(clip))
    return signal
