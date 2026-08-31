"""Forecast-quality and performance metrics, including the honesty metrics.

Three families live here.

**Forecast metrics** (:func:`regression_metrics`, :func:`information_coefficient`,
:func:`rank_information_coefficient`) answer "is the model's signal real?".  For
daily fixed-income returns, R-squared is close to useless -- a genuinely
profitable daily forecast explains well under 1% of variance, and an
out-of-sample R-squared of *exactly* zero is what a coin flip produces.  The
information coefficient is the number that matters: an IC of 0.03-0.05 sustained
out of sample is a real, tradeable edge (Grinold's fundamental law:
``IR ~ IC * sqrt(breadth)``).  Any daily IC above ~0.15 on a liquid rates
strategy is a bug hunt, not a discovery.

**Performance metrics** (:func:`performance_metrics`, :func:`drawdown_series`)
turn a return stream into the numbers on a tearsheet.

**Multiple-testing metrics** (:func:`probabilistic_sharpe_ratio`,
:func:`deflated_sharpe_ratio`, :func:`minimum_track_record_length`).  These are
the most important functions in this file.  A backtest that searched 200
configurations and reports the best one's Sharpe is reporting the maximum of 200
draws, not an estimate of skill; under the null of zero true skill the *expected*
maximum Sharpe of 200 independent 5-year backtests is around 1.0.  The deflated
Sharpe ratio prices that selection bias in and returns the probability that the
observed Sharpe reflects genuine skill.  If a strategy in this repo cannot show a
DSR above 0.95, it does not get traded.

References
----------
.. [1] Bailey, D. H. and Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient
       Frontier." *Journal of Risk*, 15(2), 3-44.  (Probabilistic Sharpe Ratio,
       minimum track record length.)
.. [2] Bailey, D. H. and Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio:
       Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
       *Journal of Portfolio Management*, 40(5), 94-107.
.. [3] Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*,
       chapter 8 and appendix.

No look-ahead
-------------
Every function here is a pure summary of an *already realised* series: it is
called after the walk-forward loop, never inside a feature.  The only ordering
assumption is that ``returns``/``equity`` are in chronological order, which
:func:`drawdown_series` relies on (a running maximum is causal by construction --
the high-water mark on day ``t`` uses days ``0..t`` only).  These functions must
never be used to build a feature; a rolling Sharpe computed over a window that
includes day ``t`` and then used to trade day ``t`` is textbook leakage.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..logging_utils import get_logger

log = get_logger("training.metrics")

__all__ = [
    "EULER_MASCHERONI",
    "regression_metrics",
    "information_coefficient",
    "rank_information_coefficient",
    "performance_metrics",
    "drawdown_series",
    "equity_curve",
    "expected_maximum_sharpe",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "minimum_track_record_length",
]

#: Euler-Mascheroni constant, used in the expected-maximum-of-N-Gaussians
#: approximation that underpins the deflated Sharpe ratio.
EULER_MASCHERONI: float = 0.5772156649015329

_TINY = 1e-15


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_1d(x: Any) -> np.ndarray:
    """Coerce anything array-like to a flat float array."""
    if isinstance(x, (pd.Series, pd.Index)):
        arr = x.to_numpy(dtype=float, copy=False)
    elif isinstance(x, pd.DataFrame):
        arr = x.to_numpy(dtype=float, copy=False).ravel()
    else:
        arr = np.asarray(x, dtype=float).ravel()
    return arr


def _aligned(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    """Flatten both inputs and drop positions where either is not finite.

    Pairwise deletion is the right call here: a NaN prediction means the model
    abstained that day, and an abstention is neither a hit nor a miss.  Silently
    filling it with zero would flatter directional accuracy.
    """
    a, b = _as_1d(y_true), _as_1d(y_pred)
    if a.shape != b.shape:
        raise ValueError(f"y_true and y_pred have different shapes: {a.shape} vs {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation with an explicit zero-variance guard.

    A constant prediction (a model that always says "+2bp") has no correlation
    with anything -- the coefficient is 0/0.  Returning NaN says "undefined";
    returning 0.0 would falsely claim we measured no skill.
    """
    if a.size < 2:
        return float("nan")
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = math.sqrt(float(a_c @ a_c) * float(b_c @ b_c))
    if denom <= _TINY:
        return float("nan")
    return float(a_c @ b_c / denom)


def _safe_div(num: float, den: float) -> float:
    """Divide, returning NaN rather than raising or returning +/-inf."""
    if not np.isfinite(den) or abs(den) <= _TINY:
        return float("nan")
    return float(num / den)


def _rf_per_period(rf: float, periods: int) -> float:
    """Convert an annual risk-free rate to a per-period rate, geometrically.

    ``(1 + rf)**(1/periods) - 1`` rather than ``rf / periods``: compounding a
    simple division back up overstates the annual rate by ~2bp at 5% and 252
    periods.  Small, but this is a rates shop.
    """
    if rf == 0.0:
        return 0.0
    return float((1.0 + rf) ** (1.0 / periods) - 1.0)


# --------------------------------------------------------------------------- #
# Forecast quality
# --------------------------------------------------------------------------- #
def information_coefficient(y_true: Any, y_pred: Any) -> float:
    """Pearson correlation between forecast and realised outcome.

    Parameters
    ----------
    y_true, y_pred : array-like
        Realised and predicted values.  Shapes must match; NaNs are dropped
        pairwise.

    Returns
    -------
    float
        Correlation in [-1, 1], or NaN if either series is constant or fewer
        than two usable observations remain.

    Notes
    -----
    The IC is the workhorse statistic of quantitative forecasting.  Its standard
    error is roughly ``1/sqrt(n)``, so over 2,000 out-of-sample days an IC needs
    to exceed ~0.045 to be two standard errors from zero.  Pearson IC is
    sensitive to outliers -- one 1994-style day can manufacture or destroy it --
    which is why :func:`rank_information_coefficient` is always reported
    alongside.  A large gap between the two is diagnostic: Pearson >> Spearman
    means the "edge" lives in a handful of extreme days.
    """
    a, b = _aligned(y_true, y_pred)
    return _pearson(a, b)


def rank_information_coefficient(y_true: Any, y_pred: Any) -> float:
    """Spearman (rank) correlation between forecast and realised outcome.

    Returns
    -------
    float
        Rank correlation in [-1, 1], NaN when undefined.

    Notes
    -----
    Computed as the Pearson correlation of average ranks, which is exactly
    Spearman's rho with the standard tie correction.  Rank IC answers the
    question a portfolio actually cares about -- "did the model order the
    opportunities correctly?" -- and is immune to the fat tails that make daily
    Treasury returns a poor fit for Pearson.
    """
    a, b = _aligned(y_true, y_pred)
    if a.size < 2:
        return float("nan")
    ra = stats.rankdata(a)
    rb = stats.rankdata(b)
    return _pearson(ra, rb)


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Standard point-forecast diagnostics for a return/yield-change model.

    Parameters
    ----------
    y_true, y_pred : array-like
        Realised and predicted values, same shape.  2-D inputs (dates x tenors)
        are flattened, i.e. pooled across tenors.

    Returns
    -------
    dict
        ``rmse``, ``mae``, ``r2``, ``directional_accuracy``, ``ic``, ``rank_ic``,
        plus ``n_obs`` and ``bias`` (mean signed error).

    Notes
    -----
    ``r2`` is the out-of-sample R-squared against the *realised* mean
    (sklearn's definition), so it can and routinely does go **negative** on daily
    data -- that simply means the model beat nothing.  Do not tune on it.

    ``directional_accuracy`` is measured only over days where the outcome was
    non-zero (a flat market has no direction to get right), and a prediction of
    exactly zero counts as a miss because it expresses no view.  For a daily
    duration signal, 51-53% is the realistic range; 60% means look-ahead has
    crept in somewhere.
    """
    a, b = _aligned(y_true, y_pred)
    n = int(a.size)
    if n == 0:
        return {
            "n_obs": 0.0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "directional_accuracy": float("nan"),
            "ic": float("nan"),
            "rank_ic": float("nan"),
            "bias": float("nan"),
        }

    err = b - a
    sse = float(err @ err)
    demeaned = a - a.mean()
    sst = float(demeaned @ demeaned)

    nz = a != 0.0
    if nz.any():
        directional = float(np.mean(np.sign(a[nz]) == np.sign(b[nz])))
    else:
        directional = float("nan")

    return {
        "n_obs": float(n),
        "rmse": float(math.sqrt(sse / n)),
        "mae": float(np.mean(np.abs(err))),
        "r2": 1.0 - _safe_div(sse, sst) if sst > _TINY else float("nan"),
        "directional_accuracy": directional,
        "ic": _pearson(a, b),
        "rank_ic": rank_information_coefficient(a, b),
        "bias": float(err.mean()),
    }


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #
def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Compound simple returns into an equity path.

    Notes
    -----
    Geometric compounding, not a cumulative sum: the backtest reinvests, and the
    difference over a decade is not cosmetic.  NaNs are treated as flat days.
    """
    r = pd.Series(returns, dtype=float).fillna(0.0)
    return initial * (1.0 + r).cumprod()


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running high-water mark.

    Parameters
    ----------
    equity : pd.Series
        Equity or NAV path in chronological order.

    Returns
    -------
    pd.Series
        ``equity / cummax(equity) - 1``: zero at every new high, negative below.

    Notes
    -----
    Causal by construction -- the high-water mark at ``t`` is the maximum over
    ``0..t`` only, which is precisely how an investor experiences a drawdown in
    real time.  Non-positive equity (a fully wiped account) makes the ratio
    meaningless, so those points return NaN rather than a nonsense number.
    """
    eq = pd.Series(equity, dtype=float)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return dd.where(peak > 0.0).rename("drawdown")


def _max_drawdown_stats(eq: np.ndarray) -> tuple[float, int]:
    """``(max_drawdown, longest_underwater_run)`` from an equity array."""
    if eq.size == 0:
        return float("nan"), 0
    peak = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0.0, eq / peak - 1.0, np.nan)
    max_dd = float(np.nanmin(dd)) if np.isfinite(dd).any() else float("nan")

    # Underwater = strictly below the prior high-water mark, with a relative
    # tolerance so float noise at a new high does not open a phantom drawdown.
    under = eq < peak - np.abs(peak) * 1e-12
    if not under.any():
        return max_dd, 0
    # Run-length encode the boolean mask in one pass rather than looping.
    padded = np.concatenate(([0], under.view(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    runs = edges[1::2] - edges[0::2]
    return max_dd, int(runs.max())


def performance_metrics(
    returns: pd.Series,
    rf: float = 0.0,
    periods: int = 252,
    turnover: pd.Series | None = None,
) -> dict[str, float]:
    """Full tearsheet statistics for a stream of periodic returns.

    Parameters
    ----------
    returns : pd.Series
        Simple (arithmetic) periodic returns as decimals, chronological order.
        NaNs are dropped -- a day the strategy did not run is not a zero return.
    rf : float, default 0.0
        **Annual** risk-free rate as a decimal, converted geometrically to a
        per-period rate.  In a rates backtest this matters: a 2023 strategy with
        a 0.5 raw Sharpe had a *negative* excess Sharpe against 5% cash.
    periods : int, default 252
        Periods per year (252 US trading days).
    turnover : pd.Series, optional
        Per-period turnover as a fraction of capital.  When supplied,
        ``ann_turnover`` is its mean times ``periods``; otherwise NaN.

    Returns
    -------
    dict
        ``total_return, cagr, ann_return, ann_vol, downside_vol, sharpe, sortino,
        calmar, max_drawdown, max_dd_duration_days, hit_rate, profit_factor,
        skew, kurtosis, excess_kurtosis, var_95, cvar_95, best_day, worst_day,
        ann_turnover, n_obs``.

    Notes
    -----
    Conventions, stated because every shop's differ:

    * ``sharpe`` uses the **sample** standard deviation (ddof=1) of *excess*
      returns and scales by ``sqrt(periods)``.  The square-root rule assumes
      serially uncorrelated returns; a trend strategy with positive
      autocorrelation has its true Sharpe understated by it, a mean-reversion
      strategy overstated.  This is why the Sharpe is always paired with
      :func:`deflated_sharpe_ratio`.
    * ``kurtosis`` is **non-excess** (Gaussian = 3.0) so it can be passed
      straight into :func:`probabilistic_sharpe_ratio`; ``excess_kurtosis`` is
      the same number minus 3 for readers used to the tearsheet convention.
    * ``var_95``/``cvar_95`` are reported as the (negative) 5th-percentile
      return and the mean of the tail beyond it -- a loss reads as a negative
      number, consistent with ``worst_day``.
    * ``hit_rate`` counts only periods with a non-zero return: days flat because
      the signal was below the trading threshold are neither wins nor losses.
    * Undefined ratios (zero volatility, no losing days, no drawdown) return
      **NaN**, never ``inf`` -- infinities poison JSON reports and downstream
      aggregation, and "undefined" is the honest answer.

    Degenerate input (empty, single observation, all-zero returns) returns the
    same key set with NaNs rather than raising, so a fold that produced no trades
    does not crash the walk-forward loop.
    """
    r = pd.Series(returns, dtype=float).dropna()
    n = int(r.size)

    out: dict[str, float] = {
        "n_obs": float(n),
        "total_return": float("nan"),
        "cagr": float("nan"),
        "ann_return": float("nan"),
        "ann_vol": float("nan"),
        "downside_vol": float("nan"),
        "sharpe": float("nan"),
        "sortino": float("nan"),
        "calmar": float("nan"),
        "max_drawdown": float("nan"),
        "max_dd_duration_days": float("nan"),
        "hit_rate": float("nan"),
        "profit_factor": float("nan"),
        "skew": float("nan"),
        "kurtosis": float("nan"),
        "excess_kurtosis": float("nan"),
        "var_95": float("nan"),
        "cvar_95": float("nan"),
        "best_day": float("nan"),
        "worst_day": float("nan"),
        "ann_turnover": float("nan"),
    }
    if turnover is not None:
        t = pd.Series(turnover, dtype=float).dropna()
        out["ann_turnover"] = float(t.mean() * periods) if t.size else float("nan")

    if n == 0:
        return out

    x = r.to_numpy(dtype=float, copy=False)
    eq = np.cumprod(1.0 + x)
    final = float(eq[-1])
    years = n / float(periods)

    out["total_return"] = final - 1.0
    out["ann_return"] = float(x.mean() * periods)
    out["best_day"] = float(x.max())
    out["worst_day"] = float(x.min())

    # A wiped-out account has no well-defined growth rate; -100% is the truth.
    out["cagr"] = float(final ** (1.0 / years) - 1.0) if final > 0.0 and years > 0 else -1.0

    max_dd, dd_len = _max_drawdown_stats(eq)
    out["max_drawdown"] = max_dd
    out["max_dd_duration_days"] = float(dd_len)

    nz = x[x != 0.0]
    out["hit_rate"] = float(np.mean(nz > 0.0)) if nz.size else float("nan")
    gains = float(x[x > 0.0].sum())
    losses = float(-x[x < 0.0].sum())
    out["profit_factor"] = _safe_div(gains, losses)

    # Historical VaR/CVaR: no distributional assumption, which matters because
    # Treasury return tails are decisively non-Gaussian.
    var95 = float(np.quantile(x, 0.05))
    out["var_95"] = var95
    tail = x[x <= var95]
    out["cvar_95"] = float(tail.mean()) if tail.size else var95

    if n < 2:
        return out

    out["skew"] = float(stats.skew(x, bias=False))
    out["excess_kurtosis"] = float(stats.kurtosis(x, fisher=True, bias=False))
    out["kurtosis"] = out["excess_kurtosis"] + 3.0

    rf_p = _rf_per_period(rf, periods)
    excess = x - rf_p
    vol = float(np.std(excess, ddof=1))
    out["ann_vol"] = float(np.std(x, ddof=1) * math.sqrt(periods))

    if vol > _TINY:
        out["sharpe"] = float(excess.mean() / vol * math.sqrt(periods))
    elif abs(float(excess.mean())) <= _TINY:
        # A genuinely flat strategy earned nothing and risked nothing: 0, not NaN.
        out["sharpe"] = 0.0

    # Sortino: only downside deviations are penalised, and the denominator uses
    # the FULL sample size (not just losing days) -- the standard definition.
    # Dividing by the count of losing days alone inflates Sortino for strategies
    # that lose rarely, which is precisely the population it is used to judge.
    downside = np.minimum(excess, 0.0)
    dd_vol = float(math.sqrt(float(downside @ downside) / n))
    out["downside_vol"] = dd_vol * math.sqrt(periods)
    if dd_vol > _TINY:
        out["sortino"] = float(excess.mean() / dd_vol * math.sqrt(periods))
    elif abs(float(excess.mean())) <= _TINY:
        out["sortino"] = 0.0

    if np.isfinite(max_dd) and abs(max_dd) > _TINY:
        out["calmar"] = float(out["cagr"] / abs(max_dd))

    return out


# --------------------------------------------------------------------------- #
# Multiple-testing / honesty metrics
# --------------------------------------------------------------------------- #
def _deannualize(sharpe: float, periods: int, annualized: bool) -> float:
    """Convert an annualised Sharpe to the per-period Sharpe the maths needs.

    The PSR/DSR algebra is written in units of *one observation*: ``n_obs``
    counts observations, so the Sharpe entering the formula must be the
    per-observation ratio ``mean(r)/sd(r)``.  Feeding an annualised Sharpe
    straight in overstates the test statistic by ``sqrt(252)`` and turns every
    strategy into a certainty.
    """
    if not annualized:
        return float(sharpe)
    return float(sharpe) / math.sqrt(float(periods))


def probabilistic_sharpe_ratio(
    sharpe: float,
    benchmark_sr: float = 0.0,
    n_obs: int = 0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    *,
    periods: int = 252,
    annualized: bool = True,
) -> float:
    r"""Probability that the true Sharpe exceeds ``benchmark_sr`` [1]_.

    .. math::
        \widehat{PSR}(SR^*) = \Phi\left(
            \frac{(\widehat{SR} - SR^*)\sqrt{n-1}}
                 {\sqrt{1 - \hat\gamma_3 \widehat{SR}
                        + \frac{\hat\gamma_4 - 1}{4}\widehat{SR}^2}}\right)

    Parameters
    ----------
    sharpe : float
        Observed Sharpe ratio.  **Annualised by default** -- see ``annualized``.
    benchmark_sr : float, default 0.0
        Threshold Sharpe :math:`SR^*`, in the same annualisation as ``sharpe``.
    n_obs : int
        Number of return observations (days, if the returns are daily).
    skew : float, default 0.0
        Skewness of the return distribution (0 = symmetric).
    kurtosis : float, default 3.0
        **Non-excess** kurtosis (3.0 = Gaussian).  This is the ``kurtosis`` key
        returned by :func:`performance_metrics`.
    periods : int, keyword-only, default 252
        Periods per year, used to de-annualise.
    annualized : bool, keyword-only, default True
        Whether ``sharpe`` and ``benchmark_sr`` are annualised.  The default is
        True because that is what :func:`performance_metrics` produces and what
        every report quotes; pass ``False`` if you already have per-period
        ratios.

    Returns
    -------
    float
        Probability in [0, 1], or NaN if fewer than two observations or the
        estimator variance is not positive.

    Notes
    -----
    The denominator is the standard error of the Sharpe estimator under
    non-normality.  Its two correction terms are economically meaningful:

    * **negative skew raises the denominator** (the ``-skew*SR`` term with
      ``skew < 0``), so a strategy that grinds out small gains and occasionally
      blows up -- selling convexity, carry trades, short-vol -- needs a much
      longer track record to prove itself.  This is the correction that flags
      "picking up nickels in front of a steamroller" *before* the steamroller.
    * **fat tails raise it too**, via ``(kurt-1)/4 * SR^2``.

    A 1.0 annualised Sharpe over one year of daily data is worth far less than
    the same Sharpe over five years, and PSR is how much less.
    """
    n = int(n_obs)
    if n < 2:
        return float("nan")

    sr = _deannualize(sharpe, periods, annualized)
    sr_star = _deannualize(benchmark_sr, periods, annualized)
    if not np.isfinite(sr) or not np.isfinite(sr_star):
        return float("nan")

    variance = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr
    if not np.isfinite(variance) or variance <= _TINY:
        # Only reachable with extreme skew/kurtosis inputs; refuse to guess.
        log.warning("PSR estimator variance non-positive (%.6g); returning NaN", variance)
        return float("nan")

    z = (sr - sr_star) * math.sqrt(n - 1) / math.sqrt(variance)
    return float(stats.norm.cdf(z))


def expected_maximum_sharpe(n_trials: int, sharpe_std: float = 1.0) -> float:
    r"""Expected maximum of ``n_trials`` independent Sharpe estimates [2]_.

    .. math::
        E[\max_N] \approx \sigma\left[(1-\gamma)\,\Phi^{-1}\!\left(1-\tfrac1N\right)
                    + \gamma\,\Phi^{-1}\!\left(1-\tfrac{1}{N e}\right)\right]

    Parameters
    ----------
    n_trials : int
        Number of independent configurations tried.
    sharpe_std : float, default 1.0
        Cross-sectional standard deviation of the trial Sharpes.  With the
        default of 1.0 the result is in units of that standard deviation.

    Returns
    -------
    float
        Expected best-of-N Sharpe under the null of zero true skill.

    Notes
    -----
    This is the extreme-value approximation to the expected maximum of ``N``
    i.i.d. standard normals (:math:`\gamma` is the Euler-Mascheroni constant).
    It grows like :math:`\sqrt{2\ln N}` -- slowly, but relentlessly: 1.54 at
    N=10, 2.51 at N=100, 3.24 at N=1000.  Multiply by the sampling standard
    deviation of a Sharpe estimate and you have the Sharpe a *worthless*
    strategy is expected to display simply because you kept looking.

    ``n_trials <= 1`` returns 0.0: with a single, pre-registered trial there is
    no selection bias to correct, and the formula's :math:`\Phi^{-1}(0)` is
    ``-inf``.
    """
    n = int(n_trials)
    if n <= 1:
        return 0.0
    g = EULER_MASCHERONI
    z1 = float(stats.norm.ppf(1.0 - 1.0 / n))
    z2 = float(stats.norm.ppf(1.0 - 1.0 / (n * math.e)))
    return float(sharpe_std) * ((1.0 - g) * z1 + g * z2)


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    *,
    sharpe_std: float | None = None,
    periods: int = 252,
    annualized: bool = True,
) -> float:
    r"""Deflated Sharpe ratio -- Bailey & Lopez de Prado (2014) [2]_.

    The probability that the observed Sharpe is genuine skill rather than the
    best of ``n_trials`` lucky draws.  This is the single most important honesty
    metric in this project: **any** backtest that searched more than one
    configuration must report it.

    Method
    ------
    1. Under the null that every trial has zero true skill, the trial Sharpes are
       centred at zero with cross-sectional dispersion :math:`\sigma_{SR}`.
    2. The *expected best* of those N draws is

       .. math::
           SR^* = \sigma_{SR}\left[(1-\gamma)\Phi^{-1}\!\left(1-\tfrac1N\right)
                  + \gamma\Phi^{-1}\!\left(1-\tfrac{1}{Ne}\right)\right]

       with :math:`\gamma` the Euler-Mascheroni constant
       (:data:`EULER_MASCHERONI`) -- see :func:`expected_maximum_sharpe`.
    3. The DSR is the probabilistic Sharpe ratio of the observed Sharpe measured
       against that inflated threshold, skew- and kurtosis-adjusted -- see
       :func:`probabilistic_sharpe_ratio`.

    Parameters
    ----------
    sharpe : float
        Observed Sharpe of the selected strategy (annualised by default).
    n_trials : int
        Number of *independent* configurations searched: models, feature sets,
        lookbacks, thresholds, universes -- everything you tried and discarded.
        Counting honestly is the hard part; when hyper-parameter grids overlap
        heavily the effective number is smaller than the raw count, so a
        defensible practice is to report DSR at both the raw count and a
        conservative fraction of it.
    n_obs : int
        Number of return observations in the backtest.
    skew : float, default 0.0
        Skewness of the strategy's returns.
    kurtosis : float, default 3.0
        Non-excess kurtosis (Gaussian = 3.0).
    sharpe_std : float, keyword-only, optional
        Cross-sectional standard deviation of the trial Sharpes, in the same
        annualisation as ``sharpe``.  If you actually stored the Sharpe of every
        configuration you tried, pass ``np.std(all_sharpes, ddof=1)`` -- that is
        the estimator in the paper.  When omitted, the null-hypothesis value
        ``sqrt(1/(n_obs-1))`` per period is used: the sampling standard deviation
        of a Sharpe estimate when the true Sharpe is zero and returns are i.i.d.
        normal.
    periods : int, keyword-only, default 252
        Periods per year for de-annualisation.
    annualized : bool, keyword-only, default True
        Whether ``sharpe``/``sharpe_std`` are annualised.

    Returns
    -------
    float
        Probability in [0, 1].  Convention: **DSR > 0.95 is the bar** for
        calling a backtest result real.

    Notes
    -----
    Two behaviours worth internalising:

    * DSR **falls monotonically as ``n_trials`` rises**.  The same 1.2 Sharpe is
      strong evidence if it was the only thing you tried and nearly worthless if
      it was the best of a thousand.
    * DSR **rises with ``n_obs``**, because a longer track record shrinks the
      sampling error faster than the selection threshold grows.

    ``n_trials <= 1`` reduces exactly to the PSR against a zero benchmark.
    """
    n = int(n_obs)
    if n < 2:
        return float("nan")

    if sharpe_std is None:
        # Sampling sd of a per-period Sharpe estimate under the zero-skill null.
        std_per_period = math.sqrt(1.0 / (n - 1))
    else:
        std_per_period = _deannualize(sharpe_std, periods, annualized)

    sr_star = expected_maximum_sharpe(n_trials, std_per_period)

    # sr_star is already per-period, so PSR is called in per-period units.
    return probabilistic_sharpe_ratio(
        _deannualize(sharpe, periods, annualized),
        sr_star,
        n,
        skew,
        kurtosis,
        periods=periods,
        annualized=False,
    )


def minimum_track_record_length(
    sharpe: float,
    benchmark_sr: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    target_prob: float = 0.95,
    *,
    periods: int = 252,
    annualized: bool = True,
) -> float:
    r"""Observations needed for PSR to reach ``target_prob`` [1]_.

    .. math::
        MinTRL = 1 + \left(1 - \hat\gamma_3 SR + \tfrac{\hat\gamma_4-1}{4}SR^2\right)
                 \left(\frac{Z_\alpha}{SR - SR^*}\right)^2

    Returns
    -------
    float
        Required number of observations, or ``inf`` when the observed Sharpe
        does not exceed the benchmark (no amount of data proves a negative edge).

    Notes
    -----
    The practical use: a strategy showing a 0.8 annualised Sharpe with negative
    skew typically needs several years of daily data before its PSR clears 95%.
    If MinTRL exceeds the history you have, the honest statement is "not yet
    demonstrable", not "0.8 Sharpe".
    """
    sr = _deannualize(sharpe, periods, annualized)
    sr_star = _deannualize(benchmark_sr, periods, annualized)
    edge = sr - sr_star
    if edge <= _TINY:
        return float("inf")
    z = float(stats.norm.ppf(target_prob))
    variance = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr
    if variance <= _TINY:
        return float("nan")
    return float(1.0 + variance * (z / edge) ** 2)
