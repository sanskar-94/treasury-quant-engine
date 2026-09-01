"""Dynamic Nelson-Siegel: forecasting the curve as an object.

Everything upstream of this module forecasts each tenor's return separately and
lets the portfolio layer sort out the cross-section. That is backwards for a
yield curve, which is not nine loosely related assets but one object with three
degrees of freedom. Diebold & Li (2006) make the point precisely: fix the decay,
and the Nelson-Siegel betas become a three-dimensional state vector whose
dynamics you can model directly. Forecast the state, and the entire curve
follows - arbitrage-free in shape by construction, because every forecast is a
curve the model can actually produce.

The estimator here is the two-step Diebold-Li procedure:

1. **Cross-section.** With the decay fixed, each day's betas are the unique
   weighted-least-squares solution to that day's quotes. Already implemented in
   :func:`tqe.curve.nelson_siegel.fit_nss_history_fixed`.
2. **Time series.** Fit a vector autoregression to the beta series and iterate
   it forward ``h`` steps. Because level, slope and curvature are strongly
   persistent (daily autocorrelations above 0.99), a VAR captures most of the
   predictable variation, and the cross-equation terms are where the economics
   live: a steepening today says something about the level tomorrow.

Why this can work where per-tenor regression did not: it forecasts three
persistent factors instead of nine noisy returns, so the ratio of parameters to
signal is far better, and the resulting yield forecasts are internally
consistent across the curve rather than nine independent guesses that have to be
reconciled afterwards.

Every estimate is strictly causal. :func:`dns_forecast_history` refits the VAR on
a rolling window that **ends before** the date being forecast, and the test suite
asserts that corrupting the future leaves earlier forecasts bit-identical.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger("curve.dynamic")

__all__ = [
    "DNSParams",
    "VARModel",
    "fit_var",
    "dns_forecast",
    "dns_forecast_history",
    "beta_to_yields",
    "FACTOR_COLUMNS",
]

FACTOR_COLUMNS = ("beta0", "beta1", "beta2", "beta3")
EPS = 1e-12


# --------------------------------------------------------------------------- #
# VAR
# --------------------------------------------------------------------------- #
@dataclass
class VARModel:
    """A fitted vector autoregression ``y_t = c + sum_l A_l y_{t-l} + e_t``.

    Deliberately hand-rolled rather than delegated to statsmodels. The estimator
    is one least-squares solve on a stacked lag matrix, the forecast is one
    iteration loop, and owning both means the causality of the rolling refit is
    auditable in twenty lines instead of hidden behind a library that has its own
    opinions about missing data and deterministic terms.
    """

    coefs: np.ndarray          # (lags, k, k)
    intercept: np.ndarray      # (k,)
    lags: int
    names: list[str] = field(default_factory=list)
    sigma: np.ndarray | None = None   # residual covariance, (k, k)
    n_obs: int = 0

    @property
    def k(self) -> int:
        return len(self.intercept)

    def forecast(self, history: np.ndarray, steps: int = 1) -> np.ndarray:
        """Iterate the VAR forward.

        ``history`` is ``(n, k)`` with the most recent observation **last**; only
        the final ``lags`` rows are used. Returns ``(steps, k)``.
        """
        hist = np.asarray(history, dtype=float)
        if hist.ndim == 1:
            hist = hist.reshape(1, -1)
        if len(hist) < self.lags:
            pad = np.repeat(hist[:1], self.lags - len(hist), axis=0)
            hist = np.vstack([pad, hist])
        window = list(hist[-self.lags:])

        out = np.empty((steps, self.k))
        for s in range(steps):
            nxt = self.intercept.copy()
            for lag in range(self.lags):
                # window[-1] is t-1, window[-2] is t-2, ...
                nxt = nxt + self.coefs[lag] @ window[-(lag + 1)]
            out[s] = nxt
            window.append(nxt)
        return out

    def is_stable(self) -> bool:
        """Whether the companion matrix has all eigenvalues inside the unit circle.

        An unstable VAR produces forecasts that diverge as the horizon grows -
        harmless at one step, absurd at twenty. Worth checking before trusting a
        multi-step forecast, and the rolling fitter falls back to a random walk
        when this fails.
        """
        k, p = self.k, self.lags
        companion = np.zeros((k * p, k * p))
        companion[:k] = np.hstack([self.coefs[i] for i in range(p)])
        if p > 1:
            companion[k:, :-k] = np.eye(k * (p - 1))
        try:
            return bool(np.max(np.abs(np.linalg.eigvals(companion))) < 1.0)
        except np.linalg.LinAlgError:
            return False


def fit_var(data: pd.DataFrame | np.ndarray, lags: int = 1, ridge: float = 1e-8) -> VARModel:
    """Least-squares VAR.

    A small ridge term is added to the normal equations. The Nelson-Siegel
    factors are highly collinear - level and slope move together most days - so
    an unregularised solve is numerically fragile on short windows and produces
    coefficient estimates that swing wildly between refits.

    Parameters
    ----------
    data:
        ``(n, k)`` observations, oldest first.
    lags:
        Autoregressive order.
    ridge:
        L2 penalty on the coefficient matrix.
    """
    names = list(data.columns) if isinstance(data, pd.DataFrame) else []
    Y = data.to_numpy(dtype=float) if isinstance(data, pd.DataFrame) else np.asarray(data, float)
    n, k = Y.shape
    if n <= lags + 1:
        raise ValueError(f"Need more than {lags + 1} observations to fit a VAR({lags}), got {n}")

    # Design: [1, y_{t-1}, ..., y_{t-p}] -> y_t
    rows, targets = [], []
    for t in range(lags, n):
        row = [1.0]
        for lag in range(1, lags + 1):
            row.extend(Y[t - lag])
        rows.append(row)
        targets.append(Y[t])
    X = np.asarray(rows)
    T = np.asarray(targets)

    gram = X.T @ X + ridge * np.eye(X.shape[1])
    try:
        beta = np.linalg.solve(gram, X.T @ T)          # (1 + k*p, k)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(X, T, rcond=None)

    intercept = beta[0]
    coefs = np.stack([beta[1 + i * k: 1 + (i + 1) * k].T for i in range(lags)])
    resid = T - X @ beta
    sigma = (resid.T @ resid) / max(len(T) - X.shape[1], 1)
    return VARModel(coefs=coefs, intercept=intercept, lags=lags, names=names,
                    sigma=sigma, n_obs=len(T))


# --------------------------------------------------------------------------- #
# Curve reconstruction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DNSParams:
    """Fixed-decay Nelson-Siegel(-Svensson) state."""

    beta: np.ndarray          # (k,) - 3 for NS, 4 for Svensson
    tau1: float
    tau2: float

    def yields(self, tenors: Sequence[float] | np.ndarray) -> np.ndarray:
        return beta_to_yields(np.asarray(self.beta, float).reshape(1, -1),
                              tenors, self.tau1, self.tau2)[0]


def _loadings(tenors: np.ndarray, tau1: float, tau2: float, k: int) -> np.ndarray:
    """Nelson-Siegel loading matrix, ``(n_tenors, k)``.

    Column 0 is the level (constant 1), column 1 the slope
    ``(1-e^{-x})/x``, column 2 the curvature ``(1-e^{-x})/x - e^{-x}``, and
    column 3 the second Svensson hump on ``tau2``. The ``x -> 0`` limit of the
    slope loading is 1 and is handled explicitly, since the 1-month point makes
    ``x`` genuinely small.
    """
    t = np.asarray(tenors, dtype=float)

    def load(tau: float) -> tuple[np.ndarray, np.ndarray]:
        x = t / max(tau, EPS)
        f = np.where(np.abs(x) < 1e-8, 1.0 - x / 2.0,
                     (1.0 - np.exp(-x)) / np.where(x == 0, 1.0, x))
        return f, np.exp(-x)

    f1, e1 = load(tau1)
    cols = [np.ones_like(t), f1, f1 - e1]
    if k >= 4:
        f2, e2 = load(tau2)
        cols.append(f2 - e2)
    return np.column_stack(cols[:k])


def beta_to_yields(
    betas: np.ndarray,
    tenors: Sequence[float] | np.ndarray,
    tau1: float,
    tau2: float,
) -> np.ndarray:
    """Reconstruct yields from factor states. ``(n_dates, k) -> (n_dates, n_tenors)``."""
    B = np.atleast_2d(np.asarray(betas, dtype=float))
    L = _loadings(np.asarray(tenors, float), tau1, tau2, B.shape[1])
    return B @ L.T


def dns_forecast(
    betas: pd.DataFrame,
    tenors: Sequence[float],
    tau1: float,
    tau2: float,
    horizon: int = 1,
    lags: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """One-shot forecast: fit a VAR on ``betas`` and project the curve forward.

    Returns ``(forecast_betas, forecast_yields)`` for the final horizon step.
    Use :func:`dns_forecast_history` for anything evaluated out of sample - this
    function fits on everything it is given, so calling it on a full history and
    reading off the forecasts would be in-sample.
    """
    cols = [c for c in FACTOR_COLUMNS if c in betas.columns]
    clean = betas[cols].dropna()
    model = fit_var(clean, lags=lags)
    path = model.forecast(clean.to_numpy(dtype=float), steps=horizon)
    fb = path[-1]
    return fb, beta_to_yields(fb.reshape(1, -1), tenors, tau1, tau2)[0]


def dns_forecast_history(
    betas: pd.DataFrame,
    tenor_map: dict[str, float],
    tau1: float,
    tau2: float,
    horizon: int = 1,
    lags: int = 1,
    window: int = 756,
    min_periods: int = 252,
    refit_every: int = 21,
    expanding: bool = False,
) -> pd.DataFrame:
    """Rolling out-of-sample curve forecasts.

    For each date ``t`` the VAR is fitted on factor observations strictly
    **before** ``t``, then iterated ``horizon`` steps from the state at ``t``.
    The slice bound is exclusive, which is the single line that makes the whole
    thing causal.

    The VAR is refitted every ``refit_every`` observations rather than daily.
    Coefficients on factors this persistent barely move between refits, and the
    saving is a factor of twenty.

    An unstable fit (companion eigenvalue outside the unit circle) is rejected
    and that block falls back to a random walk - the honest null for a near
    unit-root process, and better than a forecast that diverges.

    Returns
    -------
    pd.DataFrame
        Indexed like ``betas``. Columns: forecast factors ``f_beta*``, forecast
        yields ``f_<tenor>``, and ``dy_<tenor>`` - the predicted yield *change*
        from today, which is what a signal actually wants.
    """
    cols = [c for c in FACTOR_COLUMNS if c in betas.columns]
    clean = betas[cols].dropna()
    if len(clean) < min_periods + lags:
        return pd.DataFrame(index=betas.index)

    labels = list(tenor_map)
    tenors = np.array([tenor_map[c] for c in labels], dtype=float)
    values = clean.to_numpy(dtype=float)
    n, k = values.shape

    out_beta = np.full((n, k), np.nan)
    model: VARModel | None = None
    last_fit = -10**9
    n_unstable = 0

    for t in range(n):
        if t < min_periods:
            continue
        if model is None or (t - last_fit) >= refit_every:
            start = 0 if expanding else max(0, t - window)
            train = values[start:t]          # EXCLUSIVE of t - the causality bound
            if len(train) < lags + 5:
                continue
            try:
                candidate = fit_var(train, lags=lags)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if not candidate.is_stable():
                n_unstable += 1
                candidate = None
            model = candidate
            last_fit = t
        if model is None:
            out_beta[t] = values[t]          # random-walk fallback
            continue
        hist = values[max(0, t - model.lags + 1): t + 1]
        out_beta[t] = model.forecast(hist, steps=horizon)[-1]

    if n_unstable:
        log.info("DNS: %d refits rejected as unstable; random walk used there", n_unstable)

    fb = pd.DataFrame(out_beta, index=clean.index, columns=[f"f_{c}" for c in cols])
    fy = beta_to_yields(out_beta, tenors, tau1, tau2)
    cur = beta_to_yields(values, tenors, tau1, tau2)

    frame = pd.concat(
        [
            fb,
            pd.DataFrame(fy, index=clean.index, columns=[f"f_{c}" for c in labels]),
            pd.DataFrame(fy - cur, index=clean.index, columns=[f"dy_{c}" for c in labels]),
        ],
        axis=1,
    )
    return frame.reindex(betas.index)
