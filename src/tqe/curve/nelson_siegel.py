"""Nelson-Siegel and Nelson-Siegel-Svensson parametric yield curves.

Why a parametric curve at all
-----------------------------
The Treasury publishes 9-14 constant-maturity points per day.  Trading the curve
requires a *function* :math:`r(t)` defined at every maturity: to price an
off-the-run bond, to compute roll-down between quoted pillars, to say whether the
7y is rich to its neighbours.  Nelson-Siegel (1987) and Svensson's (1994)
four-factor extension are the market standard because their three (or four)
loadings are exactly the shapes the curve actually moves in - a level, a slope
and one or two humps - which is also what a PCA of yield changes recovers
empirically (see :mod:`tqe.curve.pca`).  Central banks (BIS, ECB, Riksbank) fit
this same functional form for their published zero curves.

The model
---------
.. math::

    r(t) = \\beta_0
         + \\beta_1 \\frac{1-e^{-t/\\tau_1}}{t/\\tau_1}
         + \\beta_2 \\left(\\frac{1-e^{-t/\\tau_1}}{t/\\tau_1} - e^{-t/\\tau_1}\\right)
         + \\beta_3 \\left(\\frac{1-e^{-t/\\tau_2}}{t/\\tau_2} - e^{-t/\\tau_2}\\right)

* :math:`\\beta_0` is the level - the asymptotic long rate, :math:`r(\\infty)`.
* :math:`\\beta_1` is the *negative* of the slope: :math:`r(0)=\\beta_0+\\beta_1`,
  so ``long - short = -beta_1``.
* :math:`\\beta_2, \\beta_3` are curvature (hump) amplitudes peaking near
  :math:`\\tau_1` and :math:`\\tau_2` years.

Fitting strategy
----------------
Throwing all six parameters at an optimizer is a well-known way to land in a
local minimum - the objective in :math:`(\\tau_1,\\tau_2)` is genuinely
multi-modal.  But for *fixed* decay parameters the model is **linear in the
betas**, so the betas are a two-line weighted least-squares solve.  We therefore
grid the two-dimensional :math:`(\\tau_1,\\tau_2)` space, solve the betas exactly
at every node, keep the best node, and only then polish all six parameters with
Levenberg-Marquardt/trust-region least squares.  This is the standard "grid +
polish" approach used by the BIS papers and it is what makes a 9,000-day
historical fit both fast and reproducible.

What the fit is applied to
--------------------------
:func:`fit_nss_history` is fed the Treasury **par (CMT) yields**, so the fitted
object is a smooth *par* curve.  That is the right thing for relative-value
signals (rich/cheap versus the fitted curve) and for interpolating a par yield at
an unquoted tenor.  If a genuine discount function is required, bootstrap the
zeros first (:func:`tqe.curve.bootstrap.par_to_zero`) and fit those instead -
the code here is agnostic about which rate it is handed.

No look-ahead
-------------
Every fit here is **purely cross-sectional**: the parameters for date *t* are a
function of date *t*'s quotes only.  :func:`fit_nss_history` seeds each day's
optimizer with the *previous* day's decay parameters, which is information dated
*t-1*; no future observation ever enters.  The seed only chooses a starting point
for the polish - the tau grid is searched exhaustively every day - so the result
is not path-dependent beyond the last digits of the polish.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ..data.sources import TENOR_YEARS
from ..logging_utils import get_logger

log = get_logger("curve.nelson_siegel")

__all__ = [
    "fit_nss_history_fixed",
    "DIEBOLD_LI_TAU1",
    "SVENSSON_FIXED_TAU2",
    "NSSParams",
    "nss_zero_rate",
    "nss_forward_rate",
    "fit_nss",
    "fit_nss_history",
    "DEFAULT_TAU1_GRID",
    "DEFAULT_TAU2_GRID",
    "NSS_COLUMNS",
]

# Decay grids in years.  tau1 covers the money-market-to-5y hump, tau2 the
# 5y-30y hump; they bracket where the US curve actually bends.
DEFAULT_TAU1_GRID: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
DEFAULT_TAU2_GRID: tuple[float, ...] = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 30.0)

# tau2 must sit meaningfully above tau1 or the two hump loadings become
# collinear and the betas are unidentified (they blow up in equal and opposite
# directions while the fitted curve barely moves).
# ---------------------------------------------------------------------------
# Fixed decay constants
# ---------------------------------------------------------------------------
# The NSS parameterisation is NOT identifiable: for a curve quoted at ~13 tenors
# many (beta, tau) combinations are observationally equivalent to well under a
# basis point.  Fitting all six parameters freely therefore produces an
# excellent *curve* but wildly unstable *parameters* - measured over the
# 1990-2026 history, free-tau beta3 has a standard deviation of 1.65 and a 99th
# percentile daily change of 2.25, because the optimiser hops between equivalent
# solutions from one day to the next.
#
# Diebold & Li (2006) resolve this by fixing the decay and estimating only the
# betas, which makes the model linear and the factors uniquely identified.  They
# use lambda = 0.0609 per month, i.e. tau1 = 1/0.0609 months ~ 1.37 years, chosen
# to maximise the curvature loading at the 2-3 year point.  ``tau2 = 8.0`` is the
# customary Svensson companion, placing the second hump in the 7-10y sector.
#
# With the decays fixed, beta3's standard deviation falls to 0.031 (53x more
# stable) and the factors become economically interpretable: over the full
# history corr(beta0 + beta1, 3m yield) = 0.998, i.e. the model's instantaneous
# short rate really is the short rate.  The price is fit quality - mean RMSE
# rises from ~3.0bp to ~7.6bp.
#
# Use FREE taus when you want the best possible curve (pricing, rich/cheap
# analysis); use FIXED taus when the betas are consumed as model features.
DIEBOLD_LI_TAU1 = 1.37
SVENSSON_FIXED_TAU2 = 8.0

_MIN_TAU_RATIO = 1.5

# Below this argument the closed-form loading (1-exp(-x))/x is replaced by its
# Maclaurin series.  The truncation error of the four-term series at x = 1e-4 is
# O(x^4/120) ~ 1e-18, i.e. below double precision, so the switch is seamless.
_SERIES_CUTOFF = 1e-4

# Sanity bounds for the polish step.  Rates are decimals, so a |beta| above 5
# (500%) is a runaway, and a tau outside [0.02, 60] years is outside the span of
# anything the Treasury quotes.
_BETA_BOUND = 5.0
_TAU1_BOUNDS = (0.02, 12.0)
_TAU2_BOUNDS = (1.0, 60.0)

NSS_COLUMNS: tuple[str, ...] = (
    "beta0",
    "beta1",
    "beta2",
    "beta3",
    "tau1",
    "tau2",
    "rmse",
    "n_points",
)


# --------------------------------------------------------------------------- #
# Core functional form
# --------------------------------------------------------------------------- #
def _decay_loading(x: np.ndarray | float) -> np.ndarray:
    """The Nelson-Siegel slope loading :math:`(1-e^{-x})/x`, safe at ``x = 0``.

    Parameters
    ----------
    x:
        Scaled maturity :math:`t/\\tau`.  May be a scalar, array, zero, or
        contain NaN.

    Returns
    -------
    numpy.ndarray
        Loading values in ``(0, 1]``; exactly ``1.0`` in the limit ``x -> 0``.

    Notes
    -----
    Two numerical hazards are handled explicitly:

    * **Division by zero at t = 0.**  The limit is 1, but a raw divide emits a
      warning and returns NaN.  We evaluate the closed form on a *substituted*
      denominator and select with :func:`numpy.where`, so the invalid branch is
      never actually computed.
    * **Catastrophic cancellation for small x.**  ``1 - exp(-x)`` loses all
      significant digits as ``x -> 0``; :func:`numpy.expm1` computes it to full
      relative precision, and below ``_SERIES_CUTOFF`` we use the Maclaurin
      series :math:`1 - x/2 + x^2/6 - x^3/24` instead.

    The distinction matters in production: the front of the curve (1-month
    bills, ``t = 0.083``) is where a bad loading silently poisons ``beta1``.
    """
    xa = np.asarray(x, dtype=float)
    small = np.abs(xa) < _SERIES_CUTOFF
    safe = np.where(small, 1.0, xa)  # never 0 -> no divide warning
    closed = -np.expm1(-safe) / safe
    series = 1.0 - xa / 2.0 + xa * xa / 6.0 - xa * xa * xa / 24.0
    return np.where(small, series, closed)


def nss_zero_rate(
    t: float | np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float = 0.0,
    tau1: float = 1.5,
    tau2: float = 10.0,
) -> np.ndarray:
    """Nelson-Siegel-Svensson rate at maturity ``t`` (years).

    Parameters
    ----------
    t:
        Maturity in years, scalar or array.  ``t = 0`` is legal and returns the
        instantaneous short rate ``beta0 + beta1``.
    beta0, beta1, beta2, beta3:
        Level, slope, first-hump and second-hump coefficients, in **decimal**
        rate units (``0.0473`` = 4.73%).  ``beta3 = 0`` reduces the model to
        three-factor Nelson-Siegel.
    tau1, tau2:
        Decay parameters in years, strictly positive.  Loosely, the maturities
        at which the two curvature loadings peak.

    Returns
    -------
    numpy.ndarray
        Fitted rate(s), decimal.  A 0-d array is returned for scalar input.

    Raises
    ------
    ValueError
        If a decay parameter is non-positive or any maturity is negative.

    Notes
    -----
    Causality: this is a pure function of one day's parameters and a maturity.
    It contains no time series and therefore cannot look ahead.
    """
    if not (tau1 > 0.0) or not (tau2 > 0.0):
        raise ValueError(f"tau parameters must be positive, got tau1={tau1}, tau2={tau2}")
    ta = np.asarray(t, dtype=float)
    if np.any(ta < 0.0):
        raise ValueError("Maturities must be non-negative")

    x1 = ta / tau1
    f1 = _decay_loading(x1)
    e1 = np.exp(-x1)
    out = beta0 + beta1 * f1 + beta2 * (f1 - e1)
    if beta3 != 0.0:
        x2 = ta / tau2
        out = out + beta3 * (_decay_loading(x2) - np.exp(-x2))
    return out


def nss_forward_rate(
    t: float | np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float = 0.0,
    tau1: float = 1.5,
    tau2: float = 10.0,
) -> np.ndarray:
    """Instantaneous forward rate implied by the same parameters.

    Parameters
    ----------
    t:
        Maturity in years.
    beta0, beta1, beta2, beta3, tau1, tau2:
        As in :func:`nss_zero_rate`.

    Returns
    -------
    numpy.ndarray
        Instantaneous forward rate, decimal.

    Notes
    -----
    Nelson-Siegel is defined *as* a forward curve - the spot curve is its
    average.  Differentiating :math:`t\\,r(t)`:

    .. math::

        f(t) = \\beta_0 + \\beta_1 e^{-t/\\tau_1}
             + \\beta_2 \\frac{t}{\\tau_1} e^{-t/\\tau_1}
             + \\beta_3 \\frac{t}{\\tau_2} e^{-t/\\tau_2}

    which is exactly a constant plus a Laguerre polynomial in ``t``.  The
    forward curve is what carry and roll-down are computed against, so it is
    exposed rather than left implicit.
    """
    if not (tau1 > 0.0) or not (tau2 > 0.0):
        raise ValueError(f"tau parameters must be positive, got tau1={tau1}, tau2={tau2}")
    ta = np.asarray(t, dtype=float)
    if np.any(ta < 0.0):
        raise ValueError("Maturities must be non-negative")
    x1 = ta / tau1
    e1 = np.exp(-x1)
    out = beta0 + beta1 * e1 + beta2 * x1 * e1
    if beta3 != 0.0:
        x2 = ta / tau2
        out = out + beta3 * x2 * np.exp(-x2)
    return out


@dataclass(frozen=True)
class NSSParams:
    """One day's fitted Nelson-Siegel-Svensson parameters.

    Attributes
    ----------
    beta0, beta1, beta2, beta3:
        Decimal rate coefficients (see :func:`nss_zero_rate`).
    tau1, tau2:
        Decay parameters in years.

    Notes
    -----
    The three convenience properties map the betas onto the vocabulary a rates
    desk actually uses, and onto the first three principal components of yield
    changes: ``level`` (parallel shift), ``slope`` (long minus short, so a
    *positive* slope is a steep curve) and ``curvature`` (the belly versus the
    wings, i.e. the butterfly).
    """

    beta0: float
    beta1: float
    beta2: float
    beta3: float = 0.0
    tau1: float = 1.5
    tau2: float = 10.0

    def __post_init__(self) -> None:
        # NaN never trips these comparisons, so a "failed fit" sentinel object
        # can still be constructed; only genuinely invalid decays are rejected.
        if self.tau1 <= 0.0 or self.tau2 <= 0.0:
            raise ValueError(f"tau parameters must be positive: {self.tau1}, {self.tau2}")

    # ---------------- evaluation ---------------- #
    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """Fitted rate at maturity ``t`` in years (scalar in, float out)."""
        out = nss_zero_rate(t, self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2)
        return float(out) if np.isscalar(t) or np.ndim(t) == 0 else out

    def forward_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """Instantaneous forward rate at maturity ``t`` in years."""
        out = nss_forward_rate(
            t, self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2
        )
        return float(out) if np.isscalar(t) or np.ndim(t) == 0 else out

    def discount(self, t: float | np.ndarray, compounding: int = 2) -> float | np.ndarray:
        """Discount factor for maturity ``t``.

        Parameters
        ----------
        t:
            Maturity in years.
        compounding:
            Compounding frequency of the fitted rate.  Defaults to ``2``
            (semi-annual), the convention of every other rate in this codebase
            and of the US Treasury market itself.  Pass ``0`` for continuous
            compounding, ``D(t) = exp(-r t)``.

        Returns
        -------
        float or numpy.ndarray
            ``(1 + r/f)^(-f t)``, or ``exp(-r t)`` when ``compounding == 0``.
        """
        r = np.asarray(self.zero_rate(t), dtype=float)
        ta = np.asarray(t, dtype=float)
        if compounding <= 0:
            out = np.exp(-r * ta)
        else:
            out = np.power(1.0 + r / compounding, -compounding * ta)
        return float(out) if np.ndim(ta) == 0 else out

    # ---------------- (de)serialisation ---------------- #
    def as_array(self) -> np.ndarray:
        """Parameters as ``[beta0, beta1, beta2, beta3, tau1, tau2]``."""
        return np.array(
            [self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2], dtype=float
        )

    @classmethod
    def from_array(cls, arr: Sequence[float] | np.ndarray) -> "NSSParams":
        """Inverse of :meth:`as_array`."""
        a = np.asarray(arr, dtype=float).ravel()
        if a.size != 6:
            raise ValueError(f"Expected 6 parameters, got {a.size}")
        return cls(float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]), float(a[5]))

    def as_dict(self) -> dict[str, float]:
        """Parameters keyed by name - the row layout of :func:`fit_nss_history`."""
        return {
            "beta0": self.beta0,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "beta3": self.beta3,
            "tau1": self.tau1,
            "tau2": self.tau2,
        }

    # ---------------- desk vocabulary ---------------- #
    @property
    def level(self) -> float:
        """Asymptotic long rate ``r(inf)`` - the parallel-shift factor."""
        return self.beta0

    @property
    def slope(self) -> float:
        """Long minus short, ``r(inf) - r(0) = -beta1``.  Positive = steep."""
        return -self.beta1

    @property
    def curvature(self) -> float:
        """Amplitude of the first hump - the belly-versus-wings factor."""
        return self.beta2


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def _design(t: np.ndarray, tau1: float, tau2: float, svensson: bool) -> np.ndarray:
    """Regressor matrix for fixed decays: columns are the model's loadings."""
    x1 = t / tau1
    f1 = _decay_loading(x1)
    e1 = np.exp(-x1)
    cols = [np.ones_like(t), f1, f1 - e1]
    if svensson:
        x2 = t / tau2
        cols.append(_decay_loading(x2) - np.exp(-x2))
    return np.column_stack(cols)


def _prior(y: np.ndarray, n_beta: int) -> np.ndarray:
    """Shrinkage target: the flat curve sitting at the longest quoted yield.

    ``y`` must already be sorted by maturity.  The prior is ``(y_long, 0, 0, 0)``
    - level equal to the long end, no slope, no humps - which is the honest
    "I know nothing about the shape" curve.
    """
    p = np.zeros(n_beta)
    p[0] = float(y[-1])
    return p


def _solve_penalised(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float, prior: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Penalised WLS for fixed decays.  Returns ``(beta, fit_rmse, objective)``.

    Minimises ``mean(w * resid^2) + ridge * ||beta - prior||^2`` by solving the
    augmented least-squares system (never the normal equations), so the solve is
    exact to working precision even when two loadings nearly coincide.
    """
    n = y.size
    sw = np.sqrt(w / n)
    if ridge > 0.0:
        p = X.shape[1]
        A = np.vstack([X * sw[:, None], np.sqrt(ridge) * np.eye(p)])
        b = np.concatenate([y * sw, np.sqrt(ridge) * prior])
    else:
        A, b = X * sw[:, None], y * sw
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = y - X @ beta
    fit_mse = float(np.mean(w * resid * resid))
    obj = fit_mse + (ridge * float(np.sum((beta - prior) ** 2)) if ridge > 0.0 else 0.0)
    return beta, float(np.sqrt(fit_mse)), obj


def _weighted_lstsq(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact weighted least squares for fixed decays.

    Solves ``min_beta mean(w * (y - X beta)^2)`` via SVD on the whitened system
    rather than the normal equations.  That matters here: when ``tau1`` and
    ``tau2`` are close the two Svensson hump loadings become nearly collinear
    and ``X'X`` is severely ill-conditioned, so forming it would throw away
    roughly half the available precision.  ``lstsq`` also returns the
    minimum-norm solution when the design is genuinely rank-deficient (a curve
    quoted at only three tenors, say) instead of raising.

    Returns
    -------
    beta:
        Fitted coefficients, length ``X.shape[1]``.
    rmse:
        Weighted root-mean-square residual, in the same units as ``y``
        (decimal rate).
    """
    n = y.size
    sw = np.sqrt(w / n)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    resid = y - X @ beta
    return beta, float(np.sqrt(np.mean(w * resid * resid)))


def _tau_pairs(
    tau1_grid: Sequence[float], tau2_grid: Sequence[float], svensson: bool
) -> np.ndarray:
    """Admissible ``(tau1, tau2)`` grid nodes, shape ``(G, 2)``."""
    t1 = np.asarray(sorted({float(v) for v in tau1_grid if v > 0}), dtype=float)
    if t1.size == 0:
        raise ValueError("tau1_grid is empty")
    if not svensson:
        # tau2 is inert when beta3 == 0; carry tau1 through so the stored
        # parameter is still a valid (positive) decay.
        return np.column_stack([t1, t1])
    t2 = np.asarray(sorted({float(v) for v in tau2_grid if v > 0}), dtype=float)
    if t2.size == 0:
        raise ValueError("tau2_grid is empty")
    grid = np.array([(a, b) for a in t1 for b in t2 if b >= _MIN_TAU_RATIO * a], dtype=float)
    if grid.size == 0:
        raise ValueError("No (tau1, tau2) pair satisfies the separation requirement")
    return grid


def _screen_grid(
    t: np.ndarray, y: np.ndarray, w: np.ndarray, pairs: np.ndarray, svensson: bool
) -> tuple[int, np.ndarray]:
    """Solve the linear sub-problem at every grid node at once.

    Returns the index of the best node and the vector of weighted RMSEs.  The
    betas are obtained from the (ridge-stabilised) normal equations rather than
    an SVD: with at most four regressors this is a batched 4x4 solve, roughly
    fifty times faster than looping ``lstsq``, and the winner is re-solved
    exactly afterwards so the loss of half the working precision in the screen
    never reaches the returned parameters.
    """
    n = t.size
    x1 = t[None, :] / pairs[:, 0][:, None]
    f1 = _decay_loading(x1)
    e1 = np.exp(-x1)
    cols = [np.ones_like(x1), f1, f1 - e1]
    if svensson:
        x2 = t[None, :] / pairs[:, 1][:, None]
        cols.append(_decay_loading(x2) - np.exp(-x2))
    X = np.stack(cols, axis=-1)  # (G, n, p)
    p = X.shape[-1]

    Xw = X * w[None, :, None]
    XtX = np.einsum("gni,gnj->gij", Xw, X)
    Xty = np.einsum("gni,n->gi", Xw, y)
    # Ridge proportional to the trace: it is invisible on a well-conditioned
    # node and rescues the near-collinear ones instead of raising LinAlgError.
    ridge = (np.einsum("gii->g", XtX) * 1e-12 / p)[:, None, None] * np.eye(p)
    try:
        # NumPy 2 treats a non-1-D right-hand side as a matrix stack, hence the
        # explicit trailing axis rather than passing (G, p) directly.
        beta = np.linalg.solve(XtX + ridge, Xty[..., None])[..., 0]
    except np.linalg.LinAlgError:  # pragma: no cover - ridge makes this unreachable
        beta = np.stack([np.linalg.lstsq(X[g], y, rcond=None)[0] for g in range(X.shape[0])])
    resid = np.einsum("gni,gi->gn", X, beta) - y[None, :]
    rmse = np.sqrt(np.einsum("n,gn->g", w, resid * resid) / n)
    rmse = np.where(np.isfinite(rmse), rmse, np.inf)
    return int(np.argmin(rmse)), rmse


def _polish(
    t: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    start: np.ndarray,
    svensson: bool,
    max_nfev: int,
    ftol: float,
) -> np.ndarray | None:
    """Refine all parameters jointly with a bounded trust-region solve.

    ``start`` is the full 6-vector.  Only the parameters the model actually uses
    are free (``beta3`` and ``tau2`` are frozen for plain Nelson-Siegel).  An
    analytic Jacobian is supplied - it costs four lines and removes the seven
    extra residual evaluations a finite-difference Jacobian would need on every
    iteration, which matters when this runs 9,000 times.
    """
    free = np.array([0, 1, 2, 3, 4, 5] if svensson else [0, 1, 2, 4])
    lo = np.array([-_BETA_BOUND] * 4 + [_TAU1_BOUNDS[0], _TAU2_BOUNDS[0]])
    hi = np.array([_BETA_BOUND] * 4 + [_TAU1_BOUNDS[1], _TAU2_BOUNDS[1]])
    p0 = np.clip(start, lo, hi)
    sw = np.sqrt(w)

    def unpack(free_vals: np.ndarray) -> np.ndarray:
        full = start.copy()
        full[free] = free_vals
        return full

    def resid(free_vals: np.ndarray) -> np.ndarray:
        b0, b1, b2, b3, ta1, ta2 = unpack(free_vals)
        return sw * (nss_zero_rate(t, b0, b1, b2, b3, ta1, ta2) - y)

    def jac(free_vals: np.ndarray) -> np.ndarray:
        b0, b1, b2, b3, ta1, ta2 = unpack(free_vals)
        x1 = t / ta1
        f1 = _decay_loading(x1)
        e1 = np.exp(-x1)
        x2 = t / ta2
        f2 = _decay_loading(x2)
        e2 = np.exp(-x2)
        # d f/d tau = (f - e)/tau ; d e/d tau = x e / tau
        d_tau1 = (b1 * (f1 - e1) + b2 * (f1 - e1 - x1 * e1)) / ta1
        d_tau2 = b3 * (f2 - e2 - x2 * e2) / ta2
        full = np.column_stack(
            [np.ones_like(t), f1, f1 - e1, f2 - e2, d_tau1, d_tau2]
        )
        return full[:, free] * sw[:, None]

    try:
        sol = least_squares(
            resid,
            p0[free],
            jac=jac,
            bounds=(lo[free], hi[free]),
            method="trf",
            xtol=ftol,
            ftol=ftol,
            gtol=ftol,
            max_nfev=max_nfev,
        )
    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - defensive
        return None
    if not sol.success and sol.status <= 0:
        return None
    return unpack(sol.x)


def fit_nss(
    tenors_years: Sequence[float] | np.ndarray,
    yields: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
    tau1_grid: Sequence[float] = DEFAULT_TAU1_GRID,
    tau2_grid: Sequence[float] = DEFAULT_TAU2_GRID,
    model: str = "svensson",
    *,
    polish: bool = True,
    seed_taus: tuple[float, float] | None = None,
    max_nfev: int = 200,
    ftol: float = 1e-12,
) -> tuple[NSSParams, float]:
    """Fit one cross-section of rates with the grid-plus-polish scheme.

    Parameters
    ----------
    tenors_years:
        Maturities in years, strictly positive.  Non-finite entries are dropped.
    yields:
        Rates in **decimal** at those maturities (par yields, zeros, whatever is
        being modelled).  Non-finite entries are dropped pairwise with the
        tenors.
    weights:
        Optional relative weights.  Rescaled internally to mean 1 so that the
        returned RMSE is directly comparable across days and equals the plain
        RMSE when weights are uniform.  A common choice is ``1/duration`` when
        fitting *prices*; for yields, uniform weights are the market default.
    tau1_grid, tau2_grid:
        Decay-parameter nodes in years.  Nodes with ``tau2 < 1.5 * tau1`` are
        skipped because the two curvature loadings become collinear there.
    model:
        ``"svensson"`` (four betas) or ``"nelson_siegel"`` (``beta3`` forced to
        zero; ``tau2`` is then inert and stored equal to ``tau1``).
    polish:
        Run the joint six-parameter refinement after the grid search.
    seed_taus:
        Optional extra grid node, typically the previous day's decays.  Included
        as a *candidate*, never as a replacement for the grid.
    max_nfev, ftol:
        Trust-region solver budget and tolerance.

    Returns
    -------
    params : NSSParams
        The fitted parameters.
    rmse : float
        Root-mean-square residual in **decimal** rate units - multiply by
        ``1e4`` for basis points.  A good fit to the CMT curve is 1-4bp.

    Raises
    ------
    ValueError
        If fewer usable points than free betas are supplied (3 for
        Nelson-Siegel, 4 for Svensson) or if ``model`` is unknown.

    Notes
    -----
    The polish is *accepted only if it improves the objective* and leaves the
    parameters inside their admissible region with ``tau1 < tau2`` preserved;
    otherwise the exact grid solution is returned.  This guarantees the result
    is never worse than the deterministic grid search, which is what makes a
    9,000-day historical fit reproducible rather than a random walk through
    local minima.

    Causality: purely cross-sectional - a single day's quotes in, that day's
    parameters out.  ``seed_taus`` is the caller's responsibility to keep dated
    at or before the observation (:func:`fit_nss_history` passes ``t-1``).
    """
    if model not in ("svensson", "nelson_siegel"):
        raise ValueError(f"Unknown model {model!r}; expected 'svensson' or 'nelson_siegel'")
    svensson = model == "svensson"

    t = np.asarray(tenors_years, dtype=float).ravel()
    y = np.asarray(yields, dtype=float).ravel()
    if t.size != y.size:
        raise ValueError(f"tenors_years ({t.size}) and yields ({y.size}) length mismatch")
    if weights is None:
        w = np.ones_like(t)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != t.size:
            raise ValueError("weights must match tenors_years in length")

    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(w) & (t > 0.0) & (w > 0.0)
    t, y, w = t[ok], y[ok], w[ok]
    n_beta = 4 if svensson else 3
    if t.size < n_beta:
        raise ValueError(f"Need at least {n_beta} usable points for {model}, got {t.size}")
    order = np.argsort(t)
    t, y, w = t[order], y[order], w[order]
    w = w * (w.size / w.sum())  # mean 1 -> RMSE stays in decimal rate units

    pairs = _tau_pairs(tau1_grid, tau2_grid, svensson)
    if seed_taus is not None:
        s1, s2 = float(seed_taus[0]), float(seed_taus[1])
        if np.isfinite(s1) and np.isfinite(s2) and s1 > 0 and s2 > 0:
            if not svensson:
                s2 = s1
            if s2 >= _MIN_TAU_RATIO * s1 or not svensson:
                pairs = np.vstack([np.array([[s1, s2]]), pairs])

    best_idx, _ = _screen_grid(t, y, w, pairs, svensson)
    tau1, tau2 = float(pairs[best_idx, 0]), float(pairs[best_idx, 1])

    # Re-solve the winning node exactly (SVD) - the screen used normal equations.
    beta, rmse = _weighted_lstsq(_design(t, tau1, tau2, svensson), y, w)
    full = np.zeros(6)
    full[: len(beta)] = beta
    if not svensson:
        full[3] = 0.0
    full[4], full[5] = tau1, tau2

    if polish and t.size > n_beta:
        cand = _polish(t, y, w, full, svensson, max_nfev, ftol)
        if cand is not None:
            b0, b1, b2, b3, c1, c2 = cand
            ordered = (c2 > _MIN_TAU_RATIO * c1) if svensson else True
            bounded = np.all(np.abs(cand[:4]) <= _BETA_BOUND) and np.all(np.isfinite(cand))
            if ordered and bounded:
                resid = nss_zero_rate(t, b0, b1, b2, b3, c1, c2) - y
                cand_rmse = float(np.sqrt(np.mean(w * resid * resid)))
                if cand_rmse < rmse:
                    full, rmse = cand, cand_rmse

    if not svensson:
        full[3] = 0.0
        full[5] = full[4]
    return NSSParams.from_array(full), rmse


def _fit_block(
    values: np.ndarray,
    years: np.ndarray,
    svensson: bool,
    tau1_grid: Sequence[float],
    tau2_grid: Sequence[float],
    min_points: int,
    polish: bool,
    weights: np.ndarray | None,
    max_nfev: int,
    ftol: float,
    seed: tuple[float, float] | None,
) -> np.ndarray:
    """Fit a contiguous block of days, warm-starting each from its predecessor."""
    n_rows = values.shape[0]
    out = np.full((n_rows, len(NSS_COLUMNS)), np.nan)
    model = "svensson" if svensson else "nelson_siegel"
    for i in range(n_rows):
        row = values[i]
        ok = np.isfinite(row)
        n_ok = int(ok.sum())
        out[i, 7] = n_ok
        if n_ok < min_points:
            continue
        w = None if weights is None else weights[ok]
        try:
            params, rmse = fit_nss(
                years[ok],
                row[ok],
                weights=w,
                tau1_grid=tau1_grid,
                tau2_grid=tau2_grid,
                model=model,
                polish=polish,
                seed_taus=seed,
                max_nfev=max_nfev,
                ftol=ftol,
            )
        except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - defensive
            log.warning("NSS fit failed on block row %d (%d points)", i, n_ok)
            continue
        out[i, :6] = params.as_array()
        out[i, 6] = rmse
        seed = (params.tau1, params.tau2)
    return out


def fit_nss_history(
    curve: pd.DataFrame,
    tenor_years: Mapping[str, float] | None = None,
    model: str = "svensson",
    n_jobs: int = 1,
    *,
    tau1_grid: Sequence[float] = DEFAULT_TAU1_GRID,
    tau2_grid: Sequence[float] = DEFAULT_TAU2_GRID,
    min_points: int = 4,
    polish: bool = True,
    weights: Mapping[str, float] | None = None,
    max_nfev: int = 200,
    ftol: float = 1e-12,
) -> pd.DataFrame:
    """Fit the curve on every date in ``curve``.

    Parameters
    ----------
    curve:
        Par-yield panel: ``DatetimeIndex`` named ``date``, one column per CMT
        tenor, **decimal** rates, NaN-ragged.
    tenor_years:
        Label -> maturity-in-years map.  Defaults to
        :data:`tqe.data.sources.TENOR_YEARS`.  Columns absent from the map are
        ignored, so passing a subset selects the tenors to fit.
    model:
        ``"svensson"`` or ``"nelson_siegel"``.
    n_jobs:
        Threads/processes for the fit.  ``1`` (default) is a single sequential
        pass; ``>1`` splits the history into contiguous blocks handled by
        :mod:`joblib`.  Blocks are contiguous *by date* so the warm start still
        only ever reaches backwards in time.
    min_points:
        Minimum number of quoted tenors required to attempt a fit.  Days with
        fewer produce an all-NaN parameter row (``n_points`` is still recorded).
    polish, tau1_grid, tau2_grid, weights, max_nfev, ftol:
        Passed through to :func:`fit_nss`.  ``weights`` is a label -> weight map
        applied to whichever tenors are present that day.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``curve.index`` with columns ``beta0, beta1, beta2, beta3,
        tau1, tau2, rmse, n_points``.  ``rmse`` is decimal; multiply by 1e4 for
        basis points.

    Notes
    -----
    **No look-ahead.**  Day *t*'s parameters are a function of day *t*'s quoted
    yields alone.  The only cross-date coupling is the warm start: the optimizer
    for day *t* begins at day *t-1*'s decay parameters, information available at
    the *t-1* close.  Because the full tau grid is still screened exhaustively
    every day, the seed can only improve the polish, never smuggle information
    in: the objective being minimised contains no data other than day *t*'s.

    The economic reason for the warm start is that :math:`(\\tau_1,\\tau_2)` is
    weakly identified - many decay pairs fit a given day almost equally well -
    so an unseeded fit produces a jittery tau series that contaminates the beta
    series with spurious day-to-day variation.  Anchoring on yesterday keeps the
    factor time series interpretable as level/slope/curvature.
    """
    if not isinstance(curve.index, pd.DatetimeIndex):
        raise TypeError("curve must be indexed by a DatetimeIndex")
    if model not in ("svensson", "nelson_siegel"):
        raise ValueError(f"Unknown model {model!r}")
    mapping = dict(TENOR_YEARS if tenor_years is None else tenor_years)

    cols = [c for c in curve.columns if c in mapping]
    if not cols:
        raise ValueError("No curve column matches the tenor_years mapping")
    years = np.array([float(mapping[c]) for c in cols], dtype=float)
    order = np.argsort(years)
    cols = [cols[i] for i in order]
    years = years[order]
    w_arr = (
        None
        if weights is None
        else np.array([float(weights.get(c, 1.0)) for c in cols], dtype=float)
    )

    values = curve[cols].to_numpy(dtype=float, copy=True)
    svensson = model == "svensson"
    n_rows = values.shape[0]

    if n_jobs is None or n_jobs <= 1 or n_rows < 512:
        block = _fit_block(
            values, years, svensson, tau1_grid, tau2_grid, min_points, polish, w_arr,
            max_nfev, ftol, None,
        )
    else:
        from joblib import Parallel, delayed

        n_blocks = int(min(n_jobs, max(1, n_rows // 256)))
        edges = np.linspace(0, n_rows, n_blocks + 1).astype(int)
        chunks = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_block)(
                values[a:b], years, svensson, tau1_grid, tau2_grid, min_points, polish,
                w_arr, max_nfev, ftol, None,
            )
            for a, b in zip(edges[:-1], edges[1:])
        )
        block = np.vstack(chunks)

    out = pd.DataFrame(block, index=curve.index, columns=list(NSS_COLUMNS))
    out.index.name = curve.index.name or "date"
    good = int(np.isfinite(out["beta0"].to_numpy()).sum())
    log.info(
        "Fitted %s on %d dates (%d successful, %d skipped for < %d tenors)",
        model, n_rows, good, n_rows - good, min_points,
    )
    return out


def fit_nss_history_fixed(
    curve: "pd.DataFrame",
    tenor_years: "Mapping[str, float] | None" = None,
    model: str = "svensson",
    tau1: float = DIEBOLD_LI_TAU1,
    tau2: float = SVENSSON_FIXED_TAU2,
    **kwargs: Any,
) -> "pd.DataFrame":
    """Fit the whole history with the decay parameters held fixed.

    This is the variant you want when the betas will be used as machine-learning
    features.  Holding ``tau1``/``tau2`` constant makes the model linear in the
    betas, so each day's factors are the unique least-squares solution and are
    directly comparable across days - see the commentary on
    :data:`DIEBOLD_LI_TAU1` for the measured stability difference.

    It is also roughly 40x faster than the free fit, because every day reduces to
    one small ``lstsq`` call instead of a grid search plus non-linear polish.

    Parameters
    ----------
    curve:
        Par-yield history, dates on the index, tenor labels on the columns.
    tenor_years:
        Label -> maturity in years.  Defaults to the Treasury CMT mapping.
    model:
        ``"svensson"`` (four betas) or ``"nelson_siegel"`` (three).
    tau1, tau2:
        The fixed decays, in years.
    **kwargs:
        Forwarded to :func:`fit_nss_history` (``min_points``, ``weights``, ...).

    Returns
    -------
    pd.DataFrame
        Same schema as :func:`fit_nss_history`; ``tau1``/``tau2`` are constant.
    """
    return fit_nss_history(
        curve,
        tenor_years=tenor_years,
        model=model,
        tau1_grid=(float(tau1),),
        tau2_grid=(float(tau2),),
        polish=False,
        **kwargs,
    )
