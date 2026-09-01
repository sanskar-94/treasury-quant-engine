"""Hamilton-style Markov regime switching for rates.

Why this module exists
----------------------
The carry benchmark in this project is the motivating measurement. A mechanical
relative-value carry book returned **+1.79% in 2020** when 2s10s was +80bp,
**-4.29% in 2022** when the hiking cycle inverted the curve to -53bp, and
**+2.05% in 2025** back at +71bp. Averaged over the window it looks like a
factor that does not work. Year by year it is obvious that it is a factor that
works in one state of the world and fails in another: a carry book is long the
carry-rich long end, and the long end is exactly what sells off when the front
end is repriced upwards.

Averaging incompatible regimes is the modelling error. A single linear model
fitted across both states estimates a coefficient that is right in neither, and
its residual variance is dominated by the regime it is not currently in. The fix
is to make the state an explicit latent variable and let the parameters switch
with it - Hamilton (1989), *A New Approach to the Economic Analysis of
Nonstationary Time Series and the Business Cycle*.

What is implemented
-------------------
A Gaussian hidden Markov model estimated by Baum-Welch EM, from first
principles:

* :func:`fit_hmm` - forward-backward with **scaling**, and the M-step in closed
  form. The scaling matters more than it sounds: the unscaled recursion
  multiplies a density by a transition probability at every step, so the
  forward variable decays geometrically and hits the double-precision floor
  (~1e-308) after a few hundred observations. On 5,000 daily yield changes the
  unscaled version silently returns zeros, a log-likelihood of ``-inf``, and
  posterior probabilities of ``nan`` - it does not raise, which is what makes it
  dangerous. Every recursion here is normalised at each step and the
  log-likelihood is accumulated from the normalisers.
* :func:`viterbi` - the single most likely state *path*, which is not the same
  object as the sequence of per-date most likely states.
* :func:`rolling_regime_probs` - the causal, tradable version.
* :class:`RegimeSwitchingModel` - one sub-learner per regime, predictions
  blended by the regime probability.

Filtered versus smoothed: the whole game
----------------------------------------
The forward-backward algorithm produces two posteriors over the state:

``filtered``   P(s_t | y_1 ... y_t)     - conditions on the past and today
``smoothed``   P(s_t | y_1 ... y_T)     - conditions on the **entire sample**

Smoothed probabilities are the right answer for a historical narrative ("when
did the 2022 regime actually begin?") and are **look-ahead by construction** for
anything traded. The smoother propagates information backwards: the smoothed
probability on 2022-01-03 is influenced by what happened in October 2022. Any
strategy conditioned on it is reading a newspaper from the future, and because
the effect is a gentle sharpening rather than an obvious leak it produces a
plausible-looking Sharpe rather than an implausible one. Every published
regime-switching result that does not say which posterior it used should be
assumed to have used the wrong one.

Only :attr:`HMMResult.filtered_probs` is admissible as a signal, and
:func:`rolling_regime_probs` goes further: the parameters themselves are refitted
on a window that **ends strictly before** the date being labelled, so neither the
state estimate nor the parameters that produced it have seen the future.

Causality
---------
:func:`fit_hmm` and :func:`viterbi` are full-sample estimators and are *not*
causal - they are research tools, and using their output directly as a feature is
a leak. :func:`rolling_regime_probs` is the causal wrapper: for date ``t`` the
parameters come from ``y[t - window : t]`` (exclusive upper bound) and the state
estimate is the filtered probability, which uses observations up to and including
``t`` but never beyond it. Day ``t``'s label is therefore known at day ``t``'s
close, which is the same convention as every other feature block in this system;
:func:`tqe.features.builder.build_features` applies the single central
``shift(feature_lag)`` that moves it to prediction time. The test suite corrupts
the tail of the input by a factor of 50 and asserts that every earlier
probability is bit-identical.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .base import BaseModel

log = get_logger("models.regime_switching")

__all__ = [
    "HMMResult",
    "fit_hmm",
    "viterbi",
    "rolling_regime_probs",
    "RegimeSwitchingModel",
    "gaussian_log_emissions",
    "forward_filter",
]

_LOG_2PI = float(np.log(2.0 * np.pi))
# Floor for the forward normalisers. Emissions are rescaled so that the largest
# is exactly 1.0 at every date (see `_rescaled_emissions`), so a normaliser can
# only collapse if the transition matrix has an unreachable state; the floor
# turns that into a harmless uniform step instead of a divide-by-zero.
_TINY = 1e-300


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)  # numpy fields make a generated __eq__ ambiguous
class HMMResult:
    """A fitted Gaussian hidden Markov model and its posteriors.

    Attributes
    ----------
    means : np.ndarray, shape (n_states,)
        Conditional mean of the observation in each state, in the units of the
        input. For daily yield changes these are decimals, so ``1e-4`` is 1bp.
    variances : np.ndarray, shape (n_states,)
        Conditional variance in each state. The *ratio* of these is what a rates
        regime model is really estimating: the conditional means of daily yield
        changes are statistically indistinguishable from zero, while the
        volatilities differ by a factor of two or more between calm and stressed
        states.
    transition_matrix : np.ndarray, shape (n_states, n_states)
        ``P[i, j] = Pr(s_t = j | s_{t-1} = i)``. Rows sum to one. The diagonal is
        the economically interesting part: ``1 / (1 - P[i, i])`` is the expected
        duration of regime ``i`` in observations, and a regime model whose
        implied durations are a handful of days has found noise, not a regime.
    initial_probs : np.ndarray, shape (n_states,)
        Estimated distribution of the state at the first observation.
    filtered_probs : np.ndarray, shape (n_obs, n_states)
        ``P(s_t | y_1 ... y_t)`` - the only posterior that may be traded on.
    smoothed_probs : np.ndarray, shape (n_obs, n_states)
        ``P(s_t | y_1 ... y_T)`` - conditions on the whole sample, so it is
        look-ahead by construction. For description and diagnosis only.
    log_likelihood : float
        Exact log-likelihood of the data under the fitted model, accumulated from
        the forward normalisers. Finite by construction at any sample length.
    n_iter : int
        EM iterations actually run.
    converged : bool
        Whether the relative change in the log-likelihood fell below ``tol``
        before ``max_iter`` was exhausted.
    states_sorted_by : str
        ``"mean"`` or ``"variance"`` - the criterion used to fix the labelling.
        State identity in an HMM is arbitrary (the likelihood is invariant to
        permutation), so an unsorted model relabels its states between refits
        and the resulting "feature" is a coin flip.
    """

    means: np.ndarray
    variances: np.ndarray
    transition_matrix: np.ndarray
    initial_probs: np.ndarray
    filtered_probs: np.ndarray
    smoothed_probs: np.ndarray
    log_likelihood: float
    n_iter: int
    converged: bool
    states_sorted_by: str

    @property
    def n_states(self) -> int:
        return int(len(self.means))

    @property
    def volatilities(self) -> np.ndarray:
        """Conditional standard deviations - the natural units for a rates desk."""
        return np.sqrt(self.variances)

    def expected_durations(self) -> np.ndarray:
        """Expected regime length in observations, ``1 / (1 - p_ii)``.

        The sojourn time of a Markov chain in state ``i`` is geometric with
        success probability ``1 - p_ii``, so its mean is ``1 / (1 - p_ii)``. On
        daily data this reads directly in trading days, and it is the sanity
        check that separates a regime from a fitted artefact: a "regime" lasting
        three days is a volatility spike with a Greek letter attached.
        """
        p_ii = np.clip(np.diag(self.transition_matrix), 0.0, 1.0 - 1e-12)
        return 1.0 / (1.0 - p_ii)

    def stationary_distribution(self) -> np.ndarray:
        """Unconditional state probabilities - the left eigenvector of ``P``.

        Solved as the constrained linear system ``(P' - I) pi = 0``,
        ``sum(pi) = 1`` rather than by eigendecomposition, which returns complex
        arithmetic and an arbitrary normalisation for a real problem with a known
        answer.
        """
        k = self.n_states
        a = np.vstack([self.transition_matrix.T - np.eye(k), np.ones(k)])
        b = np.zeros(k + 1)
        b[-1] = 1.0
        pi, *_ = np.linalg.lstsq(a, b, rcond=None)
        pi = np.clip(pi, 0.0, None)
        total = pi.sum()
        return pi / total if total > 0 else np.full(k, 1.0 / k)

    def summary(self) -> pd.DataFrame:
        """One row per state: mean, volatility, persistence, duration, weight."""
        return pd.DataFrame(
            {
                "mean": self.means,
                "volatility": self.volatilities,
                "persistence": np.diag(self.transition_matrix),
                "expected_duration": self.expected_durations(),
                "stationary_prob": self.stationary_distribution(),
                "smoothed_share": self.smoothed_probs.mean(axis=0),
            },
            index=pd.Index(range(self.n_states), name="state"),
        )


# --------------------------------------------------------------------------- #
# Emissions and the scaled recursions
# --------------------------------------------------------------------------- #
def _as_1d(y: pd.Series | np.ndarray | Sequence[float]) -> np.ndarray:
    """Coerce an observation series to a finite 1-D float array."""
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"expected a single observation series, got {y.shape[1]} columns")
        y = y.iloc[:, 0]
    arr = y.to_numpy(dtype=float) if isinstance(y, (pd.Series, pd.Index)) else np.asarray(y, dtype=float)
    arr = np.ravel(arr)
    if arr.size == 0:
        raise ValueError("cannot fit an HMM to an empty series")
    if not np.isfinite(arr).all():
        raise ValueError("observation series contains non-finite values; drop or impute them first")
    return arr


def gaussian_log_emissions(y: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Log density ``log N(y_t; mu_k, sigma^2_k)`` for every date and state.

    Parameters
    ----------
    y : np.ndarray, shape (n_obs,)
    means, variances : np.ndarray, shape (n_states,)

    Returns
    -------
    np.ndarray, shape (n_obs, n_states)

    Notes
    -----
    Computed in logs. A stressed-regime observation is routinely 8 standard
    deviations away from the calm regime's mean under the calm regime's
    variance; ``exp(-32)`` is representable but the products that follow are not.
    """
    z = y[:, None] - means[None, :]
    return -0.5 * (_LOG_2PI + np.log(variances)[None, :] + z * z / variances[None, :])


def _rescaled_emissions(log_b: np.ndarray) -> tuple[np.ndarray, float]:
    """Exponentiate log-emissions after removing a per-date constant.

    Subtracting the per-date maximum before exponentiating multiplies the
    emission vector at date ``t`` by a constant that is the same for every state.
    Every posterior (filtered, smoothed, the pairwise xi) is a ratio in which
    that constant cancels exactly, so the recursions are unchanged; the
    log-likelihood shifts by the sum of the removed constants, which is added
    back. The payoff is that the largest emission is exactly 1.0 at every date,
    which is what makes the forward normaliser impossible to underflow.

    Returns
    -------
    (b, offset) : (np.ndarray, float)
        Rescaled emissions in ``[0, 1]`` and the total log offset removed.
    """
    row_max = log_b.max(axis=1)
    return np.exp(log_b - row_max[:, None]), float(row_max.sum())


def _scaled_forward(b: np.ndarray, trans: np.ndarray, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scaled forward recursion.

    Returns ``(alpha, scale)`` where ``alpha[t]`` is the normalised forward
    variable - which *is* the filtered posterior ``P(s_t | y_1..y_t)`` - and
    ``scale[t]`` is the normaliser, whose logs sum to the log-likelihood.

    The recursion is sequential by definition (``alpha[t]`` depends on
    ``alpha[t-1]``), so the loop over dates is irreducible; the work inside it is
    a single ``k``-vector matrix product.
    """
    n, k = b.shape
    alpha = np.empty((n, k))
    scale = np.empty(n)

    a = pi * b[0]
    s = a.sum()
    scale[0] = s if s > _TINY else _TINY
    alpha[0] = a / scale[0] if s > _TINY else np.full(k, 1.0 / k)

    for t in range(1, n):
        a = (alpha[t - 1] @ trans) * b[t]
        s = a.sum()
        if s > _TINY:
            scale[t] = s
            alpha[t] = a / s
        else:  # unreachable in a well-formed model; keep it defined rather than nan
            scale[t] = _TINY
            alpha[t] = np.full(k, 1.0 / k)
    return alpha, scale


def _scaled_backward(b: np.ndarray, trans: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Scaled backward recursion, using the forward pass's own normalisers.

    Reusing ``scale`` rather than normalising the backward variable
    independently is what makes ``alpha[t] * beta[t]`` proportional to the
    smoothed posterior with a proportionality constant of one.
    """
    n, k = b.shape
    beta = np.ones((n, k))
    for t in range(n - 2, -1, -1):
        beta[t] = trans @ (b[t + 1] * beta[t + 1]) / scale[t + 1]
    return beta


def forward_filter(
    y: pd.Series | np.ndarray | Sequence[float],
    means: np.ndarray,
    variances: np.ndarray,
    transition_matrix: np.ndarray,
    initial_probs: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Filtered probabilities and exact log-likelihood for given parameters.

    Exposed separately because the causal wrapper needs to run the filter under
    parameters that were estimated on an *earlier* window - the one operation
    that :func:`fit_hmm` does not provide.

    Returns
    -------
    (filtered, log_likelihood) : (np.ndarray of shape (n_obs, n_states), float)
    """
    obs = _as_1d(y)
    b, offset = _rescaled_emissions(gaussian_log_emissions(obs, means, variances))
    alpha, scale = _scaled_forward(b, transition_matrix, initial_probs)
    return alpha, float(np.log(scale).sum() + offset)


# --------------------------------------------------------------------------- #
# Baum-Welch
# --------------------------------------------------------------------------- #
def _initial_parameters(
    y: np.ndarray, n_states: int, seed: int, attempt: int, var_floor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic starting values for EM.

    Attempt 0 partitions the sorted observations into ``n_states`` equal
    quantile groups and takes each group's moments. For a mixture of Gaussians
    that differ mainly in scale this lands close to the answer immediately, and
    it is fully deterministic - no seed dependence at all, which is what a
    rolling refit wants. Later attempts perturb it with the seeded generator so
    that ``n_init > 1`` explores rather than repeats.

    The transition matrix starts strongly persistent (0.95 on the diagonal)
    because that is the truth for every financial regime worth the name, and
    because a near-uniform start lets EM converge to a degenerate "regime"
    that flips every other day.
    """
    order = np.sort(y)
    chunks = np.array_split(order, n_states)
    means = np.array([c.mean() for c in chunks])
    variances = np.array([max(c.var(), var_floor) for c in chunks])

    if attempt > 0:
        rng = np.random.default_rng(seed + attempt)
        spread = float(np.std(y)) or 1.0
        means = means + rng.normal(0.0, 0.25 * spread, n_states)
        variances = variances * np.exp(rng.normal(0.0, 0.25, n_states))
        variances = np.maximum(variances, var_floor)

    off = 0.05 / max(1, n_states - 1)
    trans = np.full((n_states, n_states), off)
    np.fill_diagonal(trans, 0.95)
    trans /= trans.sum(axis=1, keepdims=True)
    pi = np.full(n_states, 1.0 / n_states)
    return means, variances, trans, pi


def _sort_order(means: np.ndarray, variances: np.ndarray, sort_by: str) -> np.ndarray:
    if sort_by == "mean":
        return np.argsort(means, kind="stable")
    if sort_by == "variance":
        return np.argsort(variances, kind="stable")
    raise ValueError(f"sort_by must be 'mean' or 'variance', got {sort_by!r}")


def fit_hmm(
    y: pd.Series | np.ndarray | Sequence[float],
    n_states: int = 2,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
    *,
    sort_by: str = "mean",
    n_init: int = 1,
    var_floor_frac: float = 1e-6,
) -> HMMResult:
    """Fit a Gaussian hidden Markov model by Baum-Welch EM.

    Parameters
    ----------
    y : pd.Series or array-like, shape (n_obs,)
        Observation series. Must be finite - drop NaNs before calling. For rates
        work this is a *change* series (daily yield changes, structure returns),
        never a level: a level series is non-stationary and the HMM will label
        the 1990s and the 2020s as different "regimes" purely on mean.
    n_states : int, default 2
        Number of latent regimes. Two is the honest default on daily data; each
        extra state costs ``2 n_states + n_states^2`` parameters and daily rate
        changes do not support many.
    max_iter : int, default 200
    tol : float, default 1e-6
        Convergence threshold on the **relative** change in the log-likelihood,
        ``|L - L'| <= tol * (1 + |L'|)``. Relative, because the log-likelihood of
        9,000 daily yield changes measured in decimals is around +60,000 while
        the same series in basis points is around -20,000; an absolute threshold
        means something different in each and would silently stop early or never
        stop at all depending on units.
    seed : int, default 42
        Seeds the perturbations used when ``n_init > 1``. The default single
        start is deterministic regardless of the seed.
    sort_by : {"mean", "variance"}, default "mean"
        Criterion for fixing the state labelling. See notes.
    n_init : int, default 1
        Number of EM restarts; the fit with the highest log-likelihood wins. EM
        finds a local optimum, so restarts are insurance, not decoration.
    var_floor_frac : float, default 1e-6
        Variance floor as a fraction of the sample variance. Expressed relative
        to the scale of the data on purpose: this project has twice shipped a
        bug where an absolute constant was applied to quantities of order 1e-4,
        and a fixed floor of, say, 1e-8 is a no-op on returns in percent and a
        hard constraint on the same returns in decimals. The floor prevents the
        classic Gaussian-mixture degeneracy where one state collapses onto a
        single observation and drives the likelihood to infinity.

    Returns
    -------
    HMMResult

    Notes
    -----
    **Causality.** This is a full-sample estimator: every returned quantity has
    seen every observation. It is a research and diagnosis tool. Using
    ``filtered_probs`` from a full-sample fit as a trading feature still leaks,
    because the *parameters* were estimated on the future even though the state
    estimate was not. :func:`rolling_regime_probs` is the causal version.

    **Scaling.** The forward and backward recursions are normalised at every
    date and the log-likelihood is accumulated from the normalisers, so the
    estimator is numerically identical at 100 observations and at 100,000. The
    unscaled textbook recursion underflows to exactly zero after a few hundred
    dates and returns ``nan`` posteriors without raising.

    **State identity.** The likelihood is invariant to permuting the states, so
    an unconstrained fit labels them arbitrarily. States are sorted after
    convergence, and the transition matrix is permuted on **both** axes to match.
    Sort by ``"mean"`` when the states are separated by drift; sort by
    ``"variance"`` for financial regimes, where the conditional means are noise
    and the volatilities are not - a mean-sorted daily-yield-change model
    permutes its own labels between refits and produces a feature that is
    uncorrelated with itself.
    """
    obs = _as_1d(y)
    n = obs.size
    if n_states < 1:
        raise ValueError("n_states must be >= 1")
    if n < 2 * n_states:
        raise ValueError(f"need at least {2 * n_states} observations for {n_states} states, got {n}")

    sample_var = float(np.var(obs))
    if sample_var <= 0.0:
        # A constant series has no regimes to find, and every Gaussian density
        # diverges as the variance is driven to zero. Fail loudly here rather
        # than return a model with an infinite log-likelihood.
        raise ValueError("observation series is constant; there is no regime structure to identify")
    var_floor = sample_var * float(var_floor_frac)

    best: dict[str, Any] | None = None
    for attempt in range(max(1, int(n_init))):
        means, variances, trans, pi = _initial_parameters(obs, n_states, seed, attempt, var_floor)
        prev_ll = -np.inf
        converged = False
        used_iter = 0

        for it in range(1, int(max_iter) + 1):
            used_iter = it
            # ---- E step -------------------------------------------------- #
            b, offset = _rescaled_emissions(gaussian_log_emissions(obs, means, variances))
            alpha, scale = _scaled_forward(b, trans, pi)
            beta = _scaled_backward(b, trans, scale)
            ll = float(np.log(scale).sum() + offset)

            gamma = alpha * beta
            gamma /= np.clip(gamma.sum(axis=1, keepdims=True), _TINY, None)

            # Pairwise posterior, summed over time in one matrix product:
            #   xi_sum[i, j] = sum_t alpha[t, i] P[i, j] b[t+1, j] beta[t+1, j] / scale[t+1]
            # The time sum is an outer-product accumulation, so it vectorises
            # completely - no second loop over 9,000 dates.
            tail = b[1:] * beta[1:] / scale[1:, None]
            xi_sum = trans * (alpha[:-1].T @ tail)

            # ---- M step -------------------------------------------------- #
            row = xi_sum.sum(axis=1, keepdims=True)
            trans = np.where(row > _TINY, xi_sum / np.clip(row, _TINY, None), trans)
            pi = gamma[0].copy()

            weight = gamma.sum(axis=0)
            safe = weight > _TINY
            new_means = np.where(safe, (gamma * obs[:, None]).sum(axis=0) / np.clip(weight, _TINY, None), means)
            resid2 = (obs[:, None] - new_means[None, :]) ** 2
            new_vars = np.where(safe, (gamma * resid2).sum(axis=0) / np.clip(weight, _TINY, None), variances)
            means, variances = new_means, np.maximum(new_vars, var_floor)

            # The first iteration has no previous likelihood to compare against.
            # Testing anyway is not harmless: with prev_ll = -inf the criterion
            # evaluates to `inf <= inf`, which is True, so EM declared success
            # after a single pass and returned essentially its initialisation.
            # The symptom was subtle - Viterbi still recovered 97.8% of states
            # from a good init - but the transition matrix was badly wrong
            # (0.98 estimated as 0.925, a regime duration of 50 days reported as
            # 13) and on real data both states collapsed to the same volatility.
            if np.isfinite(prev_ll) and abs(ll - prev_ll) <= tol * (1.0 + abs(prev_ll)):
                converged = True
                break
            prev_ll = ll

        # Final E step so the reported posteriors correspond to the reported
        # parameters rather than to the previous iteration's.
        b, offset = _rescaled_emissions(gaussian_log_emissions(obs, means, variances))
        alpha, scale = _scaled_forward(b, trans, pi)
        beta = _scaled_backward(b, trans, scale)
        ll = float(np.log(scale).sum() + offset)
        smoothed = alpha * beta
        smoothed /= np.clip(smoothed.sum(axis=1, keepdims=True), _TINY, None)

        if best is None or ll > best["ll"]:
            best = {
                "ll": ll, "means": means, "variances": variances, "trans": trans,
                "pi": pi, "filtered": alpha, "smoothed": smoothed,
                "n_iter": used_iter, "converged": converged,
            }

    assert best is not None  # noqa: S101 - the loop runs at least once
    order = _sort_order(best["means"], best["variances"], sort_by)
    return HMMResult(
        means=best["means"][order],
        variances=best["variances"][order],
        # Permute both axes: P[i, j] is a joint statement about two states.
        transition_matrix=best["trans"][np.ix_(order, order)],
        initial_probs=best["pi"][order],
        filtered_probs=best["filtered"][:, order],
        smoothed_probs=best["smoothed"][:, order],
        log_likelihood=best["ll"],
        n_iter=int(best["n_iter"]),
        converged=bool(best["converged"]),
        states_sorted_by=sort_by,
    )


def viterbi(y: pd.Series | np.ndarray | Sequence[float], result: HMMResult) -> np.ndarray:
    """Most likely state *path* under a fitted model.

    Parameters
    ----------
    y : pd.Series or array-like, shape (n_obs,)
    result : HMMResult
        Supplies the parameters. The returned labels use ``result``'s sorted
        state convention.

    Returns
    -------
    np.ndarray of int, shape (n_obs,)

    Notes
    -----
    This is not ``argmax(smoothed_probs, axis=1)``. That gives the most likely
    state at each date taken one at a time, and the resulting sequence can be
    one the chain cannot produce - it will happily jump through a transition of
    probability zero. Viterbi maximises the joint probability of the whole path,
    so what comes out is a sequence the model could actually have generated.
    For regime *dating* that distinction is the difference between a clean
    history and one that flickers.

    Run entirely in logs, which for a maximisation is exact rather than merely
    stable: no normalisers are needed because there is no sum to renormalise.
    """
    obs = _as_1d(y)
    log_b = gaussian_log_emissions(obs, result.means, result.variances)
    with np.errstate(divide="ignore"):
        log_a = np.log(result.transition_matrix)
        log_pi = np.log(np.clip(result.initial_probs, _TINY, None))

    n, k = log_b.shape
    delta = np.empty((n, k))
    psi = np.zeros((n, k), dtype=int)
    delta[0] = log_pi + log_b[0]
    for t in range(1, n):
        scores = delta[t - 1][:, None] + log_a  # (from, to)
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores[psi[t], np.arange(k)] + log_b[t]

    path = np.empty(n, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


# --------------------------------------------------------------------------- #
# The causal wrapper - the only version that may touch a signal
# --------------------------------------------------------------------------- #
def rolling_regime_probs(
    y: pd.Series,
    n_states: int = 2,
    window: int = 1260,
    min_periods: int = 504,
    refit_every: int = 63,
    *,
    sort_by: str = "variance",
    seed: int = 42,
    max_iter: int = 100,
    tol: float = 1e-6,
    prefix: str = "hmm",
) -> pd.DataFrame:
    """Causal regime probabilities from a rolling HMM refit.

    Parameters
    ----------
    y : pd.Series
        Observation series with a ``DatetimeIndex``. NaNs are dropped for the
        estimation and the output is reindexed back onto ``y.index``.
    n_states : int, default 2
    window : int, default 1260
        Estimation window in observations - five trading years, long enough to
        contain both a calm and a stressed episode, short enough that the model
        is not still describing the taper tantrum in 2026.
    min_periods : int, default 504
        Observations required before any label is produced. Two years: an HMM
        estimating ``2 n + n^2`` parameters from one year of daily data fits the
        window, not the market.
    refit_every : int, default 63
        Refit cadence in observations (a quarter). Between refits the parameters
        are held fixed and the filter is simply run forward, which is exactly
        what a desk would do and is ~20x cheaper than refitting daily.
    sort_by : {"variance", "mean"}, default "variance"
        Defaults to variance here, unlike :func:`fit_hmm`. Across a rolling
        refit the labelling has to be comparable through time, and for daily
        rate changes the conditional means are indistinguishable from zero while
        the conditional volatilities differ by a factor of two or more. Sorting
        on the noisier statistic permutes state identity between refits and
        turns the feature into noise.
    seed, max_iter, tol
        Passed to :func:`fit_hmm`. ``max_iter`` is lower than the default
        because there are ~140 refits over the full history and the parameters
        barely move between them.
    prefix : str, default "hmm"
        Column-name prefix.

    Returns
    -------
    pd.DataFrame
        Indexed like ``y``. Columns ``{prefix}_p0 ... {prefix}_p{k-1}`` (filtered
        probabilities) plus ``{prefix}_state`` (the argmax) and
        ``{prefix}_hi_vol`` (probability of the highest-variance state, the one
        a risk overlay actually wants). Rows before ``min_periods`` are NaN.

    Notes
    -----
    **This function is where the causality of the whole module lives.** Two
    separate boundaries are enforced:

    1. *Parameters.* The model labelling date ``t`` is fitted on
       ``values[t - window : t]``. The upper bound is **exclusive**; date ``t``
       is out of sample for its own parameters, as is everything after it.
    2. *State estimate.* The **filtered** probability is used, never the
       smoothed one. Smoothing runs a backward pass over the sample and so
       conditions date ``t`` on dates after ``t``; the resulting series would
       identify regime turning points before they happened. Filtering
       conditions only on ``y_1 ... y_t``.

    After each refit the filter is restarted from the beginning of the
    estimation window under the new parameters and run forward to ``t``; between
    refits it is advanced one step per date. Both paths touch only observations
    at or before the date being labelled, so corrupting the future leaves every
    earlier probability bit-identical - which the test suite asserts.

    Day ``t``'s probability uses day ``t``'s observation and is therefore known
    at day ``t``'s close, the same convention as every other feature block here.
    The single central ``shift(feature_lag)`` in
    :func:`tqe.features.builder.build_features` moves it to prediction time.
    Applying a second shift is how this project once threw away three quarters
    of its signal.
    """
    if not isinstance(y, pd.Series):
        raise TypeError("rolling_regime_probs expects a pd.Series with a DatetimeIndex")
    if min_periods < 2 * n_states:
        raise ValueError(f"min_periods={min_periods} is too small for {n_states} states")
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")

    cols = [f"{prefix}_p{i}" for i in range(n_states)]
    clean = y.dropna()
    values = clean.to_numpy(dtype=float)
    n = values.size
    out = np.full((n, n_states), np.nan)

    if n <= min_periods:
        frame = pd.DataFrame(out, index=clean.index, columns=cols)
        frame[f"{prefix}_state"] = np.nan
        frame[f"{prefix}_hi_vol"] = np.nan
        return frame.reindex(y.index)

    params: HMMResult | None = None
    alpha_prev: np.ndarray | None = None
    last_fit = -(10**9)
    n_failed = 0

    for t in range(min_periods, n):
        if params is None or (t - last_fit) >= refit_every:
            start = max(0, t - window)
            train = values[start:t]  # EXCLUSIVE of t - the causality bound
            try:
                candidate = fit_hmm(
                    train, n_states=n_states, max_iter=max_iter, tol=tol,
                    seed=seed, sort_by=sort_by,
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                n_failed += 1
                log.debug("HMM refit at position %d failed (%s); carrying previous fit", t, exc)
                candidate = None
            if candidate is not None:
                params = candidate
                last_fit = t
                # Warm-start: re-run the filter over the estimation window under
                # the new parameters so that alpha_prev is P(s_{t-1} | window).
                # Purely historical data - no observation at or after t is used.
                alpha_prev = params.filtered_probs[-1] if len(train) else params.initial_probs
        if params is None or alpha_prev is None:
            continue

        # One filter step onto date t. Emission is rescaled by its own max, which
        # cancels in the normalisation, so this cannot underflow either.
        log_b = gaussian_log_emissions(values[t : t + 1], params.means, params.variances)
        b = np.exp(log_b - log_b.max())[0]
        a = (alpha_prev @ params.transition_matrix) * b
        s = a.sum()
        alpha_prev = a / s if s > _TINY else np.full(n_states, 1.0 / n_states)
        out[t] = alpha_prev

    if n_failed:
        log.info("rolling_regime_probs: %d refits failed and reused the previous fit", n_failed)

    frame = pd.DataFrame(out, index=clean.index, columns=cols)
    valid = np.isfinite(out).all(axis=1)
    state = np.full(n, np.nan)
    state[valid] = np.argmax(out[valid], axis=1)
    frame[f"{prefix}_state"] = state
    # With sort_by="variance" the last column IS the highest-variance state by
    # construction; name it explicitly so downstream code never has to guess
    # which index the risk overlay should read.
    if sort_by == "variance" or params is None:
        hi = n_states - 1
    else:
        hi = int(np.argmax(params.variances))
    frame[f"{prefix}_hi_vol"] = frame[cols[hi]]
    frame.index.name = y.index.name or "date"
    return frame.reindex(y.index)


# --------------------------------------------------------------------------- #
# The learner
# --------------------------------------------------------------------------- #
def _default_submodel() -> BaseModel:
    """Default per-regime learner: ridge with a CV-selected penalty.

    A module-level function rather than a lambda so that :meth:`BaseModel.save`
    can pickle the fitted object.
    """
    from .linear import RidgeModel

    return RidgeModel()


class RegimeSwitchingModel(BaseModel):
    """One sub-learner per regime, blended by the regime probability.

    The economics
    -------------
    Carry made +1.79% in 2020 and -4.29% in 2022. A single model fitted across
    both estimates the average of two incompatible relationships and is wrong in
    each. This learner fits a separate model per regime and predicts

    .. math:: \\hat{y}_t = \\sum_k P(s_t = k \\mid \\mathcal{F}_t)\\, \\hat{y}_{t,k}

    which is the conditional expectation under the regime posterior. When the
    regime is unambiguous the blend collapses to that regime's model; when it is
    genuinely uncertain the prediction is shrunk towards the average of the
    regimes' views, which is the correct behaviour and something a hard
    switch does not do.

    Where the regime comes from
    ---------------------------
    The probabilities are read from **columns of the design matrix**, not
    computed inside the model. That is deliberate and follows the project's rule
    that there is one causality boundary: the probabilities are produced by
    :func:`rolling_regime_probs` in the feature layer, lagged once by
    :func:`tqe.features.builder.build_features` along with everything else, and
    the learner never touches a raw time series. A model that recomputed the
    regime internally at predict time would need the full history at inference
    and would be a second place for look-ahead to enter.

    Parameters
    ----------
    model_factory : callable, optional
        Zero-argument factory returning an unfitted :class:`BaseModel`, called
        once per regime. Defaults to :class:`~tqe.models.linear.RidgeModel`.
    regime_cols : sequence of str or int, optional
        Columns holding the regime probabilities. Names require ``X`` to be a
        DataFrame at fit time; integers index positionally. If omitted, columns
        whose name starts with ``prefix`` are used.
    prefix : str, default "hmm_p"
        Auto-detection prefix.
    drop_regime_cols : bool, default True
        Exclude the probability columns from the sub-models' own design
        matrices. Within a regime the probability is close to constant, so it
        contributes collinearity rather than information, and leaving it in lets
        a sub-model rediscover the switch it has already been conditioned on.
    min_obs_per_state : int, default 120
        A regime with fewer training rows than this falls back to a model fitted
        on the pooled sample. Half a trading year is the minimum at which a
        regime-specific fit is worth more than the extra estimation error, and
        without the guard a rare third state gets a model fitted on nine
        observations and dominates the blend whenever it fires.

    Attributes
    ----------
    models_ : list[BaseModel]
        Fitted sub-model per regime.
    pooled_model_ : BaseModel
        Fitted on every row; the fallback for thin regimes.
    state_counts_ : np.ndarray
        Training rows hard-assigned to each regime.
    """

    name = "regime_switching"

    def __init__(
        self,
        model_factory: Callable[[], BaseModel] | None = None,
        regime_cols: Sequence[str | int] | None = None,
        prefix: str = "hmm_p",
        drop_regime_cols: bool = True,
        min_obs_per_state: int = 120,
        **kw: Any,
    ) -> None:
        super().__init__(
            prefix=prefix,
            drop_regime_cols=bool(drop_regime_cols),
            min_obs_per_state=int(min_obs_per_state),
            **kw,
        )
        # Kept off ``params`` so the JSON sidecar written by ``save`` stays
        # serialisable; both are restored by joblib with the rest of the object.
        self.model_factory: Callable[[], BaseModel] = model_factory or _default_submodel
        self.regime_cols = list(regime_cols) if regime_cols is not None else None
        self.models_: list[BaseModel] = []
        self.pooled_model_: BaseModel | None = None
        self.regime_idx_: list[int] = []
        self.state_counts_: np.ndarray = np.empty(0, dtype=int)
        self.pooled_states_: list[int] = []

    # -- helpers ----------------------------------------------------------- #
    def _resolve_regime_idx(self, n_features: int) -> list[int]:
        """Locate the probability columns positionally, once, at fit time."""
        names = self.feature_names_
        if self.regime_cols is not None:
            idx: list[int] = []
            for c in self.regime_cols:
                if isinstance(c, (int, np.integer)):
                    idx.append(int(c))
                elif names is None:
                    raise ValueError(
                        "regime_cols given as names but X was not a DataFrame at fit time"
                    )
                elif c not in names:
                    raise ValueError(f"regime column {c!r} is not in the design matrix")
                else:
                    idx.append(names.index(c))
        else:
            if names is None:
                raise ValueError(
                    "RegimeSwitchingModel needs either a DataFrame X (to auto-detect "
                    f"columns starting with {self.params['prefix']!r}) or explicit regime_cols"
                )
            idx = [i for i, c in enumerate(names) if str(c).startswith(self.params["prefix"])]
        if len(idx) < 2:
            raise ValueError(
                f"found {len(idx)} regime probability columns; at least 2 are required. "
                "Build them with rolling_regime_probs() in the feature layer."
            )
        if max(idx) >= n_features:
            raise ValueError("regime column index is out of range for the design matrix")
        return idx

    def _probabilities(self, X: np.ndarray) -> np.ndarray:
        """Extract and renormalise the regime posterior from the design matrix."""
        p = np.clip(X[:, self.regime_idx_], 0.0, None)
        total = p.sum(axis=1, keepdims=True)
        # A row whose probabilities are all zero (a NaN survived upstream, or the
        # feature was not yet warm) is treated as maximally uncertain rather than
        # being silently assigned to state 0.
        uniform = np.full(p.shape[1], 1.0 / p.shape[1])
        return np.where(total > 1e-12, p / np.clip(total, 1e-12, None), uniform)

    def _sub_design(self, X: np.ndarray) -> np.ndarray:
        if not self.params["drop_regime_cols"]:
            return X
        keep = np.setdiff1d(np.arange(X.shape[1]), np.asarray(self.regime_idx_, dtype=int))
        return X[:, keep]

    # -- BaseModel contract ------------------------------------------------ #
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.regime_idx_ = self._resolve_regime_idx(X.shape[1])
        probs = self._probabilities(X)
        n_states = probs.shape[1]
        design = self._sub_design(X)

        # Pooled fit first: it is both the fallback for thin regimes and the
        # control the regime split has to beat.
        self.pooled_model_ = self.model_factory().fit(design, y)

        # Hard assignment for *fitting* (the BaseModel contract has no
        # sample_weight, so soft weighting is not available for an arbitrary
        # sub-learner), soft weighting for *predicting*. Fitting on the argmax
        # and blending on the posterior is the standard practical compromise and
        # keeps every sub-model an ordinary, testable learner.
        assign = np.argmax(probs, axis=1)
        counts = np.bincount(assign, minlength=n_states)
        self.state_counts_ = counts
        self.models_ = []
        self.pooled_states_ = []
        floor = int(self.params["min_obs_per_state"])
        for k in range(n_states):
            rows = assign == k
            if counts[k] >= floor:
                self.models_.append(self.model_factory().fit(design[rows], y[rows]))
            else:
                log.info(
                    "regime %d has %d training rows (< %d); using the pooled model",
                    k, int(counts[k]), floor,
                )
                self.models_.append(self.pooled_model_)
                self.pooled_states_.append(k)
        self.n_targets_ = y.shape[1]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        probs = self._probabilities(X)
        design = self._sub_design(X)
        # (n_states, n_obs, n_targets) -> probability-weighted blend.
        stacked = np.stack([m.predict(design) for m in self.models_], axis=0)
        return np.einsum("nk,kno->no", probs, stacked)

    # -- introspection ----------------------------------------------------- #
    @property
    def feature_importance(self) -> pd.Series | None:
        """Regime-averaged importance, weighted by each regime's training share."""
        if not self.models_:
            return None
        parts = [(m.feature_importance, c) for m, c in zip(self.models_, self.state_counts_)]
        parts = [(imp, c) for imp, c in parts if imp is not None and c > 0]
        if not parts:
            return None
        total = sum(c for _, c in parts)
        agg = sum(imp * (c / total) for imp, c in parts)
        return agg.sort_values(ascending=False)

    def regime_summary(self) -> pd.DataFrame:
        """Training rows per regime and whether the regime got its own model."""
        return pd.DataFrame(
            {
                "train_rows": self.state_counts_,
                "share": self.state_counts_ / max(1, int(self.state_counts_.sum())),
                "own_model": [k not in self.pooled_states_ for k in range(len(self.state_counts_))],
            },
            index=pd.Index(range(len(self.state_counts_)), name="state"),
        )
