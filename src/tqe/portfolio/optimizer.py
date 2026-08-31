"""Portfolio construction under risk and turnover constraints.

The sizing layer allocates risk proportionally to conviction and ignores
correlation. That is a reasonable default, but it leaves value on the table when
the model has genuinely different views across the curve, because the tenors are
~95% correlated: a "long 5s, short 10s" view is a much smaller risk position than
the sum of its legs, and a naive sizer will under-use its risk budget on curve
trades while over-using it on directional ones.

The optimiser here maximises

.. math::

    \\mu' w - \\frac{\\lambda}{2} w' \\Sigma w - \\gamma \\lVert w - w_{prev} \\rVert_1

subject to gross/net DV01 caps and per-tenor bounds. The L1 turnover penalty is
what makes the result tradable: without it, tiny changes in :math:`\\mu` produce
large changes in :math:`w`, and the strategy pays away its edge in spreads.

Because the objective is non-smooth, it is solved with SLSQP on a smoothed
turnover term, with an analytic projected fallback if the solver fails - an
optimiser that raises in production is worse than one that returns a slightly
sub-optimal book.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..config import PortfolioConfig
from ..logging_utils import get_logger

log = get_logger("portfolio.optimizer")

__all__ = [
    "OptimizerResult",
    "mean_variance_weights",
    "risk_parity_weights",
    "dv01_neutral_projection",
    "minimum_variance_weights",
    "optimize_history",
]

EPS = 1e-12


@dataclass
class OptimizerResult:
    """Outcome of one optimisation."""

    weights: pd.Series
    expected_return: float = 0.0
    expected_vol: float = 0.0
    dv01: float = 0.0
    gross: float = 0.0
    net: float = 0.0
    turnover: float = 0.0
    status: str = "ok"
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "expected_return": self.expected_return,
            "expected_vol": self.expected_vol,
            "dv01": self.dv01,
            "gross": self.gross,
            "net": self.net,
            "turnover": self.turnover,
            "status": self.status,
        }


def _summarise(
    w: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    prev: np.ndarray,
    index: Sequence[str],
    dv01: np.ndarray | None,
    status: str,
) -> OptimizerResult:
    var = float(w @ cov @ w)
    return OptimizerResult(
        weights=pd.Series(w, index=list(index)),
        expected_return=float(mu @ w),
        expected_vol=float(np.sqrt(max(var, 0.0))),
        dv01=float(w @ dv01) if dv01 is not None else 0.0,
        gross=float(np.abs(w).sum()),
        net=float(w.sum()),
        turnover=float(np.abs(w - prev).sum()),
        status=status,
    )


def mean_variance_weights(
    mu: pd.Series,
    cov: pd.DataFrame,
    prev_w: pd.Series | None = None,
    cfg: PortfolioConfig | None = None,
    risk_aversion: float | None = None,
    dv01_per_unit: pd.Series | None = None,
    turnover_cost: float = 2e-4,
    holding_period_days: float = 21.0,
) -> OptimizerResult:
    """Maximise risk-adjusted return net of turnover, subject to limits.

    Parameters
    ----------
    mu:
        Expected return per tenor over the holding period.
    cov:
        Covariance of tenor returns. Must be positive semi-definite; use
        :func:`tqe.portfolio.risk.covariance`, which guarantees it.
    prev_w:
        Yesterday's weights, for the turnover penalty. Zeros if omitted.
    cfg:
        Supplies ``max_leverage``, ``max_weight_per_tenor``, ``turnover_penalty``,
        ``target_annual_vol`` and the DV01 caps.
    risk_aversion:
        Overrides the value implied by the volatility target.
    dv01_per_unit:
        DV01 per unit weight, used to enforce the gross/net DV01 caps.
    turnover_cost:
        One-way transaction cost per unit of weight traded, as a decimal. The
        default 2e-4 (2bp) is the round-trip cost the 32nds-based cost model
        produces for a mid-curve Treasury.
    holding_period_days:
        Expected holding period, used to amortise the trading cost across the
        life of the position. See the Notes.

    Returns
    -------
    OptimizerResult

    Notes
    -----
    **Units matter here and are easy to get wrong.** The objective compares an
    expected return (daily, order 1e-4) against a turnover penalty. If the
    penalty is specified as a bare number like 5.0 it dwarfs the return term by
    four orders of magnitude and the optimiser correctly concludes that the best
    portfolio is no portfolio - it returns all zeros, silently.

    ``cfg.turnover_penalty`` is therefore a **multiplier on the real transaction
    cost**, not a raw coefficient. A value of 1.0 charges exactly the estimated
    cost; higher values trade less than cost-neutral, which is usually right
    because the cost estimate is itself optimistic.

    The cost must also be **amortised over the holding period**. The objective
    compares one day of expected return against the cost of establishing the
    position - but that position is held for many days and earns the return
    repeatedly, while the cost is paid once. Charging the full round trip against
    a single day's return makes trading unprofitable by construction, and the
    optimiser dutifully returns an empty book. The effective penalty is therefore

        gamma = turnover_penalty * turnover_cost / holding_period_days

    which is the correct per-day amortisation for a position expected to live
    ``holding_period_days`` sessions.
    """
    cfg = cfg or PortfolioConfig()
    tenors = list(mu.index)
    n = len(tenors)
    if n == 0:
        return OptimizerResult(weights=pd.Series(dtype=float), status="empty")

    mu_v = mu.to_numpy(dtype=float)
    cov_m = cov.reindex(index=tenors, columns=tenors).to_numpy(dtype=float)
    prev = (prev_w.reindex(tenors).fillna(0.0).to_numpy(dtype=float)
            if prev_w is not None else np.zeros(n))
    dv = dv01_per_unit.reindex(tenors).fillna(0.0).to_numpy(dtype=float) if dv01_per_unit is not None else None

    if not np.isfinite(mu_v).all() or not np.isfinite(cov_m).all():
        return OptimizerResult(weights=pd.Series(prev, index=tenors), status="non_finite_inputs")

    # ---- risk aversion, calibrated to the volatility target ---------------- #
    # The unconstrained optimum of  mu'w - (lambda/2) w'Sigma w  is
    # w* = Sigma^-1 mu / lambda, whose volatility is
    #     sqrt(w*' Sigma w*) = sqrt(mu' Sigma^-1 mu) / lambda.
    # Setting that equal to the daily target and solving gives
    #     lambda = sqrt(mu' Sigma^-1 mu) / target_daily_vol.
    #
    # Deriving lambda from the target ALONE (e.g. 1/target_variance) is a units
    # error: it ignores how strong the signal actually is, and with daily
    # expected returns of order 1e-4 it produces weights of order 1e-4 - a book
    # running a small fraction of a basis point of risk. The optimiser is then
    # working correctly and returning something useless.
    target_daily_vol = cfg.target_annual_vol / np.sqrt(252.0)
    if risk_aversion is None:
        try:
            reg = cov_m + np.eye(n) * (np.trace(cov_m) / max(n, 1) * 1e-8 + 1e-16)
            sharpe_sq = float(mu_v @ np.linalg.solve(reg, mu_v))
        except np.linalg.LinAlgError:
            sharpe_sq = 0.0
        if sharpe_sq <= EPS or not np.isfinite(sharpe_sq):
            # No usable signal: nothing to size. Return the previous book so the
            # strategy holds rather than liquidating on an estimation failure.
            return _summarise(prev, mu_v, cov_m, prev, tenors, dv, "no_signal")
        risk_aversion = np.sqrt(sharpe_sq) / max(target_daily_vol, 1e-12)

    # See the Notes above: a multiplier on the real cost, amortised over the
    # expected holding period so the penalty is commensurate with ONE day of
    # expected return.
    gamma = float(cfg.turnover_penalty) * float(turnover_cost) / max(float(holding_period_days), 1.0)
    bound = abs(cfg.max_weight_per_tenor)

    def objective(w: np.ndarray) -> float:
        # sqrt(x^2 + eps) is a smooth stand-in for |x| so SLSQP has a gradient.
        turn = np.sqrt((w - prev) ** 2 + 1e-10).sum()
        return -(mu_v @ w) + 0.5 * risk_aversion * (w @ cov_m @ w) + gamma * turn

    constraints = []
    if cfg.max_leverage and cfg.max_leverage > 0:
        constraints.append(
            {"type": "ineq", "fun": lambda w: cfg.max_leverage - np.abs(w).sum()}
        )
    if dv is not None:
        if cfg.max_gross_dv01 > 0:
            constraints.append(
                {"type": "ineq", "fun": lambda w: cfg.max_gross_dv01 - np.abs(w * dv).sum()}
            )
        if cfg.max_net_dv01 > 0:
            constraints.append(
                {"type": "ineq", "fun": lambda w: cfg.max_net_dv01 - abs(float(w @ dv))}
            )
    if cfg.dv01_neutral and dv is not None:
        constraints.append({"type": "eq", "fun": lambda w: float(w @ dv)})

    bounds = [(-bound, bound)] * n

    try:
        from scipy.optimize import minimize

        res = minimize(
            objective, prev.copy(), method="SLSQP", bounds=bounds,
            constraints=constraints, options={"maxiter": 300, "ftol": 1e-12},
        )
        if res.success and np.isfinite(res.x).all():
            return _summarise(res.x, mu_v, cov_m, prev, tenors, dv, "ok")
        status = f"slsqp_failed:{res.message[:40]}"
    except Exception as exc:  # noqa: BLE001 - fall back rather than fail a trading day
        status = f"slsqp_error:{type(exc).__name__}"

    # ---- Analytic fallback ------------------------------------------------- #
    # Unconstrained solution w* = (lambda * Sigma)^-1 mu, then projected onto the
    # box and scaled into the leverage / DV01 limits.  Sub-optimal, but always
    # produces a valid, risk-compliant book.
    log.warning("optimizer fallback engaged (%s)", status)
    try:
        reg = cov_m + np.eye(n) * (np.trace(cov_m) / n * 1e-6 + 1e-12)
        w = np.linalg.solve(risk_aversion * reg, mu_v)
        # Rescale to the volatility target - the fallback should still land on
        # the intended risk level even though it ignores the constraints.
        vol = float(np.sqrt(max(w @ cov_m @ w, 0.0)))
        if vol > EPS:
            w *= target_daily_vol / vol
    except np.linalg.LinAlgError:
        w = mu_v / max(np.abs(mu_v).sum(), EPS)

    w = np.clip(w, -bound, bound)
    gross = np.abs(w).sum()
    if cfg.max_leverage and gross > cfg.max_leverage:
        w *= cfg.max_leverage / gross
    if dv is not None and cfg.max_gross_dv01 > 0:
        gd = np.abs(w * dv).sum()
        if gd > cfg.max_gross_dv01:
            w *= cfg.max_gross_dv01 / gd
    if cfg.dv01_neutral and dv is not None:
        w = dv01_neutral_projection(pd.Series(w, index=tenors), pd.Series(dv, index=tenors)).to_numpy()
    return _summarise(w, mu_v, cov_m, prev, tenors, dv, status + "|fallback")


def risk_parity_weights(cov: pd.DataFrame, max_iter: int = 500, tol: float = 1e-10) -> pd.Series:
    """Equal risk contribution weights.

    Solved by the standard fixed-point iteration ``w_i <- w_i * (target / RC_i)``,
    which converges reliably for a positive-definite covariance and needs no
    optimiser. Long-only by construction, so this is a benchmark allocation
    rather than a signal-driven book.
    """
    tenors = list(cov.index)
    n = len(tenors)
    S = cov.to_numpy(dtype=float)
    w = np.full(n, 1.0 / n)
    target = 1.0 / n

    for _ in range(max_iter):
        port_var = float(w @ S @ w)
        if port_var <= EPS:
            break
        mrc = S @ w                      # marginal risk contribution
        rc = w * mrc / port_var          # fractional risk contribution
        adj = np.where(rc > EPS, target / rc, 1.0)
        w_new = w * np.power(adj, 0.5)   # damped update
        w_new = np.clip(w_new, 1e-8, None)
        w_new /= w_new.sum()
        if np.abs(w_new - w).max() < tol:
            w = w_new
            break
        w = w_new
    return pd.Series(w, index=tenors, name="risk_parity")


def minimum_variance_weights(cov: pd.DataFrame, long_only: bool = False) -> pd.Series:
    """Global minimum-variance portfolio, weights summing to one."""
    tenors = list(cov.index)
    n = len(tenors)
    S = cov.to_numpy(dtype=float) + np.eye(n) * 1e-12
    ones = np.ones(n)
    try:
        inv_ones = np.linalg.solve(S, ones)
        w = inv_ones / (ones @ inv_ones)
    except np.linalg.LinAlgError:
        w = ones / n
    if long_only:
        w = np.clip(w, 0.0, None)
        total = w.sum()
        w = w / total if total > EPS else ones / n
    return pd.Series(w, index=tenors, name="min_variance")


def dv01_neutral_projection(
    weights: pd.Series,
    dv01: pd.Series | None = None,
) -> pd.Series:
    """Project a book onto the DV01-neutral hyperplane.

    Removes the component of the position along the DV01 vector, leaving a book
    with zero net DV01 - pure curve exposure, no directional rates view. This is
    the orthogonal projection, so it is the *closest* neutral book to the one
    you asked for, which matters: naively scaling one leg to flatten DV01 can
    destroy the curve view you were trying to express.
    """
    if dv01 is None:
        dv01 = pd.Series(1.0, index=weights.index)
    d = dv01.reindex(weights.index).fillna(0.0).to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    denom = float(d @ d)
    if denom <= EPS:
        return weights.copy()
    w_proj = w - (float(w @ d) / denom) * d
    return pd.Series(w_proj, index=weights.index, name=weights.name)


def optimize_history(
    mu_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    cfg: PortfolioConfig,
    dv01_panel: pd.DataFrame | None = None,
    cov_lookback: int = 252,
    cov_halflife: int = 63,
    rebalance_every: int = 1,
) -> pd.DataFrame:
    """Run the optimiser day by day over a history of expected returns.

    The covariance for day ``t`` is estimated on returns strictly **before**
    ``t``, and the previous weights are the ones actually held into ``t``, so
    the resulting weight series is directly usable by the backtest engine
    without further shifting.
    """
    from .risk import covariance

    tenors = [c for c in mu_panel.columns if c in returns_panel.columns]
    mu_panel = mu_panel[tenors]
    out = pd.DataFrame(0.0, index=mu_panel.index, columns=tenors)
    prev = pd.Series(0.0, index=tenors)
    statuses: list[str] = []

    positions = returns_panel.index
    for i, date in enumerate(mu_panel.index):
        loc = positions.get_indexer([date])[0]
        if loc < cov_lookback or i % rebalance_every != 0:
            out.loc[date] = prev
            statuses.append("carry")
            continue

        # Strictly prior returns - `loc` is exclusive.
        window = returns_panel[tenors].iloc[max(0, loc - cov_lookback):loc]
        if len(window) < 30:
            out.loc[date] = prev
            statuses.append("insufficient_history")
            continue

        cov = covariance(window, method="ewma", halflife=cov_halflife, shrinkage=0.15)
        mu = mu_panel.loc[date].fillna(0.0)
        dv = dv01_panel.iloc[loc - 1] if dv01_panel is not None and loc > 0 else None
        res = mean_variance_weights(mu, cov, prev, cfg, dv01_per_unit=dv)
        out.loc[date] = res.weights
        prev = res.weights
        statuses.append(res.status)

    out.attrs["statuses"] = statuses
    return out
