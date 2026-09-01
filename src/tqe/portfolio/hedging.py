"""Hedging a rates book.

Neutralising *total* DV01 is the easy half and it is not enough. A book that is
long the 2-year and short the 10-year in DV01-neutral size has zero net duration
and is still badly exposed: it makes money on a flattening and loses on a
steepening, and a parallel-shift hedge says nothing about either.

The right question is **where on the curve** the exposure sits, which is what
key-rate durations answer. :func:`krd_hedge` solves for the hedge notionals that
minimise exposure across every key rate simultaneously, not just their sum.

Two solvers, because the situation differs:

* **Exact**, when there are at least as many hedge instruments as key rates
  being hedged. A linear solve.
* **Least-squares with a size penalty**, when there are fewer - the usual case,
  since a desk hedges a nine-point curve with two or three liquid futures. The
  penalty matters: an unconstrained least-squares hedge will happily take
  enormous offsetting positions to shave the last basis point of residual, which
  costs more in spread than the residual was worth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger("portfolio.hedging")

__all__ = [
    "HedgeResult",
    "dv01_hedge",
    "krd_hedge",
    "hedge_effectiveness",
    "minimum_variance_hedge",
]

EPS = 1e-12


@dataclass
class HedgeResult:
    """A hedge and an honest account of what it left behind."""

    notionals: pd.Series
    residual_dv01: float = 0.0
    residual_krd: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    hedge_cost_notional: float = 0.0
    method: str = ""
    reduction: float = 0.0          # fraction of exposure removed

    def summary(self) -> str:
        return (
            f"{self.method}: residual DV01 {self.residual_dv01:+,.2f}, "
            f"exposure cut {self.reduction:.1%}, "
            f"hedge notional ${self.hedge_cost_notional:,.0f}"
        )


def dv01_hedge(
    portfolio_dv01: float,
    hedge_dv01_per_100: float,
) -> float:
    """Face of a single instrument needed to flatten total DV01.

    The one-line hedge. Correct for a parallel shift and blind to everything
    else - see :func:`krd_hedge` for why that is usually insufficient.
    """
    if abs(hedge_dv01_per_100) < EPS:
        raise ValueError("Hedge instrument has ~zero DV01")
    return -portfolio_dv01 / hedge_dv01_per_100 * 100.0


def krd_hedge(
    portfolio_krd: pd.Series,
    hedge_krd: pd.DataFrame,
    size_penalty: float = 1e-4,
    max_notional: float | None = None,
) -> HedgeResult:
    """Solve for hedge notionals that neutralise key-rate exposure.

    Parameters
    ----------
    portfolio_krd:
        The book's DV01 at each key rate, indexed by key tenor.
    hedge_krd:
        ``key_tenor x instrument`` matrix of each hedge instrument's DV01 at
        each key rate, per 100 face.
    size_penalty:
        Ridge weight on hedge size, expressed **relative to the scale of the
        problem**. Without a penalty, a least-squares hedge takes arbitrarily
        large offsetting positions to remove a residual worth less than the
        spread paid to establish it.

        The relative scaling is not cosmetic. DV01 per unit notional is of order
        1e-4, so ``A'A`` has entries around 1e-8; an absolute penalty of 1e-4
        would be ten thousand times larger than the term it regularises and
        would shrink the hedge to nothing while reporting success. The penalty
        applied is ``size_penalty * trace(A'A) / n_instruments``, which means the
        same value behaves identically whether notionals are quoted in dollars
        or millions.
    max_notional:
        Optional per-instrument cap; the whole hedge is scaled down (never up)
        if it binds, which preserves the hedge's shape.

    Returns
    -------
    HedgeResult
        Including the residual key-rate exposure, so the caller can see what the
        hedge could not reach rather than assuming it reached everything.
    """
    keys = [k for k in portfolio_krd.index if k in hedge_krd.index]
    if not keys:
        raise ValueError("portfolio and hedge instruments share no key rates")

    A = hedge_krd.loc[keys].to_numpy(dtype=float) / 100.0   # DV01 per unit notional
    b = portfolio_krd.loc[keys].to_numpy(dtype=float)
    n_inst = A.shape[1]

    # Solve  min || A h + b ||^2 + lambda || h ||^2, with lambda scaled to the
    # problem (see the `size_penalty` docstring - an absolute penalty here is
    # four orders of magnitude off and silently produces a null hedge).
    ata = A.T @ A
    scale = float(np.trace(ata)) / max(n_inst, 1)
    lam = size_penalty * scale
    gram = ata + lam * np.eye(n_inst)
    try:
        h = np.linalg.solve(gram, -A.T @ b)
        method = "krd_ridge" if size_penalty > 0 else "krd_exact"
    except np.linalg.LinAlgError:
        h, *_ = np.linalg.lstsq(A, -b, rcond=None)
        method = "krd_lstsq"

    if max_notional is not None:
        worst = float(np.abs(h).max())
        if worst > max_notional > 0:
            h = h * (max_notional / worst)
            method += "_capped"

    residual = b + A @ h
    before = float(np.sum(np.abs(b)))
    after = float(np.sum(np.abs(residual)))
    reduction = 1.0 - after / before if before > EPS else 0.0

    return HedgeResult(
        notionals=pd.Series(h, index=hedge_krd.columns, name="hedge_notional"),
        residual_dv01=float(residual.sum()),
        residual_krd=pd.Series(residual, index=keys, name="residual_krd"),
        hedge_cost_notional=float(np.abs(h).sum()),
        method=method,
        reduction=float(reduction),
    )


def minimum_variance_hedge(
    portfolio_returns: pd.Series,
    hedge_returns: pd.DataFrame,
    lookback: int | None = None,
) -> HedgeResult:
    """Regression hedge: the beta that minimises realised portfolio variance.

    An empirical alternative to the analytic key-rate hedge, and a useful check
    on it. Where the KRD hedge assumes the book's sensitivities are correctly
    measured, this one asks what actually co-moved. They should broadly agree;
    when they do not, the KRD inputs are usually stale.

    ``lookback`` restricts the estimate to the most recent observations, which
    is the causal choice when this is used to size a live hedge.
    """
    df = pd.concat([portfolio_returns.rename("_p"), hedge_returns], axis=1).dropna()
    if lookback:
        df = df.tail(int(lookback))
    if len(df) < max(20, hedge_returns.shape[1] * 5):
        raise ValueError(f"Not enough overlapping observations ({len(df)}) to fit a hedge")

    y = df["_p"].to_numpy(dtype=float)
    X = df.drop(columns="_p").to_numpy(dtype=float)
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()

    gram = Xc.T @ Xc
    try:
        beta = np.linalg.solve(gram + 1e-10 * np.eye(X.shape[1]), Xc.T @ yc)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)

    hedged = yc - Xc @ beta
    var_before = float(np.var(yc))
    var_after = float(np.var(hedged))
    reduction = 1.0 - var_after / var_before if var_before > EPS else 0.0

    return HedgeResult(
        notionals=pd.Series(-beta, index=hedge_returns.columns, name="hedge_ratio"),
        residual_dv01=0.0,
        hedge_cost_notional=float(np.abs(beta).sum()),
        method="min_variance",
        reduction=float(reduction),
    )


def hedge_effectiveness(
    unhedged: pd.Series,
    hedged: pd.Series,
) -> dict[str, float]:
    """How much risk the hedge actually removed.

    Reports variance reduction, the volatility ratio and the correlation of the
    two P&L streams. A hedge that cuts volatility while leaving correlation near
    one has mostly just scaled the position down rather than neutralising it,
    which is a distinction worth catching.
    """
    df = pd.concat([unhedged.rename("u"), hedged.rename("h")], axis=1).dropna()
    if len(df) < 3:
        return {"variance_reduction": np.nan, "vol_ratio": np.nan, "correlation": np.nan}
    vu, vh = float(df["u"].var()), float(df["h"].var())
    return {
        "variance_reduction": 1.0 - vh / vu if vu > EPS else np.nan,
        "vol_ratio": float(np.sqrt(vh / vu)) if vu > EPS else np.nan,
        "correlation": float(df["u"].corr(df["h"])),
        "unhedged_vol_ann": float(np.sqrt(vu * 252)),
        "hedged_vol_ann": float(np.sqrt(vh * 252)),
    }
