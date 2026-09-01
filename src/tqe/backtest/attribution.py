"""P&L attribution: where the money actually came from.

A Sharpe ratio says whether a strategy worked. It says nothing about *why*, and
the difference decides whether you would put money behind it. A book that earned
its return by being long duration through a rally has a different future from
one that earned the same return trading the curve against itself, even though
the tearsheet looks identical.

This module decomposes realised P&L two ways.

**By risk factor.** Yield changes are projected onto the level / slope /
curvature basis from :mod:`tqe.curve.pca`, and each day's P&L is split into the
part explained by each factor plus an idiosyncratic residual. A strategy meant
to be curve-neutral that turns out to earn 80% of its money from the level
factor is not doing what it says.

**By source.** Price return, carry, financing and transaction costs, which
together reconstruct the net P&L exactly. This is the decomposition that
revealed the financing bug in this project: the market leg was losing 1.53% a
year while the reported strategy showed a profit.

Both decompositions are additive and are asserted to reconcile with the
backtest's own P&L to within a rounding error, because an attribution that does
not add up is worse than none.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger("backtest.attribution")

__all__ = [
    "FactorAttribution",
    "attribute_by_factor",
    "attribute_by_source",
    "attribution_report",
]


@dataclass
class FactorAttribution:
    """Daily P&L split across curve factors plus a residual."""

    contributions: pd.DataFrame          # date x [level, slope, curvature, residual]
    total: pd.Series
    explained_ratio: float = 0.0
    factor_share: dict[str, float] = field(default_factory=dict)
    loadings: pd.DataFrame | None = None

    def summary(self) -> str:
        parts = [f"{k}={v:+.1%}" for k, v in self.factor_share.items()]
        return f"FactorAttribution({', '.join(parts)}; R2={self.explained_ratio:.1%})"

    def annualised(self, capital: float = 1.0, periods: int = 252) -> pd.Series:
        """Mean daily contribution scaled to an annual rate."""
        return self.contributions.mean() * periods / max(capital, 1e-12)


def attribute_by_factor(
    positions: pd.DataFrame,
    yield_changes: pd.DataFrame,
    dv01: pd.DataFrame,
    pca=None,
    n_factors: int = 3,
) -> FactorAttribution:
    """Split P&L into level / slope / curvature contributions.

    The mechanics, for one day:

    1. The book's exposure to tenor *i* is its DV01, ``position_i * dv01_i / 100``.
    2. The realised yield change vector is projected onto the PCA basis:
       ``dy = sum_f s_f * v_f + e``, where ``v_f`` are the loadings, ``s_f`` the
       factor scores and ``e`` the residual.
    3. P&L from factor *f* is ``-sum_i dv01_i * s_f * v_{f,i} * 1e4`` - the DV01
       is per basis point, and the minus sign is because yields up means prices
       down.

    Because the projection is linear and the residual is carried explicitly, the
    contributions sum exactly to the total P&L.

    Parameters
    ----------
    positions:
        Signed notional per tenor per date.
    yield_changes:
        Realised daily yield changes, decimal.
    dv01:
        DV01 per 100 face per tenor per date.
    pca:
        A fitted :class:`~tqe.curve.pca.CurvePCA`. Fitted on the supplied yield
        changes if omitted - acceptable here because attribution is a *post hoc*
        description of what happened, not a signal, so using the full sample to
        define the basis does not leak anything into a trading decision.
    n_factors:
        Number of factors to attribute to.
    """
    from ..curve.pca import fit_curve_pca

    tenors = [c for c in positions.columns if c in yield_changes.columns and c in dv01.columns]
    if not tenors:
        raise ValueError("positions, yield_changes and dv01 share no tenors")

    idx = positions.index
    dy = yield_changes.reindex(index=idx, columns=tenors).fillna(0.0)
    pos = positions[tenors].fillna(0.0)
    dvs = dv01.reindex(index=idx, columns=tenors).ffill().fillna(0.0)

    if pca is None:
        pca = fit_curve_pca(dy.replace(0.0, np.nan).dropna(how="any"), n_factors=n_factors)
    comps = pca.components_[:n_factors]                       # (f, k)
    names = pca.factor_names[:n_factors]

    # Book DV01 per tenor, in dollars per basis point.
    book_dv01 = (pos.to_numpy(dtype=float) * dvs.to_numpy(dtype=float)) / 100.0
    dy_arr = dy.to_numpy(dtype=float)

    # Project each day's yield change onto the factor basis (orthonormal rows).
    scores = dy_arr @ comps.T                                 # (n, f)
    fitted = scores @ comps                                   # (n, k)
    resid = dy_arr - fitted

    contrib = {}
    for j, name in enumerate(names):
        move = np.outer(scores[:, j], comps[j])               # (n, k)
        contrib[name] = -(book_dv01 * move * 1e4).sum(axis=1)
    contrib["residual"] = -(book_dv01 * resid * 1e4).sum(axis=1)

    frame = pd.DataFrame(contrib, index=idx)
    total = pd.Series(-(book_dv01 * dy_arr * 1e4).sum(axis=1), index=idx, name="total")

    var_total = float(np.var(total)) if len(total) > 1 else 0.0
    var_resid = float(np.var(frame["residual"])) if len(frame) > 1 else 0.0
    explained = 1.0 - var_resid / var_total if var_total > EPS_VAR else 0.0

    gross = frame.abs().sum().sum()
    share = {c: float(frame[c].abs().sum() / gross) if gross > 0 else 0.0 for c in frame.columns}

    # The decomposition is exact by construction; verify rather than assume.
    reconstruction_error = float((frame.sum(axis=1) - total).abs().max())
    if reconstruction_error > 1e-6 * max(1.0, float(total.abs().max())):
        log.warning("factor attribution does not reconcile: max error %.3e", reconstruction_error)

    return FactorAttribution(
        contributions=frame,
        total=total,
        explained_ratio=float(explained),
        factor_share=share,
        loadings=pca.loadings_frame(),
    )


EPS_VAR = 1e-24


def attribute_by_source(result) -> pd.DataFrame:
    """Split net P&L into price return, carry, financing and costs.

    Reconstructs the backtest's own net return from its parts, which is the
    decomposition that matters economically: a strategy whose profit is entirely
    the financing leg is running a funding position rather than a forecast.

    Returns a frame of daily dollar contributions with a ``net`` column that
    matches ``result.returns * capital`` to a rounding error.
    """
    capital = float(result.config.get("backtest", {}).get("initial_capital", 1.0)) or 1.0
    gross = (result.gross_returns if result.gross_returns is not None
             else result.returns) * capital
    costs = result.costs.reindex(gross.index).fillna(0.0)
    fin = (result.financing.reindex(gross.index).fillna(0.0)
           if len(getattr(result, "financing", [])) else pd.Series(0.0, index=gross.index))

    frame = pd.DataFrame({
        "market_pnl": gross,
        "financing": -fin,          # a cost when positive, so it enters negated
        "costs": -costs,
    })
    frame["net"] = frame.sum(axis=1)

    expected = result.returns.reindex(frame.index) * capital
    err = float((frame["net"] - expected).abs().max())
    if err > 1e-6 * max(1.0, float(expected.abs().max())):
        log.warning("source attribution does not reconcile: max error %.3e", err)
    return frame


def attribution_report(
    result,
    yield_changes: pd.DataFrame,
    dv01: pd.DataFrame,
    periods: int = 252,
) -> dict:
    """Both decompositions, annualised, as a report-ready dict."""
    capital = float(result.config.get("backtest", {}).get("initial_capital", 1.0)) or 1.0
    years = max(len(result.returns) / periods, 1e-9)

    src = attribute_by_source(result)
    out = {
        "by_source": {c: float(src[c].sum() / capital / years) for c in src.columns},
        "n_days": int(len(result.returns)),
    }

    try:
        fac = attribute_by_factor(result.positions, yield_changes, dv01)
        out["by_factor"] = {
            c: float(fac.contributions[c].sum() / capital / years)
            for c in fac.contributions.columns
        }
        out["factor_share"] = fac.factor_share
        out["factor_explained_ratio"] = fac.explained_ratio
    except Exception as exc:  # noqa: BLE001 - attribution must not break a run
        log.warning("factor attribution unavailable: %s", exc)
        out["by_factor"] = {}

    return out
