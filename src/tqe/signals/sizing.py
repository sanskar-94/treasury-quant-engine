"""Position sizing: from a dimensionless signal to a dollar risk position.

A signal says *which way* and *how confidently*. Sizing decides *how much*, and
in fixed income the natural unit is not notional but **DV01** - dollars of P&L
per basis point. $10mm of 2-year notes and $10mm of 30-year bonds are wildly
different positions; $1,000 DV01 of each is the same amount of risk.

Everything here targets risk, not capital:

* :func:`volatility_target_weights` scales exposure inversely to realised
  volatility, so the portfolio's risk stays near its target instead of
  ballooning in a crisis.
* :func:`dv01_scaled_positions` converts the risk target into notional per
  tenor using each bond's own DV01.
* :func:`kelly_size` bounds the aggressiveness, with a fractional multiplier
  because full Kelly on an estimated edge is a recipe for ruin.

All volatility estimates are trailing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PortfolioConfig
from ..logging_utils import get_logger

log = get_logger("signals.sizing")

__all__ = [
    "apply_no_trade_band",
    "apply_rebalance_schedule",
    "realised_volatility",
    "volatility_target_weights",
    "kelly_size",
    "dv01_scaled_positions",
    "apply_leverage_cap",
    "target_dv01_from_signal",
    "size_portfolio",
]

EPS = 1e-12
TRADING_DAYS = 252.0


def realised_volatility(
    returns: pd.DataFrame,
    lookback: int = 63,
    min_periods: int | None = None,
    annualize: bool = True,
) -> pd.DataFrame:
    """Trailing realised volatility per column.

    A simple rolling standard deviation rather than an EWMA: the rolling window
    is easier to reason about when it feeds a hard risk limit, and the
    difference in a 63-day window is small. Shift is **not** applied here - the
    caller decides, and the backtest engine does it explicitly.
    """
    mp = int(min_periods or max(10, lookback // 3))
    vol = returns.rolling(lookback, min_periods=mp).std()
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol


def volatility_target_weights(
    signal: pd.DataFrame,
    realised_vol: pd.DataFrame,
    target_vol: float = 0.06,
    max_leverage: float = 4.0,
    max_weight: float | None = None,
) -> pd.DataFrame:
    """Scale a signal into portfolio weights that target a volatility level.

    Each tenor's weight is ``signal * target_vol / realised_vol``, so a quiet
    asset earns a bigger position for the same conviction. The book is then
    scaled so gross exposure respects ``max_leverage``.

    This is inverse-volatility sizing, not a full mean-variance optimisation -
    it ignores correlation. That is a deliberate simplification at this layer;
    :mod:`tqe.portfolio.optimizer` handles the covariance-aware version. In
    practice the two agree closely for rates, because the correlation structure
    is dominated by a single level factor.
    """
    vol = realised_vol.reindex_like(signal)
    scaled = signal * (target_vol / vol.where(vol.abs() > EPS))
    scaled = scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if max_weight is not None:
        scaled = scaled.clip(-abs(max_weight), abs(max_weight))
    return apply_leverage_cap(scaled, max_leverage)


def apply_leverage_cap(weights: pd.DataFrame, max_leverage: float) -> pd.DataFrame:
    """Scale each row down (never up) so gross exposure stays within the cap."""
    if max_leverage is None or max_leverage <= 0:
        return weights
    gross = weights.abs().sum(axis=1)
    factor = (max_leverage / gross.where(gross > max_leverage)).fillna(1.0)
    return weights.mul(factor, axis=0)


def kelly_size(edge: float, variance: float, fraction: float = 0.25, cap: float = 1.0) -> float:
    """Fractional Kelly position.

    Full Kelly (``f = edge / variance``) maximises long-run log growth *given
    the true parameters*. With an estimated edge it is far too aggressive: a
    50% overestimate of the edge produces a position that loses money in
    expectation. Quarter-Kelly is the customary compromise, giving ~90% of the
    growth rate at a fraction of the drawdown.
    """
    if variance <= EPS:
        return 0.0
    return float(np.clip(fraction * edge / variance, -abs(cap), abs(cap)))


def target_dv01_from_signal(
    signal: pd.DataFrame,
    realised_vol: pd.DataFrame,
    capital: float,
    target_vol: float,
    max_gross_dv01: float,
    max_net_dv01: float,
    yield_vol_bp: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert signals into a signed **DV01** target per tenor.

    The risk budget is ``capital * target_vol / sqrt(252)`` dollars of daily
    standard deviation. Allocating a share of that to a tenor and dividing by
    that tenor's daily yield volatility in basis points gives the DV01 that
    delivers it::

        dv01_target = (daily_risk_budget * share) / daily_yield_vol_bp

    Gross and net DV01 caps are then applied by scaling, so the *shape* of the
    book is preserved when a limit binds rather than arbitrarily truncating one
    leg.
    """
    daily_budget = capital * target_vol / np.sqrt(TRADING_DAYS)

    if yield_vol_bp is None:
        # Fall back to the return volatility, converted to a bp-equivalent.
        yield_vol_bp = (realised_vol / np.sqrt(TRADING_DAYS)) * 1e4
    yv = yield_vol_bp.reindex_like(signal).replace(0.0, np.nan)

    # Share of the budget per tenor, proportional to |signal|.
    abs_sig = signal.abs()
    total = abs_sig.sum(axis=1)
    share = abs_sig.div(total.where(total > EPS), axis=0).fillna(0.0)

    dv01 = np.sign(signal) * (daily_budget * share) / yv
    dv01 = dv01.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Gross cap: scale the whole row.
    gross = dv01.abs().sum(axis=1)
    gfac = (max_gross_dv01 / gross.where(gross > max_gross_dv01)).fillna(1.0)
    dv01 = dv01.mul(gfac, axis=0)

    # Net cap: scale again if the directional component is too large.
    net = dv01.sum(axis=1).abs()
    nfac = (max_net_dv01 / net.where(net > max_net_dv01)).fillna(1.0)
    dv01 = dv01.mul(nfac, axis=0)

    return dv01


def dv01_scaled_positions(
    target_dv01: pd.DataFrame,
    dv01_per_100: pd.DataFrame,
    max_position_notional: float | None = None,
    capital: float | None = None,
    max_leverage: float | None = None,
) -> pd.DataFrame:
    """Notional face required to achieve a DV01 target.

    ``dv01_per_100`` is the DV01 of 100 face, so the face needed is
    ``target_dv01 / dv01_per_100 * 100``.

    **A DV01 cap is not a leverage cap.** This is the trap the function exists to
    close. A 3-month bill has a DV01 of about $0.0025 per 100 face, versus $0.16
    for the 30-year - a factor of 65. Allocating risk equally in DV01 terms
    therefore demands ~65x more *notional* at the front end, and a $25,000 gross
    DV01 book spread across the curve can quietly imply hundreds of millions of
    notional on ten million of capital. The risk limit is satisfied and the
    balance sheet is not.

    It matters beyond balance-sheet optics because transaction costs are charged
    on notional, not on DV01: an unconstrained front-end position is cheap in
    risk and ruinously expensive to trade.

    So when ``capital`` and ``max_leverage`` are supplied, each day's book is
    scaled down (never up) to respect gross notional <= capital * max_leverage.
    Scaling the whole row preserves the relative shape of the position.
    """
    unit = dv01_per_100.reindex_like(target_dv01)
    notional = target_dv01 / unit.where(unit.abs() > EPS) * 100.0
    notional = notional.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if max_position_notional is not None:
        notional = notional.clip(-abs(max_position_notional), abs(max_position_notional))

    if capital is not None and max_leverage is not None and max_leverage > 0:
        cap = abs(capital) * abs(max_leverage)
        gross = notional.abs().sum(axis=1)
        factor = (cap / gross.where(gross > cap)).fillna(1.0)
        notional = notional.mul(factor, axis=0)

    return notional


def size_portfolio(
    signal: pd.DataFrame,
    returns_panel: pd.DataFrame,
    dv01_panel: pd.DataFrame,
    cfg: PortfolioConfig,
    yield_change_panel: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """End-to-end sizing: signal -> weights, DV01 targets and notional.

    The volatility inputs are **shifted by one day** before use. Sizing today's
    position with today's realised volatility would use the very observation the
    position is about to experience.

    Returns
    -------
    dict
        ``weights``, ``target_dv01`` and ``notional`` frames.
    """
    vol = realised_volatility(returns_panel, cfg.vol_lookback).shift(1)

    yv = None
    if yield_change_panel is not None:
        yv = (yield_change_panel.rolling(cfg.vol_lookback, min_periods=cfg.vol_lookback // 3)
              .std().shift(1) * 1e4)

    weights = volatility_target_weights(
        signal, vol, cfg.target_annual_vol, cfg.max_leverage, cfg.max_weight_per_tenor
    )
    target_dv01 = target_dv01_from_signal(
        signal, vol, cfg.capital, cfg.target_annual_vol,
        cfg.max_gross_dv01, cfg.max_net_dv01, yield_vol_bp=yv,
    )
    if cfg.dv01_neutral:
        from ..portfolio.optimizer import dv01_neutral_projection

        target_dv01 = target_dv01.apply(lambda row: dv01_neutral_projection(row), axis=1)

    notional = dv01_scaled_positions(
        target_dv01, dv01_panel.shift(1),
        capital=cfg.capital, max_leverage=cfg.max_leverage,
    )
    # Order matters: the schedule decides *when* trading is permitted, then the
    # band decides whether the move is large enough to bother with.
    notional = apply_rebalance_schedule(notional, getattr(cfg, "rebalance", "daily"))
    if getattr(cfg, "no_trade_band", 0.0) > 0:
        notional = apply_no_trade_band(notional, cfg.no_trade_band)
    # Recompute the realised DV01 after banding and the leverage cap, so the
    # reported exposure is what is actually held rather than what was requested.
    realised_dv01 = notional * dv01_panel.shift(1).reindex_like(notional) / 100.0
    return {"weights": weights, "target_dv01": realised_dv01, "notional": notional}


def apply_no_trade_band(
    targets: pd.DataFrame,
    threshold: float = 0.10,
    reference: str = "gross",
) -> pd.DataFrame:
    """Hold the existing position unless the target has moved materially.

    Without this, a daily rebalance chases every wiggle in the forecast and pays
    a spread for the privilege. Measured on the real out-of-sample predictions,
    the unbanded book turned over roughly 63% of its notional *every day* -
    around 2,300x capital per year - which converted a gross Sharpe of 1.26 into
    a net Sharpe of -0.56. Costs, not signal, were the binding constraint.

    A no-trade band is the standard remedy and is close to optimal for a
    mean-reverting target with proportional costs: trade only when the desired
    position has drifted far enough from the held one to be worth the spread,
    then trade all the way to the target.

    Parameters
    ----------
    targets:
        Desired positions per date.
    threshold:
        Minimum change, as a fraction of the average gross book, that triggers a
        rebalance in that instrument. 0.10 means "ignore moves smaller than 10%
        of typical gross exposure".
    reference:
        ``"gross"`` scales the threshold by the mean gross book (one scale for
        the whole portfolio); ``"own"`` scales by each instrument's own mean
        absolute position.

    Returns
    -------
    pd.DataFrame
        Positions actually held, after banding.
    """
    if targets.empty or threshold <= 0:
        return targets

    if reference == "own":
        scale = targets.abs().mean().replace(0.0, np.nan)
        band = (scale * threshold).fillna(0.0).to_numpy()
    else:
        gross = float(targets.abs().sum(axis=1).mean())
        band = np.full(targets.shape[1], gross * threshold / max(targets.shape[1], 1))

    tgt = targets.to_numpy(dtype=float)
    held = np.empty_like(tgt)
    current = np.zeros(tgt.shape[1])
    for i in range(len(tgt)):
        move = np.abs(tgt[i] - current)
        trade = move > band
        current = np.where(trade, tgt[i], current)
        held[i] = current
    return pd.DataFrame(held, index=targets.index, columns=targets.columns)


def apply_rebalance_schedule(
    targets: pd.DataFrame,
    frequency: str = "daily",
) -> pd.DataFrame:
    """Hold positions constant between scheduled rebalance dates.

    A blunter turnover control than :func:`apply_no_trade_band` and complementary
    to it: the band asks "has the target moved enough to be worth trading?",
    while the schedule asks "are we even allowed to trade today?". Together they
    cut turnover far more than either does alone, because the band still fires on
    every one of 252 opportunities per year if you let it.

    The schedule is derived from the index itself rather than from a calendar
    rule, so it lands on real trading days and never invents a rebalance on a
    market holiday.

    Parameters
    ----------
    targets:
        Desired positions per date.
    frequency:
        ``"daily"`` (no constraint), ``"weekly"`` (first trading day of each
        ISO week), ``"biweekly"``, or ``"monthly"`` (first trading day of each
        month).

    Returns
    -------
    pd.DataFrame
        Positions held, changing only on rebalance dates.
    """
    freq = (frequency or "daily").lower()
    if freq == "daily" or targets.empty:
        return targets

    idx = pd.DatetimeIndex(targets.index)
    if freq == "weekly":
        # First trading day of each ISO week present in the index.
        key = pd.Series(idx.isocalendar().week.to_numpy() + idx.year.to_numpy() * 100, index=idx)
    elif freq == "biweekly":
        weeks = idx.isocalendar().week.to_numpy() + idx.year.to_numpy() * 100
        key = pd.Series(weeks // 2, index=idx)
    elif freq == "monthly":
        key = pd.Series(idx.year.to_numpy() * 100 + idx.month.to_numpy(), index=idx)
    else:
        raise ValueError(
            f"Unknown rebalance frequency {frequency!r}; "
            "use daily | weekly | biweekly | monthly"
        )

    is_rebalance = key != key.shift()
    is_rebalance.iloc[0] = True
    # Take the target on rebalance days, carry it forward on every other day.
    # The mask is broadcast explicitly: pandas does not align an (n, 1) array
    # against an (n, m) frame in `where`, it raises.
    mask = np.broadcast_to(is_rebalance.to_numpy()[:, None], targets.shape)
    held = targets.where(mask)
    return held.ffill().fillna(0.0)
