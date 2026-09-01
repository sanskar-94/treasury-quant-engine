"""Event-driven backtest.

The engine is deliberately simple, because complexity in a backtester is mostly
a way to hide look-ahead. One loop, one timing convention, stated explicitly:

============ ==========================================================
Day ``t-1``  close: features observed, model predicts, signal formed
Day ``t``    open:  the book is moved from ``position[t-1]`` to
                    ``position[t]``; costs are charged on the traded amount
Day ``t``    close: ``position[t]`` earns ``return[t]``
============ ==========================================================

Because :func:`tqe.features.build_features` already lagged the features by one
day, ``signal[t]`` is a function of information available at ``t-1``'s close, so
``position[t]`` is knowable before ``return[t]`` is realised. No further shifting
happens inside this module - doing it in two places is how double-lagging (and
silently destroyed signal) happens.

The engine also runs a **look-ahead canary**. It re-runs the identical machinery
on a signal built from the realised future return itself - perfect foresight,
the best any leak could possibly achieve. The honest run must land far below it.
Reporting a Sharpe without this number is reporting half the result, so it is
computed by default rather than being an optional extra.

Note that shifting the *signal* forward is NOT a valid canary, though it is a
tempting one to write: signal[t+1] forecasts return[t+1], so using it to trade
day t merely misaligns it and scores near zero regardless of whether the
pipeline leaks. The canary has to inject the future *outcome*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from ..training.metrics import performance_metrics
from .costs import CostModel

log = get_logger("backtest.engine")

__all__ = ["BacktestResult", "run_backtest", "buy_and_hold"]

EPS = 1e-12


@dataclass
class BacktestResult:
    """Everything a backtest produced."""

    equity: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    costs: pd.Series
    financing: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    metrics: dict[str, Any] = field(default_factory=dict)
    exposures: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark: pd.Series | None = None
    gross_returns: pd.Series | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"Period          {self.equity.index.min().date()} .. {self.equity.index.max().date()}"
            f"  ({len(self.equity)} trading days)",
            f"Total return    {m.get('total_return', 0):>10.2%}",
            f"CAGR            {m.get('cagr', 0):>10.2%}",
            f"Ann. vol        {m.get('ann_vol', 0):>10.2%}",
            f"Sharpe          {m.get('sharpe', 0):>10.2f}",
            f"Sortino         {m.get('sortino', 0):>10.2f}",
            f"Calmar          {m.get('calmar', 0):>10.2f}",
            f"Max drawdown    {m.get('max_drawdown', 0):>10.2%}",
            f"Hit rate        {m.get('hit_rate', 0):>10.2%}",
            f"Ann. turnover   {m.get('ann_turnover', 0):>10.2f}x",
            f"Total costs     {m.get('total_costs', 0):>10,.0f}  "
            f"({m.get('cost_drag_annual', 0):.2%} p.a.)",
            f"Financing       {m.get('total_financing', 0):>10,.0f}  "
            f"({m.get('financing_drag_annual', 0):.2%} p.a.)",
        ]
        if "sharpe_gross" in m:
            lines.append(f"Sharpe (gross)  {m['sharpe_gross']:>10.2f}")
        if "deflated_sharpe" in m:
            lines.append(f"Deflated Sharpe {m['deflated_sharpe']:>10.4f}")
        if "lookahead_canary_sharpe" in m:
            lines.append(
                f"Canary Sharpe   {m['lookahead_canary_sharpe']:>10.2f}  "
                f"(perfect foresight; honest/canary = {m.get('canary_ratio', float('nan')):.3f})"
            )
        if self.benchmark is not None and "benchmark_sharpe" in m:
            lines.append(f"Benchmark Sharpe{m['benchmark_sharpe']:>10.2f}")
        return "\n".join(lines)

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.equity.to_frame("equity").to_parquet(out / "equity.parquet")
        self.returns.to_frame("returns").to_parquet(out / "returns.parquet")
        self.positions.to_parquet(out / "positions.parquet")
        if not self.trades.empty:
            self.trades.to_parquet(out / "trades.parquet")
        self.costs.to_frame("costs").to_parquet(out / "costs.parquet")
        if len(self.financing):
            self.financing.to_frame("financing").to_parquet(out / "financing.parquet")
        if not self.exposures.empty:
            self.exposures.to_parquet(out / "exposures.parquet")
        (out / "metrics.json").write_text(json.dumps(self.metrics, indent=2, default=str))
        (out / "summary.txt").write_text(self.summary())
        log.info("backtest artefacts written to %s", out)
        return out


def _funding_from_curve(cfg: Config, index: pd.Index) -> pd.Series | None:
    """Annualised funding rate per date: the shortest bill yield plus repo spread.

    Read from the cached curve rather than passed in, so a caller who does not
    think about financing still gets it charged. Returns ``None`` when no curve
    is available, in which case the engine simply cannot fund the book and says
    so via ``total_financing == 0``.
    """
    path = cfg.processed_dir / "curve.parquet"
    if not path.exists():
        return None
    try:
        from ..data.sources import TENOR_YEARS

        curve = pd.read_parquet(path)
        avail = [c for c in curve.columns if c in TENOR_YEARS]
        if not avail:
            return None
        short = min(avail, key=lambda c: TENOR_YEARS[c])
        rate = curve[short].reindex(index.union(curve.index)).ffill().reindex(index)
        return rate + cfg.costs.repo_spread_bp * 1e-4
    except Exception as exc:  # noqa: BLE001 - financing must not break a run
        log.warning("could not derive a funding rate (%s); financing not charged", exc)
        return None


def _bucket_map(tenors) -> dict[str, str]:
    from ..data.sources import TENOR_YEARS
    from ..data.universe import bucket_for_years

    return {t: bucket_for_years(TENOR_YEARS.get(t, 10.0)) for t in tenors}


def _core_loop(
    positions: pd.DataFrame,
    returns_panel: pd.DataFrame,
    cost_model: CostModel | None,
    buckets: dict[str, str],
    capital: float,
    include_costs: bool,
    slippage_multiplier: float,
    funding_rate: pd.Series | None = None,
    include_financing: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """Run the P&L loop.

    ``positions`` holds signed **notional face** per tenor for each day, already
    aligned so that row ``t`` is the book carried through day ``t``.

    **Financing is not optional bookkeeping - it is what makes the P&L an
    excess return.** A bond position bought with borrowed money earns its total
    return and pays repo on the borrowed amount; the strategy only keeps the
    difference. Omitting the funding leg hands a leveraged book the risk-free
    rate for nothing, and the distortion is not small: over 2018-2026 a
    three-month bill returned 2.81% at almost zero volatility, so an unfunded
    backtest scores holding cash at a Sharpe above 12. Any strategy with a net
    long bias then inherits a large, entirely fictitious edge.

    The charge is ``net_notional * funding_rate * days/360`` (ACT/360, the repo
    convention). It is levied on the **net** book: longs pay, shorts receive.

    Returns ``(net_returns, gross_returns, costs, trades, financing)``.
    """
    tenors = list(positions.columns)
    rets = returns_panel.reindex(index=positions.index, columns=tenors).fillna(0.0)

    pos = positions.to_numpy(dtype=float)
    ret = rets.to_numpy(dtype=float)

    # Trades: change in notional from the previous day's book.
    prev = np.vstack([np.zeros((1, pos.shape[1])), pos[:-1]])
    trade = pos - prev

    # Gross P&L: the book carried through day t earns day t's return.
    gross_pnl = (pos * ret).sum(axis=1)

    cost_arr = np.zeros(len(pos))
    if include_costs and cost_model is not None:
        for j, tenor in enumerate(tenors):
            bucket = buckets.get(tenor, "10y")
            traded = np.abs(trade[:, j])
            nz = traded > EPS
            if not nz.any():
                continue
            costs_j = np.array(
                [cost_model.total_cost(float(v), bucket) for v in traded[nz]], dtype=float
            )
            cost_arr[nz] += costs_j * slippage_multiplier

    # ---- financing on the net borrowed notional ---- #
    idx = positions.index
    fin_arr = np.zeros(len(pos))
    if include_financing and funding_rate is not None:
        rate = funding_rate.reindex(idx).ffill().fillna(0.0).to_numpy(dtype=float)
        # Actual calendar days between marks, so weekends are funded too.
        days = np.empty(len(idx))
        days[0] = 1.0
        days[1:] = np.diff(idx.to_numpy().astype("datetime64[D]").astype(float))
        days = np.clip(days, 0.0, 10.0)
        net_notional = pos.sum(axis=1)
        fin_arr = net_notional * rate * days / 360.0

    net_pnl = gross_pnl - cost_arr - fin_arr

    return (
        pd.Series(net_pnl / capital, index=idx, name="returns"),
        pd.Series(gross_pnl / capital, index=idx, name="gross_returns"),
        pd.Series(cost_arr, index=idx, name="costs"),
        pd.DataFrame(trade, index=idx, columns=tenors),
        pd.Series(fin_arr, index=idx, name="financing"),
    )


def run_backtest(
    signals: pd.DataFrame,
    returns_panel: pd.DataFrame,
    dv01_panel: pd.DataFrame,
    cfg: Config | None = None,
    cost_model: CostModel | None = None,
    benchmark: pd.Series | None = None,
    yield_change_panel: pd.DataFrame | None = None,
    positions: pd.DataFrame | None = None,
    funding_rate: pd.Series | None = None,
    run_canary: bool = True,
    n_trials: int = 1,
) -> BacktestResult:
    """Simulate the strategy.

    Parameters
    ----------
    signals:
        ``date x tenor`` standardised signals. Row ``t`` must be computable from
        information at ``t-1``.
    returns_panel:
        Realised total returns per tenor (fractional, per unit notional).
    dv01_panel:
        DV01 per 100 face per tenor, used for sizing and exposure reporting.
    cfg:
        Configuration; ``cfg.portfolio`` drives sizing, ``cfg.backtest`` the run.
    cost_model:
        Transaction costs. Built from ``cfg.costs`` if omitted.
    benchmark:
        Optional comparison series (e.g. buy-and-hold 10y total returns).
    yield_change_panel:
        Daily yield changes, used for a bp-denominated volatility estimate when
        sizing. Falls back to return volatility if absent.
    positions:
        Supply a pre-computed notional book to bypass the sizing layer entirely.
    funding_rate:
        Annualised repo/funding rate per date. Longs pay it, shorts receive it.
        Defaults to the shortest available tenor's yield plus
        ``cfg.costs.repo_spread_bp`` - a bill yield is the closest thing to a
        risk-free funding rate in this dataset. Without it the backtest reports
        a total return where it should report an excess return.
    run_canary:
        Re-run with a deliberately look-ahead signal and report its Sharpe.
    n_trials:
        Number of strategy configurations searched before arriving at this one.
        Feeds the deflated Sharpe ratio - be honest here.

    Returns
    -------
    BacktestResult
    """
    cfg = cfg or Config()
    bc, pc = cfg.backtest, cfg.portfolio
    cost_model = cost_model or CostModel(cfg.costs)

    tenors = [c for c in signals.columns if c in returns_panel.columns]
    if not tenors:
        raise ValueError("signals and returns_panel share no tenors")

    sig = signals[tenors].copy()
    if bc.start_date:
        sig = sig.loc[pd.Timestamp(bc.start_date):]
    if bc.end_date:
        sig = sig.loc[: pd.Timestamp(bc.end_date)]
    sig = sig.dropna(how="all")
    if sig.empty:
        raise ValueError("no signal rows inside the requested backtest window")

    # Deadband before sizing: forecasts too weak to justify their cost.
    if pc.min_signal_to_trade > 0:
        sig = sig.where(sig.abs() >= pc.min_signal_to_trade, 0.0)

    if positions is None:
        from ..signals.sizing import size_portfolio

        sized = size_portfolio(
            sig,
            returns_panel[tenors].reindex(sig.index),
            dv01_panel[tenors].reindex(sig.index),
            pc,
            yield_change_panel[tenors].reindex(sig.index) if yield_change_panel is not None else None,
        )
        positions = sized["notional"]
        target_dv01 = sized["target_dv01"]
    else:
        positions = positions.reindex(index=sig.index, columns=tenors).fillna(0.0)
        target_dv01 = positions * dv01_panel[tenors].reindex(sig.index) / 100.0

    positions = positions.fillna(0.0)
    buckets = _bucket_map(tenors)

    # Funding rate: the shortest tenor available is the best risk-free proxy in
    # this dataset, plus the configured repo spread over general collateral.
    if funding_rate is None and bc.include_financing:
        funding_rate = _funding_from_curve(cfg, returns_panel.index)

    net_r, gross_r, costs, trades, financing = _core_loop(
        positions, returns_panel, cost_model, buckets,
        bc.initial_capital, bc.include_costs, bc.slippage_multiplier,
        funding_rate=funding_rate, include_financing=bc.include_financing,
    )

    equity = bc.initial_capital * (1.0 + net_r).cumprod()
    equity.name = "equity"

    metrics = performance_metrics(net_r)
    gross_metrics = performance_metrics(gross_r)
    metrics["sharpe_gross"] = gross_metrics.get("sharpe", np.nan)
    metrics["ann_return_gross"] = gross_metrics.get("ann_return", np.nan)

    # Turnover, expressed as multiples of capital traded per year.
    daily_turnover = trades.abs().sum(axis=1) / max(bc.initial_capital, EPS)
    metrics["ann_turnover"] = float(daily_turnover.mean() * 252.0)
    metrics["total_costs"] = float(costs.sum())
    metrics["total_financing"] = float(financing.sum())
    metrics["financing_drag_annual"] = float(
        financing.sum() / bc.initial_capital / max(len(net_r) / 252.0, EPS)
    )
    years = max(len(net_r) / 252.0, EPS)
    metrics["cost_drag_annual"] = float(costs.sum() / bc.initial_capital / years)
    metrics["avg_gross_dv01"] = float(target_dv01.abs().sum(axis=1).mean())
    metrics["avg_net_dv01"] = float(target_dv01.sum(axis=1).mean())
    metrics["avg_gross_notional"] = float(positions.abs().sum(axis=1).mean())
    metrics["pct_days_invested"] = float((positions.abs().sum(axis=1) > EPS).mean())

    # Deflated Sharpe: honest adjustment for however many configurations were tried.
    from ..training.metrics import deflated_sharpe_ratio

    metrics["n_trials"] = int(n_trials)
    metrics["deflated_sharpe"] = float(
        deflated_sharpe_ratio(
            metrics.get("sharpe", 0.0), n_trials, len(net_r),
            metrics.get("skew", 0.0), metrics.get("kurtosis", 3.0),
        )
    )

    if benchmark is not None:
        bench = benchmark.reindex(net_r.index).fillna(0.0)
        bm = performance_metrics(bench)
        metrics["benchmark_sharpe"] = bm.get("sharpe", np.nan)
        metrics["benchmark_return"] = bm.get("ann_return", np.nan)
        metrics["benchmark_max_dd"] = bm.get("max_drawdown", np.nan)
        active = net_r - bench
        metrics["information_ratio"] = float(
            active.mean() / active.std() * np.sqrt(252) if active.std() > EPS else np.nan
        )
        metrics["correlation_to_benchmark"] = float(net_r.corr(bench))

    # ---- look-ahead canary ------------------------------------------------- #
    if run_canary:
        # A clean ceiling: keep the strategy's OWN position sizes, but flip each
        # one to the sign of the return that is about to be realised. Same risk,
        # same turnover constraints, same funding - perfect direction.
        #
        # Two earlier definitions were discarded. Shifting the signal forward
        # only misaligns it and scores near zero whether or not the pipeline
        # leaks. Re-sizing a sign(return) signal through the full pipeline lets
        # the no-trade band and the monthly schedule throttle the canary itself,
        # so under tight turnover limits "perfect foresight" scored below the
        # honest run and the ratio became meaningless.
        try:
            realised = returns_panel[tenors].reindex(positions.index).fillna(0.0)
            # Cash-neutral perfect foresight has to target RELATIVE returns, not
            # absolute ones: long whatever outperforms the cross-section, short
            # whatever lags. Aligning with the absolute sign and then demeaning
            # destroys the alignment - on a day when every tenor rallies, the
            # demeaned book is forced short half of them.
            excess = realised.sub(realised.mean(axis=1), axis=0)
            cheat_pos = positions.abs() * np.sign(excess)
            cheat_pos = cheat_pos.sub(cheat_pos.mean(axis=1), axis=0)
            neutral_pos = positions.sub(positions.mean(axis=1), axis=0)

            c_net, _, _, _, _ = _core_loop(
                cheat_pos, returns_panel, cost_model, buckets,
                bc.initial_capital, bc.include_costs, bc.slippage_multiplier,
                funding_rate=funding_rate, include_financing=bc.include_financing,
            )
            h_net, _, _, _, _ = _core_loop(
                neutral_pos, returns_panel, cost_model, buckets,
                bc.initial_capital, bc.include_costs, bc.slippage_multiplier,
                funding_rate=funding_rate, include_financing=bc.include_financing,
            )
            canary = performance_metrics(c_net).get("sharpe", np.nan)
            honest_neutral = performance_metrics(h_net).get("sharpe", np.nan)
            metrics["lookahead_canary_sharpe"] = float(canary)
            metrics["cash_neutral_sharpe"] = float(honest_neutral)
            metrics["canary_ratio"] = (
                float(honest_neutral / canary) if abs(canary) > EPS else np.nan
            )
            if abs(canary) > EPS and honest_neutral / canary > 0.35:
                log.warning(
                    "LOOK-AHEAD SUSPECTED: the cash-neutral run scores %.2f against a "
                    "perfect-foresight ceiling of %.2f - a clean pipeline should be far below it",
                    honest_neutral, canary,
                )
        except Exception as exc:  # noqa: BLE001 - the canary must never break the run
            log.warning("look-ahead canary failed to run: %s", exc)

    exposures = pd.DataFrame(
        {
            "gross_notional": positions.abs().sum(axis=1),
            "net_notional": positions.sum(axis=1),
            "gross_dv01": target_dv01.abs().sum(axis=1),
            "net_dv01": target_dv01.sum(axis=1),
            "leverage": positions.abs().sum(axis=1) / bc.initial_capital,
            "n_active": (positions.abs() > EPS).sum(axis=1),
        }
    )

    result = BacktestResult(
        equity=equity,
        returns=net_r,
        positions=positions,
        trades=trades,
        costs=costs,
        financing=financing,
        metrics=metrics,
        exposures=exposures,
        benchmark=benchmark.reindex(net_r.index) if benchmark is not None else None,
        gross_returns=gross_r,
        config={"portfolio": pc.__dict__, "backtest": bc.__dict__, "costs": cfg.costs.__dict__},
    )
    log.info("backtest complete\n%s", result.summary())
    return result


def buy_and_hold(
    returns_panel: pd.DataFrame,
    tenor: str = "10 Yr",
    index: pd.Index | None = None,
) -> pd.Series:
    """Passive long-one-tenor benchmark.

    The honest comparison for a rates strategy is not cash - it is simply owning
    duration, which has earned a positive risk premium for most of history. A
    model that cannot beat buy-and-hold 10-year Treasuries risk-adjusted has not
    justified its complexity.
    """
    if tenor not in returns_panel.columns:
        raise KeyError(f"{tenor!r} not in the returns panel")
    s = returns_panel[tenor].fillna(0.0)
    return s.reindex(index).fillna(0.0) if index is not None else s
