"""Backtest reporting: a markdown tearsheet plus charts.

The report is opinionated about what a reader needs to *disbelieve* the result,
not just admire it. Alongside the equity curve it always shows:

* gross vs net performance, so the cost drag is visible rather than buried,
* the deflated Sharpe ratio and the number of configurations searched,
* the look-ahead canary, which is the single fastest way for a reviewer to
  check the pipeline is honest,
* per-fold and per-year breakdowns, because one good year can carry a decade.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..training.metrics import drawdown_series

log = get_logger("backtest.report")

__all__ = ["tearsheet", "yearly_table", "monthly_table", "write_markdown", "plot_tearsheet"]


def yearly_table(returns: pd.Series, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Calendar-year returns, volatility, Sharpe and drawdown."""
    rows = []
    for year, grp in returns.groupby(returns.index.year):
        eq = (1 + grp).cumprod()
        dd = drawdown_series(eq)
        row = {
            "year": int(year),
            "return": float(eq.iloc[-1] - 1.0),
            "vol": float(grp.std() * np.sqrt(252)),
            "sharpe": float(grp.mean() / grp.std() * np.sqrt(252)) if grp.std() > 0 else np.nan,
            "max_dd": float(dd.min()),
            "days": int(len(grp)),
        }
        if benchmark is not None:
            b = benchmark.reindex(grp.index).fillna(0.0)
            row["benchmark"] = float((1 + b).prod() - 1.0)
            row["excess"] = row["return"] - row["benchmark"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")


def monthly_table(returns: pd.Series) -> pd.DataFrame:
    """Month-by-year grid of returns - the classic hedge-fund presentation."""
    m = (1 + returns).groupby([returns.index.year, returns.index.month]).prod() - 1.0
    m.index.names = ["year", "month"]
    grid = m.unstack("month")
    grid.columns = [
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][c - 1]
        for c in grid.columns
    ]
    grid["Year"] = (1 + m).groupby("year").prod() - 1.0
    return grid


def _fmt_pct(x: Any) -> str:
    try:
        return f"{float(x):.2%}"
    except (TypeError, ValueError):
        return str(x)


def _fmt_num(x: Any, nd: int = 2) -> str:
    try:
        return f"{float(x):,.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def write_markdown(result, out_dir: str | Path, title: str = "Backtest Report") -> Path:
    """Render the tearsheet as markdown."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = result.metrics
    r = result.returns

    lines = [f"# {title}", ""]
    lines += [
        f"**Period** {r.index.min().date()} to {r.index.max().date()}  ",
        f"**Observations** {len(r):,} trading days  ",
        "",
        "## Headline",
        "",
        "| Metric | Net | Gross |",
        "| --- | ---: | ---: |",
        f"| Annualised return | {_fmt_pct(m.get('ann_return'))} | {_fmt_pct(m.get('ann_return_gross'))} |",
        f"| Annualised volatility | {_fmt_pct(m.get('ann_vol'))} | - |",
        f"| Sharpe ratio | {_fmt_num(m.get('sharpe'))} | {_fmt_num(m.get('sharpe_gross'))} |",
        f"| Sortino ratio | {_fmt_num(m.get('sortino'))} | - |",
        f"| Calmar ratio | {_fmt_num(m.get('calmar'))} | - |",
        f"| Maximum drawdown | {_fmt_pct(m.get('max_drawdown'))} | - |",
        f"| Max DD duration | {int(m.get('max_dd_duration_days', 0))} days | - |",
        f"| Hit rate | {_fmt_pct(m.get('hit_rate'))} | - |",
        f"| Profit factor | {_fmt_num(m.get('profit_factor'))} | - |",
        f"| Skew / excess kurtosis | {_fmt_num(m.get('skew'))} / {_fmt_num(m.get('excess_kurtosis'))} | - |",
        f"| VaR 95 / CVaR 95 (daily) | {_fmt_pct(m.get('var_95'))} / {_fmt_pct(m.get('cvar_95'))} | - |",
        "",
        "## Costs and turnover",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Annualised turnover | {_fmt_num(m.get('ann_turnover'))}x |",
        f"| Total transaction costs | ${_fmt_num(m.get('total_costs'), 0)} |",
        f"| Cost drag | {_fmt_pct(m.get('cost_drag_annual'))} p.a. |",
        f"| Average gross DV01 | ${_fmt_num(m.get('avg_gross_dv01'), 0)} |",
        f"| Average net DV01 | ${_fmt_num(m.get('avg_net_dv01'), 0)} |",
        f"| Average gross notional | ${_fmt_num(m.get('avg_gross_notional'), 0)} |",
        f"| Days invested | {_fmt_pct(m.get('pct_days_invested'))} |",
        "",
        "## Statistical honesty",
        "",
        "These are the numbers that decide whether the headline means anything.",
        "",
        "| Check | Value | Reading |",
        "| --- | ---: | --- |",
        f"| Configurations searched | {int(m.get('n_trials', 1))} | feeds the deflation below |",
        *([f"| Deflated Sharpe (observed trial dispersion) | "
           f"{_fmt_num(m.get('deflated_sharpe_observed_sd'), 4)} | "
           f"using the measured sd of trial Sharpes ({_fmt_num(m.get('trial_sharpe_std'), 3)}) "
           f"rather than the theoretical one |"]
          if m.get("deflated_sharpe_observed_sd") is not None else []),
        f"| Deflated Sharpe ratio | {_fmt_num(m.get('deflated_sharpe'), 4)} | "
        + ("probability the Sharpe survives multiple testing |"
           if int(m.get("n_trials", 1)) > 1
           else "**NOT adjusted** - only one configuration was counted |"),
    ]
    if "lookahead_canary_sharpe" in m:
        ratio = m.get("canary_ratio", float("nan"))
        verdict = (
            "clean - the honest run is a small fraction of perfect foresight"
            if np.isfinite(ratio) and ratio < 0.35
            else "SUSPICIOUS - investigate for leakage"
        )
        lines.append(
            f"| Perfect-foresight Sharpe | {_fmt_num(m.get('lookahead_canary_sharpe'))} | "
            f"honest/foresight = {_fmt_num(ratio, 3)}; {verdict} |"
        )
    if "benchmark_sharpe" in m:
        lines += [
            f"| Benchmark Sharpe | {_fmt_num(m.get('benchmark_sharpe'))} | buy-and-hold duration |",
            f"| Information ratio | {_fmt_num(m.get('information_ratio'))} | active return per unit tracking error |",
            f"| Correlation to benchmark | {_fmt_num(m.get('correlation_to_benchmark'))} | |",
        ]
    lines.append("")

    yt = yearly_table(result.returns, result.benchmark)
    lines += ["## Calendar years", "", yt.round(4).to_markdown(), ""]

    try:
        mt = monthly_table(result.returns)
        lines += ["## Monthly returns", "", mt.round(4).to_markdown(), ""]
    except Exception:  # noqa: BLE001 - cosmetic section only
        pass

    lines += [
        "## Method",
        "",
        "- Features are lagged one business day, so every prediction uses only",
        "  information available at the prior close.",
        "- Walk-forward evaluation with purging and an embargo between train and",
        "  test blocks; the split scheme is audited before training starts.",
        "- Transaction costs are charged in 32nds of a point per the cash Treasury",
        "  convention, plus square-root market impact and commission.",
        "- The position held through day *t* earns day *t*'s return and is set from",
        "  the signal formed at *t-1*'s close.",
        "",
    ]

    path = out / "tearsheet.md"
    path.write_text("\n".join(lines))
    log.info("wrote %s", path)
    return path


def plot_tearsheet(result, out_dir: str | Path) -> list[Path]:
    """Render the standard chart set. Returns the paths written.

    Matplotlib is imported lazily and failures are non-fatal - a missing chart
    should never break a backtest run on a headless box.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping charts")
        return []

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

    try:
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2, 2, 2]})

        ax = axes[0]
        eq = result.equity / result.equity.iloc[0]
        ax.plot(eq.index, eq.to_numpy(), lw=1.3, label="Strategy (net)")
        if result.gross_returns is not None:
            g = (1 + result.gross_returns).cumprod()
            ax.plot(g.index, g.to_numpy(), lw=0.9, alpha=0.6, label="Strategy (gross)")
        if result.benchmark is not None:
            b = (1 + result.benchmark.fillna(0.0)).cumprod()
            ax.plot(b.index, b.to_numpy(), lw=1.0, alpha=0.8, label="Buy & hold 10y")
        ax.set_yscale("log")
        ax.set_ylabel("Growth of $1 (log)")
        ax.legend(loc="upper left", frameon=False)
        ax.set_title("Equity curve")

        ax = axes[1]
        dd = drawdown_series(result.equity)
        ax.fill_between(dd.index, dd.to_numpy() * 100, 0, alpha=0.4, color="#b03030")
        ax.set_ylabel("Drawdown (%)")
        ax.set_title("Drawdown")

        ax = axes[2]
        roll = result.returns.rolling(252).mean() / result.returns.rolling(252).std() * np.sqrt(252)
        ax.plot(roll.index, roll.to_numpy(), lw=1.0, color="#2c6fbb")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("Sharpe")
        ax.set_title("Rolling 1-year Sharpe ratio")

        ax = axes[3]
        if not result.exposures.empty:
            ax.plot(result.exposures.index, result.exposures["gross_dv01"].to_numpy(),
                    lw=0.9, label="Gross DV01")
            ax.plot(result.exposures.index, result.exposures["net_dv01"].to_numpy(),
                    lw=0.9, alpha=0.8, label="Net DV01")
            ax.legend(loc="upper left", frameon=False)
        ax.set_ylabel("DV01 ($)")
        ax.set_title("Risk exposure")

        fig.tight_layout()
        p = out / "tearsheet.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        written.append(p)
    except Exception as exc:  # noqa: BLE001
        log.warning("tearsheet chart failed: %s", exc)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(result.returns.to_numpy() * 100, bins=120, color="#2c6fbb", alpha=0.8)
        axes[0].set_title("Daily return distribution")
        axes[0].set_xlabel("Return (%)")

        yt = yearly_table(result.returns, result.benchmark)
        colors = ["#2e7d32" if v >= 0 else "#b03030" for v in yt["return"]]
        axes[1].bar(yt.index.astype(str), yt["return"] * 100, color=colors)
        axes[1].set_title("Calendar-year returns")
        axes[1].set_ylabel("%")
        axes[1].tick_params(axis="x", rotation=90)

        fig.tight_layout()
        p = out / "distribution.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        written.append(p)
    except Exception as exc:  # noqa: BLE001
        log.warning("distribution chart failed: %s", exc)

    return written


def tearsheet(result, out_dir: str | Path, title: str = "Backtest Report", plots: bool = True) -> Path:
    """Write the full report - markdown, charts and the raw artefacts."""
    out = Path(out_dir)
    result.save(out)
    write_markdown(result, out, title)
    if plots:
        plot_tearsheet(result, out)
    return out
