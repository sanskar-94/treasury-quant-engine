"""Diagnostic charts.

Deliberately diagnostic rather than decorative. Each function answers a question
you would otherwise have to take on trust:

* :func:`plot_curve_surface` - did the curve invert, and when
* :func:`plot_curve_fit` - is the NSS fit actually tracking the market
* :func:`plot_factors` - do the PCA loadings look like level/slope/curvature
* :func:`plot_attribution` - where did the P&L come from
* :func:`plot_signal_diagnostics` - is the signal persistent enough to trade

Matplotlib is imported lazily inside each function and every failure is caught,
so a missing backend degrades to "no chart" rather than taking down a run on a
headless box. All charts use a colour-blind-safe palette and are readable in
greyscale, because they end up in printed reports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger("viz.plots")

__all__ = [
    "plot_curve_surface",
    "plot_curve_fit",
    "plot_factors",
    "plot_attribution",
    "plot_signal_diagnostics",
    "plot_all",
]

# Okabe-Ito: distinguishable with any common form of colour blindness.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]


def _mpl():
    """Import matplotlib with a headless backend, or return None."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        })
        return plt
    except ImportError:
        log.warning("matplotlib not installed; skipping charts")
        return None


def _save(fig, out: Path, name: str) -> Path | None:
    try:
        out.mkdir(parents=True, exist_ok=True)
        path = out / name
        fig.savefig(path, bbox_inches="tight")
        return path
    except Exception as exc:  # noqa: BLE001
        log.warning("could not write %s: %s", name, exc)
        return None
    finally:
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:  # noqa: BLE001
            pass


def plot_curve_surface(curve: pd.DataFrame, out_dir: str | Path, tenor_years=None) -> Path | None:
    """The curve through time, plus the 2s10s slope with inversions shaded.

    The shading is the point: it makes the 2000, 2006-07, 2019 and 2022-24
    inversions visible at a glance, and those regimes are where most of this
    project's strategies changed behaviour.
    """
    plt = _mpl()
    if plt is None:
        return None
    if tenor_years is None:
        from ..data.sources import TENOR_YEARS

        tenor_years = TENOR_YEARS

    cols = [c for c in curve.columns if c in tenor_years]
    cols = sorted(cols, key=lambda c: tenor_years[c])
    try:
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
        ax = axes[0]
        for i, c in enumerate(cols):
            ax.plot(curve.index, curve[c] * 100, lw=0.7,
                    color=plt.cm.viridis(i / max(len(cols) - 1, 1)), label=c)
        ax.set_ylabel("Yield (%)")
        ax.set_title("US Treasury par yield curve")
        ax.legend(ncol=5, fontsize=7, frameon=False, loc="upper right")

        ax = axes[1]
        if "2 Yr" in curve.columns and "10 Yr" in curve.columns:
            slope = (curve["10 Yr"] - curve["2 Yr"]) * 1e4
            ax.plot(slope.index, slope.to_numpy(), lw=0.8, color=PALETTE[0])
            ax.axhline(0, color="k", lw=0.7)
            ax.fill_between(slope.index, slope.to_numpy(), 0,
                            where=(slope < 0).to_numpy(), color=PALETTE[1], alpha=0.35,
                            label="inverted")
            ax.legend(frameon=False, fontsize=8)
            ax.set_ylabel("2s10s (bp)")
            ax.set_title("Curve slope; shaded where inverted")
        fig.tight_layout()
        return _save(fig, Path(out_dir), "curve_surface.png")
    except Exception as exc:  # noqa: BLE001
        log.warning("curve surface chart failed: %s", exc)
        return None


def plot_curve_fit(
    curve: pd.DataFrame,
    nss: pd.DataFrame,
    out_dir: str | Path,
    as_of=None,
    tenor_years=None,
) -> Path | None:
    """Market versus fitted curve for one date, with residuals in basis points.

    The residual panel is the useful one: it shows which sector is rich or cheap
    to the fitted curve, which is exactly the signal
    :func:`~tqe.features.technical.curve_residual_features` extracts.
    """
    plt = _mpl()
    if plt is None:
        return None
    if tenor_years is None:
        from ..data.sources import TENOR_YEARS

        tenor_years = TENOR_YEARS
    from ..curve.nelson_siegel import NSSParams

    try:
        valid = nss.dropna(subset=["beta0"])
        d = pd.Timestamp(as_of) if as_of else valid.index[-1]
        row = nss.loc[d]
        p = NSSParams(row.beta0, row.beta1, row.beta2, row.beta3, row.tau1, row.tau2)

        obs = curve.loc[d].dropna()
        cols = [c for c in obs.index if c in tenor_years]
        t = np.array([tenor_years[c] for c in cols])
        mkt = obs[cols].to_numpy(dtype=float) * 100
        grid = np.linspace(max(t.min(), 1e-3), t.max(), 300)

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
        ax = axes[0]
        ax.plot(grid, np.asarray(p.zero_rate(grid)) * 100, lw=1.5, color=PALETTE[0],
                label="NSS fit")
        ax.scatter(t, mkt, s=32, color=PALETTE[1], zorder=3, label="market")
        ax.set_ylabel("Yield (%)")
        ax.set_title(f"Curve fit, {d.date()}  (RMSE {row.get('rmse', np.nan) * 1e4:.2f}bp)")
        ax.legend(frameon=False)

        resid = (mkt / 100 - np.asarray([float(p.zero_rate(x)) for x in t])) * 1e4
        ax = axes[1]
        ax.bar(t, resid, width=np.maximum(t * 0.08, 0.12),
               color=[PALETTE[2] if r >= 0 else PALETTE[1] for r in resid])
        ax.axhline(0, color="k", lw=0.7)
        ax.set_ylabel("Residual (bp)")
        ax.set_xlabel("Maturity (years)")
        ax.set_title("Market minus fitted: positive = cheap")
        ax.set_xscale("log")
        fig.tight_layout()
        return _save(fig, Path(out_dir), "curve_fit.png")
    except Exception as exc:  # noqa: BLE001
        log.warning("curve fit chart failed: %s", exc)
        return None


def plot_factors(pca, factors: pd.DataFrame, out_dir: str | Path) -> Path | None:
    """PCA loadings and the cumulative factor paths.

    The loadings panel is a correctness check as much as a chart - level should
    be flat and positive, slope monotone, curvature humped. If it does not look
    like that, the sign convention has failed.
    """
    plt = _mpl()
    if plt is None:
        return None
    try:
        load = pca.loadings_frame()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        for i, c in enumerate(load.columns):
            ax.plot(range(len(load)), load[c].to_numpy(), marker="o", ms=4,
                    color=PALETTE[i], label=f"{c} ({pca.explained_variance_ratio_[i]:.1%})")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(range(len(load)))
        ax.set_xticklabels(load.index, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Loading")
        ax.set_title("Curve factor loadings")
        ax.legend(frameon=False, fontsize=8)

        ax = axes[1]
        cum = factors.dropna().cumsum()
        for i, c in enumerate(cum.columns):
            ax.plot(cum.index, cum[c].to_numpy() * 1e4, lw=0.9, color=PALETTE[i], label=c)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_ylabel("Cumulative factor move (bp)")
        ax.set_title("Factor paths")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        return _save(fig, Path(out_dir), "factors.png")
    except Exception as exc:  # noqa: BLE001
        log.warning("factor chart failed: %s", exc)
        return None


def plot_attribution(attribution, out_dir: str | Path, capital: float = 1e7) -> Path | None:
    """Cumulative P&L by factor, and the share of gross risk each explains."""
    plt = _mpl()
    if plt is None:
        return None
    try:
        contrib = attribution.contributions
        fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                                 gridspec_kw={"width_ratios": [2, 1]})
        ax = axes[0]
        cum = contrib.cumsum() / capital * 100
        for i, c in enumerate(cum.columns):
            ax.plot(cum.index, cum[c].to_numpy(), lw=1.0, color=PALETTE[i], label=c)
        ax.plot(cum.index, (attribution.total.cumsum() / capital * 100).to_numpy(),
                lw=1.4, color="k", ls="--", label="total")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_ylabel("Cumulative P&L (% of capital)")
        ax.set_title("P&L attribution by curve factor")
        ax.legend(frameon=False, fontsize=8)

        ax = axes[1]
        share = pd.Series(attribution.factor_share)
        ax.barh(range(len(share)), share.to_numpy() * 100,
                color=[PALETTE[i] for i in range(len(share))])
        ax.set_yticks(range(len(share)))
        ax.set_yticklabels(share.index)
        ax.set_xlabel("Share of gross P&L (%)")
        ax.set_title("Risk attribution")
        fig.tight_layout()
        return _save(fig, Path(out_dir), "attribution.png")
    except Exception as exc:  # noqa: BLE001
        log.warning("attribution chart failed: %s", exc)
        return None


def plot_signal_diagnostics(signal: pd.DataFrame, out_dir: str | Path) -> Path | None:
    """Signal autocorrelation, distribution and turnover.

    Autocorrelation is the one to read first. A signal with a good information
    coefficient but daily autocorrelation near zero flips sign constantly and
    will not survive transaction costs, and that is invisible in an accuracy
    statistic.
    """
    plt = _mpl()
    if plt is None:
        return None
    try:
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))

        ax = axes[0]
        lags = range(1, 41)
        for i, c in enumerate(signal.columns[:4]):
            s = signal[c].dropna()
            ax.plot(list(lags), [s.autocorr(k) for k in lags], lw=1.0,
                    color=PALETTE[i], label=c)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("Signal persistence")
        ax.legend(frameon=False, fontsize=7)

        ax = axes[1]
        ax.hist(signal.to_numpy().ravel(), bins=80, color=PALETTE[0], alpha=0.85)
        ax.set_title("Signal distribution")
        ax.set_xlabel("Signal")

        ax = axes[2]
        turn = signal.diff().abs().sum(axis=1)
        ax.plot(turn.index, turn.rolling(63, min_periods=10).mean().to_numpy(),
                lw=1.0, color=PALETTE[3])
        ax.set_title("Rolling signal turnover (63d mean)")
        ax.set_ylabel("Sum |change|")
        fig.tight_layout()
        return _save(fig, Path(out_dir), "signal_diagnostics.png")
    except Exception as exc:  # noqa: BLE001
        log.warning("signal chart failed: %s", exc)
        return None


def plot_all(
    out_dir: str | Path,
    curve: pd.DataFrame | None = None,
    nss: pd.DataFrame | None = None,
    pca=None,
    factors: pd.DataFrame | None = None,
    attribution=None,
    signal: pd.DataFrame | None = None,
    capital: float = 1e7,
) -> list[Path]:
    """Render every chart for which the inputs were supplied."""
    out = Path(out_dir)
    written: list[Path] = []
    if curve is not None:
        written.append(plot_curve_surface(curve, out))
        if nss is not None:
            written.append(plot_curve_fit(curve, nss, out))
    if pca is not None and factors is not None:
        written.append(plot_factors(pca, factors, out))
    if attribution is not None:
        written.append(plot_attribution(attribution, out, capital))
    if signal is not None:
        written.append(plot_signal_diagnostics(signal, out))
    return [p for p in written if p is not None]
