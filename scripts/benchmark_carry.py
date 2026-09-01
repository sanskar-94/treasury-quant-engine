#!/usr/bin/env python
"""Does the model beat carry?

This is the benchmark the project was missing, and it is the one that matters.
Carry is the dominant systematic return driver in fixed income: a bond held on a
positively sloped curve earns the yield pickup over funding plus the roll down
the curve, whether or not anyone forecasts anything. A machine-learning model
that does not beat that has not earned its complexity, and comparing against
cash or against buy-and-hold flatters it by leaving out the one factor everybody
in the market already harvests.

Three signals go through the *same* honest evaluation - walk-forward-derived or
mechanical, then a double-neutral funded book (zero net cash, zero net DV01) with
costs and financing charged, then a block sign-flip null:

    carry      yield pickup over funding, plus roll-down, cross-sectionally
               demeaned so it is a relative-value view rather than a directional
               long-duration bet
    model      the trained ensemble's out-of-sample predictions
    combined   the model's forecast added to carry, equally weighted after
               standardisation

All three are lagged one day: the carry known at yesterday's close sizes today's
book, exactly as the model's forecast does.

    python scripts/benchmark_carry.py [--placebos 40]
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqe.backtest.costs import CostModel  # noqa: E402
from tqe.backtest.engine import run_backtest  # noqa: E402
from tqe.config import load_config  # noqa: E402
from tqe.data.sources import TENOR_YEARS  # noqa: E402
from tqe.data.universe import constant_maturity_total_return, universe_panel  # noqa: E402
from tqe.logging_utils import setup_logging  # noqa: E402
from tqe.signals.alpha import predictions_to_signal, signal_decay  # noqa: E402
from tqe.signals.sizing import size_portfolio  # noqa: E402


def carry_signal(curve: pd.DataFrame, rets: dict, tenors: list[str]) -> pd.DataFrame:
    """Expected return from carry and roll-down, per tenor, in basis points.

    Two components, both standard desk arithmetic:

    * **Carry** - the tenor's yield less the funding rate (the shortest bill
      here). What you earn for holding the bond if nothing moves.
    * **Roll-down** - as the bond ages it is repriced at the yield of a shorter
      maturity. On an upward-sloping curve that is a capital gain, worth
      ``(dy/dt) x duration`` per year, where ``dy/dt`` is the local slope
      against the next shorter quoted tenor.

    The result is demeaned across tenors so the signal expresses *which part of
    the curve carries best*, not "own duration". A directional carry bet is
    really a bet that the term premium is positive, which is a different and much
    slower-moving claim than a relative-value one.

    Causality: everything is computed from the curve as of each date and then
    shifted one day by the caller, so nothing uses same-day information.
    """
    y = universe_panel(rets, "yield")[tenors]
    dur = universe_panel(rets, "duration")[tenors]
    order = sorted(tenors, key=lambda c: TENOR_YEARS[c])
    funding = y[order[0]]

    out = {}
    for i, col in enumerate(order):
        carry = (y[col] - funding) * 1e4
        if i == 0:
            roll = pd.Series(0.0, index=y.index)
        else:
            prev = order[i - 1]
            dt = TENOR_YEARS[col] - TENOR_YEARS[prev]
            roll = (y[col] - y[prev]) / dt * dur[col] * 1e4
        out[col] = carry + roll

    frame = pd.DataFrame(out)[tenors]
    return frame.sub(frame.mean(axis=1), axis=0)


def double_neutral(pos: pd.DataFrame, dv01: pd.DataFrame) -> pd.DataFrame:
    """Project each day's book onto the null space of [cash, DV01]."""
    out = pos.to_numpy(dtype=float).copy()
    D = dv01.reindex_like(pos).to_numpy(dtype=float)
    for i in range(len(out)):
        w, d = out[i], D[i]
        if not np.isfinite(d).all():
            continue
        A = np.vstack([np.ones_like(d), d])
        with contextlib.suppress(np.linalg.LinAlgError):
            w = w - A.T @ np.linalg.solve(A @ A.T, A @ w)
        out[i] = w
    return pd.DataFrame(out, index=pos.index, columns=pos.columns)


def block_sign_flip(signal: pd.DataFrame, block: int = 63, seed: int = 0) -> pd.DataFrame:
    """Randomise direction in contiguous blocks, preserving everything else."""
    rng = np.random.default_rng(seed)
    n, k = signal.shape
    flips = rng.choice([-1.0, 1.0], size=(int(np.ceil(n / block)), k))
    return pd.DataFrame(signal.to_numpy() * np.repeat(flips, block, axis=0)[:n],
                        index=signal.index, columns=signal.columns)


def to_signal(raw: pd.DataFrame, cfg) -> pd.DataFrame:
    s = predictions_to_signal(raw, "vol_scale", 252, cfg.portfolio.signal_clip, 0.0)
    s = s.sub(s.mean(axis=1), axis=0)
    s = s.where(s.abs() >= cfg.portfolio.min_signal_to_trade, 0.0)
    return signal_decay(s, cfg.portfolio.signal_halflife).fillna(0.0)


def evaluate(s, tr, dv, yc, cfg, target_vol=0.03):
    idx = s.index
    pos = size_portfolio(s, tr.reindex(idx), dv.reindex(idx), cfg.portfolio, yc.reindex(idx))["notional"]
    pos = double_neutral(pos, dv.shift(1))
    pnl = (pos * tr.reindex(idx).fillna(0.0)).sum(axis=1) / cfg.backtest.initial_capital
    sd = pnl.std() * np.sqrt(252)
    if sd > 0:
        pos = pos * (target_vol / sd)
    return run_backtest(s, tr, dv, cfg, CostModel(cfg.costs),
                        yield_change_panel=yc, positions=pos, run_canary=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--placebos", type=int, default=40)
    ap.add_argument("--out", default="artifacts/reports/carry_benchmark.csv")
    args = ap.parse_args()

    setup_logging("WARNING")
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")

    curve = pd.read_parquet(root / "data/processed/curve.parquet")
    preds = pd.read_parquet(root / "data/processed/oos_predictions.parquet")
    tenors = [t for t in cfg.data.core_tenors if t in curve.columns]
    rets = constant_maturity_total_return(curve, tenors)
    tr = universe_panel(rets, "total_return")
    dv = universe_panel(rets, "dv01")
    yc = universe_panel(rets, "yield_change")

    idx = preds.index
    # Lag by one day: yesterday's carry sizes today's book, like the forecast.
    carry = carry_signal(curve, rets, tenors).shift(1).reindex(idx)

    sig_carry = to_signal(carry, cfg)
    sig_model = to_signal(preds[tenors], cfg)
    # Equal-weight blend of two standardised views.
    sig_both = to_signal(
        sig_carry.div(sig_carry.std().replace(0, np.nan), axis=1).fillna(0.0)
        + sig_model.div(sig_model.std().replace(0, np.nan), axis=1).fillna(0.0),
        cfg,
    )

    print("=" * 74)
    print("  Does the model beat carry?  Double-neutral, funded, costs charged")
    print("=" * 74)
    print(f"  window {idx.min().date()} .. {idx.max().date()}  ({len(idx)} days)")
    print(f"  placebos per signal: {args.placebos}\n")

    rows = []
    for name, s in [("carry", sig_carry), ("model", sig_model), ("carry+model", sig_both)]:
        m = evaluate(s, tr, dv, yc, cfg).metrics
        pl = [evaluate(block_sign_flip(s, 63, i), tr, dv, yc, cfg).metrics["sharpe"]
              for i in range(args.placebos)]
        beat = sum(1 for v in pl if v >= m["sharpe"])
        rows.append({
            "signal": name, "sharpe": m["sharpe"], "ann_return": m["ann_return"],
            "ann_vol": m["ann_vol"], "max_dd": m["max_drawdown"],
            "turnover": m["ann_turnover"], "hit_rate": m["hit_rate"],
            "placebo_mean": float(np.mean(pl)), "placebo_sd": float(np.std(pl)),
            "p_value": (beat + 1) / (len(pl) + 1),
            "z_vs_placebo": float((m["sharpe"] - np.mean(pl)) / max(np.std(pl), 1e-9)),
        })
        print(f"  {name:12s} Sharpe={m['sharpe']:+6.3f}  ret={m['ann_return']:+6.2%}  "
              f"turn={m['ann_turnover']:5.1f}x  placebo={np.mean(pl):+.3f}+-{np.std(pl):.3f}  "
              f"p={rows[-1]['p_value']:.3f}")

    df = pd.DataFrame(rows)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n" + "=" * 74)
    print(df.round(4).to_string(index=False))

    c = df[df.signal == "carry"].iloc[0]
    m_ = df[df.signal == "model"].iloc[0]
    b = df[df.signal == "carry+model"].iloc[0]
    print("\n=== VERDICT ===")
    print(f"  carry alone       Sharpe {c.sharpe:+.3f}  (p={c.p_value:.3f})")
    print(f"  model alone       Sharpe {m_.sharpe:+.3f}  (p={m_.p_value:.3f})")
    print(f"  carry + model     Sharpe {b.sharpe:+.3f}  (p={b.p_value:.3f})")
    if m_.sharpe > c.sharpe:
        print("\n  The model beats carry on this window.")
    else:
        print(f"\n  The model does NOT beat carry ({m_.sharpe:+.3f} vs {c.sharpe:+.3f}).")
        print("  A mechanical signal computable from the current curve with no")
        print("  fitting, no features and no training outperforms the ensemble.")
    if b.sharpe > max(c.sharpe, m_.sharpe):
        print("  The blend beats both, so the forecast adds something to carry.")
    else:
        print("  The blend does not beat both, so the forecast adds nothing to carry.")
    print(f"\n  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
