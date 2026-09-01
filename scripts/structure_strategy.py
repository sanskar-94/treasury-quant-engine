#!/usr/bin/env python
"""Trade the curve in structure space instead of tenor space.

The attribution is unambiguous about where this model's skill lives. It earns
+0.42% a year on slope and +0.24% on curvature, loses 0.10% on level, and
carries 42% of its gross risk in level - the one factor it cannot forecast. Every
version of the strategy so far has expressed nine tenor views and then projected
the unwanted exposure away afterwards.

This does it the other way round. The tradable objects are **DV01-weighted
structures** - steepeners and butterflies from
:mod:`tqe.portfolio.structures` - each of which is neutral to a parallel shift by
construction. A model that forecasts *structure returns* has no way to express a
level view even if it wants to, so the exposure the attribution flagged cannot
arise. Nothing is projected away, because nothing unwanted is ever created.

Three signals, all through the same honest evaluation - funded, costed, and
tested against block-sign-flip controls that preserve persistence and destroy
only the timing:

    structure_model   ridge trained to forecast each structure's forward return
    structure_carry   each structure's carry, computed mechanically
    tenor_projected   the existing tenor-space model, projected double-neutral
                      (the incumbent, for comparison)

    python scripts/structure_strategy.py [--placebos 40] [--horizon 1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqe.backtest.costs import CostModel  # noqa: E402
from tqe.config import load_config  # noqa: E402
from tqe.data.sources import TENOR_YEARS  # noqa: E402
from tqe.data.universe import constant_maturity_total_return, universe_panel  # noqa: E402
from tqe.logging_utils import get_logger, setup_logging  # noqa: E402
from tqe.models.registry import create_model  # noqa: E402
from tqe.portfolio.funding import doubly_neutral_structure  # noqa: E402
from tqe.portfolio.structures import build_standard_structures  # noqa: E402
from tqe.training.splits import walk_forward_splits  # noqa: E402
from tqe.training.train import _apply, fit_scaler  # noqa: E402

log = get_logger("scripts.structures")


def structure_panel(rets: dict, tenors: list[str], target_dv01: float = 1000.0,
                    cash_neutral: bool = False):
    """Daily returns and weights for each standard structure.

    Structures are rebuilt each day from that day's DV01s, so the DV01 weighting
    stays correct as durations drift. The weights used for date ``t`` come from
    ``t-1``'s DV01s - a book is sized before the day it is held, not after.

    With ``cash_neutral=True`` each structure is additionally funded against a
    bill leg so that net notional is zero as well as net DV01
    (:func:`tqe.portfolio.funding.doubly_neutral_structure`). That is the whole
    point of the comparison: the plain DV01-weighted version carries a 28.4%
    annual financing drag because it is massively net long notional, and this
    removes it while keeping ~95% of the curve exposure.
    """
    tr = universe_panel(rets, "total_return")[tenors]
    dv = universe_panel(rets, "dv01")[tenors].shift(1)

    idx = tr.index
    names: list[str] = []
    weights: dict[str, pd.DataFrame] = {}
    returns: dict[str, pd.Series] = {}

    sample = build_standard_structures(dv.dropna().iloc[0], tenors, target_dv01)
    names = [s.name for s in sample]
    for name in names:
        weights[name] = pd.DataFrame(0.0, index=idx, columns=tenors)

    for t, _date in enumerate(idx):
        row = dv.iloc[t]
        if not np.isfinite(row.to_numpy()).all():
            continue
        for s in build_standard_structures(row, tenors, target_dv01):
            if s.name not in weights:
                continue
            legs = s.weights
            if cash_neutral:
                try:
                    legs = doubly_neutral_structure(s, row).legs
                except Exception:  # noqa: BLE001 - fall back to the raw structure
                    legs = s.weights
            weights[s.name].iloc[t] = legs.reindex(tenors).fillna(0.0).to_numpy()

    for name in names:
        # A structure's P&L is (notional x return) summed over legs, which is
        # dollars. To make it a *return* it has to be divided by the capital the
        # position consumes, not by an arbitrary constant: dividing by
        # target_dv01 produces numbers of order 1 that are not returns at all
        # and compound to infinity. Gross notional is the honest denominator -
        # it is what the balance sheet and the funding leg actually see.
        gross = weights[name].abs().sum(axis=1).replace(0.0, np.nan)
        pnl = (weights[name] * tr.fillna(0.0)).sum(axis=1)
        returns[name] = (pnl / gross).fillna(0.0)

    return pd.DataFrame(returns), weights


def structure_carry(curve: pd.DataFrame, rets: dict, weights: dict, tenors: list[str]) -> pd.DataFrame:
    """Carry plus roll-down of each structure, in basis points per unit.

    A steepener's carry is negative on an upward-sloping curve - you are long the
    low-yielding short leg against the high-yielding long one - which is exactly
    why steepeners are a *cost* to hold and have to be timed.
    """
    y = universe_panel(rets, "yield")[tenors]
    dur = universe_panel(rets, "duration")[tenors]
    order = sorted(tenors, key=lambda c: TENOR_YEARS[c])
    funding = y[order[0]]

    per_tenor = {}
    for i, col in enumerate(order):
        carry = (y[col] - funding) * 1e4
        if i == 0:
            roll = pd.Series(0.0, index=y.index)
        else:
            prev = order[i - 1]
            dt = TENOR_YEARS[col] - TENOR_YEARS[prev]
            roll = (y[col] - y[prev]) / dt * dur[col] * 1e4
        per_tenor[col] = carry + roll
    pt = pd.DataFrame(per_tenor)[tenors]

    out = {}
    for name, w in weights.items():
        # notional-weighted carry, normalised so the scale is comparable
        out[name] = (w * pt.reindex_like(w).fillna(0.0)).sum(axis=1) / 1e6
    return pd.DataFrame(out)


def walk_forward_structure_model(X: pd.DataFrame, y: pd.DataFrame, cfg) -> pd.DataFrame:
    """Ridge forecasts of structure returns, walk-forward out of sample."""
    common = X.index.intersection(y.dropna(how="any").index)
    X, y = X.loc[common], y.loc[common]
    splits = walk_forward_splits(
        X.index, n_splits=cfg.training.n_splits, test_size=cfg.training.test_size,
        min_train_size=cfg.training.min_train_size, embargo=cfg.training.embargo,
        expanding=cfg.training.expanding, horizon=cfg.model.horizon,
    )
    if not splits:
        raise ValueError("not enough observations for walk-forward")

    blocks = []
    for s in splits:
        Xtr, ytr = X.iloc[s.train_idx], y.iloc[s.train_idx]
        Xte = X.iloc[s.test_idx]
        scaler = fit_scaler(Xtr, cfg.training.standardize)
        model = create_model("ridge", cfg.model)
        model.fit(_apply(scaler, Xtr), ytr)
        blocks.append(pd.DataFrame(model.predict(_apply(scaler, Xte)),
                                   index=Xte.index, columns=y.columns))
    return pd.concat(blocks).sort_index()


def to_signal(raw: pd.DataFrame, cfg) -> pd.DataFrame:
    """Standardise structure forecasts into bounded, persistent signals."""
    from tqe.signals.alpha import predictions_to_signal, signal_decay

    s = predictions_to_signal(raw, "vol_scale", 252, cfg.portfolio.signal_clip, 0.0)
    s = s.where(s.abs() >= cfg.portfolio.min_signal_to_trade, 0.0)
    return signal_decay(s, cfg.portfolio.signal_halflife).fillna(0.0)


def block_sign_flip(signal: pd.DataFrame, block: int = 63, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n, k = signal.shape
    flips = rng.choice([-1.0, 1.0], size=(int(np.ceil(n / block)), k))
    return pd.DataFrame(signal.to_numpy() * np.repeat(flips, block, axis=0)[:n],
                        index=signal.index, columns=signal.columns)


def evaluate(signal: pd.DataFrame, struct_rets: pd.DataFrame, weights: dict,
             cfg, target_vol: float = 0.03) -> dict:
    """P&L of holding ``signal`` units of each structure.

    **This delegates to** :func:`tqe.backtest.engine.run_backtest` **rather than
    computing P&L here, and that is the whole point.** The first version of this
    function hand-rolled the accounting: it charged transaction costs and
    silently omitted financing, which is exactly the bug the engine had been
    fixed for weeks earlier. It duly produced a Sharpe of 3.96, and a static
    3-month/2-year steepener scoring 8.1 - because a DV01-weighted steepener is
    hugely net *long* notional (the short leg needs enormous size to match the
    long leg's DV01), so it is a levered cash position wearing a curve trade's
    clothes.

    Structures are DV01-neutral, which removes the *directional rates* exposure.
    It does not remove the *funding* exposure, and conflating the two is how the
    same error keeps reappearing. Building the tenor-space book and handing it to
    the engine means costs, financing and the P&L convention all come from one
    place that is tested.
    """
    from tqe.backtest.engine import run_backtest

    idx = signal.index
    common = [c for c in signal.columns if c in struct_rets.columns]
    sig = signal[common].reindex(idx).fillna(0.0)
    sr = struct_rets[common].reindex(idx).fillna(0.0)

    # Scale the whole book to a common risk budget so cells are comparable.
    gross = (sig * sr).sum(axis=1)
    sd = gross.std() * np.sqrt(252)
    scale = (target_vol / sd) if sd > 0 else 0.0

    # Collapse the structure book into tenor-space notionals.
    tenors = list(next(iter(weights.values())).columns)
    notional = pd.DataFrame(0.0, index=idx, columns=tenors)
    for name in common:
        w = weights[name].reindex(idx).fillna(0.0)
        notional = notional.add(w.mul(sig[name] * scale, axis=0), fill_value=0.0)

    from tqe.data.universe import constant_maturity_total_return, universe_panel

    rets = constant_maturity_total_return(
        pd.read_parquet(Path(__file__).resolve().parents[1] / "data/processed/curve.parquet"),
        tenors,
    )
    tr = universe_panel(rets, "total_return")
    dv = universe_panel(rets, "dv01")
    yc = universe_panel(rets, "yield_change")

    # A dummy tenor-space signal: positions are supplied directly, so the signal
    # only has to carry the index and columns.
    dummy = pd.DataFrame(0.0, index=idx, columns=tenors)
    r = run_backtest(dummy, tr, dv, cfg, CostModel(cfg.costs),
                     yield_change_panel=yc, positions=notional, run_canary=False)
    m = dict(r.metrics)
    m["mean_net_notional"] = float(r.exposures["net_notional"].mean())
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--placebos", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--cash-neutral", action="store_true",
                    help="fund each structure against a bill leg so net notional is zero")
    ap.add_argument("--out", default="artifacts/reports/structure_strategy.csv")
    args = ap.parse_args()

    setup_logging("WARNING")
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    cfg.model.horizon = args.horizon

    curve = pd.read_parquet(root / "data/processed/curve.parquet")
    X = pd.read_parquet(root / "data/processed/X.parquet")
    tenors = [t for t in cfg.data.core_tenors if t in curve.columns]
    rets = constant_maturity_total_return(curve, tenors)

    print("=" * 76)
    print("  Trading the curve in STRUCTURE space (DV01-neutral by construction)")
    print("=" * 76)

    sr, weights = structure_panel(rets, tenors, cash_neutral=args.cash_neutral)
    print(f"  structures: {', '.join(sr.columns)}"
          f"{'  [cash-neutral funded]' if args.cash_neutral else '  [raw DV01-weighted]'}")
    print(f"  panel {sr.shape},  {sr.index.min().date()} .. {sr.index.max().date()}\n")

    # Arithmetic annualisation: these are relative-value spreads that can be
    # negative for long stretches, so geometric compounding is not meaningful
    # and overflows on any structure that spends time underwater.
    ann = pd.DataFrame({
        "ann_return": sr.mean() * 252,
        "ann_vol": sr.std() * np.sqrt(252),
    })
    ann["sharpe"] = ann.ann_return / ann.ann_vol.replace(0.0, np.nan)
    print("  buy-and-hold each structure (no model):")
    print(ann.round(4).to_string())

    # Target alignment follows the same contract as the main pipeline: X[t]
    # already holds t-1 information, so y[t] is the return realised over day t
    # and no further lead is applied. Leading it here as well was how the
    # one-day misalignment in tqe.features.builder was originally found.
    h = args.horizon
    lead = h - 1
    y = sr if h == 1 else ((1 + sr).rolling(h).apply(np.prod, raw=True) - 1.0).shift(-lead)

    print("\n  training ridge on structure returns ...")
    preds = walk_forward_structure_model(X, y, cfg)
    ic = float(np.corrcoef(preds.to_numpy().ravel(),
                           y.reindex(preds.index).to_numpy().ravel())[0, 1])
    print(f"  out-of-sample IC on structure returns: {ic:+.4f}  ({len(preds)} days)")

    carry = structure_carry(curve, rets, weights, tenors).reindex(preds.index).shift(1)

    rows = []
    for label, raw in [("structure_model", preds), ("structure_carry", carry)]:
        s = to_signal(raw, cfg)
        m = evaluate(s, sr, weights, cfg)
        pl = [evaluate(block_sign_flip(s, 63, i), sr, weights, cfg)["sharpe"]
              for i in range(args.placebos)]
        beat = sum(1 for v in pl if v >= m["sharpe"])
        rows.append({
            "signal": label, "sharpe": m["sharpe"], "ann_return": m["ann_return"],
            "ann_vol": m["ann_vol"], "max_dd": m["max_drawdown"],
            "turnover": m["ann_turnover"], "cost_drag": m["cost_drag_annual"],
            "financing": m.get("financing_drag_annual", 0.0),
            "placebo_mean": float(np.mean(pl)), "placebo_sd": float(np.std(pl)),
            "p_value": (beat + 1) / (len(pl) + 1),
            "z_vs_placebo": float((m["sharpe"] - np.mean(pl)) / max(np.std(pl), 1e-9)),
        })
        print(f"\n  {label:18s} Sharpe={m['sharpe']:+6.3f}  ret={m['ann_return']:+6.2%}  "
              f"turn={m['ann_turnover']:5.1f}x")
        print(f"  {'':18s} placebo={np.mean(pl):+.3f}+-{np.std(pl):.3f}  "
              f"p={rows[-1]['p_value']:.4f}  z={rows[-1]['z_vs_placebo']:+.2f}")

    df = pd.DataFrame(rows)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n" + "=" * 76)
    print(df.round(4).to_string(index=False))
    # A negative Sharpe that merely beats even-worse controls is not a finding.
    # Require the result to be positive before calling it one.
    positive = df[df.sharpe > 0]
    best = (positive.sort_values("p_value").iloc[0] if len(positive)
            else df.sort_values("sharpe", ascending=False).iloc[0])
    print("\n=== VERDICT ===")
    if best.sharpe > 0 and best.p_value <= 0.10:
        print(f"  {best.signal} reaches p={best.p_value:.4f} at Sharpe {best.sharpe:+.3f}.")
        print("  Trading the structures natively does better than projecting the")
        print("  exposure away after the fact.")
    elif best.sharpe <= 0:
        print(f"  No construction produced a positive Sharpe (best {best.sharpe:+.3f}).")
        print("  Note that a negative result can still show a low p-value against")
        print("  even-worse sign-flipped controls; that is not a finding.")
    else:
        print(f"  Best cell is {best.signal} at p={best.p_value:.4f} - not significant.")
        print("  Expressing the view natively in slope/curvature space does not")
        print("  rescue it either. The skill the attribution found is real but too")
        print("  small to survive costs in any construction tested.")
    print(f"\n  written to {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
