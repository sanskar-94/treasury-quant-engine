#!/usr/bin/env python
"""Where on the curve is the term premium best paid, and how should it be held?

Everything else in this project tried to *forecast* something and failed. The one
robust positive result was passive: holding 10-year duration, funded and costed,
earned a Sharpe of 0.47 across 1993-2026. That reframes the question. Not "when
should I own duration" - the timing test answered that, badly - but "which point
on the curve pays best per unit of risk, and does sizing it by volatility beat
sizing it by duration".

Three axes, all funded through ``run_backtest`` and costed:

    tenor          each core maturity held at constant DV01
    sizing         constant DV01 vs constant volatility (vol targeted)
    diversified    equal-risk across the curve rather than a single point

This is a selection and sizing question, not a forecasting one, so there is no
signal to sign-flip. The controls are the alternatives themselves: a choice only
counts if it beats the others out of sample, and the whole window is out of
sample because nothing here is fitted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqe.backtest.costs import CostModel  # noqa: E402
from tqe.backtest.engine import run_backtest  # noqa: E402
from tqe.config import load_config  # noqa: E402
from tqe.data.universe import constant_maturity_total_return, universe_panel  # noqa: E402
from tqe.logging_utils import setup_logging  # noqa: E402


def hold(pos: pd.DataFrame, tr, dv, yc, cfg) -> dict:
    """Funded, costed P&L of holding a given notional book."""
    dummy = pd.DataFrame(0.0, index=pos.index, columns=tr.columns)
    return run_backtest(dummy, tr, dv, cfg, CostModel(cfg.costs),
                        yield_change_panel=yc, positions=pos, run_canary=False).metrics


def constant_dv01_book(tenors, tr, dv, idx, cfg, target_vol=0.05, weights=None):
    """Hold a fixed DV01 in each tenor, scaled to a common volatility."""
    unit = (100.0 / dv[tenors].shift(1)).reindex(idx)      # notional per $1 DV01
    w = pd.Series(weights if weights is not None else 1.0, index=tenors, dtype=float)
    raw = unit.mul(w, axis=1)
    pnl = (raw * tr[tenors].reindex(idx)).sum(axis=1).fillna(0.0) / cfg.backtest.initial_capital
    sd = pnl.std() * np.sqrt(252)
    scaled = raw * (target_vol / sd) if sd > 0 else raw * 0.0
    pos = pd.DataFrame(0.0, index=idx, columns=tr.columns)
    pos[tenors] = scaled
    return pos


def vol_targeted_book(tenor, tr, dv, idx, cfg, target_vol=0.05, lookback=63):
    """Hold duration sized to a constant *volatility* rather than constant DV01.

    Trailing realised volatility, shifted a day - sizing today's position with
    today's volatility would use the observation the position is about to
    experience. The claim being tested is that constant risk beats constant
    duration, because a fixed DV01 runs far more risk in 2022 than in 2019.
    """
    unit = (100.0 / dv[tenor].shift(1)).reindex(idx)
    vol = tr[tenor].rolling(lookback, min_periods=lookback // 3).std().shift(1).reindex(idx)
    raw = unit / vol.where(vol.abs() > 1e-12)
    raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    pnl = (raw * tr[tenor].reindex(idx)).fillna(0.0) / cfg.backtest.initial_capital
    sd = pnl.std() * np.sqrt(252)
    scaled = raw * (target_vol / sd) if sd > 0 else raw * 0.0
    pos = pd.DataFrame(0.0, index=idx, columns=tr.columns)
    pos[tenor] = scaled
    return pos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="artifacts/reports/duration_harvest.csv")
    args = ap.parse_args()

    setup_logging("WARNING")
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    curve = pd.read_parquet(root / "data/processed/curve.parquet")
    tenors = [t for t in cfg.data.core_tenors if t in curve.columns]
    rets = constant_maturity_total_return(curve, tenors)
    tr, dv, yc = (universe_panel(rets, f) for f in ("total_return", "dv01", "yield_change"))
    idx = tr.dropna(how="any").index

    print("=" * 78)
    print("  Harvesting the term premium: where on the curve, and sized how?")
    print("=" * 78)
    print(f"  window {idx.min().date()} .. {idx.max().date()}  ({len(idx)} days)")
    print("  every arm funded and costed; nothing is fitted, so all of it is out of sample\n")

    rows = []
    print(f"  {'arm':<28} {'Sharpe':>8} {'ann ret':>9} {'ann vol':>8} {'maxDD':>8} {'fin':>7}")
    print("  " + "-" * 72)

    for t in tenors:
        m = hold(constant_dv01_book([t], tr, dv, idx, cfg), tr, dv, yc, cfg)
        rows.append({"arm": f"constant DV01 {t}", "kind": "single", "tenor": t, **_pick(m)})
        _show(f"constant DV01 {t}", m)

    print()
    for t in tenors:
        m = hold(vol_targeted_book(t, tr, dv, idx, cfg), tr, dv, yc, cfg)
        rows.append({"arm": f"vol targeted {t}", "kind": "voltgt", "tenor": t, **_pick(m)})
        _show(f"vol targeted {t}", m)

    print()
    m = hold(constant_dv01_book(tenors, tr, dv, idx, cfg), tr, dv, yc, cfg)
    rows.append({"arm": "equal DV01 across curve", "kind": "diversified", "tenor": "", **_pick(m)})
    _show("equal DV01 across curve", m)

    # Inverse-volatility weights across tenors - risk parity without a covariance.
    vol = tr[tenors].std()
    inv = (1.0 / vol) / (1.0 / vol).sum()
    m = hold(constant_dv01_book(tenors, tr, dv, idx, cfg, weights=inv), tr, dv, yc, cfg)
    rows.append({"arm": "inverse-vol across curve", "kind": "diversified", "tenor": "", **_pick(m)})
    _show("inverse-vol across curve", m)

    df = pd.DataFrame(rows)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    best = df.sort_values("sharpe", ascending=False).iloc[0]
    single = df[df.kind == "single"].sort_values("sharpe", ascending=False)
    print("\n=== VERDICT ===")
    print(f"  best single point : {single.iloc[0].arm} at {single.iloc[0].sharpe:+.3f}")
    print(f"  best overall      : {best.arm} at {best.sharpe:+.3f}")
    div = df[df.kind == "diversified"].sharpe.max()
    print(f"  diversifying across the curve {'beats' if div > single.iloc[0].sharpe else 'does not beat'} "
          f"the best single point ({div:+.3f} vs {single.iloc[0].sharpe:+.3f})")
    vt = df[df.kind == "voltgt"].sharpe.max()
    print(f"  volatility targeting {'beats' if vt > single.iloc[0].sharpe else 'does not beat'} "
          f"constant DV01 ({vt:+.3f} vs {single.iloc[0].sharpe:+.3f})")
    print(f"\n  written to {out}")
    return 0


def _pick(m: dict) -> dict:
    return {"sharpe": m["sharpe"], "ann_return": m["ann_return"], "ann_vol": m["ann_vol"],
            "max_dd": m["max_drawdown"], "financing": m.get("financing_drag_annual", 0.0),
            "turnover": m["ann_turnover"]}


def _show(name: str, m: dict) -> None:
    print(f"  {name:<28} {m['sharpe']:>8.3f} {m['ann_return']:>9.2%} {m['ann_vol']:>8.2%} "
          f"{m['max_drawdown']:>8.2%} {m.get('financing_drag_annual', 0.0):>7.2%}")


if __name__ == "__main__":
    raise SystemExit(main())
