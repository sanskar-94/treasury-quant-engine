#!/usr/bin/env python
"""Can the term premium time duration?

The integration experiment scored the term premium at -0.22 and the test was not
a fair one. A term premium is intrinsically *directional* - it says whether
owning duration is paid - and the double-neutral book used everywhere else in
this project cannot hold a directional view by construction. Demeaning it
cross-sectionally to fit that book strips out exactly the information it carries.

So this evaluates it the way it actually claims to work: a book allowed to hold
net duration, funded honestly, benchmarked against simply owning duration all the
time. The question is not "does duration pay" - over this window it barely does -
but "does the term premium tell you WHEN it pays".

Three arms, all funded and costed through run_backtest:

    static          constant long 10y-equivalent duration; the thing to beat
    tp_timed        long duration scaled by the standardised term premium
    tp_binary       long duration only when the premium is above its median

Controls are block sign-flips of the timing signal, which preserve persistence
and destroy only the timing - precisely the hypothesis under test.
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
from tqe.curve.bootstrap import bootstrap_history  # noqa: E402
from tqe.curve.term_premium import decompose_term_premium  # noqa: E402
from tqe.data.universe import constant_maturity_total_return, universe_panel  # noqa: E402
from tqe.logging_utils import setup_logging  # noqa: E402


def block_sign_flip(s: pd.Series, block: int = 63, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    n = len(s)
    flips = rng.choice([-1.0, 1.0], size=int(np.ceil(n / block)))
    return pd.Series(s.to_numpy() * np.repeat(flips, block)[:n], index=s.index)


def run(scale: pd.Series, tr, dv, yc, cfg, tenor: str, target_vol: float = 0.05) -> dict:
    """Hold `scale` units of duration in one tenor, funded and costed.

    Sized to a common volatility so the arms are comparable, then handed to the
    engine - which charges financing on the net notional, as it must for a book
    that is deliberately net long.
    """
    idx = scale.index
    unit = (100.0 / dv[tenor].shift(1)).reindex(idx)      # notional per $1 DV01
    raw = scale * unit
    pnl = (raw * tr[tenor].reindex(idx)).fillna(0.0) / cfg.backtest.initial_capital
    sd = pnl.std() * np.sqrt(252)
    if sd > 0:
        raw = raw * (target_vol / sd)

    pos = pd.DataFrame(0.0, index=idx, columns=tr.columns)
    pos[tenor] = raw
    dummy = pd.DataFrame(0.0, index=idx, columns=tr.columns)
    return run_backtest(dummy, tr, dv, cfg, CostModel(cfg.costs),
                        yield_change_panel=yc, positions=pos, run_canary=False).metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--placebos", type=int, default=40)
    ap.add_argument("--tenor", default="10 Yr")
    ap.add_argument("--out", default="artifacts/reports/duration_timing.csv")
    args = ap.parse_args()

    setup_logging("WARNING")
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    curve = pd.read_parquet(root / "data/processed/curve.parquet")
    tenors = [t for t in cfg.data.core_tenors if t in curve.columns]
    rets = constant_maturity_total_return(curve, tenors)
    tr, dv, yc = (universe_panel(rets, f) for f in ("total_return", "dv01", "yield_change"))

    zero = bootstrap_history(curve)[tenors]
    res = decompose_term_premium(zero, n_factors=5, lags=1, window=1260,
                                 min_periods=504, refit_every=63)
    tp = res.term_premium[args.tenor].dropna()
    mu = tp.rolling(504, min_periods=252).mean()
    sd = tp.rolling(504, min_periods=252).std()
    z = ((tp - mu) / sd.where(sd.abs() > 1e-12)).dropna().clip(-3, 3)
    idx = z.index

    print("=" * 74)
    print(f"  Can the term premium time duration?  ({args.tenor}, funded, costed)")
    print("=" * 74)
    print(f"  window {idx.min().date()} .. {idx.max().date()}  ({len(idx)} days)")
    print(f"  10y term premium: mean {tp.mean()*1e4:.0f}bp  sd {tp.std()*1e4:.0f}bp\n")

    arms = {
        "static long duration": pd.Series(1.0, index=idx),
        "tp_timed": z.clip(lower=0.0),                       # long only when paid
        "tp_binary": (z > 0).astype(float),
    }
    rows = []
    for name, sc in arms.items():
        m = run(sc, tr, dv, yc, cfg, args.tenor)
        if name == "static long duration":
            pl = []
        else:
            pl = [run(block_sign_flip(sc, 63, i).clip(lower=0.0), tr, dv, yc, cfg,
                      args.tenor)["sharpe"] for i in range(args.placebos)]
        beat = sum(1 for v in pl if v >= m["sharpe"]) if pl else 0
        rows.append({
            "arm": name, "sharpe": m["sharpe"], "ann_return": m["ann_return"],
            "ann_vol": m["ann_vol"], "max_dd": m["max_drawdown"],
            "financing": m.get("financing_drag_annual", 0.0),
            "turnover": m["ann_turnover"],
            "placebo_mean": float(np.mean(pl)) if pl else np.nan,
            "p_value": ((beat + 1) / (len(pl) + 1)) if pl else np.nan,
        })
        pm = f"placebo={np.mean(pl):+.3f}  p={rows[-1]['p_value']:.4f}" if pl else "(benchmark)"
        print(f"  {name:22s} Sharpe={m['sharpe']:+6.3f}  ret={m['ann_return']:+6.2%}  "
              f"fin={m.get('financing_drag_annual',0):+5.2%}  {pm}")

    df = pd.DataFrame(rows)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    base = df[df.arm == "static long duration"].iloc[0]
    print("\n=== VERDICT ===")
    print(f"  static duration Sharpe {base.sharpe:+.3f} - the hurdle")
    better = df[(df.arm != "static long duration") & (df.sharpe > base.sharpe)]
    if len(better):
        for _, r in better.iterrows():
            print(f"  {r.arm} beats it: {r.sharpe:+.3f} (p={r.p_value:.3f})")
    else:
        print("  Neither timing rule beats simply holding duration. The term premium")
        print("  is a sound decomposition; it is not, on this evidence, a timing signal.")
    print(f"\n  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
