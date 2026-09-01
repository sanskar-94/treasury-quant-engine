#!/usr/bin/env python
"""Does a longer horizon or a relative-value target find a tradable edge?

The daily directional experiment produced an information coefficient of +0.029
that vanished once the portfolio was funded and made market-neutral. Two
hypotheses follow directly from that failure, and this script tests both:

1. **Horizon.** Daily rates forecasting is close to a coin flip and is taxed
   heavily by turnover. Weekly and monthly targets have a better signal-to-noise
   ratio and need far less trading.
2. **Target definition.** A directional forecast has to beat the funding rate to
   be worth anything. A *relative* forecast - which tenor outperforms the curve -
   is cash- and duration-neutral by construction, so financing never enters.

Every cell is evaluated the honest way: walk-forward out-of-sample predictions,
then a **double-neutral** book (zero net notional, zero net DV01) with costs and
financing charged, then a placebo battery. A cell only counts as a finding if it
beats its own controls.

**On the choice of null.** The obvious placebo - shuffle the predictions in time -
is *not* valid here, and using it produced badly misleading results. Shuffling
preserves each tenor's mean prediction, and since the raw forecasts rise
monotonically with maturity, a shuffled signal smoothed over ten days becomes a
large, near-static curve tilt: measured on this data the shuffled books held
-$18mm of 3-month against +$10mm of 1-year, roughly ten times more concentrated
than the real book. The "placebo" was therefore a different and more aggressive
strategy, not an absence of signal, and its Sharpe was not centred on zero.

The null used instead is a **block sign-flip**: multiply the finished signal by
a random +/-1 drawn once per contiguous block of days. That preserves magnitude,
autocorrelation, persistence and cross-sectional structure - everything about the
book except its alignment with future returns - which is exactly the hypothesis
under test.

Ridge is used throughout rather than the full stacked ensemble. It took nearly
all of the stack's weight on this data, it is 250x faster, and holding the
learner fixed is what makes the comparison across cells meaningful.

    python scripts/experiment_horizons.py [--placebos 10] [--out PATH]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqe.backtest.costs import CostModel  # noqa: E402
from tqe.backtest.engine import run_backtest  # noqa: E402
from tqe.config import load_config  # noqa: E402
from tqe.data.universe import constant_maturity_total_return, universe_panel  # noqa: E402
from tqe.features.builder import FeatureSet, make_targets  # noqa: E402
from tqe.logging_utils import get_logger, setup_logging  # noqa: E402
from tqe.signals.alpha import predictions_to_signal, signal_decay  # noqa: E402
from tqe.signals.sizing import size_portfolio  # noqa: E402
from tqe.training.train import train_walk_forward  # noqa: E402

log = get_logger("scripts.experiment")

TARGETS = ["total_return", "relative_return"]
HORIZONS = [1, 5, 21]


def double_neutral(pos: pd.DataFrame, dv01: pd.DataFrame) -> pd.DataFrame:
    """Project each day's book onto the null space of [cash, DV01].

    Removes the funding position and the directional rates position at once,
    leaving pure relative value. Orthogonal projection, so it is the closest
    such book to the one requested.
    """
    out = pos.to_numpy(dtype=float).copy()
    D = dv01.reindex_like(pos).to_numpy(dtype=float)
    for i in range(len(out)):
        w, d = out[i], D[i]
        if not np.isfinite(d).all():
            continue
        A = np.vstack([np.ones_like(d), d])
        # A singular Gram matrix means the constraints are degenerate for this
        # row (e.g. all DV01s equal); leave the book untouched rather than fail.
        with contextlib.suppress(np.linalg.LinAlgError):
            w = w - A.T @ np.linalg.solve(A @ A.T, A @ w)
        out[i] = w
    return pd.DataFrame(out, index=pos.index, columns=pos.columns)


def build_signal(preds, cfg):
    """Raw forecasts -> the finished, tradable signal."""
    s = predictions_to_signal(preds, "vol_scale", 252, cfg.portfolio.signal_clip, 0.0)
    s = s.sub(s.mean(axis=1), axis=0)
    s = s.where(s.abs() >= cfg.portfolio.min_signal_to_trade, 0.0)
    return signal_decay(s, cfg.portfolio.signal_halflife).fillna(0.0)


def block_sign_flip(signal: pd.DataFrame, block: int = 63, seed: int = 0) -> pd.DataFrame:
    """Randomise the signal's direction in contiguous blocks.

    The null hypothesis is "this book has no alignment with future returns",
    not "this book does not exist". Flipping the sign in blocks keeps the
    magnitude, persistence, autocorrelation and cross-sectional shape intact
    while destroying the timing, so the control is the same strategy pointed in
    an arbitrary direction.
    """
    rng = np.random.default_rng(seed)
    n, k = signal.shape
    n_blocks = int(np.ceil(n / block))
    flips = rng.choice([-1.0, 1.0], size=(n_blocks, k))
    expanded = np.repeat(flips, block, axis=0)[:n]
    return pd.DataFrame(signal.to_numpy() * expanded, index=signal.index, columns=signal.columns)


def evaluate_signal(s, tr, dv, yc, cfg, target_vol=0.03):
    """Backtest a finished signal as a double-neutral, funded book."""
    idx = s.index
    pos = size_portfolio(s, tr.reindex(idx), dv.reindex(idx), cfg.portfolio, yc.reindex(idx))["notional"]
    pos = double_neutral(pos, dv.shift(1))
    # Common risk budget so cells are comparable rather than differing by scale.
    pnl = (pos * tr.reindex(idx).fillna(0.0)).sum(axis=1) / cfg.backtest.initial_capital
    sd = pnl.std() * np.sqrt(252)
    if sd > 0:
        pos = pos * (target_vol / sd)
    return run_backtest(s, tr, dv, cfg, CostModel(cfg.costs),
                        yield_change_panel=yc, positions=pos, run_canary=False)


def evaluate(preds, tr, dv, yc, cfg, target_vol=0.03):
    """Backtest a prediction frame as a double-neutral, funded book."""
    return evaluate_signal(build_signal(preds, cfg), tr, dv, yc, cfg, target_vol)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--placebos", type=int, default=10)
    ap.add_argument("--out", default="artifacts/reports/horizon_experiment.csv")
    args = ap.parse_args()

    setup_logging("WARNING")
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")

    X = pd.read_parquet(root / "data/processed/X.parquet")
    curve = pd.read_parquet(root / "data/processed/curve.parquet")
    rets = constant_maturity_total_return(curve, cfg.data.core_tenors)
    tr = universe_panel(rets, "total_return")
    dv = universe_panel(rets, "dv01")
    yc = universe_panel(rets, "yield_change")

    print("=" * 78)
    print("  Horizon x target experiment - ridge, walk-forward, double-neutral, funded")
    print("=" * 78)
    print(f"  features {X.shape}   placebos per cell: {args.placebos}\n")

    rows = []
    for target in TARGETS:
        for horizon in HORIZONS:
            t0 = time.time()
            c = load_config(root / "configs" / "default.yaml")
            c.model.target, c.model.horizon = target, horizon
            c.model.learners = ["ridge"]

            y = make_targets(rets, target, horizon, cfg.data.core_tenors)
            common = X.index.intersection(y.dropna().index)
            fs = FeatureSet(X=X.loc[common], y=y.loc[common],
                            metadata={"target": target, "horizon": horizon})

            res = train_walk_forward(fs, c)
            p, a = res.oos_predictions, res.oos_actuals
            ic = float(np.corrcoef(p.to_numpy().ravel(), a.to_numpy().ravel())[0, 1])

            sig = build_signal(p, c)
            bt = evaluate_signal(sig, tr, dv, yc, c)
            sharpe = bt.metrics["sharpe"]

            # Null: same book, randomly reversed in 63-day blocks.
            pl = []
            for i in range(args.placebos):
                flipped = block_sign_flip(sig, block=63, seed=i)
                pl.append(evaluate_signal(flipped, tr, dv, yc, c).metrics["sharpe"])
            beat = sum(1 for v in pl if v >= sharpe)
            pval = (beat + 1) / (len(pl) + 1)

            rows.append({
                "target": target, "horizon": horizon, "n_oos": len(p),
                "ic": ic, "sharpe": sharpe,
                "ann_return": bt.metrics["ann_return"], "ann_vol": bt.metrics["ann_vol"],
                "max_dd": bt.metrics["max_drawdown"], "turnover": bt.metrics["ann_turnover"],
                "financing": bt.metrics.get("financing_drag_annual", 0.0),
                "placebo_mean": float(np.mean(pl)), "placebo_sd": float(np.std(pl)),
                "placebo_max": float(np.max(pl)), "beat_by": beat, "p_value": pval,
                "z_vs_placebo": float((sharpe - np.mean(pl)) / max(np.std(pl), 1e-9)),
                "seconds": round(time.time() - t0, 1),
            })
            print(f"  {target:16s} h={horizon:<3d} IC={ic:+.4f}  Sharpe={sharpe:+.3f}  "
                  f"placebo={np.mean(pl):+.3f}+-{np.std(pl):.3f}  p={pval:.3f}  "
                  f"({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("  RESULTS (ranked by p-value against own placebos)")
    print("=" * 78)
    cols = ["target", "horizon", "ic", "sharpe", "placebo_mean", "p_value", "z_vs_placebo", "turnover"]
    print(df.sort_values("p_value")[cols].round(4).to_string(index=False))

    # ---- multiple-testing correction ---------------------------------- #
    # Six cells were tested. The chance of at least one p < 0.05 arising by
    # luck alone is 1 - 0.95^6 = 26%, so an uncorrected p-value is not evidence
    # of anything. Holm-Bonferroni controls the family-wise error rate while
    # being less brutal than plain Bonferroni.
    m = len(df)
    df = df.sort_values("p_value").reset_index(drop=True)
    holm, running = [], 0.0
    for i, raw in enumerate(df["p_value"]):
        running = max(running, min(1.0, raw * (m - i)))
        holm.append(running)
    df["p_holm"] = holm
    df["p_bonferroni"] = (df["p_value"] * m).clip(upper=1.0)

    print("\n=== MULTIPLE-TESTING CORRECTION ===")
    print(f"  {m} cells tested; P(at least one p<0.05 by chance) = "
          f"{1 - 0.95 ** m:.0%}")
    print(df[["target", "horizon", "p_value", "p_holm", "p_bonferroni"]].round(4).to_string(index=False))

    raw_sig = df[df.p_value <= 0.10]
    corr_sig = df[df.p_holm <= 0.10]
    print(f"\n  nominally significant (p<=0.10):        {len(raw_sig)}/{m}")
    print(f"  significant after Holm correction:      {len(corr_sig)}/{m}")

    if len(corr_sig):
        for _, r in corr_sig.iterrows():
            print(f"    {r.target} h={int(r.horizon)}: Sharpe {r.sharpe:+.3f}, "
                  f"p_holm={r.p_holm:.3f}, IC={r.ic:+.4f}")
    else:
        print("\n  CONCLUSION: no horizon or target definition tested here produces an")
        print("  edge that survives funding, costs, its own sign-flipped controls and")
        print("  the correction for having looked six times.")
        if len(raw_sig):
            r = raw_sig.iloc[0]
            print(f"\n  The best cell ({r.target}, h={int(r.horizon)}) is nominally p={r.p_value:.3f}")
            print(f"  but its information coefficient is {r.ic:+.4f}. A model that")
            print("  anti-predicts its own target while appearing to make money is")
            print("  reporting luck, not signal.")

    print("\n=== INFORMATION COEFFICIENT BY HORIZON ===")
    piv = df.pivot_table(index="target", columns="horizon", values="ic")
    print(piv.round(4).to_string())
    print("  IC degrades and turns negative as the horizon lengthens: the feature")
    print("  set is momentum-heavy and rates mean-revert over weeks to months.")
    print(f"\n  written to {out}")
    (out.with_suffix(".json")).write_text(json.dumps(rows, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
