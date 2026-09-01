#!/usr/bin/env python
"""Do the new modules actually help?

Four modules were added because measurements pointed at them - term premium,
regime switching, cash-neutral funding, execution scheduling. Building them is
not the same as showing they earn their place, and a system that accumulates
machinery nothing uses is worse than a smaller one that does.

This tests the two that make a testable claim about the *signal*:

``baseline``
    the shipped ensemble's out-of-sample predictions, unchanged.
``+ term premium``
    blended with a term-premium signal. The model has no directional skill; the
    term premium is a direct estimate of what owning duration pays, so it should
    supply exactly what the forecast lacks.
``regime conditional``
    the baseline signal scaled by the HMM's filtered probability of the calm
    state. Carry returned +1.79% in 2020 at +80bp slope and -4.29% in 2022 at
    -53bp; if that regime dependence is real, damping the signal in the
    high-volatility state should help.
``both``
    term premium blended and regime scaled.

Every variant goes through the identical honest evaluation: a double-neutral
funded book (zero net cash, zero net DV01, financing and costs charged by
``run_backtest``) scored against block-sign-flip controls that preserve
persistence and destroy only the timing.

    python scripts/integration_experiment.py [--placebos 40]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_carry import block_sign_flip, evaluate, to_signal  # noqa: E402

from tqe.config import load_config  # noqa: E402
from tqe.curve.bootstrap import bootstrap_history  # noqa: E402
from tqe.curve.term_premium import decompose_term_premium  # noqa: E402
from tqe.data.universe import constant_maturity_total_return, universe_panel  # noqa: E402
from tqe.logging_utils import get_logger, setup_logging  # noqa: E402
from tqe.models.regime_switching import rolling_regime_probs  # noqa: E402

log = get_logger("scripts.integration")


def zscore(frame: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Trailing standardisation - causal, like everything else here."""
    mp = max(20, window // 4)
    mu = frame.rolling(window, min_periods=mp).mean()
    sd = frame.rolling(window, min_periods=mp).std()
    return ((frame - mu) / sd.where(sd.abs() > 1e-12)).replace([np.inf, -np.inf], np.nan)


def build_term_premium_signal(curve: pd.DataFrame, tenors: list[str]) -> pd.DataFrame:
    """Term premium per tenor, standardised, as a tradable view.

    A high term premium means duration is well paid relative to the expected
    path of short rates, so the signal is long the tenors whose premium is
    stretched. Cross-sectionally demeaned so it expresses relative value rather
    than a directional duration bet, which is what the double-neutral book can
    actually hold.
    """
    zero = bootstrap_history(curve)[tenors]
    res = decompose_term_premium(zero, n_factors=5, lags=1, window=1260,
                                 min_periods=504, refit_every=63)
    tp = res.term_premium[tenors]
    z = zscore(tp, 252)
    return z.sub(z.mean(axis=1), axis=0)


def build_regime_scale(curve: pd.DataFrame, anchor: str = "10 Yr") -> pd.Series:
    """Filtered probability of the calm state, used to damp the signal.

    **Filtered, not smoothed.** The smoothed probability conditions on the whole
    sample and would be look-ahead by construction; the filtered one uses only
    information up to each date. This distinction is the single easiest way to
    manufacture a regime model that works beautifully and is worthless.

    Returned as a multiplier in ``[0, 1]``: full size in the calm state, damped
    in the volatile one.
    """
    dy = curve[anchor].diff().dropna()
    probs = rolling_regime_probs(dy, n_states=2, window=1260, min_periods=504,
                                 refit_every=63)
    cols = [c for c in probs.columns if c.startswith("regime_p")]
    if not cols:
        return pd.Series(1.0, index=curve.index)
    # States are sorted, so column 0 is the low-volatility regime.
    calm = probs[cols[0]]
    return calm.reindex(curve.index).ffill().fillna(0.5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--placebos", type=int, default=40)
    ap.add_argument("--tp-weight", type=float, default=0.5,
                    help="weight on the term-premium signal in the blend")
    ap.add_argument("--out", default="artifacts/reports/integration_experiment.csv")
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

    print("=" * 78)
    print("  Do the new modules earn their place?  Double-neutral, funded, costed")
    print("=" * 78)
    print(f"  window {idx.min().date()} .. {idx.max().date()}  ({len(idx)} days)")
    print(f"  placebos per variant: {args.placebos}\n")

    base_sig = to_signal(preds[tenors], cfg)

    print("  building term premium ...", flush=True)
    tp_raw = build_term_premium_signal(curve, tenors).reindex(idx)
    tp_sig = to_signal(tp_raw, cfg)

    print("  fitting rolling regime probabilities ...", flush=True)
    calm = build_regime_scale(curve).reindex(idx).fillna(0.5)
    print(f"  calm-state probability: mean {calm.mean():.3f}, "
          f"{(calm > 0.5).mean():.1%} of days above 0.5\n")

    # Blend on standardised signals so neither dominates by scale alone.
    def blend(a: pd.DataFrame, b: pd.DataFrame, w: float) -> pd.DataFrame:
        an = a.div(a.std().replace(0.0, np.nan), axis=1).fillna(0.0)
        bn = b.div(b.std().replace(0.0, np.nan), axis=1).fillna(0.0)
        return to_signal((1.0 - w) * an + w * bn, cfg)

    variants = {
        "baseline": base_sig,
        "term_premium_only": tp_sig,
        "+ term premium": blend(base_sig, tp_sig, args.tp_weight),
        "regime conditional": to_signal(base_sig.mul(calm, axis=0), cfg),
        "both": to_signal(blend(base_sig, tp_sig, args.tp_weight).mul(calm, axis=0), cfg),
    }

    rows = []
    for name, sig in variants.items():
        m = evaluate(sig, tr, dv, yc, cfg).metrics
        pl = [evaluate(block_sign_flip(sig, 63, i), tr, dv, yc, cfg).metrics["sharpe"]
              for i in range(args.placebos)]
        beat = sum(1 for v in pl if v >= m["sharpe"])
        rows.append({
            "variant": name, "sharpe": m["sharpe"], "ann_return": m["ann_return"],
            "ann_vol": m["ann_vol"], "max_dd": m["max_drawdown"],
            "turnover": m["ann_turnover"],
            "placebo_mean": float(np.mean(pl)), "placebo_sd": float(np.std(pl)),
            "p_value": (beat + 1) / (len(pl) + 1),
            "z_vs_placebo": float((m["sharpe"] - np.mean(pl)) / max(np.std(pl), 1e-9)),
        })
        print(f"  {name:20s} Sharpe={m['sharpe']:+6.3f}  ret={m['ann_return']:+6.2%}  "
              f"turn={m['ann_turnover']:5.1f}x  placebo={np.mean(pl):+.3f}+-{np.std(pl):.3f}  "
              f"p={rows[-1]['p_value']:.4f}")

    df = pd.DataFrame(rows)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # ---- multiple testing: five variants were compared ---- #
    m_tests = len(df)
    ranked = df.sort_values("p_value").reset_index(drop=True)
    holm, running = [], 0.0
    for i, raw in enumerate(ranked["p_value"]):
        running = max(running, min(1.0, raw * (m_tests - i)))
        holm.append(running)
    ranked["p_holm"] = holm

    print("\n" + "=" * 78)
    print(ranked[["variant", "sharpe", "p_value", "p_holm", "z_vs_placebo", "turnover"]]
          .round(4).to_string(index=False))

    base = df[df.variant == "baseline"].iloc[0]
    print("\n=== VERDICT ===")
    print(f"  baseline Sharpe {base.sharpe:+.3f} (p={base.p_value:.3f})")
    improved = [r for _, r in df.iterrows()
                if r.variant != "baseline" and r.sharpe > base.sharpe]
    if improved:
        for r in improved:
            print(f"  {r.variant:20s} improves to {r.sharpe:+.3f} (p={r.p_value:.3f})")
    else:
        print("  No variant improves on the baseline.")
    survivors = ranked[(ranked.p_holm <= 0.10) & (ranked.sharpe > 0)]
    print(f"\n  variants significant after Holm correction for {m_tests} comparisons: "
          f"{len(survivors)}/{m_tests}")
    if len(survivors) == 0:
        print("  Neither the term premium nor regime conditioning turns this into a")
        print("  tradable strategy. Both are correct implementations of things that")
        print("  should have helped; that they do not is the result.")
    print(f"\n  written to {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
