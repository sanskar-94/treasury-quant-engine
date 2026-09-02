"""Calibration of the block_sign_flip p-value on many KNOWN-ZERO-SKILL signals,
plus an exchangeability diagnostic and a corrected null."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/Users/sanskarawasthi/trade bot 4")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from tqe.logging_utils import setup_logging
setup_logging("ERROR")
from tqe.config import load_config
from tqe.data.universe import constant_maturity_total_return, universe_panel
import benchmark_carry as bc

cfg = load_config(ROOT / "configs" / "default.yaml")
curve = pd.read_parquet(ROOT / "data/processed/curve.parquet")
preds = pd.read_parquet(ROOT / "data/processed/oos_predictions.parquet")
tenors = [t for t in cfg.data.core_tenors if t in curve.columns]
rets = constant_maturity_total_return(curve, tenors)
tr = universe_panel(rets, "total_return"); dv = universe_panel(rets, "dv01")
yc = universe_panel(rets, "yield_change")
sig = bc.to_signal(preds[tenors], cfg)
n = len(sig)

def sh(s):
    return bc.evaluate(s, tr, dv, yc, cfg).metrics["sharpe"]

def pval(s, k=40):
    m = sh(s)
    pl = [sh(bc.block_sign_flip(s, 63, i)) for i in range(k)]
    beat = sum(1 for v in pl if v >= m)
    return m, float(np.mean(pl)), float(np.std(pl)), (beat + 1) / (k + 1)

rng = np.random.default_rng(777)
offs = rng.integers(300, n - 300, size=60)
rows = []
for k, off in enumerate(offs):
    rot = pd.DataFrame(np.roll(sig.to_numpy(), int(off), axis=0),
                       index=sig.index, columns=sig.columns)
    m, pm, ps, p = pval(rot)
    rows.append({"kind": "rotation", "off": int(off), "sharpe": m,
                 "placebo_mean": pm, "p": p})
    print(f"rot {k:2d} p={p:.4f} sh={m:+.4f} pm={pm:+.4f}", flush=True)

# --- exchangeability control: use a FLIPPED signal as the "observed" one.
# If the null were exchangeable, p here must be uniform BY CONSTRUCTION.
for k in range(30):
    obs = bc.block_sign_flip(sig, 63, 5000 + k)
    m = sh(obs)
    pl = [sh(bc.block_sign_flip(sig, 63, i)) for i in range(40)]
    beat = sum(1 for v in pl if v >= m)
    p = (beat + 1) / 41
    rows.append({"kind": "flipped_obs", "off": 5000 + k, "sharpe": m,
                 "placebo_mean": float(np.mean(pl)), "p": p})
    print(f"flipobs {k:2d} p={p:.4f} sh={m:+.4f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(ROOT / "artifacts/audit2/siglens/null_big.csv", index=False)
for kind, g in df.groupby("kind"):
    print(f"\n=== {kind}  n={len(g)} ===")
    print(f"  mean p {g.p.mean():.4f}  median {g.p.median():.4f}  "
          f"frac<=0.10 {(g.p<=0.10).mean():.3f}  frac<=0.05 {(g.p<=0.0488).mean():.3f} "
          f" frac at floor {(g.p<=0.0245).mean():.3f}")
    print(f"  mean sharpe {g.sharpe.mean():+.4f}  mean placebo_mean {g.placebo_mean.mean():+.4f}")
