"""Is block_sign_flip a valid null? Feed it signals with KNOWN ZERO skill."""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/Users/sanskarawasthi/trade bot 4")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tqe.logging_utils import setup_logging
setup_logging("ERROR")
from tqe.config import load_config
from tqe.data.universe import constant_maturity_total_return, universe_panel

import benchmark_carry as bc   # reuse the EXACT procedure under audit

cfg = load_config(ROOT / "configs" / "default.yaml")
curve = pd.read_parquet(ROOT / "data/processed/curve.parquet")
preds = pd.read_parquet(ROOT / "data/processed/oos_predictions.parquet")
tenors = [t for t in cfg.data.core_tenors if t in curve.columns]
rets = constant_maturity_total_return(curve, tenors)
tr = universe_panel(rets, "total_return")
dv = universe_panel(rets, "dv01")
yc = universe_panel(rets, "yield_change")
idx = preds.index
print("n days", len(idx), "tenors", tenors)

sig_model = bc.to_signal(preds[tenors], cfg)

def pval_for(sig, n_placebo=40):
    m = bc.evaluate(sig, tr, dv, yc, cfg).metrics
    pl = [bc.evaluate(bc.block_sign_flip(sig, 63, i), tr, dv, yc, cfg).metrics["sharpe"]
          for i in range(n_placebo)]
    beat = sum(1 for v in pl if v >= m["sharpe"])
    return m["sharpe"], float(np.mean(pl)), float(np.std(pl)), (beat + 1) / (len(pl) + 1), m

t0 = time.time()
s0, pm0, ps0, p0, m0 = pval_for(sig_model)
print(f"REAL model signal: sharpe={s0:+.4f} placebo={pm0:+.4f}+-{ps0:.4f} p={p0:.4f}  ({time.time()-t0:.1f}s)")

# ---- known-zero-skill signals: circular rotations of the real signal -------
n = len(sig_model)
rows = []
rng = np.random.default_rng(12345)
offsets = rng.integers(252, n - 252, size=24)
for k, off in enumerate(offsets):
    arr = np.roll(sig_model.to_numpy(), int(off), axis=0)
    rot = pd.DataFrame(arr, index=sig_model.index, columns=sig_model.columns)
    sh, pm, ps, p, _ = pval_for(rot)
    rows.append({"offset": int(off), "sharpe": sh, "placebo_mean": pm,
                 "placebo_sd": ps, "p": p, "z": (sh - pm) / max(ps, 1e-9)})
    print(f"  rot {k:2d} off={off:5d} sharpe={sh:+.4f} placebo={pm:+.4f}+-{ps:.4f} "
          f"p={p:.4f} z={(sh-pm)/max(ps,1e-9):+.2f}")

df = pd.DataFrame(rows)
df.to_csv(ROOT / "artifacts/audit2/siglens/null_rotations.csv", index=False)
print("\n=== NULL VALIDITY (24 zero-skill signals, 40 placebos each) ===")
print(f"  mean p               {df.p.mean():.4f}   (should be ~0.50)")
print(f"  median p             {df.p.median():.4f}   (should be ~0.50)")
print(f"  frac p<=0.10         {(df.p<=0.10).mean():.3f}   (should be ~0.10)")
print(f"  frac p<=0.05         {(df.p<=0.05).mean():.3f}   (should be ~0.05)")
print(f"  frac p<=0.025 (floor){(df.p<=0.0244).mean():.3f}  (should be ~0.024)")
print(f"  mean sharpe (real)   {df.sharpe.mean():+.4f}")
print(f"  mean placebo mean    {df.placebo_mean.mean():+.4f}")
print(f"  mean z               {df.z.mean():+.3f}   (should be ~0)")
print(f"  total time {time.time()-t0:.0f}s")
