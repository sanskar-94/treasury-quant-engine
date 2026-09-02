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

def stats(s):
    r = bc.evaluate(s, tr, dv, yc, cfg)
    m = r.metrics
    return dict(sharpe=m["sharpe"], ann_ret=m["ann_return"], vol=m["ann_vol"],
                turn=m["ann_turnover"], cost=m["cost_drag_annual"],
                fin=m["financing_drag_annual"], gross_sharpe=m["sharpe_gross"],
                gross_ret=m["ann_return_gross"], gross_not=m["avg_gross_notional"])

base = stats(sig)
print("REAL  ", {k: round(v, 5) for k, v in base.items()})
rows = []
for i in range(20):
    rows.append(stats(bc.block_sign_flip(sig, 63, i)))
pf = pd.DataFrame(rows)
print("PLACEBO mean:")
print(pf.mean().round(5).to_string())
print("\nDELTA (placebo mean - real):")
print((pf.mean() - pd.Series(base)).round(5).to_string())
print("\n--- decomposition of the Sharpe gap ---")
print(f"real   net ret {base['ann_ret']:+.5f} = gross {base['gross_ret']:+.5f} "
      f"- cost {base['cost']:.5f} - fin {base['fin']:.5f}")
print(f"plac   net ret {pf.ann_ret.mean():+.5f} = gross {pf.gross_ret.mean():+.5f} "
      f"- cost {pf.cost.mean():.5f} - fin {pf.fin.mean():.5f}")
print(f"\ngross-Sharpe gap  real {base['gross_sharpe']:+.4f} vs placebo {pf.gross_sharpe.mean():+.4f}")
print(f"cost gap in Sharpe units: {(pf.cost.mean()-base['cost'])/base['vol']:+.4f}")
print(f"fin  gap in Sharpe units: {(pf.fin.mean()-base['fin'])/base['vol']:+.4f}")
