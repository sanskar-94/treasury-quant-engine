"""benchmark_carry: (1) full-sample std used to blend; (2) placebo turnover bias."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
root = Path("/Users/sanskarawasthi/trade bot 4")
sys.path.insert(0, str(root/"src")); sys.path.insert(0, str(root/"scripts"))
from tqe.config import load_config
from tqe.data.universe import constant_maturity_total_return, universe_panel
from tqe.logging_utils import setup_logging
setup_logging("ERROR")
from benchmark_carry import carry_signal, to_signal, evaluate, block_sign_flip

cfg = load_config(root/"configs"/"default.yaml")
curve = pd.read_parquet(root/"data/processed/curve.parquet")
preds = pd.read_parquet(root/"data/processed/oos_predictions.parquet")
tenors = [t for t in cfg.data.core_tenors if t in curve.columns]
rets = constant_maturity_total_return(curve, tenors)
tr = universe_panel(rets,"total_return"); dv = universe_panel(rets,"dv01"); yc = universe_panel(rets,"yield_change")
idx = preds.index
carry = carry_signal(curve, rets, tenors).shift(1).reindex(idx)
sig_carry = to_signal(carry, cfg); sig_model = to_signal(preds[tenors], cfg)

def blend_full(a,b):
    return to_signal(a.div(a.std().replace(0,np.nan),axis=1).fillna(0.0)
                     + b.div(b.std().replace(0,np.nan),axis=1).fillna(0.0), cfg)
def blend_causal(a,b):
    sa = a.expanding(min_periods=63).std().shift(1).bfill()
    sb = b.expanding(min_periods=63).std().shift(1).bfill()
    return to_signal(a.div(sa.replace(0,np.nan)).fillna(0.0)
                     + b.div(sb.replace(0,np.nan)).fillna(0.0), cfg)

print("=== (1) full-sample std in the blend ===")
for nm, s in [("carry+model FULL-SAMPLE std", blend_full(sig_carry, sig_model)),
              ("carry+model CAUSAL std     ", blend_causal(sig_carry, sig_model))]:
    m = evaluate(s, tr, dv, yc, cfg).metrics
    print(f"  {nm}  Sharpe {m['sharpe']:+.4f}  vol {m['ann_vol']:.4%}  turn {m['ann_turnover']:.1f}")
print("  full-sample per-column std of sig_carry:\n", sig_carry.std().round(4).to_dict())
print("  full-sample per-column std of sig_model:\n", sig_model.std().round(4).to_dict())

print("\n=== (2) placebo turnover bias (model arm) ===")
r = evaluate(sig_model, tr, dv, yc, cfg)
m = r.metrics
print(f"  REAL   sharpe {m['sharpe']:+.4f}  gross_sharpe {m['sharpe_gross']:+.4f}  "
      f"turn {m['ann_turnover']:7.2f}  costdrag {m['cost_drag_annual']:.4%}  fin {m['financing_drag_annual']:+.4%}")
rows=[]
for i in range(40):
    pm = evaluate(block_sign_flip(sig_model,63,i), tr, dv, yc, cfg).metrics
    rows.append(dict(sharpe=pm["sharpe"], gross=pm["sharpe_gross"], turn=pm["ann_turnover"],
                     cost=pm["cost_drag_annual"], fin=pm["financing_drag_annual"], vol=pm["ann_vol"]))
p = pd.DataFrame(rows)
print(f"  PLACEBO mean sharpe {p.sharpe.mean():+.4f} sd {p.sharpe.std():.4f}")
print(f"          mean gross  {p.gross.mean():+.4f} sd {p.gross.std():.4f}")
print(f"          mean turn   {p.turn.mean():7.2f}   (real {m['ann_turnover']:.2f})")
print(f"          mean cost   {p.cost.mean():.4%}    (real {m['cost_drag_annual']:.4%})")
print(f"          mean fin    {p.fin.mean():+.4%}    (real {m['financing_drag_annual']:+.4%})")
print(f"          mean vol    {p.vol.mean():.4%}     (real {m['ann_vol']:.4%})")
beat = int((p.sharpe >= m["sharpe"]).sum())
print(f"  p_value = {(beat+1)/41:.4f}   beat={beat}")
beat_g = int((p.gross >= m["sharpe_gross"]).sum())
print(f"  p_value on GROSS sharpe = {(beat_g+1)/41:.4f}  beat={beat_g}")
p.to_csv(root/"artifacts/audit2/carry_placebos.csv", index=False)
