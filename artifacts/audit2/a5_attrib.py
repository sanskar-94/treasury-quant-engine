import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
root = Path("/Users/sanskarawasthi/trade bot 4"); sys.path.insert(0, str(root/"src"))
from tqe.config import load_config
from tqe.data.universe import constant_maturity_total_return, universe_panel
from tqe.backtest.engine import _funding_from_curve
from tqe.logging_utils import setup_logging
setup_logging("ERROR")
cfg = load_config(root/"configs"/"default.yaml")
curve = pd.read_parquet(root/"data/processed/curve.parquet")
tenors = [t for t in cfg.data.core_tenors if t in curve.columns]
rets = constant_maturity_total_return(curve, tenors)
tr = universe_panel(rets,"total_return")
pos = pd.read_parquet(root/"artifacts/audit2/repro_positions.parquet")
idx = pos.index
r = tr.reindex(index=idx, columns=pos.columns).fillna(0.0)
gross = pos*r
yrs = len(idx)/252.0
print("gross P&L per tenor ($/yr):")
print((gross.sum()/yrs).round(0).to_string())
print("total gross $/yr", round(gross.sum().sum()/yrs))
fr = _funding_from_curve(cfg, tr.index).reindex(idx).ffill()
days = np.empty(len(idx)); days[0]=1.0
days[1:] = np.diff(idx.to_numpy().astype("datetime64[D]").astype(float))
days = np.clip(days,0,10)
fin = pos.sum(axis=1).to_numpy()*fr.to_numpy()*days/360.0
print("financing $/yr", round(fin.sum()/yrs))
print("funding rate mean %.4f%%" % (100*fr.mean()))
# per-tenor financing share (allocate by signed notional)
share = pos.div(pos.sum(axis=1).replace(0,np.nan), axis=0)
fin_s = share.mul(pd.Series(fin, index=idx), axis=0)
print("\nfinancing per tenor ($/yr):")
print((fin_s.sum()/yrs).round(0).to_string())
print("\nnet (gross - financing) per tenor ($/yr):")
print(((gross.sum()-fin_s.sum())/yrs).round(0).to_string())
# 3Mo carry check: CMT vs funding
y3 = curve["3 Mo"].reindex(idx)
print("\nmean 3Mo CMT %.4f%%  mean funding %.4f%%  spread %.1f bp" %
      (100*y3.mean(), 100*fr.mean(), 1e4*(y3.mean()-fr.mean())))
# unit test: hold $1mm of each tenor, funded, sharpe
print("\nunit funded carry per tenor (hold $10mm, funded):")
for t in pos.columns:
    p = pd.Series(10e6, index=idx)
    g = p*r[t]
    f = p*fr*days/360.0
    n = (g-f)/1e7
    print(f"  {t:6s} ann_ret {n.mean()*252:+.4%}  vol {n.std()*np.sqrt(252):.4%}  sharpe {n.mean()/n.std()*np.sqrt(252):+.3f}")
