import pandas as pd, numpy as np, sys
sys.path.insert(0,"src")
from tqe.config import load_config
cfg = load_config("configs/default.yaml")
tr = pd.read_parquet("artifacts/audit2/tr.parquet")
c = pd.read_parquet(cfg.processed_dir/"curve.parquet")
m = pd.read_parquet(cfg.processed_dir/"macro.parquet")
pos = pd.read_parquet("artifacts/backtests/latest/positions.parquet")
w = pos.index
print("window", w.min().date(), w.max().date(), len(w))
yrs = (w[-1]-w[0]).days/365.25
for t in ["3 Mo","6 Mo","1 Yr"]:
    cum = (1+tr[t].reindex(w).fillna(0)).prod()
    print(f"{t:6s} realised ann total return = {cum**(1/yrs)-1: .4%}   mean quoted yield = {c[t].reindex(w).mean(): .4%}")
ff = m['fed_funds'].reindex(w).ffill()/100.0
print(f"mean fed funds = {ff.mean():.4%}   +5bp = {ff.mean()+0.0005:.4%}")
# compounded financing rate ACT/360 over window
days = np.empty(len(w)); days[0]=1.0
days[1:]=np.diff(w.to_numpy().astype('datetime64[D]').astype(float))
fin_rate_total = (ff.to_numpy()*days/360.0).sum()
print(f"total financing accrual over window (simple) = {fin_rate_total:.4%}  -> ann {fin_rate_total/yrs:.4%}")
for t in ["3 Mo","6 Mo","1 Yr"]:
    cum = (1+tr[t].reindex(w).fillna(0)).prod()-1
    print(f"{t:6s} cum total return {cum:.4%}  minus financing {fin_rate_total:.4%} = excess {cum-fin_rate_total:+.4%} ({(cum-fin_rate_total)/yrs:+.4%}/yr)")
