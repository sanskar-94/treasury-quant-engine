import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0,"src")
from tqe.config import load_config
from tqe.backtest.engine import _funding_from_curve
from tqe.data.universe import constant_maturity_total_return
cfg=load_config(Path("configs/default.yaml").resolve())
d="artifacts/backtests/latest/"
pos=pd.read_parquet(d+"positions.parquet"); idx=pos.index
fin=pd.read_parquet(d+"financing.parquet")["financing"]
net=pos.sum(axis=1); dow=idx.dayofweek
fr=_funding_from_curve(cfg, idx).reindex(idx).ffill()
dd=np.empty(len(idx)); dd[0]=1.0; dd[1:]=np.diff(idx.to_numpy().astype('datetime64[D]').astype(float)); dd=np.clip(dd,0,10)
t=pd.DataFrame({"net_notional":net.values,"rate":fr.values,"days":dd,"fin":fin.values},index=idx)
print(t.groupby(dow).mean().round(6).to_string())
print("\nimplied: net*rate*days/360 by dow:")
print((t.groupby(dow).apply(lambda g:(g.net_notional*g.rate*g.days/360).mean())).round(2).to_string())
curve=pd.read_parquet(cfg.processed_dir/"curve.parquet")
tn=[x for x in cfg.data.core_tenors if x in curve.columns]
rets=constant_maturity_total_return(curve,tn)
c=pd.DataFrame({x: rets[x]["carry_1d"] for x in tn}).reindex(idx)
print("\nmean carry_1d per 100 face by dow, per tenor (pre-2024 sub-sample):")
sub=c.loc[:"2024-05-24"]; print(sub.groupby(sub.index.dayofweek).mean().round(4).to_string())
print("\npositions net notional by dow, pre-2024 vs post:")
for a,b in ((None,"2024-05-24"),("2024-05-29",None)):
    s=net.loc[a:b]; print(a,b, s.groupby(s.index.dayofweek).mean().round(0).to_dict())
