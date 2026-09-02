import pandas as pd, numpy as np, sys, json
sys.path.insert(0,"src")
from tqe.config import Config
from tqe.backtest.costs import CostModel
from tqe.data.universe import bucket_for_years
from tqe.data.sources import TENOR_YEARS
from tqe.config import load_config
cfg = load_config()
cm = CostModel(cfg.costs)
base="artifacts/backtests/latest/"
pos=pd.read_parquet(base+"positions.parquet"); tr=pd.read_parquet(base+"trades.parquet")
cost=pd.read_parquet(base+"costs.parquet")["costs"]
buckets={t:bucket_for_years(TENOR_YEARS.get(t,10.0)) for t in pos.columns}
print("buckets:",buckets)
print("\nper-tenor traded notional and expected costs")
tot=0.0
rows=[]
for t in pos.columns:
    traded=tr[t].abs()
    b=buckets[t]
    c=np.array([float(cm.total_cost(float(v),b)) for v in traded])
    rows.append((t,b,traded.sum(),c.sum(), c.sum()/max(traded.sum(),1)*1e4))
    tot+=c.sum()
df=pd.DataFrame(rows,columns=["tenor","bucket","traded_$","cost_$","bp"])
print(df.to_string())
print("\nrecomputed total cost:", tot, " reported:", cost.sum(), " diff:", tot-cost.sum())
# component split
sp=0.0;im=0.0;co=0.0
for t in pos.columns:
    traded=tr[t].abs().to_numpy(); b=buckets[t]
    sp+=float(np.sum(cm.spread_cost(traded,b))); im+=float(np.sum(cm.impact_cost(traded,b))); co+=float(np.sum(cm.commission(traded)))
print(f"\nspread ${sp:,.0f}  impact ${im:,.0f}  commission ${co:,.0f}  total ${sp+im+co:,.0f}")
print("gross notional share by tenor (mean):")
print((pos.abs().mean()/pos.abs().mean().sum()).to_string())
print("\ntraded share by tenor:")
print((tr.abs().sum()/tr.abs().sum().sum()).to_string())
