import sys, json, math
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0,"src")
from tqe.config import load_config
from tqe.backtest.engine import _funding_from_curve
from tqe.data.calendar import settlement_date
from tqe.training.metrics import performance_metrics

cfg=load_config(Path("configs/default.yaml").resolve())
cap=cfg.backtest.initial_capital
d="artifacts/backtests/latest/"
pos=pd.read_parquet(d+"positions.parquet"); idx=pos.index
R=pd.read_parquet(d+"returns.parquet")["returns"]
fin=pd.read_parquet(d+"financing.parquet")["financing"]
cst=pd.read_parquet(d+"costs.parquet")["costs"]
M=json.load(open(d+"metrics.json"))
gross = R + (cst+fin)/cap

fr=_funding_from_curve(cfg, idx).reindex(idx).ffill()
rate=fr.to_numpy(float)
net_not=pos.sum(axis=1).to_numpy(float)

# engine convention: trade-date calendar diff, clipped
d_trade=np.empty(len(idx)); d_trade[0]=1.0
d_trade[1:]=np.diff(idx.to_numpy().astype('datetime64[D]').astype(float))
d_trade=np.clip(d_trade,0,10)

# settlement convention (matches how carry_1d is accrued in the returns panel)
sd=[settlement_date(ts.date(), None) for ts in idx]
so=np.array([pd.Timestamp(s).to_datetime64().astype('datetime64[D]').astype(float) for s in sd])
d_settle=np.empty(len(idx)); d_settle[0]=1.0
d_settle[1:]=np.diff(so); d_settle=np.clip(d_settle,0,10)

print("sum trade-date days", d_trade.sum(), " sum settlement days", d_settle.sum())
dow=pd.Series(idx.dayofweek, index=idx)
cmp=pd.DataFrame({"dow":dow.values,"d_trade":d_trade,"d_settle":d_settle}, index=idx)
print(cmp.groupby("dow")[["d_trade","d_settle"]].mean().to_string())

fin_old = net_not*rate*d_trade/360.0
fin_new = net_not*rate*d_settle/360.0
print("\ntotal fin engine", fin_old.sum(), " (stored", fin.sum(), ")   settlement-aligned", fin_new.sum())

net_old = gross - (cst.to_numpy()+fin_old)/cap
net_new = gross - (cst.to_numpy()+fin_new)/cap
for lab, s in (("engine (trade-date days)", net_old), ("settlement-aligned days", net_new)):
    m=performance_metrics(pd.Series(s.values if hasattr(s,'values') else s, index=idx))
    print(f"{lab:28s} sharpe={m['sharpe']:.4f} ann_ret={m['ann_return']:.5f} ann_vol={m['ann_vol']:.5f} "
          f"sortino={m['sortino']:.4f} maxdd={m['max_drawdown']:.5f} calmar={m['calmar']:.4f}")

# day-of-week means of net returns under each
for lab, s in (("engine", net_old), ("aligned", net_new)):
    ss = pd.Series(np.asarray(s), index=idx)
    print(lab, "dow mean x1e5:", np.round(ss.groupby(idx.dayofweek).mean().to_numpy()*1e5,2))

# autocorrelation
def acf(x,k):
    x=np.asarray(x,float); x=x-x.mean(); return float((x[k:]@x[:-k])/(x@x))
print("engine  acf1..10", np.round([acf(net_old,k) for k in range(1,11)],3))
print("aligned acf1..10", np.round([acf(net_new,k) for k in range(1,11)],3))
