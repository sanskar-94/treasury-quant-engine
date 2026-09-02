import sys; sys.path.insert(0,"src")
import pandas as pd, numpy as np
from tqe.training.metrics import deflated_sharpe_ratio, performance_metrics
base="artifacts/backtests/latest/"
ret=pd.read_parquet(base+"returns.parquet")["returns"]
import json; m=json.load(open(base+"metrics.json"))
# corrected returns from a5
pos=pd.read_parquet(base+"positions.parquet")
idx=pos.index
from tqe.config import load_config
from tqe.backtest.engine import _funding_from_curve
from tqe.data.universe import universe_panel, constant_maturity_total_return
cfg=load_config()
curve=pd.read_parquet(cfg.processed_dir/"curve.parquet")
tr=universe_panel(constant_maturity_total_return(curve,cfg.data.core_tenors),"total_return")
rate=_funding_from_curve(cfg,tr.index).reindex(idx).ffill().to_numpy(float)
days=np.empty(len(idx)); days[0]=1.0
days[1:]=np.diff(idx.to_numpy().astype("datetime64[D]").astype(float)); days=np.clip(days,0,10)
s=cfg.costs.repo_spread_bp/1e4
net=pos.sum(axis=1).to_numpy(float); gross=pos.abs().sum(axis=1).to_numpy(float)
extra=(gross-net)*s*days/360.0
new=ret - pd.Series(extra/1e7,index=idx)
mm=performance_metrics(new)
print("corrected metrics: sharpe %.4f  ann_ret %.4f%%  vol %.4f%%  maxdd %.2f%%"%(mm['sharpe'],mm['ann_return']*100,mm['ann_vol']*100,mm['max_drawdown']*100))
print("DSR reported : %.4f"%deflated_sharpe_ratio(m['sharpe'],219,2016,m['skew'],m['kurtosis']))
print("DSR corrected: %.4f"%deflated_sharpe_ratio(mm['sharpe'],219,2016,mm['skew'],mm['kurtosis']))
print("total extra financing $%.0f over 8y (vs reported transaction costs $%.0f)"%(extra.sum(), m['total_costs']))
print("financing drag: %.4f%% -> %.4f%%"%(m['financing_drag_annual']*100,(m['total_financing']+extra.sum())/1e7/8*100))
