import sys, json, math
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0,"src")
from tqe.config import load_config
from tqe.backtest.engine import _funding_from_curve
from tqe.data.universe import constant_maturity_total_return, universe_panel
from tqe.training.metrics import performance_metrics
cfg=load_config(Path("configs/default.yaml").resolve())
R=pd.read_parquet("artifacts/backtests/latest/returns.parquet")["returns"]; idx=R.index
M=json.load(open("artifacts/backtests/latest/metrics.json"))
curve=pd.read_parquet(cfg.processed_dir/"curve.parquet")
rets=constant_maturity_total_return(curve,[t for t in cfg.data.core_tenors if t in curve.columns])
tr=universe_panel(rets,"total_return")
b=tr[cfg.backtest.benchmark].reindex(idx).fillna(0.0)
fr=_funding_from_curve(cfg, idx).reindex(idx).ffill()
dt=np.empty(len(idx)); dt[0]=1.0; dt[1:]=np.diff(idx.to_numpy().astype('datetime64[D]').astype(float)); dt=np.clip(dt,0,10)
rf=pd.Series(fr.to_numpy()*dt/360.0, index=idx)
bt=performance_metrics(b); bx=performance_metrics(b-rf); st=performance_metrics(R+rf)
print("strategy returns are EXCESS (financing charged inside). Reported strategy sharpe:", M["sharpe"])
print("benchmark TOTAL-return  ratio (what the engine reports):", bt["sharpe"], " ann_ret", bt["ann_return"])
print("benchmark EXCESS Sharpe (consistent with the strategy) :", bx["sharpe"], " ann_ret", bx["ann_return"])
print("strategy TOTAL-return ratio (other consistent pairing) :", st["sharpe"], " ann_ret", st["ann_return"])
a1=R-b; a2=R-(b-rf)
print("reported information_ratio (excess vs total):", M["information_ratio"], "hand", a1.mean()/a1.std()*math.sqrt(252))
print("information_ratio on a consistent basis      :", a2.mean()/a2.std()*math.sqrt(252))
print("mean annual funding rate used:", float(fr.mean()))
