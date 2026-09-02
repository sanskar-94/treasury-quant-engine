import sys; sys.path.insert(0,"src")
import pandas as pd, numpy as np, json
from tqe.config import load_config
from tqe.backtest.engine import _funding_from_curve
cfg = load_config()
base="artifacts/backtests/latest/"
pos=pd.read_parquet(base+"positions.parquet")
fin=pd.read_parquet(base+"financing.parquet")["financing"]
ret=pd.read_parquet(base+"returns.parquet")["returns"]
costs=pd.read_parquet(base+"costs.parquet")["costs"]
met=json.load(open(base+"metrics.json"))
idx=pos.index
print("positions columns:", list(pos.columns))
print("index", idx[0], idx[-1], len(idx))
print("capital cfg:", cfg.backtest.initial_capital, "repo_spread_bp:", cfg.costs.repo_spread_bp)

fr = _funding_from_curve(cfg, idx)
print("funding rate is None?", fr is None)
aligned = fr.reindex(idx).ffill()
print("nan:", int(aligned.isna().sum()), "mean %%: %.4f"%(100*aligned.mean()), "min %.4f max %.4f"%(100*aligned.min(),100*aligned.max()))
rate = aligned.to_numpy(float)
days=np.empty(len(idx)); days[0]=1.0
days[1:]=np.diff(idx.to_numpy().astype("datetime64[D]").astype(float))
days=np.clip(days,0,10)
net=pos.sum(axis=1).to_numpy(float)
grossleg=pos.abs().sum(axis=1).to_numpy(float)
fin_eng = net*rate*days/360.0
print("\nreported total_financing: %.6f"%met["total_financing"])
print("financing.parquet  sum  : %.6f"%fin.sum())
print("recomputed net*rate*d/360: %.6f"%fin_eng.sum())
print("max abs diff vs parquet  : %.3e"%np.abs(fin_eng-fin.to_numpy()).max())
print("\nnet notional: mean %.0f  median %.0f  |net| mean %.0f"%(net.mean(), np.median(net), np.abs(net).mean()))
print("gross(sum|leg|) mean %.0f  (metrics avg_gross_notional %.0f)"%(grossleg.mean(), met["avg_gross_notional"]))
print("days net<0: %d / %d"%(int((net<0).sum()), len(net)))
