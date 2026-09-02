import pandas as pd, numpy as np, sys
sys.path.insert(0,"src")
from tqe.config import load_config
from tqe.training.metrics import performance_metrics
cfg = load_config("configs/default.yaml")
m = pd.read_parquet(cfg.processed_dir/"macro.parquet")
pos = pd.read_parquet("artifacts/backtests/latest/positions.parquet")
rets = pd.read_parquet("artifacts/backtests/latest/returns.parquet")["returns"]
fin = pd.read_parquet("artifacts/backtests/latest/financing.parquet")["financing"]
idx = pos.index
ff = m["fed_funds"].dropna()
base = (ff.reindex(idx.union(ff.index)).ffill().reindex(idx)/100.0)
spread = cfg.costs.repo_spread_bp*1e-4
days = np.empty(len(idx)); days[0]=1.0
days[1:] = np.diff(idx.to_numpy().astype("datetime64[D]").astype(float)); days=np.clip(days,0,10)
net = pos.sum(axis=1).to_numpy()
r = base.to_numpy()
cur  = net*(r+spread)*days/360.0                 # engine
corr = (net*r + spread*np.abs(net))*days/360.0   # CostModel.financing convention
print("engine financing total :", cur.sum())
print("costmodel convention   :", corr.sum())
print("difference (understated cost):", corr.sum()-cur.sum())
print("days net<0:", int((net<0).sum()), "of", len(net))
new_r = rets + (cur-corr)/cfg.backtest.initial_capital
print("\nbaseline sharpe:", performance_metrics(rets)["sharpe"])
print("corrected sharpe:", performance_metrics(new_r)["sharpe"])
