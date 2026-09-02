import pandas as pd, numpy as np, sys
sys.path.insert(0,"src")
from tqe.config import load_config
cfg = load_config("configs/default.yaml")
m = pd.read_parquet(cfg.processed_dir/"macro.parquet")
pos = pd.read_parquet("artifacts/backtests/latest/positions.parquet")
fin = pd.read_parquet("artifacts/backtests/latest/financing.parquet")["financing"]
idx = pos.index
ff = m["fed_funds"].dropna()
rate = ff.reindex(idx.union(ff.index)).ffill().reindex(idx)/100.0 + cfg.costs.repo_spread_bp*1e-4
print("rate coverage", rate.notna().mean(), "mean rate", rate.mean())
days = np.empty(len(idx)); days[0]=1.0
days[1:] = np.diff(idx.to_numpy().astype("datetime64[D]").astype(float))
days = np.clip(days,0,10)
net = pos.sum(axis=1).to_numpy()
mine = net*rate.to_numpy()*days/360.0
print("recomputed financing sum", mine.sum(), " stored", fin.sum(), " diff", mine.sum()-fin.sum())
print("total days", days.sum(), " span", (idx[-1]-idx[0]).days+1)
print("\nnet notional stats: mean", net.mean(), "median", np.median(net), "min", net.min(), "max", net.max())
print("frac days net<0:", (net<0).mean())
# dollar-weighted rate
print("dollar-weighted rate:", (net*rate.to_numpy()*days).sum()/ (net*days).sum())
print("simple mean rate:", rate.mean())
