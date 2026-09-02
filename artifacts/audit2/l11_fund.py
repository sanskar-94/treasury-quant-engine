import pandas as pd, numpy as np, sys
sys.path.insert(0,"src")
from tqe.config import load_config
cfg = load_config("configs/default.yaml")
m = pd.read_parquet(cfg.processed_dir/"macro.parquet")
print(m.columns.tolist())
ff = m["fed_funds"].dropna()
print("fed_funds range", ff.min(), ff.max(), ff.loc["2019-01-02":"2019-01-10"].to_dict())
print("3 Mo curve 2019-01-02:", pd.read_parquet(cfg.processed_dir/"curve.parquet")["3 Mo"].loc["2019-01-02"])
pos = pd.read_parquet("artifacts/backtests/latest/positions.parquet")
fin = pd.read_parquet("artifacts/backtests/latest/financing.parquet")["financing"]
net = pos.sum(axis=1)
print("\navg net notional", net.mean(), "avg gross", pos.abs().sum(axis=1).mean())
print("financing sum", fin.sum(), " implied avg rate =", fin.sum()/ (net.mean()* (pos.index[-1]-pos.index[0]).days/360))
