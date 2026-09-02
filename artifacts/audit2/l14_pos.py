import pandas as pd, numpy as np, sys
sys.path.insert(0,"src")
pos = pd.read_parquet("artifacts/backtests/latest/positions.parquet")
tr_ = pd.read_parquet("artifacts/audit2/tr.parquet").reindex(pos.index)
print("mean |notional| per tenor ($m):")
print((pos.abs().mean()/1e6).round(2).to_dict())
print("share of gross notional:")
print((pos.abs().mean()/pos.abs().mean().sum()).round(4).to_dict())
trades = pos.diff(); trades.iloc[0]=pos.iloc[0]
print("\ntraded notional per tenor ($bn over sample):")
print((trades.abs().sum()/1e9).round(3).to_dict())
print("share of traded notional:", (trades.abs().sum()/trades.abs().sum().sum()).round(4).to_dict())
pnl = (pos*tr_.fillna(0))
print("\ntotal gross P&L by tenor ($):")
print(pnl.sum().round(0).to_dict())
print("total gross pnl", pnl.sum().sum())
# yearly
print("\ngross P&L by year:")
print(pnl.sum(axis=1).groupby(pnl.index.year).sum().round(0).to_dict())
