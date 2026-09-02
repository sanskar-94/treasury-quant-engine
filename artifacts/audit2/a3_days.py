import pandas as pd, numpy as np
base="artifacts/backtests/latest/"
pos=pd.read_parquet(base+"positions.parquet"); tr=pd.read_parquet(base+"trades.parquet")
nz=(tr.abs().sum(axis=1)>1e-9)
print("days with a trade:", nz.sum(), "of", len(tr))
d=tr.index[nz]
print("first 15 trade dates:", [str(x.date()) for x in d[:15]])
# months
print("unique year-months in trade dates:", len(pd.Series(d).dt.to_period('M').unique()))
print("trade dates per month counts:\n", pd.Series(d).dt.to_period('M').value_counts().value_counts())
gross = pos.abs().sum(axis=1)
print("\ngross notional stats:", gross.describe())
print("leverage max:", gross.max()/1e7)
