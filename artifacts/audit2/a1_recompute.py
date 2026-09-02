import pandas as pd, numpy as np, json, sys
sys.path.insert(0, "src")
base = "artifacts/backtests/latest/"
pos = pd.read_parquet(base+"positions.parquet")
tr  = pd.read_parquet(base+"trades.parquet")
cost= pd.read_parquet(base+"costs.parquet")["costs"]
m = json.load(open(base+"metrics.json"))
print("positions shape", pos.shape, "cols", list(pos.columns))
print("index", pos.index.min(), pos.index.max())
print("\nfirst 6 rows gross notional:")
gn = pos.abs().sum(axis=1)
print(gn.head(10))
print("first nonzero day:", gn[gn>1e-9].index[0])
i0 = gn.index.get_loc(gn[gn>1e-9].index[0])
print("index pos of first nonzero:", i0)
print("\ntrades on that day (abs sum):", tr.abs().sum(axis=1).iloc[i0])
print("positions on that day (abs sum):", gn.iloc[i0])
print("cost on that day:", cost.iloc[i0])
print("\ncost head:", cost.head(85).sum(), "sum of first 85")
# recompute trades from positions
tr2 = pos.diff()
tr2.iloc[0] = pos.iloc[0]
print("\nmax |trades_reported - diff(pos)|:", (tr - tr2).abs().max().max())
# turnover
dt = tr.abs().sum(axis=1)/1e7
print("ann_turnover recomputed (mean*252):", dt.mean()*252, " reported:", m["ann_turnover"])
tot_traded = tr.abs().sum().sum()
print("total notional traded $:", tot_traded)
years = len(pos)/252
print("years:", years)
print("turnover check: total_traded/(capital*years) =", tot_traded/(1e7*years))
print("total costs reported:", m["total_costs"], " costs.parquet sum:", cost.sum())
print("implied cost per $ traded (bp):", m["total_costs"]/tot_traded*1e4)
