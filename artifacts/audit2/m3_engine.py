import sys, json, math
import numpy as np, pandas as pd
sys.path.insert(0,"src")

d="artifacts/backtests/latest/"
pos=pd.read_parquet(d+"positions.parquet")
fin=pd.read_parquet(d+"financing.parquet")["financing"]
cst=pd.read_parquet(d+"costs.parquet")["costs"]
ret=pd.read_parquet(d+"returns.parquet")["returns"]
trd=pd.read_parquet(d+"trades.parquet")
exp=pd.read_parquet(d+"exposures.parquet")
M=json.load(open(d+"metrics.json"))
cap=1e7
idx=pos.index
print("index gaps (calendar days) value_counts:")
gaps=pd.Series(np.diff(idx.to_numpy().astype('datetime64[D]').astype(float)))
print(gaps.value_counts().sort_index().to_string())
print("max gap", gaps.max())

net_not = pos.sum(axis=1)
print("avg net notional", net_not.mean(), " avg gross", pos.abs().sum(axis=1).mean(), "stored gross", M["avg_gross_notional"])
print("pct net long", (net_not>0).mean())

# implied funding rate from financing / (net_notional*days/360)
days=np.empty(len(idx)); days[0]=1.0; days[1:]=np.diff(idx.to_numpy().astype('datetime64[D]').astype(float)); days=np.clip(days,0,10)
den = net_not.to_numpy()*days/360.0
mask = np.abs(den)>1e-9
rate = fin.to_numpy()[mask]/den[mask]
print("implied funding rate: min %.4f med %.4f max %.4f" % (np.nanmin(rate), np.nanmedian(rate), np.nanmax(rate)))
print("total financing", fin.sum(), "stored", M["total_financing"])
print("total costs", cst.sum(), "stored", M["total_costs"])

# turnover
dt = trd.abs().sum(axis=1)/cap
print("ann_turnover hand", dt.mean()*252, "stored", M["ann_turnover"])
print("trades row0 == pos row0 ?", np.allclose(trd.iloc[0].to_numpy(), pos.iloc[0].to_numpy()))

# reconcile gross = net + (costs+fin)/cap
gross = ret + (cst+fin)/cap
gm_mean = gross.mean()*252
print("gross ann_return hand", gm_mean, "stored", M["ann_return_gross"])
print("gross sharpe hand", gross.mean()/gross.std(ddof=1)*math.sqrt(252), "stored", M["sharpe_gross"])
print("cost_drag", cst.sum()/cap/(len(ret)/252), "stored", M["cost_drag_annual"])
print("fin_drag", fin.sum()/cap/(len(ret)/252), "stored", M["financing_drag_annual"])
print()
print("ARITHMETIC CHECK on tearsheet: gross - costdrag - findrag = ", M["ann_return_gross"]-M["cost_drag_annual"]-M["financing_drag_annual"], " vs net ann_return", M["ann_return"])
print("Tearsheet-only (gross - costdrag) =", M["ann_return_gross"]-M["cost_drag_annual"])
