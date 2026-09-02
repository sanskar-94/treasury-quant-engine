import pandas as pd, numpy as np, sys
sys.path.insert(0,"src")
from tqe.data.universe import constant_maturity_total_return, universe_panel
c = pd.read_parquet("data/processed/curve.parquet")
core = ["3 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","30 Yr"]
rets = constant_maturity_total_return(c, core)
tr = universe_panel(rets,"total_return")
dv = universe_panel(rets,"dv01")
yc = universe_panel(rets,"yield_change")
pr = universe_panel(rets,"price_return")
ca = universe_panel(rets,"carry_1d")
print("tr shape", tr.shape)
print("\nann vol of total_return (full sample):")
print((tr.std()*np.sqrt(252)).round(4))
print("\nann mean:")
print((tr.mean()*252).round(4))
print("\nsign check: corr(yield_change, price_return) per tenor")
for t in core:
    m = yc[t].notna()&pr[t].notna()
    print(f"  {t:6s} corr={np.corrcoef(yc[t][m],pr[t][m])[0,1]: .4f}  beta(pr~dy) per bp={np.polyfit(yc[t][m],pr[t][m],1)[0]*1e-4: .6f}  dv01/100={dv[t].mean()/100:.6f}")
print("\nDV01 means:", dv.mean().round(4).to_dict())
tr.to_parquet("artifacts/audit2/tr.parquet"); dv.to_parquet("artifacts/audit2/dv.parquet")
yc.to_parquet("artifacts/audit2/yc.parquet")
