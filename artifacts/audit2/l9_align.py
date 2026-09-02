import pandas as pd, numpy as np, sys
sys.path.insert(0,"src")
from tqe.config import load_config
from tqe.data.universe import constant_maturity_total_return, universe_panel
cfg = load_config("configs/default.yaml")
preds = pd.read_parquet(cfg.processed_dir/"oos_predictions.parquet")
c = pd.read_parquet(cfg.processed_dir/"curve.parquet")
rets = constant_maturity_total_return(c, cfg.data.core_tenors)
tr = universe_panel(rets,"total_return").reindex(preds.index)
pr = universe_panel(rets,"price_return").reindex(preds.index)
print("model target:", cfg.model.target, "horizon", cfg.model.horizon)
for lag in (-2,-1,0,1,2):
    ics=[]
    for t in preds.columns:
        a = preds[t]; b = pr[t].shift(-lag)   # lag=0 -> same row
        m = a.notna()&b.notna()
        ics.append(np.corrcoef(a[m],b[m])[0,1])
    print(f" shift {lag:+d}: mean IC vs price_return = {np.mean(ics):+.4f}   per-tenor {[round(x,3) for x in ics]}")
