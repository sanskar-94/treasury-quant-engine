import pandas as pd, numpy as np, sys
sys.path.insert(0,"src")
from tqe.config import load_config
from tqe.data.universe import constant_maturity_total_return, universe_panel
cfg = load_config("configs/default.yaml")
preds = pd.read_parquet(cfg.processed_dir/"oos_predictions.parquet")
c = pd.read_parquet(cfg.processed_dir/"curve.parquet")
lo, hi = preds.index.min(), preds.index.max()
cw = c.loc[lo:hi]
print("curve rows in window:", len(cw), " preds rows:", len(preds))
missing = cw.index.difference(preds.index)
print("curve dates missing from preds:", len(missing), list(missing[:20]))
rets = constant_maturity_total_return(c, cfg.data.core_tenors)
tr = universe_panel(rets,"total_return")
sub = tr.reindex(preds.index)
print("\nNaN counts in returns panel over preds window:")
print(sub.isna().sum().to_dict())
# gap in days
g = pd.Series(np.diff(preds.index.to_numpy().astype('datetime64[D]').astype(float)), index=preds.index[1:])
print("\ncalendar-day gaps distribution:", g.value_counts().sort_index().to_dict())
