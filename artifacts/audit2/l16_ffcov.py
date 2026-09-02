import pandas as pd, numpy as np, sys
sys.path.insert(0,"src")
from tqe.config import load_config
cfg = load_config("configs/default.yaml")
m = pd.read_parquet(cfg.processed_dir/"macro.parquet")
ff = m["fed_funds"].dropna()
print("fed_funds:", ff.index.min(), "->", ff.index.max(), "n=",len(ff))
c = pd.read_parquet(cfg.processed_dir/"curve.parquet")
print("curve last:", c.index.max())
print("last 10 ff:", ff.tail(10).to_dict())
# is the ff series daily calendar?
idx = c.index
missing = idx.difference(ff.index)
print("curve dates with no ff obs:", len(missing), list(missing[-10:]))
# compare 3Mo bill yield vs fed funds recent
comp = pd.DataFrame({"ff":ff/100.0, "b3":c["3 Mo"]}).dropna()
print("\nmean(3Mo - ff) by year (bp):")
print(((comp["b3"]-comp["ff"])*1e4).groupby(comp.index.year).mean().round(1).tail(12).to_dict())
