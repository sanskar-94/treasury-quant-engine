import sys, numpy as np, pandas as pd
sys.path.insert(0,"src")
from pathlib import Path
from tqe.config import load_config
from tqe.data.universe import constant_maturity_total_return, universe_panel
cfg=load_config(Path("configs/default.yaml").resolve())
curve=pd.read_parquet(cfg.processed_dir/"curve.parquet")
tenors=[t for t in cfg.data.core_tenors if t in curve.columns]
rets=constant_maturity_total_return(curve, tenors)
print("tenors", tenors)
f=rets["10 Yr"]
f=f.loc["2018-08-06":"2026-08-28"]
df=pd.DataFrame({"carry":f["carry_1d"], "pr":f["price_return"], "tr":f["total_return"]})
df["dow"]=df.index.dayofweek
print("10Y carry_1d mean by dow (per 100 face):")
print(df.groupby("dow")["carry"].agg(["mean","count"]).to_string())
print()
f2=rets["3 Mo"].loc["2018-08-06":"2026-08-28"]
d2=pd.DataFrame({"carry":f2["carry_1d"]}); d2["dow"]=d2.index.dayofweek
print("3Mo carry_1d mean by dow:"); print(d2.groupby("dow")["carry"].agg(["mean","count"]).to_string())
print()
# split by settlement regime
for lab, sl in (("pre-2024 T+2","2018-08-06:2024-05-24"),("post T+1","2024-05-29:2026-08-28")):
    a,b=sl.split(":")
    g=df.loc[a:b]
    print(lab, "10Y carry by dow:"); print(g.groupby("dow")["carry"].mean().to_string())
