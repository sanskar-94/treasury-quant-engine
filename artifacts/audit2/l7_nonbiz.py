import pandas as pd, numpy as np, sys
sys.path.insert(0,"src")
from tqe.data.calendar import is_business_day
c = pd.read_parquet("data/processed/curve.parquet")
core = ["3 Mo","2 Yr","10 Yr","30 Yr"]
nb = [d for d in c.index if not is_business_day(d.date())]
d = c[core].diff()*1e4
for t in nb:
    i = c.index.get_loc(t)
    print(t.date(), t.day_name(), "  move bp:", d.loc[t].round(1).to_dict())
