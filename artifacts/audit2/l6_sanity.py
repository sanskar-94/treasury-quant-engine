import pandas as pd, numpy as np
c = pd.read_parquet("data/processed/curve.parquet")
core = ["3 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","30 Yr"]
print("min/max per tenor (%):")
print((c[core].min()*100).round(3).to_dict())
print((c[core].max()*100).round(3).to_dict())
print("\nnegative yields:", int((c[core]<0).sum().sum()))
d = c[core].diff().abs()*1e4
print("\nlargest 1d moves (bp) per tenor:")
print(d.max().round(1).to_dict())
for t in core:
    top = (c[t].diff()*1e4).abs().nlargest(3)
    print(t, [(str(i.date()), round(float(v),1)) for i,v in top.items()])
sp = (c["10 Yr"]-c["2 Yr"])*1e4
print("\n10y-2y spread bp: min",round(sp.min(),1), sp.idxmin().date(), " max", round(sp.max(),1), sp.idxmax().date())
sp2 = (c["30 Yr"]-c["3 Mo"])*1e4
print("30y-3m: min",round(sp2.min(),1), " max", round(sp2.max(),1))
# monotonic violations / inversion extremes are fine. check curve crossing weirdness
print("\nrows where any core yield is NaN:", int(c[core].isna().any(axis=1).sum()))
