import pandas as pd, numpy as np, sys
sys.path.insert(0,"src")
from tqe.data.calendar import is_business_day, trading_index
c = pd.read_parquet("data/processed/curve.parquet")
core = ["3 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","30 Yr"]
idx = c.index
# holiday rows
hol = [d for d in idx if not is_business_day(d.date())]
print("rows on non-business days:", len(hol), hol[:20])
# missing trading days
ti = trading_index(idx.min().date(), idx.max().date())
missing = ti.difference(idx)
print("trading days missing from curve:", len(missing), list(missing[:20]))
extra = idx.difference(ti)
print("curve dates not trading days:", len(extra), list(extra[:20]))

# exact duplicate consecutive rows across the core tenors
sub = c[core]
same = (sub.diff().abs() < 1e-12).all(axis=1) & sub.notna().all(axis=1)
print("\nexact-duplicate consecutive rows (all 9 core tenors identical):", int(same.sum()), f"{same.mean():.4%}")
print(sub.index[same][:30].tolist())
# per-tenor zero-change fraction
print("\nper-tenor zero yield-change fraction (consecutive equal):")
for t in core:
    s = c[t].dropna()
    d = s.diff().abs()
    print(f"  {t:6s} zero-change {(d<1e-12).mean():.4%}  n={len(s)}")
