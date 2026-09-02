import pandas as pd, numpy as np, glob
files = sorted(glob.glob("data/cache/treasury_yield_curve_*.parquet"))
raw = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
raw = raw[~raw.index.duplicated()]
print("raw shape", raw.shape, raw.index.min(), raw.index.max())
proc = pd.read_parquet("data/processed/curve.parquet")
print("proc shape", proc.shape)
print("index equal:", raw.index.equals(proc.index))
d = raw.index.difference(proc.index); print("in raw not proc:", len(d), list(d[:10]))
d2 = proc.index.difference(raw.index); print("in proc not raw:", len(d2), list(d2[:10]))
common = raw.index.intersection(proc.index)
cols = [c for c in proc.columns if c in raw.columns]
a = raw.loc[common, cols]; b = proc.loc[common, cols]
# raw might be in percent
print("raw sample:", raw[cols].iloc[0].to_dict())
diff = (a.values - b.values)
print("max abs diff (raw vs proc):", np.nanmax(np.abs(diff)))
diff2 = (a.values/100.0 - b.values)
print("max abs diff (raw/100 vs proc):", np.nanmax(np.abs(diff2)))
# NaN pattern changes: were NaNs filled?
na_raw = a.isna().sum().sum(); na_pro = b.isna().sum().sum()
print("NaNs raw:", na_raw, " proc:", na_pro)
filled = (a.isna() & b.notna()).sum()
print("cells NaN in raw but filled in proc:\n", filled)
blanked = (a.notna() & b.isna()).sum()
print("cells present in raw but NaN in proc:\n", blanked)
