import sys, numpy as np, pandas as pd, math
sys.path.insert(0,"src")
R=pd.read_parquet("artifacts/backtests/latest/returns.parquet")["returns"]
x=R.to_numpy(float); x=x-x.mean()
n=len(x)
def acf(x,k):
    return float((x[k:]@x[:-k])/(x@x))
print("acf lags 1..21:", np.round([acf(x,k) for k in range(1,22)],4))
print("se ~", 1/math.sqrt(n))

# is it the financing weekday pattern?
fin=pd.read_parquet("artifacts/backtests/latest/financing.parquet")["financing"]/1e7
cst=pd.read_parquet("artifacts/backtests/latest/costs.parquet")["costs"]/1e7
g=(R+fin+cst).to_numpy(float); g=g-g.mean()
print("gross acf 1..10:", np.round([acf(g,k) for k in range(1,11)],4))
f=fin.to_numpy(float); f=f-f.mean()
print("financing acf 1..10:", np.round([acf(f,k) for k in range(1,11)],4))
print("std net %.3e  std gross %.3e  std fin %.3e  std cost %.3e" % (R.std(), (R+fin+cst).std(), fin.std(), cst.std()))

# weekday effect
df=pd.DataFrame({"r":R})
df["dow"]=R.index.dayofweek
print(df.groupby("dow")["r"].agg(["mean","std","count"]))
