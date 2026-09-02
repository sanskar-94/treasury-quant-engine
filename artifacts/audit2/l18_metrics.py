import pandas as pd, numpy as np, sys
sys.path.insert(0,"src")
r = pd.read_parquet("artifacts/backtests/latest/returns.parquet")["returns"]
x = r.to_numpy()
print("n", len(x))
print("mean*252", x.mean()*252)
print("std(ddof=1)*sqrt(252)", x.std(ddof=1)*np.sqrt(252))
print("sharpe", x.mean()/x.std(ddof=1)*np.sqrt(252))
eq = (1+r).cumprod()
print("total return", eq.iloc[-1]-1)
print("cagr", eq.iloc[-1]**(252/len(x))-1)
dd = eq/eq.cummax()-1
print("max dd", dd.min())
print("skew", pd.Series(x).skew(), "kurt", pd.Series(x).kurt()+3)
print("hit rate", (x>0).mean())
print("hit rate excl zeros", (x[x!=0]>0).mean(), " zeros:", int((x==0).sum()))
