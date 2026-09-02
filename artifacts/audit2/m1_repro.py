import json, math
import numpy as np, pandas as pd, sys
sys.path.insert(0, "src")
from tqe.training.metrics import performance_metrics, drawdown_series, equity_curve

R = pd.read_parquet("artifacts/backtests/latest/returns.parquet")["returns"]
E = pd.read_parquet("artifacts/backtests/latest/equity.parquet")["equity"]
M = json.load(open("artifacts/backtests/latest/metrics.json"))

print("n returns", len(R), "n equity", len(E))
m = performance_metrics(R)
bad=[]
for k,v in m.items():
    if k in M:
        ok = (np.isnan(v) and np.isnan(M[k])) or abs(v-M[k])<1e-9*max(1,abs(v))
        if not ok: bad.append((k, v, M[k]))
print("mismatches vs stored metrics.json:", bad)

# --- independent hand computation ---
x = R.to_numpy(float)
n = len(x)
print("mean", x.mean(), "std(ddof1)", x.std(ddof=1))
print("hand sharpe", x.mean()/x.std(ddof=1)*math.sqrt(252), " stored", M["sharpe"])
eq = np.cumprod(1+x)
print("hand total_return", eq[-1]-1, "stored", M["total_return"])
print("hand cagr", eq[-1]**(252/n)-1, "stored", M["cagr"])
peak = np.maximum.accumulate(eq)
print("hand maxdd (no initial 1.0)", (eq/peak-1).min(), "stored", M["max_drawdown"])
eq2 = np.concatenate(([1.0], eq))
peak2 = np.maximum.accumulate(eq2)
print("hand maxdd (with initial 1.0)", (eq2/peak2-1).min())
# sum-of-returns drawdown for comparison
cs = np.cumsum(x)
print("cumsum-based dd", (cs - np.maximum.accumulate(cs)).min())

# equity parquet consistency
print("equity[0]", E.iloc[0], "cap*(1+r0)", 1e7*(1+x[0]))
dd_e = drawdown_series(E)
print("dd from equity.parquet min", dd_e.min())

# sortino hand
down = np.minimum(x,0.0)
dvol = math.sqrt((down@down)/n)
print("hand sortino", x.mean()/dvol*math.sqrt(252), "stored", M["sortino"])
print("downside_vol ann", dvol*math.sqrt(252), "stored", M["downside_vol"])

# hit rate
nz = x[x!=0]
print("hand hit", (nz>0).mean(), "stored", M["hit_rate"], "n_nonzero", nz.size, "n", n)
