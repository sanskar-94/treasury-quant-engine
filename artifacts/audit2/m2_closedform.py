import math, sys
import numpy as np, pandas as pd
sys.path.insert(0,"src")
from tqe.training.metrics import (performance_metrics, probabilistic_sharpe_ratio,
    deflated_sharpe_ratio, expected_maximum_sharpe, minimum_track_record_length,
    drawdown_series, equity_curve)
from scipy import stats

print("=== T1 constant positive return: everything analytic ===")
c = 0.001
r = pd.Series([c]*252)
m = performance_metrics(r)
print("  total_return", m["total_return"], "expected", (1+c)**252-1)
print("  cagr", m["cagr"], "expected", (1+c)**252-1)
print("  ann_return", m["ann_return"], "expected", c*252)
print("  vol", m["ann_vol"], "sharpe", m["sharpe"], "sortino", m["sortino"])
print("  maxdd", m["max_drawdown"], "calmar", m["calmar"], "hit", m["hit_rate"], "pf", m["profit_factor"])

print("=== T2 deterministic sawtooth: known max drawdown & duration ===")
# up 10 days of +1%, then 5 days of -2%, then flat
x = [0.01]*10 + [-0.02]*5 + [0.0]*3 + [0.01]*10
r = pd.Series(x)
eq = np.cumprod([1+v for v in x])
peak = np.maximum.accumulate(eq)
print("  true maxdd", (eq/peak-1).min(), "  (1.02^0 ... ) analytic:", 0.98**5-1)
m = performance_metrics(r)
print("  reported maxdd", m["max_drawdown"], "duration", m["max_dd_duration_days"])
# underwater from day 10 (idx10) until eq exceeds peak again
under = eq < peak - 1e-15
print("  true underwater run lengths:", np.diff(np.flatnonzero(np.diff(np.concatenate(([0],under.astype(int),[0]))))) [::2] if under.any() else 0)
print("  underwater mask:", under.astype(int))

print("=== T3 Sortino closed form ===")
# returns: +2% on 3 days, -1% on 1 day, repeated
x = np.array([0.02,0.02,0.02,-0.01]*63)
r = pd.Series(x)
m = performance_metrics(r)
n=len(x)
dd = math.sqrt((0.01**2*63)/n)
print("  hand downside dev(full n)", dd, " ann", dd*math.sqrt(252))
print("  reported downside_vol", m["downside_vol"])
print("  hand sortino", x.mean()/dd*math.sqrt(252), " reported", m["sortino"])
dd_only = math.sqrt((0.01**2*63)/63)
print("  (if it divided by downside count only) sortino would be", x.mean()/dd_only*math.sqrt(252))

print("=== T4 rf handling ===")
rng=np.random.default_rng(0)
x=pd.Series(rng.normal(0.0005,0.005,5000))
m0=performance_metrics(x, rf=0.0); m5=performance_metrics(x, rf=0.05)
print("  sharpe rf=0", m0["sharpe"], " rf=5%", m5["sharpe"])
rfp=(1.05)**(1/252)-1
print("  hand rf=5% sharpe", (x-rfp).mean()/ (x-rfp).std(ddof=1)*math.sqrt(252))
print("  ann_vol unchanged?", m0["ann_vol"], m5["ann_vol"])
print("  ann_return unchanged (NOT excess)?", m0["ann_return"], m5["ann_return"])

print("=== T5 PSR vs Bailey closed form ===")
# known: SR_hat per-period, n, skew 0, kurt 3 -> z = SR*sqrt(n-1)/sqrt(1-0+0.5*SR^2)
for srann,n,sk,ku in [(1.0,1000,0.0,3.0),(1.5,252,-1.0,8.0),(0.5,2016,0.19,9.515)]:
    sr = srann/math.sqrt(252)
    var = 1 - sk*sr + (ku-1)/4*sr*sr
    z = sr*math.sqrt(n-1)/math.sqrt(var)
    print(f"  ann={srann} n={n} skew={sk} kurt={ku}: hand={stats.norm.cdf(z):.10f} code={probabilistic_sharpe_ratio(srann,0.0,n,sk,ku):.10f}")

print("=== T6 minimum_track_record_length round trip ===")
for srann in (0.8,1.2):
    L = minimum_track_record_length(srann, 0.0, -0.5, 6.0, 0.95)
    print("  MinTRL", srann, L, "-> PSR at that n:", probabilistic_sharpe_ratio(srann,0.0,int(math.ceil(L)),-0.5,6.0))

print("=== T7 expected_maximum_sharpe sanity vs simulation ===")
rng=np.random.default_rng(1)
for N in (10,100,1000):
    sim = np.mean([rng.standard_normal(N).max() for _ in range(20000)])
    print(f"  N={N} formula={expected_maximum_sharpe(N,1.0):.4f} simulated={sim:.4f}")
