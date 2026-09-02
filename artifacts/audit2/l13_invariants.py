import pandas as pd, numpy as np, sys
sys.path.insert(0,"src")
from tqe.config import load_config
from tqe.data.universe import constant_maturity_total_return, universe_panel
cfg = load_config("configs/default.yaml")
c = pd.read_parquet(cfg.processed_dir/"curve.parquet")
rets = constant_maturity_total_return(c, cfg.data.core_tenors)
tr = universe_panel(rets,"total_return"); ca = universe_panel(rets,"carry_1d")
yc = universe_panel(rets,"yield_change"); dv = universe_panel(rets,"dv01")
dur = universe_panel(rets,"duration"); cvx = universe_panel(rets,"convexity")
y = c[cfg.data.core_tenors]

print("INVARIANT 1: total_return ~ carry - Dmod*dy + 0.5*C*dy^2  (Dmod, C from t-1 bond)")
for t in ["2 Yr","10 Yr","30 Yr"]:
    approx = ca[t]/100.0 - dur[t].shift(1)*yc[t] + 0.5*cvx[t].shift(1)*yc[t]**2
    r = (tr[t]-approx).dropna()
    print(f"  {t:6s} resid rms={r.std():.3e}  max={r.abs().max():.3e}  vs tr rms={tr[t].std():.3e}")

print("\nINVARIANT 2: annual carry sum / 100 vs average yield (per calendar year, 10 Yr)")
for yr in [1995, 2005, 2015, 2022, 2024]:
    m = ca.index.year==yr
    print(f"  {yr}: sum(carry)/100={ca['10 Yr'][m].sum()/100:.5f}  mean y={y['10 Yr'][m].mean():.5f}  ratio={ca['10 Yr'][m].sum()/100/y['10 Yr'][m].mean():.4f}")

print("\nINVARIANT 3: round-trip. Find (t0,t1) with equal 10Y yield; cum total return vs avg-yield accrual")
s = y["10 Yr"]
tot = (1+tr["10 Yr"].fillna(0)).cumprod()
found=0
for i in range(0, len(s)-1500, 373):
    y0 = s.iloc[i]
    j_candidates = np.where(np.abs(s.values[i+1000:]-y0) < 1e-6)[0]
    if len(j_candidates)==0: continue
    j = i+1000+j_candidates[-1]
    dt = (s.index[j]-s.index[i]).days/365.25
    realised = tot.iloc[j]/tot.iloc[i]-1
    # accrual approximation: geometric mean of yields over the period
    avg_y = s.iloc[i:j+1].mean()
    print(f"  {s.index[i].date()} -> {s.index[j].date()}  y={y0*100:.2f}%  dt={dt:.2f}y  realised={realised*100:+.2f}%  avg-y accrual={( (1+avg_y/2)**(2*dt)-1)*100:+.2f}%")
    found+=1
    if found>=6: break
