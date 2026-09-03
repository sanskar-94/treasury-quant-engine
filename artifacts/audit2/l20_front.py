import pandas as pd, numpy as np, sys
sys.path.insert(0,"src")
from tqe.config import load_config
from tqe.data.universe import constant_maturity_total_return, universe_panel
cfg = load_config("configs/default.yaml")
c = pd.read_parquet(cfg.processed_dir/"curve.parquet")
rets = constant_maturity_total_return(c, cfg.data.core_tenors)
tr=universe_panel(rets,"total_return"); ca=universe_panel(rets,"carry_1d")
yc=universe_panel(rets,"yield_change"); dur=universe_panel(rets,"duration"); cvx=universe_panel(rets,"convexity")
w = pd.read_parquet("artifacts/backtests/latest/positions.parquet").index
print("closed-form residual check, backtest window, front end:")
for t in ["3 Mo","6 Mo","1 Yr"]:
    approx = ca[t]/100.0 - dur[t].shift(1)*yc[t] + 0.5*cvx[t].shift(1)*yc[t]**2
    r=(tr[t]-approx).reindex(w).dropna()
    print(f"  {t:6s} resid rms={r.std():.3e} max={r.abs().max():.3e} vs tr rms={tr[t].reindex(w).std():.3e}")
print("\nzero-yield-change fraction in backtest window:")
for t in ["3 Mo","6 Mo","1 Yr"]:
    print(f"  {t:6s} {(yc[t].reindex(w).abs()<1e-12).mean():.2%}")
print("\nzero total_return days in window (should be ~0 because carry never zero):")
for t in ["3 Mo","6 Mo","1 Yr"]:
    print(f"  {t:6s} {(tr[t].reindex(w).abs()<1e-15).mean():.3%}")
