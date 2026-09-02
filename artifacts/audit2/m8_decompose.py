import sys, json, math
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0,"src")
from tqe.config import load_config
from tqe.backtest.engine import _funding_from_curve
from tqe.data.calendar import settlement_date
from tqe.data.universe import constant_maturity_total_return, universe_panel
from tqe.training.metrics import performance_metrics, deflated_sharpe_ratio

cfg=load_config(Path("configs/default.yaml").resolve()); cap=cfg.backtest.initial_capital
d="artifacts/backtests/latest/"
pos=pd.read_parquet(d+"positions.parquet"); idx=pos.index
R=pd.read_parquet(d+"returns.parquet")["returns"]
fin=pd.read_parquet(d+"financing.parquet")["financing"]; cst=pd.read_parquet(d+"costs.parquet")["costs"]
curve=pd.read_parquet(cfg.processed_dir/"curve.parquet")
tenors=[t for t in cfg.data.core_tenors if t in curve.columns]
rets=constant_maturity_total_return(curve,tenors)
cols=list(pos.columns)
carry=pd.DataFrame({t: rets[t]["carry_1d"]/100.0 for t in cols}).reindex(idx).fillna(0.0)
pr   =pd.DataFrame({t: rets[t]["price_return"] for t in cols}).reindex(idx).fillna(0.0)

carry_pnl=(pos*carry).sum(axis=1)/cap
price_pnl=(pos*pr).sum(axis=1)/cap
fin_r=fin/cap
dow=idx.dayofweek
print("day-of-week means x1e5  (n per dow:", np.bincount(dow),")")
tab=pd.DataFrame({"price_pnl":price_pnl,"carry_pnl":carry_pnl,"financing":fin_r,
                  "carry-fin":carry_pnl-fin_r,"net":R}).groupby(dow).mean()*1e5
print(tab.round(2).to_string())
print("\nstd by dow x1e5")
print((pd.DataFrame({"price":price_pnl,"carry-fin":carry_pnl-fin_r,"net":R}).groupby(dow).std()*1e5).round(2).to_string())

# settlement-aligned financing
sd=[settlement_date(ts.date(),None) for ts in idx]
so=np.array([pd.Timestamp(s).to_datetime64().astype('datetime64[D]').astype(float) for s in sd])
ds=np.empty(len(idx)); ds[0]=1.0; ds[1:]=np.diff(so); ds=np.clip(ds,0,10)
print("\nn days where settlement gap == 0:", int((ds==0).sum()))
fr=_funding_from_curve(cfg, idx).reindex(idx).ffill().to_numpy(float)
fin_new=pd.Series(pos.sum(axis=1).to_numpy(float)*fr*ds/360.0, index=idx)
gross=R+(cst+fin)/cap
net_new=gross-(cst+fin_new)/cap

M=json.load(open(d+"metrics.json"))
for lab, s in (("REPORTED (trade-date financing)", R), ("settlement-aligned financing", net_new)):
    m=performance_metrics(s)
    dsr=deflated_sharpe_ratio(m["sharpe"],219,len(s),m["skew"],m["kurtosis"])
    dsr2=deflated_sharpe_ratio(m["sharpe"],219,len(s),m["skew"],m["kurtosis"],sharpe_std=M["trial_sharpe_std"])
    print(f"\n{lab}")
    for k in ("total_return","cagr","ann_return","ann_vol","downside_vol","sharpe","sortino",
              "calmar","max_drawdown","max_dd_duration_days","hit_rate","profit_factor","var_95","cvar_95","skew","kurtosis"):
        print(f"   {k:22s} {m[k]:.6f}")
    print(f"   {'deflated_sharpe':22s} {dsr:.6f}   (observed-sd {dsr2:.3e})")
    print(f"   {'total financing':22s} {(fin if lab.startswith('REPORTED') else fin_new).sum():,.0f}")
