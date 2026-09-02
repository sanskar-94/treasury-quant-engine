import sys; sys.path.insert(0,"src")
import pandas as pd, numpy as np
from tqe.config import load_config
from tqe.backtest.engine import _funding_from_curve
cfg = load_config()
base="artifacts/backtests/latest/"
pos=pd.read_parquet(base+"positions.parquet")
fin=pd.read_parquet(base+"financing.parquet")["financing"]
idx=pos.index
# reconstruct funding rate exactly as engine does
from tqe.data.universe import universe_panel, constant_maturity_total_return
curve = pd.read_parquet(cfg.processed_dir/"curve.parquet")
returns = constant_maturity_total_return(curve, cfg.data.core_tenors)
tr = universe_panel(returns, "total_return")
fr = _funding_from_curve(cfg, tr.index)
aligned = fr.reindex(idx).ffill()
print("funding rate nan:", aligned.isna().sum(), "mean %:", 100*aligned.mean())
rate = aligned.to_numpy(float)
days = np.empty(len(idx)); days[0]=1.0
days[1:] = np.diff(idx.to_numpy().astype("datetime64[D]").astype(float))
days = np.clip(days,0,10)
net = pos.sum(axis=1).to_numpy(float)
gross = pos.abs().sum(axis=1).to_numpy(float)
fin_engine = net*rate*days/360.0
print("engine financing recomputed sum:", fin_engine.sum(), " reported:", fin.sum())

s = cfg.costs.repo_spread_bp/1e4
print("repo spread decimal:", s)
gc = rate - s     # general collateral component
fin_correct = net*gc*days/360.0 + gross*s*days/360.0
print("\nCORRECT (costs.py CostModel.financing convention: repo*net + spread*gross):")
print("  total financing:", fin_correct.sum())
print("  engine total   :", fin_engine.sum())
undercharge = fin_correct.sum() - fin_engine.sum()
print("  UNDER-CHARGE   :", undercharge, "  = $%.0f/yr"%(undercharge/8))
print("  as %% of capital p.a.: %.4f%%"%(undercharge/1e7/8*100))
print("\ndays net-short (spread credited instead of charged):", int((net<0).sum()), "/", len(net))
print("spread credited on those days: $%.0f"%(-(net[net<0]*s*days[net<0]/360.0).sum()))
# impact on Sharpe
ret = pd.read_parquet(base+"returns.parquet")["returns"]
cap = 1e7
new_ret = ret - pd.Series((fin_correct-fin_engine)/cap, index=idx)
def sharpe(x): return x.mean()/x.std()*np.sqrt(252)
print("\nSharpe reported : %.4f"%sharpe(ret))
print("Sharpe corrected: %.4f"%sharpe(new_ret))
print("ann return reported : %.4f%%"%(ret.mean()*252*100))
print("ann return corrected: %.4f%%"%(new_ret.mean()*252*100))
print("total return reported : %.3f%%"%(((1+ret).prod()-1)*100))
print("total return corrected: %.3f%%"%(((1+new_ret).prod()-1)*100))
