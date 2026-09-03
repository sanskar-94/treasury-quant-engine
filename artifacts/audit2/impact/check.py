import numpy as np, pandas as pd
from tqe.config import load_config
from tqe.backtest.costs import CostModel
from tqe.backtest.engine import _funding_from_curve

cfg = load_config(None); cm = CostModel(cfg.costs)
s = float(cfg.costs.repo_spread_bp)/1e4

# ---- 1. closed form: perfectly cash-neutral book, one year, GC=4% ----
gc, days = 0.04, 365.0
long_, short_ = 50e6, -50e6
eng = (long_+short_) * (gc+s) * days/360.0          # engine algebra: net*(GC+s)
cost = cm.financing(long_, days, gc) + cm.financing(short_, days, gc)
cf = (abs(long_)+abs(short_)) * s * days/360.0
print("CASH-NEUTRAL $100mm gross, 1yr, GC=4.00%, spread=5bp")
print("  engine  _core_loop algebra : $%,.2f".replace(",","") % eng if False else "  engine  _core_loop algebra : $%.2f" % eng)
print("  CostModel.financing        : $%.2f" % cost)
print("  closed form gross*s*365/360: $%.2f" % cf)

# ---- 2. headline book: net-short days ----
pos = pd.read_parquet("artifacts/backtests/latest/positions.parquet")
fin = pd.read_parquet("artifacts/backtests/latest/financing.parquet")["financing"]
rate = _funding_from_curve(cfg, pos.index).reindex(pos.index).ffill()
d = np.empty(len(pos)); d[0]=1.0
d[1:] = np.diff(pos.index.to_numpy().astype("datetime64[D]").astype(float))
d = np.clip(d,0,10)
net = pos.sum(axis=1).to_numpy(); gross = pos.abs().sum(axis=1).to_numpy()
r = rate.to_numpy()
eng_fin = net*r*d/360.0
fix_fin = (net*(r-s) + gross*s)*d/360.0
print("\nreproduce engine financing: %.10f vs stored %.10f  (max abs diff %.3e)"
      % (eng_fin.sum(), fin.sum(), np.abs(eng_fin-fin.to_numpy()).max()))
ns = net < 0
print("net-short days: %d / %d" % (ns.sum(), len(net)))
print("  spread term on those days, engine (credit) : $%.0f" % (net[ns]*s*d[ns]/360.0).sum())
print("  spread term on those days, correct (charge): $%.0f" % (gross[ns]*s*d[ns]/360.0).sum())
print("  swing: $%.0f" % ((gross[ns]*s*d[ns]/360.0).sum() - (net[ns]*s*d[ns]/360.0).sum()))
print("\navg gross $%.0f  avg |net| $%.0f  avg net $%.0f" % (gross.mean(), np.abs(net).mean(), net.mean()))
print("TOTAL engine $%.0f  correct $%.0f  undercharge $%.0f" % (eng_fin.sum(), fix_fin.sum(), fix_fin.sum()-eng_fin.sum()))

# ---- 3. monotonicity invariant: financing must not fall as gross rises at fixed net
for g_extra in [0, 10e6, 50e6]:
    p = np.array([50e6 + g_extra, -g_extra])  # net fixed at 50mm
    print("net $50mm, gross $%.0fmm -> engine fin(1d,4%%) = $%.2f"
          % ((abs(p).sum())/1e6, (p.sum()*(gc+s)*1/360.0)))
