"""Does re-imposing the configured leverage cap AFTER banding change the headline?"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
root = Path("/Users/sanskarawasthi/trade bot 4"); sys.path.insert(0,str(root/"src"))
from tqe.config import load_config
from tqe.backtest.costs import CostModel
from tqe.backtest.engine import buy_and_hold, run_backtest
from tqe.data.universe import constant_maturity_total_return, universe_panel
from tqe.signals.alpha import predictions_to_signal, signal_decay
from tqe.logging_utils import setup_logging
setup_logging("ERROR")
cfg = load_config(root/"configs"/"default.yaml")
curve = pd.read_parquet(root/"data/processed/curve.parquet")
preds = pd.read_parquet(root/"data/processed/oos_predictions.parquet")
tenors=[t for t in cfg.data.core_tenors if t in curve.columns]
rets = constant_maturity_total_return(curve, tenors)
tr,dv,yc = (universe_panel(rets,f) for f in ("total_return","dv01","yield_change"))
sig = predictions_to_signal(preds, "vol_scale", 252, cfg.portfolio.signal_clip, cfg.portfolio.min_signal_to_trade)
sig = signal_decay(sig, cfg.portfolio.signal_halflife)
bench = buy_and_hold(tr, cfg.backtest.benchmark, sig.index)

base = run_backtest(sig, tr, dv, cfg, CostModel(cfg.costs), benchmark=bench, yield_change_panel=yc)
pos = base.positions.copy()
cap = cfg.portfolio.capital*cfg.portfolio.max_leverage
g = pos.abs().sum(axis=1)
fac = (cap/g.where(g>cap)).fillna(1.0)
pos_capped = pos.mul(fac, axis=0)
capped = run_backtest(sig, tr, dv, cfg, CostModel(cfg.costs), benchmark=bench,
                      yield_change_panel=yc, positions=pos_capped, run_canary=False)
for nm, m in [("as shipped (cap breached)", base.metrics), ("cap enforced after band", capped.metrics)]:
    print(f"  {nm:28s} sharpe {m['sharpe']:+.4f}  ann_ret {m['ann_return']:+.4%}  "
          f"vol {m['ann_vol']:.4%}  dd {m['max_drawdown']:+.4%}  turn {m['ann_turnover']:.2f}")
print("  days over cap: %.1f%%   max leverage %.3f" % (100*(g>cap+1e-6).mean(), g.max()/cfg.portfolio.capital))
