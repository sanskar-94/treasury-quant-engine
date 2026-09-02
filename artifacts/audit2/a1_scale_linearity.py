"""Is the reported Sharpe invariant to a constant position multiplier?

If it is, the full-sample vol scaling in scripts/duration_harvest.py is a
look-ahead that cannot move the headline Sharpe.  If not, quantify.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
root = Path("/Users/sanskarawasthi/trade bot 4")
sys.path.insert(0, str(root / "src"))

from tqe.backtest.costs import CostModel
from tqe.backtest.engine import run_backtest
from tqe.config import load_config
from tqe.data.universe import constant_maturity_total_return, universe_panel
from tqe.logging_utils import setup_logging
setup_logging("ERROR")

cfg = load_config(root / "configs" / "default.yaml")
curve = pd.read_parquet(root / "data/processed/curve.parquet")
tenors = [t for t in cfg.data.core_tenors if t in curve.columns]
rets = constant_maturity_total_return(curve, tenors)
tr, dv, yc = (universe_panel(rets, f) for f in ("total_return", "dv01", "yield_change"))
idx = tr.dropna(how="any").index
print("window", idx.min().date(), idx.max().date(), len(idx))

def hold(pos):
    dummy = pd.DataFrame(0.0, index=pos.index, columns=tr.columns)
    return run_backtest(dummy, tr, dv, cfg, CostModel(cfg.costs),
                        yield_change_panel=yc, positions=pos, run_canary=False).metrics

def constant_dv01_book(tenors_, target_vol=0.05, weights=None):
    unit = (100.0 / dv[tenors_].shift(1)).reindex(idx)
    w = pd.Series(weights if weights is not None else 1.0, index=tenors_, dtype=float)
    raw = unit.mul(w, axis=1)
    pnl = (raw * tr[tenors_].reindex(idx)).sum(axis=1).fillna(0.0) / cfg.backtest.initial_capital
    sd = pnl.std() * np.sqrt(252)
    scaled = raw * (target_vol / sd) if sd > 0 else raw * 0.0
    pos = pd.DataFrame(0.0, index=idx, columns=tr.columns)
    pos[tenors_] = scaled
    return pos, float(sd)

base, sd = constant_dv01_book(["10 Yr"])
print("full-sample sd of unit-DV01 pnl (ann):", sd)
print(f"{'k':>7} {'sharpe':>10} {'ann_vol':>9} {'ann_ret':>9} {'maxDD':>9} {'costdrag':>9}")
for k in [0.25, 0.5, 1.0, 2.0, 4.0, 10.0]:
    m = hold(base * k)
    print(f"{k:>7.2f} {m['sharpe']:>10.6f} {m['ann_vol']:>9.4%} {m['ann_return']:>9.4%} "
          f"{m['max_drawdown']:>9.4%} {m['cost_drag_annual']:>9.4%}")
