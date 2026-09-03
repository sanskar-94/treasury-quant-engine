"""How much Sharpe does an UNAMBIGUOUS look-ahead earn, and does the canary see it?"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/Users/sanskarawasthi/trade bot 4")
sys.path.insert(0, str(ROOT / "src"))
from tqe.logging_utils import setup_logging
setup_logging("ERROR")
from tqe.config import load_config
from tqe.data.universe import constant_maturity_total_return, universe_panel
from tqe.backtest.costs import CostModel
from tqe.backtest.engine import run_backtest

cfg = load_config(ROOT / "configs" / "default.yaml")
curve = pd.read_parquet(ROOT / "data/processed/curve.parquet")
preds = pd.read_parquet(ROOT / "data/processed/oos_predictions.parquet")
tenors = [t for t in cfg.data.core_tenors if t in curve.columns]
rets = constant_maturity_total_return(curve, tenors)
tr = universe_panel(rets, "total_return"); dv = universe_panel(rets, "dv01")
yc = universe_panel(rets, "yield_change")
idx = preds.index
fut = tr[tenors].reindex(idx).fillna(0.0)

def show(tag, sig, **kw):
    m = run_backtest(sig, tr, dv, cfg, CostModel(cfg.costs),
                     yield_change_panel=yc, run_canary=True, **kw).metrics
    r = m.get("canary_ratio", float("nan"))
    print(f"{tag:38s} sharpe={m['sharpe']:+8.3f} cashneut={m['cash_neutral_sharpe']:+8.3f} "
          f"canary={m['lookahead_canary_sharpe']:+8.2f} ratio={r:+8.4f} "
          f"turn={m['ann_turnover']:7.1f} {'FIRES' if r>0.35 else '.'}", flush=True)

# 1. Perfect foresight signal, straight into the engine at default settings
show("sign(future ret), default cfg", np.sign(fut) * 1.0)
# 2. Same, but relaxing the throttles one at a time
import copy
for rb, band, hl in [("daily", 2.0, 10.0), ("monthly", 0.0, 10.0), ("daily", 0.0, 10.0)]:
    c = copy.deepcopy(cfg)
    c.portfolio.rebalance = rb
    c.portfolio.no_trade_band = band
    m = run_backtest(np.sign(fut) * 1.0, tr, dv, c, CostModel(c.costs),
                     yield_change_panel=yc, run_canary=True).metrics
    r = m.get("canary_ratio", float("nan"))
    print(f"{'sign(future) rb=%s band=%.1f' % (rb, band):38s} sharpe={m['sharpe']:+8.3f} "
          f"cashneut={m['cash_neutral_sharpe']:+8.3f} canary={m['lookahead_canary_sharpe']:+8.2f} "
          f"ratio={r:+8.4f} turn={m['ann_turnover']:7.1f} {'FIRES' if r>0.35 else '.'}", flush=True)
