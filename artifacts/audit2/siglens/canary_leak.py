"""Does the look-ahead canary actually fire when the future leaks in?"""
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
from tqe.signals.alpha import predictions_to_signal, signal_decay

cfg = load_config(ROOT / "configs" / "default.yaml")
curve = pd.read_parquet(ROOT / "data/processed/curve.parquet")
preds = pd.read_parquet(ROOT / "data/processed/oos_predictions.parquet")
tenors = [t for t in cfg.data.core_tenors if t in curve.columns]
rets = constant_maturity_total_return(curve, tenors)
tr = universe_panel(rets, "total_return")
dv = universe_panel(rets, "dv01")
yc = universe_panel(rets, "yield_change")
idx = preds.index

raw = preds[tenors].reindex(idx)
future = tr[tenors].reindex(idx)            # <-- the return realised on day t
# standardise both to comparable scale
rz = (raw - raw.mean()) / raw.std()
fz = (future - future.mean()) / future.std()

def mk(w):
    """w = leak weight in [0,1]; w=0 honest, w=1 pure look-ahead."""
    blended = (1 - w) * rz + w * fz
    s = predictions_to_signal(blended, "vol_scale", 252,
                              cfg.portfolio.signal_clip, cfg.portfolio.min_signal_to_trade)
    return signal_decay(s, cfg.portfolio.signal_halflife).fillna(0.0)

print(f"{'leak_w':>7} {'sharpe':>9} {'cashneut':>9} {'canary':>9} {'ratio':>9}  fires?")
for w in [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]:
    r = run_backtest(mk(w), tr, dv, cfg, CostModel(cfg.costs),
                     yield_change_panel=yc, run_canary=True)
    m = r.metrics
    ratio = m.get("canary_ratio", float("nan"))
    print(f"{w:7.2f} {m['sharpe']:+9.3f} {m['cash_neutral_sharpe']:+9.3f} "
          f"{m['lookahead_canary_sharpe']:+9.3f} {ratio:9.4f}  "
          f"{'YES' if ratio > 0.35 else 'no'}")
