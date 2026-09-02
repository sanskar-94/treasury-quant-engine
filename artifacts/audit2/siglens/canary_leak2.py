"""A leak the pipeline's throttling CANNOT destroy: the forward 21-day return.
Does the look-ahead canary fire?"""
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
tr = universe_panel(rets, "total_return"); dv = universe_panel(rets, "dv01")
yc = universe_panel(rets, "yield_change")
idx = preds.index

raw = preds[tenors].reindex(idx)
rz = (raw - raw.mean()) / raw.std()
# LEAK: mean of the NEXT 21 days' returns, known only in the future.
fwd = tr[tenors].reindex(idx).shift(-1).rolling(21).mean().shift(-20)
fz = ((fwd - fwd.mean()) / fwd.std()).fillna(0.0)

def mk(w):
    b = (1 - w) * rz + w * fz
    s = predictions_to_signal(b, "vol_scale", 252, cfg.portfolio.signal_clip,
                              cfg.portfolio.min_signal_to_trade)
    return signal_decay(s, cfg.portfolio.signal_halflife).fillna(0.0)

print(f"{'leak_w':>7} {'sharpe':>9} {'cashneut':>9} {'canary':>9} {'ratio':>9}  fires(>0.35)?")
for w in [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]:
    m = run_backtest(mk(w), tr, dv, cfg, CostModel(cfg.costs),
                     yield_change_panel=yc, run_canary=True).metrics
    r = m.get("canary_ratio", float("nan"))
    print(f"{w:7.2f} {m['sharpe']:+9.3f} {m['cash_neutral_sharpe']:+9.3f} "
          f"{m['lookahead_canary_sharpe']:+9.3f} {r:9.4f}  {'YES' if r > 0.35 else 'no'}",
          flush=True)
