import pandas as pd, numpy as np, sys
sys.path.insert(0,"src")
from tqe.config import load_config
from tqe.backtest.costs import CostModel
from tqe.backtest.engine import run_backtest
from tqe.data.universe import constant_maturity_total_return, universe_panel
from tqe.signals.alpha import predictions_to_signal, signal_decay
import logging; logging.disable(logging.WARNING)

cfg = load_config("configs/default.yaml")
preds = pd.read_parquet(cfg.processed_dir/"oos_predictions.parquet")
c = pd.read_parquet(cfg.processed_dir/"curve.parquet")
rets = constant_maturity_total_return(c, cfg.data.core_tenors)
tr = universe_panel(rets,"total_return"); dv = universe_panel(rets,"dv01"); yc = universe_panel(rets,"yield_change")
sig = predictions_to_signal(preds, method="vol_scale", window=252, clip=cfg.portfolio.signal_clip, min_abs=cfg.portfolio.min_signal_to_trade)
sig = signal_decay(sig, halflife=cfg.portfolio.signal_halflife)

def run(sg, trp, tag):
    r = run_backtest(sg, trp, dv, cfg, CostModel(cfg.costs), yield_change_panel=yc, n_trials=1, run_canary=False)
    m=r.metrics
    print(f"{tag:34s} sharpe={m['sharpe']:+.4f} ann_ret={m['ann_return']:+.5f} vol={m['ann_vol']:.5f} gross_sh={m['sharpe_gross']:+.3f}")
    return r

run(sig, tr, "baseline")
run(sig.shift(1).dropna(how='all'), tr, "signal lagged 1d (should drop)")
run(sig.shift(-1).dropna(how='all'), tr, "signal ADVANCED 1d (leak test)")
