import sys, json, argparse
sys.path.insert(0, "/Users/sanskarawasthi/trade bot 4/artifacts/audit2/impact")
import pandas as pd, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--fix", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()

from tqe.config import Config
from tqe.backtest import engine as E
from tqe.backtest.costs import CostModel
from tqe.backtest.engine import buy_and_hold, run_backtest
from tqe.data.universe import universe_panel
from tqe.signals.alpha import predictions_to_signal, signal_decay
from tqe import cli as CLI

from tqe.config import load_config
cfg = load_config(None)
spread = float(cfg.costs.repo_spread_bp) / 1e4
print("repo_spread_bp =", cfg.costs.repo_spread_bp, "capital =", cfg.backtest.initial_capital)

if a.fix:
    from patched_loop import make_patched_core_loop
    E._core_loop = make_patched_core_loop(spread)
    print(">>> PATCHED: spread on gross")

preds = pd.read_parquet(cfg.processed_dir / "oos_predictions.parquet")
_, returns = CLI._load_returns(cfg)
tr = universe_panel(returns, "total_return")
dv = universe_panel(returns, "dv01")
yc = universe_panel(returns, "yield_change")

signal = predictions_to_signal(
    preds, method="vol_scale", window=252,
    clip=cfg.portfolio.signal_clip, min_abs=cfg.portfolio.min_signal_to_trade,
)
hl = cfg.portfolio.signal_halflife
if hl and hl > 0:
    signal = signal_decay(signal, halflife=hl)

bench = buy_and_hold(tr, cfg.backtest.benchmark, signal.index)
res = run_backtest(signal, tr, dv, cfg, CostModel(cfg.costs), benchmark=bench,
                   yield_change_panel=yc, n_trials=1)
json.dump({k: (float(v) if isinstance(v,(int,float,np.floating)) else str(v))
           for k, v in res.metrics.items()}, open(a.out, "w"), indent=2)
res.financing.to_frame("financing").to_parquet(a.out.replace(".json", "_fin.parquet"))
res.returns.to_frame("r").to_parquet(a.out.replace(".json", "_ret.parquet"))
res.positions.to_parquet(a.out.replace(".json", "_pos.parquet"))
print(res.summary())
