import sys, json
sys.path.insert(0,"src")
import pandas as pd, numpy as np
from tqe.config import load_config
from tqe.backtest.costs import CostModel
from tqe.backtest.engine import buy_and_hold, run_backtest
from tqe.data.universe import universe_panel, constant_maturity_total_return
from tqe.signals.alpha import predictions_to_signal, signal_decay

cfg = load_config()
curve = pd.read_parquet(cfg.processed_dir/"curve.parquet")
returns = constant_maturity_total_return(curve, cfg.data.core_tenors)
tr = universe_panel(returns, "total_return")
dv = universe_panel(returns, "dv01")
yc = universe_panel(returns, "yield_change")
preds = pd.read_parquet(cfg.processed_dir/"oos_predictions.parquet")
signal = predictions_to_signal(preds, method="vol_scale", window=252,
                               clip=cfg.portfolio.signal_clip, min_abs=cfg.portfolio.min_signal_to_trade)
signal = signal_decay(signal, halflife=cfg.portfolio.signal_halflife)
bench = buy_and_hold(tr, cfg.backtest.benchmark, signal.index)
res = run_backtest(signal, tr, dv, cfg, CostModel(cfg.costs), benchmark=bench,
                   yield_change_panel=yc, n_trials=1, run_canary=False)
print(res.summary())
import pickle
pickle.dump({"metrics":res.metrics}, open("artifacts/audit2/base_metrics.pkl","wb"))
res.positions.to_parquet("artifacts/audit2/pos_rerun.parquet")
res.returns.to_frame("r").to_parquet("artifacts/audit2/ret_rerun.parquet")
# consistency: net = gross - cost - fin
cap = cfg.backtest.initial_capital
lhs = res.returns*cap
rhs = res.gross_returns*cap - res.costs - res.financing
print("max abs mismatch net vs gross-cost-fin:", (lhs-rhs).abs().max())
# gross pnl recomputed from positions x returns panel
rp = tr.reindex(index=res.positions.index, columns=res.positions.columns).fillna(0.0)
g2 = (res.positions*rp).sum(axis=1)
print("max abs mismatch gross pnl:", (g2 - res.gross_returns*cap).abs().max())
