import sys, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
root = Path("/Users/sanskarawasthi/trade bot 4")
sys.path.insert(0, str(root/"src"))
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
tenors = [t for t in cfg.data.core_tenors if t in curve.columns]
rets = constant_maturity_total_return(curve, tenors)
tr, dv, yc = (universe_panel(rets, f) for f in ("total_return","dv01","yield_change"))

signal = predictions_to_signal(preds, method="vol_scale", window=252,
                               clip=cfg.portfolio.signal_clip,
                               min_abs=cfg.portfolio.min_signal_to_trade)
print("signal all-NaN rows:", int(signal.isna().all(axis=1).sum()))
signal = signal_decay(signal, halflife=cfg.portfolio.signal_halflife)
bench = buy_and_hold(tr, cfg.backtest.benchmark, signal.index)
res = run_backtest(signal, tr, dv, cfg, CostModel(cfg.costs), benchmark=bench,
                   yield_change_panel=yc, n_trials=1)
print(res.summary())
m = res.metrics
ref = json.loads((root/"results/metrics.json").read_text())
print("\nreproduced sharpe %.6f   reported %.6f" % (m["sharpe"], ref["sharpe"]))
res.positions.to_parquet(root/"artifacts/audit2/repro_positions.parquet")
res.returns.to_frame("r").to_parquet(root/"artifacts/audit2/repro_returns.parquet")
res.exposures.to_parquet(root/"artifacts/audit2/repro_exposures.parquet")
