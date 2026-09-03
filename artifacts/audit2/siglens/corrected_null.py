"""Shipped block_sign_flip null vs an exchangeable block sign-flip null.

Exchangeable version: keep the book, the turnover, the costs and the financing
EXACTLY as traded; flip the sign of the day's gross P&L in contiguous blocks.
The observed statistic is the all-(+1) pattern, which IS a member of the
reference set, so the test is exact under H0 by construction.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/Users/sanskarawasthi/trade bot 4")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from tqe.logging_utils import setup_logging
setup_logging("ERROR")
from tqe.config import load_config
from tqe.data.universe import constant_maturity_total_return, universe_panel
from tqe.training.metrics import performance_metrics
import benchmark_carry as bc

cfg = load_config(ROOT / "configs" / "default.yaml")
curve = pd.read_parquet(ROOT / "data/processed/curve.parquet")
preds = pd.read_parquet(ROOT / "data/processed/oos_predictions.parquet")
tenors = [t for t in cfg.data.core_tenors if t in curve.columns]
rets = constant_maturity_total_return(curve, tenors)
tr = universe_panel(rets, "total_return"); dv = universe_panel(rets, "dv01")
yc = universe_panel(rets, "yield_change")
idx = preds.index
carry = bc.carry_signal(curve, rets, tenors).shift(1).reindex(idx)
sig_carry = bc.to_signal(carry, cfg)
sig_model = bc.to_signal(preds[tenors], cfg)
sig_both = bc.to_signal(
    sig_carry.div(sig_carry.std().replace(0, np.nan), axis=1).fillna(0.0)
    + sig_model.div(sig_model.std().replace(0, np.nan), axis=1).fillna(0.0), cfg)

def exch_p(sig, k=40, block=63, seed0=0):
    r = bc.evaluate(sig, tr, dv, yc, cfg)
    cap = cfg.backtest.initial_capital
    gross = r.gross_returns.to_numpy()
    drag = (r.costs.to_numpy() + r.financing.to_numpy()) / cap
    n = len(gross); nb = int(np.ceil(n / block))
    obs = performance_metrics(pd.Series(gross - drag, index=r.returns.index))["sharpe"]
    pl = []
    for i in range(k):
        eps = np.random.default_rng(seed0 + i).choice([-1.0, 1.0], size=nb)
        e = np.repeat(eps, block)[:n]
        pl.append(performance_metrics(pd.Series(e * gross - drag, index=r.returns.index))["sharpe"])
    beat = sum(1 for v in pl if v >= obs)
    return obs, float(np.mean(pl)), float(np.std(pl)), (beat + 1) / (k + 1)

def shipped_p(sig, k=40):
    m = bc.evaluate(sig, tr, dv, yc, cfg).metrics["sharpe"]
    pl = [bc.evaluate(bc.block_sign_flip(sig, 63, i), tr, dv, yc, cfg).metrics["sharpe"]
          for i in range(k)]
    beat = sum(1 for v in pl if v >= m)
    return m, float(np.mean(pl)), float(np.std(pl)), (beat + 1) / (k + 1)

print(f"{'signal':13s} {'sharpe':>8} | {'SHIPPED null':^30} | {'EXCHANGEABLE null':^30}")
print(f"{'':13s} {'':>8} | {'plc_mean':>9}{'plc_sd':>8}{'p':>8}{'z':>6} | "
      f"{'plc_mean':>9}{'plc_sd':>8}{'p':>8}{'z':>6}")
for name, s in [("carry", sig_carry), ("model", sig_model), ("carry+model", sig_both)]:
    m1, pm1, ps1, p1 = shipped_p(s)
    m2, pm2, ps2, p2 = exch_p(s)
    print(f"{name:13s} {m1:+8.3f} | {pm1:+9.3f}{ps1:8.3f}{p1:8.4f}{(m1-pm1)/max(ps1,1e-9):+6.2f} | "
          f"{pm2:+9.3f}{ps2:8.3f}{p2:8.4f}{(m2-pm2)/max(ps2,1e-9):+6.2f}", flush=True)

# calibration of the EXCHANGEABLE null on zero-skill rotations
n = len(sig_model)
rng = np.random.default_rng(777)
ps = []
for off in rng.integers(300, n - 300, size=60):
    rot = pd.DataFrame(np.roll(sig_model.to_numpy(), int(off), axis=0),
                       index=sig_model.index, columns=sig_model.columns)
    ps.append(exch_p(rot)[3])
ps = np.array(ps)
print(f"\nEXCHANGEABLE null on 60 zero-skill rotations: mean p {ps.mean():.3f} "
      f"median {np.median(ps):.3f} frac<=0.10 {(ps<=0.10).mean():.3f} "
      f"frac<=0.05 {(ps<=0.0488).mean():.3f} frac at floor {(ps<=0.0245).mean():.3f}")
