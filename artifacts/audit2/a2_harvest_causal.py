"""Re-run every duration_harvest arm with a CAUSAL expanding-window vol estimate."""
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
TV = 0.05
MINP = 252   # one year of warm-up before the causal estimate is trusted

def hold(pos):
    dummy = pd.DataFrame(0.0, index=pos.index, columns=tr.columns)
    return run_backtest(dummy, tr, dv, cfg, CostModel(cfg.costs),
                        yield_change_panel=yc, positions=pos, run_canary=False).metrics

def scale_full(raw, pnl):
    sd = pnl.std() * np.sqrt(252)
    return raw * (TV / sd) if sd > 0 else raw * 0.0

def scale_causal(raw, pnl):
    # expanding std of pnl through t-1 only
    sd = pnl.expanding(min_periods=MINP).std().shift(1) * np.sqrt(252)
    sd = sd.replace(0.0, np.nan).ffill()
    k = (TV / sd)
    k = k.replace([np.inf, -np.inf], np.nan)
    k = k.bfill()           # warm-up: use the first available causal estimate
    return raw.mul(k, axis=0).fillna(0.0)

def const_dv01_raw(tt, weights=None):
    unit = (100.0 / dv[tt].shift(1)).reindex(idx)
    w = pd.Series(weights if weights is not None else 1.0, index=tt, dtype=float)
    raw = unit.mul(w, axis=1)
    pnl = (raw * tr[tt].reindex(idx)).sum(axis=1).fillna(0.0) / cfg.backtest.initial_capital
    return raw, pnl

def voltgt_raw(t, lookback=63):
    unit = (100.0 / dv[t].shift(1)).reindex(idx)
    vol = tr[t].rolling(lookback, min_periods=lookback // 3).std().shift(1).reindex(idx)
    raw = unit / vol.where(vol.abs() > 1e-12)
    raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    pnl = (raw * tr[t].reindex(idx)).fillna(0.0) / cfg.backtest.initial_capital
    return raw.to_frame(t), pnl

def to_pos(scaled, cols):
    pos = pd.DataFrame(0.0, index=idx, columns=tr.columns)
    pos[cols] = scaled
    return pos

rows = []
def run(name, kind, raw, pnl, cols):
    mf = hold(to_pos(scale_full(raw, pnl), cols))
    mc = hold(to_pos(scale_causal(raw, pnl), cols))
    rows.append(dict(arm=name, kind=kind,
                     sharpe_reported=mf["sharpe"], sharpe_causal=mc["sharpe"],
                     vol_reported=mf["ann_vol"], vol_causal=mc["ann_vol"],
                     dd_reported=mf["max_drawdown"], dd_causal=mc["max_drawdown"],
                     ret_reported=mf["ann_return"], ret_causal=mc["ann_return"]))
    r = rows[-1]
    print(f"  {name:<28} {r['sharpe_reported']:>8.3f} -> {r['sharpe_causal']:>8.3f}   "
          f"vol {r['vol_reported']:>7.2%} -> {r['vol_causal']:>7.2%}   "
          f"dd {r['dd_reported']:>7.2%} -> {r['dd_causal']:>7.2%}")

print(f"  {'arm':<28} {'reported':>8}    {'causal':>8}")
for t in tenors:
    raw, pnl = const_dv01_raw([t]); run(f"constant DV01 {t}", "single", raw, pnl, [t])
print()
for t in tenors:
    raw, pnl = voltgt_raw(t); run(f"vol targeted {t}", "voltgt", raw, pnl, [t])
print()
raw, pnl = const_dv01_raw(tenors); run("equal DV01 across curve", "div", raw, pnl, tenors)
vol = tr[tenors].std(); inv = (1.0/vol)/(1.0/vol).sum()
raw, pnl = const_dv01_raw(tenors, weights=inv); run("inverse-vol across curve", "div", raw, pnl, tenors)

df = pd.DataFrame(rows)
df.to_csv(root/"artifacts/audit2/harvest_causal.csv", index=False)
print("\nmax |dSharpe| =", (df.sharpe_reported - df.sharpe_causal).abs().max())
print("mean |dSharpe| =", (df.sharpe_reported - df.sharpe_causal).abs().mean())
best_rep = df.sort_values("sharpe_reported", ascending=False).iloc[0]
best_cau = df.sort_values("sharpe_causal", ascending=False).iloc[0]
print("best by reported:", best_rep.arm, round(best_rep.sharpe_reported,3))
print("best by causal  :", best_cau.arm, round(best_cau.sharpe_causal,3))
