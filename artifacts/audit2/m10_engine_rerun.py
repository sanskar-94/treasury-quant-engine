"""End-to-end confirmation through the real run_backtest, no source edits.

Trick: the engine computes fin = net_notional * rate * d_trade/360.  Feeding it
rate * (d_settle/d_trade) makes it compute net_notional * rate * d_settle/360,
i.e. the settlement-aligned window the return leg actually accrues over.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0,"src")
from tqe.config import load_config
from tqe.backtest.engine import run_backtest, _funding_from_curve, buy_and_hold
from tqe.backtest.costs import CostModel
from tqe.data.calendar import settlement_date
from tqe.data.universe import constant_maturity_total_return, universe_panel
from tqe.signals.alpha import predictions_to_signal, signal_decay

cfg=load_config(Path("configs/default.yaml").resolve())
curve=pd.read_parquet(cfg.processed_dir/"curve.parquet")
rets=constant_maturity_total_return(curve, cfg.data.core_tenors)
tr=universe_panel(rets,"total_return"); dv=universe_panel(rets,"dv01"); yc=universe_panel(rets,"yield_change")
preds=pd.read_parquet(cfg.processed_dir/"oos_predictions.parquet")
sig=predictions_to_signal(preds, method="vol_scale", window=252,
                          clip=cfg.portfolio.signal_clip, min_abs=cfg.portfolio.min_signal_to_trade)
if cfg.portfolio.signal_halflife:
    sig=signal_decay(sig, halflife=cfg.portfolio.signal_halflife)
bench=buy_and_hold(tr, cfg.backtest.benchmark, sig.index)

base=run_backtest(sig,tr,dv,cfg,CostModel(cfg.costs),benchmark=bench,yield_change_panel=yc,run_canary=False)
idx=base.returns.index
fr=_funding_from_curve(cfg, tr.index).reindex(idx).ffill()
dt=np.empty(len(idx)); dt[0]=1.0; dt[1:]=np.diff(idx.to_numpy().astype('datetime64[D]').astype(float)); dt=np.clip(dt,0,10)
so=np.array([pd.Timestamp(settlement_date(ts.date(),None)).to_datetime64().astype('datetime64[D]').astype(float) for ts in idx])
ds=np.empty(len(idx)); ds[0]=1.0; ds[1:]=np.diff(so); ds=np.clip(ds,0,10)
fr_adj=pd.Series(fr.to_numpy()*ds/dt, index=idx)

fixed=run_backtest(sig,tr,dv,cfg,CostModel(cfg.costs),benchmark=bench,yield_change_panel=yc,
                   funding_rate=fr_adj, run_canary=False)

for lab,r in (("AS SHIPPED", base),("SETTLEMENT-ALIGNED FINANCING", fixed)):
    m=r.metrics
    print(f"\n{lab}: n={int(m['n_obs'])} sharpe={m['sharpe']:.4f} ann_vol={m['ann_vol']:.5f} "
          f"ann_ret={m['ann_return']:.5f} sortino={m['sortino']:.4f} calmar={m['calmar']:.4f} "
          f"maxdd={m['max_drawdown']:.5f} DSR={m['deflated_sharpe']:.4f} fin={m['total_financing']:,.0f} "
          f"n_trials={m['n_trials']}")
