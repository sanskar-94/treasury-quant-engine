"""Corrected _core_loop: repo spread charged on GROSS notional, GC on NET."""
import numpy as np
import pandas as pd
from tqe.backtest.engine import EPS, log


def make_patched_core_loop(spread_decimal: float):
    def _core_loop_fixed(
        positions, returns_panel, cost_model, buckets, capital,
        include_costs, slippage_multiplier, funding_rate=None, include_financing=True,
    ):
        tenors = list(positions.columns)
        rets = returns_panel.reindex(index=positions.index, columns=tenors).fillna(0.0)
        pos = positions.to_numpy(dtype=float)
        ret = rets.to_numpy(dtype=float)
        prev = np.vstack([np.zeros((1, pos.shape[1])), pos[:-1]])
        trade = pos - prev
        gross_pnl = (pos * ret).sum(axis=1)

        cost_arr = np.zeros(len(pos))
        if include_costs and cost_model is not None:
            for j, tenor in enumerate(tenors):
                bucket = buckets.get(tenor, "10y")
                traded = np.abs(trade[:, j])
                nz = traded > EPS
                if not nz.any():
                    continue
                costs_j = np.array(
                    [cost_model.total_cost(float(v), bucket) for v in traded[nz]], dtype=float
                )
                cost_arr[nz] += costs_j * slippage_multiplier

        idx = positions.index
        fin_arr = np.zeros(len(pos))
        if include_financing and funding_rate is not None:
            aligned = funding_rate.reindex(idx).ffill()
            missing = int(aligned.isna().sum())
            if missing:
                fallback = float(aligned.max()) if aligned.notna().any() else 0.0
                aligned = aligned.fillna(fallback)
            rate = aligned.to_numpy(dtype=float)
            days = np.empty(len(idx))
            days[0] = 1.0
            days[1:] = np.diff(idx.to_numpy().astype("datetime64[D]").astype(float))
            days = np.clip(days, 0.0, 10.0)
            net_notional = pos.sum(axis=1)
            gross_notional = np.abs(pos).sum(axis=1)
            # ---- THE FIX ----
            # incoming `rate` is GC + spread (engine folds the spread in).
            gc = rate - spread_decimal
            fin_arr = (net_notional * gc + gross_notional * spread_decimal) * days / 360.0

        net_pnl = gross_pnl - cost_arr - fin_arr
        return (
            pd.Series(net_pnl / capital, index=idx, name="returns"),
            pd.Series(gross_pnl / capital, index=idx, name="gross_returns"),
            pd.Series(cost_arr, index=idx, name="costs"),
            pd.DataFrame(trade, index=idx, columns=tenors),
            pd.Series(fin_arr, index=idx, name="financing"),
        )
    return _core_loop_fixed
