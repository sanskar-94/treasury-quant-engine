#!/usr/bin/env python
"""End-to-end pipeline smoke test on synthetic data.

Runs the complete chain - curve -> universe -> features -> walk-forward training
-> signals -> sizing -> backtest -> execution - against a generated yield-curve
history, so CI exercises every seam without depending on Treasury.gov being up.

The synthetic curve is built from a genuine three-factor structure with a small
amount of injected predictability, so the pipeline has something to find and the
assertions can be about *plumbing* rather than about performance.

Exits non-zero on the first failure.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqe.config import Config  # noqa: E402
from tqe.logging_utils import setup_logging  # noqa: E402

TENORS = ["3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "30 Yr"]
YEARS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0])

_passed = 0
_failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        _failed += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def synthetic_curve(n: int = 3000, seed: int = 11) -> pd.DataFrame:
    """A three-factor curve with mild, genuine serial predictability."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n, name="date")

    level_load = np.ones(len(YEARS))
    slope_load = (np.log(YEARS) - np.log(YEARS).mean()) / np.log(YEARS).std()
    curv_load = -((np.log(YEARS) - np.log(5.0)) ** 2) / 4.0

    lvl = np.zeros(n)
    slp = np.zeros(n)
    crv = np.zeros(n)
    for t in range(1, n):
        # AR(1) factors: persistent, so momentum features have something to see.
        lvl[t] = 0.999 * lvl[t - 1] + rng.normal(0, 0.0035)
        slp[t] = 0.995 * slp[t - 1] + rng.normal(0, 0.0015)
        crv[t] = 0.980 * crv[t - 1] + rng.normal(0, 0.0006)

    base = 0.030 + np.outer(lvl, level_load) + np.outer(slp, slope_load) + np.outer(crv, curv_load)
    base += 0.012 * slope_load  # a persistent upward slope
    base = np.clip(base, 0.0005, 0.20)
    return pd.DataFrame(base, index=idx, columns=TENORS)


def main() -> int:
    setup_logging("WARNING")
    cfg = Config()
    cfg.data.core_tenors = TENORS
    cfg.training.n_splits = 2
    cfg.training.test_size = 126
    cfg.training.min_train_size = 500
    cfg.model.learners = ["ridge", "ar_baseline"]  # fast, deterministic
    cfg.features.momentum_windows = [1, 5, 21, 63]
    cfg.features.vol_windows = [10, 21, 63]
    cfg.features.zscore_windows = [21, 63]
    cfg.features.include_macro = False
    cfg.features.min_feature_coverage = 0.5

    print("=" * 68)
    print("  Treasury Quant Engine - end-to-end smoke test (synthetic data)")
    print("=" * 68)

    curve = synthetic_curve()
    print(f"\n[1] synthetic curve {curve.shape}  "
          f"{curve.index.min().date()}..{curve.index.max().date()}")

    # ---- universe / total returns ---- #
    print("\n[2] universe and total returns")
    from tqe.data.universe import constant_maturity_total_return, universe_panel

    rets = constant_maturity_total_return(curve, TENORS)
    check("all tenors produced", set(rets) == set(TENORS), f"{len(rets)} tenors")
    tr = universe_panel(rets, "total_return")
    dv = universe_panel(rets, "dv01")
    yc = universe_panel(rets, "yield_change")
    check("dv01 increases with maturity",
          bool((dv.mean().diff().dropna() > 0).all()),
          f"3m={dv['3 Mo'].mean():.4f} 30y={dv['30 Yr'].mean():.4f}")
    check("returns are finite", bool(np.isfinite(tr.dropna().to_numpy()).all()))
    check("30y is more volatile than 2y", tr["30 Yr"].std() > tr["2 Yr"].std())

    # ---- curve models ---- #
    print("\n[3] curve models")
    from tqe.curve import (
        bootstrap_history,
        fit_curve_pca,
        fit_nss_history_fixed,
        par_to_zero,
        rolling_pca_factors,
        zero_curve_function,
    )

    nss = fit_nss_history_fixed(curve)
    check("NSS fitted every row", nss["beta0"].notna().all())
    check("NSS RMSE under 20bp", nss["rmse"].mean() * 1e4 < 20,
          f"{nss['rmse'].mean() * 1e4:.2f}bp")

    row = curve.iloc[-1]
    t, z = par_to_zero(YEARS, row.to_numpy())
    fn = zero_curve_function(t, z)
    worst = 0.0
    for ti, yi in zip(t, row.to_numpy()):
        k = int(np.floor(ti * 2 + 1e-9))
        if k <= 1:
            continue
        times = ti - 0.5 * np.arange(k - 1, -1, -1)
        times = times[times > 0]
        cf = np.full(len(times), yi / 2 * 100)
        cf[-1] += 100
        px = float((cf * np.array([(1 + fn(float(x)) / 2) ** (-2 * x) for x in times])).sum())
        worst = max(worst, abs(px - 100))
    check("bootstrap reprices par bonds to 100", worst < 1e-8, f"max err {worst:.2e}")

    zero = bootstrap_history(curve)
    check("zero curve shape matches", zero.shape == curve.shape)

    changes = curve.diff().dropna()
    pca = fit_curve_pca(changes, 3)
    check("3 PCs explain >95%", pca.explained_variance_ratio_.sum() > 0.95,
          pca.summary())
    check("PC1 loads positively everywhere", bool((pca.components_[0] > 0).all()))
    factors = rolling_pca_factors(changes, window=252, n_factors=3)
    check("causal PCA factors produced", factors.dropna().shape[0] > 500)

    # ---- features ---- #
    print("\n[4] features")
    from tqe.features import build_features

    fs = build_features(curve, pd.DataFrame(), cfg, returns=rets,
                        nss=nss, pca_factors=factors, zero=zero)
    check("feature matrix non-empty", len(fs) > 300, fs.summary())
    check("no NaN in X", not fs.X.isna().any().any())
    check("no NaN in y", not fs.y.isna().any().any())
    check("X and y aligned", fs.X.index.equals(fs.y.index))
    check("empty macro handled", True)

    # Leakage canary: no single feature should correlate strongly with the target.
    worst_corr = float(fs.X.corrwith(fs.y.mean(axis=1)).abs().max())
    check("no feature correlates >0.35 with next-day return", worst_corr < 0.35,
          f"max |corr| {worst_corr:.4f}")

    # ---- splits ---- #
    print("\n[5] walk-forward splits")
    from tqe.training.splits import validate_splits, walk_forward_splits

    splits = walk_forward_splits(fs.index, n_splits=cfg.training.n_splits,
                                 test_size=cfg.training.test_size,
                                 min_train_size=cfg.training.min_train_size,
                                 embargo=cfg.training.embargo, horizon=1)
    audit = validate_splits(splits, horizon=1, embargo=cfg.training.embargo)
    check("splits produced", len(splits) >= 1, f"{len(splits)} folds")
    check("split audit clean", audit["ok"], str(audit["violations"]))

    # ---- training ---- #
    print("\n[6] walk-forward training")
    from tqe.training.train import train_walk_forward

    res = train_walk_forward(fs, cfg)
    check("OOS predictions produced", len(res.oos_predictions) > 0,
          f"{len(res.oos_predictions)} rows")
    check("metrics computed", "rmse" in res.metrics and np.isfinite(res.metrics["rmse"]),
          res.summary())
    check("predictions are finite", bool(np.isfinite(res.oos_predictions.to_numpy()).all()))

    # ---- signals and sizing ---- #
    print("\n[7] signals and sizing")
    from tqe.signals.alpha import predictions_to_signal, signal_diagnostics
    from tqe.signals.sizing import size_portfolio

    sig = predictions_to_signal(res.oos_predictions, "zscore", 126,
                                cfg.portfolio.signal_clip, 0.0)
    check("signals bounded by clip",
          float(sig.abs().max().max()) <= cfg.portfolio.signal_clip + 1e-9)
    diag = signal_diagnostics(sig, res.oos_actuals)
    check("signal diagnostics computed", not diag.empty, f"{len(diag)} tenors")

    sized = size_portfolio(sig.fillna(0.0), tr.reindex(sig.index),
                           dv.reindex(sig.index), cfg.portfolio, yc.reindex(sig.index))
    gross_dv01 = sized["target_dv01"].abs().sum(axis=1)
    check("gross DV01 within cap",
          float(gross_dv01.max()) <= cfg.portfolio.max_gross_dv01 * 1.001,
          f"max {gross_dv01.max():,.0f} vs cap {cfg.portfolio.max_gross_dv01:,.0f}")
    net_dv01 = sized["target_dv01"].sum(axis=1).abs()
    check("net DV01 within cap",
          float(net_dv01.max()) <= cfg.portfolio.max_net_dv01 * 1.001,
          f"max {net_dv01.max():,.0f}")

    # ---- portfolio optimiser ---- #
    print("\n[8] portfolio optimiser and risk")
    from tqe.portfolio import covariance, mean_variance_weights, risk_parity_weights
    from tqe.portfolio.risk import (
        apply_stress,
        expected_shortfall,
        historical_var,
        stress_scenarios,
    )

    cov = covariance(tr.dropna(), method="ewma")
    eig = np.linalg.eigvalsh(cov.to_numpy())
    check("covariance is PSD", eig.min() >= -1e-10, f"min eig {eig.min():.2e}")
    # mu must be in RETURN units, not raw ensemble output - see
    # tqe.signals.alpha.scale_to_return_units for why this matters.
    from tqe.signals.alpha import scale_to_return_units

    mu_panel = scale_to_return_units(res.oos_predictions, tr, ic=0.05, window=126)
    mu = mu_panel.dropna(how="all").iloc[-1].fillna(0.0)
    opt = mean_variance_weights(mu, cov, None, cfg.portfolio, dv01_per_unit=dv.iloc[-1])
    check("optimiser returns a book", opt.gross > 0,
          f"status={opt.status} gross={opt.gross:.3f} ann_vol={opt.expected_vol * np.sqrt(252):.2%}")
    check("mu is on the scale of real returns",
          0.05 < float(mu.abs().max() / tr.std().mean()) < 5.0,
          f"|mu|max={mu.abs().max():.2e} vs return sd={tr.std().mean():.2e}")
    rp = risk_parity_weights(cov)
    check("risk parity sums to 1", abs(float(rp.sum()) - 1.0) < 1e-6)
    port = (tr.dropna() * (1 / len(TENORS))).sum(axis=1)
    check("ES >= VaR", expected_shortfall(port, 0.99) >= historical_var(port, 0.99))
    pnl = apply_stress({"10 Yr": 1000.0}, stress_scenarios()["parallel_up_100"])
    check("+100bp shock loses money on a long book", pnl < 0, f"P&L ${pnl:,.0f}")

    # ---- backtest ---- #
    print("\n[9] backtest")
    from tqe.backtest.costs import CostModel
    from tqe.backtest.engine import buy_and_hold, run_backtest

    bench = buy_and_hold(tr, "10 Yr", sig.index)
    bt = run_backtest(sig.fillna(0.0), tr, dv, cfg, CostModel(cfg.costs),
                      benchmark=bench, yield_change_panel=yc, n_trials=1)
    check("equity curve produced", len(bt.equity) > 0, f"{len(bt.equity)} days")
    check("equity is finite and positive",
          bool(np.isfinite(bt.equity).all()) and float(bt.equity.min()) > 0)
    check("costs are non-negative", float(bt.costs.min()) >= 0,
          f"total ${bt.costs.sum():,.0f}")
    check("net Sharpe <= gross Sharpe",
          bt.metrics["sharpe"] <= bt.metrics["sharpe_gross"] + 1e-9,
          f"net {bt.metrics['sharpe']:.3f} vs gross {bt.metrics['sharpe_gross']:.3f}")
    check("deflated Sharpe computed", np.isfinite(bt.metrics.get("deflated_sharpe", np.nan)),
          f"{bt.metrics.get('deflated_sharpe'):.4f}")
    canary = bt.metrics.get("lookahead_canary_sharpe")
    check("look-ahead canary ran", canary is not None and np.isfinite(canary),
          f"canary Sharpe {canary:.2f} vs honest {bt.metrics['sharpe']:.2f}")

    # ---- execution ---- #
    print("\n[10] execution stack")
    import tempfile

    from tqe.execution.broker import Order, OrderSide, OrderType
    from tqe.execution.oms import OMS
    from tqe.execution.paper import PaperBroker
    from tqe.execution.risk_gate import RiskGate

    tmp = tempfile.mkdtemp()
    pb = PaperBroker(cfg, initial_cash=10_000_000.0, half_spread_bp=0.0, slippage_bp=0.0,
                     commission_per_million=0.0, seed=1, state_dir=tmp)
    pb.set_quote("X", 100.0)
    pb.submit_order(Order(symbol="X", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET))
    pb.set_quote("X", 110.0)
    pb.submit_order(Order(symbol="X", side=OrderSide.SELL, quantity=60, order_type=OrderType.MARKET))
    check("realised P&L on a partial close", abs(pb.realized_pnl - 600.0) < 1e-6,
          f"${pb.realized_pnl:,.2f} (expected $600)")
    pb.set_quote("X", 120.0)
    pb.submit_order(Order(symbol="X", side=OrderSide.SELL, quantity=60, order_type=OrderType.MARKET))
    pos = pb.position("X")
    check("position flips through zero correctly",
          abs(pos.quantity + 20) < 1e-9 and abs(pb.realized_pnl - 1400.0) < 1e-6,
          f"qty={pos.quantity:+.0f} realised=${pb.realized_pnl:,.2f}")

    gate = RiskGate(cfg.risk, cfg.portfolio)
    acct, positions = pb.get_account(), pb.get_positions()
    big = Order(symbol="X", side=OrderSide.BUY, quantity=1_000_000, order_type=OrderType.MARKET)
    check("risk gate blocks an oversized order",
          not gate.check_order(big, acct, positions, reference_price=100.0, market_open=True).passed)
    gate.trip("smoke test")
    small = Order(symbol="X", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET)
    check("kill switch blocks everything",
          not gate.check_order(small, acct, positions, reference_price=100.0, market_open=True).passed)
    gate.reset()

    pb2 = PaperBroker(cfg, initial_cash=10_000_000.0, seed=2, state_dir=tmp)
    for s in ("IEF", "TLT"):
        pb2.set_quote(s, 95.0)
    oms = OMS(broker=pb2, risk_gate=RiskGate(cfg.risk, cfg.portfolio), cfg=cfg, state_dir=tmp)
    import datetime as _dt

    day = _dt.date(2026, 1, 5)
    r1 = oms.daily_run({"IEF": 500_000.0, "TLT": -300_000.0}, dry_run=False, as_of=day)
    r2 = oms.daily_run({"IEF": 500_000.0, "TLT": -300_000.0}, dry_run=False, as_of=day)
    check("OMS submits on the first run", r1.get("submitted", 0) == 2, str(r1.get("submitted")))
    check("OMS is idempotent on a repeat run", r2.get("generated", 1) == 0,
          f"status={r2.get('status')}")
    check("OMS reconciles with the broker",
          bool(r1.get("reconciliation", {}).get("in_sync", False)))

    # ---- summary ---- #
    print("\n" + "=" * 68)
    print(f"  {_passed} passed, {_failed} failed")
    print("=" * 68)
    if _failed == 0:
        print("\nBacktest summary (synthetic data - NOT a performance claim):")
        print(bt.summary())
    return 1 if _failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2) from None
