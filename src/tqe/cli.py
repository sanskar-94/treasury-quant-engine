"""Command-line interface.

The whole pipeline is reachable from one binary::

    tqe data pull            # download and cache Treasury + FRED history
    tqe data status          # what is on disk, and how fresh
    tqe curve fit            # NSS betas, bootstrapped zeros, PCA factors
    tqe features build       # assemble the design matrix
    tqe train                # walk-forward evaluation + deployable bundle
    tqe backtest             # simulate with costs, write a tearsheet
    tqe predict              # today's forecast from the saved model
    tqe attribute            # where the P&L came from, by factor and by source
    tqe tune                 # nested walk-forward hyper-parameter search
    tqe charts               # curve, factor, attribution and signal diagnostics
    tqe trade --dry-run      # one live session, no orders sent
    tqe serve                # FastAPI service

Stages are deliberately separate and each persists its output, so a failure in
training does not cost you the 37-year data pull, and a backtest can be re-run
against a fixed model without refitting anything.

Built on ``argparse`` rather than a CLI framework: one less dependency for
something that has to run unattended on a schedule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, load_config
from .logging_utils import get_logger, setup_logging

log = get_logger("cli")

PROCESSED = "data/processed"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _cfg_from_args(args: argparse.Namespace) -> Config:
    overrides: dict[str, Any] = {}
    for item in getattr(args, "set", None) or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()
    cfg = load_config(getattr(args, "config", None), overrides=overrides or None)
    if getattr(args, "log_level", None):
        cfg.log_level = args.log_level
    return cfg


def _p(cfg: Config, name: str) -> Path:
    return cfg.processed_dir / name


def _require(path: Path, hint: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run `{hint}` first.")
    return pd.read_parquet(path)


def _load_returns(cfg: Config):
    from .data.universe import constant_maturity_total_return

    curve = _require(_p(cfg, "curve.parquet"), "tqe data pull")
    return curve, constant_maturity_total_return(curve, cfg.data.core_tenors)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def cmd_data_pull(args: argparse.Namespace) -> int:
    from .data.sources import clean_curve, curve_coverage, fetch_treasury_curve, load_market_data

    cfg = _cfg_from_args(args)
    if args.curve_only:
        start = int(args.start or pd.Timestamp(cfg.data.start_date).year)
        end = int(args.end or pd.Timestamp.today().year)
        curve = clean_curve(fetch_treasury_curve(start, end, cfg.cache_dir, force=args.force))
        curve.to_parquet(_p(cfg, "curve.parquet"))
        macro = pd.DataFrame()
    else:
        curve, macro = load_market_data(cfg, force=args.force)
        curve.to_parquet(_p(cfg, "curve.parquet"))
        if not macro.empty:
            macro.to_parquet(_p(cfg, "macro.parquet"))

    print(f"curve  {curve.shape}  {curve.index.min().date()} .. {curve.index.max().date()}")
    print(curve_coverage(curve).to_string())
    if not macro.empty:
        print(f"\nmacro  {macro.shape}")
        print(macro.notna().sum().to_string())
    else:
        print("\nmacro  not retrieved (run scripts/fetch_macro.py if FRED was rate-limited)")
    return 0


def cmd_data_status(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    rows = []
    for name in ("curve", "macro", "nss", "zero", "pca_factors", "X", "y",
                 "oos_predictions", "oos_actuals"):
        path = _p(cfg, f"{name}.parquet")
        if path.exists():
            df = pd.read_parquet(path)
            rows.append({
                "dataset": name, "rows": len(df), "cols": df.shape[1],
                "start": df.index.min(), "end": df.index.max(),
                "mb": round(path.stat().st_size / 1e6, 2),
            })
        else:
            rows.append({"dataset": name, "rows": 0, "cols": 0, "start": None, "end": None, "mb": 0})
    print(pd.DataFrame(rows).to_string(index=False))

    from .models.registry import latest_bundle

    bundle = latest_bundle(cfg.training.artifacts_dir)
    print(f"\nlatest model bundle: {bundle if bundle else 'none - run `tqe train`'}")
    return 0


# --------------------------------------------------------------------------- #
# curve
# --------------------------------------------------------------------------- #
def cmd_curve_fit(args: argparse.Namespace) -> int:
    from .curve import (
        bootstrap_history,
        fit_curve_pca,
        fit_nss,
        fit_nss_history_fixed,
        rolling_pca_factors,
    )
    from .data.sources import TENOR_YEARS

    cfg = _cfg_from_args(args)
    curve = _require(_p(cfg, "curve.parquet"), "tqe data pull")

    if args.date:
        row = curve.loc[pd.Timestamp(args.date)].dropna()
        t = np.array([TENOR_YEARS[c] for c in row.index])
        params, rmse = fit_nss(t, row.to_numpy(), model=cfg.curve.model)
        print(f"NSS fit {args.date}  rmse={rmse * 1e4:.3f}bp")
        print(f"  beta0={params.beta0:+.6f} beta1={params.beta1:+.6f} "
              f"beta2={params.beta2:+.6f} beta3={params.beta3:+.6f}")
        print(f"  tau1={params.tau1:.4f} tau2={params.tau2:.4f}")
        print(f"  level={params.level:+.5f} slope={params.slope:+.5f} curvature={params.curvature:+.5f}")
        for c, tv in zip(row.index, t):
            print(f"   {c:10s} mkt={row[c] * 100:6.3f}%  fit={float(params.zero_rate(tv)) * 100:6.3f}%")
        return 0

    log.info("fitting NSS (fixed decays) over %d dates", len(curve))
    nss = fit_nss_history_fixed(curve, model=cfg.curve.model)
    nss.to_parquet(_p(cfg, "nss.parquet"))
    ok = nss.dropna(subset=["beta0"])
    print(f"NSS      {len(ok)}/{len(nss)} fitted, mean RMSE {ok.rmse.mean() * 1e4:.2f}bp")

    zero = bootstrap_history(curve)
    zero.to_parquet(_p(cfg, "zero.parquet"))
    print(f"zeros    {zero.shape}")

    core = [c for c in cfg.data.core_tenors if c in curve.columns]
    changes = curve[core].diff().dropna(how="any")
    pca = fit_curve_pca(changes, cfg.curve.n_pca_factors)
    factors = rolling_pca_factors(changes, window=252, n_factors=cfg.curve.n_pca_factors)
    factors.to_parquet(_p(cfg, "pca_factors.parquet"))
    print(f"PCA      {pca.summary()}")
    print(f"         causal factors {factors.shape}, first valid {factors.dropna().index.min().date()}")
    return 0


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def cmd_features_build(args: argparse.Namespace) -> int:
    from .features import build_features

    cfg = _cfg_from_args(args)
    curve, returns = _load_returns(cfg)

    def opt(name):
        p = _p(cfg, f"{name}.parquet")
        return pd.read_parquet(p) if p.exists() else None

    fs = build_features(
        curve, opt("macro"), cfg, returns=returns,
        nss=opt("nss"), pca_factors=opt("pca_factors"), zero=opt("zero"),
    )
    fs.X.to_parquet(_p(cfg, "X.parquet"))
    fs.y.to_parquet(_p(cfg, "y.parquet"))
    (cfg.processed_dir / "feature_meta.json").write_text(json.dumps(fs.metadata, indent=2, default=str))
    print(fs.summary())
    print(json.dumps(fs.metadata, indent=2, default=str))
    return 0


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> int:
    from .features.builder import FeatureSet
    from .training.train import train_walk_forward

    cfg = _cfg_from_args(args)
    if args.learners:
        cfg.model.learners = [s.strip() for s in args.learners.split(",")]

    X = _require(_p(cfg, "X.parquet"), "tqe features build")
    y = _require(_p(cfg, "y.parquet"), "tqe features build")
    meta_path = cfg.processed_dir / "feature_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    fs = FeatureSet(X=X, y=y, metadata=meta)

    out = Path(args.out or Path(cfg.training.artifacts_dir) / "ensemble_v1")
    res = train_walk_forward(fs, cfg, save_dir=str(out))

    res.oos_predictions.to_parquet(_p(cfg, "oos_predictions.parquet"))
    res.oos_actuals.to_parquet(_p(cfg, "oos_actuals.parquet"))

    print("\n=== OUT-OF-SAMPLE ===")
    for k, v in res.metrics.items():
        print(f"  {k:26s} {round(v, 6) if isinstance(v, float) else v}")
    print("\n=== FOLDS ===")
    cols = ["n_train", "n_test", "rmse", "ic", "rank_ic", "directional_accuracy"]
    print(res.fold_metrics[[c for c in cols if c in res.fold_metrics.columns]].round(5).to_string())
    if res.feature_importance is not None:
        print("\n=== TOP FEATURES ===")
        print(res.feature_importance["mean"].head(20).round(6).to_string())
    print(f"\nbundle -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# backtest
# --------------------------------------------------------------------------- #
def cmd_backtest(args: argparse.Namespace) -> int:
    from .backtest.costs import CostModel
    from .backtest.engine import buy_and_hold, run_backtest
    from .backtest.report import tearsheet
    from .data.universe import universe_panel
    from .signals.alpha import predictions_to_signal, signal_decay

    cfg = _cfg_from_args(args)
    if args.start:
        cfg.backtest.start_date = args.start
    if args.end:
        cfg.backtest.end_date = args.end
    if args.no_costs:
        cfg.backtest.include_costs = False

    preds = _require(_p(cfg, "oos_predictions.parquet"), "tqe train")
    _, returns = _load_returns(cfg)
    tr = universe_panel(returns, "total_return")
    dv = universe_panel(returns, "dv01")
    yc = universe_panel(returns, "yield_change")

    signal = predictions_to_signal(
        preds, method=args.signal_method, window=args.signal_window,
        clip=cfg.portfolio.signal_clip, min_abs=cfg.portfolio.min_signal_to_trade,
    )
    halflife = args.decay if args.decay and args.decay > 0 else cfg.portfolio.signal_halflife
    if halflife and halflife > 0:
        signal = signal_decay(signal, halflife=halflife)

    bench = buy_and_hold(tr, cfg.backtest.benchmark, signal.index)
    result = run_backtest(
        signal, tr, dv, cfg, CostModel(cfg.costs), benchmark=bench,
        yield_change_panel=yc, n_trials=args.n_trials,
    )
    print("\n" + result.summary())

    out = Path(args.out or Path(cfg.backtest.output_dir) / "latest")
    tearsheet(result, out, title="Treasury Quant Engine - Backtest", plots=not args.no_plots)
    print(f"\nreport -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# predict / trade / serve
# --------------------------------------------------------------------------- #
def cmd_attribute(args: argparse.Namespace) -> int:
    """Decompose the latest backtest's P&L by curve factor and by source."""
    from .backtest.attribution import attribute_by_factor, attribute_by_source
    from .data.universe import universe_panel

    cfg = _cfg_from_args(args)
    d = Path(args.backtest_dir or Path(cfg.backtest.output_dir) / "latest")
    pos_path = d / "positions.parquet"
    if not pos_path.exists():
        raise SystemExit(f"No backtest at {d}. Run `tqe backtest` first.")

    positions = pd.read_parquet(pos_path)
    _, returns = _load_returns(cfg)
    yc = universe_panel(returns, "yield_change")
    dv = universe_panel(returns, "dv01")

    capital = cfg.backtest.initial_capital
    years = max(len(positions) / 252.0, 1e-9)

    fa = attribute_by_factor(positions, yc, dv)
    print("=== BY CURVE FACTOR (annualised, % of capital) ===")
    print(f"  {'factor':<12} {'P&L':>10} {'share of gross risk':>21}")
    for c in fa.contributions.columns:
        pnl = fa.contributions[c].sum() / capital / years
        print(f"  {c:<12} {pnl:>9.3%} {fa.factor_share[c]:>21.1%}")
    print(f"\n  three factors explain {fa.explained_ratio:.1%} of P&L variance")

    # Reuse the tested source decomposition rather than re-deriving it here.
    # Reconstructing a minimal result object is cheap; duplicating P&L logic is
    # how this project acquired five separate financing bugs.
    rp = d / "returns.parquet"
    if rp.exists():
        from types import SimpleNamespace

        rets = pd.read_parquet(rp)["returns"]
        costs = (pd.read_parquet(d / "costs.parquet")["costs"]
                 if (d / "costs.parquet").exists() else pd.Series(0.0, index=rets.index))
        fin = (pd.read_parquet(d / "financing.parquet")["financing"]
               if (d / "financing.parquet").exists() else pd.Series(0.0, index=rets.index))
        gross = rets + (costs + fin) / capital

        shim = SimpleNamespace(
            returns=rets, gross_returns=gross, costs=costs, financing=fin,
            config={"backtest": {"initial_capital": capital}},
        )
        src = attribute_by_source(shim)
        print("\n=== BY SOURCE (annualised, % of capital) ===")
        for col in ("market_pnl", "financing", "costs"):
            print(f"  {col:<14} {src[col].sum() / capital / years:>9.3%}")
        print(f"  {'-' * 24}")
        print(f"  {'net':<14} {src['net'].sum() / capital / years:>9.3%}")
        err = float((src["net"] - rets * capital).abs().max())
        print(f"\n  reconciles with the engine's own P&L to {err:.2e}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    """Nested walk-forward hyper-parameter search."""
    import json as _json

    from .features.builder import FeatureSet
    from .training.tune import DEFAULT_GRIDS, nested_walk_forward

    cfg = _cfg_from_args(args)
    X = _require(_p(cfg, "X.parquet"), "tqe features build")
    y = _require(_p(cfg, "y.parquet"), "tqe features build")
    meta_path = cfg.processed_dir / "feature_meta.json"
    meta = _json.loads(meta_path.read_text()) if meta_path.exists() else {}
    fs = FeatureSet(X=X, y=y, metadata=meta)

    grid = DEFAULT_GRIDS.get(args.model)
    print(f"nested walk-forward: {args.model}, grid {grid}")
    print("  inner folds select the hyper-parameters; outer folds score the procedure")
    res = nested_walk_forward(fs, args.model, grid, cfg, inner_splits=args.inner_splits,
                              metric=args.metric)
    print(f"\n{res.summary()}")
    print("\n=== SELECTED PER OUTER FOLD ===")
    print(res.leaderboard.to_string(index=False))
    print("\n=== HONEST OUT-OF-SAMPLE METRICS ===")
    for k, v in res.outer_metrics.items():
        print(f"  {k:24s} {round(v, 6) if isinstance(v, float) else v}")
    print(f"\n  {res.n_candidates} configurations evaluated in total - pass this to")
    print("  `tqe backtest --n-trials` so the deflated Sharpe accounts for the search.")
    return 0


def cmd_charts(args: argparse.Namespace) -> int:
    """Render the diagnostic chart set."""
    from .curve.pca import fit_curve_pca, rolling_pca_factors
    from .viz import plot_all

    cfg = _cfg_from_args(args)
    curve = _require(_p(cfg, "curve.parquet"), "tqe data pull")

    def opt(name):
        path = _p(cfg, f"{name}.parquet")
        return pd.read_parquet(path) if path.exists() else None

    nss = opt("nss")
    core = [c for c in cfg.data.core_tenors if c in curve.columns]
    changes = curve[core].diff().dropna(how="any")
    pca = fit_curve_pca(changes, cfg.curve.n_pca_factors)
    factors = rolling_pca_factors(changes, window=252, n_factors=cfg.curve.n_pca_factors)

    out = Path(args.out or cfg.artifacts("reports", "charts"))
    paths = plot_all(out, curve=curve, nss=nss, pca=pca, factors=factors,
                     capital=cfg.backtest.initial_capital)
    for path in paths:
        print(f"  wrote {path}")
    if not paths:
        print("  no charts written (matplotlib missing?)")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .models.registry import latest_bundle, load_bundle

    cfg = _cfg_from_args(args)
    bundle_path = Path(args.model) if args.model else latest_bundle(cfg.training.artifacts_dir)
    if bundle_path is None:
        raise SystemExit("No model bundle found. Run `tqe train` first.")
    model, scaler, meta = load_bundle(bundle_path)

    from .models.registry import align_to_schema

    X = align_to_schema(
        _require(_p(cfg, "X.parquet"), "tqe features build"),
        meta.get("feature_names") or model.feature_names_,
    )
    row = X.loc[[pd.Timestamp(args.date)]] if args.date else X.tail(1)

    Xs = row
    if scaler is not None:
        Xs = pd.DataFrame(
            np.nan_to_num(scaler.transform(row.to_numpy(dtype=float))),
            index=row.index, columns=row.columns,
        )
    pred = model.predict_frame(Xs)
    asof = row.index[-1].date()

    # The raw ensemble output is heavily shrunk (its NNLS weights sum to ~0.003),
    # so printing it alone shows a column of zeros. What actually drives trading
    # is the standardised signal, so compute it over recent history and show both.
    from .data.universe import universe_panel
    from .signals.alpha import predictions_to_signal, scale_to_return_units

    hist = X.tail(min(len(X), 512))
    hist_s = hist
    if scaler is not None:
        hist_s = pd.DataFrame(
            np.nan_to_num(scaler.transform(hist.to_numpy(dtype=float))),
            index=hist.index, columns=hist.columns,
        )
    pred_hist = model.predict_frame(hist_s)
    signal = predictions_to_signal(
        pred_hist, "vol_scale", 252, cfg.portfolio.signal_clip, cfg.portfolio.min_signal_to_trade
    )

    _, returns = _load_returns(cfg)
    tr = universe_panel(returns, "total_return").reindex(pred_hist.index)
    mu = scale_to_return_units(pred_hist, tr, ic=0.05, window=252)

    print(f"model      {bundle_path}")
    print(f"features   as of {asof}  (forecasting the next session)")
    print(f"trained    {meta.get('saved_at', 'unknown')}  git {meta.get('git_sha', '?')}")
    print()
    print(f"  {'tenor':<8} {'raw':>12} {'expected':>11} {'signal':>8}  view")
    print(f"  {'':<8} {'':>12} {'return (bp)':>11} {'':>8}")
    print("  " + "-" * 52)
    latest_sig = signal.iloc[-1] if len(signal) else pred.iloc[0] * 0
    latest_mu = mu.iloc[-1] if len(mu) else pred.iloc[0] * 0
    for tenor in pred.columns:
        raw = float(pred.iloc[0][tenor])
        e_ret = float(latest_mu.get(tenor, float("nan"))) * 1e4
        sig = float(latest_sig.get(tenor, float("nan")))
        if not np.isfinite(sig) or abs(sig) < cfg.portfolio.min_signal_to_trade:
            view = "flat"
        else:
            view = "LONG " + "+" * min(int(abs(sig)) + 1, 3) if sig > 0 else \
                   "SHORT " + "-" * min(int(abs(sig)) + 1, 3)
        print(f"  {tenor:<8} {raw:12.3e} {e_ret:+11.3f} {sig:+8.2f}  {view}")

    active = int((latest_sig.abs() >= cfg.portfolio.min_signal_to_trade).sum())
    print(f"\n  {active}/{len(pred.columns)} tenors above the {cfg.portfolio.min_signal_to_trade} "
          f"trading threshold")
    print("  (raw = ensemble output; expected return rescales it to real return units)")
    return 0


def cmd_trade(args: argparse.Namespace) -> int:
    from .live.runner import LiveRunner

    cfg = _cfg_from_args(args)
    if args.live and not args.yes:
        raise SystemExit(
            "Refusing to trade live without --yes. This sends real orders to the "
            "configured broker. Re-run with:  tqe trade --live --yes"
        )
    runner = LiveRunner(cfg)
    out = runner.run_once(as_of=args.date, dry_run=not args.live)
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    d = Path(args.backtest_dir or Path(cfg.backtest.output_dir) / "latest")
    summary = d / "summary.txt"
    if not summary.exists():
        raise SystemExit(f"No backtest at {d}. Run `tqe backtest` first.")
    print(summary.read_text())
    tear = d / "tearsheet.md"
    if tear.exists():
        print(f"\nfull tearsheet: {tear}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("uvicorn is not installed. pip install 'tqe[api]'") from None
    from .api.server import create_app

    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level=cfg.log_level.lower())
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Run the whole thing end to end."""
    for name, fn, extra in [
        ("data pull", cmd_data_pull, {"force": False, "curve_only": False, "start": None, "end": None}),
        ("curve fit", cmd_curve_fit, {"date": None}),
        ("features build", cmd_features_build, {}),
        ("train", cmd_train, {"learners": None, "out": None}),
        ("backtest", cmd_backtest, {
            "start": None, "end": None, "no_costs": False, "out": None,
            "signal_method": "vol_scale", "signal_window": 252, "decay": 0.0,
            "n_trials": 1, "no_plots": False,
        }),
    ]:
        print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")
        ns = argparse.Namespace(**{**vars(args), **extra})
        rc = fn(ns)
        if rc != 0:
            return rc
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tqe",
        description="Treasury Quant Engine - US bond prediction, backtesting and execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", help="path to a YAML config (default: configs/default.yaml)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override a config key, e.g. --set portfolio.capital=5e6")
    p.add_argument("--log-level", default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="command", required=True)

    # data
    d = sub.add_parser("data", help="download and inspect market data").add_subparsers(
        dest="subcommand", required=True)
    dp = d.add_parser("pull", help="download Treasury + FRED history")
    dp.add_argument("--force", action="store_true", help="ignore the cache")
    dp.add_argument("--curve-only", action="store_true", help="skip the FRED macro bundle")
    dp.add_argument("--start", type=int, help="first calendar year")
    dp.add_argument("--end", type=int, help="last calendar year")
    dp.set_defaults(func=cmd_data_pull)
    d.add_parser("status", help="what is cached on disk").set_defaults(func=cmd_data_status)

    # curve
    c = sub.add_parser("curve", help="yield-curve modelling").add_subparsers(
        dest="subcommand", required=True)
    cf = c.add_parser("fit", help="fit NSS, bootstrap zeros, compute PCA factors")
    cf.add_argument("--date", help="fit a single date and print the detail")
    cf.set_defaults(func=cmd_curve_fit)

    # features
    f = sub.add_parser("features", help="feature engineering").add_subparsers(
        dest="subcommand", required=True)
    f.add_parser("build", help="assemble the design matrix").set_defaults(func=cmd_features_build)

    # train
    t = sub.add_parser("train", help="walk-forward training")
    t.add_argument("--learners", help="comma-separated learner names")
    t.add_argument("--out", help="bundle output directory")
    t.set_defaults(func=cmd_train)

    # backtest
    b = sub.add_parser("backtest", help="simulate the strategy with costs")
    b.add_argument("--start")
    b.add_argument("--end")
    b.add_argument("--no-costs", action="store_true", help="run gross of transaction costs")
    b.add_argument("--out", help="report output directory")
    b.add_argument("--signal-method", default="vol_scale", choices=["zscore", "vol_scale", "rank", "raw"])
    b.add_argument("--signal-window", type=int, default=252)
    b.add_argument("--decay", type=float, default=0.0, help="signal smoothing half-life in days")
    b.add_argument("--n-trials", type=int, default=1,
                   help="configurations searched, for the deflated Sharpe ratio")
    b.add_argument("--no-plots", action="store_true")
    b.set_defaults(func=cmd_backtest)

    # attribute / tune / charts
    at = sub.add_parser("attribute", help="decompose the backtest P&L")
    at.add_argument("--backtest-dir")
    at.set_defaults(func=cmd_attribute)

    tu = sub.add_parser("tune", help="nested walk-forward hyper-parameter search")
    tu.add_argument("--model", default="ridge")
    tu.add_argument("--inner-splits", type=int, default=3)
    tu.add_argument("--metric", default="ic", choices=["ic", "rank_ic", "rmse", "r2"])
    tu.set_defaults(func=cmd_tune)

    ch = sub.add_parser("charts", help="render diagnostic charts")
    ch.add_argument("--out")
    ch.set_defaults(func=cmd_charts)

    # predict
    pr = sub.add_parser("predict", help="forecast the next session")
    pr.add_argument("--date", help="as-of date (default: the latest features row)")
    pr.add_argument("--model", help="path to a model bundle")
    pr.set_defaults(func=cmd_predict)

    # trade
    tr = sub.add_parser("trade", help="run one live/paper trading session")
    tr.add_argument("--date")
    tr.add_argument("--live", action="store_true", help="actually send orders")
    tr.add_argument("--dry-run", action="store_true", default=True)
    tr.add_argument("--yes", action="store_true", help="confirm live trading")
    tr.set_defaults(func=cmd_trade)

    # report / serve / pipeline
    rp = sub.add_parser("report", help="print the latest backtest summary")
    rp.add_argument("--backtest-dir")
    rp.set_defaults(func=cmd_report)

    sv = sub.add_parser("serve", help="run the FastAPI service")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)

    sub.add_parser("pipeline", help="data -> curve -> features -> train -> backtest").set_defaults(
        func=cmd_pipeline)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level or "INFO", log_file="logs/tqe.log")
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - top level, report and exit non-zero
        log.error("%s: %s", type(exc).__name__, exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
