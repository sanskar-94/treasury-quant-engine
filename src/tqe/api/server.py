"""FastAPI service exposing the engine.

Endpoints
---------
``GET  /health``            liveness plus what is actually loaded
``GET  /curve/latest``      most recent par curve
``GET  /curve/fit``         NSS fit for a date, with fitted vs market yields
``GET  /curve/zero``        bootstrapped zero curve for a date
``POST /predict``           model forecast for a date
``GET  /signals``           standardised signals over a window
``GET  /portfolio``         target DV01 and notional for the latest session
``GET  /risk``              VaR, expected shortfall and the stress table
``GET  /backtest/summary``  metrics from the last saved backtest
``POST /trade/dry-run``     run a full session with orders suppressed

Data and the model are loaded once into application state at start-up: the
37-year curve, the NSS fits and a bundle deserialisation are far too slow to
repeat per request.

The service is read-only by design. ``/trade/dry-run`` exists so the decision
path can be inspected over HTTP, but it can never send an order - live trading
goes through the CLI, deliberately, so that it requires a human at a terminal.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..logging_utils import get_logger

log = get_logger("api.server")

__all__ = ["create_app"]


def _json_safe(obj: Any) -> Any:
    """Make numpy/pandas scalars JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (pd.Timestamp, date)):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


class _State:
    """Everything loaded once at start-up."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.curve: pd.DataFrame | None = None
        self.macro: pd.DataFrame | None = None
        self.nss: pd.DataFrame | None = None
        self.zero: pd.DataFrame | None = None
        self.X: pd.DataFrame | None = None
        self.returns: dict[str, pd.DataFrame] = {}
        self.model = None
        self.scaler = None
        self.model_meta: dict[str, Any] = {}
        self.errors: list[str] = []

    def load(self) -> None:
        p = self.cfg.processed_dir

        def opt(name):
            path = p / f"{name}.parquet"
            try:
                return pd.read_parquet(path) if path.exists() else None
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"{name}: {exc}")
                return None

        self.curve = opt("curve")
        self.macro = opt("macro")
        self.nss = opt("nss")
        self.zero = opt("zero")
        self.X = opt("X")

        if self.curve is not None:
            from ..data.universe import constant_maturity_total_return

            try:
                self.returns = constant_maturity_total_return(self.curve, self.cfg.data.core_tenors)
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"returns: {exc}")

        try:
            from ..models.registry import latest_bundle, load_bundle

            bundle = latest_bundle(self.cfg.training.artifacts_dir)
            if bundle is not None:
                self.model, self.scaler, self.model_meta = load_bundle(bundle)
                self.model_meta["bundle_path"] = str(bundle)
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"model: {exc}")

        log.info(
            "API state loaded: curve=%s model=%s errors=%d",
            None if self.curve is None else self.curve.shape,
            self.model_meta.get("model_name", "none"),
            len(self.errors),
        )

    def scale(self, rows: pd.DataFrame) -> pd.DataFrame:
        from ..models.registry import align_to_schema

        rows = align_to_schema(rows, self.model_meta.get("feature_names"))
        if self.scaler is None:
            return rows
        return pd.DataFrame(
            np.nan_to_num(self.scaler.transform(rows.to_numpy(dtype=float))),
            index=rows.index, columns=rows.columns,
        )


def create_app(cfg: Config | None = None):
    """Build the FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException, Query
    except ImportError as exc:  # pragma: no cover
        raise ImportError("FastAPI is not installed. pip install 'tqe[api]'") from exc

    cfg = cfg or load_config()
    state = _State(cfg)

    app = FastAPI(
        title="Treasury Quant Engine",
        description="US Treasury yield-curve modelling, return forecasting and execution",
        version="1.0.0",
    )

    @app.on_event("startup")
    def _startup() -> None:
        state.load()

    # ------------------------------------------------------------------ #
    @app.get("/health")
    def health() -> dict:
        return _json_safe({
            "status": "ok",
            "curve_rows": 0 if state.curve is None else len(state.curve),
            "curve_through": None if state.curve is None else state.curve.index.max(),
            "features_rows": 0 if state.X is None else len(state.X),
            "model": state.model_meta.get("model_name"),
            "model_bundle": state.model_meta.get("bundle_path"),
            "model_trained_at": state.model_meta.get("saved_at"),
            "git_sha": state.model_meta.get("git_sha"),
            "tenors": cfg.data.core_tenors,
            "load_errors": state.errors,
        })

    @app.get("/curve/latest")
    def curve_latest(n: int = Query(1, ge=1, le=250)) -> dict:
        if state.curve is None:
            raise HTTPException(503, "curve data not loaded")
        tail = state.curve.tail(n)
        return _json_safe({
            "as_of": tail.index[-1],
            "rows": [
                {"date": str(idx.date()), **{c: row[c] for c in tail.columns if pd.notna(row[c])}}
                for idx, row in tail.iterrows()
            ],
        })

    @app.get("/curve/fit")
    def curve_fit(as_of: str | None = None) -> dict:
        if state.curve is None:
            raise HTTPException(503, "curve data not loaded")
        from ..curve import fit_nss
        from ..data.sources import TENOR_YEARS

        try:
            row = (state.curve.loc[pd.Timestamp(as_of)] if as_of else state.curve.iloc[-1]).dropna()
        except KeyError:
            raise HTTPException(404, f"no curve observation for {as_of}") from None
        t = np.array([TENOR_YEARS[c] for c in row.index])
        params, rmse = fit_nss(t, row.to_numpy(), model=cfg.curve.model)
        return _json_safe({
            "as_of": as_of or state.curve.index[-1],
            "rmse_bp": rmse * 1e4,
            "params": {"beta0": params.beta0, "beta1": params.beta1, "beta2": params.beta2,
                       "beta3": params.beta3, "tau1": params.tau1, "tau2": params.tau2},
            "factors": {"level": params.level, "slope": params.slope,
                        "curvature": params.curvature},
            "fitted": {c: float(params.zero_rate(tv)) for c, tv in zip(row.index, t)},
            "market": {c: float(row[c]) for c in row.index},
        })

    @app.get("/curve/zero")
    def curve_zero(as_of: str | None = None) -> dict:
        if state.zero is None:
            raise HTTPException(503, "zero curve not built; run `tqe curve fit`")
        row = (state.zero.loc[pd.Timestamp(as_of)] if as_of else state.zero.iloc[-1]).dropna()
        return _json_safe({"as_of": as_of or state.zero.index[-1],
                           "zero_rates": {c: float(row[c]) for c in row.index}})

    @app.post("/predict")
    def predict(as_of: str | None = None) -> dict:
        if state.model is None or state.X is None:
            raise HTTPException(503, "model or features not loaded; run `tqe train`")
        rows = state.X.loc[[pd.Timestamp(as_of)]] if as_of else state.X.tail(1)
        if rows.empty:
            raise HTTPException(404, f"no feature row for {as_of}")
        pred = state.model.predict_frame(state.scale(rows)).iloc[0]
        return _json_safe({
            "features_as_of": rows.index[-1],
            "horizon_days": cfg.model.horizon,
            "target": cfg.model.target,
            "predictions_bp": {k: float(v) * 1e4 for k, v in pred.items()},
        })

    @app.get("/signals")
    def signals(window: int = Query(20, ge=1, le=500)) -> dict:
        if state.model is None or state.X is None:
            raise HTTPException(503, "model or features not loaded")
        from ..signals.alpha import predictions_to_signal

        hist = state.X.tail(max(300, window + 260))
        pred = state.model.predict_frame(state.scale(hist))
        sig = predictions_to_signal(pred, "zscore", 252, cfg.portfolio.signal_clip,
                                    cfg.portfolio.min_signal_to_trade).tail(window)
        return _json_safe({
            "as_of": sig.index[-1],
            "latest": {k: float(v) for k, v in sig.iloc[-1].items()},
            "history": [
                {"date": str(i.date()), **{c: float(r[c]) for c in sig.columns if pd.notna(r[c])}}
                for i, r in sig.iterrows()
            ],
        })

    @app.get("/portfolio")
    def portfolio() -> dict:
        if state.model is None or state.X is None or not state.returns:
            raise HTTPException(503, "model, features or returns not loaded")
        from ..data.universe import universe_panel
        from ..signals.alpha import predictions_to_signal
        from ..signals.sizing import size_portfolio

        hist = state.X.tail(512)
        pred = state.model.predict_frame(state.scale(hist))
        sig = predictions_to_signal(pred, "zscore", 252, cfg.portfolio.signal_clip,
                                    cfg.portfolio.min_signal_to_trade)
        tr = universe_panel(state.returns, "total_return").reindex(sig.index)
        dv = universe_panel(state.returns, "dv01").reindex(sig.index)
        yc = universe_panel(state.returns, "yield_change").reindex(sig.index)
        sized = size_portfolio(sig, tr, dv, cfg.portfolio, yc)
        dv01 = sized["target_dv01"].iloc[-1]
        notional = sized["notional"].iloc[-1]
        return _json_safe({
            "as_of": sig.index[-1],
            "capital": cfg.portfolio.capital,
            "target_dv01": {k: float(v) for k, v in dv01.items()},
            "target_notional": {k: float(v) for k, v in notional.items()},
            "gross_dv01": float(dv01.abs().sum()),
            "net_dv01": float(dv01.sum()),
            "gross_notional": float(notional.abs().sum()),
            "limits": {"max_gross_dv01": cfg.portfolio.max_gross_dv01,
                       "max_net_dv01": cfg.portfolio.max_net_dv01,
                       "max_leverage": cfg.portfolio.max_leverage},
        })

    @app.get("/risk")
    def risk() -> dict:
        if not state.returns:
            raise HTTPException(503, "returns not loaded")
        from ..data.universe import universe_panel
        from ..portfolio.risk import (
            apply_stress,
            covariance,
            expected_shortfall,
            historical_var,
            parametric_var,
            stress_scenarios,
        )

        tr = universe_panel(state.returns, "total_return").dropna(how="any").tail(756)
        w = pd.Series(1.0 / tr.shape[1], index=tr.columns)
        cov = covariance(tr, method="ewma")
        port = (tr * w).sum(axis=1)
        dv01 = {c: 1000.0 for c in tr.columns}
        return _json_safe({
            "as_of": tr.index[-1],
            "lookback_days": len(tr),
            "parametric_var_99": float(parametric_var(w, cov, 0.99)),
            "historical_var_99": float(historical_var(port, 0.99)),
            "expected_shortfall_99": float(expected_shortfall(port, 0.99)),
            "stress": {name: float(apply_stress(dv01, sc))
                       for name, sc in stress_scenarios().items()},
            "stress_note": "P&L in dollars for a book long $1,000 DV01 in every tenor",
        })

    @app.get("/backtest/summary")
    def backtest_summary() -> dict:
        import json as _json

        path = Path(cfg.backtest.output_dir) / "latest" / "metrics.json"
        if not path.exists():
            raise HTTPException(404, "no saved backtest; run `tqe backtest`")
        return _json_safe(_json.loads(path.read_text()))

    @app.post("/trade/dry-run")
    def trade_dry_run(as_of: str | None = None) -> dict:
        """Run a full session with execution suppressed.

        Hard-wired to ``dry_run=True``: this endpoint cannot place an order no
        matter what it is sent.
        """
        from ..live.runner import LiveRunner

        runner = LiveRunner(cfg)
        return _json_safe(runner.run_once(as_of=as_of, dry_run=True, refresh=False))

    return app


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
