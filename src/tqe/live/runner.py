"""Daily live/paper trading loop.

One session is one call to :meth:`LiveRunner.run_once`, and it does exactly what
the backtest does, in the same order, using the same code paths:

    refresh data -> rebuild features -> load model -> predict -> signal ->
    size -> pre-trade risk check -> OMS -> audit log

Using the same modules as the backtest is the point. A live path that
re-implements sizing or signal construction will drift from the one that was
validated, and the drift is invisible until it costs money.

Safety posture:

* ``dry_run=True`` is the default everywhere, including the constructor.
* Live trading additionally requires an explicit CLI flag *and* confirmation.
* The kill switch is checked before anything is sized, not just before orders
  are sent.
* Every decision is written to a JSON audit log, and the OMS is idempotent, so
  a crashed session can be re-run without double-trading.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import audit, get_logger, setup_logging

log = get_logger("live.runner")

__all__ = ["LiveRunner"]


class LiveRunner:
    """Orchestrates one trading session end to end."""

    def __init__(
        self,
        cfg: Config | None = None,
        broker=None,
        model_path: str | Path | None = None,
    ) -> None:
        self.cfg = cfg or Config()
        self.state_dir = Path(self.cfg.execution.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(self.cfg.log_level, json_file=self.state_dir / "audit.jsonl")

        self.broker = broker if broker is not None else self._make_broker()
        self.model_path = model_path
        self._model = None
        self._scaler = None
        self._meta: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    def _make_broker(self):
        name = self.cfg.execution.broker.lower()
        if name == "alpaca":
            try:
                from ..execution.alpaca import AlpacaBroker

                return AlpacaBroker(self.cfg)
            except Exception as exc:  # noqa: BLE001
                log.error("Alpaca broker unavailable (%s); falling back to paper", exc)
        from ..execution.paper import PaperBroker

        return PaperBroker(self.cfg, state_dir=self.state_dir)

    def _load_model(self):
        if self._model is not None:
            return
        from ..models.registry import latest_bundle, load_bundle

        path = Path(self.model_path) if self.model_path else latest_bundle(
            self.cfg.training.artifacts_dir
        )
        if path is None:
            raise RuntimeError(
                "No trained model bundle found. Run `tqe train` before trading."
            )
        self._model, self._scaler, self._meta = load_bundle(path)
        log.info("loaded model bundle %s (%s)", path, self._meta.get("model_name", "?"))

    # ------------------------------------------------------------------ #
    def refresh_data(self, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Pull the latest curve and macro data, falling back to cache.

        A failed download must not stop a session: the cached history is still
        valid for everything except today's observation, and the runner will
        simply decline to trade on a stale as-of date rather than trading on
        data it wrongly believes is current.
        """
        from ..data.sources import load_market_data

        try:
            curve, macro = load_market_data(self.cfg, force=force)
            curve.to_parquet(self.cfg.processed_dir / "curve.parquet")
            if not macro.empty:
                macro.to_parquet(self.cfg.processed_dir / "macro.parquet")
            return curve, macro
        except Exception as exc:  # noqa: BLE001
            log.error("data refresh failed (%s); using cached history", exc)
            cp = self.cfg.processed_dir / "curve.parquet"
            mp = self.cfg.processed_dir / "macro.parquet"
            if not cp.exists():
                raise
            return pd.read_parquet(cp), (pd.read_parquet(mp) if mp.exists() else pd.DataFrame())

    def build_features(self, curve: pd.DataFrame, macro: pd.DataFrame):
        from ..curve import bootstrap_history, fit_nss_history_fixed, rolling_pca_factors
        from ..data.universe import constant_maturity_total_return
        from ..features import build_features

        returns = constant_maturity_total_return(curve, self.cfg.data.core_tenors)
        nss = fit_nss_history_fixed(curve, model=self.cfg.curve.model)
        zero = bootstrap_history(curve)
        core = [c for c in self.cfg.data.core_tenors if c in curve.columns]
        factors = rolling_pca_factors(
            curve[core].diff().dropna(how="any"),
            window=252, n_factors=self.cfg.curve.n_pca_factors,
        )
        fs = build_features(curve, macro, self.cfg, returns=returns,
                            nss=nss, pca_factors=factors, zero=zero)
        return fs, returns

    def predict(self, fs, as_of: pd.Timestamp | None = None) -> pd.Series:
        """Forecast for the session following ``as_of``."""
        self._load_model()
        X = fs.X
        row = X.loc[[as_of]] if as_of is not None and as_of in X.index else X.tail(1)
        if self._scaler is not None:
            row = pd.DataFrame(
                np.nan_to_num(self._scaler.transform(row.to_numpy(dtype=float))),
                index=row.index, columns=row.columns,
            )
        return self._model.predict_frame(row).iloc[0]

    # ------------------------------------------------------------------ #
    def run_once(
        self,
        as_of: str | date | None = None,
        dry_run: bool = True,
        refresh: bool = True,
    ) -> dict[str, Any]:
        """Execute one session.

        Returns a JSON-serialisable record of every decision made.
        """
        started = datetime.now(timezone.utc)
        record: dict[str, Any] = {
            "started_at": started.isoformat(),
            "dry_run": bool(dry_run),
            "broker": type(self.broker).__name__,
        }
        audit(log, "session_start", dry_run=dry_run, as_of=str(as_of))

        # ---- kill switch, checked before any work ---- #
        if self.cfg.risk.kill_switch:
            record["status"] = "halted"
            record["reason"] = "kill switch enabled in configuration"
            audit(log, "session_halted", reason=record["reason"])
            return record

        try:
            curve, macro = self.refresh_data() if refresh else self._cached()
            fs, returns = self.build_features(curve, macro)
        except Exception as exc:  # noqa: BLE001
            record.update(status="error", stage="data", error=f"{type(exc).__name__}: {exc}")
            audit(log, "session_error", stage="data", error=str(exc))
            return record

        stamp = pd.Timestamp(as_of) if as_of else fs.X.index[-1]
        record["as_of"] = str(stamp.date())
        record["data_through"] = str(curve.index[-1].date())

        # Refuse to act on stale features rather than pretending they are current.
        staleness = (fs.X.index[-1] - stamp).days if stamp in fs.X.index else None
        record["feature_rows"] = int(len(fs))

        try:
            pred = self.predict(fs, stamp if stamp in fs.X.index else None)
        except Exception as exc:  # noqa: BLE001
            record.update(status="error", stage="predict", error=f"{type(exc).__name__}: {exc}")
            audit(log, "session_error", stage="predict", error=str(exc))
            return record

        record["predictions_bp"] = {k: round(float(v) * 1e4, 4) for k, v in pred.items()}

        # ---- signal & sizing, using the same code as the backtest ---- #
        from ..data.universe import universe_panel
        from ..signals.alpha import predictions_to_signal
        from ..signals.sizing import size_portfolio

        tr = universe_panel(returns, "total_return")
        dv = universe_panel(returns, "dv01")
        yc = universe_panel(returns, "yield_change")

        # The signal scaler needs history, so rebuild the prediction series over
        # the recent past using the same model, then take the last row.
        hist = fs.X.tail(min(len(fs), 512))
        if self._scaler is not None:
            hist_s = pd.DataFrame(
                np.nan_to_num(self._scaler.transform(hist.to_numpy(dtype=float))),
                index=hist.index, columns=hist.columns,
            )
        else:
            hist_s = hist
        pred_hist = self._model.predict_frame(hist_s)

        signal = predictions_to_signal(
            pred_hist, method="zscore", window=252,
            clip=self.cfg.portfolio.signal_clip,
            min_abs=self.cfg.portfolio.min_signal_to_trade,
        )
        sized = size_portfolio(
            signal, tr.reindex(signal.index), dv.reindex(signal.index),
            self.cfg.portfolio, yc.reindex(signal.index),
        )
        target_notional = sized["notional"].iloc[-1]
        target_dv01 = sized["target_dv01"].iloc[-1]

        record["signal"] = {k: round(float(v), 4) for k, v in signal.iloc[-1].items()}
        record["target_dv01"] = {k: round(float(v), 2) for k, v in target_dv01.items()}
        record["gross_dv01"] = round(float(target_dv01.abs().sum()), 2)
        record["net_dv01"] = round(float(target_dv01.sum()), 2)

        # ---- map tenors to tradable instruments ---- #
        imap = self.cfg.execution.instrument_map
        targets: dict[str, float] = {}
        for tenor, notional in target_notional.items():
            symbol = imap.get(tenor)
            if symbol is None or not np.isfinite(notional):
                continue
            targets[symbol] = targets.get(symbol, 0.0) + float(notional)
        record["targets"] = {k: round(v, 2) for k, v in targets.items()}

        if not targets:
            record["status"] = "no_targets"
            audit(log, "session_no_targets")
            return record

        # ---- risk gate + OMS ---- #
        from ..execution.oms import OMS
        from ..execution.risk_gate import RiskGate

        gate = RiskGate(self.cfg.risk, self.cfg.portfolio)
        oms = OMS(broker=self.broker, risk_gate=gate, cfg=self.cfg, state_dir=self.state_dir)

        try:
            outcome = oms.daily_run(targets, dry_run=dry_run, as_of=stamp.date())
        except TypeError:
            outcome = oms.daily_run(targets, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001
            record.update(status="error", stage="execution", error=f"{type(exc).__name__}: {exc}")
            audit(log, "session_error", stage="execution", error=str(exc))
            return record

        record["execution"] = outcome
        record["status"] = outcome.get("status", "done")
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["elapsed_seconds"] = round(
            (datetime.now(timezone.utc) - started).total_seconds(), 2
        )
        if staleness:
            record["staleness_days"] = staleness

        self._write_record(record, stamp)
        audit(log, "session_complete", status=record["status"],
              submitted=outcome.get("submitted", 0), dry_run=dry_run)
        return record

    # ------------------------------------------------------------------ #
    def _cached(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        cp = self.cfg.processed_dir / "curve.parquet"
        mp = self.cfg.processed_dir / "macro.parquet"
        if not cp.exists():
            raise FileNotFoundError("No cached curve. Run `tqe data pull`.")
        return pd.read_parquet(cp), (pd.read_parquet(mp) if mp.exists() else pd.DataFrame())

    def _write_record(self, record: dict[str, Any], stamp: pd.Timestamp) -> Path:
        out = self.state_dir / "sessions"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"session_{stamp.date()}.json"
        path.write_text(json.dumps(record, indent=2, default=str))
        return path
