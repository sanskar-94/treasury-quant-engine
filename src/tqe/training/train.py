"""Walk-forward training harness.

Produces the one number the whole project exists to justify: a set of
**out-of-sample** predictions covering many years, where every prediction was
made by a model that had never seen the day it was predicting, nor anything
after it.

The loop is deliberately boring:

    for each fold:
        fit scaler on train only
        fit model  on train only
        predict the test block
        record

Two details are load-bearing and easy to get wrong:

* **The scaler is refitted per fold on the training block only.** Standardising
  the whole sample first leaks the future mean and variance into every row. It
  is a small leak, but it is a leak, and it is the single most common one in
  published backtests.
* **Predictions are stored, not scores.** Fold-level metrics are computed from
  the stored predictions afterwards, so the backtest consumes exactly the series
  that was evaluated - no possibility of the reported metric and the traded
  signal diverging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..features.builder import FeatureSet
from ..logging_utils import get_logger
from ..models.base import BaseModel
from ..models.registry import build_ensemble, save_bundle
from .metrics import regression_metrics
from .splits import Split, describe_splits, validate_splits, walk_forward_splits

log = get_logger("training.train")

__all__ = ["TrainResult", "train_walk_forward", "train_final_model", "fit_scaler"]


@dataclass
class TrainResult:
    """Everything a walk-forward run produced."""

    model: BaseModel | None
    scaler: Any
    metrics: dict[str, float] = field(default_factory=dict)
    fold_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    oos_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    oos_actuals: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_importance: pd.DataFrame | None = None
    splits: list[Split] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        return (
            f"OOS {len(self.oos_predictions)} rows | "
            f"RMSE {m.get('rmse', float('nan')):.6g} | "
            f"IC {m.get('ic', float('nan')):+.4f} | "
            f"rank-IC {m.get('rank_ic', float('nan')):+.4f} | "
            f"dir-acc {m.get('directional_accuracy', float('nan')):.4f}"
        )

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if not self.oos_predictions.empty:
            self.oos_predictions.to_parquet(out / "oos_predictions.parquet")
            self.oos_actuals.to_parquet(out / "oos_actuals.parquet")
        if not self.fold_metrics.empty:
            self.fold_metrics.to_csv(out / "fold_metrics.csv")
        if self.feature_importance is not None:
            self.feature_importance.to_csv(out / "feature_importance.csv")
        if self.model is not None:
            save_bundle(self.model, self.scaler, {**self.config, "metrics": self.metrics}, out)
        return out


def fit_scaler(X: pd.DataFrame, standardize: bool = True):
    """Fit a feature scaler on a training block.

    Uses ``RobustScaler`` (median / IQR) rather than ``StandardScaler``. Bond
    return features have fat tails - a single 2020 observation can be twenty
    standard deviations out - and a mean/variance scaler lets that one day
    dominate the scaling of every feature it touches.
    """
    if not standardize:
        return None
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    scaler.fit(X.to_numpy(dtype=float))
    return scaler


def _apply(scaler, X: pd.DataFrame) -> pd.DataFrame:
    if scaler is None:
        return X
    arr = scaler.transform(X.to_numpy(dtype=float))
    # A feature that was constant in the training block yields 0/0; make it 0.
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(arr, index=X.index, columns=X.columns)


def train_walk_forward(
    fs: FeatureSet,
    cfg: Config | None = None,
    model_factory=None,
    save_dir: str | Path | None = None,
) -> TrainResult:
    """Run the full walk-forward evaluation.

    Parameters
    ----------
    fs:
        Aligned features and targets from :func:`tqe.features.build_features`.
    cfg:
        Configuration; ``cfg.training`` drives the split scheme and
        ``cfg.model`` the learners.
    model_factory:
        Callable returning a fresh unfitted :class:`BaseModel`. Defaults to
        :func:`tqe.models.registry.build_ensemble` on the configured learners.
    save_dir:
        If given, the result (including a deployable bundle refitted on all
        data) is written here.

    Returns
    -------
    TrainResult
    """
    cfg = cfg or Config()
    tc, mc = cfg.training, cfg.model
    factory = model_factory or (lambda: build_ensemble(mc))

    splits = walk_forward_splits(
        fs.index,
        n_splits=tc.n_splits,
        test_size=tc.test_size,
        min_train_size=tc.min_train_size,
        embargo=tc.embargo,
        expanding=tc.expanding,
        horizon=mc.horizon,
    )
    if not splits:
        raise ValueError(
            f"No walk-forward folds for {len(fs)} observations "
            f"(min_train_size={tc.min_train_size}, test_size={tc.test_size})"
        )

    audit = validate_splits(splits, horizon=mc.horizon, embargo=tc.embargo)
    if not audit["ok"]:
        raise RuntimeError(f"Split scheme failed its leakage audit: {audit['violations']}")
    log.info("walk-forward: %d folds, split audit clean", len(splits))

    pred_blocks: list[pd.DataFrame] = []
    actual_blocks: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    importances: list[pd.Series] = []
    last_model: BaseModel | None = None
    last_scaler = None

    for s in splits:
        t0 = time.time()
        Xtr, ytr = fs.X.iloc[s.train_idx], fs.y.iloc[s.train_idx]
        Xte, yte = fs.X.iloc[s.test_idx], fs.y.iloc[s.test_idx]

        scaler = fit_scaler(Xtr, tc.standardize)
        Xtr_s, Xte_s = _apply(scaler, Xtr), _apply(scaler, Xte)

        model = factory()
        model.fit(Xtr_s, ytr)
        pred = pd.DataFrame(model.predict(Xte_s), index=Xte.index, columns=ytr.columns)

        pred_blocks.append(pred)
        actual_blocks.append(yte)
        last_model, last_scaler = model, scaler

        fold_m = regression_metrics(yte.to_numpy().ravel(), pred.to_numpy().ravel())
        fold_m.update(
            {
                "fold": s.fold,
                "n_train": len(s.train_idx),
                "n_test": len(s.test_idx),
                "test_start": s.test_start,
                "test_end": s.test_end,
                "fit_seconds": round(time.time() - t0, 2),
            }
        )
        rows.append(fold_m)
        log.info(
            "fold %d  %s..%s  train=%d  RMSE=%.3e  IC=%+.4f  dir=%.3f  (%.1fs)",
            s.fold, s.test_start.date(), s.test_end.date(), len(s.train_idx),
            fold_m["rmse"], fold_m["ic"], fold_m["directional_accuracy"], fold_m["fit_seconds"],
        )

        imp = model.feature_importance
        if imp is not None:
            importances.append(imp.rename(f"fold{s.fold}"))

    oos_pred = pd.concat(pred_blocks).sort_index()
    oos_act = pd.concat(actual_blocks).sort_index()
    overall = regression_metrics(oos_act.to_numpy().ravel(), oos_pred.to_numpy().ravel())

    # Per-tenor breakdown - a headline IC can hide one tenor doing all the work.
    per_tenor = {}
    for col in oos_pred.columns:
        per_tenor[col] = regression_metrics(oos_act[col].to_numpy(), oos_pred[col].to_numpy())
    overall["per_tenor_ic"] = {k: round(v["ic"], 5) for k, v in per_tenor.items()}

    fold_df = pd.DataFrame(rows).set_index("fold") if rows else pd.DataFrame()
    imp_df = pd.concat(importances, axis=1) if importances else None
    if imp_df is not None:
        imp_df["mean"] = imp_df.mean(axis=1)
        imp_df = imp_df.sort_values("mean", ascending=False)

    result = TrainResult(
        model=last_model,
        scaler=last_scaler,
        metrics=overall,
        fold_metrics=fold_df,
        oos_predictions=oos_pred,
        oos_actuals=oos_act,
        feature_importance=imp_df,
        splits=splits,
        config={
            "model": mc.__dict__,
            "training": tc.__dict__,
            "features": fs.metadata,
            "split_audit": audit,
        },
    )
    log.info("walk-forward complete: %s", result.summary())

    if save_dir:
        # The deployable model is refitted on ALL data - walk-forward measures
        # the process, the final model is what actually trades.
        final = train_final_model(fs, cfg, model_factory)
        result.model, result.scaler = final.model, final.scaler
        result.save(save_dir)
    return result


def train_final_model(
    fs: FeatureSet,
    cfg: Config | None = None,
    model_factory=None,
) -> TrainResult:
    """Fit one model on the entire sample, for live prediction.

    This model has **no honest performance estimate** - it has seen everything.
    Its accuracy is the walk-forward number from :func:`train_walk_forward`, and
    that is the only figure that should ever be quoted.
    """
    cfg = cfg or Config()
    factory = model_factory or (lambda: build_ensemble(cfg.model))

    scaler = fit_scaler(fs.X, cfg.training.standardize)
    Xs = _apply(scaler, fs.X)
    model = factory()
    t0 = time.time()
    model.fit(Xs, fs.y)
    log.info("final model fitted on %d rows in %.1fs", len(fs), time.time() - t0)

    return TrainResult(
        model=model,
        scaler=scaler,
        metrics={},
        config={"model": cfg.model.__dict__, "training": cfg.training.__dict__,
                "features": fs.metadata, "note": "fitted on the full sample; not an OOS estimate"},
    )
