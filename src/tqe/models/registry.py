"""Model registry and artefact bundling.

The registry lets `configs/default.yaml` name learners as plain strings
(``learners: [ridge, gbm, ...]``) without the config layer importing scikit-learn
or knowing anything about model classes.

A saved *bundle* is the deployable unit: the fitted model, the fitted scaler,
the exact feature schema, and the metadata needed to reproduce the run. Live
trading loads a bundle and nothing else - if a prediction needs something that
is not in the bundle, that is a bug, because it means production depends on
state that was never versioned.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from ..config import ModelConfig
from ..logging_utils import get_logger
from .base import BaseModel
from .ensemble import AverageEnsemble, BestModelSelector, StackedEnsemble
from .linear import ARBaselineModel, ElasticNetModel, LassoModel, OLSModel, RidgeModel, ZeroModel
from .trees import ExtraTreesModel, GBMModel, RandomForestModel

log = get_logger("models.registry")

__all__ = [
    "MODEL_REGISTRY",
    "register",
    "create_model",
    "build_ensemble",
    "available_models",
    "save_bundle",
    "load_bundle",
]

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "ridge": RidgeModel,
    "elasticnet": ElasticNetModel,
    "lasso": LassoModel,
    "ols": OLSModel,
    "random_forest": RandomForestModel,
    "extra_trees": ExtraTreesModel,
    "gbm": GBMModel,
    "ar_baseline": ARBaselineModel,
    "zero": ZeroModel,
}


def register(name: str) -> Callable[[type[BaseModel]], type[BaseModel]]:
    """Decorator registering a learner under ``name``."""

    def wrap(cls: type[BaseModel]) -> type[BaseModel]:
        if name in MODEL_REGISTRY:
            log.warning("overwriting registered model %r", name)
        MODEL_REGISTRY[name] = cls
        return cls

    return wrap


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


def _params_for(name: str, cfg: ModelConfig) -> dict[str, Any]:
    """Map the flat :class:`ModelConfig` onto each learner's constructor."""
    if name == "ridge":
        return {"alphas": tuple(cfg.ridge_alphas)}
    if name in ("elasticnet", "lasso"):
        return {"random_state": cfg.random_state}
    if name in ("random_forest", "extra_trees"):
        return {
            "n_estimators": cfg.rf_n_estimators,
            "max_depth": cfg.rf_max_depth,
            "random_state": cfg.random_state,
        }
    if name == "gbm":
        return {
            "max_iter": cfg.gbm_max_iter,
            "learning_rate": cfg.gbm_learning_rate,
            "max_depth": cfg.gbm_max_depth,
            "l2_regularization": cfg.gbm_l2,
            "random_state": cfg.random_state,
        }
    return {}


def create_model(name: str, cfg: ModelConfig | None = None, **overrides: Any) -> BaseModel:
    """Instantiate a registered learner by name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {available_models()}")
    cfg = cfg or ModelConfig()
    params = _params_for(name, cfg)
    params.update(overrides)
    return MODEL_REGISTRY[name](**params)


def build_ensemble(cfg: ModelConfig | None = None, learners: Sequence[str] | None = None) -> BaseModel:
    """Construct the configured ensemble over the configured base learners.

    A single learner short-circuits to that learner rather than wrapping it, so
    ``learners: [ridge]`` really does train a bare ridge.
    """
    cfg = cfg or ModelConfig()
    names = list(learners) if learners else list(cfg.learners)
    if cfg.use_torch_lstm:
        try:
            from .lstm import LSTMModel  # noqa: F401

            if "lstm" not in names:
                names.append("lstm")
        except ImportError:
            log.warning("use_torch_lstm is set but PyTorch is not installed; skipping the LSTM")

    bases = [create_model(n, cfg) for n in names]
    if len(bases) == 1:
        return bases[0]

    mode = cfg.ensemble
    if mode == "average":
        return AverageEnsemble(bases)
    if mode == "best":
        return BestModelSelector(bases)
    if mode == "stacked":
        return StackedEnsemble(bases)
    raise ValueError(f"Unknown ensemble mode {mode!r}; use stacked | average | best")


# --------------------------------------------------------------------------- #
# Bundles
# --------------------------------------------------------------------------- #
def save_bundle(
    model: BaseModel,
    scaler: Any,
    metadata: dict[str, Any],
    path: str | Path,
) -> Path:
    """Write a deployable bundle directory.

    Layout::

        <path>/
          model.joblib      the fitted learner
          scaler.joblib     the fitted feature scaler (may be None)
          metadata.json     features, targets, config, git sha, timestamp

    Returns the directory.
    """
    import joblib

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    joblib.dump(scaler, out / "scaler.joblib")

    meta = dict(metadata)
    meta.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
    meta.setdefault("model_name", model.name)
    meta.setdefault("model_class", type(model).__name__)
    meta.setdefault("feature_names", model.feature_names_)
    meta.setdefault("target_names", model.target_names_)
    meta.setdefault("git_sha", _git_sha())
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    log.info("saved model bundle to %s", out)
    return out


def load_bundle(path: str | Path) -> tuple[BaseModel, Any, dict[str, Any]]:
    """Load ``(model, scaler, metadata)`` written by :func:`save_bundle`."""
    import joblib

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No model bundle at {p}")
    model = joblib.load(p / "model.joblib")
    scaler_path = p / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    meta_path = p / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, scaler, metadata


def latest_bundle(root: str | Path) -> Path | None:
    """Most recently modified bundle directory under ``root``."""
    r = Path(root)
    if not r.exists():
        return None
    candidates = [d for d in r.iterdir() if d.is_dir() and (d / "model.joblib").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _git_sha() -> str:
    """Current commit, so a saved model can be traced back to its code."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001 - absence of git is not an error
        return "unknown"
