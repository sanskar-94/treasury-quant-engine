"""Model interface.

Every learner in the system - a ridge regression, a gradient-boosted tree
ensemble, the stacked meta-model, even the trivial autoregressive baseline -
implements the same small protocol. That uniformity is what lets the training
harness, the backtest and the live runner treat "the model" as a black box and
swap learners from config without touching any other layer.

Two design choices worth stating:

**Multi-output by default.** The system forecasts one return per tenor, and the
tenors are ~95% correlated. Fitting them jointly (or at least in a single object)
keeps the prediction frame rectangular and makes portfolio construction simple.
Learners that cannot do multi-output natively fit one internal model per column.

**Models carry their own feature names.** A model reloaded six months later must
be able to reject a design matrix whose columns have drifted, rather than
silently predicting from misaligned data. :meth:`BaseModel.align` enforces that.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger("models.base")

__all__ = ["BaseModel", "to_array", "to_frame"]


def to_array(X: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Coerce a design matrix to a float array."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.to_numpy(dtype=float)
    return np.asarray(X, dtype=float)


def to_frame(values: np.ndarray, index, columns) -> pd.DataFrame:
    """Wrap predictions back into a labelled frame."""
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return pd.DataFrame(arr, index=index, columns=list(columns)[: arr.shape[1]])


class BaseModel(ABC):
    """Abstract learner."""

    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)
        self.feature_names_: list[str] | None = None
        self.target_names_: list[str] | None = None
        self.fitted_: bool = False

    # ------------------------------------------------------------------ #
    # Required interface
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit on plain arrays. Subclasses implement this, not :meth:`fit`."""

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict from a plain array, returning ``(n_samples, n_targets)``."""

    # ------------------------------------------------------------------ #
    # Public wrappers that handle labels and validation
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame | pd.Series) -> "BaseModel":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        if isinstance(y, pd.DataFrame):
            self.target_names_ = list(y.columns)
        elif isinstance(y, pd.Series):
            self.target_names_ = [y.name or "target"]

        Xa, ya = to_array(X), to_array(y)
        if ya.ndim == 1:
            ya = ya.reshape(-1, 1)
        if len(Xa) != len(ya):
            raise ValueError(f"X has {len(Xa)} rows but y has {len(ya)}")
        if len(Xa) == 0:
            raise ValueError("Cannot fit on an empty design matrix")
        if not np.isfinite(Xa).all():
            raise ValueError(f"{self.name}: design matrix contains non-finite values")

        self._fit(Xa, ya)
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError(f"{self.name} is not fitted")
        if isinstance(X, pd.DataFrame):
            X = self.align(X)
        pred = self._predict(to_array(X))
        return pred if pred.ndim == 2 else pred.reshape(-1, 1)

    def predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict, returning a frame indexed like ``X`` with target columns."""
        pred = self.predict(X)
        cols = self.target_names_ or [f"target{i}" for i in range(pred.shape[1])]
        return to_frame(pred, X.index, cols)

    def align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reorder / validate columns against the training schema.

        Raises if a training feature is missing. Extra columns are dropped with a
        warning rather than an error, so an upstream feature addition does not
        break inference on an existing model.
        """
        if self.feature_names_ is None:
            return X
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(
                f"{self.name}: {len(missing)} training features missing at predict time "
                f"(first few: {missing[:5]})"
            )
        extra = [c for c in X.columns if c not in self.feature_names_]
        if extra:
            log.debug("%s: dropping %d unseen columns at predict time", self.name, len(extra))
        return X[self.feature_names_]

    # ------------------------------------------------------------------ #
    # Introspection & persistence
    # ------------------------------------------------------------------ #
    @property
    def feature_importance(self) -> pd.Series | None:
        """Per-feature importance, if the learner exposes one."""
        return None

    def save(self, path: str | Path) -> Path:
        """Persist with joblib, alongside a human-readable JSON sidecar."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        meta = {
            "name": self.name,
            "class": type(self).__name__,
            "params": {k: v for k, v in self.params.items() if isinstance(v, (int, float, str, bool, type(None)))},
            "n_features": len(self.feature_names_ or []),
            "targets": self.target_names_,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        import joblib

        obj = joblib.load(Path(path))
        if not isinstance(obj, BaseModel):
            raise TypeError(f"{path} does not contain a BaseModel")
        return obj

    def __repr__(self) -> str:
        state = "fitted" if self.fitted_ else "unfitted"
        return f"{type(self).__name__}(name={self.name!r}, {state})"
