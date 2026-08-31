"""Tree-ensemble learners.

Trees earn their place here for one specific reason: **interactions**. The
linear models cannot express "momentum works, but only when realised volatility
is in its bottom tercile and the curve is not inverted", and that kind of
state-dependence is exactly what the regime features were built to expose.

They also overfit ferociously on this data if left unconstrained, so the
defaults are deliberately timid - shallow trees, a small learning rate, heavy
L2, and early stopping against an internal validation tail. The training
harness's walk-forward split is the real defence, but a model that needs the
outer split to save it is a model that will disappoint out of sample.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .base import BaseModel

log = get_logger("models.trees")

__all__ = ["RandomForestModel", "GBMModel", "ExtraTreesModel"]


class _MultiOutputTrees(BaseModel):
    """Base for tree learners, handling the one-model-per-target case."""

    def _make_estimator(self):  # pragma: no cover - overridden
        raise NotImplementedError

    #: True when the underlying estimator handles multi-output natively.
    native_multioutput: bool = False

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.native_multioutput or y.shape[1] == 1:
            self.estimator_ = self._make_estimator()
            self.estimator_.fit(X, y if y.shape[1] > 1 else y.ravel())
            self.estimators_ = None
        else:
            # Boosting is single-output in sklearn, so fit one per tenor. The
            # tenors are ~95% correlated, so this is mildly wasteful but keeps
            # each forecast independent and easy to reason about.
            self.estimator_ = None
            self.estimators_ = []
            for j in range(y.shape[1]):
                est = self._make_estimator()
                est.fit(X, y[:, j])
                self.estimators_.append(est)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self.estimators_ is not None:
            return np.column_stack([e.predict(X) for e in self.estimators_])
        pred = self.estimator_.predict(X)
        return pred if np.ndim(pred) == 2 else np.asarray(pred).reshape(-1, 1)

    @property
    def feature_importance(self) -> pd.Series | None:
        ests = self.estimators_ or ([self.estimator_] if getattr(self, "estimator_", None) else [])
        vals = [e.feature_importances_ for e in ests if hasattr(e, "feature_importances_")]
        if not vals:
            return None
        imp = np.mean(np.vstack(vals), axis=0)
        names = self.feature_names_ or [f"f{i}" for i in range(len(imp))]
        return pd.Series(imp, index=names).sort_values(ascending=False)


class RandomForestModel(_MultiOutputTrees):
    """Bagged decision trees.

    Natively multi-output, and its variance-reduction-by-averaging is a good fit
    for a very noisy target. ``max_features`` is kept low so individual trees
    decorrelate despite the heavily redundant feature set.
    """

    name = "random_forest"
    native_multioutput = True

    def __init__(self, n_estimators: int = 400, max_depth: int = 8, min_samples_leaf: int = 20,
                 max_features: float | str = 0.3, random_state: int = 42, n_jobs: int = -1,
                 **kw: Any) -> None:
        super().__init__(n_estimators=n_estimators, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf, max_features=max_features,
                         random_state=random_state, n_jobs=n_jobs, **kw)

    def _make_estimator(self):
        from sklearn.ensemble import RandomForestRegressor

        p = self.params
        return RandomForestRegressor(
            n_estimators=int(p["n_estimators"]),
            max_depth=int(p["max_depth"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            max_features=p["max_features"],
            random_state=int(p["random_state"]),
            n_jobs=int(p["n_jobs"]),
        )

    @property
    def feature_importance(self) -> pd.Series | None:
        est = getattr(self, "estimator_", None)
        if est is None or not hasattr(est, "feature_importances_"):
            return super().feature_importance
        names = self.feature_names_ or [f"f{i}" for i in range(len(est.feature_importances_))]
        return pd.Series(est.feature_importances_, index=names).sort_values(ascending=False)


class ExtraTreesModel(RandomForestModel):
    """Extremely randomised trees - more variance reduction, less fitting power."""

    name = "extra_trees"

    def _make_estimator(self):
        from sklearn.ensemble import ExtraTreesRegressor

        p = self.params
        return ExtraTreesRegressor(
            n_estimators=int(p["n_estimators"]),
            max_depth=int(p["max_depth"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            max_features=p["max_features"],
            random_state=int(p["random_state"]),
            n_jobs=int(p["n_jobs"]),
        )


class GBMModel(_MultiOutputTrees):
    """Histogram-based gradient boosting.

    ``HistGradientBoostingRegressor`` is used rather than the classic
    ``GradientBoostingRegressor`` because it is roughly an order of magnitude
    faster on this many features and handles NaN natively - which keeps the door
    open to feeding it the late-starting macro series that the linear models
    cannot accept.

    Early stopping validates on the **final** ``validation_fraction`` of the
    training block. Since the data arrives in date order and the harness never
    shuffles, that tail is the most recent stretch of the training window - a
    chronologically sensible validation set, not a random sample from the middle
    of history.
    """

    name = "gbm"
    native_multioutput = False

    def __init__(self, max_iter: int = 400, learning_rate: float = 0.03, max_depth: int = 4,
                 l2_regularization: float = 1.0, min_samples_leaf: int = 40,
                 max_leaf_nodes: int = 15, early_stopping: bool = True,
                 validation_fraction: float = 0.15, n_iter_no_change: int = 25,
                 random_state: int = 42, **kw: Any) -> None:
        super().__init__(max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth,
                         l2_regularization=l2_regularization, min_samples_leaf=min_samples_leaf,
                         max_leaf_nodes=max_leaf_nodes, early_stopping=early_stopping,
                         validation_fraction=validation_fraction,
                         n_iter_no_change=n_iter_no_change, random_state=random_state, **kw)

    def _make_estimator(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        p = self.params
        return HistGradientBoostingRegressor(
            max_iter=int(p["max_iter"]),
            learning_rate=float(p["learning_rate"]),
            max_depth=int(p["max_depth"]),
            l2_regularization=float(p["l2_regularization"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            max_leaf_nodes=int(p["max_leaf_nodes"]),
            early_stopping=bool(p["early_stopping"]),
            validation_fraction=float(p["validation_fraction"]),
            n_iter_no_change=int(p["n_iter_no_change"]),
            random_state=int(p["random_state"]),
        )

    @property
    def feature_importance(self) -> pd.Series | None:
        """Permutation importance is not computed here - it is expensive and
        would need a held-out block. ``HistGradientBoosting`` exposes no native
        ``feature_importances_``, so this returns ``None`` and the training
        harness falls back to the linear models' coefficients for the report.
        """
        return None
