"""Model combination.

Averaging several weak, differently-biased forecasts is close to free accuracy
when the signal is as faint as it is here - the errors are largely independent
while the (tiny) signal is shared, so the noise averages down faster than the
edge does.

The subtlety is in **stacking**. The meta-learner must be trained on base-model
predictions the base models did not see, otherwise it learns to trust whichever
learner overfits hardest. The standard fix is out-of-fold prediction, and for
time series it has to be an out-of-fold scheme that respects the arrow of time:
a shuffled K-fold would let the base models train on the future of the very rows
the meta-learner is scoring. :class:`StackedEnsemble` therefore builds its
out-of-fold matrix with forward-chaining splits and an embargo.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .base import BaseModel

log = get_logger("models.ensemble")

__all__ = ["AverageEnsemble", "StackedEnsemble", "BestModelSelector"]


class AverageEnsemble(BaseModel):
    """Weighted average of several fitted base models.

    With ``weights=None`` this is a plain equal-weight blend, which is a
    famously hard baseline to beat: estimating combination weights on noisy data
    usually costs more in estimation error than it gains in optimality.
    """

    name = "average"

    def __init__(self, models: Sequence[BaseModel], weights: Sequence[float] | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.models = list(models)
        if weights is None:
            self.weights = np.full(len(self.models), 1.0 / max(len(self.models), 1))
        else:
            w = np.asarray(weights, dtype=float)
            if len(w) != len(self.models):
                raise ValueError("weights and models must be the same length")
            total = w.sum()
            self.weights = w / total if total != 0 else np.full(len(w), 1.0 / len(w))

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for m in self.models:
            m.fit(X, y)
            m.feature_names_ = self.feature_names_
            m.target_names_ = self.target_names_

    def _predict(self, X: np.ndarray) -> np.ndarray:
        preds = [m.predict(X) for m in self.models]
        stack = np.stack(preds, axis=0)
        return np.tensordot(self.weights, stack, axes=(0, 0))

    @property
    def feature_importance(self) -> pd.Series | None:
        parts = [(w, m.feature_importance) for w, m in zip(self.weights, self.models)]
        parts = [(w, imp) for w, imp in parts if imp is not None]
        if not parts:
            return None
        total = sum(w for w, _ in parts)
        combined = sum((imp / imp.sum() * (w / total) for w, imp in parts if imp.sum() > 0))
        return combined.sort_values(ascending=False) if combined is not None else None


class StackedEnsemble(BaseModel):
    """Stacking with time-series-safe out-of-fold base predictions.

    Fitting proceeds in two passes:

    1. **Out-of-fold pass.** The training block is cut into ``n_folds``
       forward-chaining segments. For each fold the base models are fitted on
       everything strictly before it (minus an embargo) and predict the fold.
       The result is a matrix of predictions no base model has seen.
    2. **Refit pass.** The base models are refitted on the *full* training block
       so that inference uses all available data, and the meta-learner is fitted
       on the stage-1 out-of-fold matrix.

    The meta-learner is a non-negative least squares ridge: forecasts are
    combined, not contrasted. Allowing negative weights lets the meta-learner
    build a spurious long/short of two nearly-identical base models, which is a
    reliable way to manufacture in-sample performance that does not survive.
    """

    name = "stacked"

    def __init__(
        self,
        models: Sequence[BaseModel],
        meta_alpha: float = 1.0,
        n_folds: int = 5,
        embargo: int = 5,
        non_negative: bool = True,
        include_mean: bool = True,
        **kw: Any,
    ) -> None:
        super().__init__(meta_alpha=meta_alpha, n_folds=n_folds, embargo=embargo,
                         non_negative=non_negative, include_mean=include_mean, **kw)
        self.models = list(models)
        self.meta_weights_: np.ndarray | None = None
        self.oof_score_: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def _oof_predictions(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward-chaining out-of-fold predictions.

        Returns ``(oof, mask)`` where ``oof`` has shape
        ``(n_samples, n_models, n_targets)`` and ``mask`` marks the rows that
        actually received a prediction (the first fold has no history to train
        on, so it is left out).
        """
        n = len(X)
        n_folds = int(self.params["n_folds"])
        embargo = int(self.params["embargo"])
        n_targets = y.shape[1]

        oof = np.full((n, len(self.models), n_targets), np.nan)
        bounds = np.linspace(0, n, n_folds + 1).astype(int)

        for k in range(1, n_folds):
            test_lo, test_hi = bounds[k], bounds[k + 1]
            train_hi = max(0, test_lo - embargo)  # embargo gap before the fold
            if train_hi < 50 or test_hi <= test_lo:
                continue
            Xtr, ytr = X[:train_hi], y[:train_hi]
            Xte = X[test_lo:test_hi]
            for j, proto in enumerate(self.models):
                clone = type(proto)(**proto.params)
                try:
                    clone.fit(Xtr, ytr)
                    oof[test_lo:test_hi, j, :] = clone.predict(Xte)
                except Exception as exc:  # noqa: BLE001 - one weak learner must not kill the stack
                    log.warning("base model %s failed on OOF fold %d: %s", proto.name, k, exc)

        mask = ~np.isnan(oof).any(axis=(1, 2))
        return oof, mask

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        oof, mask = self._oof_predictions(X, y)
        n_models, n_targets = len(self.models), y.shape[1]

        # --- stage 2a: refit base models on everything ---
        for m in self.models:
            m.fit(X, y)
            # Base models are fitted on bare arrays inside the stack, so they
            # never see the column labels. Propagate them, otherwise every
            # feature-importance report comes back as f0, f1, f2 ... and is
            # useless for actually understanding the model.
            m.feature_names_ = self.feature_names_
            m.target_names_ = self.target_names_

        # --- stage 2b: meta-learner on the OOF matrix ---
        if mask.sum() < 50:
            log.warning("only %d usable OOF rows; falling back to an equal-weight blend", int(mask.sum()))
            self.meta_weights_ = np.full(n_models, 1.0 / n_models)
            self.n_targets_ = n_targets
            return

        # Pool across targets: one shared set of combination weights is far more
        # stable than fitting per-tenor weights on ~95%-correlated series.
        A = oof[mask].transpose(0, 2, 1).reshape(-1, n_models)   # (rows*targets, models)
        b = y[mask].reshape(-1)

        for j, m in enumerate(self.models):
            resid = b - A[:, j]
            ss = float(np.sum((b - b.mean()) ** 2))
            self.oof_score_[m.name] = 1.0 - float(np.sum(resid**2)) / ss if ss > 0 else np.nan

        alpha = float(self.params["meta_alpha"])
        if self.params["non_negative"]:
            from scipy.optimize import nnls

            # Ridge-regularised NNLS via an augmented system.
            aug_A = np.vstack([A, np.sqrt(alpha) * np.eye(n_models)])
            aug_b = np.concatenate([b, np.zeros(n_models)])
            w, _ = nnls(aug_A, aug_b)
            if w.sum() <= 0:
                w = np.full(n_models, 1.0 / n_models)
        else:
            gram = A.T @ A + alpha * np.eye(n_models)
            w = np.linalg.solve(gram, A.T @ b)

        self.meta_weights_ = w
        self.n_targets_ = n_targets
        log.info(
            "stacked weights: %s",
            {m.name: round(float(wi), 4) for m, wi in zip(self.models, w)},
        )

    def _predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.stack([m.predict(X) for m in self.models], axis=1)  # (n, models, targets)
        w = self.meta_weights_
        if w is None:
            w = np.full(preds.shape[1], 1.0 / preds.shape[1])
        return np.einsum("nmt,m->nt", preds, w)

    @property
    def weights_frame(self) -> pd.Series:
        """Fitted combination weight per base model."""
        w = self.meta_weights_ if self.meta_weights_ is not None else []
        return pd.Series(w, index=[m.name for m in self.models], name="stack_weight")

    @property
    def feature_importance(self) -> pd.Series | None:
        w = self.meta_weights_
        if w is None:
            return None
        parts = [(wi, m.feature_importance) for wi, m in zip(w, self.models)]
        parts = [(wi, imp) for wi, imp in parts if imp is not None and imp.sum() > 0 and wi > 0]
        if not parts:
            return None
        total = sum(wi for wi, _ in parts)
        combined = sum(imp / imp.sum() * (wi / total) for wi, imp in parts)
        return combined.sort_values(ascending=False)


class BestModelSelector(BaseModel):
    """Pick the single best base model by out-of-fold score.

    Simpler than stacking and sometimes better, because it spends no degrees of
    freedom on combination weights. Reuses :class:`StackedEnsemble`'s
    forward-chaining machinery to score candidates honestly.
    """

    name = "best"

    def __init__(self, models: Sequence[BaseModel], n_folds: int = 5, embargo: int = 5, **kw: Any) -> None:
        super().__init__(n_folds=n_folds, embargo=embargo, **kw)
        self.models = list(models)
        self.best_: BaseModel | None = None
        self.scores_: dict[str, float] = {}

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        helper = StackedEnsemble(self.models, n_folds=int(self.params["n_folds"]),
                                 embargo=int(self.params["embargo"]))
        oof, mask = helper._oof_predictions(X, y)
        if mask.sum() < 50:
            self.best_ = self.models[0]
        else:
            b = y[mask]
            for j, m in enumerate(self.models):
                resid = b - oof[mask][:, j, :]
                self.scores_[m.name] = float(np.sqrt(np.mean(resid**2)))
            best_name = min(self.scores_, key=lambda k: self.scores_[k])
            self.best_ = next(m for m in self.models if m.name == best_name)
            log.info("best model by OOF RMSE: %s (%s)", best_name,
                     {k: round(v, 8) for k, v in self.scores_.items()})
        self.best_.fit(X, y)
        self.best_.feature_names_ = self.feature_names_
        self.best_.target_names_ = self.target_names_

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_ is None:
            raise RuntimeError("BestModelSelector is not fitted")
        return self.best_.predict(X)

    @property
    def feature_importance(self) -> pd.Series | None:
        return self.best_.feature_importance if self.best_ else None
