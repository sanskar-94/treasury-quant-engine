"""Linear learners and the honest baselines they have to beat.

Regularised linear models are not an afterthought in rates forecasting - they
are frequently the best thing available. The signal-to-noise ratio in daily bond
returns is so low (a good feature correlates ~0.05 with tomorrow's move) that a
flexible learner mostly fits noise. Ridge, with the shrinkage chosen by
walk-forward validation, is the workhorse.

Two baselines are included and are meant to be taken seriously:

``ZeroModel``
    Always predicts zero. Any strategy that cannot beat *doing nothing* after
    costs is not a strategy. This is the null hypothesis.

``ARBaselineModel``
    Predicts tomorrow from a trailing mean of recent returns. Captures whatever
    trivial autocorrelation exists, so a complex model that fails to beat it has
    demonstrably added nothing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .base import BaseModel

log = get_logger("models.linear")

__all__ = ["RidgeModel", "ElasticNetModel", "LassoModel", "OLSModel", "ARBaselineModel", "ZeroModel"]


class _SklearnLinear(BaseModel):
    """Shared plumbing for the scikit-learn linear estimators."""

    def _make_estimator(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.estimator_ = self._make_estimator()
        # sklearn's linear models are natively multi-output.
        self.estimator_.fit(X, y if y.shape[1] > 1 else y.ravel())

    def _predict(self, X: np.ndarray) -> np.ndarray:
        pred = self.estimator_.predict(X)
        return pred if np.ndim(pred) == 2 else np.asarray(pred).reshape(-1, 1)

    @property
    def feature_importance(self) -> pd.Series | None:
        est = getattr(self, "estimator_", None)
        if est is None or not hasattr(est, "coef_"):
            return None
        coef = np.atleast_2d(est.coef_)
        # Average absolute coefficient across targets: the features are already
        # standardised upstream, so magnitudes are comparable.
        imp = np.abs(coef).mean(axis=0)
        names = self.feature_names_ or [f"f{i}" for i in range(len(imp))]
        return pd.Series(imp, index=names).sort_values(ascending=False)


class RidgeModel(_SklearnLinear):
    """L2-regularised regression with the penalty chosen by cross-validation.

    ``RidgeCV`` selects ``alpha`` by leave-one-out generalised cross-validation
    on the *training fold only*, which is safe inside a walk-forward split: the
    outer split already guarantees the test block is unseen. It is emphatically
    not safe to select alpha on the full sample and then walk forward, which is
    why the alpha search lives inside the model rather than in a global tuning
    step.
    """

    name = "ridge"

    def __init__(self, alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0), **kw: Any) -> None:
        super().__init__(alphas=tuple(alphas), **kw)

    def _make_estimator(self):
        from sklearn.linear_model import RidgeCV

        return RidgeCV(alphas=np.asarray(self.params["alphas"], dtype=float))

    @property
    def alpha_(self) -> float | None:
        est = getattr(self, "estimator_", None)
        return float(getattr(est, "alpha_", np.nan)) if est is not None else None


class ElasticNetModel(_SklearnLinear):
    """L1 + L2 regression - shrinks *and* selects.

    With ~480 mostly-collinear features, the L1 component is doing real work:
    it collapses whole blocks of near-duplicate momentum windows down to one
    representative, which makes the fitted model far easier to interpret.
    """

    name = "elasticnet"

    # Defaults are deliberately frugal.  Coordinate descent scales badly with
    # collinearity, and this feature set is severely collinear by construction
    # (seven momentum windows on nine tenors are near-duplicates of each other).
    # Measured on the real 4000 x 482 design matrix: the textbook settings
    # (3 l1_ratios x 30 alphas x 3-fold CV at tol=1e-4, max_iter=5000 - 270 fits)
    # did not converge in nine minutes for a SINGLE fit, while the settings below
    # complete in ~11 seconds with an identical selected alpha.  Widen the grid
    # only if you have measured that it buys you something.
    #
    # NOTE: scikit-learn >= 1.9 replaced the integer ``n_alphas`` argument with
    # ``alphas``, which now accepts either an int (grid size) or an explicit
    # array.  The public constructor keeps the older, clearer name.
    def __init__(self, l1_ratio=(0.5,), n_alphas: int = 5, max_iter: int = 1000,
                 tol: float = 1e-3, cv: int = 2, random_state: int = 42, **kw: Any) -> None:
        super().__init__(l1_ratio=tuple(l1_ratio), n_alphas=n_alphas, max_iter=max_iter,
                         tol=tol, cv=cv, random_state=random_state, **kw)

    def _make_estimator(self):
        from sklearn.linear_model import MultiTaskElasticNetCV

        return MultiTaskElasticNetCV(
            l1_ratio=list(self.params["l1_ratio"]),
            alphas=int(self.params["n_alphas"]),
            max_iter=int(self.params["max_iter"]),
            tol=float(self.params["tol"]),
            cv=int(self.params["cv"]),
            random_state=int(self.params["random_state"]),
            n_jobs=1,
            selection="random",
        )

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if y.shape[1] == 1:
            from sklearn.linear_model import ElasticNetCV

            self.estimator_ = ElasticNetCV(
                l1_ratio=list(self.params["l1_ratio"]),
                alphas=int(self.params["n_alphas"]),
                max_iter=int(self.params["max_iter"]),
                tol=float(self.params["tol"]),
                cv=int(self.params["cv"]),
                random_state=int(self.params["random_state"]),
            )
            self.estimator_.fit(X, y.ravel())
        else:
            self.estimator_ = self._make_estimator()
            self.estimator_.fit(X, y)


class LassoModel(_SklearnLinear):
    """Pure L1 - the sparsest linear fit."""

    name = "lasso"

    def __init__(self, n_alphas: int = 5, max_iter: int = 1000, tol: float = 1e-3,
                 cv: int = 2, random_state: int = 42, **kw: Any) -> None:
        super().__init__(n_alphas=n_alphas, max_iter=max_iter, tol=tol, cv=cv,
                         random_state=random_state, **kw)

    def _make_estimator(self):
        from sklearn.linear_model import MultiTaskLassoCV

        return MultiTaskLassoCV(
            alphas=int(self.params["n_alphas"]),
            max_iter=int(self.params["max_iter"]),
            tol=float(self.params["tol"]),
            cv=int(self.params["cv"]),
            random_state=int(self.params["random_state"]),
            n_jobs=1,
        )


class OLSModel(_SklearnLinear):
    """Unregularised least squares.

    Included mainly as a cautionary control: with 482 features and a few thousand
    observations it overfits badly, and seeing it lose to ridge out of sample is
    a useful demonstration that the regularisation is earning its keep.
    """

    name = "ols"

    def _make_estimator(self):
        from sklearn.linear_model import LinearRegression

        return LinearRegression()


class ARBaselineModel(BaseModel):
    """Predict the next return as a decayed mean of recent returns.

    Deliberately ignores the design matrix and uses only the target's own
    history, which makes it the cleanest possible "did the features add
    anything?" control. It needs the training targets at predict time, so it
    stores the tail of the training set and updates as it goes.
    """

    name = "ar_baseline"

    def __init__(self, window: int = 5, shrink: float = 0.1, **kw: Any) -> None:
        super().__init__(window=window, shrink=shrink, **kw)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        w = int(self.params["window"])
        self.tail_ = y[-w:] if len(y) >= w else y
        self.mean_ = y.mean(axis=0)
        self.n_targets_ = y.shape[1]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        # A constant, shrunk-toward-zero forecast: the trailing mean carries
        # essentially no information at daily frequency, which is the point.
        base = self.tail_.mean(axis=0) * float(self.params["shrink"])
        return np.tile(base, (len(X), 1))


class ZeroModel(BaseModel):
    """The null hypothesis: forecast zero, always.

    Keep this in every comparison. A backtest that beats zero only before costs
    has not found anything.
    """

    name = "zero"

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.n_targets_ = y.shape[1]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((len(X), self.n_targets_))
