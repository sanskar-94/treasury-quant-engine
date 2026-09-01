"""Hyper-parameter search that does not lie about its own performance.

Choosing hyper-parameters by walk-forward score and then reporting that same
walk-forward score is a leak, and it is the most common one in quantitative
research because it does not look like cheating. The outer folds saw every
candidate; the winner's score is the maximum of many draws, not an unbiased
estimate of anything.

The fix is **nested** cross-validation. Each outer fold's training block is
itself split into inner folds; the hyper-parameters are chosen on the inner
folds alone, and the outer test block scores only that choice. The outer score
is then an honest estimate of the whole *procedure* - "fit this model class,
tune it this way, then predict" - which is the thing that would actually be
deployed.

It costs ``n_outer x n_inner x n_candidates`` fits, which is why it is a
separate entry point rather than the default. Reporting the tuned score without
it is the cheaper option and the wrong one.

The number of candidates searched is returned in :class:`TuneResult` so it can
be fed to the deflated Sharpe ratio - a search is only honest if its width is
declared.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..features.builder import FeatureSet
from ..logging_utils import get_logger
from ..models.registry import create_model
from .metrics import regression_metrics
from .splits import walk_forward_splits
from .train import _apply, fit_scaler

log = get_logger("training.tune")

__all__ = ["TuneResult", "grid_search", "nested_walk_forward", "DEFAULT_GRIDS"]


#: Sensible search spaces per learner. Deliberately small - a wide grid on a
#: signal this faint buys multiple-testing penalty, not accuracy.
DEFAULT_GRIDS: dict[str, dict[str, Sequence[Any]]] = {
    "ridge": {"alphas": [(0.1, 1.0, 10.0), (1.0, 10.0, 100.0), (10.0, 100.0, 1000.0)]},
    "gbm": {
        "learning_rate": [0.01, 0.03, 0.1],
        "max_depth": [2, 3, 4],
        "l2_regularization": [0.1, 1.0, 10.0],
    },
    "random_forest": {"max_depth": [4, 6, 8], "min_samples_leaf": [10, 20, 50]},
    "elasticnet": {"l1_ratio": [(0.2,), (0.5,), (0.8,)]},
}


@dataclass
class TuneResult:
    """Outcome of a hyper-parameter search."""

    best_params: dict[str, Any]
    best_score: float
    leaderboard: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_candidates: int = 0
    n_fits: int = 0
    seconds: float = 0.0
    outer_metrics: dict[str, float] = field(default_factory=dict)
    outer_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    selected_per_fold: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"TuneResult(best={self.best_params}, inner_score={self.best_score:.6g}, "
            f"{self.n_candidates} candidates, {self.n_fits} fits, {self.seconds:.0f}s)"
        )


def _expand(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a parameter grid."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def _score(
    model_name: str,
    params: dict[str, Any],
    fs: FeatureSet,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: Config,
    metric: str,
) -> float:
    """Fit one candidate on one split and return its score (higher is better)."""
    Xtr, ytr = fs.X.iloc[train_idx], fs.y.iloc[train_idx]
    Xte, yte = fs.X.iloc[test_idx], fs.y.iloc[test_idx]

    scaler = fit_scaler(Xtr, cfg.training.standardize)
    model = create_model(model_name, cfg.model, **params)
    model.fit(_apply(scaler, Xtr), ytr)
    pred = model.predict(_apply(scaler, Xte))

    m = regression_metrics(yte.to_numpy().ravel(), np.asarray(pred).ravel())
    # RMSE is a loss, so negate it to keep "higher is better" everywhere.
    return -m["rmse"] if metric == "rmse" else m.get(metric, np.nan)


def grid_search(
    fs: FeatureSet,
    model_name: str = "ridge",
    grid: Mapping[str, Sequence[Any]] | None = None,
    cfg: Config | None = None,
    n_splits: int = 3,
    metric: str = "ic",
    verbose: bool = True,
) -> TuneResult:
    """Walk-forward grid search over one learner's hyper-parameters.

    **This is the inner loop, not a performance estimate.** The returned score is
    the best of ``n_candidates`` draws and is biased upward by exactly the amount
    you would expect from taking a maximum. Use :func:`nested_walk_forward` when
    the number reported has to mean something.
    """
    cfg = cfg or Config()
    grid = grid if grid is not None else DEFAULT_GRIDS.get(model_name, {})
    candidates = _expand(grid)

    splits = walk_forward_splits(
        fs.index, n_splits=n_splits,
        test_size=max(len(fs) // (n_splits + 2), 60),
        min_train_size=max(len(fs) // 3, 120),
        embargo=cfg.training.embargo, horizon=cfg.model.horizon,
    )
    if not splits:
        raise ValueError("Not enough observations to build inner folds")

    t0 = time.time()
    rows, n_fits = [], 0
    for params in candidates:
        scores = []
        for s in splits:
            try:
                scores.append(_score(model_name, params, fs, s.train_idx, s.test_idx, cfg, metric))
                n_fits += 1
            except Exception as exc:  # noqa: BLE001 - one bad candidate must not stop the search
                log.debug("candidate %s failed on fold %d: %s", params, s.fold, exc)
        if not scores:
            continue
        rows.append({**params, "score": float(np.mean(scores)),
                     "score_sd": float(np.std(scores)), "n_folds": len(scores)})
        if verbose:
            log.info("  %s -> %s %.6f", params, metric, rows[-1]["score"])

    if not rows:
        raise RuntimeError(f"No candidate could be fitted for {model_name!r}")

    board = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    best = {k: v for k, v in board.iloc[0].items() if k not in ("score", "score_sd", "n_folds")}
    return TuneResult(
        best_params=best,
        best_score=float(board.iloc[0]["score"]),
        leaderboard=board,
        n_candidates=len(candidates),
        n_fits=n_fits,
        seconds=time.time() - t0,
    )


def nested_walk_forward(
    fs: FeatureSet,
    model_name: str = "ridge",
    grid: Mapping[str, Sequence[Any]] | None = None,
    cfg: Config | None = None,
    inner_splits: int = 3,
    metric: str = "ic",
    model_factory: Callable[..., Any] | None = None,
) -> TuneResult:
    """Nested cross-validation: tune inside, score outside.

    For each outer walk-forward fold:

    1. take that fold's **training block only**,
    2. run :func:`grid_search` on it, using inner folds carved from that block,
    3. refit the winning configuration on the whole training block,
    4. predict the outer test block.

    The outer test blocks never influence which hyper-parameters are chosen, so
    concatenating their predictions gives an honest out-of-sample record for the
    tuning procedure as a whole.

    ``TuneResult.n_candidates`` reports the total number of configurations
    evaluated across all outer folds - the figure to hand to
    :func:`~tqe.training.metrics.deflated_sharpe_ratio`.
    """
    cfg = cfg or Config()
    grid = grid if grid is not None else DEFAULT_GRIDS.get(model_name, {})
    n_candidates = len(_expand(grid))

    outer = walk_forward_splits(
        fs.index, n_splits=cfg.training.n_splits, test_size=cfg.training.test_size,
        min_train_size=cfg.training.min_train_size, embargo=cfg.training.embargo,
        expanding=cfg.training.expanding, horizon=cfg.model.horizon,
    )
    if not outer:
        raise ValueError("Not enough observations for nested validation")

    t0 = time.time()
    preds, actuals, chosen = [], [], []
    total_fits = 0

    for s in outer:
        inner_fs = FeatureSet(
            X=fs.X.iloc[s.train_idx], y=fs.y.iloc[s.train_idx],
            metadata={"outer_fold": s.fold},
        )
        try:
            inner = grid_search(inner_fs, model_name, grid, cfg,
                                n_splits=inner_splits, metric=metric, verbose=False)
            params = inner.best_params
            total_fits += inner.n_fits
        except Exception as exc:  # noqa: BLE001
            log.warning("inner search failed on outer fold %d (%s); using defaults", s.fold, exc)
            params = {}

        Xtr, ytr = fs.X.iloc[s.train_idx], fs.y.iloc[s.train_idx]
        Xte, yte = fs.X.iloc[s.test_idx], fs.y.iloc[s.test_idx]
        scaler = fit_scaler(Xtr, cfg.training.standardize)
        model = (model_factory(**params) if model_factory
                 else create_model(model_name, cfg.model, **params))
        model.fit(_apply(scaler, Xtr), ytr)
        total_fits += 1

        preds.append(pd.DataFrame(model.predict(_apply(scaler, Xte)),
                                  index=Xte.index, columns=ytr.columns))
        actuals.append(yte)
        chosen.append({"fold": s.fold, **params})
        log.info("outer fold %d: chose %s", s.fold, params)

    oos_pred = pd.concat(preds).sort_index()
    oos_act = pd.concat(actuals).sort_index()
    outer_metrics = regression_metrics(oos_act.to_numpy().ravel(), oos_pred.to_numpy().ravel())

    # Which configuration won most often - the one you would ship.
    counts: dict[str, int] = {}
    for c in chosen:
        key = str({k: v for k, v in c.items() if k != "fold"})
        counts[key] = counts.get(key, 0) + 1
    modal = max(counts, key=lambda k: counts[k]) if counts else "{}"

    return TuneResult(
        best_params=next(({k: v for k, v in c.items() if k != "fold"}
                          for c in chosen if str({k: v for k, v in c.items() if k != "fold"}) == modal),
                         {}),
        best_score=float(outer_metrics.get(metric, np.nan)),
        leaderboard=pd.DataFrame(chosen),
        n_candidates=n_candidates * len(outer),
        n_fits=total_fits,
        seconds=time.time() - t0,
        outer_metrics=outer_metrics,
        outer_predictions=oos_pred,
        selected_per_fold=chosen,
    )
