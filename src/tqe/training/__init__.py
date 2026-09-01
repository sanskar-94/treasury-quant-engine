"""Walk-forward training, validation splits and performance metrics."""

from .metrics import (
    deflated_sharpe_ratio,
    drawdown_series,
    information_coefficient,
    performance_metrics,
    probabilistic_sharpe_ratio,
    rank_information_coefficient,
    regression_metrics,
)
from .splits import (
    Split,
    describe_splits,
    purged_kfold_splits,
    validate_splits,
    walk_forward_splits,
)
from .train import TrainResult, train_final_model, train_walk_forward
from .tune import TuneResult, grid_search, nested_walk_forward

__all__ = [
    "Split", "walk_forward_splits", "purged_kfold_splits", "describe_splits", "validate_splits",
    "TrainResult", "train_walk_forward", "train_final_model",
    "TuneResult", "grid_search", "nested_walk_forward",
    "regression_metrics", "performance_metrics", "information_coefficient",
    "rank_information_coefficient", "drawdown_series", "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
]
