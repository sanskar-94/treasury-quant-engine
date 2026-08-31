"""Feature engineering: technical, macro, regime blocks and the design matrix."""

from .builder import FeatureSet, build_features, feature_report, make_targets
from .macro import PUBLICATION_LAG_DAYS, apply_publication_lag, macro_features
from .regime import regime_features, rolling_regime_labels
from .technical import (
    carry_rolldown_features,
    cross_tenor_features,
    curve_shape_features,
    mean_reversion_features,
    momentum_features,
    volatility_features,
    zscore_features,
)

__all__ = [
    "FeatureSet", "build_features", "make_targets", "feature_report",
    "momentum_features", "volatility_features", "zscore_features",
    "mean_reversion_features", "curve_shape_features", "carry_rolldown_features",
    "cross_tenor_features", "macro_features", "apply_publication_lag",
    "PUBLICATION_LAG_DAYS", "regime_features", "rolling_regime_labels",
]
