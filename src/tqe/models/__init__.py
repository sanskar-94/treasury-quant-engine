"""Learners, ensembles and the model registry."""

from .base import BaseModel
from .ensemble import AverageEnsemble, BestModelSelector, StackedEnsemble
from .linear import (
    ARBaselineModel,
    ElasticNetModel,
    LassoModel,
    OLSModel,
    RidgeModel,
    ZeroModel,
)
from .registry import (
    MODEL_REGISTRY,
    available_models,
    build_ensemble,
    create_model,
    latest_bundle,
    load_bundle,
    register,
    save_bundle,
)
from .trees import ExtraTreesModel, GBMModel, RandomForestModel

__all__ = [
    "BaseModel", "RidgeModel", "ElasticNetModel", "LassoModel", "OLSModel",
    "ARBaselineModel", "ZeroModel", "RandomForestModel", "ExtraTreesModel", "GBMModel",
    "AverageEnsemble", "StackedEnsemble", "BestModelSelector",
    "MODEL_REGISTRY", "create_model", "build_ensemble", "register", "available_models",
    "save_bundle", "load_bundle", "latest_bundle",
]
