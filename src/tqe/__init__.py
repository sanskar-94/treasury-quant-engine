"""Treasury Quant Engine.

An end-to-end quantitative system for the US Treasury market: yield-curve
modelling, machine-learning return forecasting, risk-aware portfolio
construction, cost-aware backtesting and automated execution.
"""

__version__ = "1.0.0"

from .config import Config, load_config
from .logging_utils import get_logger, setup_logging

__all__ = ["Config", "load_config", "setup_logging", "get_logger", "__version__"]
