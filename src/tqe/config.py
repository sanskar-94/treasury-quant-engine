"""Typed configuration for the Treasury Quant Engine.

Configuration is layered, lowest priority first:

1. dataclass defaults defined here,
2. a YAML file (``configs/default.yaml`` unless overridden),
3. environment variables prefixed ``TQE_`` (e.g. ``TQE_MAX_GROSS_DV01``),
4. explicit keyword overrides passed by the CLI (``--set key=value``).

Keeping this in dataclasses rather than dicts means a typo in a config key fails
loudly at start-up instead of silently changing trading behaviour.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Where market data comes from and where it is cached."""

    data_dir: str = "data"
    start_date: str = "1990-01-02"
    end_date: str | None = None  # None -> today
    # Treasury.gov par-yield tenors we model.  Short tenors (1/2/4 Mo) only exist
    # for part of the history, so they are marked optional.
    tenors: list[str] = field(
        default_factory=lambda: [
            "1 Mo",
            "2 Mo",
            "3 Mo",
            "4 Mo",
            "6 Mo",
            "1 Yr",
            "2 Yr",
            "3 Yr",
            "5 Yr",
            "7 Yr",
            "10 Yr",
            "20 Yr",
            "30 Yr",
        ]
    )
    core_tenors: list[str] = field(
        default_factory=lambda: ["3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "30 Yr"]
    )
    # FRED series pulled for macro features.  Keys become column names.
    fred_series: dict[str, str] = field(
        default_factory=lambda: {
            "fed_funds": "DFF",
            "cpi_yoy": "CPIAUCSL",
            "core_cpi": "CPILFESL",
            "unemployment": "UNRATE",
            "industrial_prod": "INDPRO",
            "breakeven_10y": "T10YIE",
            "breakeven_5y": "T5YIE",
            "real_10y": "DFII10",
            "term_premium_proxy": "T10Y2Y",
            "credit_spread": "BAA10Y",
            "hy_spread": "BAMLH0A0HYM2",
            "vix": "VIXCLS",
            "sp500": "SP500",
            "dollar_index": "DTWEXBGS",
            "m2": "M2SL",
            "recession": "USREC",
        }
    )
    cache_days: int = 1  # re-download raw sources at most once per N days
    request_timeout: int = 45
    max_retries: int = 5


@dataclass
class CurveConfig:
    """Yield-curve fitting parameters."""

    model: str = "svensson"  # nelson_siegel | svensson
    n_pca_factors: int = 3
    # Multi-start bounds for the NSS decay parameters (in years).
    tau1_grid: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 3.0, 5.0])
    tau2_grid: list[float] = field(default_factory=lambda: [4.0, 8.0, 12.0, 20.0])
    max_fit_iter: int = 400
    fit_tolerance: float = 1e-10


@dataclass
class FeatureConfig:
    """Feature-engineering knobs."""

    momentum_windows: list[int] = field(default_factory=lambda: [1, 5, 10, 21, 63, 126, 252])
    vol_windows: list[int] = field(default_factory=lambda: [10, 21, 63, 252])
    zscore_windows: list[int] = field(default_factory=lambda: [21, 63, 252])
    include_macro: bool = True
    include_pca: bool = True
    include_curve_shape: bool = True
    include_carry_rolldown: bool = True
    include_regime: bool = True
    regime_states: int = 3
    # Features are lagged by this many business days before being used to predict
    # the *next* day's move.  1 = strictly information available at t-1 close.
    feature_lag: int = 1


@dataclass
class ModelConfig:
    """Learner selection and hyper-parameters."""

    target: str = "price_return"  # price_return | yield_change | direction
    horizon: int = 1  # business days ahead
    # Which base learners take part in the stacked ensemble.
    learners: list[str] = field(
        default_factory=lambda: ["ridge", "elasticnet", "random_forest", "gbm", "ar_baseline"]
    )
    use_torch_lstm: bool = False
    ensemble: str = "stacked"  # stacked | average | best
    ridge_alphas: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    gbm_max_iter: int = 400
    gbm_learning_rate: float = 0.03
    gbm_max_depth: int = 4
    gbm_l2: float = 1.0
    rf_n_estimators: int = 400
    rf_max_depth: int = 8
    lstm_hidden: int = 64
    lstm_layers: int = 2
    lstm_epochs: int = 40
    lstm_seq_len: int = 30
    random_state: int = 42


@dataclass
class TrainingConfig:
    """Walk-forward validation scheme."""

    scheme: str = "walk_forward"  # walk_forward | expanding | purged_kfold
    n_splits: int = 8
    test_size: int = 252  # one trading year per out-of-sample fold
    min_train_size: int = 1260  # five years of history before the first fold
    embargo: int = 5  # business days dropped between train and test
    expanding: bool = True
    standardize: bool = True
    artifacts_dir: str = "artifacts/models"


@dataclass
class CostConfig:
    """Transaction-cost model, quoted the way the cash treasury market does."""

    # Half-spread in 32nds of a point, by bucket.  On-the-run 10y ~ 1/2 of 1/32.
    half_spread_32nds: dict[str, float] = field(
        default_factory=lambda: {
            "bill": 0.10,
            "2y": 0.25,
            "5y": 0.35,
            "10y": 0.50,
            "30y": 1.00,
        }
    )
    # Square-root market-impact coefficient applied to (order DV01 / ADV DV01).
    impact_coefficient: float = 0.15
    # Financing: repo cost applied to leveraged notional, annualised.
    repo_spread_bp: float = 5.0
    commission_per_million: float = 12.5


@dataclass
class PortfolioConfig:
    """Sizing and portfolio-construction limits."""

    capital: float = 10_000_000.0
    target_annual_vol: float = 0.06
    max_leverage: float = 4.0
    max_gross_dv01: float = 25_000.0
    max_net_dv01: float = 15_000.0
    max_weight_per_tenor: float = 0.5
    dv01_neutral: bool = False  # if True, force sum(signed DV01) == 0
    turnover_penalty: float = 5.0
    signal_clip: float = 3.0
    min_signal_to_trade: float = 0.15
    rebalance: str = "daily"  # daily | weekly
    vol_lookback: int = 63
    kelly_fraction: float = 0.25


@dataclass
class BacktestConfig:
    """Backtest harness settings."""

    start_date: str | None = None
    end_date: str | None = None
    initial_capital: float = 10_000_000.0
    include_costs: bool = True
    include_financing: bool = True
    slippage_multiplier: float = 1.0
    benchmark: str = "10 Yr"  # buy-and-hold duration benchmark
    risk_free_series: str = "3 Mo"
    output_dir: str = "artifacts/backtests"


@dataclass
class RiskConfig:
    """Pre-trade risk gate - hard limits that block orders."""

    max_order_notional: float = 2_000_000.0
    max_position_notional: float = 20_000_000.0
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    var_confidence: float = 0.99
    var_limit_pct: float = 0.025
    max_orders_per_day: int = 50
    kill_switch: bool = False
    require_market_open: bool = True


@dataclass
class ExecutionConfig:
    """Broker wiring."""

    broker: str = "paper"  # paper | alpaca
    # ETF proxies used when trading through an equities broker.
    instrument_map: dict[str, str] = field(
        default_factory=lambda: {
            "2 Yr": "SHY",
            "3 Yr": "SHY",
            "5 Yr": "IEI",
            "7 Yr": "IEF",
            "10 Yr": "IEF",
            "20 Yr": "TLT",
            "30 Yr": "TLT",
        }
    )
    order_type: str = "limit"
    limit_offset_bp: float = 1.0
    time_in_force: str = "day"
    state_dir: str = "state"
    dry_run: bool = True


@dataclass
class Config:
    """Root configuration object."""

    data: DataConfig = field(default_factory=DataConfig)
    curve: CurveConfig = field(default_factory=CurveConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    log_level: str = "INFO"
    seed: int = 42

    # ---------------- convenience paths ---------------- #
    @property
    def root(self) -> Path:
        return REPO_ROOT

    @property
    def raw_dir(self) -> Path:
        return self._sub("raw")

    @property
    def processed_dir(self) -> Path:
        return self._sub("processed")

    @property
    def cache_dir(self) -> Path:
        return self._sub("cache")

    def _sub(self, name: str) -> Path:
        base = Path(self.data.data_dir)
        if not base.is_absolute():
            base = REPO_ROOT / base
        p = base / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def artifacts(self, *parts: str) -> Path:
        p = REPO_ROOT.joinpath("artifacts", *parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---------------- serialisation ---------------- #
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return path


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _coerce(value: Any, target_type: Any) -> Any:
    """Best-effort cast of a string (env var / CLI) into the dataclass field type."""
    if not isinstance(value, str):
        return value
    origin = getattr(target_type, "__origin__", None)
    if target_type is bool or target_type == "bool":
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if origin in (list, dict) or target_type in (list, dict):
        return yaml.safe_load(value)
    # Optional[...] and str fall through unchanged.
    return value


def _apply_mapping(obj: Any, mapping: dict[str, Any], path: str = "") -> None:
    """Recursively overlay ``mapping`` on a dataclass instance, validating keys."""
    valid = {f.name: f for f in fields(obj)}
    for key, value in mapping.items():
        if key not in valid:
            where = f"{path}.{key}" if path else key
            raise KeyError(
                f"Unknown configuration key {where!r}. Valid keys here: {sorted(valid)}"
            )
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_mapping(current, value, f"{path}.{key}" if path else key)
        else:
            setattr(obj, key, _coerce(value, valid[key].type))


def _apply_env(cfg: Config) -> None:
    """Overlay ``TQE_<SECTION>_<FIELD>`` (or a few well-known flat aliases)."""
    aliases = {
        "TQE_BROKER": ("execution", "broker"),
        "TQE_MAX_GROSS_DV01": ("portfolio", "max_gross_dv01"),
        "TQE_MAX_NET_DV01": ("portfolio", "max_net_dv01"),
        "TQE_MAX_ORDER_NOTIONAL": ("risk", "max_order_notional"),
        "TQE_MAX_DAILY_LOSS_PCT": ("risk", "max_daily_loss_pct"),
        "TQE_KILL_SWITCH": ("risk", "kill_switch"),
        "TQE_DATA_DIR": ("data", "data_dir"),
        "TQE_LOG_LEVEL": (None, "log_level"),
    }
    for env_name, (section, key) in aliases.items():
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        target = cfg if section is None else getattr(cfg, section)
        ftype = {f.name: f.type for f in fields(target)}[key]
        setattr(target, key, _coerce(raw, ftype))


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    use_env: bool = True,
) -> Config:
    """Build a :class:`Config` from defaults + YAML + env + explicit overrides."""
    cfg = Config()

    if path is None:
        default = REPO_ROOT / "configs" / "default.yaml"
        path = default if default.exists() else None

    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file {path} must contain a YAML mapping")
        _apply_mapping(cfg, data)

    if use_env:
        _apply_env(cfg)

    if overrides:
        _apply_mapping(cfg, _explode_dotted(overrides))

    return cfg


def _explode_dotted(flat: dict[str, Any]) -> dict[str, Any]:
    """``{"portfolio.capital": 5e6}`` -> ``{"portfolio": {"capital": 5e6}}``."""
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(".")
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return nested
