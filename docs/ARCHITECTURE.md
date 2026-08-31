# Architecture & Module Contracts

This document is the single source of truth for the public API of every module in
`src/tqe`. It was written *before* the modules, as a contract, so that layers
could be built and tested independently and still compose. Everything specified
here is now implemented; where the implementation diverged from the original
plan the divergence is noted inline and the reason is recorded in the module's
own docstring.

**Divergences worth knowing about:**

| Contract said | What was built | Why |
| --- | --- | --- |
| `fit_nss_history` for features | `fit_nss_history_fixed` (Diebold-Li fixed decays) | free-tau betas are not identifiable; beta3 sd 1.65 vs 0.031 |
| bootstrap solves `DF_T` in closed form | bisection on each tenor's own zero rate | closed form flat-extrapolates between sparse nodes, biasing the 20y by 38bp |
| `predictions_to_signal(method="zscore")` default | `vol_scale` default | demeaning a return forecast destroys its zero point; 32/32 configs positive vs 0/32 |
| turnover penalty as a raw coefficient | multiplier on cost, amortised over the holding period | a raw coefficient is four orders of magnitude off daily returns and returns an empty book |
| canary = signal shifted forward | canary = realised future return | shifting the signal only misaligns it and proves nothing |
| DV01 caps only | DV01 caps **and** a gross-notional leverage cap | a DV01 limit permitted 14.9x leverage via the front end |

## 0. Conventions that hold everywhere

| Concept | Representation |
| --- | --- |
| Rates / yields | **decimal** floats (`0.0473` = 4.73%) |
| Prices | per **100** face, decimal (`99.515625`) |
| DV01 | **positive** dollars lost per **+1bp**, per 100 face unless stated |
| Dates | `datetime.date` in pricing, `pd.Timestamp` in frames |
| Time series | `pd.DataFrame`, `DatetimeIndex` named `date`, ascending, no duplicates |
| Tenor labels | Treasury CMT strings: `"3 Mo"`, `"1 Yr"`, `"10 Yr"`, `"30 Yr"` |
| Tenor → years | `tqe.data.sources.TENOR_YEARS` |
| Missing data | `NaN`, never forward-filled inside the data layer |

**The cardinal rule: no look-ahead.** Any value used to predict day *t* must be
computable strictly from information available at the close of day *t-1*. Every
module that touches time series states in its docstring how it satisfies this.

## 1. Canonical datasets

```python
curve: pd.DataFrame   # (9172, 14) 1990-01-02 .. today, decimal par yields, NaN-ragged
macro: pd.DataFrame   # FRED bundle, mixed daily/monthly frequency, NaN-ragged
```

Columns present in `curve`: `1 Mo, 1.5 Month, 2 Mo, 3 Mo, 4 Mo, 6 Mo, 1 Yr, 2 Yr,
3 Yr, 5 Yr, 7 Yr, 10 Yr, 20 Yr, 30 Yr`. Coverage is ragged — `1 Mo` starts
2001-07-31, `2 Mo` 2018-10-16, `4 Mo` 2022-10-19, `1.5 Month` 2025-02-18, `20 Yr`
starts 1993-10-01, and `30 Yr` has a genuine publication gap from 2002-02 to
2006-02. **Do not forward-fill across those gaps**; select tenors by coverage.

`cfg.data.core_tenors` is the always-available set used for modelling.

---

## 2. `tqe.pricing` — implemented

```python
# daycount.py
class DayCount(str, Enum): ACT_ACT_ICMA, ACT_360, ACT_365F, THIRTY_360
year_fraction(start, end, convention, period_start=None, period_end=None, frequency=2) -> float
accrual_fraction(settlement, period_start, period_end) -> float
add_months(d: date, months: int) -> date      # clamps day-of-month
days_between(start, end) -> int

# bond.py
@dataclass(frozen=True)
class Bond:
    maturity: date; coupon: float          # coupon in PERCENT (4.25 == 4.25%)
    face: float = 100.0; frequency: int = 2
    issue_date: date | None = None; first_coupon: date | None = None
    label: str = ""; cusip: str = ""
    coupon_dates(after=None) -> list[date]
    period_bounds(settlement) -> tuple[date, date]
    w(settlement) -> float                 # fraction of period REMAINING
    n_remaining(settlement) -> int
    time_to_maturity(settlement) -> float  # years, (n-1+w)/freq
    semi_coupon -> float
    cashflows(settlement) -> list[tuple[date, float]]
    cashflow_times(settlement) -> np.ndarray   # years
    cashflow_amounts(settlement) -> np.ndarray

accrued_interest(bond, settlement) -> float
dirty_price_from_yield(bond, settlement, ytm) -> float
price_from_yield(bond, settlement, ytm) -> float          # CLEAN price
yield_from_price(bond, settlement, clean_price, guess=None) -> float
price_from_discount_curve(bond, settlement, discount: Callable[[float], float], clean=True) -> float
par_bond(settlement, tenor_years, par_yield, face=100.0) -> Bond
bill_price_from_discount(d, days, face=100.) / bill_discount_from_price / bill_bond_equivalent_yield
format_32nds(price) -> str  ;  parse_32nds(quote) -> float

# analytics.py
@dataclass(frozen=True)
class BondRisk:
    clean_price, dirty_price, accrued, ytm, macaulay_duration,
    modified_duration, convexity, dv01, time_to_maturity   # all float
    as_dict() -> dict[str, float]

macaulay_duration(bond, settlement, ytm) -> float
modified_duration(bond, settlement, ytm) -> float
convexity(bond, settlement, ytm) -> float
dv01(bond, settlement, ytm, face=None) -> float           # POSITIVE
effective_duration(bond, settlement, ytm, bump_bp=1.0) -> float
effective_convexity(bond, settlement, ytm, bump_bp=1.0) -> float
key_rate_durations(bond, settlement, zero_rate: Callable[[float], float],
                   key_tenors=DEFAULT_KEY_TENORS, bump_bp=1.0, compounding=2) -> dict[float, float]
bond_risk(bond, settlement, ytm=None, clean_price=None) -> BondRisk
price_change_estimate(risk, delta_yield, include_convexity=True) -> float
carry_and_rolldown(bond, settlement, ytm, horizon_days=90, repo_rate=0.0,
                   forward_yield=None) -> dict[str, float]   # carry/rolldown/total/financing/coupon_income
portfolio_dv01(positions: dict[str, float], dv01_per_unit: dict[str, float]) -> float
hedge_ratio(target_dv01, hedge_dv01) -> float
```

## 3. `tqe.data` — implemented

```python
# sources.py
TENOR_YEARS: dict[str, float]
load_market_data(cfg, force=False) -> tuple[curve_df, macro_df]
fetch_treasury_curve(start_year, end_year, cache_dir, series="yield_curve", ...) -> pd.DataFrame
fetch_fred_bundle(series_map, cache_dir, ...) -> pd.DataFrame
clean_curve(df, max_daily_move_bp=150.0) -> pd.DataFrame
curve_coverage(df) -> pd.DataFrame

# calendar.py
is_business_day(d) / next_business_day(d, n=1) / previous_business_day(d, n=1)
business_days_between(start, end) -> int
business_day_range(start, end) -> list[date]
settlement_date(trade_date, lag=None) -> date       # T+1 from 2024-05-28, else T+2
trading_index(start, end) -> pd.DatetimeIndex
annualization_factor(index) -> float
holidays_for_year(year) -> frozenset[date]
```

### 3a. `data/universe.py`

Turns the CMT par-yield series into **investable instruments with total
returns**. This is the bridge from "a yield went from 4.70% to 4.73%" to "the
strategy made or lost this many dollars".

```python
@dataclass(frozen=True)
class TenorSpec:
    label: str          # "10 Yr"
    years: float        # 10.0
    bucket: str         # "bill" | "2y" | "5y" | "10y" | "30y"  (cost bucket)

CORE_SPECS: tuple[TenorSpec, ...]

def build_universe(curve: pd.DataFrame, tenors: Sequence[str] | None = None) -> list[TenorSpec]
    """Tenors with enough coverage to trade, ordered short -> long."""

def constant_maturity_total_return(
    curve: pd.DataFrame, tenors: Sequence[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Per-tenor daily analytics frame. Returns {tenor: DataFrame} where each
    frame is indexed by date and has columns:

        yield          decimal par yield that day
        price          clean price of the synthetic par bond (== 100.0 by construction)
        dirty_price    price + accrued (== 100.0, bond issued that day)
        duration       modified duration (years)
        dv01           per 100 face, positive
        convexity      years^2
        carry_1d       one-day coupon accrual per 100 face
        price_return   one-day CLEAN price return from repricing YESTERDAY's bond
                       at TODAY's yield with one day less to maturity
        total_return   price_return + carry_1d/100
        yield_change   today's yield - yesterday's yield (decimal, so 0.0001 = 1bp)

    Method: on each day t-1 build the par bond `par_bond(t-1, tenor, y_{t-1})`
    which prices at exactly 100. On day t reprice THAT SAME bond (unchanged
    maturity, unchanged coupon) at yield y_t with settlement t. The clean price
    change plus one day of accrual is the realised total return of holding the
    on-the-run bond overnight. This is the standard CMT replication and is what
    makes the backtest P&L economically real.

    Must be vectorised enough to run 9000 days x 9 tenors in < 60s.
    """

def universe_panel(returns: dict[str, pd.DataFrame], field: str) -> pd.DataFrame
    """Pivot {tenor: frame} -> single DataFrame of one field, columns = tenors."""

def butterfly_weights(short_dv01, belly_dv01, long_dv01) -> tuple[float, float, float]
    """50/50 DV01-neutral fly weights (short, belly, long); belly is +1 unit."""
```

### 3b. `data/cache.py`
Small Parquet cache helper: `save_frame(df, path)`, `load_frame(path)`,
`cached(key, builder, cache_dir, max_age_days=None)`, `clear_cache(cache_dir)`.

---

## 4. `tqe.curve`

```python
# nelson_siegel.py
@dataclass(frozen=True)
class NSSParams:
    beta0: float; beta1: float; beta2: float; beta3: float
    tau1: float; tau2: float
    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray
    def forward_rate(self, t) -> ...
    def discount(self, t) -> ...
    def as_array(self) -> np.ndarray
    @classmethod
    def from_array(cls, arr) -> "NSSParams"
    @property
    def level(self) -> float        # beta0
    @property
    def slope(self) -> float        # -beta1  (long minus short)
    @property
    def curvature(self) -> float    # beta2

def nss_zero_rate(t, beta0, beta1, beta2, beta3, tau1, tau2) -> np.ndarray
    """r(t) = b0 + b1*f1 + b2*(f1 - exp(-t/tau1)) + b3*(f2 - exp(-t/tau2))
    where f1 = (1-exp(-t/tau1))/(t/tau1), f2 = (1-exp(-t/tau2))/(t/tau2).
    MUST handle t -> 0 (limit f -> 1) without dividing by zero."""

def fit_nss(tenors_years, yields, weights=None, tau1_grid=..., tau2_grid=...,
            model="svensson") -> tuple[NSSParams, float]
    """Multi-start fit. For FIXED (tau1, tau2) the model is LINEAR in the betas,
    so solve the betas by weighted least squares (np.linalg.lstsq) over a grid of
    taus and keep the best — this is far more robust than throwing all 6
    parameters at an optimizer. Optionally polish with scipy.optimize.least_squares.
    Returns (params, rmse_in_decimal). model="nelson_siegel" forces beta3=0."""

def fit_nss_history(curve: pd.DataFrame, tenor_years: dict[str, float],
                    model="svensson", n_jobs=1) -> pd.DataFrame
    """Fit every row. Returns DataFrame indexed by date with columns
    beta0,beta1,beta2,beta3,tau1,tau2,rmse,n_points. Rows with <4 valid tenors
    yield NaN. Seed each day's fit from the previous day's taus for stability."""

# bootstrap.py
def par_to_zero(tenors_years, par_yields, frequency=2, max_tenor=None) -> tuple[np.ndarray, np.ndarray]
    """Bootstrap zero (spot) rates from par yields. Par bond of tenor T with
    coupon c prices at 100 => 100 = sum_{i} (c/f)*DF_i + 100*DF_T. Solve DF_T
    recursively, interpolating intermediate DFs from already-solved points
    (log-linear on DF). Return (tenors, zero_rates) semi-annually compounded."""

def zero_to_forward(tenors, zeros, frequency=2) -> np.ndarray
def zero_to_discount(tenors, zeros, frequency=2) -> np.ndarray
def interpolate_curve(tenors, values, targets, method="linear") -> np.ndarray
    # method in {"linear", "log_linear_df", "cubic", "monotone_cubic"}
def forward_rate(zeros_fn, t1, t2, frequency=2) -> float
def bootstrap_history(curve: pd.DataFrame, tenor_years: dict) -> pd.DataFrame
    """Zero curve per day; columns = tenor labels, values = zero rates."""

# pca.py
@dataclass
class CurvePCA:
    components_: np.ndarray      # (n_factors, n_tenors)
    explained_variance_ratio_: np.ndarray
    mean_: np.ndarray
    tenors: list[str]
    def transform(self, X) -> np.ndarray
    def inverse_transform(self, F) -> np.ndarray
    def factor_frame(self, X: pd.DataFrame) -> pd.DataFrame   # columns level/slope/curvature/pc4..

def fit_curve_pca(changes: pd.DataFrame, n_factors=3, sign_convention=True) -> CurvePCA
    """PCA on DAILY YIELD CHANGES (not levels — levels are non-stationary and the
    first PC would just be the mean). Fit on the covariance of changes.
    sign_convention=True flips signs so PC1 loads positively everywhere (level),
    PC2 is increasing in tenor (slope), PC3 is a hump (curvature).
    Measured on this dataset (9 core tenors, 3m-30y, 1990-2026 daily changes):
    77.2% / 13.3% / 5.0%, cumulative 95.5%. The often-quoted ~90/7/2 applies to a
    narrower 2y-30y set; including the 3-month bill, which decouples from the long
    end when the Fed is pinned, moves variance out of level and into slope."""

def rolling_pca_factors(changes: pd.DataFrame, window=252, n_factors=3) -> pd.DataFrame
    """Expanding/rolling PCA scores computed WITHOUT look-ahead: the loadings
    used on day t are fitted only on data up to t-1."""
```

---

## 5. `tqe.features`

```python
# builder.py
@dataclass
class FeatureSet:
    X: pd.DataFrame          # features, DatetimeIndex, all float, no NaN rows
    y: pd.DataFrame          # targets, same index, columns = tenors
    feature_names: list[str]
    target_names: list[str]
    metadata: dict

def build_features(curve, macro, cfg, returns=None, nss=None, pca_factors=None) -> FeatureSet
    """Assemble every feature block, apply cfg.features.feature_lag, align, and
    drop rows with NaN. Targets are NEXT-day values so row t holds
    (features known at t, target realised at t+horizon)."""

def make_targets(returns: dict[str, pd.DataFrame], target="price_return",
                 horizon=1, tenors=None) -> pd.DataFrame
    """Shift(-horizon) so row t carries the FUTURE return. Column names = tenors."""

# technical.py  — all take a DataFrame of levels/returns, return a DataFrame
momentum_features(prices_or_yields, windows) -> pd.DataFrame
volatility_features(returns, windows) -> pd.DataFrame
zscore_features(series, windows) -> pd.DataFrame
mean_reversion_features(series, windows) -> pd.DataFrame
curve_shape_features(curve) -> pd.DataFrame
    """2s10s, 3m10y, 5s30s slopes; 2-5-10 and 5-10-30 butterflies; level; a
    steepness ratio; and an inversion indicator."""
carry_rolldown_features(curve, returns) -> pd.DataFrame

# macro.py
PUBLICATION_LAG_DAYS: dict[str, int]     # e.g. cpi 14, unemployment 7, m2 30
macro_features(macro: pd.DataFrame, index: pd.DatetimeIndex, cfg) -> pd.DataFrame
    """CRITICAL: monthly series must be shifted by their real publication lag
    BEFORE being reindexed onto the daily grid, otherwise the model sees CPI
    before it was published. Use reindex(index).ffill() only AFTER the lag."""

# regime.py
def fit_regime_model(features: pd.DataFrame, n_states=3, random_state=42) -> Any
def regime_features(curve, returns, n_states=3, window=252) -> pd.DataFrame
    """Rolling volatility regime + trend regime + inversion regime, all causal."""
```

---

## 6. `tqe.models` and `tqe.training`

```python
# models/base.py
class BaseModel(ABC):
    name: str
    def fit(self, X: np.ndarray | pd.DataFrame, y) -> "BaseModel"
    def predict(self, X) -> np.ndarray
    def save(self, path: Path) -> Path
    @classmethod
    def load(cls, path: Path) -> "BaseModel"
    @property
    def feature_importance(self) -> pd.Series | None

# models/registry.py
MODEL_REGISTRY: dict[str, type[BaseModel]]
def create_model(name: str, cfg: ModelConfig, **kw) -> BaseModel
def register(name)  # decorator
def save_bundle(models, scaler, metadata, path) -> Path
def load_bundle(path) -> tuple[...]

# models/linear.py    -> RidgeModel, ElasticNetModel, ARBaselineModel
# models/trees.py     -> RandomForestModel, GBMModel (sklearn HistGradientBoostingRegressor)
# models/ensemble.py  -> StackedEnsemble(base_models, meta_model)
#     fit(): out-of-fold predictions from base models via a PURGED time-series
#     split feed the meta learner. Never fit the meta learner on in-sample base preds.
# models/lstm.py      -> optional torch; import guarded, module must import fine without torch.

# training/splits.py
@dataclass(frozen=True)
class Split:
    train_idx: np.ndarray; test_idx: np.ndarray
    train_start, train_end, test_start, test_end   # pd.Timestamp

def walk_forward_splits(index, n_splits=8, test_size=252, min_train_size=1260,
                        embargo=5, expanding=True) -> list[Split]
def purged_kfold_splits(index, n_splits=5, embargo=5) -> list[Split]
    """Purging removes training samples whose target window overlaps the test
    window; the embargo additionally drops `embargo` observations immediately
    after the test block. Both are required when targets are forward-looking."""

# training/metrics.py
def regression_metrics(y_true, y_pred) -> dict     # rmse, mae, r2, directional_accuracy, ic, rank_ic
def information_coefficient(y_true, y_pred) -> float          # Pearson
def rank_information_coefficient(y_true, y_pred) -> float     # Spearman
def performance_metrics(returns: pd.Series, rf=0.0, periods=252) -> dict
    """total_return, cagr, ann_vol, sharpe, sortino, calmar, max_drawdown,
    max_dd_duration_days, hit_rate, profit_factor, skew, kurtosis, var_95,
    cvar_95, best_day, worst_day, ann_turnover(optional)"""
def drawdown_series(equity: pd.Series) -> pd.Series
def deflated_sharpe_ratio(sharpe, n_trials, n_obs, skew=0.0, kurtosis=3.0) -> float
    """Bailey & Lopez de Prado — the probability the Sharpe survives multiple
    testing. Any honest backtest that searched N configurations must report it."""
def probabilistic_sharpe_ratio(sharpe, benchmark_sr, n_obs, skew, kurtosis) -> float

# training/train.py
@dataclass
class TrainResult:
    model: BaseModel; scaler: Any; metrics: dict; fold_metrics: pd.DataFrame
    oos_predictions: pd.DataFrame     # index=date, columns=tenors, OOS only
    feature_importance: pd.DataFrame | None; config: dict
def train_walk_forward(fs: FeatureSet, cfg) -> TrainResult
def train_final_model(fs: FeatureSet, cfg) -> TrainResult   # fit on ALL data for live use
```

---

## 7. `tqe.signals`, `tqe.portfolio`, `tqe.backtest`

```python
# signals/alpha.py
def predictions_to_signal(preds: pd.DataFrame, method="zscore", window=252,
                          clip=3.0, min_abs=0.0) -> pd.DataFrame
    """Standardise raw forecasts into comparable, bounded signals using ONLY a
    trailing window (expanding stats would leak)."""
def blend_signals(signals: dict[str, pd.DataFrame], weights=None) -> pd.DataFrame
def signal_decay(signal, halflife) -> pd.DataFrame

# signals/sizing.py
def volatility_target_weights(signal, realised_vol, target_vol, max_leverage) -> pd.DataFrame
def kelly_size(edge, variance, fraction=0.25) -> float
def dv01_scaled_positions(signal, dv01_per_100, capital, target_dv01) -> pd.DataFrame

# portfolio/optimizer.py
@dataclass
class OptimizerResult:
    weights: pd.Series; expected_return: float; expected_vol: float
    dv01: float; gross: float; net: float; turnover: float; status: str
def mean_variance_weights(mu, cov, prev_w=None, cfg=...) -> OptimizerResult
    """Maximise mu'w - (lambda/2) w'Sw - turnover_penalty*|w - prev_w|_1 subject
    to gross/net DV01 caps and per-tenor bounds. Solve with scipy.optimize.minimize
    (SLSQP); fall back to a projected analytic solution if SLSQP fails."""
def risk_parity_weights(cov) -> pd.Series
def dv01_neutral_projection(weights, dv01) -> pd.Series

# portfolio/risk.py
def covariance(returns, method="ewma", halflife=63, shrinkage=0.1) -> pd.DataFrame
    """EWMA with Ledoit-Wolf style shrinkage toward a diagonal target."""
def parametric_var(weights, cov, confidence=0.99, horizon=1) -> float
def historical_var(portfolio_returns, confidence=0.99) -> float
def expected_shortfall(portfolio_returns, confidence=0.99) -> float
def stress_scenarios() -> dict[str, dict[str, float]]
    """Named historical shocks in bp per tenor: 1994 bond massacre, 2008 flight
    to quality, 2013 taper tantrum, 2020 COVID, 2022 hiking cycle, plus
    parallel +/-100bp, bear/bull steepener and flattener."""
def apply_stress(positions_dv01: dict[str, float], scenario: dict[str, float]) -> float
def risk_report(weights, returns, cov, positions_dv01=None) -> dict

# backtest/costs.py
@dataclass
class CostModel:
    cfg: CostConfig
    def half_spread(self, bucket: str) -> float          # in PRICE points, not 32nds
    def spread_cost(self, notional, bucket) -> float
    def impact_cost(self, notional, bucket, adv=None) -> float   # sqrt law
    def commission(self, notional) -> float
    def financing(self, notional, days, repo_rate) -> float
    def total_cost(self, trade_notional, bucket, adv=None) -> float
def turnover_cost_series(positions: pd.DataFrame, cost_model, buckets) -> pd.Series

# backtest/engine.py
@dataclass
class BacktestResult:
    equity: pd.Series; returns: pd.Series; positions: pd.DataFrame
    trades: pd.DataFrame; costs: pd.Series; metrics: dict
    exposures: pd.DataFrame; benchmark: pd.Series | None
    def summary(self) -> str
    def save(self, out_dir) -> Path
def run_backtest(signals, returns_panel, dv01_panel, cfg, cost_model=None,
                 benchmark=None) -> BacktestResult
    """Event-driven, day by day. On day t: use the signal computed from data up
    to t-1, size it, compute the trade against yesterday's position, charge
    costs on the traded notional, then apply day t's realised returns. The
    signal must NEVER see day t's return."""
# backtest/report.py -> tearsheet(result, out_dir) writing markdown + PNGs
```

---

## 8. `tqe.execution` and `tqe.live`

```python
# execution/broker.py
class OrderSide(str, Enum): BUY="buy"; SELL="sell"
class OrderType(str, Enum): MARKET="market"; LIMIT="limit"
class OrderStatus(str, Enum): NEW, SUBMITTED, PARTIAL, FILLED, CANCELLED, REJECTED
@dataclass
class Order:
    id: str; symbol: str; side: OrderSide; quantity: float
    order_type: OrderType = LIMIT; limit_price: float | None = None
    status: OrderStatus = NEW; filled_quantity: float = 0.0
    avg_fill_price: float = 0.0; created_at: datetime; updated_at: datetime
    tag: str = ""; reject_reason: str = ""
@dataclass
class Fill: order_id, symbol, side, quantity, price, timestamp, commission, slippage
@dataclass
class Position: symbol, quantity, avg_price, market_price, unrealized_pnl, realized_pnl
@dataclass
class AccountState: cash, equity, buying_power, positions: dict[str, Position], timestamp
class Broker(Protocol):
    def submit_order(self, order: Order) -> Order
    def cancel_order(self, order_id: str) -> bool
    def get_order(self, order_id: str) -> Order | None
    def list_orders(self, open_only=False) -> list[Order]
    def get_account(self) -> AccountState
    def get_positions(self) -> dict[str, Position]
    def get_quote(self, symbol: str) -> tuple[float, float]   # (bid, ask)
    def is_market_open(self) -> bool

# execution/paper.py -> PaperBroker(Broker): deterministic simulated fills with
#     configurable spread + slippage + partial fills; full position/PnL accounting;
#     persists state to JSON so a restart resumes the same book.
# execution/alpaca.py -> AlpacaBroker(Broker): REST adapter, import-guarded, never
#     required at import time. Reads keys from env. Defaults to the paper endpoint.
# execution/risk_gate.py
@dataclass
class RiskCheck: passed: bool; reason: str; checks: dict[str, bool]
class RiskGate:
    def __init__(self, cfg: RiskConfig, portfolio_cfg: PortfolioConfig)
    def check_order(self, order, account, positions, dv01_map=None) -> RiskCheck
    def check_portfolio(self, positions, dv01_map, equity, peak_equity) -> RiskCheck
    def trip(self, reason) / def reset(self) / def is_tripped -> bool
    """Hard limits: kill switch, max order notional, max position notional,
    gross/net DV01 caps, daily loss, drawdown, orders-per-day. Any failure blocks
    the order and is written to the audit log."""
# execution/oms.py
class OMS:
    """Order lifecycle + reconciliation. target_positions -> diff vs broker ->
    orders through the RiskGate -> submit -> track -> reconcile. Idempotent:
    re-running the same day must not double-trade. Persists to state_dir."""
    def reconcile(self) -> dict
    def generate_orders(self, targets: dict[str, float]) -> list[Order]
    def execute(self, orders: list[Order], dry_run=True) -> list[Order]
    def daily_run(self, targets, dry_run=True) -> dict

# live/runner.py
class LiveRunner:
    """Daily loop: pull data -> features -> load model -> predict -> signal ->
    size -> risk-check -> OMS. Writes a JSON audit trail and a daily report.
    `dry_run=True` by default; live trading requires an explicit flag."""
    def run_once(self, as_of=None, dry_run=True) -> dict
```

---

## 9. `tqe.cli` and `tqe.api`

```
tqe data pull [--force] [--start YEAR]
tqe data status
tqe curve fit  [--date YYYY-MM-DD] [--model svensson|nelson_siegel]
tqe features build [--out PATH]
tqe train [--model ...] [--walk-forward] [--out PATH]
tqe backtest [--start] [--end] [--no-costs] [--out DIR]
tqe predict [--date] [--model PATH]
tqe trade [--dry-run/--live] [--date]
tqe report [--backtest-dir]
tqe serve [--port 8000]
```

FastAPI (`api/server.py`): `GET /health`, `GET /curve/latest`, `GET /curve/fit`,
`POST /predict`, `GET /signals`, `GET /portfolio`, `GET /risk`, `GET /backtest/summary`,
`POST /trade/dry-run`. Model + data loaded once at startup into app state.

---

## 10. Testing standard

Every module ships with tests in `tests/test_<module>.py`. Numerical code is
tested against **closed-form or invariant checks**, never against a value copied
from its own output:

* zero-coupon Macaulay duration == maturity exactly,
* par bond prices to exactly 100 at its coupon,
* `sum(key_rate_durations) ≈ effective_duration`,
* NSS refits its own synthetic curve to < 0.5bp,
* bootstrapped zeros reprice the input par bonds back to 100,
* PCA on yield changes explains 77/13/5% (measured), PC1 loads positively everywhere,
* walk-forward splits never overlap and always respect the embargo,
* a backtest fed a signal shifted forward in time must produce a *worse* Sharpe
  than the correctly-lagged one (the look-ahead canary).


---

## 11. Implementation status

| Module | Status | Tests |
| --- | --- | --- |
| `pricing/` | complete | 101 |
| `curve/` | complete | 37 |
| `data/` | complete | covered via universe/backtest |
| `features/` | complete | 35 |
| `models/`, `training/` | complete | 62 |
| `signals/`, `portfolio/` | complete | covered via backtest + smoke |
| `backtest/` | complete | in `test_training.py` |
| `execution/` | complete (paper + Alpaca) | 25 |
| `live/`, `api/`, `cli.py` | complete | smoke test + manual |

260 tests, plus a 45-check end-to-end smoke test on synthetic data
(`scripts/smoke_test.py`) that needs no network and runs in about a minute.
