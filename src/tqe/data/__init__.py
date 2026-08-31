"""Market data: acquisition, the trading calendar, the tradable universe, caching.

The layering is deliberate and one-directional::

    sources   -> raw Treasury CMT par yields and the FRED macro bundle
    calendar  -> trading days and settlement conventions
    universe  -> par yields turned into investable total returns and DV01s
    cache     -> Parquet memoisation for anything expensive above

Nothing in this package forward-fills a missing observation and nothing looks
ahead: every derived value at date ``t`` is a function of raw data at ``t`` and
earlier.  See :mod:`tqe.data.universe` for the total-return construction, which
is where the whole system's P&L comes from.
"""

from __future__ import annotations

from .cache import (
    cache_info,
    cache_path,
    cached,
    clear_cache,
    load_frame,
    save_frame,
)
from .calendar import (
    annualization_factor,
    business_day_range,
    business_days_between,
    holidays_for_year,
    is_business_day,
    next_business_day,
    previous_business_day,
    settlement_date,
    trading_index,
)
from .sources import (
    TENOR_YEARS,
    DataFetchError,
    clean_curve,
    curve_coverage,
    fetch_fred_bundle,
    fetch_treasury_curve,
    load_market_data,
)
from .universe import (
    ANALYTICS_COLUMNS,
    CORE_SPECS,
    SPEC_BY_LABEL,
    TenorSpec,
    bucket_for_years,
    build_universe,
    butterfly_weights,
    constant_maturity_total_return,
    tenor_buckets,
    universe_panel,
)

__all__ = [
    # sources
    "TENOR_YEARS",
    "DataFetchError",
    "clean_curve",
    "curve_coverage",
    "fetch_fred_bundle",
    "fetch_treasury_curve",
    "load_market_data",
    # calendar
    "annualization_factor",
    "business_day_range",
    "business_days_between",
    "holidays_for_year",
    "is_business_day",
    "next_business_day",
    "previous_business_day",
    "settlement_date",
    "trading_index",
    # universe
    "ANALYTICS_COLUMNS",
    "CORE_SPECS",
    "SPEC_BY_LABEL",
    "TenorSpec",
    "bucket_for_years",
    "build_universe",
    "butterfly_weights",
    "constant_maturity_total_return",
    "tenor_buckets",
    "universe_panel",
    # cache
    "cache_info",
    "cache_path",
    "cached",
    "clear_cache",
    "load_frame",
    "save_frame",
]
