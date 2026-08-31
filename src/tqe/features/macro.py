"""Macro-economic features, with publication lags applied honestly.

This module exists because of one specific, very expensive mistake.

CPI for January is stamped ``1990-01-01`` in FRED, but it is not *published*
until roughly the middle of February.  Reindexing that series onto a daily grid
and forward-filling gives a model that knows January inflation on 2 January -
three weeks before the market did.  Backtests built that way look superb and
lose money immediately in production.

Every series here is therefore shifted by its real release lag **before** being
placed on the daily grid.  The lags are conservative (rounded up), because
being a few days late costs a little signal while being one day early
invalidates the entire result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger("features.macro")

__all__ = [
    "PUBLICATION_LAG_DAYS",
    "macro_features",
    "apply_publication_lag",
]

# Calendar days between a series' reference period and its public release.
# Sources: BLS/BEA/Federal Reserve release calendars.  Daily market series
# (VIX, breakevens, the dollar index) are same-day observable and get a lag of
# 1 day only because the close is not tradable until the next session.
PUBLICATION_LAG_DAYS: dict[str, int] = {
    # Monthly statistical releases, stamped to the FIRST day of the reference month.
    "cpi_yoy": 45,          # January CPI lands ~13 Feb -> 44 days after 1 Jan
    "core_cpi": 45,
    "unemployment": 38,     # employment report, first Friday of the next month
    "industrial_prod": 47,  # ~15th of the following month
    "m2": 60,               # H.6 release, roughly four weeks after month end
    "recession": 400,       # NBER dates are declared with a ~1 year lag. Non-negotiable.
    # Daily market data - observable at the close, actionable next session.
    "fed_funds": 2,         # effective rate is published the following morning
    "breakeven_10y": 1,
    "breakeven_5y": 1,
    "real_10y": 1,
    "term_premium_proxy": 1,
    "credit_spread": 1,
    "hy_spread": 2,         # ICE index, published with a one-day lag
    "vix": 1,
    "sp500": 1,
    "dollar_index": 2,      # H.10 is released with a lag
}

DEFAULT_LAG_DAYS = 30  # anything unrecognised is treated as a monthly release


def apply_publication_lag(
    macro: pd.DataFrame,
    lags: dict[str, int] | None = None,
    default_lag: int = DEFAULT_LAG_DAYS,
) -> pd.DataFrame:
    """Shift each column forward in time by its publication lag.

    The shift is by *calendar days on the index*, not by a number of rows, which
    matters because the macro frame mixes daily and monthly series - shifting a
    monthly series by "45 rows" would move it forward by almost four years.

    Returns a frame whose value stamped at date ``d`` was genuinely public on
    ``d``.
    """
    lags = lags or PUBLICATION_LAG_DAYS
    out = {}
    for col in macro.columns:
        lag = int(lags.get(col, default_lag))
        s = macro[col].dropna()
        if s.empty:
            out[col] = macro[col]
            continue
        shifted = s.copy()
        shifted.index = s.index + pd.Timedelta(days=lag)
        out[col] = shifted
    frame = pd.DataFrame(out)
    frame.index.name = macro.index.name or "date"
    return frame.sort_index()


def macro_features(
    macro: pd.DataFrame,
    index: pd.DatetimeIndex,
    lags: dict[str, int] | None = None,
    transforms: bool = True,
) -> pd.DataFrame:
    """Build the macro feature block on the daily trading grid.

    Parameters
    ----------
    macro:
        Raw FRED bundle, mixed frequency, stamped to the reference period.
    index:
        Target daily (trading-day) index.
    lags:
        Override the publication lags.
    transforms:
        Also emit year-on-year changes for level series (CPI as an index is
        meaningless to a model; CPI inflation is the signal), plus trailing
        changes for the market series.

    Returns
    -------
    pd.DataFrame
        Indexed by ``index``.  Empty (but correctly indexed) if ``macro`` is
        empty, so the caller never has to special-case a missing FRED pull.

    Notes
    -----
    The order of operations is the whole point and must not be rearranged:

    1. apply the publication lag on the *native* index,
    2. reindex onto the daily grid with ``ffill``,
    3. only then compute transforms.

    Doing (2) before (1) forward-fills a value into days before it existed.
    """
    if macro is None or macro.empty:
        return pd.DataFrame(index=index)

    lagged = apply_publication_lag(macro, lags)

    # Step 2: onto the daily grid.  ffill is correct HERE and only here - the
    # most recent *published* value is genuinely the market's best information
    # until the next release.
    union = lagged.index.union(index)
    daily = lagged.reindex(union).ffill().reindex(index)

    out: dict[str, pd.Series] = {f"macro_{c}": daily[c] for c in daily.columns}

    if transforms:
        # Year-on-year inflation from the CPI index level.
        for col in ("cpi_yoy", "core_cpi", "m2", "industrial_prod"):
            if col in daily.columns:
                s = daily[col]
                out[f"macro_{col}_yoy"] = s / s.shift(252) - 1.0
                out[f"macro_{col}_chg63"] = s.diff(63)

        # Trailing changes for the market-priced series.
        for col in ("fed_funds", "vix", "credit_spread", "hy_spread", "breakeven_10y",
                    "breakeven_5y", "real_10y", "term_premium_proxy", "dollar_index"):
            if col in daily.columns:
                s = daily[col]
                out[f"macro_{col}_chg21"] = s.diff(21)
                out[f"macro_{col}_chg63"] = s.diff(63)

        # Equity momentum as a risk-appetite proxy.
        if "sp500" in daily.columns:
            s = daily["sp500"]
            out["macro_sp500_ret21"] = s.pct_change(21)
            out["macro_sp500_ret63"] = s.pct_change(63)
            out["macro_sp500_vol21"] = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(252)

        # Real policy stance: nominal funds less trailing inflation.
        if "fed_funds" in daily.columns and "cpi_yoy" in daily.columns:
            infl = daily["cpi_yoy"] / daily["cpi_yoy"].shift(252) - 1.0
            out["macro_real_policy_rate"] = daily["fed_funds"] / 100.0 - infl

    frame = pd.DataFrame(out, index=index)
    frame.index.name = index.name or "date"

    n_all_nan = int(frame.isna().all().sum())
    if n_all_nan:
        log.info("macro block: %d/%d columns are entirely NaN over this window",
                 n_all_nan, frame.shape[1])
    return frame
