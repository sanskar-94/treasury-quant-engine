"""US bond-market (SIFMA) trading calendar.

The Treasury cash market observes the Federal Reserve holiday schedule *plus*
Good Friday, which the equity market also observes but which the Federal Reserve
does not.  Getting this right matters for two reasons:

* settlement dates (T+1 for Treasuries since May 2024, T+2 before that), and
* the walk-forward backtest, which must step on real trading days so that a
  "1 business day ahead" target is genuinely one trading session ahead.

Holidays are computed rather than tabulated so the calendar keeps working for
future dates without maintenance.
"""

from __future__ import annotations

import functools
from datetime import date, timedelta

import numpy as np
import pandas as pd

# Treasury settlement moved from T+2 to T+1 on 28 May 2024 (SEC rule 15c6-1).
T_PLUS_ONE_EFFECTIVE = date(2024, 5, 28)


def easter_sunday(year: int) -> date:
    """Gregorian Easter (Anonymous / Meeus-Jones-Butcher algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month, day = divmod(h + m - 7 * n + 114, 31)
    return date(year, month, day + 1)


def good_friday(year: int) -> date:
    return easter_sunday(year) - timedelta(days=2)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """``n``-th ``weekday`` (Mon=0) of a month; ``n=-1`` means the last one."""
    if n > 0:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))
    # last occurrence
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    last = nxt - timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - timedelta(days=offset)


def _observed(d: date) -> date:
    """Federal observance rule: Saturday -> Friday, Sunday -> Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@functools.lru_cache(maxsize=512)
def holidays_for_year(year: int) -> frozenset[date]:
    """Full SIFMA/Treasury holiday set for a calendar year."""
    days: set[date] = set()

    days.add(_observed(date(year, 1, 1)))                       # New Year's Day
    if year >= 1986:
        days.add(_nth_weekday(year, 1, 0, 3))                   # MLK Jr. Day
    if year >= 1971:
        days.add(_nth_weekday(year, 2, 0, 3))                   # Washington's Birthday
    else:
        days.add(_observed(date(year, 2, 22)))
    days.add(good_friday(year))                                 # Good Friday (SIFMA)
    if year >= 1971:
        days.add(_nth_weekday(year, 5, 0, -1))                  # Memorial Day
    else:
        days.add(_observed(date(year, 5, 30)))
    if year >= 2021:
        days.add(_observed(date(year, 6, 19)))                  # Juneteenth
    days.add(_observed(date(year, 7, 4)))                       # Independence Day
    days.add(_nth_weekday(year, 9, 0, 1))                       # Labor Day
    if year >= 1971:
        days.add(_nth_weekday(year, 10, 0, 2))                  # Columbus Day
    days.add(_observed(date(year, 11, 11)))                     # Veterans Day
    days.add(_nth_weekday(year, 11, 3, 4))                      # Thanksgiving
    days.add(_observed(date(year, 12, 25)))                     # Christmas

    # One-off national days of mourning / closures the market actually observed.
    one_offs = {
        2001: [date(2001, 9, 11), date(2001, 9, 12), date(2001, 9, 13), date(2001, 9, 14)],
        2004: [date(2004, 6, 11)],   # President Reagan
        2007: [date(2007, 1, 2)],    # President Ford
        2012: [date(2012, 10, 30)],  # Hurricane Sandy
        2018: [date(2018, 12, 5)],   # President G.H.W. Bush
        2025: [date(2025, 1, 9)],    # President Carter
    }
    days.update(one_offs.get(year, []))
    return frozenset(days)


def is_holiday(d: date) -> bool:
    return d in holidays_for_year(d.year)


def is_business_day(d: date) -> bool:
    """Weekday that is not a bond-market holiday."""
    return d.weekday() < 5 and not is_holiday(d)


def next_business_day(d: date, n: int = 1) -> date:
    """Advance ``n`` business days (``n`` may be negative)."""
    step = 1 if n >= 0 else -1
    remaining = abs(n)
    cur = d
    while remaining:
        cur += timedelta(days=step)
        if is_business_day(cur):
            remaining -= 1
    return cur


def previous_business_day(d: date, n: int = 1) -> date:
    return next_business_day(d, -n)


def business_days_between(start: date, end: date) -> int:
    """Number of business days strictly after ``start`` up to and including ``end``."""
    if end < start:
        return -business_days_between(end, start)
    count, cur = 0, start
    while cur < end:
        cur += timedelta(days=1)
        if is_business_day(cur):
            count += 1
    return count


def business_day_range(start: date, end: date) -> list[date]:
    """All bond-market trading days in ``[start, end]``."""
    out, cur = [], start
    while cur <= end:
        if is_business_day(cur):
            out.append(cur)
        cur += timedelta(days=1)
    return out


def settlement_date(trade_date: date, lag: int | None = None) -> date:
    """Regular-way settlement for a cash Treasury trade.

    ``lag=None`` uses the historically correct convention: T+1 from 28 May 2024
    onward, T+2 before that.
    """
    if lag is None:
        lag = 1 if trade_date >= T_PLUS_ONE_EFFECTIVE else 2
    return next_business_day(trade_date, lag)


def trading_index(start: str | date, end: str | date) -> pd.DatetimeIndex:
    """A ``DatetimeIndex`` of bond-market trading days, for reindexing frames."""
    s = pd.Timestamp(start).date()
    e = pd.Timestamp(end).date()
    return pd.DatetimeIndex([pd.Timestamp(d) for d in business_day_range(s, e)], name="date")


def annualization_factor(index: pd.DatetimeIndex) -> float:
    """Trading days per year implied by an index - used to annualise vol/returns.

    Falls back to 252 when the index is too short to estimate reliably.
    """
    if len(index) < 60:
        return 252.0
    span_years = (index[-1] - index[0]).days / 365.25
    if span_years <= 0:
        return 252.0
    return float(np.clip(len(index) / span_years, 200.0, 262.0))
