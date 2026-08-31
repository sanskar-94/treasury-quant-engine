"""Day-count conventions used in the US Treasury market.

The conventions that actually matter here:

``ACT/ACT (ICMA)``
    Accrued interest on Treasury notes and bonds.  The year fraction inside a
    coupon period is ``days_accrued / days_in_period`` where both counts are
    actual calendar days and the period is the *actual* coupon period the
    settlement date falls in.  This is the convention the Treasury's own
    ``Price / Yield`` formulas use.

``ACT/360``
    Quoted discount rate on Treasury bills, and repo.

``ACT/365F``
    Bond-equivalent yield on bills with more than 182 days to maturity, and the
    convention most analytics libraries use for a generic "actual" year.

``30/360``
    Included for completeness / comparison with corporate conventions; not used
    for Treasuries but frequently asked about, and used by the swap leg helpers.
"""

from __future__ import annotations

from datetime import date, timedelta
from enum import Enum


class DayCount(str, Enum):
    ACT_ACT_ICMA = "ACT/ACT ICMA"
    ACT_360 = "ACT/360"
    ACT_365F = "ACT/365F"
    THIRTY_360 = "30/360"


def days_between(start: date, end: date) -> int:
    """Actual calendar days between two dates (may be negative)."""
    return (end - start).days


def is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def _thirty_360_days(start: date, end: date) -> int:
    """US (NASD) 30/360 day difference."""
    d1, d2 = start.day, end.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    return 360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)


def year_fraction(
    start: date,
    end: date,
    convention: DayCount | str = DayCount.ACT_365F,
    period_start: date | None = None,
    period_end: date | None = None,
    frequency: int = 2,
) -> float:
    """Year fraction between ``start`` and ``end`` under ``convention``.

    For :attr:`DayCount.ACT_ACT_ICMA` the surrounding coupon period must be
    supplied via ``period_start`` / ``period_end``; the result is then
    ``(end - start) / (period_end - period_start) / frequency`` expressed in
    years.  Without a period the function falls back to a reference-period-free
    ACT/ACT (ISDA-style) calculation that splits the interval by calendar year.
    """
    conv = DayCount(convention) if not isinstance(convention, DayCount) else convention

    if conv is DayCount.ACT_360:
        return days_between(start, end) / 360.0
    if conv is DayCount.ACT_365F:
        return days_between(start, end) / 365.0
    if conv is DayCount.THIRTY_360:
        return _thirty_360_days(start, end) / 360.0

    # ---- ACT/ACT ---- #
    if period_start is not None and period_end is not None:
        period_days = days_between(period_start, period_end)
        if period_days <= 0:
            raise ValueError("Coupon period must have positive length")
        return days_between(start, end) / period_days / frequency

    # ACT/ACT ISDA: split across calendar years by their actual lengths.
    if end < start:
        return -year_fraction(end, start, conv)
    if start.year == end.year:
        return days_between(start, end) / (366.0 if is_leap(start.year) else 365.0)

    total = 0.0
    # Stub from `start` to the first 1 Jan.
    first_new_year = date(start.year + 1, 1, 1)
    total += days_between(start, first_new_year) / (366.0 if is_leap(start.year) else 365.0)
    # Whole years in between.
    total += end.year - start.year - 1
    # Stub from the last 1 Jan to `end`.
    last_new_year = date(end.year, 1, 1)
    total += days_between(last_new_year, end) / (366.0 if is_leap(end.year) else 365.0)
    return total


def accrual_fraction(
    settlement: date,
    period_start: date,
    period_end: date,
) -> float:
    """Fraction of the current coupon period that has *elapsed* at settlement.

    Returns a value in ``[0, 1)``; ``1 - fraction`` is the ``w`` used in the
    street price/yield formula.
    """
    period_days = days_between(period_start, period_end)
    if period_days <= 0:
        raise ValueError("Coupon period must have positive length")
    elapsed = days_between(period_start, settlement)
    return elapsed / period_days


# --------------------------------------------------------------------------- #
# Date arithmetic helpers used by the coupon schedule generator
# --------------------------------------------------------------------------- #
def add_months(d: date, months: int) -> date:
    """Add calendar months, clamping the day to the end of the target month.

    Treasury coupon dates roll on the same day-of-month as maturity, clamped for
    short months (a 31 Aug maturity pays on 28/29 Feb).
    """
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    day = min(d.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year + (month // 12), month % 12 + 1, 1) - timedelta(days=1)).day
