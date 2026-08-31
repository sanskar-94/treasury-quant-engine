"""The tradable universe: turning CMT par yields into realised total returns.

This is the bridge between *market data* and *P&L*.  Everything downstream -
features, targets, the optimiser, the backtest - is denominated in the numbers
this module produces, so the replication method matters more here than anywhere
else in the system.

The economics
-------------
Treasury.gov publishes the **Constant Maturity Treasury** (CMT) series: the
*par yield* of a hypothetical on-the-run security at each tenor.  "Par yield"
means precisely this: a semi-annual bond of that maturity whose coupon equals
the quoted rate settles at exactly 100.  A yield series alone is not investable
- you cannot hold 4.73% - so it has to be converted into the return of an
actual position.

The standard replication (what the Bloomberg/ICE CMT total-return indices do)
is a **daily-roll constant-maturity portfolio**:

1. At the close of day ``t-1`` you buy the par bond
   ``par_bond(settle_{t-1}, T, y_{t-1})``.  By construction its clean price is
   100 and its accrued interest is 0 (settlement falls on the dated date).
2. Overnight nothing about the *instrument* changes.  On day ``t`` you mark
   **that same bond** - same maturity, same coupon - at the new market yield
   ``y_t``, settling ``settle_t``.  It is now a day shorter and carries a day of
   accrued interest.
3. The clean price move is the capital gain/loss; the change in accrued interest
   is the coupon carry actually earned.  Their sum is the realised overnight
   total return.
4. At the close of day ``t`` you roll: sell the seasoned bond, buy the new
   on-the-run par bond, and repeat.  The roll is P&L-neutral in this
   idealisation (it is where the transaction-cost model in
   :mod:`tqe.backtest.costs` bites in the real backtest).

The single most common bug in home-made bond backtests is rebuilding the bond at
the *new* date before repricing - that silently prices a brand-new par bond at
the new yield, gets 100 again, and reports a total return of exactly the carry.
The tell is a return series with no volatility.  Step 2 above is the whole
point: **do not rebuild the bond**.

Causality / look-ahead
----------------------
Row ``t`` of every frame produced here contains the return realised **over**
``(t-1, t]`` - it is the payoff of a position that was already on at the close of
``t-1``.  Nothing in the frame is shifted backwards in time: every column at row
``t`` is a function of ``curve`` rows ``t-1`` and ``t`` only, and never of any
row after ``t``.  Consequently a backtest that multiplies ``total_return[t]`` by
a position sized from information up to ``t-1`` is exactly correct, and a
feature builder must *lag* these columns by at least one day before using them
as predictors.  The risk columns (``duration``/``dv01``/``convexity``) describe
the bond you would buy at the close of day ``t``, i.e. the position you carry
into ``t+1``; hedge ratios applied to ``total_return[t]`` therefore want
``dv01.shift(1)``.

Ragged coverage
---------------
The CMT panel has real holes: the 30Y was not published between Feb-2002 and
Feb-2006, the 20Y starts in Oct-1993, the short bills start in 2001/2018/2022.
Returns are computed by *positional* shift on the trading-day index, so a
missing quote on either leg produces ``NaN`` rather than a fabricated return
that bridges a four-year publication gap.  Nothing is forward-filled here.

Performance
-----------
Calling the scalar :mod:`tqe.pricing` API once per (tenor, day) would mean
~80,000 coupon-schedule constructions and ~80,000 Python-level discounting
loops.  Instead the schedule geometry is derived analytically (a par bond's
coupon dates are just month arithmetic off its maturity) with vectorised civil
date maths, and the whole history is priced as a single ``(n_days, n_coupons)``
cashflow grid per tenor.  The full 9,172 x 9 panel builds in ~1 second.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .calendar import settlement_date
from .sources import TENOR_YEARS

__all__ = [
    "TenorSpec",
    "CORE_SPECS",
    "SPEC_BY_LABEL",
    "ANALYTICS_COLUMNS",
    "bucket_for_years",
    "build_universe",
    "tenor_buckets",
    "constant_maturity_total_return",
    "universe_panel",
    "butterfly_weights",
]

log = get_logger("data.universe")

FACE: float = 100.0
"""Treasuries are quoted per 100 of face; every price/DV01 here is per 100."""

FREQUENCY: int = 2
"""US Treasury notes and bonds pay semi-annually."""

ANALYTICS_COLUMNS: tuple[str, ...] = (
    "yield",
    "price",
    "dirty_price",
    "duration",
    "dv01",
    "convexity",
    "carry_1d",
    "price_return",
    "total_return",
    "yield_change",
)


# --------------------------------------------------------------------------- #
# Instrument specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TenorSpec:
    """One tradable point on the curve.

    Parameters
    ----------
    label:
        Treasury CMT column name, e.g. ``"10 Yr"``.
    years:
        Nominal maturity in years (``10.0``).
    bucket:
        Liquidity/cost bucket used by :class:`tqe.backtest.costs.CostModel`.
        One of ``"bill"``, ``"2y"``, ``"5y"``, ``"10y"``, ``"30y"``.  Buckets
        exist because bid/ask is a step function of the sector, not a smooth
        function of maturity: the on-the-run 10y trades in half a 32nd while the
        30y is a full 32nd wide and off-the-run bills are quoted in a tenth.
    """

    label: str
    years: float
    bucket: str

    @property
    def months(self) -> int:
        """Maturity in whole calendar months - the unit coupon schedules roll on."""
        return int(round(self.years * 12))

    @property
    def n_coupons(self) -> int:
        """Coupons remaining on a freshly issued par bond of this tenor.

        A bond whose maturity is fewer than six months away still lives inside a
        *quasi* coupon period (the Treasury's own convention for short paper):
        one terminal cashflow, discounted over a fractional period.
        """
        return max(1, -(-self.months // (12 // FREQUENCY)))  # ceil division

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.label} ({self.years:g}y, {self.bucket})"


def bucket_for_years(years: float) -> str:
    """Map a maturity to its transaction-cost bucket.

    The cut points follow how the cash desk is actually organised: money-market
    (bills and the 52-week, all sub-1y), the 2s/3s front end, the 5y belly, the
    7s/10s, and the 20s/30s long end.
    """
    if years <= 1.0:
        return "bill"
    if years <= 3.5:
        return "2y"
    if years <= 6.5:
        return "5y"
    if years <= 13.0:
        return "10y"
    return "30y"


def _spec(label: str) -> TenorSpec:
    years = TENOR_YEARS[label]
    return TenorSpec(label=label, years=float(years), bucket=bucket_for_years(float(years)))


# The nine tenors quoted continuously since 1990 (``cfg.data.core_tenors``).
# The 20Y is deliberately excluded from the default universe: it is missing
# 1990-1993 and its liquidity is materially worse than the 10s and 30s.
CORE_SPECS: tuple[TenorSpec, ...] = tuple(
    _spec(label)
    for label in ("3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "30 Yr")
)

SPEC_BY_LABEL: dict[str, TenorSpec] = {label: _spec(label) for label in TENOR_YEARS}


def build_universe(
    curve: pd.DataFrame,
    tenors: Sequence[str] | None = None,
    min_coverage: float = 0.5,
) -> list[TenorSpec]:
    """Tenors with enough coverage to trade, ordered short -> long.

    Parameters
    ----------
    curve:
        Par-yield panel, ``DatetimeIndex`` named ``date``, decimal rates.
    tenors:
        Explicit labels to consider.  ``None`` uses :data:`CORE_SPECS`.
    min_coverage:
        Minimum fraction of rows that must carry a quote.  A tenor quoted on
        fewer than half the sample cannot support a continuously-held position,
        and admitting it would make every cross-sectional statistic a
        composition effect (2 Mo, for example, only exists from 2018 and would
        otherwise silently restrict the whole panel to the post-2018 sample).

    Returns
    -------
    list[TenorSpec]
        Sorted ascending by maturity.

    Notes
    -----
    Causality: this inspects only the *presence* of quotes, never their values,
    and is meant to be called once on the full sample to fix the instrument set.
    Selecting tenors by realised performance would be look-ahead of the worst
    kind; selecting them by data availability is a data-engineering decision, not
    a forecast.
    """
    if curve.empty:
        return []
    labels = [s.label for s in CORE_SPECS] if tenors is None else list(tenors)

    n_rows = len(curve)
    kept: list[TenorSpec] = []
    for label in labels:
        if label not in curve.columns:
            log.warning("tenor %r requested but absent from the curve panel", label)
            continue
        if label not in TENOR_YEARS:
            raise KeyError(f"Unknown Treasury tenor label {label!r}; expected one of {sorted(TENOR_YEARS)}")
        coverage = float(curve[label].notna().sum()) / n_rows
        if coverage < min_coverage:
            log.info("dropping %r from the universe: coverage %.1f%% < %.0f%%",
                     label, 100 * coverage, 100 * min_coverage)
            continue
        kept.append(SPEC_BY_LABEL[label])

    return sorted(kept, key=lambda s: s.years)


def tenor_buckets(specs: Sequence[TenorSpec]) -> dict[str, str]:
    """``{label: cost bucket}`` - the mapping the cost model and OMS need."""
    return {s.label: s.bucket for s in specs}


# --------------------------------------------------------------------------- #
# Vectorised civil-date arithmetic
# --------------------------------------------------------------------------- #
# Coupon schedules are pure month arithmetic off the maturity date, so the whole
# 9,172-day history of schedule geometry can be computed with array ops instead
# of 250,000 `datetime` round-trips.  These mirror `daycount.add_months` (which
# clamps the day-of-month) and `date.toordinal()` exactly; the test-suite asserts
# agreement against the scalar implementations.
_DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int64)
_ORDINAL_EPOCH_OFFSET = 719163  # date(1970, 1, 1).toordinal()


def _is_leap(year: np.ndarray) -> np.ndarray:
    return ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)


def _days_in_month(year: np.ndarray, month: np.ndarray) -> np.ndarray:
    base = _DAYS_IN_MONTH[month - 1]
    return np.where((month == 2) & _is_leap(year), 29, base)


def _add_months(ymd: tuple[np.ndarray, np.ndarray, np.ndarray], months: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised :func:`tqe.pricing.daycount.add_months`.

    Treasury coupon dates roll on maturity's day-of-month, clamped into short
    months (a 31-Aug maturity pays on 28/29 Feb).
    """
    year, month, day = ymd
    total = (month - 1) + months
    new_year = year + np.floor_divide(total, 12)
    new_month = np.mod(total, 12) + 1
    new_day = np.minimum(day, _days_in_month(new_year, new_month))
    return new_year, new_month, new_day


def _to_ordinal(ymd: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Proleptic-Gregorian day number, identical to :meth:`datetime.date.toordinal`.

    Hinnant's ``days_from_civil``: shift the year to start in March so the
    leap day lands at the end, then the month-length pattern is exactly
    ``(153*m + 2) // 5``.  Branch-free, so it vectorises.
    """
    year, month, day = ymd
    y = year - (month <= 2)
    era = np.floor_divide(y, 400)
    yoe = y - era * 400
    mp = np.mod(month + 9, 12)  # March -> 0
    doy = np.floor_divide(153 * mp + 2, 5) + day - 1
    doe = yoe * 365 + np.floor_divide(yoe, 4) - np.floor_divide(yoe, 100) + doy
    return era * 146097 + doe - 719468 + _ORDINAL_EPOCH_OFFSET


def _dates_to_ymd(dates: Sequence[date]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.array([(d.year, d.month, d.day) for d in dates], dtype=np.int64)
    return arr[:, 0], arr[:, 1], arr[:, 2]


# --------------------------------------------------------------------------- #
# Vectorised price / risk kernel
# --------------------------------------------------------------------------- #
def _pv_grid(coupon: np.ndarray, w: np.ndarray, ytm: np.ndarray, n: int) -> np.ndarray:
    """Present values of every cashflow, for every day, in one array.

    Implements the street price formula

    .. math::

        P_{dirty} = \\sum_{k=0}^{n-1} A_k\\,(1 + y/f)^{-(k + w)}

    where ``w`` is the fraction of the current coupon period still remaining and
    ``A_k`` is the coupon (plus redemption on the last flow).  Exponents are
    ``k + w`` in *periods*, which is what makes the formula agree with
    :func:`tqe.pricing.bond.dirty_price_from_yield` to machine precision.

    Parameters
    ----------
    coupon:
        ``(n_days,)`` semi-annual coupon amount per 100 face.
    w:
        ``(n_days,)`` fraction of the coupon period remaining at settlement.
    ytm:
        ``(n_days,)`` semi-annually compounded yield, decimal.
    n:
        Number of remaining cashflows (constant across the history because the
        bond is rebuilt at a constant maturity every day).

    Returns
    -------
    np.ndarray
        ``(n_days, n)`` present values.  Rows whose inputs are not finite are
        ``NaN``; NaNs are substituted out *before* the power so the exponentiation
        never raises or warns on invalid input.
    """
    valid = np.isfinite(coupon) & np.isfinite(w) & np.isfinite(ytm)
    d = np.where(valid, ytm, 0.0) / FREQUENCY
    if np.any(d[valid] <= -1.0):
        raise ValueError("Yield below -200% implies a non-positive discount factor")

    k = np.arange(n, dtype=float)
    exponent = k[None, :] + np.where(valid, w, 0.0)[:, None]  # (days, n), in periods
    discount = np.power(1.0 + d[:, None], -exponent)

    pv = np.where(valid, coupon, 0.0)[:, None] * discount
    pv[:, -1] += FACE * discount[:, -1]  # redemption rides on the final coupon
    pv[~valid, :] = np.nan
    return pv


def _dirty_from_pv(pv: np.ndarray) -> np.ndarray:
    return pv.sum(axis=1)


def _risk_from_pv(pv: np.ndarray, w: np.ndarray, ytm: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(modified_duration, dv01, convexity)`` from a cashflow-PV grid.

    Mirrors :mod:`tqe.pricing.analytics` exactly:

    * Macaulay ``D = sum(t_i PV_i) / P``  with ``t_i`` in **years**,
    * modified ``D_mod = D / (1 + y/f)``,
    * convexity ``C = sum(PV_i t_i (t_i + 1/f)) / (P (1 + y/f)^2)``,
    * ``DV01 = D_mod * P_dirty * 1e-4``, positive, per 100 face.

    DV01 is quoted off the **dirty** price because that is the money actually at
    risk; on a par bond bought at issue the two coincide.
    """
    dirty = _dirty_from_pv(pv)
    k = np.arange(n, dtype=float)
    tau = (k[None, :] + w[:, None]) / FREQUENCY  # cashflow times in years

    with np.errstate(invalid="ignore", divide="ignore"):
        macaulay = np.sum(tau * pv, axis=1) / dirty
        one_plus_d = 1.0 + ytm / FREQUENCY
        modified = macaulay / one_plus_d
        convexity = np.sum(pv * tau * (tau + 1.0 / FREQUENCY), axis=1) / (dirty * one_plus_d**2)
    dv01 = modified * dirty * 1e-4
    return modified, dv01, convexity


def _shift1(a: np.ndarray) -> np.ndarray:
    """Lag by one row, filling the first observation with ``NaN``.

    Used for both legs of the return: row ``t`` of the result is the value from
    row ``t-1``.  The float cast is deliberate - integer ordinals have to become
    floats to carry the leading ``NaN``, and that ``NaN`` is what guarantees the
    first day of the sample reports no return instead of a fabricated one.
    """
    out = np.empty(a.shape, dtype=float)
    out[0] = np.nan
    out[1:] = a[:-1]
    return out


# --------------------------------------------------------------------------- #
# The main event
# --------------------------------------------------------------------------- #
def constant_maturity_total_return(
    curve: pd.DataFrame,
    tenors: Sequence[str] | None = None,
    settlement_lag: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Daily total returns and risk analytics for the constant-maturity universe.

    Parameters
    ----------
    curve:
        Par-yield panel: ``DatetimeIndex`` named ``date``, ascending, decimal
        rates, ``NaN`` where a tenor was not published.
    tenors:
        Labels to build.  ``None`` runs :func:`build_universe` on the default
        core set.
    settlement_lag:
        Override the settlement convention (``None`` = the historically correct
        T+2 before 2024-05-28 and T+1 after, via
        :func:`tqe.data.calendar.settlement_date`).  Exposed only so tests can
        pin a convention; production should leave it ``None``.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{tenor_label: frame}``.  Each frame is indexed by **trade date** and
        carries :data:`ANALYTICS_COLUMNS`:

        ==================  ====================================================
        ``yield``           par yield quoted that day (decimal)
        ``price``           clean price of that day's synthetic par bond (100)
        ``dirty_price``     clean + accrued (also 100 - issued at settlement)
        ``duration``        modified duration of that day's bond, years
        ``dv01``            positive dollars per +1bp, per 100 face
        ``convexity``       years squared
        ``carry_1d``        accrual earned overnight, per 100 face
        ``price_return``    clean price return of *yesterday's* bond
        ``total_return``    ``price_return + carry_1d / 100``
        ``yield_change``    ``y_t - y_{t-1}`` in decimal (0.0001 = 1bp)
        ==================  ====================================================

    Notes
    -----
    **Method.**  For consecutive trading days ``t-1, t``:

    1. build ``B = par_bond(settle_{t-1}, T, y_{t-1})`` - clean price exactly
       100, accrued exactly 0;
    2. reprice **that same** ``B`` (unchanged maturity, unchanged coupon) at
       ``y_t`` settling ``settle_t``;
    3. ``price_return = (clean_t - clean_{t-1}) / clean_{t-1}``;
    4. ``carry_1d = accrued_t - accrued_{t-1}``;
    5. ``total_return = price_return + carry_1d / 100``.

    Both legs settle through :func:`tqe.data.calendar.settlement_date`, so the
    accrual window is the true settlement gap - three calendar days over a
    weekend, and exactly zero on 2024-05-28 when the market moved from T+2 to
    T+1 and two consecutive trade dates settled on the same day.

    **Look-ahead.**  Row ``t`` uses ``curve`` rows ``t-1`` and ``t`` and nothing
    else.  The return at row ``t`` is realised P&L on a position that existed at
    the close of ``t-1``; the risk columns at row ``t`` describe the bond bought
    at the close of ``t``.  No column is shifted backwards, so a strategy that
    multiplies a signal formed at ``t-1`` by ``total_return[t]`` is causal by
    construction.

    **Gaps.**  Returns come from a *positional* shift on the trading index, so a
    tenor that stops being published (the 30Y, Feb-2002 to Feb-2006) simply
    produces ``NaN`` on both the last day before and the first day after the
    hole.  Nothing is interpolated or forward-filled across it.

    **Exactness.**  The clean price at issue is 100 to machine precision whenever
    the tenor is a whole number of coupon periods.  For the 3-month point (and
    any tenor that is not a multiple of six months) the Treasury's *quasi coupon
    period* convention puts settlement mid-period, and the clean price lands
    within a couple of cents of par - a genuine convention artefact, not an
    error.  Returns divide by the actual computed base price rather than
    assuming 100, so they stay internally consistent either way.
    """
    if curve.empty:
        return {}
    if not isinstance(curve.index, pd.DatetimeIndex):
        raise TypeError("curve must be indexed by a DatetimeIndex")
    if not curve.index.is_monotonic_increasing:
        raise ValueError("curve index must be ascending; sort before pricing")
    if curve.index.has_duplicates:
        raise ValueError("curve index has duplicate dates; returns would be ill-defined")

    specs = build_universe(curve, tenors)
    if not specs:
        log.warning("no tenor met the coverage bar; returning an empty universe")
        return {}

    index = curve.index
    # Settlement is a property of the trade date alone, so resolve it once for
    # the whole history and reuse across every tenor.
    trade_dates = [ts.date() for ts in index]
    settle_dates = [settlement_date(d, settlement_lag) for d in trade_dates]
    settle_ymd = _dates_to_ymd(settle_dates)
    settle_ord = _to_ordinal(settle_ymd)
    if np.any(np.diff(settle_ord) < 0):
        raise ValueError("settlement dates are not monotonic; the calendar is inconsistent")

    out: dict[str, pd.DataFrame] = {}
    for spec in specs:
        out[spec.label] = _tenor_frame(curve[spec.label], index, spec, settle_ymd, settle_ord)
    return out


def _tenor_frame(
    yields: pd.Series,
    index: pd.DatetimeIndex,
    spec: TenorSpec,
    settle_ymd: tuple[np.ndarray, np.ndarray, np.ndarray],
    settle_ord: np.ndarray,
) -> pd.DataFrame:
    """Build one tenor's analytics frame with two vectorised pricing passes."""
    n = spec.n_coupons
    step = 12 // FREQUENCY

    # ---- schedule geometry -------------------------------------------------
    # A par bond issued on `settle` matures `months` later; its coupon dates are
    # generated *backward from maturity* (the Treasury convention), so every date
    # is `add_months(maturity, -6k)` and the clamping never compounds.
    maturity = _add_months(settle_ymd, spec.months)
    first_coupon = _add_months(maturity, -step * (n - 1))
    period_start = _add_months(first_coupon, -step)

    first_coupon_ord = _to_ordinal(first_coupon)
    period_days = (first_coupon_ord - _to_ordinal(period_start)).astype(float)
    if np.any(period_days <= 0):
        raise ValueError(f"{spec.label}: degenerate coupon period in the generated schedule")

    # Fraction of the coupon period still *remaining* at settlement.  Equal to
    # 1.0 (accrued = 0) whenever the tenor is a whole number of coupon periods -
    # settlement then coincides with the dated date.
    w_own = (first_coupon_ord - settle_ord) / period_days

    y = yields.to_numpy(dtype=float, copy=False)
    coupon = FACE * y / FREQUENCY  # semi-annual coupon per 100 face == par yield

    # ---- leg 1: today's freshly issued par bond, for risk -------------------
    pv_own = _pv_grid(coupon, w_own, y, n)
    dirty_own = _dirty_from_pv(pv_own)
    accrued_own = coupon * (1.0 - w_own)
    clean_own = dirty_own - accrued_own
    duration, dv01, convexity = _risk_from_pv(pv_own, w_own, y, n)

    # ---- leg 2: yesterday's bond, repriced at today's yield -----------------
    # This is the whole method.  The *coupon* and *maturity* come from row t-1;
    # only the yield and the settlement date advance.  Rebuilding the bond here
    # would collapse the price return to zero.
    coupon_prev = _shift1(coupon)
    first_coupon_prev = _shift1(first_coupon_ord)
    period_days_prev = _shift1(period_days)

    # A settlement that jumped past yesterday's first coupon would change the
    # cashflow count and invalidate the fixed-`n` grid.  It cannot happen (the
    # nearest coupon is at least a month out) but a silent mispricing here would
    # be invisible in the output, so it is checked rather than assumed.
    crossed = np.zeros(len(index), dtype=bool)
    crossed[1:] = settle_ord[1:] >= first_coupon_prev[1:]
    if crossed.any():
        raise ValueError(
            f"{spec.label}: settlement crossed a coupon date on "
            f"{index[crossed][0].date()} - the fixed cashflow grid is invalid"
        )

    w_reprice = (first_coupon_prev - settle_ord) / period_days_prev
    pv_reprice = _pv_grid(coupon_prev, w_reprice, y, n)
    dirty_reprice = _dirty_from_pv(pv_reprice)
    accrued_reprice = coupon_prev * (1.0 - w_reprice)
    clean_reprice = dirty_reprice - accrued_reprice

    # ---- realised P&L -------------------------------------------------------
    clean_base = _shift1(clean_own)
    accrued_base = _shift1(accrued_own)

    with np.errstate(invalid="ignore", divide="ignore"):
        price_return = clean_reprice / clean_base - 1.0
    carry_1d = accrued_reprice - accrued_base
    total_return = price_return + carry_1d / FACE
    yield_change = y - _shift1(y)

    frame = pd.DataFrame(
        {
            "yield": y,
            "price": clean_own,
            "dirty_price": dirty_own,
            "duration": duration,
            "dv01": dv01,
            "convexity": convexity,
            "carry_1d": carry_1d,
            "price_return": price_return,
            "total_return": total_return,
            "yield_change": yield_change,
        },
        index=index,
    )
    frame.index.name = "date"
    frame.attrs["tenor"] = spec.label
    frame.attrs["years"] = spec.years
    frame.attrs["bucket"] = spec.bucket
    return frame[list(ANALYTICS_COLUMNS)]


# --------------------------------------------------------------------------- #
# Panels and relative-value weights
# --------------------------------------------------------------------------- #
def universe_panel(returns: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    """Pivot ``{tenor: frame}`` into a single frame of one field.

    Parameters
    ----------
    returns:
        Output of :func:`constant_maturity_total_return`.
    field:
        Column to extract, e.g. ``"total_return"`` or ``"dv01"``.

    Returns
    -------
    pd.DataFrame
        Index = union of the per-tenor dates (they share one index in practice),
        columns = tenors ordered short -> long.  ``NaN`` is preserved: it marks
        days the tenor was not quoted and must stay visible so the portfolio
        layer can drop the instrument rather than trade a stale price.

    Notes
    -----
    Causality: a pure reshape.  Row ``t`` of the panel is row ``t`` of each
    input frame, so whatever timing convention the field carries is unchanged.
    """
    if not returns:
        return pd.DataFrame()
    missing = [t for t, f in returns.items() if field not in f.columns]
    if missing:
        raise KeyError(f"field {field!r} missing from tenor frame(s): {missing}")

    ordered = sorted(returns, key=lambda t: TENOR_YEARS.get(t, float("inf")))
    panel = pd.concat([returns[t][field].rename(t) for t in ordered], axis=1)
    panel.index.name = "date"
    return panel


def butterfly_weights(
    short_dv01: float, belly_dv01: float, long_dv01: float
) -> tuple[float, float, float]:
    """50/50 DV01-neutral butterfly weights, belly normalised to ``+1`` unit.

    A butterfly isolates *curvature*: long the belly, short the wings, with the
    wing notionals chosen so the package has zero net DV01 and therefore no
    first-order exposure to a parallel shift.  The "50/50" split puts half the
    belly's risk on each wing, which is the desk default because it also leaves
    the trade roughly neutral to a pure steepening (the two wings' slope
    exposures cancel).

    Parameters
    ----------
    short_dv01, belly_dv01, long_dv01:
        DV01 per 100 face of each leg, positive, e.g. from the ``dv01`` column of
        :func:`constant_maturity_total_return`.

    Returns
    -------
    tuple[float, float, float]
        ``(w_short, w_belly, w_long)`` in units of face / 100.  The wings come
        back negative: ``w_short * short_dv01 + 1 * belly_dv01 +
        w_long * long_dv01 == 0``.

    Examples
    --------
    >>> w = butterfly_weights(1.9, 4.4, 8.5)
    >>> round(w[0] * 1.9 + w[1] * 4.4 + w[2] * 8.5, 12)
    0.0
    """
    if not np.isfinite([short_dv01, belly_dv01, long_dv01]).all():
        raise ValueError("butterfly legs need finite DV01s")
    if abs(short_dv01) < 1e-12 or abs(long_dv01) < 1e-12:
        raise ValueError("wing DV01 is ~zero; the fly cannot be hedged")
    half = 0.5 * belly_dv01
    return (-half / short_dv01, 1.0, -half / long_dv01)
