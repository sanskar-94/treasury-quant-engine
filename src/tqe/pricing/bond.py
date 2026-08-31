"""Cash Treasury instrument definitions, cashflows and price/yield conversion.

Everything follows the street conventions the Treasury publishes in its
*Price, Yield and Rate Calculations for a Treasury Note or Bond* memoranda:

* semi-annual coupons, dates generated **backward** from maturity,
* accrued interest on ACT/ACT (ICMA) within the actual coupon period,
* the dirty price of a note settling mid-period is

  .. math::

     P_{dirty} = \\sum_{k=0}^{n-1} \\frac{c}{(1+y/2)^{k+w}}
                 + \\frac{100}{(1+y/2)^{n-1+w}}

  where :math:`c` is the semi-annual coupon per 100 face and
  :math:`w = \\frac{\\text{days to next coupon}}{\\text{days in coupon period}}`,

* clean price = dirty price - accrued, and accrued = :math:`c\\,(1-w)`.

Bills are quoted on a bank-discount basis (ACT/360) and converted to a
bond-equivalent yield with the Treasury's own coupon-equivalent formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Iterable, Sequence

import numpy as np

from .daycount import add_months, days_between

__all__ = [
    "Bond",
    "accrued_interest",
    "dirty_price_from_yield",
    "price_from_yield",
    "yield_from_price",
    "price_from_discount_curve",
    "par_bond",
    "bill_price_from_discount",
    "bill_discount_from_price",
    "bill_bond_equivalent_yield",
    "format_32nds",
    "parse_32nds",
    "decimal_to_32nds_float",
]

_MAX_COUPONS = 4000  # 30y semi-annual = 60; generous guard against runaway loops


# --------------------------------------------------------------------------- #
# Instrument
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Bond:
    """A US Treasury note or bond.

    Parameters
    ----------
    maturity:
        Redemption date.
    coupon:
        Annual coupon **in percent** of face (``4.25`` means 4.25%).  Zero makes
        the instrument a zero-coupon / STRIP.
    face:
        Redemption amount per quoted unit; Treasuries quote per 100.
    frequency:
        Coupon payments per year (2 for Treasuries).
    issue_date:
        Optional dated date.  When supplied, coupon dates before it are dropped
        and the first period may be short or long.
    label:
        Free-form identifier carried through the portfolio and OMS.
    """

    maturity: date
    coupon: float
    face: float = 100.0
    frequency: int = 2
    issue_date: date | None = None
    first_coupon: date | None = None
    label: str = ""
    cusip: str = ""
    _schedule_cache: dict = field(default_factory=dict, repr=False, compare=False, hash=False)

    # ---------------- schedule ---------------- #
    def coupon_dates(self, after: date | None = None) -> list[date]:
        """All coupon dates, generated backward from maturity.

        If ``after`` is given only dates strictly greater than it are returned -
        this is exactly the set of cashflows a buyer settling on ``after``
        receives (a coupon paid *on* the settlement date belongs to the seller).
        """
        key = "all"
        sched = self._schedule_cache.get(key)
        if sched is None:
            step = 12 // self.frequency
            sched = []
            cur = self.maturity
            start_bound = self.issue_date or date(1900, 1, 1)
            for _ in range(_MAX_COUPONS):
                sched.append(cur)
                prev = add_months(self.maturity, -step * len(sched))
                if prev <= start_bound:
                    break
                cur = prev
            sched.reverse()
            if self.first_coupon is not None:
                sched = [d for d in sched if d >= self.first_coupon]
            self._schedule_cache[key] = sched
        if after is None:
            return list(sched)
        return [d for d in sched if d > after]

    def period_bounds(self, settlement: date) -> tuple[date, date]:
        """``(period_start, period_end)`` of the coupon period containing settlement."""
        remaining = self.coupon_dates(after=settlement)
        if not remaining:
            raise ValueError(f"Bond matured on {self.maturity}; settlement {settlement} is after it")
        nxt = remaining[0]
        step = 12 // self.frequency
        prev = add_months(nxt, -step)
        # Guard against a pathological schedule where clamping produced prev >= nxt.
        if prev >= nxt:
            prev = add_months(nxt, -step - 1)
        return prev, nxt

    def w(self, settlement: date) -> float:
        """Fraction of the current coupon period still **remaining** at settlement."""
        prev, nxt = self.period_bounds(settlement)
        total = days_between(prev, nxt)
        if total <= 0:
            raise ValueError("Degenerate coupon period")
        return days_between(settlement, nxt) / total

    def n_remaining(self, settlement: date) -> int:
        return len(self.coupon_dates(after=settlement))

    def time_to_maturity(self, settlement: date) -> float:
        """Years to maturity measured in coupon periods (the pricing time axis)."""
        n = self.n_remaining(settlement)
        if n == 0:
            return 0.0
        return (n - 1 + self.w(settlement)) / self.frequency

    @property
    def semi_coupon(self) -> float:
        """Coupon amount per period, per unit of ``face``."""
        return self.face * (self.coupon / 100.0) / self.frequency

    def cashflows(self, settlement: date) -> list[tuple[date, float]]:
        """Remaining ``(date, amount)`` cashflows, principal folded into the last."""
        dates = self.coupon_dates(after=settlement)
        if not dates:
            return []
        flows = [(d, self.semi_coupon) for d in dates]
        last_date, last_amt = flows[-1]
        flows[-1] = (last_date, last_amt + self.face)
        return flows

    def cashflow_times(self, settlement: date) -> np.ndarray:
        """Cashflow times in **years** on the street ``(k + w)/freq`` axis."""
        n = self.n_remaining(settlement)
        if n == 0:
            return np.zeros(0)
        w = self.w(settlement)
        return (np.arange(n) + w) / self.frequency

    def cashflow_amounts(self, settlement: date) -> np.ndarray:
        n = self.n_remaining(settlement)
        if n == 0:
            return np.zeros(0)
        amounts = np.full(n, self.semi_coupon, dtype=float)
        amounts[-1] += self.face
        return amounts

    def __hash__(self) -> int:  # _schedule_cache is excluded from equality/hash
        return hash((self.maturity, self.coupon, self.face, self.frequency, self.issue_date))


# --------------------------------------------------------------------------- #
# Accrued interest
# --------------------------------------------------------------------------- #
def accrued_interest(bond: Bond, settlement: date) -> float:
    """Accrued interest per unit face, ACT/ACT (ICMA)."""
    if bond.coupon == 0.0:
        return 0.0
    if bond.n_remaining(settlement) == 0:
        return 0.0
    return bond.semi_coupon * (1.0 - bond.w(settlement))


# --------------------------------------------------------------------------- #
# Price <-> yield
# --------------------------------------------------------------------------- #
def dirty_price_from_yield(bond: Bond, settlement: date, ytm: float) -> float:
    """Full (invoice) price per unit face given a semi-annually compounded yield.

    ``ytm`` is a decimal (0.0425 for 4.25%).  Yields at or below
    ``-frequency`` are rejected because the discount factor is undefined there.
    """
    n = bond.n_remaining(settlement)
    if n == 0:
        return 0.0
    d = ytm / bond.frequency
    if d <= -1.0:
        raise ValueError(f"Yield {ytm} implies a non-positive discount factor")
    times = bond.cashflow_times(settlement) * bond.frequency  # exponents in periods
    amounts = bond.cashflow_amounts(settlement)
    return float(np.sum(amounts * (1.0 + d) ** (-times)))


def price_from_yield(bond: Bond, settlement: date, ytm: float) -> float:
    """Clean price per unit face (the quoted price)."""
    return dirty_price_from_yield(bond, settlement, ytm) - accrued_interest(bond, settlement)


def _price_and_derivative(bond: Bond, settlement: date, ytm: float) -> tuple[float, float]:
    """Dirty price and ``dP/dy`` - shared by the Newton solver and analytics."""
    d = ytm / bond.frequency
    times = bond.cashflow_times(settlement) * bond.frequency
    amounts = bond.cashflow_amounts(settlement)
    disc = (1.0 + d) ** (-times)
    price = float(np.sum(amounts * disc))
    # dP/dy = -(1/f) * sum(amount * t_periods * (1+d)^-(t+1))
    deriv = float(-np.sum(amounts * times * disc / (1.0 + d)) / bond.frequency)
    return price, deriv


def yield_from_price(
    bond: Bond,
    settlement: date,
    clean_price: float,
    guess: float | None = None,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """Invert the price formula for the semi-annual yield to maturity.

    Newton-Raphson seeded with the current-yield approximation, falling back to a
    bracketed bisection when Newton leaves the admissible region.  The bisection
    fallback is what makes this safe on deep-discount long bonds and on the
    negative-yield regime that briefly appeared in 2020 bill quotes.
    """
    n = bond.n_remaining(settlement)
    if n == 0:
        raise ValueError("Cannot compute yield on a matured bond")
    target = clean_price + accrued_interest(bond, settlement)
    if target <= 0:
        raise ValueError(f"Non-positive dirty price {target}")

    ttm = max(bond.time_to_maturity(settlement), 1e-6)
    if guess is None:
        annual_coupon = bond.face * bond.coupon / 100.0
        guess = (annual_coupon + (bond.face - clean_price) / ttm) / max(
            (bond.face + clean_price) / 2.0, 1e-9
        )
        guess = float(np.clip(guess, -0.02, 1.0))

    y = guess
    for _ in range(max_iter):
        price, deriv = _price_and_derivative(bond, settlement, y)
        diff = price - target
        if abs(diff) < tol:
            return y
        if deriv == 0 or not math.isfinite(deriv):
            break
        step = diff / deriv
        y_new = y - step
        if not math.isfinite(y_new) or y_new <= -bond.frequency * 0.999:
            break
        if abs(y_new - y) < 1e-15:
            return y_new
        y = y_new
    else:
        return y

    # ---- bracketed fallback ---- #
    lo, hi = -0.95 * bond.frequency, 2.0
    f_lo = dirty_price_from_yield(bond, settlement, lo) - target
    f_hi = dirty_price_from_yield(bond, settlement, hi) - target
    expand = 0
    while f_lo * f_hi > 0 and expand < 20:
        hi *= 2.0
        f_hi = dirty_price_from_yield(bond, settlement, hi) - target
        expand += 1
    if f_lo * f_hi > 0:
        raise ValueError(
            f"Could not bracket a yield for clean price {clean_price} on {bond.label or bond.maturity}"
        )
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        f_mid = dirty_price_from_yield(bond, settlement, mid) - target
        if abs(f_mid) < tol or (hi - lo) < 1e-15:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Curve pricing
# --------------------------------------------------------------------------- #
def price_from_discount_curve(
    bond: Bond,
    settlement: date,
    discount: Callable[[float], float],
    clean: bool = True,
) -> float:
    """Price a bond off an arbitrary discount function ``discount(t_years)``.

    Used to value the whole universe consistently off one fitted curve, which is
    what makes rich/cheap (spread-to-curve) signals meaningful.
    """
    times = bond.cashflow_times(settlement)
    amounts = bond.cashflow_amounts(settlement)
    if times.size == 0:
        return 0.0
    dfs = np.asarray([discount(float(t)) for t in times], dtype=float)
    dirty = float(np.sum(amounts * dfs))
    return dirty - accrued_interest(bond, settlement) if clean else dirty


def par_bond(settlement: date, tenor_years: float, par_yield: float, face: float = 100.0) -> Bond:
    """Construct the synthetic on-the-run par bond used to replicate a CMT tenor.

    The Treasury's constant-maturity series *is* a par yield: a bond of that
    tenor whose coupon equals the quoted yield prices at exactly 100.  Building
    that bond each day and repricing yesterday's bond at today's yield is the
    standard way to turn a yield series into an investable total return.
    """
    months = int(round(tenor_years * 12))
    maturity = add_months(settlement, months)
    coupon = par_yield * 100.0 if abs(par_yield) < 1.0 else par_yield
    return Bond(
        maturity=maturity,
        coupon=float(coupon),
        face=face,
        frequency=2,
        issue_date=settlement,
        label=f"PAR-{tenor_years:g}Y",
    )


# --------------------------------------------------------------------------- #
# Bills
# --------------------------------------------------------------------------- #
def bill_price_from_discount(discount_rate: float, days_to_maturity: int, face: float = 100.0) -> float:
    """Bank-discount basis: ``P = F (1 - d * t/360)``."""
    return face * (1.0 - discount_rate * days_to_maturity / 360.0)


def bill_discount_from_price(price: float, days_to_maturity: int, face: float = 100.0) -> float:
    if days_to_maturity <= 0:
        raise ValueError("days_to_maturity must be positive")
    return (face - price) / face * 360.0 / days_to_maturity


def bill_bond_equivalent_yield(
    discount_rate: float, days_to_maturity: int, face: float = 100.0
) -> float:
    """Treasury coupon-equivalent yield of a bill.

    For <= 182 days the simple ACT/365 conversion applies.  Beyond that the
    Treasury uses the quadratic that accounts for the intervening coupon.
    """
    if days_to_maturity <= 0:
        raise ValueError("days_to_maturity must be positive")
    price = bill_price_from_discount(discount_rate, days_to_maturity, face)
    if days_to_maturity <= 182:
        return (face - price) / price * 365.0 / days_to_maturity

    t = days_to_maturity
    year = 366.0 if _spans_leap_day(t) else 365.0
    a = t / (2.0 * year) - 0.25
    b = t / year
    c = (price - face) / price
    disc = b * b - 4.0 * a * c
    if a == 0 or disc < 0:
        return (face - price) / price * 365.0 / t
    return (-b + math.sqrt(disc)) / (2.0 * a)


def _spans_leap_day(days: int) -> bool:
    """Conservative flag for the 366-day denominator on long bills."""
    return days > 365


# --------------------------------------------------------------------------- #
# 32nds quoting
# --------------------------------------------------------------------------- #
def format_32nds(price: float, ticks: int = 32) -> str:
    """Render a decimal price the way a Treasury desk quotes it.

    ``99.515625 -> '99-16+'`` (99 and 16.5/32).  Eighths of a 32nd are shown as
    ``+`` (a half) or a trailing digit for quarter increments.
    """
    whole = math.floor(price)
    frac = price - whole
    thirty_seconds = frac * ticks
    base = math.floor(thirty_seconds + 1e-9)
    remainder = thirty_seconds - base
    eighths = round(remainder * 8)
    if eighths == 8:
        base += 1
        eighths = 0
    if base >= ticks:
        whole += 1
        base -= ticks
    suffix = ""
    if eighths == 4:
        suffix = "+"
    elif eighths != 0:
        suffix = str(eighths)
    return f"{whole}-{base:02d}{suffix}"


def parse_32nds(quote: str, ticks: int = 32) -> float:
    """Inverse of :func:`format_32nds`; also accepts a plain decimal string."""
    q = quote.strip()
    if "-" not in q:
        return float(q)
    whole_str, frac_str = q.split("-", 1)
    whole = float(whole_str)
    suffix = 0.0
    if frac_str.endswith("+"):
        suffix, frac_str = 0.5, frac_str[:-1]
    elif len(frac_str) == 3:
        suffix, frac_str = int(frac_str[2]) / 8.0, frac_str[:2]
    thirty_seconds = float(frac_str) + suffix
    sign = -1.0 if whole_str.strip().startswith("-") else 1.0
    return whole + sign * thirty_seconds / ticks


def decimal_to_32nds_float(price: float, ticks: int = 32) -> float:
    """Price expressed in ticks - convenient for cost models quoted in 32nds."""
    return price * ticks


def price_series_from_yields(
    bond_factory: Callable[[date, float], Bond],
    settlements: Sequence[date],
    yields: Iterable[float],
) -> list[float]:
    """Helper: clean prices for a sequence of (settlement, yield) observations."""
    out = []
    for settle, y in zip(settlements, yields):
        bond = bond_factory(settle, y)
        out.append(price_from_yield(bond, settle, y))
    return out
