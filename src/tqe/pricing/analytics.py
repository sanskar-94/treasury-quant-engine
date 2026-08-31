"""Fixed-income risk analytics: duration, convexity, DV01, key-rate durations,
carry and roll-down.

Two families of measure live here and they are deliberately kept separate:

*Analytic* measures (:func:`macaulay_duration`, :func:`convexity`, ...) are
closed-form derivatives of the price/yield formula with respect to the bond's
*own* yield.  They are exact, fast and are what a desk quotes.

*Effective* measures (:func:`effective_duration`, :func:`key_rate_durations`)
re-price the bond off a shifted **curve**.  These are what you hedge with when a
portfolio spans several tenors, because a parallel yield shift is not the same
thing as a parallel zero-curve shift once coupons differ.

The identity ``sum(key_rate_durations) ~= effective_duration`` is asserted in the
test-suite; it is the standard sanity check that a KRD implementation is right.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from .bond import (
    Bond,
    accrued_interest,
    dirty_price_from_yield,
    price_from_discount_curve,
    price_from_yield,
    yield_from_price,
)

__all__ = [
    "BondRisk",
    "macaulay_duration",
    "modified_duration",
    "convexity",
    "dv01",
    "pvbp",
    "effective_duration",
    "effective_convexity",
    "key_rate_durations",
    "bond_risk",
    "carry_and_rolldown",
    "price_change_estimate",
    "portfolio_dv01",
    "hedge_ratio",
]

DEFAULT_KEY_TENORS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)


@dataclass(frozen=True)
class BondRisk:
    """A full risk snapshot for one bond at one settlement date."""

    clean_price: float
    dirty_price: float
    accrued: float
    ytm: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    dv01: float
    time_to_maturity: float

    def as_dict(self) -> dict[str, float]:
        return {
            "clean_price": self.clean_price,
            "dirty_price": self.dirty_price,
            "accrued": self.accrued,
            "ytm": self.ytm,
            "macaulay_duration": self.macaulay_duration,
            "modified_duration": self.modified_duration,
            "convexity": self.convexity,
            "dv01": self.dv01,
            "time_to_maturity": self.time_to_maturity,
        }


# --------------------------------------------------------------------------- #
# Analytic (yield-based) measures
# --------------------------------------------------------------------------- #
def _pv_components(bond: Bond, settlement: date, ytm: float) -> tuple[np.ndarray, np.ndarray, float]:
    """``(times_in_years, present_values, dirty_price)``."""
    times = bond.cashflow_times(settlement)
    amounts = bond.cashflow_amounts(settlement)
    d = ytm / bond.frequency
    if d <= -1.0:
        raise ValueError(f"Yield {ytm} implies a non-positive discount factor")
    pv = amounts * (1.0 + d) ** (-times * bond.frequency)
    return times, pv, float(pv.sum())


def macaulay_duration(bond: Bond, settlement: date, ytm: float) -> float:
    """PV-weighted average time to cashflow, in years."""
    times, pv, dirty = _pv_components(bond, settlement, ytm)
    if dirty <= 0 or times.size == 0:
        return 0.0
    return float(np.sum(times * pv) / dirty)


def modified_duration(bond: Bond, settlement: date, ytm: float) -> float:
    """``-1/P * dP/dy``; the standard first-order price sensitivity."""
    return macaulay_duration(bond, settlement, ytm) / (1.0 + ytm / bond.frequency)


def convexity(bond: Bond, settlement: date, ytm: float) -> float:
    """``1/P * d2P/dy2`` in years-squared."""
    times, pv, dirty = _pv_components(bond, settlement, ytm)
    if dirty <= 0 or times.size == 0:
        return 0.0
    d = 1.0 + ytm / bond.frequency
    return float(np.sum(pv * times * (times + 1.0 / bond.frequency)) / (dirty * d * d))


def dv01(bond: Bond, settlement: date, ytm: float, face: float | None = None) -> float:
    """Dollar value of a basis point, for ``face`` of notional.

    Returned as a **positive** number: the price loss from a +1bp yield move.
    With the default ``face=None`` the bond's own face (100) is used, so the
    result is "dollars per 100 face", the market convention.
    """
    _, _, dirty = _pv_components(bond, settlement, ytm)
    mod = modified_duration(bond, settlement, ytm)
    scale = 1.0 if face is None else face / bond.face
    return mod * dirty * 1e-4 * scale


def pvbp(bond: Bond, settlement: date, ytm: float, face: float | None = None) -> float:
    """Price value of a basis point - an alias for :func:`dv01`."""
    return dv01(bond, settlement, ytm, face)


# --------------------------------------------------------------------------- #
# Effective (curve-based) measures
# --------------------------------------------------------------------------- #
def effective_duration(
    bond: Bond,
    settlement: date,
    ytm: float,
    bump_bp: float = 1.0,
) -> float:
    """Central-difference duration from a parallel yield bump.

    Agrees with :func:`modified_duration` to ~1e-6 for an option-free bond; the
    difference is the second-order term the analytic formula drops.
    """
    h = bump_bp * 1e-4
    p_up = dirty_price_from_yield(bond, settlement, ytm + h)
    p_dn = dirty_price_from_yield(bond, settlement, ytm - h)
    p_0 = dirty_price_from_yield(bond, settlement, ytm)
    if p_0 <= 0:
        return 0.0
    return float((p_dn - p_up) / (2.0 * h * p_0))


def effective_convexity(
    bond: Bond,
    settlement: date,
    ytm: float,
    bump_bp: float = 1.0,
) -> float:
    h = bump_bp * 1e-4
    p_up = dirty_price_from_yield(bond, settlement, ytm + h)
    p_dn = dirty_price_from_yield(bond, settlement, ytm - h)
    p_0 = dirty_price_from_yield(bond, settlement, ytm)
    if p_0 <= 0:
        return 0.0
    return float((p_up + p_dn - 2.0 * p_0) / (h * h * p_0))


def _tent_weight(t: float, key_tenors: Sequence[float], idx: int) -> float:
    """Triangular ('tent') interpolation weight of key tenor ``idx`` at time ``t``.

    Weights across all key tenors sum to 1 for any ``t`` inside the grid, which
    is exactly why the key-rate durations add up to the parallel duration.
    """
    k = key_tenors[idx]
    if t <= key_tenors[0]:
        return 1.0 if idx == 0 else 0.0
    if t >= key_tenors[-1]:
        return 1.0 if idx == len(key_tenors) - 1 else 0.0
    if idx > 0 and key_tenors[idx - 1] <= t <= k:
        left = key_tenors[idx - 1]
        return (t - left) / (k - left)
    if idx < len(key_tenors) - 1 and k <= t <= key_tenors[idx + 1]:
        right = key_tenors[idx + 1]
        return (right - t) / (right - k)
    return 0.0


def key_rate_durations(
    bond: Bond,
    settlement: date,
    zero_rate: Callable[[float], float],
    key_tenors: Sequence[float] = DEFAULT_KEY_TENORS,
    bump_bp: float = 1.0,
    compounding: int = 2,
) -> dict[float, float]:
    """Partial durations to a 1bp bump at each key tenor of the **zero** curve.

    ``zero_rate(t)`` must return the continuously-quoted-per-``compounding``
    zero rate (decimal) for maturity ``t`` in years.  Bumps are applied with tent
    weights so that a simultaneous bump of every key rate reproduces a parallel
    shift; consequently ``sum(result.values()) ~= effective_duration``.
    """
    h = bump_bp * 1e-4
    tenors = list(key_tenors)

    def make_discount(shift_idx: int | None, sign: float) -> Callable[[float], float]:
        def disc(t: float) -> float:
            r = zero_rate(t)
            if shift_idx is not None:
                r += sign * h * _tent_weight(t, tenors, shift_idx)
            return (1.0 + r / compounding) ** (-compounding * t)

        return disc

    base = price_from_discount_curve(bond, settlement, make_discount(None, 0.0), clean=False)
    if base <= 0:
        return dict.fromkeys(tenors, 0.0)

    out: dict[float, float] = {}
    for i, tenor in enumerate(tenors):
        up = price_from_discount_curve(bond, settlement, make_discount(i, +1.0), clean=False)
        dn = price_from_discount_curve(bond, settlement, make_discount(i, -1.0), clean=False)
        out[tenor] = float((dn - up) / (2.0 * h * base))
    return out


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #
def bond_risk(
    bond: Bond,
    settlement: date,
    ytm: float | None = None,
    clean_price: float | None = None,
) -> BondRisk:
    """Full risk snapshot from either a yield or a price.

    Exactly one of ``ytm`` / ``clean_price`` is required.
    """
    if (ytm is None) == (clean_price is None):
        raise ValueError("Provide exactly one of ytm or clean_price")
    if ytm is None:
        ytm = yield_from_price(bond, settlement, float(clean_price))
    clean = price_from_yield(bond, settlement, ytm)
    ai = accrued_interest(bond, settlement)
    return BondRisk(
        clean_price=clean,
        dirty_price=clean + ai,
        accrued=ai,
        ytm=ytm,
        macaulay_duration=macaulay_duration(bond, settlement, ytm),
        modified_duration=modified_duration(bond, settlement, ytm),
        convexity=convexity(bond, settlement, ytm),
        dv01=dv01(bond, settlement, ytm),
        time_to_maturity=bond.time_to_maturity(settlement),
    )


def price_change_estimate(
    risk: BondRisk, delta_yield: float, include_convexity: bool = True
) -> float:
    """Second-order Taylor approximation of the dirty-price change.

    ``dP ~= -D_mod * P * dy + 0.5 * C * P * dy^2``
    """
    linear = -risk.modified_duration * risk.dirty_price * delta_yield
    if not include_convexity:
        return linear
    return linear + 0.5 * risk.convexity * risk.dirty_price * delta_yield * delta_yield


def carry_and_rolldown(
    bond: Bond,
    settlement: date,
    ytm: float,
    horizon_days: int = 90,
    repo_rate: float = 0.0,
    forward_yield: float | None = None,
) -> dict[str, float]:
    """Decompose the expected holding-period return into carry and roll-down.

    * **Carry** - coupon accrued over the horizon less the financing cost of the
      dirty price at ``repo_rate`` (ACT/360, the repo convention).
    * **Roll-down** - the price change from the bond simply becoming shorter,
      holding the yield curve fixed, i.e. repricing at ``forward_yield`` (the
      curve's yield for the shortened maturity) instead of ``ytm``.

    Together they are the return you earn if the curve does not move - the
    hurdle any directional forecast has to clear.
    """
    from .daycount import add_months  # local import keeps the module import graph flat

    horizon_date = date.fromordinal(settlement.toordinal() + int(horizon_days))
    if bond.n_remaining(horizon_date) == 0:
        return {"carry": 0.0, "rolldown": 0.0, "total": 0.0, "financing": 0.0, "coupon_income": 0.0}

    ai_now = accrued_interest(bond, settlement)
    ai_then = accrued_interest(bond, horizon_date)
    # Coupons actually paid between the two dates.
    paid = sum(amt for d, amt in bond.cashflows(settlement) if settlement < d <= horizon_date)
    # A coupon payment resets accrued to ~0, so income is (accrual change + coupons paid).
    coupon_income = (ai_then - ai_now) + paid

    dirty_now = price_from_yield(bond, settlement, ytm) + ai_now
    financing = dirty_now * repo_rate * horizon_days / 360.0
    carry = coupon_income - financing

    y_fwd = ytm if forward_yield is None else forward_yield
    clean_then = price_from_yield(bond, horizon_date, y_fwd)
    clean_now = price_from_yield(bond, settlement, ytm)
    rolldown = clean_then - clean_now

    del add_months  # silence linters; kept for signature parity with curve helpers
    return {
        "carry": float(carry),
        "rolldown": float(rolldown),
        "total": float(carry + rolldown),
        "financing": float(financing),
        "coupon_income": float(coupon_income),
    }


# --------------------------------------------------------------------------- #
# Portfolio level
# --------------------------------------------------------------------------- #
def portfolio_dv01(positions: dict[str, float], dv01_per_unit: dict[str, float]) -> float:
    """Signed net DV01 of a book.

    ``positions`` maps instrument -> face notional (negative = short);
    ``dv01_per_unit`` maps instrument -> DV01 per 100 face.
    """
    total = 0.0
    for key, notional in positions.items():
        unit = dv01_per_unit.get(key)
        if unit is None:
            continue
        total += notional / 100.0 * unit
    return float(total)


def hedge_ratio(target_dv01: float, hedge_dv01: float) -> float:
    """Face of the hedge instrument needed to neutralise ``target_dv01``.

    Returns the *negative* of the DV01 ratio so that adding
    ``hedge_ratio * hedge_face`` flattens the book.
    """
    if abs(hedge_dv01) < 1e-12:
        raise ValueError("Hedge instrument has ~zero DV01")
    return -target_dv01 / hedge_dv01
