"""Par -> zero -> forward curve bootstrapping.

The Treasury publishes a **par yield** curve (CMT): for each tenor, the coupon a
bond of that maturity would need to price at exactly 100.  Almost every useful
calculation - discounting an arbitrary cashflow, computing a forward rate,
valuing a bond that is not on the run - needs the **zero** (spot) curve instead.

Bootstrapping recovers it recursively.  A par bond of tenor :math:`T` with
semi-annual coupon :math:`c = y_T / 2` satisfies

.. math::

    100 = \\sum_{i=1}^{2T} \\frac{100 c}{(1 + z_{t_i}/2)^{2 t_i}}
          + \\frac{100}{(1 + z_T/2)^{2T}}

When every coupon date coincides with an already-solved node this rearranges to
a closed form.  It generally does **not**: between the quoted 10-year and
20-year points sit twenty unquoted coupon dates, and discounting them off the
10-year alone biases the 20-year solution by tens of basis points.  So each
tenor is instead solved by bisection on its own zero rate, with the intermediate
dates interpolated off the solved nodes *including the trial point* - a
self-consistent bootstrap.  Price is strictly monotone in the zero rate, so
bisection is unconditionally safe.

Intermediate coupon dates that fall between quoted tenors are interpolated
**log-linearly on the discount factor**, which is equivalent to piecewise-constant
forward rates and is the market-standard choice: it guarantees positive forwards
and never produces the saw-toothed forward curve that linear-on-zeros gives.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger("curve.bootstrap")

__all__ = [
    "par_to_zero",
    "zero_to_discount",
    "zero_to_forward",
    "discount_to_zero",
    "interpolate_curve",
    "forward_rate",
    "bootstrap_history",
    "zero_curve_function",
]

INTERP_METHODS = ("linear", "log_linear_df", "cubic", "monotone_cubic")


# --------------------------------------------------------------------------- #
# Conversions
# --------------------------------------------------------------------------- #
def zero_to_discount(
    tenors: Sequence[float] | np.ndarray,
    zeros: Sequence[float] | np.ndarray,
    frequency: int = 2,
) -> np.ndarray:
    """Discount factors from compounded zero rates: ``(1 + z/f)^(-f t)``."""
    t = np.asarray(tenors, dtype=float)
    z = np.asarray(zeros, dtype=float)
    base = 1.0 + z / frequency
    if np.any(base <= 0.0):
        raise ValueError("Zero rate implies a non-positive discount base")
    return base ** (-frequency * t)


def discount_to_zero(
    tenors: Sequence[float] | np.ndarray,
    discounts: Sequence[float] | np.ndarray,
    frequency: int = 2,
) -> np.ndarray:
    """Inverse of :func:`zero_to_discount`."""
    t = np.asarray(tenors, dtype=float)
    df = np.asarray(discounts, dtype=float)
    out = np.full(t.shape, np.nan)
    ok = (t > 0) & (df > 0)
    out[ok] = frequency * (df[ok] ** (-1.0 / (frequency * t[ok])) - 1.0)
    return out


def zero_to_forward(
    tenors: Sequence[float] | np.ndarray,
    zeros: Sequence[float] | np.ndarray,
    frequency: int = 2,
) -> np.ndarray:
    """Implied forward rates between consecutive tenors.

    ``f(t_i, t_{i+1})`` is the rate that makes rolling the shorter zero into the
    forward reproduce the longer zero exactly.  The first element repeats the
    first zero rate, since there is no interval before it.
    """
    t = np.asarray(tenors, dtype=float)
    df = zero_to_discount(t, zeros, frequency)
    fwd = np.empty_like(t)
    fwd[0] = np.asarray(zeros, dtype=float)[0]
    dt = np.diff(t)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = df[:-1] / df[1:]
        fwd[1:] = frequency * (ratio ** (1.0 / (frequency * dt)) - 1.0)
    return fwd


def forward_rate(
    zero_fn: Callable[[float], float],
    t1: float,
    t2: float,
    frequency: int = 2,
) -> float:
    """Forward rate between ``t1`` and ``t2`` implied by a zero-curve function."""
    if t2 <= t1:
        raise ValueError("t2 must be strictly greater than t1")
    df1 = (1.0 + zero_fn(t1) / frequency) ** (-frequency * t1)
    df2 = (1.0 + zero_fn(t2) / frequency) ** (-frequency * t2)
    return float(frequency * ((df1 / df2) ** (1.0 / (frequency * (t2 - t1))) - 1.0))


# --------------------------------------------------------------------------- #
# Interpolation
# --------------------------------------------------------------------------- #
def interpolate_curve(
    tenors: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    method: str = "linear",
    frequency: int = 2,
) -> np.ndarray:
    """Interpolate a curve onto ``targets``.

    Parameters
    ----------
    method:
        ``"linear"``
            Straight line in zero-rate space.  Simple, but produces a
            discontinuous forward curve at every node.
        ``"log_linear_df"``
            Linear in ``log(DF)``, i.e. piecewise-constant forward rates.  The
            market default, and the one bootstrapping uses internally.
        ``"cubic"``
            Natural cubic spline through the zeros - smooth forwards, but can
            overshoot and generate negative forwards on a kinked curve.
        ``"monotone_cubic"``
            PCHIP.  Smooth *and* shape-preserving, so it cannot invent a hump
            the quotes do not support.

    Extrapolation beyond the quoted range is flat in all methods; extrapolating a
    spline is how curve libraries produce absurd 50-year forwards.
    """
    t = np.asarray(tenors, dtype=float)
    v = np.asarray(values, dtype=float)
    tgt = np.asarray(targets, dtype=float)

    order = np.argsort(t)
    t, v = t[order], v[order]
    finite = np.isfinite(t) & np.isfinite(v)
    t, v = t[finite], v[finite]
    if t.size == 0:
        return np.full(tgt.shape, np.nan)
    if t.size == 1:
        return np.full(tgt.shape, v[0])

    if method == "linear":
        out = np.interp(tgt, t, v)
    elif method == "log_linear_df":
        # Interpolating log(DF) linearly == constant forward rate between nodes.
        df = zero_to_discount(t, v, frequency)
        log_df = np.log(df)
        interp_log_df = np.interp(tgt, t, log_df)
        out = np.where(tgt > 0, discount_to_zero(tgt, np.exp(interp_log_df), frequency), v[0])
    elif method in ("cubic", "monotone_cubic"):
        from scipy.interpolate import CubicSpline, PchipInterpolator

        spline = (
            PchipInterpolator(t, v, extrapolate=False)
            if method == "monotone_cubic"
            else CubicSpline(t, v, extrapolate=False)
        )
        out = spline(tgt)
        # Flat extrapolation outside the quoted range.
        out = np.where(tgt < t[0], v[0], out)
        out = np.where(tgt > t[-1], v[-1], out)
    else:
        raise ValueError(f"Unknown interpolation method {method!r}; choose from {INTERP_METHODS}")

    out = np.asarray(out, dtype=float)
    out = np.where(tgt < t[0], v[0], out)
    out = np.where(tgt > t[-1], v[-1], out)
    return out


# --------------------------------------------------------------------------- #
# Bootstrapping
# --------------------------------------------------------------------------- #
def par_to_zero(
    tenors_years: Sequence[float] | np.ndarray,
    par_yields: Sequence[float] | np.ndarray,
    frequency: int = 2,
    max_tenor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap semi-annually compounded zero rates from par yields.

    Tenors shorter than one coupon period are treated as single-payment money
    market instruments (a 3-month bill has no intermediate coupon), so their zero
    rate equals the quoted rate.  From the first tenor with an intermediate
    coupon onward, the recursion above applies, with intermediate discount
    factors interpolated log-linearly from the zeros solved so far.

    Parameters
    ----------
    tenors_years:
        Maturities in years, any order; NaNs are dropped.
    par_yields:
        Par (CMT) yields as decimals.
    frequency:
        Coupon frequency, 2 for Treasuries.
    max_tenor:
        Optionally truncate the curve.

    Returns
    -------
    (tenors, zeros):
        Sorted maturities and their zero rates.

    Notes
    -----
    The bootstrap is exact by construction: repricing each input par bond off the
    resulting zeros returns 100 to machine precision.  ``tests/test_curve.py``
    asserts exactly that, because a bootstrap that does not round-trip is worse
    than useless - it silently biases every downstream valuation.
    """
    t = np.asarray(tenors_years, dtype=float)
    y = np.asarray(par_yields, dtype=float)
    if t.shape != y.shape:
        raise ValueError("tenors and yields must have the same shape")

    keep = np.isfinite(t) & np.isfinite(y) & (t > 0)
    t, y = t[keep], y[keep]
    if max_tenor is not None:
        keep = t <= max_tenor
        t, y = t[keep], y[keep]
    if t.size == 0:
        return np.zeros(0), np.zeros(0)

    order = np.argsort(t)
    t, y = t[order], y[order]
    # Collapse duplicate tenors (the Treasury file has none, but be defensive).
    t, idx = np.unique(t, return_index=True)
    y = y[idx]

    period = 1.0 / frequency
    zeros = np.full(t.size, np.nan)
    solved_t: list[float] = []
    solved_z: list[float] = []

    def par_price(ti: float, coupon: float, z_trial: float) -> float:
        """Price the tenor-``ti`` par bond given a trial zero rate at ``ti``.

        Intermediate coupon dates are discounted off the curve built from the
        already-solved nodes *plus the trial node itself*.  Including the trial
        node is what makes the bootstrap self-consistent when quoted tenors are
        far apart: between the 10y and the 20y there are twenty coupon dates and
        no quotes, so pricing them off the 10y alone (flat extrapolation) throws
        the 20y solution out by tens of basis points.
        """
        n_cf = int(np.floor(ti * frequency + 1e-9))
        inter = ti - period * np.arange(n_cf - 1, 0, -1)
        inter = inter[inter > 0]

        knots_t = np.asarray(solved_t + [ti], dtype=float)
        knots_z = np.asarray(solved_z + [z_trial], dtype=float)
        order_k = np.argsort(knots_t)
        knots_t, knots_z = knots_t[order_k], knots_z[order_k]

        if inter.size:
            z_inter = interpolate_curve(
                knots_t, knots_z, inter, method="log_linear_df", frequency=frequency
            )
            annuity = float(np.sum(zero_to_discount(inter, z_inter, frequency)))
        else:
            annuity = 0.0
        df_final = float(zero_to_discount(np.array([ti]), np.array([z_trial]), frequency)[0])
        return coupon * (annuity + df_final) + df_final

    for i, (ti, yi) in enumerate(zip(t, y)):
        n_cf = int(np.floor(ti * frequency + 1e-9))
        if n_cf <= 1:
            # Single cashflow (bills, and any tenor inside one coupon period):
            # there is nothing to bootstrap, the par rate *is* the zero rate.
            zeros[i] = yi
            solved_t.append(float(ti))
            solved_z.append(float(yi))
            continue

        coupon = yi / frequency
        # Solve par_price(z) == 1.0 for the trial zero.  Price is strictly
        # decreasing in the zero rate, so a bracketed bisection is unconditionally
        # safe - far more robust here than Newton, which can step outside the
        # admissible region on a steeply inverted curve.
        lo, hi = max(yi - 0.05, -0.9 * frequency + 1e-9), yi + 0.05
        f_lo, f_hi = par_price(ti, coupon, lo) - 1.0, par_price(ti, coupon, hi) - 1.0
        expand = 0
        while f_lo * f_hi > 0.0 and expand < 40:
            lo = max(lo - 0.05, -0.9 * frequency + 1e-9)
            hi += 0.05
            f_lo = par_price(ti, coupon, lo) - 1.0
            f_hi = par_price(ti, coupon, hi) - 1.0
            expand += 1
        if f_lo * f_hi > 0.0:
            log.warning("could not bracket the zero rate at tenor %.3f; skipping", ti)
            continue

        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f_mid = par_price(ti, coupon, mid) - 1.0
            if abs(f_mid) < 1e-15 or (hi - lo) < 1e-15:
                break
            if f_lo * f_mid <= 0.0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        z_solved = 0.5 * (lo + hi)

        zeros[i] = z_solved
        solved_t.append(float(ti))
        solved_z.append(float(z_solved))

    return t, zeros


def zero_curve_function(
    tenors: Sequence[float] | np.ndarray,
    zeros: Sequence[float] | np.ndarray,
    method: str = "log_linear_df",
    frequency: int = 2,
) -> Callable[[float], float]:
    """Wrap a discrete zero curve as a callable ``z(t)``.

    This is the shape :func:`tqe.pricing.analytics.key_rate_durations` and
    :func:`tqe.pricing.bond.price_from_discount_curve` expect.
    """
    t = np.asarray(tenors, dtype=float)
    z = np.asarray(zeros, dtype=float)
    finite = np.isfinite(t) & np.isfinite(z)
    t, z = t[finite], z[finite]

    def fn(x: float) -> float:
        return float(interpolate_curve(t, z, np.asarray([x], dtype=float), method, frequency)[0])

    return fn


def bootstrap_history(
    curve: pd.DataFrame,
    tenor_years: Mapping[str, float] | None = None,
    frequency: int = 2,
) -> pd.DataFrame:
    """Bootstrap the zero curve for every date in a par-yield history.

    Ragged coverage is handled per row: each day is bootstrapped from whichever
    tenors it actually quotes, so the 2002-2006 30-year gap and the late-arriving
    1/2/4-month bills neither break the recursion nor contaminate it with
    forward-filled stale quotes.

    Returns
    -------
    pd.DataFrame
        Same index and columns as ``curve``, holding zero rates instead of par
        yields.  A cell is NaN wherever the input was NaN.
    """
    if tenor_years is None:
        from ..data.sources import TENOR_YEARS

        tenor_years = TENOR_YEARS

    cols = [c for c in curve.columns if c in tenor_years]
    if not cols:
        raise ValueError("No recognised tenor columns in the supplied curve")
    years = np.array([tenor_years[c] for c in cols], dtype=float)

    values = curve[cols].to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)

    for i in range(values.shape[0]):
        row = values[i]
        mask = np.isfinite(row)
        if mask.sum() < 2:
            continue
        t_i, z_i = par_to_zero(years[mask], row[mask], frequency)
        # Map the solved zeros back onto the original column positions.
        pos = np.flatnonzero(mask)
        lookup = {float(tt): zz for tt, zz in zip(t_i, z_i)}
        for p in pos:
            out[i, p] = lookup.get(float(years[p]), np.nan)

    result = pd.DataFrame(out, index=curve.index, columns=cols)
    result.index.name = curve.index.name or "date"
    return result
