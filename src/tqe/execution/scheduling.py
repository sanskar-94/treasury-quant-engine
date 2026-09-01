"""Execution scheduling: working an order instead of dropping it.

The backtest charges a square-root impact cost on the whole order as though it
were executed in one print. A desk does not do that. It works a large order over
hours or days, which reduces impact - and takes on the risk that the price moves
away while it is still working. Those two costs pull in opposite directions and
the trade-off has a closed form.

Almgren & Chriss (2000) set it up as: minimise

    E[cost] + lambda * Var[cost]

over trading trajectories. With linear temporary impact and arithmetic Brownian
price dynamics the optimal path is

    x(t) = X * sinh(kappa (T - t)) / sinh(kappa T),    kappa = sqrt(lambda sigma^2 / eta)

where ``x(t)`` is the quantity *remaining*, ``eta`` is the temporary impact
coefficient and ``sigma`` the price volatility. The two limits are the whole
intuition and they are asserted in the tests:

* ``lambda -> 0`` (risk-neutral): ``kappa -> 0`` and the trajectory becomes
  linear - trade evenly, minimise impact, accept the timing risk. This is TWAP.
* ``lambda -> infinity`` (risk-averse): ``kappa -> infinity`` and the trajectory
  collapses to the front - dump the order, pay the impact, eliminate the risk.

Anything in between is a genuine choice about how much timing risk to buy back
with impact cost.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger("execution.scheduling")

__all__ = [
    "ExecutionSchedule",
    "twap_schedule",
    "vwap_schedule",
    "almgren_chriss_schedule",
    "implementation_shortfall",
    "optimal_participation",
]

EPS = 1e-12

#: Practical ceiling on participation. Above roughly a fifth of volume a trader
#: stops being a price taker: the order becomes a visible share of the tape,
#: other participants infer the direction, and impact stops behaving like the
#: square-root law that every cost model here assumes.
MAX_PARTICIPATION = 0.20


@dataclass
class ExecutionSchedule:
    """A plan for working one order, and what it is expected to cost."""

    quantities: np.ndarray            # per slice, signed like the parent order
    times: np.ndarray                 # slice end times, in the horizon's units
    expected_cost: float = 0.0        # currency, impact + spread
    expected_risk: float = 0.0        # currency, std dev of execution price risk
    participation_rate: float = 0.0
    horizon: float = 0.0
    strategy: str = ""
    diagnostics: dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        return float(self.quantities.sum())

    @property
    def n_slices(self) -> int:
        return int(len(self.quantities))

    def remaining(self) -> np.ndarray:
        """Quantity still to trade after each slice - the Almgren-Chriss ``x(t)``."""
        return self.total - np.cumsum(self.quantities)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "time": self.times,
            "quantity": self.quantities,
            "cumulative": np.cumsum(self.quantities),
            "remaining": self.remaining(),
        })

    def summary(self) -> str:
        front = self.quantities[0] / self.total if abs(self.total) > EPS else float("nan")
        return (
            f"{self.strategy}: {self.n_slices} slices over {self.horizon:g}, "
            f"first slice {front:.1%} of the order, "
            f"E[cost] {self.expected_cost:,.0f}, E[risk] {self.expected_risk:,.0f}"
        )


def _finalise(q: np.ndarray, total: float) -> np.ndarray:
    """Force the slices to sum to the order exactly.

    Any schedule that leaves a residual has failed at its one job. Rounding is
    absorbed into the largest slice rather than spread around, so the shape of
    the trajectory is preserved.
    """
    q = np.asarray(q, dtype=float)
    resid = total - q.sum()
    if abs(resid) > EPS:
        q[int(np.argmax(np.abs(q)))] += resid
    return q


def twap_schedule(quantity: float, n_slices: int = 10, horizon: float = 1.0) -> ExecutionSchedule:
    """Equal slices across the horizon.

    The baseline every other strategy is measured against. Minimises impact for
    a given horizon and is completely indifferent to price risk.
    """
    if n_slices < 1:
        raise ValueError("n_slices must be at least 1")
    q = _finalise(np.full(n_slices, quantity / n_slices), quantity)
    return ExecutionSchedule(
        quantities=q,
        times=np.linspace(horizon / n_slices, horizon, n_slices),
        horizon=horizon,
        strategy="TWAP",
    )


def vwap_schedule(
    quantity: float,
    volume_profile: Sequence[float] | np.ndarray,
    horizon: float = 1.0,
) -> ExecutionSchedule:
    """Slices proportional to expected volume.

    Trading in proportion to volume keeps participation - and therefore impact -
    roughly constant through the day, instead of taking an outsized share of a
    thin period. A flat profile reduces this to TWAP.
    """
    profile = np.asarray(volume_profile, dtype=float)
    if profile.size == 0:
        raise ValueError("volume_profile is empty")
    if np.any(profile < 0):
        raise ValueError("volume_profile cannot be negative")
    total_vol = profile.sum()
    if total_vol <= EPS:
        raise ValueError("volume_profile sums to zero")

    n = profile.size
    q = _finalise(quantity * profile / total_vol, quantity)
    return ExecutionSchedule(
        quantities=q,
        times=np.linspace(horizon / n, horizon, n),
        horizon=horizon,
        strategy="VWAP",
        diagnostics={"volume_share": (profile / total_vol).tolist()},
    )


def almgren_chriss_schedule(
    quantity: float,
    horizon: float = 1.0,
    n_slices: int = 10,
    volatility: float = 0.01,
    temp_impact: float = 1e-6,
    perm_impact: float = 0.0,
    risk_aversion: float = 1e-6,
) -> ExecutionSchedule:
    """The closed-form optimal trading trajectory.

    Parameters
    ----------
    quantity:
        Signed parent order.
    horizon:
        Total time available, in the same units as ``volatility``.
    n_slices:
        Number of discrete child orders.
    volatility:
        Price volatility per unit time. Drives the timing risk.
    temp_impact:
        Temporary impact coefficient ``eta``: cost per unit of trading *rate*.
        Higher means trading fast is more expensive, so the optimum stretches out.
    perm_impact:
        Permanent impact ``gamma``. Enters the expected cost but **not** the
        trajectory - a standard and initially surprising result: permanent
        impact is paid on the whole order however you slice it, so it cannot be
        optimised away and does not change the optimal path.
    risk_aversion:
        ``lambda``. Zero recovers TWAP; large values front-load. See the module
        docstring for the two limits.

    Returns
    -------
    ExecutionSchedule

    Notes
    -----
    ``kappa = sqrt(risk_aversion * volatility^2 / temp_impact)`` has units of
    1/time, and ``kappa * horizon`` is the only thing that matters for the
    *shape* of the trajectory. When it is small the ``sinh`` is nearly linear
    and the schedule is TWAP; when it is large the ``sinh`` is nearly
    exponential and almost everything trades in the first slice.
    """
    if n_slices < 1:
        raise ValueError("n_slices must be at least 1")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if temp_impact <= 0:
        raise ValueError("temp_impact must be positive")

    kappa = float(np.sqrt(max(risk_aversion, 0.0) * volatility**2 / temp_impact))
    times = np.linspace(horizon / n_slices, horizon, n_slices)
    grid = np.concatenate([[0.0], times])

    kt = kappa * horizon
    if kt < 1e-8:
        # sinh(k(T-t))/sinh(kT) -> (T-t)/T as k -> 0. Evaluating the ratio
        # directly here would be 0/0.
        remaining = (horizon - grid) / horizon
    elif kt > 300:
        # sinh overflows past ~710 in float64 and the ratio underflows long
        # before that; the limit is a pure exponential decay.
        remaining = np.exp(-kappa * grid)
    else:
        remaining = np.sinh(kappa * (horizon - grid)) / np.sinh(kt)

    q = _finalise(quantity * (-np.diff(remaining)), quantity)

    # Expected cost: temporary impact on each slice's trading rate, plus
    # permanent impact on the full order.
    dt = horizon / n_slices
    rate = np.abs(q) / dt
    temp_cost = float(temp_impact * np.sum(rate * np.abs(q)))
    perm_cost = float(0.5 * perm_impact * quantity**2)

    # Timing risk: volatility of the value of what is still outstanding.
    held = quantity * remaining[:-1]
    risk = float(volatility * np.sqrt(dt * np.sum(held**2)))

    return ExecutionSchedule(
        quantities=q,
        times=times,
        expected_cost=temp_cost + perm_cost,
        expected_risk=risk,
        horizon=horizon,
        strategy="Almgren-Chriss",
        diagnostics={"kappa": kappa, "kappa_times_horizon": kt,
                     "temp_cost": temp_cost, "perm_cost": perm_cost,
                     "risk_aversion": risk_aversion},
    )


def optimal_participation(order_size: float, adv: float, urgency: float = 0.5) -> float:
    """Participation rate to target, as a fraction of average daily volume.

    Scales with urgency but is capped at :data:`MAX_PARTICIPATION`. The cap is
    the point of the function: beyond roughly a fifth of volume the order stops
    being anonymous, other participants infer its direction, and impact stops
    following the square-root law that the cost model assumes. A number above
    the cap is not a more aggressive plan, it is a plan whose cost estimate has
    become fiction.
    """
    if adv <= EPS:
        raise ValueError("adv must be positive")
    urgency = float(np.clip(urgency, 0.0, 1.0))
    natural = abs(order_size) / adv
    target = urgency * MAX_PARTICIPATION + (1.0 - urgency) * min(natural, MAX_PARTICIPATION)
    return float(min(target, MAX_PARTICIPATION))


def implementation_shortfall(
    schedule: ExecutionSchedule,
    arrival_price: float,
    fill_prices: Sequence[float] | np.ndarray,
    final_price: float | None = None,
    unfilled: float = 0.0,
) -> dict[str, float]:
    """Decompose realised cost against the arrival price.

    Implementation shortfall is the honest measure of execution quality: it
    compares what was actually paid against the price at the moment the decision
    was made, so it captures both the impact of trading and the cost of not
    having traded yet.

    Returns the standard three-way split, in currency:

    ``execution_cost``
        Fills against the arrival price - what the trading itself cost.
    ``opportunity_cost``
        The unfilled residual marked at the final price - what hesitating cost.
    ``total_shortfall``
        Their sum, and the number that goes in the TCA report.
    """
    fills = np.asarray(fill_prices, dtype=float)
    q = np.asarray(schedule.quantities, dtype=float)
    if fills.shape != q.shape:
        raise ValueError(f"fill_prices has shape {fills.shape}, expected {q.shape}")

    # Signed by construction: buying above arrival is a cost (positive), selling
    # below arrival is also a cost, and q carries the sign of the parent order.
    execution_cost = float(np.sum(q * (fills - arrival_price)))

    opportunity = 0.0
    if abs(unfilled) > EPS:
        end = float(final_price if final_price is not None else fills[-1])
        opportunity = float(unfilled * (end - arrival_price))

    filled = float(np.sum(np.abs(q)))
    avg = float(np.sum(q * fills) / np.sum(q)) if abs(np.sum(q)) > EPS else float("nan")
    return {
        "execution_cost": execution_cost,
        "opportunity_cost": opportunity,
        "total_shortfall": execution_cost + opportunity,
        "average_fill_price": avg,
        "arrival_price": float(arrival_price),
        "slippage_bp": float((avg / arrival_price - 1.0) * 1e4) if arrival_price else float("nan"),
        "filled_quantity": filled,
        "unfilled_quantity": float(unfilled),
    }
