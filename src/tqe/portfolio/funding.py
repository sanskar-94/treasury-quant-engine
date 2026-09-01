"""Cash-neutral trade construction - paying for a curve trade the way a desk does.

The problem this module exists to solve
---------------------------------------
A DV01-weighted steepener matches the *risk* of its two legs, and matching risk
across the curve forces the notionals wildly apart. The 2-year carries roughly a
quarter of the 10-year's DV01, so equalising DV01 means holding about four times
as much of it: a 2s10s steepener sized to $1,000 per basis point per leg is
about **+$5.2mm of 2-year against -$1.2mm of 10-year**, and therefore about
**+$4.0mm net long notional**.

That residual is not a rounding error, it is the whole trade's economics.
Financing in :mod:`tqe.backtest.engine` is charged on **net** notional (ACT/360,
the repo convention - longs pay, shorts receive), so a book of DV01-neutral
structures runs a large permanent long-cash position that must be funded. Run on
this project's own data, the structure-space strategy carried a **28.4% annual
financing drag**, and the first version of that experiment - which computed its
own P&L and forgot the funding leg - reported a Sharpe of 3.96 where the truth
was 0.04. *DV01-neutral is not cash-neutral, and conflating the two is how a
curve trade turns into a levered cash position wearing a curve trade's clothes.*

What a desk actually does
-------------------------
It does not borrow $4mm to hold the steepener. It **funds the trade against a
bill position**: short enough 3-month paper that the book raises its own cash and
the net notional is zero. Two things then change, and both matter.

1. The financing line goes to zero, because there is no net borrowed balance.
   The cost of money has not disappeared - it now shows up where it belongs, as
   the realised total return of the short bill inside market P&L, at the bill's
   actual return rather than as an unbounded charge on a net position.
2. What is left is the *slope* carry the steepener is supposed to earn, instead
   of the front-end-versus-funding carry it was accidentally running. The net
   long notional was a directional bet on the level of the curve against the
   repo rate. Removing it removes a view the trade never intended to express.

The catch, and why there are two constructors
---------------------------------------------
A 3-month bill is not DV01-free. Shorting $4.0mm of it removes roughly $99 per
basis point of duration - about 5% of the steepener's gross DV01. So:

* :func:`cash_neutral_structure` zeroes net notional with a single bill leg and
  **measures** the DV01 it costs you (``dv01_disturbance``). Nothing here assumes
  the bill's duration is negligible; the number is computed and returned so the
  caller can decide whether to care.
* :func:`doubly_neutral_structure` spends the second free parameter - a rescale
  of one cash leg - to put net DV01 back to exactly zero as well. Two unknowns,
  two linear constraints, solved as a 2x2 in closed form.

Look-ahead
----------
Every function here is a **pure, static, single-date construction**: it maps a
(structure, DV01 snapshot) pair to notionals with no time index and no rolling
statistic anywhere in the module, so it cannot look forward on its own. Causality
is therefore entirely a property of the DV01 vector the caller passes in. The
risk columns produced by :func:`tqe.data.universe.constant_maturity_total_return`
describe the bond you would buy at the *close* of day ``t``, i.e. the position
carried into ``t+1``; a book that will earn ``total_return[t]`` must therefore be
sized from ``dv01_panel.shift(1)``. Passing ``dv01_panel.loc[t]`` and harvesting
``return[t]`` is a one-day leak, and it is the caller's job to avoid it because
this module has no way to detect it.

Scale
-----
Every tolerance in this module is **relative to the size of the book** (Rule 4 of
this codebase: two separate bugs came from absolute thresholds applied to
quantities of order 1e-4). Net notional is judged against gross notional, net
DV01 against gross DV01, and the 2x2 determinant against the natural scale of its
own entries - never against a bare epsilon.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .structures import Structure

log = get_logger("portfolio.funding")

__all__ = [
    "CashNeutralTrade",
    "cash_neutral_structure",
    "doubly_neutral_structure",
    "funding_cost",
    "build_cash_neutral_book",
]

EPS: float = 1e-12

ACT_360: float = 360.0
"""Repo, fed funds and T-bills all accrue ACT/360. Bonds do not; funding does."""

CASH_NEUTRAL_TOL: float = 1e-9
"""``|net notional| / gross notional`` below which a book needs no financing."""

DV01_NEUTRAL_TOL: float = 1e-9
"""``|net DV01| / gross DV01`` below which a book carries no directional view."""

DEGENERATE_TOL: float = 1e-6
"""Relative floor on the 2x2 determinant - see :func:`doubly_neutral_structure`."""


def _require(condition: bool, message: str) -> None:
    """Raise :class:`AssertionError` unconditionally - ``python -O`` cannot strip it.

    A bare ``assert`` in library code is a comment with delusions of grandeur: it
    vanishes under optimisation, which is exactly the build where a silent
    financing error would be most expensive.
    """
    if not condition:
        raise AssertionError(message)


def _dv01_series(dv01: Mapping[str, float] | pd.Series) -> pd.Series:
    """Coerce a DV01 mapping to a float Series (per 100 face, positive)."""
    d = dv01 if isinstance(dv01, pd.Series) else pd.Series(dict(dv01), dtype=float)
    return d.astype(float)


def _dollars_per_bp(legs: pd.Series, dv01: pd.Series) -> float:
    """Signed DV01 of a notional book, in dollars per basis point.

    ``dv01`` is quoted per 100 face (the market convention throughout this
    project), so a position of ``N`` face carries ``N * dv01 / 100`` of risk.
    Positive means long duration: the book loses when yields rise.
    """
    d = dv01.reindex(legs.index).fillna(0.0)
    return float((legs * d).sum() / 100.0)


def _ordered_index(labels: Sequence[str]) -> list[str]:
    """Tenor labels shortest-first, with unknown labels kept in given order."""
    from ..data.sources import TENOR_YEARS

    known = [t for t in labels if t in TENOR_YEARS]
    unknown = [t for t in labels if t not in TENOR_YEARS]
    return sorted(known, key=lambda t: TENOR_YEARS[t]) + unknown


@dataclass(frozen=True)
class CashNeutralTrade:
    """A relative-value trade together with the bill position that pays for it.

    Attributes
    ----------
    legs : pd.Series
        Signed notional **face** per tenor, positive long, *including* the
        funding leg. This is the object you hand to
        :func:`tqe.backtest.engine.run_backtest` as a row of ``positions``.
    net_notional : float
        ``legs.sum()``. The quantity financing is charged on, so the number this
        whole module exists to drive to zero.
    net_dv01 : float
        Signed dollars per basis point across every leg, bill included. Zero
        means the trade has no view on the level of the curve.
    funding_leg : float
        Notional of the bill position. Negative is the normal case: a
        DV01-weighted steepener is net long, so it is funded by *shorting* bills.
    description : str
        What the trade is and what it is neutral to.
    funding_tenor : str
        The bill used, e.g. ``"3 Mo"``.
    base_net_notional, base_net_dv01 : float
        The underlying structure's exposures *before* funding, so the effect of
        the construction is auditable rather than asserted.
    dv01_disturbance : float
        ``net_dv01 - base_net_dv01``: the duration the funding leg dragged in.
        For :func:`cash_neutral_structure` this is the price of cash neutrality;
        for :func:`doubly_neutral_structure` it is cancelled by the rescale and
        the trade lands on zero.
    dv01_disturbance_frac : float
        The same number as a fraction of gross DV01 - the scale-free version, and
        the one to look at, because "$99 per basis point" means nothing without
        knowing whether the book is $2,000 or $2mm per basis point.
    rescaled_tenor : str
        Leg whose size was solved for, empty when no leg was rescaled.
    rescale_factor : float
        Multiplier applied to that leg. Distance from 1.0 is how much the funding
        construction distorted the original view.
    name : str
        Identifier inherited from the underlying structure.
    """

    legs: pd.Series
    net_notional: float
    net_dv01: float
    funding_leg: float
    description: str = ""
    funding_tenor: str = ""
    base_net_notional: float = 0.0
    base_net_dv01: float = 0.0
    dv01_disturbance: float = 0.0
    dv01_disturbance_frac: float = 0.0
    rescaled_tenor: str = ""
    rescale_factor: float = 1.0
    name: str = ""

    # ---------------- measured scale ---------------- #
    @property
    def gross_notional(self) -> float:
        """Sum of absolute leg notionals - what the trade costs to put on."""
        return float(self.legs.abs().sum())

    @property
    def leverage(self) -> float:
        """Gross notional per dollar of net notional; infinite when cash-neutral.

        The whole point of the construction: a cash-neutral book has unbounded
        notional leverage against a *zero* funded balance, which is a different
        and far safer animal than the same gross book run against a real one.
        """
        n = abs(self.net_notional)
        return float("inf") if n < EPS else self.gross_notional / n

    def gross_dv01(self, dv01: Mapping[str, float] | pd.Series) -> float:
        """Sum of absolute leg DV01s, in dollars per basis point."""
        d = _dv01_series(dv01).reindex(self.legs.index).fillna(0.0)
        return float((self.legs * d).abs().sum() / 100.0)

    # ---------------- independent re-measurement ---------------- #
    def dv01_of(self, dv01: Mapping[str, float] | pd.Series) -> float:
        """Recompute net DV01 from the legs and a DV01 vector.

        Deliberately does not read :attr:`net_dv01`. Tests and risk checks should
        re-derive exposures from the positions rather than trust a cached field
        that was written by the same code they are meant to be checking.
        """
        return _dollars_per_bp(self.legs, _dv01_series(dv01))

    def is_cash_neutral(self, tol: float = CASH_NEUTRAL_TOL) -> bool:
        """``|net notional|`` negligible **relative to gross notional**."""
        return abs(float(self.legs.sum())) <= tol * max(self.gross_notional, EPS)

    def is_dv01_neutral(
        self,
        dv01: Mapping[str, float] | pd.Series,
        tol: float = DV01_NEUTRAL_TOL,
    ) -> bool:
        """``|net DV01|`` negligible **relative to gross DV01**."""
        return abs(self.dv01_of(dv01)) <= tol * max(self.gross_dv01(dv01), EPS)

    def summary(self) -> str:
        legs = "  ".join(f"{k}:{v:+,.0f}" for k, v in self.legs.items() if abs(v) > EPS)
        return (
            f"{self.name or 'trade'} [{self.funding_tenor} funded]\n"
            f"  legs            {legs}\n"
            f"  net notional    {self.net_notional:+,.2f}  "
            f"(before funding {self.base_net_notional:+,.0f})\n"
            f"  net DV01        {self.net_dv01:+,.4f} $/bp  "
            f"(before funding {self.base_net_dv01:+,.4f})\n"
            f"  DV01 disturbed  {self.dv01_disturbance:+,.4f} $/bp  "
            f"({self.dv01_disturbance_frac:+.4%} of gross DV01)"
        )

    def __repr__(self) -> str:
        return (
            f"CashNeutralTrade({self.name!r}, net_notional={self.net_notional:+,.2f}, "
            f"net_dv01={self.net_dv01:+,.4f}, funding_leg={self.funding_leg:+,.0f})"
        )


def cash_neutral_structure(
    structure: Structure,
    dv01: Mapping[str, float] | pd.Series,
    funding_tenor: str = "3 Mo",
) -> CashNeutralTrade:
    r"""Fund a DV01-weighted structure with an offsetting bill position.

    The construction is one line of algebra and a great deal of consequence: set
    the bill notional to :math:`b = -\sum_t w_t` so the book raises exactly the
    cash it spends. There is then no net borrowed balance, financing in
    :func:`tqe.backtest.engine.run_backtest` is charged on zero, and the cost of
    money reappears where it economically belongs - inside the realised total
    return of the short bill.

    A bill is not duration-free, so this is not free. Shorting :math:`b` of paper
    with DV01 :math:`d_f` per 100 face moves net DV01 by :math:`b d_f / 100`,
    which for a $4mm bill leg against a 3-month bill is about $99 per basis
    point. **That number is computed and returned, never assumed away** - see
    ``dv01_disturbance`` and its scale-free companion ``dv01_disturbance_frac``.
    If the leakage matters for your purpose, use
    :func:`doubly_neutral_structure`, which removes it exactly.

    Parameters
    ----------
    structure : Structure
        Any structure from :mod:`tqe.portfolio.structures` - steepener,
        butterfly, or a bespoke set of DV01-weighted legs.
    dv01 : mapping or pd.Series
        DV01 **per 100 face**, positive, by tenor label. Must cover every leg of
        ``structure`` and ``funding_tenor``. Pass the DV01 observable at the
        close *before* the trade is put on; see the module docstring on
        causality.
    funding_tenor : str, default ``"3 Mo"``
        The bill the trade is financed against. The 3-month bill is the natural
        choice because it is the closest cash instrument to general-collateral
        repo, which is what the engine's funding rate is built from.

    Returns
    -------
    CashNeutralTrade
        ``net_notional`` is zero to floating-point; ``net_dv01`` is the
        structure's own net DV01 plus the measured bill disturbance.

    Raises
    ------
    ValueError
        If ``funding_tenor`` has no DV01, or the funding bill has essentially
        zero DV01 (which would make the disturbance meaningless to report).

    Notes
    -----
    The original ``structure`` is never mutated; ``legs`` is a fresh Series.
    """
    d = _dv01_series(dv01)
    if funding_tenor not in d.index:
        raise ValueError(
            f"No DV01 for funding tenor {funding_tenor!r}; have {sorted(d.index)}"
        )
    d_f = float(d[funding_tenor])
    if abs(d_f) < EPS:
        raise ValueError(f"Funding tenor {funding_tenor!r} has ~zero DV01 ({d_f})")

    idx = list(structure.weights.index)
    if funding_tenor not in idx:
        idx = _ordered_index([*idx, funding_tenor])
    legs = structure.weights.reindex(idx).fillna(0.0).astype(float)

    base_notional = float(legs.sum())
    base_dv01 = _dollars_per_bp(legs, d)

    # The entire construction: short exactly the cash the structure consumes.
    b = -base_notional
    legs = legs.copy()
    legs[funding_tenor] = float(legs[funding_tenor]) + b
    legs.name = structure.name

    net_notional = float(legs.sum())
    net_dv01 = _dollars_per_bp(legs, d)
    disturbance = net_dv01 - base_dv01

    gross_dv01 = float((legs * d.reindex(legs.index).fillna(0.0)).abs().sum() / 100.0)
    gross_notional = float(legs.abs().sum())

    # Relative, not absolute (Rule 4): a $10 residual is nothing on a $10mm book
    # and a catastrophe on a $100 one.
    _require(
        abs(net_notional) <= CASH_NEUTRAL_TOL * max(gross_notional, EPS),
        f"cash neutralisation failed: net notional {net_notional:+.6g} on gross "
        f"{gross_notional:,.0f}",
    )
    # The disturbance must be exactly the bill leg's own DV01 - nothing else moved.
    _require(
        abs(disturbance - b * d_f / 100.0) <= 1e-9 * max(abs(disturbance), 1.0),
        "DV01 disturbance is not attributable to the funding leg alone",
    )

    return CashNeutralTrade(
        legs=legs,
        net_notional=net_notional,
        net_dv01=net_dv01,
        funding_leg=b,
        description=(
            f"{structure.description}; funded by {'short' if b < 0 else 'long'} "
            f"{abs(b):,.0f} of {funding_tenor} so net notional is zero. "
            f"Net DV01 moves {disturbance:+.2f} $/bp "
            f"({disturbance / max(gross_dv01, EPS):+.2%} of gross) - the bill's own duration."
        ),
        funding_tenor=funding_tenor,
        base_net_notional=base_notional,
        base_net_dv01=base_dv01,
        dv01_disturbance=disturbance,
        dv01_disturbance_frac=float(disturbance / max(gross_dv01, EPS)),
        rescaled_tenor="",
        rescale_factor=1.0,
        name=structure.name,
    )


def doubly_neutral_structure(
    structure: Structure,
    dv01: Mapping[str, float] | pd.Series,
    funding_tenor: str = "3 Mo",
    rescale_tenor: str | None = None,
) -> CashNeutralTrade:
    r"""Zero net notional **and** zero net DV01, solved exactly as a 2x2.

    :func:`cash_neutral_structure` buys cash neutrality with a few per cent of
    the book's DV01. Getting both neutralities needs two free parameters, and a
    funded structure has exactly two available: the bill notional :math:`b` and a
    rescale :math:`s` of one cash leg. With :math:`k` the rescaled leg,
    :math:`w_k` its notional, :math:`f = d_f/100` the bill's dollars of DV01 per
    unit face, :math:`g_k = w_k d_k / 100` the leg's DV01, and
    :math:`n_{\text{rest}}, d_{\text{rest}}` the notional and DV01 of everything
    else:

    .. math::

        \begin{pmatrix} w_k & 1 \\ g_k & f \end{pmatrix}
        \begin{pmatrix} s \\ b \end{pmatrix}
        = \begin{pmatrix} -n_{\text{rest}} \\ -d_{\text{rest}} \end{pmatrix}

    Two equations, two unknowns, one direct solve. No iteration, no optimiser, no
    tolerance to tune - the answer is exact up to floating point, which is the
    only acceptable standard for a constraint a risk system will later assert on.

    The determinant is :math:`\det = w_k (f - d_k/100)`. It vanishes when the
    rescaled leg has the bill's own DV01 per unit face, which is the honest
    economic statement of the problem: if your two free instruments are
    indistinguishable, you cannot hit two independent targets with them. For the
    same reason the funding tenor itself is never eligible as the rescale leg -
    that choice makes the two columns exactly collinear.

    Parameters
    ----------
    structure : Structure
        The DV01-weighted trade to fund.
    dv01 : mapping or pd.Series
        DV01 per 100 face by tenor, as of the close before the trade goes on.
    funding_tenor : str, default ``"3 Mo"``
        The bill used to raise cash.
    rescale_tenor : str, optional
        Leg to solve for. Defaults to the leg maximising :math:`|\det|`, i.e. the
        position that is most distinguishable from the bill *weighted by how much
        of it you hold*. That is simultaneously the best-conditioned choice and
        the one that needs the smallest proportional change, so the trade you get
        back is the closest doubly-neutral trade to the one you asked for.

    Returns
    -------
    CashNeutralTrade
        ``net_notional`` and ``net_dv01`` both zero to floating point;
        ``rescale_factor`` records how far the view had to be distorted.

    Raises
    ------
    ValueError
        If there is no eligible rescale leg, or the chosen one is degenerate
        against the bill.

    Notes
    -----
    Because both constraints are linear and homogeneous in the legs, any weighted
    sum of trades built here is itself doubly neutral - which is what makes
    :func:`build_cash_neutral_book` a one-line aggregation rather than a second
    optimisation.
    """
    d = _dv01_series(dv01)
    if funding_tenor not in d.index:
        raise ValueError(
            f"No DV01 for funding tenor {funding_tenor!r}; have {sorted(d.index)}"
        )

    idx = list(structure.weights.index)
    if funding_tenor not in idx:
        idx = _ordered_index([*idx, funding_tenor])
    base = structure.weights.reindex(idx).fillna(0.0).astype(float)
    dv = d.reindex(idx).fillna(0.0)

    base_notional = float(base.sum())
    base_dv01 = _dollars_per_bp(base, dv)
    f = float(dv[funding_tenor]) / 100.0

    # ---- pick the leg to solve for ---- #
    # The funding tenor is excluded on principle, not on taste: rescaling it
    # would make the two columns of the 2x2 identical up to a factor.
    candidates = [
        t for t in idx if t != funding_tenor and abs(float(base[t])) > EPS
    ]
    if not candidates:
        raise ValueError(
            f"{structure.name!r} has no leg outside the funding tenor "
            f"{funding_tenor!r} to rescale"
        )
    if rescale_tenor is not None:
        if rescale_tenor not in candidates:
            raise ValueError(
                f"{rescale_tenor!r} is not an eligible rescale leg; "
                f"eligible: {candidates}"
            )
        k = rescale_tenor
    else:
        k = max(candidates, key=lambda t: abs(float(base[t]) * (f - float(dv[t]) / 100.0)))

    w_k = float(base[k])
    g_k = w_k * float(dv[k]) / 100.0
    n_rest = base_notional - w_k
    d_rest = base_dv01 - g_k

    A = np.array([[w_k, 1.0], [g_k, f]], dtype=float)
    rhs = np.array([-n_rest, -d_rest], dtype=float)
    det = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])

    # Relative degeneracy check (Rule 4). |det| = |w_k| * |f - d_k/100|, so the
    # natural yardstick is |w_k| times the larger of the two DV01 densities.
    scale = abs(w_k) * max(abs(f), abs(float(dv[k])) / 100.0)
    if abs(det) < DEGENERATE_TOL * max(scale, EPS):
        raise ValueError(
            f"Degenerate 2x2: leg {k!r} (DV01 {float(dv[k]):.6f}) is "
            f"indistinguishable from the {funding_tenor!r} bill "
            f"(DV01 {float(dv[funding_tenor]):.6f}); pick another rescale_tenor"
        )

    s, b = (float(v) for v in np.linalg.solve(A, rhs))

    legs = base.copy()
    legs[k] = w_k * s
    legs[funding_tenor] = float(legs[funding_tenor]) + b
    legs.name = structure.name

    net_notional = float(legs.sum())
    net_dv01 = _dollars_per_bp(legs, dv)
    gross_notional = float(legs.abs().sum())
    gross_dv01 = float((legs * dv).abs().sum() / 100.0)

    _require(
        abs(net_notional) <= CASH_NEUTRAL_TOL * max(gross_notional, EPS),
        f"double neutralisation left net notional {net_notional:+.6g} on gross "
        f"{gross_notional:,.0f}",
    )
    _require(
        abs(net_dv01) <= DV01_NEUTRAL_TOL * max(gross_dv01, EPS),
        f"double neutralisation left net DV01 {net_dv01:+.6g} on gross DV01 "
        f"{gross_dv01:,.4f}",
    )

    return CashNeutralTrade(
        legs=legs,
        net_notional=net_notional,
        net_dv01=net_dv01,
        funding_leg=b,
        description=(
            f"{structure.description}; doubly neutral - {funding_tenor} leg "
            f"{b:+,.0f} and the {k} leg rescaled x{s:.4f} solve net notional = 0 "
            f"and net DV01 = 0 simultaneously."
        ),
        funding_tenor=funding_tenor,
        base_net_notional=base_notional,
        base_net_dv01=base_dv01,
        dv01_disturbance=net_dv01 - base_dv01,
        dv01_disturbance_frac=float((net_dv01 - base_dv01) / max(gross_dv01, EPS)),
        rescaled_tenor=k,
        rescale_factor=s,
        name=structure.name,
    )


def funding_cost(
    trade: CashNeutralTrade,
    funding_rate: float,
    days: float = 1,
) -> float:
    r"""Repo charge on a trade's net notional over ``days``, ACT/360.

    .. math:: \text{cost} = \text{net notional} \times r \times \frac{\text{days}}{360}

    This is deliberately the *same* arithmetic the P&L engine applies in
    :func:`tqe.backtest.engine._core_loop`, expressed on a single trade so that a
    construction can be checked before it is ever run. It is **not** an
    alternative P&L path: it prices one line item (the financing leg), it never
    touches returns, and every reported strategy number in this project still
    comes from :func:`tqe.backtest.engine.run_backtest`. Five separate times a
    bespoke P&L helper here inflated a result by omitting exactly this term; the
    fix is not to hide the term, it is to make it impossible to forget.

    Sign convention follows the engine: **positive means a cost** (subtracted
    from P&L). A net short book returns a negative number, which is financing
    income.

    Parameters
    ----------
    trade : CashNeutralTrade
        The funded trade. Net notional is **recomputed from ``trade.legs``**
        rather than read from the cached field, and the two must agree - a trade
        whose stored exposure disagrees with its own positions is exactly the
        kind of object that has previously produced a fictitious Sharpe.
    funding_rate : float
        Annualised repo rate as a **decimal** (0.0425 = 4.25%), i.e. the bill
        yield plus ``cfg.costs.repo_spread_bp``.
    days : float, default 1
        **Calendar** days, not business days. A position held over a weekend is
        funded for three days; charging one is a systematic understatement of
        about 40%.

    Returns
    -------
    float
        Dollars of financing. Approximately zero for a cash-neutral trade, and
        that is asserted, not hoped for.

    Raises
    ------
    AssertionError
        If ``trade.net_notional`` disagrees with ``trade.legs.sum()``, or if a
        trade that reports itself cash-neutral is nevertheless charged a
        non-negligible amount.
    """
    net = float(trade.legs.sum())
    gross = trade.gross_notional
    _require(
        abs(net - trade.net_notional) <= 1e-9 * max(gross, 1.0),
        f"trade.net_notional ({trade.net_notional:+,.6g}) disagrees with its own "
        f"legs ({net:+,.6g}) - the stored exposure cannot be trusted",
    )

    cost = net * float(funding_rate) * float(days) / ACT_360

    if trade.is_cash_neutral():
        # The natural scale of the charge is what the *gross* book would pay;
        # comparing to an absolute dollar figure would be a Rule 4 violation.
        scale = gross * abs(float(funding_rate)) * abs(float(days)) / ACT_360
        _require(
            abs(cost) <= max(1e-8 * scale, 1e-6),
            f"a cash-neutral trade was charged {cost:+,.6g} of financing "
            f"(net notional {net:+,.6g} on gross {gross:,.0f})",
        )
    return float(cost)


def build_cash_neutral_book(
    structures: Sequence[Structure],
    dv01: Mapping[str, float] | pd.Series,
    weights: Sequence[float] | Mapping[str, float],
    funding_tenor: str = "3 Mo",
) -> pd.Series:
    """Combine weighted structures into one tenor-space book, doubly neutral.

    A rates book is a set of views, not a single trade: some slope, some
    curvature, each with its own conviction. This aggregates them into the
    notional vector a backtest or an OMS actually consumes.

    Each structure is funded through :func:`doubly_neutral_structure` *first* and
    the results are then summed. That ordering is the point. Both constraints -
    ``sum(w) = 0`` and ``sum(w * dv01) = 0`` - are linear and homogeneous, so a
    weighted sum of doubly-neutral trades is doubly neutral by construction, for
    any weights whatsoever. Aggregating first and neutralising afterwards would
    also reach a neutral book, but by projecting the *combined* view onto the
    constraint null space, which silently reallocates risk between structures
    that had nothing to do with each other. Neutralising each trade on its own
    terms leaves every view intact and needs no second solve.

    Parameters
    ----------
    structures : sequence of Structure
        Typically :func:`tqe.portfolio.structures.build_standard_structures`.
    dv01 : mapping or pd.Series
        DV01 per 100 face by tenor, observable before the book is put on.
    weights : sequence of float or mapping
        Conviction per structure. A sequence must align with ``structures``
        positionally; a mapping is keyed by ``Structure.name`` and missing names
        get zero. Units are "units of the structure", so a structure built at
        $1,000 DV01 per leg and weighted 2.0 risks $2,000 per basis point of its
        own shape.
    funding_tenor : str, default ``"3 Mo"``
        Bill used to fund every constituent.

    Returns
    -------
    pd.Series
        Signed notional face per tenor, shortest tenor first, named
        ``"notional"``. Empty input returns an empty Series rather than raising,
        so a day on which no structure passes its filters is a flat book rather
        than a crash.

    Raises
    ------
    ValueError
        If a positional ``weights`` sequence does not match ``structures``.
    """
    d = _dv01_series(dv01)
    structures = list(structures)

    if isinstance(weights, Mapping):
        w_list = [float(weights.get(s.name, 0.0)) for s in structures]
    else:
        w_list = [float(x) for x in weights]
        if len(w_list) != len(structures):
            raise ValueError(
                f"{len(w_list)} weights for {len(structures)} structures"
            )

    if not structures:
        return pd.Series(dtype=float, name="notional")

    trades = [
        doubly_neutral_structure(s, d, funding_tenor=funding_tenor)
        for s in structures
    ]

    labels: list[str] = []
    for t in trades:
        labels.extend(x for x in t.legs.index if x not in labels)
    idx = _ordered_index(labels)

    book = pd.Series(0.0, index=idx, name="notional")
    for w, t in zip(w_list, trades, strict=True):
        book = book.add(t.legs.reindex(idx).fillna(0.0) * w, fill_value=0.0)
    book.name = "notional"

    gross_notional = float(book.abs().sum())
    dvv = d.reindex(idx).fillna(0.0)
    gross_dv01 = float((book * dvv).abs().sum() / 100.0)
    net_notional = float(book.sum())
    net_dv01 = _dollars_per_bp(book, dvv)

    # Linearity guarantees both of these; they are asserted anyway, because the
    # cost of being wrong about financing in this codebase is a Sharpe of 3.96.
    _require(
        abs(net_notional) <= CASH_NEUTRAL_TOL * max(gross_notional, EPS),
        f"book is not cash-neutral: net {net_notional:+,.6g} on gross "
        f"{gross_notional:,.0f}",
    )
    _require(
        abs(net_dv01) <= DV01_NEUTRAL_TOL * max(gross_dv01, EPS),
        f"book is not DV01-neutral: net {net_dv01:+,.6g} $/bp on gross DV01 "
        f"{gross_dv01:,.4f}",
    )
    log.debug(
        "cash-neutral book: %d structures, gross $%.0f, gross DV01 %.1f $/bp",
        len(structures), gross_notional, gross_dv01,
    )
    return book
