"""Transaction-cost model for cash US Treasuries.

Why this module exists
----------------------
A fixed-income backtest that ignores costs is a fiction.  The strategies in this
repo trade duration daily, so the *only* thing standing between a 1.5 Sharpe on
paper and a 0.3 Sharpe in production is an honest execution model.  Everything
here is deliberately conservative: when in doubt the model charges more, not
less.

Market conventions encoded here
-------------------------------
**Why 32nds.**  US Treasury notes and bonds are quoted in points of par and
*thirty-seconds* of a point ("handles and ticks"), not decimals -- a legacy of
the colonial-era Spanish dollar, which was cut into eight reales; American
securities markets inherited the binary fractions and subdivided them further
(halves of eighths of a point = 32nds, then 64ths and 128ths for the front end).
Equities decimalised in 2001; cash Treasuries never did, because the tick grid
*is* the price grid a voice/RFQ market negotiates on.  A quote of ``99-16+``
means 99 + 16.5/32 = 99.515625.  One 32nd = 0.03125 price points = 3.125bp of
price.  Bills are the exception: they are quoted on an ACT/360 *discount rate*,
not a price, which is why the bill bucket's half-spread here is best read as its
price-equivalent.

**What an on-the-run 10y actually costs.**  In normal conditions the OTR 10y
note is quoted about one tick wide on the interdealer screens, i.e. a
**half-spread of 0.5/32 = 0.015625 price points = 1.56bp of price**.  With a
10y modified duration near 8 (DV01 ~ 0.08 points per bp per 100 face) that is
about **0.2bp of yield** -- roughly two tenths of one basis point to cross the
market.  That number is the sanity anchor for this whole module: if a cost model
tells you the 10y costs 5bp of yield to trade, it is wrong by more than an order
of magnitude; if it tells you the 30y costs the same as the 10y, it is also
wrong.  Off-the-runs, the 20y (reintroduced in 2020 and structurally the
cheapest-to-hold, widest-to-trade coupon point) and month/quarter-end trade
materially wider than these defaults.

Cost components
---------------
``spread``      Half the quoted bid/offer, paid on every trade in either
                direction.  Linear in size.
``impact``      Square-root market impact, ``coef * spread * sqrt(size / ADV)``.
                Sub-linear in *price* terms but super-linear in *dollar* terms
                (dollars scale as ``size^1.5``), which is what stops the
                optimiser from sizing infinitely.
``commission``  Flat dollars per million -- brokerage/clearing/FICC.
``financing``   Repo on ACT/360, the money-market convention.  A holding cost,
                not a trading cost, so it is deliberately *excluded* from
                :meth:`CostModel.total_cost`.

No look-ahead
-------------
Nothing in this module reads a time series.  :func:`turnover_cost_series` is the
one time-aware function and it charges day ``t``'s cost on
``position_t - position_{t-1}``, i.e. on a trade whose size is known the moment
the day-``t`` target is formed from day ``t-1`` information.  It never peeks at
``t+1`` and it never re-prices a past trade with a later spread.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import CostConfig
from ..data.sources import TENOR_YEARS
from ..logging_utils import get_logger

log = get_logger("backtest.costs")

__all__ = [
    "BUCKETS",
    "DEFAULT_ADV",
    "TICKS_PER_POINT",
    "tenor_bucket",
    "CostModel",
    "turnover_cost_series",
]

# One point of par = 32 ticks.  Exposed as a constant so nothing below has a
# bare ``/ 32`` whose meaning has to be guessed.
TICKS_PER_POINT: float = 32.0

#: Cost buckets, short -> long.  Matches ``TenorSpec.bucket`` in
#: :mod:`tqe.data.universe` so a bucket string is interchangeable between the
#: universe layer and this one.
BUCKETS: tuple[str, ...] = ("bill", "2y", "5y", "10y", "30y")

#: Average daily *dollar* volume by bucket, order-of-magnitude figures consistent
#: with SIFMA/primary-dealer aggregate Treasury turnover (~$900bn/day across the
#: complex in recent years).  These are only ever used inside a square root, so
#: being off by 2x moves the impact charge by 40% of a already-small number.
#: The long end is by far the thinnest: the 20y/30y sector turns over roughly a
#: quarter of what the 10y sector does, which is why it is bucketed separately.
DEFAULT_ADV: dict[str, float] = {
    "bill": 250e9,
    "2y": 180e9,
    "5y": 180e9,
    "10y": 200e9,
    "30y": 60e9,
}

# Upper edge (in years) of each bucket.  The 7y sits in the "5y" bucket because
# its liquidity profile tracks the 5y far more closely than the 10y benchmark;
# the 20y is charged at 30y levels, which if anything flatters it -- since its
# 2020 reintroduction the 20y has consistently traded *wider* than the 30y.
_BUCKET_EDGES: tuple[tuple[float, str], ...] = (
    (1.0, "bill"),
    (4.0, "2y"),
    (8.5, "5y"),
    (15.0, "10y"),
    (np.inf, "30y"),
)


def tenor_bucket(tenor: str | float) -> str:
    """Map a tenor label or a maturity in years to a cost bucket.

    Parameters
    ----------
    tenor : str or float
        Either a Treasury CMT label (``"10 Yr"``, ``"3 Mo"``) or a maturity in
        years (``10.0``).

    Returns
    -------
    str
        One of :data:`BUCKETS`.

    Notes
    -----
    Liquidity in Treasuries is organised around the *on-the-run* issues, so the
    right granularity for a cost model is the auction sector, not the exact
    maturity.  Anything under a year is money-market paper (bills / rolling
    coupons) and trades on a discount basis; 1-3y prices off the 2y and 3y
    auctions; 5y and 7y trade together; the 10y is the global benchmark and the
    tightest coupon point; 20y and 30y are the illiquid long end.

    Examples
    --------
    >>> tenor_bucket("10 Yr"), tenor_bucket("6 Mo"), tenor_bucket(20.0)
    ('10y', 'bill', '30y')
    """
    if isinstance(tenor, str):
        if tenor in BUCKETS:  # already a bucket -- idempotent
            return tenor
        try:
            years = TENOR_YEARS[tenor]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(
                f"Unknown tenor label {tenor!r}; expected one of {sorted(TENOR_YEARS)} "
                f"or a bucket in {BUCKETS}"
            ) from exc
    else:
        years = float(tenor)

    if not np.isfinite(years) or years <= 0:
        raise ValueError(f"Tenor must be a positive number of years, got {years!r}")

    for edge, bucket in _BUCKET_EDGES:
        if years < edge:
            return bucket
    return "30y"  # pragma: no cover - unreachable, np.inf edge catches everything


@dataclass
class CostModel:
    """Execution costs for cash Treasuries, in dollars.

    Parameters
    ----------
    cfg : CostConfig
        Half-spreads (in 32nds, by bucket), the square-root impact coefficient,
        the repo spread and the per-million commission.
    adv : Mapping[str, float], optional
        Average daily dollar volume per bucket.  Defaults to :data:`DEFAULT_ADV`.
    slippage_multiplier : float, default 1.0
        Global stress dial applied to spread *and* impact (not to commission,
        which is contractual).  ``BacktestConfig.slippage_multiplier`` feeds
        this; run the backtest at 2.0 to see whether the edge survives a market
        that is twice as wide as assumed.

    Notes
    -----
    All costs are returned as **positive dollars** -- they are subtracted from
    P&L by the caller.  Notionals are dollars of face; a sign on the notional is
    ignored (crossing the spread costs the same whether you are buying or
    selling), except in :meth:`financing`, where direction genuinely matters.
    """

    cfg: CostConfig
    adv: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_ADV))
    slippage_multiplier: float = 1.0

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: Any) -> "CostModel":
        """Build from either a root :class:`~tqe.config.Config` or a ``CostConfig``.

        Picks up ``cfg.backtest.slippage_multiplier`` when handed a root config so
        the stress dial does not have to be wired twice.
        """
        costs = getattr(cfg, "costs", cfg)
        mult = float(getattr(getattr(cfg, "backtest", None), "slippage_multiplier", 1.0))
        return cls(cfg=costs, slippage_multiplier=mult)

    # ------------------------------------------------------------------ #
    # primitives
    # ------------------------------------------------------------------ #
    def _bucket(self, bucket: str) -> str:
        """Accept either a bucket or a tenor label, always return a bucket."""
        return bucket if bucket in self.cfg.half_spread_32nds else tenor_bucket(bucket)

    def half_spread(self, bucket: str) -> float:
        """Half the quoted bid/offer, in **price points** (not 32nds).

        Parameters
        ----------
        bucket : str
            A cost bucket (:data:`BUCKETS`) or a tenor label, which is bucketed
            for you.

        Returns
        -------
        float
            Half-spread in points of par per 100 face, e.g. ``0.015625`` for the
            10y (= 0.5/32).

        Notes
        -----
        The config stores 32nds because that is how a Treasury desk states a
        market ("the ten-year is half a tick wide"); every calculation below
        needs points, so the conversion happens here exactly once.  Multiply by
        100 to read the answer in basis points *of price*; divide by the bond's
        DV01 (points per bp) to read it in basis points *of yield* -- for the
        10y that is 1.5625bp of price / 0.08 = ~0.20bp of yield.
        """
        b = self._bucket(bucket)
        try:
            ticks = float(self.cfg.half_spread_32nds[b])
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(
                f"No half-spread configured for bucket {b!r}; have "
                f"{sorted(self.cfg.half_spread_32nds)}"
            ) from exc
        return ticks / TICKS_PER_POINT * self.slippage_multiplier

    def quoted_spread(self, bucket: str) -> float:
        """Full quoted bid/offer width in price points (``2 * half_spread``)."""
        return 2.0 * self.half_spread(bucket)

    def adv_for(self, bucket: str, adv: float | None = None) -> float:
        """Average daily dollar volume used to normalise trade size."""
        if adv is not None:
            return float(adv)
        b = self._bucket(bucket)
        return float(self.adv.get(b, DEFAULT_ADV[b]))

    # ------------------------------------------------------------------ #
    # cost components
    # ------------------------------------------------------------------ #
    def spread_cost(self, notional: float | np.ndarray, bucket: str) -> float | np.ndarray:
        """Cost of crossing half the bid/offer, in dollars.

        Parameters
        ----------
        notional : float or ndarray
            Dollars of face traded.  The sign is ignored.
        bucket : str
            Cost bucket or tenor label.

        Returns
        -------
        float or ndarray
            ``|notional| * half_spread / 100``.

        Notes
        -----
        Half-spreads are quoted per **100 face**, hence the ``/ 100``: a
        half-spread of 0.015625 points on $10mm costs
        ``10e6 * 0.015625/100 = $1,562.50``.  A round trip pays this twice --
        once on the way in, once on the way out -- which the backtest gets for
        free because it charges every position *change*.
        """
        return np.abs(notional) * self.half_spread(bucket) / 100.0

    def impact_cost(
        self,
        notional: float | np.ndarray,
        bucket: str,
        adv: float | None = None,
    ) -> float | np.ndarray:
        r"""Square-root market impact, in dollars.

        .. math::
            \text{impact (points)} = c \cdot s \cdot \sqrt{Q / V}

        with :math:`c` the impact coefficient, :math:`s` the full quoted spread
        in price points, :math:`Q` the order size and :math:`V` average daily
        volume.

        Parameters
        ----------
        notional : float or ndarray
            Dollars of face traded (sign ignored).
        bucket : str
            Cost bucket or tenor label.
        adv : float, optional
            Override the bucket's average daily dollar volume.

        Returns
        -------
        float or ndarray
            Impact cost in dollars.

        Notes
        -----
        The square-root law is the most robust empirical regularity in
        market-microstructure: price impact grows with the *square root* of
        participation rate, not linearly (Torre & Ferrari; Almgren et al. 2005;
        Toth et al. 2011 across asset classes).  Two consequences the backtest
        depends on:

        1. In **dollar** terms the charge scales as :math:`Q^{1.5}`, so doubling
           trade size nearly triples the impact bill.  That convexity is what
           makes a turnover penalty meaningful in the optimiser.
        2. Impact is scaled by the *spread*, so it automatically widens in the
           30y bucket and in stressed regimes (via ``slippage_multiplier``).

        At realistic sizes in Treasuries impact is genuinely small -- a $10mm
        clip is ~0.005% of the 10y sector's daily volume -- and the model says
        so rather than inventing a penalty.  It only bites at hundreds of
        millions, which is exactly the point of a size-aware cost model.
        """
        size = np.abs(notional)
        participation = size / self.adv_for(bucket, adv)
        impact_points = (
            float(self.cfg.impact_coefficient) * self.quoted_spread(bucket) * np.sqrt(participation)
        )
        return size * impact_points / 100.0

    def commission(self, notional: float | np.ndarray) -> float | np.ndarray:
        """Brokerage / clearing fees, in dollars.

        Notes
        -----
        Quoted per million of face -- the market convention for a cash Treasury
        ticket.  Unlike spread and impact this is contractual, so the
        ``slippage_multiplier`` stress dial does not touch it.
        """
        return np.abs(notional) / 1e6 * float(self.cfg.commission_per_million)

    def financing(self, notional: float, days: float, repo_rate: float) -> float:
        """Repo financing over ``days``, on **ACT/360**, in dollars.

        Parameters
        ----------
        notional : float
            **Signed** dollars of face: positive = long (you borrow cash and
            pledge the bond), negative = short (you lend cash via reverse repo
            and borrow the bond).
        days : float
            Calendar days held.  Repo accrues on calendar days, including
            weekends -- a Friday trade finances three days over the weekend,
            which is why the convention is ACT/360 and not a business-day count.
        repo_rate : float
            General-collateral repo rate as a decimal (``0.0430`` = 4.30%).

        Returns
        -------
        float
            Positive = cost to the strategy, negative = a financing credit.

        Notes
        -----
        ACT/360 is the money-market convention: repo, fed funds and bill
        discount rates all accrue actual days over a 360-day year, which is why
        an "overnight 4.30%" actually pays ``4.30% * 1/360`` and a repo rate is
        never directly comparable to a bond yield (ACT/ACT, semi-annual) without
        conversion.

        The spread is charged on **gross** notional in both directions:

        * long the bond, you finance at GC + spread;
        * short the bond, you must borrow it in the repo market, and a
          sought-after issue goes *special* -- the reverse-repo rate you earn
          drops below GC, sometimes to the fails charge floor.  Either way the
          desk pays the spread, so the algebra is
          ``repo * notional + spread * |notional|``.

        This is a *holding* cost and is intentionally not part of
        :meth:`total_cost`; the backtest applies it to the carried position, not
        to the day's trade.
        """
        spread = float(self.cfg.repo_spread_bp) / 10_000.0
        return (float(repo_rate) * float(notional) + spread * abs(float(notional))) * float(days) / 360.0

    # ------------------------------------------------------------------ #
    # aggregate
    # ------------------------------------------------------------------ #
    def total_cost(
        self,
        trade_notional: float | np.ndarray,
        bucket: str,
        adv: float | None = None,
    ) -> float | np.ndarray:
        """Total round-of-execution cost in **dollars** for a traded notional.

        Parameters
        ----------
        trade_notional : float or ndarray
            Dollars of face traded (sign ignored -- both directions cross the
            spread).
        bucket : str
            Cost bucket or tenor label.
        adv : float, optional
            Override average daily dollar volume.

        Returns
        -------
        float or ndarray
            ``spread + impact + commission``, in dollars.  Financing is excluded
            (see :meth:`financing`).

        Examples
        --------
        A $10mm on-the-run 10y trade at the default config:
        $1,562.50 spread + ~$4 impact + $125 commission = ~$1,691 -- about
        1.7bp of price, or a fifth of a basis point of yield.
        """
        return (
            self.spread_cost(trade_notional, bucket)
            + self.impact_cost(trade_notional, bucket, adv)
            + self.commission(trade_notional)
        )

    def cost_breakdown(
        self,
        trade_notional: float,
        bucket: str,
        adv: float | None = None,
    ) -> dict[str, float]:
        """Per-component dollar costs plus the same numbers expressed in bp.

        Returns
        -------
        dict
            ``spread``, ``impact``, ``commission``, ``total`` (dollars);
            ``total_bp_of_notional`` and ``bucket``-level diagnostics.  Handy in
            a tearsheet, where "we paid 1.9bp per trade" is the number a PM
            actually challenges.
        """
        b = self._bucket(bucket)
        spread = float(self.spread_cost(trade_notional, b))
        impact = float(self.impact_cost(trade_notional, b, adv))
        comm = float(self.commission(trade_notional))
        total = spread + impact + comm
        notional = abs(float(trade_notional))
        return {
            "bucket": b,
            "notional": notional,
            "spread": spread,
            "impact": impact,
            "commission": comm,
            "total": total,
            "half_spread_points": self.half_spread(b),
            "half_spread_32nds": self.half_spread(b) * TICKS_PER_POINT,
            "total_bp_of_notional": (total / notional * 10_000.0) if notional > 0 else 0.0,
            "participation": (notional / self.adv_for(b, adv)) if notional > 0 else 0.0,
        }


# --------------------------------------------------------------------------- #
# Time series
# --------------------------------------------------------------------------- #
def _resolve_buckets(
    columns: Sequence[str],
    buckets: Mapping[str, str] | Sequence[str] | None,
) -> dict[str, str]:
    """Normalise the ``buckets`` argument to ``{column: bucket}``."""
    if buckets is None:
        return {c: tenor_bucket(c) for c in columns}
    if isinstance(buckets, Mapping):
        return {c: buckets.get(c, tenor_bucket(c)) for c in columns}
    if len(buckets) != len(columns):
        raise ValueError(
            f"buckets has length {len(buckets)} but positions has {len(columns)} columns"
        )
    return dict(zip(columns, buckets))


def turnover_cost_series(
    positions: pd.DataFrame,
    cost_model: CostModel,
    buckets: Mapping[str, str] | Sequence[str] | None = None,
) -> pd.Series:
    """Daily dollar transaction cost implied by a path of positions.

    Parameters
    ----------
    positions : pd.DataFrame
        Signed **dollar notional** per instrument, ``DatetimeIndex`` named
        ``date``, columns = tenor labels (or bucket names).  NaN is read as
        "flat" -- a tenor that has not started publishing yet cannot be held.
    cost_model : CostModel
        The cost model to charge with.
    buckets : Mapping or Sequence, optional
        Column -> cost bucket.  Defaults to :func:`tenor_bucket` on each column
        label.

    Returns
    -------
    pd.Series
        Dollar cost per date, named ``cost``, aligned to ``positions.index``.

    Notes
    -----
    **Causality.**  The cost on day ``t`` is charged on
    ``position_t - position_{t-1}``.  Both quantities are known at the moment the
    day-``t`` order is sent (the target came from a signal built on data up to
    ``t-1``), so no future information enters.  The first row is charged in full:
    establishing the initial book is a real trade and pretending otherwise is a
    small, systematic overstatement of the first fold's returns.

    Costs are computed per column and summed, because the half-spread is
    bucket-dependent and the impact term is non-linear -- netting a long 2y
    against a short 30y before costing would understate the bill.
    """
    if positions.empty:
        return pd.Series(dtype=float, index=positions.index, name="cost")

    columns = list(positions.columns)
    mapping = _resolve_buckets(columns, buckets)

    # Copy-on-write safe: ``pos`` and ``trades`` are fresh objects, so the
    # ``.iloc[0]`` assignment below is a direct write, never chained.
    pos = positions.astype(float).fillna(0.0)
    trades = pos.diff()
    trades.iloc[0] = pos.iloc[0]

    total = np.zeros(len(pos), dtype=float)
    for col in columns:
        traded = trades[col].to_numpy(dtype=float, copy=False)
        total += np.asarray(cost_model.total_cost(traded, mapping[col]), dtype=float)

    return pd.Series(total, index=pos.index, name="cost")
