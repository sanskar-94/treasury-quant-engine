"""Relative-value trade structures.

A rates desk does not think in terms of nine independent tenor positions. It
thinks in **structures**: a 2s10s steepener, a 5s10s30s butterfly, a duration-
weighted asset swap. Each is a specific combination of legs weighted so that the
trade expresses exactly one view and is neutral to everything else.

The weighting is the whole point and is where these trades are usually got
wrong. "Long 10mm of 2-year against 10mm of 10-year" is not a curve trade - it
is a large short-duration position with a curve view attached, because the
10-year carries four times the DV01. A steepener has to be **DV01-weighted** so
that a parallel shift nets to zero and only the *slope* moves the P&L.

Every constructor here returns notional weights per tenor that satisfy their
stated neutrality exactly, and the test suite verifies the neutrality rather
than trusting the algebra.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger("portfolio.structures")

__all__ = [
    "Structure",
    "steepener",
    "butterfly",
    "duration_neutral_pair",
    "cash_and_duration_neutral",
    "structure_returns",
    "STANDARD_STRUCTURES",
]

EPS = 1e-12


@dataclass(frozen=True)
class Structure:
    """A named combination of legs with its neutrality properties.

    ``weights`` are notional per unit of the structure, positive long. The
    ``net_dv01`` and ``net_notional`` fields record what the structure is
    actually neutral to, computed rather than asserted.
    """

    name: str
    weights: pd.Series
    net_dv01: float = 0.0
    net_notional: float = 0.0
    description: str = ""

    def dv01_of(self, dv01: pd.Series) -> float:
        """Net DV01 of one unit of the structure."""
        d = dv01.reindex(self.weights.index).fillna(0.0)
        return float((self.weights * d / 100.0).sum())

    def __repr__(self) -> str:
        legs = ", ".join(f"{k}:{v:+.3f}" for k, v in self.weights.items() if abs(v) > EPS)
        return f"Structure({self.name}, {legs})"


def _series(weights: Mapping[str, float], tenors: Sequence[str]) -> pd.Series:
    return pd.Series({t: float(weights.get(t, 0.0)) for t in tenors}, dtype=float)


def steepener(
    short_tenor: str,
    long_tenor: str,
    dv01: Mapping[str, float] | pd.Series,
    tenors: Sequence[str] | None = None,
    target_dv01: float = 1000.0,
) -> Structure:
    """A DV01-neutral curve steepener: long the short leg, short the long leg.

    Profits when the curve steepens - the long tenor's yield rises relative to
    the short one's. Sized so the two legs carry equal and opposite DV01, which
    is what makes a parallel shift a non-event and leaves only the slope.

    ``target_dv01`` is the DV01 *per leg*, so the structure risks that many
    dollars per basis point of slope change.
    """
    d = pd.Series(dict(dv01), dtype=float) if not isinstance(dv01, pd.Series) else dv01
    tenors = list(tenors) if tenors else list(d.index)
    ds, dl = float(d.get(short_tenor, 0.0)), float(d.get(long_tenor, 0.0))
    if abs(ds) < EPS or abs(dl) < EPS:
        raise ValueError(f"Zero DV01 on a leg ({short_tenor}={ds}, {long_tenor}={dl})")

    # notional = target_dv01 / (dv01 per 100 face) * 100
    w = _series({short_tenor: target_dv01 / ds * 100.0,
                 long_tenor: -target_dv01 / dl * 100.0}, tenors)
    s = Structure(
        name=f"steepener_{short_tenor}_{long_tenor}".replace(" ", ""),
        weights=w,
        description=f"long {short_tenor} / short {long_tenor}, DV01-neutral; profits on steepening",
    )
    return Structure(s.name, w, net_dv01=s.dv01_of(d), net_notional=float(w.sum()),
                     description=s.description)


def butterfly(
    short_tenor: str,
    belly_tenor: str,
    long_tenor: str,
    dv01: Mapping[str, float] | pd.Series,
    tenors: Sequence[str] | None = None,
    target_dv01: float = 1000.0,
    fifty_fifty: bool = True,
) -> Structure:
    """A DV01-neutral butterfly: the belly against the two wings.

    Long belly / short wings profits when the belly **richens** relative to the
    wings - the classic bet that a kink in the curve closes. Sized so total DV01
    nets to zero, making the trade neutral to both level and (with the
    fifty-fifty weighting) slope, leaving pure curvature.

    ``fifty_fifty=True`` splits the wing DV01 equally, which is the desk default
    and gives a trade neutral to a linear twist as well as to a parallel shift.
    Setting it False weights the wings by their distance from the belly, which
    keeps the trade neutral to a shift but leaves a small slope exposure.
    """
    d = pd.Series(dict(dv01), dtype=float) if not isinstance(dv01, pd.Series) else dv01
    tenors = list(tenors) if tenors else list(d.index)
    ds, db, dl = (float(d.get(t, 0.0)) for t in (short_tenor, belly_tenor, long_tenor))
    if min(abs(ds), abs(db), abs(dl)) < EPS:
        raise ValueError("Zero DV01 on a butterfly leg")

    if fifty_fifty:
        wing_short = wing_long = 0.5
    else:
        from ..data.sources import TENOR_YEARS

        ys, yb, yl = (TENOR_YEARS.get(t, np.nan) for t in (short_tenor, belly_tenor, long_tenor))
        span = (yl - ys) or 1.0
        wing_short, wing_long = (yl - yb) / span, (yb - ys) / span

    w = _series({
        belly_tenor: target_dv01 / db * 100.0,
        short_tenor: -target_dv01 * wing_short / ds * 100.0,
        long_tenor: -target_dv01 * wing_long / dl * 100.0,
    }, tenors)
    name = f"fly_{short_tenor}_{belly_tenor}_{long_tenor}".replace(" ", "")
    s = Structure(name, w, description=(
        f"long {belly_tenor} vs {short_tenor}/{long_tenor} wings, DV01-neutral; "
        "profits when the belly richens"))
    return Structure(name, w, net_dv01=s.dv01_of(d), net_notional=float(w.sum()),
                     description=s.description)


def duration_neutral_pair(
    tenor_a: str,
    tenor_b: str,
    dv01: Mapping[str, float] | pd.Series,
    tenors: Sequence[str] | None = None,
    target_dv01: float = 1000.0,
) -> Structure:
    """Long ``tenor_a`` against ``tenor_b``, DV01-neutral. Alias of a steepener."""
    return steepener(tenor_a, tenor_b, dv01, tenors, target_dv01)


def cash_and_duration_neutral(
    weights: pd.Series,
    dv01: pd.Series,
) -> pd.Series:
    """Project a book onto zero net notional **and** zero net DV01.

    The orthogonal projection onto the null space of the two constraint vectors,
    so it is the closest doubly-neutral book to the one requested rather than an
    arbitrary rescaling of one leg.

    This is what turns an arbitrary set of views into a fundable relative-value
    book: zero net notional means no financing position, zero net DV01 means no
    directional rates bet, and what remains is a pure view on the shape of the
    curve.
    """
    idx = weights.index
    w = weights.to_numpy(dtype=float)
    d = dv01.reindex(idx).fillna(0.0).to_numpy(dtype=float)
    A = np.vstack([np.ones_like(d), d])
    try:
        w = w - A.T @ np.linalg.solve(A @ A.T, A @ w)
    except np.linalg.LinAlgError:
        log.warning("degenerate constraints; book returned unchanged")
    return pd.Series(w, index=idx, name=weights.name)


def structure_returns(
    structure: Structure,
    returns_panel: pd.DataFrame,
    capital: float = 1.0,
) -> pd.Series:
    """Daily return of holding one unit of a structure."""
    w = structure.weights.reindex(returns_panel.columns).fillna(0.0)
    return (returns_panel.fillna(0.0) * w).sum(axis=1) / max(capital, EPS)


#: The structures a rates desk quotes by name.
STANDARD_STRUCTURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("steepener", ("2 Yr", "10 Yr")),
    ("steepener", ("5 Yr", "30 Yr")),
    ("steepener", ("3 Mo", "2 Yr")),
    ("butterfly", ("2 Yr", "5 Yr", "10 Yr")),
    ("butterfly", ("5 Yr", "10 Yr", "30 Yr")),
    ("butterfly", ("1 Yr", "3 Yr", "7 Yr")),
)


def build_standard_structures(
    dv01: Mapping[str, float] | pd.Series,
    tenors: Sequence[str] | None = None,
    target_dv01: float = 1000.0,
) -> list[Structure]:
    """Every standard structure whose legs are present in ``dv01``."""
    d = pd.Series(dict(dv01), dtype=float) if not isinstance(dv01, pd.Series) else dv01
    tenors = list(tenors) if tenors else list(d.index)
    out: list[Structure] = []
    for kind, legs in STANDARD_STRUCTURES:
        if not all(leg in d.index and abs(float(d[leg])) > EPS for leg in legs):
            continue
        try:
            if kind == "steepener":
                out.append(steepener(legs[0], legs[1], d, tenors, target_dv01))
            else:
                out.append(butterfly(legs[0], legs[1], legs[2], d, tenors, target_dv01))
        except ValueError as exc:
            log.debug("skipping %s %s: %s", kind, legs, exc)
    return out
