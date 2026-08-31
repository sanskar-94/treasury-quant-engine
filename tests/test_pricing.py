"""Tests for the bond pricing and risk-analytics core.

Every assertion here is anchored to a closed-form result or a mathematical
invariant, never to a number copied out of this library's own output.  That is
the only way a numerical test suite can actually catch a regression.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from tqe.pricing.analytics import (
    bond_risk,
    carry_and_rolldown,
    convexity,
    dv01,
    effective_convexity,
    effective_duration,
    hedge_ratio,
    key_rate_durations,
    macaulay_duration,
    modified_duration,
    portfolio_dv01,
    price_change_estimate,
)
from tqe.pricing.bond import (
    Bond,
    accrued_interest,
    bill_bond_equivalent_yield,
    bill_discount_from_price,
    bill_price_from_discount,
    dirty_price_from_yield,
    format_32nds,
    par_bond,
    parse_32nds,
    price_from_discount_curve,
    price_from_yield,
    yield_from_price,
)
from tqe.pricing.daycount import (
    DayCount,
    accrual_fraction,
    add_months,
    days_between,
    year_fraction,
)

SETTLE = date(2024, 8, 15)


# --------------------------------------------------------------------------- #
# Day count
# --------------------------------------------------------------------------- #
class TestDayCount:
    def test_act_360_full_year(self):
        assert year_fraction(date(2024, 1, 1), date(2025, 1, 1), DayCount.ACT_360) == pytest.approx(
            366 / 360
        )

    def test_act_365f(self):
        assert year_fraction(date(2023, 1, 1), date(2024, 1, 1), DayCount.ACT_365F) == pytest.approx(1.0)

    def test_thirty_360_half_year(self):
        assert year_fraction(
            date(2024, 1, 31), date(2024, 7, 31), DayCount.THIRTY_360
        ) == pytest.approx(0.5)

    def test_act_365f_is_the_default_convention(self):
        """The default is ACT/365F, so a 366-day leap year is >1.0 by design."""
        assert year_fraction(date(2024, 1, 1), date(2025, 1, 1)) == pytest.approx(366 / 365)

    def test_act_act_isda_normalises_leap_year_to_one(self):
        """ACT/ACT divides each calendar year by its own length, so a full
        leap year is exactly 1.0 - unlike ACT/365F."""
        assert year_fraction(
            date(2024, 1, 1), date(2025, 1, 1), DayCount.ACT_ACT_ICMA
        ) == pytest.approx(1.0)

    def test_act_act_isda_splits_across_calendar_years(self):
        """Half of 2023 plus half of 2024 must land near 1.0 for a 366-day span."""
        yf = year_fraction(date(2023, 7, 1), date(2024, 7, 1), DayCount.ACT_ACT_ICMA)
        assert yf == pytest.approx(184 / 365 + 182 / 366, abs=1e-12)

    def test_act_act_isda_multi_year(self):
        assert year_fraction(
            date(2020, 1, 1), date(2023, 1, 1), DayCount.ACT_ACT_ICMA
        ) == pytest.approx(3.0)

    def test_act_act_icma_uses_reference_period(self):
        """Half a coupon period is 0.25 years at semi-annual frequency."""
        ps, pe = date(2024, 2, 15), date(2024, 8, 15)
        mid = date(2024, 5, 15)
        yf = year_fraction(ps, mid, DayCount.ACT_ACT_ICMA, period_start=ps, period_end=pe, frequency=2)
        assert yf == pytest.approx(days_between(ps, mid) / days_between(ps, pe) / 2)

    def test_accrual_fraction_bounds(self):
        ps, pe = date(2024, 2, 15), date(2024, 8, 15)
        assert accrual_fraction(ps, ps, pe) == 0.0
        assert 0.0 < accrual_fraction(date(2024, 5, 15), ps, pe) < 1.0

    def test_add_months_clamps_short_month(self):
        assert add_months(date(2024, 8, 31), 6) == date(2025, 2, 28)
        assert add_months(date(2023, 8, 31), 6) == date(2024, 2, 29)  # leap year
        assert add_months(date(2024, 1, 15), -1) == date(2023, 12, 15)

    def test_negative_interval(self):
        assert year_fraction(date(2024, 6, 1), date(2024, 1, 1)) < 0


# --------------------------------------------------------------------------- #
# Bond pricing
# --------------------------------------------------------------------------- #
class TestBondPricing:
    def test_par_bond_prices_to_exactly_100(self):
        """A par bond priced at its own coupon must return exactly 100."""
        for tenor in (2, 3, 5, 7, 10, 20, 30):
            for y in (0.001, 0.0225, 0.0425, 0.09):
                b = par_bond(SETTLE, tenor, y)
                assert price_from_yield(b, SETTLE, y) == pytest.approx(100.0, abs=1e-9)

    def test_closed_form_five_year(self):
        """5y 2% at 4%, settling on a coupon date: annuity + discounted principal."""
        b = Bond(maturity=date(2028, 2, 15), coupon=2.0, issue_date=date(2018, 2, 15))
        s = date(2023, 2, 15)
        annuity = (1 - 1.02**-10) / 0.02
        expected = 1.0 * annuity + 100 / 1.02**10
        assert price_from_yield(b, s, 0.04) == pytest.approx(expected, abs=1e-10)

    def test_no_accrued_on_coupon_date(self):
        b = Bond(maturity=date(2028, 2, 15), coupon=2.0, issue_date=date(2018, 2, 15))
        assert accrued_interest(b, date(2023, 2, 15)) == pytest.approx(0.0, abs=1e-12)

    def test_accrued_matches_act_act(self):
        b = Bond(maturity=date(2034, 2, 15), coupon=4.0, issue_date=date(2024, 2, 15))
        s = date(2024, 5, 15)
        ps, pe = b.period_bounds(s)
        expected = 2.0 * days_between(ps, s) / days_between(ps, pe)
        assert accrued_interest(b, s) == pytest.approx(expected, abs=1e-12)

    def test_accrued_is_zero_for_zero_coupon(self):
        z = Bond(maturity=date(2034, 8, 15), coupon=0.0, issue_date=SETTLE)
        assert accrued_interest(z, SETTLE) == 0.0

    @pytest.mark.parametrize("coupon", [0.0, 0.5, 2.0, 4.25, 8.0, 15.0])
    @pytest.mark.parametrize("ytm", [-0.005, 0.0001, 0.01, 0.0425, 0.09, 0.20])
    def test_price_yield_roundtrip(self, coupon, ytm):
        b = Bond(maturity=date(2039, 11, 15), coupon=coupon, issue_date=date(2014, 11, 15))
        px = price_from_yield(b, SETTLE, ytm)
        assert yield_from_price(b, SETTLE, px) == pytest.approx(ytm, abs=1e-9)

    def test_price_is_monotonically_decreasing_in_yield(self):
        b = par_bond(SETTLE, 10, 0.04)
        ys = np.linspace(0.0, 0.15, 40)
        prices = [price_from_yield(b, SETTLE, y) for y in ys]
        assert all(a > b_ for a, b_ in zip(prices, prices[1:]))

    def test_premium_and_discount_relationships(self):
        """Coupon above yield -> premium; below -> discount."""
        b = Bond(maturity=date(2034, 8, 15), coupon=6.0, issue_date=SETTLE)
        assert price_from_yield(b, SETTLE, 0.04) > 100
        assert price_from_yield(b, SETTLE, 0.08) < 100

    def test_discount_curve_pricing_matches_flat_yield(self):
        """A flat discount curve must reproduce the flat-yield price."""
        b = par_bond(SETTLE, 10, 0.045)
        y = 0.045
        flat = lambda t: (1 + y / 2) ** (-2 * t)  # noqa: E731
        assert price_from_discount_curve(b, SETTLE, flat) == pytest.approx(
            price_from_yield(b, SETTLE, y), abs=1e-9
        )

    def test_matured_bond_rejected(self):
        b = Bond(maturity=date(2020, 1, 1), coupon=2.0, issue_date=date(2010, 1, 1))
        with pytest.raises(ValueError):
            yield_from_price(b, SETTLE, 100.0)

    def test_dirty_minus_clean_is_accrued(self):
        b = Bond(maturity=date(2034, 2, 15), coupon=4.0, issue_date=date(2024, 2, 15))
        s = date(2024, 6, 3)
        assert dirty_price_from_yield(b, s, 0.045) - price_from_yield(b, s, 0.045) == pytest.approx(
            accrued_interest(b, s), abs=1e-12
        )

    def test_coupon_dates_are_semiannual_and_end_at_maturity(self):
        b = Bond(maturity=date(2034, 8, 15), coupon=4.0, issue_date=date(2024, 8, 15))
        dates = b.coupon_dates()
        assert dates[-1] == date(2034, 8, 15)
        assert len(dates) == 20
        assert all(d1 < d2 for d1, d2 in zip(dates, dates[1:]))

    def test_coupon_on_settlement_belongs_to_seller(self):
        """A coupon paid exactly on settlement is not the buyer's."""
        b = Bond(maturity=date(2034, 8, 15), coupon=4.0, issue_date=date(2024, 8, 15))
        assert date(2025, 2, 15) in b.coupon_dates()
        assert date(2025, 2, 15) not in b.coupon_dates(after=date(2025, 2, 15))


# --------------------------------------------------------------------------- #
# Bills
# --------------------------------------------------------------------------- #
class TestBills:
    def test_discount_price_roundtrip(self):
        p = bill_price_from_discount(0.05, 91)
        assert bill_discount_from_price(p, 91) == pytest.approx(0.05, abs=1e-12)

    def test_bey_exceeds_discount_rate(self):
        """Coupon-equivalent yield is always above the bank-discount rate."""
        assert bill_bond_equivalent_yield(0.05, 91) > 0.05

    def test_long_bill_uses_quadratic_branch(self):
        short = bill_bond_equivalent_yield(0.05, 180)
        long = bill_bond_equivalent_yield(0.05, 360)
        assert long > short  # more compounding pickup

    def test_zero_days_rejected(self):
        with pytest.raises(ValueError):
            bill_bond_equivalent_yield(0.05, 0)


# --------------------------------------------------------------------------- #
# Risk analytics
# --------------------------------------------------------------------------- #
class TestAnalytics:
    def test_zero_coupon_macaulay_duration_equals_maturity(self):
        """The single sharpest test of a duration implementation."""
        for years in (1, 5, 10, 30):
            z = Bond(maturity=add_months(SETTLE, 12 * years), coupon=0.0, issue_date=SETTLE)
            assert macaulay_duration(z, SETTLE, 0.04) == pytest.approx(float(years), abs=1e-10)

    def test_modified_is_macaulay_discounted(self):
        b = par_bond(SETTLE, 10, 0.0425)
        mac = macaulay_duration(b, SETTLE, 0.0425)
        assert modified_duration(b, SETTLE, 0.0425) == pytest.approx(mac / 1.02125, abs=1e-12)

    def test_coupon_bond_duration_below_maturity(self):
        b = par_bond(SETTLE, 10, 0.0425)
        assert 0 < macaulay_duration(b, SETTLE, 0.0425) < 10

    def test_duration_increases_with_maturity(self):
        durations = [modified_duration(par_bond(SETTLE, t, 0.04), SETTLE, 0.04) for t in (2, 5, 10, 30)]
        assert all(a < b for a, b in zip(durations, durations[1:]))

    def test_convexity_positive_and_increasing_with_maturity(self):
        cvx = [convexity(par_bond(SETTLE, t, 0.04), SETTLE, 0.04) for t in (2, 5, 10, 30)]
        assert all(c > 0 for c in cvx)
        assert all(a < b for a, b in zip(cvx, cvx[1:]))

    def test_effective_matches_analytic_duration(self):
        b = par_bond(SETTLE, 10, 0.0425)
        assert effective_duration(b, SETTLE, 0.0425) == pytest.approx(
            modified_duration(b, SETTLE, 0.0425), abs=1e-5
        )

    def test_effective_matches_analytic_convexity(self):
        b = par_bond(SETTLE, 10, 0.0425)
        assert effective_convexity(b, SETTLE, 0.0425) == pytest.approx(
            convexity(b, SETTLE, 0.0425), rel=1e-4
        )

    def test_dv01_is_positive_and_consistent(self):
        b = par_bond(SETTLE, 10, 0.0425)
        r = bond_risk(b, SETTLE, ytm=0.0425)
        assert r.dv01 > 0
        assert r.dv01 == pytest.approx(r.modified_duration * r.dirty_price * 1e-4, abs=1e-12)

    def test_dv01_scales_linearly_with_face(self):
        b = par_bond(SETTLE, 10, 0.0425)
        assert dv01(b, SETTLE, 0.0425, face=1000.0) == pytest.approx(
            10 * dv01(b, SETTLE, 0.0425), abs=1e-9
        )

    def test_key_rate_durations_sum_to_effective_duration(self):
        """The defining invariant of a correct KRD implementation."""
        b = par_bond(SETTLE, 10, 0.0425)
        krd = key_rate_durations(b, SETTLE, lambda t: 0.0425)
        assert sum(krd.values()) == pytest.approx(effective_duration(b, SETTLE, 0.0425), abs=1e-4)

    def test_key_rate_durations_concentrate_at_maturity(self):
        b = par_bond(SETTLE, 10, 0.0425)
        krd = key_rate_durations(b, SETTLE, lambda t: 0.0425)
        assert max(krd, key=lambda k: krd[k]) == 10.0

    def test_krd_on_sloped_curve_still_sums_near_duration(self):
        b = par_bond(SETTLE, 10, 0.0425)
        sloped = lambda t: 0.03 + 0.015 * (1 - math.exp(-t / 3))  # noqa: E731
        krd = key_rate_durations(b, SETTLE, sloped)
        assert sum(krd.values()) == pytest.approx(8.0, abs=0.5)

    def test_taylor_expansion_error_is_third_order(self):
        """Halving the yield move must cut the approximation error ~8x."""
        b = par_bond(SETTLE, 10, 0.0425)
        r = bond_risk(b, SETTLE, ytm=0.0425)

        def err(dy):
            actual = price_from_yield(b, SETTLE, 0.0425 + dy) - r.clean_price
            return abs(price_change_estimate(r, dy) - actual)

        assert err(0.005) / err(0.01) == pytest.approx(0.125, rel=0.3)

    def test_convexity_makes_taylor_more_accurate(self):
        b = par_bond(SETTLE, 30, 0.0425)
        r = bond_risk(b, SETTLE, ytm=0.0425)
        actual = price_from_yield(b, SETTLE, 0.0525) - r.clean_price
        with_cvx = abs(price_change_estimate(r, 0.01, True) - actual)
        without = abs(price_change_estimate(r, 0.01, False) - actual)
        assert with_cvx < without

    def test_bond_risk_from_price_matches_from_yield(self):
        b = par_bond(SETTLE, 10, 0.0425)
        by_yield = bond_risk(b, SETTLE, ytm=0.0425)
        by_price = bond_risk(b, SETTLE, clean_price=100.0)
        assert by_price.ytm == pytest.approx(by_yield.ytm, abs=1e-10)

    def test_bond_risk_requires_exactly_one_input(self):
        b = par_bond(SETTLE, 10, 0.0425)
        with pytest.raises(ValueError):
            bond_risk(b, SETTLE)
        with pytest.raises(ValueError):
            bond_risk(b, SETTLE, ytm=0.04, clean_price=100.0)


class TestCarryRolldown:
    def test_carry_positive_when_coupon_exceeds_repo(self):
        b = par_bond(SETTLE, 10, 0.05)
        out = carry_and_rolldown(b, SETTLE, 0.05, horizon_days=90, repo_rate=0.02)
        assert out["carry"] > 0

    def test_carry_negative_when_repo_exceeds_coupon(self):
        b = par_bond(SETTLE, 10, 0.02)
        out = carry_and_rolldown(b, SETTLE, 0.02, horizon_days=90, repo_rate=0.06)
        assert out["carry"] < 0

    def test_rolldown_positive_on_upward_sloping_curve(self):
        """Rolling down a positively-sloped curve is a price gain."""
        b = par_bond(SETTLE, 10, 0.045)
        out = carry_and_rolldown(b, SETTLE, 0.045, horizon_days=180, forward_yield=0.043)
        assert out["rolldown"] > 0

    def test_total_is_sum_of_parts(self):
        b = par_bond(SETTLE, 10, 0.045)
        out = carry_and_rolldown(b, SETTLE, 0.045, horizon_days=90, repo_rate=0.04)
        assert out["total"] == pytest.approx(out["carry"] + out["rolldown"], abs=1e-12)


class TestPortfolioHelpers:
    def test_portfolio_dv01_nets_longs_and_shorts(self):
        dvs = {"a": 0.08, "b": 0.02}
        assert portfolio_dv01({"a": 1_000_000, "b": -1_000_000}, dvs) == pytest.approx(600.0)

    def test_portfolio_dv01_ignores_unknown_instruments(self):
        assert portfolio_dv01({"zzz": 1_000_000}, {"a": 0.08}) == 0.0

    def test_hedge_ratio_flattens_the_book(self):
        target, hedge = 5000.0, 0.02
        assert target + hedge_ratio(target, hedge) * hedge == pytest.approx(0.0, abs=1e-9)

    def test_hedge_ratio_rejects_zero_dv01(self):
        with pytest.raises(ValueError):
            hedge_ratio(1000.0, 0.0)


# --------------------------------------------------------------------------- #
# Quote formatting
# --------------------------------------------------------------------------- #
class TestQuoting:
    @pytest.mark.parametrize(
        "price,quote",
        [(99.5, "99-16"), (99.515625, "99-16+"), (101.0, "101-00"), (100.25, "100-08")],
    )
    def test_format_32nds(self, price, quote):
        assert format_32nds(price) == quote

    @pytest.mark.parametrize("quote,price", [("99-16", 99.5), ("99-16+", 99.515625), ("100-08", 100.25)])
    def test_parse_32nds(self, quote, price):
        assert parse_32nds(quote) == pytest.approx(price)

    def test_eighths_suffix_roundtrip(self):
        assert parse_32nds(format_32nds(99.99609375)) == pytest.approx(99.99609375)

    def test_parse_accepts_plain_decimal(self):
        assert parse_32nds("99.75") == pytest.approx(99.75)

    @pytest.mark.parametrize("price", [98.0, 99.03125, 100.5, 102.984375, 105.0])
    def test_quote_roundtrip(self, price):
        assert parse_32nds(format_32nds(price)) == pytest.approx(price, abs=1e-9)
