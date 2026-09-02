"""Independent hand-computation of the constant-maturity total return.

Rebuilt from scratch with plain datetime arithmetic and the street price
formula, no tqe pricing code involved.
"""
import pandas as pd, numpy as np, sys
from datetime import date, timedelta
sys.path.insert(0,"src")
from tqe.data.calendar import settlement_date
from tqe.data.universe import constant_maturity_total_return

def add_months(d, m):
    y = d.year + (d.month - 1 + m)//12
    mo = (d.month - 1 + m)%12 + 1
    # clamp
    import calendar as cal
    dd = min(d.day, cal.monthrange(y, mo)[1])
    return date(y, mo, dd)

def price(settle, maturity, coupon_rate, ytm, freq=2):
    """dirty, clean, accrued of a semiannual bond, street convention."""
    # coupon dates backward from maturity
    dates = []
    k = 0
    while True:
        cd = add_months(maturity, -6*k)
        if cd <= settle:
            break
        dates.append(cd)
        k += 1
    dates = sorted(dates)
    prev = add_months(dates[0], -6)
    period = (dates[0]-prev).days
    w = (dates[0]-settle).days / period
    c = 100.0*coupon_rate/freq
    d = ytm/freq
    dirty = 0.0
    for i, cd in enumerate(dates):
        cf = c + (100.0 if i == len(dates)-1 else 0.0)
        dirty += cf * (1.0+d)**(-(i+w))
    accrued = c*(1.0-w)
    return dirty, dirty-accrued, accrued, len(dates), w

c = pd.read_parquet("data/processed/curve.parquet")
core = ["3 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","30 Yr"]
rets = constant_maturity_total_return(c, core)

TESTS = [("10 Yr",120,"2016-11-10"), ("10 Yr",120,"2020-03-09"), ("2 Yr",24,"2023-03-13"),
         ("30 Yr",360,"2011-08-08"), ("5 Yr",60,"2005-06-15"), ("3 Mo",3,"2008-10-15")]
for tenor, months, ds in TESTS:
    ts = pd.Timestamp(ds)
    idx = c.index
    i = idx.get_loc(ts)
    t0, t1 = idx[i-1].date(), idx[i].date()
    y0, y1 = c[tenor].iloc[i-1], c[tenor].iloc[i]
    s0, s1 = settlement_date(t0), settlement_date(t1)
    m0 = add_months(s0, months)
    # leg1: yesterday's par bond at yesterday's yield
    d0, cl0, a0, n0, w0 = price(s0, m0, y0, y0)
    # leg2: SAME bond at today's yield, today's settle
    d1, cl1, a1, n1, w1 = price(s1, m0, y0, y1)
    pr = cl1/cl0 - 1.0
    carry = a1 - a0
    tot = pr + carry/100.0
    f = rets[tenor]
    print(f"\n{tenor} {ds}: prev={t0} y0={y0*100:.3f}%  y1={y1*100:.3f}%  dy={(y1-y0)*1e4:+.1f}bp")
    print(f"  settle {s0} -> {s1}, maturity {m0}, ncf {n0}/{n1}")
    print(f"  HAND  clean0={cl0:.8f} clean1={cl1:.8f} acc0={a0:.8f} acc1={a1:.8f}")
    print(f"  HAND  price_return={pr:+.8f}  carry={carry:+.8f}  total={tot:+.8f}")
    print(f"  CODE  price_return={f['price_return'].iloc[i]:+.8f}  carry={f['carry_1d'].iloc[i]:+.8f}  total={f['total_return'].iloc[i]:+.8f}")
    print(f"  diff total = {tot - f['total_return'].iloc[i]:+.2e}")
