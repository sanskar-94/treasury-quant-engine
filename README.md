# Treasury Quant Engine

An end-to-end quantitative trading system for the US Treasury market: yield-curve
modelling, machine-learning return forecasting, risk-aware portfolio
construction, cost-accurate backtesting, and automated execution with pre-trade
risk controls.

Built on **36 years of real Treasury data** (9,172 trading days, 1990–2026) pulled
from Treasury.gov and FRED. No paid data, no vendor libraries — the bond maths,
the curve models and the backtester are implemented from first principles.

```bash
git clone https://github.com/sanskar-94/treasury-quant-engine.git
cd treasury-quant-engine
python3 -m venv .venv && ./.venv/bin/pip install -e ".[all]"
./.venv/bin/tqe pipeline        # data → curve → features → train → backtest
```

---

## Headline result

Walk-forward out-of-sample, **2018-08-03 to 2026-08-27** (2,016 trading days,
8 non-overlapping annual folds). Every prediction was made by a model that had
never seen the day it was predicting, nor anything after it.

| | Strategy (net) | Strategy (gross) | Buy & hold 10y |
| --- | ---: | ---: | ---: |
| Sharpe ratio | **1.23** | 2.35 | 0.18 |
| Annualised return | 2.39% | 4.52% | 1.40% |
| Annualised volatility | 1.94% | 1.92% | 7.69% |
| Maximum drawdown | −10.28% | — | −27.11% |
| Hit rate | 56.14% | — | — |
| Sortino / Calmar | 1.89 / 0.23 | — | — |

Information ratio vs the benchmark is 0.14 with a correlation of only 0.31, so
this is largely an independent return stream rather than levered duration.
Average gross notional is $37.2mm on $10mm of capital (3.7×, inside the 4× cap),
and the book is invested on 96.9% of days.

**And the numbers that decide whether the above means anything:**

| Check | Value | Reading |
| --- | ---: | --- |
| Configurations searched | 64 | every one is in `artifacts/reports/parameter_study.csv` |
| **Deflated Sharpe ratio** | **0.871** | 87% probability the Sharpe survives multiple testing (Bailey & López de Prado) |
| Perfect-foresight Sharpe | 12.30 | the ceiling a total leak would reach |
| honest / perfect-foresight | **0.100** | a leaking pipeline scores near 1.0; this is clean |
| Annualised turnover | 444× | high — see [What I'd fix next](#what-id-fix-next) |
| Cost drag | 2.12% p.a. | charged in 32nds per cash-Treasury convention |
| Model IC (pooled, OOS) | +0.029 | positive on **all nine** tenors |

Performance is **not** uniform. 2022 lost 7.0% during the hiking cycle (the
benchmark lost 16.4%); 2024 and 2025 each gained over 9%. A strategy that only
works in some regimes is the normal case, and the calendar-year table is in
[`artifacts/backtests/latest/tearsheet.md`](artifacts/backtests/latest/tearsheet.md).

![Tearsheet](artifacts/backtests/latest/tearsheet.png)

---

## Why the honest numbers are the interesting ones

Most backtests are wrong in the same few ways. This project treats defending
against those as the actual engineering problem, and three of the four bugs
worth reporting were found by *running* the system rather than by reading it.

### 1. A DV01 cap is not a leverage cap

The sizing layer allocated risk equally in DV01 terms and enforced a $25,000
gross DV01 limit. It was also running **$148mm of notional on $10mm of capital**
— 14.9× leverage — entirely inside its risk limit.

A 3-month bill has a DV01 of ~$0.0025 per 100 face against ~$0.16 for the
30-year, a factor of 65. Equal *risk* therefore means 65× the *notional* at the
front end. Since transaction costs are charged on notional, the front-end leg was
cheap in risk and ruinous to trade. `dv01_scaled_positions` now enforces gross
notional against `max_leverage`, and says why in its docstring.

### 2. Turnover, not signal, was the binding constraint

The first honest backtest returned **gross Sharpe +1.26, net −0.56**. Turnover
was 2,365× capital per year and costs ate 11.4% p.a. The alpha was real; the
implementation gave all of it away.

`apply_no_trade_band` holds a position until the target drifts materially, then
trades all the way to it. Turnover fell to 444× and cost drag to 2.1%.

### 3. Z-scoring a return forecast destroys it

The default signal transform was a trailing z-score. Across the full
64-configuration sweep the result was unambiguous:

| Signal transform | Configs | Net Sharpe (median) | Positive |
| --- | ---: | ---: | ---: |
| `vol_scale` | 32 | **+0.54** | **32 / 32** |
| `zscore` | 32 | −1.08 | 0 / 32 |

This is structural, not fitted. The model forecasts a *return*, so zero means
"no move expected" — a meaningful point. Subtracting a trailing mean replaces
that with "unusually bullish relative to how bullish the model has been lately",
which is a different and actively harmful statement. Demean a forecast only when
you distrust its calibration more than you trust its level.

### 4. The look-ahead canary was testing the wrong thing

The original canary re-ran the backtest with the signal shifted one day forward
and expected a large positive Sharpe. It got −4.53, which looks reassuring and
proves nothing: `signal[t+1]` forecasts `return[t+1]`, so using it to trade day
*t* merely misaligns it and scores near zero whether or not the pipeline leaks.

The canary now trades the **realised future return** — perfect foresight, the
ceiling any leak could reach — and reports `honest / foresight`. At **0.100**,
the pipeline is clean. A leak would push that ratio toward 1.

---

## How look-ahead bias is prevented

The system assumes it is wrong and tries to catch itself:

- **One causality boundary.** Every feature block is written causal-as-of-its-own
  close, and exactly one line — `X.shift(feature_lag)` in
  [`features/builder.py`](src/tqe/features/builder.py) — moves the whole matrix
  to prediction time. Enforcing it in two places is how double-lagging happens.
- **Real publication lags.** CPI for January is stamped `1990-01-01` in FRED but
  released in mid-February. Forward-filling it onto a daily grid hands the model
  three weeks of foresight.
  [`features/macro.py`](src/tqe/features/macro.py) shifts each series by its
  actual release lag *before* it reaches the daily grid. NBER recession dates get
  a 400-day lag, because that is roughly when they are actually declared.
- **Purging and embargo.** Training rows whose forward-looking label overlaps the
  test block are removed; a further embargo is dropped after it. `validate_splits`
  audits the scheme before training starts and raises on violation — and the test
  suite feeds it a deliberately leaky split to confirm it catches one.
- **Causal PCA.** Curve factor loadings for day *t* are fitted only on data
  through *t−1*. The test corrupts the last third of the sample by 25× and asserts
  the earlier factor scores are bit-identical.
- **Per-fold scaling.** The feature scaler is refitted inside each fold on
  training data only. Standardising the full sample first leaks its mean and
  variance into every row.
- **Deflated Sharpe.** 64 configurations were searched, and the reported figure
  is adjusted for all 64.

---

## Architecture

```
src/tqe/
├── pricing/       bond maths — day counts, price/yield, duration, convexity, DV01, KRD
├── data/          Treasury.gov + FRED loaders, SIFMA calendar, total-return universe
├── curve/         Nelson-Siegel-Svensson, bootstrapping, PCA
├── features/      technical, macro (lag-aware), regime blocks → design matrix
├── models/        ridge/elastic-net/RF/GBM + baselines, stacked ensemble, registry
├── training/      walk-forward splits with purging, metrics, training harness
├── signals/       forecast → signal → DV01-sized position
├── portfolio/     mean-variance optimiser, covariance, VaR/ES, stress scenarios
├── backtest/      event-driven engine, 32nds cost model, tearsheet reporting
├── execution/     broker protocol, paper broker, risk gate, idempotent OMS
├── live/          daily trading loop
├── api/           FastAPI service
└── cli.py         tqe data | curve | features | train | backtest | predict | trade | serve
```

Full interface contracts: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### Pricing core

Implemented from the Treasury's own price/yield conventions and validated
against closed-form results, never against its own output:

| Property | Result |
| --- | --- |
| Zero-coupon Macaulay duration == maturity | exact to 1e-10 |
| Par bond prices at its own coupon | exactly 100.000000 |
| Price ↔ yield round trip | 9.1e-14 |
| Σ key-rate durations == effective duration | 8.077947 vs 8.077948 |
| Bootstrapped zeros reprice input par bonds | 1.4e-13 |

Accrued interest uses ACT/ACT (ICMA) within the actual coupon period; settlement
follows the real T+1 / T+2 split at 2024-05-28; the calendar computes Good Friday
via the Gregorian Easter algorithm and includes one-off closures like the 2012
Hurricane Sandy shutdown.

### From yields to P&L

The Treasury publishes *par yields*, not prices. To make a backtest economically
real, `constant_maturity_total_return` builds the synthetic on-the-run par bond
each day — which prices at exactly 100 by construction — then reprices **that
same bond** at the next day's yield, one day shorter, and adds the coupon
accrual.

The replication checks out against reality over 1990–2026:

| Tenor | Ann. return | Ann. vol | Mean DV01 (per 100) |
| --- | ---: | ---: | ---: |
| 3 Mo | 2.89% | 0.28% | 0.0025 |
| 2 Yr | 3.47% | 1.71% | 0.0192 |
| 10 Yr | 4.88% | 7.46% | 0.0813 |
| 30 Yr | 5.26% | 14.24% | 0.1648 |

Monotone in tenor for both return and risk, and the 10-year matches published
Treasury index history. Cross-checked against `−D·Δy + ½C·Δy²` to 0.025bp.

### Curve modelling — and a note on identifiability

Nelson-Siegel-Svensson fits the real curve to **1.0bp RMSE** across 14 tenors.
But NSS is **not identifiable**: many (β, τ) combinations are observationally
equivalent to well under a basis point. Fitting all six parameters freely gives
an excellent curve and useless parameters — measured over the full history, free
τ gives β₃ a standard deviation of **1.65** with 99th-percentile daily jumps of
2.25, as the optimiser hops between equivalent solutions.

Following Diebold & Li (2006), fixing the decays makes the model linear in β and
the factors uniquely identified:

| | Free τ | Fixed τ (1.37, 8.0) |
| --- | ---: | ---: |
| β₃ standard deviation | 1.6479 | **0.0308** (53× more stable) |
| β₃ 99th-pct daily change | 2.2534 | **0.0100** (226×) |
| Fit RMSE | 2.98bp | 7.62bp |
| Runtime, 9,172 days | 43.6s | **1.0s** |

The fixed-τ factors are also economically meaningful: `corr(β₀+β₁, 3-month
yield) = 0.998`. Free τ is available for pricing, where fit quality is what
matters; fixed τ feeds the model, where stability is.

PCA on daily yield changes gives **77.2% / 13.3% / 5.0%** for level / slope /
curvature. (The often-quoted 90/7/2 applies to a narrower 2y–30y set; including
the 3-month bill, which decouples when the Fed is pinned, moves variance out of
level and into slope.)

### Cost model

Quoted the way the cash market actually quotes: half-spreads in **32nds of a
point**, plus square-root market impact and commission.

```
$10mm on-the-run 10y   spread 0.5/32 = 1.56bp of price
                       spread cost   $1,562.50
                       impact           $3.31
                       commission     $125.00
                       total        $1,690.81  (1.69bp)
```

### Execution

- `PaperBroker` with correct position accounting through a sign flip — buy 100 @
  100, sell 60 @ 110, sell 60 @ 120 realises exactly $1,400 and leaves a short
  20 at an average of 120. (This is the classic accounting bug, so it is a test.)
- `RiskGate` enforcing hard limits — order and position notional, gross/net DV01,
  daily loss, drawdown, orders per day — plus a kill switch that stays tripped
  until explicitly reset. Every rejection hits the audit log.
- `OMS` that is **idempotent**: running the same day twice generates zero orders
  the second time, verified in the smoke test. It reconciles against broker state
  and persists to disk, so a crashed session can be safely re-run.
- Live trading is `dry_run=True` everywhere by default and additionally requires
  `--live --yes`. The HTTP API physically cannot place an order.

---

## Usage

```bash
tqe data pull                    # 36 years of curve + 16 FRED series (cached)
tqe curve fit                    # NSS betas, bootstrapped zeros, causal PCA factors
tqe curve fit --date 2026-08-28  # inspect a single day's fit
tqe features build               # 482 features × 6,946 rows
tqe train                        # walk-forward + deployable bundle
tqe backtest --n-trials 64       # costs, tearsheet, deflated Sharpe
tqe predict                      # next session's forecast per tenor
tqe trade --dry-run              # full live path, no orders sent
tqe serve --port 8000            # FastAPI
```

`make all` runs the whole pipeline. `python scripts/smoke_test.py` exercises
every seam on synthetic data in about a minute — 45 checks, no network.

---

## Testing

```bash
make test
```

Numerical code is tested against closed-form results and invariants, never
against values copied from its own output. The suite includes negative
controls — a deliberately leaky split that `validate_splits` must reject, and a
future-corruption test that the causal PCA must survive unchanged.

---

## What I'd fix next

Being specific about the limitations is more useful than hiding them:

1. **Turnover is still 444× capital per year.** It is survivable at 2.1% cost
   drag but it is the strategy's main fragility — a wider bid-ask than modelled
   would hurt disproportionately. The next step is optimising the no-trade band
   against a properly estimated impact function rather than a fixed threshold.
2. **The stacked ensemble shrinks hard.** Its NNLS weights sum to ~0.003, so the
   raw output carries almost no scale (`scale_to_return_units` exists to repair
   this for the optimiser). A calibrated-probability formulation would be cleaner
   than post-hoc rescaling.
3. **ETF proxies, not cash bonds.** Execution maps tenors to TLT/IEF/IEI/SHY
   because they are reachable from a retail broker. Real implementation would
   trade the on-the-run issues or futures, where the cost model already applies.
4. **Directional accuracy is 47.9% while IC is +0.029.** The model gets magnitude
   right more reliably than sign, which is why `vol_scale` sizing works and naive
   sign-following does not. Worth understanding rather than working around.
5. **Macro coverage forces a trade-off.** TIPS breakevens start in 2003 and the
   broad dollar index in 2006, so keeping them truncates training to 2007
   onwards. The default (`min_feature_coverage: 0.80`) drops them and trains from
   1993 instead; both are one config line apart.

---

## Data sources

- **US Treasury** — Daily Treasury Par Yield Curve Rates (CMT), 1990–present.
  Coverage is genuinely ragged and is handled rather than papered over: the 20y
  starts 1993-10, the 30y has a real publication gap from 2002-02 to 2006-02, and
  a 1.5-month tenor appeared on 2025-02-18.
- **FRED** — 16 macro and market series (Fed funds, CPI, unemployment,
  breakevens, credit spreads, VIX, dollar index, NBER recession dates).

Both are public and need no API key. `scripts/fetch_macro.py` handles FRED's
undocumented burst rate-limiting, which manifests as hung connections rather
than HTTP 429.

---

## Disclaimer

Research and educational software. Not investment advice. The backtest is a
simulation, and simulated results are not a promise about the future. Do not
trade real money with this without doing your own work on it first.

## Licence

MIT
