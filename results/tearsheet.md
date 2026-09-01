# Treasury Quant Engine - Backtest

**Period** 2018-08-06 to 2026-08-28  
**Observations** 2,016 trading days  

## Headline

| Metric | Net | Gross |
| --- | ---: | ---: |
| Annualised return | 0.56% | 2.44% |
| Annualised volatility | 1.09% | - |
| Sharpe ratio | 0.51 | 2.13 |
| Sortino ratio | 0.76 | - |
| Calmar ratio | 0.24 | - |
| Maximum drawdown | -2.27% | - |
| Max DD duration | 870 days | - |
| Hit rate | 51.21% | - |
| Profit factor | 1.11 | - |
| Skew / excess kurtosis | 0.26 / 7.30 | - |
| VaR 95 / CVaR 95 (daily) | -0.11% / -0.16% | - |

## Costs and turnover

| Metric | Value |
| --- | ---: |
| Annualised turnover | 18.82x |
| Total transaction costs | $66,007 |
| Cost drag | 0.08% p.a. |
| Average gross DV01 | $1,419 |
| Average net DV01 | $631 |
| Average gross notional | $32,062,173 |
| Days invested | 95.98% |

## Statistical honesty

These are the numbers that decide whether the headline means anything.

| Check | Value | Reading |
| --- | ---: | --- |
| Configurations searched | 145 | feeds the deflation below |
| Deflated Sharpe ratio | 0.1110 | probability the Sharpe survives multiple testing |
| Perfect-foresight Sharpe | 14.63 | honest/foresight = 0.006; clean - the honest run is a small fraction of perfect foresight |
| Benchmark Sharpe | 0.17 | buy-and-hold duration |
| Information ratio | -0.10 | active return per unit tracking error |
| Correlation to benchmark | -0.02 | |

## Calendar years

|   year |   return |    vol |   sharpe |   max_dd |   days |   benchmark |   excess |
|-------:|---------:|-------:|---------:|---------:|-------:|------------:|---------:|
|   2018 |  -0.0005 | 0.0043 |  -0.311  |  -0.0031 |    100 |      0.0351 |  -0.0357 |
|   2019 |   0.004  | 0.0063 |   0.6402 |  -0.0023 |    250 |      0.0929 |  -0.0889 |
|   2020 |   0.0318 | 0.0107 |   2.9496 |  -0.0035 |    251 |      0.1045 |  -0.0727 |
|   2021 |  -0.0008 | 0.0018 |  -0.4545 |  -0.0026 |    251 |     -0.041  |   0.0402 |
|   2022 |   0.0203 | 0.011  |   1.8555 |  -0.0063 |    249 |     -0.1637 |   0.184  |
|   2023 |   0.0014 | 0.0143 |   0.1071 |  -0.0096 |    250 |      0.0378 |  -0.0364 |
|   2024 |  -0.0085 | 0.0154 |  -0.5497 |  -0.0163 |    250 |     -0.016  |   0.0075 |
|   2025 |   0.0001 | 0.0128 |   0.0163 |  -0.0063 |    249 |      0.0766 |  -0.0765 |
|   2026 |  -0.003  | 0.0101 |  -0.4481 |  -0.006  |    166 |     -0.0157 |   0.0126 |

## Monthly returns

|   year |      Jan |      Feb |      Mar |      Apr |      May |      Jun |      Jul |     Aug |      Sep |      Oct |      Nov |      Dec |    Year |
|-------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|--------:|---------:|---------:|---------:|---------:|--------:|
|   2018 | nan      | nan      | nan      | nan      | nan      | nan      | nan      |  0      |   0      |   0      |   0      |  -0.0005 | -0.0005 |
|   2019 |   0.0003 |  -0.0005 |   0.0003 |  -0.0008 |   0.001  |   0.0017 |  -0.0004 |  0.0011 |  -0.0001 |   0.0028 |  -0.0014 |   0.0001 |  0.004  |
|   2020 |   0.0018 |   0.0091 |   0.0208 |  -0.0006 |  -0.0004 |  -0.0002 |   0.0012 | -0.0004 |   0.0001 |  -0.0002 |   0.0002 |  -0      |  0.0318 |
|   2021 |   0.0001 |   0.0003 |   0      |   0.0003 |  -0.0001 |  -0.0008 |  -0.0001 | -0.0001 |  -0.0003 |  -0.0008 |   0.0003 |   0.0003 | -0.0008 |
|   2022 |   0.0016 |   0.0003 |   0.0003 |   0.0031 |   0.0014 |   0.0075 |  -0.0006 | -0.0016 |   0.0031 |   0.0067 |   0.0003 |  -0.0021 |  0.0203 |
|   2023 |   0.0026 |   0.0032 |  -0.0039 |   0.0002 |  -0.0005 |   0.001  |   0.0019 | -0.0008 |   0.0012 |   0.0019 |  -0.0034 |  -0.0018 |  0.0014 |
|   2024 |  -0.0025 |   0.0007 |  -0.0025 |  -0.0103 |  -0.0003 |   0.0002 |  -0.0008 |  0.0021 |   0.0025 |  -0.0008 |   0.0005 |   0.0029 | -0.0085 |
|   2025 |  -0.0009 |   0.0003 |  -0.0012 |  -0.0001 |  -0.0014 |  -0.0007 |  -0.001  |  0.0037 |  -0      |   0.0003 |  -0.0002 |   0.0013 |  0.0001 |
|   2026 |  -0.0004 |  -0.0004 |  -0.003  |  -0.0003 |  -0.0007 |   0.0012 |  -0.0003 |  0.001  | nan      | nan      | nan      | nan      | -0.003  |

## Method

- Features are lagged one business day, so every prediction uses only
  information available at the prior close.
- Walk-forward evaluation with purging and an embargo between train and
  test blocks; the split scheme is audited before training starts.
- Transaction costs are charged in 32nds of a point per the cash Treasury
  convention, plus square-root market impact and commission.
- The position held through day *t* earns day *t*'s return and is set from
  the signal formed at *t-1*'s close.
