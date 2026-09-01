# Treasury Quant Engine - Backtest

**Period** 2018-08-03 to 2026-08-27  
**Observations** 2,016 trading days  

## Headline

| Metric | Net | Gross |
| --- | ---: | ---: |
| Annualised return | 0.13% | 2.78% |
| Annualised volatility | 1.12% | - |
| Sharpe ratio | 0.12 | 2.43 |
| Sortino ratio | 0.18 | - |
| Calmar ratio | 0.04 | - |
| Maximum drawdown | -3.56% | - |
| Max DD duration | 1605 days | - |
| Hit rate | 48.62% | - |
| Profit factor | 1.03 | - |
| Skew / excess kurtosis | 0.58 / 10.14 | - |
| VaR 95 / CVaR 95 (daily) | -0.11% / -0.16% | - |

## Costs and turnover

| Metric | Value |
| --- | ---: |
| Annualised turnover | 20.93x |
| Total transaction costs | $73,418 |
| Cost drag | 0.09% p.a. |
| Average gross DV01 | $1,473 |
| Average net DV01 | $922 |
| Average gross notional | $32,253,538 |
| Days invested | 94.94% |

## Statistical honesty

These are the numbers that decide whether the headline means anything.

| Check | Value | Reading |
| --- | ---: | --- |
| Configurations searched | 142 | feeds the deflation below |
| Deflated Sharpe ratio | 0.0102 | probability the Sharpe survives multiple testing |
| Perfect-foresight Sharpe | 14.65 | honest/foresight = 0.009; clean - the honest run is a small fraction of perfect foresight |
| Benchmark Sharpe | 0.18 | buy-and-hold duration |
| Information ratio | -0.17 | active return per unit tracking error |
| Correlation to benchmark | 0.26 | |

## Calendar years

|   year |   return |    vol |   sharpe |   max_dd |   days |   benchmark |   excess |
|-------:|---------:|-------:|---------:|---------:|-------:|------------:|---------:|
|   2018 |  -0.001  | 0.0051 |  -0.4783 |  -0.0039 |    101 |      0.0379 |  -0.0389 |
|   2019 |   0.0085 | 0.0078 |   1.1086 |  -0.0027 |    250 |      0.0929 |  -0.0843 |
|   2020 |   0.0327 | 0.0111 |   2.9215 |  -0.0042 |    251 |      0.1045 |  -0.0718 |
|   2021 |  -0.0021 | 0.0017 |  -1.2494 |  -0.003  |    251 |     -0.041  |   0.039  |
|   2022 |  -0.0061 | 0.0106 |  -0.5775 |  -0.0149 |    249 |     -0.1637 |   0.1577 |
|   2023 |  -0.0074 | 0.0161 |  -0.455  |  -0.0165 |    250 |      0.0378 |  -0.0451 |
|   2024 |  -0.0087 | 0.017  |  -0.5125 |  -0.0178 |    250 |     -0.016  |   0.0072 |
|   2025 |  -0.0037 | 0.0099 |  -0.3761 |  -0.0082 |    249 |      0.0766 |  -0.0803 |
|   2026 |  -0.0015 | 0.0096 |  -0.2286 |  -0.0046 |    165 |     -0.0113 |   0.0099 |

## Monthly returns

|   year |      Jan |      Feb |      Mar |      Apr |      May |      Jun |      Jul |     Aug |      Sep |      Oct |      Nov |      Dec |    Year |
|-------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|--------:|---------:|---------:|---------:|---------:|--------:|
|   2018 | nan      | nan      | nan      | nan      | nan      | nan      | nan      |  0      |   0      |   0      |   0      |  -0.001  | -0.001  |
|   2019 |   0.0008 |  -0.0005 |   0.0008 |  -0.001  |   0.0018 |   0.0031 |  -0.001  |  0.0022 |  -0.0001 |   0.0038 |  -0.0014 |   0      |  0.0085 |
|   2020 |   0.002  |   0.0097 |   0.0215 |  -0.0006 |  -0.0011 |  -0.0001 |   0.0014 | -0.0005 |   0.0001 |  -0.0001 |   0.0003 |  -0      |  0.0327 |
|   2021 |   0.0001 |   0.0003 |   0.0001 |   0.0002 |  -0.0001 |  -0.0007 |  -0.0001 | -0.0001 |  -0.0003 |  -0.0009 |  -0.0004 |  -0.0002 | -0.0021 |
|   2022 |  -0.0002 |  -0.0004 |  -0.0016 |   0.0003 |   0.0004 |  -0.0052 |  -0.0036 | -0.0035 |   0.0026 |   0.0072 |   0.0004 |  -0.0023 | -0.0061 |
|   2023 |   0.0029 |  -0.0052 |   0.0028 |  -0.0018 |  -0.0074 |  -0.0014 |   0.0001 | -0.0005 |   0.0006 |   0.0017 |  -0.0024 |   0.0032 | -0.0074 |
|   2024 |  -0.0028 |  -0.0017 |  -0.002  |  -0.0103 |  -0.0002 |   0.0005 |   0.0029 |  0.0049 |   0.0052 |  -0.0047 |  -0.0001 |  -0.0004 | -0.0087 |
|   2025 |  -0.0001 |   0.0002 |  -0.0004 |   0.0003 |  -0.0012 |   0.0003 |  -0.0013 | -0.0028 |  -0.0003 |   0.0003 |  -0.0002 |   0.0014 | -0.0037 |
|   2026 |  -0.0004 |  -0.0004 |  -0.0025 |  -0.0004 |  -0.0004 |   0.0023 |   0.0002 |  0.0001 | nan      | nan      | nan      | nan      | -0.0015 |

## Method

- Features are lagged one business day, so every prediction uses only
  information available at the prior close.
- Walk-forward evaluation with purging and an embargo between train and
  test blocks; the split scheme is audited before training starts.
- Transaction costs are charged in 32nds of a point per the cash Treasury
  convention, plus square-root market impact and commission.
- The position held through day *t* earns day *t*'s return and is set from
  the signal formed at *t-1*'s close.
