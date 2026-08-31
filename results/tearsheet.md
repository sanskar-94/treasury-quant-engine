# Treasury Quant Engine - Backtest

**Period** 2018-08-03 to 2026-08-27  
**Observations** 2,016 trading days  

## Headline

| Metric | Net | Gross |
| --- | ---: | ---: |
| Annualised return | 2.39% | 4.52% |
| Annualised volatility | 1.94% | - |
| Sharpe ratio | 1.23 | 2.35 |
| Sortino ratio | 1.89 | - |
| Calmar ratio | 0.23 | - |
| Maximum drawdown | -10.28% | - |
| Max DD duration | 1122 days | - |
| Hit rate | 56.14% | - |
| Profit factor | 1.28 | - |
| Skew / excess kurtosis | 0.50 / 10.60 | - |
| VaR 95 / CVaR 95 (daily) | -0.17% / -0.28% | - |

## Costs and turnover

| Metric | Value |
| --- | ---: |
| Annualised turnover | 444.26x |
| Total transaction costs | $1,698,622 |
| Cost drag | 2.12% p.a. |
| Average gross DV01 | $2,667 |
| Average net DV01 | $1,109 |
| Average gross notional | $37,194,483 |
| Days invested | 96.92% |

## Statistical honesty

These are the numbers that decide whether the headline means anything.

| Check | Value | Reading |
| --- | ---: | --- |
| Configurations searched | 64 | feeds the deflation below |
| Deflated Sharpe ratio | 0.8711 | probability the Sharpe survives multiple testing |
| Perfect-foresight Sharpe | 12.30 | honest/foresight = 0.100; clean - the honest run is a small fraction of perfect foresight |
| Benchmark Sharpe | 0.18 | buy-and-hold duration |
| Information ratio | 0.14 | active return per unit tracking error |
| Correlation to benchmark | 0.31 | |

## Calendar years

|   year |   return |    vol |   sharpe |   max_dd |   days |   benchmark |   excess |
|-------:|---------:|-------:|---------:|---------:|-------:|------------:|---------:|
|   2018 |  -0.0019 | 0.0055 |  -0.8612 |  -0.0082 |    101 |      0.0379 |  -0.0398 |
|   2019 |   0.071  | 0.0116 |   5.961  |  -0.0095 |    250 |      0.0929 |  -0.0219 |
|   2020 |   0.0071 | 0.0146 |   0.4919 |  -0.0225 |    251 |      0.1045 |  -0.0974 |
|   2021 |  -0.0007 | 0.0053 |  -0.1264 |  -0.0043 |    251 |     -0.041  |   0.0403 |
|   2022 |  -0.0703 | 0.0232 |  -3.1632 |  -0.0668 |    249 |     -0.1637 |   0.0935 |
|   2023 |   0.017  | 0.0279 |   0.6231 |  -0.0505 |    250 |      0.0378 |  -0.0208 |
|   2024 |   0.0942 | 0.0222 |   4.0963 |  -0.0153 |    250 |     -0.016  |   0.1102 |
|   2025 |   0.0916 | 0.0167 |   5.3253 |  -0.0048 |    249 |      0.0766 |   0.015  |
|   2026 |  -0.0047 | 0.0271 |  -0.2528 |  -0.0385 |    165 |     -0.0113 |   0.0066 |

## Monthly returns

|   year |      Jan |      Feb |      Mar |      Apr |      May |      Jun |      Jul |     Aug |      Sep |      Oct |      Nov |      Dec |    Year |
|-------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|--------:|---------:|---------:|---------:|---------:|--------:|
|   2018 | nan      | nan      | nan      | nan      | nan      | nan      | nan      |  0      |   0      |   0      |  -0.0071 |   0.0052 | -0.0019 |
|   2019 |   0.0102 |   0.0062 |   0.0115 |   0.0073 |   0.0112 |   0.0109 |   0.006  |  0.0101 |  -0.0027 |  -0.0053 |   0.0017 |   0.0018 |  0.071  |
|   2020 |   0.0067 |   0.0126 |  -0.0164 |   0.002  |  -0.0005 |  -0.0003 |   0.0013 | -0.0004 |   0.0005 |   0.0006 |   0.0004 |   0.0006 |  0.0071 |
|   2021 |   0.0008 |   0.0011 |   0.0006 |   0.0002 |  -0      |  -0.001  |   0.0001 | -0.0004 |   0.0003 |  -0.0008 |  -0.0013 |  -0.0003 | -0.0007 |
|   2022 |  -0.0159 |  -0.0008 |  -0.0117 |  -0.0051 |   0.0044 |  -0.0111 |   0.0046 | -0.0053 |  -0.0122 |  -0.0012 |  -0.0084 |  -0.0097 | -0.0703 |
|   2023 |   0.0164 |  -0.0134 |   0.0186 |   0.0002 |   0.0031 |  -0.0105 |  -0.0039 | -0.0152 |  -0.015  |   0.0026 |   0.0144 |   0.0206 |  0.017  |
|   2024 |   0.0136 |   0.0095 |   0.0065 |  -0.0072 |   0.0004 |   0.0099 |   0.0182 |  0.023  |   0.0189 |  -0.0039 |   0.004  |  -0.002  |  0.0942 |
|   2025 |   0.0051 |   0.0081 |   0.0058 |   0.0163 |  -0.0011 |   0.004  |   0.0057 |  0.0054 |   0.0077 |   0.0097 |   0.013  |   0.0082 |  0.0916 |
|   2026 |   0.0095 |   0.0076 |   0.0128 |  -0.0113 |  -0.008  |   0.0034 |  -0.0072 | -0.0112 | nan      | nan      | nan      | nan      | -0.0047 |

## Method

- Features are lagged one business day, so every prediction uses only
  information available at the prior close.
- Walk-forward evaluation with purging and an embargo between train and
  test blocks; the split scheme is audited before training starts.
- Transaction costs are charged in 32nds of a point per the cash Treasury
  convention, plus square-root market impact and commission.
- The position held through day *t* earns day *t*'s return and is set from
  the signal formed at *t-1*'s close.
