# Treasury Quant Engine - Backtest

**Period** 2018-08-06 to 2026-08-28  
**Observations** 2,016 trading days  

## Headline

| Metric | Net | Gross |
| --- | ---: | ---: |
| Annualised return | 0.51% | 2.57% |
| Annualised volatility | 1.16% | - |
| Sharpe ratio | 0.44 | 2.09 |
| Sortino ratio | 0.64 | - |
| Calmar ratio | 0.18 | - |
| Maximum drawdown | -2.75% | - |
| Max DD duration | 870 days | - |
| Hit rate | 52.25% | - |
| Profit factor | 1.09 | - |
| Skew / excess kurtosis | 0.17 / 6.54 | - |
| VaR 95 / CVaR 95 (daily) | -0.12% / -0.17% | - |

## Costs and turnover

| Metric | Value |
| --- | ---: |
| Annualised turnover | 18.99x |
| Total transaction costs | $66,898 |
| Cost drag | 0.08% p.a. |
| Average gross DV01 | $2,154 |
| Average net DV01 | $512 |
| Average gross notional | $34,622,344 |
| Days invested | 95.98% |

## Statistical honesty

These are the numbers that decide whether the headline means anything.

| Check | Value | Reading |
| --- | ---: | --- |
| Configurations searched | 219 | feeds the deflation below |
| Deflated Sharpe (observed trial dispersion) | 0.0000 | using the measured sd of trial Sharpes (0.782) rather than the theoretical one |
| Deflated Sharpe ratio | 0.0601 | probability the Sharpe survives multiple testing |
| Perfect-foresight Sharpe | 15.32 | honest/foresight = 0.003; clean - the honest run is a small fraction of perfect foresight |
| Benchmark Sharpe | 0.17 | buy-and-hold duration |
| Information ratio | -0.10 | active return per unit tracking error |
| Correlation to benchmark | -0.15 | |

## Calendar years

|   year |   return |    vol |   sharpe |   max_dd |   days |   benchmark |   excess |
|-------:|---------:|-------:|---------:|---------:|-------:|------------:|---------:|
|   2018 |  -0.0033 | 0.0057 |  -1.462  |  -0.0058 |    100 |      0.0351 |  -0.0384 |
|   2019 |   0.005  | 0.0072 |   0.7052 |  -0.0023 |    250 |      0.0929 |  -0.0879 |
|   2020 |   0.0246 | 0.0097 |   2.5316 |  -0.0045 |    251 |      0.1045 |  -0.0799 |
|   2021 |   0.0002 | 0.0027 |   0.0826 |  -0.0016 |    251 |     -0.041  |   0.0412 |
|   2022 |   0.0259 | 0.0133 |   1.9529 |  -0.0071 |    249 |     -0.1637 |   0.1896 |
|   2023 |  -0.0056 | 0.0171 |  -0.321  |  -0.0142 |    250 |      0.0378 |  -0.0434 |
|   2024 |  -0.0032 | 0.0157 |  -0.1991 |  -0.0156 |    250 |     -0.016  |   0.0128 |
|   2025 |  -0.0014 | 0.0119 |  -0.1135 |  -0.0069 |    249 |      0.0766 |  -0.078  |
|   2026 |  -0.0012 | 0.0102 |  -0.1668 |  -0.0048 |    166 |     -0.0157 |   0.0145 |

## Monthly returns

|   year |      Jan |      Feb |      Mar |      Apr |      May |      Jun |      Jul |     Aug |      Sep |      Oct |      Nov |      Dec |    Year |
|-------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|--------:|---------:|---------:|---------:|---------:|--------:|
|   2018 | nan      | nan      | nan      | nan      | nan      | nan      | nan      |  0      |   0      |   0      |   0      |  -0.0033 | -0.0033 |
|   2019 |   0.0001 |  -0.0003 |   0.0002 |  -0.0009 |   0.0009 |   0.0022 |  -0.0014 |  0.0016 |  -0      |   0.0038 |  -0.0012 |  -0.0002 |  0.005  |
|   2020 |   0.0013 |   0.0088 |   0.0159 |  -0.0009 |  -0.001  |  -0      |   0.0016 | -0.0011 |   0.0002 |  -0.0004 |   0.0004 |  -0.0003 |  0.0246 |
|   2021 |  -0.0003 |   0.0003 |  -0.0003 |   0.0003 |  -0.0005 |   0.0001 |  -0.0005 | -0.0002 |  -0.0002 |   0.0004 |   0.0005 |   0.0005 |  0.0002 |
|   2022 |   0.0027 |   0.0007 |   0.0029 |   0.0034 |   0.0006 |   0.0084 |  -0.0013 |  0.0003 |   0.0049 |   0.0058 |  -0.0009 |  -0.0017 |  0.0259 |
|   2023 |   0.0011 |   0.0047 |  -0.006  |  -0.0004 |   0.0006 |   0.0029 |   0.0009 | -0.0017 |   0.0007 |   0.0009 |  -0.0055 |  -0.0038 | -0.0056 |
|   2024 |  -0.0019 |   0.0024 |  -0.002  |  -0.0085 |  -0.0017 |  -0.0009 |  -0.0006 |  0.004  |   0.0039 |  -0.0015 |   0.0004 |   0.0033 | -0.0032 |
|   2025 |  -0.0012 |  -0.0007 |  -0.0015 |  -0.0013 |  -0.0005 |  -0.0013 |  -0.0002 |  0.0029 |   0.0007 |   0.0005 |  -0.0003 |   0.0016 | -0.0014 |
|   2026 |   0      |  -0.0008 |  -0.0013 |   0.0001 |  -0.0006 |   0.0013 |  -0.0006 |  0.0007 | nan      | nan      | nan      | nan      | -0.0012 |

## Method

- Features are lagged one business day, so every prediction uses only
  information available at the prior close.
- Walk-forward evaluation with purging and an embargo between train and
  test blocks; the split scheme is audited before training starts.
- Transaction costs are charged in 32nds of a point per the cash Treasury
  convention, plus square-root market impact and commission.
- The position held through day *t* earns day *t*'s return and is set from
  the signal formed at *t-1*'s close.
