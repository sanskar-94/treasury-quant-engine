# Published results

The committed output of the run documented in the top-level README, produced
with financing charged. Everything here is regenerable with `tqe pipeline`; it
is checked in so the results can be read without running anything.

| File | Contents |
| --- | --- |
| `summary.txt` | headline backtest metrics |
| `metrics.json` | every metric the engine computed, machine-readable |
| `tearsheet.md` | full tearsheet: headline, costs, honesty checks, calendar years, monthly grid |
| `tearsheet.png` | equity curve, drawdown, rolling Sharpe, DV01 exposure |
| `distribution.png` | daily return distribution and calendar-year bars |
| `walk_forward_folds.csv` | per-fold out-of-sample RMSE, IC, rank-IC and directional accuracy |
| `parameter_study.csv` | 64 signal-transform configurations |
| `turnover_study.csv` | 72 turnover-control configurations, scored at 1x and 2x assumed costs |

**Read the two study files first** if you are checking whether the headline was
cherry-picked. Together they are the 136 configurations that feed the deflated
Sharpe ratio, and they contain every configuration tried rather than only the
winner.

Note that the headline result is **negative**: after financing is charged the
strategy has no economically significant edge. The top-level README explains how
that was established and why the pre-financing numbers were wrong.
