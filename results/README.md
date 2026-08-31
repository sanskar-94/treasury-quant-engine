# Published results

The committed output of the run documented in the top-level README. Everything
here is regenerable with `tqe pipeline`; it is checked in so the results can be
read without running anything.

| File | Contents |
| --- | --- |
| `summary.txt` | headline backtest metrics |
| `metrics.json` | every metric the engine computed, machine-readable |
| `tearsheet.md` | full tearsheet: headline, costs, honesty checks, calendar years, monthly grid |
| `tearsheet.png` | equity curve, drawdown, rolling Sharpe, DV01 exposure |
| `distribution.png` | daily return distribution and calendar-year bars |
| `walk_forward_folds.csv` | per-fold out-of-sample RMSE, IC, rank-IC and directional accuracy |
| `parameter_study.csv` | all 64 searched configurations - the input to the deflated Sharpe |

`parameter_study.csv` is the one to read first if you are checking whether the
headline Sharpe was cherry-picked. It contains every configuration tried, not
just the winner.
