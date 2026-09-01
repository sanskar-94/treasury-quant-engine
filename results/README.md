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
| `horizon_experiment.csv` | 6 target x horizon cells (momentum features only), 40 block-sign-flip controls, Holm-corrected |
| `horizon_experiment_v2.csv` | the same six cells after adding rich/cheap and reversal features |
| `horizon_experiment_v3.csv` | horizon x target, re-run after the alignment fix |
| `carry_benchmark_v2.csv` | carry benchmark, re-run after the alignment fix |
| `turnover_study_v2.csv` | 54 turnover configs, re-run after the alignment fix |
| `carry_benchmark.csv` | carry vs model vs blend, each against 40 block-sign-flip controls |
| `structure_strategy.csv` | DV01-neutral steepeners and butterflies, funded, vs 40 controls |
| `charts/curve_surface.png` | the full curve 1990-2026 with inversions shaded |
| `charts/curve_fit.png` | NSS fit vs market for the latest date, with rich/cheap residuals |
| `charts/factors.png` | PCA loadings and cumulative level/slope/curvature paths |
| `charts/attribution.png` | cumulative P&L by curve factor, and share of gross risk |
| `charts/signal_diagnostics.png` | signal persistence, distribution and turnover |

**Read the two study files first** if you are checking whether the headline was
cherry-picked. Together they are the 136 configurations that feed the deflated
Sharpe ratio, and they contain every configuration tried rather than only the
winner.

Note that the headline result is **negative**: after financing is charged the
strategy has no economically significant edge. The top-level README explains how
that was established and why the pre-financing numbers were wrong.
