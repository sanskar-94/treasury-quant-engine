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
| `duration_timing.csv` | can the term premium time duration? static vs timed, funded |
| `duration_harvest.csv` | where on the curve duration is best paid: 9 tenors x constant-DV01 and vol-targeted, plus two diversified arms, all funded and costed |
| `integration_experiment.csv` | term premium and regime conditioning vs baseline, 40 controls each |
| `structure_strategy_cashneutral.csv` | structures funded against a bill leg (net notional zero) |
| `charts/curve_surface.png` | the full curve 1990-2026 with inversions shaded |
| `charts/curve_fit.png` | NSS fit vs market for the latest date, with rich/cheap residuals |
| `charts/factors.png` | PCA loadings and cumulative level/slope/curvature paths |
| `charts/attribution.png` | cumulative P&L by curve factor, and share of gross risk |
| `charts/signal_diagnostics.png` | signal persistence, distribution and turnover |

**Read the study files first** if you are checking whether the headline was
cherry-picked. Every row of every study file here is one configuration that was
evaluated and compared: **219** in total (64 parameter, 72 + 54 turnover, 18
horizon, 5 integration, 6 carry). They contain every configuration tried rather
than only the winner, and `backtest/trials.py` counts them automatically so the
deflated Sharpe cannot be computed against a smaller number by accident.

That mattered: the tearsheet once reported a deflated Sharpe of 0.9043 while
declaring "configurations searched: 1". Against the real 219 it is **0.100**, or
**0.000** using the observed dispersion of the trial Sharpes (0.782) rather than
the theoretical i.i.d. one (0.354).

Note that the headline result is **negative**: after financing is charged the
strategy has no economically significant edge. The top-level README explains how
that was established and why the pre-financing numbers were wrong.
