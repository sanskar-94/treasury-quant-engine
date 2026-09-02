"""Counting how many configurations were actually searched.

The deflated Sharpe ratio answers "is this Sharpe just the best of N lucky
draws?", and it is only as honest as the N fed into it. ``deflated_sharpe_ratio``
defaults ``n_trials`` to 1, which deflates by nothing at all: the deflated Sharpe
then equals the probabilistic Sharpe and the label on it is a lie.

That is exactly what happened here. The shipped tearsheet reported a deflated
Sharpe of 0.9043 under the heading "probability the Sharpe survives multiple
testing", computed with ``n_trials=1``, while this project had in fact searched
219 configurations across its parameter, turnover, horizon and integration
studies. At the true count the number is 0.067 - and using the observed
dispersion of the trial Sharpes rather than the theoretical i.i.d. one, it
rounds to zero.

The honest count is not something to remember to pass on the command line. It is
recoverable from the study files this project already writes, so it is recovered
here and used by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger("backtest.trials")

__all__ = ["TrialCensus", "count_configurations_searched"]

#: Files whose every row is one configuration that was evaluated and compared.
#: A row here is a draw from the search, whether or not it was kept.
STUDY_GLOBS = ("*study*.csv", "*experiment*.csv", "*benchmark*.csv")

#: Column names under which a study stores the Sharpe of each configuration.
SHARPE_COLUMNS = ("sharpe", "sharpe_ratio", "ann_sharpe", "net_sharpe")


@dataclass(frozen=True)
class TrialCensus:
    """How many configurations were searched, and how much they varied."""

    n_trials: int
    sharpe_std: float | None       # observed cross-sectional sd of trial Sharpes
    sources: dict[str, int]

    def describe(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(self.sources.items()))
        sd = f"{self.sharpe_std:.4f}" if self.sharpe_std is not None else "n/a"
        return f"{self.n_trials} configurations (observed Sharpe sd {sd}) from {parts}"


def count_configurations_searched(results_dir: str | Path) -> TrialCensus:
    """Count evaluated configurations from the study files in ``results_dir``.

    Every row of every study file is one configuration that was run and compared
    against the others, which is precisely the quantity the deflated Sharpe
    needs. Where a study also records the Sharpe of each configuration, their
    cross-sectional standard deviation is collected too: Bailey & Lopez de Prado
    note that passing the *observed* dispersion is strictly better than assuming
    the i.i.d. theoretical one, and here it is markedly wider (0.95 against
    0.35), which makes the honest number smaller still.

    Returns a census with ``n_trials >= 1`` so the result is always safe to pass
    on. A directory with no studies yields ``n_trials=1`` and a warning, because
    that means the count could not be established rather than that only one
    configuration was tried.
    """
    root = Path(results_dir)
    sources: dict[str, int] = {}
    sharpes: list[float] = []

    if root.is_dir():
        seen: set[Path] = set()
        for pattern in STUDY_GLOBS:
            for path in sorted(root.glob(pattern)):
                if path in seen:
                    continue
                seen.add(path)
                try:
                    frame = pd.read_csv(path)
                except Exception as exc:  # noqa: BLE001 - a bad file must not stop a run
                    log.warning("could not read study %s (%s)", path.name, exc)
                    continue
                if frame.empty:
                    continue
                sources[path.name] = len(frame)
                for col in SHARPE_COLUMNS:
                    if col in frame.columns:
                        vals = pd.to_numeric(frame[col], errors="coerce").dropna()
                        sharpes.extend(vals.tolist())
                        break

    total = sum(sources.values())
    if total == 0:
        log.warning(
            "no study files found in %s; deflating by 1 trial, which deflates by "
            "nothing. Any reported deflated Sharpe is really a probabilistic Sharpe.",
            root,
        )
        return TrialCensus(1, None, {})

    sd = float(np.std(sharpes, ddof=1)) if len(sharpes) > 2 else None
    census = TrialCensus(max(total, 1), sd, sources)
    log.info("trial census: %s", census.describe())
    return census
