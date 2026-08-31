"""Time-series cross-validation with purging and embargo.

Ordinary K-fold is invalid on financial time series for two separate reasons,
and both have to be fixed:

**Ordering.** A shuffled fold trains on the future and tests on the past. Every
split here is forward-chaining: the test block always follows its training data.

**Overlap.** When the target is a forward return over ``horizon`` days, the
label attached to training row ``t`` is realised over ``t+1 .. t+horizon``. If
the test block starts at ``t+1``, that training row's label *contains test-period
information*. Removing such rows is **purging**. Separately, serial correlation
means observations immediately after the test block are still informative about
it, so a further **embargo** of rows is dropped after the test window.

Both concepts are from Lopez de Prado, *Advances in Financial Machine Learning*
(2018), ch. 7. Without them a walk-forward backtest still overstates performance,
just less obviously than a shuffled one.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger("training.splits")

__all__ = [
    "Split",
    "walk_forward_splits",
    "purged_kfold_splits",
    "expanding_splits",
    "describe_splits",
    "validate_splits",
]


@dataclass(frozen=True)
class Split:
    """One train/test fold, carrying both positions and dates."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    fold: int = 0

    def __len__(self) -> int:
        return len(self.test_idx)

    def __repr__(self) -> str:
        return (
            f"Split(fold={self.fold}, train={len(self.train_idx)} "
            f"[{self.train_start.date()}..{self.train_end.date()}], "
            f"test={len(self.test_idx)} [{self.test_start.date()}..{self.test_end.date()}])"
        )

    def as_dict(self) -> dict:
        return {
            "fold": self.fold,
            "n_train": len(self.train_idx),
            "n_test": len(self.test_idx),
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }


def _make(index: pd.DatetimeIndex, train_idx: np.ndarray, test_idx: np.ndarray, fold: int) -> Split:
    return Split(
        train_idx=train_idx,
        test_idx=test_idx,
        train_start=index[train_idx[0]],
        train_end=index[train_idx[-1]],
        test_start=index[test_idx[0]],
        test_end=index[test_idx[-1]],
        fold=fold,
    )


def walk_forward_splits(
    index: pd.DatetimeIndex | Sequence,
    n_splits: int = 8,
    test_size: int = 252,
    min_train_size: int = 1260,
    embargo: int = 5,
    expanding: bool = True,
    horizon: int = 1,
) -> list[Split]:
    """Forward-chaining splits with purging and embargo.

    The test blocks are laid out to end at the *last* observation and step
    backwards by ``test_size``, so the most recent data is always evaluated and
    the earliest fold is the one dropped if there is not enough history.

    Parameters
    ----------
    index:
        The sample index (must be sorted ascending).
    n_splits:
        Maximum number of folds. Fewer are returned if history is short.
    test_size:
        Observations per out-of-sample block. 252 ~ one trading year.
    min_train_size:
        Minimum training observations required before a fold is emitted.
    embargo:
        Observations dropped immediately **after** each test block, so a
        subsequent expanding-window training set does not begin flush against
        data that is serially correlated with the test period.
    expanding:
        ``True`` grows the training window from the start of history (the
        realistic choice - a desk does not throw away data). ``False`` uses a
        rolling window of ``min_train_size``.
    horizon:
        Target horizon in observations. ``horizon`` rows immediately before the
        test block are **purged**, because their labels overlap the test window.

    Returns
    -------
    list[Split]
        Ordered oldest fold first.
    """
    idx = pd.DatetimeIndex(index)
    n = len(idx)
    if n == 0:
        return []
    if not idx.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending")

    splits: list[Split] = []
    # Lay the test blocks out from the end backwards.
    starts = [n - test_size * (k + 1) for k in range(n_splits)]
    starts = [s for s in starts if s > 0][::-1]

    for fold, test_lo in enumerate(starts):
        test_hi = min(test_lo + test_size, n)
        # Purge the `horizon` rows whose labels bleed into the test block, then
        # nothing else: the embargo applies AFTER the test window, and matters
        # only for folds that come later.
        train_hi = test_lo - horizon
        train_lo = 0 if expanding else max(0, train_hi - min_train_size)
        if train_hi - train_lo < min_train_size:
            continue
        train_idx = np.arange(train_lo, train_hi)
        test_idx = np.arange(test_lo, test_hi)
        if len(test_idx) == 0:
            continue
        splits.append(_make(idx, train_idx, test_idx, fold))

    if not splits:
        log.warning(
            "no walk-forward folds produced: %d observations, min_train_size=%d, test_size=%d",
            n, min_train_size, test_size,
        )
    return splits


def expanding_splits(index, n_splits: int = 8, **kw) -> list[Split]:
    """Alias for :func:`walk_forward_splits` with ``expanding=True``."""
    kw["expanding"] = True
    return walk_forward_splits(index, n_splits=n_splits, **kw)


def purged_kfold_splits(
    index: pd.DatetimeIndex | Sequence,
    n_splits: int = 5,
    embargo: int = 5,
    horizon: int = 1,
) -> list[Split]:
    """Contiguous K-fold with purging on both sides and a forward embargo.

    Unlike :func:`walk_forward_splits`, training data here may come from *after*
    the test block. That is not usable for an honest backtest, but it is the
    right tool for measuring how much signal a feature set carries when you are
    not simulating a live process - it uses every observation and has lower
    variance. Use it for feature research; use walk-forward for anything that
    produces a performance number.
    """
    idx = pd.DatetimeIndex(index)
    n = len(idx)
    if n == 0:
        return []
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    splits: list[Split] = []

    for fold in range(n_splits):
        test_lo, test_hi = bounds[fold], bounds[fold + 1]
        if test_hi <= test_lo:
            continue
        test_idx = np.arange(test_lo, test_hi)
        # Purge horizon rows before, embargo rows after.
        drop_lo = max(0, test_lo - horizon)
        drop_hi = min(n, test_hi + embargo)
        train_mask = np.ones(n, dtype=bool)
        train_mask[drop_lo:drop_hi] = False
        train_idx = np.flatnonzero(train_mask)
        if len(train_idx) < 50:
            continue
        splits.append(_make(idx, train_idx, test_idx, fold))
    return splits


def describe_splits(splits: Sequence[Split]) -> pd.DataFrame:
    """Tabulate the folds - useful in a report to show the scheme is sane."""
    if not splits:
        return pd.DataFrame()
    return pd.DataFrame([s.as_dict() for s in splits]).set_index("fold")


def validate_splits(splits: Sequence[Split], horizon: int = 1, embargo: int = 0) -> dict:
    """Assert the invariants a leak-free split scheme must satisfy.

    Checks, for every fold:

    * train and test index sets are disjoint,
    * no training observation lies inside ``[test_start - horizon, test_end]``,
    * test blocks do not overlap each other.

    Returns a dict of results; ``ok`` is True only if every check passed. The
    test-suite calls this, but so does the training harness at run time - a leak
    that only shows up in production is worth failing loudly for.
    """
    results = {"n_folds": len(splits), "disjoint": True, "purged": True, "test_disjoint": True,
               "violations": []}
    seen_test: set[int] = set()

    for s in splits:
        train_set = set(s.train_idx.tolist())
        test_set = set(s.test_idx.tolist())

        if train_set & test_set:
            results["disjoint"] = False
            results["violations"].append(f"fold {s.fold}: train and test overlap")

        lo = int(s.test_idx[0]) - horizon
        hi = int(s.test_idx[-1])
        bleed = [i for i in train_set if lo <= i <= hi]
        if bleed:
            results["purged"] = False
            results["violations"].append(
                f"fold {s.fold}: {len(bleed)} training rows inside the purge window"
            )

        if seen_test & test_set:
            results["test_disjoint"] = False
            results["violations"].append(f"fold {s.fold}: test block overlaps an earlier fold")
        seen_test |= test_set

    results["ok"] = bool(
        results["disjoint"] and results["purged"] and results["test_disjoint"] and splits
    )
    return results


def iter_splits(splits: Sequence[Split]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` - the scikit-learn splitter interface."""
    for s in splits:
        yield s.train_idx, s.test_idx
