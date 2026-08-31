"""Optional PyTorch sequence model.

An LSTM is the obvious thing to reach for on a time series, and it is included
so the comparison can be made honestly rather than asserted. The expectation is
that it will **not** beat ridge here, and the reason is worth stating plainly:
the target has an information coefficient of roughly 0.03, so the signal is a
rounding error next to the noise, and a model with tens of thousands of
parameters will spend nearly all of its capacity memorising the noise. Sequence
models earn their keep when there is genuine non-linear temporal structure and
enough signal to identify it; daily Treasury returns have neither in abundance.

The module is written so that the rest of the system does not depend on it:

* ``torch`` is imported lazily inside methods, so ``import tqe.models.lstm``
  succeeds on a machine without PyTorch and the registry can advertise the
  learner without requiring it,
* it is excluded from the default ``learners`` list and only joins the ensemble
  when ``model.use_torch_lstm`` is set,
* everything is seeded, and cuDNN is put in deterministic mode, so a reported
  result can be reproduced.

Install with ``pip install 'tqe[deep]'``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .base import BaseModel

log = get_logger("models.lstm")

__all__ = ["LSTMModel", "torch_available"]


def torch_available() -> bool:
    """Whether PyTorch can be imported in this environment."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


class LSTMModel(BaseModel):
    """Stacked LSTM over a trailing window of features.

    The design matrix arrives as ``(n_samples, n_features)`` with one row per
    day. This wraps it into overlapping sequences of length ``seq_len`` so the
    network sees a window of history: sample ``t`` becomes
    ``X[t-seq_len+1 : t+1]`` and predicts ``y[t]``.

    Crucially the windows are built **backwards from t**, never forwards, so a
    sequence for day ``t`` contains nothing after ``t``. The first
    ``seq_len - 1`` rows have insufficient history and are padded by repeating
    the earliest available row, which is the only choice that does not either
    drop data or leak.

    Parameters
    ----------
    hidden: hidden units per layer.
    layers: stacked LSTM layers.
    seq_len: lookback window in days.
    epochs / batch_size / learning_rate / dropout / weight_decay: training knobs.
    patience: early-stopping patience on the validation tail.
    validation_fraction: final fraction of the training block held out. Taken
        from the end, not sampled at random, so validation is chronologically
        after training.
    device: ``"auto"`` picks CUDA, then Apple MPS, then CPU.
    """

    name = "lstm"

    def __init__(
        self,
        hidden: int = 64,
        layers: int = 2,
        seq_len: int = 30,
        epochs: int = 40,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        dropout: float = 0.2,
        weight_decay: float = 1e-4,
        patience: int = 8,
        validation_fraction: float = 0.15,
        device: str = "auto",
        random_state: int = 42,
        **kw: Any,
    ) -> None:
        super().__init__(
            hidden=hidden, layers=layers, seq_len=seq_len, epochs=epochs,
            batch_size=batch_size, learning_rate=learning_rate, dropout=dropout,
            weight_decay=weight_decay, patience=patience,
            validation_fraction=validation_fraction, device=device,
            random_state=random_state, **kw,
        )
        self.net_ = None
        self.history_: list[dict[str, float]] = []
        self._tail: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def _resolve_device(self):
        import torch

        want = str(self.params["device"])
        if want != "auto":
            return torch.device(want)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _windows(X: np.ndarray, seq_len: int) -> np.ndarray:
        """Build ``(n, seq_len, n_features)`` causal windows ending at each row.

        Uses a strided view rather than materialising every window, which keeps
        memory to a single copy of the padded array instead of ``seq_len`` copies.
        """
        n, f = X.shape
        pad = np.repeat(X[:1], seq_len - 1, axis=0) if seq_len > 1 else np.empty((0, f))
        padded = np.concatenate([pad, X], axis=0)
        strides = (padded.strides[0], padded.strides[0], padded.strides[1])
        return np.lib.stride_tricks.as_strided(
            padded, shape=(n, seq_len, f), strides=strides, writeable=False
        )

    def _build(self, n_features: int, n_targets: int):
        import torch
        from torch import nn

        p = self.params

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=int(p["hidden"]),
                    num_layers=int(p["layers"]),
                    batch_first=True,
                    dropout=float(p["dropout"]) if int(p["layers"]) > 1 else 0.0,
                )
                self.drop = nn.Dropout(float(p["dropout"]))
                self.head = nn.Linear(int(p["hidden"]), n_targets)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out, _ = self.lstm(x)
                return self.head(self.drop(out[:, -1, :]))  # last timestep only

        return Net()

    # ------------------------------------------------------------------ #
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LSTMModel needs PyTorch. Install with: pip install 'tqe[deep]'"
            ) from exc

        p = self.params
        seed = int(p["random_state"])
        torch.manual_seed(seed)
        # PyTorch's DataLoader and several ops consult the legacy global NumPy
        # RNG, so it has to be seeded too - a Generator instance would not reach
        # them. This is the one place in the codebase that uses it.
        np.random.seed(seed)  # noqa: NPY002
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        device = self._resolve_device()
        seq_len = int(p["seq_len"])
        n_targets = y.shape[1]

        seqs = np.ascontiguousarray(self._windows(X, seq_len), dtype=np.float32)
        targets = y.astype(np.float32)

        # Chronological split - the validation block follows the training block.
        n_val = max(1, int(len(seqs) * float(p["validation_fraction"])))
        n_train = max(1, len(seqs) - n_val)
        Xtr = torch.from_numpy(seqs[:n_train]).to(device)
        ytr = torch.from_numpy(targets[:n_train]).to(device)
        Xva = torch.from_numpy(seqs[n_train:]).to(device)
        yva = torch.from_numpy(targets[n_train:]).to(device)

        self.net_ = self._build(X.shape[1], n_targets).to(device)
        opt = torch.optim.AdamW(
            self.net_.parameters(),
            lr=float(p["learning_rate"]),
            weight_decay=float(p["weight_decay"]),
        )
        loss_fn = nn.MSELoss()
        batch = int(p["batch_size"])
        patience = int(p["patience"])

        best_loss = float("inf")
        best_state = None
        stale = 0
        self.history_ = []

        for epoch in range(int(p["epochs"])):
            self.net_.train()
            # Shuffle only the *order of batches*, not the contents of a window.
            perm = torch.randperm(len(Xtr), device=device)
            total = 0.0
            for i in range(0, len(perm), batch):
                idx = perm[i : i + batch]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.net_(Xtr[idx]), ytr[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
                total += float(loss.item()) * len(idx)

            self.net_.eval()
            with torch.no_grad():
                val = float(loss_fn(self.net_(Xva), yva).item()) if len(Xva) else float("nan")
            self.history_.append({"epoch": epoch, "train": total / max(len(Xtr), 1), "val": val})

            if np.isfinite(val) and val < best_loss - 1e-9:
                best_loss, stale = val, 0
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    log.info("early stopping at epoch %d (best val %.6e)", epoch, best_loss)
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.net_.eval()
        self._device = device
        # Keep the tail so predict() can build windows for rows that arrive later.
        self._tail = X[-(seq_len - 1):].copy() if seq_len > 1 else np.empty((0, X.shape[1]))
        self.n_targets_ = n_targets

    def _predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.net_ is None:
            raise RuntimeError("LSTMModel is not fitted")
        seq_len = int(self.params["seq_len"])

        # Prepend the training tail so early prediction rows have real history
        # rather than padding - this is history from the past, never the future.
        if self._tail is not None and len(self._tail):
            joined = np.concatenate([self._tail, X], axis=0)
            seqs = self._windows(joined, seq_len)[len(self._tail):]
        else:
            seqs = self._windows(X, seq_len)

        seqs = np.ascontiguousarray(seqs, dtype=np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(seqs), 1024):
                chunk = torch.from_numpy(seqs[i : i + 1024]).to(self._device)
                out.append(self.net_(chunk).cpu().numpy())
        return np.vstack(out) if out else np.zeros((len(X), self.n_targets_))

    # ------------------------------------------------------------------ #
    @property
    def feature_importance(self) -> pd.Series | None:
        """Not exposed.

        Gradient- or permutation-based attributions would be possible but are
        expensive and, on a signal this faint, not stable enough between seeds to
        be worth reporting. The training harness falls back to the linear models'
        coefficients for its importance table.
        """
        return None

    def training_curve(self) -> pd.DataFrame:
        """Per-epoch train and validation loss, for diagnosing the fit."""
        return pd.DataFrame(self.history_)


def _register() -> None:
    """Add the LSTM to the registry when PyTorch is present."""
    if not torch_available():
        return
    from .registry import MODEL_REGISTRY

    MODEL_REGISTRY.setdefault("lstm", LSTMModel)


_register()
