"""Loading of processed .npz datasets and the torch Dataset wrapper."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_processed(npz_path: str | Path) -> dict:
    """Load a processed .npz file and return its fields as a dict."""
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def split_indices(fold: np.ndarray) -> dict[str, np.ndarray]:
    """Return train/val/test sample indices from the fold column (train=1-8, val=9, test=10)."""
    fold = np.asarray(fold).astype(int)
    return {
        "train": np.where(fold <= 8)[0],
        "val": np.where(fold == 9)[0],
        "test": np.where(fold == 10)[0],
    }


class ECGDataset:
    """torch Dataset yielding (signal[12,T] float32, label int64).

    torch is imported lazily so this module stays importable without torch installed.
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        import torch  # noqa: F401  lazy import
        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        import torch
        return torch.from_numpy(self.signals[i]), torch.tensor(self.labels[i])
