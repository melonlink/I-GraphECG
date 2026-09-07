"""PTB-XL adapter. Wraps the existing processed median-beat NPZ (behavior unchanged).

The processed NPZ already contains R-aligned 12-lead median beats at 100 Hz produced by
the original pipeline (scripts/10_prepare_ptbxl.py); this adapter exposes them through the
common ECGSource interface so PTB-XL and external databases share one downstream path.
"""
from __future__ import annotations
from typing import Iterator
import numpy as np

from ..config import CANONICAL_LEADS, TARGET_FS, PTBXL_NPZ
from .base import ECGRecord, ECGSource


class PTBXLSource(ECGSource):
    """Iterates the processed PTB-XL clean four-class median beats.

    Note: these are already median beats at 100 Hz; `already_median_beat=True` tells the
    preprocessing pipeline to skip re-extraction and only (re)apply the saved scaler, so the
    published PTB-XL numbers are reproduced bit-for-bit.
    """
    already_median_beat = True

    def __init__(self, npz_path=PTBXL_NPZ, split: str | None = None):
        import sys
        from igraphecg.data.dataset import load_processed, split_indices
        self._data = load_processed(npz_path)
        self.fold = self._data["fold"]
        self.signals = self._data["signal_12lead"]    # [N,12,T]
        self.labels = self._data["label"].astype(int)
        self.label_names = self._data["label_name"]
        self.record_ids = self._data.get("record_id", np.arange(len(self.labels)))
        if split is not None:
            idx = split_indices(self.fold)[split]
            self._sel = idx
        else:
            self._sel = np.arange(len(self.labels))

    @property
    def name(self) -> str:
        return "ptbxl"

    def raw(self) -> dict:
        """Expose the full processed dict (scaler stats, folds) for the regression path."""
        return self._data

    def iter_records(self) -> Iterator[ECGRecord]:
        for i in self._sel:
            yield ECGRecord(
                signal_12lead=self.signals[i],
                fs=TARGET_FS,
                lead_names=list(CANONICAL_LEADS),
                source_labels=[str(self.label_names[i])],
                record_id=str(self.record_ids[i]),
                source_name="ptbxl",
            )
