"""Frozen preprocessing pipeline, shared by PTB-XL and all external databases.

make_median_beat(record, target_fs=100, scaler=None, external=False):
  1. resample to target_fs (Challenge is usually 500 Hz; some PTB 1000 Hz)
  2. zero-phase 0.5-40 Hz band-pass
  3. R-peak detection (NeuroKit2 lead II, fallback V5/multi-lead) -- via existing code
  4. R-aligned -300..+700 ms 12-lead median beat
  5. robust median/IQR scaling using the PASSED-IN scaler (PTB-XL train-fit)

Hard rule (external validation): `scaler` must be a fitted RobustLeadScaler loaded from
artifacts; refitting on external data is forbidden. With external=True we assert the
scaler is provided and never call .fit().
"""
from __future__ import annotations
import numpy as np
from scipy.signal import resample_poly

from ..config import (TARGET_FS, BANDPASS_LOW, BANDPASS_HIGH,
                      WINDOW_PRE_MS, WINDOW_POST_MS)
from ..sources.base import ECGRecord


def _resample_12lead(sig: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return sig.astype(np.float32)
    from math import gcd
    g = gcd(int(fs_in), int(fs_out))
    up, down = fs_out // g, fs_in // g
    # resample each lead; NaN leads stay NaN
    out = []
    for row in sig:
        if np.all(np.isnan(row)):
            n_out = int(round(len(row) * fs_out / fs_in))
            out.append(np.full(n_out, np.nan, dtype=np.float32))
        else:
            out.append(resample_poly(np.nan_to_num(row), up, down).astype(np.float32))
    n = min(len(r) for r in out)
    return np.stack([r[:n] for r in out]).astype(np.float32)


def make_median_beat(record: ECGRecord, target_fs: int = TARGET_FS,
                     scaler=None, external: bool = False) -> np.ndarray | None:
    """Return a scaled [12, T] median beat, or None if R-peak/quality fails.

    For external sources, `external=True` enforces a provided (pre-fit) scaler.
    """
    from igraphecg.data.preprocess import bandpass_filter
    from igraphecg.data.beat_extraction import extract_median_beat
    from igraphecg.data.ptbxl_loader import LEAD_TO_IDX

    if external and scaler is None:
        raise AssertionError("external preprocessing requires a pre-fit scaler "
                             "(refitting on external data is forbidden)")

    sig = np.asarray(record.signal_12lead, dtype=np.float32)
    # 1. resample to target_fs
    sig = _resample_12lead(sig, record.fs, target_fs)
    # 2. band-pass (zero-phase)
    sig = bandpass_filter(np.nan_to_num(sig), target_fs, BANDPASS_LOW, BANDPASS_HIGH)
    # 3-4. R detection + median beat (existing verified code; lead II primary)
    res = extract_median_beat(sig, target_fs, pre_ms=WINDOW_PRE_MS, post_ms=WINDOW_POST_MS,
                              lead_idx=LEAD_TO_IDX)
    if res is None or getattr(res, "median_beat", None) is None:
        return None
    beat = np.asarray(res.median_beat, dtype=np.float32)   # [12, T] physical mV
    # 5. scale with the provided (PTB-XL train-fit) scaler
    if scaler is not None:
        if external:
            assert not _will_fit(scaler), "scaler must be already fit for external data"
        beat = scaler.transform(beat[None])[0]
    return beat


def _will_fit(scaler) -> bool:
    # RobustLeadScaler is 'fit' iff it has median_/iqr_ set
    return getattr(scaler, "median_", None) is None
