"""Hand-crafted feature vectors from the 12-lead median beat (for the B0 baseline).

Stage-1 feature set: QRS width, QT, ST segment, T wave, R/S ratio, peak-to-peak amplitude,
and the key morphology of II/V1/V5. One fixed-length vector per record.
"""
from __future__ import annotations

import numpy as np

from . import morphology as morph
from ..data.ptbxl_loader import CANONICAL_LEADS, LEAD_TO_IDX

# scalar feature names extracted per lead
_PER_LEAD = ["ptp", "R_amp", "Q_amp", "S_amp", "RS_ratio", "ST_mean", "ST_slope", "T_amp", "T_area"]
_GLOBAL = ["QRS_width_ms", "QT_ms", "QTc_proxy"]


def feature_names() -> list[str]:
    names = []
    for lead in CANONICAL_LEADS:
        for f in _PER_LEAD:
            names.append(f"{lead}_{f}")
    names.extend(_GLOBAL)
    return names


def _per_lead_features(beat: np.ndarray, fs: float, r0: int) -> list[float]:
    """Per-lead features. beat: [T]."""
    n = len(beat)
    qrs_pre = beat[morph.window_slice(morph.QRS_PRE_MS, fs, r0, n)]
    qrs_post = beat[morph.window_slice(morph.QRS_POST_MS, fs, r0, n)]
    st = beat[morph.window_slice(morph.ST_MS, fs, r0, n)]
    t = beat[morph.window_slice(morph.T_MS, fs, r0, n)]

    ptp = float(np.nanmax(beat) - np.nanmin(beat))
    r_amp = float(beat[r0]) if 0 <= r0 < n else float(np.nanmax(beat))
    q_amp = float(np.nanmin(qrs_pre)) if qrs_pre.size else 0.0
    s_amp = float(np.nanmin(qrs_post)) if qrs_post.size else 0.0
    rs_ratio = float(r_amp / (abs(s_amp) + 1e-6))
    st_mean = float(np.nanmean(st)) if st.size else 0.0
    st_slope = morph.safe_slope(np.nan_to_num(st))
    # T-wave amplitude: signed value at the largest absolute value inside the window
    if t.size:
        t_amp = float(t[int(np.nanargmax(np.abs(t)))])
        t_area = float(np.nansum(t) / fs)
    else:
        t_amp, t_area = 0.0, 0.0
    return [ptp, r_amp, q_amp, s_amp, rs_ratio, st_mean, st_slope, t_amp, t_area]


def extract_features(beat12: np.ndarray, fs: float, pre_ms: float = 300.0) -> np.ndarray:
    """Extract the feature vector from one 12-lead median beat.

    beat12: [12, T], with the R peak at r0 = round(pre_ms/1000*fs).
    """
    beat12 = np.nan_to_num(beat12.astype(np.float64))
    n_t = beat12.shape[1]
    r0 = int(round(pre_ms / 1000.0 * fs))
    r0 = min(max(r0, 0), n_t - 1)

    feats: list[float] = []
    for li in range(beat12.shape[0]):
        feats.extend(_per_lead_features(beat12[li], fs, r0))

    # global: QRS width and QT estimated from the vector magnitude
    vm = np.sqrt(np.sum(beat12 ** 2, axis=0))
    qrs_w = morph.qrs_width_ms(vm, fs, r0)
    qt = morph.qt_interval_ms(vm, fs, r0)
    # no RR available; QTc is a Bazett placeholder at a fixed reference rate (75bpm, RR=0.8s)
    qtc = qt / np.sqrt(0.8) if qt > 0 else 0.0
    feats.extend([qrs_w, qt, qtc])
    return np.asarray(feats, dtype=np.float32)


def extract_features_batch(beats: np.ndarray, fs: float, pre_ms: float = 300.0) -> np.ndarray:
    """Batch extraction. beats: [N, 12, T] -> [N, F]."""
    return np.stack([extract_features(b, fs, pre_ms) for b in beats], axis=0)
