"""R-peak detection and median-beat extraction.

R peaks are detected on lead II, falling back to V5 and then to multi-lead energy.
neurokit2 is used when available, otherwise the built-in Pan-Tompkins-style detector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

try:
    import neurokit2 as _nk
    _HAS_NK = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_NK = False


def _pan_tompkins_like(lead: np.ndarray, fs: float) -> np.ndarray:
    """Built-in R-peak detection: band-pass, derivative, square, integrate; returns peak indices."""
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * fs
    b, a = butter(2, [5 / nyq, min(15, nyq * 0.99) / nyq], btype="band")
    x = filtfilt(b, a, lead)
    diff = np.diff(x, prepend=x[0])
    sq = diff ** 2
    win = max(1, int(0.15 * fs))
    integ = np.convolve(sq, np.ones(win) / win, mode="same")
    thr = np.mean(integ) + 0.5 * np.std(integ)
    min_dist = int(0.25 * fs)  # minimum RR interval ~240ms
    peaks, _ = find_peaks(integ, height=thr, distance=min_dist)
    # refine each peak to the local maximum of the raw signal
    refined = []
    half = int(0.05 * fs)
    for p in peaks:
        lo, hi = max(0, p - half), min(len(lead), p + half)
        if hi > lo:
            refined.append(lo + int(np.argmax(np.abs(lead[lo:hi]))))
    return np.array(sorted(set(refined)), dtype=int)


def detect_rpeaks(lead: np.ndarray, fs: float) -> np.ndarray:
    """Single-lead R-peak detection; returns an array of sample indices."""
    lead = np.nan_to_num(lead.astype(float))
    if _HAS_NK:
        try:
            _, info = _nk.ecg_peaks(lead, sampling_rate=int(fs))
            rp = np.asarray(info["ECG_R_Peaks"], dtype=int)
            if len(rp) >= 2:
                return rp
        except Exception:
            pass
    return _pan_tompkins_like(lead, fs)


@dataclass
class BeatResult:
    median_beat: np.ndarray | None   # [12, T_window]
    n_beats_used: int
    rpeak_lead: str
    quality: float                   # 0-1, higher is more reliable
    success: bool


def _rpeaks_with_fallback(sig12: np.ndarray, fs: float, lead_idx: dict[str, int]
                          ) -> tuple[np.ndarray, str]:
    """Try R-peak detection in the order II -> V5 -> multi-lead RMS energy."""
    for name in ("II", "V5"):
        idx = lead_idx.get(name)
        if idx is None:
            continue
        rp = detect_rpeaks(sig12[idx], fs)
        if len(rp) >= 2:
            return rp, name
    energy = np.sqrt(np.nanmean(sig12 ** 2, axis=0))
    return detect_rpeaks(energy, fs), "multi"


def extract_median_beat(
    sig12: np.ndarray,
    fs: float,
    lead_idx: dict[str, int],
    pre_ms: float = 300.0,
    post_ms: float = 700.0,
) -> BeatResult:
    """Extract the median beat over the window [-pre, +post] around the R peak at t = 0.

    sig12: [12, T]. Returns [12, T_window], T_window = round((pre+post)/1000*fs).
    The median beat is the point-wise median over beats at each time sample (noise robust).
    """
    n_leads, n_t = sig12.shape
    pre_n = int(round(pre_ms / 1000.0 * fs))
    post_n = int(round(post_ms / 1000.0 * fs))
    win = pre_n + post_n

    rpeaks, rlead = _rpeaks_with_fallback(sig12, fs, lead_idx)
    beats = []
    for r in rpeaks:
        lo, hi = r - pre_n, r + post_n
        if lo < 0 or hi > n_t:
            continue
        beats.append(sig12[:, lo:hi])
    if len(beats) == 0:
        return BeatResult(None, 0, rlead, 0.0, False)

    stack = np.stack(beats, axis=0)            # [n_beats, 12, win]
    median_beat = np.median(stack, axis=0).astype(np.float32)

    # quality: consistency, as the mean correlation of each beat with the median beat
    if len(beats) >= 2:
        flat = stack.reshape(len(beats), -1)
        med = median_beat.reshape(-1)
        corrs = []
        for b in flat:
            denom = (np.std(b) * np.std(med))
            if denom > 1e-8:
                corrs.append(np.corrcoef(b, med)[0, 1])
        quality = float(np.clip(np.mean(corrs), 0, 1)) if corrs else 0.5
    else:
        quality = 0.4
    return BeatResult(median_beat, len(beats), rlead, quality, True)
