"""Median-beat morphology helpers: QRS / ST / T windows located relative to the R peak.

The median beat has the R peak at t = 0, r0 = index(pre_ms); windows are given in ms from it.
"""
from __future__ import annotations

import numpy as np

# feature windows relative to the R peak (t=0), in ms
QRS_PRE_MS = (-50.0, 0.0)
QRS_POST_MS = (0.0, 60.0)
QRS_FULL_MS = (-50.0, 60.0)
ST_MS = (60.0, 120.0)
T_MS = (150.0, 400.0)


def ms_to_idx(ms: float, fs: float, r0: int) -> int:
    return int(round(r0 + ms / 1000.0 * fs))


def window_slice(ms_range: tuple[float, float], fs: float, r0: int, n_t: int) -> slice:
    lo = max(0, ms_to_idx(ms_range[0], fs, r0))
    hi = min(n_t, ms_to_idx(ms_range[1], fs, r0))
    if hi <= lo:
        hi = min(n_t, lo + 1)
    return slice(lo, hi)


def safe_slope(y: np.ndarray) -> float:
    """Least-squares linear fit of a segment; returns the slope per sample."""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n)
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom < 1e-12:
        return 0.0
    return float((x * (y - y.mean())).sum() / denom)


def qrs_width_ms(vm: np.ndarray, fs: float, r0: int, frac: float = 0.2) -> float:
    """QRS width (ms) from vm(t): contiguous run around the R peak where vm exceeds frac*peak."""
    n = len(vm)
    qrs_region = window_slice((-80.0, 100.0), fs, r0, n)
    seg = vm[qrs_region]
    if seg.size == 0:
        return 0.0
    peak = float(seg.max())
    if peak < 1e-8:
        return 0.0
    thr = frac * peak
    above = seg >= thr
    # take the contiguous True run containing the local peak
    peak_pos = int(np.argmax(seg))
    lo = peak_pos
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = peak_pos
    while hi < len(above) - 1 and above[hi + 1]:
        hi += 1
    return (hi - lo + 1) / fs * 1000.0


def qt_interval_ms(vm: np.ndarray, fs: float, r0: int) -> float:
    """Approximate QT: QRS onset to T-wave end, where vm falls back to the baseline threshold."""
    n = len(vm)
    qrs_start = max(0, ms_to_idx(-50.0, fs, r0))
    t_region = window_slice((150.0, 500.0), fs, r0, n)
    seg = vm[t_region]
    if seg.size == 0:
        return 0.0
    base = float(np.median(vm[window_slice((-280.0, -200.0), fs, r0, n)])) if n > 0 else 0.0
    peak = float(seg.max())
    thr = base + 0.1 * (peak - base)
    # locate the fall-back point after the T peak
    t_peak = int(np.argmax(seg))
    end = t_region.start + t_peak
    for i in range(t_region.start + t_peak, t_region.stop):
        if vm[i] <= thr:
            end = i
            break
    else:
        end = t_region.stop - 1
    return (end - qrs_start) / fs * 1000.0
