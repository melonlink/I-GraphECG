"""Signal preprocessing: band-pass, optional notch, quality check, lead-wise robust scaling."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(sig: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0,
                    order: int = 4) -> np.ndarray:
    """Zero-phase 0.5-40 Hz band-pass (Butterworth + filtfilt) along the last axis.

    sig: [..., T]. high is clipped to stay below Nyquist.
    """
    nyq = 0.5 * fs
    high = min(high, nyq * 0.99)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    # filtfilt requires a sufficiently long signal
    if sig.shape[-1] <= 3 * max(len(a), len(b)):
        return sig
    return filtfilt(b, a, sig, axis=-1).astype(np.float32)


def notch_filter(sig: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Power-line notch filter (50 Hz by default)."""
    nyq = 0.5 * fs
    if freq >= nyq:
        return sig
    b, a = iirnotch(freq / nyq, q)
    return filtfilt(b, a, sig, axis=-1).astype(np.float32)


@dataclass
class QualityReport:
    has_nan: bool = False
    n_constant_leads: int = 0
    max_abs_amplitude: float = 0.0
    abnormal_amplitude: bool = False
    is_ok: bool = True


def check_quality(sig12: np.ndarray, max_mv: float = 20.0) -> QualityReport:
    """Check for NaNs, constant leads and abnormal amplitudes. sig12: [12, T] in mV."""
    rep = QualityReport()
    rep.has_nan = bool(np.isnan(sig12).any())
    # constant lead: peak-to-peak range near 0
    ptp = np.nanmax(sig12, axis=-1) - np.nanmin(sig12, axis=-1)
    rep.n_constant_leads = int(np.sum(ptp < 1e-6))
    rep.max_abs_amplitude = float(np.nanmax(np.abs(sig12))) if sig12.size else 0.0
    rep.abnormal_amplitude = rep.max_abs_amplitude > max_mv
    rep.is_ok = (not rep.has_nan) and (rep.n_constant_leads == 0) and (not rep.abnormal_amplitude)
    return rep


@dataclass
class RobustLeadScaler:
    """Lead-wise robust scaling: (x - median) / IQR. Statistics fitted on the training set only.

    Operates on [N, 12, T]: per-lead median and IQR estimated over all training time points.
    """
    median_: np.ndarray = field(default=None)
    iqr_: np.ndarray = field(default=None)

    def fit(self, x: np.ndarray) -> "RobustLeadScaler":
        # x: [N, 12, T] -> flatten N*T per lead
        n_leads = x.shape[1]
        flat = x.transpose(1, 0, 2).reshape(n_leads, -1)
        self.median_ = np.nanmedian(flat, axis=1).astype(np.float32)
        q75 = np.nanpercentile(flat, 75, axis=1)
        q25 = np.nanpercentile(flat, 25, axis=1)
        iqr = (q75 - q25).astype(np.float32)
        iqr[iqr < 1e-6] = 1.0  # guard against division by zero
        self.iqr_ = iqr
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        m = self.median_[None, :, None]
        s = self.iqr_[None, :, None]
        return ((x - m) / s).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        m = self.median_[None, :, None]
        s = self.iqr_[None, :, None]
        return (x * s + m).astype(np.float32)

    def to_dict(self) -> dict:
        return {"median": self.median_, "iqr": self.iqr_}

    @classmethod
    def from_dict(cls, d: dict) -> "RobustLeadScaler":
        obj = cls()
        obj.median_ = np.asarray(d["median"], dtype=np.float32)
        obj.iqr_ = np.asarray(d["iqr"], dtype=np.float32)
        return obj
