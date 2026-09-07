"""Repolarization descriptors for the graph-Eikonal surrogate.

(1) Parameter domain: APD / tau_rep / repolarization end time T_end / repolarization synchrony
    Sync_rep (ventricular nodes).
(2) ECG domain: T-wave morphology features from the (real or reconstructed) median beat,
    aggregated across leads.

Named parameters are unpacked through :class:`ParamSpace`. In particular, the
first eight graph-Eikonal parameters are ``delta_root`` and seven edge delays;
they are not eight free nodal activation times. Nodal ``delta`` values are
therefore obtained from the decoder path matrix rather than positional slices.
"""
from __future__ import annotations

import numpy as np
import torch

from ..models.surrogate_decoder import ParamSpace, VENT_NODES

VENT = VENT_NODES

# ECG T-wave / ST / QRS windows (ms, relative to the R peak)
T_WIN = (160.0, 450.0)
ST_WIN = (60.0, 120.0)


def _cv(x, axis=-1, eps=1e-6):
    return x.std(axis=axis) / (np.abs(x.mean(axis=axis)) + eps)


def repolarization_param_features(theta: np.ndarray, c_rep: float = 2.0,
                                  tau0: float = 0.05) -> dict[str, np.ndarray]:
    """Parameter-domain repolarization descriptors from theta[N,47], all on ventricular nodes."""
    theta_t = torch.as_tensor(theta, dtype=torch.float32)
    p = ParamSpace.unpack(theta_t)
    apd = p["APD"][..., VENT].cpu().numpy()
    taur = p["tau_rep"][..., VENT].cpu().numpy()
    delta = p["delta"][..., VENT].cpu().numpy()
    tend = delta + apd + c_rep * taur
    return {
        "APD_mean": apd.mean(1), "APD_disp": apd.max(1) - apd.min(1), "APD_cv": _cv(apd),
        "tau_rep_mean": taur.mean(1), "tau_rep_disp": taur.max(1) - taur.min(1), "tau_rep_cv": _cv(taur),
        "Tend_mean": tend.mean(1), "Tend_disp": tend.max(1) - tend.min(1), "Tend_cv": _cv(tend),
        "Sync_rep": np.exp(-tend.std(1) / tau0),
    }


def _win_idx(win_ms, fs, r0, n):
    lo = max(0, int(round(r0 + win_ms[0] / 1000 * fs)))
    hi = min(n, int(round(r0 + win_ms[1] / 1000 * fs)))
    return lo, max(lo + 1, hi)


def twave_features(signals: np.ndarray, fs: float = 100, pre_ms: float = 300.0
                   ) -> dict[str, np.ndarray]:
    """T-wave morphology from [N,12,T] median beats, aggregated over 12 leads; returns dict[N]."""
    N, L, T = signals.shape
    r0 = int(round(pre_ms / 1000 * fs))
    tlo, thi = _win_idx(T_WIN, fs, r0, T)
    slo, shi = _win_idx(ST_WIN, fs, r0, T)
    tseg = signals[:, :, tlo:thi]                       # [N,12,Tt]
    twin_t = (np.arange(tlo, thi) - r0) / fs * 1000     # ms

    # Per-lead T-wave features
    amax_i = np.abs(tseg).argmax(axis=2)                # [N,12] index of the absolute peak
    peak_amp = np.take_along_axis(tseg, amax_i[:, :, None], axis=2)[:, :, 0]   # signed peak
    peak_time = twin_t[amax_i]
    area = tseg.sum(axis=2) / fs
    abs_area = np.abs(tseg).sum(axis=2) / fs
    polarity = np.sign(peak_amp)
    # Width proxy: fraction of samples with |T| > 0.3 * peak
    thr = 0.3 * np.abs(peak_amp)[:, :, None]
    width = (np.abs(tseg) >= np.maximum(thr, 1e-6)).mean(axis=2)
    # Asymmetry: (area after peak - area before peak), normalized
    asym = np.zeros((N, L))
    for n in range(N):
        for l in range(L):
            k = amax_i[n, l]
            before = np.abs(tseg[n, l, :k]).sum(); after = np.abs(tseg[n, l, k:]).sum()
            asym[n, l] = (after - before) / (before + after + 1e-6)
    low_amp = np.exp(-np.abs(peak_amp) / 0.2)           # lower amplitude -> higher score
    st_seg = signals[:, :, slo:shi]
    st_mean = st_seg.mean(axis=2)
    # ST slope (linear fit)
    xs = np.arange(st_seg.shape[2]) - (st_seg.shape[2] - 1) / 2
    st_slope = (st_seg * xs).sum(axis=2) / (xs @ xs + 1e-9)

    def agg(M, name):
        return {f"{name}_mean": M.mean(1), f"{name}_std": M.std(1),
                f"{name}_maxabs": np.abs(M).max(1), f"{name}_range": M.max(1) - M.min(1)}

    out: dict[str, np.ndarray] = {}
    for M, nm in [(peak_amp, "Tpeak"), (peak_time, "Tptime"), (area, "Tarea"),
                  (abs_area, "Tabsarea"), (width, "Twidth"), (asym, "Tasym"),
                  (low_amp, "Tlowamp"), (st_mean, "STmean"), (st_slope, "STslope")]:
        out.update(agg(M, nm))
    # T-wave polarity discordance rate: leads whose polarity differs from lead II (index 1)
    ref = polarity[:, 1:2]
    out["T_sign_discordance_rate"] = (polarity != ref).mean(axis=1)
    return out
