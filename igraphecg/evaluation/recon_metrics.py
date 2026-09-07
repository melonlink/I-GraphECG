"""ECG reconstruction metrics (ROUND2 §7.1 / §10): NRMSE, per-lead Pearson r, QRS/ST/T errors."""
from __future__ import annotations

import numpy as np

QRS_WIN = (-0.06, 0.10)
ST_WIN = (0.06, 0.14)
T_WIN = (0.12, 0.45)


def _time_mask(t: np.ndarray, win) -> np.ndarray:
    return (t >= win[0]) & (t <= win[1])


def per_sample_nrmse(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """y,yhat:[N,12,T] -> per-sample NRMSE [N] (pooled over all leads)."""
    num = np.sqrt(((y - yhat) ** 2).sum(axis=(1, 2)))
    den = np.sqrt((y ** 2).sum(axis=(1, 2))) + eps
    return num / den


def per_sample_lead_corr(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Per-sample mean of the Pearson r over the 12 leads [N]."""
    N, L, T = y.shape
    out = np.zeros(N)
    for i in range(N):
        rs = []
        for l in range(L):
            a, b = y[i, l], yhat[i, l]
            sa, sb = a.std(), b.std()
            if sa > 1e-8 and sb > 1e-8:
                rs.append(np.corrcoef(a, b)[0, 1])
        out[i] = np.mean(rs) if rs else 0.0
    return out


def masked_nrmse(y, yhat, t, win, eps: float = 1e-8) -> float:
    m = _time_mask(t, win)
    yy, yh = y[:, :, m], yhat[:, :, m]
    num = np.sqrt(((yy - yh) ** 2).sum(axis=(1, 2)))
    den = np.sqrt((yy ** 2).sum(axis=(1, 2))) + eps
    return float(np.median(num / den))


def st_mean_abs_error(y, yhat, t) -> float:
    m = _time_mask(t, ST_WIN)
    return float(np.median(np.abs(y[:, :, m].mean(axis=2) - yhat[:, :, m].mean(axis=2)).mean(axis=1)))


def compute_recon_metrics(y, yhat, t) -> dict:
    nr = per_sample_nrmse(y, yhat)
    cr = per_sample_lead_corr(y, yhat)
    return {
        "median_nrmse": float(np.median(nr)),
        "mean_nrmse": float(np.mean(nr)),
        "median_corr": float(np.median(cr)),
        "mean_corr": float(np.mean(cr)),
        "qrs_nrmse": masked_nrmse(y, yhat, t, QRS_WIN),
        "st_nrmse": masked_nrmse(y, yhat, t, ST_WIN),
        "t_nrmse": masked_nrmse(y, yhat, t, T_WIN),
        "st_mean_abs_err": st_mean_abs_error(y, yhat, t),
    }
