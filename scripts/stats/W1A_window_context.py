"""W1A supplement — context for the four segment windows.

The P window NRMSE lands above 1.0, i.e. worse than the flat/zero predictor that the
NRMSE denominator implies. Before the manuscript prints (or refuses to print) that number
we need to know whether it is a real reconstruction failure or a small-denominator artefact.

Emits, per seed and per window (P/QRS/ST/T):
  median per-record NRMSE                       (the headline convention)
  median per-record signal RMS  (scaled + mV)   (denominator size)
  median per-record residual RMS (scaled + mV)  (absolute error size)
  median per-record windowed mean-lead Pearson r (is the shape captured at all?)
plus a per-lead breakdown of window NRMSE.

Writes W1A_window_context.csv and W1A_window_by_lead.csv into runs/v7rev_stats.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro
REPO = repro.repo()
sys.path.insert(0, str(REPO))
# siblings now live beside this file; no sys.path surgery needed

from igraphecg.data.dataset import load_processed, split_indices        # noqa: E402
from igraphecg.data.preprocess import RobustLeadScaler                  # noqa: E402
from igraphecg.data.ptbxl_loader import CANONICAL_LEADS                 # noqa: E402
from igraphecg.evaluation.recon_metrics import _time_mask               # noqa: E402

from W1A_per_record_recon import (CKPT_DIR, NPZ, OUT, SEEDS,            # noqa: E402
                                  WINDOWS, infer, masked_nrmse_vec)


def windowed_corr(y, yhat):
    """per-record mean over leads of Pearson r inside the window (same drop rule as
    recon_metrics.per_sample_lead_corr: skip leads whose std <= 1e-8)."""
    N, L, _ = y.shape
    out = np.zeros(N)
    for i in range(N):
        rs = []
        for l in range(L):
            a, b = y[i, l], yhat[i, l]
            if a.std() > 1e-8 and b.std() > 1e-8:
                rs.append(np.corrcoef(a, b)[0, 1])
        out[i] = np.mean(rs) if rs else 0.0
    return out


def main():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_processed(NPZ)
    signals = data["signal_12lead"].astype(np.float32)
    fold = data["fold"].astype(int)
    stats = {"median": data["scaler_median"], "iqr": data["scaler_iqr"]}
    scaler = RobustLeadScaler.from_dict(stats)
    scaled = scaler.transform(signals)
    te = split_indices(fold)["test"]

    ctx_rows, lead_rows = [], []
    for seed in SEEDS:
        yh, t_sec, *_ = infer(repro.lineage.checkpoint(seed), scaled, te, stats, device)
        y = scaled[te]
        y_mV, yh_mV = signals[te], scaler.inverse_transform(yh)
        for wn, wv in WINDOWS.items():
            m = _time_mask(t_sec, wv)
            n_s = int(m.sum())
            ys, yhs = y[:, :, m], yh[:, :, m]
            ym, yhm = y_mV[:, :, m], yh_mV[:, :, m]
            nr = masked_nrmse_vec(y, yh, t_sec, wv)
            ctx_rows.append({
                "seed": seed, "window": wn, "t_lo_s": wv[0], "t_hi_s": wv[1], "n_samples": n_s,
                "median_nrmse": float(np.median(nr)),
                "median_signal_rms_scaled": float(np.median(np.sqrt((ys ** 2).mean(axis=(1, 2))))),
                "median_resid_rms_scaled": float(np.median(np.sqrt(((ys - yhs) ** 2).mean(axis=(1, 2))))),
                "median_signal_rms_mV": float(np.median(np.sqrt((ym ** 2).mean(axis=(1, 2))))),
                "median_resid_rms_mV": float(np.median(np.sqrt(((ym - yhm) ** 2).mean(axis=(1, 2))))),
                "median_windowed_corr": float(np.median(windowed_corr(ys, yhs))),
                "frac_records_nrmse_gt1": float((nr > 1.0).mean()),
            })
            for li, ln in enumerate(CANONICAL_LEADS):
                v = masked_nrmse_vec(y[:, li:li + 1], yh[:, li:li + 1], t_sec, wv)
                lead_rows.append({"seed": seed, "window": wn, "lead": ln,
                                  "median_nrmse": float(np.median(v))})
        print(f"[seed {seed}] done")

    pd.DataFrame(ctx_rows).to_csv(OUT / "W1A_window_context.csv", index=False)
    pd.DataFrame(lead_rows).to_csv(OUT / "W1A_window_by_lead.csv", index=False)
    d = pd.DataFrame(ctx_rows)
    g = d.groupby("window").agg(nrmse_mean=("median_nrmse", "mean"),
                                nrmse_sd=("median_nrmse", lambda s: float(np.std(s, ddof=1))),
                                sig_rms_mV=("median_signal_rms_mV", "mean"),
                                res_rms_mV=("median_resid_rms_mV", "mean"),
                                corr=("median_windowed_corr", "mean"),
                                frac_gt1=("frac_records_nrmse_gt1", "mean"))
    print(g.round(4).to_string())
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()
