"""W1A — per-record / per-lead reconstruction metrics on fold-10 (test), 5 seeds.

Frozen lineage, read-only w.r.t. everything outside the output directory:
  checkpoints  <output root>/checkpoints/d1s_r3_s{42,1,2,3,4}_best.pt
  data         data/processed/ptbxl_medianbeat_clean_100hz.npz
  metrics      igraphecg.evaluation.recon_metrics (window convention reused verbatim)

Adds ONE new window, P_WIN = (-0.30, -0.10) s relative to R, using exactly the
same masking convention as QRS/ST/T (inclusive both ends, `_time_mask`).

Writes into <output root>/runs/v7rev_stats:
  W1A_per_record_recon_s{seed}.csv    one row per fold-10 record
  W1A_summary.csv                     tidy per-seed medians + 5-seed mean/sd (ddof=1)
  W1A_controls.json                   control reproduction check

Usage:
  python <this file>
"""
from __future__ import annotations

import json
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

from igraphecg.data.dataset import load_processed, split_indices          # noqa: E402
from igraphecg.data.preprocess import RobustLeadScaler                    # noqa: E402
from igraphecg.data.ptbxl_loader import CANONICAL_LEADS                   # noqa: E402
from igraphecg.evaluation.recon_metrics import (QRS_WIN, ST_WIN, T_WIN,   # noqa: E402
                                                _time_mask, masked_nrmse,
                                                per_sample_lead_corr,
                                                per_sample_nrmse)

# NEW window for the revision: P wave, same convention as QRS/ST/T
P_WIN = (-0.30, -0.10)
WINDOWS = {"P": P_WIN, "QRS": QRS_WIN, "ST": ST_WIN, "T": T_WIN}

NPZ = REPO / "data/processed/ptbxl_medianbeat_clean_100hz.npz"
CKPT_DIR = repro.outputs() / "checkpoints"
OUT = repro.outdir("runs/v7rev_stats")
SEEDS = [42, 1, 2, 3, 4]

CONTROLS = {                       # seed 42, fold-10 (from the frozen run tables)
    "median_nrmse": 0.4009,
    "median_corr": 0.9115,
    "qrs_nrmse": 0.2437,
    "lead_median_corr": {"I": 0.9364, "aVF": 0.8946, "V5": 0.9544},
}
TOL = 5e-4


def masked_nrmse_vec(y, yhat, t, win, eps: float = 1e-8) -> np.ndarray:
    """Per-record window NRMSE. Byte-identical to recon_metrics.masked_nrmse
    minus the final np.median (verified below)."""
    m = _time_mask(t, win)
    yy, yh = y[:, :, m], yhat[:, :, m]
    num = np.sqrt(((yy - yh) ** 2).sum(axis=(1, 2)))
    den = np.sqrt((yy ** 2).sum(axis=(1, 2))) + eps
    return num / den


def infer(ckpt_path: Path, scaled: np.ndarray, idx: np.ndarray, scaler_stats: dict, device):
    import torch
    from igraphecg.models.phys_encoder import PhysEncoder
    from igraphecg.models.surrogate_decoder import PARAM_DIM, SurrogateDecoder

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    rank = int(cfg.get("leadfield_rank", 3))
    scaled_deriv = bool(cfg.get("scaled_derivation", False))
    # exactly as scripts/31_surrogate_encoder.py builds the decoder
    dec_scaler = scaler_stats if scaled_deriv else None
    enc = PhysEncoder(in_ch=12, out_dim=PARAM_DIM)
    dec = SurrogateDecoder(n_t=scaled.shape[2], leadfield_rank=rank, scaler=dec_scaler)
    enc.load_state_dict(ck["encoder"])
    dec.load_state_dict(ck["decoder"])
    enc.to(device).eval()
    dec.to(device).eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(idx), 512):
            xb = torch.from_numpy(scaled[idx[s:s + 512]]).to(device)
            yh, _ = dec.forward_from_z(enc(xb))
            outs.append(yh.cpu().numpy())
    return np.concatenate(outs), dec.t.cpu().numpy(), rank, scaled_deriv, int(cfg.get("seed", -1))


def main():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_processed(NPZ)
    signals = data["signal_12lead"].astype(np.float32)
    fold = data["fold"].astype(int)
    labels = data["label"].astype(int)
    label_name = data["label_name"].astype(str)
    record_id = data["record_id"].astype(int)
    leads = [str(x) for x in data["lead_names"]]
    assert leads == list(CANONICAL_LEADS), (leads, CANONICAL_LEADS)

    stats = {"median": data["scaler_median"], "iqr": data["scaler_iqr"]}
    scaler = RobustLeadScaler.from_dict(stats)
    scaled = scaler.transform(signals)
    idx = split_indices(fold)
    te = idx["test"]
    print(f"device={device}  N={len(fold)}  test(fold-10) n={len(te)}")

    rows_summary = []
    per_seed = {}
    controls_report = {}

    for seed in SEEDS:
        ck = repro.lineage.checkpoint(seed)
        yh, t_sec, rank, sdv, cfg_seed = infer(ck, scaled, te, stats, device)
        y = scaled[te]
        assert y.shape == yh.shape

        nr = per_sample_nrmse(y, yh)
        cr = per_sample_lead_corr(y, yh)

        rec = pd.DataFrame({
            "record_id": record_id[te], "fold": fold[te],
            "label": labels[te], "label_name": label_name[te],
            "seed": seed,
            "nrmse_scaled_12lead": nr,
            "corr_meanlead_12lead": cr,
        })

        # per-lead (single-lead slice, same convention as scripts/05 per-lead table)
        lead_med = {}
        for li, ln in enumerate(CANONICAL_LEADS):
            c_l = per_sample_lead_corr(y[:, li:li + 1, :], yh[:, li:li + 1, :])
            n_l = per_sample_nrmse(y[:, li:li + 1, :], yh[:, li:li + 1, :])
            rec[f"corr_{ln}"] = c_l
            rec[f"nrmse_{ln}"] = n_l
            lead_med[ln] = (float(np.median(c_l)), float(np.median(n_l)))

        # windows (12-lead Frobenius inside the window), per record
        win_med = {}
        for wn, wv in WINDOWS.items():
            v = masked_nrmse_vec(y, yh, t_sec, wv)
            rec[f"nrmse_win_{wn}"] = v
            win_med[wn] = float(np.median(v))
            if wn in ("QRS", "ST", "T"):     # convention identity check vs repo function
                ref = masked_nrmse(y, yh, t_sec, wv)
                assert abs(ref - win_med[wn]) < 1e-12, (wn, ref, win_med[wn])

        rec.to_csv(OUT / f"W1A_per_record_recon_s{seed}.csv", index=False)
        per_seed[seed] = {
            "median_nrmse": float(np.median(nr)), "mean_nrmse": float(np.mean(nr)),
            "median_corr": float(np.median(cr)), "mean_corr": float(np.mean(cr)),
            "lead": lead_med, "win": win_med,
            "rank": rank, "scaled_derivation": sdv, "cfg_seed": cfg_seed, "ckpt": "checkpoints/" + ck.name,   # output-root relative, same in both trees
        }
        print(f"[seed {seed}] rank={rank} sdv={sdv} cfg_seed={cfg_seed} n={len(nr)}  "
              f"medNRMSE={np.median(nr):.6f} medCORR={np.median(cr):.6f}  "
              + "  ".join(f"{k}={v:.4f}" for k, v in win_med.items()))

        rows_summary.append({"seed": str(seed), "metric": "median_nrmse", "scope": "all_12lead",
                             "value": per_seed[seed]["median_nrmse"]})
        rows_summary.append({"seed": str(seed), "metric": "mean_nrmse", "scope": "all_12lead",
                             "value": per_seed[seed]["mean_nrmse"]})
        rows_summary.append({"seed": str(seed), "metric": "median_corr", "scope": "all_12lead",
                             "value": per_seed[seed]["median_corr"]})
        rows_summary.append({"seed": str(seed), "metric": "mean_corr", "scope": "all_12lead",
                             "value": per_seed[seed]["mean_corr"]})
        for ln, (c, n) in lead_med.items():
            rows_summary.append({"seed": str(seed), "metric": "lead_median_corr",
                                 "scope": ln, "value": c})
            rows_summary.append({"seed": str(seed), "metric": "lead_median_nrmse",
                                 "scope": ln, "value": n})
        for wn, v in win_med.items():
            rows_summary.append({"seed": str(seed), "metric": "window_median_nrmse",
                                 "scope": wn, "value": v})

    # ---- controls (seed 42) ----
    p42 = per_seed[42]
    checks = {
        "median_nrmse": (p42["median_nrmse"], CONTROLS["median_nrmse"]),
        "median_corr": (p42["median_corr"], CONTROLS["median_corr"]),
        "qrs_window_nrmse": (p42["win"]["QRS"], CONTROLS["qrs_nrmse"]),
    }
    for ln, exp in CONTROLS["lead_median_corr"].items():
        checks[f"lead_median_corr_{ln}"] = (p42["lead"][ln][0], exp)
    ok = True
    for k, (got, exp) in checks.items():
        good = abs(got - exp) <= TOL
        ok &= good
        controls_report[k] = {"got": got, "expected": exp, "abs_diff": abs(got - exp),
                              "pass": bool(good)}
        print(f"CONTROL {k:26s} got={got:.6f} expected={exp:.4f} "
              f"diff={abs(got-exp):.2e} {'OK' if good else 'FAIL'}")

    # ---- five-seed aggregate (mean/sd over the five per-seed medians, ddof=1) ----
    df = pd.DataFrame(rows_summary)
    agg = (df.groupby(["metric", "scope"])["value"]
             .agg(mean="mean", sd=lambda s: float(np.std(s, ddof=1)))
             .reset_index())
    for _, r in agg.iterrows():
        rows_summary.append({"seed": "mean5", "metric": r["metric"], "scope": r["scope"],
                             "value": float(r["mean"])})
        rows_summary.append({"seed": "sd5", "metric": r["metric"], "scope": r["scope"],
                             "value": float(r["sd"])})
    out_df = pd.DataFrame(rows_summary)
    out_df.to_csv(OUT / "W1A_summary.csv", index=False)

    payload = {
        "task": "W1A",
        "n_test_fold10": int(len(te)),
        "seeds": SEEDS,
        "windows_s": {k: list(v) for k, v in WINDOWS.items()},
        "t_axis": {"t0": -0.30, "t1": 0.69, "T": int(signals.shape[2]),
                   "mask_convention": "(t >= win[0]) & (t <= win[1])  [inclusive both ends]"},
        "controls": controls_report,
        "controls_all_pass": bool(ok),
        "per_seed": per_seed,
        "five_seed": {f"{m}|{s}": {"mean": float(a), "sd": float(b)}
                      for m, s, a, b in zip(agg["metric"], agg["scope"], agg["mean"], agg["sd"])},
    }
    (OUT / "W1A_controls.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"\nwrote -> {OUT}")
    if not ok:
        print("!!! CONTROL MISMATCH")
        sys.exit(2)


if __name__ == "__main__":
    main()
