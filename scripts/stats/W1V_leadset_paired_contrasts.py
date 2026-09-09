"""W1V: paired patient-clustered contrasts between the reduced-lead models (Reviewer 3, Comment 7).

Table 6 reports the macro-AUROC of each lead-set encoder with its own patient-clustered interval;
overlapping intervals are not a pairwise test. This script retrains the three lead-set encoders of
Table 6 with the locked recipe of scripts/55_leadset_classification.py (full-12 teacher loss,
stock initialization, seed 42), keeps their per-record C5 probabilities on fold 10 together with
those of the locked 12-lead model, and applies the W1-H protocol of Table S16: 2,000 joint
patient-clustered bootstrap resamples, two-sided bootstrap p-values for every pairwise contrast and
Benjamini-Hochberg control over the six contrasts.

The retrained encoders are by-products (like those of W1O); only the summary is registered.
Writes runs/v7rev_stats/W1V_leadset_paired_contrasts.csv and W1V_headline.json.
Run (repository root, torch environment): python scripts/stats/W1V_leadset_paired_contrasts.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from igraphecg import repro  # noqa: E402
from igraphecg.data.dataset import load_processed  # noqa: E402
from igraphecg.data.label_utils import TARGET_CLASSES  # noqa: E402
from igraphecg.data.preprocess import RobustLeadScaler  # noqa: E402
from igraphecg.evaluation.round2_io import infer_theta, load_encoder_decoder  # noqa: E402

OUT = repro.outdir("runs/v7rev_stats")
NPZ = ROOT / "data/processed/ptbxl_medianbeat_clean_100hz.npz"
N = len(TARGET_CLASSES)
N_BOOT = 2000
MODELS = ["12 leads", "I+V1+V4", "II+V1+V5", "II"]


def _load_script(name: str):
    p = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def macro_auroc(y, P):
    from sklearn.metrics import roc_auc_score
    return float(np.mean([roc_auc_score((y == c).astype(int), P[:, c]) for c in range(N)]))


def cluster_boot_idx(groups, rng, n_boot):
    ug = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ug}
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        yield np.concatenate([members[g] for g in pick])


def main():
    t0 = time.time()
    s55 = _load_script("55_leadset_classification.py")
    data = load_processed(NPZ)
    signals = data["signal_12lead"].astype(np.float32)
    fs, n_t = int(data["sampling_rate"]), signals.shape[2]
    label, fold = data["label"].astype(int), data["fold"].astype(int)
    record_id = data["record_id"].astype(np.int64)
    db = pd.read_csv(s55.PATIENT_CSV, usecols=["ecg_id", "patient_id"]).set_index("ecg_id")
    patient = db.reindex(record_id)["patient_id"].to_numpy(dtype=np.int64)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals)
    idx = s55.split_indices(fold)
    train_mask = fold <= 8
    enc_main, dec = load_encoder_decoder(s55.CKPT, n_t, 3, s55.DEV)
    H_ind = (dec.A @ dec.B).detach().cpu().numpy()
    te = fold == 10
    y = label[te]

    probs, point, aux = {}, {}, {}
    th12, _ = infer_theta(enc_main, dec, scaled, device=s55.DEV)
    rec12 = s55.reconstruct_physical(dec, th12, scaler)
    frame = s55.build_c5(th12, rec12, H_ind, fs, train_mask)
    m, lo, hi, prob = s55.classify(frame, label, fold, patient)
    probs["12 leads"], point["12 leads"] = prob, (m["macro_auroc"], lo, hi)
    print(f"[W1V] 12 leads: {m['macro_auroc']:.4f} [{lo:.3f}, {hi:.3f}]  ({time.time()-t0:.0f}s)")
    for name in ("I+V1+V4", "II+V1+V5", "II"):
        leads = s55.SETS[name]
        enc, vn, eps = s55.train_leadset_encoder(scaled, idx, leads, dec)
        th = s55.encode_theta(enc, dec, scaled, leads)
        rec = s55.reconstruct_physical(dec, th, scaler)
        frame = s55.build_c5(th, rec, H_ind, fs, train_mask)
        m, lo, hi, prob = s55.classify(frame, label, fold, patient)
        probs[name], point[name], aux[name] = prob, (m["macro_auroc"], lo, hi), {"val_nrmse": vn, "epochs": eps}
        print(f"[W1V] {name}: {m['macro_auroc']:.4f} [{lo:.3f}, {hi:.3f}]  val NRMSE {vn:.4f} ({eps} epochs)  ({time.time()-t0:.0f}s)")

    rng = np.random.default_rng(20260909)
    boot = {k: [] for k in MODELS}
    kept = 0
    pid = patient[te]
    for ii in cluster_boot_idx(pid, rng, N_BOOT):
        yy = y[ii]
        if len(np.unique(yy)) < N:
            continue
        kept += 1
        for k in MODELS:
            boot[k].append(macro_auroc(yy, probs[k][ii]))
    B = {k: np.asarray(v) for k, v in boot.items()}
    rows = []
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            d = B[a] - B[b]
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            p = min(1.0, max(p, 1.0 / kept))
            rows.append({"model_a": a, "model_b": b, "auroc_a": point[a][0], "auroc_b": point[b][0],
                         "delta": point[a][0] - point[b][0], "ci_lo": float(np.percentile(d, 2.5)),
                         "ci_hi": float(np.percentile(d, 97.5)), "p_raw": float(p)})
    res = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    mm = len(res)
    ranked = np.arange(1, mm + 1)
    bh = res.p_raw.values * mm / ranked
    res["p_bh"] = np.minimum.accumulate(bh[::-1])[::-1].clip(0, 1)
    res["sig_bh_05"] = res.p_bh < 0.05
    res["n_boot_usable"] = kept
    res.to_csv(OUT / "W1V_leadset_paired_contrasts.csv", index=False, float_format="%.6g")
    head = {"models": {k: {"macro_auroc": point[k][0], "ci_lo": point[k][1], "ci_hi": point[k][2], **aux.get(k, {})} for k in MODELS},
            "n_boot": N_BOOT, "n_boot_usable": kept,
            "contrasts": res[["model_a", "model_b", "delta", "ci_lo", "ci_hi", "p_raw", "p_bh", "sig_bh_05"]].to_dict("records")}
    (OUT / "W1V_headline.json").write_text(json.dumps(head, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)), encoding="utf-8")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"[W1V] done ({time.time()-t0:.0f}s) -> {OUT}")


if __name__ == "__main__":
    main()
