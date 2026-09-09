"""Revision classification analysis for descriptor groups C0--C6.

Models are fitted on folds 1--8 and evaluated once on fold 10. Confidence
intervals and paired group differences use patient-clustered bootstrap
resampling. All XGBoost predictions used for the summary, per-class metrics,
calibration, and confusion matrix are saved from the same fitted estimators.

Usage: python scripts/42_classification.py --config configs/v1fix/r3_s42.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igraphecg import repro   # output keys in configs are output-root relative

from igraphecg.data.dataset import load_processed
from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.evaluation.metrics import (bootstrap_ci, compute_metrics,
                                          paired_bootstrap_difference,
                                          per_class_metrics)
from igraphecg.evaluation.recon_metrics import compute_recon_metrics
from igraphecg.evaluation.round2_io import infer_theta, load_encoder_decoder
from igraphecg.utils.config import parse_args_with_config
from igraphecg.utils.logging import get_logger
from igraphecg.utils.paths import PROJECT_ROOT, ensure_dirs
from igraphecg.utils.seed import set_seed

log = get_logger("round3_cls")
N = len(TARGET_CLASSES)

THETA = [f"theta_{i}" for i in range(47)]
OLD = ["D_act", "D_rep_old_APDdisp", "S_ST_internal_L2", "mean_vent_APD", "vent_delta_spread"]
REP = ["APD_mean", "APD_disp", "APD_cv", "tau_rep_mean", "tau_rep_disp", "tau_rep_cv",
       "Tend_mean", "Tend_disp", "Tend_cv", "Sync_rep", "Tpeak_maxabs", "Tarea_mean",
       "Tabsarea_mean", "Tasym_mean", "Tlowamp_mean", "T_sign_discordance_rate"]
ST = ["ST_projected_L2", "ST_projected_max_abs", "ST_projected_anterior_leads",
      "ST_projected_lateral_leads", "ST_projected_inferior_leads", "ST_projected_sign_discordance"]
STAB = ["m_proxy", "V_rec"]
# C6 is a compact heuristic descriptor. It is not called an identifiable set.
IDENT = [f"theta_{i}" for i in list(range(0, 8)) + list(range(16, 24)) + list(range(32, 40)) + [46]] \
        + ["D_act", "T_sign_discordance_rate", "ST_projected_max_abs", "Tpeak_maxabs", "m_proxy", "Sync_rep"]

GROUPS = {"C0_theta": THETA, "C1_theta_old": THETA + OLD, "C2_theta_rep": THETA + REP,
          "C3_theta_st": THETA + ST, "C4_theta_stab": THETA + STAB,
          "C5_theta_rep_st_stab": THETA + REP + ST + STAB, "C6_compact_heuristic": IDENT}


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> tuple[float, float]:
    """Multiclass Brier score and top-label expected calibration error."""
    one_hot = np.eye(y_prob.shape[1], dtype=float)[y_true]
    brier = float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))
    confidence = y_prob.max(axis=1)
    correct = (y_prob.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return brier, float(ece)


def patient_groups(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Map records to official PTB-XL patients; fail on missing metadata."""
    # PTB-XL ships the patient mapping in ptbxl_database.csv. Accept either a path
    # relative to the project root or a bare filename to be found under the data
    # root, so the config need not encode where the database was unpacked.
    from igraphecg.utils.paths import find_ptbxl_root
    metadata = PROJECT_ROOT / cfg["patient_metadata"]
    if not metadata.exists():
        root = find_ptbxl_root()
        if root is not None and (root / Path(cfg["patient_metadata"]).name).exists():
            metadata = root / Path(cfg["patient_metadata"]).name
    if not metadata.exists():
        # Last resort: the two-column subset shipped with this archive. The paper states
        # that every classification result reproduces from the archive alone, without the
        # raw recordings -- and the patient-clustered bootstrap needs this mapping, so
        # without the fallback that claim fails on the first bootstrap.
        shipped = PROJECT_ROOT / "results" / "patient_map.csv"
        if shipped.exists():
            metadata = shipped
    if not metadata.exists():
        raise FileNotFoundError(
            f"patient metadata not found: {cfg['patient_metadata']}. Set "
            "ECG_DATA_DIR to the directory holding ptbxl_database.csv.")
    db = pd.read_csv(metadata, usecols=["ecg_id", "patient_id"]).set_index("ecg_id")
    patient = db.reindex(df["record_id"].to_numpy())["patient_id"]
    if patient.isna().any():
        missing = df.loc[patient.isna().to_numpy(), "record_id"].head().tolist()
        raise RuntimeError(f"missing patient identifiers for records {missing}")
    return patient.to_numpy(dtype=np.int64)


def make_clf(kind, seed):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="multinomial")
    import xgboost as xgb
    return xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, objective="multi:softprob", num_class=N,
                             tree_method="hist", eval_metric="mlogloss", random_state=seed, n_jobs=-1)


def main():
    _, cfg = parse_args_with_config("Round3 classification")
    ensure_dirs(); set_seed(cfg["seed"])
    out = repro.outputs() / cfg["out_dir"]
    # do not rely on 07 having run first: make our own output subdirectories
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(out / "tables" / "round3_features.csv")
    stab = pd.read_csv(out / "tables" / "stability_proxy_values.csv")
    # Row-order alignment: 41_stability.py writes its rows in the order of the
    # same round3_features.csv. Prefer the record_id key; fall back to the (fold,label)
    # sequence for older outputs that lack the column. This is what stops a re-sort on
    # either side from silently mismatching C4's m_proxy / V_rec columns.
    if len(stab) != len(df):
        raise RuntimeError(f"stability_proxy_values.csv has {len(stab)} rows but round3_features.csv "
                           f"has {len(df)}; rerun scripts/41_stability.py")
    if "record_id" in stab.columns and "record_id" in df.columns:
        aligned = (stab["record_id"].to_numpy() == df["record_id"].to_numpy()).all()
        key = "record_id"
    else:
        aligned = ((stab["fold"].to_numpy() == df["fold"].to_numpy()).all()
                   and (stab["label"].to_numpy() == df["label"].to_numpy()).all())
        key = "(fold,label) sequence — weaker check; rerun 11 to emit record_id"
    if not aligned:
        raise RuntimeError(f"stability_proxy_values.csv is not row-aligned with round3_features.csv "
                           f"[checked on {key}]; rerun scripts/41_stability.py")
    df["m_proxy"] = stab["m_proxy"].values; df["V_rec"] = stab["V_rec"].values
    label = df["label"].values.astype(int); fold = df["fold"].values.astype(int)
    patient = patient_groups(df, cfg)
    tr, te = fold <= 8, fold == 10

    from sklearn.preprocessing import StandardScaler
    rows, percls, predictions = [], [], {}
    for gname, cols in GROUPS.items():
        cols = [c for c in cols if c in df.columns]
        X = np.nan_to_num(df[cols].values.astype(np.float32))
        sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
        for kind in ("logreg", "xgboost"):
            try:
                clf = make_clf(kind, cfg["seed"])
            except ImportError:
                continue
            clf.fit(Xs[tr], label[tr]); prob = clf.predict_proba(Xs[te])
            m = compute_metrics(label[te], prob, N)
            lo, hi = bootstrap_ci(label[te], prob, N, "macro_auroc", groups=patient[te])
            brier, ece = calibration_metrics(label[te], prob)
            rows.append({"feature_group": gname, "n_features": len(cols), "classifier": kind,
                         "macro_auroc": m["macro_auroc"], "macro_auprc": m["macro_auprc"],
                         "macro_f1": m["macro_f1"], "balanced_accuracy": m["balanced_accuracy"],
                         "multiclass_brier": brier, "toplabel_ece": ece,
                         "auroc_ci_lo": lo, "auroc_ci_hi": hi,
                         "bootstrap_unit": "patient", "n_boot": 1000})
            if kind == "xgboost":
                predictions[gname] = prob
                for pc in per_class_metrics(label[te], prob, N, TARGET_CLASSES):
                    percls.append({"feature_group": gname, **pc})
            log.info(f"[{gname}/{kind}] AUROC={m['macro_auroc']:.4f} F1={m['macro_f1']:.4f} bACC={m['balanced_accuracy']:.4f}")

    cdf = pd.DataFrame(rows)
    cdf.to_csv(out / "tables" / "round3_classification_summary.csv", index=False)
    pd.DataFrame(percls).to_csv(out / "tables" / "round3_classification_perclass.csv", index=False)
    pd.DataFrame({"feature": IDENT}).to_csv(out / "tables" / "compact_heuristic_features.csv", index=False)

    pred_rows = []
    test_df = df.loc[te, ["record_id", "label", "label_name", "fold"]].reset_index(drop=True)
    test_patient = patient[te]
    for group_name, prob in predictions.items():
        for index, row in test_df.iterrows():
            pred_rows.append({"feature_group": group_name, "record_id": int(row.record_id),
                              "patient_id": int(test_patient[index]), "label": int(row.label),
                              "label_name": row.label_name, "predicted_label": int(prob[index].argmax()),
                              **{f"prob_{TARGET_CLASSES[c]}": float(prob[index, c]) for c in range(N)}})
    pd.DataFrame(pred_rows).to_csv(out / "tables" / "round3_xgboost_test_predictions.csv", index=False)

    paired_rows = []
    for metric in ("macro_auroc", "macro_f1", "balanced_accuracy"):
        point, lo, hi = paired_bootstrap_difference(
            label[te], predictions["C5_theta_rep_st_stab"], predictions["C0_theta"], N,
            metric=metric, groups=patient[te])
        paired_rows.append({"comparison": "C5_minus_C0", "metric": metric,
                            "difference": point, "ci_lo": lo, "ci_hi": hi,
                            "bootstrap_unit": "patient", "n_boot": 1000})
    pd.DataFrame(paired_rows).to_csv(out / "tables" / "paired_group_differences.csv", index=False)

    c5_metrics = compute_metrics(label[te], predictions["C5_theta_rep_st_stab"], N)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ConfusionMatrixDisplay(c5_metrics["confusion_matrix"], display_labels=TARGET_CLASSES).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("C5 XGBoost, PTB-XL fold 10")
    fig.tight_layout()
    fig.savefig(out / "figures" / "c5_xgboost_confusion_matrix.png", dpi=300)
    plt.close(fig)

    # ---- Round 2 reproduction: recompute test NRMSE/corr + C0 AUROC ----
    # This block reconstructs waveforms and so needs the median-beat npz. Everything the
    # paper claims reproduces "from the archive alone" -- classification, identifiability,
    # lead selection -- has already been written above from the derived tables. Skip with
    # a clear message rather than crashing, so nobody running from the archive gets a
    # traceback after the results they came for are already on disk.
    npz_path = PROJECT_ROOT / cfg["npz"]
    if not npz_path.exists():
        log.warning(f"skipping the Round-2 reconstruction re-check: {cfg['npz']} not found. "
                    "It needs the median-beat dataset (built by scripts/10_prepare_ptbxl.py); "
                    "the classification results above are unaffected.")
        log.info(f"Task G done -> {out}")
        return
    import torch
    data = load_processed(npz_path)
    signals = data["signal_12lead"].astype(np.float32)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals)
    enc, dec = load_encoder_decoder(repro.outputs() / cfg["encoder_ckpt"], signals.shape[2],
                                    cfg["leadfield_rank"], "cuda" if torch.cuda.is_available() else "cpu")
    theta_all, _ = infer_theta(enc, dec, scaled, device="cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        yh = []
        for s in range(0, len(theta_all), 1024):
            y12, _ = dec.forward(torch.from_numpy(theta_all[s:s+1024].astype(np.float32)).to(dev))
            yh.append(y12.cpu().numpy())
        yh = np.concatenate(yh)
    t_sec = dec.t.cpu().numpy()
    mrec = compute_recon_metrics(scaled[te], yh[te], t_sec)
    c0_auroc = cdf[(cdf.feature_group == "C0_theta") & (cdf.classifier == "xgboost")]["macro_auroc"].values[0]
    rep = pd.DataFrame([{"reproduced_test_nrmse": mrec["median_nrmse"], "reproduced_test_corr": mrec["median_corr"],
                         "reproduced_theta_macro_auroc": float(c0_auroc)}])
    rep.to_csv(out / "tables" / "round2_reproduction_metrics.csv", index=False)
    log.info(f"Round-2 re-check: NRMSE={mrec['median_nrmse']:.4f} corr={mrec['median_corr']:.4f} C0-AUROC={c0_auroc:.4f}")
    best = cdf.loc[cdf.macro_auroc.idxmax()]
    log.info(f"best feature group: {best['feature_group']}/{best['classifier']} AUROC={best['macro_auroc']:.4f}")
    log.info(f"Task G done -> {out}")


if __name__ == "__main__":
    main()
