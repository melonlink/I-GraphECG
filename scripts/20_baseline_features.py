"""B0: handcrafted ECG features + classical classifier (auto: xgboost / hist_gb / rf / logreg).

Run:
    python scripts/20_baseline_features.py --config configs/baseline_features.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igraphecg import repro   # output keys in configs are output-root relative

from igraphecg.data.dataset import load_processed, split_indices
from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.evaluation.metrics import bootstrap_ci, compute_metrics, per_class_metrics
from igraphecg.evaluation.plots import plot_confusion_matrix
from igraphecg.features.ecg_features import extract_features_batch, feature_names
from igraphecg.utils.config import parse_args_with_config
from igraphecg.utils.logging import get_logger
from igraphecg.utils.paths import PROJECT_ROOT, TABLES_DIR, ensure_dirs
from igraphecg.utils.seed import set_seed

log = get_logger("baseline_features")
N_CLASSES = len(TARGET_CLASSES)


def build_classifier(kind: str, cfg: dict, n_train: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    if kind == "auto":
        try:
            import xgboost   # noqa: F401
            kind = "xgboost"
        except ImportError:
            kind = "hist_gb"
    log.info(f"classifier = {kind}")
    if kind == "logreg":
        c = cfg.get("logreg", {})
        return kind, LogisticRegression(C=c.get("C", 1.0), max_iter=c.get("max_iter", 2000),
                                        class_weight="balanced", multi_class="multinomial")
    if kind == "rf":
        c = cfg.get("rf", {})
        return kind, RandomForestClassifier(n_estimators=c.get("n_estimators", 400),
                                            max_depth=c.get("max_depth"), class_weight="balanced",
                                            n_jobs=-1, random_state=cfg.get("seed", 42))
    if kind == "hist_gb":
        c = cfg.get("hist_gb", {})
        return kind, HistGradientBoostingClassifier(max_iter=c.get("max_iter", 400),
                                                    learning_rate=c.get("learning_rate", 0.05),
                                                    random_state=cfg.get("seed", 42))
    if kind == "xgboost":
        import xgboost as xgb
        c = cfg.get("xgboost", {})
        return kind, xgb.XGBClassifier(
            n_estimators=c.get("n_estimators", 600), max_depth=c.get("max_depth", 5),
            learning_rate=c.get("learning_rate", 0.05), subsample=c.get("subsample", 0.8),
            colsample_bytree=c.get("colsample_bytree", 0.8), objective="multi:softprob",
            num_class=N_CLASSES, tree_method="hist", eval_metric="mlogloss",
            random_state=cfg.get("seed", 42), n_jobs=-1)
    raise ValueError(f"unknown classifier {kind}")


def main():
    _, cfg = parse_args_with_config("Train handcrafted-feature baseline (B0)")
    ensure_dirs()
    set_seed(cfg.get("seed", 42))

    data = load_processed(PROJECT_ROOT / cfg["npz"])
    signals = data["signal_12lead"]   # [N,12,T] physical mV
    labels = data["label"].astype(int)
    fs = int(data["sampling_rate"])
    pre_ms = float(cfg.get("window_pre_ms", 300))
    idx = split_indices(data["fold"])
    log.info(f"N={len(labels)}  train/val/test = {len(idx['train'])}/{len(idx['val'])}/{len(idx['test'])}")

    log.info("extracting handcrafted features ...")
    X = extract_features_batch(signals, fs, pre_ms)   # [N,F]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    log.info(f"feature dimension F={X.shape[1]} ({len(feature_names())} names)")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X[idx["train"]])   # fitted on train only
    Xs = scaler.transform(X)

    kind, clf = build_classifier(cfg.get("classifier", "auto"), cfg, len(idx["train"]))
    clf.fit(Xs[idx["train"]], labels[idx["train"]])

    rows, percls_rows = [], []
    cm_test = None
    for split in ("val", "test"):
        ii = idx[split]
        prob = clf.predict_proba(Xs[ii])
        m = compute_metrics(labels[ii], prob, N_CLASSES)
        lo, hi = bootstrap_ci(labels[ii], prob, N_CLASSES, "macro_auroc")
        rows.append({"model": f"B0_{kind}", "split": split,
                     "macro_auroc": m["macro_auroc"], "macro_auprc": m["macro_auprc"],
                     "macro_f1": m["macro_f1"], "balanced_accuracy": m["balanced_accuracy"],
                     "auroc_ci_lo": lo, "auroc_ci_hi": hi, "n": len(ii)})
        for pc in per_class_metrics(labels[ii], prob, N_CLASSES, TARGET_CLASSES):
            percls_rows.append({"model": f"B0_{kind}", "split": split, **pc})
        log.info(f"[{split}] AUROC={m['macro_auroc']:.4f} F1={m['macro_f1']:.4f} "
                 f"bACC={m['balanced_accuracy']:.4f}  (95%CI AUROC {lo:.3f}-{hi:.3f})")
        if split == "test":
            cm_test = m["confusion_matrix"]

    out_metrics = repro.outputs() / cfg["out_metrics"]
    pd.DataFrame(rows).to_csv(out_metrics, index=False)
    pd.DataFrame(percls_rows).to_csv(
        str(out_metrics).replace(".csv", "_perclass.csv"), index=False)
    plot_confusion_matrix(cm_test, TARGET_CLASSES, repro.outputs() / cfg["out_cm"],
                          title=f"B0 {kind} (test)")
    log.info(f"saved metrics {out_metrics} and confusion matrix {cfg['out_cm']}")


if __name__ == "__main__":
    main()
