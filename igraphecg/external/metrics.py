"""Evaluation metrics (re-exported / thin wrappers over the verified src code)."""
from __future__ import annotations
import numpy as np


def reconstruction_metrics(real: np.ndarray, recon: np.ndarray) -> dict:
    """real, recon: [N,12,T] (scaled). Uses the paper's definitions
    (src.evaluation.recon_metrics): median over records of the per-record mean per-lead
    Pearson r, and median per-record whole-beat NRMSE."""
    from igraphecg.evaluation.recon_metrics import per_sample_lead_corr, per_sample_nrmse
    cr = np.asarray(per_sample_lead_corr(real, recon), dtype=float)
    nr = np.asarray(per_sample_nrmse(real, recon), dtype=float)
    cr = cr[np.isfinite(cr)]; nr = nr[np.isfinite(nr)]   # drop non-finite (flat/missing leads)
    return {"median_corr": float(np.median(cr)) if cr.size else float("nan"),
            "scaled_median_nrmse": float(np.median(nr)) if nr.size else float("nan"),
            "n_finite_corr": int(cr.size)}


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, n_classes: int = 4) -> dict:
    """Wrap src.evaluation.metrics (macro AUROC etc.) + per-class precision/recall/F1."""
    from igraphecg.evaluation.metrics import compute_metrics
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    m = compute_metrics(y_true, prob, n_classes)
    pred = prob.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, labels=list(range(n_classes)),
                                                  zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=list(range(n_classes)))
    recall_by_class = (cm.diagonal() / cm.sum(1).clip(min=1)).tolist()
    m["per_class_precision"] = p.tolist()
    m["per_class_recall"] = r.tolist()
    m["per_class_f1"] = f1.tolist()
    m["balanced_accuracy"] = float(np.mean(recall_by_class))
    m["confusion_matrix"] = cm.tolist()
    return m
