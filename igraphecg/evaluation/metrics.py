"""Classification metrics: macro AUROC / AUPRC / F1 / balanced accuracy, per-class sensitivity /
specificity, confusion matrix and bootstrap 95% CI, all from predicted probabilities and labels.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    oh = np.zeros((len(y), n_classes), dtype=int)
    oh[np.arange(len(y)), y] = 1
    return oh


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> dict:
    """Return the core metrics as a dict. y_prob: [N, n_classes]."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = y_prob.argmax(axis=1)
    y_oh = _one_hot(y_true, n_classes)

    present = [c for c in range(n_classes) if (y_true == c).sum() > 0]
    try:
        macro_auroc = roc_auc_score(y_oh[:, present], y_prob[:, present], average="macro", multi_class="ovr")
    except ValueError:
        macro_auroc = float("nan")
    try:
        macro_auprc = average_precision_score(y_oh[:, present], y_prob[:, present], average="macro")
    except ValueError:
        macro_auprc = float("nan")

    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=list(range(n_classes)), zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    return {
        "macro_auroc": float(macro_auroc),
        "macro_auprc": float(macro_auprc),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(bal_acc),
        "confusion_matrix": cm,
    }


def per_class_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int,
                      class_names: list[str]) -> list[dict]:
    """Per-class AUROC / AUPRC / F1 / sensitivity / specificity."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = y_prob.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    out = []
    total = cm.sum()
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        bin_true = (y_true == c).astype(int)
        try:
            auroc = roc_auc_score(bin_true, y_prob[:, c]) if bin_true.sum() > 0 else float("nan")
        except ValueError:
            auroc = float("nan")
        try:
            auprc = average_precision_score(bin_true, y_prob[:, c]) if bin_true.sum() > 0 else float("nan")
        except ValueError:
            auprc = float("nan")
        f1 = f1_score(bin_true, (y_pred == c).astype(int), zero_division=0)
        out.append({
            "class": class_names[c],
            "auroc": float(auroc), "auprc": float(auprc), "f1": float(f1),
            "sensitivity": float(sens), "specificity": float(spec),
            "support": int((y_true == c).sum()),
        })
    return out


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int,
                 metric: str = "macro_auroc", n_boot: int = 1000, seed: int = 42,
                 groups: np.ndarray | None = None) -> tuple[float, float]:
    """Return a percentile 95% bootstrap CI.

    When ``groups`` is supplied, complete groups (for example patients) are
    resampled with replacement. This preserves within-patient clustering.
    """
    import warnings

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    n = len(y_true)
    groups = None if groups is None else np.asarray(groups)
    if groups is not None and len(groups) != n:
        raise ValueError("groups must have the same length as y_true")
    unique_groups = np.unique(groups) if groups is not None else None
    vals = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # resampling may drop a class; the warnings are noise
        for _ in range(n_boot):
            if unique_groups is None:
                idx = rng.integers(0, n, n)
            else:
                sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
                idx = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
            try:
                m = compute_metrics(y_true[idx], y_prob[idx], n_classes)[metric]
                if not np.isnan(m):
                    vals.append(m)
            except (ValueError, KeyError):
                continue
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def paired_bootstrap_difference(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    n_classes: int,
    metric: str = "macro_auroc",
    n_boot: int = 1000,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Estimate ``metric(A) - metric(B)`` and its paired bootstrap CI."""
    import warnings

    y_true = np.asarray(y_true).astype(int)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)
    if len(y_prob_a) != len(y_true) or len(y_prob_b) != len(y_true):
        raise ValueError("probability arrays must align with y_true")
    groups = None if groups is None else np.asarray(groups)
    if groups is not None and len(groups) != len(y_true):
        raise ValueError("groups must have the same length as y_true")

    point = (compute_metrics(y_true, y_prob_a, n_classes)[metric]
             - compute_metrics(y_true, y_prob_b, n_classes)[metric])
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups) if groups is not None else None
    diffs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_boot):
            if unique_groups is None:
                idx = rng.integers(0, len(y_true), len(y_true))
            else:
                sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
                idx = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
            try:
                a = compute_metrics(y_true[idx], y_prob_a[idx], n_classes)[metric]
                b = compute_metrics(y_true[idx], y_prob_b[idx], n_classes)[metric]
                if np.isfinite(a) and np.isfinite(b):
                    diffs.append(a - b)
            except (ValueError, KeyError):
                continue
    if not diffs:
        return float(point), float("nan"), float("nan")
    return float(point), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
