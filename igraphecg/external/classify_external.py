"""External classification: PTB-XL-trained parameter-only (C0, theta-47) classifier applied
to external theta dumps. Inference/transfer only — the classifier is trained ONCE on PTB-XL
train fold (seed-42), never on external data.

C0 (theta only) is used for cross-database transfer because it depends solely on the inferred
equivalent parameters (no signal-domain T-wave features whose scale may differ across devices).

Run (from repo root):
  <py> -m igraphecg.external.classify_external --theta results/ext_georgia.theta.npz \
       --dataset georgia --out results/cls_georgia.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from ..config import SUPERCLASSES
from .. import repro

# The published feature table. This used to default to round3_eikonal -- a superseded
# lineage -- and was corrected only by a driver's monkey patch.
PTBXL_FEATURES = repro.runs() / "v1fix_r3_s42" / "tables" / "round3_features.csv"
THETA_COLS = [f"theta_{i}" for i in range(47)]


def _train_ptbxl_c0(seed=42):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    df = pd.read_csv(PTBXL_FEATURES)
    tr = df["fold"].values <= 8
    X = np.nan_to_num(df[THETA_COLS].values.astype(np.float32)); y = df["label"].values.astype(int)
    sc = StandardScaler().fit(X[tr])
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, objective="multi:softprob", num_class=4,
                            tree_method="hist", eval_metric="mlogloss", random_state=seed, n_jobs=-1)
    clf.fit(sc.transform(X[tr]), y[tr])
    return sc, clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix

    d = np.load(a.theta)
    theta, y = d["theta"].astype(np.float32), d["labels"].astype(int)
    sc, clf = _train_ptbxl_c0()
    prob = clf.predict_proba(sc.transform(np.nan_to_num(theta)))
    pred = prob.argmax(1)

    present = sorted(set(y.tolist()))
    res = {"dataset": a.dataset, "classifier": "C0 (theta-47, PTB-XL-trained, seed-42)",
           "n": int(len(y)), "classes_present": [SUPERCLASSES[c] for c in present], "per_class": {}}
    p, r, f1, sup = precision_recall_fscore_support(y, pred, labels=list(range(4)), zero_division=0)
    for c in range(4):
        name = SUPERCLASSES[c]
        entry = {"support": int(sup[c]), "precision": round(float(p[c]), 4),
                 "recall": round(float(r[c]), 4), "f1": round(float(f1[c]), 4)}
        yb = (y == c).astype(int)
        if yb.sum() > 0 and yb.sum() < len(yb):
            entry["auroc"] = round(float(roc_auc_score(yb, prob[:, c])), 4)
            entry["auprc"] = round(float(average_precision_score(yb, prob[:, c])), 4)
        else:
            entry["auroc"] = None; entry["auprc"] = None
            entry["note"] = "class absent in this dataset" if yb.sum() == 0 else ""
        res["per_class"][name] = entry
    # macro over present classes with positives
    aurocs = [v["auroc"] for v in res["per_class"].values() if v["auroc"] is not None]
    res["macro_auroc_present"] = round(float(np.mean(aurocs)), 4) if aurocs else None
    res["balanced_acc_present"] = round(float(np.mean([res["per_class"][SUPERCLASSES[c]]["recall"]
                                                       for c in present])), 4)
    res["confusion_matrix"] = confusion_matrix(y, pred, labels=list(range(4))).tolist()
    res["confusion_labels"] = SUPERCLASSES
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
