"""W1F step 3: fold-10 disposition of the multi-label pool and the W1F headline JSON.

Reads   data/processed/ptbxl_medianbeat_multilabel_100hz_metadata.csv  (11_prepare_ptbxl_multilabel)
        runs/v7rev_stats/W1F_multilabel_recon.csv, W1F_multilabel_auroc.csv,
        W1F_multilabel_recon_delta.csv                                   (70_multilabel_eval)
Writes  runs/v7rev_stats/W1F_multilabel_fold10_disposition.csv
        runs/v7rev_stats/W1F_multilabel_headline.json
Run after 70_multilabel_eval.py and before W1F_multilabel_numeric_provenance.py.
"""
from __future__ import annotations

import json

import pandas as pd

import sys as _sys
from pathlib import Path as _Path
# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro

OUT = repro.outdir("runs/v7rev_stats")

meta = pd.read_csv(repro.processed() / "ptbxl_medianbeat_multilabel_100hz_metadata.csv")
te = meta[meta.fold == 10]
print("fold10 n =", len(te))
print("cardinality:", te.n_superclass.value_counts().sort_index().to_dict())
print("clean:", int(te.is_clean.sum()), " excluded:", int((~te.is_clean).sum()))
sub = te[(te.n_superclass == 1) & (~te.is_clean)]
print("single-superclass but NOT clean:", len(sub))
print("  of which HYP-only:", int(sub.y_HYP.sum()))
print("  label mix:", {c: int(sub['y_' + c].sum()) for c in ["NORM", "MI", "STTC", "CD", "HYP"]})
print("patients fold10:", te.patient_id.nunique())
print("\nwhole pool cardinality:", meta.n_superclass.value_counts().sort_index().to_dict())
print("pool patients:", meta.patient_id.nunique())

disp = pd.DataFrame([
    {"stratum": "fold10_total_ge1_superclass", "n": len(te)},
    {"stratum": "fold10_cardinality_1", "n": int((te.n_superclass == 1).sum())},
    {"stratum": "fold10_cardinality_2", "n": int((te.n_superclass == 2).sum())},
    {"stratum": "fold10_cardinality_3", "n": int((te.n_superclass == 3).sum())},
    {"stratum": "fold10_cardinality_4", "n": int((te.n_superclass == 4).sum())},
    {"stratum": "fold10_clean_single_label", "n": int(te.is_clean.sum())},
    {"stratum": "fold10_excluded_by_clean_filter", "n": int((~te.is_clean).sum())},
    {"stratum": "fold10_single_superclass_but_excluded", "n": len(sub)},
    {"stratum": "fold10_unique_patients", "n": int(te.patient_id.nunique())},
])
disp.to_csv(OUT / "W1F_multilabel_fold10_disposition.csv", index=False)

r = pd.read_csv(OUT / "W1F_multilabel_recon.csv").set_index("subset")
a = pd.read_csv(OUT / "W1F_multilabel_auroc.csv")
dl = pd.read_csv(OUT / "W1F_multilabel_recon_delta.csv")


def rrow(s):
    x = r.loc[s]
    return {"n": int(x.n), "median_nrmse_scaled": round(float(x.median_nrmse_scaled), 4),
            "median_corr": round(float(x.median_corr), 4),
            "nrmse_iqr": [round(float(x.nrmse_q25_scaled), 4), round(float(x.nrmse_q75_scaled), 4)],
            "corr_iqr": [round(float(x.corr_q25), 4), round(float(x.corr_q75), 4)]}


def arows(g, k):
    s = a[(a.feature_group == g) & (a.classifier == k)]
    return {row.label: {"auroc": round(row.auroc, 4),
                        "ci": [round(row.auroc_ci_lo, 4), round(row.auroc_ci_hi, 4)],
                        "auprc": round(row.auprc, 4),
                        "auprc_ci": [round(row.auprc_ci_lo, 4), round(row.auprc_ci_hi, 4)],
                        "n_pos": int(row.n_pos)} for row in s.itertuples()}


head = {
    "task": "W1F multi-label PTB-XL, encoder frozen (zero-shot), seed-42 d1s_r3_s42",
    "pool": {"total_ptbxl": 21799, "ge1_superclass": 21388, "processed": len(meta),
             "dropped_by_beat_quality_gate": 21388 - len(meta),
             "clean_single_label_subset": int(meta.is_clean.sum()),
             "split_train_val_test": [int((meta.fold <= 8).sum()), int((meta.fold == 9).sum()),
                                      int((meta.fold == 10).sum())]},
    "fold10_disposition": disp.set_index("stratum")["n"].to_dict(),
    "recon": {"a_all_ge1_superclass": rrow("fold10_all_ge1_superclass"),
              "b_multilabel_gt1": rrow("fold10_multilabel_gt1"),
              "single_superclass_reference": rrow("fold10_single_superclass"),
              "clean_subset_CONTROL": rrow("fold10_clean_subset"),
              "excluded_by_clean_filter": rrow("fold10_excluded_by_clean_filter"),
              "cardinality_2": rrow("fold10_card2"), "cardinality_3plus": rrow("fold10_card3plus")},
    "recon_degradation": dl.to_dict(orient="records"),
    "auroc": {"C0_theta_xgboost": arows("C0_theta", "xgboost"),
              "C0_theta_logreg": arows("C0_theta", "logreg"),
              "C5_xgboost": arows("C5_theta_rep_st_stab", "xgboost"),
              "C5_logreg": arows("C5_theta_rep_st_stab", "logreg")},
    "notes": ["encoder + decoder frozen; only the one-vs-rest downstream classifier is fitted",
              "frozen PTB-XL training robust scaler reused verbatim, never refit",
              "split = official strat_fold 1-8 train / 9 val / 10 test",
              "bootstrap = 1000 patient-clustered resamples, percentile 95% CI"],
}
(OUT / "W1F_multilabel_headline.json").write_text(json.dumps(head, indent=2), encoding="utf-8")
print("\nwrote", OUT / "W1F_multilabel_headline.json")
