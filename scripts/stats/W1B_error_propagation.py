"""W1B -- does reconstruction error propagate to descriptors and classification?

TEST 1  descriptor-substitution oracle (C5 vs C5-obs and matched descriptor-only heads)
TEST 2  stratification of fold-10 by scaled reconstruction NRMSE, pooled and within class

Protocol is copied from scripts/42_classification.py:
  features   = <run>/tables/round3_features.csv  +  m_proxy/V_rec from stability_proxy_values.csv
  scaler     = StandardScaler fit on fold <= 8
  classifier = XGBClassifier(400, depth 4, lr 0.05, subsample .8, colsample .8,
               multi:softprob, num_class 4, hist, mlogloss, random_state=seed, n_jobs=-1)
  train fold <= 8, test fold == 10, patient-clustered bootstrap (1000 draws, seed 42)

Nothing under runs/v1fix_*, checkpoints/ or FINAL_Sensors/ is written to.

Usage:
  python W1B_error_propagation.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sys as _sys
from pathlib import Path as _Path
# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro
REPO = repro.repo()
sys.path.insert(0, str(REPO))

from igraphecg.data.label_utils import TARGET_CLASSES  # noqa: E402
from igraphecg.evaluation.metrics import (compute_metrics, paired_bootstrap_difference,  # noqa: E402
                                          per_class_metrics)

RUNS = repro.runs()
OUT = RUNS / "v7rev_stats"
def _patient_csv():
    """PTB-XL patient map. This used to read a copy kept under Revision_v2/Verification;
    that copy was byte-identical to ptbxl_database.csv under the data root and went with
    the Revision v2-v4 clean-up, so the data root is read directly."""
    from igraphecg.utils.paths import find_ptbxl_root
    root = find_ptbxl_root()
    if root is None or not (root / "ptbxl_database.csv").exists():
        raise FileNotFoundError(
            "PTB-XL patient metadata (ptbxl_database.csv) not found under the data root. "
            "Set ECG_DATA_DIR to the directory holding it.")
    return root / "ptbxl_database.csv"


PATIENT_CSV = _patient_csv()
SEEDS = [42, 1, 2, 3, 4]
N = len(TARGET_CLASSES)          # 4: NORM MI STTC CD

# ---- feature blocks, verbatim from scripts/42_classification.py -------------------
THETA = [f"theta_{i}" for i in range(47)]
REP = ["APD_mean", "APD_disp", "APD_cv", "tau_rep_mean", "tau_rep_disp", "tau_rep_cv",
       "Tend_mean", "Tend_disp", "Tend_cv", "Sync_rep", "Tpeak_maxabs", "Tarea_mean",
       "Tabsarea_mean", "Tasym_mean", "Tlowamp_mean", "T_sign_discordance_rate"]
ST = ["ST_projected_L2", "ST_projected_max_abs", "ST_projected_anterior_leads",
      "ST_projected_lateral_leads", "ST_projected_inferior_leads", "ST_projected_sign_discordance"]
STAB = ["m_proxy", "V_rec"]
C5 = THETA + REP + ST + STAB

# The 37 descriptors that feature_provenance.csv marks `reconstructed_ecg`; each has an
# `obs_` twin computed by the identical routine on the OBSERVED median beat.
FAMILIES = ["Tpeak", "Tptime", "Tarea", "Tabsarea", "Twidth", "Tasym", "Tlowamp",
            "STmean", "STslope"]
STATS = ["mean", "std", "maxabs", "range"]
REC37 = [f"{f}_{s}" for f in FAMILIES for s in STATS] + ["T_sign_discordance_rate"]
OBS37 = ["obs_" + c for c in REC37]
# recon-domain descriptors that actually appear in C5 (the only ones the oracle can swap)
C5_SWAPPABLE = [c for c in C5 if c in REC37]


def load_seed(seed: int) -> pd.DataFrame:
    run = RUNS / f"v1fix_r3_s{seed}"
    df = pd.read_csv(run / "tables" / "round3_features.csv")
    stab = pd.read_csv(run / "tables" / "stability_proxy_values.csv")
    if len(stab) != len(df):
        raise RuntimeError(f"seed {seed}: stability rows {len(stab)} != feature rows {len(df)}")
    if not (stab["record_id"].to_numpy() == df["record_id"].to_numpy()).all():
        raise RuntimeError(f"seed {seed}: stability_proxy_values.csv not row-aligned")
    df["m_proxy"] = stab["m_proxy"].values
    df["V_rec"] = stab["V_rec"].values
    return df


def patient_groups(df: pd.DataFrame) -> np.ndarray:
    db = pd.read_csv(PATIENT_CSV, usecols=["ecg_id", "patient_id"]).set_index("ecg_id")
    pid = db.reindex(df["record_id"].to_numpy())["patient_id"]
    if pid.isna().any():
        raise RuntimeError("missing patient identifiers")
    return pid.to_numpy(dtype=np.int64)


def fit_predict(df: pd.DataFrame, cols: list[str], tr: np.ndarray, te: np.ndarray,
                label: np.ndarray, seed: int) -> np.ndarray:
    """Exact 42_classification.py fit path for one feature list."""
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    cols = [c for c in cols if c in df.columns]
    X = np.nan_to_num(df[cols].values.astype(np.float32))
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X)
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, objective="multi:softprob", num_class=N,
                            tree_method="hist", eval_metric="mlogloss", random_state=seed,
                            n_jobs=-1)
    clf.fit(Xs[tr], label[tr])
    return clf.predict_proba(Xs[te])


# ============================== TEST 1 =====================================================
def test1() -> tuple[pd.DataFrame, dict]:
    variants = {
        # name                     : (feature list, note)
        "C5":        (C5, "locked submitted model (47 theta + 16 rep + 6 ST_projected + 2 stab)"),
        "C5_obs":    (["obs_" + c if c in REC37 else c for c in C5],
                      "oracle: the 6 reconstruction-ECG descriptors in C5 swapped for obs_ twins"),
        "C5_drop":   ([c for c in C5 if c not in REC37],
                      "the same 6 descriptors deleted (no replacement)"),
        "REC37_only": (REC37, "descriptor-only head, 37 reconstruction-domain descriptors"),
        "OBS37_only": (OBS37, "descriptor-only head, the 37 observation-domain twins"),
        "C5_plus_OBS37": (C5 + [c for c in OBS37 if c not in C5],
                          "supplementary, NOT dimension-matched: C5 augmented with all 37 obs twins"),
    }
    contrasts = [("C5_obs", "C5"), ("C5_drop", "C5"), ("OBS37_only", "REC37_only"),
                 ("C5_plus_OBS37", "C5")]

    rows, controls = [], {}
    for seed in SEEDS:
        df = load_seed(seed)
        label = df["label"].values.astype(int)
        fold = df["fold"].values.astype(int)
        pat = patient_groups(df)
        tr, te = fold <= 8, fold == 10
        probs = {}
        for name, (cols, note) in variants.items():
            present = [c for c in cols if c in df.columns]
            if len(present) != len(cols):
                raise RuntimeError(f"{name}: missing {set(cols) - set(present)}")
            prob = fit_predict(df, cols, tr, te, label, seed)
            probs[name] = prob
            m = compute_metrics(label[te], prob, N)
            pc = per_class_metrics(label[te], prob, N, TARGET_CLASSES)
            rec = {p["class"]: p["sensitivity"] for p in pc}
            auc = {p["class"]: p["auroc"] for p in pc}
            rows.append({"test": "T1_variant", "seed": seed, "variant": name,
                         "n_features": len(cols), "note": note,
                         "macro_auroc": m["macro_auroc"], "macro_auprc": m["macro_auprc"],
                         "macro_f1": m["macro_f1"], "balanced_accuracy": m["balanced_accuracy"],
                         **{f"recall_{c}": rec[c] for c in TARGET_CLASSES},
                         **{f"auroc_{c}": auc[c] for c in TARGET_CLASSES},
                         "MI_to_NORM": int(m["confusion_matrix"][1, 0]),
                         "MI_support": int((label[te] == 1).sum())})
        if seed == 42:
            m5 = compute_metrics(label[te], probs["C5"], N)
            controls = {"C5_macro_auroc": m5["macro_auroc"],
                        "C5_balanced_accuracy": m5["balanced_accuracy"],
                        "C5_MI_to_NORM": int(m5["confusion_matrix"][1, 0]),
                        "C5_recall": {p["class"]: p["sensitivity"]
                                      for p in per_class_metrics(label[te], probs["C5"], N,
                                                                 TARGET_CLASSES)}}
        for a, b in contrasts:
            for metric in ("macro_auroc", "balanced_accuracy", "macro_f1"):
                pt, lo, hi = paired_bootstrap_difference(label[te], probs[a], probs[b], N,
                                                         metric=metric, groups=pat[te],
                                                         n_boot=1000, seed=42)
                rows.append({"test": "T1_paired_delta", "seed": seed,
                             "variant": f"{a}_minus_{b}", "metric": metric,
                             "difference": pt, "ci_lo": lo, "ci_hi": hi,
                             "crosses_zero": bool(lo <= 0.0 <= hi),
                             "bootstrap_unit": "patient", "n_boot": 1000})
        print(f"  [T1] seed {seed} done", flush=True)
    return pd.DataFrame(rows), controls


# ============================== TEST 2 =====================================================
def recon_table(seed: int) -> pd.DataFrame:
    f = OUT / f"W1A_per_record_recon_s{seed}.csv"
    r = pd.read_csv(f, usecols=["record_id", "label", "label_name", "nrmse_scaled_12lead",
                                "corr_meanlead_12lead"])
    return r


def test2(t1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    pooled_rows, within_rows, sp_rows = [], [], []
    class_assoc = {}
    for seed in SEEDS:
        df = load_seed(seed)
        label = df["label"].values.astype(int)
        fold = df["fold"].values.astype(int)
        pat = patient_groups(df)
        tr, te = fold <= 8, fold == 10
        prob = fit_predict(df, C5, tr, te, label, seed)
        test = df.loc[te, ["record_id", "label", "label_name"]].reset_index(drop=True)
        test["patient_id"] = pat[te]
        for c in range(N):
            test[f"prob_{TARGET_CLASSES[c]}"] = prob[:, c]
        test["pred"] = prob.argmax(axis=1)
        test["p_true"] = prob[np.arange(len(test)), test["label"].values]
        rec = recon_table(seed)
        merged = test.merge(rec[["record_id", "nrmse_scaled_12lead", "corr_meanlead_12lead"]],
                            on="record_id", how="left", validate="one_to_one")
        if merged["nrmse_scaled_12lead"].isna().any():
            raise RuntimeError(f"seed {seed}: unmatched records in W1A recon table")

        y = merged["label"].values
        p = merged[[f"prob_{c}" for c in TARGET_CLASSES]].values

        # ---------- pooled quartiles ----------
        q = pd.qcut(merged["nrmse_scaled_12lead"], 4, labels=[1, 2, 3, 4]).astype(int).values
        edges = np.quantile(merged["nrmse_scaled_12lead"], [0, .25, .5, .75, 1.0])
        for k in (1, 2, 3, 4):
            m = q == k
            mm = compute_metrics(y[m], p[m], N)
            pc = per_class_metrics(y[m], p[m], N, TARGET_CLASSES)
            row = {"seed": seed, "quartile": k, "n": int(m.sum()),
                   "nrmse_lo": float(edges[k - 1]), "nrmse_hi": float(edges[k]),
                   "nrmse_median": float(np.median(merged["nrmse_scaled_12lead"][m])),
                   "corr_median": float(np.median(merged["corr_meanlead_12lead"][m])),
                   "macro_auroc": mm["macro_auroc"],
                   "balanced_accuracy": mm["balanced_accuracy"],
                   "accuracy": float((p.argmax(1)[m] == y[m]).mean()),
                   "mean_p_true": float(merged["p_true"][m].mean())}
            for pcc in pc:
                row[f"recall_{pcc['class']}"] = pcc["sensitivity"]
                row[f"n_{pcc['class']}"] = pcc["support"]
            for cname, ci in zip(TARGET_CLASSES, range(N)):
                row[f"frac_{cname}"] = float((y[m] == ci).mean())
            pooled_rows.append(row)

        # ---------- within-class quartiles ----------
        for ci, cname in enumerate(TARGET_CLASSES):
            sub = merged[merged["label"] == ci].copy()
            qq = pd.qcut(sub["nrmse_scaled_12lead"], 4, labels=[1, 2, 3, 4]).astype(int).values
            ed = np.quantile(sub["nrmse_scaled_12lead"], [0, .25, .5, .75, 1.0])
            for k in (1, 2, 3, 4):
                s = sub[qq == k]
                within_rows.append({
                    "seed": seed, "class": cname, "quartile": k, "n": int(len(s)),
                    "nrmse_lo": float(ed[k - 1]), "nrmse_hi": float(ed[k]),
                    "nrmse_median": float(s["nrmse_scaled_12lead"].median()),
                    "corr_median": float(s["corr_meanlead_12lead"].median()),
                    "recall": float((s["pred"].values == ci).mean()),
                    "mean_p_true": float(s["p_true"].mean()),
                    "median_p_true": float(s["p_true"].median()),
                    "frac_pred_NORM": float((s["pred"].values == 0).mean())})
            # ---------- Spearman recon-error vs P(true class) ----------
            for err_name, err in (("nrmse_scaled_12lead", sub["nrmse_scaled_12lead"].values),
                                  ("corr_meanlead_12lead", sub["corr_meanlead_12lead"].values)):
                rho, pv = spearmanr(err, sub["p_true"].values)
                sp_rows.append({"seed": seed, "class": cname, "error_metric": err_name,
                                "n": int(len(sub)), "spearman_rho": float(rho),
                                "p_value": float(pv)})
        # ---------- class-level association (the honest confound check) ----------
        med_corr, rec_c = [], []
        for ci, cname in enumerate(TARGET_CLASSES):
            sub = merged[merged["label"] == ci]
            med_corr.append(float(sub["corr_meanlead_12lead"].median()))
            rec_c.append(float((sub["pred"].values == ci).mean()))
        rho_c, pv_c = spearmanr(med_corr, rec_c)
        rho_n, pv_n = spearmanr([float(merged[merged.label == ci]["nrmse_scaled_12lead"].median())
                                 for ci in range(N)], rec_c)
        class_assoc[seed] = {"median_corr_by_class": dict(zip(TARGET_CLASSES, med_corr)),
                             "median_nrmse_by_class": dict(zip(
                                 TARGET_CLASSES,
                                 [float(merged[merged.label == ci]["nrmse_scaled_12lead"].median())
                                  for ci in range(N)])),
                             "recall_by_class": dict(zip(TARGET_CLASSES, rec_c)),
                             "spearman_medcorr_vs_recall": float(rho_c), "p_corr": float(pv_c),
                             "spearman_mednrmse_vs_recall": float(rho_n), "p_nrmse": float(pv_n)}
        print(f"  [T2] seed {seed} done", flush=True)
    return (pd.DataFrame(pooled_rows), pd.DataFrame(within_rows), pd.DataFrame(sp_rows),
            class_assoc)


def add_mean_rows(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).columns.difference(["seed"])
    agg = df.groupby(keys)[list(num)].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.insert(0, "seed", "MEAN_5SEED")
    return pd.concat([df, agg], ignore_index=True)


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"C5 swappable recon descriptors ({len(C5_SWAPPABLE)}): {C5_SWAPPABLE}")

    print("TEST 1 -- descriptor-substitution oracle")
    t1, controls = test1()
    t1.to_csv(OUT / "W1B_substitution_oracle.csv", index=False)

    print("TEST 2 -- stratification by reconstruction quality")
    pooled, within, sp, class_assoc = test2(t1)
    pooled = add_mean_rows(pooled, ["quartile"])
    within = add_mean_rows(within, ["class", "quartile"])
    pooled.to_csv(OUT / "W1B_strata_pooled.csv", index=False)
    within.to_csv(OUT / "W1B_strata_within_class.csv", index=False)

    agg = (sp.groupby(["class", "error_metric"])["spearman_rho"]
             .agg(["mean", "std", "median", "min", "max"]).reset_index())
    agg.columns = ["class", "error_metric", "rho_mean", "rho_std_ddof1", "rho_median",
                   "rho_min", "rho_max"]
    agg["n_seeds"] = 5
    agg["seed"] = "AGG_5SEED"
    sp_out = pd.concat([sp, agg], ignore_index=True)
    sp_out.to_csv(OUT / "W1B_recon_vs_trueprob_spearman.csv", index=False)

    json.dump({"controls_seed42": controls,
               "C5_swappable_descriptors": C5_SWAPPABLE,
               "n_rec37": len(REC37), "n_obs37": len(OBS37),
               "class_level_association": class_assoc},
              open(OUT / "W1B_controls_and_class_association.json", "w"), indent=1)
    print("\ncontrols seed42:", json.dumps(controls, indent=1))
    print("wrote 4 CSVs + 1 JSON to", OUT)


if __name__ == "__main__":
    main()
