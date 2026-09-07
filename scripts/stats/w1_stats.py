"""Wave-1 statistics for the sensors-4543399 revision.

Covers W1-D (MI operating points), W1-H (paired ablation + BH-FDR),
W1-I (five-seed group values and mean confusion), W1-L (the untraceable 0.919).

Protocol is replicated exactly from scripts/42_classification.py:
XGBoost fitted on folds 1-8, evaluated on fold 10; fold 9 is untouched by
training and is therefore free for operating-point selection.

Outputs go to a NEW run directory; the submitted lineage is never overwritten.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

import sys as _sys
from pathlib import Path as _Path
# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro
REPO = repro.repo()
RUNS = repro.runs()
OUT = repro.outdir("runs/v7rev_stats")
OUT.mkdir(parents=True, exist_ok=True)

CLS = ["NORM", "MI", "STTC", "CD"]
N = 4
SEEDS = [42, 1, 2, 3, 4]

# --- feature groups, verbatim from scripts/42_classification.py ---
THETA = [f"theta_{i}" for i in range(47)]
OLD = ["D_act", "D_rep_old_APDdisp", "S_ST_internal_L2", "mean_vent_APD", "vent_delta_spread"]
REP = ["APD_mean", "APD_disp", "APD_cv", "tau_rep_mean", "tau_rep_disp", "tau_rep_cv",
       "Tend_mean", "Tend_disp", "Tend_cv", "Sync_rep", "Tpeak_maxabs", "Tarea_mean",
       "Tabsarea_mean", "Tasym_mean", "Tlowamp_mean", "T_sign_discordance_rate"]
ST = ["ST_projected_L2", "ST_projected_max_abs", "ST_projected_anterior_leads",
      "ST_projected_lateral_leads", "ST_projected_inferior_leads", "ST_projected_sign_discordance"]
STAB = ["m_proxy", "V_rec"]
IDENT = [f"theta_{i}" for i in list(range(0, 8)) + list(range(16, 24)) + list(range(32, 40)) + [46]] \
        + ["D_act", "T_sign_discordance_rate", "ST_projected_max_abs", "Tpeak_maxabs", "m_proxy", "Sync_rep"]
GROUPS = {"C0_theta": THETA, "C1_theta_old": THETA + OLD, "C2_theta_rep": THETA + REP,
          "C3_theta_st": THETA + ST, "C4_theta_stab": THETA + STAB,
          "C5_theta_rep_st_stab": THETA + REP + ST + STAB, "C6_compact_heuristic": IDENT}
MAIN_GROUPS = ["C0_theta", "C2_theta_rep", "C3_theta_st", "C4_theta_stab",
               "C5_theta_rep_st_stab", "C6_compact_heuristic"]

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


PATIENT_META = _patient_csv()


def make_xgb(seed):
    import xgboost as xgb
    return xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, objective="multi:softprob", num_class=N,
                             tree_method="hist", eval_metric="mlogloss", random_state=seed, n_jobs=-1)


def load_seed(seed):
    """Features + stability proxies for one seed, with the same alignment check as script 13."""
    t = RUNS / f"v1fix_r3_s{seed}" / "tables"
    df = pd.read_csv(t / "round3_features.csv")
    stab = pd.read_csv(t / "stability_proxy_values.csv")
    if len(stab) != len(df):
        raise RuntimeError(f"seed {seed}: stability rows {len(stab)} != features {len(df)}")
    if "record_id" in stab.columns:
        if not (stab["record_id"].to_numpy() == df["record_id"].to_numpy()).all():
            raise RuntimeError(f"seed {seed}: stability not row-aligned with features")
    df["m_proxy"] = stab["m_proxy"].values
    df["V_rec"] = stab["V_rec"].values
    return df


def patient_of(record_ids):
    db = pd.read_csv(PATIENT_META, usecols=["ecg_id", "patient_id"]).set_index("ecg_id")
    pid = db.reindex(np.asarray(record_ids))["patient_id"]
    if pid.isna().any():
        raise RuntimeError("missing patient ids")
    return pid.to_numpy(dtype=np.int64)


def fit_predict(df, seed, groups=MAIN_GROUPS, sample_weight=None):
    """Return {group: (prob_fold9, prob_fold10)}; weights apply to training rows only."""
    label = df["label"].values.astype(int)
    fold = df["fold"].values.astype(int)
    tr, va, te = fold <= 8, fold == 9, fold == 10
    out = {}
    for g in groups:
        cols = [c for c in GROUPS[g] if c in df.columns]
        X = np.nan_to_num(df[cols].values.astype(np.float32))
        sc = StandardScaler().fit(X[tr])
        Xs = sc.transform(X)
        clf = make_xgb(seed)
        if sample_weight is None:
            clf.fit(Xs[tr], label[tr])
        else:
            clf.fit(Xs[tr], label[tr], sample_weight=sample_weight[tr])
        out[g] = (clf.predict_proba(Xs[va]), clf.predict_proba(Xs[te]))
    return out, label, tr, va, te


def macro_auroc(y, p):
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro"))


def summarize(y, pred, prob=None):
    cm = confusion_matrix(y, pred, labels=range(N))
    rec = cm.diagonal() / cm.sum(1)
    prec = np.divide(cm.diagonal(), cm.sum(0), out=np.zeros(N), where=cm.sum(0) > 0)
    d = {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
         "macro_f1": float(f1_score(y, pred, average="macro"))}
    for i, c in enumerate(CLS):
        d[f"recall_{c}"] = float(rec[i])
        d[f"precision_{c}"] = float(prec[i])
    if prob is not None:
        d["macro_auroc"] = macro_auroc(y, prob)
    d["MI_to_NORM"] = int(cm[1, 0])
    d["MI_total"] = int(cm[1].sum())
    return d, cm


def cluster_boot_idx(groups, rng, n_boot):
    """Patient-clustered bootstrap: resample patients, take all their records."""
    uniq = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in uniq}
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        yield np.concatenate([by[g] for g in drawn])


# =====================================================================
def w1d_mi_operating_points(n_boot=1000):
    """W1-D: MI sensitivity remedies, all on the locked seed-42 C5 model."""
    print("\n" + "=" * 78)
    print("W1-D  MI operating points (seed-42, C5, fold-10 test; fold 9 for selection)")
    print("=" * 78)
    df = load_seed(42)
    probs, label, tr, va, te = fit_predict(df, 42, groups=["C5_theta_rep_st_stab"])
    p9, p10 = probs["C5_theta_rep_st_stab"]
    y9, y10 = label[va], label[te]
    pid10 = patient_of(df.loc[te, "record_id"].to_numpy())

    rows = []

    # --- (0) control: unweighted argmax, the paper's operating point ---
    base_pred = p10.argmax(1)
    d, cm = summarize(y10, base_pred, p10)
    d.update(rule="argmax (paper)", selected_on="-")
    rows.append(d)
    print(f"control  macro-AUROC {d['macro_auroc']:.4f}  MI recall {d['recall_MI']:.3f}  "
          f"MI->NORM {d['MI_to_NORM']}/{d['MI_total']}  balacc {d['balanced_accuracy']:.4f}")

    # --- (1) MI-specific thresholds chosen on fold 9 at target sensitivity ---
    # Decision rule: flag MI when P(MI) >= t, else argmax over the remaining classes.
    def apply_mi_threshold(p, t):
        pred = p.argmax(1)
        mi = p[:, 1] >= t
        alt = p.copy()
        alt[:, 1] = -1.0
        pred = np.where(mi, 1, alt.argmax(1))
        return pred

    for target in (0.70, 0.80, 0.90):
        # smallest threshold on fold 9 achieving >= target MI sensitivity
        ts = np.unique(np.round(p9[:, 1], 4))
        best_t = None
        for t in ts:
            sens9 = recall_score(y9, apply_mi_threshold(p9, t), labels=[1], average="macro",
                                 zero_division=0)
            if sens9 >= target:
                best_t = t  # keep raising t while target still met -> take the largest such t
        if best_t is None:
            best_t = float(ts.min())
        pred = apply_mi_threshold(p10, best_t)
        d, cm = summarize(y10, pred, p10)
        d.update(rule=f"MI threshold @fold-9 sens>={target:.2f}", selected_on=f"fold 9, t={best_t:.4f}")
        rows.append(d)
        print(f"thr>={target:.2f}  t={best_t:.4f}  MI recall {d['recall_MI']:.3f}  "
              f"MI prec {d['precision_MI']:.3f}  NORM recall {d['recall_NORM']:.3f}  "
              f"balacc {d['balanced_accuracy']:.4f}")

    # --- (2) inverse-frequency class weights ---
    counts = np.bincount(label[tr], minlength=N)
    w_per_class = counts.sum() / (N * counts)
    sw = w_per_class[label]
    probs_w, _, _, _, _ = fit_predict(df, 42, groups=["C5_theta_rep_st_stab"], sample_weight=sw)
    _, p10w = probs_w["C5_theta_rep_st_stab"]
    d, cm = summarize(y10, p10w.argmax(1), p10w)
    d.update(rule="inverse-frequency class weights", selected_on="-")
    rows.append(d)
    print(f"invfreq  macro-AUROC {d['macro_auroc']:.4f}  MI recall {d['recall_MI']:.3f}  "
          f"balacc {d['balanced_accuracy']:.4f}  macroF1 {d['macro_f1']:.4f}")

    # --- (3) class-weighted multinomial logistic regression ---
    from sklearn.linear_model import LogisticRegression
    cols = [c for c in GROUPS["C5_theta_rep_st_stab"] if c in df.columns]
    X = np.nan_to_num(df[cols].values.astype(np.float32))
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(Xs[tr], label[tr])
    plr = lr.predict_proba(Xs[te])
    d, cm = summarize(y10, plr.argmax(1), plr)
    d.update(rule="class-weighted logistic regression", selected_on="-")
    rows.append(d)
    print(f"logreg   macro-AUROC {d['macro_auroc']:.4f}  MI recall {d['recall_MI']:.3f}  "
          f"balacc {d['balanced_accuracy']:.4f}")

    # --- (4) prior-corrected argmax: rescale posteriors to full-PTB-XL class frequencies ---
    # Turns "the filter set the prior" from an assertion into a measurement.
    prior_train = counts / counts.sum()
    # Measured by W1-E over all PTB-XL records carrying >=1 of the four superclasses
    # (records with several superclasses count in each): [0.3788, 0.2177, 0.2084, 0.1950].
    prior_path = OUT / "w1e_prior_full.npy"
    if not prior_path.exists():
        raise RuntimeError("run w1e_inclusion.py first -- it measures the full-PTB-XL prior")
    prior_full = np.load(prior_path)
    prior_full = prior_full / prior_full.sum()
    print(f"         full-PTB-XL prior {np.round(prior_full,4).tolist()}  "
          f"vs clean-train prior {np.round(prior_train,4).tolist()}")
    adj = p10 * (prior_full / prior_train)[None, :]
    adj = adj / adj.sum(1, keepdims=True)
    d, cm = summarize(y10, adj.argmax(1), adj)
    d.update(rule="prior-corrected argmax (full-PTB-XL prior)", selected_on="-")
    rows.append(d)
    print(f"prior    macro-AUROC {d['macro_auroc']:.4f}  MI recall {d['recall_MI']:.3f}  "
          f"balacc {d['balanced_accuracy']:.4f}")

    # --- (5) patient-clustered bootstrap CIs on MI sensitivity for each rule ---
    rng = np.random.default_rng(42)
    idx_sets = list(cluster_boot_idx(pid10, rng, n_boot))
    rule_preds = {"argmax (paper)": base_pred,
                  "inverse-frequency class weights": p10w.argmax(1),
                  "prior-corrected argmax (full-PTB-XL prior)": adj.argmax(1)}
    for r in rows:
        if r["rule"].startswith("MI threshold"):
            t = float(r["selected_on"].split("t=")[1])
            rule_preds[r["rule"]] = apply_mi_threshold(p10, t)
    for r in rows:
        pred = rule_preds.get(r["rule"])
        if pred is None:
            continue
        sens, spec_norm = [], []
        for ii in idx_sets:
            yy, pp = y10[ii], pred[ii]
            m = yy == 1
            if m.sum() == 0:
                continue
            sens.append((pp[m] == 1).mean())
            spec_norm.append((pp[~m] != 1).mean())
        r["recall_MI_lo"], r["recall_MI_hi"] = float(np.percentile(sens, 2.5)), float(np.percentile(sens, 97.5))
        r["MI_specificity"] = float(np.mean(spec_norm))

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "w1d_mi_operating_points.csv", index=False)
    print(f"\n-> {OUT / 'w1d_mi_operating_points.csv'}")
    return out


# =====================================================================
def w1h_paired_ablation(n_boot=2000):
    """W1-H: all 15 pairs among C0,C2..C6 by joint patient-clustered resampling + BH-FDR."""
    print("\n" + "=" * 78)
    print("W1-H  Paired ablation contrasts with BH-FDR (seed-42, fold-10)")
    print("=" * 78)
    t = RUNS / "v1fix_r3_s42" / "tables"
    pr = pd.read_csv(t / "round3_xgboost_test_predictions.csv")
    pcol = [f"prob_{c}" for c in CLS]
    piv = {g: pr[pr.feature_group == g].sort_values("record_id") for g in MAIN_GROUPS}
    ref = piv[MAIN_GROUPS[0]]
    y = ref.label.values
    pid = ref.patient_id.values
    for g in MAIN_GROUPS:
        assert (piv[g].record_id.values == ref.record_id.values).all(), g
    P = {g: piv[g][pcol].values for g in MAIN_GROUPS}
    obs = {g: macro_auroc(y, P[g]) for g in MAIN_GROUPS}
    print("point estimates:", {g.split('_')[0]: round(v, 4) for g, v in obs.items()})

    rng = np.random.default_rng(20260903)
    boot = {g: [] for g in MAIN_GROUPS}
    kept = 0
    for ii in cluster_boot_idx(pid, rng, n_boot):
        yy = y[ii]
        if len(np.unique(yy)) < N:
            continue
        kept += 1
        for g in MAIN_GROUPS:
            boot[g].append(macro_auroc(yy, P[g][ii]))
    B = {g: np.asarray(v) for g, v in boot.items()}
    print(f"usable bootstrap replicates: {kept}/{n_boot}")

    rows = []
    for i, a in enumerate(MAIN_GROUPS):
        for b in MAIN_GROUPS[i + 1:]:
            d = B[a] - B[b]
            delta = obs[a] - obs[b]
            # two-sided bootstrap p: proportion of replicates on the other side of 0, doubled
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            p = min(1.0, max(p, 1.0 / kept))
            rows.append({"group_a": a, "group_b": b, "auroc_a": obs[a], "auroc_b": obs[b],
                         "delta": delta, "ci_lo": float(np.percentile(d, 2.5)),
                         "ci_hi": float(np.percentile(d, 97.5)), "p_raw": float(p)})
    res = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    m = len(res)
    res["p_bonferroni"] = np.minimum(1.0, res.p_raw * m)
    ranked = np.arange(1, m + 1)
    bh = res.p_raw.values * m / ranked
    res["p_bh"] = np.minimum.accumulate(bh[::-1])[::-1].clip(0, 1)
    res["sig_bh_05"] = res.p_bh < 0.05
    res["sig_bonf_05"] = res.p_bonferroni < 0.05
    res.to_csv(OUT / "w1h_paired_ablation_fdr.csv", index=False)
    print(res[["group_a", "group_b", "delta", "ci_lo", "ci_hi", "p_raw", "p_bh", "sig_bh_05"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nBH q<0.05: {int(res.sig_bh_05.sum())}/{m}   Bonferroni p<0.05: {int(res.sig_bonf_05.sum())}/{m}")
    print(f"-> {OUT / 'w1h_paired_ablation_fdr.csv'}")
    return res


# =====================================================================
def w1i_five_seed():
    """W1-I: five-seed macro-AUROC per group and the five-seed mean confusion matrix."""
    print("\n" + "=" * 78)
    print("W1-I  Five-seed group values and mean confusion (ddof=1)")
    print("=" * 78)
    pcol = [f"prob_{c}" for c in CLS]
    per_seed, cms = {g: [] for g in MAIN_GROUPS}, []
    for s in SEEDS:
        t = RUNS / f"v1fix_r3_s{s}" / "tables" / "round3_xgboost_test_predictions.csv"
        if not t.exists():
            print(f"  ! missing seed {s}: {t}")
            continue
        pr = pd.read_csv(t)
        for g in MAIN_GROUPS:
            sub = pr[pr.feature_group == g]
            per_seed[g].append(macro_auroc(sub.label.values, sub[pcol].values))
        c5 = pr[pr.feature_group == "C5_theta_rep_st_stab"]
        cms.append(confusion_matrix(c5.label.values, c5.predicted_label.values, labels=range(N)))
    rows = []
    for g in MAIN_GROUPS:
        v = np.asarray(per_seed[g])
        rows.append({"group": g, "n_seeds": len(v), "seed42": v[0] if len(v) else np.nan,
                     "mean": v.mean(), "std_ddof1": v.std(ddof=1), "min": v.min(), "max": v.max()})
        print(f"{g:24s} seed42={v[0]:.4f}  mean={v.mean():.4f} +- {v.std(ddof=1):.4f}  "
              f"[{v.min():.4f}, {v.max():.4f}]")
    pd.DataFrame(rows).to_csv(OUT / "w1i_group_auroc_five_seed.csv", index=False)

    A = np.stack(cms).astype(float)
    mean_cm, sd_cm = A.mean(0), A.std(0, ddof=1)
    mi_norm = A[:, 1, 0]
    print(f"\nMI->NORM per seed: {mi_norm.astype(int).tolist()}  "
          f"mean {mi_norm.mean():.1f} +- {mi_norm.std(ddof=1):.1f}  "
          f"= {mi_norm.mean()/A[0,1].sum()*100:.1f}% of {int(A[0,1].sum())}")
    pd.DataFrame(mean_cm, index=CLS, columns=CLS).to_csv(OUT / "w1i_mean_confusion.csv")
    pd.DataFrame(sd_cm, index=CLS, columns=CLS).to_csv(OUT / "w1i_sd_confusion.csv")
    print(f"-> {OUT / 'w1i_group_auroc_five_seed.csv'}")
    return rows


# =====================================================================
def w1l_class_matched():
    """W1-L: recompute the class-matched three-class internal macro-AUROC (the '0.919')."""
    print("\n" + "=" * 78)
    print("W1-L  Class-matched internal three-class macro-AUROC (paper claims 0.919)")
    print("=" * 78)
    pcol = [f"prob_{c}" for c in CLS]
    keep = [0, 2, 3]  # NORM, STTC, CD -- MI absent externally
    rows = []
    for g in ["C0_theta", "C5_theta_rep_st_stab"]:
        for s in SEEDS:
            t = RUNS / f"v1fix_r3_s{s}" / "tables" / "round3_xgboost_test_predictions.csv"
            if not t.exists():
                continue
            pr = pd.read_csv(t)
            sub = pr[pr.feature_group == g]
            y, P = sub.label.values, sub[pcol].values

            # (a) drop MI records, restrict to 3 columns, renormalize
            m = np.isin(y, keep)
            PP = P[m][:, keep]
            PP = PP / PP.sum(1, keepdims=True)
            remap = {c: i for i, c in enumerate(keep)}
            a_drop = roc_auc_score(np.vectorize(remap.get)(y[m]), PP,
                                   multi_class="ovr", average="macro")

            # (b) keep all 1594 records; average the one-vs-rest AUROC of the 3 present classes
            a_ovr3 = float(np.mean([roc_auc_score((y == c).astype(int), P[:, c]) for c in keep]))

            # (c) drop MI records but score each class one-vs-rest within that subset
            a_drop_ovr = float(np.mean([roc_auc_score((y[m] == c).astype(int), P[m][:, c])
                                        for c in keep]))

            rows.append({"group": g, "seed": s, "drop_renorm": a_drop,
                         "ovr3_all_records": a_ovr3, "drop_ovr": a_drop_ovr, "n_dropped": int(m.sum())})
            print(f"  {g:24s} seed {s:>2}:  drop+renorm {a_drop:.4f}   "
                  f"OvR-3 all records {a_ovr3:.4f}   drop+OvR {a_drop_ovr:.4f}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "w1l_class_matched_variants.csv", index=False)
    print("\nPaper states 0.919 at L761 and in Table 7.")
    hit = res[(res[["drop_renorm", "ovr3_all_records", "drop_ovr"]].round(3) == 0.919).any(axis=1)]
    print("variants rounding to 0.919:", "NONE" if hit.empty else hit.to_string(index=False))
    print(f"-> {OUT / 'w1l_class_matched_variants.csv'}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "d"):
        w1d_mi_operating_points()
    if which in ("all", "h"):
        w1h_paired_ablation()
    if which in ("all", "i"):
        w1i_five_seed()
    if which in ("all", "l"):
        w1l_class_matched()
    print(f"\nAll outputs under {OUT}")
