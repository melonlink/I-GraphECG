"""W1F stage 2: zero-shot multi-label PTB-XL evaluation with the encoder FROZEN.

Nothing is retrained. The locked seed-42 surrogate
(``outputs/checkpoints/d1s_r3_s42_best.pt``, the paper model) is applied to the
canonical PTB-XL superdiagnostic pool (every record carrying >=1 diagnostic
superclass, 21,388 records), including the records the single-label "clean"
filter excluded. The frozen PTB-XL training robust scaler is reused verbatim --
never refit. Only the downstream one-vs-rest classifier is fitted, on the
multi-label training folds 1--8.

Outputs (into outputs/runs/v7rev_stats):
  W1F_multilabel_recon.csv       reconstruction quality, fold 10, sliced by label cardinality
  W1F_multilabel_auroc.csv       per-label + macro AUROC/AUPRC with patient-clustered CIs
  W1F_multilabel_controls.csv    every control this task had to reproduce
  W1F_multilabel_perrecord.csv   per-record fold-10 recon + predicted probabilities
  W1F_multilabel_recon_delta.csv the (b)-minus-(single-label) degradation, with CI

Usage:
    ECG_DATA_DIR=<...>/data_PTB-XL python scripts/70_multilabel_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igraphecg import repro

from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.evaluation.recon_metrics import (compute_recon_metrics, per_sample_lead_corr,
                                                per_sample_nrmse)
from igraphecg.evaluation.round2_io import infer_theta, load_encoder_decoder
from igraphecg.evaluation.stability import TrainZScorer
from igraphecg.utils.logging import get_logger
from igraphecg.utils.paths import PROJECT_ROOT
from igraphecg.utils.seed import set_seed

log = get_logger("w1f_eval")

SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
SEED = 42
N_BOOT = 1000
POOL_EXPECTED = 21388   # CONTROL: canonical >=1-superclass pool
POOL_FOLD10_EXPECTED = 2158   # CONTROL: its fold-10 part
CLEAN_EXPECTED = 15709   # CONTROL: the frozen single-label clean subset

ML_NPZ = PROJECT_ROOT / "data" / "processed" / "ptbxl_medianbeat_multilabel_100hz.npz"
CLEAN_NPZ = PROJECT_ROOT / "data" / "processed" / "ptbxl_medianbeat_clean_100hz.npz"
CKPT = repro.lineage.checkpoint(42)
FROZEN_RUN = repro.runs() / "v1fix_r3_s42"
FROZEN_D1S = repro.runs() / "d1s_r3_s42"
OUT = repro.outdir("runs/v7rev_stats")

# ---- frozen descriptor definition (verbatim from scripts/42_classification.py) ----
THETA = [f"theta_{i}" for i in range(47)]
REP = ["APD_mean", "APD_disp", "APD_cv", "tau_rep_mean", "tau_rep_disp", "tau_rep_cv",
       "Tend_mean", "Tend_disp", "Tend_cv", "Sync_rep", "Tpeak_maxabs", "Tarea_mean",
       "Tabsarea_mean", "Tasym_mean", "Tlowamp_mean", "T_sign_discordance_rate"]
ST = ["ST_projected_L2", "ST_projected_max_abs", "ST_projected_anterior_leads",
      "ST_projected_lateral_leads", "ST_projected_inferior_leads", "ST_projected_sign_discordance"]
STAB = ["m_proxy", "V_rec"]
GROUPS = {"C0_theta": THETA, "C5_theta_rep_st_stab": THETA + REP + ST + STAB}
C_REP = 2.0   # configs/v1fix/r3_s42.yaml
LEADFIELD_RANK = 3   # configs/v1fix/r3_s42.yaml


def build_features(theta, recon_mV, H_ind, fs, c_rep=C_REP):
    """Exactly scripts/40_descriptors.py's surrogate-side descriptor build
    (the ``obs_`` observed-ECG features are omitted: no C0/C5 descriptor uses them)."""
    import torch

    from igraphecg.evaluation.repolarization_features import (repolarization_param_features,
                                                              twave_features)
    from igraphecg.evaluation.st_features import internal_st_features, projected_st_features
    from igraphecg.models.phys_features import compute_phys_features

    feats = {}
    old = compute_phys_features(torch.from_numpy(theta.astype(np.float32)))
    feats["D_act"] = old["D_act"].numpy()
    feats["D_rep_old_APDdisp"] = old["D_rep"].numpy()
    feats["S_ST_internal_L2"] = old["S_ST"].numpy()
    feats["mean_vent_APD"] = old["mean_vent_APD"].numpy()
    feats["vent_delta_spread"] = old["vent_delta_spread"].numpy()
    feats["prox_delay"] = old["prox_delay"].numpy()
    feats["leaf_delay_spread"] = old["leaf_delay_spread"].numpy()
    feats.update(repolarization_param_features(theta, c_rep=c_rep))
    feats.update(twave_features(recon_mV, fs=fs))
    feats.update(internal_st_features(theta))
    feats.update(projected_st_features(theta, H_ind))
    return feats


def make_binary_clf(kind, seed):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    import xgboost as xgb
    return xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, objective="binary:logistic",
                             tree_method="hist", eval_metric="logloss", random_state=seed,
                             n_jobs=-1)


def _patient_index(patients):
    """Return (unique patients, list of index arrays) for cluster resampling."""
    uniq, inv = np.unique(patients, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    starts = np.searchsorted(sorted_inv, np.arange(len(uniq)), side="left")
    ends = np.searchsorted(sorted_inv, np.arange(len(uniq)), side="right")
    return uniq, [order[a:b] for a, b in zip(starts, ends)]


def patient_bootstrap_ci(y_true, score, patients, n_boot=N_BOOT, seed=SEED, metric="auroc"):
    """Percentile 95% CI, resampling whole patients (clusters) with replacement."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(seed)
    uniq, groups = _patient_index(patients)
    fn = roc_auc_score if metric == "auroc" else average_precision_score
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([groups[p] for p in pick])
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        vals.append(float(fn(yt, score[idx])))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def macro_bootstrap_ci(Y, S, patients, n_boot=N_BOOT, seed=SEED, metric="auroc"):
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(seed)
    uniq, groups = _patient_index(patients)
    fn = roc_auc_score if metric == "auroc" else average_precision_score
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([groups[p] for p in pick])
        per = []
        for k in range(Y.shape[1]):
            yt = Y[idx, k]
            if yt.min() == yt.max():
                per = []
                break
            per.append(float(fn(yt, S[idx, k])))
        if per:
            vals.append(float(np.mean(per)))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def median_delta_ci(values, mask_a, mask_b, patients, n_boot=N_BOOT, seed=SEED):
    """median(values[mask_b]) - median(values[mask_a]) with patient-clustered percentile CI.

    masks are boolean arrays over the same index space as ``values``/``patients``.
    """
    rng = np.random.default_rng(seed)
    uniq, groups = _patient_index(patients)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([groups[p] for p in pick])
        a = values[idx][mask_a[idx]]
        b = values[idx][mask_b[idx]]
        if len(a) < 5 or len(b) < 5:
            continue
        vals.append(float(np.median(b) - np.median(a)))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    import torch
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    controls = []

    def ctl(name, expected, observed, ok):
        controls.append({"control": name, "expected": expected, "observed": observed,
                         "ok": bool(ok)})
        log.info(f"CONTROL {name}: expected={expected} observed={observed} ok={ok}")

    d = np.load(ML_NPZ, allow_pickle=True)
    signals = d["signal_12lead"].astype(np.float32)
    Y = d["y_multilabel"].astype(int)
    fold = d["fold"].astype(int)
    rec = d["record_id"].astype(np.int64)
    pat = d["patient_id"].astype(np.int64)
    is_clean = d["is_clean"].astype(bool)
    fs = int(d["sampling_rate"])
    assert list(d["superclasses"]) == SUPER, list(d["superclasses"])
    n_lab = Y.sum(axis=1)
    log.info(f"multi-label pool processed N={len(rec)}  fold10={int((fold == 10).sum())}")

    ctl("pool_ge1_superclass_processed", POOL_EXPECTED, int(len(rec)),
        POOL_EXPECTED - int(len(rec)) <= 1)   # 1 record fails the frozen beat-quality gate
    ctl("pool_fold10", POOL_FOLD10_EXPECTED, int((fold == 10).sum()),
        int((fold == 10).sum()) == POOL_FOLD10_EXPECTED)
    ctl("clean_records_present", CLEAN_EXPECTED, int(is_clean.sum()),
        int(is_clean.sum()) == CLEAN_EXPECTED)

    # ---- frozen training scaler: copied from the submitted npz by stage 1, never refit ----
    with np.load(CLEAN_NPZ) as f:
        med_f, iqr_f = f["scaler_median"], f["scaler_iqr"]
        clean_sig, clean_rec = f["signal_12lead"], f["record_id"]
    same_scaler = bool(np.array_equal(med_f, d["scaler_median"])
                       and np.array_equal(iqr_f, d["scaler_iqr"]))
    ctl("frozen_scaler_reused_not_refit", "bit-identical",
        "bit-identical" if same_scaler else "DIFFERENT", same_scaler)
    assert same_scaler

    # ---- median beats of the clean records must be bit-identical to the frozen npz ----
    pos = pd.Series(np.arange(len(rec)), index=rec).reindex(clean_rec).to_numpy()
    assert not np.isnan(pos).any()
    beats_same = bool(np.array_equal(clean_sig, signals[pos.astype(int)]))
    ctl("clean_median_beats_bit_identical", "identical",
        "identical" if beats_same else "DIFFERENT", beats_same)

    scaler = RobustLeadScaler.from_dict({"median": med_f, "iqr": iqr_f})
    scaled = scaler.transform(signals)

    # ---- frozen encoder, zero-shot ----
    enc, dec = load_encoder_decoder(CKPT, n_t=signals.shape[2], rank=LEADFIELD_RANK,
                                    device=device, scaler={"median": med_f, "iqr": iqr_f})
    theta, _z = infer_theta(enc, dec, scaled, device=device)
    H_ind = (dec.A @ dec.B).detach().cpu().numpy()
    with torch.no_grad():
        yh = []
        for s in range(0, len(theta), 1024):
            y12, _ = dec.forward(torch.from_numpy(theta[s:s + 1024].astype(np.float32)).to(device))
            yh.append(y12.cpu().numpy())
    yhat_scaled = np.concatenate(yh)
    recon_mV = scaler.inverse_transform(yhat_scaled)
    t_sec = dec.t.cpu().numpy()
    log.info(f"theta {theta.shape}  yhat {yhat_scaled.shape}  device={device}")

    # ================= 1. reconstruction quality =================
    nrmse = per_sample_nrmse(scaled, yhat_scaled)   # scaled space (primary)
    nrmse_mV = per_sample_nrmse(signals, recon_mV)   # physical mV
    corr = per_sample_lead_corr(scaled, yhat_scaled)   # Pearson: scale-invariant

    def slice_row(name, mask, note=""):
        if mask.sum() == 0:
            return None
        m = compute_recon_metrics(scaled[mask], yhat_scaled[mask], t_sec)
        return {"subset": name, "n": int(mask.sum()),
                "median_nrmse_scaled": m["median_nrmse"], "mean_nrmse_scaled": m["mean_nrmse"],
                "nrmse_q25_scaled": float(np.percentile(nrmse[mask], 25)),
                "nrmse_q75_scaled": float(np.percentile(nrmse[mask], 75)),
                "median_nrmse_mV": float(np.median(nrmse_mV[mask])),
                "median_corr": m["median_corr"], "mean_corr": m["mean_corr"],
                "corr_q25": float(np.percentile(corr[mask], 25)),
                "corr_q75": float(np.percentile(corr[mask], 75)),
                "qrs_nrmse_scaled": m["qrs_nrmse"], "st_nrmse_scaled": m["st_nrmse"],
                "t_nrmse_scaled": m["t_nrmse"], "st_mean_abs_err_scaled": m["st_mean_abs_err"],
                "note": note}

    te = fold == 10
    rows = [
        slice_row("fold10_all_ge1_superclass", te,
                  "(a) every fold-10 record with >=1 diagnostic superclass"),
        slice_row("fold10_multilabel_gt1", te & (n_lab > 1),
                  "(b) fold-10 records carrying MORE THAN ONE superclass"),
        slice_row("fold10_single_superclass", te & (n_lab == 1),
                  "reference: exactly 1 superclass"),
        slice_row("fold10_clean_subset", te & is_clean,
                  "CONTROL: the frozen single-label test set (n must be 1594)"),
        slice_row("fold10_excluded_by_clean_filter", te & ~is_clean,
                  "records the study's clean filter removed"),
        slice_row("fold10_card2", te & (n_lab == 2), "cardinality exactly 2"),
        slice_row("fold10_card3plus", te & (n_lab >= 3), "cardinality >=3"),
        slice_row("all_folds_ge1_superclass", np.ones(len(rec), bool), "whole pool, all folds"),
    ]
    for c in SUPER:
        k = SUPER.index(c)
        rows.append(slice_row(f"fold10_label_{c}", te & (Y[:, k] == 1),
                              f"fold-10 records carrying {c} (any cardinality)"))
    recon_df = pd.DataFrame([r for r in rows if r is not None])
    recon_df.to_csv(OUT / "W1F_multilabel_recon.csv", index=False)
    log.info("\n" + recon_df[["subset", "n", "median_nrmse_scaled", "median_corr"]]
             .to_string(index=False))

    # degradation of (b) relative to the single-label reference, patient-clustered CI
    m_single, m_multi = (te & (n_lab == 1)), (te & (n_lab > 1))
    m_clean, m_excl = (te & is_clean), (te & ~is_clean)
    from scipy.stats import mannwhitneyu
    delta_rows = []
    for nm, ma, mb in [("multi_gt1_minus_single", m_single, m_multi),
                       ("excluded_minus_clean", m_clean, m_excl)]:
        for metric, vals in [("median_nrmse_scaled", nrmse), ("median_corr", corr)]:
            lo, hi = median_delta_ci(vals[te], ma[te], mb[te], pat[te])
            u = mannwhitneyu(vals[mb], vals[ma], alternative="two-sided")
            delta_rows.append({
                "comparison": nm, "metric": metric,
                "n_a": int(ma.sum()), "n_b": int(mb.sum()),
                "median_a": float(np.median(vals[ma])), "median_b": float(np.median(vals[mb])),
                "delta_b_minus_a": float(np.median(vals[mb]) - np.median(vals[ma])),
                "delta_ci_lo": lo, "delta_ci_hi": hi,
                "mannwhitney_u": float(u.statistic), "mannwhitney_p": float(u.pvalue),
                "bootstrap_unit": "patient", "n_boot": N_BOOT})
    pd.DataFrame(delta_rows).to_csv(OUT / "W1F_multilabel_recon_delta.csv", index=False)
    log.info("\n" + pd.DataFrame(delta_rows)[
        ["comparison", "metric", "median_a", "median_b", "delta_b_minus_a",
         "delta_ci_lo", "delta_ci_hi", "mannwhitney_p"]].to_string(index=False))

    # CONTROL: the clean fold-10 subset must reproduce the frozen d1s_r3_s42 test numbers
    fz = pd.read_csv(FROZEN_D1S / "tables" / "reconstruction_metrics.csv")
    fz = fz[(fz.split == "test") & (fz.space == "scaled")].iloc[0]
    got = recon_df[recon_df.subset == "fold10_clean_subset"].iloc[0]
    for nm, exp, obs, tol in [("clean_test_n", fz["n"], got["n"], 0.5),
                              ("clean_test_median_nrmse_scaled", fz["median_nrmse"],
                               got["median_nrmse_scaled"], 2e-3),
                              ("clean_test_median_corr", fz["median_corr"],
                               got["median_corr"], 2e-3)]:
        ctl(nm, float(exp), float(obs), abs(float(exp) - float(obs)) < tol)

    # ================= 2. multi-label classification with FROZEN descriptors =================
    feats = build_features(theta, recon_mV, H_ind, fs)
    df = pd.DataFrame(feats)
    for j in range(theta.shape[1]):
        df[f"theta_{j}"] = theta[:, j]

    # stability proxy: frozen definition, z-scored on the *frozen clean training folds*
    zkeys = ["D_act", "Tend_disp", "tau_rep_disp", "ST_projected_L2"]
    zs = TrainZScorer().fit({k: df[k].values for k in zkeys},
                            train_mask=(is_clean & (fold <= 8)))
    V = sum(zs.z(k, df[k].values) for k in zkeys)
    df["V_rec"] = V
    df["m_proxy"] = -V

    # ---- CONTROL: descriptors on the clean records must match the frozen feature table ----
    #
    # Two levels, because the pooled forward pass cannot be bit-identical to the frozen one:
    # the pool has 21,387 records instead of 15,709, so the final partial batch has a
    # different shape and cuDNN selects a different convolution kernel for it. That is
    # float32 noise, not a pipeline difference. So we check BOTH:
    #   (i) batching-matched: rerun the encoder over the clean records ALONE, exactly the
    #       frozen batch composition -> must agree with the frozen theta to ~1e-7;
    #   (ii) pooled: the theta actually used below -> must agree to <1% of each
    #       descriptor's own standard deviation.
    fz_feat = pd.read_csv(FROZEN_RUN / "tables" / "round3_features.csv")
    fz_stab = pd.read_csv(FROZEN_RUN / "tables" / "stability_proxy_values.csv")
    mine = df.copy()
    mine["record_id"] = rec

    theta_clean_only, _ = infer_theta(enc, dec, scaled[pos.astype(int)], device=device)
    theta_frozen = fz_feat[[f"theta_{i}" for i in range(47)]].to_numpy()
    assert np.array_equal(fz_feat["record_id"].to_numpy(), clean_rec)
    dth = float(np.max(np.abs(theta_clean_only - theta_frozen)))
    ctl("theta_batching_matched_max_abs_diff", "<1e-5 (float32 exact)", dth, dth < 1e-5)
    dth_pool = float(np.max(np.abs(theta[pos.astype(int)] - theta_frozen)))
    ctl("theta_pooled_max_abs_diff", "<1e-3", dth_pool, dth_pool < 1e-3)

    check_cols = ["theta_0", "theta_23", "ST_projected_L2", "Sync_rep", "D_act", "Tend_disp"]
    j = fz_feat[["record_id"] + check_cols].merge(mine[["record_id"] + check_cols],
                                                  on="record_id", suffixes=("_frozen", "_new"))
    log.info(f"descriptor control over {len(j)} shared records")
    for col in check_cols:
        dmax = float(np.max(np.abs(j[f"{col}_frozen"] - j[f"{col}_new"])))
        sd = float(fz_feat[col].std())
        rel = dmax / sd
        ctl(f"descriptor_{col}_max_abs_diff_over_sd", "<0.01", rel, rel < 0.01)
        controls[-1]["max_abs_diff"] = dmax
        controls[-1]["frozen_sd"] = sd
    js = fz_stab[["record_id", "m_proxy"]].merge(mine[["record_id", "m_proxy"]], on="record_id",
                                                 suffixes=("_frozen", "_new"))
    dmax = float(np.max(np.abs(js["m_proxy_frozen"] - js["m_proxy_new"])))
    sd = float(fz_stab["m_proxy"].std())
    ctl("m_proxy_max_abs_diff_over_sd", "<0.01", dmax / sd, dmax / sd < 0.01)
    controls[-1]["max_abs_diff"] = dmax
    controls[-1]["frozen_sd"] = sd

    tr, va, teM = fold <= 8, fold == 9, fold == 10
    log.info(f"multi-label split train/val/test = {tr.sum()}/{va.sum()}/{teM.sum()}")

    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    auroc_rows, prob_store = [], {}
    for gname, cols in GROUPS.items():
        cols = [c for c in cols if c in df.columns]
        X = np.nan_to_num(df[cols].values.astype(np.float32))
        sc = StandardScaler().fit(X[tr])
        Xs = sc.transform(X)
        for kind in ("logreg", "xgboost"):
            S_te = np.zeros((int(teM.sum()), len(SUPER)))
            S_va = np.zeros((int(va.sum()), len(SUPER)))
            for k in range(len(SUPER)):
                clf = make_binary_clf(kind, SEED)
                clf.fit(Xs[tr], Y[tr, k])
                S_te[:, k] = clf.predict_proba(Xs[teM])[:, 1]
                S_va[:, k] = clf.predict_proba(Xs[va])[:, 1]
            per_a = [float(roc_auc_score(Y[teM, k], S_te[:, k])) for k in range(len(SUPER))]
            per_p = [float(average_precision_score(Y[teM, k], S_te[:, k]))
                     for k in range(len(SUPER))]
            macro_auroc, macro_auprc = float(np.mean(per_a)), float(np.mean(per_p))
            mlo, mhi = macro_bootstrap_ci(Y[teM], S_te, pat[teM], metric="auroc")
            plo, phi = macro_bootstrap_ci(Y[teM], S_te, pat[teM], metric="auprc")
            auroc_rows.append({
                "feature_group": gname, "n_features": len(cols), "classifier": kind,
                "strategy": "one-vs-rest (5 independent binary models, encoder frozen)",
                "label": "MACRO", "n_pos": int(Y[teM].sum()), "n_test": int(teM.sum()),
                "prevalence": np.nan,
                "auroc": macro_auroc, "auroc_ci_lo": mlo, "auroc_ci_hi": mhi,
                "auprc": macro_auprc, "auprc_ci_lo": plo, "auprc_ci_hi": phi,
                "val_macro_auroc": float(np.mean([roc_auc_score(Y[va, k], S_va[:, k])
                                                  for k in range(len(SUPER))])),
                "bootstrap_unit": "patient", "n_boot": N_BOOT,
                "n_patients_test": int(len(np.unique(pat[teM])))})
            for k, cname in enumerate(SUPER):
                lo, hi = patient_bootstrap_ci(Y[teM, k], S_te[:, k], pat[teM], metric="auroc")
                aplo, aphi = patient_bootstrap_ci(Y[teM, k], S_te[:, k], pat[teM], metric="auprc")
                auroc_rows.append({
                    "feature_group": gname, "n_features": len(cols), "classifier": kind,
                    "strategy": "one-vs-rest (5 independent binary models, encoder frozen)",
                    "label": cname, "n_pos": int(Y[teM, k].sum()), "n_test": int(teM.sum()),
                    "prevalence": float(Y[teM, k].mean()),
                    "auroc": per_a[k], "auroc_ci_lo": lo, "auroc_ci_hi": hi,
                    "auprc": per_p[k], "auprc_ci_lo": aplo, "auprc_ci_hi": aphi,
                    "val_macro_auroc": np.nan,
                    "bootstrap_unit": "patient", "n_boot": N_BOOT,
                    "n_patients_test": int(len(np.unique(pat[teM])))})
            log.info(f"[{gname}/{kind}] macro-AUROC={macro_auroc:.4f} [{mlo:.4f},{mhi:.4f}] "
                     f"macro-AUPRC={macro_auprc:.4f}  per-label="
                     + " ".join(f"{c}:{a:.4f}" for c, a in zip(SUPER, per_a)))
            prob_store[(gname, kind)] = S_te

    pd.DataFrame(auroc_rows).to_csv(OUT / "W1F_multilabel_auroc.csv", index=False)

    # ---- per-record fold-10 dump (C5 / xgboost probabilities) ----
    S = prob_store[("C5_theta_rep_st_stab", "xgboost")]
    per_rec = pd.DataFrame({
        "record_id": rec[teM], "patient_id": pat[teM], "fold": fold[teM],
        "n_superclass": n_lab[teM], "is_clean": is_clean[teM],
        "nrmse_scaled": nrmse[teM], "nrmse_mV": nrmse_mV[teM], "mean_lead_corr": corr[teM],
        **{f"y_{c}": Y[teM, i] for i, c in enumerate(SUPER)},
        **{f"prob_{c}": S[:, i] for i, c in enumerate(SUPER)},
    })
    per_rec.to_csv(OUT / "W1F_multilabel_perrecord.csv", index=False)

    cdf = pd.DataFrame(controls)
    cdf.to_csv(OUT / "W1F_multilabel_controls.csv", index=False)
    bad = cdf[~cdf["ok"].astype(bool)]
    if len(bad):
        log.error("CONTROLS FAILED:\n" + bad.to_string(index=False))
    else:
        log.info("all controls reproduced")
    log.info(f"done -> {OUT}")


if __name__ == "__main__":
    main()
