"""W1S: do the classifiers depend on weakly identifiable directions? (Reviewer 3, Comment 2)

Two tests on the locked seed-42 model and the fold-10 test records, with the C0 (47 parameters)
and C5 (parameters + descriptors) XGBoost classifiers refitted on folds 1-8 exactly as in
scripts/42_classification.py.

(1) Perturbations along Fisher eigendirections. The population FIM of Section 2.8 (400-record
    design subset, radius-scaled parameters, all twelve leads, Sigma = I) is eigendecomposed.
    Each test record's parameter estimate is displaced by a Gaussian step of standard deviation
    `step` radius units along a random combination of the m least-informative directions
    (m = 6, 12, 24) and, as a control, of the 6 most-informative directions; K = 5 draws per
    record, clipped to the admissible box. Reported per condition: the parameter displacement,
    the change of the reconstruction it causes (median |delta NRMSE|), and what happens to the
    predictions: argmax flip rate, mean |delta p(true class)|, and the macro-AUROC of the
    perturbed features under the unchanged classifier.

(2) An alternative inverse. Every fold-10 record is refitted by per-record optimization with the
    locked decoder held fixed (the scripts/32 fixed-H protocol: Adam on z from the parameter
    centre, early stopping on the median NRMSE), giving a second parameter estimate that is not
    the encoder's. Reported: argmax agreement and macro-AUROC of the refitted features under the
    encoder-trained classifiers, and the record-level agreement of each parameter (Pearson r
    across records, encoder vs refit) grouped by identifiability tier (Table S11).

Writes runs/v7rev_stats/W1S_perturbation.csv, W1S_alt_inverse_params.csv, W1S_headline.json.
Run (repository root, torch environment): python scripts/stats/W1S_nullspace_perturbation.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from igraphecg import repro  # noqa: E402
from igraphecg.data.dataset import load_processed  # noqa: E402
from igraphecg.data.label_utils import TARGET_CLASSES  # noqa: E402
from igraphecg.data.preprocess import RobustLeadScaler  # noqa: E402
from igraphecg.evaluation.round2_io import infer_theta, load_encoder_decoder  # noqa: E402
from igraphecg.evaluation.stability import surrogate_stability_proxy  # noqa: E402
from igraphecg.training.losses import build_masks  # noqa: E402

OUT = repro.outdir("runs/v7rev_stats")
LOCKED = repro.runs() / "v1fix_r3_s42"
NPZ = ROOT / "data/processed/ptbxl_medianbeat_clean_100hz.npz"
LAM = 1e-3
SEED = 42
STEPS = (0.25, 0.5)
M_NULL = (6, 12, 24)
K_DRAWS = 5
N = len(TARGET_CLASSES)


def _load_script(name: str):
    p = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixed_subset(label, fold, per_class=100, seed=42):
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(N):
        pool = np.where((label == c) & (fold == 10))[0]
        idx.append(pool if len(pool) <= per_class else rng.choice(pool, per_class, replace=False))
    return np.concatenate(idx)


def population_fim(dec, thetas, radius, device):
    from torch.func import jacfwd
    r = radius.to(device).view(1, -1)

    def f(th):
        y12, _ = dec.forward(th.unsqueeze(0))
        return y12[0].reshape(-1)

    F = torch.zeros(thetas.shape[1], thetas.shape[1], dtype=torch.float64, device=device)
    for s in range(thetas.shape[0]):
        J = (jacfwd(f)(thetas[s].to(device)) * r).double()
        F += J.T @ J
    return (F / thetas.shape[0]).detach().cpu().numpy()


def nrmse(y, yh):
    return np.sqrt(((y - yh) ** 2).sum(axis=(1, 2))) / (np.sqrt((y ** 2).sum(axis=(1, 2))) + 1e-8)


def features_for(theta, yhat_scaled, y_phys, dec, s40, zs_train, c_rep, fs, scaler):
    """C0 and C5 feature frames from a parameter matrix, its (scaled) reconstruction and the observed
    beats in physical units, as scripts/40 builds them (the reconstruction is inverse-scaled first);
    the stability proxy is z-scored on the training-fold statistics."""
    H = dec.H_ind.detach().cpu().numpy()
    feats = s40.build_all_features(theta, scaler.inverse_transform(yhat_scaled), y_phys, H, fs, c_rep)
    df = pd.DataFrame({k: np.asarray(v, dtype=np.float64) for k, v in feats.items()})
    for j in range(theta.shape[1]):
        df[f"theta_{j}"] = theta[:, j]
    keys = ["D_act", "Tend_disp", "tau_rep_disp", "ST_projected_L2"]
    V = sum(zs_train.z(k, df[k].values) for k in keys)
    df["V_rec"], df["m_proxy"] = V, -V
    return df


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s40, s42, s32 = _load_script("40_descriptors.py"), _load_script("42_classification.py"), _load_script("32_oracle_fold10.py")
    from igraphecg.evaluation.stability import TrainZScorer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    data = load_processed(NPZ)
    signals = data["signal_12lead"].astype(np.float32)
    label, fold = data["label"].astype(int), data["fold"].astype(int)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals).astype(np.float32)
    n_t = signals.shape[2]
    fs = int(data["sampling_rate"])
    c_rep = 2.0
    enc, dec = load_encoder_decoder(repro.lineage.checkpoint(42), n_t, 3, dev,
                                    scaler={"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    center, radius = dec.pspace.center.cpu().numpy(), dec.pspace.radius.cpu().numpy()
    te = np.where(fold == 10)[0]
    y_te = scaled[te]
    y_te_phys = signals[te]

    # ---- classifiers refitted exactly as in scripts/42 on the registered feature table ------
    feat = pd.read_csv(LOCKED / "tables" / "round3_features.csv")
    stab = pd.read_csv(LOCKED / "tables" / "stability_proxy_values.csv")
    feat["m_proxy"], feat["V_rec"] = stab["m_proxy"].values, stab["V_rec"].values
    tr = feat["fold"].values <= 8
    y_tr = feat["label"].values.astype(int)[tr]
    clfs = {}
    for g in ("C0_theta", "C5_theta_rep_st_stab"):
        cols = [c for c in s42.GROUPS[g] if c in feat.columns]
        X = np.nan_to_num(feat[cols].values.astype(np.float32))
        sc = StandardScaler().fit(X[tr])
        clf = s42.make_clf("xgboost", SEED).fit(sc.transform(X[tr]), y_tr)
        clfs[g] = (cols, sc, clf)
    # z-scorer of the stability proxy on training folds (as scripts/41)
    zs_train = TrainZScorer().fit({k: feat[k].values for k in ["D_act", "Tend_disp", "tau_rep_disp", "ST_projected_L2"]}, tr)

    def predict(df):
        out = {}
        for g, (cols, sc, clf) in clfs.items():
            X = np.nan_to_num(df[cols].values.astype(np.float32))
            out[g] = clf.predict_proba(sc.transform(X))
        return out

    def macro_auroc(P, y):
        return float(np.mean([roc_auc_score((y == c).astype(int), P[:, c]) for c in range(N)]))

    # ---- baseline: encoder estimates on fold 10, features recomputed by the same builder ----
    theta_enc, _ = infer_theta(enc, dec, y_te, device=dev)
    with torch.no_grad():
        yhat_enc = dec.forward(torch.from_numpy(theta_enc).to(dev))[0].cpu().numpy()
    df0 = features_for(theta_enc, yhat_enc, y_te_phys, dec, s40, zs_train, c_rep, fs, scaler)
    P0 = predict(df0)
    y_true = label[te]
    base = {g: {"macro_auroc": macro_auroc(P0[g], y_true), "argmax": P0[g].argmax(1)} for g in P0}
    nrmse0 = nrmse(y_te, yhat_enc)
    print(f"[W1S] baseline C0 {base['C0_theta']['macro_auroc']:.4f}  C5 {base['C5_theta_rep_st_stab']['macro_auroc']:.4f}  "
          f"median NRMSE {np.median(nrmse0):.4f}  ({time.time()-t0:.0f}s)")

    # ---- population FIM on the design subset, eigendirections in radius-scaled coordinates -----
    idx = fixed_subset(label, fold)
    th_sub, _ = infer_theta(enc, dec, scaled[idx], device=dev)
    F = population_fim(dec, torch.from_numpy(th_sub), dec.pspace.radius, dev)
    nu, V = np.linalg.eigh(F + LAM * np.eye(F.shape[0]))          # ascending
    print(f"[W1S] FIM eigenvalues: min {nu[0]:.3g} max {nu[-1]:.3g}; 6th/12th/24th smallest "
          f"{nu[5]:.3g}/{nu[11]:.3g}/{nu[23]:.3g}  ({time.time()-t0:.0f}s)")
    u_enc = (theta_enc - center) / radius                           # radius units, |u| <= 1

    rows = []
    conditions = [(f"least informative {m}", V[:, :m], m) for m in M_NULL] + [("most informative 6", V[:, -6:], 6)]
    for cname, basis, m in conditions:
        for step in STEPS:
            flips = {g: [] for g in clfs}
            dp = {g: [] for g in clfs}
            aur = {g: [] for g in clfs}
            disp, dn = [], []
            for k in range(K_DRAWS):
                a = rng.normal(0.0, step, size=(len(te), m))
                u = np.clip(u_enc + a @ basis.T, -1.0, 1.0)
                th = (center + radius * u).astype(np.float32)
                with torch.no_grad():
                    yh = dec.forward(torch.from_numpy(th).to(dev))[0].cpu().numpy()
                df = features_for(th, yh, y_te_phys, dec, s40, zs_train, c_rep, fs, scaler)
                P = predict(df)
                disp.append(np.linalg.norm(u - u_enc, axis=1))
                dn.append(np.abs(nrmse(y_te, yh) - nrmse0))
                for g in clfs:
                    flips[g].append((P[g].argmax(1) != base[g]["argmax"]).mean())
                    dp[g].append(np.abs(P[g][np.arange(len(te)), y_true] - P0[g][np.arange(len(te)), y_true]).mean())
                    aur[g].append(macro_auroc(P[g], y_true))
            row = {"condition": cname, "n_directions": m, "step_radius_units": step, "n_draws": K_DRAWS,
                   "eigenvalue_max_in_set": float(nu[m - 1] if "least" in cname else nu[-1]),
                   "eigenvalue_min_in_set": float(nu[0] if "least" in cname else nu[-6]),
                   "median_displacement_radius_units": float(np.median(np.concatenate(disp))),
                   "median_abs_delta_nrmse": float(np.median(np.concatenate(dn))),
                   "p90_abs_delta_nrmse": float(np.percentile(np.concatenate(dn), 90))}
            for g, short in (("C0_theta", "C0"), ("C5_theta_rep_st_stab", "C5")):
                row[f"{short}_flip_rate"] = float(np.mean(flips[g]))
                row[f"{short}_mean_abs_delta_p_true"] = float(np.mean(dp[g]))
                row[f"{short}_macro_auroc"] = float(np.mean(aur[g]))
                row[f"{short}_delta_macro_auroc"] = float(np.mean(aur[g]) - base[g]["macro_auroc"])
            rows.append(row)
            print(f"[W1S] {cname:22s} step {step}: disp {row['median_displacement_radius_units']:.3f}  "
                  f"dNRMSE {row['median_abs_delta_nrmse']:.4f}  C0 flip {row['C0_flip_rate']:.3f} dAUROC {row['C0_delta_macro_auroc']:+.4f}  "
                  f"C5 flip {row['C5_flip_rate']:.3f} dAUROC {row['C5_delta_macro_auroc']:+.4f}  ({time.time()-t0:.0f}s)")
    pert = pd.DataFrame(rows)
    pert.insert(0, "baseline_C5_macro_auroc", base["C5_theta_rep_st_stab"]["macro_auroc"])
    pert.insert(0, "baseline_C0_macro_auroc", base["C0_theta"]["macro_auroc"])
    pert.to_csv(OUT / "W1S_perturbation.csv", index=False, float_format="%.6g")

    # ---- alternative inverse: per-record refit with the locked decoder fixed -----------------
    cfg = {"lr_theta": 0.02, "lr_H": 0.005, "max_epochs": 20000, "early_stop_patience": 150}
    masks = {k: v.to(dev) for k, v in build_masks(dec.t).items()}
    weights = {"rec": 1.0, "slope": 0.2, "qrs": 1.0, "st": 0.5, "t": 0.5, "phys": 0.05, "H": 0.001}
    yt = torch.tensor(y_te, device=dev)
    z = torch.zeros(len(te), theta_enc.shape[1], device=dev, requires_grad=True)
    dec_fit = load_encoder_decoder(repro.lineage.checkpoint(42), n_t, 3, dev,
                                   scaler={"median": data["scaler_median"], "iqr": data["scaler_iqr"]})[1]
    z_fit, best, ep = s32.fit(dec_fit, z, yt, masks, weights, cfg, False, dev)
    with torch.no_grad():
        theta_alt = dec_fit.pspace.theta(z_fit).cpu().numpy()
        yhat_alt = dec_fit.forward(torch.from_numpy(theta_alt).to(dev))[0].cpu().numpy()
    print(f"[W1S] refit: median NRMSE {best:.4f} after {ep} epochs ({time.time()-t0:.0f}s)")
    df_alt = features_for(theta_alt, yhat_alt, y_te_phys, dec, s40, zs_train, c_rep, fs, scaler)
    P_alt = predict(df_alt)
    tiers = pd.read_csv(OUT / "W1G_parameter_table.csv", encoding="utf-8-sig")
    tiers = tiers.rename(columns={"index": "param_index", "tier_recommended": "tier"})[["param_index", "name", "tier"]]
    prow = []
    for j in range(theta_enc.shape[1]):
        r = float(np.corrcoef(theta_enc[:, j], theta_alt[:, j])[0, 1])
        rel = float(np.median(np.abs(theta_alt[:, j] - theta_enc[:, j]) / radius[j]))
        prow.append({"param_index": j, "pearson_r_encoder_vs_refit": r, "median_abs_diff_radius_units": rel})
    alt = pd.DataFrame(prow)
    alt = alt.merge(tiers, on="param_index", how="left")
    alt.to_csv(OUT / "W1S_alt_inverse_params.csv", index=False, float_format="%.6g")

    head = {
        "test_records": int(len(te)), "design_subset": int(len(idx)), "lambda": LAM,
        "fim_eigenvalues_smallest_6": [float(x) for x in nu[:6]], "fim_eigenvalues_largest_6": [float(x) for x in nu[-6:]],
        "baseline": {"C0_macro_auroc": base["C0_theta"]["macro_auroc"], "C5_macro_auroc": base["C5_theta_rep_st_stab"]["macro_auroc"],
                     "median_nrmse": float(np.median(nrmse0))},
        "alternative_inverse": {
            "median_nrmse_refit": float(np.median(nrmse(y_te, yhat_alt))), "epochs": int(ep),
            "C0_macro_auroc_refit_features": macro_auroc(P_alt["C0_theta"], y_true),
            "C5_macro_auroc_refit_features": macro_auroc(P_alt["C5_theta_rep_st_stab"], y_true),
            "C0_argmax_agreement": float((P_alt["C0_theta"].argmax(1) == base["C0_theta"]["argmax"]).mean()),
            "C5_argmax_agreement": float((P_alt["C5_theta_rep_st_stab"].argmax(1) == base["C5_theta_rep_st_stab"]["argmax"]).mean()),
            "median_param_r_by_tier": ({t: float(alt[alt.tier == t]["pearson_r_encoder_vs_refit"].median()) for t in sorted(alt.tier.dropna().unique())}
                                       if "tier" in alt.columns else None),
        },
    }
    (OUT / "W1S_headline.json").write_text(json.dumps(head, indent=2), encoding="utf-8")
    print(json.dumps(head["alternative_inverse"], indent=2))
    print(f"[W1S] done ({time.time()-t0:.0f}s) -> {OUT}")


if __name__ == "__main__":
    main()
