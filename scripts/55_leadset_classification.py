"""Controlled lead-masked classification on pre-aligned median beats.

For each lead set: train a lead-set-specific encoder (frozen eikonal-B decoder, full-12 teacher loss,
ONLY the listed leads as encoder input), derive the C5 descriptor set, and report fold-10 macro-AUROC
with a 95% bootstrap CI. 12-lead row = the locked main eikonal-B encoder (no retrain). A random-3-lead
baseline reports mean+/-std over 10 random independent-lead triples. The structured feature set is computed
only from theta and the decoder reconstruction. It never reads unavailable observed channels at inference.

The experiment starts from median beats that were aligned in the common 12-channel preprocessing pipeline.
It is therefore a controlled representation sensitivity analysis, not an end-to-end wearable deployment test.

Run: python PUBLIC/scripts/55_leadset_classification.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from igraphecg import repro
import torch
from torch.utils.data import DataLoader, TensorDataset

from igraphecg.data.dataset import load_processed, split_indices
from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.data.ptbxl_loader import CANONICAL_LEADS
from igraphecg.evaluation.metrics import bootstrap_ci, compute_metrics
from igraphecg.evaluation.repolarization_features import repolarization_param_features, twave_features
from igraphecg.evaluation.st_features import internal_st_features, projected_st_features
from igraphecg.evaluation.stability import surrogate_stability_proxy
from igraphecg.evaluation.round2_io import infer_theta, load_encoder_decoder
from igraphecg.models.phys_features import compute_phys_features
from igraphecg.models.phys_encoder import PhysEncoder
from igraphecg.models.surrogate_decoder import PARAM_DIM
from igraphecg.training.losses import build_masks, total_loss

N = len(TARGET_CLASSES)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = repro.lineage.checkpoint(42)
NPZ = ROOT / "data/processed/ptbxl_medianbeat_clean_100hz.npz"
# PTB-XL patient mapping. This used to read a copy kept under the paper directory, which
# the archive does not ship; that copy was byte-identical (sha256) to ptbxl_database.csv in
# the data root. Resolve it there instead -- no error at import, an actionable one at read.
def _resolve_patient_csv() -> Path | None:
    from igraphecg.utils.paths import find_ptbxl_root
    root = find_ptbxl_root()
    if root is not None and (root / "ptbxl_database.csv").exists():
        return root / "ptbxl_database.csv"
    return None


PATIENT_CSV = _resolve_patient_csv()
# The published location. Resolved lazily -- importing this module must not create directories.
def _default_out():
    return repro.outdir("runs/v1fix_leadclf")
SEED = 42
C_REP = 2.0
W = {"rec": 1.0, "slope": 0.2, "qrs": 1.0, "st": 0.5, "t": 0.5, "phys": 0.05, "H": 0.001}
TR = {"batch_size": 128, "lr": 1e-3, "wd": 1e-4, "max_epochs": 80, "patience": 15}

LI = {n: i for i, n in enumerate(CANONICAL_LEADS)}
SETS = {   # name -> canonical 12-lead channel indices (encoder input)
    "II+V1+V5":  [LI["II"], LI["V1"], LI["V5"]],
    "I+V1+V4":   [LI["I"], LI["V1"], LI["V4"]],
    "I+V1+V5":   [LI["I"], LI["V1"], LI["V5"]],
    "II":        [LI["II"]],
}
INDEP = [LI[n] for n in ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")]

THETA = [f"theta_{i}" for i in range(47)]
REP = ["APD_mean", "APD_disp", "APD_cv", "tau_rep_mean", "tau_rep_disp", "tau_rep_cv",
       "Tend_mean", "Tend_disp", "Tend_cv", "Sync_rep", "Tpeak_maxabs", "Tarea_mean",
       "Tabsarea_mean", "Tasym_mean", "Tlowamp_mean", "T_sign_discordance_rate"]
ST = ["ST_projected_L2", "ST_projected_max_abs", "ST_projected_anterior_leads",
      "ST_projected_lateral_leads", "ST_projected_inferior_leads", "ST_projected_sign_discordance"]
STAB = ["m_proxy", "V_rec"]


def build_c5(theta, reconstructed_signals, H_ind, fs, train_mask):
    """Structured features from theta and its decoder reconstruction only."""
    feats = {}
    old = compute_phys_features(torch.from_numpy(theta.astype(np.float32)))
    feats["D_act"] = old["D_act"].numpy()
    feats.update(repolarization_param_features(theta, c_rep=C_REP))
    feats.update(twave_features(reconstructed_signals, fs=fs))
    feats.update(internal_st_features(theta))
    feats.update(projected_st_features(theta, H_ind))
    feats.update(surrogate_stability_proxy(feats, train_mask))
    cols = {f"theta_{j}": theta[:, j] for j in range(theta.shape[1])}
    for k in REP + ST + STAB:
        cols[k] = feats[k]
    return pd.DataFrame(cols)


def classify(df, label, fold, patient):
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    tr, te = fold <= 8, fold == 10
    X = np.nan_to_num(df.values.astype(np.float32))
    sc = StandardScaler().fit(X[tr]); X = sc.transform(X)
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, objective="multi:softprob", num_class=N,
                            tree_method="hist", eval_metric="mlogloss", random_state=SEED, n_jobs=-1)
    clf.fit(X[tr], label[tr]); prob = clf.predict_proba(X[te])
    m = compute_metrics(label[te], prob, N)
    lo, hi = bootstrap_ci(label[te], prob, N, "macro_auroc", groups=patient[te])
    return m, lo, hi, prob


@torch.no_grad()
def reconstruct_physical(dec, theta, scaler):
    yhat = []
    for s in range(0, len(theta), 1024):
        th = torch.from_numpy(theta[s:s + 1024].astype(np.float32)).to(DEV)
        yhat.append(dec.forward(th)[0].cpu().numpy())
    return scaler.inverse_transform(np.concatenate(yhat))


def train_leadset_encoder(scaled, idx, leads, dec, seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    enc = PhysEncoder(in_ch=len(leads), out_dim=PARAM_DIM).to(DEV)
    for p in dec.parameters():
        p.requires_grad_(False)
    masks = {k: v.to(DEV) for k, v in build_masks(dec.t).items()}
    opt = torch.optim.AdamW(enc.parameters(), lr=TR["lr"], weight_decay=TR["wd"])
    use_amp = DEV.type == "cuda"; gs = torch.amp.GradScaler("cuda", enabled=use_amp)
    full = torch.from_numpy(scaled).to(DEV); lead_t = torch.tensor(leads, device=DEV)
    dl = DataLoader(TensorDataset(torch.from_numpy(idx["train"].astype(np.int64))),
                    batch_size=TR["batch_size"], shuffle=True)

    @torch.no_grad()
    def val_nrmse():
        enc.eval(); ii = idx["val"]; num = []
        for s in range(0, len(ii), 512):
            b = ii[s:s + 512]; yh, _ = dec.forward_from_z(enc(full[b][:, lead_t]))
            num.append((torch.sqrt(((full[b] - yh) ** 2).sum((1, 2))) /
                        (torch.sqrt((full[b] ** 2).sum((1, 2))) + 1e-8)).cpu().numpy())
        return float(np.median(np.concatenate(num)))

    best, best_state, patience = 1e9, None, 0
    for ep in range(TR["max_epochs"]):
        enc.train()
        for (b,) in dl:
            b = b.to(DEV); opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                z = enc(full[b][:, lead_t]); yhat, aux = dec.forward_from_z(z)
                loss, _ = total_loss(full[b], yhat, z, aux["theta"], dec.H_ind, masks, W)
            gs.scale(loss).backward(); gs.step(opt); gs.update()
        vn = val_nrmse()
        if vn < best - 1e-4:
            best, patience, best_state = vn, 0, {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
        else:
            patience += 1
            if patience >= TR["patience"]:
                break
    enc.load_state_dict(best_state); enc.eval()
    return enc, best, ep + 1


@torch.no_grad()
def encode_theta(enc, dec, scaled, leads):
    lead_t = torch.tensor(leads, device=DEV); th = []
    for s in range(0, len(scaled), 512):
        xin = torch.from_numpy(scaled[s:s + 512][:, leads].astype(np.float32)).to(DEV)
        th.append(dec.pspace.theta(enc(xin)).cpu().numpy())
    return np.concatenate(th)


def main(out=None):
    OUT = Path(out) if out else _default_out()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_processed(NPZ)
    signals = data["signal_12lead"].astype(np.float32); fs = int(data["sampling_rate"]); n_t = signals.shape[2]
    label = data["label"].astype(int); fold = data["fold"].astype(int)
    record_id = data["record_id"].astype(np.int64)
    if PATIENT_CSV is None:
        raise FileNotFoundError(
            "PTB-XL patient metadata (ptbxl_database.csv) not found under the data root. "
            "Place the PTB-XL v1.0.3 release under data/ptbxl/raw/ (see data/README.md) "
            "or set ECG_DATA_DIR to the directory holding it.")
    db = pd.read_csv(PATIENT_CSV, usecols=["ecg_id", "patient_id"]).set_index("ecg_id")
    patient_s = db.reindex(record_id)["patient_id"]
    if patient_s.isna().any():
        raise RuntimeError("official patient metadata does not cover every processed record")
    patient = patient_s.to_numpy(dtype=np.int64)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals); idx = split_indices(fold); train_mask = fold <= 8
    enc_main, dec = load_encoder_decoder(CKPT, n_t, 3, DEV)
    H_ind = (dec.A @ dec.B).detach().cpu().numpy()

    rows, perclass_rows = [], []

    def evaluate(name, kind, theta, reconstructed, n_leads, val_nrmse=np.nan, epochs=0):
        groups = {
            "theta_only": pd.DataFrame({f"theta_{j}": theta[:, j] for j in range(theta.shape[1])}),
            "theta_plus_reconstruction": build_c5(theta, reconstructed, H_ind, fs, train_mask),
        }
        for group, frame in groups.items():
            metrics, lo, hi, prob = classify(frame, label, fold, patient)
            row = {"lead_set": name, "n_leads": n_leads, "feature_group": group,
                   "macro_auroc": metrics["macro_auroc"], "macro_auprc": metrics["macro_auprc"],
                   "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["balanced_accuracy"],
                   "ci_lo": lo, "ci_hi": hi, "kind": kind,
                   "val_nrmse": val_nrmse, "epochs": epochs,
                   "preprocessing_scope": "common_full_record_alignment",
                   "bootstrap_unit": "patient", "n_boot": 1000}
            rows.append(row)
            cm = metrics["confusion_matrix"]
            for class_index, class_name in enumerate(TARGET_CLASSES):
                support = int(cm[class_index].sum())
                recall = float(cm[class_index, class_index] / support) if support else np.nan
                perclass_rows.append({"lead_set": name, "feature_group": group,
                                      "class": class_name, "support": support, "recall": recall})
            print(f"[{name}/{group}] macro-AUROC={metrics['macro_auroc']:.4f} "
                  f"[{lo:.3f},{hi:.3f}] bACC={metrics['balanced_accuracy']:.3f}")
        return rows[-1]["macro_auroc"]

    # 12-lead = locked main encoder (no retrain)
    th12, _ = infer_theta(enc_main, dec, scaled, device=DEV)
    rec12 = reconstruct_physical(dec, th12, scaler)
    evaluate("12 displayed channels", "main", th12, rec12, 12)

    for name, leads in SETS.items():
        enc, vn, eps = train_leadset_encoder(scaled, idx, leads, dec)
        th = encode_theta(enc, dec, scaled, leads)
        rec = reconstruct_physical(dec, th, scaler)
        evaluate(name, "specific", th, rec, len(leads), vn, eps)

    # random 3-lead baseline
    rng = np.random.default_rng(SEED); rand_a = []
    for r in range(10):
        leads = sorted(rng.choice(INDEP, size=3, replace=False).tolist())
        enc, vn, eps = train_leadset_encoder(scaled, idx, leads, dec, seed=SEED + r)
        th = encode_theta(enc, dec, scaled, leads)
        rec = reconstruct_physical(dec, th, scaler)
        names = "+".join(CANONICAL_LEADS[i] for i in leads)
        a = evaluate(f"random3_{names}", "random", th, rec, 3, vn, eps)
        rand_a.append(a)
    rmean, rstd = float(np.mean(rand_a)), float(np.std(rand_a, ddof=1))
    rows.append({"lead_set": "random 3-lead (mean±std)", "n_leads": 3,
                 "feature_group": "theta_plus_reconstruction", "macro_auroc": rmean,
                 "ci_lo": rmean - rstd, "ci_hi": rmean + rstd, "kind": "random_summary",
                 "preprocessing_scope": "common_full_record_alignment"})
    print(f"[random 3-lead] structured mean±std = {rmean:.4f} ± {rstd:.4f}  (n=10)")

    pd.DataFrame(rows).to_csv(OUT / "leadclf_m10.csv", index=False)
    pd.DataFrame(perclass_rows).to_csv(OUT / "leadclf_m10_perclass.csv", index=False)
    print(f"\nSaved -> {OUT/'leadclf_m10.csv'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="output directory; default = the published run under the output root")
    main(out=ap.parse_args().out)
