"""W1O -- Teacher-free three-lead encoders (Reviewer 1 comment 5).

Table 6's reduced-lead AUROCs come from lead-set-specific encoders trained with a
full-12 teacher loss (scripts/55_leadset_classification.py driven by scripts/55_leadset_classification.py on the
B1-fixed d1s_r3_s42 checkpoint).  This script quantifies how much of that number is
teacher-supervision optimism by retraining the same encoders with the loss restricted to
the OBSERVED leads (the "obs" mode of the former lead-set retrain script, since removed; the mode lives in W1O_teacherfree_leadsets.py and the
L1-no-teacher-obs variant of scripts/44_single_lead_sanity.py, Table S2), and evaluating
descriptor-based four-class classification exactly as Table 6 does.

Arms per montage (montages: II+V1+V5, I+V1+V4, II):
  teacher_stock   full-12 teacher loss, stock encoder init          <- reproduces Table 6
  teacher_stable  full-12 teacher loss, head_init_scale + grad_clip <- init-matched teacher control
  obs_stock       observed-lead loss,   stock encoder init          <- Table S2 precedent, exact
  obs_stable      observed-lead loss,   head_init_scale + grad_clip <- headline teacher-free

The "stable" arms carry the documented initialisation fix (configs/d1s_r3_s42.yaml:
head_init_scale=0.01, grad_clip=1.0), which is what removed the encoder's
tanh-saturation divergence failure mode.

Read-only on the frozen lineage; every artefact goes to runs/v7rev_stats/ and
checkpoints/v7rev_teacherfree/.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro
REPO = repro.repo()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import torch
from torch.utils.data import DataLoader, TensorDataset

m = importlib.import_module("55_leadset_classification")            # reuse the Table-6 pipeline verbatim

from igraphecg.data.dataset import load_processed, split_indices
from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.data.ptbxl_loader import CANONICAL_LEADS
from igraphecg.evaluation.identifiability import compute_fim, fim_metrics, lead_rows
from igraphecg.evaluation.recon_metrics import per_sample_lead_corr, per_sample_nrmse
from igraphecg.evaluation.round2_io import infer_theta, load_encoder_decoder
from igraphecg.models.phys_encoder import PhysEncoder
from igraphecg.models.surrogate_decoder import PARAM_DIM
from igraphecg.training.losses import build_masks, total_loss

OUT = repro.outdir("runs/v7rev_stats")
CKPT_OUT = repro.outputs() / "checkpoints" / "v7rev_teacherfree"
OUT.mkdir(parents=True, exist_ok=True)
CKPT_OUT.mkdir(parents=True, exist_ok=True)

# --- frozen lineage inputs (read-only) ---------------------------------------------
CKPT = repro.lineage.checkpoint(42)
m.CKPT = CKPT                                        # same override as 55_leadset_classification.py
NPZ = m.NPZ
PATIENT_CSV = m.PATIENT_CSV
SEED = m.SEED                                        # 42
W = m.W                                              # loss weights, byte-for-byte from 55
TR = m.TR                                            # batch 128 / lr 1e-3 / wd 1e-4 / 80 ep / pat 15
DEV = m.DEV
N = m.N

SAT_THRESH = 0.90                                    # scripts/31_surrogate_encoder.py
SAT_SEPARATOR = 0.35                                 # converged / diverged separator
HEAD_INIT_SCALE = 0.01                               # configs/d1s_r3_s42.yaml
GRAD_CLIP = 1.0                                      # configs/d1s_r3_s42.yaml

LI = {n: i for i, n in enumerate(CANONICAL_LEADS)}
MONTAGES = {
    "II+V1+V5": [LI["II"], LI["V1"], LI["V5"]],      # clinical
    "I+V1+V4": [LI["I"], LI["V1"], LI["V4"]],        # observability-optimal
    "II": [LI["II"]],                                # single-lead control
}
ARMS = [  # (arm name, loss mode, head_init_scale, grad_clip)
    ("teacher_stock", "full", 1.0, 0.0),
    ("teacher_stable", "full", HEAD_INIT_SCALE, GRAD_CLIP),
    ("obs_stock", "obs", 1.0, 0.0),
    ("obs_stable", "obs", HEAD_INIT_SCALE, GRAD_CLIP),
]
TABLE6_TEACHER = {"12 leads": 0.901, "II+V1+V5": 0.887, "I+V1+V4": 0.878, "II": 0.844}
FIM_PER_CLASS = 100


# ---------------------------------------------------------------------------------
def fixed_subset(label, fold, per_class=FIM_PER_CLASS, seed=SEED):
    """scripts/43_identifiability.py:subset_idx -- fold-10, 100 per class, seed 42."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(len(TARGET_CLASSES)):
        pool = np.where((label == c) & (fold == 10))[0]
        idx.append(pool if len(pool) <= per_class else rng.choice(pool, per_class, replace=False))
    return np.concatenate(idx)


def train_encoder(scaled, idx, leads, dec, mode, head_init_scale, grad_clip, seed=SEED):
    """scripts/55_leadset_classification.py:train_leadset_encoder + obs mode (the former retrain script / 18) + init fix (05)."""
    torch.manual_seed(seed); np.random.seed(seed)
    enc = PhysEncoder(in_ch=len(leads), out_dim=PARAM_DIM).to(DEV)
    if head_init_scale != 1.0:                        # stabilisation 1: small head init
        with torch.no_grad():
            enc.head[-1].weight.mul_(head_init_scale)
            if enc.head[-1].bias is not None:
                enc.head[-1].bias.zero_()
    for p in dec.parameters():
        p.requires_grad_(False)
    masks = {k: v.to(DEV) for k, v in build_masks(dec.t).items()}
    opt = torch.optim.AdamW(enc.parameters(), lr=TR["lr"], weight_decay=TR["wd"])
    use_amp = DEV.type == "cuda"
    gs = torch.amp.GradScaler("cuda", enabled=use_amp)
    full = torch.from_numpy(scaled).to(DEV)
    lead_t = torch.tensor(leads, device=DEV)
    dl = DataLoader(TensorDataset(torch.from_numpy(idx["train"].astype(np.int64))),
                    batch_size=TR["batch_size"], shuffle=True)

    @torch.no_grad()
    def val_stats():
        """(early-stop NRMSE, observed-lead NRMSE, full-12 NRMSE, sat090) on the val split."""
        enc.eval(); ii = idx["val"]
        n_obs, n_full, sat = [], [], []
        for s in range(0, len(ii), 512):
            b = ii[s:s + 512]
            z = enc(full[b][:, lead_t])
            yh, _ = dec.forward_from_z(z)
            o = full[b][:, lead_t]; oh = yh[:, lead_t]
            n_obs.append((torch.sqrt(((o - oh) ** 2).sum((1, 2))) /
                          (torch.sqrt((o ** 2).sum((1, 2))) + 1e-8)).cpu().numpy())
            n_full.append((torch.sqrt(((full[b] - yh) ** 2).sum((1, 2))) /
                           (torch.sqrt((full[b] ** 2).sum((1, 2))) + 1e-8)).cpu().numpy())
            sat.append((torch.tanh(z).abs() > SAT_THRESH).float().mean().item())
        vo = float(np.median(np.concatenate(n_obs)))
        vf = float(np.median(np.concatenate(n_full)))
        return (vo if mode == "obs" else vf), vo, vf, float(np.mean(sat))

    best, best_state, best_diag, patience, ep = 1e9, None, None, 0, -1
    for ep in range(TR["max_epochs"]):
        enc.train()
        for (b,) in dl:
            b = b.to(DEV); opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                z = enc(full[b][:, lead_t])
                yhat, aux = dec.forward_from_z(z)
                tgt = full[b][:, lead_t] if mode == "obs" else full[b]
                yy = yhat[:, lead_t] if mode == "obs" else yhat
                loss, _ = total_loss(tgt, yy, z, aux["theta"], dec.H_ind, masks, W)
            gs.scale(loss).backward()
            if grad_clip and grad_clip > 0:           # stabilisation 2: gradient clipping
                gs.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(enc.parameters(), grad_clip)
            gs.step(opt); gs.update()
        vsel, vo, vf, vsat = val_stats()
        if vsel < best - 1e-4:
            best, patience = vsel, 0
            best_state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
            best_diag = {"val_nrmse_selected": vsel, "val_nrmse_observed": vo,
                         "val_nrmse_full12": vf, "val_sat090": vsat, "best_epoch": ep + 1}
        else:
            patience += 1
            if patience >= TR["patience"]:
                break
    enc.load_state_dict(best_state); enc.eval()
    best_diag["epochs_run"] = ep + 1
    return enc, best_diag


@torch.no_grad()
def encode_all(enc, dec, scaled, leads):
    """theta (bounded) and scaled-space 12-lead reconstruction from the encoder."""
    lead_t = torch.tensor(leads, device=DEV)
    th, yh = [], []
    for s in range(0, len(scaled), 512):
        xin = torch.from_numpy(scaled[s:s + 512][:, leads].astype(np.float32)).to(DEV)
        z = enc(xin)
        y12, _ = dec.forward_from_z(z)
        th.append(dec.pspace.theta(z).cpu().numpy()); yh.append(y12.cpu().numpy())
    return np.concatenate(th), np.concatenate(yh)


def fim_effrank(dec, theta_subset, radius, leads, n_t):
    rows = torch.tensor(lead_rows(leads, n_t), dtype=torch.long, device=DEV)
    F = compute_fim(dec, torch.from_numpy(theta_subset.astype(np.float32)).to(DEV),
                    radius, observed_rows=rows, device=DEV)
    return fim_metrics(F)


def main():
    t0 = time.time()
    data = load_processed(NPZ)
    signals = data["signal_12lead"].astype(np.float32)
    fs = int(data["sampling_rate"]); n_t = signals.shape[2]
    label = data["label"].astype(int); fold = data["fold"].astype(int)
    record_id = data["record_id"].astype(np.int64)
    db = pd.read_csv(PATIENT_CSV, usecols=["ecg_id", "patient_id"]).set_index("ecg_id")
    patient_s = db.reindex(record_id)["patient_id"]
    if patient_s.isna().any():
        raise RuntimeError("official patient metadata does not cover every processed record")
    patient = patient_s.to_numpy(dtype=np.int64)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals)
    idx = split_indices(fold)
    train_mask = fold <= 8
    te = fold == 10
    enc_main, dec = load_encoder_decoder(CKPT, n_t, 3, DEV)
    H_ind = (dec.A @ dec.B).detach().cpu().numpy()
    radius = dec.pspace.radius
    sub = fixed_subset(label, fold)
    print(f"[setup] N={len(label)} n_t={n_t} fs={fs} test={int(te.sum())} "
          f"fim_subset={len(sub)} device={DEV}", flush=True)

    # --- resume: keep whatever a previous (interrupted) run already finished -------
    csv_path = OUT / "W1O_teacherfree_leadsets.csv"
    pc_path = OUT / "W1O_teacherfree_leadsets_perclass.csv"
    rows, perclass_rows, done = [], [], set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        rows = prev.to_dict("records")
        done = set(zip(prev["lead_set"], prev["arm"]))
        if pc_path.exists():
            perclass_rows = pd.read_csv(pc_path).to_dict("records")
        print(f"[resume] {len(done)} (lead_set, arm) pairs already done: {sorted(done)}", flush=True)

    def evaluate(name, arm, mode, leads, theta, yh_scaled, diag, effrank_row):
        rec_mV = scaler.inverse_transform(yh_scaled)
        obs_corr = float(np.median(per_sample_lead_corr(scaled[te][:, leads], yh_scaled[te][:, leads])))
        full_nrmse = float(np.median(per_sample_nrmse(scaled[te], yh_scaled[te])))
        groups = {
            "theta_only": pd.DataFrame({f"theta_{j}": theta[:, j] for j in range(theta.shape[1])}),
            "theta_plus_reconstruction": m.build_c5(theta, rec_mV, H_ind, fs, train_mask),
        }
        out = {}
        for group, frame in groups.items():
            metrics, lo, hi, _ = m.classify(frame, label, fold, patient)
            row = {
                "lead_set": name, "n_leads": len(leads), "arm": arm, "loss_mode": mode,
                "head_init_scale": diag.get("head_init_scale"), "grad_clip": diag.get("grad_clip"),
                "feature_group": group,
                "macro_auroc": metrics["macro_auroc"], "ci_lo": lo, "ci_hi": hi,
                "macro_auprc": metrics["macro_auprc"], "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "observed_corr": obs_corr, "full12_nrmse": full_nrmse,
                "fim_effective_rank": effrank_row["effective_rank"],
                "fim_rank95": effrank_row["rank95"], "fim_logdet": effrank_row["logdet"],
                "val_nrmse_selected": diag.get("val_nrmse_selected"),
                "val_nrmse_observed": diag.get("val_nrmse_observed"),
                "val_nrmse_full12": diag.get("val_nrmse_full12"),
                "val_sat090": diag.get("val_sat090"),
                "best_epoch": diag.get("best_epoch"), "epochs_run": diag.get("epochs_run"),
                "diverged": diag.get("diverged"),
                "table6_teacher_auroc": TABLE6_TEACHER.get(name, np.nan),
                "bootstrap_unit": "patient", "n_boot": 1000, "seed": SEED,
            }
            rows.append(row)
            out[group] = row
            cm = metrics["confusion_matrix"]
            for ci, cn in enumerate(TARGET_CLASSES):
                support = int(cm[ci].sum())
                perclass_rows.append({
                    "lead_set": name, "arm": arm, "feature_group": group, "class": cn,
                    "support": support,
                    "recall": float(cm[ci, ci] / support) if support else np.nan})
        c5 = out["theta_plus_reconstruction"]
        print(f"  [{name}|{arm}] C5 AUROC={c5['macro_auroc']:.4f} "
              f"[{c5['ci_lo']:.3f},{c5['ci_hi']:.3f}] theta-only={out['theta_only']['macro_auroc']:.4f} "
              f"obs_corr={obs_corr:.3f} full12_NRMSE={full_nrmse:.3f} "
              f"effrank={effrank_row['effective_rank']:.3f} sat090={diag.get('val_sat090'):.3f} "
              f"diverged={diag.get('diverged')}  [{time.time()-t0:.0f}s]", flush=True)

    # ---- control: locked 12-lead encoder, no retrain (Table 6 row 1) -------------
    if ("12 leads", "locked_main") not in done:
        print("[control] locked 12-lead encoder (d1s_r3_s42, no retrain)", flush=True)
        th12, _ = infer_theta(enc_main, dec, scaled, device=DEV)
        yh12 = []
        with torch.no_grad():
            for s in range(0, len(scaled), 512):
                xb = torch.from_numpy(scaled[s:s + 512].astype(np.float32)).to(DEV)
                yh12.append(dec.forward_from_z(enc_main(xb))[0].cpu().numpy())
        yh12 = np.concatenate(yh12)
        er12 = fim_effrank(dec, th12[sub], radius, list(range(12)), n_t)
        print(f"[control] locked-model FIM eff rank, 12 leads = {er12['effective_rank']:.4f} "
              f"(Table 6 seed-42: 4.36)", flush=True)
        locked_effrank = {"12 leads": er12}
        for name, leads in MONTAGES.items():
            er = fim_effrank(dec, th12[sub], radius, leads, n_t)
            locked_effrank[name] = er
            print(f"[control] locked-model FIM eff rank, {name} = {er['effective_rank']:.4f}",
                  flush=True)
        evaluate("12 leads", "locked_main", "n/a", list(range(12)), th12, yh12,
                 {"head_init_scale": None, "grad_clip": None, "diverged": False,
                  "val_sat090": float(np.nan), "best_epoch": 0, "epochs_run": 0}, er12)

        with open(OUT / "W1O_locked_model_fim_effrank.json", "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ("eig_top", "eig_bottom")}
                       for k, v in locked_effrank.items()}, f, indent=2)
    else:
        print("[resume] locked 12-lead control already done, skipping", flush=True)

    # ---- lead-set-specific encoders --------------------------------------------
    for name, leads in MONTAGES.items():
        for arm, mode, his, gc in ARMS:
            if (name, arm) in done:
                print(f"[resume] {name} | {arm} already done, skipping", flush=True)
                continue
            print(f"[train] {name} | {arm} (mode={mode} head_init={his} clip={gc})", flush=True)
            enc, diag = train_encoder(scaled, idx, leads, dec, mode, his, gc)
            diag["head_init_scale"] = his
            diag["grad_clip"] = gc
            diag["diverged"] = bool(diag["val_sat090"] > SAT_SEPARATOR)
            tag = name.replace("+", "_")
            torch.save({"encoder": enc.state_dict(), "leads": leads, "mode": mode,
                        "arm": arm, "head_init_scale": his, "grad_clip": gc,
                        "diag": diag, "base_ckpt": repro.lineage.lock()["checkpoints"][42]["path"]},
                       CKPT_OUT / f"leadset_{tag}_{arm}_best.pt")
            theta, yh = encode_all(enc, dec, scaled, leads)
            er = fim_effrank(dec, theta[sub], radius, leads, n_t)
            evaluate(name, arm, mode, leads, theta, yh, diag, er)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            pd.DataFrame(perclass_rows).to_csv(pc_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    pd.DataFrame(perclass_rows).to_csv(pc_path, index=False)
    print("\n" + df[df.feature_group == "theta_plus_reconstruction"][
        ["lead_set", "arm", "macro_auroc", "ci_lo", "ci_hi", "observed_corr",
         "full12_nrmse", "fim_effective_rank", "val_sat090", "diverged"]].to_string(index=False))
    print(f"\nSaved -> {OUT/'W1O_teacherfree_leadsets.csv'}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
