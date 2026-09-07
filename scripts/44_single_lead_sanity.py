"""Round 5 Task C: re-check of the single-lead (II) θ-AUROC.

Variants (decoder frozen throughout, only the lead-II encoder is trained):
  L1-baseline-full   : default capacity, full-12 teacher (reproduces Round 4's 0.843)
  L1-small-full      : encoder width halved, full-12 teacher
  L1-regularized-full: default capacity + strong dropout + weight_decay, full-12 teacher
  L1-no-teacher-obs  : default capacity, observed lead-II loss only (no full-12 teacher)
Tests whether the high θ-AUROC is an artefact of encoder capacity / teacher amortization.

Run: python scripts/44_single_lead_sanity.py --config configs/v1fix/round5_s42.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igraphecg import repro   # output keys in configs are output-root relative

from igraphecg.data.dataset import load_processed, split_indices
from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.evaluation.metrics import bootstrap_ci, compute_metrics
from igraphecg.evaluation.recon_metrics import per_sample_lead_corr, per_sample_nrmse
from igraphecg.evaluation.repolarization_features import twave_features
from igraphecg.evaluation.round2_io import load_encoder_decoder
from igraphecg.evaluation.st_features import projected_st_features
from igraphecg.evaluation.stats_tests import cliffs_vs_norm
from igraphecg.models.phys_features import compute_phys_features
from igraphecg.utils.config import parse_args_with_config
from igraphecg.utils.logging import get_logger
from igraphecg.utils.paths import PROJECT_ROOT, ensure_dirs
from igraphecg.utils.seed import set_seed

log = get_logger("l1_sanity")
LEAD_II = [1]
N = len(TARGET_CLASSES)


def train_l1(scaled, idx, dec, cfg, device, widths, dropout, wd, mode):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from igraphecg.models.phys_encoder import PhysEncoder
    from igraphecg.models.surrogate_decoder import PARAM_DIM
    from igraphecg.training.losses import build_masks, total_loss
    tr = cfg["train"]; w = dict(cfg["loss_weights"])
    enc = PhysEncoder(in_ch=1, out_dim=PARAM_DIM, widths=widths, dropout=dropout).to(device)
    for p in dec.parameters():
        p.requires_grad_(False)
    masks = {k: v.to(device) for k, v in build_masks(dec.t).items()}
    opt = torch.optim.AdamW(enc.parameters(), lr=tr["lr_encoder"], weight_decay=wd)
    use_amp = bool(tr.get("amp", True)) and device.type == "cuda"
    gs = torch.amp.GradScaler("cuda", enabled=use_amp)
    full = torch.from_numpy(scaled).to(device); lead_t = torch.tensor(LEAD_II, device=device)
    dl = DataLoader(TensorDataset(torch.from_numpy(idx["train"].astype(np.int64))),
                    batch_size=tr["batch_size"], shuffle=True)

    @torch.no_grad()
    def val_nrmse():
        enc.eval(); ii = idx["val"]; num = []
        for s in range(0, len(ii), 512):
            b = ii[s:s+512]; yh, _ = dec.forward_from_z(enc(full[b][:, lead_t]))
            tgt = full[b][:, lead_t] if mode == "obs" else full[b]
            yy = yh[:, lead_t] if mode == "obs" else yh
            num.append((torch.sqrt(((tgt-yy)**2).sum((1,2)))/(torch.sqrt((tgt**2).sum((1,2)))+1e-8)).cpu().numpy())
        return float(np.median(np.concatenate(num)))

    best, best_state, patience = 1e9, None, 0
    for ep in range(tr["max_epochs"]):
        enc.train()
        for (b,) in dl:
            b = b.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                z = enc(full[b][:, lead_t]); yhat, aux = dec.forward_from_z(z)
                tgt = full[b][:, lead_t] if mode == "obs" else full[b]
                yy = yhat[:, lead_t] if mode == "obs" else yhat
                loss, _ = total_loss(tgt, yy, z, aux["theta"], dec.H_ind, masks, w)
            gs.scale(loss).backward(); gs.step(opt); gs.update()
        vn = val_nrmse()
        if vn < best - 1e-4:
            best, patience, best_state = vn, 0, {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
        else:
            patience += 1
            if patience >= tr["early_stop_patience"]:
                break
    enc.load_state_dict(best_state); enc.eval()
    return enc


def evaluate(enc, dec, scaled, signals, scaler, label, fold, H_ind, fs, device, seed):
    import torch
    lead_t = torch.tensor(LEAD_II, device=device)
    th, yh = [], []
    with torch.no_grad():
        for s in range(0, len(scaled), 512):
            xin = torch.from_numpy(scaled[s:s+512][:, LEAD_II].astype(np.float32)).to(device)
            z = enc(xin); y12, _ = dec.forward_from_z(z)
            th.append(dec.pspace.theta(z).cpu().numpy()); yh.append(y12.cpu().numpy())
    theta = np.concatenate(th); yh = np.concatenate(yh)
    te = fold == 10
    obs_corr = float(np.median(per_sample_lead_corr(scaled[te][:, LEAD_II], yh[te][:, LEAD_II])))
    full_nrmse = float(np.median(per_sample_nrmse(scaled[te], yh[te])))
    from sklearn.preprocessing import StandardScaler
    trm = fold <= 8
    sc = StandardScaler().fit(theta[trm]); X = sc.transform(np.nan_to_num(theta))
    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, objective="multi:softprob", num_class=N,
                                tree_method="hist", eval_metric="mlogloss", random_state=seed, n_jobs=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=400, random_state=seed)
    clf.fit(X[trm], label[trm]); prob = clf.predict_proba(X[te])
    auroc = compute_metrics(label[te], prob, N)["macro_auroc"]
    d_act = compute_phys_features(torch.from_numpy(theta[te].astype(np.float32)))["D_act"].numpy()
    tw = twave_features(scaler.inverse_transform(yh[te]), fs=fs)
    stp = projected_st_features(theta[te], H_ind)["ST_projected_max_abs"]
    return {"theta_macro_auroc": auroc, "observed_corr": obs_corr, "full12_nrmse": full_nrmse,
            "D_act_CD": cliffs_vs_norm(d_act, label[te], "CD"),
            "STTC_Tsign": cliffs_vs_norm(tw["T_sign_discordance_rate"], label[te], "STTC"),
            "MI_STproj": cliffs_vs_norm(stp, label[te], "MI")}


def main():
    _, cfg = parse_args_with_config("Round5 single-lead sanity")
    ensure_dirs(); set_seed(cfg["seed"])
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_processed(PROJECT_ROOT / cfg["npz"])
    signals = data["signal_12lead"].astype(np.float32); fs = int(data["sampling_rate"])
    label = data["label"].astype(int); fold = data["fold"].astype(int)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals); idx = split_indices(fold)
    _, dec = load_encoder_decoder(repro.outputs() / cfg["encoder_ckpt"], signals.shape[2], cfg["leadfield_rank"], device)
    H_ind = (dec.A @ dec.B).detach().cpu().numpy()
    out = repro.outputs() / cfg["out_dir"]; (out / "tables").mkdir(parents=True, exist_ok=True)

    variants = [
        ("L1-baseline-full", (32, 64, 128, 256), 0.2, 1e-4, "full"),
        ("L1-small-full", (16, 32, 64, 128), 0.2, 1e-4, "full"),
        ("L1-regularized-full", (32, 64, 128, 256), 0.5, 1e-3, "full"),
        ("L1-no-teacher-obs", (32, 64, 128, 256), 0.2, 1e-4, "obs"),
    ]
    rows = []
    for name, widths, dropout, wd, mode in variants:
        set_seed(cfg["seed"])
        log.info(f"training {name} (widths={widths} dropout={dropout} wd={wd} mode={mode}) ...")
        enc = train_l1(scaled, idx, dec, cfg, device, widths, dropout, wd, mode)
        m = evaluate(enc, dec, scaled, signals, scaler, label, fold, H_ind, fs, device, cfg["seed"])
        rows.append({"variant": name, "mode": mode, "fim_effrank_L1": 2.814, **m})
        log.info(f"  -> theta_AUROC={m['theta_macro_auroc']:.4f} full12_NRMSE={m['full12_nrmse']:.3f} "
                 f"obs_corr={m['observed_corr']:.3f} D_act={m['D_act_CD']:.2f} T_sign={m['STTC_Tsign']:.2f} MI={m['MI_STproj']:.2f}")
    df = pd.DataFrame(rows); df.to_csv(out / "tables" / "single_lead_sanity.csv", index=False)
    log.info("\n" + df[["variant","theta_macro_auroc","full12_nrmse","observed_corr","STTC_Tsign","MI_STproj"]].to_string(index=False))
    log.info(f"Task C done -> {out}")


if __name__ == "__main__":
    main()
