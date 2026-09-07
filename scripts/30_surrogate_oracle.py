"""Round 2 Step 2: per-sample oracle fitting.

Optimize the per-sample parameters z_n directly on a small subset (with a shared low-rank
lead field H), to check whether the surrogate decoder can express the real median beat.
The encoder is not trained here.

Run:
    python scripts/30_surrogate_oracle.py --config configs/v1fix/oracle_s42.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igraphecg import repro   # output keys in configs are output-root relative

from igraphecg.data.dataset import load_processed
from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.evaluation.recon_metrics import compute_recon_metrics
from igraphecg.evaluation.plots import (plot_heatmap, plot_param_histograms, plot_recon_compare)
from igraphecg.utils.config import parse_args_with_config
from igraphecg.utils.logging import get_logger
from igraphecg.utils.paths import PROJECT_ROOT, ensure_dirs
from igraphecg.utils.seed import set_seed

log = get_logger("oracle")


def select_subset(labels, fold, per_class, split_fold, seed):
    rng = np.random.default_rng(seed)
    idx = []
    mask = (fold <= 8) if split_fold == "train" else (fold == 9)
    for c in range(len(TARGET_CLASSES)):
        pool = np.where((labels == c) & mask)[0]
        take = pool if len(pool) <= per_class else rng.choice(pool, per_class, replace=False)
        idx.append(take)
    return np.concatenate(idx)


def main():
    import torch

    from igraphecg.models.surrogate_decoder import PARAM_DIM, SurrogateDecoder

    _, cfg = parse_args_with_config("Round2 oracle fitting")
    ensure_dirs()
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device = {device}")

    data = load_processed(PROJECT_ROOT / cfg["npz"])
    signals = data["signal_12lead"].astype(np.float32)   # physical mV
    labels = data["label"].astype(int)
    fold = data["fold"].astype(int)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals)

    tr_idx = select_subset(labels, fold, cfg["subset_train_per_class"], "train", cfg["seed"])
    va_idx = select_subset(labels, fold, cfg["subset_val_per_class"], "val", cfg["seed"] + 1)
    log.info(f"oracle subset: train={len(tr_idx)} val={len(va_idx)}")

    all_idx = np.concatenate([tr_idx, va_idx])
    y = torch.tensor(scaled[all_idx], device=device)   # [M,12,T] scaled
    n_tr = len(tr_idx)

    # scaled_derivation: the same B1-fix gate as scripts/05 -- when True the limb leads are
    # derived with the affine coefficients correct for the scaled space; default False keeps
    # the historical behaviour bit-for-bit.
    _scaler_for_dec = None
    if cfg.get("scaled_derivation", False):
        _scaler_for_dec = {"median": data["scaler_median"], "iqr": data["scaler_iqr"]}
    dec = SurrogateDecoder(n_t=signals.shape[2], leadfield_rank=cfg["leadfield_rank"],
                           scaler=_scaler_for_dec).to(device)
    z = torch.zeros(len(all_idx), PARAM_DIM, device=device, requires_grad=True)   # init at param center
    from igraphecg.training.losses import build_masks, total_loss
    masks = {k: v.to(device) for k, v in build_masks(dec.t).items()}
    w = cfg["loss_weights"]

    opt = torch.optim.Adam([
        {"params": [z], "lr": cfg["lr_theta"]},
        {"params": [dec.A, dec.B], "lr": cfg["lr_H"]},
    ])

    def val_nrmse():
        with torch.no_grad():
            yh, _ = dec.forward_from_z(z[n_tr:])
            num = torch.sqrt(((y[n_tr:] - yh) ** 2).sum(dim=(1, 2)))
            den = torch.sqrt((y[n_tr:] ** 2).sum(dim=(1, 2))) + 1e-8
            return float((num / den).median().item())

    best, best_state, patience = 1e9, None, 0
    for epoch in range(cfg["max_epochs"]):
        dec.train()
        opt.zero_grad()
        yhat, aux = dec.forward_from_z(z)
        loss, comps = total_loss(y, yhat, z, aux["theta"], dec.H_ind, masks, w)
        loss.backward()
        opt.step()
        if (epoch + 1) % 25 == 0 or epoch == 0:
            vnr = val_nrmse()
            log.info(f"epoch {epoch+1:03d}  loss={loss.item():.4f} rec={comps['rec'].item():.4f} "
                     f"val_median_NRMSE={vnr:.4f}")
            if vnr < best - 1e-4:
                best, patience = vnr, 0
                best_state = (z.detach().clone(), {k: v.detach().clone() for k, v in dec.state_dict().items()})
            else:
                patience += 1
                if patience >= cfg["early_stop_patience"] // 25 + 1:
                    log.info(f"early stop at epoch {epoch+1}")
                    break

    if best_state is not None:
        z = best_state[0].requires_grad_(False)
        dec.load_state_dict(best_state[1])
    log.info(f"best val median NRMSE (scaled) = {best:.4f}")

    # ---- evaluation ----
    out_dir = repro.outputs() / cfg["out_dir"]
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures" / "reconstruction_examples").mkdir(parents=True, exist_ok=True)

    dec.eval()
    with torch.no_grad():
        yhat_all, aux = dec.forward_from_z(z)
    theta = aux["theta"].cpu().numpy()
    yhat_np = yhat_all.cpu().numpy()
    y_np = scaled[all_idx]
    t_sec = dec.t.cpu().numpy()
    t_ms = t_sec * 1000

    # scaled / physical metrics (reported on the val subset)
    vs = slice(n_tr, len(all_idx))
    m_scaled = compute_recon_metrics(y_np[vs], yhat_np[vs], t_sec)
    y_phys = signals[all_idx]
    yhat_phys = scaler.inverse_transform(yhat_np)
    m_phys = compute_recon_metrics(y_phys[vs], yhat_phys[vs], t_sec)
    boundary = dec.pspace.boundary_rate(z[vs])

    rows = [{"space": "scaled", **m_scaled}, {"space": "physical_mV", **m_phys}]
    df = pd.DataFrame(rows)
    df["theta_boundary_rate"] = boundary
    df.to_csv(out_dir / "tables" / "reconstruction_metrics.csv", index=False)
    log.info("\n" + df.to_string(index=False))
    log.info(f"theta boundary rate = {boundary:.3f}")

    # reconstruction examples: 3 per class, taken from the val subset
    val_labels = labels[all_idx][vs]
    for c, cname in enumerate(TARGET_CLASSES):
        cand = np.where(val_labels == c)[0][:3]
        for k, j in enumerate(cand):
            gj = n_tr + j
            plot_recon_compare(y_np[gj], yhat_np[gj], t_ms,
                               out_dir / "figures" / "reconstruction_examples" / f"{cname}_{k}.png",
                               title=f"{cname} oracle recon (scaled) | NRMSE={m_scaled['median_nrmse']:.3f}")
    # parameter histograms + lead-field heatmap
    from igraphecg.models.surrogate_decoder import _SEG
    pnames = []
    for nm, n in _SEG:
        pnames += [f"{nm}{i}" for i in range(n)] if n > 1 else [nm]
    plot_param_histograms(theta, pnames, out_dir / "figures" / "parameter_distributions.png",
                          title="oracle theta distributions")
    plot_heatmap(dec.H_ind.detach().cpu().numpy(), out_dir / "figures" / "leadfield_heatmap.png",
                 title="lead-field H_ind (A@B)",
                 xticks=[f"n{i}" for i in range(8)],
                 yticks=["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"])
    log.info(f"oracle done, outputs -> {out_dir}")


if __name__ == "__main__":
    main()
