"""Round 2 Step 3: encoder-amortized fitting.

ECG -> PhysEncoder -> z -> SurrogateDecoder -> reconstructed ECG. Reconstruction only, no
classification. The lead field is frozen for the first leadfield_warmup_epochs and only the
encoder is trained; H is then released with a small learning rate.

Optional stabilization switches (all config-driven, off by default => bit-for-bit identical
to the historical behaviour):
  head_init_scale     scale the encoder's last Linear weight by `scale`, zero its bias
                      (default/1.0 = unchanged)
  grad_clip           clip_grad_norm_ bound (default/<=0 = disabled)
  abort_sat_threshold boundary-saturation abort threshold (default = never abort; see below)
  recon_channels      lead subset scored by the five waveform terms (rec/slope/qrs/st/t)
                      (default/absent = all 12 leads); "non_derived" = the 8 non-derived
                      channels I,II,V1..V6 only, which strips the implicit over-weighting of
                      the limb subspace introduced by III/aVR/aVL/aVF. **Affects the loss
                      only**; every reconstruction metric below is still computed over all 12
                      displayed leads.
Mechanism ported verbatim from experiments/robustness_universality/train_encoder_stable.py.

Diagnostic: each epoch logs the **boundary saturation** on val = fraction of theta coordinates
with |tanh z| > 0.90 (same convention as SAT_THRESH in
experiments/02_observability_control/validation/d1_eval_checkpoints.py).

Run:
    python scripts/31_surrogate_encoder.py --config configs/d1s_r3_s42.yaml
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
from igraphecg.data.ptbxl_loader import CANONICAL_LEADS
from igraphecg.evaluation.plots import (plot_heatmap, plot_loss_curve, plot_param_histograms,
                                  plot_recon_compare)
from igraphecg.evaluation.recon_metrics import (compute_recon_metrics, per_sample_lead_corr,
                                          per_sample_nrmse)
from igraphecg.utils.config import parse_args_with_config
from igraphecg.utils.logging import get_logger
from igraphecg.utils.paths import PROJECT_ROOT, ensure_dirs
from igraphecg.utils.seed import set_seed

log = get_logger("encoder")
N_CLASSES = len(TARGET_CLASSES)

SAT_THRESH = 0.90   # |tanh z| > SAT_THRESH counts as saturated (cf. d1_eval_checkpoints.py)
SAT_SEPARATOR = 0.35   # converged / diverged separator from the D1 evaluation (logging only)
ABORT_SAT_PATIENCE = 5   # consecutive post-warmup epochs above threshold needed to abort


def main():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from igraphecg.models.phys_encoder import PhysEncoder
    from igraphecg.models.surrogate_decoder import PARAM_DIM, SurrogateDecoder
    from igraphecg.training.losses import build_masks, resolve_recon_rows, total_loss

    _, cfg = parse_args_with_config("Round2 encoder fitting")
    ensure_dirs()
    set_seed(cfg.get("seed", 42))
    tr = cfg["train"]
    # --- optional stabilization switches (defaults = off = historical behaviour) ---
    head_init_scale = float(cfg.get("head_init_scale", 1.0))
    grad_clip = float(cfg.get("grad_clip", 0.0) or 0.0)
    abort_sat_threshold = cfg.get("abort_sat_threshold", None)
    # recon_channels absent => recon_rows=None => historical all-12-lead total_loss, bit-identical
    recon_channels = cfg.get("recon_channels", None)
    recon_rows = resolve_recon_rows(recon_channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device = {device}  seed={cfg.get('seed', 42)}  head_init_scale={head_init_scale} "
             f"grad_clip={grad_clip} abort_sat_threshold={abort_sat_threshold}")
    log.info(f"recon_channels={recon_channels!r} -> loss rows="
             f"{'all 12 displayed leads (default)' if recon_rows is None else list(recon_rows)}"
             f"  [evaluation metrics below are ALWAYS over all 12 displayed leads]")

    data = load_processed(PROJECT_ROOT / cfg["npz"])
    signals = data["signal_12lead"].astype(np.float32)
    labels = data["label"].astype(np.int64)
    fold = data["fold"].astype(int)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals)
    idx = split_indices(fold)
    log.info(f"N={len(labels)} train/val/test={len(idx['train'])}/{len(idx['val'])}/{len(idx['test'])}")

    def loader(split, shuffle):
        ii = idx[split]
        ds = TensorDataset(torch.from_numpy(scaled[ii]), torch.from_numpy(ii.astype(np.int64)))
        return DataLoader(ds, batch_size=tr["batch_size"], shuffle=shuffle)
    tl = loader("train", True)

    enc = PhysEncoder(in_ch=12, out_dim=PARAM_DIM).to(device)
    # --- stabilization 1: small head init avoids tanh saturation at start (1.0 = baseline) ---
    if head_init_scale != 1.0:
        with torch.no_grad():
            enc.head[-1].weight.mul_(head_init_scale)
            if enc.head[-1].bias is not None:
                enc.head[-1].bias.zero_()
    # scaled_derivation: True applies the Einthoven-Goldberger relations with the affine
    # coefficients of the **scaled space** (target is scaled); False = old flawed physical coeffs.
    _scaler_for_dec = None
    if cfg.get("scaled_derivation", False):
        _scaler_for_dec = {"median": data["scaler_median"], "iqr": data["scaler_iqr"]}
    # structural ablations (Reviewer 3): both keys absent in the published configs
    if not cfg.get("graph_delays", True):
        from igraphecg.models.surrogate_decoder import ParamSpace
        ParamSpace.GRAPH_DELAYS = False
    dec = SurrogateDecoder(n_t=signals.shape[2], leadfield_rank=cfg["leadfield_rank"],
                           scaler=_scaler_for_dec, direct_12_leads=cfg.get("direct_12_leads", False),
                           graph_delays=cfg.get("graph_delays", True),
                           leadfield_init_scale=cfg.get("leadfield_init_scale", 0.3)).to(device)
    masks = {k: v.to(device) for k, v in build_masks(dec.t).items()}
    w = cfg["loss_weights"]

    opt = torch.optim.AdamW([
        {"params": enc.parameters(), "lr": tr["lr_encoder"]},
        {"params": [dec.A, dec.B], "lr": tr["lr_decoder_global"]},
    ], weight_decay=tr["weight_decay"])
    use_amp = bool(tr.get("amp", True)) and device.type == "cuda"
    gscaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    full = torch.from_numpy(scaled).to(device)

    @torch.no_grad()
    def infer(split):
        enc.eval()
        ii = idx[split]
        zs, yhs = [], []
        for s in range(0, len(ii), 512):
            xb = full[ii[s:s + 512]]
            z = enc(xb)
            yh, _ = dec.forward_from_z(z)
            zs.append(z.cpu().numpy()); yhs.append(yh.cpu().numpy())
        return np.concatenate(zs), np.concatenate(yhs), ii

    def val_nrmse_and_sat():
        """(median NRMSE, boundary saturation) on val; z from infer, no extra RNG/forward pass."""
        z, yh, ii = infer("val")
        return (float(np.median(per_sample_nrmse(scaled[ii], yh))),
                float((np.abs(np.tanh(z)) > SAT_THRESH).mean()))

    history = {"train_loss": [], "val_loss": [], "val_nrmse": [], "val_sat": []}
    best, best_state, patience = 1e9, None, 0
    sat_streak, sat_below_epoch, aborted = 0, None, False
    for epoch in range(tr["max_epochs"]):
        # lead-field warmup: freeze A,B for the first few epochs
        warm = epoch < tr["leadfield_warmup_epochs"]
        dec.A.requires_grad_(not warm); dec.B.requires_grad_(not warm)
        enc.train()
        ep = 0.0
        for xb, _ in tl:
            xb = xb.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                z = enc(xb)
                yhat, aux = dec.forward_from_z(z)
                loss, comps = total_loss(xb, yhat, z, aux["theta"], dec.H_ind, masks, w,
                                         recon_rows=recon_rows,
                                         precordial_rows=dec.precordial_rows)
            gscaler.scale(loss).backward()
            # --- stabilization 2: gradient clipping (disabled when grad_clip<=0) ---
            if grad_clip and grad_clip > 0:
                gscaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]], grad_clip)
            gscaler.step(opt); gscaler.update()
            ep += loss.item() * len(xb)
        ep /= len(idx["train"])
        vnr, vsat = val_nrmse_and_sat()
        history["train_loss"].append(ep); history["val_nrmse"].append(vnr)
        history["val_sat"].append(vsat)
        log.info(f"epoch {epoch+1:03d}/{tr['max_epochs']}  train_loss={ep:.4f}  val_median_NRMSE={vnr:.4f}"
                 f"  sat090={vsat:.4f}{'  [Hwarm]' if warm else ''}")
        if sat_below_epoch is None and vsat < SAT_SEPARATOR:
            sat_below_epoch = epoch + 1
            log.info(f"SAT_BELOW_{SAT_SEPARATOR}: first at epoch {sat_below_epoch} (sat090={vsat:.4f})")
        if vnr < best - 1e-4:
            best, patience = vnr, 0
            best_state = ({k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                          {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()})
        else:
            patience += 1
            if patience >= tr["early_stop_patience"]:
                log.info(f"early stop at epoch {epoch+1}"); break
        # --- saturation abort (off unless abort_sat_threshold set); post-warmup epochs only ---
        if abort_sat_threshold is not None and not warm:
            sat_streak = sat_streak + 1 if vsat > float(abort_sat_threshold) else 0
            if sat_streak >= ABORT_SAT_PATIENCE:
                aborted = True
                log.info(f"ABORT: boundary saturation sat090={vsat:.4f} > {float(abort_sat_threshold)} "
                         f"for {ABORT_SAT_PATIENCE} consecutive post-warmup epochs; "
                         f"stopping run early at epoch {epoch+1}.")
                break

    if best_state is not None:
        enc.load_state_dict(best_state[0]); dec.load_state_dict(best_state[1])
    log.info(f"best val median NRMSE = {best:.4f}")
    log.info(f"SAT_SUMMARY epochs={len(history['val_sat'])} "
             f"sat090_epoch1={history['val_sat'][0]:.4f} sat090_end={history['val_sat'][-1]:.4f} "
             f"first_epoch_below_{SAT_SEPARATOR}={sat_below_epoch} aborted={aborted}"
             if history["val_sat"] else "SAT_SUMMARY no epochs completed")

    out_dir = repro.outputs() / cfg["out_dir"]
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures" / "reconstruction_examples").mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(), "cfg": cfg},
               repro.outputs() / cfg["out_ckpt"])

    # per-epoch diagnostic curves (train_loss / val NRMSE / boundary saturation) for review
    pd.DataFrame({"epoch": np.arange(1, len(history["train_loss"]) + 1),
                  "train_loss": history["train_loss"],
                  "val_median_nrmse": history["val_nrmse"],
                  f"val_sat{int(SAT_THRESH*100):03d}": history["val_sat"]}).to_csv(
        out_dir / "tables" / "epoch_log.csv", index=False)

    t_sec = dec.t.cpu().numpy(); t_ms = t_sec * 1000
    rows, perclass_rows, perlead_rows = [], [], []
    theta_store = {}
    for split in ("train", "val", "test"):
        z, yh, ii = infer(split)
        theta = dec.pspace.theta(torch.from_numpy(z).to(device)).cpu().numpy()
        theta_store[split] = (theta, labels[ii], ii)
        y_s = scaled[ii]
        m_s = compute_recon_metrics(y_s, yh, t_sec)
        y_p = signals[ii]; yh_p = scaler.inverse_transform(yh)
        m_p = compute_recon_metrics(y_p, yh_p, t_sec)
        boundary = float((np.abs(np.tanh(z)) > 0.95).mean())
        sat090 = float((np.abs(np.tanh(z)) > SAT_THRESH).mean())
        rows.append({"split": split, "space": "scaled", **m_s, "boundary_rate": boundary,
                     "sat090": sat090, "n": len(ii)})
        rows.append({"split": split, "space": "physical_mV", **m_p, "boundary_rate": boundary,
                     "sat090": sat090, "n": len(ii)})
        log.info(f"[{split}] scaled median NRMSE={m_s['median_nrmse']:.4f} corr={m_s['median_corr']:.4f} "
                 f"boundary={boundary:.3f} sat090={sat090:.3f}")
        if split in ("val", "test"):
            # per class
            for c, cn in enumerate(TARGET_CLASSES):
                cm = labels[ii] == c
                if cm.sum() == 0:
                    continue
                perclass_rows.append({"split": split, "class": cn,
                                      "median_nrmse": float(np.median(per_sample_nrmse(y_s[cm], yh[cm]))),
                                      "median_corr": float(np.median(per_sample_lead_corr(y_s[cm], yh[cm]))),
                                      "n": int(cm.sum())})
        if split == "test":
            # per lead
            for l, ln in enumerate(CANONICAL_LEADS):
                cr = per_sample_lead_corr(y_s[:, l:l+1, :], yh[:, l:l+1, :])
                nr = per_sample_nrmse(y_s[:, l:l+1, :], yh[:, l:l+1, :])
                perlead_rows.append({"lead": ln, "median_corr": float(np.median(cr)),
                                     "median_nrmse": float(np.median(nr))})

    pd.DataFrame(rows).to_csv(out_dir / "tables" / "reconstruction_metrics.csv", index=False)
    pd.DataFrame(perclass_rows).to_csv(out_dir / "tables" / "reconstruction_by_class.csv", index=False)
    pd.DataFrame(perlead_rows).to_csv(out_dir / "tables" / "reconstruction_by_lead.csv", index=False)

    # save theta for Step 4/5
    theta_all = np.concatenate([theta_store[s][0] for s in ("train", "val", "test")])
    lab_all = np.concatenate([theta_store[s][1] for s in ("train", "val", "test")])
    fold_all = fold[np.concatenate([theta_store[s][2] for s in ("train", "val", "test")])]
    np.savez_compressed(PROJECT_ROOT / cfg["out_theta"], theta=theta_all, label=lab_all, fold=fold_all)
    log.info(f"theta saved -> {cfg['out_theta']}  shape={theta_all.shape}")

    # figures: recon examples (3/class, test), parameter distributions, lead field, train curve
    z_te, yh_te, ii_te = infer("test")
    lab_te = labels[ii_te]
    for c, cn in enumerate(TARGET_CLASSES):
        cand = np.where(lab_te == c)[0][:3]
        for k, j in enumerate(cand):
            plot_recon_compare(scaled[ii_te][j], yh_te[j], t_ms,
                               out_dir / "figures" / "reconstruction_examples" / f"{cn}_{k}.png",
                               title=f"{cn} encoder recon (scaled, test)")
    from igraphecg.models.surrogate_decoder import _SEG
    pnames = []
    for nm, n in _SEG:
        pnames += [f"{nm}{i}" for i in range(n)] if n > 1 else [nm]
    plot_param_histograms(theta_store["test"][0], pnames,
                          out_dir / "figures" / "parameter_distributions.png",
                          title="encoder theta distributions (test)")
    plot_heatmap(dec.H_ind.detach().cpu().numpy(), out_dir / "figures" / "leadfield_heatmap.png",
                 title="lead-field H_ind", xticks=[f"n{i}" for i in range(8)],
                 yticks=["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"])
    plot_loss_curve({"train_loss": history["train_loss"], "val_loss": history["val_nrmse"]},
                    out_dir / "figures" / "training_curve.png", title="encoder: train loss / val NRMSE")
    log.info(f"encoder done, outputs -> {out_dir}")


if __name__ == "__main__":
    main()
