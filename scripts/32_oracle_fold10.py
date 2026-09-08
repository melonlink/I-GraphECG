"""Per-record oracle fit on the whole fold-10 test set, like-for-like with the encoder.

Section 3.1 sets the oracle's capacity number beside the amortized encoder's test-fold
numbers. scripts/30_surrogate_oracle.py fits its oracle on a 128/32-per-class subset of
folds 1-8/9, so the two figures were never computed on the same records. This script fits
the oracle on every fold-10 record (1,594) under two protocols:

  refit_H   z per record and the shared lead field A, B optimized jointly from a fresh
            decoder, exactly the scripts/30 protocol applied to the test records: the
            expressive-capacity ceiling of the parameterization on those records.
  fixed_H   lead field frozen at the locked seed-42 decoder; z per record initialized at
            the encoder's own estimate and refined by the same objective: the amortization
            gap, i.e. how much the per-record optimum improves on the feed-forward estimate
            with the very same decoder, starting from where the encoder put it.
  fixed_H_center
            lead field frozen at the locked decoder, z per record from the parameter
            centre (the scripts/30 initialization): the per-record optimum of the locked
            decoder without the encoder's help, which separates the lead-field contribution
            to the capacity ceiling from the encoder's initialization.

All arms use the loss weights, learning rates and early-stopping rule of
configs/v1fix/oracle_fold10_s42.yaml, whose epoch budget is large enough for the refit arm to
early-stop (scripts/30's 2,500-epoch budget leaves it still improving). Early stopping watches
the median NRMSE of the records being fitted, which is in-sample by design: an oracle is a
best-fit reference, not a predictor.

A fourth block asks whether the refit lead field is a fold-10 artefact: H is refit (with
per-record z) on one interleaved half of the fold, frozen, and used for per-record fits on the
other half, beside the locked lead field on that same half.

Outputs (output-root relative, registered in paper.lock.yaml):
  runs/v1fix_oracle_fold10/tables/oracle_fold10_metrics.csv     arm x space, Table 2 metrics
  runs/v1fix_oracle_fold10/tables/oracle_fold10_per_record.csv  per-record NRMSE and
                                                                correlation, encoder and arms
  runs/v1fix_oracle_fold10/tables/oracle_fold10_leadfield_transfer.csv
                                                                half-split transfer of the
                                                                refit lead field

Run: <myPyTorch python> scripts/32_oracle_fold10.py --config configs/v1fix/oracle_fold10_s42.yaml
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igraphecg import repro
from igraphecg.data.dataset import load_processed
from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.evaluation.recon_metrics import (compute_recon_metrics, per_sample_lead_corr,
                                                per_sample_nrmse)
from igraphecg.evaluation.round2_io import load_encoder_decoder
from igraphecg.utils.config import parse_args_with_config
from igraphecg.utils.logging import get_logger
from igraphecg.utils.paths import PROJECT_ROOT
from igraphecg.utils.seed import set_seed

log = get_logger("oracle_fold10")
TEST_FOLD = 10


def fit(dec, z, y, masks, weights, cfg, train_leadfield: bool, device):
    """Optimize z (and A, B when train_leadfield) with the scripts/30 loop; returns the best
    state under early stopping on the in-sample median NRMSE."""
    import torch
    from igraphecg.training.losses import total_loss

    groups = [{"params": [z], "lr": cfg["lr_theta"]}]
    dec.A.requires_grad_(train_leadfield)
    dec.B.requires_grad_(train_leadfield)
    if train_leadfield:
        groups.append({"params": [dec.A, dec.B], "lr": cfg["lr_H"]})
    opt = torch.optim.Adam(groups)

    def median_nrmse():
        with torch.no_grad():
            yh, _ = dec.forward_from_z(z)
            num = torch.sqrt(((y - yh) ** 2).sum(dim=(1, 2)))
            den = torch.sqrt((y ** 2).sum(dim=(1, 2))) + 1e-8
            return float((num / den).median().item())

    best, best_state, patience, epochs_run = median_nrmse(), None, 0, 0
    best_state = (z.detach().clone(), {k: v.detach().clone() for k, v in dec.state_dict().items()})
    log.info(f"  start median NRMSE = {best:.4f}")
    for epoch in range(cfg["max_epochs"]):
        opt.zero_grad()
        yhat, aux = dec.forward_from_z(z)
        loss, comps = total_loss(y, yhat, z, aux["theta"], dec.H_ind, masks, weights)
        loss.backward()
        opt.step()
        epochs_run = epoch + 1
        if (epoch + 1) % 25 == 0 or epoch == 0:
            vnr = median_nrmse()
            if (epoch + 1) % 250 == 0 or epoch == 0:
                log.info(f"  epoch {epoch + 1:04d}  loss={loss.item():.4f} rec={comps['rec'].item():.4f} "
                         f"median_NRMSE={vnr:.4f}")
            if vnr < best - 1e-4:
                best, patience = vnr, 0
                best_state = (z.detach().clone(),
                              {k: v.detach().clone() for k, v in dec.state_dict().items()})
            else:
                patience += 1
                if patience >= cfg["early_stop_patience"] // 25 + 1:
                    log.info(f"  early stop at epoch {epoch + 1}")
                    break
    z = best_state[0].requires_grad_(False)
    dec.load_state_dict(best_state[1])
    return z, best, epochs_run


def main() -> None:
    import torch
    from igraphecg.models.surrogate_decoder import PARAM_DIM, SurrogateDecoder
    from igraphecg.training.losses import build_masks

    _, cfg = parse_args_with_config("Fold-10 oracle: like-for-like capacity and amortization gap")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device = {device}")

    data = load_processed(PROJECT_ROOT / cfg["npz"])
    signals = data["signal_12lead"].astype(np.float32)
    labels = data["label"].astype(int)
    fold = data["fold"].astype(int)
    record_id = data["record_id"]
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals)
    n_t = signals.shape[2]

    te = np.where(fold == TEST_FOLD)[0]
    log.info(f"fold-{TEST_FOLD} records: {len(te)} "
             f"({', '.join(f'{c}={int((labels[te] == i).sum())}' for i, c in enumerate(TARGET_CLASSES))})")
    y = torch.tensor(scaled[te], device=device)
    w = cfg["loss_weights"]
    dec_scaler = {"median": data["scaler_median"], "iqr": data["scaler_iqr"]} if cfg.get("scaled_derivation") else None

    # ---- the encoder's own fold-10 reconstruction (the Table 2 numbers) ------------------
    ckpt = repro.lineage.checkpoint(42)
    enc, dec_locked = load_encoder_decoder(ckpt, n_t, cfg["leadfield_rank"], device,
                                           scaler={"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    masks = {k: v.to(device) for k, v in build_masks(dec_locked.t).items()}
    t_sec = dec_locked.t.cpu().numpy()
    with torch.no_grad():
        z_enc = torch.cat([enc(y[s:s + 512]) for s in range(0, len(te), 512)])
        yhat_enc, _ = dec_locked.forward_from_z(z_enc)
    arms = {"encoder": (yhat_enc.cpu().numpy(), z_enc, None, 0)}
    log.info(f"[encoder] median NRMSE = {np.median(per_sample_nrmse(scaled[te], arms['encoder'][0])):.4f}")

    # ---- arm 1: refit_H (the scripts/30 protocol on the test records) ------------------
    set_seed(cfg.get("seed", 42))
    dec_fresh = SurrogateDecoder(n_t=n_t, leadfield_rank=cfg["leadfield_rank"], scaler=dec_scaler).to(device)
    z = torch.zeros(len(te), PARAM_DIM, device=device, requires_grad=True)
    t0 = time.time()
    log.info("[refit_H] z per record + shared A,B from a fresh decoder")
    z, best, ep = fit(dec_fresh, z, y, masks, w, cfg, True, device)
    with torch.no_grad():
        yh, _ = dec_fresh.forward_from_z(z)
    arms["oracle_refit_H"] = (yh.cpu().numpy(), z, dec_fresh, ep)
    log.info(f"[refit_H] best median NRMSE = {best:.4f}  epochs={ep}  ({time.time() - t0:.0f}s)")

    # ---- arm 2: fixed_H (locked decoder, z initialised at the encoder estimate) --------
    set_seed(cfg.get("seed", 42))
    z = z_enc.detach().clone().requires_grad_(True)
    t0 = time.time()
    log.info("[fixed_H] z per record from the encoder estimate, lead field frozen at the locked decoder")
    z, best, ep = fit(dec_locked, z, y, masks, w, cfg, False, device)
    with torch.no_grad():
        yh, _ = dec_locked.forward_from_z(z)
    arms["oracle_fixed_H"] = (yh.cpu().numpy(), z, dec_locked, ep)
    log.info(f"[fixed_H] best median NRMSE = {best:.4f}  epochs={ep}  ({time.time() - t0:.0f}s)")

    # ---- arm 3: fixed_H_center (locked decoder, z from the parameter centre) -----------
    set_seed(cfg.get("seed", 42))
    enc2, dec_locked2 = load_encoder_decoder(ckpt, n_t, cfg["leadfield_rank"], device,
                                             scaler={"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    z = torch.zeros(len(te), PARAM_DIM, device=device, requires_grad=True)
    t0 = time.time()
    log.info("[fixed_H_center] z per record from the parameter centre, lead field frozen at the locked decoder")
    z, best, ep = fit(dec_locked2, z, y, masks, w, cfg, False, device)
    with torch.no_grad():
        yh, _ = dec_locked2.forward_from_z(z)
    arms["oracle_fixed_H_center"] = (yh.cpu().numpy(), z, dec_locked2, ep)
    log.info(f"[fixed_H_center] best median NRMSE = {best:.4f}  epochs={ep}  ({time.time() - t0:.0f}s)")

    # ---- tables -------------------------------------------------------------------------
    out = repro.outdir("runs/v1fix_oracle_fold10")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    y_np, y_phys = scaled[te], signals[te]
    rows, per = [], {"record_id": record_id[te], "fold": fold[te], "label": labels[te],
                     "label_name": [TARGET_CLASSES[i] for i in labels[te]]}
    for arm, (yh_np, z_arm, dec_arm, ep) in arms.items():
        pspace = (dec_arm or dec_locked).pspace
        boundary = pspace.boundary_rate(z_arm)
        for space, (yy, yh) in {"scaled": (y_np, yh_np),
                                "physical_mV": (y_phys, scaler.inverse_transform(yh_np))}.items():
            rows.append({"arm": arm, "space": space, **compute_recon_metrics(yy, yh, t_sec),
                         "theta_boundary_rate": boundary, "n": len(te), "epochs": ep})
        per[f"nrmse_{arm}"] = per_sample_nrmse(y_np, yh_np)
        per[f"corr_{arm}"] = per_sample_lead_corr(y_np, yh_np)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "tables" / "oracle_fold10_metrics.csv", index=False)
    per_df = pd.DataFrame(per)
    per_df.to_csv(out / "tables" / "oracle_fold10_per_record.csv", index=False)

    # ---- paired comparison, printed ----------------------------------------------------
    log.info("\n" + metrics[metrics.space == "scaled"][["arm", "median_nrmse", "median_corr", "qrs_nrmse",
                                                        "st_nrmse", "t_nrmse", "theta_boundary_rate",
                                                        "epochs"]].to_string(index=False))
    for arm in ("oracle_refit_H", "oracle_fixed_H", "oracle_fixed_H_center"):
        d = per_df["nrmse_encoder"] - per_df[f"nrmse_{arm}"]
        log.info(f"encoder minus {arm}: median NRMSE difference {d.median():+.4f} "
                 f"(IQR {d.quantile(.25):+.4f}..{d.quantile(.75):+.4f}); "
                 f"oracle better on {(d > 0).mean() * 100:.1f}% of records")
    # ---- lead-field transfer: refit H on half A, freeze it, fit z on half B --------------
    A_idx, B_idx = te[0::2], te[1::2]      # interleaved halves keep the class mix
    yA, yB = y[0::2], y[1::2]
    rows = []

    def half_arm(name, dec, z0, yy, train_H):
        zz, best, ep = fit(dec, z0, yy, masks, w, cfg, train_H, device)
        with torch.no_grad():
            yh, _ = dec.forward_from_z(zz)
        yh = yh.cpu().numpy(); yy_np = yy.cpu().numpy()
        rows.append({"arm": name, "n": len(yy_np), "median_nrmse": float(np.median(per_sample_nrmse(yy_np, yh))),
                     "median_corr": float(np.median(per_sample_lead_corr(yy_np, yh))), "epochs": ep,
                     "H_singular_values": " ".join(f"{v:.3f}" for v in np.linalg.svd(dec.H_ind.detach().cpu().numpy(), compute_uv=False)[:3])})
        log.info(f"[transfer] {name}: median NRMSE {rows[-1]['median_nrmse']:.4f}  epochs={ep}")
        return dec

    set_seed(cfg.get("seed", 42))
    decA = SurrogateDecoder(n_t=n_t, leadfield_rank=cfg["leadfield_rank"], scaler=dec_scaler).to(device)
    decA = half_arm("refit_H_on_A_in_sample", decA, torch.zeros(len(A_idx), PARAM_DIM, device=device, requires_grad=True), yA, True)
    set_seed(cfg.get("seed", 42))
    half_arm("H_A_frozen_z_on_B_out_of_sample", decA, torch.zeros(len(B_idx), PARAM_DIM, device=device, requires_grad=True), yB, False)
    set_seed(cfg.get("seed", 42))
    _, decL = load_encoder_decoder(ckpt, n_t, cfg["leadfield_rank"], device,
                                   scaler={"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    half_arm("locked_H_z_on_B", decL, torch.zeros(len(B_idx), PARAM_DIM, device=device, requires_grad=True), yB, False)
    HA, HL = decA.H_ind.detach().cpu().numpy(), decL.H_ind.detach().cpu().numpy()
    for r in rows:
        r["relative_frobenius_distance_refit_vs_locked_H"] = float(np.linalg.norm(HA - HL) / np.linalg.norm(HL))
    pd.DataFrame(rows).to_csv(out / "tables" / "oracle_fold10_leadfield_transfer.csv", index=False)
    log.info(f"outputs -> {out / 'tables'}")


if __name__ == "__main__":
    main()
