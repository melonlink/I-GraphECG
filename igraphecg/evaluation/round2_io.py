"""Load the Round 2 encoder+decoder and recompute theta in the original record order (so theta
stays aligned with the signals).

B1/B3 fix: when the checkpoint cfg carries ``scaled_derivation: true`` (the repaired models from
D1 onwards), the decoder must derive the limb leads with the **affine Einthoven-Goldberger
coefficients of the scaled space** - bit-for-bit as during training.
This module decides from the checkpoint's own cfg whether to attach a scaler to SurrogateDecoder:
  - flag absent (legacy v1/v2/v3 checkpoints) -> never attach; bit-for-bit legacy behaviour;
  - flag present -> use the ``scaler`` passed by the caller if given; otherwise read
    scaler_median/iqr from the npz path recorded in cfg (relative to the repository root);
    if neither is available, **raise** - silently falling back to the wrong coordinate
    system is exactly the B1 bug.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _resolve_scaler_from_cfg(cfg_ck: dict) -> dict:
    """Read the robust-scaler statistics from the npz path recorded in the checkpoint cfg."""
    from igraphecg.utils.paths import PROJECT_ROOT
    npz_rel = cfg_ck.get("npz")
    if not npz_rel:
        raise RuntimeError(
            "checkpoint was trained with scaled_derivation=true but its cfg has no "
            "'npz' path and no scaler was passed to load_encoder_decoder(); "
            "pass scaler={'median':..., 'iqr':...} explicitly")
    npz_path = (PROJECT_ROOT / npz_rel).resolve()
    if not npz_path.exists():
        raise RuntimeError(
            f"checkpoint needs the training scaler (scaled_derivation=true) but "
            f"{npz_path} does not exist; pass scaler=... explicitly")
    with np.load(npz_path) as d:
        return {"median": d["scaler_median"].copy(), "iqr": d["scaler_iqr"].copy()}


def load_encoder_decoder(ckpt_path: str | Path, n_t: int, rank: int = 3, device="cpu",
                         scaler=None):
    """Load encoder+decoder. ``scaler`` is used only if the checkpoint sets scaled_derivation."""
    import torch
    from igraphecg.models.phys_encoder import PhysEncoder
    from igraphecg.models.surrogate_decoder import PARAM_DIM, SurrogateDecoder
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_ck = ck.get("cfg") or {}
    dec_scaler = None
    if bool(cfg_ck.get("scaled_derivation", False)):
        dec_scaler = scaler if scaler is not None else _resolve_scaler_from_cfg(cfg_ck)
    enc = PhysEncoder(in_ch=12, out_dim=PARAM_DIM).to(device)
    dec = SurrogateDecoder(n_t=n_t, leadfield_rank=rank, scaler=dec_scaler).to(device)
    enc.load_state_dict(ck["encoder"]); dec.load_state_dict(ck["decoder"])
    enc.eval(); dec.eval()
    return enc, dec


def infer_theta(enc, dec, scaled: np.ndarray, device="cpu", batch: int = 512):
    """scaled:[N,12,T] -> (theta[N,47], z[N,47]) as numpy, in the original order."""
    import torch
    zs, ths = [], []
    with torch.no_grad():
        for s in range(0, len(scaled), batch):
            xb = torch.from_numpy(scaled[s:s + batch].astype(np.float32)).to(device)
            z = enc(xb)
            th = dec.pspace.theta(z)
            zs.append(z.cpu().numpy()); ths.append(th.cpu().numpy())
    return np.concatenate(ths), np.concatenate(zs)
