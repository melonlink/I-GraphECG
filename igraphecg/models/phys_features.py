"""Interpretable descriptors derived from the surrogate parameters theta (ROUND2 §5.5 / §8 Step4).

V = {2,3,4,5,6,7}, the ventricular nodes:
  D_act = max δ_i - min δ_i        activation dispersion (expected to rise in CD)
  D_rep = max APD_i - min APD_i     repolarization dispersion (expected to rise in STTC)
  S_ST  = ||alpha_ST||_2            ST source strength (expected to rise in MI/STTC)
"""
from __future__ import annotations

import numpy as np
import torch

from .surrogate_decoder import ParamSpace, VENT_NODES


def compute_phys_features(theta: torch.Tensor) -> dict[str, torch.Tensor]:
    """theta:[B,PARAM_DIM] -> derived descriptors, each [B] (torch)."""
    p = ParamSpace.unpack(theta)
    dv = p["delta"][..., VENT_NODES]
    apdv = p["APD"][..., VENT_NODES]
    aST = p["alpha_ST"]  # [B,6]
    ed = p["edge_delay"]  # [B,7] edge delays; e0,e1 proximal (AV/His), e2..e6 leaf (vent. walls)
    prox_delay = ed[..., 0] + ed[..., 1]                      # proximal delay (mechanistic CD)
    leaf = ed[..., 2:7]
    leaf_delay_spread = leaf.max(dim=-1).values - leaf.min(dim=-1).values  # branch imbalance
    return {
        "D_act": dv.max(dim=-1).values - dv.min(dim=-1).values,
        "D_rep": apdv.max(dim=-1).values - apdv.min(dim=-1).values,
        "S_ST": torch.sqrt((aST ** 2).sum(dim=-1) + 1e-12),
        "mean_vent_APD": apdv.mean(dim=-1),
        "vent_delta_spread": dv.std(dim=-1),
        "mean_global_gain": p["global_gain"],
        "prox_delay": prox_delay,
        "leaf_delay_spread": leaf_delay_spread,
    }


def phys_feature_matrix(theta: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Concatenate theta with derived descriptors into a feature matrix. [N,PARAM_DIM] -> [N,F]."""
    th = torch.as_tensor(theta, dtype=torch.float32)
    feats = compute_phys_features(th)
    keys = ["D_act", "D_rep", "S_ST", "mean_vent_APD", "vent_delta_spread",
            "prox_delay", "leaf_delay_spread"]
    derived = torch.stack([feats[k] for k in keys], dim=1)
    X = torch.cat([th, derived], dim=1).cpu().numpy()
    names = [f"theta_{i}" for i in range(th.shape[1])] + keys
    return X, names
