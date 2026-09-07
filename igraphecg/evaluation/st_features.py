"""Round 3 Task B: interpretation of the MI/ST source term.

Internal alpha_ST features (6 ventricular nodes) + regional grouping + lead-projected ST
features ST_lead = H_ind·alpha_ST.
alpha_ST is slice 40:46 of theta and corresponds to ventricular nodes [2,3,4,5,6,7].
Independent lead order I,II,V1..V6; the 12 leads are then derived.
"""
from __future__ import annotations

import numpy as np

_AST = slice(40, 46)              # alpha_ST (6 ventricular nodes)
VENT = [2, 3, 4, 5, 6, 7]
# Regions (indices 0..5 into the alpha_ST vector for the 6 ventricular nodes = nodes 2..7)
REGION = {"anterior": [2, 5], "lateral": [3], "inferior": [4], "right_septal": [0, 1]}
# Index map: node 2(septum)=0, 3(RV)=1, 4(LVant)=2, 5(LVlat)=3, 6(LVinf)=4, 7(apex)=5
IND_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
OUT_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _entropy(p):
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    return -(p * np.log(p + 1e-12)).sum(axis=1)


def internal_st_features(theta: np.ndarray) -> dict[str, np.ndarray]:
    a = theta[:, _AST]                                  # [N,6]
    feats = {
        "ST_L2": np.sqrt((a ** 2).sum(1)),
        "ST_L1": np.abs(a).sum(1),
        "ST_max_abs": np.abs(a).max(1),
        "ST_signed_sum": a.sum(1),
        "ST_spatial_disp": a.max(1) - a.min(1),
        "ST_entropy": _entropy(np.abs(a)),
    }
    for rname, idx in REGION.items():
        feats[f"ST_{rname}_strength"] = np.sqrt((a[:, idx] ** 2).sum(1))
    region_str = np.stack([feats[f"ST_{r}_strength"] for r in REGION], axis=1)
    feats["ST_region_max"] = region_str.max(1)
    feats["ST_region_argmax"] = region_str.argmax(1).astype(float)
    return feats


def _derive12(vec8: np.ndarray) -> np.ndarray:
    """vec8:[N,8] (I,II,V1..V6) -> [N,12] (OUT_LEADS)."""
    I, II = vec8[:, 0], vec8[:, 1]
    III = II - I; aVR = -(I + II) / 2; aVL = I - II / 2; aVF = II - I / 2
    return np.stack([I, II, III, aVR, aVL, aVF,
                     vec8[:, 2], vec8[:, 3], vec8[:, 4], vec8[:, 5], vec8[:, 6], vec8[:, 7]], axis=1)


def projected_st_features(theta: np.ndarray, H_ind: np.ndarray) -> dict[str, np.ndarray]:
    """ST_lead = H_ind @ alpha_ST_full (projected onto the 8 independent leads), then derive 12."""
    N = theta.shape[0]
    a_full = np.zeros((N, 8))
    a_full[:, VENT] = theta[:, _AST]
    st_ind = a_full @ H_ind.T                           # [N,8] independent leads
    st12 = _derive12(st_ind)                            # [N,12]
    idx = {l: i for i, l in enumerate(OUT_LEADS)}
    ant = np.abs(st12[:, [idx[l] for l in ["V1", "V2", "V3", "V4"]]]).mean(1)
    lat = np.abs(st12[:, [idx[l] for l in ["I", "aVL", "V5", "V6"]]]).mean(1)
    inf = np.abs(st12[:, [idx[l] for l in ["II", "III", "aVF"]]]).mean(1)
    sign = np.sign(st12)
    maj = np.sign(np.sum(sign, axis=1, keepdims=True) + 1e-9)
    return {
        "ST_projected_L2": np.sqrt((st12 ** 2).sum(1)),
        "ST_projected_max_abs": np.abs(st12).max(1),
        "ST_projected_anterior_leads": ant,
        "ST_projected_lateral_leads": lat,
        "ST_projected_inferior_leads": inf,
        "ST_projected_sign_discordance": (sign != maj).mean(1),
    }
