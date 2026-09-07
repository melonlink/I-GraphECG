"""Round 3 Task E: lead ablation - lead subsets and masking (strategy B: unobserved -> zero)."""
from __future__ import annotations

import numpy as np

from ..data.ptbxl_loader import CANONICAL_LEADS

_idx = {l: i for i, l in enumerate(CANONICAL_LEADS)}
LEAD_SETS: dict[str, list[int]] = {
    "L12": list(range(12)),
    "L3a_II_V1_V5": [_idx["II"], _idx["V1"], _idx["V5"]],
    "L3b_I_II_V5": [_idx["I"], _idx["II"], _idx["V5"]],
    "L1_II": [_idx["II"]],
}


def mask_signal(scaled: np.ndarray, observed: list[int]) -> np.ndarray:
    """Strategy B: keep observed leads, zero the rest (input stays 12-channel). scaled:[N,12,T]."""
    out = np.zeros_like(scaled)
    out[:, observed, :] = scaled[:, observed, :]
    return out
