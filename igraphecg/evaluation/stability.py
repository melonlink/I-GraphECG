"""Round 3 Task F: stability metrics.

Level 1 (surrogate stability proxy, not a strict Floquet analysis):
  V_rec = z(D_act) + z(Tend_rep_disp) + z(tau_rep_disp) + z(ST_projected_L2)
  m_proxy = -V_rec       (z-score fitted on train only)
Level 2 (AP-ODE finite-time monodromy) is in src/models/ap_ode.py.
"""
from __future__ import annotations

import numpy as np


class TrainZScorer:
    """z-score: mean/std fitted on the train subset only, then applied to all samples."""

    def __init__(self):
        self.mean_, self.std_ = {}, {}

    def fit(self, feats: dict[str, np.ndarray], train_mask: np.ndarray):
        for k, v in feats.items():
            self.mean_[k] = float(np.mean(v[train_mask]))
            self.std_[k] = float(np.std(v[train_mask]) + 1e-8)
        return self

    def z(self, key, v):
        return (v - self.mean_[key]) / self.std_[key]


def surrogate_stability_proxy(feats: dict[str, np.ndarray], train_mask: np.ndarray
                              ) -> dict[str, np.ndarray]:
    """feats needs D_act, Tend_disp, tau_rep_disp, ST_projected_L2; returns V_rec and m_proxy."""
    zs = TrainZScorer().fit(feats, train_mask)
    keys = ["D_act", "Tend_disp", "tau_rep_disp", "ST_projected_L2"]
    V = sum(zs.z(k, feats[k]) for k in keys)
    return {"V_rec": V, "m_proxy": -V}
