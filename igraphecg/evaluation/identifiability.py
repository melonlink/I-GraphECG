"""Round 3 Task C: identifiability analysis (Jacobian / Fisher information matrix / effective
rank / parameter confidence).

J = d vec(yhat) / d theta (forward-mode autodiff; theta is 47-dim < 1200 outputs, so jacfwd wins).
Parameter-scale standardization: J_scaled[:,j] = J[:,j] * radius_j.
FIM F = mean_s J_s^T J_s (W=I).
"""
from __future__ import annotations

import numpy as np


def lead_rows(lead_idx: list[int], n_t: int) -> np.ndarray:
    """Row indices of the selected leads after flattening the 12 leads."""
    return np.concatenate([np.arange(l * n_t, (l + 1) * n_t) for l in lead_idx])


def compute_fim(dec, thetas, radius, observed_rows=None, device="cpu"):
    """thetas:[M,47] (torch). Returns the standardized FIM F:[47,47] (numpy)."""
    import torch
    from torch.func import jacfwd

    dec.eval()
    r = radius.to(device).view(1, -1)

    def f(th):  # th:[47] -> flat yhat
        y12, _ = dec.forward(th.unsqueeze(0))
        return y12[0].reshape(-1)

    P = thetas.shape[1]
    F = torch.zeros(P, P, device=device)
    for s in range(thetas.shape[0]):
        J = jacfwd(f)(thetas[s].to(device))            # [1200,47]
        if observed_rows is not None:
            J = J[observed_rows]
        J = J * r                                       # column-scale standardization
        F += J.T @ J
    F /= thetas.shape[0]
    return F.detach().cpu().numpy()


def fim_metrics(F: np.ndarray) -> dict:
    eig = np.linalg.eigvalsh(F)
    eig = np.clip(eig, 0, None)[::-1]                   # descending
    s = eig.sum() + 1e-12
    p = eig / s
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    cum = np.cumsum(eig) / s
    rank95 = int(np.searchsorted(cum, 0.95) + 1)
    rank99 = int(np.searchsorted(cum, 0.99) + 1)
    eps = eig.max() * 1e-6 + 1e-12
    cond = float(eig.max() / max(eig.min(), eps))
    cond = min(cond, 1e12)
    logdet = float(np.sum(np.log(eig + eps)))
    return {"effective_rank": eff_rank, "rank95": rank95, "rank99": rank99,
            "condition_number": cond, "logdet": logdet,
            "eig_top": eig[:10].tolist(), "eig_bottom": eig[-10:].tolist()}


def identifiability_scores(F: np.ndarray, lam: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Tikhonov covariance Cov=(F+λI)^-1 -> (identifiability_score, Fisher correlation matrix)."""
    P = F.shape[0]
    cov = np.linalg.inv(F + lam * np.eye(P))
    d = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    score = 1.0 / d
    corr = cov / (d[:, None] * d[None, :] + 1e-18)
    return score, corr


# Parameter groups (in _SEG order). Timing splits into delta_root / edge_prox (AV-bundle) /
# edge_leaf (ventricular walls), so the identifiability plot shows the "proximal > leaf" tiers.
PARAM_GROUPS = {"delta_root": range(0, 1), "edge_prox": range(1, 3), "edge_leaf": range(3, 8),
                "APD": range(8, 16), "tau_dep": range(16, 24), "tau_rep": range(24, 32),
                "q": range(32, 40), "alpha_ST": range(40, 46), "global_gain": range(46, 47)}


def param_names() -> list[str]:
    names = []
    for nm, rng in PARAM_GROUPS.items():
        n = len(list(rng))
        names += [f"{nm}{i}" for i in range(n)] if n > 1 else [nm]
    return names


def group_scores(score: np.ndarray) -> dict[str, float]:
    return {g: float(np.mean(score[list(rng)])) for g, rng in PARAM_GROUPS.items()}
