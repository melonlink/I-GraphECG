"""Observability analysis wrappers (FIM / effective rank / CRB / D-A-E / lead-field SVD).

Thin layer over src.evaluation.identifiability so the same analysis can run on any source's
inferred theta (PTB-XL or external), for cross-database observability reproduction.
"""
from __future__ import annotations
import numpy as np

from ..config import CANONICAL_LEADS


def run_observability(dec, thetas, radius, lead_set=None, device="cpu") -> dict:
    """Compute FIM + effective rank (+ optional lead subset) for given inferred thetas.

    dec: decoder; thetas:[M,47] torch tensor; radius: param radius tensor.
    lead_set: list of canonical lead names (default = all 12).
    """
    import torch
    from igraphecg.evaluation.identifiability import compute_fim, fim_metrics, lead_rows
    n_t = dec.t.shape[0] if hasattr(dec, "t") else None
    observed_rows = None
    if lead_set is not None and n_t is not None:
        idx = [CANONICAL_LEADS.index(l) for l in lead_set]
        observed_rows = lead_rows(idx, n_t)
    th = thetas if torch.is_tensor(thetas) else torch.from_numpy(np.asarray(thetas, np.float32))
    F = compute_fim(dec, th, radius, observed_rows=observed_rows, device=device)
    return {"fim": F, **fim_metrics(F)}


def effrank_for_leadsets(dec, thetas, radius, lead_sets, device="cpu") -> dict:
    """{name: effective_rank} for several lead sets (name -> list of canonical lead names)."""
    out = {}
    for name, leads in lead_sets.items():
        out[name] = run_observability(dec, thetas, radius, lead_set=leads, device=device)["effective_rank"]
    return out


def greedy_dopt_3lead(dec, thetas, radius, candidate_leads, n_select=3, device="cpu"):
    """D-optimal greedy selection (max log det F over a lead subset) among candidate leads.
    Returns (selected_leads, logdet_of_selected)."""
    import torch
    from igraphecg.evaluation.identifiability import compute_fim, lead_rows
    n_t = dec.t.shape[0]
    th = thetas if torch.is_tensor(thetas) else torch.from_numpy(np.asarray(thetas, np.float32))
    selected, remaining = [], list(candidate_leads)

    def _logdet(leads):
        rows = lead_rows([CANONICAL_LEADS.index(l) for l in leads], n_t)
        F = compute_fim(dec, th, radius, observed_rows=rows, device=device)
        ev = np.linalg.eigvalsh(F); ev = np.clip(ev, 1e-12, None)
        return float(np.log(ev).sum())

    best_ld = None
    for _ in range(n_select):
        scored = [(l, _logdet(selected + [l])) for l in remaining]
        l, best_ld = max(scored, key=lambda x: x[1])
        selected.append(l); remaining.remove(l)
    return selected, best_ld


def leadfield_svd(dec):
    """Singular values of the source-to-lead operator H=AB and the ST-source transmission gain.
    Model-fixed (independent of the dataset)."""
    import torch
    if not (hasattr(dec, "A") and hasattr(dec, "B")):
        return None
    with torch.no_grad():
        H = (dec.A @ dec.B).detach().cpu().numpy()   # [8,8] source-to-lead operator
    sv = np.linalg.svd(H, compute_uv=False)
    return {"singular_values": sv.tolist()}
