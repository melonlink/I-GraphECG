"""M4: Fisher-information noise-model sensitivity (reviewer item M4).

Compares D-optimal lead selection under identity and empirical diagonal noise
after fixing the global-gain gauge. The design subset is fold 9; fold 10 is not
used for montage selection in this analysis.

Baseline FIM (already published):  F = mean_s J_s^T J_s  (identity noise, W=I)
New FIM (empirical diagonal Sigma): F_w = mean_s J_s^T diag(sigma_inv) J_s
  where sigma_inv[l] = 1 / var_l, var_l = per-lead isoelectric-segment variance
  (TP/PR baseline: first PRE_SAMPLES samples of the median beat, 0..PRE_T-1).

Outputs:
  runs/v1fix_noise_fim/m4_noise_variance.csv   -- per-lead noise std
  runs/v1fix_noise_fim/m4_leadset_ranking.csv  -- D-optimal greedy ranking for both W=I and W=Sigma^-1
  runs/v1fix_noise_fim/RESULT.md
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from igraphecg import repro

import torch
from torch.func import jacfwd

from igraphecg.data.dataset import load_processed
from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.data.ptbxl_loader import CANONICAL_LEADS
from igraphecg.evaluation.round2_io import load_encoder_decoder
from igraphecg.models.surrogate_decoder import PARAM_DIM

SEED    = 42
DEV     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT    = repro.lineage.checkpoint(42)
NPZ     = ROOT / "data/processed/ptbxl_medianbeat_clean_100hz.npz"
# The published location. Resolved lazily -- importing this module must not create directories.
def _default_out():
    return repro.outdir("runs/v1fix_noise_fim")

# Isoelectric baseline: first 20 samples (200 ms at 100 Hz, covering TP+PR)
PRE_SAMPLES = 20
N_FIM       = 400   # same subset as in the paper (per-class balanced)
GREEDY_N    = 5   # how many leads to select greedily
INDEP_LEADS = ["I","II","V1","V2","V3","V4","V5","V6"]   # 8 independent leads
FIM_COLS = list(range(PARAM_DIM - 1))   # condition on global_gain to remove the exact scale gauge
FIM_DIM = len(FIM_COLS)
LAM = 1e-3


def lead_rows(lead_idx: list[int], n_t: int) -> np.ndarray:
    return np.concatenate([np.arange(l * n_t, (l + 1) * n_t) for l in lead_idx])


def compute_fim_weighted(dec, thetas, radius, sigma_inv_vec, device=DEV):
    """F_w = mean_s J_s^T diag(sigma_inv_vec) J_s  (all 12 leads, all time steps).

    sigma_inv_vec: [12*T] weight vector (1/var per output dimension)
    """
    dec.eval()
    r = radius.to(device).view(1, -1)
    w = torch.tensor(sigma_inv_vec, dtype=torch.float32, device=device)   # [12*T]

    def f(th):
        y12, _ = dec.forward(th.unsqueeze(0))
        return y12[0].reshape(-1)   # [12*T]

    P = len(FIM_COLS)
    F = torch.zeros(P, P, device=device)
    for s in range(thetas.shape[0]):
        J = jacfwd(f)(thetas[s].to(device))[:, FIM_COLS]   # [12*T, P]
        J = J * r   # scale columns
        Jw = J * w.unsqueeze(1)   # [12*T, P] * [12*T, 1]
        F += Jw.T @ J   # J^T diag(w) J
    F /= thetas.shape[0]
    return F.detach().cpu().numpy()


def logdet_fim_subset(F_full, lead_idxs, n_t):
    """Extract sub-FIM for given leads and return log-det (D-criterion)."""
    rows = lead_rows(lead_idxs, n_t)
    # F_sub built from Jacobian restricted to these rows; we need the per-sample Js
    # Instead, use the lead-projected FIM: F_sub ≈ W_sub^T F_full W_sub (approx)
    # For exact: recompute. Here we store per-sample J -> too expensive.
    # Use eigenvalue-based trick: logdet(F) already computed from full J.
    # We reuse the stored per-sample Js for the subset.
    raise NotImplementedError   # handled differently below


def greedy_logdet(per_sample_Js, radius, sigma_inv_vec, n_t,
                  candidate_leads, n_select=5, device=DEV):
    """D-optimal greedy selection over candidate_leads.

    per_sample_Js: list of [12*T, 47] tensors (one per sample, scaled by radius)
    sigma_inv_vec: [12*T] or None (identity)
    """
    r = radius.to(device)
    w = (torch.tensor(sigma_inv_vec, dtype=torch.float32, device=device)
         if sigma_inv_vec is not None else None)
    selected = []
    remaining = list(candidate_leads)
    lam = LAM
    p = per_sample_Js[0].shape[1]

    for k in range(n_select):
        best_lead, best_val = None, -1e30
        for lead in remaining:
            trial = selected + [lead]
            rows = lead_rows(trial, n_t)
            F = torch.zeros(p, p, device=device)
            for J in per_sample_Js:
                Jl = J[rows]   # [n_rows, 47]
                if w is not None:
                    wl = w[rows].unsqueeze(1)
                    F += (Jl * wl).T @ Jl
                else:
                    F += Jl.T @ Jl
            F /= len(per_sample_Js)
            eig = torch.linalg.eigvalsh(F + lam * torch.eye(p, device=device))
            ld = float(eig.clamp(min=1e-15).log().sum())
            if ld > best_val:
                best_val, best_lead = ld, lead
        selected.append(best_lead)
        remaining.remove(best_lead)
        print(f"  step {k+1}: added lead {CANONICAL_LEADS[best_lead]:3s}  "
              f"logdet={best_val:.4f}  set={[CANONICAL_LEADS[i] for i in selected]}")
    return selected


def main(out=None):
    OUT = Path(out) if out else _default_out()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    # ── Load data & model ────────────────────────────────────────────────
    data   = load_processed(NPZ)
    sig_np = data["signal_12lead"]   # [N,12,T] physical mV
    labels = data["label"].astype(int)
    folds  = data["fold"]
    n_t    = sig_np.shape[2]   # 100

    # Scale signals (same as training)
    scaler = RobustLeadScaler()
    scaler.median_ = data["scaler_median"]
    scaler.iqr_    = data["scaler_iqr"]
    sig_scaled = scaler.transform(sig_np)   # [N,12,T]

    enc, dec = load_encoder_decoder(CKPT, n_t=n_t, device=DEV)
    enc.eval(); dec.eval()

    radius = dec.pspace.radius[FIM_COLS].detach().cpu()

    # ── 1. Estimate per-lead noise variance from isoelectric baseline ─────
    print(f"Estimating per-lead noise from first {PRE_SAMPLES} samples (TP/PR baseline) …")
    baseline = sig_scaled[folds <= 8, :, :PRE_SAMPLES]   # training records only
    # Remove per-sample mean (DC offset) before variance
    baseline = baseline - baseline.mean(axis=2, keepdims=True)
    var_per_sample = baseline.var(axis=2)   # [N,12]
    mean_var = var_per_sample.mean(axis=0)   # [12]
    noise_std = np.sqrt(mean_var)   # [12]

    lead_names = list(CANONICAL_LEADS)
    df_noise = pd.DataFrame({"lead": lead_names,
                             "noise_std": noise_std,
                             "noise_var": mean_var})
    df_noise.to_csv(OUT / "m4_noise_variance.csv", index=False)
    print("Per-lead noise std:")
    for i, (name, s) in enumerate(zip(lead_names, noise_std)):
        print(f"  {name:3s}: std={s:.5f}")

    # ── 2. Build design FIM subset from fold 9 ────────────────────────────
    np.random.seed(SEED)
    idx_list = []
    for c in range(len(TARGET_CLASSES)):
        cand = np.where((labels == c) & (folds == 9))[0]
        n = min(N_FIM // len(TARGET_CLASSES), len(cand))
        idx_list.append(np.random.choice(cand, n, replace=False))
    fim_idx = np.concatenate(idx_list)
    print(f"\nFIM subset: {len(fim_idx)} samples (balanced)")

    # Infer bounded physical theta for the subset.
    sig_t = torch.from_numpy(sig_scaled[fim_idx].astype(np.float32)).to(DEV)
    with torch.no_grad():
        thetas = dec.pspace.theta(enc(sig_t)).cpu()   # [M,47]

    # ── 3. Build per-sample Jacobian list (scaled) ───────────────────────
    print("Computing Jacobians …")
    dec.eval()
    r = radius.to(DEV).view(1, -1)

    def f_full(th):
        y12, _ = dec.forward(th.unsqueeze(0))
        return y12[0].reshape(-1)

    per_sample_Js = []
    for s in range(len(thetas)):
        J = jacfwd(f_full)(thetas[s].to(DEV))[:, FIM_COLS]   # [12*T,46]
        J = J * r   # scale columns
        per_sample_Js.append(J.detach())
        if (s + 1) % 50 == 0:
            print(f"  {s+1}/{len(thetas)}")

    # ── 4. Build sigma_inv_vec ────────────────────────────────────────────
    sigma_inv = 1.0 / np.maximum(mean_var, 1e-8)   # [12]
    # Expand to [12*T]: each lead's T samples share the same weight
    sigma_inv_vec = np.repeat(sigma_inv, n_t)   # [12*T]
    # Normalise so overall scale is comparable to W=I
    sigma_inv_vec = sigma_inv_vec / sigma_inv_vec.mean()

    # ── 5. Greedy D-optimal selection ─────────────────────────────────────
    indep_idx = [list(CANONICAL_LEADS).index(l) for l in INDEP_LEADS]

    print("\n=== Greedy D-optimal (W = I, identity noise) ===")
    sel_I = greedy_logdet(per_sample_Js, radius, None,
                          n_t, indep_idx, n_select=GREEDY_N, device=DEV)

    print("\n=== Greedy D-optimal (W = Sigma^{-1}, empirical noise) ===")
    sel_W = greedy_logdet(per_sample_Js, radius, sigma_inv_vec,
                          n_t, indep_idx, n_select=GREEDY_N, device=DEV)

    sel_I_names = [CANONICAL_LEADS[i] for i in sel_I]
    sel_W_names = [CANONICAL_LEADS[i] for i in sel_W]

    # ── 6. Report 3-lead comparison without a predeclared winner ─────────
    same_top3 = set(sel_I_names[:3]) == set(sel_W_names[:3])
    print(f"\nW=I    top-3: {sel_I_names[:3]}")
    print(f"W=Σ⁻¹  top-3: {sel_W_names[:3]}")

    # ── 7. Save ranking CSV ───────────────────────────────────────────────
    max_k = max(len(sel_I_names), len(sel_W_names))
    df_rank = pd.DataFrame({
        "rank": list(range(1, max_k + 1)),
        "W=I (identity)": sel_I_names + [""] * (max_k - len(sel_I_names)),
        "W=Sigma^-1 (empirical)": sel_W_names + [""] * (max_k - len(sel_W_names)),
    })
    df_rank.to_csv(OUT / "m4_leadset_ranking.csv", index=False)

    # ── 8. RESULT.md ──────────────────────────────────────────────────────
    md = f"""# M4 — Fisher-Information Noise-Model Sensitivity

## Per-lead noise (scaled signal, isoelectric baseline first {PRE_SAMPLES} samples)

| lead | noise std |
|---|---|
""" + "".join(f"| {n} | {s:.5f} |\n" for n, s in zip(lead_names, noise_std)) + f"""
Noise varies ~{noise_std.max()/noise_std.min():.1f}× across leads (max/min).

## D-optimal greedy selection: top-{GREEDY_N} leads

| rank | W = I (identity) | W = Σ⁻¹ (empirical noise) |
|---|---|---|
""" + "".join(f"| {k+1} | {sel_I_names[k] if k < len(sel_I_names) else ''} | {sel_W_names[k] if k < len(sel_W_names) else ''} |\n"
              for k in range(max_k)) + f"""

## 3-lead set comparison
- W=I top-3 = {sel_I_names[:3]}
- W=Σ⁻¹ top-3 = {sel_W_names[:3]}
- Same unordered top-3 = {same_top3}

## Conclusion
The empirical noise model {'preserved' if same_top3 else 'changed'} the unordered identity-noise top-3 set.
Exact montage claims are conditional on this decoder, fold-9 design subset, parameter scaling, gauge choice, and noise model.

## Method
- Empirical noise variance: per lead, the variance of the first {PRE_SAMPLES} samples of the scaled signal (100 Hz, TP/PR isoelectric segment)
- Jacobian: as in the manuscript, computed with jacfwd on the locked decoder
- FIM subset: {len(fim_idx)} class-balanced fold-9 records; independent lead set: {INDEP_LEADS}
- Gauge: global_gain held fixed in the local analysis; FIM dimension = {FIM_DIM}
- Regularization: lambda = {LAM}
- Script: scripts/53_noise_model_sensitivity.py
"""
    (OUT / "RESULT.md").write_text(md, encoding="utf-8")
    print(f"\n✅ Saved to {OUT}/")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="output directory; default = the published run under the output root")
    main(out=ap.parse_args().out)
