"""v3 Phase 1 (no retrain): control-theoretic posterior analysis on the frozen Eikonal model.

Candidates (v3 queue):
  1.2  SVD of lead-field H=AB + FIM eigen-decomposition -> unobservable parameter combinations
       (mechanistic "why" alpha_ST is least observable: near-zero source->lead gain / transmission zero).
  1.3  Optimal sensor (lead) selection on the parameter observability Gramian (=scaled FIM):
       D-optimal (logdet, submodular -> greedy 1-1/e), A-optimal (trace inv), E-optimal (lambda_min);
       greedy vs exhaustive near-optimal gap; vs clinical II+V1+V5; observability-vs-#leads curve;
       per-lead param-group identifiability heatmap.
  1.5  Cramer-Rao lower bound (CRB) per parameter group per lead set.

All read-only on a frozen checkpoint (default = boundary-B). seed=42, official folds, fixed 400-sample
FIM subset (100/class) for comparability.

Run: PY="python"
     "$PY" PUBLIC/scripts/50_observability_phase1.py --ckpt outputs/checkpoints/d1s_r3_s42_best.pt
"""
from __future__ import annotations
import argparse, itertools, sys
from pathlib import Path
try:
    import torch   # noqa
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]   # PUBLIC
sys.path.insert(0, str(ROOT))
from igraphecg import repro
from igraphecg.data.dataset import load_processed
from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.data.ptbxl_loader import CANONICAL_LEADS
from igraphecg.evaluation.identifiability import lead_rows, fim_metrics, PARAM_GROUPS, param_names
from igraphecg.evaluation.round2_io import load_encoder_decoder, infer_theta

mpl.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.titlesize": 9})
LAM = 1e-3   # Tikhonov for inv/logdet stability
NLEAD = 12


def fixed_subset(label, fold, per_class=100, seed=42):
    rng = np.random.default_rng(seed); idx = []
    for c in range(len(TARGET_CLASSES)):
        pool = np.where((label == c) & (fold == 10))[0]
        idx.append(pool if len(pool) <= per_class else rng.choice(pool, per_class, replace=False))
    return np.concatenate(idx)


def compute_J_all(dec, thetas, radius, n_t, device):
    """Return the scaled Jacobian J_all:[N, 12*n_t, 47] (J * radius, columns normalized)."""
    from torch.func import jacfwd
    r = radius.to(device).view(1, -1)

    def f(th):
        y12, _ = dec.forward(th.unsqueeze(0))
        return y12[0].reshape(-1)

    Js = []
    for s in range(thetas.shape[0]):
        J = jacfwd(f)(thetas[s].to(device)) * r   # [12*n_t, 47]
        Js.append(J.detach())
    return torch.stack(Js)   # [N, R, 47]


def F_from_J(J_all, rows):
    Js = J_all[:, rows, :]   # [N, |rows|, 47]
    return torch.einsum("nrp,nrq->pq", Js, Js) / J_all.shape[0]


def dopt(F):   # logdet (D-optimal)
    P = F.shape[0]
    return float(torch.logdet(F + LAM * torch.eye(P, device=F.device)))


def aopt(F):   # trace inv (A-optimal) ; smaller is better
    P = F.shape[0]
    return float(torch.trace(torch.linalg.inv(F + LAM * torch.eye(P, device=F.device))))


def eopt(F):   # lambda_min (E-optimal)
    P = F.shape[0]
    return float(torch.linalg.eigvalsh(F + LAM * torch.eye(P, device=F.device))[0])


def main():
    ap = argparse.ArgumentParser()
    # The locked lineage, not a path: this default used to name the superseded
    # eikonal_B weights and was corrected only by a driver's monkey patch.
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--npz", default="data/processed/ptbxl_medianbeat_clean_100hz.npz")
    # None -> the published run for the locked seed; the old default was the
    # pre-v1fix phase1 run, which the paper does not report.
    ap.add_argument("--out", default=None)
    ap.add_argument("--per_class", type=int, default=100)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = ROOT / (a.out or repro.outdir("runs/v1fix_phase1_s42")); (out / "tables").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(parents=True, exist_ok=True)

    data = load_processed(ROOT / a.npz)
    signals = data["signal_12lead"].astype(np.float32); n_t = signals.shape[2]
    label = data["label"].astype(int); fold = data["fold"].astype(int)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals)
    enc, dec = load_encoder_decoder(ROOT / (a.ckpt or repro.lineage.checkpoint(42)), n_t, 3, dev)

    idx = fixed_subset(label, fold, a.per_class)
    theta, _ = infer_theta(enc, dec, scaled[idx], device=dev)
    thetas = torch.from_numpy(theta).to(dev)
    radius = dec.pspace.radius
    print(f"[phase1] ckpt={(a.ckpt or repro.lineage.checkpoint(42))}  FIM subset n={len(idx)}  device={dev}")
    J_all = compute_J_all(dec, thetas, radius, n_t, dev)
    print(f"[phase1] J_all {tuple(J_all.shape)} computed")
    names = param_names(); groups = list(PARAM_GROUPS.keys())

    # ============== 1.2 unobservable directions + lead-field SVD ==============
    F12 = F_from_J(J_all, torch.tensor(lead_rows(list(range(12)), n_t), device=dev))
    evals, evecs = torch.linalg.eigh(F12)   # ascending
    evals = evals.cpu().numpy(); evecs = evecs.cpu().numpy()
    # bottom-5 eigenvectors -> per-group energy (which params form the unobservable subspace)
    k_un = 5
    Eu = evecs[:, :k_un]   # [47,5] smallest
    rows12 = []
    for gi, g in enumerate(groups):
        cols = list(PARAM_GROUPS[g])
        energy = float((Eu[cols, :] ** 2).sum() / k_un)
        rows12.append({"param_group": g, "unobs_energy_bottom5": energy,
                       "n_params": len(cols)})
    gdf = pd.DataFrame(rows12).sort_values("unobs_energy_bottom5", ascending=False)
    gdf.to_csv(out / "tables" / "unobservable_subspace_group_energy.csv", index=False)
    # lead-field SVD: source(node)->independent-lead gain
    H = (dec.A @ dec.B).detach().cpu().numpy()   # [8 leads_ind, 8 nodes]
    Uh, Sh, Vh = np.linalg.svd(H)
    np.savetxt(out / "tables" / "leadfield_singular_values.csv", Sh, delimiter=",", header="sigma", comments="")
    VENT = [2, 3, 4, 5, 6, 7]
    # ST source mode = uniform over ventricular nodes; its transmitted gain = ||H @ e_ST|| (normalized)
    e_st = np.zeros(8); e_st[VENT] = 1.0; e_st /= np.linalg.norm(e_st)
    st_gain = float(np.linalg.norm(H @ e_st))
    # persisted so that the manuscript's transmission-gain figure (Section 3.6) has a registered source
    pd.DataFrame([{"mode": "uniform_ventricular_ST", "transmitted_gain": st_gain,
                   "leading_sigma": float(Sh[0]), "gain_over_leading_sigma": st_gain / float(Sh[0]),
                   "smallest_nonzero_sigma": float(Sh[2])}]).to_csv(out / "tables" / "st_mode_gain.csv", index=False)
    qrs_mode = np.zeros(8); qrs_mode[VENT] = 1.0; qrs_mode /= np.linalg.norm(qrs_mode)   # same support; use top sing for contrast
    print(f"[1.2] smallest FIM eigenvalues: {np.round(evals[:5],3)}  | unobs top group: {gdf.iloc[0]['param_group']} "
          f"({gdf.iloc[0]['unobs_energy_bottom5']:.2f})")
    print(f"[1.2] lead-field sigma: {np.round(Sh,3)}  cond={Sh[0]/max(Sh[-1],1e-9):.1f}  | ST-source transmitted gain={st_gain:.3f} "
          f"(vs top sigma {Sh[0]:.3f})")
    # figure: unobservable-subspace group energy
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(range(len(gdf)), gdf["unobs_energy_bottom5"],
           color=["#c0392b" if g == "alpha_ST" else "#4c72b0" for g in gdf["param_group"]])
    ax.set_xticks(range(len(gdf))); ax.set_xticklabels(gdf["param_group"], rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("energy in bottom-5\nunobservable eigvecs"); ax.set_title("(1.2) Which parameters form the unobservable subspace (12-lead)")
    ax.grid(alpha=.3, axis="y"); fig.tight_layout()
    fig.savefig(out / "figures" / "unobservable_subspace.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ============== 1.3 optimal sensor selection (D/A/E) ==============
    def F_leads(ls):
        return F_from_J(J_all, torch.tensor(lead_rows(ls, n_t), device=dev))
    # greedy D-optimal
    remaining = list(range(NLEAD)); chosen = []; greedy_rows = []
    for step in range(6):
        best_l, best_s = None, -1e18
        for l in remaining:
            s = dopt(F_leads(chosen + [l]))
            if s > best_s:
                best_s, best_l = s, l
        chosen.append(best_l); remaining.remove(best_l)
        Fc = F_leads(chosen); m = fim_metrics(Fc.cpu().numpy())
        greedy_rows.append({"k": len(chosen), "added_lead": CANONICAL_LEADS[best_l],
                            "lead_set": "+".join(CANONICAL_LEADS[i] for i in chosen),
                            "D_logdet": best_s, "A_trace_inv": aopt(Fc), "E_lambda_min": eopt(Fc),
                            "eff_rank": m["effective_rank"]})
        print(f"[1.3] greedy k={len(chosen)} +{CANONICAL_LEADS[best_l]:<3} D={best_s:.2f} effrank={m['effective_rank']:.2f}")
    gr = pd.DataFrame(greedy_rows); gr.to_csv(out / "tables" / "greedy_lead_selection.csv", index=False)
    # exhaustive best (D-opt) for k=2,3,4 -> near-optimal gap
    exh = []
    for k in (2, 3, 4):
        best, bestset = -1e18, None
        for combo in itertools.combinations(range(NLEAD), k):
            s = dopt(F_leads(list(combo)))
            if s > best:
                best, bestset = s, combo
        g_at_k = gr[gr.k == k]["D_logdet"].values[0]
        exh.append({"k": k, "exhaustive_best_D": best,
                    "exhaustive_set": "+".join(CANONICAL_LEADS[i] for i in bestset),
                    "greedy_D": g_at_k, "near_opt_gap": g_at_k - best})
        print(f"[1.3] k={k} exhaustive best D={best:.2f} ({'+'.join(CANONICAL_LEADS[i] for i in bestset)}) "
              f"| greedy={g_at_k:.2f} gap={g_at_k-best:+.3f}")
    pd.DataFrame(exh).to_csv(out / "tables" / "greedy_vs_exhaustive_gap.csv", index=False)
    # clinical II+V1+V5 vs greedy-3
    clin = [CANONICAL_LEADS.index(x) for x in ("II", "V1", "V5")]
    Fcl = F_leads(clin); F12m = fim_metrics(F12.cpu().numpy())
    cmp_rows = [
        {"set": "greedy-3", "leads": gr[gr.k == 3]["lead_set"].values[0], "D_logdet": gr[gr.k == 3]["D_logdet"].values[0],
         "A_trace_inv": aopt(F_leads([CANONICAL_LEADS.index(x) for x in gr[gr.k == 3]["lead_set"].values[0].split("+")])),
         "E_lambda_min": eopt(F_leads([CANONICAL_LEADS.index(x) for x in gr[gr.k == 3]["lead_set"].values[0].split("+")])),
         "eff_rank": gr[gr.k == 3]["eff_rank"].values[0]},
        {"set": "clinical II+V1+V5", "leads": "II+V1+V5", "D_logdet": dopt(Fcl), "A_trace_inv": aopt(Fcl),
         "E_lambda_min": eopt(Fcl), "eff_rank": fim_metrics(Fcl.cpu().numpy())["effective_rank"]},
        {"set": "all-12", "leads": "12", "D_logdet": dopt(F12), "A_trace_inv": aopt(F12),
         "E_lambda_min": eopt(F12), "eff_rank": F12m["effective_rank"]},
    ]
    pd.DataFrame(cmp_rows).to_csv(out / "tables" / "greedy3_vs_clinical.csv", index=False)
    print(f"[1.3] greedy-3={cmp_rows[0]['leads']} D={cmp_rows[0]['D_logdet']:.2f} | clinical II+V1+V5 D={cmp_rows[1]['D_logdet']:.2f}")
    # observability-vs-#leads curve (greedy)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(gr.k, gr.D_logdet, "o-", label="greedy D-optimal (logdet F)")
    for r in exh:
        ax.scatter([r["k"]], [r["exhaustive_best_D"]], marker="x", s=60, color="#c0392b",
                   label="exhaustive best" if r["k"] == 2 else None)
    ax.scatter([3], [cmp_rows[1]["D_logdet"]], marker="s", s=50, color="#2f6f4f", label="clinical II+V1+V5")
    ax.set_xlabel("number of leads k"); ax.set_ylabel("D-optimality  log det F(S)")
    ax.set_title("(1.3) Observability vs lead count"); ax.legend(fontsize=7, frameon=False); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "figures" / "observability_vs_nleads.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    # per-lead param-group identifiability heatmap (single-lead F -> group score)
    from igraphecg.evaluation.identifiability import identifiability_scores, group_scores
    heat = np.zeros((NLEAD, len(groups)))
    for li in range(NLEAD):
        Fl = F_leads([li]).cpu().numpy()
        sc, _ = identifiability_scores(Fl, lam=LAM); gs = group_scores(sc)
        heat[li] = [gs[g] for g in groups]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(np.log10(heat + 1e-9), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(NLEAD)); ax.set_yticklabels(CANONICAL_LEADS, fontsize=7)
    ax.set_title("(1.3) Per-lead parameter-group identifiability (log10 score)"); fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(out / "figures" / "per_lead_group_identifiability.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ============== 1.5 Cramer-Rao lower bound per group per lead set ==============
    crb_rows = []
    lead_sets = {"L12": list(range(12)), "II+V1+V5": clin,
                 "I+II+V5": [CANONICAL_LEADS.index(x) for x in ("I", "II", "V5")],
                 "II": [CANONICAL_LEADS.index("II")]}
    for name, ls in lead_sets.items():
        F = F_leads(ls).cpu().numpy()
        cov = np.linalg.inv(F + LAM * np.eye(F.shape[0]))
        crb = np.sqrt(np.clip(np.diag(cov), 0, None))   # CRB (scaled units) per param
        row = {"lead_set": name}
        for g in groups:
            row[f"CRB_{g}"] = float(np.mean(crb[list(PARAM_GROUPS[g])]))
        crb_rows.append(row)
    pd.DataFrame(crb_rows).to_csv(out / "tables" / "crb_by_group_by_leadset.csv", index=False)
    print("[1.5] CRB (scaled) alpha_ST by lead set: " +
          ", ".join(f"{r['lead_set']}={r['CRB_alpha_ST']:.2f}" for r in crb_rows))
    print(f"[phase1] DONE -> {out}")


if __name__ == "__main__":
    main()
