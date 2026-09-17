"""v3 Phase 1 — optimal sensor selection restricted to the 8 INDEPENDENT leads {I,II,V1..V6}.

Derived leads III/aVR/aVL/aVF are exact linear combinations of I,II — excluded (range error as
"sensors" + would break Fisher additivity / submodularity). On the 8 independent leads:
  - verify F(S)=sum_{l in S} F_l (additivity);
  - exhaustive k=2,3,4 (C(8,k)<=70) for D-/A-/E-optimal; greedy-D vs exhaustive;
  - agreement across criteria; each vs clinical II+V1+V5;
  - bootstrap (200x over 400 records): selection frequency, logdet(optimal)-logdet(clinical) 95% CI,
    eigenvalue-floor sensitivity (clip FIM eig<1e-3 to 1e-3);
  - CRB: optimal-3 vs clinical-3 (avg + alpha_ST).

Frozen boundary-B checkpoint; no retrain; seed=42; fixed 400-sample subset.
Run: "$PY" PUBLIC/scripts/51_leadset8_exhaustive.py --ckpt outputs/checkpoints/d1s_r3_s42_best.pt
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from igraphecg import repro
from igraphecg.data.dataset import load_processed
from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.data.ptbxl_loader import CANONICAL_LEADS
from igraphecg.evaluation.identifiability import lead_rows, PARAM_GROUPS
from igraphecg.evaluation.round2_io import load_encoder_decoder, infer_theta

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 9})
LAM = 1e-3
IND = [CANONICAL_LEADS.index(x) for x in ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")]
INAME = {l: CANONICAL_LEADS[l] for l in IND}
CLIN = [CANONICAL_LEADS.index(x) for x in ("II", "V1", "V5")]


def fixed_subset(label, fold, per_class=100, seed=42):
    rng = np.random.default_rng(seed); idx = []
    for c in range(len(TARGET_CLASSES)):
        pool = np.where((label == c) & (fold == 10))[0]
        idx.append(pool if len(pool) <= per_class else rng.choice(pool, per_class, replace=False))
    return np.concatenate(idx)


def compute_J_all(dec, thetas, radius, device):
    from torch.func import jacfwd
    r = radius.to(device).view(1, -1)
    def f(th):
        y12, _ = dec.forward(th.unsqueeze(0)); return y12[0].reshape(-1)
    return torch.stack([jacfwd(f)(thetas[s].to(device)) * r for s in range(thetas.shape[0])]).detach()


def logdet(F): return float(np.linalg.slogdet(F + LAM * np.eye(F.shape[0]))[1])
def trinv(F):  return float(np.trace(np.linalg.inv(F + LAM * np.eye(F.shape[0]))))
def lmin(F):   return float(np.linalg.eigvalsh(F + LAM * np.eye(F.shape[0]))[0])
def logdet_floor(F, floor=1e-3):
    w = np.linalg.eigvalsh(F); return float(np.sum(np.log(np.clip(w, floor, None))))


def main():
    ap = argparse.ArgumentParser()
    # The locked lineage, not a path: this default used to name the superseded
    # eikonal_B weights and was corrected only by a driver's monkey patch.
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--npz", default="data/processed/ptbxl_medianbeat_clean_100hz.npz")
    # None -> the published run for the locked seed; the old default was the
    # pre-v1fix phase1 run, which the paper does not report.
    ap.add_argument("--out", default=None)
    ap.add_argument("--nboot", type=int, default=200)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = ROOT / (a.out or repro.outdir("runs/v1fix_ls8_s42")); (out / "tables").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(parents=True, exist_ok=True)

    data = load_processed(ROOT / a.npz)
    signals = data["signal_12lead"].astype(np.float32); n_t = signals.shape[2]
    label = data["label"].astype(int); fold = data["fold"].astype(int)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals)
    enc, dec = load_encoder_decoder(ROOT / (a.ckpt or repro.lineage.checkpoint(42)), n_t, 3, dev)
    idx = fixed_subset(label, fold)
    theta, _ = infer_theta(enc, dec, scaled[idx], device=dev)
    thetas = torch.from_numpy(theta).to(dev); radius = dec.pspace.radius
    print(f"[ls8] ckpt={(a.ckpt or repro.lineage.checkpoint(42))} n={len(idx)} dev={dev}")
    J_all = compute_J_all(dec, thetas, radius, dev)   # [N,1200,47]
    N = J_all.shape[0]

    # per-lead per-sample Fisher contributions G[l]:[N,47,47] (independent leads only)
    G = {}
    for l in IND:
        rows = torch.tensor(lead_rows([l], n_t), device=dev)
        Jl = J_all[:, rows, :]   # [N,|rows|,47]
        G[l] = torch.einsum("nrp,nrq->npq", Jl, Jl).cpu().numpy()   # [N,47,47]
    print(f"[ls8] G computed for {len(IND)} independent leads")

    def F_of(ls, sidx=None):
        sl = slice(None) if sidx is None else sidx
        return sum(G[l][sl].mean(0) for l in ls)

    # ---- additivity check: F(all8) via union rows == sum_l F_l ----
    rows8 = torch.tensor(lead_rows(IND, n_t), device=dev)
    Ju = J_all[:, rows8, :]
    F_union = torch.einsum("nrp,nrq->pq", Ju, Ju).cpu().numpy() / N
    F_sum = F_of(IND)
    add_err = float(np.abs(F_union - F_sum).max())
    print(f"[ls8] additivity F(S)=sum F_l  max|diff| = {add_err:.2e}  (should be ~0)")

    # ---- exhaustive k=2,3,4 for D/A/E ----
    crit_rows = []; best_sets = {}
    for k in (2, 3, 4):
        combos = list(itertools.combinations(IND, k))
        Ds = {c: logdet(F_of(c)) for c in combos}
        As = {c: trinv(F_of(c)) for c in combos}
        Es = {c: lmin(F_of(c)) for c in combos}
        bD = max(Ds, key=Ds.get); bA = min(As, key=As.get); bE = max(Es, key=Es.get)
        best_sets[k] = {"D": bD, "A": bA, "E": bE}
        for crit, b, val in [("D", bD, Ds[bD]), ("A", bA, As[bA]), ("E", bE, Es[bE])]:
            crit_rows.append({"k": k, "criterion": crit,
                              "best_set": "+".join(INAME[i] for i in b),
                              "score": val, "agree_across_DAE": None})
        agree = (bD == bA == bE)
        for r in crit_rows[-3:]:
            r["agree_across_DAE"] = agree
        print(f"[ls8] k={k} D*={'+'.join(INAME[i] for i in bD)}({Ds[bD]:.2f}) "
              f"A*={'+'.join(INAME[i] for i in bA)} E*={'+'.join(INAME[i] for i in bE)} agree={agree}")
    pd.DataFrame(crit_rows).to_csv(out / "tables" / "best_sets_DAE_k234.csv", index=False)

    # ---- greedy-D vs exhaustive ----
    chosen, rem, greedy = [], list(IND), []
    for _ in range(4):
        bl = max(rem, key=lambda l: logdet(F_of(chosen + [l])))
        chosen.append(bl); rem.remove(bl)
        greedy.append({"k": len(chosen), "set": "+".join(INAME[i] for i in chosen), "D": logdet(F_of(chosen))})
    gdf = pd.DataFrame(greedy); gdf.to_csv(out / "tables" / "greedy_D_8indep.csv", index=False)
    for k in (2, 3, 4):
        gd = gdf[gdf.k == k]["D"].values[0]; ed = logdet(F_of(best_sets[k]["D"]))
        reached = abs(gd - ed) < 1e-6
        print(f"[ls8] greedy k={k} D={gd:.3f} | exhaustive D*={ed:.3f} | greedy_reached_optimum={reached}")

    # ---- optimal-3 (D) vs clinical II+V1+V5 across D/A/E ----
    opt3 = best_sets[3]["D"]
    cmp = []
    for nm, S in [("optimal-3 (D)", opt3), ("clinical II+V1+V5", tuple(CLIN))]:
        F = F_of(list(S))
        cmp.append({"set": nm, "leads": "+".join(INAME[i] for i in S),
                    "D_logdet": logdet(F), "A_trace_inv": trinv(F), "E_lambda_min": lmin(F)})
    pd.DataFrame(cmp).to_csv(out / "tables" / "optimal3_vs_clinical.csv", index=False)
    print(f"[ls8] optimal-3={cmp[0]['leads']}  clinical={cmp[1]['leads']}  "
          f"D {cmp[0]['D_logdet']:.2f} vs {cmp[1]['D_logdet']:.2f}")

    # ---- bootstrap (200x) ----
    rng = np.random.default_rng(42)
    combos3 = list(itertools.combinations(IND, 3))
    win_count = {c: 0 for c in combos3}; lead_incl = {l: 0 for l in IND}
    gap_boot = []
    for _ in range(a.nboot):
        bi = rng.integers(0, N, N)
        Fl = {l: G[l][bi].mean(0) for l in IND}
        def Fb(S): return sum(Fl[l] for l in S)
        # (a) argmax-D 3-set this resample
        wd = max(combos3, key=lambda c: logdet(Fb(c)))
        win_count[wd] += 1
        for l in wd:
            lead_incl[l] += 1
        # (b) gap of fixed optimal-3 vs clinical
        gap_boot.append(logdet(Fb(opt3)) - logdet(Fb(tuple(CLIN))))
    gap_boot = np.array(gap_boot)
    ci = (float(np.percentile(gap_boot, 2.5)), float(np.percentile(gap_boot, 97.5)))
    incl = pd.DataFrame([{"lead": INAME[l], "inclusion_rate": lead_incl[l] / a.nboot} for l in IND]
                        ).sort_values("inclusion_rate", ascending=False)
    incl.to_csv(out / "tables" / "bootstrap_lead_inclusion.csv", index=False)
    topwin = sorted(win_count.items(), key=lambda kv: -kv[1])[:5]
    pd.DataFrame([{"set": "+".join(INAME[i] for i in c), "win_freq": n / a.nboot} for c, n in topwin]
                 ).to_csv(out / "tables" / "bootstrap_top_winning_sets.csv", index=False)
    # (c) eigenvalue-floor sensitivity (full data)
    gap_lam = logdet(F_of(list(opt3))) - logdet(F_of(list(CLIN)))
    gap_floor = logdet_floor(F_of(list(opt3))) - logdet_floor(F_of(list(CLIN)))
    print(f"[ls8] bootstrap gap(opt3-clinical) mean={gap_boot.mean():.3f} 95%CI=[{ci[0]:.3f},{ci[1]:.3f}] "
          f"crosses0={ci[0] <= 0 <= ci[1]}")
    print(f"[ls8] top winning 3-set: {topwin[0][0]} freq={topwin[0][1]/a.nboot:.2f} | lead incl: "
          + ", ".join(f"{INAME[l]}={lead_incl[l]/a.nboot:.2f}" for l in IND))
    print(f"[ls8] gap with-lambda={gap_lam:.3f}  gap eig-floored(1e-3)={gap_floor:.3f}  (both >0 => robust)")
    pd.DataFrame([{"bootstrap_gap_mean": float(gap_boot.mean()), "ci_lo": ci[0], "ci_hi": ci[1],
                   "crosses_zero": bool(ci[0] <= 0 <= ci[1]), "gap_lambda": gap_lam,
                   "gap_eig_floored": gap_floor}]).to_csv(out / "tables" / "bootstrap_gap_summary.csv", index=False)

    # ---- CRB optimal vs clinical ----
    crb_rows = []
    aST = list(PARAM_GROUPS["alpha_ST"])
    for nm, S in [("optimal-3 (D)", opt3), ("clinical II+V1+V5", tuple(CLIN)), ("all-8 independent", tuple(IND))]:
        cov = np.linalg.inv(F_of(list(S)) + LAM * np.eye(47))
        crb = np.sqrt(np.clip(np.diag(cov), 0, None))
        crb_rows.append({"set": nm, "leads": "+".join(INAME[i] for i in S),
                         "CRB_mean_allparams": float(crb.mean()), "CRB_alpha_ST": float(crb[aST].mean())})
    pd.DataFrame(crb_rows).to_csv(out / "tables" / "crb_optimal_vs_clinical.csv", index=False)
    print("[ls8] CRB mean/alphaST: " + " | ".join(f"{r['set']}: {r['CRB_mean_allparams']:.2f}/{r['CRB_alpha_ST']:.2f}" for r in crb_rows))

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(range(len(incl)), incl["inclusion_rate"], color="#4c72b0")
    ax.set_xticks(range(len(incl))); ax.set_xticklabels(incl["lead"], fontsize=8)
    ax.set_ylabel("inclusion rate in\nbootstrap optimal-3"); ax.set_ylim(0, 1)
    ax.set_title("(8-indep) Bootstrap lead inclusion in D-optimal 3-set"); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(out / "figures" / "bootstrap_lead_inclusion.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.hist(gap_boot, bins=30, color="#2f6f4f", alpha=.8)
    ax.axvline(0, color="k", lw=1); ax.axvline(ci[0], color="#c0392b", ls="--", lw=1); ax.axvline(ci[1], color="#c0392b", ls="--", lw=1)
    ax.set_xlabel("logdet(optimal-3) - logdet(clinical)"); ax.set_ylabel("bootstrap count")
    ax.set_title(f"(8-indep) Optimal vs clinical gap\n95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
    fig.tight_layout(); fig.savefig(out / "figures" / "bootstrap_gap_hist.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[ls8] DONE -> {out}")


if __name__ == "__main__":
    main()
