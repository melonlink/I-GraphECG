"""W1R design-fold check -- does the three-lead selection depend on which fold the
Fisher design subset is drawn from?

The identity-noise selection of scripts/51_leadset8_exhaustive.py draws its class-balanced
400-record design subset from fold 10. This script repeats the exhaustive three-lead D/A/E
selection (8 independent leads, 56 triples, F + LAM*I, per-record Fisher contributions
averaged over the subset) for every seed with the subset drawn from fold 10 (reproduces the
frozen numbers), from the validation fold 9, and from the training folds 1-8, and records
the selected sets, the D-optimal margin over the clinical set II+V1+V5 and the clinical
set's rank among the 56 triples.

Output: runs/v7rev_stats/W1R_design_fold_check.csv (Supplementary Table S18).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys as _sys
from pathlib import Path as _Path
# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro
REPO = repro.repo()
sys.path.insert(0, str(REPO))

from igraphecg.data.dataset import load_processed
from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.data.ptbxl_loader import CANONICAL_LEADS
from igraphecg.evaluation.identifiability import lead_rows
from igraphecg.evaluation.round2_io import load_encoder_decoder, infer_theta

OUT = repro.outdir("runs/v7rev_stats")
NPZ = REPO / "data/processed/ptbxl_medianbeat_clean_100hz.npz"
LAM = 1e-3
SEEDS = (42, 1, 2, 3, 4)
IND_NAMES = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
IND = [CANONICAL_LEADS.index(x) for x in IND_NAMES]
INAME = {l: CANONICAL_LEADS[l] for l in IND}
CLIN = tuple(CANONICAL_LEADS.index(x) for x in ("II", "V1", "V5"))
DESIGNS = (("fold 10 (test)", lambda fold: fold == 10),
           ("fold 9 (validation)", lambda fold: fold == 9),
           ("folds 1-8 (training)", lambda fold: fold <= 8))


def subset(label, fold_sel, per_class=100, seed=42):
    """Class-balanced subset, the sampling rule of scripts/51 with the fold predicate free."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(len(TARGET_CLASSES)):
        pool = np.where((label == c) & fold_sel)[0]
        idx.append(pool if len(pool) <= per_class else rng.choice(pool, per_class, replace=False))
    return np.concatenate(idx)


def compute_J_all(dec, thetas, radius, device):
    from torch.func import jacfwd
    r = radius.to(device).view(1, -1)

    def f(th):
        y12, _ = dec.forward(th.unsqueeze(0))
        return y12[0].reshape(-1)
    return torch.stack([jacfwd(f)(thetas[s].to(device)) * r for s in range(thetas.shape[0])]).detach()


def logdet(F):
    return float(np.linalg.slogdet(F + LAM * np.eye(F.shape[0]))[1])


def trinv(F):
    return float(np.trace(np.linalg.inv(F + LAM * np.eye(F.shape[0]))))


def lmin(F):
    return float(np.linalg.eigvalsh(F + LAM * np.eye(F.shape[0]))[0])


def names(c):
    return "+".join(INAME[i] for i in c)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_processed(NPZ)
    signals = data["signal_12lead"].astype(np.float32)
    n_t = signals.shape[2]
    label = data["label"].astype(int)
    fold = data["fold"].astype(int)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals)
    combos = list(itertools.combinations(IND, 3))
    rows = []
    for seed in SEEDS:
        enc, dec = load_encoder_decoder(REPO / repro.lineage.checkpoint(seed), n_t, 3, dev)
        radius = dec.pspace.radius
        for design, sel in DESIGNS:
            idx = subset(label, sel(fold), seed=seed)
            theta, _ = infer_theta(enc, dec, scaled[idx], device=dev)
            J_all = compute_J_all(dec, torch.from_numpy(theta).to(dev), radius, dev)
            G = {}
            for l in IND:
                Jl = J_all[:, torch.tensor(lead_rows([l], n_t), device=dev), :]
                G[l] = torch.einsum("nrp,nrq->pq", Jl, Jl).cpu().numpy() / J_all.shape[0]

            def F_of(ls):
                return sum(G[l] for l in ls)
            Ds = {c: logdet(F_of(c)) for c in combos}
            As = {c: trinv(F_of(c)) for c in combos}
            Es = {c: lmin(F_of(c)) for c in combos}
            bD = max(Ds, key=Ds.get)
            bA = min(As, key=As.get)
            bE = max(Es, key=Es.get)
            rank = sorted(Ds, key=Ds.get, reverse=True)
            rows.append({"seed": seed, "design_subset": design, "n_records": int(len(idx)),
                         "D_optimal": names(bD), "A_optimal": names(bA), "E_optimal": names(bE),
                         "D_logdet_optimal": Ds[bD], "D_logdet_clinical": Ds[CLIN],
                         "D_margin_over_clinical": Ds[bD] - Ds[CLIN],
                         "D_margin_over_second": Ds[bD] - Ds[rank[1]],
                         "clinical_rank_of_56": rank.index(CLIN) + 1})
            print(f"[W1R] seed {seed:>2} {design:<22} D*={rows[-1]['D_optimal']:<9} "
                  f"A*={rows[-1]['A_optimal']:<9} E*={rows[-1]['E_optimal']:<9} "
                  f"margin={rows[-1]['D_margin_over_clinical']:.2f} clinical rank {rows[-1]['clinical_rank_of_56']}",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "W1R_design_fold_check.csv", index=False)
    same_D = all(g.D_optimal.nunique() == 1 for _, g in df.groupby("seed"))
    same_A = all(g.A_optimal.nunique() == 1 for _, g in df.groupby("seed"))
    spread = df.groupby("seed").D_margin_over_clinical.agg(lambda x: x.max() - x.min()).max()
    print(f"[W1R] D-optimal set identical across design subsets at every seed: {same_D}; "
          f"A-optimal: {same_A}; max within-seed spread of the margin: {spread:.3f}")
    print(f"[W1R] wrote {OUT / 'W1R_design_fold_check.csv'}")


if __name__ == "__main__":
    main()
