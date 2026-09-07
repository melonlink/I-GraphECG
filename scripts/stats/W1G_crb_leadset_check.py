# -*- coding: utf-8 -*-
"""W1G step 4: verify per-parameter CRB = 1 / identifiability_score, and reproduce every
row of the frozen crb_by_group_by_leadset.csv from per-parameter CRBs.

Replicates scripts/50_observability_phase1.py's path exactly (one J_all, then row selection),
which is the code that produced the frozen file, and cross-checks against
scripts/43_identifiability.py's path (compute_fim per lead set) used for the score table.

Emits W1G_crb_per_parameter_by_leadset.csv and W1G_crb_identity_check.csv.
"""
from __future__ import annotations

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

from igraphecg.data.dataset import load_processed                        # noqa: E402
from igraphecg.data.label_utils import TARGET_CLASSES                    # noqa: E402
from igraphecg.data.preprocess import RobustLeadScaler                   # noqa: E402
from igraphecg.data.ptbxl_loader import CANONICAL_LEADS                  # noqa: E402
from igraphecg.evaluation.identifiability import (                       # noqa: E402
    PARAM_GROUPS, identifiability_scores, lead_rows, param_names)
from igraphecg.evaluation.round2_io import infer_theta, load_encoder_decoder  # noqa: E402

OUT = repro.outdir("runs/v7rev_stats")
RUNS = repro.runs()
LAM = 1e-3


def fixed_subset(label, fold, per_class=100, seed=42):
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(len(TARGET_CLASSES)):
        pool = np.where((label == c) & (fold == 10))[0]
        idx.append(pool if len(pool) <= per_class else rng.choice(pool, per_class, replace=False))
    return np.concatenate(idx)


def compute_J_all(dec, thetas, radius, device):
    from torch.func import jacfwd
    r = radius.to(device).view(1, -1)

    def f(th):
        y12, _ = dec.forward(th.unsqueeze(0))
        return y12[0].reshape(-1)

    Js = []
    for s in range(thetas.shape[0]):
        Js.append((jacfwd(f)(thetas[s]) * r).detach())
    return torch.stack(Js)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_processed(REPO / "data/processed/ptbxl_medianbeat_clean_100hz.npz")
    signals = data["signal_12lead"].astype(np.float32)
    n_t = signals.shape[2]
    label = data["label"].astype(int); fold = data["fold"].astype(int)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals)
    ck = repro.lineage.checkpoint(42)
    enc, dec = load_encoder_decoder(ck, n_t, 3, dev)
    idx = fixed_subset(label, fold, 100)
    theta, _ = infer_theta(enc, dec, scaled[idx], device=dev)
    thetas = torch.from_numpy(theta).to(dev)
    J_all = compute_J_all(dec, thetas, dec.pspace.radius, dev)
    print(f"J_all {tuple(J_all.shape)}")

    names = param_names()
    lead_sets = {"L12": list(range(12)),
                 "II+V1+V5": [CANONICAL_LEADS.index(x) for x in ("II", "V1", "V5")],
                 "I+II+V5": [CANONICAL_LEADS.index(x) for x in ("I", "II", "V5")],
                 "II": [CANONICAL_LEADS.index("II")]}

    per_rows, chk_rows = [], []
    frozen = pd.read_csv(RUNS / "v1fix_phase1_s42/tables/crb_by_group_by_leadset.csv")
    frozen = frozen.set_index("lead_set")
    for lname, ls in lead_sets.items():
        rows = torch.tensor(lead_rows(ls, n_t), device=dev)
        Js = J_all[:, rows, :]
        F = (torch.einsum("nrp,nrq->pq", Js, Js) / J_all.shape[0]).cpu().numpy()
        cov = np.linalg.inv(F + LAM * np.eye(47))
        crb = np.sqrt(np.clip(np.diag(cov), 0, None))       # script 41's definition
        score, _ = identifiability_scores(F, lam=LAM)       # script 09's definition
        ident = np.abs(crb - 1.0 / score) / crb             # the relation under test
        for i, nm in enumerate(names):
            per_rows.append({"lead_set": lname, "index": i, "param": nm,
                             "crb_scaled": float(crb[i]),
                             "identifiability_score": float(score[i]),
                             "one_over_score": float(1.0 / score[i]),
                             "rel_diff_crb_vs_1_over_score": float(ident[i])})
        for g, r in PARAM_GROUPS.items():
            ours = float(np.mean(crb[list(r)]))
            theirs = float(frozen.loc[lname, f"CRB_{g}"])
            chk_rows.append({"lead_set": lname, "group": g,
                             "crb_group_mean_recomputed": ours,
                             "crb_group_mean_frozen": theirs,
                             "abs_diff": abs(ours - theirs),
                             "rel_diff": abs(ours - theirs) / abs(theirs)})
        print(f"{lname:9s} max rel|CRB - 1/score| = {ident.max():.3e}")

    pdf = pd.DataFrame(per_rows); cdf = pd.DataFrame(chk_rows)
    pdf.to_csv(OUT / "W1G_crb_per_parameter_by_leadset.csv", index=False, encoding="utf-8-sig")
    cdf.to_csv(OUT / "W1G_crb_identity_check.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(cdf.to_string(index=False))
    print(f"\nMAX rel diff, per-parameter identity CRB == 1/score : "
          f"{pdf['rel_diff_crb_vs_1_over_score'].max():.3e}")
    print(f"MAX rel diff, group CRB vs frozen phase-1 table     : {cdf['rel_diff'].max():.3e}")


if __name__ == "__main__":
    main()
