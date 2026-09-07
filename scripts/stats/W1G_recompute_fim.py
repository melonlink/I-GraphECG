# -*- coding: utf-8 -*-
"""W1G step 1: recompute the 12-lead FIM for the five frozen seeds so that the
FULL 47x47 Fisher correlation matrix is available (runs/ only stored the top-20 pairs).

Reproduces scripts/43_identifiability.py bit-for-bit for the L12 lead set, then
CHECKS the recomputed identifiability score against the frozen CSV. If the check
fails the script aborts -- no papering over.

Outputs (into this directory):
  W1G_score_recheck.csv        per-seed max |recomputed - frozen| on the 47 scores
  W1G_corr_s{seed}.npy         full 47x47 Fisher correlation matrix
  W1G_score_s{seed}.npy        47 identifiability scores in canonical param order
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

from igraphecg.data.dataset import load_processed                      # noqa: E402
from igraphecg.data.label_utils import TARGET_CLASSES                  # noqa: E402
from igraphecg.data.preprocess import RobustLeadScaler                 # noqa: E402
from igraphecg.evaluation.identifiability import (                     # noqa: E402
    compute_fim, identifiability_scores, lead_rows, param_names, PARAM_GROUPS)
from igraphecg.evaluation.lead_ablation import LEAD_SETS               # noqa: E402
from igraphecg.evaluation.round2_io import infer_theta, load_encoder_decoder  # noqa: E402
from igraphecg.utils.seed import set_seed                              # noqa: E402

OUT = repro.outdir("runs/v7rev_stats")
RUNS = repro.runs()
SEEDS = [42, 1, 2, 3, 4]


def subset_idx(label, fold, per_class, seed):
    """verbatim from scripts/43_identifiability.py"""
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(len(TARGET_CLASSES)):
        pool = np.where((label == c) & (fold == 10))[0]
        idx.append(pool if len(pool) <= per_class else rng.choice(pool, per_class, replace=False))
    return np.concatenate(idx)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    data = load_processed(REPO / "data/processed/ptbxl_medianbeat_clean_100hz.npz")
    signals = data["signal_12lead"].astype(np.float32)
    n_t = signals.shape[2]
    label = data["label"].astype(int)
    fold = data["fold"].astype(int)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals)
    names = param_names()
    assert len(names) == 47, len(names)

    rows = []
    for sd in SEEDS:
        set_seed(sd)
        ck = repro.lineage.checkpoint(sd)
        enc, dec = load_encoder_decoder(ck, n_t, 3, device)
        idx = subset_idx(label, fold, 100, sd)
        theta, _ = infer_theta(enc, dec, scaled[idx], device=device)
        thetas = torch.from_numpy(theta).to(device)
        radius = dec.pspace.radius
        rws = torch.tensor(lead_rows(LEAD_SETS["L12"], n_t), dtype=torch.long, device=device)
        F12 = compute_fim(dec, thetas, radius, observed_rows=rws, device=device)
        score, corr = identifiability_scores(F12, lam=1e-3)

        frozen = pd.read_csv(RUNS / f"v1fix_r3_s{sd}/tables/fim_parameter_identifiability.csv")
        fz = frozen.set_index("param")["identifiability_score"].reindex(names).to_numpy()
        adiff = np.abs(score - fz)
        rdiff = adiff / np.abs(fz)
        rows.append({"seed": sd, "n_subset": int(len(idx)),
                     "max_abs_diff": float(adiff.max()),
                     "max_rel_diff": float(rdiff.max()),
                     "worst_param": names[int(np.argmax(rdiff))]})
        print(f"seed {sd}: n={len(idx)} max_abs={adiff.max():.3e} max_rel={rdiff.max():.3e} "
              f"worst={names[int(np.argmax(rdiff))]}")
        np.save(OUT / f"W1G_corr_s{sd}.npy", corr)
        np.save(OUT / f"W1G_score_s{sd}.npy", score)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "W1G_score_recheck.csv", index=False)
    print(df.to_string(index=False))
    if df["max_rel_diff"].max() > 1e-3:
        print("!! RECOMPUTATION DOES NOT REPRODUCE THE FROZEN SCORES -- STOP")
        sys.exit(2)
    print("OK: recomputed identifiability scores reproduce the frozen tables.")


if __name__ == "__main__":
    main()
