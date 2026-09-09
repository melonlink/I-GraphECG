"""W1U: lead selection under additional noise models and wearable-feasible families
(Reviewer 3, Comment 5).

Section 3.6 selects three-lead sets under two noise models (identity, empirical diagonal). This
script repeats the exhaustive 56-subset D-optimal selection of scripts/stats/W1J on the same
five checkpoints and the same 400-record design subset under five further models, all of them
expressed through the noise covariance Sigma that enters F = S^T Sigma^{-1} S:

  identity            Sigma = I                                        (Section 3.6, control)
  empirical_diag      per-lead pre-QRS variance, diagonal               (Section 3.6, control)
  empirical_full      the full 8x8 pre-QRS covariance of the independent leads (cross-lead
                      correlation of the unmodeled baseline), whitened jointly
  limb_artifact_x4    limb-lead (I, II) noise variance four times the precordial (motion /
                      muscle artifact on limb electrodes, the wearable's usual failure mode)
  chest_artifact_x4   the mirror image: precordial variance four times the limb
  ar1_rho0.8          coloured noise: AR(1) temporal correlation (rho = 0.8) within every lead,
                      equal variance across leads
  empirical_full_ar1  the empirical full lead covariance combined with the AR(1) temporal
                      correlation (Kronecker structure), the most realistic model here

For each model and seed: the D-optimal triple, its log det F_lambda margin over the clinical set
II+V1+V5, the clinical set's rank among the 56, whether lead II enters the optimum, and whether at
least one of V2-V5 does. Wearable-feasible families are then scored under every model: at most one
precordial lead (limb-based device with one chest electrode), precordial leads only (chest patch),
and triples that must contain lead II (a lead-II device extended by two leads).

Writes runs/v7rev_stats/W1U_noise_models.csv, W1U_families.csv, W1U_headline.json.
Run (repository root, torch environment): python scripts/stats/W1U_noise_models.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from igraphecg import repro  # noqa: E402
from igraphecg.data.dataset import load_processed  # noqa: E402
from igraphecg.data.label_utils import TARGET_CLASSES  # noqa: E402
from igraphecg.data.preprocess import RobustLeadScaler  # noqa: E402
from igraphecg.data.ptbxl_loader import CANONICAL_LEADS  # noqa: E402
from igraphecg.evaluation.identifiability import lead_rows  # noqa: E402
from igraphecg.evaluation.round2_io import infer_theta, load_encoder_decoder  # noqa: E402

OUT = repro.outdir("runs/v7rev_stats")
NPZ = ROOT / "data/processed/ptbxl_medianbeat_clean_100hz.npz"
LAM = 1e-3
SEEDS = [42, 1, 2, 3, 4]
PRE_SAMPLES = 20
RHO = 0.8
IND_NAMES = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
IND = [CANONICAL_LEADS.index(x) for x in IND_NAMES]
NAME = {l: CANONICAL_LEADS[l] for l in IND}
CLIN = tuple(CANONICAL_LEADS.index(x) for x in ("II", "V1", "V5"))
PRECORDIAL = {"V1", "V2", "V3", "V4", "V5", "V6"}
II = CANONICAL_LEADS.index("II")


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

    return torch.stack([jacfwd(f)(thetas[s].to(device)) * r for s in range(thetas.shape[0])]).detach()


def logdet(F):
    return float(np.linalg.slogdet(F + LAM * np.eye(F.shape[0]))[1])


def setname(leads):
    return "+".join(NAME[i] for i in leads)


def nprec(leads):
    return sum(1 for l in leads if NAME[l] in PRECORDIAL)


def ar1_inverse(n_t, rho):
    """Inverse of the AR(1) correlation matrix, scaled to unit mean diagonal."""
    t = np.arange(n_t)
    R = rho ** np.abs(t[:, None] - t[None, :])
    Ri = np.linalg.inv(R)
    return Ri / np.mean(np.diag(Ri))


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_processed(NPZ)
    signals = data["signal_12lead"].astype(np.float32)
    n_t = signals.shape[2]
    label, fold = data["label"].astype(int), data["fold"].astype(int)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals).astype(np.float32)

    # ---- noise statistics of the pre-QRS segment on the training folds (scaled space) --------
    b = scaled[fold <= 8][:, IND, :PRE_SAMPLES]
    b = b - b.mean(axis=2, keepdims=True)
    var8 = b.var(axis=2).mean(axis=0)                                   # [8]
    cov8 = np.einsum("nlt,nmt->lm", b, b) / (b.shape[0] * b.shape[2])  # [8,8]
    corr8 = cov8 / np.sqrt(np.outer(np.diag(cov8), np.diag(cov8)))
    Ri = ar1_inverse(n_t, RHO)

    def diag_model(w):
        w = np.asarray(w, dtype=float)
        return np.diag(w / w.mean())                                     # Sigma^{-1}, unit mean

    Sinv_emp_diag = diag_model(1.0 / var8)
    Sinv_full = np.linalg.inv(cov8)
    Sinv_full = Sinv_full / np.mean(np.diag(Sinv_full))
    limb = np.array([0.25 if NAME[l] not in PRECORDIAL else 1.0 for l in IND])
    chest = np.array([1.0 if NAME[l] not in PRECORDIAL else 0.25 for l in IND])
    models = {
        "identity": (np.eye(8), None),
        "empirical_diag": (Sinv_emp_diag, None),
        "empirical_full": (Sinv_full, None),
        "limb_artifact_x4": (diag_model(limb), None),
        "chest_artifact_x4": (diag_model(chest), None),
        "ar1_rho0.8": (np.eye(8), Ri),
        "empirical_full_ar1": (Sinv_full, Ri),
    }
    combos = list(itertools.combinations(range(8), 3))
    assert len(combos) == 56
    families = {
        "unconstrained": lambda c: True,
        "at_most_one_precordial": lambda c: nprec([IND[i] for i in c]) <= 1,
        "precordial_only": lambda c: nprec([IND[i] for i in c]) == 3,
        "contains_II": lambda c: IND.index(II) in c,
    }
    clin_c = tuple(IND.index(l) for l in CLIN)

    rows, fam_rows, all_rows = [], [], []
    for seed in SEEDS:
        ck = repro.lineage.checkpoint(seed)
        enc, dec = load_encoder_decoder(ck, n_t, 3, dev)
        idx = fixed_subset(label, fold)
        theta, _ = infer_theta(enc, dec, scaled[idx], device=dev)
        J = compute_J_all(dec, torch.from_numpy(theta), dec.pspace.radius, dev)   # [N,1200,47]
        Jl = [J[:, torch.tensor(lead_rows([l], n_t), device=dev), :] for l in IND]      # 8 x [N,T,47]
        Ri_t = torch.tensor(Ri, dtype=J.dtype, device=dev)
        # cross-lead blocks, white and AR(1)-whitened in time, averaged over records
        G = {}
        for a in range(8):
            for c in range(8):
                G[("white", a, c)] = torch.einsum("ntp,ntq->pq", Jl[a], Jl[c]).double().cpu().numpy() / len(idx)
                G[("ar1", a, c)] = torch.einsum("ntp,ts,nsq->pq", Jl[a], Ri_t, Jl[c]).double().cpu().numpy() / len(idx)
        del J, Jl
        if dev == "cuda":
            torch.cuda.empty_cache()
        print(f"[W1U] seed {seed}: FIM blocks ready ({time.time()-t0:.0f}s)")

        for mname, (Sinv, temporal) in models.items():
            kind = "ar1" if temporal is not None else "white"

            def F_of(c):
                sub = np.ix_(list(c), list(c))
                # F(S) = sum_{l,m in S} [Sigma_S^{-1}]_{lm} G_lm, with Sigma_S the S x S block
                Ssub_inv = np.linalg.inv(np.linalg.inv(Sinv)[sub])
                F = np.zeros((47, 47))
                for i, a in enumerate(c):
                    for j, d in enumerate(c):
                        F += Ssub_inv[i, j] * G[(kind, a, d)]
                return F

            scores = {c: logdet(F_of(c)) for c in combos}
            order = sorted(scores, key=scores.get, reverse=True)
            opt = order[0]
            for rank, c in enumerate(order, start=1):
                all_rows.append({"seed": seed, "noise_model": mname, "leads": setname([IND[i] for i in c]),
                                 "D_logdet": scores[c], "rank": rank, "n_precordial": nprec([IND[i] for i in c])})
            rows.append({
                "seed": seed, "noise_model": mname, "optimal_set": setname([IND[i] for i in opt]),
                "D_logdet_optimal": scores[opt], "D_logdet_clinical": scores[clin_c],
                "margin_over_clinical": scores[opt] - scores[clin_c],
                "clinical_rank_of_56": order.index(clin_c) + 1,
                "II_in_optimum": IND.index(II) in opt,
                "II_in_top5": any(IND.index(II) in c for c in order[:5]),
                "n_precordial_in_optimum": nprec([IND[i] for i in opt]),
                "V2_to_V5_in_optimum": any(NAME[IND[i]] in ("V2", "V3", "V4", "V5") for i in opt),
                "second_set": setname([IND[i] for i in order[1]]),
                "margin_first_to_second": scores[opt] - scores[order[1]],
            })
            for fname, ok in families.items():
                cand = [c for c in combos if ok(c)]
                best = max(cand, key=scores.get)
                fam_rows.append({
                    "seed": seed, "noise_model": mname, "family": fname, "n_candidates": len(cand),
                    "best_set": setname([IND[i] for i in best]), "D_logdet_best": scores[best],
                    "loss_vs_unconstrained_optimum": scores[best] - scores[opt],
                    "margin_over_clinical": scores[best] - scores[clin_c],
                    "rank_of_56": order.index(best) + 1,
                })
            print(f"[W1U]   {mname:20s} optimum {setname([IND[i] for i in opt]):10s} margin {scores[opt]-scores[clin_c]:+7.2f} "
                  f"clinical rank {order.index(clin_c)+1:2d}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "W1U_noise_models.csv", index=False, float_format="%.6g")
    fam = pd.DataFrame(fam_rows)
    fam.to_csv(OUT / "W1U_families.csv", index=False, float_format="%.6g")
    pd.DataFrame(all_rows).to_csv(OUT / "W1U_all56_by_model.csv", index=False, float_format="%.6g")
    head = {
        "seeds": SEEDS, "lambda": LAM, "ar1_rho": RHO, "pre_qrs_samples": PRE_SAMPLES,
        "noise_models": list(models),
        "empirical_lead_correlation_8": {f"{NAME[IND[a]]}-{NAME[IND[c]]}": float(corr8[a, c]) for a in range(8) for c in range(a + 1, 8)},
        "max_abs_offdiag_lead_correlation": float(np.max(np.abs(corr8 - np.eye(8)))),
        "II_never_in_optimum": bool((~df.II_in_optimum).all()),
        "II_never_in_top5": bool((~df.II_in_top5).all()),
        "V2_to_V5_in_every_optimum": bool(df.V2_to_V5_in_optimum.all()),
        "min_margin_over_clinical": float(df.margin_over_clinical.min()),
        "max_clinical_rank": int(df.clinical_rank_of_56.max()),
        "min_clinical_rank": int(df.clinical_rank_of_56.min()),
        "optima_seed42": {m: str(df[(df.seed == 42) & (df.noise_model == m)].optimal_set.iloc[0]) for m in models},
        "distinct_optima_all_seeds_models": sorted(df.optimal_set.unique().tolist()),
        "families_seed42": {f: {m: {"best_set": str(r.best_set), "loss_vs_optimum": float(r.loss_vs_unconstrained_optimum),
                                    "margin_over_clinical": float(r.margin_over_clinical)}
                                for m, r in ((m, fam[(fam.seed == 42) & (fam.noise_model == m) & (fam.family == f)].iloc[0]) for m in models)}
                           for f in families},
    }
    (OUT / "W1U_headline.json").write_text(json.dumps(head, indent=2), encoding="utf-8")
    print(df[["seed", "noise_model", "optimal_set", "margin_over_clinical", "clinical_rank_of_56", "II_in_optimum", "V2_to_V5_in_optimum"]].to_string(index=False))
    print(f"[W1U] done ({time.time()-t0:.0f}s) -> {OUT}")


if __name__ == "__main__":
    main()
