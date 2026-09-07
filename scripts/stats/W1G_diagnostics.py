# -*- coding: utf-8 -*-
"""W1G step 3: diagnostics that decide the tier rule and flag manuscript risks.

  (1) threshold sensitivity of the recommended per-parameter min-across-seeds rule
  (2) per-seed group means -> does 'the activation-delay group is the most identifiable'
      hold at the locked seed 42, or only in the five-seed mean?
  (3) the seed-42 edge_leaf collapse: within-group Fisher collinearity per seed
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro
REPO = repro.repo()
sys.path.insert(0, str(REPO))
from igraphecg.evaluation.identifiability import PARAM_GROUPS, param_names   # noqa: E402

OUT = repro.outdir("runs/v7rev_stats")
RUNS = repro.runs()
SEEDS = [42, 1, 2, 3, 4]
names = param_names()

S = np.zeros((5, 47))
for k, sd in enumerate(SEEDS):
    d = pd.read_csv(RUNS / f"v1fix_r3_s{sd}/tables/fim_parameter_identifiability.csv")
    S[k] = d.set_index("param")["identifiability_score"].reindex(names).to_numpy()
mean, sdv, mn, mx = S.mean(0), S.std(0, ddof=1), S.min(0), S.max(0)
cv = sdv / mean
C = np.stack([np.load(OUT / f"W1G_corr_s{sd}.npy") for sd in SEEDS])

# (1) threshold sensitivity -------------------------------------------------------
rows = []
for thr in [5, 8, 10, 12, 15, 18, 20, 25]:
    for stat, lab in [(mn, "min_across_seeds"), (mean, "five_seed_mean")]:
        m = stat >= thr
        rows.append({"statistic": lab, "threshold": thr, "n_tierA": int(m.sum()),
                     "members": ";".join(np.array(names)[m])})
sens = pd.DataFrame(rows)

# (2) per-seed group means --------------------------------------------------------
grows = []
for g, r in PARAM_GROUPS.items():
    idx = list(r)
    gm = S[:, idx].mean(1)
    row = {"group": g, "n_params": len(idx)}
    for k, sd in enumerate(SEEDS):
        row[f"mean_score_s{sd}"] = float(gm[k])
    row["five_seed_mean"] = float(gm.mean())
    row["five_seed_sd"] = float(gm.std(ddof=1))
    row["rank_at_seed42_1_is_highest"] = 0
    grows.append(row)
gdf = pd.DataFrame(grows)
gdf["rank_at_seed42_1_is_highest"] = gdf["mean_score_s42"].rank(ascending=False).astype(int)
gdf["rank_five_seed_mean_1_is_highest"] = gdf["five_seed_mean"].rank(ascending=False).astype(int)

# (3) within-group Fisher collinearity -------------------------------------------
crows = []
for g, r in PARAM_GROUPS.items():
    idx = list(r)
    if len(idx) < 2:
        continue
    for k, sd in enumerate(SEEDS):
        sub = np.abs(C[k][np.ix_(idx, idx)])
        iu = np.triu_indices(len(idx), 1)
        ev = np.linalg.eigvalsh(C[k][np.ix_(idx, idx)])[::-1]
        crows.append({"group": g, "seed": sd, "n_params": len(idx),
                      "mean_abs_pairwise_fisher_corr": float(sub[iu].mean()),
                      "max_abs_pairwise_fisher_corr": float(sub[iu].max()),
                      "top_eig_frac_of_subblock": float(ev[0] / ev.sum()),
                      "group_mean_ident_score": float(S[k, idx].mean())})
cdf = pd.DataFrame(crows)

sens.to_csv(OUT / "W1G_tier_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
gdf.to_csv(OUT / "W1G_group_scores_by_seed.csv", index=False, encoding="utf-8-sig")
cdf.to_csv(OUT / "W1G_within_group_collinearity.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250)
print("== threshold sensitivity ==")
print(sens[["statistic", "threshold", "n_tierA"]].pivot(
    index="threshold", columns="statistic", values="n_tierA").to_string())
print("\n== group means by seed ==")
print(gdf.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
print("\n== edge_leaf / q collinearity ==")
print(cdf[cdf["group"].isin(["edge_leaf", "edge_prox", "q", "tau_dep"])].to_string(
    index=False, float_format=lambda x: f"{x:8.4f}"))

extra = {
    "seed42_group_rank": dict(zip(gdf["group"], gdf["rank_at_seed42_1_is_highest"].astype(int))),
    "five_seed_group_rank": dict(zip(gdf["group"], gdf["rank_five_seed_mean_1_is_highest"].astype(int))),
    "edge_leaf_mean_abs_pairwise_corr_by_seed": {
        int(r.seed): round(float(r.mean_abs_pairwise_fisher_corr), 4)
        for r in cdf[cdf["group"] == "edge_leaf"].itertuples()},
    "edge_leaf_group_score_by_seed": {
        int(r.seed): round(float(r.group_mean_ident_score), 3)
        for r in cdf[cdf["group"] == "edge_leaf"].itertuples()},
}
(OUT / "W1G_diagnostics.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")
print("\n", json.dumps(extra, indent=2))
