"""W1B finalize: fix the boolean-count bug in the band-split JSON and emit a single
headline JSON holding every number the manuscript will quote."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro
OUT = repro.outdir("runs/v7rev_stats")
CLS = ["NORM", "MI", "STTC", "CD"]


def tobool(s):
    return s.astype(str).str.strip().str.lower().eq("true")


def ms(x):
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "sd_ddof1": float(x.std(ddof=1)),
            "min": float(x.min()), "max": float(x.max()), "n_seeds": int(len(x))}


d = pd.read_csv(OUT / "W1B_substitution_oracle.csv")
var = d[d.test.isin(["T1_variant", "T1b_variant"])]
# T1b re-evaluates the C5 baseline; count each (variant, seed) once, preferring T1
var = var.sort_values("test").drop_duplicates(["variant", "seed"], keep="first")
del_ = d[d.test.isin(["T1_paired_delta", "T1b_paired_delta"])].copy()
del_["cz"] = tobool(del_["crosses_zero"])

variants = {}
for name, g in var.groupby("variant"):
    variants[name] = {
        "n_features": int(g["n_features"].iloc[0]),
        "seed42_macro_auroc": float(g[g.seed.astype(str) == "42"]["macro_auroc"].iloc[0]),
        "macro_auroc": ms(g["macro_auroc"]),
        "balanced_accuracy": ms(g["balanced_accuracy"]),
        "recall": {c: ms(g[f"recall_{c}"]) for c in CLS},
        "MI_to_NORM": ms(g["MI_to_NORM"]),
    }

deltas = {}
for (name, metric), g in del_.groupby(["variant", "metric"]):
    deltas[f"{name}|{metric}"] = {
        **ms(g["difference"]),
        "n_seeds_positive": int((g["difference"] > 0).sum()),
        "n_seeds_ci_excludes_zero": int((~g["cz"]).sum()),
        "seed42": {"difference": float(g[g.seed.astype(str) == "42"]["difference"].iloc[0]),
                   "ci_lo": float(g[g.seed.astype(str) == "42"]["ci_lo"].iloc[0]),
                   "ci_hi": float(g[g.seed.astype(str) == "42"]["ci_hi"].iloc[0])},
    }

# ---- within-class quartile trends -------------------------------------------------------
w = pd.read_csv(OUT / "W1B_strata_within_class.csv")
w = w[w.seed != "MEAN_5SEED"].copy()
w["seed"] = w["seed"].astype(int)
within = {}
for c in CLS:
    s = w[w["class"] == c]
    p = s.pivot(index="seed", columns="quartile", values="recall")
    mp = s.pivot(index="seed", columns="quartile", values="mean_p_true")
    within[c] = {
        "recall_by_quartile_mean": [float(p[q].mean()) for q in (1, 2, 3, 4)],
        "recall_by_quartile_sd": [float(p[q].std(ddof=1)) for q in (1, 2, 3, 4)],
        "Q4_minus_Q1_recall": ms(p[4] - p[1]),
        "n_seeds_Q4_gt_Q1": int(((p[4] - p[1]) > 0).sum()),
        "mean_p_true_by_quartile_mean": [float(mp[q].mean()) for q in (1, 2, 3, 4)],
        "n_per_quartile": [int(s[s.quartile == q]["n"].iloc[0]) for q in (1, 2, 3, 4)],
        "median_nrmse_by_quartile_mean": [float(s[s.quartile == q]["nrmse_median"].mean())
                                          for q in (1, 2, 3, 4)],
    }
mi = w[w["class"] == "MI"].copy()
mi["n_to_NORM"] = mi["frac_pred_NORM"] * mi["n"]
t = mi.groupby("quartile")["n_to_NORM"].mean()
within["MI"]["MI_to_NORM_by_quartile_mean"] = [float(t[q]) for q in (1, 2, 3, 4)]
within["MI"]["MI_to_NORM_total_mean"] = float(t.sum())
within["MI"]["pct_of_MI_to_NORM_in_two_best_quartiles"] = float(100 * t[[1, 2]].sum() / t.sum())

# ---- pooled quartiles --------------------------------------------------------------------
pl = pd.read_csv(OUT / "W1B_strata_pooled.csv")
plm = pl[pl.seed == "MEAN_5SEED"]
pooled = {k: [float(plm[plm.quartile == q][f"{k}_mean"].iloc[0]) for q in (1, 2, 3, 4)]
          for k in ["n", "nrmse_median", "corr_median", "macro_auroc", "balanced_accuracy",
                    "accuracy", "mean_p_true", "recall_NORM", "recall_MI", "recall_STTC",
                    "recall_CD", "frac_NORM", "frac_MI", "frac_STTC", "frac_CD"]}

# ---- Spearman ----------------------------------------------------------------------------
sp = pd.read_csv(OUT / "W1B_recon_vs_trueprob_spearman.csv")
spa = sp[sp.seed == "AGG_5SEED"]
spp = sp[sp.seed != "AGG_5SEED"]
spear = {}
for c in CLS:
    spear[c] = {}
    for em in ["nrmse_scaled_12lead", "corr_meanlead_12lead"]:
        r = spa[(spa["class"] == c) & (spa.error_metric == em)].iloc[0]
        pv = spp[(spp["class"] == c) & (spp.error_metric == em)]["p_value"].astype(float)
        spear[c][em] = {"rho_mean": float(r.rho_mean), "rho_sd_ddof1": float(r.rho_std_ddof1),
                        "rho_min": float(r.rho_min), "rho_max": float(r.rho_max),
                        "max_p_across_seeds": float(pv.max()),
                        "n_seeds_p_lt_0.05": int((pv < 0.05).sum())}

ca = json.load(open(OUT / "W1B_controls_and_class_association.json"))
head = {
    "task": "W1B",
    "question": "does reconstruction error propagate to descriptors and classification?",
    "controls_seed42": ca["controls_seed42"],
    "C5_swappable_recon_descriptors": ca["C5_swappable_descriptors"],
    "TEST1_variants": variants,
    "TEST1_paired_deltas": deltas,
    "TEST2_pooled_quartiles_five_seed_mean": pooled,
    "TEST2_within_class": within,
    "TEST2_spearman_recon_vs_P_true_class": spear,
    "class_level_confound": {
        "spearman_median_corr_vs_recall_per_seed": [
            ca["class_level_association"][s]["spearman_medcorr_vs_recall"]
            for s in ["42", "1", "2", "3", "4"]],
        "spearman_median_nrmse_vs_recall_per_seed": [
            ca["class_level_association"][s]["spearman_mednrmse_vs_recall"]
            for s in ["42", "1", "2", "3", "4"]],
        "note": "n = 4 classes, so p = 0.20 for rho = 0.8; the association is real but "
                "cannot by itself be given a causal reading.",
        "seed42_median_corr_by_class": ca["class_level_association"]["42"]["median_corr_by_class"],
        "seed42_median_nrmse_by_class": ca["class_level_association"]["42"]["median_nrmse_by_class"],
        "seed42_recall_by_class": ca["class_level_association"]["42"]["recall_by_class"],
    },
}
json.dump(head, open(OUT / "W1B_headline.json", "w"), indent=1)

bs = json.load(open(OUT / "W1B_bandsplit_headline.json"))
for k in bs["delta_macro_auroc_vs_C5"]:
    bs["delta_macro_auroc_vs_C5"][k]["n_seeds_ci_excludes_0"] = \
        deltas[f"{k}|macro_auroc"]["n_seeds_ci_excludes_zero"]
    bs["delta_macro_auroc_vs_C5"][k]["n_seeds_positive"] = \
        deltas[f"{k}|macro_auroc"]["n_seeds_positive"]
bs["per_descriptor_delta"] = {"band_A_4_descriptors": bs["delta_macro_auroc_vs_C5"]
                              ["C5_obsA_minus_C5"]["mean"] / 4,
                              "band_C_2_descriptors": bs["delta_macro_auroc_vs_C5"]
                              ["C5_obsC_minus_C5"]["mean"] / 2}
json.dump(bs, open(OUT / "W1B_bandsplit_headline.json", "w"), indent=1)
print(json.dumps(bs, indent=1))
print("wrote W1B_headline.json")
