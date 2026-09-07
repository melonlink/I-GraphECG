# -*- coding: utf-8 -*-
"""W1G step 2: build the 47-row parameter table with a seed-robust identifiability tiering.

Reads the FROZEN per-seed identifiability tables (v1fix_r3_s{42,1,2,3,4}), the bounds from
igraphecg.models.surrogate_decoder._bounds(), the grouping from
igraphecg.evaluation.identifiability.PARAM_GROUPS, and the full 47x47 Fisher correlation
matrices recomputed (and verified bit-exact) by W1G_recompute_fim.py.

Emits:
  W1G_parameter_table.csv   47 rows
  W1G_tier_rules.csv        candidate tier rules with sizes and membership
  W1G_crb_verification.csv  per-parameter CRB = 1/score, checked against
                            v1fix_phase1_s42/tables/crb_by_group_by_leadset.csv
  W1G_headline.json         numbers the manuscript will quote
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
from igraphecg.models.surrogate_decoder import _bounds, EDGES, VENT_NODES    # noqa: E402

OUT = repro.outdir("runs/v7rev_stats")
RUNS = repro.runs()
SEEDS = [42, 1, 2, 3, 4]

# ---------------------------------------------------------------- semantics
NODE = {0: "SA / atrial source (graph root)", 1: "AV node - His entry",
        2: "interventricular septum", 3: "right-ventricular free wall",
        4: "LV anterior wall", 5: "LV lateral wall", 6: "LV inferior wall", 7: "LV apex"}
NODE_SHORT = {0: "SA/atrium", 1: "AV/His", 2: "septum", 3: "RV", 4: "LV-ant",
              5: "LV-lat", 6: "LV-inf", 7: "apex"}
# ST region assignment, from igraphecg/evaluation/st_features.py REGION
ST_REGION = {0: "right_septal", 1: "right_septal", 2: "anterior",
             3: "lateral", 4: "inferior", 5: "anterior"}

UNIT_S = "s (seconds, physical parameter unit)"
UNIT_MV = "mV-equivalent source strength (amplitude defined up to the per-lead robust scaling)"
UNIT_1 = "dimensionless (relative source amplitude)"
UNIT_G = "dimensionless (global amplitude scale)"


def meanings():
    """(meaning, unit, physical_unit_short) for each of the 47 coordinates, in _SEG order."""
    m, u, us = [], [], []
    m.append("Activation onset of the graph root (SA / atrial source) relative to the R peak; "
             "a global time origin for the whole activation sequence")
    u.append(UNIT_S); us.append("s")
    for k, (a, b) in enumerate(EDGES):
        lab = f"conduction delay on graph edge e{k} ({NODE_SHORT[a]} -> {NODE_SHORT[b]})"
        if k == 0:
            lab += "; carries the AV-nodal delay (PR-segment analogue)"
        elif k == 1:
            lab += "; His-Purkinje entry into the septum"
        else:
            lab += "; fascicular / branch delay to that ventricular wall"
        m.append(lab); u.append(UNIT_S); us.append("s")
    for i in range(8):
        m.append(f"Equivalent action-potential duration of node {i} ({NODE[i]})")
        u.append(UNIT_S); us.append("s")
    for i in range(8):
        m.append(f"Depolarization (upstroke) time constant tau_dep of node {i} ({NODE[i]}); "
                 f"sets the QRS-limb slope contributed by that node")
        u.append(UNIT_S); us.append("s")
    for i in range(8):
        m.append(f"Repolarization (downstroke) time constant tau_rep of node {i} ({NODE[i]}); "
                 f"sets the T-wave limb slope contributed by that node")
        u.append(UNIT_S); us.append("s")
    for i in range(8):
        m.append(f"Nodal source gain q of node {i} ({NODE[i]}); scales dU_i/dt, i.e. that node's "
                 f"contribution to the QRS/T dipole")
        u.append(UNIT_1); us.append("1")
    for k in range(6):
        n = VENT_NODES[k]
        m.append(f"ST-platform source amplitude alpha_ST of ventricular node {n} ({NODE[n]}); "
                 f"ST region '{ST_REGION[k]}'. Signed: positive = ST elevation-like source")
        u.append(UNIT_MV); us.append("mV-eq")
    m.append("Global amplitude gain g applied to the reconstructed 12-lead beat")
    u.append(UNIT_G); us.append("1")
    return m, u, us


# clinical interpretation status in the v6 manuscript (Section 4.4 / Table 5 / Fig. 5)
def clinical_use(name, group):
    if group == "APD" and name in [f"APD{i}" for i in VENT_NODES]:
        return ("stated theta-domain result",
                "node-level APD dispersion D_rep (max-min APD over ventricular nodes); "
                "manuscript reports it as LOWER in STTC, used as the counter-constructive finding")
    if group == "APD":
        return ("no", "atrial / AV-node APD; enters no interpreted descriptor "
                      "(D_rep and mean_vent_APD are ventricular-only)")
    if group == "edge_prox":
        return ("negative example only",
                "prox_delay = e0 + e1; manuscript reports it as seed-unstable (sign flip) and "
                "explicitly NOT claim-supporting")
    if group == "alpha_ST":
        return ("observability-limited, not a marker",
                "||alpha_ST|| (S_ST) and lead-projected ST = H_ind*alpha_ST; manuscript uses it "
                "as the near-null direction explaining MI under-detection, and reports the "
                "lead-projected ST as weak/seed-variable (negative example)")
    if group == "edge_leaf":
        return ("indirect only",
                "enters D_act / leaf_delay_spread / vent_delta_spread, which are computed but not "
                "interpreted; conduction is interpreted via QRS duration measured on the "
                "reconstructed waveform instead")
    if group == "tau_rep":
        return ("indirect only",
                "ventricular tau_rep enters tau_rep_disp / Tend / Sync_rep and shapes the "
                "reconstructed T wave, from which the interpreted T-wave descriptors are measured; "
                "no claim is made on tau_rep itself")
    if group == "delta_root":
        return ("no", "global time origin; cancels from every dispersion descriptor "
                      "(D_act, Tend_disp, Sync_rep) and the beat is R-peak aligned")
    if group in ("tau_dep", "q", "global_gain"):
        return ("indirect only",
                "shapes the reconstructed beat from which the interpreted ECG-domain descriptors "
                "(QRS duration, T-peak amplitude, T-wave sign discordance) are measured; "
                "never read as a clinical quantity on its own")
    raise ValueError(name)


def main():
    names = param_names()
    lo, hi = _bounds()
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    radius = (hi - lo) / 2.0
    center = (hi + lo) / 2.0
    assert len(names) == len(lo) == 47

    group = [g for g, r in PARAM_GROUPS.items() for _ in r]

    # ---- frozen per-seed scores -------------------------------------------------
    S = np.zeros((len(SEEDS), 47))
    for k, sd in enumerate(SEEDS):
        d = pd.read_csv(RUNS / f"v1fix_r3_s{sd}/tables/fim_parameter_identifiability.csv")
        S[k] = d.set_index("param")["identifiability_score"].reindex(names).to_numpy()
    s42 = S[0]
    mean = S.mean(0); sdv = S.std(0, ddof=1); mn = S.min(0); mx = S.max(0)
    cv = sdv / mean

    # ---- CRB (scaled units = fractions of the parameter's own half-range) --------
    crb42 = 1.0 / s42
    crb_mean5 = (1.0 / S).mean(0)
    crb_phys42 = crb42 * radius

    # ---- Fisher coupling --------------------------------------------------------
    C = np.stack([np.load(OUT / f"W1G_corr_s{sd}.npy") for sd in SEEDS])   # [5,47,47]
    A = np.abs(C).copy()
    for k in range(len(SEEDS)):
        np.fill_diagonal(A[k], -1.0)
    partner_idx = A[0].argmax(1)
    partner = [names[j] for j in partner_idx]
    partner_corr42 = np.array([C[0, i, partner_idx[i]] for i in range(47)])
    partner_abs_mean5 = np.array([np.abs(C[:, i, partner_idx[i]]).mean() for i in range(47)])
    partner_agree = np.array([int(sum(A[k].argmax(1)[i] == partner_idx[i]
                                      for k in range(len(SEEDS)))) for i in range(47)])
    max_abs_corr_5seed_mean = A.max(2).mean(0)

    # ---- table ------------------------------------------------------------------
    mng, unit, ushort = meanings()
    cu = [clinical_use(n, g) for n, g in zip(names, group)]
    df = pd.DataFrame({
        "index": np.arange(47),
        "name": names,
        "group": group,
        "physiological_meaning": mng,
        "bound_lo": lo, "bound_hi": hi,
        "bound_center": center, "bound_radius": radius,
        "units": unit,
        "unit_short": ushort,
        "ident_score_seed42": s42,
        "ident_score_5seed_mean": mean,
        "ident_score_5seed_sd": sdv,
        "ident_score_5seed_min": mn,
        "ident_score_5seed_max": mx,
        "ident_score_5seed_cv": cv,
        "crb_12lead_seed42_scaled": crb42,
        "crb_12lead_5seed_mean_scaled": crb_mean5,
        "crb_12lead_seed42_physical": crb_phys42,
        "crb_physical_unit": ushort,
        "crb_units_note": "scaled = standard deviation in units of the parameter's own bound "
                          "radius (half-width); physical = scaled x bound_radius",
        "strongest_coupling_partner": partner,
        "partner_corr_seed42": partner_corr42,
        "partner_abs_corr_seed42": np.abs(partner_corr42),
        "partner_abs_corr_5seed_mean": partner_abs_mean5,
        "partner_same_in_n_of_5_seeds": partner_agree,
        "max_abs_fisher_corr_5seed_mean": max_abs_corr_5seed_mean,
        "clinical_interpretation_in_manuscript": [c[0] for c in cu],
        "interpreting_descriptor": [c[1] for c in cu],
    })

    # ---- candidate tier rules ---------------------------------------------------
    def tier_from_masks(a_mask, c_mask):
        t = np.where(a_mask, "A_strongly_identifiable",
                     np.where(c_mask, "C_primarily_regularized", "B_weakly_identifiable"))
        return t

    gmean = {g: S[:, list(r)].mean(1) for g, r in PARAM_GROUPS.items()}   # per-seed group mean
    gcv = {g: float(v.std(ddof=1) / v.mean()) for g, v in gmean.items()}
    gm = {g: float(v.mean()) for g, v in gmean.items()}
    gA = np.array([(gm[g] >= 15.0) and (gcv[g] <= 0.20) for g in group])
    gC = np.array([gm[g] < 1.0 for g in group])

    rules = {}
    rules["R1_group_mean_cv (the flawed rule)"] = dict(
        desc="Tier A if the parameter's GROUP has five-seed mean score >= 15 and group-level "
             "CV <= 0.20; Tier C if group mean < 1. Group-mean CV hides per-parameter spread.",
        tier=tier_from_masks(gA, gC))
    rules["R2_min_across_seeds"] = dict(
        desc="Tier A if min-across-seeds score >= 15 (per-parameter); "
             "Tier C if max-across-seeds score < 1 (CRB > 1 radius in every seed).",
        tier=tier_from_masks(mn >= 15.0, mx < 1.0))
    rules["R3_per_param_cv"] = dict(
        desc="Tier A if five-seed mean score >= 15 AND per-parameter CV <= 0.35; "
             "Tier C if five-seed mean score < 1.",
        tier=tier_from_masks((mean >= 15.0) & (cv <= 0.35), mean < 1.0))
    rules["R4_mean_only"] = dict(
        desc="Tier A if five-seed mean score >= 15 (no robustness term); Tier C if mean < 1.",
        tier=tier_from_masks(mean >= 15.0, mean < 1.0))
    rules["R5_min_and_cv"] = dict(
        desc="Tier A if min-across-seeds score >= 15 AND per-parameter CV <= 0.50; "
             "Tier C if max-across-seeds score < 1. Identical to R2 on this data: the CV term "
             "is non-binding once min-across-seeds is enforced, so R2 is preferred as simpler.",
        tier=tier_from_masks((mn >= 15.0) & (cv <= 0.50), mx < 1.0))

    rows = []
    for rname, r in rules.items():
        t = r["tier"]
        df[f"tier_{rname.split()[0]}"] = t
        for tl in ["A_strongly_identifiable", "B_weakly_identifiable", "C_primarily_regularized"]:
            mem = [names[i] for i in range(47) if t[i] == tl]
            rows.append({"rule": rname, "rule_definition": r["desc"], "tier": tl,
                         "n_parameters": len(mem), "members": ";".join(mem)})
    tiers = pd.DataFrame(rows)
    df["tier_recommended"] = rules["R2_min_across_seeds"]["tier"]
    df["one_sigma_pct_of_admissible_range_seed42"] = 50.0 * crb42
    df["one_sigma_pct_of_admissible_range_5seed_mean"] = 50.0 * crb_mean5
    note = []
    for i in range(47):
        n = []
        if abs(s42[i] - mean[i]) > 1.5 * sdv[i]:
            n.append(f"locked seed 42 is an outlier vs the five-seed distribution "
                     f"({s42[i]:.2f} vs {mean[i]:.2f}+-{sdv[i]:.2f})")
        if cv[i] >= 0.6:
            n.append(f"per-parameter CV {100*cv[i]:.1f}% - not seed-stable at the parameter level")
        if mx[i] < 1.0:
            n.append("CRB exceeds the parameter's own bound half-width in every seed "
                     "-> determined by the bounds/Tikhonov prior, not by the data")
        note.append("; ".join(n))
    df["notes"] = note

    # ---- CRB verification against the frozen phase-1 group table ----------------
    ver = []
    crbf = pd.read_csv(RUNS / "v1fix_phase1_s42/tables/crb_by_group_by_leadset.csv")
    l12 = crbf[crbf["lead_set"] == "L12"].iloc[0]
    for g, r in PARAM_GROUPS.items():
        idx = list(r)
        ours = float(np.mean(1.0 / s42[idx]))
        theirs = float(l12[f"CRB_{g}"])
        ver.append({"lead_set": "L12", "group": g, "n_params": len(idx),
                    "crb_from_1_over_identifiability": ours,
                    "crb_in_frozen_phase1_table": theirs,
                    "abs_diff": abs(ours - theirs),
                    "rel_diff": abs(ours - theirs) / abs(theirs)})
    # also check the group-mean identifiability table
    gfroz = pd.read_csv(RUNS / "v1fix_r3_s42/tables/identifiability_by_param_group.csv")
    gfroz = gfroz.set_index("group")["mean_score"]
    for g, r in PARAM_GROUPS.items():
        ours = float(np.mean(s42[list(r)]))
        theirs = float(gfroz[g])
        ver.append({"lead_set": "L12(group-mean-score check)", "group": g, "n_params": len(list(r)),
                    "crb_from_1_over_identifiability": ours,
                    "crb_in_frozen_phase1_table": theirs,
                    "abs_diff": abs(ours - theirs),
                    "rel_diff": abs(ours - theirs) / abs(theirs)})
    verdf = pd.DataFrame(ver)

    # ---- node-relabelling diagnostic -------------------------------------------
    relab = []
    for g, r in PARAM_GROUPS.items():
        idx = list(r)
        if len(idx) < 2:
            continue
        Ss = np.sort(S[:, idx], axis=1)
        cvs = Ss.std(0, ddof=1) / Ss.mean(0)
        relab.append({"group": g, "n": len(idx),
                      "unsorted_cv_mean": float(cv[idx].mean()),
                      "unsorted_cv_max": float(cv[idx].max()),
                      "sorted_cv_mean": float(cvs.mean()),
                      "sorted_cv_max": float(cvs.max()),
                      "cv_reduction_by_sorting": float(cv[idx].mean() - cvs.mean())})
    relabdf = pd.DataFrame(relab)

    # ---- write ------------------------------------------------------------------
    df.to_csv(OUT / "W1G_parameter_table.csv", index=False, encoding="utf-8-sig")
    tiers.to_csv(OUT / "W1G_tier_rules.csv", index=False, encoding="utf-8-sig")
    verdf.to_csv(OUT / "W1G_crb_verification.csv", index=False, encoding="utf-8-sig")
    relabdf.to_csv(OUT / "W1G_group_relabelling_diagnostic.csv", index=False, encoding="utf-8-sig")

    head = {
        "n_parameters": 47,
        "tier_sizes_by_rule": {r: {t: int(((rules[r]["tier"]) == t).sum())
                                   for t in ["A_strongly_identifiable", "B_weakly_identifiable",
                                             "C_primarily_regularized"]} for r in rules},
        "recommended_rule": "R2_min_across_seeds (>=15 for Tier A; max-across-seeds <1 for Tier C). "
                            "R5 adds a CV<=0.50 term and returns the identical partition, so the "
                            "simpler R2 is recommended.",
        "recommended_tierA": [names[i] for i in range(47)
                              if df["tier_recommended"][i] == "A_strongly_identifiable"],
        "recommended_tierC": [names[i] for i in range(47)
                              if df["tier_recommended"][i] == "C_primarily_regularized"],
        "max_per_parameter_cv": {"param": names[int(cv.argmax())], "cv": float(cv.max())},
        "second_max_per_parameter_cv": {"param": names[int(np.argsort(cv)[-2])],
                                        "cv": float(np.sort(cv)[-2])},
        "group_level_cv": gcv,
        "group_level_five_seed_mean_score": gm,
        "crb_verification_max_rel_diff": float(verdf["rel_diff"].max()),
        "units_statement": (
            "identifiability score and CRB are computed on RADIUS-SCALED parameter columns "
            "(J_scaled[:,j] = J[:,j] * radius_j, radius_j = (hi_j-lo_j)/2), so CRB_j = "
            "1/score_j is a standard deviation expressed in units of parameter j's own bound "
            "half-width, NOT in seconds or millivolts. Multiply by bound_radius for the "
            "physical-unit value. A Tikhonov term lambda=1e-3 is added before inversion, so "
            "the score floor is 1/sqrt(1/lambda) = 0.0316 and the values are a regularized "
            "uncertainty proxy, not a strict unbiased-estimator bound."),
    }
    (OUT / "W1G_headline.json").write_text(json.dumps(head, indent=2, ensure_ascii=False),
                                           encoding="utf-8")

    pd.set_option("display.width", 250)
    print("== CRB verification (L12) ==")
    print(verdf.to_string(index=False))
    print("\n== tier rules ==")
    print(tiers[["rule", "tier", "n_parameters"]].to_string(index=False))
    print("\n== recommended Tier A ==", head["recommended_tierA"])
    print("== recommended Tier C ==", head["recommended_tierC"])
    print("\n== relabelling diagnostic ==")
    print(relabdf.to_string(index=False))


if __name__ == "__main__":
    main()
