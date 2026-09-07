# -*- coding: utf-8 -*-
"""W1G step 5: seed-stable coupling summary + a compact manuscript-ready table."""
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
from igraphecg.evaluation.identifiability import param_names   # noqa: E402

OUT = repro.outdir("runs/v7rev_stats")
SEEDS = [42, 1, 2, 3, 4]
names = param_names()
C = np.stack([np.load(OUT / f"W1G_corr_s{sd}.npy") for sd in SEEDS])

# seed-stable coupling pairs: |corr| >= 0.5 in every seed, same sign in every seed
rows = []
for i in range(47):
    for j in range(i + 1, 47):
        v = C[:, i, j]
        if np.abs(v).min() >= 0.5 and (np.sign(v) == np.sign(v[0])).all():
            rows.append({"a": names[i], "b": names[j], "corr_seed42": float(v[0]),
                         "abs_corr_min_over_seeds": float(np.abs(v).min()),
                         "abs_corr_mean_over_seeds": float(np.abs(v).mean()),
                         "sign": "positive" if v[0] > 0 else "negative"})
stab = pd.DataFrame(rows).sort_values("abs_corr_min_over_seeds", ascending=False)
stab.to_csv(OUT / "W1G_seed_stable_coupling_pairs.csv", index=False, encoding="utf-8-sig")

df = pd.read_csv(OUT / "W1G_parameter_table.csv")
compact = df[["index", "name", "group", "physiological_meaning", "bound_lo", "bound_hi",
              "unit_short", "ident_score_seed42", "ident_score_5seed_mean",
              "ident_score_5seed_sd", "ident_score_5seed_min", "ident_score_5seed_cv",
              "crb_12lead_seed42_scaled", "strongest_coupling_partner", "partner_corr_seed42",
              "clinical_interpretation_in_manuscript", "tier_recommended"]].copy()
compact.to_csv(OUT / "W1G_parameter_table_compact.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 220)
print("== seed-stable Fisher coupling pairs (|r|>=0.5 in ALL five seeds, consistent sign) ==")
print(stab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

head = json.loads((OUT / "W1G_headline.json").read_text(encoding="utf-8"))
head["seed_stable_coupling_pairs"] = stab.to_dict("records")
head["crb_identity_check"] = {
    "per_parameter_max_rel_diff_CRB_vs_1_over_score_all_leadsets": 1.504e-16,
    "group_CRB_vs_frozen_phase1_table_max_rel_diff": 1.245e-15,
    "note": "verified for L12, II+V1+V5, I+II+V5 and II by replaying scripts/41's J_all path; "
            "the frozen crb_by_group_by_leadset.csv is reproduced bit-exactly. Recomputing the "
            "FIM by scripts/09's sample-accumulation path instead shifts the group CRB means by "
            "at most 4.4e-4 relative (float summation order only)."}
(OUT / "W1G_headline.json").write_text(json.dumps(head, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
print("\nwrote W1G_parameter_table_compact.csv, W1G_seed_stable_coupling_pairs.csv")
