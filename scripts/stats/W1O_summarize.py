"""W1O summary: teacher vs teacher-free contrast table (reads W1O_teacherfree_leadsets.csv)."""
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

# frozen submitted-lineage reference (runs/v1fix_leadclf/leadclf_m10.csv, C5 row)
TABLE6 = {
    "12 leads": (0.9014108235762697, 0.887263639318269, 0.9154638653854968),
    "II+V1+V5": (0.8871095212724889, 0.8714025054189635, 0.9020683833120994),
    "I+V1+V4": (0.8780879032570988, 0.8617554597438102, 0.8932275103922185),
    "II": (0.8442931066507506, 0.8258282333212292, 0.8621974157150641),
}
# FIM eff rank of the LOCKED d1s_r3_s42 12-lead encoder on the fixed 400-record subset,
# recomputed here (control: reproduces runs/v1fix_r3_s42/tables/fim_effective_rank_by_leadset.csv
# and the Table 6 seed-42 column).
LOCKED = json.loads((OUT / "W1O_locked_model_fim_effrank.json").read_text())
TABLE6_EFFRANK = {k: v["effective_rank"] for k, v in LOCKED.items()}

df = pd.read_csv(OUT / "W1O_teacherfree_leadsets.csv")
c5 = df[df.feature_group == "theta_plus_reconstruction"].copy()
th = df[df.feature_group == "theta_only"].copy()

recs = []
for lead_set in ["12 leads", "II+V1+V5", "I+V1+V4", "II"]:
    sub = c5[c5.lead_set == lead_set]
    if sub.empty:
        continue
    ref = TABLE6[lead_set]
    for _, r in sub.iterrows():
        t = th[(th.lead_set == lead_set) & (th.arm == r["arm"])]
        recs.append({
            "lead_set": lead_set, "arm": r["arm"], "loss_mode": r["loss_mode"],
            "n_leads": int(r["n_leads"]),
            "C5_macro_auroc": r["macro_auroc"], "C5_ci_lo": r["ci_lo"], "C5_ci_hi": r["ci_hi"],
            "theta_only_macro_auroc": float(t["macro_auroc"].iloc[0]) if len(t) else np.nan,
            "balanced_accuracy": r["balanced_accuracy"], "macro_f1": r["macro_f1"],
            "observed_corr": r["observed_corr"], "full12_nrmse": r["full12_nrmse"],
            "fim_effective_rank": r["fim_effective_rank"],
            "table6_frozen_auroc": ref[0], "table6_frozen_ci_lo": ref[1], "table6_frozen_ci_hi": ref[2],
            "delta_vs_table6_frozen": r["macro_auroc"] - ref[0],
            "table6_frozen_fim_effrank_locked": TABLE6_EFFRANK[lead_set],
            "val_sat090": r["val_sat090"], "best_epoch": r["best_epoch"],
            "epochs_run": r["epochs_run"], "diverged": r["diverged"],
        })
summary = pd.DataFrame(recs)

# teacher (stable init) -> teacher-free (stable init) contrast, init-matched
contrast = []
for lead_set in ["II+V1+V5", "I+V1+V4", "II"]:
    s = summary[summary.lead_set == lead_set].set_index("arm")
    row = {"lead_set": lead_set, "table6_frozen_teacher_auroc": TABLE6[lead_set][0]}
    for arm in ["teacher_stock", "teacher_stable", "obs_stock", "obs_stable"]:
        if arm in s.index:
            row[f"{arm}_auroc"] = s.loc[arm, "C5_macro_auroc"]
            row[f"{arm}_ci"] = f"[{s.loc[arm,'C5_ci_lo']:.3f},{s.loc[arm,'C5_ci_hi']:.3f}]"
            row[f"{arm}_obs_corr"] = s.loc[arm, "observed_corr"]
            row[f"{arm}_full12_nrmse"] = s.loc[arm, "full12_nrmse"]
            row[f"{arm}_effrank"] = s.loc[arm, "fim_effective_rank"]
    if "teacher_stable" in s.index and "obs_stable" in s.index:
        row["teacher_optimism_stable"] = (s.loc["teacher_stable", "C5_macro_auroc"]
                                          - s.loc["obs_stable", "C5_macro_auroc"])
    if "teacher_stock" in s.index and "obs_stock" in s.index:
        row["teacher_optimism_stock"] = (s.loc["teacher_stock", "C5_macro_auroc"]
                                         - s.loc["obs_stock", "C5_macro_auroc"])
    if "obs_stable" in s.index:
        row["obs_stable_minus_table6"] = (s.loc["obs_stable", "C5_macro_auroc"]
                                          - TABLE6[lead_set][0])
        row["obs_stable_inside_table6_ci"] = bool(
            TABLE6[lead_set][1] <= s.loc["obs_stable", "C5_macro_auroc"] <= TABLE6[lead_set][2])
    contrast.append(row)
contrast = pd.DataFrame(contrast)

summary.to_csv(OUT / "W1O_teacherfree_summary.csv", index=False)
contrast.to_csv(OUT / "W1O_teacherfree_contrast.csv", index=False)
with open(OUT / "W1O_teacherfree_contrast.json", "w") as f:
    json.dump(json.loads(contrast.to_json(orient="records")), f, indent=2)

pd.set_option("display.width", 250)
print(summary[["lead_set", "arm", "C5_macro_auroc", "C5_ci_lo", "C5_ci_hi", "theta_only_macro_auroc",
               "observed_corr", "full12_nrmse", "fim_effective_rank", "val_sat090",
               "epochs_run", "diverged", "delta_vs_table6_frozen"]].to_string(index=False))
print()
print(contrast.to_string(index=False))
print(f"\nSaved -> {OUT/'W1O_teacherfree_summary.csv'}, {OUT/'W1O_teacherfree_contrast.csv'}")
