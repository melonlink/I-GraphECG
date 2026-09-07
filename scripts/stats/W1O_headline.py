import json
from pathlib import Path
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro
OUT = repro.outdir("runs/v7rev_stats")
d = pd.read_csv(OUT / "W1O_teacherfree_leadsets.csv")
c5 = d[d.feature_group == "theta_plus_reconstruction"].set_index(["lead_set", "arm"])
th = d[d.feature_group == "theta_only"].set_index(["lead_set", "arm"])
locked = json.loads((OUT / "W1O_locked_model_fim_effrank.json").read_text())
frozen = pd.read_csv(repro.runs() / "v1fix_leadclf" / "leadclf_m10.csv")
frozen = frozen[frozen.feature_group == "theta_plus_reconstruction"].set_index("lead_set")
FMAP = {"12 leads": "12 displayed channels", "II+V1+V5": "II+V1+V5",
        "I+V1+V4": "I+V1+V4", "II": "II"}

out = {
    "task": "W1O",
    "question": ("How much of the Table-6 reduced-lead macro-AUROC is optimism from the "
                 "full-12 teacher loss? Retrain lead-set-specific encoders with the loss "
                 "restricted to the observed leads and re-evaluate C5 descriptors."),
    "protocol": {
        # recorded relative to the output root, so this provenance reads correctly both
        # here and in the archive, where that directory is called outputs/
        "base_checkpoint": "checkpoints/d1s_r3_s42_best.pt",
        "pipeline": "scripts/55_leadset_classification.py caliber (batch 128, lr 1e-3, wd 1e-4, 80 epochs, "
                    "patience 15, frozen decoder + lead field), driver = 55_leadset_classification.py",
        "teacher_free_loss": "total_loss restricted to the observed leads "
                             "(the former lead-set retrain script, since removed; the mode lives in W1O_teacherfree_leadsets.py mode='obs'; "
                             "scripts/44_single_lead_sanity.py L1-no-teacher-obs, Table S2)",
        "init_fix": {"head_init_scale": 0.01, "grad_clip": 1.0,
                     "source": "configs/d1s_r3_s42.yaml / scripts/31_surrogate_encoder.py",
                     "arms_with_fix": ["teacher_stable", "obs_stable"]},
        "features": "C5 = theta(47) + repolarization + ST + auxiliary recovery, "
                    "computed only from theta and its decoder reconstruction",
        "classifier": "XGBoost 400x4, folds 1-8 train, fold 10 test, seed 42",
        "ci": "1000x percentile bootstrap, resampled by patient cluster",
        "fim": "fixed 400-record fold-10 subset (100/class, seed 42), scaled 47-coordinate "
               "Fisher matrix, effective rank = exp(entropy of eigenvalue spectrum)",
    },
    "controls_reproduced": {},
    "locked_model_fim_effective_rank": {k: round(v["effective_rank"], 4) for k, v in locked.items()},
    "arms": {},
    "teacher_optimism": {},
    "divergences": [],
}

for name in ["12 leads", "II+V1+V5", "I+V1+V4", "II"]:
    fr = frozen.loc[FMAP[name]]
    ctrl_arm = "locked_main" if name == "12 leads" else "teacher_stock"
    got = c5.loc[(name, ctrl_arm)]
    out["controls_reproduced"][name] = {
        "control_arm": ctrl_arm,
        "frozen_macro_auroc": float(fr["macro_auroc"]),
        "reproduced_macro_auroc": float(got["macro_auroc"]),
        "abs_difference": abs(float(got["macro_auroc"]) - float(fr["macro_auroc"])),
        "frozen_ci": [float(fr["ci_lo"]), float(fr["ci_hi"])],
        "reproduced_ci": [float(got["ci_lo"]), float(got["ci_hi"])],
        "frozen_val_nrmse": None if pd.isna(fr["val_nrmse"]) else float(fr["val_nrmse"]),
        "reproduced_val_nrmse": None if pd.isna(got["val_nrmse_selected"]) else float(got["val_nrmse_selected"]),
    }

for (name, arm), r in c5.iterrows():
    out["arms"][f"{name}|{arm}"] = {
        "loss_mode": r["loss_mode"], "n_leads": int(r["n_leads"]),
        "C5_macro_auroc": float(r["macro_auroc"]),
        "C5_ci": [float(r["ci_lo"]), float(r["ci_hi"])],
        "theta_only_macro_auroc": float(th.loc[(name, arm), "macro_auroc"]),
        "balanced_accuracy": float(r["balanced_accuracy"]),
        "observed_lead_median_corr": float(r["observed_corr"]),
        "full12_median_nrmse_scaled": float(r["full12_nrmse"]),
        "fim_effective_rank_encoder_specific": float(r["fim_effective_rank"]),
        "fim_effective_rank_locked_model": round(locked[name]["effective_rank"], 4),
        "val_sat090": None if pd.isna(r["val_sat090"]) else float(r["val_sat090"]),
        "epochs_run": int(r["epochs_run"]), "diverged": bool(r["diverged"]),
    }
    if bool(r["diverged"]):
        out["divergences"].append(f"{name}|{arm}")

for name in ["II+V1+V5", "I+V1+V4", "II"]:
    tf = c5.loc[(name, "obs_stable")]
    out["teacher_optimism"][name] = {
        "table6_teacher_auroc": float(frozen.loc[FMAP[name], "macro_auroc"]),
        "teacher_free_auroc_with_init_fix": float(tf["macro_auroc"]),
        "teacher_free_ci": [float(tf["ci_lo"]), float(tf["ci_hi"])],
        "delta_vs_table6": float(frozen.loc[FMAP[name], "macro_auroc"] - tf["macro_auroc"]),
        "delta_init_matched": float(c5.loc[(name, "teacher_stable"), "macro_auroc"]
                                    - c5.loc[(name, "obs_stable"), "macro_auroc"]),
        "delta_stock_matched": float(c5.loc[(name, "teacher_stock"), "macro_auroc"]
                                     - c5.loc[(name, "obs_stock"), "macro_auroc"]),
        "teacher_free_inside_table6_ci": bool(
            frozen.loc[FMAP[name], "ci_lo"] <= tf["macro_auroc"] <= frozen.loc[FMAP[name], "ci_hi"]),
        "observed_corr_teacher_to_free": [float(c5.loc[(name, "teacher_stock"), "observed_corr"]),
                                          float(tf["observed_corr"])],
        "full12_nrmse_teacher_to_free": [float(c5.loc[(name, "teacher_stock"), "full12_nrmse"]),
                                         float(tf["full12_nrmse"])],
    }

(OUT / "W1O_headline.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out["controls_reproduced"], indent=2))
print(json.dumps(out["teacher_optimism"], indent=2))
print("divergences:", out["divergences"])
