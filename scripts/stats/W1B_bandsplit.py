"""W1B follow-up: split the C5 substitution oracle by descriptor-fidelity band.

W1C classified the 37 reconstruction-domain descriptors by how well the reconstruction
recovers them (Pearson r against the observation-domain twin, five-seed mean):
  band A "faithful"        r >= 0.80
  band C "not recovered"   r <  0.45
Of the 6 recon descriptors that appear in C5, four are band A
(Tabsarea_mean .951, Tarea_mean .915, Tlowamp_mean .891, Tpeak_maxabs .824) and two are
band C (T_sign_discordance_rate .370, Tasym_mean .271).

If the oracle gain from C5_obs is caused by reconstruction ERROR, it should come from the
two band-C descriptors and not from the four band-A ones. This script tests that directly.

Appends rows (test = T1b_*) to W1B_substitution_oracle.csv.
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
# siblings now live beside this file; no sys.path surgery needed

from igraphecg.data.label_utils import TARGET_CLASSES  # noqa: E402
from igraphecg.evaluation.metrics import (compute_metrics, paired_bootstrap_difference,  # noqa: E402
                                          per_class_metrics)
from W1B_error_propagation import (C5, OUT, SEEDS, fit_predict, load_seed,  # noqa: E402
                                   patient_groups)

N = len(TARGET_CLASSES)
BAND_A = ["Tpeak_maxabs", "Tarea_mean", "Tabsarea_mean", "Tlowamp_mean"]   # r .824-.951
BAND_C = ["Tasym_mean", "T_sign_discordance_rate"]                          # r .271-.370


def main():
    import warnings
    warnings.filterwarnings("ignore")
    variants = {
        "C5": (C5, "control"),
        "C5_obsA": (["obs_" + c if c in BAND_A else c for c in C5],
                    "only the 4 FAITHFULLY reconstructed descriptors (r .82-.95) swapped to obs"),
        "C5_obsC": (["obs_" + c if c in BAND_C else c for c in C5],
                    "only the 2 NOT-RECOVERED descriptors (r .27-.37) swapped to obs"),
    }
    rows = []
    for seed in SEEDS:
        df = load_seed(seed)
        label = df["label"].values.astype(int)
        fold = df["fold"].values.astype(int)
        pat = patient_groups(df)
        tr, te = fold <= 8, fold == 10
        probs = {}
        for name, (cols, note) in variants.items():
            prob = fit_predict(df, cols, tr, te, label, seed)
            probs[name] = prob
            m = compute_metrics(label[te], prob, N)
            pc = {p["class"]: p["sensitivity"]
                  for p in per_class_metrics(label[te], prob, N, TARGET_CLASSES)}
            rows.append({"test": "T1b_variant", "seed": seed, "variant": name,
                         "n_features": len(cols), "note": note,
                         "macro_auroc": m["macro_auroc"], "macro_f1": m["macro_f1"],
                         "balanced_accuracy": m["balanced_accuracy"],
                         **{f"recall_{c}": pc[c] for c in TARGET_CLASSES},
                         "MI_to_NORM": int(m["confusion_matrix"][1, 0])})
        for a in ("C5_obsA", "C5_obsC"):
            pt, lo, hi = paired_bootstrap_difference(label[te], probs[a], probs["C5"], N,
                                                    metric="macro_auroc", groups=pat[te],
                                                    n_boot=1000, seed=42)
            rows.append({"test": "T1b_paired_delta", "seed": seed, "variant": f"{a}_minus_C5",
                         "metric": "macro_auroc", "difference": pt, "ci_lo": lo, "ci_hi": hi,
                         "crosses_zero": bool(lo <= 0.0 <= hi),
                         "bootstrap_unit": "patient", "n_boot": 1000})
        print(f"  [T1b] seed {seed} done", flush=True)

    new = pd.DataFrame(rows)
    f = OUT / "W1B_substitution_oracle.csv"
    old = pd.read_csv(f)
    old = old[~old["test"].astype(str).str.startswith("T1b")]
    pd.concat([old, new], ignore_index=True).to_csv(f, index=False)

    v = new[new.test == "T1b_variant"]
    g = v.groupby("variant")[["macro_auroc", "balanced_accuracy"]].agg(["mean", "std"])
    print(g.round(5).to_string())
    d = new[new.test == "T1b_paired_delta"]
    print(d.groupby("variant")[["difference"]].agg(["mean", "std", "min", "max"]).round(5).to_string())
    json.dump({"band_A_faithful": BAND_A, "band_C_not_recovered": BAND_C,
               "delta_macro_auroc_vs_C5": {
                   k: {"mean": float(d[d.variant == k]["difference"].mean()),
                       "sd_ddof1": float(d[d.variant == k]["difference"].std()),
                       "n_seeds_ci_excludes_0": int((~d[d.variant == k]["crosses_zero"].astype(bool)).sum())}
                   for k in ("C5_obsA_minus_C5", "C5_obsC_minus_C5")}},
              open(OUT / "W1B_bandsplit_headline.json", "w"), indent=1)
    print("appended T1b rows to", f)


if __name__ == "__main__":
    main()
