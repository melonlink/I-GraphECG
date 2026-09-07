"""Round 3 Task F: stability metrics.
Level 1 surrogate stability proxy (mandatory) + Level 2 AP-ODE finite-time monodromy (best-effort).

Run: python scripts/41_stability.py --config configs/v1fix/r3_s42.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igraphecg import repro   # output keys in configs are output-root relative

from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.evaluation.stability import surrogate_stability_proxy
from igraphecg.evaluation.stats_tests import by_class_table
from igraphecg.utils.config import parse_args_with_config
from igraphecg.utils.logging import get_logger
from igraphecg.utils.paths import PROJECT_ROOT, ensure_dirs
from igraphecg.utils.seed import set_seed

log = get_logger("stability")


def main():
    _, cfg = parse_args_with_config("Round3 stability")
    ensure_dirs(); set_seed(cfg["seed"])
    out = repro.outputs() / cfg["out_dir"]
    # do not rely on 07 having run first: make our own output subdirectories
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(out / "tables" / "round3_features.csv")
    label = df["label"].values.astype(int); fold = df["fold"].values.astype(int)

    # ---- Level 1: surrogate stability proxy ----
    feats_in = {k: df[k].values for k in ["D_act", "Tend_disp", "tau_rep_disp", "ST_projected_L2"]}
    proxy = surrogate_stability_proxy(feats_in, train_mask=(fold <= 8))
    df["V_rec"] = proxy["V_rec"]; df["m_proxy"] = proxy["m_proxy"]
    # carry record_id so 42_classification.py can verify alignment by key, not row order
    stab_cols = ["m_proxy", "V_rec", "label", "fold"]
    if "record_id" in df.columns:
        stab_cols.insert(0, "record_id")
    df[stab_cols].to_csv(out / "tables" / "stability_proxy_values.csv", index=False)

    te = fold == 10
    tab = by_class_table({"V_rec": proxy["V_rec"][te], "m_proxy": proxy["m_proxy"][te]}, label[te])
    tab.to_csv(out / "tables" / "stability_features_by_class.csv", index=False)
    log.info("Level1 proxy by class (test):\n" + tab[["feature", "kw_p", "fdr_q"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([proxy["m_proxy"][te][label[te] == c] for c in range(4)],
               tick_labels=TARGET_CLASSES, showfliers=False)
    ax.set_title("surrogate stability margin m_proxy by class (test)"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "figures" / "stability_by_class_boxplot.png", dpi=300); plt.close(fig)

    if not cfg.get("run_ap_ode", True):
        pd.DataFrame([{"stability_level": "surrogate_proxy_only",
                       "ap_ode_implemented": True, "ap_ode_success_rate": np.nan,
                       "note": "AP-ODE analysis disabled in this revision configuration"}]
                     ).to_csv(out / "tables" / "stability_level.csv", index=False)
        log.info(f"Task F proxy complete; AP-ODE disabled -> {out}")
        return

    # ---- Level 2: AP-ODE finite-time monodromy (best-effort, small subset) ----
    import torch

    from igraphecg.evaluation.round2_io import infer_theta, load_encoder_decoder
    from igraphecg.data.dataset import load_processed
    from igraphecg.data.preprocess import RobustLeadScaler
    from igraphecg.models.ap_ode import rule_adapter

    data = load_processed(PROJECT_ROOT / cfg["npz"])
    signals = data["signal_12lead"].astype(np.float32)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals)
    enc, dec = load_encoder_decoder(repro.outputs() / cfg["encoder_ckpt"], signals.shape[2], cfg["leadfield_rank"], "cpu")

    rng = np.random.default_rng(cfg["seed"]); per = cfg["ode_per_class"]
    sub = np.concatenate([rng.choice(np.where((label == c) & (fold == 10))[0], per, replace=False)
                          for c in range(4)])
    theta_sub, _ = infer_theta(enc, dec, scaled[sub], device="cpu")

    rows = []; n_ok = 0
    for i, th in enumerate(theta_sub):
        ode = rule_adapter(th, device="cpu")
        try:
            rho = ode.finite_time_monodromy(T=0.6)
        except Exception:
            rho = None
        if rho is not None and np.isfinite(rho):
            n_ok += 1
            rows.append({"class": TARGET_CLASSES[label[sub][i]], "rho": rho,
                         "m_stab": 1 - rho, "log_stab": -np.log(rho + 1e-9) / 0.6})
    success_rate = n_ok / len(sub)
    log.info(f"AP-ODE finite-time monodromy success: {n_ok}/{len(sub)} = {success_rate:.2%}")
    ode_level = "finite_time_monodromy" if success_rate > 0.0 else "none"
    if rows:
        odf = pd.DataFrame(rows)
        odf.to_csv(out / "tables" / "ap_ode_monodromy.csv", index=False)
        lab_ode = np.array([TARGET_CLASSES.index(r["class"]) for r in rows])
        mtab = by_class_table({"m_stab": odf["m_stab"].values}, lab_ode)
        log.info("AP-ODE m_stab by class:\n" + mtab[["feature", "kw_p", "fdr_q"]].to_string(index=False))

    pd.DataFrame([{"stability_level": "surrogate_proxy + " + ode_level,
                   "ap_ode_implemented": True, "ap_ode_success_rate": success_rate,
                   "note": "finite-time recovery multiplier under paced dynamics, NOT strict Floquet"}]
                 ).to_csv(out / "tables" / "stability_level.csv", index=False)
    log.info(f"Task F done -> {out}")


if __name__ == "__main__":
    main()
