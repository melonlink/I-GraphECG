"""Round 3 Task A+B: parameter- and reconstruction-derived descriptors.

theta is recomputed by the Round 2 encoder in original record order, strictly aligned with signals.
The main C2/C5 feature groups use T-wave features computed from the decoder
reconstruction, matching the manuscript definition ``phi(theta, yhat)``. Direct
observed-ECG features are retained only with an ``obs_`` prefix for provenance
and fair baseline analyses; they are not silently included in surrogate groups.
Run: python scripts/40_descriptors.py --config configs/v1fix/r3_s42.yaml
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

from igraphecg.data.dataset import load_processed
from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.evaluation.repolarization_features import (repolarization_param_features, twave_features)
from igraphecg.evaluation.round2_io import infer_theta, load_encoder_decoder
from igraphecg.evaluation.st_features import internal_st_features, projected_st_features
from igraphecg.evaluation.stats_tests import by_class_table
from igraphecg.models.phys_features import compute_phys_features
from igraphecg.utils.config import parse_args_with_config
from igraphecg.utils.logging import get_logger
from igraphecg.utils.paths import PROJECT_ROOT, ensure_dirs
from igraphecg.utils.seed import set_seed

log = get_logger("round3_features")

# expected direction for the new repolarization descriptors (theta + T-wave): STTC vs NORM
REP_EXPECT = {"Tend_disp": ("STTC", "up"), "tau_rep_disp": ("STTC", "up"),
              "Sync_rep": ("STTC", "down"), "Tpeak_maxabs": ("STTC", "down"),
              "T_sign_discordance_rate": ("STTC", "up")}
ST_EXPECT = {}


def build_all_features(theta, reconstructed_signals, observed_signals, H_ind, fs, c_rep):
    import torch
    feats = {}
    old = compute_phys_features(torch.from_numpy(theta.astype(np.float32)))
    feats["D_act"] = old["D_act"].numpy()
    feats["D_rep_old_APDdisp"] = old["D_rep"].numpy()
    feats["S_ST_internal_L2"] = old["S_ST"].numpy()
    feats["mean_vent_APD"] = old["mean_vent_APD"].numpy()
    feats["vent_delta_spread"] = old["vent_delta_spread"].numpy()
    # graph-Eikonal mechanistic CD descriptors (proximal conduction / leaf-branch imbalance)
    feats["prox_delay"] = old["prox_delay"].numpy()
    feats["leaf_delay_spread"] = old["leaf_delay_spread"].numpy()
    feats.update(repolarization_param_features(theta, c_rep=c_rep))
    reconstructed = twave_features(reconstructed_signals, fs=fs)
    feats.update(reconstructed)
    observed = twave_features(observed_signals, fs=fs)
    feats.update({f"obs_{k}": v for k, v in observed.items()})
    feats.update(internal_st_features(theta))
    feats.update(projected_st_features(theta, H_ind))
    return feats


def main():
    _, cfg = parse_args_with_config("Round3 features (rep + ST)")
    ensure_dirs()
    set_seed(cfg["seed"])
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_processed(PROJECT_ROOT / cfg["npz"])
    signals = data["signal_12lead"].astype(np.float32)
    fs = int(data["sampling_rate"])
    label = data["label"].astype(int); fold = data["fold"].astype(int)
    record_id = data["record_id"].astype(np.int64)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals)

    enc, dec = load_encoder_decoder(repro.outputs() / cfg["encoder_ckpt"], n_t=signals.shape[2],
                                    rank=cfg["leadfield_rank"], device=device)
    theta, _z = infer_theta(enc, dec, scaled, device=device)
    H_ind = (dec.A @ dec.B).detach().cpu().numpy()
    log.info(f"theta recomputed: {theta.shape}, aligned with signals (N={len(label)})")

    with torch.no_grad():
        yhat_scaled = []
        for start in range(0, len(theta), 1024):
            batch = torch.from_numpy(theta[start:start + 1024].astype(np.float32)).to(device)
            yhat_scaled.append(dec.forward(batch)[0].cpu().numpy())
    reconstructed = scaler.inverse_transform(np.concatenate(yhat_scaled))
    feats = build_all_features(theta, reconstructed, signals, H_ind, fs, cfg["c_rep"])

    out = repro.outputs() / cfg["out_dir"]
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # save the full feature CSV (theta included) for reuse by classification / stability
    df = pd.DataFrame({k: v for k, v in feats.items()})
    for j in range(theta.shape[1]):
        df[f"theta_{j}"] = theta[:, j]
    df["record_id"] = record_id
    df["label"] = label; df["label_name"] = np.array(TARGET_CLASSES)[label]; df["fold"] = fold
    df.to_csv(out / "tables" / "round3_features.csv", index=False)
    log.info(f"saved round3_features.csv  ({df.shape[1]} cols)")

    provenance_rows = []
    for name in feats:
        if name.startswith("obs_"):
            source = "observed_ecg"
        elif name.startswith(("Tpeak", "Tptime", "Tarea", "Tabsarea", "Twidth", "Tasym", "Tlowamp",
                              "STmean", "STslope", "T_sign_")):
            source = "reconstructed_ecg"
        elif name.startswith("ST_projected"):
            source = "theta_and_leadfield"
        elif name in {"m_proxy", "V_rec"}:
            source = "auxiliary"
        else:
            source = "named_theta"
        provenance_rows.append({"feature": name, "source": source})
    pd.DataFrame(provenance_rows).to_csv(out / "tables" / "feature_provenance.csv", index=False)

    # by-class statistics (test)
    te = fold == 10
    rep_keys = ["D_rep_old_APDdisp", "APD_mean", "APD_disp", "APD_cv", "tau_rep_mean",
                "tau_rep_disp", "tau_rep_cv", "Tend_mean", "Tend_disp", "Tend_cv", "Sync_rep",
                "Tpeak_maxabs", "Tarea_mean", "Tabsarea_mean", "Tasym_mean", "Tlowamp_mean",
                "T_sign_discordance_rate"]
    rep_feats = {k: feats[k][te] for k in rep_keys if k in feats}
    rep_tab = by_class_table(rep_feats, label[te], expect=REP_EXPECT)
    rep_tab.to_csv(out / "tables" / "repolarization_features_by_class.csv", index=False)

    st_keys = ["S_ST_internal_L2", "ST_L2", "ST_L1", "ST_max_abs", "ST_signed_sum",
               "ST_spatial_disp", "ST_entropy", "ST_anterior_strength", "ST_lateral_strength",
               "ST_inferior_strength", "ST_right_septal_strength",
               "ST_projected_L2", "ST_projected_max_abs", "ST_projected_anterior_leads",
               "ST_projected_lateral_leads", "ST_projected_inferior_leads",
               "ST_projected_sign_discordance"]
    st_feats = {k: feats[k][te] for k in st_keys if k in feats}
    st_tab = by_class_table(st_feats, label[te], expect=ST_EXPECT)
    st_tab.to_csv(out / "tables" / "st_features_by_class.csv", index=False)

    n_rep_sig = int(((rep_tab["fdr_q"] < 0.05) &
                     (rep_tab["feature"] != "D_rep_old_APDdisp")).sum())
    log.info(f"new repolarisation descriptors significant between NORM and the other classes at FDR<0.05 (excluding the old APD_disp): {n_rep_sig}")
    log.info("\n" + rep_tab[["feature", "fdr_q"] + [c for c in ("cliffs_delta", "direction_ok") if c in rep_tab]].to_string(index=False))
    log.info("\nST:\n" + st_tab[["feature", "fdr_q"] + [c for c in ("cliffs_delta", "direction_ok") if c in st_tab]].to_string(index=False))

    # coupling correlations
    coup = ["D_rep_old_APDdisp", "tau_rep_disp", "Tend_disp", "S_ST_internal_L2",
            "ST_projected_L2", "Sync_rep", "T_sign_discordance_rate", "Tpeak_maxabs"]
    M = np.stack([feats[k][te] for k in coup], axis=1)
    C = np.corrcoef(M, rowvar=False)
    pairs = [{"a": coup[i], "b": coup[j], "corr": float(C[i, j])}
             for i in range(len(coup)) for j in range(i + 1, len(coup))]
    pd.DataFrame(sorted(pairs, key=lambda r: -abs(r["corr"]))).to_csv(
        out / "tables" / "parameter_correlation_pairs.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(coup)), coup, rotation=90, fontsize=7)
    ax.set_yticks(range(len(coup)), coup, fontsize=7)
    fig.colorbar(im, fraction=0.046); ax.set_title("feature coupling (test)")
    fig.tight_layout(); fig.savefig(out / "figures" / "parameter_correlation_heatmap.png", dpi=130); plt.close(fig)

    # boxplots
    def box(keys, fname, title):
        keys = [k for k in keys if k in feats]
        fig, axes = plt.subplots(1, len(keys), figsize=(3 * len(keys), 4))
        if len(keys) == 1:
            axes = [axes]
        for ax, k in zip(axes, keys):
            ax.boxplot([feats[k][te][label[te] == c] for c in range(len(TARGET_CLASSES))],
                       tick_labels=TARGET_CLASSES, showfliers=False)
            ax.set_title(k, fontsize=9); ax.grid(alpha=0.3)
        fig.suptitle(title); fig.tight_layout()
        fig.savefig(out / "figures" / fname, dpi=130); plt.close(fig)

    box(["D_rep_old_APDdisp", "tau_rep_disp", "Tend_disp", "Sync_rep"],
        "repolarization_boxplots.png", "repolarization params by class (test)")
    box(["Tpeak_maxabs", "Tarea_mean", "Tasym_mean", "T_sign_discordance_rate"],
        "t_wave_feature_boxplots.png", "T-wave morphology by class (test)")
    box(["ST_projected_L2", "ST_projected_max_abs", "ST_projected_anterior_leads", "ST_L2"],
        "st_projected_feature_boxplots.png", "ST (projected/internal) by class (test)")
    log.info(f"Task A/B done -> {out}")


if __name__ == "__main__":
    main()
