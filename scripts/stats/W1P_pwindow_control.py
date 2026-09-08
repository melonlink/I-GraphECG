"""W1P: the atrial-window control behind Table S15.

The locked lineage never supervises the atrial region, and Section 3.1 reports that the P wave is
not reconstructed. This control asks what happens when the objective does ask for it: the
seed-42 recipe is retrained with one extra loss term over the atrial window (-300..-100 ms) at
two weights, and the resulting encoders are evaluated with the same scripts as the locked model.

Inputs, all produced by documented commands (REPRODUCIBILITY.md, section 5):
  scripts/31_surrogate_encoder.py --config configs/v1fix/d1s_r3_s42_pwin.yaml    (p = 0.5)
  scripts/31_surrogate_encoder.py --config configs/v1fix/d1s_r3_s42_pwin2.yaml   (p = 2.0)
  scripts/4{0,1,2,3}_*.py         --config configs/v1fix/r3_s42_pwin.yaml / r3_s42_pwin2.yaml
The control checkpoints and their analysis directories are by-products (not registered, like the
teacher-free encoders of W1O); only the summary written here is registered.

Per model: window NRMSE (P, QRS, ST, T) and overall median NRMSE / correlation on fold 10; the
across-record spread of the atrioventricular delay d0 and root onset; the share of records whose
atrial source gain q0 sits at its lower bound; C5 macro-AUROC, CD recall and first-degree AV-block
recall from the classification tables; and the identifiability scores of delta_root and edge_prox0.

Writes runs/v7rev_stats/W1P_pwindow_control.csv.
"""
from __future__ import annotations

import ast
import sys as _sys
from pathlib import Path as _Path

import numpy as np
import pandas as pd
import yaml

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro
from igraphecg.data.dataset import load_processed
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.evaluation.recon_metrics import per_sample_lead_corr, per_sample_nrmse
from igraphecg.evaluation.round2_io import infer_theta, load_encoder_decoder
from igraphecg.utils.paths import find_ptbxl_root

ROOT = _Path(__file__).resolve().parents[2]
OUT = repro.outdir("runs/v7rev_stats")
NPZ = repro.processed() / "ptbxl_medianbeat_clean_100hz.npz"
WINDOWS = {"P": (-0.30, -0.10), "QRS": (-0.06, 0.10), "ST": (0.06, 0.14), "T": (0.12, 0.45)}
IDX = {"delta_root": 0, "edge_prox0": 1, "q0": 32}
Q0_LO = 0.05


def models():
    """(label, checkpoint path, analysis tables dir) for the locked model and the two controls."""
    rows = [("locked", repro.lineage.checkpoint(42), repro.runs() / "v1fix_r3_s42" / "tables")]
    for train_cfg, ana_cfg, label in (("d1s_r3_s42_pwin.yaml", "r3_s42_pwin.yaml", "atrial window, weight 0.5"),
                                      ("d1s_r3_s42_pwin2.yaml", "r3_s42_pwin2.yaml", "atrial window, weight 2.0")):
        t = yaml.safe_load(open(ROOT / "configs/v1fix" / train_cfg, encoding="utf-8"))
        a = yaml.safe_load(open(ROOT / "configs/v1fix" / ana_cfg, encoding="utf-8"))
        ck = repro.outputs() / t["out_ckpt"]
        if not ck.exists():
            raise SystemExit(f"{ck} missing: run scripts/31_surrogate_encoder.py --config configs/v1fix/{train_cfg}")
        rows.append((label, ck, repro.outputs() / a["out_dir"] / "tables"))
    return rows


def main() -> None:
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_processed(NPZ)
    sig = data["signal_12lead"].astype(np.float32)
    fold = data["fold"].astype(int)
    rid = data["record_id"]
    sc = {"median": data["scaler_median"], "iqr": data["scaler_iqr"]}
    scaled = RobustLeadScaler.from_dict(sc).transform(sig)
    n_t = sig.shape[2]
    te = np.where(fold == 10)[0]
    y = scaled[te]

    raw = find_ptbxl_root()
    if raw is None:
        raise SystemExit("PTB-XL not found: set ECG_DATA_DIR (needed for the AV-block subclass)")
    codes = pd.read_csv(raw / "ptbxl_database.csv", index_col="ecg_id").loc[rid[te], "scp_codes"].apply(ast.literal_eval)
    is_1avb = pd.Series([("1AVB" in d) for d in codes], index=rid[te])

    rows = []
    for label, ck, tables in models():
        enc, dec = load_encoder_decoder(ck, n_t, 3, dev, scaler=sc)
        theta, z = infer_theta(enc, dec, y, device=dev)
        with torch.no_grad():
            yh = np.concatenate([dec.forward_from_z(torch.from_numpy(z[s:s + 512]).to(dev))[0].cpu().numpy()
                                 for s in range(0, len(te), 512)])
        t = dec.t.cpu().numpy()
        r = {"model": label, "n_test": len(te),
             "median_nrmse": float(np.median(per_sample_nrmse(y, yh))),
             "median_corr": float(np.median(per_sample_lead_corr(y, yh)))}
        for w, (a, b) in WINDOWS.items():
            m = (t >= a) & (t <= b)
            nr = np.sqrt(((y[:, :, m] - yh[:, :, m]) ** 2).sum(axis=(1, 2))) / (np.sqrt((y[:, :, m] ** 2).sum(axis=(1, 2))) + 1e-8)
            r[f"nrmse_{w}_window"] = float(np.median(nr))
        r["sd_edge_prox0_s"] = float(theta[:, IDX["edge_prox0"]].std())
        r["sd_delta_root_s"] = float(theta[:, IDX["delta_root"]].std())
        r["frac_q0_at_lower_bound"] = float(np.mean(theta[:, IDX["q0"]] < Q0_LO + 0.05 * (3.0 - Q0_LO)))

        summ = pd.read_csv(tables / "round3_classification_summary.csv")
        c5 = summ[(summ.feature_group == "C5_theta_rep_st_stab") & (summ.classifier == "xgboost")].iloc[0]
        r["C5_macro_auroc"] = float(c5.macro_auroc)
        r["C5_auroc_ci_lo"], r["C5_auroc_ci_hi"] = float(c5.auroc_ci_lo), float(c5.auroc_ci_hi)
        pred = pd.read_csv(tables / "round3_xgboost_test_predictions.csv")
        pred = pred[pred.feature_group == "C5_theta_rep_st_stab"].set_index("record_id")
        hit = pred[["prob_NORM", "prob_MI", "prob_STTC", "prob_CD"]].values.argmax(1) == pred.label.values
        cd = pred.label.values == 3
        r["CD_recall"] = float(hit[cd].mean())
        avb = is_1avb.reindex(pred.index).fillna(False).values & cd
        r["n_1AVB"], r["recall_1AVB"] = int(avb.sum()), float(hit[avb].mean())
        fim = pd.read_csv(tables / "fim_parameter_identifiability.csv").set_index("param")
        r["ident_score_delta_root"] = float(fim.loc["delta_root", "identifiability_score"])
        r["ident_score_edge_prox0"] = float(fim.loc["edge_prox0", "identifiability_score"])
        rows.append(r)
        print(f"[W1P] {label}: P-window NRMSE {r['nrmse_P_window']:.3f}, overall {r['median_nrmse']:.3f}, "
              f"C5 AUROC {r['C5_macro_auroc']:.4f}, sd(d0) {r['sd_edge_prox0_s']:.4f} s")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "W1P_pwindow_control.csv", index=False)
    print(df.to_string(index=False))
    print(f"wrote -> {OUT / 'W1P_pwindow_control.csv'}")


if __name__ == "__main__":
    main()
