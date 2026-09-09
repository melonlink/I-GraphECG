"""W1T: structural ablation of the surrogate (Reviewer 3, Comment 3).

The published ablation (Table 3) varies the descriptor set fed to the classifier; this one varies
the model itself. Six single-seed variants of the locked seed-42 recipe are retrained, each with one
structural element changed and everything else byte-identical to configs/d1s_r3_s42.yaml:

  rank1, rank5, rank8   lead-field rank 1 / 5 / 8 instead of 3 (8 = unconstrained on 8 leads); the
                        random initial lead field is scaled so that its entry variance equals the
                        rank-3 recipe's (s = 0.3 (3/r)^(1/4)), because a larger initial field drives
                        the encoder into the tanh-saturation failure mode the recipe was tuned against;
                        rank 8 also needs the early-stopping patience extended by the 20 warm-up
                        epochs (its frozen random field yields no validation gain before then) and
                        a longer schedule (400 instead of 150 epochs: it leaves the saturation
                        regime only after about 90 epochs and is still improving at 150)
  nophys                physics loss switched off (loss_weights.phys = 0)
  physderiv             textbook Einthoven-Goldberger coefficients applied in scaled coordinates
                        (scaled_derivation = false), i.e. the derived leads are not exact
  direct12              a [12, rank] lead field projecting onto all twelve leads directly, no exact
                        limb-lead relations (direct_12_leads = true)
  freedelays            eight free node activation times instead of root onset + seven edge
                        delays on the conduction graph (graph_delays = false)

Each variant is then evaluated by the same scripts as the locked model (40 descriptors,
41 stability proxy, 42 classification, 43 identifiability). The checkpoints and analysis
directories are by-products (runs/v7rev_abl_*, not registered, like the W1O and W1P controls);
only the summary written here is registered.

Run (from the repository root, torch environment):
    python scripts/stats/W1T_structural_ablation.py --train      # scripts/31 per variant, ~4 min each
    python scripts/stats/W1T_structural_ablation.py --analyze    # scripts/40-43 per variant
    python scripts/stats/W1T_structural_ablation.py              # summary only
Writes runs/v7rev_stats/W1T_structural_ablation.csv and W1T_headline.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from igraphecg import repro  # noqa: E402

PY = sys.executable
CFG = ROOT / "configs" / "ablation"
VARIANTS = {
    # rank variants keep the initial lead-field norm of the locked recipe: s = 0.3 (3 / r)^(1/4)
    "rank1": {"leadfield_rank": 1, "leadfield_init_scale": round(0.3 * (3 / 1) ** 0.25, 4)},
    "rank5": {"leadfield_rank": 5, "leadfield_init_scale": round(0.3 * (3 / 5) ** 0.25, 4)},
    # rank 8: with the lead field frozen for the 20 warm-up epochs, the random full-rank field gives
    # no validation improvement over epoch 1, so the stock patience (20) would stop the run at the
    # first epoch in which the field is trainable; the patience is extended by the warm-up length.
    # The unconstrained field also leaves the tanh-saturation regime only after about 90 epochs and
    # is still improving at the stock 150-epoch cap (val NRMSE 0.47), so the cap is raised to 400.
    "rank8": {"leadfield_rank": 8, "leadfield_init_scale": round(0.3 * (3 / 8) ** 0.25, 4),
              "train": {"early_stop_patience": 40, "max_epochs": 400}},
    "nophys": {"loss_weights": {"phys": 0.0}},
    "physderiv": {"scaled_derivation": False},
    "direct12": {"direct_12_leads": True},
    "freedelays": {"graph_delays": False},
}
LABEL = {
    "locked": "locked model (rank 3, graph, exact relations, physics loss)",
    "rank1": "lead-field rank 1",
    "rank5": "lead-field rank 5",
    "rank8": "lead-field rank 8 (unconstrained)",
    "nophys": "no physics loss",
    "physderiv": "physical Einthoven coefficients in scaled space",
    "direct12": "no exact limb-lead relations (12-row lead field)",
    "freedelays": "no conduction graph (free activation times)",
}


def write_configs() -> None:
    """Training and analysis configs per variant, derived from the locked seed-42 files."""
    CFG.mkdir(exist_ok=True)
    train = yaml.safe_load((ROOT / "configs/d1s_r3_s42.yaml").read_text(encoding="utf-8"))
    anal = yaml.safe_load((ROOT / "configs/v1fix/r3_s42.yaml").read_text(encoding="utf-8"))
    for v, changes in VARIANTS.items():
        t = json.loads(json.dumps(train))
        for k, val in changes.items():
            if isinstance(val, dict):
                t[k].update(val)
            else:
                t[k] = val
        t["out_dir"] = f"runs/v7rev_abl_{v}"
        t["out_theta"] = f"data/processed/ptbxl_theta_abl_{v}.npz"
        t["out_ckpt"] = f"checkpoints/abl_{v}.pt"
        (CFG / f"train_{v}.yaml").write_text(
            f"# W1T structural ablation '{v}': {LABEL[v]}; every other setting is configs/d1s_r3_s42.yaml.\n"
            + yaml.safe_dump(t, sort_keys=False), encoding="utf-8")
        a = json.loads(json.dumps(anal))
        a["encoder_ckpt"] = f"checkpoints/abl_{v}.pt"
        a["leadfield_rank"] = t["leadfield_rank"]
        a["out_dir"] = f"runs/v7rev_abl_{v}"
        (CFG / f"{v}.yaml").write_text(
            f"# W1T structural ablation '{v}': analysis of checkpoints/abl_{v}.pt with the locked\n"
            f"# model's analysis settings (configs/v1fix/r3_s42.yaml).\n" + yaml.safe_dump(a, sort_keys=False),
            encoding="utf-8")
    print(f"configs -> {CFG}")


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}")


def train(variants) -> None:
    for v in variants:
        run([PY, "scripts/31_surrogate_encoder.py", "--config", str(CFG / f"train_{v}.yaml")])


def analyze(variants) -> None:
    for v in variants:
        for s in ("40_descriptors", "41_stability", "42_classification", "43_identifiability"):
            run([PY, f"scripts/{s}.py", "--config", str(CFG / f"{v}.yaml")])


def collect(run_dir: Path, recon_dir: Path) -> dict:
    rec = pd.read_csv(recon_dir / "tables" / "reconstruction_metrics.csv")
    r = rec[(rec.split == "test") & (rec.space == "scaled")].iloc[0]
    rp = rec[(rec.split == "test") & (rec.space == "physical_mV")].iloc[0]
    cls = pd.read_csv(run_dir / "tables" / "round3_classification_summary.csv")
    cls = cls[cls.classifier == "xgboost"].set_index("feature_group")
    fim = pd.read_csv(run_dir / "tables" / "fim_effective_rank_by_leadset.csv").set_index("lead_set")
    out = {
        "test_median_nrmse_scaled": float(r.median_nrmse), "test_median_corr": float(r.median_corr),
        "test_qrs_nrmse_scaled": float(r.qrs_nrmse), "test_st_nrmse_scaled": float(r.st_nrmse),
        "test_t_nrmse_scaled": float(r.t_nrmse), "test_st_mean_abs_err_mV": float(rp.st_mean_abs_err),
        "boundary_rate": float(r.boundary_rate),
        "C0_macro_auroc": float(cls.loc["C0_theta", "macro_auroc"]),
        "C0_auroc_ci_lo": float(cls.loc["C0_theta", "auroc_ci_lo"]), "C0_auroc_ci_hi": float(cls.loc["C0_theta", "auroc_ci_hi"]),
        "C5_macro_auroc": float(cls.loc["C5_theta_rep_st_stab", "macro_auroc"]),
        "C5_auroc_ci_lo": float(cls.loc["C5_theta_rep_st_stab", "auroc_ci_lo"]), "C5_auroc_ci_hi": float(cls.loc["C5_theta_rep_st_stab", "auroc_ci_hi"]),
        "fim_effrank_12": float(fim.loc["L12", "effective_rank"]),
        "fim_effrank_II": float(fim.loc[[i for i in fim.index if i.endswith("_II")][0], "effective_rank"]) if any(i.endswith("_II") for i in fim.index) else float("nan"),
    }
    return out


def physical_consistency(ckpt: Path, rank: int) -> dict:
    """Two quantities the constrained design fixes by construction, evaluated on the fold-10
    reconstructions of a checkpoint: the relative error of the four derived limb leads against
    the Einthoven-Goldberger relations in physical units (median over records of
    ||d_hat - d(I_hat, II_hat)|| / ||d_hat||, averaged over III, aVR, aVL, aVF), and the fraction of
    records in which some node activates before its parent on the conduction tree."""
    import numpy as np
    import torch
    from igraphecg.data.dataset import load_processed
    from igraphecg.data.preprocess import RobustLeadScaler
    from igraphecg.evaluation.round2_io import infer_theta, load_encoder_decoder
    from igraphecg.models.leadfield import OUTPUT_LEADS
    from igraphecg.models.surrogate_decoder import EDGES, ParamSpace

    data = load_processed(ROOT / "data/processed/ptbxl_medianbeat_clean_100hz.npz")
    signals = data["signal_12lead"].astype(np.float32)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    te = np.where(data["fold"].astype(int) == 10)[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc, dec = load_encoder_decoder(ckpt, signals.shape[2], rank, dev,
                                    scaler={"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    theta, _ = infer_theta(enc, dec, scaler.transform(signals[te]).astype(np.float32), device=dev)
    with torch.no_grad():
        y = dec.forward(torch.from_numpy(theta).to(dev))[0].cpu().numpy()
    y = scaler.inverse_transform(y)                              # [n, 12, T] in mV
    L = {n: y[:, OUTPUT_LEADS.index(n)] for n in OUTPUT_LEADS}
    ref = {"III": L["II"] - L["I"], "aVR": -(L["I"] + L["II"]) / 2.0,
           "aVL": L["I"] - L["II"] / 2.0, "aVF": L["II"] - L["I"] / 2.0}
    rel = []
    for n, r in ref.items():
        num = np.sqrt(((L[n] - r) ** 2).sum(1))
        den = np.sqrt((L[n] ** 2).sum(1)) + 1e-8
        rel.append(np.median(num / den))
    delta = ParamSpace.unpack(torch.from_numpy(theta), dec.graph_delays)["delta"].numpy()  # [n, 8]
    bad = np.zeros(len(te), dtype=bool)
    for a, b in EDGES:
        bad |= delta[:, b] < delta[:, a]
    ParamSpace.GRAPH_DELAYS = True                               # restore the process default
    return {"limb_relation_rel_error": float(np.mean(rel)), "activation_order_violation_rate": float(bad.mean())}


def summarize() -> None:
    rows = [{"variant": "locked", "description": LABEL["locked"],
             **collect(repro.runs() / "v1fix_r3_s42", repro.runs() / "d1s_r3_s42"),
             **physical_consistency(repro.lineage.checkpoint(42), 3)}]
    for v in VARIANTS:
        d = repro.runs() / f"v7rev_abl_{v}"
        if not (d / "tables" / "round3_classification_summary.csv").exists():
            print(f"  {v}: not analysed yet, skipped")
            continue
        rows.append({"variant": v, "description": LABEL[v], **collect(d, d),
                     **physical_consistency(repro.outputs() / "checkpoints" / f"abl_{v}.pt", VARIANTS[v].get("leadfield_rank", 3))})
    df = pd.DataFrame(rows)
    out = repro.runs() / "v7rev_stats"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "W1T_structural_ablation.csv", index=False, float_format="%.6g")
    base = df.iloc[0]
    head = {"n_variants": int(len(df) - 1), "seed": 42,
            "locked": {k: float(base[k]) for k in ("test_median_nrmse_scaled", "test_median_corr", "C0_macro_auroc", "C5_macro_auroc",
                                                   "fim_effrank_12", "fim_effrank_II", "boundary_rate",
                                                   "limb_relation_rel_error", "activation_order_violation_rate")}}
    for _, r in df.iloc[1:].iterrows():
        head[r.variant] = {"delta_nrmse": float(r.test_median_nrmse_scaled - base.test_median_nrmse_scaled),
                           "delta_corr": float(r.test_median_corr - base.test_median_corr),
                           "delta_C0_auroc": float(r.C0_macro_auroc - base.C0_macro_auroc),
                           "delta_C5_auroc": float(r.C5_macro_auroc - base.C5_macro_auroc),
                           "fim_effrank_12": float(r.fim_effrank_12), "fim_effrank_II": float(r.fim_effrank_II),
                           "boundary_rate": float(r.boundary_rate),
                           "limb_relation_rel_error": float(r.limb_relation_rel_error),
                           "activation_order_violation_rate": float(r.activation_order_violation_rate),
                           "C5_inside_locked_ci": bool(base.C5_auroc_ci_lo <= r.C5_macro_auroc <= base.C5_auroc_ci_hi)}
    (out / "W1T_headline.json").write_text(json.dumps(head, indent=2), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"-> {out / 'W1T_structural_ablation.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    variants = a.only or list(VARIANTS)
    write_configs()
    if a.train:
        train(variants)
    if a.analyze:
        analyze(variants)
    summarize()


if __name__ == "__main__":
    main()
