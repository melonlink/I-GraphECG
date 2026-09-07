"""Round 3 Task C: identifiability (Jacobian/FIM, effective rank, parameter confidence, coupling).
Also computes the decoder-level FIM effective rank for each lead subset (Task E §8.2.1).

Run: python scripts/43_identifiability.py --config configs/v1fix/r3_s42.yaml
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
from igraphecg.evaluation.identifiability import (compute_fim, fim_metrics, group_scores,
                                            identifiability_scores, lead_rows, param_names)
from igraphecg.evaluation.lead_ablation import LEAD_SETS
from igraphecg.evaluation.round2_io import infer_theta, load_encoder_decoder
from igraphecg.utils.config import parse_args_with_config
from igraphecg.utils.logging import get_logger
from igraphecg.utils.paths import PROJECT_ROOT, ensure_dirs
from igraphecg.utils.seed import set_seed

log = get_logger("identifiability")


def subset_idx(label, fold, per_class, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(len(TARGET_CLASSES)):
        pool = np.where((label == c) & (fold == 10))[0]
        idx.append(pool if len(pool) <= per_class else rng.choice(pool, per_class, replace=False))
    return np.concatenate(idx)


def main():
    _, cfg = parse_args_with_config("Round3 identifiability")
    ensure_dirs()
    set_seed(cfg["seed"])
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_processed(PROJECT_ROOT / cfg["npz"])
    signals = data["signal_12lead"].astype(np.float32)
    n_t = signals.shape[2]
    label = data["label"].astype(int); fold = data["fold"].astype(int)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals)
    enc, dec = load_encoder_decoder(repro.outputs() / cfg["encoder_ckpt"], n_t, cfg["leadfield_rank"], device)

    idx = subset_idx(label, fold, cfg["fim_per_class"], cfg["seed"])
    theta, _ = infer_theta(enc, dec, scaled[idx], device=device)
    thetas = torch.from_numpy(theta).to(device)
    radius = dec.pspace.radius
    log.info(f"FIM subset n={len(idx)} (per_class={cfg['fim_per_class']})")

    out = repro.outputs() / cfg["out_dir"]
    (out / "tables").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(parents=True, exist_ok=True)

    # ---- decoder-level FIM effective rank for each lead subset ----
    rows = []
    F12 = None
    for name, leads in LEAD_SETS.items():
        rws = torch.tensor(lead_rows(leads, n_t), dtype=torch.long, device=device)
        F = compute_fim(dec, thetas, radius, observed_rows=rws, device=device)
        m = fim_metrics(F)
        rows.append({"lead_set": name, "n_leads": len(leads), "n_samples": len(idx),
                     "effective_rank": m["effective_rank"], "rank95": m["rank95"],
                     "rank99": m["rank99"], "condition_number": m["condition_number"],
                     "logdet": m["logdet"]})
        if name == "L12":
            F12 = F
        log.info(f"[{name}] eff_rank={m['effective_rank']:.2f} rank95={m['rank95']} cond={m['condition_number']:.2e}")
    pd.DataFrame(rows).to_csv(out / "tables" / "fim_effective_rank_by_leadset.csv", index=False)

    # per-class FIM effective rank (L12)
    cls_rows = []
    for c, cn in enumerate(TARGET_CLASSES):
        sel = label[idx] == c
        Fc = compute_fim(dec, thetas[sel], radius, device=device)
        cls_rows.append({"class": cn, "effective_rank": fim_metrics(Fc)["effective_rank"], "n": int(sel.sum())})
    pd.DataFrame(cls_rows).to_csv(out / "tables" / "fim_effective_rank_by_class.csv", index=False)

    # ---- parameter confidence + coupling (L12) ----
    score, corr = identifiability_scores(F12, lam=1e-3)
    names = param_names()
    sdf = pd.DataFrame({"param": names, "identifiability_score": score}).sort_values(
        "identifiability_score", ascending=False)
    sdf.to_csv(out / "tables" / "fim_parameter_identifiability.csv", index=False)
    sdf.head(10).to_csv(out / "tables" / "most_identifiable_parameters.csv", index=False)
    sdf.tail(10).to_csv(out / "tables" / "least_identifiable_parameters.csv", index=False)
    gs = group_scores(score)
    pd.DataFrame([{"group": g, "mean_score": v} for g, v in gs.items()]).to_csv(
        out / "tables" / "identifiability_by_param_group.csv", index=False)
    log.info("param group identifiability: " + ", ".join(f"{g}={v:.2e}" for g, v in gs.items()))

    # top coupled pairs
    P = len(names); pairs = []
    for i in range(P):
        for j in range(i + 1, P):
            pairs.append({"a": names[i], "b": names[j], "abs_corr": abs(corr[i, j]), "corr": corr[i, j]})
    pd.DataFrame(sorted(pairs, key=lambda r: -r["abs_corr"])[:20]).to_csv(
        out / "tables" / "top_20_coupled_parameter_pairs.csv", index=False)

    # figures: FIM spectrum + identifiability by parameter group + correlation heatmap
    eig = np.clip(np.linalg.eigvalsh(F12), 1e-12, None)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4)); ax.semilogy(eig, "o-", ms=3)
    ax.set_title("FIM spectrum (12-lead)"); ax.set_xlabel("index"); ax.set_ylabel("eigenvalue"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "figures" / "fim_spectrum_12lead.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(gs)), list(gs.values())); ax.set_yscale("log")
    ax.set_xticks(range(len(gs)), list(gs.keys()), rotation=30); ax.set_title("identifiability by param group")
    fig.tight_layout(); fig.savefig(out / "figures" / "identifiability_by_param_group.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Fisher parameter correlation (12-lead)"); fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(out / "figures" / "parameter_correlation_heatmap_fim.png", dpi=130); plt.close(fig)

    # lead-set spectrum comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, leads in LEAD_SETS.items():
        rws = torch.tensor(lead_rows(leads, n_t), dtype=torch.long, device=device)
        F = compute_fim(dec, thetas, radius, observed_rows=rws, device=device)
        e = np.clip(np.linalg.eigvalsh(F), 1e-12, None)[::-1]
        ax.semilogy(e, label=name)
    ax.legend(); ax.set_title("FIM spectrum by lead set"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "figures" / "fim_spectrum_leadsets.png", dpi=130); plt.close(fig)
    log.info(f"Task C done -> {out}")


if __name__ == "__main__":
    main()
