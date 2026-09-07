"""W1J precision audit — is the seed-42 residual against the frozen v1fix_ls8_s42
artifact a protocol difference or float32 accumulation noise?

Recomputes the seed-42 identity arm three ways on the identical fixed subset:
  (i)   float32 Jacobians, GPU einsum   (what scripts/42 and W1J do)
  (ii)  float64 Jacobians, GPU einsum   (numerically converged reference)
  (iii) float32 Jacobians, CPU einsum   (different accumulation order)
and reports logdet(I+V1+V4), logdet(II+V1+V5) and their gap for each, together
with the frozen artifact's own internal inconsistency (its greedy table vs its
exhaustive table report different logdet for the SAME set I+V1+V4).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys as _sys
from pathlib import Path as _Path
# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro
REPO = repro.repo()
sys.path.insert(0, str(REPO))

from igraphecg.data.dataset import load_processed
from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.data.ptbxl_loader import CANONICAL_LEADS
from igraphecg.evaluation.identifiability import lead_rows
from igraphecg.evaluation.round2_io import load_encoder_decoder, infer_theta

OUT = repro.outdir("runs/v7rev_stats")
CKPT = repro.lineage.checkpoint(42)
NPZ = REPO / "data/processed/ptbxl_medianbeat_clean_100hz.npz"
FROZEN = repro.runs() / "v1fix_ls8_s42/tables"
LAM = 1e-3
IND_NAMES = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
IND = [CANONICAL_LEADS.index(x) for x in IND_NAMES]
OPT = tuple(CANONICAL_LEADS.index(x) for x in ("I", "V1", "V4"))
CLIN = tuple(CANONICAL_LEADS.index(x) for x in ("II", "V1", "V5"))


def fixed_subset(label, fold, per_class=100, seed=42):
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(len(TARGET_CLASSES)):
        pool = np.where((label == c) & (fold == 10))[0]
        idx.append(pool if len(pool) <= per_class else rng.choice(pool, per_class, replace=False))
    return np.concatenate(idx)


def logdet(F):
    return float(np.linalg.slogdet(F + LAM * np.eye(F.shape[0]))[1])


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_processed(NPZ)
    signals = data["signal_12lead"].astype(np.float32)
    n_t = signals.shape[2]
    label = data["label"].astype(int)
    fold = data["fold"].astype(int)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals)

    enc, dec = load_encoder_decoder(CKPT, n_t, 3, dev)
    idx = fixed_subset(label, fold)

    # theta reproducibility: GPU vs CPU encoder pass
    th_gpu, _ = infer_theta(enc, dec, scaled[idx], device=dev)
    enc_c, dec_c = load_encoder_decoder(CKPT, n_t, 3, "cpu")
    th_cpu, _ = infer_theta(enc_c, dec_c, scaled[idx], device="cpu")
    theta_max_abs = float(np.abs(th_gpu - th_cpu).max())
    theta_max_rel = float((np.abs(th_gpu - th_cpu) / np.maximum(np.abs(th_gpu), 1e-12)).max())
    print(f"[audit] theta GPU-vs-CPU max|diff|={theta_max_abs:.3e}  max rel={theta_max_rel:.3e}")

    from torch.func import jacfwd
    thetas = torch.from_numpy(th_gpu).to(dev)
    radius = dec.pspace.radius
    r = radius.to(dev).view(1, -1)

    def f(th):
        y12, _ = dec.forward(th.unsqueeze(0))
        return y12[0].reshape(-1)

    J = torch.stack([jacfwd(f)(thetas[s].to(dev)) * r
                     for s in range(thetas.shape[0])]).detach()      # [400,1200,47] float32
    N = J.shape[0]

    rows = {}
    variants = {}
    for name, Jv, where in [("f32_gpu", J, dev),
                            ("f64_gpu", J.double(), dev),
                            ("f32_cpu", J.cpu(), "cpu")]:
        G = {}
        for l in IND:
            ridx = torch.tensor(lead_rows([l], n_t), device=Jv.device)
            Jl = Jv[:, ridx, :]
            G[l] = torch.einsum("nrp,nrq->npq", Jl, Jl).mean(0).cpu().numpy().astype(np.float64)
        variants[name] = G
        d_opt = logdet(sum(G[l] for l in OPT))
        d_clin = logdet(sum(G[l] for l in CLIN))
        rows[name] = {"logdet_I+V1+V4": d_opt, "logdet_II+V1+V5": d_clin, "gap": d_opt - d_clin}
        print(f"[audit] {name:8s} opt={d_opt:.6f} clin={d_clin:.6f} gap={d_opt-d_clin:.6f}")
        del where

    # order-of-summation sensitivity inside a single variant (the frozen run's own noise source)
    G = variants["f32_gpu"]
    perms = [(OPT[0], OPT[1], OPT[2]), (OPT[2], OPT[1], OPT[0]), (OPT[1], OPT[0], OPT[2])]
    order_vals = [logdet(sum(G[l] for l in p)) for p in perms]
    print(f"[audit] summation-order spread within f32_gpu: {max(order_vals)-min(order_vals):.2e}")

    frozen_ex = pd.read_csv(FROZEN / "optimal3_vs_clinical.csv")
    frozen_gr = pd.read_csv(FROZEN / "greedy_D_8indep.csv")
    frozen_gap = pd.read_csv(FROZEN / "bootstrap_gap_summary.csv")
    fz_ex = float(frozen_ex.iloc[0]["D_logdet"])
    fz_gr = float(frozen_gr[frozen_gr.k == 3]["D"].values[0])
    fz_clin = float(frozen_ex.iloc[1]["D_logdet"])
    print(f"[audit] FROZEN internal inconsistency for the SAME set I+V1+V4: "
          f"exhaustive={fz_ex:.9f} greedy={fz_gr:.9f} |diff|={abs(fz_ex-fz_gr):.3e}")

    ref = rows["f64_gpu"]
    out = []
    for k, v in rows.items():
        out.append({"variant": k, **v,
                    "abs_diff_vs_f64": v["logdet_I+V1+V4"] - ref["logdet_I+V1+V4"],
                    "gap_diff_vs_f64": v["gap"] - ref["gap"]})
    out.append({"variant": "FROZEN_v1fix_ls8_s42_exhaustive",
                "logdet_I+V1+V4": fz_ex, "logdet_II+V1+V5": fz_clin,
                "gap": float(frozen_gap.iloc[0]["gap_lambda"]),
                "abs_diff_vs_f64": fz_ex - ref["logdet_I+V1+V4"],
                "gap_diff_vs_f64": float(frozen_gap.iloc[0]["gap_lambda"]) - ref["gap"]})
    out.append({"variant": "FROZEN_v1fix_ls8_s42_greedy_same_set",
                "logdet_I+V1+V4": fz_gr, "logdet_II+V1+V5": np.nan, "gap": np.nan,
                "abs_diff_vs_f64": fz_gr - ref["logdet_I+V1+V4"], "gap_diff_vs_f64": np.nan})
    df = pd.DataFrame(out)
    df["theta_gpu_cpu_max_abs_diff"] = theta_max_abs
    df["frozen_internal_same_set_diff"] = abs(fz_ex - fz_gr)
    df["f32_summation_order_spread"] = max(order_vals) - min(order_vals)
    df.to_csv(OUT / "W1J_precision_audit.csv", index=False)
    print(df.to_string(index=False))
    print(f"[audit] -> {OUT/'W1J_precision_audit.csv'}")


if __name__ == "__main__":
    main()
