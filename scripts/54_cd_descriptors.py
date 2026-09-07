"""Find a seed-robust CD mechanistic descriptor (replace unstable prox_delay). No training.

On each of the 5 checkpoints (seed 42/1/2/3/4), recompute CD-vs-NORM Cliff's delta for:
  1. QRS_dur_recon   : QRS width measured from the RECONSTRUCTED 12-lead beat (observable, leaf-driven)
  2. leaf_delay_spread: max-min of leaf edge delays (more identifiable leaf edges)
  3. total_activation_time: delta_max - delta_root
  4. prox_delay      : proximal edge sum (unstable baseline, control)
Also: QRS_dur_real (gold standard, from REAL beat) + consistency (corr QRS_recon vs QRS_real),
and corr(each candidate, prox_delay).

Run: "$PY" PUBLIC/scripts/54_cd_descriptors.py
"""
from __future__ import annotations
import sys
from pathlib import Path
try:
    import torch   # noqa
except Exception:
    pass
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from igraphecg import repro
from igraphecg.data.dataset import load_processed
from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.evaluation.round2_io import load_encoder_decoder, infer_theta

SEEDS = [42, 1, 2, 3, 4]
NORM, MI, STTC, CD = 0, 1, 2, 3


def ckpt_of(s):
    """The locked checkpoint for this seed: no branch, no fallback, sha256-checked."""
    return repro.lineage.checkpoint(s)


def cliffs(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return (int((a[:, None] > b[None, :]).sum()) - int((a[:, None] < b[None, :]).sum())) / (len(a) * len(b))


def qrs_width_ms(beats, t):
    """beats:[N,12,T] (mV) -> QRS width (ms) per record via cross-lead RMS threshold around R."""
    dt = (t[-1] - t[0]) / (len(t) - 1) * 1000.0
    rms = np.sqrt((beats ** 2).mean(axis=1))   # [N,T]
    win = np.where((t >= -0.10) & (t <= 0.16))[0]
    out = np.zeros(len(beats))
    for i in range(len(beats)):
        seg = rms[i, win]
        rk = win[seg.argmax()]
        thr = 0.15 * rms[i, rk] + 1e-9
        on = rk
        while on > win[0] and rms[i, on] > thr:
            on -= 1
        off = rk
        while off < win[-1] and rms[i, off] > thr:
            off += 1
        out[i] = (off - on) * dt
    return out


def main(out=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_processed(ROOT / "data/processed/ptbxl_medianbeat_clean_100hz.npz")
    signals = data["signal_12lead"].astype(np.float32); n_t = signals.shape[2]
    label = data["label"].astype(int); fold = data["fold"].astype(int)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals); te = np.where(fold == 10)[0]
    t = np.linspace(-0.30, 0.69, n_t); labte = label[te]
    qrs_real = qrs_width_ms(signals[te], t)   # gold standard (model-independent)

    cand_names = ["QRS_dur_recon", "leaf_delay_spread", "total_activation_time", "prox_delay"]
    perseed = {c: {} for c in cand_names + ["QRS_dur_real_gold"]}
    cons = {"qrs_recon_vs_real_corr": {}, **{f"{c}_vs_prox_corr": {} for c in cand_names}}

    for s in SEEDS:
        enc, dec = load_encoder_decoder(ckpt_of(s), n_t, 3, dev)
        theta, _ = infer_theta(enc, dec, scaled[te], device=dev)
        with torch.no_grad():
            yh = np.concatenate([dec.forward(torch.from_numpy(theta[i:i+512]).to(dev))[0].cpu().numpy()
                                 for i in range(0, len(theta), 512)])
        yh_mv = scaler.inverse_transform(yh)
        feats = {}
        feats["QRS_dur_recon"] = qrs_width_ms(yh_mv, t)
        ed = theta[:, 1:8]   # edge_delay e0..e6
        feats["prox_delay"] = ed[:, 0] + ed[:, 1]
        leaf = ed[:, 2:7]
        feats["leaf_delay_spread"] = leaf.max(1) - leaf.min(1)
        feats["total_activation_time"] = ed[:, 0] + ed[:, 1] + leaf.max(1)
        feats["QRS_dur_real_gold"] = qrs_real
        for c in cand_names + ["QRS_dur_real_gold"]:
            a = feats[c][labte == CD]; b = feats[c][labte == NORM]
            d = cliffs(a, b); _, p = mannwhitneyu(a, b, alternative="two-sided")
            perseed[c][s] = (d, p)
        cons["qrs_recon_vs_real_corr"][s] = float(np.corrcoef(feats["QRS_dur_recon"], feats["QRS_dur_real_gold"])[0, 1])
        for c in cand_names:
            cons[f"{c}_vs_prox_corr"][s] = float(np.corrcoef(feats[c], feats["prox_delay"])[0, 1])
        print(f"[seed {s}] " + " ".join(f"{c}:δ={perseed[c][s][0]:+.3f}" for c in cand_names))

    out = Path(out) if out else repro.outdir("v1fix/runs/cd_descriptors")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cand_names + ["QRS_dur_real_gold"]:
        ds = np.array([perseed[c][s][0] for s in SEEDS])
        rows.append({"candidate": c, **{f"s{ss}": round(perseed[c][ss][0], 3) for ss in SEEDS},
                     "mean": ds.mean(), "std": ds.std(ddof=1),
                     "all5_positive": bool((ds > 0).all()), "std_lt_0.1": bool(ds.std(ddof=1) < 0.1)})
    df = pd.DataFrame(rows); df.to_csv(out / "cd_candidates_byseed.csv", index=False)
    print("\n=== CD-vs-NORM Cliff δ across seeds ===")
    print(df.to_string(index=False))
    print("\n=== consistency (mean across seeds) ===")
    print(f"  QRS_recon vs QRS_real corr = {np.mean(list(cons['qrs_recon_vs_real_corr'].values())):.3f}")
    for c in cand_names:
        print(f"  {c} vs prox_delay corr = {np.mean(list(cons[f'{c}_vs_prox_corr'].values())):.3f}")
    pd.DataFrame(cons).to_csv(out / "cd_candidates_consistency.csv")
    # verdict
    ok = df[(df.candidate.isin(cand_names)) & (df.all5_positive) & (df["std_lt_0.1"])]
    print("\n=== PASS (5/5 positive & std<0.1) ===")
    print(ok[["candidate", "mean", "std"]].to_string(index=False) if len(ok) else "  NONE")
    print(f"DONE -> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="output directory; default = the published run under the output root")
    main(out=ap.parse_args().out)
