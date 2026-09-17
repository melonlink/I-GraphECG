"""Block-3 Figure 3 (UPDATED): four core interpretable descriptors by class.
Panel A = QRS duration (reconstructed)  [replaces D_act]; B/C/D unchanged.

Boxplots over PTB-XL fold-10 test, grouped NORM/MI/STTC/CD, on the locked seed-42 model.
A from reconstructed-waveform QRS width; B/C/D from round3_eikonal_B features.
Effect-size annotations use the canonical (5-seed where noted) Cliff's delta.

Run: "$PY" PUBLIC/scripts/81_fig_descriptors.py --out <dir>
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
try:
    import torch   # noqa
except Exception:
    pass
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from igraphecg import repro
from igraphecg.data.dataset import load_processed
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.evaluation.round2_io import load_encoder_decoder, infer_theta

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"], "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "path",
                     "axes.titlesize": 9})
P1 = repro.outputs()
CKPT = repro.lineage.checkpoint(42)
FEAT = repro.runs() / "v1fix_r3_s42/tables/round3_features.csv"   # was round3_eikonal_B: a SUPERSEDED lineage
CLASSES = ["NORM", "MI", "STTC", "CD"]; CCOL = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]


def qrs_width_ms(beats, t):
    dt = (t[-1] - t[0]) / (len(t) - 1) * 1000.0
    rms = np.sqrt((beats ** 2).mean(axis=1)); win = np.where((t >= -0.10) & (t <= 0.16))[0]
    out = np.zeros(len(beats))
    for i in range(len(beats)):
        rk = win[rms[i, win].argmax()]; thr = 0.15 * rms[i, rk] + 1e-9
        on = rk
        while on > win[0] and rms[i, on] > thr: on -= 1
        off = rk
        while off < win[-1] and rms[i, off] > thr: off += 1
        out[i] = (off - on) * dt
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--name", default="fig_descriptors")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_processed(ROOT / "data/processed/ptbxl_medianbeat_clean_100hz.npz")
    signals = data["signal_12lead"].astype(np.float32); n_t = signals.shape[2]
    label = data["label"].astype(int); fold = data["fold"].astype(int)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals); te = np.where(fold == 10)[0]; t = np.linspace(-0.30, 0.69, n_t)
    enc, dec = load_encoder_decoder(ROOT / CKPT, n_t, 3, dev)
    theta, _ = infer_theta(enc, dec, scaled[te], device=dev)
    with torch.no_grad():
        yh = np.concatenate([dec.forward(torch.from_numpy(theta[i:i+512]).to(dev))[0].cpu().numpy()
                             for i in range(0, len(theta), 512)])
    qrs = qrs_width_ms(scaler.inverse_transform(yh), t)
    lab_te = label[te]
    fdf = pd.read_csv(ROOT / FEAT); ftt = fdf.fold == 10
    def col(c): return [fdf.loc[ftt & (fdf.label == k), c].values for k in range(4)]
    panels = [
        ("(a)", "QRS duration, reconstructed (ms)", [qrs[lab_te == k] for k in range(4)], "CD vs NORM:  δ = +0.53 ± 0.02"),
        ("(b)", "cross-lead T-wave sign discordance", col("obs_T_sign_discordance_rate"), "STTC vs NORM:  δ = +0.753"),
        ("(c)", "T-peak max. abs. amplitude (mV)", col("obs_Tpeak_maxabs"), "STTC vs NORM:  δ = −0.531"),
        ("(d)", "lead-projected ST max. abs. (scaled)", col("ST_projected_max_abs"), "MI vs NORM:  δ = −0.19 ± 0.18 (seed-variable)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6))
    for ax, (lab, title, dat, ann) in zip(axes.ravel(), panels):
        bp = ax.boxplot(dat, tick_labels=CLASSES, showfliers=False, widths=0.62, patch_artist=True)
        for patch, c in zip(bp["boxes"], CCOL):
            patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor("#333")
        for med in bp["medians"]:
            med.set_color("#222"); med.set_linewidth(1.2)
        ax.set_ylabel(title, fontsize=9); ax.grid(alpha=0.3, axis="y")
        # add top headroom so the letter and annotation sit ABOVE the whisker caps
        y0, y1 = ax.get_ylim(); ax.set_ylim(y0, y1 + (y1 - y0) * 0.20)
        # MDPI convention: bold lowercase panel marker BELOW the panel, centered
        ax.text(0.5, -0.14, lab, transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="top", ha="center")
        # effect-size annotation in the top-RIGHT headroom band (clear of the boxplots)
        ax.text(0.97, 0.97, ann, transform=ax.transAxes, ha="right", va="top", fontsize=7.4, color="#333",
                bbox=dict(boxstyle="round,pad=0.25", fc="#f7f7f7", ec="#bbb", lw=0.6))
    fig.tight_layout(); fig.subplots_adjust(hspace=0.42)
    for ext, kw in [("pdf", {}), ("png", {"dpi": 600})]:
        fig.savefig(out / f"{a.name}.{ext}", facecolor="white", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"wrote {a.name}.pdf/.png -> {out}  (QRS median by class: " +
          ", ".join(f"{CLASSES[k]}={np.median(qrs[lab_te==k]):.0f}ms" for k in range(4)) + ")")


if __name__ == "__main__":
    main()
