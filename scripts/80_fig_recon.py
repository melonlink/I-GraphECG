"""Regenerate main-text Figure 2 (observed vs reconstructed median beats) from the locked model.

Why this exists
---------------
``fig_recon.pdf`` shipped in the submission package has no generating script anywhere in the
repository: scripts 50/51/53/54/56 only copy it forward. It is therefore unreproducible from the
archive, the same provenance gap that Figure S1 had. It also embeds Type 3 fonts, which MDPI
production rejects and which leave the figure text unsearchable, and its legend says "real" where
the manuscript's own terminology is "observed".

This script rebuilds the figure from the locked seed-42 checkpoint
``d1s_r3_s42_best.pt`` with TrueType fonts, manuscript terminology and legible
type. The representative record per class is the one whose whole-beat correlation is nearest that
class's fold-10 median, which is the rule used by the earlier revision-figure script
(``the earlier revision-figure script (removed in the paper-only clean-up)``); ``--compare`` reports how closely the result matches
the shipped figure so the substitution can be checked rather than assumed.

Canvas sizing
-------------
The figure is placed at ``width=\\linewidth`` and ``\\linewidth`` in the MDPI class is 394.355 pt
= 5.457 in. Authoring on a wider canvas makes LaTeX scale everything down, so a nominal 14 pt
title printed at 6.9 pt. The canvas is therefore authored at exactly the printed width: the
LaTeX scale factor is 1.0 and point sizes below are point sizes on paper. Nothing is under 8 pt.

Usage (needs torch):
    python scripts/80_fig_recon.py
    ... --compare      also rasterise old and new and report the pixel difference
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from igraphecg import repro

from igraphecg.data.dataset import load_processed, split_indices
from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.data.ptbxl_loader import CANONICAL_LEADS

# PAPER root removed: figures go to the output root; the manuscript copies from there
CKPT = repro.lineage.checkpoint(42)
# The published figure location, resolved lazily; importing this module creates nothing.
def _default_out():
    return repro.outdir("figures") / "fig_recon.pdf"
# Column titles are the bare channel names. Annotating only I with "(limb)" implied that II was
# not a limb lead, which is wrong; the limb/precordial split is stated once in the caption
# instead, where it costs no panel width.
SHOW = [("I", "I"), ("II", "II"), ("V2", "V2"), ("V5", "V5")]

# \linewidth in the MDPI class is 394.355 TeX pt = 394.355/72.27 = 5.4566 in. Authoring a hair
# under that leaves the LaTeX scale factor at 1.0002, i.e. no down-scaling of any label.
FIG_W_IN = 5.456
FIG_H_IN = 4.9


def _even_out_yticks(ax, max_pad_frac: float = 0.28) -> None:
    """Nudge a panel's y limits so it shows at least three ticks, when that is cheap.

    Each panel keeps its own scale -- amplitudes genuinely differ between channels and classes,
    so a shared y range would be wrong. What made the grid read as 16 unrelated plots was the
    tick *count*, which ranged from 2 to 6. ``MaxNLocator`` caps the top; this lifts the bottom
    by extending a limit out to the next locator tick, but only when the extension costs less
    than ``max_pad_frac`` of the current span. Limits are padding, not data.
    """
    lo, hi = ax.get_ylim()
    loc = ax.yaxis.get_major_locator()
    tv = loc.tick_values(lo, hi)
    if len(tv) < 2:
        return
    step = float(tv[1] - tv[0])
    eps = 1e-9 * max(1.0, abs(hi))

    def n_inside(a, b):
        return sum(1 for t in np.arange(np.floor(a / step) * step,
                                        np.ceil(b / step) * step + step / 2, step)
                   if a - eps <= t <= b + eps)

    if n_inside(lo, hi) >= 3:
        return
    span = hi - lo
    costs = sorted([(lo - np.floor(lo / step + eps) * step, "lo"),
                    (np.ceil(hi / step - eps) * step - hi, "hi")])
    for cost, side in costs:
        if n_inside(lo, hi) >= 3:
            break
        if cost <= max_pad_frac * span:
            if side == "lo":
                lo -= cost
            else:
                hi += cost
    ax.set_ylim(lo, hi)


def _pdftoppm() -> str:
    """poppler's pdftoppm: PDF2IMAGE_POPPLER_PATH or POPPLER_BIN, else on PATH. Used only
    by --compare. The first version hard-coded the author's TeX Live path."""
    import os, shutil
    for d in (os.environ.get("PDF2IMAGE_POPPLER_PATH"), os.environ.get("POPPLER_BIN")):
        if d and (Path(d) / "pdftoppm.exe").exists():
            return str(Path(d) / "pdftoppm.exe")
        if d and (Path(d) / "pdftoppm").exists():
            return str(Path(d) / "pdftoppm")
    found = shutil.which("pdftoppm")
    if not found:
        raise SystemExit("--compare needs poppler's pdftoppm: install poppler or set POPPLER_BIN")
    return found


def main(compare: bool, out=None) -> None:
    OUT = Path(out) if out else _default_out()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    import torch
    import matplotlib
    matplotlib.use("Agg")
    # Type 42 = TrueType. matplotlib's default of Type 3 is what put unsearchable bitmap-style
    # glyphs in the shipped figure.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    from igraphecg.evaluation.round2_io import load_encoder_decoder

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    data = load_processed(ROOT / cfg["npz"])
    signals = data["signal_12lead"].astype(np.float32)
    labels = data["label"].astype(int)
    fold = data["fold"].astype(int)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals)
    te = split_indices(fold)["test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc, dec = load_encoder_decoder(CKPT, n_t=signals.shape[2], rank=cfg["leadfield_rank"],
                                    device=device)
    with torch.no_grad():
        yh = []
        for s in range(0, len(te), 256):
            z = enc(torch.from_numpy(scaled[te[s:s + 256]]).to(device))
            yh.append(dec.forward_from_z(z)[0].cpu().numpy())
    recon_mV = scaler.inverse_transform(np.concatenate(yh))
    real_mV = signals[te]
    lab = labels[te]

    # representative record per class: whole-beat correlation nearest that class's median
    corr = np.array([np.corrcoef(real_mV[i].ravel(), recon_mV[i].ravel())[0, 1]
                     for i in range(len(te))])
    rep = {}
    for c in range(len(TARGET_CLASSES)):
        idx = np.flatnonzero(lab == c)
        rep[c] = idx[np.argmin(np.abs(corr[idx] - np.median(corr[idx])))]
        print(f"  {TARGET_CLASSES[c]:<5} record {te[rep[c]]:>6}  corr {corr[rep[c]]:.4f}")

    t_ms = dec.t.cpu().numpy() * 1000
    cols = [CANONICAL_LEADS.index(k) for k, _ in SHOW]

    from matplotlib.ticker import FuncFormatter, MaxNLocator

    fig, axes = plt.subplots(4, 4, figsize=(FIG_W_IN, FIG_H_IN), sharex=True,
                             layout="constrained")
    # w_pad has to clear the widest y tick label ("-2.0") from the neighbouring panel's spine.
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.03, hspace=0.03)
    handles = None
    for r in range(4):
        i = rep[r]
        for c, (li, (_, title)) in enumerate(zip(cols, SHOW)):
            ax = axes[r, c]
            ax.plot(t_ms, real_mV[i, li], color="black", lw=0.8, label="observed")
            ax.plot(t_ms, recon_mV[i, li], color="tab:red", lw=0.8, alpha=0.85,
                    label="reconstructed")
            ax.axvline(0, color="gray", ls=":", lw=0.5)
            ax.grid(alpha=0.3, lw=0.4)
            ax.tick_params(labelsize=8, length=2, width=0.5, pad=1.5)
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)
            # Three x ticks only: at ~0.93 in of panel width the previous four collided at 8 pt,
            # and any pair straddling t=0 sat closer than a word space. These are 400 ms apart,
            # which clears the widest label. t=0 itself is marked by the dotted rule.
            ax.set_xticks([-200, 200, 600])
            # Panels keep their own y range (the amplitudes genuinely differ), but the tick
            # count is bounded so the grid reads as one figure rather than 16 unrelated plots.
            # steps excludes 2.5 so every tick is exact at one decimal (a 0.25 step would be
            # mislabelled by the one-decimal formatter).
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
            # %-formatting emits an ASCII hyphen, which the x axis (left to the default
            # formatter) sets as U+2212; the journal requires the true minus sign on both axes.
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _pos: f"{v:.1f}".replace("-", "\N{MINUS SIGN}")))
            if r == 0:
                ax.set_title(title, fontsize=9, pad=2.5)
            if c == 0:
                ax.set_ylabel(f"{TARGET_CLASSES[r]}\n(mV)", fontsize=8.5, labelpad=1.5)
            _even_out_yticks(ax)
            if handles is None:
                handles = ax.get_lines()[:2]
    # Figure-level legend: inside the axes it covered the R peaks of the (NORM, I) panel, which
    # is precisely the agreement the figure exists to show.
    fig.legend(handles=handles, labels=["observed", "reconstructed"],
               loc="outside upper center", ncols=2, frameon=False, fontsize=8.5,
               handlelength=1.8, columnspacing=1.6, borderpad=0.55)
    fig.align_ylabels(axes[:, 0])
    fig.supxlabel("time relative to R (ms)", fontsize=9)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print("written", OUT)

    if compare:
        import subprocess
        from PIL import Image
        tmp = OUT.parent / "_cmp_new.png"
        subprocess.run([_pdftoppm(), "-r", "80",
                        "-png", "-singlefile", str(OUT), str(tmp.with_suffix(""))], check=True)
        print("rendered new figure for visual comparison ->", tmp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--out", default=None, help="output file; default = <output root>/figures/fig_recon.pdf")
    a = ap.parse_args()
    main(a.compare, out=a.out)
