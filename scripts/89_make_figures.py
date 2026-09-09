"""Regenerate every data figure the manuscript includes, into one output directory.

    "$PY" scripts/89_make_figures.py                # -> <output root>/figures/
    "$PY" scripts/89_make_figures.py --out DIR --only fig_descriptors fig_recon

This replaces 89_make_figures.py and 89_make_figures.py, which produced the same figures by
READING OTHER SCRIPTS' SOURCE TEXT, applying string substitutions, and executing the result
(run_patched). That existed only because the scripts they drove held their paths, lineage and
figure text in module-level constants that could not be parameterised. Stages S1-S3 made the
published values those scripts' own defaults, so every substitution became either dead or
the default, and the drivers reduce to ordinary function calls.

What each figure is, and where it comes from:

    fig_recon                 80_fig_recon      Figure 2
    fig_classification        drawn here, from round3_classification_summary   Figure 3
    fig_confusion             drawn here, from round3_xgboost_test_predictions Figure 4
    fig_descriptors           81_fig_descriptors           Figure 5
    fig_identifiability_ab    83_fig_identifiability_selection               Figure 6  (published version)
    fig_selection_cd          83_fig_identifiability_selection               Figure 7  (published version)
    figS1_recovery            drawn here, from ap_ode_monodromy.csv (11)     Figure S1
    figS2_recon12             drawn here, one fold-10 NORM record            Figure S2

    fig_overview              NOT generated: a hand-authored schematic (Figure 1)

Two things are deliberate. Figures 6 and 7 come from 83_fig_identifiability_selection, not from
61_split_fig_identifiability: 61_split draws the Revision-v6 versions, and the documented
figure command used to regenerate those -- so a reader following the notes got two figures
that differ from the ones in the paper. And nothing here writes into a manuscript directory;
figures land under the output root and the manuscript copies them from there.
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from igraphecg import repro  # noqa: E402

TARGETS = ["fig_recon", "fig_classification", "fig_confusion", "fig_descriptors",
           "fig_identifiability_ab", "fig_selection_cd", "figS1_recovery", "figS2_recon12"]
NOT_GENERATED = {
    "fig_overview": "hand-authored schematic (Figure 1); carries no computed number",
}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "Arial", "font.size": 8, "pdf.fonttype": 42})


# ---- figures drawn here ---------------------------------------------------------------

def fig_classification(out: Path) -> None:
    """Figure 3: macro-AUROC by interpretable feature group, black-box references dashed."""
    import matplotlib.pyplot as plt
    import pandas as pd
    _style()
    summ = pd.read_csv(repro.runs() / "v1fix_r3_s42/tables/round3_classification_summary.csv")
    summ = summ[summ.classifier == "xgboost"].set_index("feature_group")
    order = ["C0_theta", "C2_theta_rep", "C3_theta_st", "C4_theta_stab",
             "C5_theta_rep_st_stab", "C6_compact_heuristic"]
    labels = ["C0", "C2", "C3", "C4", "C5", "C6"]
    au = [summ.loc[g, "macro_auroc"] for g in order]
    lo = [summ.loc[g, "auroc_ci_lo"] for g in order]
    hi = [summ.loc[g, "auroc_ci_hi"] for g in order]
    # printed at 0.58 of the text width (3.17 in): draw at that width so 8 pt is 8 pt
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    x = range(len(order))
    ax.errorbar(x, au, yerr=[[a - b for a, b in zip(au, lo)], [h - a for h, a in zip(hi, au)]],
                fmt="o", ms=4.5, capsize=3, lw=1, color="#2f4b7c")
    ax.axhline(0.9200, ls="--", color="#888", lw=1)
    ax.axhline(0.9241, ls=":", color="#888", lw=1)
    ax.set_xlim(-0.45, 5.45)
    ax.text(5.32, 0.9192, "hand-crafted+XGB 0.9200", fontsize=7, color="#666", ha="right", va="top")
    ax.text(5.32, 0.9249, "ResNet1D 0.9241 (black-box reference)", fontsize=7, color="#666",
            ha="right", va="bottom")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylim(0.87, 0.93)
    ax.set_ylabel("macro-AUROC"); ax.set_xlabel("interpretable feature group")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_classification.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_confusion(out: Path) -> None:
    """Figure 4: C5 confusion matrix on fold 10, row-normalised, no in-figure title."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    _style()
    plt.rcParams.update({"font.size": 9})
    CL = ["NORM", "MI", "STTC", "CD"]
    pred = pd.read_csv(repro.runs() / "v1fix_r3_s42/tables/round3_xgboost_test_predictions.csv")
    c5 = pred[pred.feature_group == "C5_theta_rep_st_stab"]
    M = np.zeros((4, 4), int)
    for i in range(4):
        for j in range(4):
            M[i, j] = int(((c5.label == i) & (c5.predicted_label == j)).sum())
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(M / M.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            frac = M[i, j] / M[i].sum()
            ax.text(j, i, f"{M[i, j]}\n({frac:.2f})", ha="center", va="center", fontsize=7.5,
                    color="white" if frac > 0.55 else "#222")
    ax.set_xticks(range(4), CL); ax.set_yticks(range(4), CL)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row fraction")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_confusion.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figS2_recon12(out: Path) -> None:
    """Figure S2: real vs reconstructed 12-lead median beat for one fold-10 NORM record."""
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from igraphecg.data.dataset import load_processed
    from igraphecg.data.preprocess import RobustLeadScaler
    from igraphecg.evaluation.round2_io import infer_theta, load_encoder_decoder
    _style()
    plt.rcParams.update({"font.size": 9})
    data = load_processed(repro.processed() / "ptbxl_medianbeat_clean_100hz.npz")
    sig = data["signal_12lead"].astype(np.float32)
    fold = data["fold"].astype(int)
    labn = data["label_name"]
    sc = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = sc.transform(sig)
    i = int(np.where((fold == 10) & (labn == "NORM"))[0][3])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc, dec = load_encoder_decoder(repro.lineage.checkpoint(42), sig.shape[2], 3, dev)
    th, _ = infer_theta(enc, dec, scaled[i:i + 1], device=dev)
    with torch.no_grad():
        yh = dec.forward(torch.from_numpy(th).to(dev))[0][0].cpu().numpy()
    LE = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    t = np.arange(100) * 10 - 300
    # printed at the supplement text width (5.46 in): draw at that width
    fig, axes = plt.subplots(3, 4, figsize=(5.46, 3.1), sharex=True)
    for k, ax in enumerate(axes.ravel()):
        ax.plot(t, scaled[i, k], color="k", lw=0.9, label="observed")
        ax.plot(t, yh[k], color="#c0392b", lw=0.9, label="reconstructed")
        ax.text(0.03, 0.93, LE[k], transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    for ax in axes[-1]:
        ax.set_xlabel("time from R peak (ms)", fontsize=7.5)
    for r in range(3):
        axes[r, 0].set_ylabel("scaled amplitude", fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "figS2_recon12.png", dpi=300, facecolor="white")
    plt.close(fig)


def figS1_recovery(out: Path) -> None:
    """Figure S1: paced finite-time recovery margin 1 - rho by diagnostic class.

    Data: ap_ode_monodromy.csv from 41_stability (m_stab = 1 - rho). The caption
    calls this a transparency check that supports no main-text claim. The manuscript's copy
    was drawn by a script that no longer exists in the tree, so this regenerates it from the
    same table; whether it is pixel-identical to the submitted PNG is reported by the
    figure-identity test, not assumed here.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    _style()
    d = pd.read_csv(repro.runs() / "v1fix_r3_s42/tables/ap_ode_monodromy.csv")
    classes = ["NORM", "MI", "STTC", "CD"]
    vals = [d.loc[d["class"] == c, "m_stab"].to_numpy() for c in classes]
    # Plain matplotlib boxplot, as the submitted PNG shows: one light box fill, default
    # orange medians, black whiskers, outliers drawn as open circles, no strip points.
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    bp = ax.boxplot(vals, labels=classes, widths=0.5, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#dce3ec"); patch.set_edgecolor("#333")
    ax.set_ylabel(r"finite-time recovery margin $1-\rho$")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "figS1_recovery.png", dpi=300, facecolor="white")
    plt.close(fig)


# ---- figures drawn by their own scripts, called as functions ----------------------------

def _fresh(module: str):
    """Import a drawing script, re-executing it if this process imported it before.

    81 and 83 apply their rcParams at module level, so a cached import would draw with
    whatever main() reset the defaults to; reloading re-applies the script's own style."""
    m = sys.modules.get(module)
    return importlib.reload(m) if m is not None else importlib.import_module(module)


def fig_recon(out: Path) -> None:
    m = _fresh("80_fig_recon")
    m.main(compare=False, out=out / "fig_recon.pdf")


def fig_descriptors(out: Path) -> None:
    m = _fresh("81_fig_descriptors")
    argv, sys.argv = sys.argv, ["81", "--out", str(out), "--name", "fig_descriptors"]
    try:
        m.main()
    finally:
        sys.argv = argv


def _panel_assets() -> None:
    """82 writes the panel CSVs that 83 reads; run it once, into its published location."""
    m = importlib.import_module("82_fig_identifiability_assets")
    m.main()


def fig_identifiability_and_selection(out: Path) -> None:
    """Figures 6 and 7, the PUBLISHED versions, from 83_fig_identifiability_selection."""
    m = _fresh("83_fig_identifiability_selection")
    argv, sys.argv = sys.argv, ["83", "--out", str(out)]
    try:
        m.main()
    finally:
        sys.argv = argv


def main(out: Path | None = None, only: list[str] | None = None, refresh_assets: bool = False) -> None:
    out = Path(out) if out else repro.outdir("figures")
    out.mkdir(parents=True, exist_ok=True)
    want = set(only) if only else set(TARGETS)
    unknown = want - set(TARGETS) - set(NOT_GENERATED)
    if unknown:
        raise SystemExit(f"unknown figure(s): {sorted(unknown)}; choose from {TARGETS}")
    for name in sorted(want & set(NOT_GENERATED)):
        print(f"[skip] {name}: {NOT_GENERATED[name]}")

    # fig_recon, fig_descriptors and figS2 re-run the encoder over PTB-XL median beats, so
    # they need the derived cache; the other figures draw from the shipped tables alone.
    # Say which step is missing instead of failing inside a loader with a bare path.
    NEEDS_CACHE = {"fig_recon", "fig_descriptors", "figS2_recon12"}
    cache = repro.processed() / "ptbxl_medianbeat_clean_100hz.npz"
    if (want & NEEDS_CACHE) and not cache.exists():
        raise SystemExit(
            f"{sorted(want & NEEDS_CACHE)} need the PTB-XL median-beat cache, which is not at\n"
            f"  {cache}\n"
            "Build it once with ECG_DATA_DIR pointing at PTB-XL (see data/README.md):\n"
            "  python scripts/10_prepare_ptbxl.py --config configs/ptbxl_data.yaml\n"
            "The remaining figures need no raw data: pass --only with their names.")

    steps = [
        ("fig_recon", lambda: fig_recon(out)),
        ("fig_classification", lambda: fig_classification(out)),
        ("fig_confusion", lambda: fig_confusion(out)),
        ("fig_descriptors", lambda: fig_descriptors(out)),
        ("fig_identifiability_ab", None),   # produced together with fig_selection_cd
        ("fig_selection_cd", None),
        ("figS1_recovery", lambda: figS1_recovery(out)),
        ("figS2_recon12", lambda: figS2_recon12(out)),
    ]
    if refresh_assets and (want & {"fig_identifiability_ab", "fig_selection_cd"}):
        print("### panel assets (82_fig_identifiability_assets, five checkpoints)")
        _panel_assets()
    done = set()
    import matplotlib as mpl
    for name, fn in steps:
        if name not in want or name in done:
            continue
        print(f"### {name}")
        # Every figure is drawn from matplotlib's default rcParams, whatever ran before it in
        # this process. fig_recon (80) draws with the defaults -- DejaVu Sans, size 10 -- while
        # _style() and the drawers above set Arial 8/9 and never restore it; drawn third in a
        # process, fig_recon inherited that and differed from the manuscript in 15% of pixels
        # while being exact when drawn alone. rc_context restores the caller's state after.
        with mpl.rc_context():
            mpl.rcdefaults()
            if fn is None:
                fig_identifiability_and_selection(out)
                done |= {"fig_identifiability_ab", "fig_selection_cd"}
            else:
                fn()
                done.add(name)
    print(f"figures -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=None, help="default: <output root>/figures")
    ap.add_argument("--only", nargs="*", default=None, help=f"subset of {TARGETS}")
    ap.add_argument("--refresh-assets", action="store_true",
                    help="re-run 82_fig_identifiability_assets (five checkpoints) before Figures 6-7; "
                         "otherwise the published panel CSVs under v1fix/runs/fig34 are used")
    a = ap.parse_args()
    main(out=a.out, only=a.only, refresh_assets=a.refresh_assets)
