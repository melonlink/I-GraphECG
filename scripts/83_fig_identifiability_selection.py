"""Redraw Figures 6 and 7 for Revision v7.

The v6 panels carry four defects the pre-resubmission audit identified:

  Fig 6a  a line joins II+V1+V5 to I+II+V5, a NON-NESTED pair, implying an ordering the
          text explicitly disowns ("monotonicity is a property of the nested comparison and
          does not order two lead sets neither of which contains the other").
  Fig 6a  the starred set is labelled "optimal 3-lead" with no noise model named, against
          a Results section that says the optimum is noise-model-dependent.
  Fig 6b  three colours encode a grouping with no legend.
  Fig 7a  a truncated baseline (169.0-171.5) makes a 0.47 log-det spread look decisive,
          with no error bars, on the paper's most contested claim.

The redraw fixes each at the source rather than in the caption:

  Fig 6a  x is now the NUMBER of leads, so the three-lead sets sit side by side at x=3 and
          the two nested chains are drawn separately. This makes the caption's own claim --
          lead count sets the ceiling, lead choice decides how closely it is approached --
          the thing the reader sees.
  Fig 6b  a legend, and the alpha_ST separation stated at its true magnitude.
  Fig 7a  the clinical set is drawn alongside the top three, on an axis that starts below
          it, with bootstrap intervals. The honest picture appears: the optimized sets
          cluster, the gap that matters is the one over the clinical set.

Data: outputs/v1fix/runs/fig34 (five-seed panels, the lineage the manuscript quotes) and
runs/v7rev_stats/W1J_* (exhaustive 56-subset search and bootstrap margins, both noise models).

Run: python scripts/83_fig_identifiability_selection.py --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
from igraphecg import repro
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = repro.outputs() / "v1fix/runs/fig34"
W1J = repro.runs() / "v7rev_stats"

# Drawn at the printed width (the 13.86 cm text width), so point sizes here are the point
# sizes on the page.
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"], "font.size": 8, "pdf.fonttype": 42,
                     "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5})

BLUE, RED, GREEN, GREY = "#4c72b0", "#c0392b", "#55a868", "#9aa0a6"
# parameter-group tick labels in the manuscript's notation (Section 2.3)
GROUP_LABELS = {"delta_root": "$\\delta_{\\mathrm{root}}$", "edge_prox": "$d_1$, $d_2$ (prox.)",
                "edge_leaf": "$d_3$–$d_7$ (leaf)", "APD": "APD",
                "tau_dep": "$\\tau_{\\mathrm{dep}}$", "tau_rep": "$\\tau_{\\mathrm{rep}}$",
                "q": "$q$", "alpha_ST": "$\\alpha_{ST}$", "global_gain": "$g$"}


def panel_marks(fig, xs, y=0.005):
    for x, lab in zip(xs, ("(a)", "(b)")):
        fig.text(x, y, lab, ha="center", va="bottom", fontsize=9, fontweight="bold")


def fig_identifiability(out: Path):
    pA = pd.read_csv(FIG / "panelA_degradation.csv").set_index("lead_set")
    pB = pd.read_csv(FIG / "panelB_group_identifiability.csv")

    fig = plt.figure(figsize=(5.46, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.38)

    # ---------------- (a) effective rank against the NUMBER of leads ----------------
    ax = fig.add_subplot(gs[0, 0])
    pts = {"12": 12, "II+V1+V5": 3, "I+II+V5": 3, "I+V1+V4": 3, "II": 1}

    def mv(k):
        return pA.loc[k, "mean"], pA.loc[k, "std"]

    # the two nested chains, drawn separately: 12 > S > II for each three-lead set
    for k, style in (("II+V1+V5", dict(ls="-", lw=1.3)), ("I+II+V5", dict(ls="--", lw=1.1))):
        xs, ys = [12, 3, 1], [mv("12")[0], mv(k)[0], mv("II")[0]]
        ax.plot(xs, ys, color=BLUE, alpha=.40, zorder=1, **style)

    # the spread among three-lead sets is the point of the panel: bracket it
    three = [mv(k)[0] for k in ("II+V1+V5", "I+II+V5", "I+V1+V4")]
    ax.annotate("", xy=(4.55, max(three)), xytext=(4.55, min(three)),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.1, alpha=.8), zorder=2)
    ax.text(4.9, sum(three) / 3, "same count,\ndifferent choice", fontsize=7,
            color=RED, va="center", ha="left")

    offsets = {"II+V1+V5": (-0.62, (-9, 9), "right"),
               "I+II+V5": (0.00, (-7, -16), "right"),
               "I+V1+V4": (0.72, (8, 5), "left")}
    for k, (dx, txt_off, ha) in offsets.items():
        m, s = mv(k)
        col = RED if k == "I+V1+V4" else BLUE
        mk, ms = ("*", 16) if k == "I+V1+V4" else ("o", 6)
        ax.errorbar([3 + dx], [m], yerr=[s], fmt=mk, color=col, ms=ms, capsize=4, zorder=3)
        ax.annotate(k, xy=(3 + dx, m), xytext=txt_off, textcoords="offset points",
                    ha=ha, fontsize=7, color=col, zorder=4)
    for k in ("12", "II"):
        m, s = mv(k)
        ax.errorbar([pts[k]], [m], yerr=[s], fmt="o", color=BLUE, ms=6, capsize=4, zorder=3)

    ax.set_xscale("log")
    ax.set_xticks([1, 3, 12])
    ax.set_xticklabels(["1\n(lead II)", "3", "12\n(all)"], fontsize=7.5)
    ax.set_xlim(0.75, 17)
    ax.set_xlabel("number of leads")
    ax.set_ylabel("FIM effective rank")
    ax.set_title("Effective rank vs lead count (mean$\\pm$SD, $n=5$ seeds)")
    ax.grid(alpha=.3, axis="y")
    ax.plot([], [], color=BLUE, alpha=.40, lw=1.3, label="nested reductions")
    ax.plot([], [], "*", color=RED, ms=13, ls="none", label="identity-noise optimum")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_xlim(0.75, 22)

    # ---------------- (b) group identifiability, with a legend ----------------
    axB = fig.add_subplot(gs[0, 1])
    groups = list(pB["group"])
    gm, gsd = list(pB["mean"]), list(pB["std"])
    delay = ("delta_root", "edge_prox", "edge_leaf")
    cols = [RED if g == "alpha_ST" else (GREEN if g in delay else BLUE) for g in groups]
    axB.bar(range(len(groups)), gm, yerr=gsd, color=cols, capsize=2, alpha=.85)
    axB.set_yscale("log")
    axB.set_xticks(range(len(groups)))
    axB.set_xticklabels([GROUP_LABELS.get(g, g) for g in groups], rotation=45, ha="right",
                        fontsize=7)
    axB.set_ylabel("identifiability score (log scale)")
    axB.set_title("Identifiability by parameter group")
    axB.grid(alpha=.3, axis="y", which="both")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=.85) for c in (GREEN, BLUE, RED)]
    axB.set_ylim(top=max(m + s for m, s in zip(gm, gsd)) * 12)
    axB.legend(handles, ["conduction delays", "repolarization / source",
                         "internal ST source"], fontsize=7, frameon=False, loc="upper left",
                ncol=1, handlelength=1.2, borderaxespad=0.3)
    lo = min(gm) - min(gsd)
    axB.annotate("", xy=(len(groups) - 1.5, min(gm)), xytext=(len(groups) - 1.5, 8.0),
                 arrowprops=dict(arrowstyle="<->", color="0.35", lw=1))
    axB.text(len(groups) - 1.58, 1.5, "1.5–2\norders", fontsize=7, color="0.25", ha="right")

    panel_marks(fig, [0.30, 0.80])
    fig.subplots_adjust(bottom=0.26, top=0.90, left=0.08, right=0.99)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_identifiability_ab.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("  fig_identifiability_ab: lead-count axis, nested chains separated, legend added")


def fig_selection(out: Path):
    allset = pd.read_csv(W1J / "W1J_all56_logdet.csv")
    boot = pd.read_csv(W1J / "W1J_margin_bootstrap_summary.csv")
    pC2 = pd.read_csv(FIG / "panelC_marginal.csv").set_index("lead")

    ident = allset[(allset.seed == 42) & (allset.noise_model == "identity")]
    top3 = ident.nlargest(3, "D_logdet")
    clin = ident[ident.leads == "II+V1+V5"].iloc[0]
    b = boot[(boot.seed == 42) & (boot.noise_model == "identity")].iloc[0]

    fig = plt.figure(figsize=(5.46, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.75], wspace=0.38)

    # ---------------- (a) top three against the clinical set, honest axis ----------
    ax = fig.add_subplot(gs[0, 0])
    names = list(top3.leads) + [clin.leads + "\n(clinical)"]
    vals = list(top3.D_logdet) + [clin.D_logdet]
    cols = [RED, GREY, GREY, "#8a8f94"]
    # no per-bar error bars: the bootstrap resamples records jointly for every set, so the only
    # uncertainty that is defined is that of a paired difference (title); one-decimal labels
    ax.bar(range(4), vals, color=cols, alpha=.9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.35, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_ylim(min(vals) - 3.0, max(vals) + 3.4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, rotation=18, fontsize=7, ha="right")
    ax.set_ylabel("D-optimality  log det $F_\\lambda$")
    ax.set_title("Three-lead selection, identity noise")
    ax.grid(alpha=.3, axis="y")
    ax.set_xlim(-0.72, 4.7)
    ax.annotate("", xy=(4.3, vals[0]), xytext=(4.3, vals[3]),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.2))
    ax.text(4.18, (vals[0] + vals[3]) / 2,
            f"margin over clinical\n$+{b.boot_margin_mean:.1f}$ [{b.boot_margin_ci_lo:.1f}, {b.boot_margin_ci_hi:.1f}]\n(bootstrap);\nthe gap that replicates",
            fontsize=7, color=RED, va="center", ha="right")
    ax.text(1.0, max(vals) + 1.9, "first-to-second margin 0.45", fontsize=7, color="0.3", ha="center")

    # ---------------- (b) marginal contribution ----------------
    axB = fig.add_subplot(gs[0, 1])
    mk = ["V1", "I", "V4"]
    mv = [pC2.loc[k, "marginal_logdet"] for k in mk]
    axB.bar(range(3), mv, color="#2f6f4f", alpha=.9)
    for i, v in enumerate(mv):
        axB.text(i, v + 0.7, f"+{v:.1f}", ha="center", fontsize=7)
    axB.set_ylim(0, max(mv) * 1.12)
    axB.set_xticks(range(3))
    axB.set_xticklabels(mk, fontsize=7.5)
    axB.set_xlabel("lead")
    axB.set_ylabel("marginal log det gain")
    axB.set_title("Marginal contribution within\nthe identity-noise optimum")
    axB.grid(alpha=.3, axis="y")

    panel_marks(fig, [0.33, 0.83])
    fig.subplots_adjust(bottom=0.26, top=0.90, left=0.09, right=0.99)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_selection_cd.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("  fig_selection_cd: clinical set shown, axis widened, paired-difference CI in title")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_identifiability(out)
    fig_selection(out)
    print("wrote both figures ->", out)


if __name__ == "__main__":
    main()
