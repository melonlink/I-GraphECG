"""§3.4 identifiability figure assets + optimal-set effective-rank supplement. No training.

Frozen 5 checkpoints (seed 42/1/2/3/4); fixed 400-sample subset (100/class).
- eff rank for lead sets {12, II+V1+V5, I+II+V5, II, I+V1+V4, I+V1+V5}: seed-42 + 5-seed mean±std.
- per-group identifiability scores (mean±std).
- seed-42: top-3 3-lead D-opt logdet + marginal contributions in I+V1+V4.
- One combined Figure (Panel A degradation+choice / B param-group / C optimal selection), PDF+PNG, + CSVs.

Run: "$PY" PUBLIC/scripts/82_fig_identifiability_assets.py
"""
from __future__ import annotations
import itertools, sys
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
from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.data.ptbxl_loader import CANONICAL_LEADS
from igraphecg.evaluation.identifiability import lead_rows, fim_metrics, identifiability_scores, group_scores, PARAM_GROUPS
from igraphecg.evaluation.round2_io import load_encoder_decoder, infer_theta

mpl.rcParams.update({"font.family": "Arial", "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "path",
                     "axes.titlesize": 9})
LAM = 1e-3
SEEDS = [42, 1, 2, 3, 4]
LI = {x: CANONICAL_LEADS.index(x) for x in ("I", "II", "V1", "V3", "V4", "V5")}
SETS = {"12": list(range(12)), "II+V1+V5": [LI["II"], LI["V1"], LI["V5"]],
        "I+II+V5": [LI["I"], LI["II"], LI["V5"]], "II": [LI["II"]],
        "I+V1+V4": [LI["I"], LI["V1"], LI["V4"]], "I+V1+V5": [LI["I"], LI["V1"], LI["V5"]],
        "I+V1+V3": [LI["I"], LI["V1"], LI["V3"]]}
GROUPS = list(PARAM_GROUPS.keys())


def ckpt_of(s):
    """The locked checkpoint for this seed: no branch, no fallback, sha256-checked."""
    return repro.lineage.checkpoint(s)


def fixed_subset(label, fold, per_class=100, seed=42):
    rng = np.random.default_rng(seed); idx = []
    for c in range(len(TARGET_CLASSES)):
        pool = np.where((label == c) & (fold == 10))[0]
        idx.append(pool if len(pool) <= per_class else rng.choice(pool, per_class, replace=False))
    return np.concatenate(idx)


def logdet(F): return float(np.linalg.slogdet(F + LAM * np.eye(F.shape[0]))[1])


def main(out=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(out) if out else repro.outdir("v1fix/runs/fig34")
    out.mkdir(parents=True, exist_ok=True)
    data = load_processed(ROOT / "data/processed/ptbxl_medianbeat_clean_100hz.npz")
    signals = data["signal_12lead"].astype(np.float32); n_t = signals.shape[2]
    label = data["label"].astype(int); fold = data["fold"].astype(int)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals); sub = fixed_subset(label, fold)
    from torch.func import jacfwd
    rows_cache = {k: torch.tensor(lead_rows(v, n_t), device=dev) for k, v in SETS.items()}

    eff = {k: [] for k in SETS}; gsc = {g: [] for g in GROUPS}
    seed42 = {}
    for s in SEEDS:
        enc, dec = load_encoder_decoder(ckpt_of(s), n_t, 3, dev)
        theta, _ = infer_theta(enc, dec, scaled[sub], device=dev)
        th = torch.from_numpy(theta).to(dev); r = dec.pspace.radius.to(dev).view(1, -1)
        def f(t):
            y, _ = dec.forward(t.unsqueeze(0)); return y[0].reshape(-1)
        J = torch.stack([jacfwd(f)(th[i]) * r for i in range(th.shape[0])]).detach()
        Fmat = {}
        for k, rr in rows_cache.items():
            Fk = (torch.einsum("nrp,nrq->pq", J[:, rr, :], J[:, rr, :]) / J.shape[0]).cpu().numpy()
            Fmat[k] = Fk; eff[k].append(fim_metrics(Fk)["effective_rank"])
        sc, _ = identifiability_scores(Fmat["12"], lam=LAM); gs = group_scores(sc)
        for g in GROUPS:
            gsc[g].append(gs[g])
        if s == 42:
            seed42["top3"] = {k: logdet(Fmat[k]) for k in ("I+V1+V4", "I+V1+V5", "I+V1+V3")}
            full = logdet(Fmat["I+V1+V4"])
            marg = {}
            for ld, nm in [(LI["I"], "I"), (LI["V1"], "V1"), (LI["V4"], "V4")]:
                rest = [x for x in SETS["I+V1+V4"] if x != ld]
                Fp = (torch.einsum("nrp,nrq->pq", J[:, torch.tensor(lead_rows(rest, n_t), device=dev), :],
                                   J[:, torch.tensor(lead_rows(rest, n_t), device=dev), :]) / J.shape[0]).cpu().numpy()
                marg[nm] = full - logdet(Fp)
            seed42["marg"] = marg
        print(f"[fig34] seed {s} eff: " + " ".join(f"{k}={eff[k][-1]:.2f}" for k in ("12", "II+V1+V5", "I+V1+V4", "II")))

    def ms(a): a = np.array(a); return a[0], a.mean(), a.std(ddof=1)
    # ---- supplement table: I+V1+V4 / I+V1+V5 eff rank ----
    supp = []
    for k in SETS:
        s42, mu, sd = ms(eff[k])
        supp.append({"lead_set": k, "seed42": round(s42, 3), "mean": round(mu, 3), "std": round(sd, 3)})
    pd.DataFrame(supp).to_csv(out / "leadset_effrank_supp.csv", index=False)
    iv4 = [r for r in supp if r["lead_set"] == "I+V1+V4"][0]
    iv5 = [r for r in supp if r["lead_set"] == "I+V1+V5"][0]
    print(f"[fig34] OPTRANK I+V1+V4 seed42={iv4['seed42']} mean±std={iv4['mean']}±{iv4['std']} | "
          f"I+V1+V5 seed42={iv5['seed42']} mean±std={iv5['mean']}±{iv5['std']}")

    # ---- panel data CSVs ----
    pA = pd.DataFrame([{"lead_set": k, "mean": np.mean(eff[k]), "std": np.std(eff[k], ddof=1)}
                       for k in ("12", "II+V1+V5", "I+II+V5", "II", "I+V1+V4")])
    pA.to_csv(out / "panelA_degradation.csv", index=False)
    pB = pd.DataFrame([{"group": g, "mean": np.mean(gsc[g]), "std": np.std(gsc[g], ddof=1)} for g in GROUPS])
    pB.to_csv(out / "panelB_group_identifiability.csv", index=False)
    pC2 = pd.DataFrame([{"lead": k, "marginal_logdet": v} for k, v in seed42["marg"].items()])
    pC2.to_csv(out / "panelC_marginal.csv", index=False)

    # ============ combined figure ============
    fig = plt.figure(figsize=(13.5, 4.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.0, 0.85, 0.85], wspace=0.42)
    # Panel A
    axA = fig.add_subplot(gs[0, 0])
    seq = ["12", "II+V1+V5", "I+II+V5", "II"]; xseq = [0, 1, 2, 3]
    mu = [np.mean(eff[k]) for k in seq]; sd = [np.std(eff[k], ddof=1) for k in seq]
    axA.errorbar(xseq, mu, yerr=sd, fmt="o-", color="#4c72b0", capsize=4, label="lead-count path (clinical-style)")
    o_mu, o_sd = np.mean(eff["I+V1+V4"]), np.std(eff["I+V1+V4"], ddof=1)
    axA.errorbar([1], [o_mu], yerr=[o_sd], fmt="*", color="#c0392b", ms=15, capsize=4,
                 label="optimal 3-lead I+V1+V4")
    axA.annotate("same #leads,\nbetter choice", xy=(1, o_mu), xytext=(1.5, o_mu + 0.7),
                 fontsize=7.5, color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b"))
    axA.set_xticks(xseq); axA.set_xticklabels(["12", "II+V1+V5", "I+II+V5", "II"], fontsize=8)
    axA.set_ylabel("FIM effective rank"); axA.legend(fontsize=6.5, frameon=False, loc="lower left")
    axA.set_title("(A) lead count = ceiling, lead choice = how near\n(mean±std, n=5 seeds)"); axA.grid(alpha=.3, axis="y")
    # Panel B
    axB = fig.add_subplot(gs[0, 1])
    gm = [np.mean(gsc[g]) for g in GROUPS]; gsd = [np.std(gsc[g], ddof=1) for g in GROUPS]
    cols = ["#c0392b" if g == "alpha_ST" else ("#55a868" if g in ("delta_root", "edge_prox", "edge_leaf") else "#4c72b0") for g in GROUPS]
    axB.bar(range(len(GROUPS)), gm, yerr=gsd, color=cols, capsize=2, alpha=.85)
    axB.set_yscale("log"); axB.set_xticks(range(len(GROUPS)))
    axB.set_xticklabels([g.replace("global_gain", "g").replace("alpha_ST", "α_ST") for g in GROUPS], rotation=45, ha="right", fontsize=6.5)
    axB.set_ylabel("identifiability score (log)"); axB.set_title("(B) parameter-group identifiability\nα_ST lowest (≈1 order below)"); axB.grid(alpha=.3, axis="y", which="both")
    # Panel C-left top3
    axC1 = fig.add_subplot(gs[0, 2])
    t3 = ["I+V1+V4", "I+V1+V5", "I+V1+V3"]; lv = [seed42["top3"][k] for k in t3]
    axC1.bar(range(3), lv, color=["#c0392b", "#9aa", "#9aa"])
    for i, v in enumerate(lv): axC1.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=7)
    axC1.set_ylim(min(lv) - 1.5, max(lv) + 1.2); axC1.set_xticks(range(3)); axC1.set_xticklabels(t3, rotation=20, fontsize=7)
    axC1.set_ylabel("D-opt log det F"); axC1.set_title(f"(C) optimal selection\n1st−2nd margin={lv[0]-lv[1]:.2f} ≫ boot~0.35"); axC1.grid(alpha=.3, axis="y")
    # Panel C-right marginal
    axC2 = fig.add_subplot(gs[0, 3])
    mk = ["V1", "I", "V4"]; mv = [seed42["marg"][k] for k in mk]
    axC2.bar(range(3), mv, color="#2f6f4f")
    for i, v in enumerate(mv): axC2.text(i, v + 0.6, f"+{v:.1f}", ha="center", fontsize=7)
    axC2.set_xticks(range(3)); axC2.set_xticklabels(mk, fontsize=8); axC2.set_ylabel("marginal log det gain")
    axC2.set_title("each lead's unique\ncontribution (low redundancy)"); axC2.grid(alpha=.3, axis="y")

    for ext in ("pdf", "png"):
        # No creation date in the PDF, so the registered file is byte-identical on every run.
        fig.savefig(out / f"figure_identifiability.{ext}", dpi=300, bbox_inches="tight", facecolor="white",
                    metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)
    print(f"[fig34] DONE -> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="output directory; default = the published run under the output root")
    main(out=ap.parse_args().out)
