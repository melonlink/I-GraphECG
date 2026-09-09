"""The manuscript's numbers against the registered outputs.

Every check below recomputes a quantity from the shipped tables under ``outputs/`` (no raw data,
no GPU) and asserts that the manuscript states it in the printed form: the main text and the
supplement are searched for the exact rounded string. A failure therefore means either that a
number in the paper no longer matches what the code produces, or that the paper's rounding
convention changed. The list is curated, not exhaustive: it covers every headline number of the
Results, the Discussion and the Conclusions, and the supplementary tables that were typed by hand
(S2 lower block, S14, S15) rather than generated from a file.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
STATS = OUT / "runs" / "v7rev_stats"
SEEDS = [42, 1, 2, 3, 4]
MAIN = ROOT / "paper" / "1_Manuscript" / "main.tex"
SUPP = ROOT / "paper" / "2_Supplementary" / "supplementary.tex"

pytestmark = pytest.mark.skipif(not (MAIN.exists() and STATS.exists()), reason="paper or outputs absent")


def _tex() -> str:
    t = MAIN.read_text(encoding="utf-8") + "\n" + SUPP.read_text(encoding="utf-8")
    t = t.replace("~", " ").replace("\\,", "").replace("{,}", ",")
    return re.sub(r"\s+", " ", t)


TEX = _tex() if MAIN.exists() else ""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("~", " ").replace("\\,", "").replace("{,}", ","))


def stated(s: str) -> bool:
    return _norm(s) in TEX


def fmt(x, nd):
    return f"{x:.{nd}f}"


def cnt(x) -> str:
    """Integer count as the manuscript writes it: a thousands separator from four digits on."""
    return f"{int(x):,}"


def pm(vals, nd_mean, nd_sd=None):
    v = np.asarray(vals, float)
    return f"{v.mean():.{nd_mean}f}\\pm{v.std(ddof=1):.{nd_sd or nd_mean}f}"


def recon(seed):
    d = pd.read_csv(OUT / "runs" / f"d1s_r3_s{seed}" / "tables" / "reconstruction_metrics.csv")
    return d[(d.split == "test") & (d.space == "scaled")].iloc[0], d[(d.split == "test") & (d.space == "physical_mV")].iloc[0]


def cliffs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(((a[:, None] > b[None, :]).mean() - (a[:, None] < b[None, :]).mean()))


def checks():
    """(label, string the manuscript must contain) pairs, computed from the outputs."""
    c = []

    # ---- Section 2.1: data
    split = pd.read_csv(OUT / "tables" / "data_split_summary.csv", index_col=0)
    for row, name in (("train", "train"), ("val", "validation"), ("test", "test")):
        r = split.loc[row]
        c.append((f"split {name}", f"{name} & {cnt(r.NORM)} & {cnt(r.MI)} & {cnt(r.STTC)} & {cnt(r.CD)} & {cnt(r.total)}"))
    xgb = pd.read_csv(OUT / "tables" / "baseline_features_metrics.csv")
    cnn = pd.read_csv(OUT / "tables" / "baseline_cnn_metrics.csv")
    c.append(("XGBoost ceiling", fmt(xgb[xgb.split == "test"].macro_auroc.iloc[0], 4)))
    c.append(("ResNet ceiling", fmt(cnn[cnn.split == "test"].macro_auroc.iloc[0], 4)))
    ret = pd.read_csv(STATS / "w1e_retention_by_class.csv").set_index("class")
    c.append(("NORM retention", fmt(ret.loc["NORM", "retention_pct"], 1) + "\\%"))
    c.append(("MI retention", fmt(ret.loc["MI", "retention_pct"], 1) + "\\%"))
    c.append(("NORM:MI full", fmt(ret.loc["NORM", "carrying"] / ret.loc["MI", "carrying"], 2)))
    c.append(("NORM:MI subset", fmt(ret.loc["NORM", "retained"] / ret.loc["MI", "retained"], 2)))
    c.append(("clean CD count", f"{int(ret.loc['CD', 'retained']):,}"))
    pool = json.load(open(STATS / "W1F_multilabel_headline.json"))["pool"]
    c.append(("records total", f"{pool['total_ptbxl']:,}"))
    c.append(("clean subset", f"{pool['clean_single_label_subset']:,}"))
    c.append(("multi-label pool", f"{pool['ge1_superclass']:,}"))
    c.append(("removed by filter", f"{pool['total_ptbxl'] - pool['clean_single_label_subset']:,}"))

    # ---- Section 3.1 / Table 2: reconstruction, locked value and five seeds
    sc, ph = zip(*[recon(s) for s in SEEDS])
    s42s, s42p = sc[0], ph[0]
    c.append(("NRMSE", f"{fmt(s42s.median_nrmse, 3)}$ (${pm([r.median_nrmse for r in sc], 3)}"))
    c.append(("corr", f"{fmt(s42s.median_corr, 3)}$ (${pm([r.median_corr for r in sc], 3)}"))
    c.append(("QRS NRMSE", f"{fmt(s42s.qrs_nrmse, 3)}$ (${pm([r.qrs_nrmse for r in sc], 3)}"))
    c.append(("ST MAE mV", f"{fmt(s42p.st_mean_abs_err, 3)}$ mV (${pm([r.st_mean_abs_err for r in ph], 4)}"))
    c.append(("boundary rate", f"{fmt(s42s.boundary_rate, 3)}$ (${pm([r.boundary_rate for r in sc], 3)}"))
    c.append(("Table 2 ST window", f"{fmt(s42s.st_nrmse, 3)}$ & ${pm([r.st_nrmse for r in sc], 3)}"))
    c.append(("Table 2 T window", f"{fmt(s42s.t_nrmse, 3)}$ & ${pm([r.t_nrmse for r in sc], 3)}"))
    win = pd.read_csv(STATS / "W1A_window_context.csv")
    pw = win[win.window == "P"].set_index("seed")
    c.append(("Table 2 P window", f"{fmt(pw.loc[42, 'median_nrmse'], 3)}$ & ${pm(pw.median_nrmse, 3)}"))
    c.append(("P window text", f"{fmt(pw.loc[42, 'median_nrmse'], 3)}$ (${pm(pw.median_nrmse, 2)}"))
    c.append(("P windowed corr", f"windowed correlation ${fmt(pw.median_windowed_corr.mean(), 2)}$"))
    ctl = json.load(open(STATS / "W1A_controls.json"))
    worst = [min(ctl["per_seed"][str(s)]["lead"], key=lambda k: ctl["per_seed"][str(s)]["lead"][k][0]) for s in SEEDS]
    assert set(worst) == {"aVF"}, worst
    c.append(("aVF worst lead", f"aVF least (${fmt(ctl['per_seed']['42']['lead']['aVF'][0], 2)}$"))
    best = max(ctl["per_seed"]["42"]["lead"].items(), key=lambda kv: kv[1][0])
    c.append(("best lead corr", f"up to ${fmt(best[1][0], 2)}$"))
    ora = pd.read_csv(OUT / "runs" / "v1fix_oracle_fold10" / "tables" / "oracle_fold10_metrics.csv")
    ora = ora[ora.space == "scaled"].set_index("arm").median_nrmse
    c.append(("oracle fixed H (centre)", f"median scaled NRMSE of ${fmt(ora['oracle_fixed_H_center'], 3)}$"))
    c.append(("oracle refit H", f"reaches ${fmt(ora['oracle_refit_H'], 3)}$"))
    c.append(("amortization gap", f"within ${fmt(ora['encoder'] - ora['oracle_fixed_H_center'], 3)}$ NRMSE"))
    tr = pd.read_csv(OUT / "runs" / "v1fix_oracle_fold10" / "tables" / "oracle_fold10_leadfield_transfer.csv")
    col = [x for x in tr.columns if "nrmse" in x.lower()][0]
    vals = sorted(tr[col].astype(float).round(3).tolist())
    c.append(("lead-field transfer", f"(${fmt(vals[1], 3)}$ against ${fmt(vals[-1], 3)}$"))

    # ---- Section 3.2: propagation (W1B, W1C)
    b = json.load(open(STATS / "W1B_headline.json"))
    d = b["TEST1_paired_deltas"]["C5_obs_minus_C5|macro_auroc"]
    c.append(("substitution oracle gain", f"${fmt(d['mean'], 4)}\\pm{fmt(d['sd_ddof1'], 4)}$"))
    r5, ro = b["TEST1_variants"]["C5"]["recall"], b["TEST1_variants"]["C5_obs"]["recall"]
    c.append(("STTC recall swap", f"${fmt(r5['STTC']['mean'], 3)}\\to{fmt(ro['STTC']['mean'], 3)}$"))
    c.append(("MI recall swap", f"${fmt(r5['MI']['mean'], 3)}\\to{fmt(ro['MI']['mean'], 3)}$"))
    mi = b["TEST2_within_class"]["MI"]
    c.append(("MI recall Q1->Q4", f"${fmt(mi['recall_by_quartile_mean'][0], 3)}$ in the best-reconstructed NRMSE quartile to ${fmt(mi['recall_by_quartile_mean'][3], 3)}$"))
    c.append(("MI quartile delta", f"\\Delta=+{fmt(mi['Q4_minus_Q1_recall']['mean'], 3)}\\pm{fmt(mi['Q4_minus_Q1_recall']['sd_ddof1'], 3)}$"))
    fid = pd.read_csv(STATS / "W1C_descriptor_fidelity.csv").groupby(["pair", "family"]).pearson_r.mean().reset_index()
    kind = fid.family.map(lambda f: "timing/shape" if f in ("Tptime", "Tasym", "Twidth") else "polarity" if f.startswith("T_sign") else "amplitude")
    n_amp, n_amp_ok = int((kind == "amplitude").sum()), int(((kind == "amplitude") & (fid.pearson_r >= 0.80)).sum())
    n_ts, n_ts_lo = int((kind == "timing/shape").sum()), int(((kind == "timing/shape") & (fid.pearson_r < 0.50)).sum())
    assert n_ts == n_ts_lo
    c.append(("fidelity amplitude count", f"{n_amp_ok} of the {n_amp} amplitude, area and slope descriptors"))
    c.append(("fidelity timing count", f"all {n_ts} timing and shape descriptors"))

    # ---- Section 3.3 / Table 3: classification
    g = pd.read_csv(STATS / "w1i_group_auroc_five_seed.csv").set_index("group")
    summ = pd.read_csv(OUT / "runs" / "v1fix_r3_s42" / "tables" / "round3_classification_summary.csv")
    summ = summ[summ.classifier == "xgboost"].set_index("feature_group")
    for grp, tag in (("C0_theta", "C0"), ("C2_theta_rep", "C2"), ("C3_theta_st", "C3"), ("C4_theta_stab", "C4"),
                     ("C5_theta_rep_st_stab", "C5"), ("C6_compact_heuristic", "C6")):
        s = summ.loc[grp]
        c.append((f"Table 3 {tag}", f"${fmt(s.macro_auroc, 4)}$ & $[{fmt(s.auroc_ci_lo, 3)},{fmt(s.auroc_ci_hi, 3)}]$ & ${fmt(g.loc[grp, 'mean'], 4)}\\pm{fmt(g.loc[grp, 'std_ddof1'], 4)}$ & ${fmt(s.macro_f1, 3)}$"))
    theta = g.loc[[x for x in g.index if not x.startswith("C6")]]
    c.append(("theta-group seed-42 range", f"${fmt(theta.seed42.min(), 4)}$--${fmt(theta.seed42.max(), 4)}$"))
    c.append(("theta-group five-seed range", f"${fmt(theta['mean'].min(), 4)}$--${fmt(theta['mean'].max(), 4)}$"))
    h = pd.read_csv(STATS / "w1h_paired_ablation_fdr.csv")
    c56 = h[(h.group_a == "C5_theta_rep_st_stab") & (h.group_b == "C6_compact_heuristic")].iloc[0]
    c.append(("C5-C6 contrast", f"${fmt(c56.delta, 4)}$ ($95\\%$ CI $[{fmt(c56.ci_lo, 4)},{fmt(c56.ci_hi, 4)}]$"))
    p_theta = h[~h.group_a.str.startswith("C6") & ~h.group_b.str.startswith("C6")].p_raw.min()
    c.append(("smallest theta p", f"smallest uncorrected $p$ is ${fmt(p_theta, 2)}$"))
    ops = pd.read_csv(STATS / "w1d_mi_operating_points.csv").set_index("rule")
    a = ops.loc["argmax (paper)"]
    c.append(("per-class recall", f"NORM ${fmt(a.recall_NORM, 2)}$, STTC ${fmt(a.recall_STTC, 2)}$, CD ${fmt(a.recall_CD, 2)}$, and MI only ${fmt(a.recall_MI, 2)}$ (balanced accuracy ${fmt(a.balanced_accuracy, 3)}$)"))
    c.append(("MI->NORM count", f"Of the {int(a.MI_total)} MI test records, {int(a.MI_to_NORM)} (about ${round(100 * a.MI_to_NORM / a.MI_total)}\\%$) were assigned to NORM"))
    c.append(("MI precision", f"precision ${fmt(a.precision_MI, 2)}$"))
    mc, sd = pd.read_csv(STATS / "w1i_mean_confusion.csv", index_col=0), pd.read_csv(STATS / "w1i_sd_confusion.csv", index_col=0)
    c.append(("MI->NORM five seeds", f"${int(a.MI_to_NORM)}$ against ${fmt(mc.loc['MI', 'NORM'], 1)}\\pm{fmt(sd.loc['MI', 'NORM'], 1)}$"))
    pc = {m: pd.read_csv(OUT / "tables" / f"baseline_{m}_metrics_perclass.csv") for m in ("cnn", "features")}
    pc = {m: v[v.split == "test"].set_index("class").auroc for m, v in pc.items()}
    others = pd.concat([pc["cnn"].drop("MI"), pc["features"].drop("MI")])
    c.append(("baseline MI AUROC", f"per-class AUROC ${fmt(pc['cnn']['MI'], 3)}$ for ResNet1D and ${fmt(pc['features']['MI'], 3)}$ for the hand-crafted XGBoost, against ${fmt(others.min(), 3)}$--${fmt(others.max(), 3)}$"))
    p5 = pd.read_csv(OUT / "runs" / "v1fix_r3_s42" / "tables" / "round3_classification_perclass.csv")
    p0 = p5[p5.feature_group == "C0_theta"].set_index("class").sensitivity
    c.append(("internal C0 recalls (Table 7)", " / ".join(fmt(p0[k], 2) for k in ("NORM", "STTC", "CD"))))
    p5 = p5[p5.feature_group == "C5_theta_rep_st_stab"].set_index("class").auroc
    gaps = (pc["cnn"] - p5)
    c.append(("C5 gap to ResNet", f"largest on MI (${fmt(gaps['MI'], 3)}$ against ${fmt(gaps.drop('MI').min(), 3)}$--${fmt(gaps.drop('MI').max(), 3)}$"))

    # ---- Section 3.4: decision rules (Table S7)
    def rule(name):
        return ops.loc[name]
    pr, cw, t9 = rule("prior-corrected argmax (full-PTB-XL prior)"), rule("inverse-frequency class weights"), rule("MI threshold @fold-9 sens>=0.90")
    c.append(("prior correction", f"from ${fmt(a.recall_MI, 3)}$ $[{fmt(a.recall_MI_lo, 3)},{fmt(a.recall_MI_hi, 3)}]$ to ${fmt(pr.recall_MI, 3)}$ $[{fmt(pr.recall_MI_lo, 3)},{fmt(pr.recall_MI_hi, 3)}]$"))
    c.append(("class weights", f"give ${fmt(cw.recall_MI, 3)}$ $[{fmt(cw.recall_MI_lo, 3)},{fmt(cw.recall_MI_hi, 3)}]$"))
    c.append(("threshold 0.90", f"recall is ${fmt(t9.recall_MI, 3)}$ $[{fmt(t9.recall_MI_lo, 3)},{fmt(t9.recall_MI_hi, 3)}]$ and MI$\\to$NORM errors fall from ${int(a.MI_to_NORM)}$ to ${int(t9.MI_to_NORM)}$ of ${int(a.MI_total)}$, at MI precision ${fmt(t9.precision_MI, 3)}$ and NORM recall ${fmt(t9.recall_NORM, 3)}$"))
    c.append(("AUROC unmoved", f"(${fmt(a.macro_auroc, 4)}$, ${fmt(pr.macro_auroc, 4)}$, ${fmt(cw.macro_auroc, 4)}$)"))
    c.append(("MI AUROC lowest", f"lowest per-class AUROC (${fmt(p5['MI'], 2)}$"))
    c.append(("conclusions MI numbers", f"MI recall is ${fmt(a.recall_MI, 2)}$ and ${round(100 * a.MI_to_NORM / a.MI_total)}\\%$ of MI records"))
    c.append(("conclusions remedies", f"(to ${fmt(pr.recall_MI, 2)}$ and ${fmt(t9.recall_MI, 2)}$)"))

    # ---- Section 3.5: interpretation (five seeds, fold 10)
    cd = pd.read_csv(OUT / "v1fix" / "runs" / "cd_descriptors" / "cd_candidates_byseed.csv").set_index("candidate")
    c.append(("CD QRS duration delta", f"\\delta=+{fmt(cd.loc['QRS_dur_recon', 'mean'], 2)}\\pm{fmt(cd.loc['QRS_dur_recon', 'std'], 2)}$"))
    c.append(("CD real-beat delta", f"real-beat $\\delta=+{fmt(cd.loc['QRS_dur_real_gold', 'mean'], 3)}$"))
    feats = {}
    for s in SEEDS:
        f = pd.read_csv(OUT / "runs" / f"v1fix_r3_s{s}" / "tables" / "round3_features.csv")
        feats[s] = f[f.fold == 10]
    def delta(col, cls):
        return [cliffs(feats[s][feats[s].label_name == cls][col], feats[s][feats[s].label_name == "NORM"][col]) for s in SEEDS]
    disc = delta("obs_T_sign_discordance_rate", "STTC")
    assert np.ptp(disc) < 1e-9
    c.append(("STTC discordance", f"\\delta=+{fmt(disc[0], 3)}$"))
    c.append(("STTC T-peak", f"\\delta={fmt(delta('obs_Tpeak_maxabs', 'STTC')[0], 3)}$"))
    mi_disc = delta("T_sign_discordance_rate", "MI")
    c.append(("MI recon discordance", f"\\delta=+{fmt(np.mean(mi_disc), 2)}\\pm{fmt(np.std(mi_disc, ddof=1), 2)}$"))
    st = delta("ST_projected_max_abs", "MI")
    c.append(("MI projected ST", f"\\delta={fmt(np.mean(st), 2)}\\pm{fmt(np.std(st, ddof=1), 2)}$"))
    assert max(st) <= 0
    c.append(("MI projected ST range", f"from ${fmt(min(abs(x) for x in st), 2)}$ to ${fmt(max(abs(x) for x in st), 2)}$"))

    # ---- Section 3.6: identifiability and lead selection
    # the five-seed effective ranks of Table 7 / Figure 6a evaluate all checkpoints on the locked
    # model's 400-record subset (scripts/82, v1fix/runs/fig34); the locked values must also agree
    # with the per-run table of scripts/43
    er42 = pd.read_csv(OUT / "runs" / "v1fix_r3_s42" / "tables" / "fim_effective_rank_by_leadset.csv").set_index("lead_set").effective_rank
    fig34 = pd.read_csv(OUT / "v1fix" / "runs" / "fig34" / "leadset_effrank_supp.csv").set_index("lead_set")
    for key, name in (("L12", "12"), ("L3a_II_V1_V5", "II+V1+V5"), ("L3b_I_II_V5", "I+II+V5"), ("L1_II", "II")):
        assert abs(er42[key] - fig34.loc[name, "seed42"]) < 5e-4, (key, er42[key], fig34.loc[name, "seed42"])
        c.append((f"effective rank {name}", f"${fmt(er42[key], 3)}$ (${fmt(fig34.loc[name, 'mean'], 2)}\\pm{fmt(fig34.loc[name, 'std'], 2)}$)"))
    c.append(("I+V1+V4 effective rank", f"${fmt(fig34.loc['I+V1+V4', 'seed42'], 2)}$ (${fmt(fig34.loc['I+V1+V4', 'mean'], 2)}\\pm{fmt(fig34.loc['I+V1+V4', 'std'], 2)}$) versus ${fmt(fig34.loc['II+V1+V5', 'mean'], 2)}\\pm{fmt(fig34.loc['II+V1+V5', 'std'], 2)}$"))
    W = json.load(open(STATS / "W1G_headline.json"))
    tiers = W["tier_sizes_by_rule"]["R2_min_across_seeds"]
    c.append(("tier sizes", f"& {tiers['A_strongly_identifiable']} & "))
    c.append(("tier B size", f"& {tiers['B_weakly_identifiable']} & "))
    c.append(("Tier A count text", f"Only ${tiers['A_strongly_identifiable']}$ of $47$ parameters are strongly identifiable"))
    c.append(("group-rule tier A", f"admitting ${W['tier_sizes_by_rule']['R1_group_mean_cv (the flawed rule)']['A_strongly_identifiable']}$ parameters"))
    pt = pd.read_csv(STATS / "W1G_parameter_table.csv")
    col = "one_sigma_pct_of_admissible_range_5seed_mean"   # the column Table 6 prints
    for tier, letter in (("A_strongly_identifiable", "A"), ("B_weakly_identifiable", "B"), ("C_primarily_regularized", "C")):
        v = pt[pt.tier_recommended == tier][col]
        c.append((f"tier {letter} sigma range", f"${fmt(v.min(), 1) if letter != 'C' else str(round(v.min()))}$--${fmt(v.max(), 1) if letter != 'C' else str(round(v.max()))}$"))
    gs = W["group_level_five_seed_mean_score"]
    c.append(("leaf group score", f"highest five-seed mean score (${fmt(gs['edge_leaf'], 1)}$)"))
    g42 = pd.read_csv(OUT / "runs" / "v1fix_r3_s42" / "tables" / "identifiability_by_param_group.csv").set_index("group").mean_score
    c.append(("leaf score locked", f"(${fmt(g42['edge_leaf'], 1)}$ at the locked seed)"))
    others = [v for k, v in gs.items() if k != "alpha_ST"]
    c.append(("alpha_ST group score", f"group mean ${fmt(gs['alpha_ST'], 2)}$ against ${round(min(others))}$--${round(max(others))}$"))
    c.append(("max per-parameter CV", f"reach ${round(100 * W['max_per_parameter_cv']['cv'])}\\%$"))
    en = {s: pd.read_csv(OUT / "runs" / f"v1fix_phase1_s{s}" / "tables" / "unobservable_subspace_group_energy.csv").set_index("param_group").unobs_energy_bottom5["alpha_ST"] for s in SEEDS}
    c.append(("ST group energy", f"group energy ${fmt(en[42], 2)}$ at the locked seed; ${pm(list(en.values()), 2)}$"))
    sv = np.loadtxt(OUT / "runs" / "v1fix_phase1_s42" / "tables" / "leadfield_singular_values.csv", skiprows=1)
    c.append(("singular values", f"$[{fmt(sv[0], 3)},{fmt(sv[1], 3)},{fmt(sv[2], 3)},0,\\dots]$"))
    stg = [pd.read_csv(OUT / "runs" / f"v1fix_phase1_s{s}" / "tables" / "st_mode_gain.csv").transmitted_gain.iloc[0] for s in SEEDS]
    c.append(("ST-mode transmission gain", f"transmission gain of only ${fmt(stg[0], 3)}$ (${pm(stg, 2)}$ across seeds)"))
    crb = pd.read_csv(OUT / "runs" / "v1fix_phase1_s42" / "tables" / "crb_by_group_by_leadset.csv").set_index("lead_set").CRB_alpha_ST
    crb8 = pd.read_csv(OUT / "runs" / "v1fix_ls8_s42" / "tables" / "crb_optimal_vs_clinical.csv").set_index("set").CRB_alpha_ST
    c.append(("CRB ladder", f"from ${fmt(crb['L12'], 2)}$ at 12 leads (${fmt(crb8['all-8 independent'], 2)}$ on the eight independent leads alone; under $\\Sigma=I$ the four derived leads count as additional observations of I and II) to ${fmt(crb['II+V1+V5'], 2)}$ (II+V1+V5), ${fmt(crb['I+II+V5'], 2)}$ (I+II+V5), and ${fmt(crb['II'], 2)}$ for lead II alone"))
    c.append(("Table 7 CRB", f"& ${fmt(crb['L12'], 2)}$ &"))
    ls8 = OUT / "runs" / "v1fix_ls8_s42" / "tables"
    best = pd.read_csv(ls8 / "best_sets_DAE_k234.csv").set_index(["k", "criterion"]).best_set
    assert best.loc[(3, "D")] == best.loc[(3, "A")] == "I+V1+V4" and best.loc[(3, "E")] == "V1+V2+V5"
    o3 = pd.read_csv(ls8 / "optimal3_vs_clinical.csv").set_index("leads").D_logdet
    c.append(("D-optimal scores", f"${fmt(o3['I+V1+V4'], 2)}$ versus ${fmt(o3['II+V1+V5'], 2)}$"))
    gap = pd.read_csv(ls8 / "bootstrap_gap_summary.csv").iloc[0]
    c.append(("bootstrap margin", f"margin of $+{fmt(gap.bootstrap_gap_mean, 2)}$"))
    c.append(("bootstrap CI", f"$[{fmt(gap.ci_lo, 2)},{fmt(gap.ci_hi, 2)}]$"))
    c.append(("point-estimate gap", f"point estimates is ${fmt(gap.gap_lambda, 2)}$"))
    c.append(("floored gap", f"remained ${fmt(gap.gap_eig_floored, 2)}$"))
    wins = pd.read_csv(ls8 / "bootstrap_top_winning_sets.csv").set_index("set").win_freq
    c.append(("bootstrap wins", f"selected in ${round(100 * wins['I+V1+V4'])}\\%$ of bootstrap resamples"))
    byc = {s: pd.read_csv(OUT / "runs" / f"v1fix_byclass_s{s}" / "tables" / "byclass_optimal_DAE.csv") for s in [42]}
    assert set(byc[42].D_opt) == {"I+V1+V4"}
    mg = pd.read_csv(OUT / "runs" / "v1fix_byclass_s42" / "tables" / "optimal_set_marginal_contrib.csv").set_index("lead_dropped").marginal_gain_of_lead
    c.append(("marginal contributions", f"(V1 $+{fmt(mg['V1'], 1)}$, V4 $+{fmt(mg['V4'], 1)}$, I $+{fmt(mg['I'], 1)}$)"))
    lj = pd.read_csv(STATS / "W1J_leadsets_by_noise_model.csv")
    idn = lj[lj.noise_model == "identity"]
    second = -idn[idn["rank"] == 2].set_index("seed").delta_logdet_vs_rank1
    c.append(("first-to-second margin", f"(${fmt(second[42], 2)}$ at seed 42; ${fmt(second.min(), 2)}$--${fmt(second.max(), 2)}$"))
    top = idn[idn["rank"] == 1].set_index("seed").leads
    assert all(t.startswith("I+V1+") for t in top)
    c.append(("V4/V5 alternation", f"V4 ({['two', 'three'][int((top == 'I+V1+V5').sum() == 2)]} seeds) and V5 ({['two', 'three'][int((top == 'I+V1+V5').sum() == 3)]})"))
    emp = lj[lj.noise_model == "empirical_sigma_inv"]
    etop = emp[emp["rank"] == 1].set_index("seed").leads
    c.append(("empirical optima", "the optima become " + ", ".join(etop[s] for s in [42, 1, 2, 3]) + f" and {etop[4]}"))
    clin_rank = lj[lj.leads.str.contains("clinical")].set_index(["seed", "noise_model"])["rank"]
    c.append(("clinical rank identity", f"ranks ${int(clin_rank[(42, 'identity')])}$nd of $56$ under identity noise and ${int(clin_rank[(42, 'empirical_sigma_inv')])}$th of $56$"))
    c.append(("clinical best rank", f"never better than ${int(clin_rank.xs('identity', level=1).min())}$th or ${int(clin_rank.xs('empirical_sigma_inv', level=1).min())}$th at any seed"))
    mb = pd.read_csv(STATS / "W1J_margin_bootstrap_summary.csv").set_index(["seed", "noise_model"])
    c.append(("empirical margin", f"and $+{fmt(mb.loc[(42, 'empirical_sigma_inv'), 'margin_logdet'], 2)}$ $\\log\\det$ units"))
    dfc = pd.read_csv(STATS / "W1R_design_fold_check.csv")
    same = all(g.D_optimal.nunique() == 1 and g.A_optimal.nunique() == 1 for _, g in dfc.groupby("seed"))
    assert same, "W1R: the D- or A-optimal set depends on the design fold, contradicting Section 3.6"
    spread = dfc.groupby("seed").D_margin_over_clinical.agg(lambda x: x.max() - x.min()).max()
    c.append(("design-fold margin spread", f"moves by at most ${fmt(spread, 2)}$ $\\log\\det$ units"))
    ch = json.load(open(STATS / "W1J_chance_reference.json"))
    c.append(("chance precordial", f"contains ${fmt(ch['expected_n_precordial_random_triple'], 2)}$ precordial leads on average and is all-precordial with probability ${ch['count_by_n_precordial']['3']}/{ch['n_three_lead_subsets']}={fmt(ch['prob_all_three_precordial'], 2)}$"))
    c.append(("observed precordial means", f"average exactly ${fmt(ch['observed']['identity']['mean_n_precordial_over_seeds'], 1)}$ precordial leads"))
    c.append(("empirical precordial mean", f"average ${fmt(ch['observed']['empirical_sigma_inv']['mean_n_precordial_over_seeds'], 1)}$"))
    # the chance probability of an all-precordial triple is stated in the base-rate sentence
    c.append(("all-precordial chance", f"is all-precordial with probability ${ch['count_by_n_precordial']['3']}/{ch['n_three_lead_subsets']}={fmt(ch['prob_all_three_precordial'], 2)}$"))
    tf = pd.read_csv(STATS / "W1O_teacherfree_contrast.csv").set_index("lead_set")
    d1 = tf.loc["II+V1+V5", "obs_stable_auroc"] - tf.loc["II+V1+V5", "teacher_stock_auroc"]
    d2 = tf.loc["I+V1+V4", "obs_stable_auroc"] - tf.loc["I+V1+V4", "teacher_stock_auroc"]
    c.append(("teacher-free deltas", f"by $+{fmt(d1, 3)}$ and ${fmt(d2, 3)}$"))
    c.append(("largest optimism", f"largest optimism in any matched comparison is ${fmt(tf[['teacher_optimism_stable', 'teacher_optimism_stock']].max().max(), 3)}$"))
    noise = pd.read_csv(STATS / "W1J_per_lead_noise.csv").set_index("lead").noise_std_scaled
    limb, chest = noise[["I", "II", "III", "aVR", "aVL", "aVF"]], noise[["V2", "V3", "V4", "V5"]]
    c.append(("noise ratio", f"carry a ${fmt(limb.min() / chest.max(), 1)}$--${fmt(limb.max() / chest.min(), 1)}\\times$ smaller unmodeled pre-QRS standard deviation"))

    # ---- Section 3.7 / Table 7: reduced-lead classification
    lc = pd.read_csv(OUT / "runs" / "v1fix_leadclf" / "leadclf_m10.csv")
    lc5 = lc[lc.feature_group == "theta_plus_reconstruction"]
    named = lc5[lc5.kind != "random"].set_index("lead_set")
    for ls in ("II+V1+V5", "I+V1+V4", "II"):
        r = named.loc[ls]
        c.append((f"Table 7 AUROC {ls}", f"${fmt(r.macro_auroc, 3)}$ $[{fmt(r.ci_lo, 3)},{fmt(r.ci_hi, 3)}]$"))
    c.append(("Table 7 I+V1+V5", f"${fmt(named.loc['I+V1+V5', 'macro_auroc'], 3)}$ $[{fmt(named.loc['I+V1+V5', 'ci_lo'], 3)},{fmt(named.loc['I+V1+V5', 'ci_hi'], 3)}]$"))
    rnd = lc5[lc5.kind == "random"].macro_auroc
    c.append(("random triples", f"${fmt(rnd.mean(), 3)}\\pm{fmt(rnd.std(ddof=1), 3)}$"))
    full12 = named.loc["12 displayed channels", "macro_auroc"]
    c.append(("effective-rank loss", f"lowers the FIM effective rank from ${fmt(fig34.loc['12', 'mean'], 2)}\\pm{fmt(fig34.loc['12', 'std'], 2)}$ to ${fmt(fig34.loc['II', 'mean'], 2)}\\pm{fmt(fig34.loc['II', 'std'], 2)}$"))
    c.append(("CRB ratio 4.2", f"raises the Cram\\'er--Rao bound of the ST source almost sevenfold (${fmt(crb['L12'], 2)}$ to ${fmt(crb['II'], 2)}$)"))
    # external single-label criterion: fraction of each cohort removed, and its components for Georgia
    disp = pd.read_csv(STATS / "W1K_cohort_disposition.csv").set_index(["cohort", "quantity"]).pct_of_cohort
    g_rem = 100 - disp[("georgia", "n_clean_single_label_kept")]
    c.append(("external filter", f"removed ${round(g_rem)}\\%$ of Georgia (${round(disp[('georgia', 'drop_2_OTHER_code_present')])}\\%$ for a rhythm, axis or ectopy code, ${round(disp[('georgia', 'drop_1_HYP_overlap')])}\\%$ for hypertrophy overlap, ${round(disp[('georgia', 'drop_3_multi_superclass')])}\\%$ for two target superclasses) and ${round(100 - disp[('cpsc_2018', 'n_clean_single_label_kept')])}\\%$ of CPSC2018"))
    # Table S1 (AP-ODE recovery multiplier, seed 42) and Table S2 (lead-II variant effective ranks)
    ode = pd.read_csv(OUT / "runs" / "v1fix_r3_s42" / "tables" / "ap_ode_monodromy.csv")
    for cls in ("NORM", "MI", "STTC", "CD"):
        r = ode[ode["class"] == cls].rho
        c.append((f"Table S1 {cls}", f"${fmt(r.mean(), 4)}\\pm{fmt(r.std(ddof=1), 4)}$ & ${fmt(1 - r.mean(), 4)}$"))
    sl = pd.read_csv(OUT / "runs" / "v1fix_round5" / "tables" / "single_lead_sanity.csv").set_index("variant")
    for v in sl.index:
        c.append((f"Table S2 effrank {v}", f"& {fmt(sl.loc[v, 'fim_effrank_L1'], 2)} & {fmt(sl.loc[v, 'theta_macro_auroc'], 3)} &"))
    c.append(("AUROC loss", f"only about ${round(100 * (1 - named.loc['II', 'macro_auroc'] / full12))}\\%$ of macro-AUROC (${fmt(full12, 3)}$ to ${fmt(named.loc['II', 'macro_auroc'], 3)}$)"))
    r5 = pd.read_csv(OUT / "runs" / "v1fix_round5" / "tables" / [f for f in (OUT / "runs" / "v1fix_round5" / "tables").glob("*.csv")][0].name) if (OUT / "runs" / "v1fix_round5" / "tables").exists() else None

    # ---- Section 3.8: multi-label (W1F)
    F = json.load(open(STATS / "W1F_multilabel_headline.json"))
    ra, rb, rc = F["recon"]["a_all_ge1_superclass"], F["recon"]["b_multilabel_gt1"], F["recon"]["clean_subset_CONTROL"]
    c.append(("multi-label recon", f"median scaled NRMSE ${fmt(ra['median_nrmse_scaled'], 4)}$ and correlation ${fmt(ra['median_corr'], 4)}$ over the whole fold-10 pool ($n={cnt(ra['n'])}$) against ${fmt(rc['median_nrmse_scaled'], 4)}$ and ${fmt(rc['median_corr'], 4)}$ on the clean subset, and ${fmt(rb['median_nrmse_scaled'], 4)}$ and ${fmt(rb['median_corr'], 4)}$ on the ${rb['n']}$ records"))
    au = F["auroc"]["C0_theta_xgboost"]
    c.append(("multi-label AUROC", f"${fmt(au['MACRO']['auroc'], 4)}$ $[{fmt(au['MACRO']['ci'][0], 4)},{fmt(au['MACRO']['ci'][1], 4)}]$"))
    c.append(("multi-label MI/HYP", f"with MI at ${fmt(au['MI']['auroc'], 4)}$ and hypertrophy, never trained for, at ${fmt(au['HYP']['auroc'], 4)}$"))

    # ---- Section 3.9 / Table 7 (merged): external cohorts
    ext = {k: json.load(open(OUT / "v1fix" / "external" / f"cls_{k}.json")) for k in ("georgia", "cpsc2018")}
    return c, ext


def _external_checks(ext):
    """External-cohort numbers; the JSON layout is discovered from the files themselves."""
    c = []
    for cohort, label in (("georgia", "Georgia"), ("cpsc2018", "CPSC2018")):
        e = ext[cohort]
        auroc = e.get("macro_auroc_present") or e.get("macro_auroc")
        ci = e.get("macro_auroc_present_ci95")
        if auroc is not None and ci is not None:
            c.append((f"{label} AUROC", f"{fmt(auroc, 3)} [{fmt(ci[0], 3)}, {fmt(ci[1], 3)}]"))
        n = e.get("n") or e.get("n_records")
        if n is not None:
            c.append((f"{label} n", f"$n={cnt(n)}$"))
        pc = e.get("per_class") or {}
        if all(k in pc and "recall" in pc[k] for k in ("NORM", "STTC", "CD")):
            c.append((f"{label} recall", " / ".join(fmt(pc[k]["recall"], 2) for k in ("NORM", "STTC", "CD"))))
    return c


def test_manuscript_numbers_match_outputs():
    c, ext = checks()
    c += _external_checks(ext)
    missing = [(label, s) for label, s in c if not stated(s)]
    assert not missing, "manuscript does not state:\n" + "\n".join(f"  {l}: {s}" for l, s in missing)
    assert len(c) >= 90, len(c)
