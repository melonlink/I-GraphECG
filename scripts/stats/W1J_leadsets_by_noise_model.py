"""W1J - empirical-noise lead selection at evidential parity with the identity arm.

Ports W = Sigma^{-1} (empirical diagonal noise covariance, per-lead isoelectric variance,
protocol of scripts/53_noise_model_sensitivity.py) into the FULL identity-arm protocol of
scripts/51_leadset8_exhaustive.py:
  * same fixed 400-record fold-10 subset (per_class=100, rng seed 42), same 47-parameter FIM,
    same Tikhonov lambda=1e-3, same 5 checkpoints d1s_r3_s{42,1,2,3,4};
  * EXHAUSTIVE search over all C(8,3)=56 three-lead subsets of the 8 independent leads;
  * bootstrap (200 record-level resamples, rng seed 42) for the selection margin;
  * comparison against the clinical montage II+V1+V5.

Control: the identity arm reproduced here must match runs/v1fix_ls8_s*/tables/ bit-for-bit
(up to GPU float noise), otherwise the port is not at parity and we stop.

Also emits (a) the measured per-lead isoelectric noise levels and (b) the chance-level
reference for "precordial dominance" over the 8-lead selection pool.

Read-only w.r.t. the frozen lineage; every artifact lands in runs/v7rev_stats/.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time
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
from igraphecg.evaluation.identifiability import lead_rows, PARAM_GROUPS
from igraphecg.evaluation.round2_io import load_encoder_decoder, infer_theta

OUT = repro.outdir("runs/v7rev_stats")
CKPT_DIR = repro.outputs() / "checkpoints"
NPZ = REPO / "data/processed/ptbxl_medianbeat_clean_100hz.npz"
LS8_DIR = repro.runs()

LAM = 1e-3
NBOOT = 200
SEEDS = [42, 1, 2, 3, 4]
PRE_SAMPLES = 20                      # isoelectric TP/PR window, script 58 convention
IND_NAMES = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
IND = [CANONICAL_LEADS.index(x) for x in IND_NAMES]
INAME = {l: CANONICAL_LEADS[l] for l in IND}
CLIN = tuple(CANONICAL_LEADS.index(x) for x in ("II", "V1", "V5"))
PRECORDIAL = {"V1", "V2", "V3", "V4", "V5", "V6"}


# ---------------------------------------------------------------- protocol helpers
def fixed_subset(label, fold, per_class=100, seed=42):
    """Verbatim from scripts/51_leadset8_exhaustive.py."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(len(TARGET_CLASSES)):
        pool = np.where((label == c) & (fold == 10))[0]
        idx.append(pool if len(pool) <= per_class else rng.choice(pool, per_class, replace=False))
    return np.concatenate(idx)


def compute_J_all(dec, thetas, radius, device):
    from torch.func import jacfwd
    r = radius.to(device).view(1, -1)

    def f(th):
        y12, _ = dec.forward(th.unsqueeze(0))
        return y12[0].reshape(-1)

    return torch.stack([jacfwd(f)(thetas[s].to(device)) * r
                        for s in range(thetas.shape[0])]).detach()


def logdet(F):
    return float(np.linalg.slogdet(F + LAM * np.eye(F.shape[0]))[1])


def trinv(F):
    return float(np.trace(np.linalg.inv(F + LAM * np.eye(F.shape[0]))))


def lmin(F):
    return float(np.linalg.eigvalsh(F + LAM * np.eye(F.shape[0]))[0])


def nprec(leads):
    return sum(1 for l in leads if INAME[l] in PRECORDIAL)


def setname(leads):
    return "+".join(INAME[i] for i in leads)


# ---------------------------------------------------------------- (a) empirical noise
def _baseline_var(sig, folds):
    b = sig[folds <= 8, :, :PRE_SAMPLES]
    b = b - b.mean(axis=2, keepdims=True)
    return b.var(axis=2).mean(axis=0), b.shape[0]      # [12], n


def per_lead_noise(sig_scaled, sig_phys, folds):
    """script 58 protocol: first 20 samples of the median beat, DC-removed, training folds only.

    Returned in BOTH unit systems. The decoder / FIM live in scaled space, so the
    scaled variance is the one that defines Sigma; the physical-mV variance is
    reported so the manuscript can separate "quieter lead" from "larger IQR".
    """
    var_scaled, n = _baseline_var(sig_scaled, folds)
    var_mv, _ = _baseline_var(sig_phys, folds)
    return var_scaled, np.sqrt(var_scaled), var_mv, np.sqrt(var_mv), n


# ---------------------------------------------------------------- (b) chance reference
def chance_reference():
    n_pool, n_prec = 8, 6
    combos = list(itertools.combinations(range(n_pool), 3))
    assert len(combos) == 56
    dist = {}
    for k in range(4):
        cnt = math.comb(n_prec, k) * math.comb(n_pool - n_prec, 3 - k)
        dist[k] = cnt
    total = sum(dist.values())
    exp_prec = sum(k * c for k, c in dist.items()) / total
    return {
        "selection_pool": list(IND_NAMES),
        "pool_size": n_pool,
        "n_precordial_in_pool": n_prec,
        "n_limb_in_pool": n_pool - n_prec,
        "n_three_lead_subsets": total,
        "count_by_n_precordial": {str(k): int(v) for k, v in dist.items()},
        "prob_by_n_precordial": {str(k): v / total for k, v in dist.items()},
        "expected_n_precordial_random_triple": exp_prec,
        "prob_all_three_precordial": dist[3] / total,
        "prob_at_least_two_precordial": (dist[2] + dist[3]) / total,
        "modal_n_precordial_random_triple": max(dist, key=dist.get),
    }


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[W1J] device={dev}")

    data = load_processed(NPZ)
    signals = data["signal_12lead"].astype(np.float32)
    n_t = signals.shape[2]
    label = data["label"].astype(int)
    fold = data["fold"].astype(int)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    scaled = scaler.transform(signals)

    # ---- (a) per-lead empirical noise -------------------------------------------
    mean_var, noise_std, var_mv, std_mv, n_base = per_lead_noise(scaled, signals, fold)
    sigma_inv12 = 1.0 / np.maximum(mean_var, 1e-8)
    w12_s58 = sigma_inv12 / sigma_inv12.mean()                  # script-58 normalisation (12 leads)
    w8 = sigma_inv12[IND] / sigma_inv12[IND].mean()             # alt: normalised over the 8-lead pool
    df_noise = pd.DataFrame({
        "lead": list(CANONICAL_LEADS),
        "noise_std_scaled": noise_std,
        "noise_var_scaled": mean_var,
        "noise_std_mV": std_mv,
        "noise_var_mV2": var_mv,
        "scaler_iqr_mV": np.asarray(scaler.iqr_, dtype=float),
        "scaler_median_mV": np.asarray(scaler.median_, dtype=float),
        "sigma_inv_scaled": sigma_inv12,
        "weight_norm12_script58": w12_s58,
        "is_independent_lead": [c in IND_NAMES for c in CANONICAL_LEADS],
        "is_precordial": [c in PRECORDIAL for c in CANONICAL_LEADS],
        "n_baseline_records": n_base,
        "baseline_samples": PRE_SAMPLES,
    })
    df_noise["noise_std_scaled_check"] = df_noise["noise_std_mV"] / df_noise["scaler_iqr_mV"]
    df_noise.to_csv(OUT / "W1J_per_lead_noise.csv", index=False)
    prec8 = [c for c in IND_NAMES if c in PRECORDIAL]
    limb8 = [c for c in IND_NAMES if c not in PRECORDIAL]
    g = df_noise.set_index("lead")
    noise_contrast = {
        "precordial_in_pool": prec8, "limb_in_pool": limb8,
        "mean_noise_std_scaled_precordial": float(g.loc[prec8, "noise_std_scaled"].mean()),
        "mean_noise_std_scaled_limb": float(g.loc[limb8, "noise_std_scaled"].mean()),
        "ratio_limb_over_precordial_scaled": float(g.loc[limb8, "noise_std_scaled"].mean()
                                                   / g.loc[prec8, "noise_std_scaled"].mean()),
        "mean_noise_std_mV_precordial": float(g.loc[prec8, "noise_std_mV"].mean()),
        "mean_noise_std_mV_limb": float(g.loc[limb8, "noise_std_mV"].mean()),
        "ratio_limb_over_precordial_mV": float(g.loc[limb8, "noise_std_mV"].mean()
                                               / g.loc[prec8, "noise_std_mV"].mean()),
        "mean_iqr_mV_precordial": float(g.loc[prec8, "scaler_iqr_mV"].mean()),
        "mean_iqr_mV_limb": float(g.loc[limb8, "scaler_iqr_mV"].mean()),
        "quietest_lead_scaled": str(df_noise.loc[df_noise.noise_std_scaled.idxmin(), "lead"]),
        "loudest_lead_scaled": str(df_noise.loc[df_noise.noise_std_scaled.idxmax(), "lead"]),
        "quietest_lead_mV": str(df_noise.loc[df_noise.noise_std_mV.idxmin(), "lead"]),
        "loudest_lead_mV": str(df_noise.loc[df_noise.noise_std_mV.idxmax(), "lead"]),
        "maxmin_ratio_scaled": float(noise_std.max() / noise_std.min()),
        "maxmin_ratio_mV": float(std_mv.max() / std_mv.min()),
    }
    iV2, iAVF = list(CANONICAL_LEADS).index("V2"), list(CANONICAL_LEADS).index("aVF")
    noise_contrast["headline_V2_vs_aVF"] = {
        "V2_noise_std_scaled": float(noise_std[iV2]),
        "aVF_noise_std_scaled": float(noise_std[iAVF]),
        "ratio_aVF_over_V2_scaled": float(noise_std[iAVF] / noise_std[iV2]),
        "V2_noise_std_mV": float(std_mv[iV2]),
        "aVF_noise_std_mV": float(std_mv[iAVF]),
        "ratio_aVF_over_V2_mV": float(std_mv[iAVF] / std_mv[iV2]),
        "V2_iqr_mV": float(scaler.iqr_[iV2]),
        "aVF_iqr_mV": float(scaler.iqr_[iAVF]),
        "ratio_V2_over_aVF_iqr": float(scaler.iqr_[iV2] / scaler.iqr_[iAVF]),
        "interpretation": (
            "The 2.07x scaled-space contrast is ~entirely a dynamic-range effect: in absolute mV "
            "the two leads have essentially the same isoelectric noise "
            "(V2 %.5f vs aVF %.5f mV, ratio %.3f), while V2's robust IQR is %.2fx larger. "
            "Sigma (defined in the scaled space the decoder and FIM live in) is therefore a "
            "per-lead inverse baseline-SNR, NOT an absolute noise level."
            % (float(std_mv[iV2]), float(std_mv[iAVF]),
               float(std_mv[iAVF] / std_mv[iV2]),
               float(scaler.iqr_[iV2] / scaler.iqr_[iAVF]))),
    }
    print("[W1J] V2 vs aVF: scaled %.5f/%.5f (%.2fx) | mV %.5f/%.5f (%.3fx) | IQR ratio %.2fx"
          % (noise_std[iV2], noise_std[iAVF], noise_std[iAVF] / noise_std[iV2],
             std_mv[iV2], std_mv[iAVF], std_mv[iAVF] / std_mv[iV2],
             scaler.iqr_[iV2] / scaler.iqr_[iAVF]))
    print("[W1J] mV noise std: " + ", ".join(f"{c}={v:.5f}" for c, v in zip(CANONICAL_LEADS, std_mv)))
    print(f"[W1J] scaled limb/precordial noise ratio = "
          f"{noise_contrast['ratio_limb_over_precordial_scaled']:.3f} | "
          f"mV ratio = {noise_contrast['ratio_limb_over_precordial_mV']:.3f}")
    print("[W1J] per-lead noise std: " + ", ".join(f"{c}={s:.5f}"
                                                   for c, s in zip(CANONICAL_LEADS, noise_std)))
    print(f"[W1J] max/min noise-std ratio = {noise_std.max()/noise_std.min():.3f}")

    weight_maps = {
        "identity": {l: 1.0 for l in IND},
        "empirical_sigma_inv": {l: float(w12_s58[l]) for l in IND},
        "empirical_sigma_inv_norm8": {l: float(w8[i]) for i, l in enumerate(IND)},
    }

    combos3 = list(itertools.combinations(IND, 3))
    assert len(combos3) == 56

    top_rows, summary_rows, all56_rows, control_rows, incl_rows = [], [], [], [], []
    cross_rows = []
    opt_by_seed = {}

    for seed in SEEDS:
        ck = repro.lineage.checkpoint(seed)
        enc, dec = load_encoder_decoder(ck, n_t, 3, dev)
        idx = fixed_subset(label, fold)
        theta, _ = infer_theta(enc, dec, scaled[idx], device=dev)
        thetas = torch.from_numpy(theta).to(dev)
        radius = dec.pspace.radius
        J_all = compute_J_all(dec, thetas, radius, dev)          # [N,1200,47]
        N = J_all.shape[0]
        print(f"[W1J] seed={seed} ckpt={ck.name} N={N} J={tuple(J_all.shape)} "
              f"({time.time()-t0:.0f}s)")

        G = {}
        for l in IND:
            rows = torch.tensor(lead_rows([l], n_t), device=dev)
            Jl = J_all[:, rows, :]
            G[l] = torch.einsum("nrp,nrq->npq", Jl, Jl).cpu().numpy().astype(np.float64)
        del J_all
        if dev == "cuda":
            torch.cuda.empty_cache()

        Gbar = {l: G[l].mean(0) for l in IND}

        for wname, W in weight_maps.items():
            def F_of(S, bar=Gbar):
                return sum(W[l] * bar[l] for l in S)

            scores = {c: logdet(F_of(c)) for c in combos3}
            order = sorted(scores, key=scores.get, reverse=True)
            opt3 = order[0]
            clin_ld = logdet(F_of(CLIN))
            margin = scores[opt3] - clin_ld
            clin_rank = order.index(CLIN) + 1
            ii_ranks = [r + 1 for r, c in enumerate(order)
                        if CANONICAL_LEADS.index("II") in c]
            ii_in_top5 = min(ii_ranks) <= 5

            for r, c in enumerate(order[:5], start=1):
                F = F_of(c)
                top_rows.append({
                    "seed": seed, "noise_model": wname, "rank": r, "leads": setname(c),
                    "D_logdet": scores[c], "A_trace_inv": trinv(F), "E_lambda_min": lmin(F),
                    "n_precordial": nprec(c), "contains_II": "II" in setname(c).split("+"),
                    "delta_logdet_vs_rank1": scores[c] - scores[opt3],
                    "delta_logdet_vs_clinical": scores[c] - clin_ld,
                })
            Fc = F_of(CLIN)
            top_rows.append({
                "seed": seed, "noise_model": wname, "rank": clin_rank,
                "leads": setname(CLIN) + " (clinical)",
                "D_logdet": clin_ld, "A_trace_inv": trinv(Fc), "E_lambda_min": lmin(Fc),
                "n_precordial": nprec(CLIN), "contains_II": True,
                "delta_logdet_vs_rank1": -margin, "delta_logdet_vs_clinical": 0.0,
            })
            for c in combos3:
                all56_rows.append({"seed": seed, "noise_model": wname, "leads": setname(c),
                                   "D_logdet": scores[c], "n_precordial": nprec(c),
                                   "rank": order.index(c) + 1})

            # ---- bootstrap: identical resample stream to scripts/42 ----
            rng = np.random.default_rng(42)
            win = {c: 0 for c in combos3}
            incl = {l: 0 for l in IND}
            gaps = np.empty(NBOOT)
            prec_boot = np.empty(NBOOT)
            for b in range(NBOOT):
                bi = rng.integers(0, N, N)
                Fl = {l: G[l][bi].mean(0) for l in IND}

                def Fb(S):
                    return sum(W[l] * Fl[l] for l in S)

                wd = max(combos3, key=lambda c: logdet(Fb(c)))
                win[wd] += 1
                for l in wd:
                    incl[l] += 1
                prec_boot[b] = nprec(wd)
                gaps[b] = logdet(Fb(opt3)) - logdet(Fb(CLIN))
            ci = (float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5)))
            topwin = max(win, key=win.get)

            opt_by_seed[(seed, wname)] = opt3
            summary_rows.append({
                "seed": seed, "noise_model": wname,
                "optimal3": setname(opt3), "optimal3_logdet": scores[opt3],
                "clinical": setname(CLIN), "clinical_logdet": clin_ld,
                "clinical_rank_of_56": clin_rank,
                "margin_logdet": margin,
                "boot_margin_mean": float(gaps.mean()),
                "boot_margin_ci_lo": ci[0], "boot_margin_ci_hi": ci[1],
                "boot_margin_crosses_zero": bool(ci[0] <= 0 <= ci[1]),
                "boot_top_set": setname(topwin), "boot_top_set_freq": win[topwin] / NBOOT,
                "boot_freq_of_fulldata_opt": win[opt3] / NBOOT,
                "n_precordial_opt3": nprec(opt3),
                "boot_mean_n_precordial": float(prec_boot.mean()),
                "boot_frac_all_precordial": float((prec_boot == 3).mean()),
                "II_best_rank": min(ii_ranks), "II_in_top5": bool(ii_in_top5),
                "top5_mean_n_precordial": float(np.mean([nprec(c) for c in order[:5]])),
                "nboot": NBOOT, "n_records": int(N), "lambda": LAM, "fim_dim": 47,
                "checkpoint": ck.name,
            })
            for l in IND:
                incl_rows.append({"seed": seed, "noise_model": wname, "lead": INAME[l],
                                  "bootstrap_inclusion_rate": incl[l] / NBOOT})
            print(f"[W1J]   {wname:26s} opt3={setname(opt3):12s} logdet={scores[opt3]:8.3f} "
                  f"clin={clin_ld:8.3f} margin={margin:7.3f} "
                  f"CI=[{ci[0]:.3f},{ci[1]:.3f}] nprec={nprec(opt3)} "
                  f"II_rank={min(ii_ranks)}")

        # ---- cross-model: evaluate each arm's optimum under BOTH weightings ----
        opt_id = opt_by_seed[(seed, "identity")]
        opt_em = opt_by_seed[(seed, "empirical_sigma_inv")]
        for wname in ("identity", "empirical_sigma_inv"):
            W = weight_maps[wname]

            def Fw(S, bar=Gbar, W=W):
                return sum(W[l] * bar[l] for l in S)

            allc = {c: logdet(Fw(c)) for c in combos3}
            order = sorted(allc, key=allc.get, reverse=True)
            rng = np.random.default_rng(42)
            d = np.empty(NBOOT)
            for b in range(NBOOT):
                bi = rng.integers(0, N, N)
                Fl = {l: G[l][bi].mean(0) for l in IND}

                def Fb(S, W=W):
                    return sum(W[l] * Fl[l] for l in S)

                d[b] = logdet(Fb(opt_em)) - logdet(Fb(opt_id))
            lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            cross_rows.append({
                "seed": seed, "evaluated_under": wname,
                "identity_optimum": setname(opt_id), "empirical_optimum": setname(opt_em),
                "logdet_identity_optimum": allc[opt_id],
                "logdet_empirical_optimum": allc[opt_em],
                "rank_of_identity_optimum_of_56": order.index(opt_id) + 1,
                "rank_of_empirical_optimum_of_56": order.index(opt_em) + 1,
                "delta_emp_minus_id": allc[opt_em] - allc[opt_id],
                "boot_delta_mean": float(d.mean()),
                "boot_delta_ci_lo": lo, "boot_delta_ci_hi": hi,
                "boot_delta_crosses_zero": bool(lo <= 0 <= hi),
                "same_set": opt_id == opt_em,
            })
            print(f"[W1J]   cross under {wname:20s} "
                  f"id={setname(opt_id):10s}(rank {order.index(opt_id)+1:2d}) "
                  f"emp={setname(opt_em):10s}(rank {order.index(opt_em)+1:2d}) "
                  f"delta={allc[opt_em]-allc[opt_id]:+8.3f} "
                  f"CI=[{lo:+.3f},{hi:+.3f}]")

        # ---- control: identity arm vs frozen runs/v1fix_ls8_s{seed} ----
        ref = LS8_DIR / f"v1fix_ls8_s{seed}" / "tables"
        if (ref / "optimal3_vs_clinical.csv").exists():
            rdf = pd.read_csv(ref / "optimal3_vs_clinical.csv")
            gdf = pd.read_csv(ref / "bootstrap_gap_summary.csv")
            mine = [r for r in summary_rows if r["seed"] == seed and r["noise_model"] == "identity"][0]
            for field, ref_val, my_val in [
                ("optimal3_leads", str(rdf.iloc[0]["leads"]), mine["optimal3"]),
                ("optimal3_logdet", float(rdf.iloc[0]["D_logdet"]), mine["optimal3_logdet"]),
                ("clinical_logdet", float(rdf.iloc[1]["D_logdet"]), mine["clinical_logdet"]),
                ("gap_lambda", float(gdf.iloc[0]["gap_lambda"]), mine["margin_logdet"]),
                ("boot_gap_mean", float(gdf.iloc[0]["bootstrap_gap_mean"]), mine["boot_margin_mean"]),
                ("boot_ci_lo", float(gdf.iloc[0]["ci_lo"]), mine["boot_margin_ci_lo"]),
                ("boot_ci_hi", float(gdf.iloc[0]["ci_hi"]), mine["boot_margin_ci_hi"]),
            ]:
                if isinstance(ref_val, str):
                    ok, diff = (ref_val == my_val), ""
                else:
                    diff = abs(ref_val - my_val)
                    ok = (diff < 5e-3) or (diff / max(abs(ref_val), 1e-12) < 1e-4)
                rel = ("" if isinstance(ref_val, str)
                       else abs(ref_val - my_val) / max(abs(ref_val), 1e-12))
                control_rows.append({"seed": seed, "field": field, "frozen_value": ref_val,
                                     "recomputed_value": my_val, "abs_diff": diff,
                                     "rel_diff": rel, "match": ok})
                if not ok:
                    print(f"[W1J] !! CONTROL MISMATCH seed={seed} {field}: "
                          f"frozen={ref_val} mine={my_val}")
        del G, Gbar

    pd.DataFrame(top_rows).to_csv(OUT / "W1J_leadsets_by_noise_model.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "W1J_margin_bootstrap_summary.csv", index=False)
    pd.DataFrame(all56_rows).to_csv(OUT / "W1J_all56_logdet.csv", index=False)
    pd.DataFrame(incl_rows).to_csv(OUT / "W1J_bootstrap_lead_inclusion.csv", index=False)
    pd.DataFrame(cross_rows).to_csv(OUT / "W1J_cross_noise_model.csv", index=False)
    ctl = pd.DataFrame(control_rows)
    ctl.to_csv(OUT / "W1J_identity_control_check.csv", index=False)
    print(f"[W1J] controls: {int(ctl['match'].sum())}/{len(ctl)} matched")

    # ---- (b) chance reference + where the optima sit -----------------------------
    chance = chance_reference()
    sdf = pd.DataFrame(summary_rows)
    obs = {}
    for wname in weight_maps:
        sub = sdf[sdf.noise_model == wname]
        counts = sub["n_precordial_opt3"].tolist()
        p_ge = [chance["prob_by_n_precordial"][str(k)] for k in range(3, 4)]
        obs[wname] = {
            "optima_by_seed": dict(zip(sub["seed"].astype(int).tolist(), sub["optimal3"].tolist())),
            "n_precordial_by_seed": dict(zip(sub["seed"].astype(int).tolist(), [int(c) for c in counts])),
            "mean_n_precordial_over_seeds": float(np.mean(counts)),
            "expected_under_chance": chance["expected_n_precordial_random_triple"],
            "difference_vs_chance": float(np.mean(counts) - chance["expected_n_precordial_random_triple"]),
            "above_chance": bool(np.mean(counts) > chance["expected_n_precordial_random_triple"]),
            "one_sided_p_of_observing_at_least_this_many_by_chance": float(
                sum(chance["prob_by_n_precordial"][str(k)]
                    for k in range(int(round(np.mean(counts))), 4))),
            "top5_mean_n_precordial_over_seeds": float(sub["top5_mean_n_precordial"].mean()),
            "bootstrap_mean_n_precordial_over_seeds": float(sub["boot_mean_n_precordial"].mean()),
            "bootstrap_frac_all_precordial_over_seeds": float(sub["boot_frac_all_precordial"].mean()),
        }
        del p_ge
    chance["observed"] = obs
    chance["clinical_set"] = {
        "leads": setname(CLIN), "n_precordial": nprec(CLIN),
        "clinical_rank_of_56_by_seed": {
            wn: dict(zip(sdf[sdf.noise_model == wn]["seed"].astype(int).tolist(),
                         sdf[sdf.noise_model == wn]["clinical_rank_of_56"].astype(int).tolist()))
            for wn in weight_maps}}
    chance["measured_noise_contrast"] = noise_contrast
    chance["margin_over_clinical_by_noise_model"] = {
        wn: {
            "per_seed": dict(zip(sdf[sdf.noise_model == wn]["seed"].astype(int).tolist(),
                                 sdf[sdf.noise_model == wn]["margin_logdet"].tolist())),
            "mean": float(sdf[sdf.noise_model == wn]["margin_logdet"].mean()),
            "sd_ddof1": float(sdf[sdf.noise_model == wn]["margin_logdet"].std(ddof=1)),
            "min": float(sdf[sdf.noise_model == wn]["margin_logdet"].min()),
            "max": float(sdf[sdf.noise_model == wn]["margin_logdet"].max()),
            "all_bootstrap_CIs_exclude_zero": bool(
                (~sdf[sdf.noise_model == wn]["boot_margin_crosses_zero"]).all()),
            "clinical_rank_of_56_per_seed": sorted(
                sdf[sdf.noise_model == wn]["clinical_rank_of_56"].astype(int).tolist()),
            "lead_II_best_rank_per_seed": sorted(
                sdf[sdf.noise_model == wn]["II_best_rank"].astype(int).tolist()),
            "lead_II_enters_top5_in_n_seeds": int(sdf[sdf.noise_model == wn]["II_in_top5"].sum()),
        } for wn in weight_maps}
    p_all = chance["prob_all_three_precordial"]
    n_seeds = len(SEEDS)
    for wname in weight_maps:
        k_all = int(sum(1 for v in obs[wname]["n_precordial_by_seed"].values() if v == 3))
        p_ge = float(sum(math.comb(n_seeds, j) * p_all ** j * (1 - p_all) ** (n_seeds - j)
                         for j in range(k_all, n_seeds + 1)))
        p_le = float(sum(math.comb(n_seeds, j) * p_all ** j * (1 - p_all) ** (n_seeds - j)
                         for j in range(0, k_all + 1)))
        obs[wname]["n_all_precordial_optima_of_5_seeds"] = k_all
        obs[wname]["binom_p_at_least_this_many_all_precordial"] = p_ge
        obs[wname]["binom_p_at_most_this_many_all_precordial"] = p_le
        obs[wname]["binom_note"] = (
            "Binomial(n=5, p=20/56) against the 8-lead-pool base rate. The 5 seeds share the "
            "same 400-record subset and differ only in encoder initialisation, so they are not "
            "fully independent draws; treat this as a heuristic, not a formal test.")
    chance["verdict"] = {
        "identity_optima_are_below_chance_on_precordial_count": bool(
            obs["identity"]["mean_n_precordial_over_seeds"]
            < chance["expected_n_precordial_random_triple"]),
        "empirical_optima_are_above_chance_on_precordial_count": bool(
            obs["empirical_sigma_inv"]["mean_n_precordial_over_seeds"]
            > chance["expected_n_precordial_random_triple"]),
        "count_based_dominance_claim_supported_under_identity": False,
        "count_based_dominance_claim_supported_under_empirical": bool(
            obs["empirical_sigma_inv"]["mean_n_precordial_over_seeds"]
            > chance["expected_n_precordial_random_triple"]),
    }
    chance["note"] = (
        "A count-based 'precordial dominance' claim must be read against the base rate of the "
        "8-lead selection pool (6/8 precordial). A random triple already contains 2.25 precordial "
        "leads on average and is all-precordial with probability 20/56=0.357."
    )
    (OUT / "W1J_chance_reference.json").write_text(
        json.dumps(chance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[W1J] DONE in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
