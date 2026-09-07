"""W1M -- scaled-space Einthoven/Goldberger coefficients, their cost, and the FIM difference.

Three deliverables (all on the frozen d1s_r3_s42 / v1fix lineage; nothing is overwritten):

 A) exact affine coefficients on (I_s, II_s, 1) for III, aVR, aVL, aVF, from the
    train-fitted RobustLeadScaler stored in the processed npz.
 B) counterfactual: apply the PHYSICAL coefficients to scaled signals; residual in mV.
 C) FIM / effective-rank difference between the two derivation conventions, evaluated
    with the FIXED trained d1s_r3_s42 decoder at analysis time.

Run with the torch env:
  python <this file>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.func import jacfwd

import sys as _sys
from pathlib import Path as _Path
# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro
REPO = repro.repo()
sys.path.insert(0, str(REPO))

from igraphecg.data.label_utils import TARGET_CLASSES                     # noqa: E402
from igraphecg.evaluation.identifiability import fim_metrics, lead_rows   # noqa: E402
from igraphecg.evaluation.round2_io import load_encoder_decoder           # noqa: E402
from igraphecg.models.leadfield import (DERIVED, OUTPUT_LEADS, _D_PHYS,   # noqa: E402
                                        scaled_einthoven_coeffs)
from igraphecg.models.surrogate_decoder import PARAM_DIM, SurrogateDecoder  # noqa: E402
from igraphecg.training.losses import NON_DERIVED_ROWS                    # noqa: E402

OUT = repro.outdir("runs/v7rev_stats")
NPZ = REPO / "data/processed/ptbxl_medianbeat_clean_100hz.npz"
CKPT = repro.lineage.checkpoint(42)
TEST_FOLD = 10
LEADFIELD_RANK = 3
PER_CLASS = 100          # same balanced-subset size as experiments/02_observability_control/legacy_scripts/61_revision_v2_fim.py
SEED = 42
FIM_COLS = np.arange(PARAM_DIM - 1)   # drop global_gain: fixes the (q, alpha_ST, gain) scale gauge

# expected values quoted in the task (independent earlier check)
EXPECTED = {"III": (-1.0762, +1.4223, -0.0244), "aVR": (-0.4507, -0.5957, -0.0210),
            "aVL": (+1.3937, -0.9210, -0.0390), "aVF": (-0.5093, +1.3462, -0.0404)}
COEF_TOL = 5e-4          # expected values are given to 4 d.p.

fail: list[str] = []


def main() -> int:
    torch.manual_seed(0)
    d = np.load(NPZ, allow_pickle=True)
    leads = [str(x) for x in d["lead_names"]]
    if leads != OUTPUT_LEADS:
        raise SystemExit(f"lead order mismatch npz={leads} vs OUTPUT_LEADS={OUTPUT_LEADS}")
    med = d["scaler_median"].astype(np.float64)
    iqr = d["scaler_iqr"].astype(np.float64)
    sig = d["signal_12lead"].astype(np.float64)       # physical mV
    fold = d["fold"].astype(int)
    labels = d["label"].astype(int)
    scaled = (sig - med[None, :, None]) / iqr[None, :, None]
    i_I, i_II = leads.index("I"), leads.index("II")
    test = np.flatnonzero(fold == TEST_FOLD)
    print(f"N={len(sig)}  fold-{TEST_FOLD} n={len(test)}  T={sig.shape[2]}")
    print(f"IQR    {dict(zip(leads, np.round(iqr, 6)))}")
    print(f"median {dict(zip(leads, np.round(med, 6)))}")

    # ================================================================= A) coefficients
    scaler = {"median": med, "iqr": iqr}
    co = scaled_einthoven_coeffs(scaler, leads)
    rows = []
    print("\n[A] scaled-space affine coefficients  derived_s = A*I_s + B*II_s + C")
    print(f"  {'lead':5s} {'A(I)':>10s} {'B(II)':>10s} {'C':>10s} | "
          f"{'phys A':>7s} {'phys B':>7s} | {'devA%':>8s} {'devB%':>8s} | {'|d| vs expected':>16s}")
    for k, name in enumerate(DERIVED):
        A, B, C = co[name]
        pa, pb = _D_PHYS[k]
        ea, eb, ec = EXPECTED[name]
        dmax = max(abs(A - ea), abs(B - eb), abs(C - ec))
        devA = 100.0 * (A - pa) / abs(pa)
        devB = 100.0 * (B - pb) / abs(pb)
        print(f"  {name:5s} {A:>10.4f} {B:>10.4f} {C:>10.4f} | {pa:>7.3f} {pb:>7.3f} | "
              f"{devA:>+8.1f} {devB:>+8.1f} | {dmax:>16.2e}")
        if dmax > COEF_TOL:
            fail.append(f"(A) {name}: coefficients differ from the expected values by {dmax:.2e} "
                        f"(tol {COEF_TOL})")
        rows.append({
            "derived_lead": name,
            "coef_I_scaled": A, "coef_II_scaled": B, "const_scaled": C,
            "coef_I_physical": pa, "coef_II_physical": pb, "const_physical": 0.0,
            "deviation_pct_I": devA, "deviation_pct_II": devB,
            "expected_I": ea, "expected_II": eb, "expected_C": ec,
            "max_abs_diff_vs_expected": dmax,
            "iqr_I_mV": iqr[i_I], "iqr_II_mV": iqr[i_II], "iqr_derived_mV": iqr[leads.index(name)],
            "median_I_mV": med[i_I], "median_II_mV": med[i_II],
            "median_derived_mV": med[leads.index(name)],
        })
    coef_df = pd.DataFrame(rows)
    coef_df.to_csv(OUT / "W1M_scaled_coefficients.csv", index=False)

    # CONTROL A1: equal IQR + zero median must collapse the affine map onto the physical one
    flat = {"median": np.zeros(12), "iqr": np.ones(12)}
    co_flat = scaled_einthoven_coeffs(flat, leads)
    ctrl_a1 = max(max(abs(co_flat[n][0] - _D_PHYS[k][0]), abs(co_flat[n][1] - _D_PHYS[k][1]),
                      abs(co_flat[n][2])) for k, n in enumerate(DERIVED))
    print(f"\n  [control A1] iqr=1, median=0  ->  max |scaled coef - physical coef| = {ctrl_a1:.2e}")
    if ctrl_a1 > 1e-12:
        fail.append("(A1) with unit IQR / zero median the scaled coefficients do not reduce to the "
                    "physical ones -- the derivation is wrong")

    # CONTROL A2: precordial scaler constants must not touch the limb derivation
    pert = {"median": med.copy(), "iqr": iqr.copy()}
    pert["median"][6:] += 0.37
    pert["iqr"][6:] *= 2.5
    co_pert = scaled_einthoven_coeffs(pert, leads)
    ctrl_a2 = max(max(abs(np.array(co_pert[n]) - np.array(co[n]))) for n in DERIVED)
    print(f"  [control A2] perturb V1-V6 scaler constants -> max coef change = {ctrl_a2:.2e}")
    if ctrl_a2 > 0.0:
        fail.append("(A2) precordial scaler constants leak into the limb derivation")

    # ============================================== B) counterfactual physical-on-scaled
    print("\n[B] counterfactual: PHYSICAL coefficients applied to scaled signals")
    print("    residual_mV = [(a_phys - A)*I_s + (b_phys - B)*II_s - C] * IQR_derived")
    I_s_obs, II_s_obs = scaled[test, i_I], scaled[test, i_II]

    # the model's own fold-10 reconstruction of (I, II) -- secondary source
    enc, dec_s = load_encoder_decoder(CKPT, n_t=sig.shape[2], rank=LEADFIELD_RANK, device="cpu")
    with torch.no_grad():
        yh, th = [], []
        for s in range(0, len(test), 512):
            xb = torch.from_numpy(scaled[test[s:s + 512]].astype(np.float32))
            z = enc(xb)
            y, _ = dec_s.forward_from_z(z)
            yh.append(y.numpy().astype(np.float64))
            th.append(dec_s.pspace.theta(z).numpy().astype(np.float64))
    y_hat = np.concatenate(yh)
    theta_test = np.concatenate(th)
    I_s_rec, II_s_rec = y_hat[:, i_I], y_hat[:, i_II]

    allrec = np.arange(len(sig))
    res_rows = []
    # PRIMARY = observed fold-10 test set. The all-records row is carried so the
    # whole-cohort maxima quoted in the revision plan can be reconciled.
    sources = (("observed_fold10", (I_s_obs, II_s_obs), test),
               ("d1s_r3_s42_reconstruction_fold10", (I_s_rec, II_s_rec), test),
               ("observed_all_clean_records", (scaled[:, i_I], scaled[:, i_II]), allrec))
    for source, (Is, IIs), sub in sources:
        for k, name in enumerate(DERIVED):
            A, B, C = co[name]
            pa, pb = _D_PHYS[k]
            s_d = iqr[leads.index(name)]
            r_mV = ((pa - A) * Is + (pb - B) * IIs - C) * s_d
            a = np.abs(r_mV)
            # amplitude reference: the observed derived lead in the same records
            ref_ptp = float(np.median(np.ptp(sig[sub, leads.index(name)], axis=1)))
            row = {
                "source": source, "derived_lead": name,
                "n_records": int(len(Is)), "n_samples": int(a.size),
                "median_abs_residual_mV": float(np.median(a)),
                "mean_abs_residual_mV": float(a.mean()),
                "p95_abs_residual_mV": float(np.percentile(a, 95)),
                "p99_abs_residual_mV": float(np.percentile(a, 99)),
                "p99p9_abs_residual_mV": float(np.percentile(a, 99.9)),
                "max_abs_residual_mV": float(a.max()),
                "rms_residual_mV": float(np.sqrt((r_mV ** 2).mean())),
                "median_ptp_of_lead_mV": ref_ptp,
                "median_abs_residual_pct_of_ptp": float(100 * np.median(a) / ref_ptp),
                "max_abs_residual_pct_of_ptp": float(100 * a.max() / ref_ptp),
            }
            res_rows.append(row)
            if source == "observed_fold10":
                print(f"  {name:5s} median={row['median_abs_residual_mV']:.5f} mV  "
                      f"p99.9={row['p99p9_abs_residual_mV']:.5f} mV  "
                      f"max={row['max_abs_residual_mV']:.5f} mV  "
                      f"(median = {row['median_abs_residual_pct_of_ptp']:.2f}% of the lead's "
                      f"median peak-to-peak {ref_ptp:.3f} mV)")
    res_df = pd.DataFrame(res_rows)
    res_df.to_csv(OUT / "W1M_counterfactual_residuals.csv", index=False)

    # CONTROL B1: the CORRECT scaled coefficients must reproduce the observed derived leads
    # as well as the physical coefficients do in physical space (i.e. this is a pure space
    # mismatch, not wrong physics).
    print("\n  [control B1] relative RMS residual against the OBSERVED derived leads, fold-10")
    print(f"  {'lead':5s} {'phys coef in mV space':>22s} {'scaled coef in scaled space':>28s} "
          f"{'phys coef in scaled space':>26s}")
    ctrl_b1 = {}
    for k, name in enumerate(DERIVED):
        i_d = leads.index(name)
        pa, pb = _D_PHYS[k]
        A, B, C = co[name]
        obs_p, obs_s = sig[test, i_d], scaled[test, i_d]
        r_phys = np.sqrt(((obs_p - (pa * sig[test, i_I] + pb * sig[test, i_II])) ** 2).sum()
                         / (obs_p ** 2).sum())
        r_scal = np.sqrt(((obs_s - (A * I_s_obs + B * II_s_obs + C)) ** 2).sum()
                         / (obs_s ** 2).sum())
        r_wrong = np.sqrt(((obs_s - (pa * I_s_obs + pb * II_s_obs)) ** 2).sum()
                          / (obs_s ** 2).sum())
        ctrl_b1[name] = {"physical_space_physical_coef": float(r_phys),
                         "scaled_space_scaled_coef": float(r_scal),
                         "scaled_space_physical_coef": float(r_wrong)}
        print(f"  {name:5s} {r_phys:>22.4f} {r_scal:>28.4f} {r_wrong:>26.4f}")
        if r_wrong <= r_scal:
            fail.append(f"(B1) for {name} the physical coefficients in scaled space are not worse "
                        "than the scaled coefficients -- the disclosure would be vacuous")

    # ==================================================== C) FIM under the two conventions
    print("\n[C] FIM / effective rank under the two derivation conventions, d1s_r3_s42")
    # a second decoder carrying the SAME trained weights but the physical (scaler=None) map
    dec_p = SurrogateDecoder(n_t=sig.shape[2], leadfield_rank=LEADFIELD_RANK, scaler=None)
    dec_p.load_state_dict(dec_s.state_dict())
    dec_p.eval()
    assert dec_s.scaler is not None and dec_p.scaler is None
    assert torch.allclose(dec_s.A, dec_p.A) and torch.allclose(dec_s.B, dec_p.B)

    rng = np.random.default_rng(SEED + TEST_FOLD)
    picks = []
    for ci in range(len(TARGET_CLASSES)):
        cand = np.flatnonzero((labels == ci) & (fold == TEST_FOLD))
        if len(cand) < PER_CLASS:
            raise SystemExit(f"fold {TEST_FOLD} class {ci} has only {len(cand)} records")
        picks.append(rng.choice(cand, PER_CLASS, replace=False))
    pick_global = np.concatenate(picks)
    pos = {g: i for i, g in enumerate(test)}
    theta_sub = np.stack([theta_test[pos[g]] for g in pick_global])
    print(f"  balanced fold-10 subset: {len(pick_global)} records "
          f"({PER_CLASS} per class, rng seed {SEED + TEST_FOLD})")

    n_t = sig.shape[2]
    rows8 = lead_rows(list(NON_DERIVED_ROWS), n_t)
    P = len(FIM_COLS)

    def make64(dec):
        d64 = SurrogateDecoder(n_t=n_t, leadfield_rank=LEADFIELD_RANK, scaler=dec.scaler)
        d64.load_state_dict(dec.state_dict())
        d64 = d64.double().eval()
        for p in d64.parameters():
            p.requires_grad_(False)
        return d64

    def accumulate(dec, tag):
        d64 = make64(dec)
        radius = d64.pspace.radius.double()[FIM_COLS].view(1, -1)

        def f(th):
            return d64.forward(th.unsqueeze(0))[0][0].reshape(-1)

        F8 = torch.zeros(P, P, dtype=torch.float64)
        F12 = torch.zeros(P, P, dtype=torch.float64)
        t0 = time.time()
        for i in range(len(theta_sub)):
            J = jacfwd(f)(torch.tensor(theta_sub[i], dtype=torch.float64))[:, FIM_COLS] * radius
            J8 = J[rows8]
            F8 += J8.T @ J8
            F12 += J.T @ J
            if (i + 1) % 100 == 0:
                print(f"    [{tag}] {i + 1}/{len(theta_sub)} records  ({time.time() - t0:.1f}s)")
        return (F8 / len(theta_sub)).numpy(), (F12 / len(theta_sub)).numpy()

    print("  --- scaled convention (as trained, scaler attached) ---")
    F8_s, F12_s = accumulate(dec_s, "scaled")
    print("  --- physical convention (textbook coefficients on scaled outputs) ---")
    F8_p, F12_p = accumulate(dec_p, "physical")

    def block(Fs, Fp, tag):
        ms, mp = fim_metrics(Fs), fim_metrics(Fp)
        rel_fro = float(np.linalg.norm(Fs - Fp) / np.linalg.norm(Fs))
        keys = ("effective_rank", "rank95", "rank99", "condition_number", "logdet")
        out = {"lead_rows": tag,
               "scaled": {k: ms[k] for k in keys},
               "physical": {k: mp[k] for k in keys},
               "delta_physical_minus_scaled": {k: float(mp[k] - ms[k]) for k in keys},
               "max_abs_entry_diff": float(np.abs(Fs - Fp).max()),
               "relative_frobenius_diff": rel_fro,
               "trace_scaled": float(np.trace(Fs)), "trace_physical": float(np.trace(Fp)),
               "top_eigval_scaled": ms["eig_top"][0], "top_eigval_physical": mp["eig_top"][0]}
        print(f"\n  [{tag}]")
        print(f"    eff.rank  scaled={ms['effective_rank']:.6f}  physical={mp['effective_rank']:.6f}"
              f"  delta={mp['effective_rank'] - ms['effective_rank']:+.6f}")
        print(f"    rank95 {ms['rank95']} -> {mp['rank95']}   rank99 {ms['rank99']} -> {mp['rank99']}")
        print(f"    logdet  {ms['logdet']:.4f} -> {mp['logdet']:.4f}  "
              f"(delta {mp['logdet'] - ms['logdet']:+.4f})")
        print(f"    trace   {np.trace(Fs):.6f} -> {np.trace(Fp):.6f}")
        print(f"    max|dF| = {np.abs(Fs - Fp).max():.6e}   rel ||dF||_F = {rel_fro:.6f}")
        return out

    b8 = block(F8_s, F8_p, "8 non-derived leads (I,II,V1-V6) -- the published FIM convention")
    b12 = block(F12_s, F12_p, "all 12 displayed leads -- the convention of the training loss")

    if b8["max_abs_entry_diff"] != 0.0:
        fail.append("(C) the 8-lead FIM is NOT bitwise identical across conventions -- the "
                    "structural invariance argument is broken")
    if b12["max_abs_entry_diff"] == 0.0:
        fail.append("(C) the 12-lead FIM is identical across conventions -- the convention would "
                    "have no effect on anything computed over the displayed leads")

    # CONTROL C1 (independent code path): on a handful of records, the derived ROWS of each
    # decoder's Jacobian must equal the corresponding linear combination of the (I, II) rows,
    # and the 8 non-derived rows must be bitwise identical across the two decoders.
    N_C1 = 20
    d64s, d64p = make64(dec_s), make64(dec_p)

    def flat(dec):
        return lambda th: dec.forward(th.unsqueeze(0))[0][0].reshape(-1)

    c1_derived, c1_nonderived = 0.0, 0.0
    for i in range(min(N_C1, len(theta_sub))):
        tt = torch.tensor(theta_sub[i], dtype=torch.float64)
        Js = jacfwd(flat(d64s))(tt).numpy()
        Jp = jacfwd(flat(d64p))(tt).numpy()
        c1_nonderived = max(c1_nonderived,
                            float(np.abs(Js[lead_rows(list(NON_DERIVED_ROWS), n_t)]
                                         - Jp[lead_rows(list(NON_DERIVED_ROWS), n_t)]).max()))
        JI, JII = Js[i_I * n_t:(i_I + 1) * n_t], Js[i_II * n_t:(i_II + 1) * n_t]
        for k, name in enumerate(DERIVED):
            r0 = leads.index(name) * n_t
            A, B, _ = co[name]
            pa, pb = _D_PHYS[k]
            c1_derived = max(c1_derived,
                             float(np.abs(Js[r0:r0 + n_t] - (A * JI + B * JII)).max()),
                             float(np.abs(Jp[r0:r0 + n_t] - (pa * JI + pb * JII)).max()))
    scale = max(float(np.abs(F12_s).max()), 1.0)
    print(f"\n  [control C1] over {min(N_C1, len(theta_sub))} records: derived Jacobian rows vs "
          f"the analytic linear image  max|diff| = {c1_derived:.3e}")
    print(f"               8 non-derived Jacobian rows across the two decoders  "
          f"max|diff| = {c1_nonderived:.3e}")
    cross = max(c1_derived, c1_nonderived)
    if c1_derived > 1e-9:
        fail.append(f"(C1) derived Jacobian rows are not the analytic linear image of the "
                    f"(I,II) rows (max diff {c1_derived:.2e})")
    if c1_nonderived != 0.0:
        fail.append(f"(C1) the 8 non-derived Jacobian rows differ across conventions "
                    f"(max diff {c1_nonderived:.2e}) -- they must not")

    # Evidence that the CONVENTION CHANGES TRAINING (not only analysis): the learned lead
    # field of the two lineages differs. eikonal_B was trained WITHOUT scaled_derivation.
    # Deliberately a SUPERSEDED checkpoint: this comparison is the evidence that the
    # scaled derivation changes training, so it needs the pre-change lineage. The one
    # place published code reads a non-d1s_r3 weight file, and it is a read, not a claim.
    try:
        OLD_CKPT = repro.lineage.permitted_artifact("scripts/stats/W1M_scaled_derivation.py")
    except FileNotFoundError as exc:
        print(f"\n  ! {exc}\n  ! the eikonal_B comparison arm is recorded as null", flush=True)
        OLD_CKPT = None
    sv_new = np.linalg.svd(dec_s.H_ind.detach().numpy().astype(np.float64),
                           compute_uv=False).tolist()
    sv_old = None
    if OLD_CKPT is not None:
        _, dec_old = load_encoder_decoder(OLD_CKPT, n_t=n_t, rank=LEADFIELD_RANK, device="cpu")
        sv_old = np.linalg.svd(dec_old.H_ind.detach().numpy().astype(np.float64),
                               compute_uv=False).tolist()
    print(f"\n  [training is affected] top-3 singular values of the learned lead field H:")
    print(f"    d1s_r3_s42 (trained WITH scaled derivation): "
          f"{[round(v, 4) for v in sv_new[:3]]}")
    if sv_old is not None:
        print(f"    eikonal_B  (trained WITHOUT, superseded):     "
              f"{[round(v, 4) for v in sv_old[:3]]}")

    payload = {
        "task": "W1M",
        "superseded_artifact": {
            "path": ("1_I-GraphECG-Surrogate/FINAL_Sensors/Revision_v3/Results/"
                     "review_response/structure_check_summary.json"),
            "field": "max_abs_F_diff_scaled_vs_unscaled_derivation",
            "value": 0.0,
            "checkpoint": "graphmono_encoder_eikonal_B_best.pt (superseded, NOT the paper model)",
            "status": ("recomputed here on d1s_r3_s42; the 0.0 reproduces exactly for the 8 "
                       "non-derived lead rows, but it is a structural identity, not evidence "
                       "of convention-invariance over the 12 displayed leads")},
        "learned_leadfield_singular_values": {
            "d1s_r3_s42_scaled_derivation": sv_new,
            "graphmono_encoder_eikonal_B_physical_derivation": sv_old},
        "lineage": {"checkpoint": repro.lineage.lock()["checkpoints"][42]["path"],  # output-root relative, same in both trees
                    "seed": 42, "leadfield_rank": LEADFIELD_RANK,
                    "scaled_derivation_in_cfg": True,
                    "recon_channels_in_cfg": "absent -> all 12 displayed leads in the loss"},
        "subset": {"fold": TEST_FOLD, "per_class": PER_CLASS,
                   "n_records": int(len(pick_global)), "rng_seed": SEED + TEST_FOLD,
                   "classes": TARGET_CLASSES},
        "fim_settings": {"columns": "0..45 (global_gain dropped: scale gauge fixed)",
                         "jacobian": "torch.func.jacfwd, float64, columns scaled by box radius",
                         "estimator": "F = mean_s J_s^T J_s"},
        "results": [b8, b12],
        "controls": {
            "coefficients_vs_expected_max_abs_diff":
                float(coef_df["max_abs_diff_vs_expected"].max()),
            "unit_iqr_zero_median_reduces_to_physical": ctrl_a1,
            "precordial_scaler_leak": ctrl_a2,
            "relrms_against_observed_derived_leads_fold10": ctrl_b1,
            "derived_jacobian_rows_vs_analytic_linear_image_max_diff": c1_derived,
            "non_derived_jacobian_rows_across_conventions_max_diff": c1_nonderived,
            "analytic_vs_autodiff_max_diff": cross},
        "scope_statement": (
            "The five waveform loss terms in scripts/31_surrogate_encoder.py are computed over "
            "all 12 displayed leads whenever cfg has no recon_channels key; d1s_r3_s42.yaml has "
            "no such key, so recon_rows=None and the four derived leads ARE inside the "
            "reconstruction loss. The derivation convention therefore changes the training "
            "objective and the fitted parameters. The invariance reported here is only that, "
            "for a FIXED trained decoder evaluated at analysis time, the FIM restricted to the "
            "8 non-derived leads is bitwise identical under the two conventions (the derived "
            "rows are simply not in the row set). Over all 12 displayed leads the FIM is NOT "
            "invariant. No claim is made that a model retrained under the other convention "
            "would have the same FIM."),
        "failures": fail,
    }
    (OUT / "W1M_fim_difference.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")

    print("\n" + "=" * 88)
    if fail:
        print("FAILURES:")
        for f in fail:
            print("  -", f)
        return 1
    print("all propositions and controls passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
