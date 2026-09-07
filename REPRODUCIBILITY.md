# REPRODUCIBILITY

Steps to reproduce every result in the manuscript. All commands run from the repository
root. All reported numbers come from one model lineage: five encoders (seeds 42, 1, 2,
3, 4) trained with the scaled-space Einthoven derivation and stabilized initialization
(`d1s_r3_s{42,1,2,3,4}`); seed 42 is the locked model.

## 1. Data
- PTB-XL v1.0.3 (PhysioNet), 100 Hz (`records100`). Place the extracted archive under the
  shared data root (`ECG_DATA_DIR`), containing `ptbxl_database.csv`, `scp_statements.csv`,
  `records100/`.
- External cohorts (inference only): PhysioNet/CinC Challenge 2021 `training/georgia` and
  `training/cpsc_2018`. The Challenge copies of PTB-XL/PTB are never used (no leakage).

## 2. Clean subset
Records matching exactly one target superclass in {NORM, MI, STTC, CD};
`exclude_hyp_overlap=true`; N = 15,709. Official `strat_fold`: train 1–8 (12,542),
validation 9 (1,573), test 10 (1,594).

## 3. Preprocessing
Median beat window [-300, +700] ms, R peak at index 30; 0.5–40 Hz zero-phase band-pass
(4th-order Butterworth); R detection NeuroKit2 on lead II with V5/multi-lead fallback;
lead-wise robust scaling (median/IQR) fitted on folds 1–8 only.

## 4. Environment
conda env `myPyTorch`: Python 3.10, torch 2.9 (CUDA), numpy 2.2, scipy 1.15, pandas 2.3,
scikit-learn 1.7, wfdb 4.3, xgboost 3.2, neurokit2 0.2.13.

## 5. End-to-end commands
```bash
PY=<myPyTorch python>; export PYTHONUTF8=1

# Stage 0: data
"$PY" scripts/10_prepare_ptbxl.py --config configs/ptbxl_data.yaml
"$PY" scripts/20_baseline_features.py --config configs/baseline_features.yaml
"$PY" scripts/21_baseline_cnn.py      --config configs/baseline_cnn.yaml

# Encoders (five seeds; scaled_derivation: true, head_init_scale 0.01, grad_clip 1.0)
"$PY" scripts/31_surrogate_encoder.py --config configs/d1s_r3_s42.yaml   # + s1..s4
# Oracle capacity check (fixed decoder)
"$PY" scripts/30_surrogate_oracle.py  --config configs/v1fix/oracle_s42.yaml

# Main analysis chain, per seed S in {42,1,2,3,4}
"$PY" scripts/40_descriptors.py       --config configs/v1fix/r3_sS.yaml
"$PY" scripts/41_stability.py    --config configs/v1fix/r3_sS.yaml
"$PY" scripts/42_classification.py --config configs/v1fix/r3_sS.yaml
"$PY" scripts/43_identifiability.py       --config configs/v1fix/r3_sS.yaml
"$PY" scripts/50_observability_phase1.py  --ckpt <d1s_r3_sS ckpt> --out <runs/v1fix_phase1_sS>
"$PY" scripts/51_leadset8_exhaustive.py --ckpt <d1s_r3_sS ckpt> --out <runs/v1fix_ls8_sS>
"$PY" scripts/52_leadset8_byclass.py       --ckpt <d1s_r3_sS ckpt> --out <runs/v1fix_byclass_sS>

# Noise-model sensitivity, CD descriptor sweep, lead-set retrains, single-lead sanity
"$PY" scripts/53_noise_model_sensitivity.py
"$PY" scripts/54_cd_descriptors.py
"$PY" scripts/55_leadset_classification.py
"$PY" scripts/44_single_lead_sanity.py --config configs/v1fix/round5_s42.yaml

# External validation (inference only; PTB-XL scaler and label map frozen)
"$PY" -m igraphecg.external.external_eval --dataset georgia  --data_dir <challenge_2021/training/georgia>  --checkpoint <d1s_r3_s42> --out <ext_georgia.json>
"$PY" -m igraphecg.external.external_eval --dataset cpsc2018 --data_dir <challenge_2021/training/cpsc_2018> --checkpoint <d1s_r3_s42> --out <ext_cpsc2018.json>
"$PY" scripts/71_external_classification.py --theta <ext_georgia.theta.npz>  --dataset georgia  --out <cls_georgia.json>
"$PY" scripts/71_external_classification.py --theta <ext_cpsc2018.theta.npz> --dataset cpsc2018 --out <cls_cpsc2018.json>

# Multi-label zero-shot check (Section 3.9, Table S13): encoder and decoder frozen,
# frozen PTB-XL scaler reused, only the one-vs-rest classifier fitted
"$PY" scripts/11_prepare_ptbxl_multilabel.py
"$PY" scripts/70_multilabel_eval.py

# Revision statistics: Table 6 and Supplementary Tables S7-S13.
# Each family is ordered; the _finalize / _summarize / _headline steps read what the
# steps above them wrote. All of them write into <output root>/runs/v7rev_stats.
"$PY" scripts/stats/W1A_per_record_recon.py        # per-record reconstruction, 5 seeds
"$PY" scripts/stats/W1A_window_context.py
"$PY" scripts/stats/W1B_error_propagation.py       # does reconstruction error propagate?
"$PY" scripts/stats/W1B_bandsplit.py               # appends to W1B_substitution_oracle.csv, so after it
"$PY" scripts/stats/W1B_finalize.py
"$PY" scripts/stats/W1C_fidelity.py                # descriptor fidelity, recon vs observed -> Table S(descfidelity)
"$PY" scripts/stats/W1C_bands.py
"$PY" scripts/stats/W1C_finalize.py
"$PY" scripts/stats/W1G_recompute_fim.py           # identifiability tiers -> Table 6
"$PY" scripts/stats/W1G_crb_leadset_check.py
"$PY" scripts/stats/W1G_build_table.py
"$PY" scripts/stats/W1G_diagnostics.py
"$PY" scripts/stats/W1G_finalize.py
"$PY" scripts/stats/W1J_leadsets_by_noise_model.py # exhaustive 56-subset selection -> Fig 7
"$PY" scripts/stats/W1J_precision_audit.py
"$PY" scripts/stats/W1M_scaled_derivation.py       # scaled-space Einthoven coefficients
"$PY" scripts/stats/W1O_teacherfree_leadsets.py    # teacher-free control, Section 3.7
"$PY" scripts/stats/W1O_summarize.py
"$PY" scripts/stats/W1O_headline.py
"$PY" scripts/stats/W1F_headline.py                # fold-10 disposition + headline, after 70_multilabel_eval
"$PY" scripts/stats/W1F_multilabel_numeric_provenance.py   # after scripts 101 and 102
"$PY" scripts/stats/w1_stats.py                    # MI operating points, paired ablation FDR
"$PY" scripts/stats/w1e_inclusion.py
# External cohort label composition (Supplementary): header-only SNOMED scan of the
# Challenge-2021 georgia/ and cpsc_2018/ directories (--challenge_dir or ECG_CHALLENGE_DIR).
"$PY" scripts/stats/W1K_scan.py --challenge_dir <challenge_2021/training>
"$PY" scripts/stats/W1K_addendum.py

# Figures. One command regenerates every data figure the manuscript includes, into
# <output root>/figures/. Figures 6-7 come from scripts/83_fig_identifiability_selection.py (the published
# versions); 61_split draws the earlier Revision-v6 versions and is not used here.
# fig_overview (Figure 1) is a hand-authored schematic with no generator.
# tests/test_figures_match_manuscript.py compares what this command draws with the files the
# manuscript includes, rasterised at 150 dpi: Figures 2-7 and S2 are pixel-identical; S1 is
# redrawn from the same table and differs only at anti-aliasing level (0.68% of pixels, <=17/255).
"$PY" scripts/89_make_figures.py                      # all of Figures 2-7 and S1-S2
"$PY" scripts/89_make_figures.py --refresh-assets     # also re-run 82_fig_identifiability_assets first
```

## 6. Table/figure provenance (main text)
- Table 1: dataset split (fold x class counts of `runs/v1fix_r3_s42/tables/round3_features.csv`)
  + black-box baselines `tables/baseline_features_metrics.csv`, `tables/baseline_cnn_metrics.csv`
  (scripts 20 and 21 on the script-10 cache; both reproduce the printed AUROCs exactly).
- Table 2 / Figure 2: `runs/d1s_r3_s*/tables/reconstruction_*` (test split) + oracle run;
  Figure 2 drawn by `scripts/80_fig_recon.py`.
- Tables 3–4 / Figures 3–4: `runs/v1fix_r3_s*/tables/round3_classification_*`;
  Figures 3–4 drawn in `scripts/89_make_figures.py` from those tables.
- Table 5 / Figure 5: `runs/v1fix_r3_s42/tables/round3_features.csv` (signed Cliff's
  delta over five seeds) + `v1fix/runs/cd_descriptors` (QRS descriptor sweep).
- Table 6: `runs/v7rev_stats/W1G_parameter_table.csv` (identifiability tiers of the 47
  parameters; tier rule = across-seed minimum), from `scripts/stats/W1G_*`.
- Table 7 / Figures 6–7: `runs/v1fix_phase1_s*` (effective rank, CRB, group energy),
  Figures 6–7 drawn by `scripts/83_fig_identifiability_selection.py` from `v1fix/runs/fig34` (47) and
  `runs/v7rev_stats/W1J_*` (scripts/stats);
  `runs/v1fix_ls8_s*` and `runs/v1fix_byclass_s*` (exhaustive selection, bootstrap),
  `runs/v1fix_leadclf` (lead-set-specific classification, patient-clustered bootstrap).
- Table S13: `runs/v7rev_stats/W1F_multilabel_*.csv` (scripts 101, 102;
  zero-shot multi-label, encoder frozen).
- Tables 8–9: `v1fix/external/` (Georgia, CPSC2018; zero-shot) and
  `runs/v7rev_stats/W1K_*` (cohort label composition).
- Supplementary Tables S7–S13: `runs/v7rev_stats/` (w1d, w1e, W1K, W1M, W1G, W1C, W1F),
  from `scripts/stats/` — see the command list in section 5.

> Cross-references to main-text numbers in this file and in `supplementary.tex` are
> literal, not `\ref`. Re-check them whenever a float or subsection is inserted.
- Table S2: `runs/v1fix_round5`. Noise-model paragraph: `runs/v1fix_noise_fim`.

## 7. Seeds and audit protocol
Seed 42 locked in advance; seeds 1–4 for replication. Folds 1–8 for fitting, fold 9 for
design and for any operating-point selection, fold 10 held out until all model and feature
choices were locked and thereafter scored, never searched over: the myocardial-infarction
decision rules of Section 3.4 select their threshold on fold 9 and apply it unchanged. Bootstrap: 200 resamples (lead selection, record-level),
1000 resamples (classification, patient-clustered; external record-level).

## 8. Known limitations of reproduction
- GPU nondeterminism can shift third-decimal metrics; seeds fix the order of magnitude.
- The external cohorts are not redistributed. Where the Challenge-2021 training data is
  present, the documented commands regenerate `v1fix/external/` exactly (checked 2026-09-06:
  every number, every theta array; only the wall-clock field differs).
- The strict Floquet analysis does not converge (0% periodic); only the finite-time
  recovery descriptor is reported (Table S1), supporting no main-text claim.
