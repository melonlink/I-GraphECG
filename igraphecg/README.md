# igraphecg — single, inference-ready I-GraphECG package

Unified package: research core (data/features/models/training/evaluation) + delivery layer
(sources/preprocess/inference/external/labels). **Any 12-lead ECG database** can be run through
the frozen PTB-XL-trained model with **zero retraining** (external validation = inference only).
PTB-XL behavior is locked by `tests/test_regression_ptbxl.py`.

## Layout
```
igraphecg/
  config.py              frozen constants (= manuscript / Table S3) + ref values + paths
  __init__.py            public API (lazy): make_median_beat, load_model, predict_theta, reconstruct, get_source
  inference.py           load + predict + reconstruct (eval-only; train guard)
  # --- research core (formerly src/) ---
  data/        beat_extraction, dataset, label_utils, preprocess, ptbxl_loader
  features/    ecg_features, morphology
  models/      phys_encoder, surrogate_decoder, leadfield, phys_features, resnet1d, ap_ode
  training/    losses
  evaluation/  identifiability, lead_ablation, metrics, recon_metrics, round2_io, stability,
               stats_tests, st/repolarization features, plots
  utils/       paths, logging, config, seed
  # --- delivery layer ---
  sources/     base(ECGRecord/ECGSource), ptbxl, wfdb_source, registry   (external DB adapters)
  preprocess/  pipeline: resample→bandpass→R-align→median beat→robust scaling
  external/    metrics, observability, external_eval, classify_external
  labels/      superclass(clean_single_label), map_snomed
  artifacts/   scalers/ label_maps/(snomed_to_superclass.csv)
```
(Regression + unit tests live in the repo-root `tests/`.)

## Key rules
- External databases reuse `artifacts/scalers/ptbxl_train_signal.pkl` (PTB-XL train-fit);
  refitting on external data is forbidden (asserted).
- Single-label cleaning + 4-superclass definition are one function (`labels/superclass.py`),
  used identically for all databases.
- SNOMED→superclass map is a reviewable CSV, not hard-coded.

## Run
```bash
# regression test (from repo root, myPyTorch env): reproduces paper PTB-XL numbers
python -m pytest tests/test_regression_ptbxl.py

# external validation (inference only)
python -m igraphecg.external.external_eval --dataset georgia --data_dir <challenge_2021/training/georgia> \
  --out <ext_georgia.json>          # checkpoint, scaler and label map default to the published ones
```
Adding a database = register it in `sources/registry.py` + complete the SNOMED CSV. Exclude the
Challenge corpus's PTB-XL/PTB subsets from external sets.
