# I-GraphECG

Code, configurations, trained encoder checkpoints and derived per-record tables for:

> **I-GraphECG: Observability-Based Lead Selection and Interpretable Disease Prediction
> from the 12-Lead ECG Using a Gray-Box Graph Electrophysiology Surrogate.**
> Limin Zhao, Hongtao Xu, Pengjian Wang, Weicheng Fu, Ningning Zhang. *Sensors*, 2026
> (accepted; in press).

An encoder maps a 12-lead median beat to a bounded 47-dimensional set of equivalent
electrophysiological descriptors. A constrained eight-node conduction-graph decoder --
eikonal-inspired activation on a directed tree plus a low-rank lead field -- reconstructs
the signal from those descriptors. Disease prediction and reduced-lead selection are then
computed on the descriptors rather than on the waveform, which is what lets lead choice be
posed as observability-based sensor selection on the Fisher information.

MIT licensed. See `LICENSE`; the datasets are not covered by it and are not redistributed.

## What reproduces without downloading anything

Most of it. `outputs/` ships the 35 run directories that `paper.lock.yaml` lists -- 340
files, every table the reported numbers come from, at the same output-root-relative
paths the scripts read -- and `outputs/checkpoints/` the five trained encoders (seeds 42,
1-4), so
**every classification, identifiability and lead-selection result in the paper reproduces
from this archive alone -- no raw recordings, no GPU.** The test suite runs with no data
present at all:

```bash
pip install -e .            # or: pip install -r requirements.txt
python -m pytest tests/ -q
```

What does need the raw data: rebuilding the median-beat dataset (`scripts/10_prepare_ptbxl.py`),
retraining an encoder (`scripts/31_surrogate_encoder.py`), and the external-cohort
evaluation. `REPRODUCIBILITY.md` gives the full command list in order, and section 6 maps
every table and figure in the paper to the run directory that backs it.

## Getting the data, if you want the full chain

| Dataset | Used for | Where |
|---|---|---|
| PTB-XL v1.0.3, 100 Hz | training and the main analysis | https://physionet.org/content/ptb-xl/ |
| PhysioNet/CinC Challenge 2021, `training/georgia` | external validation, inference only | https://physionet.org/content/challenge-2021/ |
| CPSC2018 | external validation, inference only | http://2018.icbeb.org/Challenge.html |

Point `ECG_DATA_DIR` at the directory holding the extracted PTB-XL release. The loader
also accepts it one level deeper, as `unzip` tends to leave it. The Challenge copies of
PTB-XL and PTB are never read, so the external cohorts stay disjoint from training.

## Layout

```
igraphecg/          the package: models, evaluation, data handling, external cohorts
scripts/            the pipeline, numbered in execution order (see REPRODUCIBILITY.md)
configs/            YAML for each stage; configs/v1fix/ is the reported analysis lineage,
                    configs/ablation/ the structural-ablation variants (Supplementary Table S20)
tests/              runs with no data present; test_manuscript_numbers.py checks every headline
                    number of the manuscript against the shipped output tables (when the
                    manuscript sources sit beside the code as paper/; otherwise it skips)
outputs/            the published run directories, exactly as paper.lock.yaml lists them,
                    plus checkpoints/ with the five trained encoders
results/            patient_map.csv, the PTB-XL record-to-patient map the tests use
paper.lock.yaml     the lineage: seeds, checksummed weights, the run-directory registry
repro.yaml          where the roots are in THIS tree; the one file that differs from the
                    repository
data/               empty; see data/README.md for what belongs where
```

## Deliberately not included

- **Raw recordings.** Not redistributable; the links above are the sources.
- **Superseded model lineages.** Earlier revisions explored rank-5/6/8 lead fields and an
  unscaled Einthoven derivation. Only the reported lineage (rank 3, scaled derivation,
  stabilized initialization) ships, so nothing here corresponds to a number the paper
  does not report.
- **Unpublished work.** A coupled hemodynamic/mechanical branch was explored and is not
  part of this paper.
- **The manuscript.** The article and its Supplementary Materials are published by the
  journal; their sources are not part of this archive. `paper.lock.yaml` and section 6 of
  `REPRODUCIBILITY.md` map every table and figure to the files here.

## Known limitations

- Myocardial infarction is under-detected (recall about 0.49). The model must not be used
  as a stand-alone rule-out for infarction. Section 4 of the paper explains why: the ST
  source has low observability in this parameterization.
- The classifier is fitted on a clean single-label four-class subset, an easier task than
  the standard multi-label PTB-XL benchmark. Absolute AUROC is therefore not comparable to
  leaderboard numbers; Section 3.8 quantifies the gap on the multi-label task.
- Disease prediction degrades on external cohorts, and infarction could not be tested
  externally at all -- neither cohort labels it usably. Reconstruction and the
  lead-selection conclusions do transfer.
- `xgboost` is optional but the reported numbers come from it; without it the code falls
  back to `HistGradientBoosting`, which does not reproduce the tables.

## Citation

See `CITATION.cff`, or cite the paper above. Release `1.0` is the state of this repository at
acceptance: the code, checkpoints and output tables behind every number in the published article.
