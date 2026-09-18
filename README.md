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

MIT licensed. See `LICENSE`; the datasets are not covered by it.

## What the repository contains, and what it does not

The repository holds code, configurations, the trained encoders and the result tables. It
holds **no dataset material**: no recordings, no copy of the PTB-XL or Challenge metadata
tables, no label or patient list taken from them. Record identifiers, patient identifiers
and class labels appear in `outputs/` only as columns of per-record result tables.

`outputs/` ships the 35 run directories that `paper.lock.yaml` lists -- every table the
reported numbers come from, at the output-root-relative paths the scripts read -- and
`outputs/checkpoints/` the five trained encoders (seeds 42, 1-4). The test suite checks
the shipped tables and runs with no data present:

```bash
pip install -e .            # or: pip install -r requirements.txt
python -m pytest tests/ -q
```

Running the pipeline itself needs the datasets below. `REPRODUCIBILITY.md` gives the full
command list in order, and its section 6 maps every table and figure in the paper to the
run directory behind it.

## Data

Three public datasets, all from PhysioNet under CC BY 4.0. Download them from the source
and place them under `data/` exactly as shown; nothing else needs configuring.

| Dataset | Required content | Records | Used for |
|---|---|---|---|
| [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) | `ptbxl_database.csv`, `scp_statements.csv`, `records100/` (100 Hz WFDB; `records500/` is not read) | 21,799 | training, every main analysis, patient-clustered bootstrap |
| [Challenge 2021 v1.0.3](https://physionet.org/content/challenge-2021/1.0.3/), `training/georgia/` | `g1/` ... `g11/`, each `E*.hea` + `E*.mat` | 10,344 | external validation, inference only |
| Challenge 2021 v1.0.3, `training/cpsc_2018/` | `g1/` ... `g7/`, each `A*.hea` + `A*.mat` | 6,877 | external validation, inference only |

```
data/
  ptbxl/raw/                    PTB-XL v1.0.3, extracted
    ptbxl_database.csv
    scp_statements.csv
    records100/00000/00001_lr.hea, 00001_lr.dat, ...
  challenge_2021/training/
    georgia/g1/E00001.hea, E00001.mat, ...    (g1 ... g11)
    cpsc_2018/g1/A0001.hea, A0001.mat, ...    (g1 ... g7)
  processed/                    written by scripts 10 and 11; never filled by hand
```

Requirements on the sources:

- **PTB-XL must be v1.0.3.** The clean four-class subset (N = 15,709) and the official
  `strat_fold` split are defined on it; another version changes the record set. Only the
  100 Hz records are read. The loader also finds the release one directory deeper, as
  `unzip` leaves it (`data/ptbxl/raw/ptb-xl-a-large-.../ptbxl_database.csv` works).
- **The external cohorts must be the Challenge 2021 copies** (WFDB header with SNOMED-CT
  `Dx` codes), not the original CPSC2018 distribution, whose files and label format
  differ. Only `georgia/` and `cpsc_2018/` are read; the Challenge copies of PTB-XL and PTB
  are never used, so the external cohorts stay disjoint from training.
- Keep the files as released: the code reads them in place and never modifies them.

What needs which dataset:

| Needs | Commands (`REPRODUCIBILITY.md`, section 5) |
|---|---|
| nothing | `pytest tests/`; reading the tables in `outputs/` |
| PTB-XL | scripts 10 and 11 first: they build `data/processed/`, which the later stages read; the patient-clustered bootstraps also read `patient_id` from `ptbxl_database.csv` |
| Challenge 2021 | `igraphecg.external.external_eval`, script 71, `scripts/stats/W1K_scan.py` |

Elsewhere on disk: the roots are declared in `repro.yaml`. `ECG_DATA_DIR` overrides the
PTB-XL location (any directory at or above `ptbxl_database.csv`);
`external_eval` takes the cohort directory as `--data_dir`, and `W1K_scan.py` takes
`--challenge_dir` or `ECG_CHALLENGE_DIR`. A script that does not find its data stops with
a message naming the path it searched, before writing anything.

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
paper.lock.yaml     the lineage: seeds, checksummed weights, the run-directory registry
repro.yaml          where the roots are in THIS tree; the one file that differs from the
                    repository
data/               not tracked except data/README.md; the datasets go here (section Data)
```

## Deliberately not included

- **Dataset material.** Neither recordings nor metadata tables; section Data gives the
  sources and where they go.
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
