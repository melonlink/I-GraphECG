# data/

Nothing here ships with the repository: this file is the only tracked entry under `data/`.
The datasets are public (PhysioNet, CC BY 4.0) and are obtained from their source;
`README.md`, section *Data*, lists the downloads, versions and requirements.

## Layout

```
data/
  ptbxl/raw/                    PTB-XL v1.0.3, extracted (read-only)
    ptbxl_database.csv
    scp_statements.csv
    records100/                 100 Hz WFDB records; records500/ is not read
  challenge_2021/training/      PhysioNet/CinC Challenge 2021 v1.0.3 (read-only)
    georgia/g1 ... g11/         E*.hea + E*.mat, 10,344 records
    cpsc_2018/g1 ... g7/        A*.hea + A*.mat, 6,877 records
  processed/                    derived cache, written by the scripts
```

`repro.yaml` declares `data_root: data/ptbxl`; PTB-XL is found under its `raw/`, also one
directory deeper as `unzip` leaves it. `ECG_DATA_DIR` points elsewhere instead.

## processed/ (derived, regenerable)

`scripts/10_prepare_ptbxl.py` writes `ptbxl_medianbeat_clean_100hz.npz` (the clean
four-class subset, N = 15,709) and `scripts/11_prepare_ptbxl_multilabel.py`
`ptbxl_medianbeat_multilabel_100hz.npz`; everything downstream reads them. Run both first:
they are stage 0 of the command list in `REPRODUCIBILITY.md`. Both are deterministic, so
the cache regenerates byte for byte from the same PTB-XL release.

The trained encoders are in `outputs/checkpoints/`, so the analysis chain runs without
repeating the training step.
