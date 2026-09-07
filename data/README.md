# data/

Nothing here ships with the archive. Two different things live under this path.

## Raw data (external, read-only)

Not redistributable. Point `ECG_DATA_DIR` at a directory holding the extracted
PTB-XL v1.0.3 release -- `ptbxl_database.csv`, `scp_statements.csv`, `records100/`.
The scripts locate it through `igraphecg.utils.paths.find_ptbxl_root`, which also
accepts the dataset one directory deeper, as `unzip` leaves it.

External cohorts, used for inference only: PhysioNet/CinC Challenge 2021
`training/georgia` and `training/cpsc_2018`.

## processed/ (derived, regenerable)

`scripts/10_prepare_ptbxl.py` writes `processed/ptbxl_medianbeat_clean_100hz.npz`
here, and everything downstream reads it. Run that script first; it is stage 0 of
the command list in `REPRODUCIBILITY.md`.

Pre-trained weights are in `outputs/checkpoints/`, so the analysis chain can be run
without repeating the training step.
