"""PTB-XL raw-data loading: database table, scp_statements, and per-record 12-lead waveforms."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# canonical 12-lead order (all outputs use this order)
CANONICAL_LEADS: list[str] = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]
LEAD_TO_IDX: dict[str, int] = {l: i for i, l in enumerate(CANONICAL_LEADS)}


def load_database(ptbxl_root: str | Path) -> pd.DataFrame:
    """Load ptbxl_database.csv, indexed by ecg_id."""
    df = pd.read_csv(Path(ptbxl_root) / "ptbxl_database.csv", index_col="ecg_id")
    return df


def load_scp_statements(ptbxl_root: str | Path) -> pd.DataFrame:
    """Load scp_statements.csv, indexed by SCP diagnostic code."""
    df = pd.read_csv(Path(ptbxl_root) / "scp_statements.csv", index_col=0)
    return df


def _normalize_lead_name(name: str) -> str:
    """Normalize a wfdb header lead name to the CANONICAL_LEADS spelling (case AVR/avr)."""
    n = name.strip()
    upper = n.upper()
    canon = {
        "I": "I", "II": "II", "III": "III",
        "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
        "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6",
    }
    return canon.get(upper, n)


def read_record_signal(
    ptbxl_root: str | Path,
    row: pd.Series,
    sampling_rate: int = 100,
) -> tuple[np.ndarray, list[str]]:
    """Read the waveform of one record; returns (signal[12, T] float32, lead_names).

    PTB-XL waveforms are already in mV. Leads are reordered to CANONICAL_LEADS.
    sampling_rate=100 uses filename_lr, 500 uses filename_hr.
    """
    import wfdb

    fname = row["filename_lr"] if sampling_rate == 100 else row["filename_hr"]
    rec_path = str(Path(ptbxl_root) / fname)
    sig, fields = wfdb.rdsamp(rec_path)  # sig: [T, n_leads]
    sig_names = [_normalize_lead_name(s) for s in fields["sig_name"]]

    # reorder to the canonical lead order
    out = np.full((len(CANONICAL_LEADS), sig.shape[0]), np.nan, dtype=np.float32)
    for i, name in enumerate(sig_names):
        if name in LEAD_TO_IDX:
            out[LEAD_TO_IDX[name]] = sig[:, i].astype(np.float32)
    return out, CANONICAL_LEADS


def assign_official_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Official PTB-XL strat_fold split: train=fold 1-8, val=fold 9, test=fold 10."""
    fold = df["strat_fold"].astype(int)
    split = np.where(fold <= 8, "train", np.where(fold == 9, "val", "test"))
    df = df.copy()
    df["split"] = split
    return df
