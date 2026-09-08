"""W1K step 3: the PTB-XL column of the STTC label-composition table (Supplementary Table S9).

The Georgia and CPSC2018 columns come from the SNOMED header scan (W1K_scan.py / W1K_addendum.py).
PTB-XL is annotated with SCP-ECG statements instead, so its column is computed here, over the
clean single-label STTC records of the analysed subset (all folds; 2,400 records), with the
label families the table's footnote states:

    T-wave abnormality or inversion   NDT
    non-specific ST or ST-T change    NST_
    territory-coded ischemia          ISC_, ISCAL, ISCAN, ISCAS, ISCIL, ISCIN, ISCLA

A statement counts when it appears in the record's scp_codes at any likelihood, the same
record-level union the cohort columns use. STD_ and STE_ are reported for information only:
PTB-XL files them as form statements with no diagnostic_class, so they are not part of the
STTC superclass and the table prints n/a for them.

Writes runs/v7rev_stats/W1K_ptbxl_sttc_families.csv. Needs the PTB-XL release (ptbxl_database.csv,
scp_statements.csv under the data root) and the median-beat metadata of scripts/10.
"""
from __future__ import annotations

import ast
import csv
import sys as _sys
from pathlib import Path as _Path

import pandas as pd

# reach igraphecg without installation; redundant after `pip install -e .`
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from igraphecg import repro
from igraphecg.utils.paths import find_ptbxl_root

OUT = repro.outdir("runs/v7rev_stats")
META = repro.processed() / "ptbxl_medianbeat_clean_100hz_metadata.csv"

FAMILIES = {
    "T-wave abnormality or inversion": (["NDT"], "STTC superclass"),
    "Non-specific ST or ST-T change": (["NST_"], "STTC superclass"),
    "Territory-coded ischemia": (["ISC_", "ISCAL", "ISCAN", "ISCAS", "ISCIL", "ISCIN", "ISCLA"],
                                 "STTC superclass"),
    "ST depression (form statement)": (["STD_"], "form statement, no diagnostic_class; n/a in Table S9"),
    "ST elevation (form statement)": (["STE_"], "form statement, no diagnostic_class; n/a in Table S9"),
}


def main() -> None:
    raw = find_ptbxl_root()
    if raw is None:
        raise SystemExit("PTB-XL not found: set ECG_DATA_DIR to the directory holding ptbxl_database.csv")
    if not META.exists():
        raise SystemExit(f"median-beat metadata not found at {META}: run scripts/10_prepare_ptbxl.py first")
    db = pd.read_csv(raw / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(raw / "scp_statements.csv", index_col=0)
    meta = pd.read_csv(META)
    sttc_ids = meta.loc[meta["label_name"] == "STTC", "record_id"].astype(int).values
    codes = db.loc[sttc_ids, "scp_codes"].apply(ast.literal_eval)
    n = len(codes)

    # every family member must be a real SCP statement, and the superclass ones must be STTC
    for fam, (members, kind) in FAMILIES.items():
        for c in members:
            if c not in scp.index:
                raise SystemExit(f"{c} is not an SCP statement in scp_statements.csv")
            cls = scp.loc[c, "diagnostic_class"]
            if kind.startswith("STTC") and cls != "STTC":
                raise SystemExit(f"{c} has diagnostic_class {cls!r}, not STTC")
            if kind.startswith("form") and isinstance(cls, str) and cls:
                raise SystemExit(f"{c} carries diagnostic_class {cls!r}; expected a form statement")

    rows = []
    for fam, (members, kind) in FAMILIES.items():
        k = int(sum(any(c in d for c in members) for d in codes))
        rows.append({"family": fam, "scp_codes": "+".join(members), "n_records": k,
                     "pct_of_clean_STTC_records": round(100.0 * k / n, 2), "note": kind})
    rows.append({"family": "Records", "scp_codes": "(clean single-label STTC, all folds)",
                 "n_records": n, "pct_of_clean_STTC_records": 100.0, "note": ""})

    dest = OUT / "W1K_ptbxl_sttc_families.csv"
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(open(dest, encoding="utf-8").read())
    print(f"wrote -> {dest}")


if __name__ == "__main__":
    main()
