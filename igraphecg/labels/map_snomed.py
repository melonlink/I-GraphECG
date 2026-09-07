"""SNOMED-CT -> four-superclass mapping for Challenge 2020/2021 records.

The mapping table lives in artifacts/label_maps/snomed_to_superclass.csv
(columns: snomed_code, term, mapped_superclass, note) so it is reviewable and can be
attached to the revision as supplementary material -- it is NOT hard-coded in logic.

Mapping is anchored on PTB-XL's SCP->superclass scheme; codes with hypertrophy are marked
HYP (used for exclusion); codes that do not correspond to the four classes are OTHER
(records carrying them are dropped together with multi-label records).

NOTE (for the manuscript): this mapping is defined by this study and may differ from the
source database's labelling conventions; it is a known noise source of cross-database transfer.
"""
from __future__ import annotations
import csv
from pathlib import Path

from ..config import LABELMAP_DIR
from .superclass import clean_single_label

DEFAULT_MAP_CSV = LABELMAP_DIR / "snomed_to_superclass.csv"


def load_snomed_map(csv_path=DEFAULT_MAP_CSV) -> dict[str, str]:
    """Load {snomed_code -> superclass|'HYP'|'OTHER'} from the reviewable CSV."""
    m: dict[str, str] = {}
    p = Path(csv_path)
    if not p.exists():
        return m
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = str(row["snomed_code"]).strip()
            if code and not code.startswith("#"):
                m[code] = str(row["mapped_superclass"]).strip()
    return m


def map_record_label(snomed_codes, snomed_map: dict[str, str]):
    """Map a record's SNOMED codes to a clean single superclass (or None to drop)."""
    mapped = {snomed_map.get(str(c).strip()) for c in snomed_codes}
    mapped.discard(None)
    return clean_single_label(mapped)
