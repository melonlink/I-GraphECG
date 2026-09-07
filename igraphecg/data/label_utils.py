"""PTB-XL label mapping: scp_codes -> diagnostic superclass, and the single-label clean subset.

Stage-1 target classes: NORM / MI / STTC / CD (HYP is not included).
"""
from __future__ import annotations

import ast
from typing import Iterable

import pandas as pd

TARGET_CLASSES: list[str] = ["NORM", "MI", "STTC", "CD"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: c for c, i in CLASS_TO_IDX.items()}


def parse_scp_codes(raw: str | dict) -> dict[str, float]:
    """The scp_codes field of ptbxl_database.csv is a string such as "{'NORM':100.0,'SR':0.0}"."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        d = ast.literal_eval(raw)
        return {str(k): float(v) for k, v in d.items()} if isinstance(d, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def build_code_to_superclass(scp_statements: pd.DataFrame) -> dict[str, str]:
    """Build {scp_code: diagnostic_class} from scp_statements.csv, diagnostic==1 codes only."""
    df = scp_statements
    if "diagnostic" in df.columns:
        df = df[df["diagnostic"] == 1]
    mapping: dict[str, str] = {}
    for code, row in df.iterrows():
        sc = row.get("diagnostic_class")
        if isinstance(sc, str) and sc:
            mapping[str(code)] = sc
    return mapping


def record_superclasses(
    scp_codes: dict[str, float],
    code_to_super: dict[str, str],
    min_likelihood: float = 0.0,
) -> set[str]:
    """Return all diagnostic superclasses of the record (including non-target ones, e.g. HYP).

    min_likelihood: likelihood threshold on scp_codes (0-100). The default 0 keeps every code
    present, matching the aggregation of the Wagner et al. (2020) PTB-XL benchmark.
    """
    supers: set[str] = set()
    for code, like in scp_codes.items():
        if like < min_likelihood:
            continue
        sc = code_to_super.get(code)
        if sc:
            supers.add(sc)
    return supers


def assign_single_label(
    database: pd.DataFrame,
    code_to_super: dict[str, str],
    min_likelihood: float = 0.0,
    exclude_hyp_overlap: bool = True,
) -> pd.DataFrame:
    """Add the superclasses / label_name / label / is_clean columns to database.

    is_clean (criterion for the single-label clean subset):
      - exactly one hit among the four target classes {NORM,MI,STTC,CD};
      - if exclude_hyp_overlap=True, the record must not also carry HYP (unambiguous label).
    """
    target_set = set(TARGET_CLASSES)
    rows = []
    for ecg_id, row in database.iterrows():
        codes = parse_scp_codes(row.get("scp_codes"))
        supers = record_superclasses(codes, code_to_super, min_likelihood)
        targets = supers & target_set
        has_hyp = "HYP" in supers
        is_clean = (len(targets) == 1) and not (exclude_hyp_overlap and has_hyp)
        label_name = next(iter(targets)) if is_clean else None
        rows.append({
            "ecg_id": ecg_id,
            "superclasses": ";".join(sorted(supers)) if supers else "",
            "n_target_super": len(targets),
            "has_hyp": has_hyp,
            "is_clean": is_clean,
            "label_name": label_name,
            "label": CLASS_TO_IDX.get(label_name, -1) if label_name else -1,
        })
    extra = pd.DataFrame(rows).set_index("ecg_id")
    return database.join(extra)


def class_count_table(df: pd.DataFrame, splits: Iterable[str] = ("train", "val", "test")) -> pd.DataFrame:
    """Build a split x class count table (df must have 'split' and 'label_name' columns)."""
    out = {}
    for sp in splits:
        sub = df[df["split"] == sp]
        out[sp] = {c: int((sub["label_name"] == c).sum()) for c in TARGET_CLASSES}
        out[sp]["total"] = int(len(sub))
    table = pd.DataFrame(out).T[TARGET_CLASSES + ["total"]]
    return table
