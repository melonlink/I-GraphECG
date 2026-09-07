"""Four-superclass definition + single-label cleaning rule (single source of truth).

Used identically for PTB-XL and every external database: keep a record only if its mapped
superclass set hits exactly one of {NORM, MI, STTC, CD}; drop records with hypertrophy (HYP)
overlap or any OTHER/multi-superclass content.
"""
from __future__ import annotations
from typing import Optional, Iterable

from ..config import SUPERCLASSES, EXCLUDE_SUPERCLASS


def clean_single_label(superclass_set: Iterable[str]) -> Optional[str]:
    """Return the single superclass if the record is clean, else None (dropped).

    Rules:
      - drop if HYP present (hypertrophy overlap excluded);
      - keep iff exactly one of the four target superclasses is present;
      - any OTHER code or >1 target superclass -> drop.
    """
    s = set(superclass_set)
    if EXCLUDE_SUPERCLASS in s:
        return None
    targets = s & set(SUPERCLASSES)
    others = s - set(SUPERCLASSES) - {EXCLUDE_SUPERCLASS}
    if others:
        return None
    if len(targets) == 1:
        return next(iter(targets))
    return None
