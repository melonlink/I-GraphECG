"""Generic WFDB adapter for PhysioNet/CinC Challenge 2020/2021 records (.hea + .mat/.dat).

Reads raw multi-lead signals, reorders leads to the canonical 12-lead order, and parses
the diagnosis codes (SNOMED-CT) from the header `# Dx:` comment line. No preprocessing,
no relabeling here — that happens in the frozen pipeline / label maps downstream.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np

from ..config import CANONICAL_LEADS
from .base import ECGRecord, ECGSource

# lead-name normalization (header names may be upper/lower case)
_NORM = {l.upper(): l for l in CANONICAL_LEADS}


def _canon_lead(name: str) -> str | None:
    return _NORM.get(name.strip().upper())


def _parse_dx(header_comments) -> list[str]:
    dx: list[str] = []
    for c in header_comments:
        c = c.strip()
        if c.lower().startswith("dx:"):
            dx = [t.strip() for t in c.split(":", 1)[1].split(",") if t.strip()]
    return dx


class WFDBSource(ECGSource):
    """Iterate all WFDB records under a directory (recursively)."""
    already_median_beat = False

    def __init__(self, data_dir, source_name: str = "wfdb"):
        self.data_dir = Path(data_dir)
        self._source_name = source_name

    @property
    def name(self) -> str:
        return self._source_name

    def _record_paths(self):
        return sorted(p.with_suffix("") for p in self.data_dir.rglob("*.hea"))

    def iter_records(self) -> Iterator[ECGRecord]:
        import wfdb
        for stem in self._record_paths():
            rec = wfdb.rdrecord(str(stem))
            sig = np.asarray(rec.p_signal, dtype=np.float32).T   # [leads, n]
            names = [str(s) for s in rec.sig_name]
            # reorder/select into canonical 12-lead; missing leads -> NaN row
            out = np.full((12, sig.shape[1]), np.nan, dtype=np.float32)
            for row, nm in zip(sig, names):
                cl = _canon_lead(nm)
                if cl is not None:
                    out[CANONICAL_LEADS.index(cl)] = row
            dx = _parse_dx(getattr(rec, "comments", []) or [])
            yield ECGRecord(
                signal_12lead=out,
                fs=int(rec.fs),
                lead_names=list(CANONICAL_LEADS),
                source_labels=dx,                 # SNOMED-CT codes
                record_id=stem.name,
                source_name=self._source_name,
            )
