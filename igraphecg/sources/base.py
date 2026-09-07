"""Unified ECG data record + source interface (adapter pattern)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator
import numpy as np


@dataclass
class ECGRecord:
    """One raw 12-lead ECG record from any source, before preprocessing."""
    signal_12lead: np.ndarray            # [12, n] in mV, lead order = config.CANONICAL_LEADS
    fs: int                              # native sampling rate (Hz)
    lead_names: list[str]                # length-12 canonical order
    source_labels: list[str] = field(default_factory=list)  # raw codes (SCP / SNOMED-CT)
    record_id: str = ""
    source_name: str = ""


class ECGSource(ABC):
    """Abstract dataset adapter. Add a new database = subclass + register in registry."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def iter_records(self) -> Iterator[ECGRecord]:
        """Yield raw ECGRecord objects (no preprocessing, no relabeling)."""
        ...
