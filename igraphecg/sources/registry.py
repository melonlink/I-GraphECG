"""Source registry: get a dataset adapter by name.

Add a new external database by registering a builder here + a SNOMED label map;
no other code needs to change.
"""
from __future__ import annotations
from .base import ECGSource
from .ptbxl import PTBXLSource
from .wfdb_source import WFDBSource

# Challenge-2021 WFDB sub-databases (all read through the generic WFDB adapter).
# IMPORTANT (per command): the Challenge corpus contains PTB-XL and a PTB subset;
# those MUST be excluded when assembling external-validation sets.
_WFDB_SETS = {"cpsc", "cpsc2", "cpsc_2018", "cpsc2018", "cpsc_2018_extra",
              "georgia", "chapman", "chapman_shaoxing", "ningbo",
              "st_petersburg_incart", "ptb", "ptb-xl", "ptbxl_challenge"}


def get_source(name: str, data_dir=None, **kwargs) -> ECGSource:
    name = name.lower()
    if name == "ptbxl":
        return PTBXLSource(**({} if data_dir is None else {"npz_path": data_dir}), **kwargs)
    if name in _WFDB_SETS:
        if data_dir is None:
            raise ValueError(f"source '{name}' requires data_dir")
        return WFDBSource(data_dir, source_name=name)
    raise KeyError(f"unknown source '{name}'. known: ptbxl, {sorted(_WFDB_SETS)}")


def available_sources() -> list[str]:
    return ["ptbxl"] + sorted(_WFDB_SETS)
