"""Project paths, by name. Every module takes its directories from here.

The roots themselves are resolved by igraphecg.repro.roots -- environment variable, then
repro.yaml at the tree root, then the built-in default -- so this module carries no path
of its own. It used to: DATA_DIR and OUTPUTS_DIR were spelled out here a second time, with
the repository's Chinese output-directory name in the literal, and the archive builder had
to rewrite this file to make it describe the archive. Now the two trees differ in
repro.yaml and nowhere else.

Layout (code and shared data are separate):
  <repo>/                    this repository (the igraphecg package and the paper tree)
    data/processed/          derived, regenerable caches (median beats, theta); gitignored
  ../PUBLIC/data/raw/        raw ECG datasets shared by three programs (read-only, not in
                             any repository); ECG_DATA_DIR overrides
Outputs default to the paper's output root; ECG_OUTPUT_DIR overrides.
"""
from __future__ import annotations

from pathlib import Path

from igraphecg.repro import roots

# <repo>/igraphecg/utils/paths.py -> <repo>
PROJECT_ROOT = roots.repo()

# Raw data: shared, read-only, outside the repository by default.
DATA_DIR = roots.data()
RAW_DIR = DATA_DIR / "raw"
# Derived caches (median beats, theta) live in the repository's data/, not next to the raw
# data, so a regenerable file never lands in the shared tree.
PROCESSED_DIR = roots.processed()

# Run outputs.
OUTPUTS_DIR = roots.outputs()
LOGS_DIR = OUTPUTS_DIR / "logs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"



def ensure_dirs() -> None:
    """Create the output and data directories (idempotent)."""
    for d in [
        RAW_DIR, PROCESSED_DIR,
        LOGS_DIR, CHECKPOINTS_DIR, FIGURES_DIR, TABLES_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def find_ptbxl_root(raw_dir: Path | None = None) -> Path | None:
    """Locate the PTB-XL root (the directory holding ptbxl_database.csv); None if absent.

    Search order (first hit wins):
      1. raw_dir (default RAW_DIR = <data root>/raw), allowing one extra level from an
         unpacked zip;
      2. the data root itself;
      3. the raw/ and top level of sibling data* directories of the data root.

    Step 3 is necessary: on the shared drive PTB-XL actually lives in
    ../PUBLIC/data_PTB-XL/raw/, while ensure_dirs() creates an empty ../PUBLIC/data/raw/,
    so looking only at step 1 misses it. The candidates are bounded -- no multi-GB
    recursive scan of the data root.
    """
    def _under(d: Path) -> Path | None:
        if not d.exists():
            return None
        direct = d / "ptbxl_database.csv"
        if direct.exists():
            return d
        hits = list(d.rglob("ptbxl_database.csv"))
        # the shortest path is the outermost copy
        return min(hits, key=lambda p: len(p.parts)).parent if hits else None

    if raw_dir is not None:
        return _under(Path(raw_dir))

    for cand in (RAW_DIR, DATA_DIR):
        found = _under(cand)
        if found is not None:
            return found

    parent = DATA_DIR.parent
    if parent.exists():
        for sib in sorted(parent.glob("data*")):
            if not sib.is_dir() or sib == DATA_DIR:
                continue
            for cand in (sib / "raw", sib):
                found = _under(cand)
                if found is not None:
                    return found
    return None
