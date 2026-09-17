"""A built wheel must carry the package's data files, not only its modules.

`igraphecg` ships two files that are read at run time -- the SNOMED-to-superclass map used by
`igraphecg.labels.map_snomed` and the training-signal scaler used by
`igraphecg.external.external_eval`. Both live one directory below `igraphecg/artifacts/`, which
the original `package-data` pattern `artifacts/*` did not match: `pip install -e .` worked
because it points at the source tree, while `pip install .` produced a package that imported
cleanly and then failed on the first external-cohort record. Nothing caught it, because no test
looked at what a build actually contains.

This test needs no build: it checks the declaration against the tree, so a data file added in a
directory no pattern covers fails here rather than in a user's install.
"""
from __future__ import annotations

import ast

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "igraphecg"

# Not read at run time: documentation, byte-code caches, and the packages' own sources.
EXEMPT_SUFFIXES = {".py", ".pyc", ".pyo", ".md"}


def _patterns() -> list[str]:
    """The `igraphecg = [...]` line of [tool.setuptools.package-data].

    Read with a regex rather than tomllib, which arrived in Python 3.11 while this package
    declares requires-python >= 3.10.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    table = re.search(r"^\[tool\.setuptools\.package-data\]\s*$(.*?)(?=^\[|\Z)",
                      text, re.M | re.S)
    assert table, "[tool.setuptools.package-data] is missing from pyproject.toml"
    line = re.search(r"^\s*igraphecg\s*=\s*(\[[^\]]*\])", table.group(1), re.M)
    assert line, "no igraphecg entry under [tool.setuptools.package-data]"
    return list(ast.literal_eval(line.group(1)))


def _shipped_data_files() -> list[Path]:
    return sorted(p for p in PKG.rglob("*")
                  if p.is_file()
                  and p.suffix.lower() not in EXEMPT_SUFFIXES
                  and "__pycache__" not in p.parts)


def test_every_runtime_data_file_is_declared_as_package_data():
    patterns = _patterns()
    # setuptools expands each pattern with glob semantics relative to the package directory,
    # where `*` stops at a path separator and only `**` descends. Path.glob matches that;
    # fnmatch does not (its `*` crosses separators and would call the broken pattern covered).
    covered = {m.relative_to(PKG).as_posix()
               for pat in patterns for m in PKG.glob(pat) if m.is_file()}
    missed = [f.relative_to(PKG).as_posix() for f in _shipped_data_files()
              if f.relative_to(PKG).as_posix() not in covered]
    assert not missed, (
        "these files are read at run time but no package-data pattern covers them, so "
        "`pip install .` would drop them:\n  " + "\n  ".join(missed)
        + f"\npatterns declared: {patterns}")


def test_the_two_known_artifacts_are_present_in_the_tree():
    """A guard on the fixture itself: if these are ever moved, the test above must be updated."""
    for rel in ("artifacts/label_maps/snomed_to_superclass.csv",
                "artifacts/scalers/ptbxl_train_signal.pkl"):
        assert (PKG / rel).is_file(), f"{rel} is gone; update this test and pyproject.toml"
