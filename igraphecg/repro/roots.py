"""Root resolution. The repository and the code archive differ here and nowhere else.

Resolution order for every root: environment variable, then `repro.yaml` at the tree root,
then the built-in default. `repro.yaml` is the single file that differs between the
repository and the published archive:

    repository (its repro.yaml)                        archive (its repro.yaml)
    output_root: <the paper's output directory>        output_root: outputs
    paper_root:  <the manuscript tree>                 paper_root:  null

The repository's directory names are declared in its repro.yaml, not here: this file is
shipped verbatim and carries no name from either tree. Without any repro.yaml the
built-in defaults below describe a bare checkout (outputs/, no manuscript).

`paper_root` being null in the archive is deliberate and load-bearing: `paper()` raises
there, so a script that tries to write into a manuscript directory fails loudly at the
source instead of silently creating an empty `paper/Revision_v2/...` tree in the deposit,
which is what used to happen.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

# <repo>/igraphecg/repro/roots.py -> <repo>
_REPO = Path(__file__).resolve().parents[2]

_DEFAULTS = {
    "output_root": "outputs",
    "paper_root": None,
    "processed_root": "data/processed",
    "data_root": None,          # resolved relative to the repo's parent, see data()
}


@lru_cache(maxsize=1)
def _conf() -> dict:
    p = _REPO / "repro.yaml"
    if not p.exists():
        return dict(_DEFAULTS)
    loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    conf = dict(_DEFAULTS)
    conf.update({k: v for k, v in loaded.items() if k in _DEFAULTS})
    return conf


def repo() -> Path:
    """The tree root: the repository, or the archive root once unpacked."""
    return _REPO


def _rooted(key: str, env: str) -> Path | None:
    override = os.environ.get(env)
    if override:
        return Path(override)
    rel = _conf()[key]
    return None if rel is None else (_REPO / rel)


def outputs() -> Path:
    """Where runs write. ECG_OUTPUT_DIR overrides."""
    return _rooted("output_root", "ECG_OUTPUT_DIR")


def runs() -> Path:
    return outputs() / "runs"


def outdir(slug: str, create: bool = True) -> Path:
    """A run directory under the output root, by slug -- never by absolute path.

    Refuses to escape the output root, so a stray '..' or an absolute slug cannot put run
    output back into a paper directory.
    """
    d = (outputs() / slug).resolve()
    if not str(d).startswith(str(outputs().resolve())):
        raise ValueError(f"run slug escapes the output root: {slug!r}")
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def processed() -> Path:
    """Derived, regenerable cache (median beats, theta). ECG_PROCESSED_DIR overrides."""
    return _rooted("processed_root", "ECG_PROCESSED_DIR")


def data() -> Path:
    """Raw datasets: external, read-only, never redistributed. ECG_DATA_DIR overrides.

    The historical default is a sibling of the repository shared by three programs; the
    archive sets it explicitly because it has no such sibling.
    """
    override = _rooted("data_root", "ECG_DATA_DIR")
    return override if override is not None else _REPO.parent / "PUBLIC" / "data"


def paper() -> Path:
    """The manuscript tree. Raises where there is none -- notably inside the archive."""
    p = _rooted("paper_root", "ECG_PAPER_DIR")
    if p is None:
        raise RuntimeError(
            "no paper root in this tree: published code must not read or write manuscript "
            "directories. Write to repro.outdir(<slug>) instead.")
    return p


def describe() -> str:
    """One line per root, for logs and for the reproduction report."""
    def show(fn):
        try:
            return str(fn())
        except RuntimeError as exc:
            return f"(unavailable: {exc.args[0].splitlines()[0]})"
    return "\n".join(f"  {n:<11}{show(f)}" for n, f in
                     [("repo", repo), ("outputs", outputs), ("processed", processed),
                      ("data", data), ("paper", paper)])
