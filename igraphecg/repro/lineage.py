"""The published lineage, read from paper.lock.yaml. The only place a checkpoint is named.

Why this exists rather than a constant in each script: scripts/45 and scripts/47 both
carried

    def ckpt_of(s):
        return f"{P1}/checkpoints/graphmono_encoder_eikonal_B_best.pt" if s == 42 \\
            else f"{P1}/checkpoints/seed_sweep/eikonal_B_seed{s}.pt"

Both branches name the SUPERSEDED eikonal_B line. The published numbers are nevertheless
correct, because 54_cd_descriptors.py -- a different file -- replaced the whole function with a
lambda pointing at d1s_r3 before calling main(). So the shipped default was wrong and the
correct value lived in a monkey patch somewhere else. Anyone running the script directly,
as a reader of the archive would, silently computed numbers from a model the paper does not
use.

`checkpoint(seed)` replaces every such helper. It has no per-seed branch to get wrong, it
resolves through the declared lineage only, and it verifies the file's sha256 the first time
it is used, so a weight that is not the locked one fails loudly instead of quietly changing
a reported number.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import yaml

from igraphecg.repro import roots

LOCK = "paper.lock.yaml"


@lru_cache(maxsize=1)
def lock() -> dict:
    p = roots.repo() / LOCK
    if not p.exists():
        raise FileNotFoundError(
            f"{LOCK} not found at {p}. It declares the published lineage; published code "
            "cannot resolve a checkpoint without it.")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def name() -> str:
    return lock()["lineage"]


def seeds() -> tuple[int, ...]:
    return tuple(lock()["seeds"])


_verified: set[str] = set()


def checkpoint(seed: int, verify: bool = True) -> Path:
    """The locked encoder weight for this seed, sha256-checked on first use."""
    entries = lock()["checkpoints"]
    if seed not in entries:
        raise KeyError(
            f"seed {seed} is not part of the {name()} lineage (declared: {list(entries)}). "
            "There is no fallback: a seed outside the lineage has no published checkpoint.")
    e = entries[seed]
    p = roots.outputs() / e["path"]
    if not p.exists():
        raise FileNotFoundError(f"locked checkpoint missing: {p} (lineage {name()}, seed {seed})")
    if verify and e["path"] not in _verified:
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != e["sha256"]:
            raise ValueError(
                f"{p} is not the locked weight.\n  expected sha256 {e['sha256']}\n"
                f"  found            {got}\n"
                f"Reported numbers come from the locked file; refusing to continue with "
                f"another one.")
        _verified.add(e["path"])
    return p


def published_runs() -> list[str]:
    """Output-root-relative run directories that belong to the published lineage."""
    return [r["dir"] for r in lock().get("runs", [])]


def is_superseded(text: str) -> str | None:
    """The superseded lineage this string names, if any. Used by the gates."""
    for lin in lock().get("superseded", {}).get("lineages", []):
        if lin in text:
            return lin
    return None


def permitted_superseded_reads() -> list[dict]:
    """Declared exceptions: a script that legitimately reads a superseded artefact."""
    return lock().get("superseded", {}).get("permitted_reads", [])


def permitted_artifact(script: str, verify: bool = True) -> Path:
    """The superseded artifact the lock permits `script` to read, sha256-checked.

    Declared in `superseded.permitted_reads`, with the reason. A script asking for an
    artifact it is not declared to read gets a KeyError; a declared one that is absent
    from this tree gets a FileNotFoundError naming the declaration, so the omission is
    never silent -- the archive once shipped without it and W1M's comparison arm became
    null with nothing said.
    """
    for r in permitted_superseded_reads():
        if r["script"] == script:
            p = roots.outputs() / r["artifact"]
            if not p.exists():
                raise FileNotFoundError(
                    f"permitted superseded artifact missing: {p}\n  declared for {script} in "
                    f"paper.lock.yaml ({r.get('reason', '').strip()[:80]})")
            if verify and r.get("sha256") and r["artifact"] not in _verified:
                got = hashlib.sha256(p.read_bytes()).hexdigest()
                if got != r["sha256"]:
                    raise ValueError(f"{p} is not the declared artifact.\n  expected sha256 "
                                     f"{r['sha256']}\n  found            {got}")
                _verified.add(r["artifact"])
            return p
    raise KeyError(f"{script} is not declared to read any superseded artifact")
