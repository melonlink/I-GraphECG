"""Shipped source is English. No CJK character anywhere a reviewer will read.

Until 2026-09-06 the repository's comments, docstrings and log messages were Chinese, and the
archive builder swapped in an English overlay at build time behind an AST-shape guard. That
guard once let corrupted colour literals through (the sync tool took "#4c72b0" for a comment),
and the overlay drifted whenever a source file changed. The overlay is gone: the English text
is the source, adopted only after every file was proved to be the same program as its Chinese
twin. This test is what stops the next Chinese comment from arriving.

Scope: the package, every script the archive manifest names plus scripts/stats, the shipped
configs, the tests, the lock, and the READMEs that ship. experiments/ and the paper tree are
not shipped and are not held to this.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile("[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")   # written as escapes so this file passes itself


def _manifest_scripts() -> set[str]:
    m = ROOT / "1_I-GraphECG-Surrogate/FINAL_Sensors/Revision_v7/Analysis/tools/build_archive.py"
    if not m.exists():
        return set()
    blk = re.search(r"PAPER_SCRIPTS = \[(.*?)\n\]", m.read_text(encoding="utf-8"), re.S)
    return set(re.findall(r'"([0-9a-z_]+\.py)"', blk.group(1))) if blk else set()


def _scope() -> list[Path]:
    ship = _manifest_scripts()
    files = [p for p in (ROOT / "igraphecg").rglob("*") if p.suffix in (".py", ".md") and "__pycache__" not in p.parts]
    files += [p for p in (ROOT / "scripts").glob("*.py") if not ship or p.name in ship]
    files += list((ROOT / "scripts/stats").glob("*.py")) + [ROOT / "scripts/README.md"]
    files += list((ROOT / "configs").rglob("*.yaml"))
    files += [p for p in (ROOT / "tests").glob("*.py")]
    files += [ROOT / "paper.lock.yaml", ROOT / "pyproject.toml", ROOT / "requirements.txt", ROOT / "README.md"]
    return sorted(p for p in files if p.exists())


def test_no_chinese_in_shipped_source():
    offenders = []
    for p in _scope():
        text = p.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.split("\n"), 1):
            if CJK.search(line):
                offenders.append(f"{p.relative_to(ROOT).as_posix()}:{i}: {line.strip()[:70]}")
    assert not offenders, "Chinese text in shipped source:\n  " + "\n  ".join(offenders[:30])


def test_the_repository_declares_its_roots_in_repro_yaml_not_in_code():
    """roots.py ships verbatim, so the repository's own directory names may not be in it."""
    src = (ROOT / "igraphecg/repro/roots.py").read_text(encoding="utf-8")
    assert "1_I-GraphECG-Surrogate" not in src
    assert (ROOT / "repro.yaml").exists()
