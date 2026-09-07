"""Published code does not know where the manuscript is. Everything resolves through repro.

This is the gate for root cause 2. Before it, three shipped scripts wrote their results INTO
frozen paper-submission directories --

    OUT = ROOT / "1_I-GraphECG-Surrogate/FINAL_Sensors/Revision_v2/Results/lead_masking"
    OUT = ROOT / "1_I-GraphECG-Surrogate/FINAL_Sensors/Revision_v2/Results/noise_fim"
    OUT = PAPER / "FINAL_Sensors/Revision_v3/Submission/1_Manuscript/fig_recon.pdf"

-- with a module-level mkdir, so merely importing them created directories, and every
revision bump (v2 -> v7) broke them. The archive builder then rewrote those strings to
paper/..., a directory the archive does not have, and the deposit shipped empty
paper/Revision_2/Results/ trees as a fossil of the whole chain.

Now igraphecg.repro resolves the roots, repro.paper() raises inside the archive by design,
and this test fails if any shipped Python file names the repository's paper layout in code.
"""
from __future__ import annotations

import ast
import io
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = re.compile(r"FINAL_Sensors|Revision_v[1-7]|1_I-GraphECG-Surrogate")

# Declared exceptions, each with its reason.
ALLOWED = {
    "igraphecg/repro/roots.py",          # describes the two trees in its docstring; repro.yaml declares
    "igraphecg/repro/__init__.py",       # prose describing the rule
    "tests/test_no_paper_paths.py",
    "tests/test_lineage_lock.py",        # reads the repository's archive manifest
}
DRIVERS = ()          # the 99_v1fix_* figure drivers are gone; 89_make_figures.py is the command


def _shipped_py() -> list[Path]:
    manifest = ROOT / "1_I-GraphECG-Surrogate/FINAL_Sensors/Revision_v7/Analysis/tools/build_archive.py"
    ship: set[str] = set()
    if manifest.exists():
        blk = re.search(r"PAPER_SCRIPTS = \[(.*?)\n\]", manifest.read_text(encoding="utf-8"), re.S)
        if blk:
            ship = set(re.findall(r'"([0-9a-z_]+\.py)"', blk.group(1)))
    out = [p for p in (ROOT / "scripts").glob("*.py") if p.name in ship]
    out += list((ROOT / "scripts/stats").glob("*.py"))
    out += [p for p in (ROOT / "igraphecg").rglob("*.py") if "__pycache__" not in p.parts]
    return sorted(out)


def _prose_lines(text: str) -> set[int]:
    """Docstring lines, found with the parser. Prose may describe the layout; code may not."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    out: set[int] = set()
    for n in ast.walk(tree):
        b = getattr(n, "body", None)
        if (isinstance(b, list) and b and isinstance(b[0], ast.Expr)
                and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str)):
            out.update(range(b[0].lineno, (b[0].end_lineno or b[0].lineno) + 1))
    return out


def _record_lines(text: str) -> set[int]:
    """A dict value is a provenance record, not a path the code opens -- W1M's
    superseded_artifact.path is a citation of where a historical number came from."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    out: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Dict):
            for v in n.values:
                if isinstance(v, (ast.Constant, ast.JoinedStr, ast.Tuple)):
                    out.update(range(v.lineno, (v.end_lineno or v.lineno) + 1))
    return out


def _hits(p: Path) -> list[str]:
    text = io.open(p, encoding="utf-8", errors="replace").read()
    skip = _prose_lines(text) | _record_lines(text)
    out = []
    for i, line in enumerate(text.split("\n"), 1):
        if i in skip or line.strip().startswith("#"):
            continue
        code = line.split("#", 1)[0]          # a trailing comment may cite history
        if LAYOUT.search(code):
            out.append(f"{p.relative_to(ROOT).as_posix()}:{i}  {code.strip()[:80]}")
    return out


def test_shipped_code_never_names_the_paper_layout():
    offenders = []
    for p in _shipped_py():
        rel = p.relative_to(ROOT).as_posix()
        if rel in ALLOWED or p.name in DRIVERS:
            continue
        offenders += _hits(p)
    assert not offenders, (
        "shipped code names the repository's paper layout instead of resolving through "
        "igraphecg.repro:\n  " + "\n  ".join(offenders))


def test_paper_root_raises_where_there_is_no_manuscript(tmp_path, monkeypatch):
    """The archive sets paper_root: null. repro.paper() must refuse, not invent a path."""
    import sys
    sys.path.insert(0, str(ROOT))
    from igraphecg.repro import roots
    monkeypatch.setattr(roots, "_conf", lambda: {**roots._DEFAULTS, "paper_root": None})
    with pytest.raises(RuntimeError):
        roots.paper()


def test_outdir_cannot_escape_the_output_root():
    import sys
    sys.path.insert(0, str(ROOT))
    from igraphecg import repro
    with pytest.raises(ValueError):
        repro.outdir("../escape", create=False)
