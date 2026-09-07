"""The lineage may be named in one place. Everything else asks.

This is the gate that makes root cause 1 structural rather than merely fixed. Before it,
scripts/45 and scripts/47 both carried

    return f"{P1}/checkpoints/graphmono_encoder_eikonal_B_best.pt" if s == 42 \\
        else f"{P1}/checkpoints/seed_sweep/eikonal_B_seed{s}.pt"

-- both branches naming a superseded lineage -- and the published numbers were correct only
because a lambda in a third file replaced the function before it ran. Nothing could have
caught that, because there was nowhere the lineage was declared to be checked against.

Now paper.lock.yaml declares it, igraphecg/repro/lineage.py reads it, and this test fails if
any other shipped file names a weight file. A superseded default cannot be reintroduced
without turning the suite red.
"""
from __future__ import annotations

import ast
import io
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "paper.lock.yaml"

# Where a checkpoint filename may legitimately appear.
ALLOWED = {
    "igraphecg/repro/lineage.py",       # reads the lock; documents the defect it replaces
    "paper.lock.yaml",                  # the declaration itself
    "scripts/54_cd_descriptors.py",     # docstrings recording the defect they no longer have
    "scripts/82_fig_identifiability_assets.py",
    "tests/test_lineage_lock.py",
}
SEARCH = ["scripts", "igraphecg", "configs", "tests"]
CKPT = re.compile(r"[A-Za-z0-9_./]*_best\.pt|eikonal_B_seed\d+\.pt|seed_sweep/[A-Za-z0-9_]+\.pt")


def _archive_scripts() -> set[str]:
    """The scripts that ship with the paper, read from the archive manifest.

    This is the gate's boundary, and it is a deliberate one. experiments/02_observability_control/legacy_scripts/61_revision_v2_fim.py,
    63 and 65 are Revision-v2/v3 analyses: eikonal_B IS their lineage, because it is what
    they were written to study. They are kept for experiments/02, they are not in the
    archive, and holding them to the published lineage would be wrong. The paper's code must
    be single-lineage; the repository may keep its history.
    """
    manifest = ROOT / "1_I-GraphECG-Surrogate/FINAL_Sensors/Revision_v7/Analysis/tools/build_archive.py"
    if not manifest.exists():
        return set()
    text = manifest.read_text(encoding="utf-8")
    block = re.search(r"PAPER_SCRIPTS = \[(.*?)\]", text, re.S)
    names = set(re.findall(r'"([0-9a-z_]+\.py)"', block.group(1))) if block else set()
    names |= {p.name for p in (ROOT / "scripts/stats").glob("*.py")}
    return names


def _shipped() -> list[Path]:
    """Files the archive carries: the package, the manifest's scripts, configs, tests."""
    ship = _archive_scripts()
    out = []
    for d in SEARCH:
        for p in sorted((ROOT / d).rglob("*")):
            if p.suffix not in (".py", ".yaml", ".yml") or "__pycache__" in p.parts:
                continue
            if p.parent.name == "scripts" and p.name not in ship:
                continue          # repository-only, see _archive_scripts
            if p.suffix in (".yaml", ".yml") and _superseded_config(p):
                continue          # not shipped; the archive selects configs by name
            out.append(p)
    return out


SUPERSEDED_CFG = re.compile(r"d1s_r[568]_|d1_r\d|limbctl|rank[24]_|_eikonal|round3\.yaml|round4\.yaml|seed_sweep")


def _superseded_config(p: Path) -> bool:
    """Configs of superseded lineages. They are not in the archive: build_archive selects
    by name (the NAMED set, d1s_r3_s*, v1fix/), so these stay in the repository only."""
    return bool(SUPERSEDED_CFG.search(p.as_posix()))


pytestmark = pytest.mark.skipif(not LOCK.exists(), reason="paper.lock.yaml absent")


def test_lock_declares_one_lineage_and_verifiable_checkpoints():
    import yaml
    lock = yaml.safe_load(LOCK.read_text(encoding="utf-8"))
    assert lock["lineage"] == "d1s_r3"
    assert set(lock["seeds"]) == {42, 1, 2, 3, 4}
    for seed, e in lock["checkpoints"].items():
        assert e["path"].endswith(f"d1s_r3_s{seed}_best.pt"), f"seed {seed} names a foreign weight"
        assert len(e["sha256"]) == 64, f"seed {seed} has no usable checksum"


def _docstring_lines(text: str) -> set[int]:
    """Line numbers inside any docstring, found with the parser rather than by guessing.

    Naming a weight file in prose -- a usage example, or a note recording the defect a
    function no longer has -- is documentation, not a resolution. The first version of this
    test only skipped lines that *began* with a quote or a hash, so it flagged the interior
    lines of every multi-line docstring.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", [])
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            d = body[0]
            out.update(range(d.lineno, (d.end_lineno or d.lineno) + 1))
    return out


def _record_lines(text: str) -> set[int]:
    """Lines where a weight name is RECORDED rather than RESOLVED.

    Three shapes are not lineage decisions and must not be flagged:
      a dict value      "base_checkpoint": "checkpoints/d1s_r3_s42_best.pt"   -- provenance
                        "checkpoint": "..._eikonal_B_best.pt (superseded, NOT the paper model)"
      a written path    CKPT_OUT / f"leadset_{tag}_{arm}_best.pt"  -- a weight this script
                        trains and saves, not one it loads
    The distinction matters: the defect was reading the wrong model, and a gate that cannot
    tell reading from writing would push people to work around it.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    out.add(v.lineno)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            src = ast.dump(node.left)
            if "_OUT" in src or "_out" in src:
                out.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return out


def _hits(p: Path) -> list[tuple[int, str]]:
    text = io.open(p, encoding="utf-8", errors="replace").read()
    skip = set()
    if p.suffix == ".py":
        skip = _docstring_lines(text) | _record_lines(text)
    out = []
    for m in CKPT.finditer(text):
        line = text[:m.start()].count("\n") + 1
        if line in skip:
            continue
        if text.split("\n")[line - 1].strip().startswith("#"):
            continue              # a comment recording history is not a resolution
        out.append((line, m.group(0)))
    return out


DRIVERS = ()          # the 99_v1fix_* drivers are gone; the notes cite the real scripts


def _is_declared_exception(rel: str, name: str) -> bool:
    """A superseded read the lock declares, with its reason recorded there."""
    import sys
    sys.path.insert(0, str(ROOT))
    from igraphecg.repro import lineage
    base = name.rsplit("/", 1)[-1]
    return any(r["script"] == rel and r["artifact"].endswith(base)
               for r in lineage.permitted_superseded_reads())


def test_no_python_file_resolves_a_checkpoint():
    """Code asks the lock. A weight filename in Python is a lineage decision in hiding.

    Configuration is held to a weaker rule below: a config naming a checkpoint is a visible,
    reviewable declaration, not a constant buried in a function. Moving those to lineage keys
    (encoder_ckpt: d1s_r3_s42) belongs to the path-constant stage.
    """
    offenders = []
    for p in _shipped():
        rel = p.relative_to(ROOT).as_posix()
        if p.suffix != ".py" or rel in ALLOWED or p.name in DRIVERS:
            continue          # drivers have their own test above, pending S3
        for ln, name in _hits(p):
            if _is_declared_exception(rel, name):
                continue
            offenders.append(f"{rel}:{ln}  {name}")
    assert not offenders, (
        "these modules resolve a checkpoint themselves instead of asking the lineage lock:\n  "
        + "\n  ".join(offenders))


def test_nothing_shipped_names_a_superseded_checkpoint():
    """The danger is not naming a weight; it is naming the WRONG one.

    Every entry here is a path that resolves to a lineage the paper does not report, so a
    reader following it computes numbers no table contains.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from igraphecg.repro import lineage
    permitted = {r["artifact"].rsplit("/", 1)[-1]: r["script"]
                 for r in lineage.permitted_superseded_reads()}
    offenders = []
    for p in _shipped():
        rel = p.relative_to(ROOT).as_posix()
        if rel in ALLOWED or p.name in DRIVERS:
            continue          # drivers are covered by the xfail above, pending S3
        for ln, name in _hits(p):
            base = name.rsplit("/", 1)[-1]
            if base in permitted and permitted[base] == rel:
                continue          # a declared exception, with its reason in the lock
            if lineage.is_superseded(name):
                offenders.append(f"{rel}:{ln}  {name}")
    assert not offenders, (
        "these name a checkpoint from a lineage the paper does not report:\n  "
        + "\n  ".join(offenders))


def test_checkpoint_resolution_refuses_a_foreign_seed():
    """There is no fallback: a seed outside the lineage has no published weight."""
    import sys
    sys.path.insert(0, str(ROOT))
    from igraphecg.repro import lineage
    with pytest.raises(KeyError):
        lineage.checkpoint(7)


def test_no_mixed_lineage_branch_survives():
    """The specific shape of the original defect: a per-seed conditional over weight paths."""
    bad = []
    for p in _shipped():
        if p.suffix != ".py":
            continue
        try:
            tree = ast.parse(io.open(p, encoding="utf-8", errors="replace").read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.IfExp):
                continue
            dump = ast.dump(node)
            if "_best" in dump or "seed_sweep" in dump:
                bad.append(f"{p.relative_to(ROOT).as_posix()}:{node.lineno}")
    assert not bad, "a per-seed conditional still chooses between weight files: " + ", ".join(bad)


def test_permitted_superseded_artifact_is_declared_with_a_checksum_and_resolves():
    """The one superseded read the lock permits (W1M <- eikonal_B) is checksummed, and
    resolves through lineage.permitted_artifact where the file is present. An undeclared
    script gets KeyError: the permission is per script, not a general licence."""
    import sys
    sys.path.insert(0, str(ROOT))
    from igraphecg.repro import lineage
    reads = lineage.permitted_superseded_reads()
    assert reads and all(len(r.get("sha256", "")) == 64 for r in reads)
    with pytest.raises(KeyError):
        lineage.permitted_artifact("scripts/54_cd_descriptors.py")
    try:
        p = lineage.permitted_artifact(reads[0]["script"])
    except FileNotFoundError:
        pytest.skip("permitted artifact not present in this tree")
    assert p.name == reads[0]["artifact"].rsplit("/", 1)[-1]
