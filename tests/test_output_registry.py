"""The lock's `runs:` registry, the code that writes, and the frozen tree agree.

paper.lock.yaml lists every output directory the paper's numbers come from, with the number
of files each holds. That list is only worth having if something checks it, so this does,
in both directions:

  code -> registry   every output location a shipped script writes or reads under the
                     output root is a registered directory. A script writing somewhere
                     unregistered is either producing something the paper does not use, or
                     the registry is incomplete; both are findings.
  registry -> tree   every registered directory exists under the output root with exactly
                     the declared number of files. Inside the code archive this is the
                     deposit gate: the archive ships those files, and a short one fails here.

The first version of this check found a summary module under igraphecg/external writing
to a pre-fix external-results directory whose PTB-XL reference numbers (0.911, 0.893) are
not the paper's, while Tables 8-9 come from `v1fix/external/`. Nothing in the documented
pipeline called it. Module and directory are both gone.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "paper.lock.yaml"
sys.path.insert(0, str(ROOT))
from igraphecg import repro  # noqa: E402

pytestmark = pytest.mark.skipif(not LOCK.exists(), reason="paper.lock.yaml absent")

# Locations under the output root that are not run directories and are not registered:
#   figures      the figure command's output, compared with the manuscript by
#                test_figures_match_manuscript.py rather than counted here
#   checkpoints  weights, declared per seed with checksums under `checkpoints:` in the lock
NOT_RUNS = {"figures", "checkpoints"}

SLUG = re.compile(
    r'repro\.outdir\(\s*"([^"]+)"'                    # a directory a script writes
    r'|repro\.runs\(\)\s*/\s*"([^"]+)"'               # a run it reads
    r'|repro\.outputs\(\)\s*/\s*"([^"]+)"')           # anything else under the root


def _lock() -> dict:
    return yaml.safe_load(LOCK.read_text(encoding="utf-8"))


def _registered() -> set[str]:
    """Registered directories, with the tables/figures leaf stripped so a script naming the
    run (runs/v1fix_r3_s42) matches an entry naming its tables (runs/v1fix_r3_s42/tables)."""
    out = set()
    for e in _lock()["runs"]:
        d = e["dir"]
        out.add(d)
        for leaf in ("/tables", "/figures"):
            if d.endswith(leaf):
                out.add(d[: -len(leaf)])
    return out


def _shipped_py() -> list[Path]:
    manifest = ROOT / "1_I-GraphECG-Surrogate/FINAL_Sensors/Revision_v7/Analysis/tools/build_archive.py"
    ship: set[str] = set()
    if manifest.exists():
        blk = re.search(r"PAPER_SCRIPTS = \[(.*?)\n\]", manifest.read_text(encoding="utf-8"), re.S)
        if blk:
            ship = set(re.findall(r'"([0-9a-z_]+\.py)"', blk.group(1)))
    out = [p for p in (ROOT / "scripts").glob("*.py") if not ship or p.name in ship]
    out += list((ROOT / "scripts/stats").glob("*.py"))
    out += [p for p in (ROOT / "igraphecg").rglob("*.py") if "__pycache__" not in p.parts]
    return sorted(out)


def _as_dir(slug: str) -> str:
    """A file path under a run (runs/x/tables/a.csv) names the run's tables directory."""
    parts = slug.split("/")
    while parts and "." in parts[-1]:
        parts.pop()
    return "/".join(parts)


def _normalise(slug: str, via: str) -> str:
    """repro.runs()/"x" is runs/x; repro.outputs()/"x" and outdir("x") are x as written."""
    return f"runs/{slug}" if via == "runs" else slug


def test_every_output_location_in_shipped_code_is_registered():
    reg = _registered()
    offenders = []
    for p in _shipped_py():
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in SLUG.finditer(text):
            via = ("outdir", "runs", "outputs")[m.lastindex - 1]
            slug = _as_dir(_normalise(m.group(m.lastindex), via))
            if slug in NOT_RUNS or slug.split("/")[0] in NOT_RUNS:
                continue
            if slug in reg or any(slug.startswith(r + "/") for r in reg):
                continue
            line = text[: m.start()].count("\n") + 1
            offenders.append(f"{p.relative_to(ROOT).as_posix()}:{line}  {slug}")
    assert not offenders, (
        "shipped code uses output locations the lock's runs: registry does not list:\n  "
        + "\n  ".join(offenders))


def test_registered_directories_exist_with_the_declared_file_counts():
    root = repro.outputs()
    if not root.exists():
        pytest.skip(f"no output root at {root}")
    problems = []
    for e in _lock()["runs"]:
        d = root / e["dir"]
        declared = e["files"]
        if not d.is_dir():
            problems.append(f"{e['dir']}: missing")
            continue
        have = sorted(f.name for f in d.iterdir() if f.is_file())
        if isinstance(declared, int):
            if len(have) != declared:
                problems.append(f"{e['dir']}: {len(have)} files, lock declares {declared}")
        else:
            missing = sorted(set(declared) - set(have))
            if missing:
                problems.append(f"{e['dir']}: missing {missing}")
    assert not problems, "the frozen tree does not match the lock's registry:\n  " + "\n  ".join(problems)


# Registered files whose names are given on the command line rather than in code: the
# external-cohort outputs, written by igraphecg.external.external_eval and
# 71_external_classification to whatever --out names (REPRODUCIBILITY.md section 5).
CLI_NAMED = {"v1fix/external"}


def test_every_registered_file_has_a_shipped_producer():
    """The reverse of the first check: a file in the deposit that no shipped script writes is a
    number nobody can regenerate. The first run of this found 33 -- the W1C, W1K and W1F
    headline generators had only ever existed in a scratch directory (their outputs back the
    Supplementary descriptor-fidelity and cohort-composition tables), one file's writer was
    lost and has been reconstructed, and ten *_VERIFY_* re-check records plus a run log
    were audit by-products with no generator, now unregistered."""
    # a producer names the file in code, or declares it in a shipped config (the baseline
    # scripts write wherever out_metrics / out_cm in their YAML say)
    code = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for p in _shipped_py() + sorted((ROOT / "configs").rglob("*.yaml"))
                     if "__pycache__" not in p.parts)
    orphans = []
    for e in _lock()["runs"]:
        if e["dir"] in CLI_NAMED:
            continue
        for name in e["files"]:
            stem = Path(name).stem
            base = re.sub(r"_s(42|[1-4])$", "", stem)   # per-seed names carry the seed in an f-string
            derived = stem.replace("_perclass", "")      # 20/21 derive it: out_metrics -> *_perclass.csv
            if name in code or stem in code or base in code or derived in code:
                continue
            orphans.append(f"{e['dir']}/{name}")
    assert not orphans, ("registered files that no shipped script writes:\n  " + "\n  ".join(orphans))


def test_registry_names_no_superseded_lineage():
    """A registered directory is a published one; a superseded lineage cannot be in it."""
    from igraphecg.repro import lineage
    bad = [e["dir"] for e in _lock()["runs"] if lineage.is_superseded(e["dir"])]
    assert not bad, f"registry lists superseded output: {bad}"
