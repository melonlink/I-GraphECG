"""The documented figure command produces the figures that are in the paper.

This is the gate for root cause 6's worst instance. Before stage S3 the reproducibility
notes told readers to run two figure drivers, and those drivers regenerated the
Revision-v6 versions of Figures 6 and 7 -- which differ from the ones in the submitted
manuscript -- while the script that actually produced the published versions was not
mentioned at all. Nothing could have caught that, because nothing compared a regenerated
figure with the file the manuscript includes.

This test regenerates every generated figure, rasterises both the regenerated PDF and the
manuscript's, and requires them to be pixel-identical at 150 dpi. PDF bytes are not
compared: matplotlib stamps a creation date into every PDF, so byte identity would fail
for a reason that has nothing to do with the figure.

Two figures need the published checkpoint and a GPU (fig_recon, figS2_recon12) or the
frozen five-seed panels (fig_descriptors, Figures 6 and 7); those run only where CUDA is
available and the seed-42 weight resolves, and skip otherwise. Figure 1 is a hand-authored
schematic and has no generator. Figure S1 is held to a stated weaker standard, see below.

It skips, rather than fails, where it cannot run: in the code archive there is no
manuscript tree to compare against, and on a machine without poppler there is no
rasteriser. A skip is reported as a skip; it is never reported as a pass.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from igraphecg import repro  # noqa: E402  -- the frozen tables resolve through the roots, like everything else

# The manuscript: the repository's tree, or paper/ in a checkout that carries the submission
# package beside the code. Where neither exists (the bare archive) every check here skips.
_CANDIDATES = [ROOT / "1_I-GraphECG-Surrogate/FINAL_Sensors/Revision_v7/Submission/1_Manuscript",
               ROOT / "paper/1_Manuscript"]
MANUSCRIPT = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])
FROZEN = repro.runs() / "v1fix_r3_s42/tables"
CPU_FIGURES = ["fig_classification", "fig_confusion"]


def _pdftoppm() -> str | None:
    for cand in (os.environ.get("PDF2IMAGE_POPPLER_PATH"), os.environ.get("POPPLER_BIN"),
                 r"D:\TOOLS\texlive\texlive\2025\bin\windows"):
        if cand and (Path(cand) / "pdftoppm.exe").exists():
            return str(Path(cand) / "pdftoppm.exe")
    return shutil.which("pdftoppm")


pytestmark = pytest.mark.skipif(
    not MANUSCRIPT.exists() or not FROZEN.exists() or _pdftoppm() is None,
    reason="needs the manuscript tree, the frozen seed-42 tables and poppler's pdftoppm")


def _raster(pdf: Path, out_stem: Path):
    subprocess.run([_pdftoppm(), "-r", "150", "-png", "-singlefile", str(pdf), str(out_stem)],
                   check=True, capture_output=True)
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(str(out_stem) + ".png").convert("L"), dtype=float)


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    out = tmp_path_factory.mktemp("figs")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    m = importlib.import_module("89_make_figures")
    m.main(out=out, only=CPU_FIGURES)
    return out


def test_figure_S1_is_regenerated_from_the_same_data_and_visually_identical(tmp_path):
    """Figure S1 is held to a WEAKER standard than the others, and the reason is stated.

    The script that drew the submitted PNG no longer exists; 89_make_figures.py redraws it
    from the same table (ap_ode_monodromy.csv). Redrawn with the manuscript's styling it
    differs from the submitted file in 0.68% of pixels by at most 17/255 -- anti-aliasing
    and font hinting, not content -- and pixel exactness would require the exact matplotlib
    build that drew the original. So this asserts near-identity, and says so, rather than
    reporting a pass it cannot earn.
    """
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    importlib.import_module("89_make_figures").main(out=tmp_path, only=["figS1_recovery"])
    new = np.asarray(Image.open(tmp_path / "figS1_recovery.png").convert("L"), dtype=float)
    ref = np.asarray(Image.open(MANUSCRIPT.parent / "2_Supplementary/figS1_recovery.png")
                     .convert("L"), dtype=float)
    assert new.shape == ref.shape
    diff = np.abs(new - ref)
    frac, mx = float((diff > 8).mean()), float(diff.max())
    assert frac < 0.01 and mx <= 24, (
        f"figS1_recovery: {frac * 100:.2f}% of pixels differ (max {mx:.0f}); more than "
        f"rendering noise -- the redraw no longer matches the submitted figure")


def _identical(name: str, new_path: Path, ref_path: Path, tmp_path: Path):
    import numpy as np
    if new_path.suffix == ".pdf":
        new = _raster(new_path, tmp_path / f"new_{name}")
        ref = _raster(ref_path, tmp_path / f"ref_{name}")
    else:
        from PIL import Image
        new = np.asarray(Image.open(new_path).convert("L"), dtype=float)
        ref = np.asarray(Image.open(ref_path).convert("L"), dtype=float)
    assert new.shape == ref.shape, f"{name}: raster size {new.shape} vs manuscript {ref.shape}"
    diff = np.abs(new - ref)
    frac = float((diff > 8).mean())
    assert frac == 0.0 and diff.max() == 0, (
        f"{name}: {frac * 100:.3f}% of pixels differ (max {diff.max():.0f}) from the manuscript's "
        f"copy -- the documented figure command no longer produces the published figure")


@pytest.mark.parametrize("name", CPU_FIGURES)
def test_regenerated_figure_is_pixel_identical_to_the_manuscript(name, regenerated, tmp_path):
    _identical(name, regenerated / f"{name}.pdf", MANUSCRIPT / f"{name}.pdf", tmp_path)


# Figures that need the published seed-42 weight on a GPU, or the frozen five-seed panels.
# fig_identifiability_ab and fig_selection_cd are one drawing call (Figures 6 and 7).
GPU_FIGURES = ["fig_recon", "fig_descriptors", "fig_identifiability_ab", "fig_selection_cd",
               "figS2_recon12"]


def _gpu_and_weight_available() -> bool:
    try:
        import torch
        sys.path.insert(0, str(ROOT))
        from igraphecg.repro import lineage
        lineage.checkpoint(42)
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@pytest.fixture(scope="module")
def regenerated_gpu(tmp_path_factory):
    """Runs AFTER the CPU figures and Figure S1 in this module, on purpose: that ordering is
    what exposed fig_recon inheriting another drawer's rcParams (15% of pixels off when
    drawn third, exact when drawn alone). 89_make_figures now resets rcParams per figure;
    this fixture keeps exercising the order that found it."""
    if not _gpu_and_weight_available():
        pytest.skip("needs CUDA and the seed-42 checkpoint named in paper.lock.yaml")
    out = tmp_path_factory.mktemp("figs_gpu")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    importlib.import_module("89_make_figures").main(out=out, only=GPU_FIGURES)
    return out


@pytest.mark.parametrize("name", GPU_FIGURES)
def test_gpu_figure_is_pixel_identical_to_the_manuscript(name, regenerated_gpu, tmp_path):
    if name == "figS2_recon12":
        ref = MANUSCRIPT.parent / "2_Supplementary/figS2_recon12.png"
        _identical(name, regenerated_gpu / f"{name}.png", ref, tmp_path)
    else:
        _identical(name, regenerated_gpu / f"{name}.pdf", MANUSCRIPT / f"{name}.pdf", tmp_path)
