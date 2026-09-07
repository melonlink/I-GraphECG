"""Frozen configuration for I-GraphECG (single source of truth for all 'magic constants').

Values here MUST match the manuscript / Supplementary Table S3. Changing them changes
published behavior and will break the regression test (tests/test_regression_ptbxl.py).
"""
from __future__ import annotations
import sys
from pathlib import Path

# --- repo root holds the igraphecg package; put it on sys.path so `import igraphecg` works ---
REPO_ROOT = Path(__file__).resolve().parents[1]   # .../I-GraphMonoECG
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Back-compat alias (older code imported PUBLIC_ROOT)
PUBLIC_ROOT = REPO_ROOT

# Roots come from igraphecg.repro.roots (environment variable > repro.yaml > default), the
# one place they are declared. Raw data is shared and outside the repository; derived
# caches (median beats, theta) are regenerable and live in the repository's data/.
from igraphecg.repro import roots as _roots
DATA_DIR = _roots.data()
PROCESSED_DIR = _roots.processed()

# --- signal / preprocessing (frozen; = manuscript §2.1 and Table S3) ---
TARGET_FS = 100   # Hz; analysis sampling rate
BANDPASS_LOW = 0.5   # Hz
BANDPASS_HIGH = 40.0   # Hz
WINDOW_PRE_MS = 300.0   # ms before R
WINDOW_POST_MS = 700.0   # ms after R
RPEAK_PRIMARY_LEAD = "II"   # NeuroKit2 on lead II, fallback V5/multi-lead

# --- canonical 12-lead order ---
CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                   "V1", "V2", "V3", "V4", "V5", "V6"]
INDEPENDENT_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]   # 8 independent

# --- task definition ---
SUPERCLASSES = ["NORM", "MI", "STTC", "CD"]
EXCLUDE_SUPERCLASS = "HYP"   # hypertrophy overlap excluded from the clean subset

# --- model ---
PARAM_DIM = 47
LEADFIELD_RANK = 3
SEED_LOCKED = 42
SEEDS_ALL = [42, 1, 2, 3, 4]

# --- artifact locations (relative to this package) ---
PKG_ROOT = Path(__file__).resolve().parent
ARTIFACTS = PKG_ROOT / "artifacts"
SCALER_DIR = ARTIFACTS / "scalers"
LABELMAP_DIR = ARTIFACTS / "label_maps"

# Repository data/checkpoints produced by the existing pipeline (gitignored)
# PAPER_ROOT is gone: published code must not know where the manuscript tree is.
# Roots come from igraphecg.repro, and in the archive repro.paper() raises by design.
PTBXL_NPZ = PROCESSED_DIR / "ptbxl_medianbeat_clean_100hz.npz"   # repo-local derived cache
def default_checkpoint(seed: int = 42):
    """The locked encoder weight. Replaces EIKONAL_B_CKPT, which named a superseded
    model and was the default for both external_eval and the public load_model()."""
    from igraphecg.repro import lineage
    return lineage.checkpoint(seed)
