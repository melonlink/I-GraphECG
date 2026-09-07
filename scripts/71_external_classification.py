"""v1fix driver for igraphecg.external.classify_external: train the PTB-XL C0 (theta-only)
classifier from the FIXED-lineage feature table (v1fix_r3_s42) instead of the old non-B
round3_eikonal table, then apply it to an external theta dump. Inference/transfer only.

Usage: <py> scripts/71_external_classification.py --theta <ext.theta.npz> --dataset georgia --out <json>
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from igraphecg.external import classify_external as ce   # noqa: E402

# PTBXL_FEATURES no longer needs overriding: the module's own default is the published table
print(f"[v1fix-extcls] PTBXL_FEATURES -> {ce.PTBXL_FEATURES}")
ce.main()
