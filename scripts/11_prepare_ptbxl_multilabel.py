"""W1F stage 1: build the canonical PTB-XL *multi-label* superdiagnostic median-beat set.

Difference from the frozen `scripts/10_prepare_ptbxl.py`:
  * record pool = every record carrying >=1 diagnostic superclass (the standard
    PTB-XL superdiagnostic benchmark pool, 21,388 records) instead of the
    single-label "clean" subset (15,709);
  * target = 5 binary labels NORM/MI/STTC/CD/HYP instead of one exclusive class;
  * the lead-wise robust scaler is **NOT refit** -- the frozen PTB-XL training
    scaler is copied verbatim from the submitted
    `data/processed/ptbxl_medianbeat_clean_100hz.npz`.

Everything else (0.5-40 Hz zero-phase bandpass, quality gates, R-peak detection,
median-beat window -300/+700 ms at 100 Hz) is the *same repository code* used by
the frozen pipeline, so the median beats of the clean records are bit-identical
to the frozen ones (verified downstream by 70_multilabel_eval.py).

Usage:
    ECG_DATA_DIR=<...>/data_PTB-XL python scripts/11_prepare_ptbxl_multilabel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igraphecg.data import beat_extraction as be
from igraphecg.data import label_utils as lu
from igraphecg.data import ptbxl_loader as pl
from igraphecg.data.preprocess import bandpass_filter, check_quality
from igraphecg.data.ptbxl_loader import CANONICAL_LEADS, LEAD_TO_IDX
from igraphecg.utils.logging import get_logger
from igraphecg.utils.paths import PROJECT_ROOT, RAW_DIR, find_ptbxl_root
from igraphecg.utils.seed import set_seed

log = get_logger("w1f_prepare")

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# frozen preprocessing constants (mirror configs/ptbxl_data.yaml exactly)
FS = 100
PRE_MS, POST_MS = 300.0, 700.0
BP_LOW, BP_HIGH, BP_ORDER = 0.5, 40.0, 4
MAX_ABS_MV = 20.0
MIN_LIKELIHOOD = 0.0
EXCLUDE_HYP_OVERLAP = True

FROZEN_NPZ = PROJECT_ROOT / "data" / "processed" / "ptbxl_medianbeat_clean_100hz.npz"
OUT_NPZ = PROJECT_ROOT / "data" / "processed" / "ptbxl_medianbeat_multilabel_100hz.npz"
OUT_META = PROJECT_ROOT / "data" / "processed" / "ptbxl_medianbeat_multilabel_100hz_metadata.csv"


def main():
    set_seed(42)
    root = find_ptbxl_root()
    if root is None:
        log.error(f"PTB-XL not found: no ptbxl_database.csv under {RAW_DIR}. Unpack PTB-XL v1.0.3 "
                  "there or set ECG_DATA_DIR (README.md, section 'Data').")
        sys.exit(2)
    log.info(f"PTB-XL root: {root}")

    db = pl.load_database(root)
    scp = pl.load_scp_statements(root)
    code2super = lu.build_code_to_superclass(scp)
    db = lu.assign_single_label(db, code2super, min_likelihood=MIN_LIKELIHOOD,
                                exclude_hyp_overlap=EXCLUDE_HYP_OVERLAP)
    db = pl.assign_official_splits(db)

    n_super = db["superclasses"].apply(lambda s: 0 if not s else len(str(s).split(";")))
    pool = db[n_super >= 1].copy()
    log.info(f"CONTROL total={len(db)}  clean_single_label={int(db['is_clean'].sum())}  "
             f">=1_superclass_pool={len(pool)}")
    assert len(db) == 21799, len(db)
    assert int(db["is_clean"].sum()) == 15709
    assert len(pool) == 21388, len(pool)
    log.info(f"CONTROL fold10 of pool = {int((pool['strat_fold'] == 10).sum())}")

    # frozen training scaler -- copied, never refit
    with np.load(FROZEN_NPZ, allow_pickle=True) as d:
        scaler_median = d["scaler_median"].copy()
        scaler_iqr = d["scaler_iqr"].copy()
        frozen_ids = d["record_id"].copy()
    log.info("frozen scaler loaded (NOT refit)")

    lead_idx = LEAD_TO_IDX
    sigs, folds, recs, pats, ages, sexes, quals, ysup, is_clean, clean_lab = ([] for _ in range(10))
    rej = {"read": 0, "nan": 0, "constant_lead": 0, "abnormal_amplitude": 0, "beat_fail": 0}

    from tqdm import tqdm
    for ecg_id, row in tqdm(pool.iterrows(), total=len(pool), desc="multilabel"):
        try:
            sig, _ = pl.read_record_signal(root, row, FS)
        except Exception:  # noqa: BLE001
            rej["read"] += 1
            continue
        if np.isnan(sig).all(axis=1).any():
            rej["nan"] += 1
            continue
        sig = bandpass_filter(sig, FS, BP_LOW, BP_HIGH, BP_ORDER)
        q = check_quality(sig, max_mv=MAX_ABS_MV)
        if q.has_nan:
            rej["nan"] += 1; continue
        if q.n_constant_leads > 0:
            rej["constant_lead"] += 1; continue
        if q.abnormal_amplitude:
            rej["abnormal_amplitude"] += 1; continue
        res = be.extract_median_beat(sig, FS, lead_idx, PRE_MS, POST_MS)
        if not res.success:
            rej["beat_fail"] += 1
            continue

        supers = set(str(row["superclasses"]).split(";"))
        ysup.append([1 if c in supers else 0 for c in SUPERCLASSES])
        sigs.append(res.median_beat)
        folds.append(int(row["strat_fold"]))
        recs.append(int(ecg_id))
        pats.append(int(row["patient_id"]))
        ages.append(float(row.get("age", np.nan)))
        sexes.append(int(row["sex"]) if not pd.isna(row.get("sex", np.nan)) else -1)
        quals.append(float(res.quality))
        is_clean.append(bool(row["is_clean"]))
        clean_lab.append(int(row["label"]))

    sigs = np.stack(sigs, 0).astype(np.float32)
    ysup = np.asarray(ysup, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int64)
    recs = np.asarray(recs, dtype=np.int64)
    log.info(f"processed {len(sigs)}/{len(pool)}  rejections={rej}")

    n_lab = ysup.sum(axis=1)
    log.info(f"label cardinality: 1={int((n_lab == 1).sum())} 2={int((n_lab == 2).sum())} "
             f">=3={int((n_lab >= 3).sum())}")
    for i, c in enumerate(SUPERCLASSES):
        log.info(f"  {c}: {int(ysup[:, i].sum())} positives")

    np.savez_compressed(
        OUT_NPZ,
        record_id=recs, patient_id=np.asarray(pats, dtype=np.int64),
        signal_12lead=sigs, y_multilabel=ysup,
        superclasses=np.asarray(SUPERCLASSES),
        fold=folds, age=np.asarray(ages, dtype=np.float32),
        sex=np.asarray(sexes, dtype=np.int64),
        sampling_rate=np.int64(FS),
        r_peak_quality=np.asarray(quals, dtype=np.float32),
        lead_names=np.asarray(CANONICAL_LEADS),
        scaler_median=scaler_median, scaler_iqr=scaler_iqr,
        # repository-relative, so the cache is byte-identical wherever the checkout lives
        scaler_source=np.asarray(FROZEN_NPZ.relative_to(PROJECT_ROOT).as_posix()),
        is_clean=np.asarray(is_clean, dtype=bool),
        clean_label=np.asarray(clean_lab, dtype=np.int64),
        window_pre_ms=np.float32(PRE_MS), window_post_ms=np.float32(POST_MS),
    )
    log.info(f"saved {OUT_NPZ}")

    meta = pd.DataFrame({
        "record_id": recs, "patient_id": pats, "fold": folds,
        "split": np.where(folds <= 8, "train", np.where(folds == 9, "val", "test")),
        "n_superclass": n_lab, "is_clean": is_clean, "r_peak_quality": quals,
        **{f"y_{c}": ysup[:, i] for i, c in enumerate(SUPERCLASSES)},
    })
    meta.to_csv(OUT_META, index=False)
    log.info(f"saved {OUT_META}")

    overlap = np.intersect1d(recs, frozen_ids)
    log.info(f"records shared with the frozen clean set: {len(overlap)} "
             f"(frozen set size {len(frozen_ids)})")


if __name__ == "__main__":
    main()
