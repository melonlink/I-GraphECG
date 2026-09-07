"""Stage 0: PTB-XL -> four-class single-label clean median-beat dataset.

Run:
    python scripts/10_prepare_ptbxl.py --config configs/ptbxl_data.yaml

Outputs:
    data/processed/ptbxl_medianbeat_clean_100hz.npz
    data/processed/ptbxl_medianbeat_clean_100hz_metadata.csv
    outputs/tables/data_split_summary.csv
    outputs/figures/random_median_beats/*.png
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
from igraphecg.data.preprocess import RobustLeadScaler, bandpass_filter, check_quality, notch_filter
from igraphecg.data.ptbxl_loader import CANONICAL_LEADS, LEAD_TO_IDX
from igraphecg.evaluation.plots import plot_median_beat
from igraphecg.utils.config import parse_args_with_config, save_config
from igraphecg.utils.logging import get_logger
from igraphecg.utils.paths import (FIGURES_DIR, PROCESSED_DIR, PROJECT_ROOT, TABLES_DIR,
                             ensure_dirs, find_ptbxl_root)
from igraphecg.utils.seed import set_seed

log = get_logger("prepare_ptbxl")


def main():
    _, cfg = parse_args_with_config("Prepare PTB-XL median-beat dataset")
    ensure_dirs()
    set_seed(cfg.get("seed", 42))

    fs = int(cfg["sampling_rate"])
    pre_ms, post_ms = cfg["window_pre_ms"], cfg["window_post_ms"]

    ptbxl_root = find_ptbxl_root()
    if ptbxl_root is None:
        log.error("PTB-XL not found (no ptbxl_database.csv under data/raw). Unpack the dataset first.")
        sys.exit(2)
    log.info(f"PTB-XL root: {ptbxl_root}")

    # 1) database and label mapping
    db = pl.load_database(ptbxl_root)
    scp = pl.load_scp_statements(ptbxl_root)
    code2super = lu.build_code_to_superclass(scp)
    db = lu.assign_single_label(db, code2super,
                                min_likelihood=cfg["min_likelihood"],
                                exclude_hyp_overlap=cfg["exclude_hyp_overlap"])
    db = pl.assign_official_splits(db)

    clean = db[db["is_clean"]].copy()
    if cfg.get("debug_limit", 0):
        clean = clean.iloc[: int(cfg["debug_limit"])]
    log.info(f"clean single-label subset: {len(clean)} / full database {len(db)}")

    # count the rejection reasons
    reject = {
        "multi_target_superclass": int((db["n_target_super"] > 1).sum()),
        "no_target_superclass": int((db["n_target_super"] == 0).sum()),
        "hyp_overlap_excluded": int(((db["n_target_super"] == 1) & db["has_hyp"]).sum())
        if cfg["exclude_hyp_overlap"] else 0,
    }

    # 2) per-record processing
    lead_idx = LEAD_TO_IDX
    signals, labels, label_names, folds, ages, sexes, rec_ids, qualities = ([] for _ in range(8))
    rej_rpeak = rej_nan = rej_const = rej_amp = rej_read = 0

    from tqdm import tqdm
    for ecg_id, row in tqdm(clean.iterrows(), total=len(clean), desc="processing"):
        try:
            sig, _ = pl.read_record_signal(ptbxl_root, row, fs)
        except Exception as e:  # noqa: BLE001
            rej_read += 1
            continue
        if np.isnan(sig).all(axis=1).any():  # an entirely NaN lead
            rej_nan += 1
            continue
        sig = bandpass_filter(sig, fs, cfg["bandpass_low"], cfg["bandpass_high"], cfg["bandpass_order"])
        if cfg.get("apply_notch", False):
            sig = notch_filter(sig, fs, cfg["notch_freq"])

        q = check_quality(sig, max_mv=cfg["max_abs_mv"])
        if q.has_nan:
            rej_nan += 1; continue
        if q.n_constant_leads > 0:
            rej_const += 1; continue
        if q.abnormal_amplitude:
            rej_amp += 1; continue

        res = be.extract_median_beat(sig, fs, lead_idx, pre_ms, post_ms)
        if not res.success or res.quality < cfg.get("min_beat_quality", 0.0):
            rej_rpeak += 1
            continue

        signals.append(res.median_beat)
        labels.append(int(row["label"]))
        label_names.append(row["label_name"])
        folds.append(int(row["strat_fold"]))
        ages.append(float(row.get("age", np.nan)))
        sexes.append(int(row.get("sex", -1)) if not pd.isna(row.get("sex", np.nan)) else -1)
        rec_ids.append(int(ecg_id))
        qualities.append(float(res.quality))

    signals = np.stack(signals, axis=0).astype(np.float32)  # [N,12,T] physical mV (unscaled)
    labels = np.asarray(labels, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int64)
    log.info(f"processed {len(signals)} records successfully, shape={signals.shape}")

    # 3) lead-wise robust scaling fitted on train only, to avoid leakage
    train_mask = folds <= 8
    scaler = RobustLeadScaler().fit(signals[train_mask])
    log.info(f"scaler median={np.round(scaler.median_,3)}")

    # 4) write npz (signals kept as unscaled physical mV; scaler stored so models can scale on demand)
    out_npz = PROJECT_ROOT / cfg["out_npz"]
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        record_id=np.asarray(rec_ids, dtype=np.int64),
        signal_12lead=signals,
        label=labels,
        label_name=np.asarray(label_names),
        fold=folds,
        age=np.asarray(ages, dtype=np.float32),
        sex=np.asarray(sexes, dtype=np.int64),
        sampling_rate=np.int64(fs),
        r_peak_quality=np.asarray(qualities, dtype=np.float32),
        lead_names=np.asarray(CANONICAL_LEADS),
        scaler_median=scaler.median_,
        scaler_iqr=scaler.iqr_,
        window_pre_ms=np.float32(pre_ms),
        window_post_ms=np.float32(post_ms),
    )
    log.info(f"saved {out_npz}")

    # 5) metadata csv
    meta = pd.DataFrame({
        "record_id": rec_ids, "label": labels, "label_name": label_names,
        "fold": folds, "split": np.where(folds <= 8, "train", np.where(folds == 9, "val", "test")),
        "age": ages, "sex": sexes, "r_peak_quality": qualities,
    })
    out_meta = PROJECT_ROOT / cfg["out_metadata"]
    meta.to_csv(out_meta, index=False)
    log.info(f"saved {out_meta}")

    # 6) split x class count table
    table = lu.class_count_table(meta)
    table.to_csv(TABLES_DIR / "data_split_summary.csv")
    log.info("\nSample counts:\n" + table.to_string())

    # rejection counts
    rej_df = pd.DataFrame({
        "reason": ["multi_target_superclass", "no_target_superclass", "hyp_overlap_excluded",
                   "read_fail", "nan", "constant_lead", "abnormal_amplitude", "rpeak/beat_fail"],
        "count": [reject["multi_target_superclass"], reject["no_target_superclass"],
                  reject["hyp_overlap_excluded"], rej_read, rej_nan, rej_const, rej_amp, rej_rpeak],
    })
    rej_df.to_csv(TABLES_DIR / "data_rejection_summary.csv", index=False)
    log.info("\nRejection counts:\n" + rej_df.to_string(index=False))

    # 7) random visual check
    n_plot = int(cfg.get("n_random_plots", 20))
    fig_dir = FIGURES_DIR / "random_median_beats"
    rng = np.random.default_rng(cfg.get("seed", 42))
    idx = rng.choice(len(signals), size=min(n_plot, len(signals)), replace=False)
    for i in idx:
        plot_median_beat(signals[i], fs, fig_dir / f"rec{rec_ids[i]}_{label_names[i]}.png",
                         title=f"record {rec_ids[i]} | {label_names[i]} | q={qualities[i]:.2f}",
                         pre_ms=pre_ms)
    log.info(f"saved {len(idx)} random median beats to {fig_dir}")

    # archive the run configuration
    save_config({**cfg, "n_processed": int(len(signals))},
                PROCESSED_DIR / "prepare_run_config.yaml")
    log.info("Stage 0 done.")


if __name__ == "__main__":
    main()
