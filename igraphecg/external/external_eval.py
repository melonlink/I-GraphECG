"""Command-2 external validation: reconstruction + observability on Georgia / CPSC2018.

Inference only, PTB-XL-trained model, PTB-XL scaler reused. Resamples 500->100 Hz inside the
frozen pipeline. Writes results JSON + manifest CSV. (Classification is added separately.)

Run (from repo root):
  <py> -m igraphecg.external.external_eval --dataset georgia \
       --data_dir challenge_2021/training/georgia --out results/ext_georgia.json
"""
from __future__ import annotations
import argparse, json, csv, pickle, time
from pathlib import Path
from collections import Counter
import numpy as np

from ..config import TARGET_FS, INDEPENDENT_LEADS, SCALER_DIR, SUPERCLASSES, default_checkpoint
from ..sources.registry import get_source
from ..preprocess.pipeline import make_median_beat
from ..labels.map_snomed import load_snomed_map, map_record_label
from ..labels.superclass import EXCLUDE_SUPERCLASS

LEAD_SETS = {"12": list("I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6".split(",")),
             "II+V1+V5": ["II", "V1", "V5"],
             "I+V1+V4": ["I", "V1", "V4"],
             "II": ["II"]}


def _load_signal_scaler(path):
    from igraphecg.data.preprocess import RobustLeadScaler
    with open(path, "rb") as f:
        return RobustLeadScaler.from_dict(pickle.load(f))


def build_external(dataset, data_dir, scaler, snomed, exclude=(), max_records=None):
    """Return (beats[N,12,T], labels[N], manifest dict)."""
    src = get_source(dataset, data_dir)
    beats, labels, ids = [], [], []
    reasons = Counter(); kept_by_class = Counter()
    n_total = 0
    for k, rec in enumerate(src.iter_records()):
        if max_records and n_total >= max_records:
            break
        n_total += 1
        y = map_record_label(rec.source_labels, snomed)
        if y is None:
            reasons["label_other_or_multilabel_or_HYP"] += 1; continue
        if y in exclude:
            reasons[f"excluded_class_{y}"] += 1; continue
        beat = make_median_beat(rec, target_fs=TARGET_FS, scaler=scaler, external=True)
        if beat is None:
            reasons["no_clean_median_beat"] += 1; continue
        beats.append(beat); labels.append(SUPERCLASSES.index(y)); ids.append(rec.record_id)
        kept_by_class[y] += 1
    manifest = {"dataset": dataset, "n_scanned": n_total, "n_kept": len(beats),
                "kept_by_class": dict(kept_by_class), "drop_reasons": dict(reasons)}
    return (np.stack(beats) if beats else np.empty((0, 12, 100), np.float32),
            np.array(labels, int), ids, manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--signal_scaler", default=str(SCALER_DIR / "ptbxl_train_signal.pkl"))
    ap.add_argument("--label_map", default=None)
    ap.add_argument("--checkpoint", default=None)   # None -> the locked lineage
    ap.add_argument("--exclude_classes", default="")     # e.g. "MI"
    ap.add_argument("--max_records", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.checkpoint is None:
        a.checkpoint = str(default_checkpoint())

    import torch
    from ..inference import load_model, predict_theta, reconstruct, assert_inference_only
    from .metrics import reconstruction_metrics
    from .observability import effrank_for_leadsets, greedy_dopt_3lead, leadfield_svd
    from ..labels.map_snomed import DEFAULT_MAP_CSV
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    scaler = _load_signal_scaler(a.signal_scaler)
    assert getattr(scaler, "median_", None) is not None, "scaler must be pre-fit (no external refit)"
    snomed = load_snomed_map(a.label_map or DEFAULT_MAP_CSV)
    exclude = tuple(x for x in a.exclude_classes.split(",") if x)

    t0 = time.time()
    beats, labels, ids, manifest = build_external(a.dataset, a.data_dir, scaler, snomed,
                                                  exclude=exclude, max_records=a.max_records)
    manifest["seconds_preprocess"] = round(time.time() - t0, 1)
    result = {"dataset": a.dataset, "manifest": manifest}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    if len(beats) == 0:
        result["note"] = "no records kept"
        Path(a.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps(result, indent=2, ensure_ascii=False)); return

    enc, dec = load_model(beats.shape[2], ckpt_path=a.checkpoint, device=dev)
    assert_inference_only(enc); assert_inference_only(dec)
    theta, _ = predict_theta(enc, dec, beats, device=dev)
    recon = reconstruct(dec, theta, device=dev)
    result["reconstruction"] = reconstruction_metrics(beats, recon)

    # observability on the external theta distribution (capped to <=400, balanced, seed-42,
    # to match the paper's FIM subset and keep jacfwd tractable)
    radius = dec.pspace.radius.detach().cpu()
    rng = np.random.default_rng(42)
    if len(theta) > 400:
        per = max(1, 400 // len(set(labels.tolist())))
        sel = []
        for c in sorted(set(labels.tolist())):
            ci = np.where(labels == c)[0]
            sel.append(rng.choice(ci, min(per, len(ci)), replace=False))
        obs_idx = np.concatenate(sel)
    else:
        obs_idx = np.arange(len(theta))
    th_t = torch.from_numpy(theta[obs_idx].astype(np.float32))
    result["observability_n"] = int(len(obs_idx))
    result["observability"] = {
        "effective_rank": effrank_for_leadsets(dec, th_t, radius, LEAD_SETS, device=dev),
    }
    sel, ld = greedy_dopt_3lead(dec, th_t, radius, INDEPENDENT_LEADS, n_select=3, device=dev)
    result["observability"]["dopt_3lead"] = {"selected": sel, "logdet": ld}
    result["observability"]["leadfield_svd"] = leadfield_svd(dec)   # model-fixed (ST near-null)

    # dump predictions/labels for later bootstrap CI
    np.savez(Path(a.out).with_suffix(".theta.npz"), theta=theta, labels=labels)
    Path(a.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    # manifest CSV
    mpath = Path(a.out).with_name(f"manifest_{a.dataset}.csv")
    with open(mpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["key", "value"])
        for k, v in manifest.items(): w.writerow([k, v])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
