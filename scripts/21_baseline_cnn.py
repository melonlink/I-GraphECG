"""B1: ResNet1D end-to-end baseline (input: raw 12-lead median beat).

Run:
    python scripts/21_baseline_cnn.py --config configs/baseline_cnn.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from igraphecg import repro   # output keys in configs are output-root relative

from igraphecg.data.dataset import load_processed, split_indices
from igraphecg.data.label_utils import TARGET_CLASSES
from igraphecg.data.preprocess import RobustLeadScaler
from igraphecg.evaluation.metrics import bootstrap_ci, compute_metrics, per_class_metrics
from igraphecg.evaluation.plots import plot_confusion_matrix, plot_loss_curve
from igraphecg.utils.config import parse_args_with_config
from igraphecg.utils.logging import get_logger
from igraphecg.utils.paths import PROJECT_ROOT, ensure_dirs
from igraphecg.utils.seed import set_seed

log = get_logger("baseline_cnn")
N_CLASSES = len(TARGET_CLASSES)


def make_loader(sig, lab, batch, shuffle, num_workers):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.from_numpy(sig.astype(np.float32)),
                       torch.from_numpy(lab.astype(np.int64)))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=num_workers)


@np.errstate(all="ignore")
def predict_proba(model, loader, device):
    import torch
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def main():
    import torch
    import torch.nn as nn

    from igraphecg.models.resnet1d import ResNet1D, count_params

    _, cfg = parse_args_with_config("Train ResNet1D baseline (B1)")
    ensure_dirs()
    set_seed(cfg.get("seed", 42))
    tr = cfg["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device = {device}")

    data = load_processed(PROJECT_ROOT / cfg["npz"])
    signals = data["signal_12lead"].astype(np.float32)
    labels = data["label"].astype(np.int64)
    # apply the train-set robust scaler (statistics stored in the npz)
    scaler = RobustLeadScaler.from_dict({"median": data["scaler_median"], "iqr": data["scaler_iqr"]})
    signals = scaler.transform(signals)
    idx = split_indices(data["fold"])
    log.info(f"N={len(labels)} train/val/test={len(idx['train'])}/{len(idx['val'])}/{len(idx['test'])}")

    nw = tr.get("num_workers", 0)
    tl = make_loader(signals[idx["train"]], labels[idx["train"]], tr["batch_size"], True, nw)
    vl = make_loader(signals[idx["val"]], labels[idx["val"]], tr["batch_size"], False, nw)
    te = make_loader(signals[idx["test"]], labels[idx["test"]], tr["batch_size"], False, nw)

    model = ResNet1D(in_ch=signals.shape[1], n_classes=N_CLASSES, **cfg.get("model", {})).to(device)
    log.info(f"ResNet1D params = {count_params(model):,}")

    # class-weighted cross entropy
    if tr.get("class_weighted_loss", True):
        counts = np.bincount(labels[idx["train"]], minlength=N_CLASSES).astype(float)
        w = counts.sum() / (N_CLASSES * np.clip(counts, 1, None))
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    else:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    use_amp = bool(tr.get("amp", True)) and device.type == "cuda"
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auroc, best_state, patience = -1.0, None, 0
    history = {"train_loss": [], "val_loss": [], "val_auroc": []}

    for epoch in range(tr["epochs"]):
        model.train()
        ep_loss = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(xb), yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(opt)
            scaler_amp.update()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(idx["train"])

        vprob, vy = predict_proba(model, vl, device)
        vloss = float(nn.functional.cross_entropy(
            torch.tensor(np.log(np.clip(vprob, 1e-7, 1))), torch.tensor(vy)).item())
        vm = compute_metrics(vy, vprob, N_CLASSES)
        history["train_loss"].append(ep_loss)
        history["val_loss"].append(vloss)
        history["val_auroc"].append(vm["macro_auroc"])
        log.info(f"epoch {epoch+1:02d}/{tr['epochs']}  train_loss={ep_loss:.4f}  "
                 f"val_AUROC={vm['macro_auroc']:.4f}  val_F1={vm['macro_f1']:.4f}")

        if vm["macro_auroc"] > best_auroc:
            best_auroc = vm["macro_auroc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= tr.get("early_stop_patience", 12):
                log.info(f"early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    out_ckpt = repro.outputs() / cfg["out_ckpt"]
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "best_val_auroc": best_auroc}, out_ckpt)
    log.info(f"best val AUROC={best_auroc:.4f}; ckpt -> {out_ckpt}")

    rows, percls = [], []
    cm_test = None
    for split, loader in (("val", vl), ("test", te)):
        prob, y = predict_proba(model, loader, device)
        m = compute_metrics(y, prob, N_CLASSES)
        lo, hi = bootstrap_ci(y, prob, N_CLASSES, "macro_auroc")
        rows.append({"model": "B1_resnet1d", "split": split, "macro_auroc": m["macro_auroc"],
                     "macro_auprc": m["macro_auprc"], "macro_f1": m["macro_f1"],
                     "balanced_accuracy": m["balanced_accuracy"], "auroc_ci_lo": lo, "auroc_ci_hi": hi,
                     "n": len(y)})
        for pc in per_class_metrics(y, prob, N_CLASSES, TARGET_CLASSES):
            percls.append({"model": "B1_resnet1d", "split": split, **pc})
        log.info(f"[{split}] AUROC={m['macro_auroc']:.4f} F1={m['macro_f1']:.4f} bACC={m['balanced_accuracy']:.4f}")
        if split == "test":
            cm_test = m["confusion_matrix"]

    out_metrics = repro.outputs() / cfg["out_metrics"]
    pd.DataFrame(rows).to_csv(out_metrics, index=False)
    pd.DataFrame(percls).to_csv(str(out_metrics).replace(".csv", "_perclass.csv"), index=False)
    plot_confusion_matrix(cm_test, TARGET_CLASSES, repro.outputs() / cfg["out_cm"], title="B1 ResNet1D (test)")
    plot_loss_curve(history, repro.outputs() / cfg["out_loss_curve"], title="ResNet1D training")
    log.info("B1 done.")


if __name__ == "__main__":
    main()
