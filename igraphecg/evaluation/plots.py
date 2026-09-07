"""Plot helpers: confusion matrix, random median-beat samples, training loss curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless environment
import matplotlib.pyplot as plt
import numpy as np

from ..data.ptbxl_loader import CANONICAL_LEADS


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: str | Path,
                          title: str = "Confusion Matrix", normalize: bool = False) -> None:
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thr = cm.max() / 2 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_median_beat(beat12: np.ndarray, fs: float, path: str | Path,
                     title: str = "", pre_ms: float = 300.0) -> None:
    """Plot one 12-lead median beat (4x3 subplots). beat12: [12, T]."""
    n_t = beat12.shape[1]
    t = (np.arange(n_t) / fs - pre_ms / 1000.0) * 1000.0  # ms, R peak at 0
    fig, axes = plt.subplots(4, 3, figsize=(11, 9), sharex=True)
    for li, ax in enumerate(axes.ravel()):
        ax.plot(t, beat12[li], lw=0.9)
        ax.axvline(0, color="r", ls=":", lw=0.6)
        ax.set_title(CANONICAL_LEADS[li], fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.supxlabel("time relative to R (ms)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_recon_compare(y_real: np.ndarray, y_recon: np.ndarray, t_ms: np.ndarray,
                       path: str | Path, title: str = "") -> None:
    """Real vs reconstructed 12-lead overlay. y_*: [12, T], t_ms: [T] (ms)."""
    fig, axes = plt.subplots(4, 3, figsize=(11, 9), sharex=True)
    for li, ax in enumerate(axes.ravel()):
        ax.plot(t_ms, y_real[li], lw=1.0, label="real", color="black")
        ax.plot(t_ms, y_recon[li], lw=1.0, label="recon", color="tab:red", alpha=0.8)
        ax.axvline(0, color="gray", ls=":", lw=0.5)
        ax.set_title(CANONICAL_LEADS[li], fontsize=9)
        ax.grid(alpha=0.3)
        if li == 0:
            ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.supxlabel("time relative to R (ms)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_param_histograms(theta: np.ndarray, names: list[str], path: str | Path,
                          title: str = "parameter distributions") -> None:
    """Grid of parameter histograms for theta:[N,P]."""
    P = theta.shape[1]
    ncol = 6
    nrow = int(np.ceil(P / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 1.8 * nrow))
    for j, ax in enumerate(axes.ravel()):
        if j < P:
            ax.hist(theta[:, j], bins=30, color="tab:blue", alpha=0.8)
            ax.set_title(names[j] if j < len(names) else str(j), fontsize=7)
            ax.tick_params(labelsize=6)
        else:
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_heatmap(M: np.ndarray, path: str | Path, title: str = "",
                 xticks: list[str] | None = None, yticks: list[str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-np.abs(M).max(), vmax=np.abs(M).max())
    if xticks:
        ax.set_xticks(range(len(xticks)), xticks, rotation=45, fontsize=8)
    if yticks:
        ax.set_yticks(range(len(yticks)), yticks, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_loss_curve(history: dict, path: str | Path, title: str = "Training") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ("train_loss", "val_loss"):
        if key in history and len(history[key]):
            ax.plot(history[key], label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
