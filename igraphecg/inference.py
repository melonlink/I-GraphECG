"""Inference-only model interface (wraps the verified surrogate; never trains).

load_model(): load the locked eikonal-B encoder/decoder (+ classifier optional).
predict_theta(): scaled beats -> theta (encoder).
reconstruct(): theta -> 12-lead reconstruction (decoder).
"""
from __future__ import annotations
import numpy as np

from .config import LEADFIELD_RANK, default_checkpoint


def load_model(n_t: int, ckpt_path=None, device="cpu"):
    """Load encoder+decoder (inference mode). Returns (enc, dec).

    ckpt_path=None means the locked lineage. This default used to be EIKONAL_B_CKPT, so the
    package's advertised inference API loaded a model the paper does not report. The
    documented reproduction commands pass --checkpoint and were unaffected; anyone following
    the README's `from igraphecg import load_model` was not.
    """
    if ckpt_path is None:
        ckpt_path = default_checkpoint()
    from igraphecg.evaluation.round2_io import load_encoder_decoder
    enc, dec = load_encoder_decoder(str(ckpt_path), n_t, LEADFIELD_RANK, device)
    enc.eval(); dec.eval()
    return enc, dec


def predict_theta(enc, dec, scaled_beats: np.ndarray, device="cpu", batch: int = 512):
    """scaled_beats:[N,12,T] -> theta:[N,47] (and z). Inference only."""
    from igraphecg.evaluation.round2_io import infer_theta
    return infer_theta(enc, dec, scaled_beats, device=device, batch=batch)


def reconstruct(dec, theta: np.ndarray, device="cpu", batch: int = 512) -> np.ndarray:
    """theta:[N,47] -> reconstructed scaled 12-lead beats [N,12,T] (decoder forward)."""
    import torch
    outs = []
    with torch.no_grad():
        for s in range(0, len(theta), batch):
            th = torch.from_numpy(theta[s:s + batch].astype(np.float32)).to(device)
            y12, _ = dec.forward(th)
            outs.append(y12.cpu().numpy())
    return np.concatenate(outs)


class InferenceError(RuntimeError):
    pass


def assert_inference_only(obj) -> None:
    """Guard used by external runs: torch modules must be in eval() mode."""
    import torch
    if isinstance(obj, torch.nn.Module) and obj.training:
        raise InferenceError("model is in training mode; external validation is inference-only")
