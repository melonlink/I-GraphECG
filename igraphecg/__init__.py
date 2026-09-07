"""I-GraphECG — interpretable graph-electrophysiology ECG modeling & inference.

Single package, two layers:
  research core    : data / features / models / training / evaluation
  delivery layer   : sources (external DB adapters) / preprocess / inference / external / labels

Real-time inference entry points are exposed lazily at the top level so that a plain
``import igraphecg`` stays light (no torch import until you actually need a model):

    from igraphecg import make_median_beat, load_model, predict_theta, reconstruct
    rec   = make_median_beat(signal_12lead, fs)        # preprocess one record
    enc, dec = load_model(n_samples)                   # frozen encoder/decoder
    theta, _ = predict_theta(enc, dec, scaled)         # identifiable parameters
"""
from __future__ import annotations

from . import config  # light: stdlib only, safe to import eagerly

__version__ = "1.0.0"

# name -> (module, attribute); imported on first access to avoid pulling torch eagerly
_LAZY = {
    "load_model": ("igraphecg.inference", "load_model"),
    "predict_theta": ("igraphecg.inference", "predict_theta"),
    "reconstruct": ("igraphecg.inference", "reconstruct"),
    "make_median_beat": ("igraphecg.preprocess.pipeline", "make_median_beat"),
    "get_source": ("igraphecg.sources.registry", "get_source"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'igraphecg' has no attribute {name!r}")
    import importlib
    module, attr = target
    return getattr(importlib.import_module(module), attr)


def __dir__():
    return sorted([*globals().keys(), *_LAZY])


__all__ = ["config", "__version__", *_LAZY]
