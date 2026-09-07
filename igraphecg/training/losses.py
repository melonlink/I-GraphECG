"""Round 2 reconstruction training losses (ROUND2 §7).

Total loss L = λ_rec·L_rec + λ_slope·L_slope + λ_qrs·L_qrs + λ_st·L_st + λ_t·L_t
               + λ_phys·L_phys + λ_H·L_H

All time windows are turned into masks from the time axis (seconds); no hard-coded indices.

Optional (off by default): the five waveform terms (rec/slope/qrs/st/t) can be restricted to a
**row subset** of the lead axis; see `total_loss(..., recon_rows=...)` and `resolve_recon_rows`.
The default recon_rows=None means all 12 output leads => bit-for-bit legacy behaviour. The
phys and H terms are unaffected.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from ..models.surrogate_decoder import ParamSpace, _SEG, PARAM_DIM

# Time windows relative to the R peak (t=0), in seconds
QRS_WIN = (-0.06, 0.10)
ST_WIN = (0.06, 0.14)
T_WIN = (0.12, 0.45)
# Note: activation-order penalty removed; eikonal propagation (edge_delay>=0) keeps delta monotone.

# --- Reconstruction channel subset (lead axis = second-to-last dim) ------------
# CANONICAL_LEADS = [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6].
# III/aVR/aVL/aVF are **exact affine functions** of I and II (Einthoven-Goldberger, see
# models/leadfield.py), so weighting all 12 channels equally counts the limb subspace error
# about 6/12 of the time while each precordial lead counts only 1/12.
# NON_DERIVED_ROWS gives the row indices of the 8 non-derived channels (I, II, V1..V6),
# which removes that implicit weighting.
NON_DERIVED_ROWS: tuple[int, ...] = (0, 1, 6, 7, 8, 9, 10, 11)

# String aliases writable from config -> row indices (None = all 12 rows = legacy behaviour)
RECON_CHANNEL_SETS: dict[str, tuple[int, ...] | None] = {
    "all": None,
    "all12": None,
    "non_derived": NON_DERIVED_ROWS,
    "nonderived": NON_DERIVED_ROWS,
}


def resolve_recon_rows(spec) -> tuple[int, ...] | None:
    """Resolve a config value to row indices; None/"all" -> None (all 12 rows, legacy behaviour).

    Accepts None, a RECON_CHANNEL_SETS string alias, or an explicit int sequence (e.g. [0, 1, 6]).
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key not in RECON_CHANNEL_SETS:
            raise ValueError(
                f"unknown recon_channels={spec!r}; expected one of "
                f"{sorted(RECON_CHANNEL_SETS)} or an explicit list of row indices")
        return RECON_CHANNEL_SETS[key]
    if isinstance(spec, Sequence):
        rows = tuple(int(i) for i in spec)
        if not rows:
            raise ValueError("recon_channels must not be an empty sequence")
        return rows
    raise TypeError(f"recon_channels must be None / str / sequence of int, got {type(spec)!r}")


def _select_rows(y: torch.Tensor, yhat: torch.Tensor, rows):
    """Select rows along lead axis (-2); rows=None returns inputs **as is** (bit-exact no-op)."""
    if rows is None:
        return y, yhat
    idx = torch.as_tensor(rows, device=y.device, dtype=torch.long)
    return y.index_select(-2, idx), yhat.index_select(-2, idx)


def build_masks(t: torch.Tensor) -> dict[str, torch.Tensor]:
    def m(win):
        return ((t >= win[0]) & (t <= win[1])).float()
    return {"qrs": m(QRS_WIN), "st": m(ST_WIN), "t": m(T_WIN)}


def _masked_mse(y, yhat, mask):
    w = mask.view(1, 1, -1)
    num = (((y - yhat) ** 2) * w).sum()
    den = w.sum() * y.shape[0] * y.shape[1] + 1e-8
    return num / den


def rec_loss(y, yhat, eps: float = 1e-6):
    return ((y - yhat) ** 2).sum() / ((y ** 2).sum() + eps)


def slope_loss(y, yhat):
    dy = y[..., 1:] - y[..., :-1]
    dyh = yhat[..., 1:] - yhat[..., :-1]
    return ((dy - dyh) ** 2).mean()


def _seg_slice(name: str) -> slice:
    i = 0
    for nm, k in _SEG:
        if nm == name:
            return slice(i, i + k)
        i += k
    raise KeyError(name)


_TAU_DEP_SL = _seg_slice("tau_dep")
_TAU_REP_SL = _seg_slice("tau_rep")


def _prior_weights(device) -> "torch.Tensor":
    w = torch.ones(PARAM_DIM, device=device)
    w[_TAU_DEP_SL] = 4.0     # tau_dep weakly identifiable -> stronger pull to center
    w[_TAU_REP_SL] = 6.0     # tau_rep least identifiable (45% at bounds) -> strongest pull
    return w


def phys_loss(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Weighted normal prior (heavier on tau_dep/tau_rep) + ST source sparsity + boundary penalty.

    Activation-order penalty removed; eikonal propagation (edge_delay>=0) keeps delta monotone.
    """
    p = ParamSpace.unpack(theta)
    w = _prior_weights(z.device)
    prior = ((z ** 2) * w).mean()                 # weighted pull-to-center
    st_sparse = p["alpha_ST"].abs().mean()        # ST source sparsity (alpha_ST least identifiable)
    tanh_z = torch.tanh(z)
    boundary = torch.relu(tanh_z.abs() - 0.9).pow(2).mean()
    return 0.1 * prior + 1.0 * st_sparse + 1.0 * boundary   # st_sparse 0.5->1.0


def leadfield_loss(H_ind: torch.Tensor, lambda_smooth: float = 1.0) -> torch.Tensor:
    """Frobenius regularization on H + V1–V6 smoothness (rows 2..7 are V1..V6)."""
    fro = (H_ind ** 2).mean()
    v = H_ind[2:8]  # V1..V6 rows
    smooth = ((v[1:] - v[:-1]) ** 2).mean()
    return fro + lambda_smooth * smooth


def total_loss(y, yhat, z, theta, H_ind, masks, weights: dict, recon_rows=None):
    """recon_rows=None (default): the waveform terms use all 12 output leads - legacy, bit-exact.

    With explicit row indices (e.g. NON_DERIVED_ROWS) only rec/slope/qrs/st/t move to that row
    subset; phys / H are lead-independent and unaffected. The normalization of each term is
    unchanged (rec = relative energy on the subset, slope = mean over subset elements, masked
    terms = in-window means over subset elements), so loss_weights need no rescaling.
    """
    yr, yhr = _select_rows(y, yhat, recon_rows)
    comps = {
        "rec": rec_loss(yr, yhr),
        "slope": slope_loss(yr, yhr),
        "qrs": _masked_mse(yr, yhr, masks["qrs"]),
        "st": _masked_mse(yr, yhr, masks["st"]),
        "t": _masked_mse(yr, yhr, masks["t"]),
        "phys": phys_loss(z, theta),
        "H": leadfield_loss(H_ind),
    }
    total = sum(weights.get(k, 0.0) * v for k, v in comps.items())
    comps["total"] = total
    return total, comps
