"""Lead field and 12-lead derivation (contribution 3: physical constraints on the lead field).

Only the 8 independent leads [I, II, V1, V2, V3, V4, V5, V6] are predicted; III, aVR, aVL
and aVF follow from the Einthoven-Goldberger relations, which enforces the limb-lead
constraints and cuts the degrees of freedom.

derive_12leads_from_independent works on numpy arrays and torch tensors alike (lead axis = -2).
"""
from __future__ import annotations

# Independent lead order
INDEPENDENT_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
# Output 12-lead order (matches ptbxl_loader.CANONICAL_LEADS)
OUTPUT_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

BASIS = ("I", "II")
DERIVED = ("III", "aVR", "aVL", "aVF")
# Einthoven-Goldberger matrix D:[4,2] in physical units; rows = DERIVED, cols = BASIS = (I, II)
#   III = -I + II ;  aVR = -I/2 - II/2 ;  aVL = I - II/2 ;  aVF = -I/2 + II
_D_PHYS = ((-1.0, 1.0), (-0.5, -0.5), (1.0, -0.5), (-0.5, 1.0))


def _scaler_stats(scaler, n_leads: int):
    """Accept RobustLeadScaler / {"median","iqr"} / (median, iqr); return the lists (med, iqr)."""
    if hasattr(scaler, "median_") and hasattr(scaler, "iqr_"):
        med, iqr = scaler.median_, scaler.iqr_
    elif hasattr(scaler, "median") and hasattr(scaler, "iqr"):
        med, iqr = scaler.median, scaler.iqr
    elif isinstance(scaler, dict):
        med, iqr = scaler["median"], scaler["iqr"]
    else:
        med, iqr = scaler
    med = [float(v) for v in med]
    iqr = [float(v) for v in iqr]
    if len(med) != n_leads or len(iqr) != n_leads:
        raise ValueError(f"scaler stats have {len(med)}/{len(iqr)} leads, expected {n_leads}")
    return med, iqr


def scaled_einthoven_coeffs(scaler, lead_names=None) -> dict:
    """Physical coefficients D -> affine coefficients in scaled space, {derived: (A_I, B_II, C)}.

    Derivation (B1): substituting y_phys = s*y_s + m into y_phys_k = sum_j D[k,j] y_phys_j and
    dividing by s_k gives
        D_scaled[k,j] = D[k,j] * iqr_j / iqr_k
        c_k          = (sum_j D[k,j]*med_j - med_k) / iqr_k
    Applying the physical coefficients directly in the scaled space is wrong: the IQR differs
    from lead to lead.
    """
    names = list(lead_names) if lead_names is not None else list(OUTPUT_LEADS)
    med, iqr = _scaler_stats(scaler, n_leads=len(names))
    bi = [names.index(n) for n in BASIS]
    out = {}
    for k, dn in enumerate(DERIVED):
        di = names.index(dn)
        a = _D_PHYS[k][0] * iqr[bi[0]] / iqr[di]
        b = _D_PHYS[k][1] * iqr[bi[1]] / iqr[di]
        c = (_D_PHYS[k][0] * med[bi[0]] + _D_PHYS[k][1] * med[bi[1]] - med[di]) / iqr[di]
        out[dn] = (float(a), float(b), float(c))
    return out


def derive_12leads_from_independent(y_ind, scaler=None, lead_names=None):
    """y_ind: [..., 8, T] (order: INDEPENDENT_LEADS) -> [..., 12, T] (order: OUTPUT_LEADS).

    III = II - I
    aVR = -(I + II) / 2
    aVL = I - II / 2
    aVF = II - I / 2
    """
    def lead(name):
        return y_ind[..., INDEPENDENT_LEADS.index(name), :]

    I, II = lead("I"), lead("II")
    if scaler is None:
        # Physical units (mV): textbook Einthoven-Goldberger, bit-identical to the legacy path
        derived = {"III": II - I, "aVR": -(I + II) / 2.0,
                   "aVL": I - II / 2.0, "aVF": II - I / 2.0}
    else:
        # Lead-wise robust-scaled: same relation as an affine map with IQR ratios and median offset
        co = scaled_einthoven_coeffs(scaler, lead_names)
        derived = {n: co[n][0] * I + co[n][1] * II + co[n][2] for n in DERIVED}

    # Stack in OUTPUT_LEADS order
    parts = {
        "I": I, "II": II, **derived,
        "V1": lead("V1"), "V2": lead("V2"), "V3": lead("V3"),
        "V4": lead("V4"), "V5": lead("V5"), "V6": lead("V6"),
    }
    seq = [parts[n] for n in OUTPUT_LEADS]

    # Both numpy and torch provide stack; dispatch by duck typing
    try:
        import torch
        if isinstance(y_ind, torch.Tensor):
            return torch.stack(seq, dim=-2)
    except ImportError:
        pass
    import numpy as np
    return np.stack(seq, axis=-2)
