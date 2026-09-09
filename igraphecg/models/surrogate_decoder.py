"""Closed-form Eikonal-APD surrogate decoder (Round 2, model one).

8 cardiac equivalent nodes; each node approximates activation-repolarization by a
double-sigmoid action potential U_i(t). Source term s_i(t) = q_i·dU_i/dt + alpha_ST_i·P_i(t);
the low-rank lead field H=A@B projects onto 8 independent leads, from which the 12 leads
follow through the Einthoven-Goldberger relations.

Parameters are constrained to a physiologically plausible range by theta =
center + radius·tanh(z), with z unbounded (encoder or oracle output).

Physical units: time in s; alpha_ST is a source strength in arbitrary units (absorbed by H and the robust scaling); q is a relative
source amplitude (dimensionless).
The double-sigmoid node action potential follows TASK_SPEC §16.1 / ROUND2 §5.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .leadfield import derive_12leads_from_independent

N_NODES = 8
VENT_NODES = [2, 3, 4, 5, 6, 7]          # ventricular nodes (D_act/D_rep/S_ST computed here)
N_ST = len(VENT_NODES)                    # alpha_ST on ventricular nodes only

# Conduction graph (directed tree, root = SA node 0). Edge order e0..e6:
EDGES = [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)]
N_EDGE = len(EDGES)                        # 7
_PARENT = {1: 0, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2}


def _path_matrix() -> torch.Tensor:
    """Path indicator matrix: M[node, edge]=1 when edge e lies on the root->node path."""
    M = torch.zeros(N_NODES, N_EDGE)
    for j in range(1, N_NODES):
        node = j
        while node != 0:
            par = _PARENT[node]
            M[j, EDGES.index((par, node))] = 1.0
            node = par
    return M


PATH_M = _path_matrix()                     # [8,7] fixed buffer

# Parameter layout (flat vector per sample) and segment lengths.
# Activation times come from eikonal propagation on the graph: delta = delta_root +
# M @ edge_delay (edge_delay>=0), replacing the 8 free deltas; total dimension is still 47.
_SEG = [("delta_root", 1), ("edge_delay", N_EDGE), ("APD", N_NODES), ("tau_dep", N_NODES),
        ("tau_rep", N_NODES), ("q", N_NODES), ("alpha_ST", N_ST), ("global_gain", 1)]
PARAM_DIM = sum(n for _, n in _SEG)        # 1 + 7 + 8*4 + 6 + 1 = 47


# Absolute activation-time window of the free-delay ablation (graph_delays=False): the seven
# edge_delay slots then hold the activation times of nodes 1..7 directly, and this window
# covers every time the graph parameterization can produce (-0.24+0.045 .. -0.12+0.33 s).
FREE_DELTA_BOUNDS = (-0.20, 0.25)


def _bounds(graph_delays: bool = True) -> tuple[list[float], list[float]]:
    """Return (lo, hi), the lower/upper parameter bounds of length PARAM_DIM, in _SEG order.

    graph_delays=False is the structural ablation without the conduction graph: the edge_delay
    slots become free absolute activation times of nodes 1..7 with the window FREE_DELTA_BOUNDS,
    so no ordering is imposed. The published lineage never sets it."""
    lo: list[float] = []
    hi: list[float] = []
    # delta_root (onset of atrial/SA activation, s): same as the former delta0
    lo += [-0.24]; hi += [-0.12]
    # edge_delay (conduction delay, s, >=0), ordered as in EDGES:
    #   e0:0->1 includes AV-nodal delay (longest); e1:1->2 His->septum; e2..e6:2->{3,4,5,6,7}
    if graph_delays:
        lo += [0.04];      hi += [0.20]        # e0 (0->1)
        lo += [0.005];     hi += [0.06]        # e1 (1->2)
        lo += [0.005] * 5; hi += [0.07] * 5    # e2..e6 (2->j)
    else:
        lo += [FREE_DELTA_BOUNDS[0]] * N_EDGE; hi += [FREE_DELTA_BOUNDS[1]] * N_EDGE
    # APD (action-potential duration, s): shorter for atria/AV node, longer for ventricles
    apd_lo = [0.08, 0.08, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18]
    apd_hi = [0.18, 0.18, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]
    lo += apd_lo; hi += apd_hi
    # tau_dep / tau_rep (rise/fall time constants, s) -- weakly identifiable, tightened in variant B
    lo += [0.007] * N_NODES; hi += [0.026] * N_NODES        # tau_dep: [0.007,0.026]
    lo += [0.035] * N_NODES; hi += [0.100] * N_NODES        # tau_rep: [0.035,0.100]
    # q (relative source amplitude) -- unchanged
    lo += [0.05] * N_NODES; hi += [3.00] * N_NODES
    # alpha_ST (ventricular nodes, arbitrary source units) -- tightened in variant B
    lo += [-0.25] * N_ST; hi += [0.25] * N_ST               # alpha_ST: [-0.25,0.25]
    # global_gain
    lo += [0.5]; hi += [2.0]
    return lo, hi


class ParamSpace:
    """Map z (unbounded) <-> theta (bounded); also unpacks segments and reports boundary rate.

    GRAPH_DELAYS is the process-wide default read by the static ``unpack`` (used by the feature
    and loss code, which see theta without a decoder). The checkpoint loader sets it from the
    checkpoint's config, so a free-delay ablation checkpoint is unpacked consistently everywhere;
    the published lineage leaves it True."""

    GRAPH_DELAYS = True

    def __init__(self, device=None, graph_delays: bool | None = None):
        self.graph_delays = ParamSpace.GRAPH_DELAYS if graph_delays is None else bool(graph_delays)
        lo, hi = _bounds(self.graph_delays)
        lo_t = torch.tensor(lo, dtype=torch.float32)
        hi_t = torch.tensor(hi, dtype=torch.float32)
        self.center = ((lo_t + hi_t) / 2.0)
        self.radius = ((hi_t - lo_t) / 2.0)
        if device is not None:
            self.center = self.center.to(device)
            self.radius = self.radius.to(device)

    def to(self, device):
        self.center = self.center.to(device)
        self.radius = self.radius.to(device)
        return self

    def theta(self, z: torch.Tensor) -> torch.Tensor:
        """z:[B,PARAM_DIM] -> theta:[B,PARAM_DIM], theta=center+radius·tanh(z)."""
        return self.center + self.radius * torch.tanh(z)

    def boundary_rate(self, z: torch.Tensor, thresh: float = 0.95) -> float:
        """Fraction of entries with |tanh(z)|>thresh, i.e. pinned against a bound."""
        with torch.no_grad():
            return float((torch.tanh(z).abs() > thresh).float().mean().item())

    @staticmethod
    def unpack(theta: torch.Tensor, graph_delays: bool | None = None) -> dict[str, torch.Tensor]:
        """Split theta into named segments. alpha_ST expands to 8 nodes (0 for atria/AV node)."""
        out, i = {}, 0
        for name, n in _SEG:
            out[name] = theta[..., i:i + n]
            i += n
        if ParamSpace.GRAPH_DELAYS if graph_delays is None else graph_delays:
            # Graph eikonal propagation: delta_j = delta_root + sum_{e in path(root->j)} edge_delay_e
            # edge_delay>=0 (enforced by bounds) -> delta_child > delta_parent holds by construction.
            M = PATH_M.to(theta.device, theta.dtype)                  # [8,7]
            out["delta"] = out["delta_root"] + torch.einsum("ne,...e->...n", M, out["edge_delay"])
        else:
            # Structural ablation: node 0 at delta_root, nodes 1..7 at free absolute times.
            out["delta"] = torch.cat([out["delta_root"], out["edge_delay"]], dim=-1)
        # alpha_ST covers ventricular nodes only -> expand to [..., 8]
        B = theta.shape[:-1]
        full = torch.zeros(*B, N_NODES, device=theta.device, dtype=theta.dtype)
        full[..., VENT_NODES] = out["alpha_ST"]
        out["alpha_ST_full"] = full
        out["global_gain"] = out["global_gain"][..., 0]  # scalar
        return out


class SurrogateDecoder(nn.Module):
    """Differentiable physical decoder: θ -> 12-lead median beat.

    Time axis t = linspace(-0.30, 0.69, T) seconds, R peak at t=0.
    Low-rank lead field H_ind = A @ B, A:[8,rank], B:[rank,8].
    """

    def __init__(self, n_t: int = 100, t0: float = -0.30, t1: float = 0.69,
                 leadfield_rank: int = 3, scaler=None, direct_12_leads: bool = False,
                 graph_delays: bool | None = None, leadfield_init_scale: float = 0.3):
        """scaler: if given (RobustLeadScaler / {"median","iqr"} / (median, iqr)), the decoder
        output is read as **lead-wise robust-scaled** coordinates and the derived leads use the
        affine coefficients of the scaled space. With None, the textbook Einthoven-Goldberger
        coefficients are applied in physical units (legacy behaviour, kept for compatibility).

        Note: the training target is the scaled signal, so a scaler **should** be passed at
        training and inference time; omitting it applies the physical coefficients in scaled
        coordinates (quantified in B1: III/aVL/aVF residuals degrade from 4-11% to 8-34%).

        Structural ablations (Reviewer 3): direct_12_leads=True replaces the [8, rank] lead field
        plus exact limb-lead relations by a [12, rank] lead field that projects onto all twelve
        displayed leads directly; graph_delays=False drops the conduction graph (free node
        activation times, see ParamSpace). Both default to the published behaviour.
        """
        super().__init__()
        self.n_t = n_t
        self.scaler = scaler
        self.direct_12_leads = bool(direct_12_leads)
        self.graph_delays = ParamSpace.GRAPH_DELAYS if graph_delays is None else bool(graph_delays)
        # rows of H holding V1..V6, for the smoothness term of the lead-field regulariser
        self.precordial_rows = slice(6, 12) if self.direct_12_leads else slice(2, 8)
        t = torch.linspace(t0, t1, n_t)
        self.register_buffer("t", t)
        # Low-rank lead field (globally shared parameters)
        # leadfield_init_scale: 0.3 in the published lineage (rank 3). An [8, r] @ [r, 8] product of
        # N(0, s^2) entries has entry variance r s^4, so a rank ablation keeps the initial lead-field
        # norm of the locked recipe with s = 0.3 (3 / r)^(1/4) (Reviewer 3, structural ablation).
        s0 = float(leadfield_init_scale)
        self.A = nn.Parameter(torch.randn(12 if self.direct_12_leads else 8, leadfield_rank) * s0)
        self.B = nn.Parameter(torch.randn(leadfield_rank, N_NODES) * s0)
        self.pspace = ParamSpace(graph_delays=self.graph_delays)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Move the ParamSpace center/radius to the device the model lives on
        self.pspace.to(self.A.device)
        return self

    @property
    def H_ind(self) -> torch.Tensor:
        return self.A @ self.B  # [8, 8] ([12, 8] with direct_12_leads)

    def node_sources(self, theta: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute the node source terms s:[B,8,T] from theta."""
        p = ParamSpace.unpack(theta, self.graph_delays)
        t = self.t.view(1, 1, -1)                 # [1,1,T]
        delta = p["delta"].unsqueeze(-1)          # [B,8,1]
        apd = p["APD"].unsqueeze(-1)
        tdep = p["tau_dep"].unsqueeze(-1)
        trep = p["tau_rep"].unsqueeze(-1)
        q = p["q"].unsqueeze(-1)
        aST = p["alpha_ST_full"].unsqueeze(-1)

        sd = torch.sigmoid((t - delta) / tdep)
        sr = torch.sigmoid((t - delta - apd) / trep)
        # Analytic dU/dt: QRS from the rising edge, T wave from the falling edge
        dU = sd * (1 - sd) / tdep - sr * (1 - sr) / trep
        # Plateau gate (ST segment)
        P = torch.sigmoid((t - delta - 0.04) / 0.02) * torch.sigmoid((delta + apd - t) / 0.04)
        s = q * dU + aST * P                       # [B,8,T]
        return s, p

    def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """theta:[B,PARAM_DIM] -> (y12:[B,12,T], aux dict)."""
        s, p = self.node_sources(theta)
        y_ind = torch.einsum("ln,bnt->blt", self.H_ind, s)   # [B,8,T] ordered I,II,V1..V6
        y_ind = y_ind * p["global_gain"].view(-1, 1, 1)
        if self.direct_12_leads:
            # ablation: the twelve displayed leads are projected directly ([B,12,T]), and the
            # limb-lead relations are left to the data
            return y_ind, {"s": s, "y_ind": y_ind, **p}
        y12 = derive_12leads_from_independent(y_ind, scaler=self.scaler)   # [B,12,T]
        return y12, {"s": s, "y_ind": y_ind, **p}

    def forward_from_z(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """z (unbounded) -> theta -> y12."""
        theta = self.pspace.theta(z)
        y12, aux = self.forward(theta)
        aux["theta"] = theta
        return y12, aux
