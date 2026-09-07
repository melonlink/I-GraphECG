"""Round 3 Task F Level 2: monodomain Aliev-Panfilov ODE skeleton on the 8-node graph,
plus finite-time monodromy.

State x=[u_1..u_8, v_1..v_8] (16-dimensional). Fixed-step RK4; internal step of 2–5 ms advised.
Note: this computes the finite-time recovery multiplier of the paced dynamics (monodromy along
a single trajectory segment), **not** a strict autonomous limit-cycle Floquet multiplier
(pacing is not repeated to a stable periodic orbit).
"""
from __future__ import annotations

import numpy as np
import torch

# 8-node graph (TASK_SPEC §4): conduction connections, made undirected
_EDGES = [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (2, 6), (4, 7), (5, 7), (6, 7)]


def build_adjacency() -> torch.Tensor:
    A = torch.zeros(8, 8)
    for i, j in _EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


class APODE:
    """Parameters: g_global (conduction), a, k, eps, b (per node or scalar). Fixed-step RK4."""

    def __init__(self, g_global=1.5, a=0.1, k=8.0, eps=None, b=0.1, dt=0.003, device="cpu"):
        self.dev = device
        self.adj = build_adjacency().to(device)
        self.g = g_global
        self.a = torch.full((8,), float(a), device=device)
        self.k = torch.full((8,), float(k), device=device)
        self.b = torch.full((8,), float(b), device=device)
        self.eps = torch.full((8,), 0.02, device=device) if eps is None else eps.to(device)
        self.dt = dt

    def pacing(self, t: float) -> torch.Tensor:
        I = torch.zeros(8, device=self.dev)
        if 0.0 <= t < 0.006:        # short pulse at the SA node
            I[0] = 1.2
        return I

    def rhs(self, x: torch.Tensor, t: float) -> torch.Tensor:
        u, v = x[:8], x[8:]
        G = self.g * self.adj
        diffusion = G @ u - u * G.sum(1)
        ionic = -self.k * u * (u - self.a) * (u - 1.0) - u * v
        du = diffusion + ionic + self.pacing(t)
        dv = self.eps * (-v - self.k * u * (u - self.b - 1.0))
        return torch.cat([du, dv])

    def rk4_step(self, x, t):
        dt = self.dt
        k1 = self.rhs(x, t)
        k2 = self.rhs(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.rhs(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.rhs(x + dt * k3, t + dt)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(self, T: float = 0.6):
        x = torch.zeros(16, device=self.dev)
        n = int(T / self.dt)
        traj = [x]
        for s in range(n):
            x = self.rk4_step(x, s * self.dt)
            if not torch.isfinite(x).all() or x.abs().max() > 1e3:
                return None  # numerical blow-up
            traj.append(x)
        return torch.stack(traj)

    def finite_time_monodromy(self, T: float = 0.6) -> float | None:
        """Integrate dΦ=A(t)Φ along the trajectory; return rho=max|eig(M)|, or None on failure."""
        from torch.func import jacrev
        x = torch.zeros(16, device=self.dev)
        Phi = torch.eye(16, device=self.dev)
        n = int(T / self.dt)
        for s in range(n):
            t = s * self.dt
            A = jacrev(lambda z: self.rhs(z, t))(x)        # [16,16]
            # Φ updated with an Euler step at the same dt (first order suffices for the magnitude)
            Phi = Phi + self.dt * (A @ Phi)
            x = self.rk4_step(x, t)
            if not torch.isfinite(x).all() or x.abs().max() > 1e3 or not torch.isfinite(Phi).all():
                return None
        eig = torch.linalg.eigvals(Phi)
        return float(eig.abs().max().item())


# ---------------------------------------------------------------------------
# Round 4 Task B: strict paced periodic monodromy (numpy + analytic Jacobian, fast)
# ---------------------------------------------------------------------------
_ADJ_NP = build_adjacency().numpy()


def ode_params_from_theta(theta_row: np.ndarray) -> dict:
    """Surrogate θ -> AP-ODE parameters φ (rule-based map)."""
    vent = [2, 3, 4, 5, 6, 7]
    delta = theta_row[0:8][vent]; taur = theta_row[24:32]
    delta_spread = float(delta.max() - delta.min())
    g = float(np.clip(1.0 + 1.5 * (0.05 - delta_spread) / 0.05, 0.3, 3.0))
    eps = np.clip(0.06 - 0.4 * (taur - 0.03), 0.005, 0.08).astype(np.float64)
    return {"G": g * _ADJ_NP, "a": np.full(8, 0.1), "k": np.full(8, 8.0),
            "b": np.full(8, 0.1), "eps": eps}


def _rhs_np(x, p, I):
    u, v = x[:8], x[8:]
    G = p["G"]
    diffusion = G @ u - u * G.sum(1)
    du = diffusion - p["k"] * u * (u - p["a"]) * (u - 1.0) - u * v + I
    dv = p["eps"] * (-v - p["k"] * u * (u - p["b"] - 1.0))
    return np.concatenate([du, dv])


def _jac_np(x, p):
    """Analytic 16x16 Jacobian ∂f/∂x."""
    u, v = x[:8], x[8:]
    G = p["G"]; k = p["k"]; a = p["a"]; b = p["b"]; eps = p["eps"]
    A = np.zeros((16, 16))
    # ∂du/∂u
    Juu = G.copy()
    diag = -G.sum(1) - k * (3 * u ** 2 - 2 * (a + 1) * u + a) - v
    np.fill_diagonal(Juu, diag)
    A[:8, :8] = Juu
    A[:8, 8:] = np.diag(-u)                               # ∂du/∂v
    A[8:, :8] = np.diag(eps * (-k * (2 * u - (b + 1))))   # ∂dv/∂u
    A[8:, 8:] = np.diag(-eps)                             # ∂dv/∂v
    return A


def _pacing_np(t, T_cycle, stim_width):
    I = np.zeros(8)
    if (t % T_cycle) < stim_width:
        I[0] = 1.2
    return I


def _rk4_np(x, t, p, dt, T_cycle, stim_width):
    f = lambda xx, tt: _rhs_np(xx, p, _pacing_np(tt, T_cycle, stim_width))
    k1 = f(x, t); k2 = f(x + 0.5*dt*k1, t+0.5*dt); k3 = f(x + 0.5*dt*k2, t+0.5*dt); k4 = f(x+dt*k3, t+dt)
    return x + dt/6*(k1+2*k2+2*k3+k4)


def paced_monodromy_np(p: dict, T_cycle=1.0, n_cycles=25, dt=0.003,
                       stim_width=0.01, conv_tol=1e-3) -> dict:
    """Pace until convergence, then take the last-cycle monodromy; returns rho/converged dict."""
    n_per = int(round(T_cycle / dt))
    x = np.zeros(16); x_prev_end = None; conv_cycle = -1; last_err = np.nan
    for c in range(n_cycles):
        for s in range(n_per):
            x = _rk4_np(x, c * T_cycle + s * dt, p, dt, T_cycle, stim_width)
            if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e3:
                return {"rho": np.nan, "converged": False, "conv_cycle": -1,
                        "final_rel_err": np.nan, "fail": "explode"}
        if x_prev_end is not None:
            last_err = np.linalg.norm(x - x_prev_end) / (np.linalg.norm(x_prev_end) + 1e-9)
            if last_err < conv_tol and conv_cycle < 0:
                conv_cycle = c
        x_prev_end = x.copy()
    converged = conv_cycle >= 0
    # Last-cycle monodromy (variational equation integrated with Euler)
    Phi = np.eye(16)
    for s in range(n_per):
        t = n_cycles * T_cycle + s * dt
        A = _jac_np(x, p)
        Phi = Phi + dt * (A @ Phi)
        x = _rk4_np(x, t, p, dt, T_cycle, stim_width)
        if not np.all(np.isfinite(Phi)) or not np.all(np.isfinite(x)):
            return {"rho": np.nan, "converged": converged, "conv_cycle": conv_cycle,
                    "final_rel_err": float(last_err), "fail": "monodromy_nan"}
    rho = float(np.max(np.abs(np.linalg.eigvals(Phi))))
    return {"rho": rho, "converged": converged, "conv_cycle": conv_cycle,
            "final_rel_err": float(last_err), "fail": ""}


def rule_adapter(theta_row: np.ndarray, device="cpu") -> APODE:
    """Surrogate theta -> AP-ODE parameters (rule-based map, §9.4).

    delta(0:8), APD(8:16), tau_rep(24:32). Ventricular nodes [2..7].
    """
    vent = [2, 3, 4, 5, 6, 7]
    delta = theta_row[0:8][vent]
    taur = theta_row[24:32]
    delta_spread = float(delta.max() - delta.min())
    g_global = float(np.clip(1.0 + 1.5 * (0.05 - delta_spread) / 0.05, 0.3, 3.0))
    # eps modulated by tau_rep: larger tau_rep means slower recovery -> smaller eps
    eps = np.clip(0.06 - 0.4 * (taur - 0.03), 0.005, 0.08).astype(np.float32)
    return APODE(g_global=g_global, eps=torch.tensor(eps, device=device), device=device)
