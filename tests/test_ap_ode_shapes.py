import numpy as np
import pytest

torch = pytest.importorskip("torch")

from igraphecg.models.ap_ode import APODE, build_adjacency, rule_adapter


def test_adjacency():
    A = build_adjacency()
    assert A.shape == (8, 8)
    assert torch.allclose(A, A.T)          # undirected


def test_simulate_and_adapter():
    ode = APODE(dt=0.005)
    traj = ode.simulate(T=0.3)
    assert traj is None or traj.shape[1] == 16
    th = np.random.default_rng(0).standard_normal(47).astype(np.float32) * 0.1 + 0.1
    ode2 = rule_adapter(th)
    assert isinstance(ode2, APODE)


def test_monodromy_returns_scalar_or_none():
    ode = APODE(dt=0.005)
    rho = ode.finite_time_monodromy(T=0.3)
    assert rho is None or (np.isfinite(rho) and rho >= 0)
