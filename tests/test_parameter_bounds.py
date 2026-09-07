"""Parameters must fall inside the physiological range after center+radius*tanh(z)."""
import pytest

torch = pytest.importorskip("torch")

from igraphecg.models.surrogate_decoder import PARAM_DIM, ParamSpace, _bounds


def test_theta_within_bounds():
    ps = ParamSpace()
    lo, hi = _bounds()
    lo_t, hi_t = torch.tensor(lo), torch.tensor(hi)
    z = torch.randn(2000, PARAM_DIM) * 10  # extreme z
    theta = ps.theta(z)
    assert torch.all(theta >= lo_t - 1e-4)
    assert torch.all(theta <= hi_t + 1e-4)


def test_boundary_rate_range():
    ps = ParamSpace()
    z0 = torch.zeros(100, PARAM_DIM)
    assert ps.boundary_rate(z0) == 0.0           # z=0 -> parameter center, no bound hits
    zbig = torch.full((100, PARAM_DIM), 5.0)
    assert ps.boundary_rate(zbig) > 0.99         # large z -> all at the bounds


def test_unpack_alpha_st_atria_zero():
    ps = ParamSpace()
    theta = ps.theta(torch.randn(4, PARAM_DIM))
    p = ParamSpace.unpack(theta)
    assert torch.allclose(p["alpha_ST_full"][:, :2], torch.zeros(4, 2))  # atria/AV node: no ST
