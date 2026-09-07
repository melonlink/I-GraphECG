"""Output shape checks for the surrogate decoder."""
import pytest

torch = pytest.importorskip("torch")

from igraphecg.models.surrogate_decoder import PARAM_DIM, SurrogateDecoder


def test_forward_shapes():
    dec = SurrogateDecoder(n_t=100)
    z = torch.randn(5, PARAM_DIM)
    y12, aux = dec.forward_from_z(z)
    assert y12.shape == (5, 12, 100)
    assert aux["y_ind"].shape == (5, 8, 100)
    assert aux["s"].shape == (5, 8, 100)
    assert dec.H_ind.shape == (8, 8)
    assert aux["theta"].shape == (5, PARAM_DIM)


def test_param_dim():
    assert PARAM_DIM == 47  # 8*5 + 6 alpha_ST + 1 global_gain
