"""Decoder and loss outputs are free of NaN/Inf on random and extreme inputs."""
import pytest

torch = pytest.importorskip("torch")

from igraphecg.models.surrogate_decoder import PARAM_DIM, SurrogateDecoder
from igraphecg.training.losses import build_masks, total_loss


def test_decoder_finite_extreme_z():
    dec = SurrogateDecoder(n_t=100)
    for scale in (0.0, 1.0, 20.0):
        z = torch.randn(8, PARAM_DIM) * scale
        y12, aux = dec.forward_from_z(z)
        assert torch.isfinite(y12).all()
        assert torch.isfinite(aux["s"]).all()


def test_loss_finite_and_backward():
    dec = SurrogateDecoder(n_t=100)
    masks = build_masks(dec.t)
    z = torch.randn(8, PARAM_DIM, requires_grad=True)
    y12, aux = dec.forward_from_z(z)
    y = torch.randn(8, 12, 100)
    w = {"rec": 1.0, "slope": 0.2, "qrs": 1.0, "st": 0.5, "t": 0.5, "phys": 0.01, "H": 0.001}
    loss, comps = total_loss(y, y12, z, aux["theta"], dec.H_ind, masks, w)
    assert torch.isfinite(loss)
    for k, v in comps.items():
        assert torch.isfinite(v), k
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
