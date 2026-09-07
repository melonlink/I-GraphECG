"""Derived leads in the decoder output satisfy the Einthoven/Goldberger relations exactly."""
import pytest

torch = pytest.importorskip("torch")

from igraphecg.models.leadfield import OUTPUT_LEADS
from igraphecg.models.surrogate_decoder import PARAM_DIM, SurrogateDecoder


def test_decoder_lead_relations():
    dec = SurrogateDecoder(n_t=100)
    y12, _ = dec.forward_from_z(torch.randn(3, PARAM_DIM))
    g = lambda n: y12[:, OUTPUT_LEADS.index(n), :]
    I, II = g("I"), g("II")
    assert torch.allclose(g("III"), II - I, atol=1e-5)
    assert torch.allclose(g("aVR"), -(I + II) / 2, atol=1e-5)
    assert torch.allclose(g("aVL"), I - II / 2, atol=1e-5)
    assert torch.allclose(g("aVF"), II - I / 2, atol=1e-5)
