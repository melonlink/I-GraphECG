import numpy as np
import pytest

torch = pytest.importorskip("torch")

from igraphecg.evaluation.identifiability import compute_fim, lead_rows
from igraphecg.models.surrogate_decoder import PARAM_DIM, SurrogateDecoder


def test_fim_shape():
    dec = SurrogateDecoder(n_t=100)
    thetas = dec.pspace.theta(torch.randn(4, PARAM_DIM))
    F = compute_fim(dec, thetas, dec.pspace.radius, device="cpu")
    assert F.shape == (PARAM_DIM, PARAM_DIM)
    assert np.isfinite(F).all()


def test_lead_rows():
    rws = lead_rows([1, 5], n_t=100)
    assert rws.shape == (200,)
    assert rws.min() == 100 and rws.max() == 599
