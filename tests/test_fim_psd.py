import numpy as np
import pytest

torch = pytest.importorskip("torch")

from igraphecg.evaluation.identifiability import compute_fim, fim_metrics, identifiability_scores
from igraphecg.models.surrogate_decoder import PARAM_DIM, SurrogateDecoder


def test_fim_psd_and_metrics():
    dec = SurrogateDecoder(n_t=100)
    thetas = dec.pspace.theta(torch.randn(6, PARAM_DIM))
    F = compute_fim(dec, thetas, dec.pspace.radius, device="cpu")
    eig = np.linalg.eigvalsh(F)
    assert eig.min() > -1e-6                       # positive semi-definite (numerical tolerance)
    m = fim_metrics(F)
    assert 1.0 <= m["effective_rank"] <= PARAM_DIM
    score, corr = identifiability_scores(F, lam=1e-3)
    assert score.shape == (PARAM_DIM,) and np.isfinite(score).all()
    assert np.allclose(np.diag(corr), 1.0, atol=1e-5)
