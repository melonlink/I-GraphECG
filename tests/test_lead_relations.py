"""Einthoven-Goldberger relations for III / aVR / aVL / aVF."""
import numpy as np

from igraphecg.models.leadfield import (INDEPENDENT_LEADS, OUTPUT_LEADS,
                                   derive_12leads_from_independent)


def test_lead_relations_numpy():
    rng = np.random.default_rng(0)
    y_ind = rng.standard_normal((4, len(INDEPENDENT_LEADS), 100)).astype(np.float32)
    y12 = derive_12leads_from_independent(y_ind)
    assert y12.shape == (4, 12, 100)

    def g(name, arr):
        return arr[:, OUTPUT_LEADS.index(name), :]

    I = y_ind[:, INDEPENDENT_LEADS.index("I"), :]
    II = y_ind[:, INDEPENDENT_LEADS.index("II"), :]
    assert np.allclose(g("III", y12), II - I, atol=1e-5)
    assert np.allclose(g("aVR", y12), -(I + II) / 2, atol=1e-5)
    assert np.allclose(g("aVL", y12), I - II / 2, atol=1e-5)
    assert np.allclose(g("aVF", y12), II - I / 2, atol=1e-5)
    # independent leads pass through unchanged
    assert np.allclose(g("V3", y12), y_ind[:, INDEPENDENT_LEADS.index("V3"), :], atol=1e-6)


def test_lead_relations_torch():
    torch = __import__("pytest").importorskip("torch")
    y_ind = torch.randn(2, len(INDEPENDENT_LEADS), 50)
    y12 = derive_12leads_from_independent(y_ind)
    assert tuple(y12.shape) == (2, 12, 50)
    I = y_ind[:, INDEPENDENT_LEADS.index("I"), :]
    II = y_ind[:, INDEPENDENT_LEADS.index("II"), :]
    III = y12[:, OUTPUT_LEADS.index("III"), :]
    assert torch.allclose(III, II - I, atol=1e-5)
