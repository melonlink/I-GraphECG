import numpy as np

from igraphecg.evaluation.lead_ablation import LEAD_SETS, mask_signal


def test_lead_sets_valid():
    assert LEAD_SETS["L12"] == list(range(12))
    assert len(LEAD_SETS["L3a_II_V1_V5"]) == 3
    assert LEAD_SETS["L1_II"] == [1]   # lead II is index 1 in the canonical order
    for v in LEAD_SETS.values():
        assert all(0 <= i < 12 for i in v)


def test_mask_zeros_unobserved():
    sig = np.random.default_rng(0).standard_normal((5, 12, 100)).astype(np.float32)
    m = mask_signal(sig, LEAD_SETS["L3a_II_V1_V5"])
    obs = LEAD_SETS["L3a_II_V1_V5"]
    assert np.allclose(m[:, obs, :], sig[:, obs, :])
    unobs = [i for i in range(12) if i not in obs]
    assert np.all(m[:, unobs, :] == 0)
