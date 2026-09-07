import numpy as np

from igraphecg.evaluation.stability import surrogate_stability_proxy


def test_proxy_finite():
    rng = np.random.default_rng(0)
    N = 200
    feats = {k: rng.standard_normal(N) + 1.0 for k in
             ["D_act", "Tend_disp", "tau_rep_disp", "ST_projected_L2"]}
    train_mask = np.arange(N) < 150
    out = surrogate_stability_proxy(feats, train_mask)
    assert out["V_rec"].shape == (N,) and np.isfinite(out["V_rec"]).all()
    assert np.allclose(out["m_proxy"], -out["V_rec"])
