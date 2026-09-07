import numpy as np
import torch

from igraphecg.evaluation.repolarization_features import (repolarization_param_features, twave_features)
from igraphecg.models.surrogate_decoder import ParamSpace


def test_param_features():
    theta = np.random.default_rng(0).standard_normal((20, 47)).astype(np.float32) * 0.1 + 0.2
    f = repolarization_param_features(theta)
    for k in ["APD_disp", "tau_rep_disp", "Tend_disp", "Sync_rep", "tau_rep_cv"]:
        assert f[k].shape == (20,) and np.isfinite(f[k]).all()


def test_param_features_use_eikonal_nodal_activation_times():
    theta = np.zeros((1, 47), dtype=np.float32)
    theta[0, 0] = -0.2
    theta[0, 1:8] = np.array([0.10, 0.02, 0.01, 0.02, 0.03, 0.04, 0.05])
    theta[0, 8:16] = 0.3
    theta[0, 24:32] = 0.05

    p = ParamSpace.unpack(torch.from_numpy(theta))
    ventricular_delta = p["delta"][0, 2:8].numpy()
    expected_tend = ventricular_delta + 0.3 + 2.0 * 0.05
    feats = repolarization_param_features(theta)

    assert np.isclose(feats["Tend_mean"][0], expected_tend.mean())
    assert np.isclose(feats["Tend_disp"][0], np.ptp(expected_tend))
    assert not np.isclose(feats["Tend_mean"][0], theta[0, 2:8].mean() + 0.4)


def test_twave_features():
    sig = np.random.default_rng(1).standard_normal((15, 12, 100)).astype(np.float32)
    f = twave_features(sig, fs=100)
    assert f["T_sign_discordance_rate"].shape == (15,)
    assert all(np.isfinite(v).all() for v in f.values())
