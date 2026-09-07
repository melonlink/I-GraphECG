import numpy as np

from igraphecg.evaluation.st_features import internal_st_features, projected_st_features


def test_internal_st():
    theta = np.random.default_rng(0).standard_normal((10, 47)).astype(np.float32)
    f = internal_st_features(theta)
    for k in ["ST_L2", "ST_max_abs", "ST_entropy", "ST_anterior_strength"]:
        assert f[k].shape == (10,) and np.isfinite(f[k]).all()


def test_projected_st():
    theta = np.random.default_rng(1).standard_normal((10, 47)).astype(np.float32)
    H = np.random.default_rng(2).standard_normal((8, 8)).astype(np.float32) * 0.1
    f = projected_st_features(theta, H)
    assert f["ST_projected_L2"].shape == (10,)
    assert 0.0 <= f["ST_projected_sign_discordance"].max() <= 1.0
