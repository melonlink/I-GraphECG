"""Check that small-sample results are identical once the seed is fixed."""
import numpy as np

from igraphecg.utils.seed import set_seed


def test_numpy_repro():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_feature_extraction_deterministic():
    from igraphecg.features.ecg_features import extract_features
    rng = np.random.default_rng(0)
    beat = rng.standard_normal((12, 100)).astype(np.float32)
    f1 = extract_features(beat, fs=100)
    f2 = extract_features(beat, fs=100)
    assert np.array_equal(f1, f2)
    assert np.isfinite(f1).all()


def test_classifier_repro():
    """HistGB with the same seed gives identical results on the same data."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(1)
    X = rng.standard_normal((120, 8)); y = (X[:, 0] + X[:, 1] > 0).astype(int)
    p1 = HistGradientBoostingClassifier(max_iter=30, random_state=7).fit(X, y).predict_proba(X)
    p2 = HistGradientBoostingClassifier(max_iter=30, random_state=7).fit(X, y).predict_proba(X)
    assert np.allclose(p1, p2)
