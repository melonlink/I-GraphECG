import numpy as np

from igraphecg.evaluation.metrics import bootstrap_ci, paired_bootstrap_difference


def _toy_predictions():
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    good = np.full((8, 4), 0.05)
    good[np.arange(8), y] = 0.85
    weak = np.full((8, 4), 0.20)
    weak[np.arange(8), y] = 0.40
    groups = np.array([10, 10, 11, 11, 12, 12, 13, 13])
    return y, good, weak, groups


def test_grouped_bootstrap_is_reproducible():
    y, good, _, groups = _toy_predictions()
    first = bootstrap_ci(y, good, 4, n_boot=50, seed=7, groups=groups)
    second = bootstrap_ci(y, good, 4, n_boot=50, seed=7, groups=groups)
    assert first == second


def test_paired_bootstrap_difference_uses_aligned_resamples():
    y, good, weak, groups = _toy_predictions()
    point, lo, hi = paired_bootstrap_difference(
        y, good, weak, 4, n_boot=50, seed=7, groups=groups
    )
    assert np.isfinite([point, lo, hi]).all()
    assert lo <= point <= hi
