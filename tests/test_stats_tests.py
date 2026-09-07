import numpy as np

from igraphecg.evaluation.stats_tests import cliffs_delta


def test_cliffs_delta_uses_average_ranks_for_ties():
    # Four pairwise comparisons: two ties and two cases with a < b.
    assert np.isclose(cliffs_delta(np.array([1.0, 1.0]), np.array([1.0, 2.0])), -0.5)


def test_cliffs_delta_is_antisymmetric():
    a = np.array([0.0, 1.0, 1.0, 4.0])
    b = np.array([1.0, 2.0, 3.0])
    assert np.isclose(cliffs_delta(a, b), -cliffs_delta(b, a))
