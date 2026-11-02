import numpy as np

from tools.analyze_evolution import allocation_shares


def test_active_region_shares_match_loss_admission():
    shares = allocation_shares(np.array([[1e-9, .2, .8], [1e-8, 0, 1e-10]]))
    np.testing.assert_array_equal(shares[0], [0, .2, .8])
    np.testing.assert_array_equal(shares[1], [0, 0, 0])


def test_common_scale_preserves_positive_shares_away_from_threshold():
    weights = np.array([[.1, .3, .6]])
    np.testing.assert_allclose(allocation_shares(weights), allocation_shares(weights * .1))
