"""Small CPU tests of the resampling unit and native pooled metric."""

import numpy as np
import pytest

from tools.evolution_fifty.bootstrap import count_array, paired_bootstrap, pooled


def test_pooled_is_not_mean_of_episode_mious():
    episodes = np.array([[[1, 0], [1, 0]], [[0, 0], [9, 0]]], dtype=np.int64)
    assert pooled(episodes.sum(axis=0)) == pytest.approx(10.)
    assert np.mean(pooled(episodes)) == pytest.approx(50.)


def test_pairing_and_chunking_share_exact_draws():
    a = np.array([[[1, 2], [4, 5]], [[2, 1], [3, 4]], [[0, 4], [3, 5]]], dtype=np.int64)
    b = a.copy()
    b[:, 0, :] //= 2
    assert np.array_equal(paired_bootstrap(a, a, replicates=111, seed=7), np.zeros(111))
    first = paired_bootstrap(a, b, replicates=111, seed=7, chunk=1)
    assert np.array_equal(first, paired_bootstrap(a, b, replicates=111, seed=7, chunk=32))
    assert np.array_equal(-first, paired_bootstrap(b, a, replicates=111, seed=7))


def test_whole_episode_manual_reference():
    a = np.array([[[1, 0], [5, 0]], [[4, 3], [5, 4]]], dtype=np.int64)
    b = a.copy()
    b[:, 0] = 0
    rng = np.random.Generator(np.random.PCG64(9))
    values = []
    for _ in range(20):
        ids = rng.integers(0, 2, size=2)
        values.append(pooled(a[ids].sum(axis=0)) - pooled(b[ids].sum(axis=0)))
    assert np.array_equal(paired_bootstrap(a, b, replicates=20, seed=9), values)


@pytest.mark.parametrize("bad", [
    dict(intersection=[True], union=[1]), dict(intersection=[2], union=[1]),
    dict(intersection=[-1], union=[1]), dict(intersection=[.5], union=[1]),
    dict(intersection=[0], union=[2**63]), dict(intersection=[], union=[]),
])
def test_counts_reject_invalid(bad):
    with pytest.raises(ValueError):
        count_array(bad)


def test_absent_classes_and_empty_replica():
    assert pooled(np.array([[1, 0], [2, 0]])) == 50.
    with pytest.raises(ValueError, match="No union"):
        pooled(np.zeros((2, 2)))


def test_overflow_and_wrong_pairing_rejected():
    a = np.array([[[2**62], [2**62]], [[0], [1]]], dtype=np.int64)
    with pytest.raises(ValueError, match="overflow"):
        paired_bootstrap(a, a, replicates=20, seed=0)
    with pytest.raises(ValueError, match="Paired"):
        paired_bootstrap(a, a[:1], replicates=20, seed=0)
