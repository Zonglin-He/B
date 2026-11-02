import math
import random

import pytest

from tools.analyze_ablation_mechanisms import dominance_allocation


def test_common_attenuation_cancels():
    result = dominance_allocation([.2, .4], [.1, .3], [.4, .6])
    assert result['raw_attenuation_fraction'] == pytest.approx(.2)
    assert result['normalized_tv_after_eps'] == pytest.approx(0)


def test_singleton_cancels():
    result = dominance_allocation([.2], [.1], [.8])
    assert result['raw_attenuation_fraction'] == pytest.approx(.6)
    assert result['normalized_tv_after_eps'] == pytest.approx(0)


def test_heterogeneous_penalties_change_relative_weights():
    result = dominance_allocation([.2, .8], [.2, .8], [.8, .8])
    assert result['raw_attenuation_fraction'] == pytest.approx(.12)
    assert result['normalized_tv_before_eps'] == pytest.approx(.1090909090909)
    assert result['normalized_tv_after_eps'] == pytest.approx(.1090909090909)


def test_eps_pruning_is_separate_from_ideal_identity():
    result = dominance_allocation([1.5e-8], [1.5e-8], [.5])
    assert result['normalized_tv_before_eps'] == pytest.approx(0)
    assert result['normalized_tv_after_eps'] == pytest.approx(1)
    assert result['eps_activity_changes'] == 1
    assert result['active_to_empty'] == 1


def test_empty_objective():
    result = dominance_allocation([0], [0], [1])
    assert result['raw_attenuation_fraction'] is None
    assert result['normalized_tv_before_eps'] is None
    assert result['normalized_tv_after_eps'] == 0


def test_randomized_identity():
    rng = random.Random(20261025)
    for count in (1, 2, 7, 100):
        for _ in range(40):
            result = dominance_allocation(*[[rng.random() for _ in range(count)]
                                            for _ in range(3)])
            assert math.isfinite(result['normalized_tv_before_eps'])
            assert result['normalized_tv_before_eps'] == pytest.approx(
                result['normalized_tv_after_eps'], abs=1e-12)


@pytest.mark.parametrize('args', [([.2], [], [.3]), ([float('nan')], [.3], [.4]),
                                  ([.2], [1.2], [.3])])
def test_invalid_supports(args):
    with pytest.raises(ValueError):
        dominance_allocation(*args)
