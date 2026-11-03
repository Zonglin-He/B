import numpy as np
import pytest

from tools.evolution_fifty.pixel_transitions import transitions, rates, confusion


def test_exact_pixel_events_ignore255_and_metric_denominators():
    gt = np.array([[0, 0, 1], [1, 255, 0]])
    before = np.array([[0, 1, 1], [0, 1, 0]])
    after = np.array([[1, 0, 1], [1, 0, 0]])
    value = transitions(before, after, gt, 2)
    np.testing.assert_array_equal(value, [5, 3, 2, 1, 2])
    result = rates(value)
    assert float(result["correct_retention_pct"]) == pytest.approx(200/3)
    assert float(result["wrong_correction_pct"]) == 100
    assert float(result["net_correct_change_pp"]) == 20
    assert int(confusion(after, gt, 2)[0].sum() - confusion(before, gt, 2)[0].sum()) == 1


def test_identity_has_zero_damage_and_net_and_explicit_undefined_rate():
    gt = np.ones((2, 2), dtype=np.int64)
    value = transitions(gt, gt, gt, 2)
    result = rates(value)
    assert float(result["correct_retention_pct"]) == 100
    assert float(result["net_correct_change_pp"]) == 0
    assert np.isnan(result["wrong_correction_pct"])


def test_batched_pooled_rates_are_not_averages_of_source_percentages():
    values = np.array([[100, 90, 10, 9, 1], [10, 1, 9, 1, 3]], dtype=np.int64)
    assert rates(values).get("net_correct_change_pp").shape == (2,)
    assert float(rates(values.sum(0))["net_correct_change_pp"]) == pytest.approx(-600/110)


@pytest.mark.parametrize("bad", [np.array([5, 4, 2, 0, 0]), np.array([5, 3, 2, 4, 0]),
                                 np.array([5, 3, 2, 0, 3]), np.array([5., 3., 2., 1., 0.])])
def test_impossible_counts_rejected(bad):
    with pytest.raises(ValueError):
        rates(bad)


def test_invalid_labels_or_geometry_rejected():
    a = np.zeros((2, 2), dtype=np.int64)
    with pytest.raises(ValueError):
        transitions(a, a[:1], a, 2)
    with pytest.raises(ValueError):
        transitions(a, a+2, a, 2)
    with pytest.raises(ValueError):
        transitions(a, a, a-1, 2)
