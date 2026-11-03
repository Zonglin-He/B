import numpy as np
import pytest

from tools.evolution_fifty.risk_outcomes import measure, matching_stratum, matched_C_groups


def test_inside_outside_continuous_support_and_pixel_changes_kept_separate():
    gt = np.array([[0, 1], [1, 255]])
    before = np.array([[0, 0], [1, 0]])
    after = np.array([[1, 1], [1, 0]])
    a = np.array([[.8, .2], [.7, np.nan]])
    b = np.array([[.5, .6], [.7, np.nan]])
    mask = np.array([[True, False], [False, True]])
    result = measure(before, after, gt, a, b, mask, 2)
    assert result["inside"]["net_correct_pixels"] == -1
    assert result["outside"]["net_correct_pixels"] == 1
    assert result["whole"]["net_correct_pixels"] == 0
    assert result["inside"]["mean_true_class_support_delta"] == pytest.approx(-.3)
    assert result["outside"]["mean_true_class_support_delta"] == pytest.approx(.2)
    assert result["inside"]["primary_support_harm"] is True


def test_empty_valid_subregion_is_explicitly_undefined_not_zero():
    gt = np.array([[255, 0]])
    a = np.array([[0, 0]])
    support = np.array([[np.nan, .7]])
    result = measure(a, a, gt, support, support, np.array([[True, False]]), 2)
    assert result["inside"]["zero_valid_pixels"] is True
    assert result["inside"]["mean_true_class_support_delta"] is None
    assert result["inside"]["primary_support_harm"] is None
    assert result["inside"]["correct_retention_pct"] is None


def test_exact_median_ties_and_sparse_strata_not_silently_used_as_matched():
    def row(c, query=1):
        return dict(C=c, confidence=.91, benchmark_class=query, area=16, shape=[16, 16])
    result = matched_C_groups([row(.1), row(.5), row(.9), row(.5, 2)])
    assert result["matched_strata"] == 1 and result["matched_non_tied_candidates"] == 2
    assert result["exact_median_tied_candidates"] == [1, 3]
    assert result["unmatched_candidates"] == [3]


def test_confidence_one_and_full_area_have_finite_final_bins():
    assert matching_stratum(1., 0, 256, [16, 16]) == (0, 19, 4)
