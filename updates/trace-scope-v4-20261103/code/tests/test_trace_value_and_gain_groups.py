import copy

import numpy as np
import pytest

from tools.evolution_fifty import gain_groups as group
from tools.evolution_fifty.value_protocol import budget_steps, protocol, score_variants


def candidates(confidence, C, areas=None):
    areas = areas or [10]*len(C)
    return [dict(confidence=a, reliability=b, area=size,
                 region=dict(shape=[10, 10], true_pixels=size))
            for a,b,size in zip(confidence,C,areas,strict=True)]


def test_budget_grid_is_prospective_does_not_select_miou_and_separates_time():
    assert budget_steps(30) == dict(image_C=[15,30], image_confidence=[30,60,120])
    assert budget_steps(7) == dict(image_C=[4,7], image_confidence=[7,14,28])
    p = protocol()
    assert p["G04"]["dataset"] == "voc20" and p["G04"]["new_cells"] == 3
    assert p["V01"]["frozen_grid_conditions"] == 10
    assert not p["executable_queue"] and not p["scientific_completion"]
    assert "not isolated candidate causality" in p["A01"]["claim"]
    for invalid in (0,-1,True,1.5):
        with pytest.raises(ValueError):
            budget_steps(invalid)


def test_absolute_relative_product_controls_are_detached_numeric_variants_not_new_default():
    confidence, C = np.asarray([.2,.8,0.]), np.asarray([.1,.4,0.])
    before = confidence.copy(), C.copy()
    scores = score_variants(confidence,C)
    np.testing.assert_allclose(scores["relative"], C/(confidence+1e-8))
    np.testing.assert_allclose(scores["product"], confidence*C)
    np.testing.assert_array_equal(confidence,before[0])
    np.testing.assert_array_equal(C,before[1])
    with pytest.raises(ValueError):
        score_variants([np.nan],[.5])
    with pytest.raises(ValueError):
        score_variants([.5],[-.1])


def test_source_ranking_separates_same_order_from_equal_normalized_weight():
    rows = candidates([.2,.8],[.1,.9])
    old = copy.deepcopy(rows)
    result = group.source_features(rows)
    assert result["ordering_group"] == "same_order" and result["discordant_pairs"] == 0
    assert result["weight_L1"] == pytest.approx(.2)
    assert result["comparable_pairs"] == 1
    assert rows == old
    swapped = group.source_features(candidates([.2,.8],[.9,.1]))
    assert swapped["ordering_group"] == "reordered" and swapped["discordant_pairs"] == 1


def test_no_comparable_ties_and_EPS_support_are_reported_not_dropped():
    result = group.source_features(candidates([.2,.8],[0.,0.]))
    assert result["ordering_group"] == "no_comparable"
    assert result["active_C"] == 0 and result["active_confidence"] == 2
    assert result["C_tied_pairs"] == 1 and result["weight_L1"] == pytest.approx(1)
    assert group.source_features([])["count_group"] == "none"
    assert group.source_features(candidates([.5],[.6]))["ordering_group"] == "single"


def test_confidence_and_area_groups_have_fixed_boundaries_and_explicit_source_semantics():
    result = group.source_features(candidates([.25,.25],[.1,.2],areas=[1,1]))
    labels = group.memberships(result)
    assert "source_mean_confidence:0.25to0.5" in labels
    assert "source_median_candidate_area:le0.01" in labels
    assert set(labels) <= set(group.all_groups())
    assert len(group.all_groups()) == len(set(group.all_groups()))


def test_sparse_bootstrap_retains_undefined_replicates_and_marks_interval_null(monkeypatch):
    monkeypatch.setattr(group,"REPLICATES",100)
    episodes = np.zeros((2,3,5),np.int64)
    episodes[0] = [10,10,0,1,0]
    result = group.grouped_intervals(episodes,"fixture")
    assert result["wrong_correction_pct"]["undefined_replicates"] == 100
    assert result["wrong_correction_pct"]["lower95_pp"] is None
    assert result["net_correct_change_pp"]["undefined_replicates"] > 0
    assert result["net_correct_change_pp"]["lower95_pp"] is None
    assert all(r["no_replicate_dropped"] for r in result.values())
