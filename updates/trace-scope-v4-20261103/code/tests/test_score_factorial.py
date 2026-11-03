import copy

import pytest

from trace_ovss.method.score_factorial import ARMS, assign_score_factorial


def sample():
    return [dict(source_id="s", candidate_id=str(i), area=2+i, kappa=k, C=c)
            for i, (k, c) in enumerate(((.9, .2), (.3, .6), (.5, .4)))]


def test_four_arms_multisets_and_orders_without_mutation():
    candidates = sample()
    before = copy.deepcopy(candidates)
    result = assign_score_factorial(candidates)["arms"]
    assert tuple(result) == ARMS and candidates == before
    assert result["Dk_Ok"]["assigned_scores"] == [.9, .3, .5]
    assert result["Dc_Oc"]["assigned_scores"] == [.2, .6, .4]
    assert result["Dc_Ok"]["assigned_scores"] == [.6, .2, .4]
    assert result["Dk_Oc"]["assigned_scores"] == [.3, .9, .5]
    for arm in ARMS:
        assert sum(result[arm]["ideal_region_coefficients"]) == pytest.approx(1.)


def test_ties_by_stable_id_not_input_order_and_no_source_mixing():
    candidates = sample()
    candidates[0]["kappa"] = candidates[1]["kappa"] = .5
    candidates += [dict(source_id="other", candidate_id="0", area=1, kappa=.01, C=.99)]
    a = assign_score_factorial(candidates)["arms"]
    reverse = list(reversed(candidates))
    b = assign_score_factorial(reverse)["arms"]
    for arm in ARMS:
        assert a[arm]["assigned_scores"] == list(reversed(b[arm]["assigned_scores"]))
        assert a[arm]["ideal_region_coefficients"][-1] == .5


def test_natural_and_common_active_are_distinct_explicit_studies():
    candidates = sample()
    candidates[0]["C"] = 0.
    natural = assign_score_factorial(candidates)
    common = assign_score_factorial(candidates, common_active=True)
    assert len(natural["arms"]["Dk_Ok"]["active_candidate_ids"]) == 3
    assert len(natural["arms"]["Dc_Oc"]["active_candidate_ids"]) == 2
    assert common["retained_candidates"] == 2
    assert all(len(v["active_candidate_ids"]) == 2 for v in common["arms"].values())
    assert natural["scope"] != common["scope"]


@pytest.mark.parametrize("damage", ["nan", "bool", "gt", "duplicate", "empty_id", "area"])
def test_invalid_or_supervised_inputs_rejected(damage):
    candidates = sample()
    if damage == "nan":
        candidates[0]["C"] = float("nan")
    elif damage == "bool":
        candidates[0]["kappa"] = True
    elif damage == "gt":
        candidates[0]["correctness"] = .8
    elif damage == "duplicate":
        candidates[1]["candidate_id"] = candidates[0]["candidate_id"]
    elif damage == "empty_id":
        candidates[0]["source_id"] = ""
    else:
        candidates[0]["area"] = 0
    with pytest.raises(ValueError):
        assign_score_factorial(candidates)


def test_empty_population_has_no_budget_or_spurious_score():
    result = assign_score_factorial([])
    assert all(v["ideal_region_coefficients"] == [] and v["active_source_ids"] == []
               for v in result["arms"].values())
