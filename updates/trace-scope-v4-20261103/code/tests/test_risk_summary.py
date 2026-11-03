import numpy as np
import pytest

from tools.evolution_fifty.risk_outcomes import measure
from tools.evolution_fifty.risk_summary import contributions, score_groups, summary


def observation(delta, *, ignored=False):
    gt = np.full((2, 2), 255 if ignored else 0, np.uint8)
    labels = np.zeros((2, 2), np.uint16)
    before = np.full((2, 2), .5)
    after = before+delta
    return measure(labels, labels, gt, before, after, np.ones((2, 2), bool), 2)["whole"]


def test_quantiles_freeze_before_outcomes_and_ties_do_not_get_forced_equal_counts():
    rows = [dict(C=.5, confidence=i/10) for i in range(10)]
    plan = score_groups(rows)
    assert plan["groups"]["C_Q1"] == [] and plan["groups"]["C_Q5"] == list(range(10))
    assert plan["cuts"]["C"] == [.5]*4
    with pytest.raises(ValueError):
        score_groups([dict(C=float("nan"), confidence=.9)])


def test_episode_bootstrap_keeps_candidate_and_pixel_metrics_separate_and_pairs_contrast():
    rows = [dict(C=i/9, confidence=1-i/9, episode=i//2, source_id=f"source{i//2}") for i in range(10)]
    groups = score_groups(rows)["groups"]
    result = summary(rows, [observation(-.01 if i < 2 else .02) for i in range(10)], groups, "fixture", replicates=400)
    all_metrics = result["groups"]["all"]["metrics"]
    assert all_metrics["candidate_mean_support_change"]["value"] == pytest.approx(.014)
    assert all_metrics["support_harm_rate_threshold_0.001"]["value"] == 20
    assert result["groups"]["all"]["coverage"]["original_episodes"] == 5
    contrast = result["contrasts"]["C_Q5_minus_C_Q1"]["support_harm_rate_threshold_0.001"]
    assert contrast["value"] == -100 and contrast["undefined_replicates"] > 0
    assert contrast["lower95"] is None and contrast["no_replicate_dropped"]
    again = summary(rows, [observation(-.01 if i < 2 else .02) for i in range(10)], groups, "fixture", replicates=400)
    assert result == again


def test_empty_groups_and_regions_stay_undefined_with_explicit_coverage():
    rows = [dict(C=.5, confidence=.9, episode=i, source_id=f"s{i}") for i in range(3)]
    groups = score_groups(rows)["groups"]
    result = summary(rows, [observation(0, ignored=True)]*3, groups, "empty", replicates=50)
    assert result["groups"]["all"]["coverage"]["zero_valid_candidates"] == 3
    metric = result["groups"]["all"]["metrics"]["candidate_mean_support_change"]
    assert metric["value"] is None and metric["undefined_replicates"] == 50
    assert result["groups"]["C_Q1"]["coverage"]["candidates"] == 0


def test_inconsistent_support_sum_or_group_population_rejected():
    bad = observation(.01)
    bad["support_delta_sum"] = 100
    with pytest.raises(ValueError, match="Inconsistent"):
        contributions(bad)
    rows = [dict(episode=0, source_id="s0"), dict(episode=1, source_id="s1")]
    with pytest.raises(ValueError, match="membership"):
        summary(rows, [observation(0)]*2, {"all":[0, 0]}, "bad", replicates=5)
