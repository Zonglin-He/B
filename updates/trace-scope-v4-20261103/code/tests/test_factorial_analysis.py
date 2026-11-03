import numpy as np
import pytest

from tools.evolution_fifty import factorial_analysis as analysis


def fixture():
    values = np.zeros((3,2,4,2,2),dtype=np.int64)
    for e in range(3):
        for s in range(2):
            for arm in range(4):
                values[e,s,arm,0] = [3+e+s+arm,2+e]
                values[e,s,arm,1] = [12+e,10+e]
    return values


def test_contrasts_use_declared_matrix_and_do_not_contaminate_unused_arms():
    values = np.array([10.,17.,13.,12.])
    expected = [7.,4.,2.,5.,3.,2.]
    assert [analysis.contrast(values,w) for w in analysis.CONTRASTS.values()] == expected
    values[-1] = np.nan
    assert analysis.contrast(values,analysis.CONTRASTS["native_C_minus_Confidence"]) == 7
    assert np.isnan(analysis.contrast(values,analysis.CONTRASTS["distribution_order_interaction"]))


def test_manual_original_episode_draws_match_shared_study_arm_bootstrap_exactly():
    values = fixture()
    result,samples = analysis.paired_summary(values,"pc60",replicates=37,chunk=4)
    rng = np.random.Generator(np.random.PCG64(result["bootstrap"]["seed_uint128"]))
    draws = rng.integers(0,len(values),size=(37,len(values)))
    expected = analysis.miou(values[draws].sum(1,dtype=np.int64))
    np.testing.assert_array_equal(samples,expected)
    assert set(result["studies"]) == set(analysis.STUDIES)
    assert set(result["studies"]["natural"]["contrasts"]) == set(analysis.CONTRASTS)
    other,other_samples = analysis.paired_summary(values,"pc60",replicates=37,chunk=11)
    assert result == other
    np.testing.assert_array_equal(samples,other_samples)


def test_sparse_union_draws_are_retained_and_interval_is_null_without_dropping():
    values = fixture()
    values[0] = 0
    result,samples = analysis.paired_summary(values,"coco_object",replicates=100,chunk=9)
    arm = result["studies"]["natural"]["arms"]["Dk_Ok"]
    assert len(samples) == 100 and arm["undefined_draws"] > 0 and arm["CI95_pp"] is None
    assert arm["total_draws"] == 100


@pytest.mark.parametrize("fault",["float","negative","intersection","shape","overflow"])
def test_invalid_counts_fail_without_metric_or_resampling(fault):
    values = fixture()
    if fault == "float":
        values = values.astype(float)
    elif fault == "negative":
        values[0,0,0,0,0] = -1
    elif fault == "intersection":
        values[0,0,0,0,0] = 1000
    elif fault == "shape":
        values = values[:,:1]
    else:
        values.fill(2**52)
    with pytest.raises(ValueError):
        analysis.paired_summary(values,"pc60",replicates=2)
