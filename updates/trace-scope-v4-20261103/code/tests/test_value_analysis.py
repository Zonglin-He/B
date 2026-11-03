import numpy as np
import pytest
import math

from tools.evolution_fifty import value_analysis as analysis


def times(dataset="pc60", values=(4.,10.,6.,10.5,30.)):
    return {p["id"]:v for p,v in zip(analysis.points(dataset),values,strict=True)}


def test_nearest_is_time_only_and_near_time_is_not_fabricated():
    result = analysis.time_match("pc60",times())
    assert result["nearest_Confidence_point"].endswith("steps14") and result["near_time"]
    result = analysis.time_match("pc60",times(values=(1.,2.,5.,6.,7.)))
    assert result["nearest_Confidence_point"].endswith("steps7") and not result["near_time"]


def test_exact_tie_uses_fewer_updates():
    result = analysis.time_match("pc60",times(values=(1.,1.,1.,1.,1.)))
    assert result["nearest_Confidence_point"].endswith("steps7")


@pytest.mark.parametrize("fault",["missing","zero","infinite","boolean"])
def test_no_time_selection_from_missing_or_invalid_points(fault):
    values = times()
    name = next(iter(values))
    if fault == "missing": values.pop(name)
    else: values[name] = dict(zero=0.,infinite=np.inf,boolean=True)[fault]
    with pytest.raises(ValueError): analysis.time_match("pc60",values)


def test_shared_episode_bootstrap_reproducible_and_undefined_not_dropped():
    counts = np.ones((3,5,2,2),dtype=np.int64)
    result = analysis.summarize("pc60",counts,times(),replicates=20)
    assert result == analysis.summarize("pc60",counts,times(),replicates=20,chunk=7)
    assert result["native_C_minus_nearest_Confidence"]["CI95_pp"] == [0.,0.]
    assert result["bootstrap"]["shared_point_draws"]
    counts[0] = 0
    result = analysis.summarize("pc60",counts,times(),replicates=1000)
    assert result["native_C_minus_nearest_Confidence"]["undefined_draws"] > 0
    assert result["native_C_minus_nearest_Confidence"]["CI95_pp"] is None


def test_unmatched_curve_retains_five_real_points_without_interpolation():
    counts = np.ones((2,5,2,2),dtype=np.int64)
    result = analysis.summarize("coco_object",counts,times("coco_object",(1.,2.,5.,6.,7.)),replicates=10)
    assert len(result["points"]) == 5 and result["near_time_accuracy_comparison"] is None


def test_pooled_counts_not_mean_image_metric_and_counts_validation():
    counts = np.zeros((2,5,2,1),dtype=np.int64)
    counts[0,:,0,:] = 1
    counts[0,:,1,:] = 1
    counts[1,:,1,:] = 9
    result = analysis.summarize("pc60",counts,times(),replicates=10)
    assert all(p["mIoU"] == 10. for p in result["points"].values())
    with pytest.raises(ValueError): analysis.summarize("pc60",counts.astype(float),times(),replicates=10)


@pytest.mark.parametrize("dataset",["coco_object","pc60"])
def test_exact_reciprocal_time_tie_selects_fewer_steps_not_rounded_log_distance(dataset):
    result = analysis.time_match(dataset,times(dataset,(1.,110.,121.,100.,1000.)))
    expected = next(p for p in analysis.points(dataset) if p["arm"] == "image_confidence")
    assert result["nearest_Confidence_point"] == expected["id"]
    counts = np.zeros((2,5,2,1),dtype=np.int64)
    counts[:,:,1] = 100
    counts[:,:,0,0] = [0,50,10,90,0]
    summary = analysis.summarize(dataset,counts,times(dataset,(1.,110.,121.,100.,1000.)),replicates=10)
    assert summary["native_C_minus_nearest_Confidence"]["difference_pp"] == 40.
    # A genuinely greater multiplicative distance is not widened into a tie.
    result = analysis.time_match(dataset,times(dataset,(1.,110.,math.nextafter(121.,math.inf),100.,1000.)))
    assert result["nearest_Confidence_point"] != expected["id"]


@pytest.mark.parametrize("dataset",["coco_object","pc60"])
def test_finite_positive_extreme_times_do_not_underflow_before_selection(dataset):
    result = analysis.time_match(dataset,times(dataset,(1.,1e300,1e-300,1e299,1e300)))
    assert result["nearest_Confidence_point"] == analysis.points(dataset)[-1]["id"]
    assert result["Confidence_to_C_time_ratio"] == 1. and result["near_time"]
    result = analysis.time_match(dataset,times(dataset,(1.,1e-300,1e300,1e301,1e302)))
    assert result["Confidence_to_C_time_ratio"] is None and not result["near_time"]
    assert result["unrepresentable_time_ratio_exact_fraction"]


@pytest.mark.parametrize("conf,near",[(9.,True),(11.,True),(math.nextafter(9.,0.),False),(math.nextafter(11.,math.inf),False)])
def test_exact_ten_percent_threshold_not_widened(conf,near):
    result = analysis.time_match("pc60",times("pc60",(1.,10.,conf,100.,1000.)))
    assert result["near_time"] is near
