import hashlib
import json

import numpy as np
import pytest

from tools.evolution_fifty import legacy_analysis as analysis
from tools.evolution_fifty import legacy_runtime
from tools.evolution_fifty.generalization_score import interval,miou


def test_all_five_prospective_contrasts_and_stream_key():
    counts = np.ones((3,4,2,2),dtype=np.int64)*10
    counts[:,:,0,:] = np.array([8,6,5,4])[None,:,None]
    result = analysis.paired_summary(counts,"pc60",replicates=23,chunk=7)
    assert analysis.MODES == legacy_runtime.MODES
    assert result["primary_contrast"] == "single_C_vs_legacy_image"
    assert {name:r["delta_pp"] for name,r in result["contrasts"].items()} == dict(
        single_C_vs_legacy_image=30.,single_C_vs_legacy_episode=20.,image_vs_episode_for_C=20.,
        image_vs_episode_for_legacy=10.,diagnostic_objective_interaction=10.)
    key = json.dumps([20261102,"M03","pc60","diagnostic_objective"],separators=(",",":"),ensure_ascii=True)
    assert result["bootstrap"]["key"] == key
    assert result["bootstrap"]["seed_uint128"] == int.from_bytes(hashlib.sha256(key.encode()).digest()[:16],"little")
    assert result == analysis.paired_summary(counts,"pc60",replicates=23,chunk=3)


def test_full10000_episode_bootstrap_matches_independent_direct_pooling_oracle():
    counts = np.ones((3,4,2,2),dtype=np.int64)*10
    counts[:,:,0,:] = np.array([2,5,8])[:,None,None]+np.arange(4)[None,:,None]//2
    result = analysis.paired_summary(counts,"coco_object")
    rng = np.random.Generator(np.random.PCG64(result["bootstrap"]["seed_uint128"]))
    draws = rng.integers(0,3,size=(10000,3))
    samples = np.stack([miou(counts[indices].sum(0)) for indices in draws])
    point = miou(counts.sum(0))
    for name,weights in analysis.CONTRASTS.items():
        active = [i for i,w in enumerate(weights) if w]
        direct = sum(weights[i]*samples[:,i] for i in active)
        actual = result["contrasts"][name]
        assert actual["delta_pp"] == pytest.approx(sum(weights[i]*point[i] for i in active))
        assert {k:actual[k] for k in interval(direct)} == interval(direct)


def test_pooled_not_image_mean_and_unused_NaN_does_not_poison_primary():
    counts = np.zeros((2,4,2,1),dtype=np.int64)
    counts[:,[0,2],1,:] = np.array([1,9])[:,None,None]
    counts[0,[0,2],0,:] = 1
    result = analysis.paired_summary(counts,"pc60",replicates=20)
    assert result["modes"]["C.image"]["mIoU"] == 10.
    assert result["modes"]["C.episode"]["mIoU"] is None
    assert result["contrasts"]["single_C_vs_legacy_image"]["CI95_pp"] == [0.,0.]
    assert result["contrasts"]["single_C_vs_legacy_episode"]["CI95_pp"] is None


def test_all_sparse_draws_retained_and_null_not_conditionally_dropped():
    counts = np.zeros((2,4,2,1),dtype=np.int64)
    counts[0,:,1] = 1
    result = analysis.paired_summary(counts,"pc60",replicates=100)
    for row in result["contrasts"].values():
        assert 0 < row["undefined_draws"] < 100 and row["total_draws"] == 100 and row["CI95_pp"] is None


@pytest.mark.parametrize("fault",["dtype","negative","I_gt_U","shape","dataset","replicates","chunk","overflow"])
def test_resource_and_population_bounds(fault):
    values = np.ones((2,4,2,2),dtype=np.int64)
    options = {}
    dataset = "pc60"
    if fault == "dtype":values = values.astype(float)
    elif fault == "negative":values[0,0,0,0] = -1
    elif fault == "I_gt_U":values[0,0,0,0] = 2
    elif fault == "shape":values = values[:,:3]
    elif fault == "dataset":dataset = "other"
    elif fault == "replicates":options["replicates"] = 10001
    elif fault == "chunk":options["chunk"] = 65
    elif fault == "overflow":values[:] = 2**52
    with pytest.raises(ValueError):analysis.paired_summary(values,dataset,**options)
