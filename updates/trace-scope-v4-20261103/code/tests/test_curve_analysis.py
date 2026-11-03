import copy
import hashlib
import json

import numpy as np
import pytest

from tools.evolution_fifty import curve_analysis as analysis
from tools.evolution_fifty.generalization_score import miou


def fixture():
    values=np.zeros((3,5,3,3,2,20),dtype=np.int64)
    for e in range(3):
        for t in range(5):
            for s in range(3):
                for a in range(3):
                    values[e,t,s,a,0,:2]=[e+t+a+1,s+2]
                    values[e,t,s,a,1,:2]=[20+3*e+t,15+s+a]
    return values


def test_all_fifteen_points_three_arms_and_shared_10000_episode_oracle():
    values=fixture();before=values.copy()
    result,samples=analysis.paired_summary(values)
    key=json.dumps([20261102,"G03","voc20","severity_curves"],separators=(",",":"),ensure_ascii=True)
    seed=int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[:16],"little")
    draws=np.random.Generator(np.random.PCG64(seed)).integers(0,3,size=(10000,3))
    oracle=np.stack([miou(values[row].sum(0,dtype=np.int64)) for row in draws])
    np.testing.assert_array_equal(samples,oracle)
    np.testing.assert_array_equal(values,before)
    assert samples.shape == (10000,5,3,3) and result["bootstrap"]["key"] == key
    for t,kind in enumerate(analysis.CURVE_TYPES):
        for s,severity in enumerate(analysis.SEVERITIES):
            r=result["curves"][kind][str(severity)]
            assert r["C_minus_Confidence_pp"] == miou(values.sum(0))[t,s,2]-miou(values.sum(0))[t,s,1]
            np.testing.assert_array_equal(r["paired_difference"]["CI95_pp"],
                np.quantile(oracle[:,t,s,2]-oracle[:,t,s,1],[.025,.975],method="linear"))
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize("chunk",[1,7,64])
def test_chunk_does_not_change_any_draw_or_result(chunk):
    baseline,samples=analysis.paired_summary(fixture(),replicates=127)
    other,actual=analysis.paired_summary(fixture(),replicates=127,chunk=chunk)
    assert other == baseline
    np.testing.assert_array_equal(samples,actual)


def test_undefined_kept_null_and_no_global_pool_or_mean_image_IOU():
    values=fixture()
    values[:,0,0]=0
    result,samples=analysis.paired_summary(values,replicates=41)
    empty=result["curves"][analysis.CURVE_TYPES[0]]["1"]
    assert empty["C_minus_Confidence_pp"] is None
    assert empty["paired_difference"] == dict(CI95_pp=None,undefined_draws=41,total_draws=41)
    assert np.isnan(samples[:,0,0]).all()
    assert result["curves"][analysis.CURVE_TYPES[1]]["1"]["paired_difference"]["undefined_draws"] == 0
    # Unequal union sizes distinguish pooled counts from the mean of episodes.
    p=miou(values.sum(0))[1,0,0]
    assert p != miou(values)[...,1,0,0].mean()
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize("fault",["float","negative","I_gt_U","classes","empty","overflow","arms","severity"])
def test_geometry_dtypes_counts_and_exact_resource_bounds(fault):
    values=fixture()
    if fault == "float":values=values.astype(float)
    elif fault == "negative":values[0,0,0,0,0,0]=-1
    elif fault == "I_gt_U":values[0,0,0,0,0,0]=1000
    elif fault == "classes":values=values[...,:19]
    elif fault == "empty":values=values[:0]
    elif fault == "overflow":values[...,1,:]=2**52
    elif fault == "arms":values=values[:,:,:,:2]
    else:values=values[:,:,:2]
    with pytest.raises(ValueError):analysis.paired_summary(values)


@pytest.mark.parametrize("kw",[dict(chunk=True),dict(chunk=65),dict(replicates=0),dict(replicates=10001)])
def test_bootstrap_limits(kw):
    with pytest.raises(ValueError):analysis.paired_summary(fixture(),**kw)


def test_fixed_protocol_reuses_S5_without_new_aggregate_or_unseen_data_claim():
    p=analysis.protocol()
    assert p["types"] == list(analysis.CURVE_TYPES) and p["severities"] == [1,3,5]
    assert p["new_prediction_cells"] == 30 and p["new_image_method_evaluations"] == 43470
    assert p["uncertainty"]["replicates"] == 10000 and "after G02S5 outcomes" in p["exposure"]
    assert "no new S5 inference" in p["severity5"] and "no macro15" in p["estimates"]
