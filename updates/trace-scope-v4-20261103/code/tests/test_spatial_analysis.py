import hashlib
import io
import json

import cv2
import numpy as np
from PIL import Image
import pytest

from tools.evolution_fifty import spatial_analysis as metric
from tools.evolution_fifty.risk_score_inputs import decode_annotation


def test_native_palette_PIL_geometry_not_G_OpenCV_geometry():
    original = np.array([[0,1,0],[1,0,1]],dtype=np.uint8)
    image = Image.fromarray(original).convert("P")
    image.putpalette([255,0,0,0,255,0]+[0]*762)
    stream = io.BytesIO()
    image.save(stream,format="PNG")
    actual = decode_annotation(stream.getvalue(),(224,224),2,False)
    assert np.array_equal(actual,np.asarray(Image.fromarray(original).resize((224,224),Image.Resampling.NEAREST)))
    assert not np.array_equal(actual,cv2.resize(original,(224,224),interpolation=cv2.INTER_NEAREST))


def test_native_zero_ignore_mapping():
    image = Image.fromarray(np.array([[0,1,2,255]],np.uint8))
    stream = io.BytesIO()
    image.save(stream,format="PNG")
    assert set(np.unique(decode_annotation(stream.getvalue(),(224,224),2,True))) == {0,1,255}


def test_fixed_contrasts_shared_episode_draws_and_key_are_reproducible():
    counts = np.ones((3,4,2,2),dtype=np.int64)*10
    counts[:,:,0,:] = np.array([3,8,5])[:,None,None]
    counts[:,0,0,:] += 1
    result = metric.paired_summary(counts,"pc60",replicates=29,chunk=7)
    assert result == metric.paired_summary(counts,"pc60",replicates=29,chunk=4)
    for contrast in result["contrasts"].values():
        assert contrast["delta_pp"] == pytest.approx(10.)
        assert contrast["CI95_pp"] == pytest.approx([10.,10.])
    key = json.dumps([20261102,"M02","pc60","spatial_preservation"],separators=(",",":"),ensure_ascii=True)
    assert result["bootstrap"]["key"] == key
    assert result["bootstrap"]["seed_uint128"] == int.from_bytes(hashlib.sha256(key.encode()).digest()[:16],"little")


def test_sparse_undefined_draws_are_retained_with_null_CI():
    counts = np.zeros((2,4,2,1),dtype=np.int64)
    counts[0,:,1,:] = 1
    result = metric.paired_summary(counts,"coco_object",replicates=100)
    assert 0 < result["contrasts"]["native_minus_global"]["undefined_draws"] < 100
    assert result["contrasts"]["native_minus_global"]["CI95_pp"] is None


def test_pooled_not_mean_image_IoU_and_unused_nan_does_not_poison_contrast():
    counts = np.zeros((2,4,2,1),dtype=np.int64)
    counts[:,:,1,:] = np.array([1,9])[:,None,None]
    counts[0,:,0,:] = 1
    counts[:,2] = 0
    result = metric.paired_summary(counts,"pc60",replicates=20)
    assert result["modes"]["native"]["mIoU"] == 10.
    assert result["modes"]["global"]["mIoU"] is None
    assert result["contrasts"]["native_minus_translated"]["CI95_pp"] == [0.,0.]
    assert result["contrasts"]["native_minus_global"]["CI95_pp"] is None


@pytest.mark.parametrize("fault",["dtype","negative","intersection","replicates","chunk","dataset"])
def test_estimator_bounds(fault):
    counts = np.ones((2,4,2,2),dtype=np.int64)
    options = {}
    if fault == "dtype": counts = counts.astype(float)
    elif fault == "negative": counts[0,0,0,0] = -1
    elif fault == "intersection": counts[0,0,0,0] = 2
    elif fault == "replicates": options["replicates"] = 10001
    elif fault == "chunk": options["chunk"] = 65
    with pytest.raises(ValueError): metric.paired_summary(counts,"other" if fault == "dataset" else "pc60",**options)
