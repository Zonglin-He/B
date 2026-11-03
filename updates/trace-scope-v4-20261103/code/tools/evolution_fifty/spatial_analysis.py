"""Prospective M02 estimators; no files, GT, models or execution permission."""

from __future__ import annotations

import numpy as np

from .bootstrap import require
from .generalization_score import finite_or_none, interval, miou, stream_seed
from .spatial_runtime import MODES

CONTRASTS = dict(native_minus_translated=(1,-1,0,0),
                 native_minus_global=(1,0,-1,0), native_minus_confidence=(1,0,0,-1))


def paired_summary(episode_counts,dataset,*,replicates=10000,chunk=32):
    values = np.asarray(episode_counts)
    require(dataset in ("coco_object","pc60") and values.ndim == 4
            and 0 < values.shape[0] <= 10000 and values.shape[1:3] == (4,2)
            and 0 < values.shape[-1] <= 81 and values.dtype == np.int64
            and values.nbytes <= 256*1024**2 and bool((values >= 0).all())
            and bool((values[:,:,0,:] <= values[:,:,1,:]).all()), "Invalid M02 paired episode I/U")
    require(type(replicates) is int and 0 < replicates <= 10000
            and type(chunk) is int and 0 < chunk <= 64, "M02 bootstrap resource bound exceeded")
    episodes = values.shape[0]
    require(int(values.max())*episodes < 2**52, "M02 totals exceed exact float64 count range")
    seed,key = stream_seed("M02",dataset,stage="spatial_preservation")
    rng = np.random.Generator(np.random.PCG64(seed))
    flat = values.reshape(episodes,-1).astype(np.float64)
    samples = np.empty((replicates,4),dtype=np.float64)
    for start in range(0,replicates,chunk):
        size = min(chunk,replicates-start)
        draws = rng.integers(0,episodes,size=(size,episodes))
        weights = np.stack([np.bincount(row,minlength=episodes) for row in draws]).astype(np.float64)
        samples[start:start+size] = miou((weights@flat).reshape((size,)+values.shape[1:]))
    point = miou(values.sum(0,dtype=np.int64))
    def contrast(array,weights):
        return sum(w*array[...,i] for i,w in enumerate(weights) if w)
    return dict(dataset=dataset,episodes=episodes,
        modes={mode:dict(mIoU=finite_or_none(point[i]),**interval(samples[:,i])) for i,mode in enumerate(MODES)},
        contrasts={name:dict(delta_pp=finite_or_none(contrast(point,weights)),
            **interval(contrast(samples,weights))) for name,weights in CONTRASTS.items()},
        bootstrap=dict(key=key,seed_uint128=seed,replicates=replicates,
            unit="Whole original source episode, identical draws across all four modes",
            interval="95pointwise linear percentile; no multiplicity adjustment; undefined draws retained/null CI"),
        scope="Native pooled class I/U at fixed historically selected configuration, not independent calibration or candidate causality")
