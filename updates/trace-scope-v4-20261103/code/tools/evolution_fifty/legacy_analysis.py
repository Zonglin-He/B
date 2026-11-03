"""Prospective M03 paired estimators; no files, GT, models or permission."""

from __future__ import annotations

import numpy as np

from .bootstrap import require
from .generalization_score import finite_or_none,interval,miou,stream_seed

MODES = ("C.image","C.episode","legacy.image","legacy.episode")
PRIMARY = "single_C_vs_legacy_image"
CONTRASTS = {
    PRIMARY:(1,0,-1,0),
    "single_C_vs_legacy_episode":(0,1,0,-1),
    "image_vs_episode_for_C":(1,-1,0,0),
    "image_vs_episode_for_legacy":(0,0,1,-1),
    "diagnostic_objective_interaction":(1,-1,-1,1),
}


def contrast(values,weights):
    return sum(weight*values[...,i] for i,weight in enumerate(weights) if weight)


def paired_summary(episode_counts,dataset,*,replicates=10000,chunk=32):
    values = np.asarray(episode_counts)
    require(dataset in ("coco_object","pc60") and values.ndim == 4
            and 0 < values.shape[0] <= 10000 and values.shape[1:3] == (4,2)
            and 0 < values.shape[-1] <= 81 and values.dtype == np.int64
            and values.nbytes <= 256*1024**2 and bool((values >= 0).all())
            and bool((values[:,:,0,:] <= values[:,:,1,:]).all()), "Invalid M03 paired original-episode I/U")
    require(type(replicates) is int and 0 < replicates <= 10000
            and type(chunk) is int and 0 < chunk <= 64, "M03 bootstrap resource bound exceeded")
    episodes = len(values)
    require(int(values.max())*episodes < 2**52, "M03 counts exceed exact float64 integer range")
    seed,key = stream_seed("M03",dataset,stage="diagnostic_objective")
    rng = np.random.Generator(np.random.PCG64(seed))
    flat = values.reshape(episodes,-1).astype(np.float64)
    samples = np.empty((replicates,4),dtype=np.float64)
    for start in range(0,replicates,chunk):
        size = min(chunk,replicates-start)
        draws = rng.integers(0,episodes,size=(size,episodes))
        weights = np.stack([np.bincount(row,minlength=episodes) for row in draws]).astype(np.float64)
        samples[start:start+size] = miou((weights@flat).reshape((size,)+values.shape[1:]))
    point = miou(values.sum(0,dtype=np.int64))
    return dict(dataset=dataset,episodes=episodes,primary_contrast=PRIMARY,
        modes={mode:dict(mIoU=finite_or_none(point[i]),**interval(samples[:,i])) for i,mode in enumerate(MODES)},
        contrasts={name:dict(delta_pp=finite_or_none(contrast(point,weights)),
            **interval(contrast(samples,weights))) for name,weights in CONTRASTS.items()},
        bootstrap=dict(key=key,seed_uint128=seed,replicates=replicates,
            unit="Whole original source episode, identical draws across all4 diagnostic/objective conditions",
            interval="95pointwise linear percentile, no multiplicity adjustment; undefined retained/nullCI"),
        scope="Current confidenceTopK pool/engine/readout with pinned diagnostic components, not complete historical reproduction or independent calibration; no best-condition selection")
