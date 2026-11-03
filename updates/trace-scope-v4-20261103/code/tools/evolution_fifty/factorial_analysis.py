"""Prospective M04 paired estimands on synthetic or separately admitted I/U.

No files, annotations, predictions, models, queue or scoring permission here.
"""

from __future__ import annotations

import numpy as np

from .bootstrap import require
from .factorial_prediction import STUDIES
from .factorial_runtime import ARMS
from .generalization_score import finite_or_none,interval,miou,stream_seed

CONTRASTS = {
    "native_C_minus_Confidence":(-1,1,0,0),
    "C_order_minus_conf_order_at_C_distribution":(0,1,-1,0),
    "C_order_minus_conf_order_at_conf_distribution":(-1,0,0,1),
    "C_distribution_minus_conf_distribution_at_C_order":(0,1,0,-1),
    "C_distribution_minus_conf_distribution_at_conf_order":(-1,0,1,0),
    "distribution_order_interaction":(1,1,-1,-1),
}


def contrast(values,weights):
    require(values.shape[-1] == len(ARMS) and len(weights) == len(ARMS), "M04 contrast dimensions differ")
    # Unused arms must not turn a well-defined contrast into NaN through 0*NaN.
    return sum(w*values[...,i] for i,w in enumerate(weights) if w)


def paired_summary(episode_counts,dataset,*,replicates=10000,chunk=32):
    values = np.asarray(episode_counts)
    require(dataset in ("coco_object","pc60") and values.ndim == 5
            and 0 < values.shape[0] <= 10000 and values.shape[1:4] == (len(STUDIES),len(ARMS),2)
            and 0 < values.shape[-1] <= 256 and values.dtype == np.int64
            and values.nbytes <= 256*1024**2 and bool((values >= 0).all())
            and bool((values[...,0,:] <= values[...,1,:]).all()), "Invalid M04 paired episode I/U")
    require(type(replicates) is int and 0 < replicates <= 10000
            and type(chunk) is int and 0 < chunk <= 64, "M04 bootstrap resource bound exceeded")
    episodes = values.shape[0]
    require(int(values.max())*episodes < 2**52, "M04 totals exceed exact float64 count range")
    seed,key = stream_seed("M04",dataset,stage="rank_distribution")
    rng = np.random.Generator(np.random.PCG64(seed))
    flat = values.reshape(episodes,-1).astype(np.float64)
    samples = np.empty((replicates,len(STUDIES),len(ARMS)),dtype=np.float64)
    for start in range(0,replicates,chunk):
        size = min(chunk,replicates-start)
        draws = rng.integers(0,episodes,size=(size,episodes))
        weights = np.stack([np.bincount(row,minlength=episodes) for row in draws]).astype(np.float64)
        samples[start:start+size] = miou((weights@flat).reshape((size,)+values.shape[1:]))
    point = miou(values.sum(0,dtype=np.int64))
    summaries = {}
    for i,study in enumerate(STUDIES):
        summaries[study] = dict(arms={arm:dict(mIoU=finite_or_none(point[i,j]),**interval(samples[:,i,j]))
            for j,arm in enumerate(ARMS)},contrasts={name:dict(delta_pp=finite_or_none(contrast(point[i],weights)),
                **interval(contrast(samples[:,i],weights))) for name,weights in CONTRASTS.items()})
    return dict(dataset=dataset,studies=summaries,episodes=episodes,
        bootstrap=dict(key=key,seed_uint128=seed,replicates=replicates,
            unit="Whole original source episode, identical draws across both studies and all4 arms",
            interval="95pointwise linear percentile, no multiplicity adjustment; undefined draws retained/nullCI"),
        scope="Fixed configuration, historical exposure; natural includes EPS admission differences, common-active is conditioned; not candidate causality"),samples
