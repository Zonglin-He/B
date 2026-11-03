"""Pure G03 fixed severity-curve estimators; no IO or scoring permission."""

from __future__ import annotations

import numpy as np

from .bootstrap import require
from .generalization import ARMS,CURVE_TYPES
from .generalization_score import describe,miou,stream_seed

SEVERITIES=(1,3,5)


def protocol():
    return dict(experiment="G03_fixed_five_severity_curves",types=list(CURVE_TYPES),
        severities=list(SEVERITIES),arms=list(ARMS),backbone="ViT-L/14",dataset="voc20",classes=20,
        source_images=1449,new_prediction_conditions=10,new_prediction_cells=30,
        new_image_method_evaluations=43470,
        severity5="Exact sealed G1954 L14 five-type predictions/counts only, no new S5 inference",
        annotation_geometry="Palette indices then OpenCV INTER_NEAREST224, identical to G primary; not native PIL nearest",
        estimates="Per type/severity pooled class I/U mIoU for all3arms and paired C-Confidence pp; no macro15, AUC, slope or winner selection",
        uncertainty=dict(replicates=10000,root_seed=20261102,key=[20261102,"G03","voc20","severity_curves"],
            seed_encoding="Compact ASCII JSON, SHA256 first16bytes little-endian uint128, PCG64",
            unit="Whole original episode, identical draws across all five types, three severities and three arms",
            interval="95pointwise linear percentile; no multiplicity correction; undefined draws retained/nullCI"),
        pairing="Exact original source order/episode grouping must match both new s1/s3 and reused s5 before concatenating counts",
        exposure="Registered after G02S5 outcomes but before G03s1/s3 predictions; fixed previously declared types/severities unchanged; historically exposed settings, not untouched calibration",
        scope="Pure estimator, not full streaming scorer or GT admission; no production/default changes")


def paired_summary(episode_counts,*,replicates=10000,chunk=32):
    values=np.asarray(episode_counts)
    require(values.ndim == 6 and values.dtype == np.int64 and 0 < values.shape[0] <= 10000
            and values.shape[1:] == (len(CURVE_TYPES),len(SEVERITIES),len(ARMS),2,20)
            and values.nbytes <= 256*1024**2 and bool((values >= 0).all())
            and bool((values[...,0,:] <= values[...,1,:]).all()),"Invalid G03 paired original-episode I/U")
    require(type(replicates) is int and 0 < replicates <= 10000 and type(chunk) is int
            and 0 < chunk <= 64,"G03 bootstrap resource bound exceeded")
    episodes=values.shape[0]
    require(int(values.max())*episodes < 2**52,"G03 counts exceed exact float64 sum bound")
    seed,key=stream_seed("G03","voc20",stage="severity_curves")
    rng=np.random.Generator(np.random.PCG64(seed))
    flat=values.reshape(episodes,-1).astype(np.float64)
    samples=np.empty((replicates,len(CURVE_TYPES),len(SEVERITIES),len(ARMS)),dtype=np.float64)
    for start in range(0,replicates,chunk):
        size=min(chunk,replicates-start)
        draws=rng.integers(0,episodes,size=(size,episodes))
        weights=np.stack([np.bincount(row,minlength=episodes) for row in draws]).astype(np.float64)
        samples[start:start+size]=miou((weights@flat).reshape((size,)+values.shape[1:]))
    points=miou(values.sum(0,dtype=np.int64))
    return dict(curves={kind:{str(severity):describe(points[t,s],samples[:,t,s])
        for s,severity in enumerate(SEVERITIES)} for t,kind in enumerate(CURVE_TYPES)},episodes=episodes,
        bootstrap=dict(key=key,seed_uint128=seed,replicates=replicates,
            unit=protocol()["uncertainty"]["unit"],interval=protocol()["uncertainty"]["interval"]),
        severity5_reused=True,aggregation="Pool I/U within each type/severity first; no cross-type or cross-severity pooling",
        exposure=protocol()["exposure"],scope=protocol()["scope"]),samples
