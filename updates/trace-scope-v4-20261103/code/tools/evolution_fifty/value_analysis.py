"""Fixed V01 offline estimators. No file, GT, prediction or model access."""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction

import numpy as np

from .bootstrap import require
from .generalization_score import finite_or_none, interval, miou
from .value_runtime import NATIVE_STEPS, points


def time_match(dataset, model_seconds):
    grid = points(dataset)
    require(set(model_seconds) == {p["id"] for p in grid}, "V01 needs all five measured times")
    require(all(type(v) in (int,float) and math.isfinite(v) and v > 0
                for v in model_seconds.values()), "V01 times must be finite positive totals")
    anchor = next(p for p in grid if p["arm"] == "image_C" and p["steps"] == NATIVE_STEPS[dataset])
    exact = {k:Fraction(v) for k,v in model_seconds.items()}
    anchor_time = exact[anchor["id"]]
    # log(max(t/c,c/t)) has the same ordering as abs(log(t/c)), without
    # float-ratio underflow or a rounded false inequality at reciprocal ties.
    chosen = min((p for p in grid if p["arm"] == "image_confidence"),
        key=lambda p:(max(exact[p["id"]]/anchor_time,anchor_time/exact[p["id"]]),p["steps"]))
    ratio_exact = exact[chosen["id"]]/anchor_time
    try:ratio = float(ratio_exact)
    except OverflowError:ratio = None
    if ratio is not None and (not math.isfinite(ratio) or ratio == 0):ratio = None
    result = dict(C_point=anchor["id"], nearest_Confidence_point=chosen["id"],
                Confidence_to_C_time_ratio=ratio, near_time=Fraction(9,10) <= ratio_exact <= Fraction(11,10),
                selection="All-episode total synchronized model time, including episode0; no accuracy input",
                unmatched_policy="If outside10percent report no near-time point and retain full curve; no interpolation")
    if ratio is None:
        result["unrepresentable_time_ratio_exact_fraction"] = dict(numerator=str(ratio_exact.numerator),denominator=str(ratio_exact.denominator))
    return result


def summarize(dataset, episode_counts, model_seconds, *, replicates=10000, chunk=32):
    """Whole original episodes, shared draws across all five fixed points."""
    grid, values = points(dataset), np.asarray(episode_counts)
    require(values.ndim == 4 and 0 < values.shape[0] <= 10000 and values.shape[1:3] == (5,2)
            and 1 <= values.shape[3] <= 81 and np.issubdtype(values.dtype,np.integer)
            and values.nbytes <= 256*1024**2
            and bool((values >= 0).all()) and bool((values[:,:,0,:] <= values[:,:,1,:]).all()),
            "Invalid V01 whole-episode fixed-grid counts")
    require(type(replicates) is int and 0 < replicates <= 10000
            and type(chunk) is int and 0 < chunk <= 64, "V01 bootstrap resource bound exceeded")
    require(int(values.max())*values.shape[0] < 2**52, "V01 count totals exceed exact float64 integer bound")
    match = time_match(dataset,model_seconds)
    key = json.dumps([20261102,"V01",dataset,"fixed_grid"],separators=(",",":"),ensure_ascii=True)
    seed = int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[:16],"little")
    rng = np.random.Generator(np.random.PCG64(seed))
    population = values.shape[0]
    flat = values.reshape(population,-1).astype(np.float64)
    samples = np.empty((replicates,5),dtype=np.float64)
    for start in range(0,replicates,chunk):
        size = min(chunk,replicates-start)
        draws = rng.integers(0,population,size=(size,population))
        weights = np.stack([np.bincount(row,minlength=population) for row in draws]).astype(np.float64)
        totals = (weights@flat).reshape((size,)+values.shape[1:])
        samples[start:start+size] = miou(totals)
    actual = miou(values.sum(0))
    names = [p["id"] for p in grid]
    c = names.index(match["C_point"])
    conf = names.index(match["nearest_Confidence_point"])
    native_conf = next(i for i,p in enumerate(grid) if p["arm"] == "image_confidence"
                       and p["steps"] == NATIVE_STEPS[dataset])
    contrast = lambda other:dict(difference_pp=finite_or_none(actual[c]-actual[other]),
                                 **interval(samples[:,c]-samples[:,other]))
    return dict(dataset=dataset,points={name:dict(point=grid[i],model_seconds=model_seconds[name],
        mIoU=finite_or_none(actual[i]),**interval(samples[:,i])) for i,name in enumerate(names)},
        time_matching=match, native_C_minus_same_steps_Confidence=contrast(native_conf),
        native_C_minus_nearest_Confidence=contrast(conf),
        near_time_accuracy_comparison=contrast(conf) if match["near_time"] else None,
        bootstrap=dict(key=key,seed_uint128=seed,replicates=replicates,original_episodes=population,
            shared_point_draws=True,interval="95percent pointwise linear percentile; any undefined draw => null CI",
            historical_exposure="Conditional on fixed previously selected configuration, not independent calibration"),
        aggregation="Native class-wise pooled I/U then class mean, not mean per-image IoU",
        causality_claim=False)
