"""Offline-only continuous candidate consequences; never an adaptation input."""

from __future__ import annotations

import math

import numpy as np

from .bootstrap import require
from .pixel_transitions import transitions, rates, FIELDS


def measure(before, after, gt, before_true_support, after_true_support, candidate_mask, classes):
    """Keep inside/outside/whole-image denominators and zero-coverage explicit.

    True-class supports must already use the frozen query-to-benchmark mapping.
    The replay layer produces these from fixed checkpoints after prediction seal.
    """
    mask = np.asarray(candidate_mask)
    require(mask.dtype == np.bool_ and mask.shape == gt.shape == before_true_support.shape == after_true_support.shape,
            "Offline support/region geometry differs")
    for support in (before_true_support, after_true_support):
        valid = gt != 255
        require(bool(np.isfinite(support[valid]).all())
                and bool(((support[valid] >= 0) & (support[valid] <= 1)).all()), "Invalid mapped true-class support")
    result = {}
    for name, selected in (("inside", mask), ("outside", ~mask), ("whole", np.ones_like(mask))):
        label = np.where(selected, gt, 255)
        counts = transitions(before, after, label, classes)
        n = int(counts[0])
        metrics = rates(counts)
        valid = selected & (gt != 255)
        delta = float((after_true_support[valid].astype(np.float64)-before_true_support[valid]).sum())
        result[name] = dict(counts=dict(zip(FIELDS, map(int, counts), strict=True)),
            **{k: float(v) if np.isfinite(v) else None for k,v in metrics.items()},
            support_delta_sum=delta, mean_true_class_support_delta=delta/n if n else None,
            net_correct_pixels=int(counts[4]-counts[3]),
            primary_support_harm=(delta/n < -.001) if n else None,
            support_harm_thresholds={str(t): (delta/n < -t) if n else None for t in (0., .001, .005)},
            zero_valid_pixels=n == 0)
    return result


def matching_stratum(confidence, benchmark_class, area, shape):
    require(type(confidence) in (float, int) and math.isfinite(confidence) and 0 <= confidence <= 1,
            "Invalid confidence")
    require(type(benchmark_class) is int and benchmark_class >= 0 and type(area) is int
            and len(shape) == 2 and all(type(x) is int and x > 0 for x in shape)
            and 0 < area <= shape[0]*shape[1], "Invalid class/area")
    confidence_bin = min(19, int(confidence/.05))
    fraction = area/(shape[0]*shape[1])
    area_bin = min(4, int(np.searchsorted([.01, .05, .2, .5, 1.], fraction, side="right")))
    return benchmark_class, confidence_bin, area_bin


def matched_C_groups(rows):
    """Prespecified median groups; preserve unmatched/tied population coverage."""
    strata = {}
    for index, row in enumerate(rows):
        c = row["C"]
        require(type(c) in (float, int) and math.isfinite(c) and 0 <= c <= 1, "Invalid detached C")
        key = matching_stratum(row["confidence"], row["benchmark_class"], row["area"], row["shape"])
        strata.setdefault(key, []).append(index)
    groups, unmatched, ties = [], [], []
    for key, indices in strata.items():
        median = float(np.median([rows[i]["C"] for i in indices]))
        high = [i for i in indices if rows[i]["C"] > median]
        low = [i for i in indices if rows[i]["C"] < median]
        tied = [i for i in indices if rows[i]["C"] == median]
        ties.extend(tied)
        if high and low:
            groups.append(dict(stratum=list(key), high=high, low=low, median_C=median, ties=tied))
        else:
            unmatched.extend(indices)
    return dict(groups=groups, total_candidates=len(rows), total_strata=len(strata),
                matched_strata=len(groups), matched_non_tied_candidates=sum(len(g["high"])+len(g["low"]) for g in groups),
                exact_median_tied_candidates=ties, unmatched_candidates=unmatched,
                policy="No post-outcome bin merging or removal of unmatched coverage")
