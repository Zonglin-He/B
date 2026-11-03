"""Prospective episode-cluster summaries of isolated candidate consequences.

These are conditional panel statistics, not full-split mIoU or candidate-IID
confidence intervals. No model, files, annotation reader or execution queue.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .bootstrap import require
from .pixel_transitions import FIELDS, rates

REPLICATES, SEED = 10000, 20261102
HARM_THRESHOLDS = (0., .001, .005)


def score_groups(rows):
    """Freeze score-only quantile membership before update outcomes are read."""
    require(bool(rows), "Nonempty candidate panel required")
    groups, cuts = {"all": list(range(len(rows)))}, {}
    for field in ("C", "confidence"):
        values = np.asarray([r[field] for r in rows], dtype=np.float64)
        require(bool(np.isfinite(values).all()) and bool(((values >= 0) & (values <= 1)).all()),
                "Invalid initial candidate score")
        edges = np.quantile(values, [.2, .4, .6, .8], method="linear")
        bins = np.searchsorted(edges, values, side="right")
        cuts[field] = edges.tolist()
        for q in range(5):
            groups[f"{field}_Q{q+1}"] = np.flatnonzero(bins == q).tolist()
    return dict(groups=groups, cuts=cuts,
        tie_policy="Exact boundary ties go to higher bin; retain empty quantiles, never force equal counts")


def contributions(measurement):
    raw_counts = [measurement["counts"][key] for key in FIELDS]
    require(all(type(v) is int and 0 <= v <= 2**53 for v in raw_counts), "Invalid exact pixel counts")
    counts = np.asarray(raw_counts, dtype=np.int64)
    rates(counts)
    n = int(counts[0])
    mean = measurement["mean_true_class_support_delta"]
    require((mean is None) == (n == 0), "Support denominator/undefined value differs")
    require(np.isfinite(measurement["support_delta_sum"]), "Nonfinite support delta sum")
    if n:
        require(np.isfinite(mean) and -1 <= mean <= 1
                and np.isclose(mean*n, measurement["support_delta_sum"], rtol=1e-6, atol=1e-6),
                "Inconsistent support sum/mean")
    else:
        require(measurement["support_delta_sum"] == 0, "Nonzero support sum with empty region")
    return np.asarray([*counts, float(n > 0), 0. if mean is None else mean,
        measurement["support_delta_sum"],
        *[int(mean < -t) if n else 0 for t in HARM_THRESHOLDS]], dtype=np.float64)


def values(totals):
    def divide(a, b, factor=1.):
        return np.divide(factor*a, b, out=np.full(np.shape(a), np.nan), where=b > 0)
    # Counts are exact nonnegative integers aggregated asfloat64 (tiny panel).
    pixel = rates(totals[..., :5].astype(np.int64))
    return dict(candidate_mean_support_change=divide(totals[..., 6], totals[..., 5]),
        valid_pixel_mean_support_change=divide(totals[..., 7], totals[..., 0]),
        **{f"support_harm_rate_threshold_{threshold}": divide(totals[..., 8+i], totals[..., 5], 100.)
           for i, threshold in enumerate(HARM_THRESHOLDS)}, **pixel)


def bounds(samples):
    samples = np.asarray(samples)
    undefined = int((~np.isfinite(samples)).sum())
    pair = np.quantile(samples, [.025, .975], method="linear") if not undefined else (None, None)
    return dict(lower95=None if pair[0] is None else float(pair[0]),
        upper95=None if pair[1] is None else float(pair[1]), undefined_replicates=undefined,
        total_replicates=len(samples), no_replicate_dropped=True)


def summary(rows, outcomes, groups, dataset, *, replicates=REPLICATES):
    """One checkpoint/region at a time; episode draws shared bydataset seed."""
    require(len(rows) == len(outcomes) > 0 and type(replicates) is int and replicates > 0,
            "Invalid summary population or replicate count")
    episodes = sorted({r["episode"] for r in rows})
    require(all(type(e) is int and e >= 0 for e in episodes) and len(episodes) >= 2,
            "At least two original episodes required")
    position = {e:i for i, e in enumerate(episodes)}
    names = list(groups)
    require("all" in groups and sorted(groups["all"]) == list(range(len(rows))), "Incomplete all-group membership")
    total = np.zeros((len(episodes), len(names), 11), dtype=np.float64)
    coverage = {}
    for j, (name, indices) in enumerate(groups.items()):
        require(len(indices) == len(set(indices)) and all(type(i) is int and 0 <= i < len(rows) for i in indices),
                "Invalid group membership")
        valid, sources = 0, set()
        for i in indices:
            contribution = contributions(outcomes[i])
            total[position[rows[i]["episode"]], j] += contribution
            valid += int(contribution[5])
            sources.add(rows[i]["source_id"])
        coverage[name] = dict(candidates=len(indices), valid_candidates=valid,
            zero_valid_candidates=len(indices)-valid, unique_sources=len(sources),
            original_episodes=sum(bool(total[e, j, 5]) for e in range(len(episodes))))
    points = values(total.sum(0))
    seed = int.from_bytes(hashlib.sha256(f"M01-risk:{SEED}:{dataset}".encode()).digest()[:8], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = {metric:[] for metric in points}
    for start in range(0, replicates, 32):
        draw = rng.integers(len(episodes), size=(min(32, replicates-start), len(episodes)))
        computed = values(total[draw].sum(1))
        for metric, estimates in computed.items():
            samples[metric].append(estimates)
    samples = {k:np.concatenate(v) for k, v in samples.items()}
    result = {}
    for j, name in enumerate(names):
        result[name] = dict(coverage=coverage[name], metrics={metric:dict(
            value=None if not np.isfinite(point[j]) else float(point[j]), **bounds(samples[metric][:, j]))
            for metric, point in points.items()})
    contrasts = {}
    for score in ("C", "confidence"):
        high, low = f"{score}_Q5", f"{score}_Q1"
        if high in names and low in names:
            a, b = names.index(high), names.index(low)
            contrasts[f"{high}_minus_{low}"] = {metric:dict(
                value=None if not np.isfinite(point[a]-point[b]) else float(point[a]-point[b]),
                **bounds(samples[metric][:, a]-samples[metric][:, b])) for metric, point in points.items()}
    return dict(groups=result, contrasts=contrasts, source_clusters=len(episodes), replicates=replicates,
        grouping="Original parent episodes, not candidate IID; candidate-weighted support/harm and separately pixel-occurrence-weighted metrics",
        limitations="Conditional confidence-TopK-selected panel and fixed updates; not causal pool independence or full-split mIoU",
        intervals="Pointwise95percentile; no multiplicity correction; undefined draws retained as null interval")
