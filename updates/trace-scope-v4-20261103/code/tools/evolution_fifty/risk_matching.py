"""Fixed-stratum offline C contrasts, with original-episode uncertainty.

Membership uses initial scores and optionally initial GT quality, never update
consequences. Equal stratum weighting is explicit; sparse draws remain undefined.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .bootstrap import require
from .risk_outcomes import matching_stratum
from .risk_summary import SEED, REPLICATES, bounds, contributions, values


def membership(rows, qualities=None, query_indices=None):
    require(qualities is None or len(qualities) == len(rows), "Quality-control population differs")
    require(query_indices is None or len(query_indices) == len(rows), "Query-control population differs")
    strata, undefined_quality = {}, []
    for i, row in enumerate(rows):
        key = matching_stratum(row["confidence"], row["benchmark_class"], row["area"], row["shape"])
        c = row["C"]
        require(type(c) in (int, float) and np.isfinite(c) and 0 <= c <= 1, "Invalid initial C")
        if query_indices is not None:
            query = query_indices[i]
            require(type(query) is int and query >= 0, "Invalid initial query class")
            key += (query,)
        if qualities is not None:
            q = qualities[i]["fraction_correct"]
            if q is None:
                undefined_quality.append(i)
                continue
            require(type(q) in (int, float) and np.isfinite(q) and 0 <= q <= 1, "Invalid initial GT quality")
            key += (int(np.searchsorted([.25, .5, .75], q, side="right")),)
        strata.setdefault(key, []).append(i)
    matched, unmatched, tied = [], [], []
    for key, indices in sorted(strata.items()):
        median = float(np.median([rows[i]["C"] for i in indices]))
        high = [i for i in indices if rows[i]["C"] > median]
        low = [i for i in indices if rows[i]["C"] < median]
        ties = [i for i in indices if rows[i]["C"] == median]
        tied.extend(ties)
        if high and low:
            matched.append(dict(stratum=list(key), median_C=median, high=high, low=low, ties=ties))
        else:
            unmatched.extend(indices)
    return dict(strata=matched, total_candidates=len(rows), total_strata=len(strata),
        matched_candidates=sum(len(s["high"])+len(s["low"]) for s in matched),
        exact_median_ties=tied, unmatched_candidates=unmatched, undefined_initial_quality=undefined_quality,
        estimator="Equal-weight mean of within-fixed-stratum high-C minus low-C metrics; no pooled unmatched comparison",
        sparse_policy="No bin widening; frozen medians/memberships; any undefined stratum makes that replicate undefined")


def summarize(rows, outcomes, grouping, dataset, *, replicates=REPLICATES):
    require(len(rows) == len(outcomes) == grouping["total_candidates"] and len(rows) > 0,
            "Matched summary population differs")
    require(type(replicates) is int and replicates > 0, "Invalid matched replicate count")
    episodes = sorted({r["episode"] for r in rows})
    require(len(episodes) >= 2 and all(type(e) is int and e >= 0 for e in episodes), "Original episode clusters required")
    strata = grouping["strata"]
    metrics = list(values(np.zeros(11)))
    if not strata:
        return dict(coverage=grouping, matched_strata=0, replicates=replicates,
            contrasts={k:dict(value=None, lower95=None, upper95=None,
                undefined_replicates=replicates, total_replicates=replicates, no_replicate_dropped=True) for k in metrics})
    position = {e:i for i, e in enumerate(episodes)}
    # Bootstrap counts per original episode, not candidate IID. Loop strata to
    # keep memory bounded even for hundreds of sparse matching cells.
    seed = int.from_bytes(hashlib.sha256(f"M01-risk:{SEED}:{dataset}".encode()).digest()[:8], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(len(episodes), size=(replicates, len(episodes)))
    weights = np.zeros((replicates, len(episodes)), dtype=np.int32)
    for i, draw in enumerate(draws):
        weights[i] = np.bincount(draw, minlength=len(episodes))
    point = {k:0. for k in metrics}
    samples = {k:np.zeros(replicates, dtype=np.float64) for k in metrics}
    seen = set()
    for stratum in strata:
        total = np.zeros((len(episodes), 2, 11), dtype=np.float64)
        for side, key in enumerate(("high", "low")):
            require(bool(stratum[key]), "Empty matched side")
            for i in stratum[key]:
                require(type(i) is int and 0 <= i < len(rows) and i not in seen, "Duplicate matched membership")
                seen.add(i)
                total[position[rows[i]["episode"]], side] += contributions(outcomes[i])
        p = values(total.sum(0))
        for k in metrics:
            point[k] += float(p[k][0]-p[k][1])/len(strata)
        for start in range(0, replicates, 32):
            sums = np.einsum("re,esk->rsk", weights[start:start+32], total)
            b = values(sums)
            for k in metrics:
                samples[k][start:start+len(sums)] += (b[k][:, 0]-b[k][:, 1])/len(strata)
    return dict(coverage=grouping, matched_strata=len(strata), replicates=replicates,
        contrasts={k:dict(value=float(v) if np.isfinite(v) else None, **bounds(samples[k])) for k, v in point.items()},
        limitations="Conditional within initial-score/quality strata; observational C, not a causal or independent-pool claim")
