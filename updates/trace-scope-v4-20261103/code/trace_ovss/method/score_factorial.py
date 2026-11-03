"""E02 assignment-only controls; not a replacement for production TRACE.

Exchange score multisets and ordering within each source image. This helper
does not run a model or claim a new adaptation result. Natural admission is
reported per arm; common-active conditioning is explicitly a separate study.
"""

from __future__ import annotations

import math

from .trace.constants import EPS

ARMS = ("Dk_Ok", "Dc_Oc", "Dc_Ok", "Dk_Oc")


def assign_score_factorial(candidates, *, common_active=False):
    """Return per-candidate scores and scalar budget metadata for four arms.

    Required immutable identifiers: source_id and candidate_id (strings), with
    unique candidate IDs inside each source. area is the original positive
    spatial token count, kappa and C are detached finite host probabilities.
    Ties use lexical candidate_id. Neither GT nor outcome fields are accepted.
    Floating coefficients here describe ideal scalar budgets; runtime dtype
    coefficients must still be measured in the real objective/gate.
    """
    if type(common_active) is not bool:
        raise ValueError("common_active must be an explicit boolean")
    groups, seen = {}, set()
    for index, candidate in enumerate(candidates):
        if set(candidate) != {"source_id", "candidate_id", "area", "kappa", "C"}:
            raise ValueError("Candidate schema must not contain outcome/GT fields")
        source, identity = candidate["source_id"], candidate["candidate_id"]
        if not isinstance(source, str) or not source or not isinstance(identity, str) or not identity:
            raise ValueError("Stable nonempty source/candidate identifiers required")
        if (source, identity) in seen:
            raise ValueError("Duplicate candidate identity")
        seen.add((source, identity))
        if type(candidate["area"]) is not int or candidate["area"] <= 0:
            raise ValueError("Positive original candidate area required")
        for key in ("kappa", "C"):
            value = candidate[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Finite host probability required")
        if not common_active or min(candidate["kappa"], candidate["C"]) > EPS:
            groups.setdefault(source, []).append(index)
    assigned = {arm: [None] * len(candidates) for arm in ARMS}
    for indices in groups.values():
        orders = {name: sorted(indices, key=lambda i: (candidates[i][name], candidates[i]["candidate_id"]))
                  for name in ("kappa", "C")}
        values = {name: sorted(float(candidates[i][name]) for i in indices) for name in ("kappa", "C")}
        for arm, distribution, ordering in (("Dk_Ok", "kappa", "kappa"), ("Dc_Oc", "C", "C"),
                                           ("Dc_Ok", "C", "kappa"), ("Dk_Oc", "kappa", "C")):
            for index, value in zip(orders[ordering], values[distribution], strict=True):
                assigned[arm][index] = value
    records = {}
    for arm, weights in assigned.items():
        active_groups = {source: [i for i in indices if weights[i] > EPS]
                         for source, indices in groups.items()}
        active_groups = {source: ids for source, ids in active_groups.items() if ids}
        coefficients = [0.] * len(candidates)
        for ids in active_groups.values():
            denominator = math.fsum(weights[i] for i in ids)
            for i in ids:
                coefficients[i] = weights[i] / denominator / len(active_groups)
        reference = assigned["Dc_Oc" if arm.startswith("Dc") else "Dk_Ok"]
        records[arm] = dict(assigned_scores=weights, ideal_region_coefficients=coefficients,
            active_candidate_ids=[(candidates[i]["source_id"], candidates[i]["candidate_id"])
                                  for i, value in enumerate(weights) if value is not None and value > EPS],
            active_source_ids=list(active_groups),
            changed_assigned_scores=sum(a != b for a, b in zip(weights, reference, strict=True)),
            non_singleton_sources=sum(len(ids) > 1 for ids in active_groups.values()))
    return dict(arms=records, scope="common_active_conditioned" if common_active else "natural_per_arm_admission",
        candidate_count=len(candidates), retained_candidates=sum(map(len, groups.values())),
        extra_C_diagnostics_required=True, coefficients_are_runtime_measured=False,
        original_candidate_order=[(r["source_id"], r["candidate_id"]) for r in candidates])
