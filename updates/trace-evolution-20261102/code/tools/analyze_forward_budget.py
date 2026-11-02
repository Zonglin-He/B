"""Offline TRACE forward-budget audit. Standard library only; never runs a model.

Only Full rows selected from final_audit/summary.csv are inspected. Saved labels
and predictions are not opened; class metrics are not recomputed or used for
selection. Thresholds are fixed, not tuned to IoU.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import struct


EPS = 1e-8
THRESHOLDS = (0.0, 0.001, 0.005, 0.01, 0.02)
DEFAULT_SUMMARY = Path("/home/wwww/hzl/results/trace_removal_suite_20261025_0730/final_audit/summary.csv")
DEFAULT_OUTPUT = Path("/home/wwww/hzl/results/trace_efficiency_diagnosis_20261025/forward_analysis")
PANELS = {"voc21": (512, "cls_voc21.txt"), "pc60": (512, "cls_context60.txt"),
          "coco_object": (512, "cls_coco_object.txt"), "coco_cd": (500, "coco_cd.txt")}
CONTRACTS = ("method/trace/constants.py", "method/trace/reliability.py",
             "method/trace/estimator.py", "method/trace/objective.py",
             "method/trace/interventions.py", "method/trace/engine.py",
             "method/trace/candidates.py", "data/vocabulary.py", "models/dense_clip.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class Evidence:
    """Read and seal exactly the inputs used; fail if any change during analysis."""

    def __init__(self):
        self.files = {}

    def read(self, path):
        path = Path(path).resolve()
        data = path.read_bytes()
        record = {"sha256": sha256(data), "bytes": len(data)}
        require(str(path) not in self.files or self.files[str(path)] == record,
                f"Evidence changed: {path}")
        self.files[str(path)] = record
        return data

    def json(self, path):
        return json.loads(self.read(path))

    def csv(self, path):
        return list(csv.DictReader(io.StringIO(self.read(path).decode("utf-8"))))

    def recheck(self):
        for path, record in self.files.items():
            require(sha256(Path(path).read_bytes()) == record["sha256"],
                    f"Evidence changed before publication: {path}")


def probability(value):
    number = float(value)
    require(math.isfinite(number) and 0 <= number <= 1,
            f"Expected finite probability, got {value!r}")
    return number


def integer(value, minimum=0):
    number = float(value)
    require(math.isfinite(number) and number == int(number) and number >= minimum,
            f"Expected integer >= {minimum}, got {value!r}")
    return int(number)


def reliability(target, counterfactual, residual):
    target, counterfactual, residual = map(probability, (target, counterfactual, residual))
    quality = min(target, counterfactual)
    return quality * (1.0 - max(residual - target, 0.0))


@dataclass(frozen=True)
class Candidate:
    episode: int
    image_id: str
    patch: int
    class_idx: int
    area: int
    mask_sha256: str
    target: float
    counterfactual: float
    residual: float
    quality: float
    rho: float

    @property
    def bound(self):
        return self.quality * (1.0 - self.target)

    @property
    def lower(self):
        return self.quality * self.target

    @property
    def exact_drop(self):
        return self.area == 0 or self.quality <= EPS or self.target == 1.0

    def whole_mask(self, tokens):
        return self.area == tokens and self.mask_sha256 == sha256(bytes([1]) * tokens)


def parse_candidates(rows, queries, tokens):
    candidates, seen = [], set()
    for row in rows:
        target, counterfactual, residual, quality, dominance, rho = (
            probability(row[key]) for key in ("target_support", "counterfactual_support",
                "residual_support", "quality", "dominance", "reliability"))
        require(quality == min(target, counterfactual), "Cached q != min(T,C)")
        require(dominance == max(residual - target, 0), "Cached dominance != max(D-T,0)")
        require(rho == reliability(target, counterfactual, residual),
                "Cached rho violates production formula")
        candidate = Candidate(integer(row["episode"]), row["image_id"], integer(row["patch_in_batch"]),
            integer(row["class_idx"]), integer(row["area"], 1), row["mask_sha256"],
            target, counterfactual, residual, quality, rho)
        require(candidate.class_idx < queries, "Candidate outside text-query class axis")
        require(candidate.area <= tokens, "Region exceeds token grid")
        require(len(candidate.mask_sha256) == 64 and
                all(char in "0123456789abcdef" for char in candidate.mask_sha256), "Invalid mask hash")
        if candidate.area == tokens:
            require(candidate.whole_mask(tokens), "Whole-mask area/hash disagree")
        key = (candidate.episode, candidate.image_id, candidate.patch,
               candidate.class_idx, candidate.mask_sha256)
        require(key not in seen, "Duplicate candidate identity")
        seen.add(key)
        require(candidate.lower <= rho + 1e-15 and rho <= quality + 1e-15,
                "Reliability outside [q*T,q]")
        require(-1e-15 <= quality - rho <= candidate.bound + 1e-15,
                "Raw skip-drop error exceeds q*(1-T)")
        candidates.append(candidate)
    return candidates


def normalized_coefficients(weights, areas):
    """Real-arithmetic objective coefficients, evaluated in binary64."""
    require(len(weights) == len(areas), "Weight/region lengths differ")
    active = []
    for weight, area in zip(weights, areas):
        weight = probability(weight)
        area = integer(area)
        active.append(weight if weight > EPS and area > 0 else 0.0)
    total = math.fsum(active)
    coefficients = [value / total for value in active] if total > EPS else [0.0] * len(active)
    require(all(math.isfinite(value) for value in coefficients), "Nonfinite coefficient")
    require(math.isclose(math.fsum(coefficients), float(total > EPS), abs_tol=1e-12),
            "Coefficient normalization failure")
    return coefficients, total


def float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def float32_objective_coefficients(weights, areas):
    """Emulate production's scalar casts and sequential normalizer, without torch.

    Coefficients are real-valued cast-weight/normalizer ratios. Numerator
    multiplication/reduction rounding is not included and is not loss evidence.
    """
    require(len(weights) == len(areas), "Weight/region lengths differ")
    active, total = [], 0.0
    for weight, area in zip(weights, areas):
        weight, area = probability(weight), integer(area)
        value = float32(weight) if weight > EPS and area > 0 else 0.0
        active.append(value)
        total = float32(total + value)
    normalizer = total if total > EPS else 1.0
    return [value / normalizer for value in active], total


def coefficient_distance(before, after, areas, queries):
    queries = integer(queries, 1)
    a, total_a = normalized_coefficients(before, areas)
    b, total_b = normalized_coefficients(after, areas)
    active_a, active_b = total_a > EPS, total_b > EPS
    l1 = math.fsum(abs(x - y) for x, y in zip(a, b))
    # An empty objective has unit mass on a zero-loss dummy term. Omitting it
    # would give TV=1/2 for empty->active and an invalid log(K)*TV bound.
    extended_l1 = l1 + float(active_a != active_b)
    tv = extended_l1 / 2
    require(tv <= 1 + 1e-12, "TV exceeds one")
    return dict(coefficient_l1=l1, coefficient_l1_with_null=extended_l1, tv=tv,
                standard_tv=l1 / 2 if active_a == active_b else None,
                entropy_loss_bound_nats=math.log(queries) * tv,
                before_weight_sum=total_a, after_weight_sum=total_b,
                before_empty=not active_a, after_empty=not active_b,
                empty_to_active=not active_a and active_b,
                active_to_empty=active_a and not active_b,
                candidates_activated=sum(x == 0 and y > 0 for x, y in zip(a, b)),
                candidates_deactivated=sum(x > 0 and y == 0 for x, y in zip(a, b)))


def singleton_certificate(candidates):
    possible = [index for index, item in enumerate(candidates) if item.area and item.quality > EPS]
    if len(possible) == 1 and float32(candidates[possible[0]].lower) > EPS:
        return possible[0]
    return None


def policy_skip(candidates, policy, threshold=None):
    require(policy in {"exact_drop", "exact_plus_certified_singleton", "bound_skip", "all_drop_sensitivity"},
            f"Unknown policy: {policy}")
    if policy == "bound_skip":
        require(threshold in THRESHOLDS, "Threshold must belong to the predeclared grid")
    certified = singleton_certificate(candidates)
    return [item.exact_drop or
            (policy == "exact_plus_certified_singleton" and index == certified) or
            (policy == "bound_skip" and item.bound <= threshold) or
            policy == "all_drop_sensitivity"
            for index, item in enumerate(candidates)]


def select_panels(rows):
    selected = []
    for dataset, (images, _) in PANELS.items():
        matches = [row for row in rows if row["stage"] == "ablation" and
                   row["dataset"] == dataset and row["variant"] == "full" and
                   integer(row["images"]) == images]
        require(len(matches) == 1, f"Expected exactly one Full ablation panel for {dataset}/{images}")
        row = matches[0]
        if dataset == "coco_cd":
            require("fresh_full_coco_cd" in Path(row["result_path"]).parts,
                    "COCO-CD must use refreshed Full, never the old external control")
        selected.append(row)
    return selected


def load_panel(row, evidence, package_root):
    result_path = Path(row["result_path"]).resolve()
    directory = result_path.parent
    protocol_path = directory / "protocol.json"
    protocol = evidence.json(protocol_path)
    result = evidence.json(result_path)
    require(result["status"] == "completed", "Incomplete evidence")
    require(result["protocol_sha256"] == evidence.files[str(protocol_path)]["sha256"],
            "Protocol seal mismatch")
    dataset = row["dataset"]
    for saved in (protocol, result):
        require(saved["dataset"] == dataset and saved["variant"] == "full", "Panel identity mismatch")
        require(saved["images"] == integer(row["images"]), "Panel size mismatch")
        require(saved["split"] == "evaluation", "Non-evaluation panel")
        require(str(saved["full_split"]) == row["full_split"], "Full-split identity mismatch")
    require(not protocol.get("force_uniform_policy", False), "Nonstandard Full policy")
    source_root = Path(protocol["package_root"])
    for relative, expected in protocol["source_hashes"].items():
        require(sha256(evidence.read(source_root / "trace_ovss" / relative)) == expected,
                f"Frozen source mismatch: {relative}")
    for relative in CONTRACTS:
        require(sha256(evidence.read(package_root / "trace_ovss" / relative)) ==
                protocol["source_hashes"][relative], f"Production contract changed: {relative}")
    for relative, expected in protocol["experiment_hashes"].items():
        require(sha256(evidence.read(source_root / "tools" / relative)) == expected,
                f"Frozen benchmark tool changed: {relative}")
    constants = ast.parse(evidence.read(source_root / "trace_ovss/method/trace/constants.py"))
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in constants.body
              if isinstance(node, ast.Assign)}
    require(values["EPS"] == EPS and values["TRACE_VIEWS"] == ("target", "residual", "counterfactual"),
            "Unknown production EPS/view contract")
    engine = ast.parse(evidence.read(source_root / "trace_ovss/method/trace/engine.py"))
    constructor = next(node for node in ast.walk(engine)
                       if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    defaults = {arg.arg: ast.literal_eval(value) for arg, value in
                zip(constructor.args.kwonlyargs, constructor.args.kw_defaults)
                if isinstance(value, ast.Constant)}
    require(defaults["backbone"] == "ViT-L/14", "Unknown token stride; do not guess whole-mask shape")
    config = protocol["config"]
    require(config["resize"] == config["patch_size"] == [224, 224],
            "Only audited one-patch 224x224 panels are supported")
    tokens = (224 // 14) ** 2
    resource = source_root / "trace_ovss/resources/class_names" / PANELS[dataset][1]
    groups = [[part.strip() for part in line.split(",")] for line in
              evidence.read(resource).decode("utf-8").splitlines() if line.strip()]
    require(groups and all(all(group) for group in groups), "Empty vocabulary")
    classes = [group[0] for group in groups]
    classes[0] = "background"
    queries = [query for group in groups for query in group] if config["class_extensions"] else classes
    mapping = [index for index, group in enumerate(groups) for _ in group] if config["class_extensions"] else None
    require(sha256(json.dumps([queries, mapping]).encode()) == protocol["vocabulary_sha256"],
            "Text-query vocabulary seal mismatch")
    candidates = parse_candidates(evidence.csv(directory / "candidate_diagnostics.csv"), len(queries), tokens)
    costs = evidence.csv(directory / "cost.csv")
    grouped = defaultdict(list)
    for item in candidates:
        grouped[item.episode].append(item)
    ids = protocol["image_ids"]
    require(len(ids) == integer(row["images"]) and len(set(ids)) == len(ids), "Invalid image cohort")
    require(len(costs) == math.ceil(len(ids) / config["batch_size"]), "Missing cost episodes")
    total_images = 0
    for index, cost in enumerate(costs):
        for field in ("episode", "images", "update_steps", "candidates", "diagnostic_views",
                      "effective_view_batch_size", "calls", "inputs"):
            cost[field] = integer(cost[field])
        require(cost["episode"] == index, "Cost episode IDs must be unique and contiguous")
        expected_ids = ids[index * config["batch_size"]:(index + 1) * config["batch_size"]]
        items = grouped[index]
        require(cost["images"] == len(expected_ids), "Cost image count mismatch")
        require(cost["candidates"] == len(items), "Candidate/cost counts disagree")
        require(cost["diagnostic_views"] == 3 * len(items), "Full diagnostic view count mismatch")
        require(cost["effective_view_batch_size"] > 0, "Invalid view batch size")
        require(cost["inputs"] == cost["diagnostic_views"] + cost["images"] * (cost["update_steps"] + 2),
                "Observed forward inputs disagree with one-patch episode accounting")
        baseline_calls = math.ceil(3 * len(items) / cost["effective_view_batch_size"])
        require(cost["calls"] == baseline_calls + cost["update_steps"] + 2,
                "Unexpected calls/OOM retries: fixed-batch call model is not valid")
        active = any(item.rho > EPS for item in items)
        _, cast_total = float32_objective_coefficients([item.rho for item in items],
                                                       [item.area for item in items])
        require(not active or cast_total > EPS,
                "Positive terms but float32 normalizer falls back to one; ideal normalization invalid")
        require(cost["update_steps"] == (config["steps"] if active else 0),
                "Update steps disagree with nonempty objective")
        for item in items:
            require(item.patch < len(expected_ids) and expected_ids[item.patch] == item.image_id,
                    "Candidate image/patch/episode identity mismatch")
        for field in ("adaptation_seconds", "inference_seconds", "total_seconds", "peak_allocated_mib"):
            cost[field] = float(cost[field])
            require(math.isfinite(cost[field]) and cost[field] >= 0, "Invalid cost measurement")
        require(math.isclose(cost["total_seconds"], cost["adaptation_seconds"] + cost["inference_seconds"],
                             rel_tol=1e-9, abs_tol=1e-9), "Timing components disagree")
        total_images += cost["images"]
    require(set(grouped) == set(range(len(costs))), "Candidate episode missing from cost")
    warmup = integer(result["warmup_episodes"])
    require(warmup == min(2, len(costs) - 1), "Unexpected warmup exclusion")
    measured = costs[warmup:]
    measured_images = sum(item["images"] for item in measured)
    for key, actual in (
        ("diagnostic_views_per_image", sum(item["diagnostic_views"] for item in measured) / measured_images),
        ("ms_per_image", 1000 * math.fsum(item["total_seconds"] for item in measured) / measured_images)):
        require(math.isclose(float(row[key]), actual, rel_tol=1e-10), f"Summary/cost {key} mismatch")
        require(math.isclose(float(result[key]), actual, rel_tol=1e-10), f"Result/cost {key} mismatch")
    return dict(dataset=dataset, candidates=candidates, grouped=grouped, costs=costs,
                queries=len(queries), tokens=tokens, warmup=warmup, config=config,
                source_row=row, directory=str(directory), images=total_images)


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def episode_analysis(panel, cost, policy, threshold=None):
    items = panel["grouped"][cost["episode"]]
    skipped = policy_skip(items, policy, threshold)
    actual = [item.rho for item in items]
    hypothetical = [item.quality if skip else item.rho for item, skip in zip(items, skipped)]
    distances = coefficient_distance(actual, hypothetical, [item.area for item in items], panel["queries"])
    ideal_a, _ = normalized_coefficients(actual, [item.area for item in items])
    ideal_b, _ = normalized_coefficients(hypothetical, [item.area for item in items])
    cast_a, _ = float32_objective_coefficients(actual, [item.area for item in items])
    cast_b, cast_total_b = float32_objective_coefficients(hypothetical, [item.area for item in items])
    require(distances["after_empty"] or cast_total_b > EPS,
            "Counterfactual float32 normalizer fallback invalidates ideal normalization")
    n, saved = len(items), sum(skipped)
    batch = cost["effective_view_batch_size"]
    original_calls = math.ceil(3 * n / batch)
    two_stage_calls = math.ceil(2 * n / batch) + math.ceil((n - saved) / batch)
    added_steps = panel["config"]["steps"] if distances["empty_to_active"] else 0
    removed_steps = cost["update_steps"] if distances["active_to_empty"] else 0
    net_saved_inputs = saved + cost["images"] * (removed_steps - added_steps)
    delta = math.fsum(after - before for before, after in zip(actual, hypothetical))
    upper = math.fsum(item.bound for item, skip in zip(items, skipped) if skip)
    require(-1e-12 <= delta <= upper + 1e-12, "Episode raw error exceeds bound")
    return dict(dataset=panel["dataset"], episode=cost["episode"], policy=policy,
        epsilon=threshold, measured=cost["episode"] >= panel["warmup"], images=cost["images"],
        candidates=n, skipped_drop_views=saved, baseline_diagnostic_views=cost["diagnostic_views"],
        remaining_diagnostic_views=cost["diagnostic_views"] - saved,
        baseline_all_forward_inputs=cost["inputs"], baseline_all_forward_calls=cost["calls"],
        additional_update_forward_inputs=cost["images"] * added_steps,
        removed_update_forward_inputs=cost["images"] * removed_steps,
        net_all_forward_inputs_saved=net_saved_inputs,
        modeled_remaining_all_forward_inputs=cost["inputs"] - net_saved_inputs,
        modeled_two_stage_diagnostic_calls=two_stage_calls,
        modeled_diagnostic_calls_saved=original_calls - two_stage_calls,
        modeled_all_forward_calls_saved=original_calls - two_stage_calls + removed_steps - added_steps,
        raw_rho_l1=delta, raw_rho_bound_sum=upper,
        raw_rho_max_error=max((a - b for a, b in zip(hypothetical, actual)), default=0),
        float32_reference_coefficient_l1_drift=math.fsum(abs(x-y) for x, y in zip(ideal_a, cast_a)),
        float32_counterfactual_coefficient_l1_drift=math.fsum(abs(x-y) for x, y in zip(ideal_b, cast_b)),
        **distances)


def summarize(rows, scope):
    selected = [row for row in rows if scope == "all_episodes" or row["measured"]]
    require(bool(selected), "No episodes in summary scope")
    first = selected[0]
    additive = ("images", "candidates", "skipped_drop_views", "baseline_diagnostic_views",
        "remaining_diagnostic_views", "baseline_all_forward_inputs", "baseline_all_forward_calls",
        "modeled_two_stage_diagnostic_calls", "modeled_diagnostic_calls_saved",
        "additional_update_forward_inputs", "removed_update_forward_inputs", "net_all_forward_inputs_saved",
        "modeled_remaining_all_forward_inputs", "modeled_all_forward_calls_saved",
        "empty_to_active", "active_to_empty", "candidates_activated", "candidates_deactivated")
    record = dict(dataset=first["dataset"], policy=first["policy"], epsilon=first["epsilon"],
                  scope=scope, episodes=len(selected))
    record.update({key: sum(row[key] for row in selected) for key in additive})
    record.update(drop_skip_fraction=record["skipped_drop_views"] / max(record["candidates"], 1),
        diagnostic_views_saved_fraction=record["skipped_drop_views"] / max(record["baseline_diagnostic_views"], 1),
        all_forward_inputs_saved_fraction=record["net_all_forward_inputs_saved"] / record["baseline_all_forward_inputs"],
        skipped_views_per_image=record["skipped_drop_views"] / record["images"],
        episodes_tv_nonzero=sum(row["tv"] > 1e-12 for row in selected))
    for key in ("coefficient_l1", "coefficient_l1_with_null", "tv", "entropy_loss_bound_nats",
                "raw_rho_l1", "raw_rho_bound_sum", "raw_rho_max_error",
                "float32_reference_coefficient_l1_drift", "float32_counterfactual_coefficient_l1_drift"):
        values = [row[key] for row in selected]
        record.update({f"{key}_mean": math.fsum(values) / len(values),
                       f"{key}_p95": quantile(values, .95), f"{key}_max": max(values)})
    return record


def audit_panel(panel):
    items = panel["candidates"]
    groups = [panel["grouped"][cost["episode"]] for cost in panel["costs"]]
    per_image = defaultdict(list)
    for item in items:
        per_image[item.image_id].append(item)
    whole = [item for item in items if item.whole_mask(panel["tokens"])]
    lower_crossings = [item for item in items if item.rho <= EPS < item.quality]
    return dict(dataset=panel["dataset"], images=panel["images"], episodes=len(groups),
        candidates=len(items), query_classes=panel["queries"], token_grid_size=panel["tokens"],
        zero_area_candidates=sum(item.area == 0 for item in items),
        q_zero=sum(item.quality == 0 for item in items), rho_zero=sum(item.rho == 0 for item in items),
        q_le_eps=sum(item.quality <= EPS for item in items), rho_le_eps=sum(item.rho <= EPS for item in items),
        rho_le_eps_but_q_gt_eps=len(lower_crossings),
        target_exactly_one=sum(item.target == 1 for item in items),
        exact_drop_views=sum(item.exact_drop for item in items),
        actual_D_le_T_oracle_only=sum(item.residual <= item.target for item in items),
        actual_positive_penalty=sum(item.residual > item.target for item in items),
        whole_mask_candidates=len(whole), whole_mask_TC_support_exact_matches=sum(
            item.target == item.counterfactual for item in whole),
        whole_mask_TC_max_abs_difference=max((abs(item.target - item.counterfactual) for item in whole), default=0),
        whole_mask_TC_structural_views_potential=2 * len(whole),
        whole_mask_plus_exact_drop_structural_views_potential=2 * len(whole) + sum(item.exact_drop for item in items),
        single_candidate_episodes=sum(len(group) == 1 for group in groups),
        single_active_candidate_episodes=sum(sum(item.rho > EPS for item in group) == 1 for group in groups),
        certified_singleton_episodes=sum(singleton_certificate(group) is not None for group in groups),
        single_candidate_images_not_normalization_units=sum(len(group) == 1 for group in per_image.values()),
        allzero_episodes=sum(not any(item.rho > EPS for item in group) for group in groups),
        naive_all_drop_allzero_to_active_episodes=sum(not any(item.rho > EPS for item in group)
            and any(item.quality > EPS for item in group) for group in groups),
        max_raw_skip_error=max((item.quality - item.rho for item in items), default=0),
        max_raw_skip_bound=max((item.bound for item in items), default=0),
        cached_baseline_ms_per_image=float(panel["source_row"]["ms_per_image"]),
        cost_comparable_to_refreshed_full=panel["source_row"]["cost_comparable_to_refreshed_full"],
        result_path=panel["source_row"]["result_path"])


def mathematics():
    return dict(
        rho="q*(1-max(D-T,0)); q=min(T,C); finite T,C,D in [0,1]",
        raw_error="0 <= q-rho = q*max(D-T,0) <= q*(1-T); also q*T <= rho <= q",
        exact_drop="After T,C: q<=production EPS means both true rho and replacement q are discarded; T==1 gives rho=q.",
        zero="Since q<=T, rho>=q*T>=q^2: in exact real arithmetic rho==0 iff q==0. rho<=EPS is a different boundary.",
        oracle="Observed D<=T implies no penalty ex post, but D is unavailable before a skipped drop forward. Not a deployable gate.",
        singleton="Weights cancel only per entire adaptation episode, not per image. Exactly one possible active q>EPS and float32(q*T)>EPS certifies its survival and normalizer for every D. Other q<=EPS terms remain inactive.",
        singleton_failure="A raw positive rho<=EPS is discarded. q>EPS alone cannot certify survival. Replacing rho with q may turn an empty/no-update episode into an active one.",
        wholemask="For verified all-one input mask: target=counterfactual=original image, residual=weak image. This saves two view inputs structurally if base logits can be reused; residual is still needed unless independently certified.",
        wholemask_caveat="Input equivalence is not bitwise model-output equivalence across batches, especially fp16/mixed-precision kernels and different batch sizes. Base probabilities are additionally renormalized and sanitized differently. CSV has no base logits/original_support; exact reuse cannot be certified here. No whole-mask savings are included in skip-policy tables.",
        coefficient="Discard rho<=EPS and empty regions, then normalize jointly over the episode; no per-image or per-class renormalization.",
        tv="For two nonempty objectives TV=0.5*coefficient L1. Add a zero-loss dummy term of mass one for an empty objective: TV=0.5*(candidate L1 + abs(active_before-active_after)). Thus empty<->active has TV=1, not 0.5.",
        entropy="At the same logits and nonempty region masks each entropy lies in [0,ln(number of TEXT QUERY classes)]. |L_after-L_before| <= ln(K)*TV. Natural logarithm; NOT label-class count, not entropy of the coefficient vector.",
        normalization_warning="A per-candidate raw error cap does not cap normalized error: small total weight and EPS crossings can amplify it. Actual per-episode distances use measured D.",
        numeric="Main TV/bounds use binary64 ideal normalized coefficients with math.fsum. Separately emulate float32 weight casts and sequential denominator and report coefficient L1 drift. Reject positive-term denominator<=EPS fallback; tests cover it. Numerator/gradient rounding is not included. No bitwise float32/fp16 or loss-trajectory identity claim.",
        unverified_other_reuse="Complementary masks on the same patch can make target(M)=residual(1-M), but the scalar cache stores mask hashes, not mask arrays. No complementary-mask equivalence or savings are inferred from equal areas or support values.",
        call_model="One patch/image and observed fixed view batch verified from cost. Two-stage calls=ceil(2*N/B)+ceil((N-skips)/B). Sequential gates can lose batching efficiency; negative saved-call counts are retained. Empty->active adds configured update steps*images forward inputs, deducted from net savings; active->empty removes the cached update steps. These are count models, not runtime measurements.",
        limitations=["No GPU, model execution, optimizer updates, production edits or queue writes.",
            "No new mIoU, prediction, gradient, optimizer-trajectory or walltime evidence.",
            "Forward input savings are counts, not measured speedup or FLOPs.",
            "Threshold grid fixed by request; no IoU/per-class metrics used for threshold selection.",
            "Candidate identities and masks are held fixed; altered updates could change subsequent trajectories.",
            "Scores are copied only in selected_source_rows.csv, never recomputed.",
            "Only the 512-image Full ablation panels and fresh 500-image CD panel are used; other full-validation panels excluded."])


def write_csv(path, rows):
    require(bool(rows), f"Refusing empty CSV without schema: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compact_summary(summaries):
    return [dict(dataset=row["dataset"], epsilon=row["epsilon"],
        diagnostic_views_saved_pct=100 * row["diagnostic_views_saved_fraction"],
        all_forward_inputs_saved_pct=100 * row["all_forward_inputs_saved_fraction"],
        candidate_activations=row["candidates_activated"], tv_max=row["tv_max"],
        entropy_loss_bound_max_nats=row["entropy_loss_bound_nats_max"],
        status="exact_objective_only" if row["epsilon"] == 0 else "NONEXACT_OFFLINE_NOT_ADOPTED")
        for row in summaries if row["policy"] == "bound_skip" and row["scope"] == "all_episodes"]


def run(summary=DEFAULT_SUMMARY, output=DEFAULT_OUTPUT, package_root=None):
    package_root = Path(package_root or Path(__file__).resolve().parents[1]).resolve()
    summary, output = Path(summary).resolve(), Path(output).resolve()
    require(not output.exists() or not any(output.iterdir()), "Output must be fresh or empty; evidence is never overwritten")
    evidence = Evidence()
    evidence.read(Path(__file__))
    evidence.read(package_root / "tests/test_forward_budget.py")
    selected = select_panels(evidence.csv(summary))
    panels = [load_panel(row, evidence, package_root) for row in selected]
    episodes, summaries, audits, candidate_rows = [], [], [], []
    policies = [("exact_drop", None), ("exact_plus_certified_singleton", None)]
    policies += [("bound_skip", threshold) for threshold in THRESHOLDS]
    policies += [("all_drop_sensitivity", None)]
    for panel in panels:
        audits.append(audit_panel(panel))
        for item in panel["candidates"]:
            candidate_rows.append(dict(dataset=panel["dataset"], episode=item.episode, image_id=item.image_id,
                patch_in_batch=item.patch, class_idx=item.class_idx, area=item.area, mask_sha256=item.mask_sha256,
                T=item.target, C=item.counterfactual, D=item.residual, q=item.quality, rho=item.rho,
                raw_skip_error=item.quality - item.rho, raw_skip_bound=item.bound,
                rho_lower_bound=item.lower, exact_drop=item.exact_drop,
                whole_mask=item.whole_mask(panel["tokens"])))
        for policy, threshold in policies:
            records = [episode_analysis(panel, cost, policy, threshold) for cost in panel["costs"]]
            episodes.extend(records)
            summaries.extend(summarize(records, scope) for scope in ("all_episodes", "warmup_excluded"))
            if policy in {"exact_drop", "exact_plus_certified_singleton"}:
                require(all(row["tv"] < 1e-12 for row in records), "Certified skip changed normalized coefficients")
    evidence.recheck()
    output.mkdir(parents=True, exist_ok=True)
    tables = {"selected_source_rows.csv": selected, "panel_audit.csv": audits,
              "candidate_analysis.csv": candidate_rows, "episode_analysis.csv": episodes,
              "strategy_summary.csv": summaries, "report_summary.csv": compact_summary(summaries)}
    for name, rows in tables.items():
        write_csv(output / name, rows)
    provenance = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        summary_source=str(summary), output_directory=str(output), production_eps=EPS,
        threshold_grid=list(THRESHOLDS), execution="stdlib-only offline cache arithmetic; no torch import",
        inputs=evidence.files, outputs={name: {"sha256": sha256((output / name).read_bytes()), "rows": len(rows)}
                                       for name, rows in tables.items()},
        mathematics=mathematics(), panel_audits=audits,
        decision={"production_adoption": False,
            "approximation_status": "Nonexact offline candidates only; user explicitly defers adoption.",
            "activation_warning": "rho<=EPS<q candidates can reactivate under rho=q. A small observed TV is not exact equivalence. Consult the counts below and the compact table.",
            "candidate_activations_at_epsilon_001": {row["dataset"]: row["candidates_activated"]
                for row in summaries if row["scope"] == "all_episodes" and
                row["policy"] == "bound_skip" and row["epsilon"] == .001},
            "wholemask_TC_reusable_view_inputs": {row["dataset"]: row["whole_mask_TC_structural_views_potential"]
                for row in audits},
            "wholemask": "In general two input-equivalent views per verified whole mask, not three; no bitwise guarantee. Counts are structural opportunities, not measured model-reuse results.",
            "recommendation": "Keep production unchanged. Judge exact savings separately from approximate input savings and fixed-logit TV. Nonzero approximation is not an equivalence, mIoU or walltime result. No threshold selected from accuracy.",
            "compact_table_scope": "All episodes, including warmup. Eight columns; both zero and nonzero thresholds shown."},
        validation="All selected candidate rows, episode counts, cost inputs/calls, source and vocabulary seals checked.",
        normalization_unit="entire adaptation episode; macro episode mean and p95; totals include explicit warmup scopes",
        review="Automated boundary/property tests and separate mathematical self-review; no separate subagent tool available.")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = run(args.summary, args.output, args.package_root)
    print(json.dumps({"output": result["output_directory"], "panels": len(result["panel_audits"]),
                      "files": result["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
