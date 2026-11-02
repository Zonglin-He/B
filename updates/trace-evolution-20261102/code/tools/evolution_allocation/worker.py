"""Five fixed C-allocation ablations. Only score may open annotation bytes.

The controller owns prepared-source admission and the two-dataset barrier.
Prediction, scoring and gates are single-attempt operations, never resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch

from tools.evolution_allocation import common
from tools.evolution import run_cell as shared
from tools.evolution_confirmation import run as confirmation
from tools.evolution_image_full import run as full
from tools.evolution_tuning import worker as tuning
from trace_ovss.method.image_balanced import ImageBalancedTRACE

ROOT, ARMS, DATASETS = common.ROOT, common.ARMS, common.DATASETS
POPULATIONS, ALGORITHM, PERMUTATION_SEED = common.POPULATIONS, common.ALGORITHM, common.PERMUTATION_SEED
require, read_json, write_json = common.require, common.read_json, common.write_json
sha256, canonical_hash = common.sha256, common.canonical_hash
admission, validate_manifest = common.admission, common.validate_manifest
EPS, COUNTERS = full.EPS, full.COUNTERS
make_data = confirmation.make_data
finite_observation, GateReference = full.finite_observation, full.GateReference
TIMING_SCOPE = (
    "Synchronized model-side wallclock: reset/state check, candidate admission, C diagnostic, "
    "allocation preparation/metadata, optimization, final inference/reconstruction/CPU transfer, "
    "finite and operation-count hooks. Excludes model setup, shared RGB decode/H2D/geometry, "
    "receipt serialization/disk writes and offline scoring. Episode 0 excluded only from timings."
)
SCORE_FILES = {"SCORING_STARTED.json", "scores_by_episode.jsonl", "scores_by_image.jsonl",
               "score_inputs.json", "result.json", "SCORED_SEALED.json"}
SHUFFLE_COUNTS = ("active_candidates", "active_sources", "eligible_candidates", "eligible_sources",
                  "moved_candidates", "moved_sources", "changed_score_candidates", "changed_score_sources")


class CostAccumulator:
    """Bounded scalar accounting, including warmup counts and shuffle coverage."""

    def __init__(self):
        self.costs = {a: {k: 0 for k in COUNTERS} for a in ARMS}
        self.measured = {a: dict(episodes=0, images=0, seconds=0.) for a in ARMS}
        self.shuffle_coverage = dict.fromkeys(SHUFFLE_COUNTS, 0)

    def add(self, row, coverage):
        arm = row["arm"]
        for key in COUNTERS:
            self.costs[arm][key] += row[key]
        if row["episode"] != 0:
            group = self.measured[arm]
            group["episodes"] += 1
            group["images"] += row["images"]
            group["seconds"] += row["seconds"]
        if arm == "image_shuffled":
            for key in SHUFFLE_COUNTS:
                require(type(coverage[key]) is int and coverage[key] >= 0, "Invalid shuffle coverage count")
                self.shuffle_coverage[key] += coverage[key]
            require(coverage["active_candidates"] == row["active_candidates"]
                    and coverage["active_sources"] <= row["images"], "Shuffle coverage/cost differs")

    def timings(self):
        result = {}
        for arm, group in self.measured.items():
            episodes, images, seconds = group["episodes"], group["images"], group["seconds"]
            result[arm] = dict(measured_episodes=episodes, measured_images=images,
                seconds=seconds if episodes else None, seconds_per_image=seconds / images if images else None,
                images_per_second=images / seconds if images and seconds > 0 else None,
                warmup_episode=0, accuracy_includes_warmup=True)
        return result


def scope(manifest):
    return dict(algorithm=ALGORITHM, dataset=manifest["dataset"], stage=manifest["stage"],
                seed=manifest["seed"], permutation_seed=manifest["permutation_seed"],
                full_split=manifest["stage"] == "full")


def arm_order(episode):
    require(type(episode) is int and episode >= 0, "Invalid episode")
    offset = episode % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def stable_sources(manifest, meta, start, stop):
    rows = manifest["panel"]["inputs"][start:stop]
    require(list(meta["img_path"]) == [r["image"] for r in rows], "Actual RGB/source order differs")
    keys = {i: f"{manifest['dataset']}/{r['official_id']}" for i, r in enumerate(rows)}
    require(len(set(keys.values())) == len(rows), "Duplicate stable source identity")
    return keys


def source_permutation(size, source_key):
    # Same public, source-keyed permutation protocol as the CPU reference.
    payload = json.dumps(["trace-allocation-v1", PERMUTATION_SEED, source_key],
                         ensure_ascii=True, separators=(",", ":")).encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(payload).digest(), "big"))
    order = list(range(size))
    rng.shuffle(order)
    return order


def coefficient_receipts(stats, arm, source_keys, mapping, *, dtype=torch.float32):
    """Independent receipt calculation; no model forwards or CPU loss offload.

    Portable CPU coefficients use the objective's scalar dtype/addition order.
    Device-rounded observations are separate allocation_metadata fields; ideal
    coefficients are separately named, not substituted for runtime weights.
    """
    require(arm in ARMS and dtype in (torch.float32, torch.float64), "Invalid allocation receipt request")
    groups, records = {}, []
    for index, row in enumerate(stats):
        require(all(type(row[k]) is int and row[k] >= 0 for k in
                    ("image_idx", "batch_idx", "class_idx", "component_idx")), "Noninteger candidate identity")
        image, crop, query = (int(row[k]) for k in ("image_idx", "batch_idx", "class_idx"))
        require(image in source_keys and crop == image and query >= 0
                and (mapping is None or query < len(mapping)), "Unknown source/crop/query")
        region = np.asarray(row["region"])
        require(region.dtype == np.bool_ and region.ndim == 2 and region.size > 0,
                "Candidate region must be native boolean HW")
        area, q = int(region.sum()), float(row["reliability"])
        require(math.isfinite(q) and 0 <= q <= 1 and area > 0, "Invalid native C/area")
        require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) and 0 <= row[k] <= 1
                    for k in ("confidence", "original_support", "counterfactual_support", "quality")),
                "Invalid original C diagnostic support")
        require(row["target_support"] is None and row["diagnostic_algorithm"] == "C", "Non-C diagnostic")
        scalar_keys = ("batch_idx", "image_idx", "class_idx", "confidence", "area", "component_idx",
                       "original_support", "target_support", "counterfactual_support", "quality", "reliability",
                       "residual_support", "dominance", "diagnostic_algorithm", "omitted")
        scalars = {k: row[k].item() if isinstance(row[k], np.generic) else row[k] for k in scalar_keys}
        record = dict(scalars, source_key=source_keys[image], source_image_idx=image,
            token_area=area, native_C=q, assigned_C=q, native_tensor_C=float(torch.tensor(q, dtype=dtype)),
            assigned_tensor_C=float(torch.tensor(q, dtype=dtype)), active=q > EPS,
            assigned_from_candidate=index, ideal_coefficient=0., objective_coefficient=0.,
            allocation_weight=0., allocation_denominator=0., allocation_normalizer=0.,
            preupdate_candidate_category_idx=query if mapping is None else int(mapping[query]),
            region=dict(shape=list(region.shape), encoding="packbits-little-hex",
                        data=np.packbits(region.reshape(-1), bitorder="little").tobytes().hex(), true_pixels=area))
        records.append(record)
        if q > EPS:
            groups.setdefault(image, []).append(index)
    eligible = moved = changed = eligible_sources = moved_sources = changed_sources = 0
    if arm == "image_shuffled":
        for image, indices in groups.items():
            order = source_permutation(len(indices), source_keys[image])
            local_moved = local_changed = 0
            for destination, donor in enumerate(order):
                target, origin = records[indices[destination]], records[indices[donor]]
                target["assigned_C"] = origin["native_C"]
                target["assigned_tensor_C"] = origin["native_tensor_C"]
                target["assigned_from_candidate"] = indices[donor]
                local_moved += destination != donor
                local_changed += target["native_tensor_C"] != origin["native_tensor_C"]
            eligible += len(indices) if len(indices) > 1 else 0
            eligible_sources += len(indices) > 1
            moved += local_moved
            changed += local_changed
            moved_sources += local_moved > 0
            changed_sources += local_changed > 0
    normalization = list(groups.values())
    if arm.startswith("episode_"):
        normalization = [[i for i, r in enumerate(records) if r["active"]]] if groups else []
    for indices in normalization:
        denominator = torch.tensor(0., dtype=dtype)
        ideal_denominator = 0.
        for i in indices:
            r = records[i]
            factor = r["token_area"] if arm.endswith("_token") else 1
            weight = torch.tensor(r["native_C"], dtype=dtype) * factor
            denominator = denominator + weight
            ideal_denominator += r["native_C"] * factor
        require(bool(torch.isfinite(denominator)), "Nonfinite allocation budget")
        normalizer = denominator if float(denominator.double()) > EPS else torch.ones_like(denominator)
        for i in indices:
            r = records[i]
            factor = r["token_area"] if arm.endswith("_token") else 1
            weight = torch.tensor(r["assigned_C"], dtype=dtype) * factor
            divisor = len(normalization) if arm.startswith("image_") else 1
            r["objective_coefficient"] = float((weight / normalizer) / divisor)
            r["ideal_coefficient"] = (r["assigned_C"] * factor / ideal_denominator) / divisor
            r["allocation_weight"] = float(weight)
            r["allocation_denominator"] = float(denominator)
            r["allocation_normalizer"] = float(normalizer)
    coverage = dict(active_candidates=sum(map(len, groups.values())), active_sources=len(groups),
        eligible_candidates=eligible, eligible_sources=eligible_sources, moved_candidates=moved,
        moved_sources=moved_sources, changed_score_candidates=changed, changed_score_sources=changed_sources,
        permutation_seed=PERMUTATION_SEED if arm == "image_shuffled" else None,
        eligible_candidate_fraction=eligible / sum(map(len, groups.values())) if groups else 0.,
        moved_candidate_fraction=moved / eligible if eligible else 0.,
        changed_score_fraction=changed / eligible if eligible else 0.)
    json.dumps(records, allow_nan=False)
    return records, coverage


def native_allocation_metadata(entropy, terms, prepared):
    """Observe the actual inherited cache once, without reevaluating the loss."""
    groups = {}
    for i, term in enumerate(terms):
        if float(term["rho"]) > EPS and term["indices"].numel():
            groups.setdefault(int(term["image_idx"]), []).append(i)
    require(len(groups) == len(prepared), "Native prepared source groups differ")
    tensors, fields = [], []
    prompts = entropy.shape[0] if entropy.ndim == 4 else 1
    for (source, indices), (rows, normalizer) in zip(groups.items(), prepared, strict=True):
        for term_index, (crop, tokens, q) in zip(indices, rows, strict=True):
            coefficient = q / normalizer / len(prepared)
            tensors.append(torch.stack((q, q, q, normalizer, coefficient,
                                        coefficient / tokens.numel() / prompts)))
            fields.append((term_index, source, crop, tokens.numel()))
    values = torch.stack(tensors).detach().cpu().tolist() if tensors else []
    records = []
    for (index, source, crop, area), numbers in zip(fields, values, strict=True):
        require(all(math.isfinite(v) for v in numbers), "Nonfinite native allocation metadata")
        records.append(dict(term_index=index, donor_term_index=index, source=source, source_key=None,
            crop=crop, area=area, diagnostic_q=float(terms[index]["rho"]),
            **dict(zip(("original_q", "assigned_q", "weight", "normalizer", "coefficient", "token_coefficient"), numbers))))
    return token_accounting_metadata(entropy, dict(control="image_region", seed=None, entropy_shape=list(entropy.shape), dtype=str(entropy.dtype),
        device=str(entropy.device), active_candidates=len(records), active_sources=len(groups),
        normalization_groups=len(prepared), candidates=records, coverage=None))


def accounting_failure(message, detail):
    detail = {k: repr(v) if isinstance(v, float) and not math.isfinite(v) else v
              for k, v in detail.items()}
    error = RuntimeError(message + ": " + json.dumps(detail, allow_nan=False))
    error.worker_context = dict(token_accounting=detail)
    raise error


def token_accounting_metadata(entropy, metadata):
    """Keep device observations distinct from portable scalar accounting.

    These are algebraic coefficients of a regional mean, not measured autograd
    entries. Reconstruct on the same device from exported coefficient bits;
    CPU reconstruction is recorded separately, never substituted for the loss.
    """
    require(metadata["device"] == str(entropy.device) and metadata["dtype"] == str(entropy.dtype)
            and metadata["entropy_shape"] == list(entropy.shape), "Token accounting device/layout differs")
    records = metadata["candidates"]
    prompts = entropy.shape[0] if entropy.ndim == 4 else 1
    groups = metadata["normalization_groups"]
    require(type(groups) is int and groups >= bool(records), "Invalid accounting group count")
    scalars = torch.tensor([[r["weight"], r["normalizer"], r["coefficient"]] for r in records],
                          dtype=entropy.dtype, device=entropy.device)
    references = torch.stack([torch.stack((values[0] / values[1] / groups,
        values[2] / r["area"] / prompts))
        for values, r in zip(scalars.unbind(), records, strict=True)]).detach().cpu().tolist() if records else []
    metadata["token_accounting"] = "same_device_exact_and_portable_v2"
    for record, (coefficient_reference, reference) in zip(records, references, strict=True):
        portable_coefficient = float(torch.tensor(record["weight"], dtype=entropy.dtype)
                                     / torch.tensor(record["normalizer"], dtype=entropy.dtype) / groups)
        portable = float(torch.tensor(portable_coefficient, dtype=entropy.dtype) / record["area"] / prompts)
        observed = record["token_coefficient"]
        detail = dict(term_index=record["term_index"], device=str(entropy.device), dtype=str(entropy.dtype),
            coefficient=record["coefficient"], area=record["area"], prompts=prompts,
            same_device_coefficient=coefficient_reference, portable_coefficient=portable_coefficient,
            coefficient_device_minus_portable=record["coefficient"] - portable_coefficient,
            observed_token_coefficient=observed, same_device_token_coefficient=reference,
            portable_token_coefficient=portable, device_minus_portable=observed - portable)
        if (not all(math.isfinite(v) and v >= 0 for v in (record["coefficient"], coefficient_reference,
                                                        portable_coefficient, observed, reference, portable))
                or record["coefficient"] != coefficient_reference or observed != reference):
            accounting_failure("Same-device token coefficient differs", dict(detail, groups=groups))
        record.update(token_coefficient_reference=reference, portable_token_coefficient=portable,
                      token_device_minus_portable=observed - portable,
                      coefficient_reference=coefficient_reference, portable_coefficient=portable_coefficient,
                      coefficient_device_minus_portable=record["coefficient"] - portable_coefficient)
    return metadata


def validate_allocation_metadata(metadata, candidates, arm, dtype):
    """Cross-check runtime coefficients, rather than merely describing ideal C."""
    active = [i for i, r in enumerate(candidates) if r["active"]]
    if not active:
        require(metadata is None, "Empty episode unexpectedly prepared an objective")
        return
    require(isinstance(metadata, dict) and metadata["control"] == arm and metadata["dtype"] == str(dtype)
            and metadata.get("token_accounting") == "same_device_exact_and_portable_v2"
            and (metadata.get("device") == "cpu" or metadata.get("device") == "cuda:0")
            and metadata["seed"] == (PERMUTATION_SEED if arm == "image_shuffled" else None)
            and metadata["active_candidates"] == len(active), "Actual allocation metadata scope differs")
    groups = {candidates[i]["image_idx"] for i in active}
    require(metadata["active_sources"] == len(groups)
            and metadata["normalization_groups"] == (len(groups) if arm.startswith("image_") else 1),
            "Actual normalization groups differ")
    seen = set()
    shape = metadata["entropy_shape"]
    require(len(shape) in (3, 4) and all(type(v) is int and v > 0 for v in shape), "Invalid entropy metadata shape")
    prompts = shape[0] if len(shape) == 4 else 1
    for record in metadata["candidates"]:
        term = record["term_index"]
        require(type(term) is int and 0 <= term < len(active) and term not in seen, "Actual allocation term identity differs")
        seen.add(term)
        row = candidates[active[term]]
        donor = record["donor_term_index"]
        require(type(donor) is int and 0 <= donor < len(active)
                and active[donor] == row["assigned_from_candidate"]
                and record["source"] == row["image_idx"] and record["crop"] == row["batch_idx"]
                and record["area"] == row["token_area"] and record["diagnostic_q"] == row["native_C"]
                and record["original_q"] == row["native_tensor_C"] and record["assigned_q"] == row["assigned_tensor_C"]
                and record["weight"] == row["allocation_weight"] and record["normalizer"] == row["allocation_normalizer"]
                and record["source_key"] == (row["source_key"] if arm == "image_shuffled" else None),
                "Actual runtime coefficient/source/score differs from independent accounting")
        token_coefficient = float(torch.tensor(row["objective_coefficient"], dtype=dtype) / row["token_area"] / prompts)
        observed, reference = record["token_coefficient"], record["token_coefficient_reference"]
        valid_accounting = (all(type(v) is float and math.isfinite(v) and v >= 0
                    for v in (observed, reference, record["portable_token_coefficient"], record["coefficient"],
                              record["coefficient_reference"], record["portable_coefficient"]))
                and record["portable_coefficient"] == row["objective_coefficient"]
                and record["portable_token_coefficient"] == token_coefficient
                and record["coefficient"] == record["coefficient_reference"]
                and record["coefficient_device_minus_portable"] == record["coefficient"] - row["objective_coefficient"]
                and observed == reference
                and record["token_device_minus_portable"] == observed - token_coefficient
                and (metadata["device"] != "cpu" or (observed == token_coefficient
                    and record["coefficient"] == row["objective_coefficient"])))
        if not valid_accounting:
            accounting_failure("Token accounting differs", dict(term_index=term, area=row["token_area"],
                prompts=prompts, groups=metadata["normalization_groups"], device=metadata["device"], dtype=str(dtype),
                observed_token_coefficient=observed, same_device_token_coefficient=reference,
                stored_portable_token_coefficient=record["portable_token_coefficient"],
                recomputed_portable_token_coefficient=token_coefficient,
                stored_token_delta=record["token_device_minus_portable"],
                recomputed_token_delta=observed - token_coefficient,
                observed_coefficient=record["coefficient"], same_device_coefficient=record["coefficient_reference"],
                stored_portable_coefficient=record["portable_coefficient"],
                recomputed_portable_coefficient=row["objective_coefficient"],
                stored_coefficient_delta=record["coefficient_device_minus_portable"],
                recomputed_coefficient_delta=record["coefficient"] - row["objective_coefficient"]))
    require(seen == set(range(len(active))), "Missing actual allocation terms")
    if arm == "image_shuffled":
        coverage = metadata["coverage"]
        expected = dict(active_candidates=len(active), active_sources=len(groups),
            exchange_eligible_candidates=sum(sum(candidates[i]["image_idx"] == s for i in active)
                for s in groups if sum(candidates[i]["image_idx"] == s for i in active) > 1),
            exchange_eligible_sources=sum(sum(candidates[i]["image_idx"] == s for i in active) > 1 for s in groups),
            actually_moved_candidates=sum(candidates[i]["assigned_from_candidate"] != i for i in active),
            actually_moved_sources=len({candidates[i]["image_idx"] for i in active if candidates[i]["assigned_from_candidate"] != i}),
            score_changed_candidates=sum(candidates[i]["native_tensor_C"] != candidates[i]["assigned_tensor_C"] for i in active),
            score_changed_sources=len({candidates[i]["image_idx"] for i in active if candidates[i]["native_tensor_C"] != candidates[i]["assigned_tensor_C"]}))
        require(coverage == expected, "Actual runtime shuffle coverage differs")
    else:
        require(metadata["coverage"] is None, "Unexpected non-shuffle runtime coverage")


def diagnostic_identity(candidates):
    """Bind original diagnostics AND activity, excluding allocation assignments."""
    fields = ("batch_idx", "image_idx", "class_idx", "confidence", "original_support",
              "counterfactual_support", "target_support", "reliability", "quality", "area",
              "component_idx", "diagnostic_algorithm", "region", "source_key", "source_image_idx",
              "token_area", "native_C", "active", "preupdate_candidate_category_idx")
    return canonical_hash([{k: row[k] for k in fields} for row in candidates])


class ExperimentTRACE(ImageBalancedTRACE):
    """Fixed C estimator, unchanged native arm, four allocation-only controls."""

    def __init__(self, **kwargs):
        require(kwargs.pop("view_mode", "C") == "C", "Only fixed C allowed")
        self.arm, self.source_keys = "image_region", {}
        self.allocation_dtype = torch.float32
        self.allocation_metadata = None
        super().__init__(view_mode="C", **kwargs)

    def bind_sources(self, source_keys):
        require(isinstance(source_keys, dict) and source_keys
                and all(type(i) is int and i >= 0 and isinstance(k, str) and k for i, k in source_keys.items())
                and len(set(source_keys.values())) == len(source_keys), "Invalid stable source keys")
        self.source_keys = dict(source_keys)
        self.allocation_metadata = None

    def prepare_trace_objective(self, entropy, terms):
        from trace_ovss.method import allocation_runtime as runtime
        require(self.arm in ARMS and self.source_keys, "Unbound allocation arm/source identity")
        self.allocation_dtype = entropy.dtype
        if self.arm == "image_region":
            prepared = ImageBalancedTRACE.prepare_trace_objective(entropy, terms)
            self.allocation_metadata = native_allocation_metadata(entropy, terms, prepared)
            return prepared
        options = {}
        if self.arm == "image_shuffled":
            active = {int(t["image_idx"]) for t in terms if float(t["rho"]) > EPS and t["indices"].numel()}
            options = dict(seed=PERMUTATION_SEED, source_keys={i: self.source_keys[i] for i in active})
        prepared = runtime.prepare_allocation_control(entropy, terms, control=self.arm, **options)
        self.allocation_metadata = token_accounting_metadata(entropy, runtime.allocation_control_metadata(prepared))
        return prepared

    def trace_region_normalized_entropy(self, entropy, terms, *, prepared=None):
        if self.arm == "image_region":
            return ImageBalancedTRACE.trace_region_normalized_entropy(entropy, terms, prepared=prepared)
        from trace_ovss.method import allocation_runtime as runtime
        return runtime.allocation_control_entropy(entropy, terms, prepared=prepared)

    _count = confirmation.prior.StandaloneTRACE._count
    count_work = confirmation.prior.StandaloneTRACE.count_work
    _forward_trace_view_batches = confirmation.prior.StandaloneTRACE._forward_trace_view_batches

    def forward_eval_logits(self, images, *args, **kwargs):
        if hasattr(self, "work"):
            self._count(("diagnostic" if getattr(self, "_diagnostic", False) else "eval") + "_forward", len(images))
        return super().forward_eval_logits(images, *args, **kwargs)

    def forward_adapt_logits(self, images):
        if hasattr(self, "work"):
            self._count("adapt_forward", len(images))
        return super().forward_adapt_logits(images)


def validate_cost(row, arm, cfg):
    require(arm in ARMS and row["arm"] == arm, "Unknown allocation cost arm")
    for key in ("images", "patches", "candidate_count", "active_candidates", "steps", *COUNTERS):
        require(type(row[key]) is int and row[key] >= 0, "Noninteger/negative actual cost")
    require(0 < row["images"] <= cfg["batch_size"] and row["patches"] == row["images"]
            and 0 <= row["active_candidates"] <= row["candidate_count"] <= row["images"] * cfg["trace_max_regions"]
            and type(row["seconds"]) in (int, float) and math.isfinite(row["seconds"])
            and row["seconds"] >= 0, "Invalid finite cost/budget")
    steps = cfg["steps"] if row["active_candidates"] else 0
    views = row["candidate_count"]
    calls = (views + cfg["trace_view_batch_size"] - 1) // cfg["trace_view_batch_size"]
    expected = dict(eval_forward_calls=2, eval_forward_inputs=2 * row["patches"],
        adapt_forward_calls=steps, adapt_forward_inputs=steps * row["patches"],
        diagnostic_forward_calls=calls, diagnostic_forward_inputs=views,
        diagnostic_encoder_calls=calls, diagnostic_encoder_inputs=views,
        encoder_calls=2 + steps + calls, encoder_inputs=(2 + steps) * row["patches"] + views)
    require(row["steps"] == steps and all(row[k] == v for k, v in expected.items()), "Actual C-only schedule differs")


def run_policy(method, arm, images, valid, ids, meta, cfg, mapping, initial, source_keys):
    if isinstance(method, ExperimentTRACE):
        method.arm = arm
        method.bind_sources(source_keys)
    tick, counts = time.perf_counter(), {}
    try:
        with finite_observation(method), method.count_work() as counts:
            shared.synchronize(method.device)
            tick = time.perf_counter()
            method.reset()
            require(shared.restored_state(method) == initial, "Pre-arm reset drift")
            losses = method.adapt(images, valid_mask=valid, image_ids=ids)
            predictions = shared.label_predictions(method, images, meta, cfg, mapping)
            shared.synchronize(method.device)
            seconds = time.perf_counter() - tick
        require(all(math.isfinite(v) for v in losses), "Nonfinite adaptation loss")
        stats = method.last_candidate_stats
        candidates, coverage = coefficient_receipts(stats, arm, source_keys, mapping,
            dtype=getattr(method, "allocation_dtype", torch.float32))
        if isinstance(method, ExperimentTRACE):
            validate_allocation_metadata(method.allocation_metadata, candidates, arm, method.allocation_dtype)
        row = dict(arm=arm, images=len(meta["img_path"]), patches=len(images), seconds=seconds,
            steps=len(losses), candidate_count=len(stats), active_candidates=sum(r["active"] for r in candidates),
            candidate_sha256=diagnostic_identity(candidates), **counts)
        validate_cost(row, arm, cfg)
        return losses, predictions, candidates, coverage, row
    except BaseException as error:
        error.worker_context = dict(getattr(error, "worker_context", {}), arm=arm, images=len(meta["img_path"]), patches=len(images),
            seconds_until_failure=time.perf_counter() - tick, actual_partial_counters=dict(counts))
        raise


def artifact_names(manifest):
    n, batch = len(manifest["panel"]["inputs"]), manifest["config"]["batch_size"]
    return ({f"predictions/{a}/{i:06d}.npy" for a in ARMS for i in range(n)}
            | {f"episodes/{e:06d}.{a}.json" for a in ARMS for e in range((n + batch - 1) // batch)}
            | {"STARTED.json", "prediction_protocol.json", "initial_state.json", "prediction_result.json", "cost.jsonl"})


def validate_prediction(prediction, classes, shape):
    require(prediction.ndim == 2 and prediction.shape == tuple(shape) and prediction.size > 0
            and np.issubdtype(prediction.dtype, np.integer) and (prediction >= 0).all()
            and (prediction < classes).all(), "Invalid prediction geometry/category")


def execute_predictions(method, data, manifest, output):
    from trace_ovss.cli import build_patch_image_ids, build_patch_valid_mask
    cfg, panel = manifest["config"], manifest["panel"]
    require(len(data) == len(panel["inputs"]), "RGB data population differs")
    for arm in ARMS:
        (output / "predictions" / arm).mkdir(parents=True)
    (output / "episodes").mkdir()
    method.reset()
    initial = shared.restored_state(method)
    write_json(output / "initial_state.json", initial)
    frozen = [(p, p._version) for p in method.model.parameters() if not p.requires_grad]
    account = CostAccumulator()
    with (output / "cost.jsonl").open("x") as stream:
        for episode, start in enumerate(range(0, len(data), cfg["batch_size"])):
            stop = min(start + cfg["batch_size"], len(data))
            batch = shared.image_collate([data[i] for i in range(start, stop)])
            images, meta = batch["img_patches"].to(method.device), batch["meta"]
            sources = stable_sources(manifest, meta, start, stop)
            valid = build_patch_valid_mask(meta, cfg["patch_size"], cfg["patch_stride"], method.device)
            ids = build_patch_image_ids(meta, method.device)
            rng, identity = shared.rng_state(method.device), None
            for position, arm in enumerate(arm_order(episode)):
                shared.restore_rng(rng, method.device)
                try:
                    losses, predictions, candidates, coverage, row = run_policy(method, arm, images,
                        valid, ids, meta, cfg, panel["query_to_class"], initial, sources)
                except BaseException as error:
                    error.worker_context = dict(getattr(error, "worker_context", {}), episode=episode,
                        order_position=position, ordered_ids=panel["ordered_ids"][start:stop], **scope(manifest))
                    raise
                require(all(p._version == version for p, version in frozen), "Frozen weights changed")
                digest = row["candidate_sha256"]
                identity = digest if identity is None else identity
                require(identity == digest, "Native C/activity/diagnostic support differs across allocation arms")
                require(len(predictions) == stop - start, "Prediction count differs")
                row.update(episode=episode, order_position=position, warmup=episode == 0, **scope(manifest))
                for index, prediction in zip(range(start, stop), predictions, strict=True):
                    validate_prediction(prediction, len(panel["classes"]), cfg["resize"])
                    with (output / f"predictions/{arm}/{index:06d}.npy").open("xb") as destination:
                        np.save(destination, prediction, allow_pickle=False)
                write_json(output / f"episodes/{episode:06d}.{arm}.json", dict(row, losses=losses,
                    ordered_ids=panel["ordered_ids"][start:stop], candidates=candidates, coverage=coverage,
                    allocation_dtype=str(method.allocation_dtype), allocation_metadata=method.allocation_metadata,
                    gt_contents_opened=False))
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                account.add(row, coverage)
                method.reset()
                require(shared.restored_state(method) == initial, "Post-arm reset drift")
    return account.timings(), account.costs, account.shuffle_coverage


def build_method(manifest):
    cfg = manifest["config"]
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    return ExperimentTRACE(classes=manifest["panel"]["queries"], lr=cfg["lr"], steps=cfg["steps"],
        trace_max_regions=cfg["trace_max_regions"], trace_view_batch_size=cfg["trace_view_batch_size"],
        vision_outputs=confirmation.VISION_OUTPUTS, device="cuda", backbone=manifest["checkpoint"]["path"])


def started(output, manifest_sha, phase):
    output.parent.mkdir(parents=True, exist_ok=True)
    # The external claim survives even deletion/interruption of the output tree.
    claim = confirmation.claim_gate_attempt(output, manifest_sha)
    output.mkdir(exist_ok=False)
    write_json(output / "STARTED.json", dict(phase=phase, manifest_sha256=manifest_sha,
        attempt_sha256=sha256(claim), retry_allowed=False, gt_contents_opened=False))
    return claim


def verify_gates(manifest):
    """Direct worker entry cannot bypass the controller's two real-model gates."""
    suite = Path(manifest["suite_root"])
    boundary = read_json(suite / "ALL_GATES_PASSED.json")
    require(boundary["complete"] is True and boundary["suite_prepared_sha256"] == manifest["suite_prepared_sha256"]
            and set(boundary["gates"]) == set(DATASETS), "Both authenticated gates required")
    for dataset in DATASETS:
        output = suite / "gates" / dataset
        path = common.manifest_path(suite, dataset, "gate")
        expected = read_json(path)
        validate_manifest(expected)
        guard_output_labels(output, expected)
        seal_path = output / "GATE_PASSED.json"
        require(sha256(seal_path) == boundary["gates"][dataset], "Gate certificate drift")
        certificate, report = read_json(seal_path), read_json(output / "gate.json")
        require(certificate == dict(report, gate_sha256=sha256(output / "gate.json"))
                and report["passed"] is True and report["images"] == 4 and report["arms"] == ARMS
                and report["gt_contents_opened"] is False and report["manifest"] == expected
                and report["manifest_path"] == str(path) and report["manifest_sha256"] == sha256(path)
                and report["output"] == str(output) and report["config"] == expected["config"]
                and report["suite_prepared_sha256"] == manifest["suite_prepared_sha256"]
                and report["checkpoint"] == manifest["checkpoint"] and report["source_sha256"] == manifest["source_sha256"]
                and report["ordered_ids"] == expected["panel"]["ordered_ids"]
                and all(report[k] == v for k, v in scope(expected).items()), "Gate source/config/input binding differs")
        claim = output.with_name(output.name + ".STARTED.json")
        attempt = read_json(output / "STARTED.json")
        require(report["attempt_sha256"] == attempt["attempt_sha256"] == sha256(claim)
                and attempt["phase"] == "gate" and attempt["manifest_sha256"] == sha256(path)
                and attempt["retry_allowed"] is False and attempt["gt_contents_opened"] is False
                and read_json(claim)["manifest_sha256"] == sha256(path), "Gate attempt identity differs")
        require(sum(report["batch_images"]) == 4 and len(report["comparisons"]) == len(report["batch_images"])
                and bool(report["comparisons"]), "Gate RGB coverage differs")
        for comparison, images in zip(report["comparisons"], report["batch_images"], strict=True):
            require(comparison["candidate_masks_equal"] is True and comparison["reset_equal"] is True
                    and comparison["changed_label_pixels"] == 0 and comparison["candidates"] > 0
                    and all(type(comparison[k]) in (int, float) and math.isfinite(comparison[k])
                        and 0 <= comparison[k] <= 1e-5 for k in ("max_support_abs_difference",
                            "max_coefficient_abs_difference", "max_loss_abs_difference",
                            "max_gradient_abs_difference", "max_parameter_abs_difference")), "Native C gate threshold differs")
            actual = comparison["actual_arm_counts"]
            require(set(actual) == {"direct_C", *ARMS}
                    and len({r["candidate_sha256"] for r in actual.values()}) == 1,
                    "Gate five-arm diagnostic coverage differs")
            for arm, row in actual.items():
                validate_cost(row, "image_region" if arm == "direct_C" else arm, expected["config"])
                require(row["images"] == images and row["active_candidates"] > 0
                        and len(row["losses"]) == expected["config"]["steps"]
                        and all(type(v) in (int, float) and math.isfinite(v) for v in row["losses"]), "Gate finite update missing")
    return boundary["gates"]


def predict(args):
    manifest, output = admission(args)
    require(manifest["stage"] == "full", "Full manifest required for predict")
    common.require_gpu_envelope(manifest)
    started(output, args.manifest_sha256, "predict")
    try:
        gates = verify_gates(manifest)
        data = verify_inputs(manifest)
        with confirmation.prior.gpu_lock() as lock:
            runtime = shared.configure_deterministic_runtime()
            write_json(output / "prediction_protocol.json", dict(manifest=manifest, manifest_path=str(args.manifest),
                manifest_sha256=args.manifest_sha256, output=str(output), gpu_lock=lock,
                deterministic_runtime=runtime, gate_seal_sha256=gates,
                timing_scope=TIMING_SCOPE, gt_contents_opened=False))
            method = build_method(manifest)
            timings, costs, shuffle_coverage = execute_predictions(method, data, manifest, output)
            admission(args)
            verify_inputs(manifest)
            require(shared.deterministic_runtime_state() == runtime, "Deterministic runtime drift")
            write_json(output / "prediction_result.json", dict(status="predicted", images=len(data), arms=ARMS,
                config=manifest["config"], timings=timings, costs=costs, shuffle_coverage=shuffle_coverage,
                timing_scope=TIMING_SCOPE, **scope(manifest)))
            names = artifact_names(manifest)
            require(inventory(output) == names, "Unexpected/incomplete prediction inventory")
            guard_output_labels(output, manifest)
            write_json(output / "PREDICTIONS_SEALED.json", dict(complete=True, gt_contents_opened=False,
                output=str(output), manifest_sha256=args.manifest_sha256, images=len(data), arms=ARMS,
                suite_prepared_sha256=manifest["suite_prepared_sha256"], **scope(manifest),
                artifact_sha256={name: sha256(output / name) for name in sorted(names)}))
    except BaseException as error:
        write_json(output / "PREDICTION_FAILED.json", tuning.failure_receipt(error, "predict", args.manifest_sha256))
        raise


def inventory(output):
    paths = list(output.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "Aliased payload directory/file")
    return {str(p.relative_to(output)) for p in paths if p.is_file()}


def guard_output_labels(output, manifest):
    """Stat-only: authenticating a hardlinked payload must not read GT bytes."""
    labels = common.annotation_inodes(manifest)
    paths = list(output.rglob("*")) + [output.with_name(output.name + ".STARTED.json")]
    for path in paths:
        if path.is_file():
            status = path.stat()
            require((status.st_dev, status.st_ino) not in labels, "Output payload aliases annotation")


def verify_inputs(manifest):
    common.guard_global_inputs(manifest)
    return tuning.verify_inputs(manifest)


def verify_prediction_seal(output, expected_manifest=None, expected_sha=None, *, hash_payload=True):
    output = tuning.absolute_path(str(output))
    bound_manifest = expected_manifest
    if bound_manifest is None:
        bound_manifest = read_json(common.manifest_path(output.parent.parent, output.name, "full"))
    validate_manifest(bound_manifest)
    guard_output_labels(output, bound_manifest)
    protocol = read_json(output / "prediction_protocol.json")
    manifest, seal = protocol["manifest"], read_json(output / "PREDICTIONS_SEALED.json")
    validate_manifest(manifest)
    require(manifest == bound_manifest, "Canonical/sealed manifest differs")
    require(manifest["stage"] == "full", "Only full prediction seals admitted")
    path = tuning.absolute_path(protocol["manifest_path"])
    require(path.is_relative_to(Path(manifest["suite_root"])) and output.is_relative_to(Path(manifest["suite_root"]))
            and sha256(path) == protocol["manifest_sha256"] == seal["manifest_sha256"]
            and read_json(path) == manifest, "Prediction manifest/source binding differs")
    require(protocol["output"] == seal["output"] == manifest["output"] == str(output), "Output binding differs")
    require(protocol["gate_seal_sha256"] == verify_gates(manifest), "Prediction gate binding differs")
    if expected_manifest is not None:
        require(manifest == expected_manifest, "CLI/sealed manifest differs")
    if expected_sha is not None:
        require(protocol["manifest_sha256"] == expected_sha, "CLI/sealed manifest hash differs")
    require(seal["complete"] is True and seal["gt_contents_opened"] is False
            and protocol["gt_contents_opened"] is False and seal["images"] == len(manifest["panel"]["inputs"])
            and seal["arms"] == ARMS and protocol["timing_scope"] == TIMING_SCOPE
            and seal["suite_prepared_sha256"] == manifest["suite_prepared_sha256"]
            and all(seal[k] == v for k, v in scope(manifest).items()), "Prediction seal scope differs")
    names, hashes = artifact_names(manifest), seal["artifact_sha256"]
    require(set(hashes) == names, "Prediction payload set differs")
    actual = inventory(output)
    require(names | {"PREDICTIONS_SEALED.json"} <= actual
            and actual <= names | SCORE_FILES | {"PREDICTIONS_SEALED.json"}, "Unexpected/missing prediction payload")
    for name in names:
        require(tuning.digest_valid(hashes[name]), "Invalid payload hash")
        if hash_payload:
            require(sha256(output / name) == hashes[name], "Prediction payload drift: " + name)
    start = read_json(output / "STARTED.json")
    claim = output.with_name(output.name + ".STARTED.json")
    require(start["phase"] == "predict" and start["manifest_sha256"] == protocol["manifest_sha256"]
            and start["attempt_sha256"] == sha256(claim) and start["retry_allowed"] is False
            and start["gt_contents_opened"] is False
            and read_json(claim)["manifest_sha256"] == protocol["manifest_sha256"], "Persistent attempt binding differs")
    return manifest


def unpack_candidate(record):
    region = record["region"]
    require(region["encoding"] == "packbits-little-hex" and isinstance(region["shape"], list)
            and len(region["shape"]) == 2 and all(type(v) is int and 0 < v <= 224 for v in region["shape"]),
            "Invalid candidate mask geometry")
    size = math.prod(region["shape"])
    payload = bytes.fromhex(region["data"])
    require(len(payload) == (size + 7) // 8, "Invalid packed candidate length")
    bits = np.unpackbits(np.frombuffer(payload, np.uint8), bitorder="little")
    require(not bits[size:].any(), "Noncanonical mask padding")
    mask = bits[:size].reshape(region["shape"]).astype(bool)
    require(int(mask.sum()) == region["true_pixels"] == record["token_area"], "Candidate area differs")
    return dict(record, region=mask)


def validate_episode_receipt(receipt, row, manifest, start, stop):
    require(all(receipt[k] == v for k, v in row.items()) and receipt["gt_contents_opened"] is False
            and receipt["ordered_ids"] == manifest["panel"]["ordered_ids"][start:stop], "Episode receipt scope differs")
    require(isinstance(receipt["losses"], list) and len(receipt["losses"]) == row["steps"]
            and all(type(v) in (int, float) and math.isfinite(v) for v in receipt["losses"]), "Invalid episode losses")
    records = receipt["candidates"]
    require(all(type(r["active"]) is bool and all(type(r[k]) is float and math.isfinite(r[k]) for k in
                ("native_C", "assigned_C", "native_tensor_C", "assigned_tensor_C", "ideal_coefficient",
                 "objective_coefficient", "allocation_weight", "allocation_denominator", "allocation_normalizer"))
                for r in records), "Invalid candidate scalar receipt types")
    require(len(records) == row["candidate_count"] and sum(r["active"] is True for r in records) == row["active_candidates"],
            "Candidate/activity count differs")
    sources = {i: f"{manifest['dataset']}/{r['official_id']}" for i, r in enumerate(manifest["panel"]["inputs"][start:stop])}
    stats = [unpack_candidate(r) for r in records]
    dtype = {"torch.float32": torch.float32, "torch.float64": torch.float64}.get(receipt["allocation_dtype"])
    require(dtype is not None, "Invalid allocation scalar dtype")
    expected, coverage = coefficient_receipts(stats, row["arm"], sources, manifest["panel"]["query_to_class"], dtype=dtype)
    require(records == expected and receipt["coverage"] == coverage, "Coefficient/source/permutation receipts differ")
    validate_allocation_metadata(receipt["allocation_metadata"], records, row["arm"], dtype)
    require(diagnostic_identity(records) == row["candidate_sha256"], "Native diagnostic identity differs")


def validate_cost_stream(output, manifest):
    output = Path(output)
    cfg, n = manifest["config"], len(manifest["panel"]["inputs"])
    episodes = (n + cfg["batch_size"] - 1) // cfg["batch_size"]
    rows = iter(full.bounded_json_lines(output / "cost.jsonl", episodes * len(ARMS)))
    account = CostAccumulator()
    for episode in range(episodes):
        group = [next(rows, None) for _ in ARMS]
        require(all(row is not None for row in group), "Incomplete actual costs")
        require([r["arm"] for r in group] == arm_order(episode)
                and len({r["candidate_sha256"] for r in group}) == 1, "Cross-arm order/native diagnostics differ")
        start, stop = episode * cfg["batch_size"], min((episode + 1) * cfg["batch_size"], n)
        for position, row in enumerate(group):
            require(type(row["episode"]) is int and type(row["order_position"]) is int
                    and row["episode"] == episode and row["order_position"] == position and row["warmup"] is (episode == 0)
                    and row["images"] == stop - start and tuning.digest_valid(row["candidate_sha256"])
                    and all(row[k] == v for k, v in scope(manifest).items()), "Cost scope differs")
            validate_cost(row, row["arm"], cfg)
            receipt = read_json(output / f"episodes/{episode:06d}.{row['arm']}.json")
            validate_episode_receipt(receipt, row, manifest, start, stop)
            account.add(row, receipt["coverage"])
    require(next(rows, None) is None, "Extra actual costs")
    predicted = read_json(output / "prediction_result.json")
    require(predicted["status"] == "predicted" and predicted["images"] == n and predicted["arms"] == ARMS
            and predicted["config"] == cfg and predicted["costs"] == account.costs and predicted["timing_scope"] == TIMING_SCOPE
            and predicted["timings"] == account.timings() and predicted["shuffle_coverage"] == account.shuffle_coverage
            and all(predicted[k] == v for k, v in scope(manifest).items()), "Aggregate measured costs/timings differ")
    return predicted


def ten_arm_scope():
    return [dict(dataset=d, arm=a, images=POPULATIONS[d]) for d in DATASETS for a in ARMS]


def pooled_result(counts):
    intersection, union = counts
    return dict(miou_unrounded=100 * math.fsum(int(i) / int(u) for i, u in zip(intersection, union) if u)
        / int((union > 0).sum()) if (union > 0).any() else None,
        intersection=intersection.tolist(), union=union.tolist())


def score(args):
    manifest, output = admission(args)
    require(manifest["stage"] == "full", "Gate cannot score")
    require(not any((output / name).exists() for name in SCORE_FILES), "No score retry/overwrite")
    verify_inputs(manifest)  # RGB/vocabulary/stat only.
    barrier_sha = common.verify_barrier(args.barrier, manifest, output, args.manifest_sha256)
    require(tuning.digest_valid(barrier_sha), "Barrier verifier must return authenticated SHA256")
    verify_prediction_seal(output, manifest, args.manifest_sha256)
    predicted = validate_cost_stream(output, manifest)
    prediction_sha = sha256(output / "PREDICTIONS_SEALED.json")
    write_json(output / "SCORING_STARTED.json", dict(manifest_sha256=args.manifest_sha256,
        prediction_seal_sha256=prediction_sha, barrier_sha256=barrier_sha, ten_arm_scope=ten_arm_scope(), **scope(manifest)))
    cfg, panel = manifest["config"], manifest["panel"]
    classes, n = len(panel["classes"]), len(panel["inputs"])
    totals = {a: np.zeros((2, classes), dtype=np.int64) for a in ARMS}
    hashes = {}
    with (output / "scores_by_episode.jsonl").open("x") as episodes, (output / "scores_by_image.jsonl").open("x") as images:
        for start in range(0, n, cfg["batch_size"]):
            counts = {a: np.zeros((2, classes), dtype=np.int64) for a in ARMS}
            stop = min(start + cfg["batch_size"], n)
            for index in range(start, stop):
                row = panel["inputs"][index]
                confirmation.verify_label_metadata(row)
                content = Path(row["label"]).read_bytes()
                hashes[row["label"]] = hashlib.sha256(content).hexdigest()
                with Image.open(io.BytesIO(content)) as source:
                    gt = np.asarray(source.resize((cfg["resize"][1], cfg["resize"][0]), Image.Resampling.NEAREST))
                if gt.ndim == 3:
                    gt = gt[..., 0]
                gt = gt.astype(np.int64, copy=True)
                if panel["reduce_zero_label"]:
                    background, valid = gt == 0, gt != 255
                    gt[valid] -= 1
                    gt[background] = 255
                require(((gt == 255) | ((gt >= 0) & (gt < classes))).all(), "Invalid native label category")
                confirmation.verify_label_metadata(row)
                per_image = {}
                for arm in ARMS:
                    prediction = np.load(output / f"predictions/{arm}/{index:06d}.npy", allow_pickle=False)
                    validate_prediction(prediction, classes, cfg["resize"])
                    value = np.asarray(shared.streaming_counts(prediction, gt, classes))
                    counts[arm] += value
                    per_image[arm] = dict(intersection=value[0].tolist(), union=value[1].tolist())
                images.write(json.dumps(dict(index=index, official_id=row["official_id"],
                    episode=start // cfg["batch_size"], arms=per_image), allow_nan=False) + "\n")
            for arm in ARMS:
                totals[arm] += counts[arm]
            episodes.write(json.dumps(dict(episode=start // cfg["batch_size"], images=stop - start,
                arms={a: dict(intersection=v[0].tolist(), union=v[1].tolist()) for a, v in counts.items()}), allow_nan=False) + "\n")
    require(common.verify_barrier(args.barrier, manifest, output, args.manifest_sha256) == barrier_sha
            and sha256(output / "PREDICTIONS_SEALED.json") == prediction_sha, "Scoring binding drift")
    admission(args)
    verify_inputs(manifest)
    result = dict(status="completed", images=n, arms=ARMS, config=cfg, **scope(manifest),
        arm_results={a: pooled_result(v) for a, v in totals.items()},
        **{k: predicted[k] for k in ("timings", "costs", "timing_scope", "shuffle_coverage")}, manifest_sha256=args.manifest_sha256,
        prediction_seal_sha256=prediction_sha, barrier_sha256=barrier_sha, ten_arm_scope=ten_arm_scope(),
        label_contents_first_opened_after_stage_predictions_sealed=True,
        development_exposed=True, full_evaluation_used_for_selection=False, normalization_variants_are_ablations=True)
    write_json(output / "result.json", result)
    write_json(output / "score_inputs.json", dict(label_sha256=hashes, manifest_sha256=args.manifest_sha256,
        prediction_seal_sha256=prediction_sha, barrier_path=str(args.barrier), barrier_sha256=barrier_sha,
        ten_arm_scope=ten_arm_scope()))
    write_json(output / "SCORED_SEALED.json", dict(complete=True, manifest_sha256=args.manifest_sha256,
        prediction_seal_sha256=prediction_sha, barrier_sha256=barrier_sha, ten_arm_scope=ten_arm_scope(),
        artifact_sha256={name: sha256(output / name) for name in sorted(SCORE_FILES - {"SCORED_SEALED.json"})}))
    return result


def count_array(record, classes):
    require(set(record) == {"intersection", "union"}, "Unexpected class-count fields")
    i, u = record["intersection"], record["union"]
    require(isinstance(i, list) and isinstance(u, list) and len(i) == len(u) == classes
            and all(type(v) is int and 0 <= v <= 2**60 for v in i + u)
            and all(x <= y for x, y in zip(i, u)), "Invalid class counts")
    return np.asarray([i, u], dtype=np.int64)


def verify_score_seal(output, manifest, manifest_sha, barrier=None):
    """CPU-only full stream reconciliation, with no annotation content reads."""
    output = Path(output)
    seal = read_json(output / "SCORED_SEALED.json")
    inputs, result = read_json(output / "score_inputs.json"), read_json(output / "result.json")
    prediction_sha = sha256(output / "PREDICTIONS_SEALED.json")
    barrier = tuning.absolute_path(str(barrier or Path(manifest["suite_root"]) / "ALL_FULL_PREDICTIONS_SEALED.json"))
    require(inputs["barrier_path"] == str(barrier), "Noncanonical score barrier binding")
    barrier_sha = common.verify_barrier(barrier, manifest, output, manifest_sha)
    require(seal["complete"] is True and set(seal["artifact_sha256"]) == SCORE_FILES - {"SCORED_SEALED.json"}, "Incomplete score seal")
    for name, expected in seal["artifact_sha256"].items():
        require(tuning.digest_valid(expected) and sha256(output / name) == expected, "Score payload drift: " + name)
    start = read_json(output / "SCORING_STARTED.json")
    for value in (seal, inputs, result, start):
        require(value["manifest_sha256"] == manifest_sha and value["prediction_seal_sha256"] == prediction_sha
                and value["barrier_sha256"] == barrier_sha and value["ten_arm_scope"] == ten_arm_scope(), "Score binding/scope differs")
    require(all(start[k] == v for k, v in scope(manifest).items()), "Scoring start scope differs")
    labels = inputs["label_sha256"]
    require(set(labels) == {r["label"] for r in manifest["panel"]["inputs"]}
            and all(tuning.digest_valid(v) for v in labels.values()), "Label hash coverage differs")
    cfg, panel = manifest["config"], manifest["panel"]
    n, classes = len(panel["inputs"]), len(panel["classes"])
    totals = {a: np.zeros((2, classes), dtype=np.int64) for a in ARMS}
    episode_stream = iter(full.bounded_json_lines(output / "scores_by_episode.jsonl", (n + cfg["batch_size"] - 1) // cfg["batch_size"]))
    image_stream = iter(full.bounded_json_lines(output / "scores_by_image.jsonl", n))
    for start_index in range(0, n, cfg["batch_size"]):
        stop = min(start_index + cfg["batch_size"], n)
        counts = {a: np.zeros((2, classes), dtype=np.int64) for a in ARMS}
        for index in range(start_index, stop):
            image = next(image_stream, None)
            require(image is not None and image["index"] == index and image["official_id"] == panel["ordered_ids"][index]
                    and image["episode"] == start_index // cfg["batch_size"] and set(image["arms"]) == set(ARMS), "Per-image score order/scope differs")
            for arm in ARMS:
                value = count_array(image["arms"][arm], classes)
                require(int(value[0].sum()) <= math.prod(cfg["resize"])
                        and int(value[1].sum()) <= 2 * math.prod(cfg["resize"]), "Per-image pixel budget exceeded")
                counts[arm] += value
        episode = next(episode_stream, None)
        require(episode is not None and episode["episode"] == start_index // cfg["batch_size"]
                and episode["images"] == stop - start_index and set(episode["arms"]) == set(ARMS), "Episode score scope differs")
        for arm in ARMS:
            require(np.array_equal(counts[arm], count_array(episode["arms"][arm], classes)), "Image/episode score counts differ")
            totals[arm] += counts[arm]
    require(next(image_stream, None) is None and next(episode_stream, None) is None, "Extra score stream records")
    predicted = validate_cost_stream(output, manifest)
    require(result["status"] == "completed" and result["images"] == n and result["arms"] == ARMS
            and result["config"] == cfg and all(result[k] == v for k, v in scope(manifest).items())
            and result["arm_results"] == {a: pooled_result(v) for a, v in totals.items()}
            and all(result[k] == predicted[k] for k in ("costs", "timings", "timing_scope", "shuffle_coverage"))
            and result["label_contents_first_opened_after_stage_predictions_sealed"] is True
            and result["development_exposed"] is True and result["full_evaluation_used_for_selection"] is False
            and result["normalization_variants_are_ablations"] is True, "Final pooled score/cost/claim scope differs")
    return result


def verify(args):
    manifest, output = admission(args)
    verify_inputs(manifest)
    verify_prediction_seal(output, manifest, args.manifest_sha256)
    validate_cost_stream(output, manifest)
    for arm in ARMS:
        for index in range(len(manifest["panel"]["inputs"])):
            validate_prediction(np.load(output / f"predictions/{arm}/{index:06d}.npy", allow_pickle=False),
                len(manifest["panel"]["classes"]), manifest["config"]["resize"])
    if any((output / name).exists() for name in SCORE_FILES):
        verify_score_seal(output, manifest, args.manifest_sha256, args.barrier)
    return dict(verified=True, scored=(output / "SCORED_SEALED.json").exists(), **scope(manifest))


def compare_gate_episode(method, reference, images, valid, ids, meta, cfg, mapping, sources, episode=0):
    """Actual native-C equivalence and four real-update finite gates, no extra forwards."""
    method.reset()
    initial = shared.restored_state(method)
    rng, outcomes, actual, identity = shared.rng_state(method.device), {}, {}, None
    # Capture and remove hooks before another wrapper uses the shared encoder.
    for arm in ["direct_C", *arm_order(episode)]:
        current = reference if arm == "direct_C" else method
        policy = "image_region" if arm == "direct_C" else arm
        current.reset()
        require(shared.restored_state(current) == initial, "Gate initial reset differs")
        shared.restore_rng(rng, current.device)
        with full.gradient_receipt(current) as gradients:
            losses, predictions, candidates, coverage, row = run_policy(current, policy, images,
                valid, ids, meta, cfg, mapping, initial, sources)
        require(len(predictions) == len(meta["img_path"]), "Gate final prediction population differs")
        parameters = {name: p.detach().clone() for name, p in current.model.named_parameters() if p.requires_grad}
        require(set(gradients) == set(parameters) and bool(parameters), "Gate trainable/gradient membership differs")
        require(len(losses) == cfg["steps"] and row["active_candidates"] > 0, "Vacuous finite gate without actual updates")
        for values in gradients.values():
            require(len(values) == len(losses), "Gate missing step gradients")
            full.require_finite(values, "gate gradients")
        full.require_finite(list(parameters.values()), "gate parameters")
        digest = row["candidate_sha256"]
        identity = digest if identity is None else identity
        require(digest == identity, "Gate original C/activity/support identity differs")
        outcomes[arm] = (losses, predictions, candidates, gradients, parameters)
        actual[arm] = dict(row, coverage=coverage, candidates=candidates, losses=losses,
                          allocation_metadata=method.allocation_metadata if current is method else None)
        current.reset()
        require(shared.restored_state(current) == initial, "Gate final reset differs")
    a, b = outcomes["direct_C"], outcomes["image_region"]
    require(len(a[0]) == len(b[0]) and set(a[3]) == set(b[3]) == set(a[4]) == set(b[4]), "Gate membership differs")
    support = max((abs(x[k] - y[k]) for x, y in zip(a[2], b[2], strict=True)
                   for k in ("confidence", "counterfactual_support", "reliability")), default=0.)
    coefficient = max((abs(x["objective_coefficient"] - y["objective_coefficient"])
                       for x, y in zip(a[2], b[2], strict=True)), default=0.)
    loss = max((abs(x - y) for x, y in zip(a[0], b[0], strict=True)), default=0.)
    gradient = max((float((x - y).abs().max()) for name in a[3]
        for x, y in zip(a[3][name], b[3][name], strict=True)), default=0.)
    parameter = max((float((a[4][name] - b[4][name]).abs().max()) for name in a[4]), default=0.)
    require(len(a[1]) == len(b[1]) and all(x.shape == y.shape for x, y in zip(a[1], b[1], strict=True)), "Gate label shapes differ")
    changed = sum(int(np.count_nonzero(x != y)) for x, y in zip(a[1], b[1], strict=True))
    require(all(math.isfinite(v) and v <= 1e-5 for v in (support, coefficient, loss, gradient, parameter))
            and changed == 0, "Direct native image-region C equivalence failed")
    return dict(candidates=len(a[2]), candidate_masks_equal=True, reset_equal=True,
        max_support_abs_difference=support, max_coefficient_abs_difference=coefficient,
        max_loss_abs_difference=loss, max_gradient_abs_difference=gradient,
        max_parameter_abs_difference=parameter, changed_label_pixels=changed, actual_arm_counts=actual,
        initial_state=initial, final_parameter_sha256={k: shared.state_digest(v[4]) for k, v in outcomes.items()})


def gate(args):
    from trace_ovss.cli import build_patch_image_ids, build_patch_valid_mask
    manifest, output = admission(args)
    require(manifest["stage"] == "gate" and len(manifest["panel"]["inputs"]) == 4, "Exactly four GT-free RGB gate inputs required")
    common.require_gpu_envelope(manifest)
    claim = started(output, args.manifest_sha256, "gate")
    report = dict(passed=False, images=4, gt_contents_opened=False, manifest_sha256=args.manifest_sha256,
        attempt_sha256=sha256(claim), output=str(output), config=manifest["config"], arms=ARMS,
        manifest=manifest, manifest_path=str(args.manifest), suite_prepared_sha256=manifest["suite_prepared_sha256"],
        ordered_ids=manifest["panel"]["ordered_ids"], **scope(manifest))
    try:
        data = verify_inputs(manifest)
        with confirmation.prior.gpu_lock() as lock:
            runtime = shared.configure_deterministic_runtime()
            method = build_method(manifest)
            method.reset()
            reference = GateReference(classes=method.classes, lr=method.lr, steps=method.steps,
                trace_max_regions=method.trace_max_regions, trace_view_batch_size=method.trace_view_batch_size,
                vision_outputs=method.vision_outputs, prompts=method.prompt_templates, model=method.model,
                tokenize_fn=method.tokenize, device=method.device, view_mode="C")
            require(torch.equal(method.text_x, reference.text_x), "Gate text embeddings differ")
            cfg, comparisons, batches = manifest["config"], [], []
            for episode, start in enumerate(range(0, 4, cfg["batch_size"])):
                stop = min(start + cfg["batch_size"], 4)
                batch = shared.image_collate([data[i] for i in range(start, stop)])
                images, meta = batch["img_patches"].to(method.device), batch["meta"]
                sources = stable_sources(manifest, meta, start, stop)
                valid = build_patch_valid_mask(meta, cfg["patch_size"], cfg["patch_stride"], method.device)
                ids = build_patch_image_ids(meta, method.device)
                comparisons.append(compare_gate_episode(method, reference, images, valid, ids, meta,
                    cfg, manifest["panel"]["query_to_class"], sources, episode))
                batches.append(stop - start)
            admission(args)
            verify_inputs(manifest)
            require(shared.deterministic_runtime_state() == runtime, "Gate deterministic runtime drift")
            report.update(passed=True, comparisons=comparisons, batch_images=batches, gpu_lock=lock,
                deterministic_runtime=runtime, checkpoint=manifest["checkpoint"], source_sha256=manifest["source_sha256"])
        write_json(output / "gate.json", report)
        write_json(output / "GATE_PASSED.json", dict(report, gate_sha256=sha256(output / "gate.json")))
    except BaseException as error:
        target = output / ("failure.json" if (output / "gate.json").exists() else "gate.json")
        write_json(target, dict(report, failure=tuning.failure_receipt(error, "gate", args.manifest_sha256)))
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("gate", "predict", "score", "verify"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--barrier", type=Path)
    args = parser.parse_args(argv)
    if args.phase == "score" and args.barrier is None:
        parser.error("--barrier is required for score")
    if args.phase != "score" and args.barrier is not None:
        parser.error("--barrier is allowed only for score")
    return dict(gate=gate, predict=predict, score=score, verify=verify)[args.phase](args)


if __name__ == "__main__":
    main()
