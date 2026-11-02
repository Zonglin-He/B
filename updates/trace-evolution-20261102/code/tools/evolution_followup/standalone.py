"""Bounded standalone probe costs: prepare a lock, predict all arms, then score.

Prepare and score do not run model inference. The inherited CLIP import may
query CUDA availability; prediction acquires the lock before model loading.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evolution import run_cell as shared
from tools.evolution.policies import state_digest
from tools.suite_inputs import EVALUATION_COUNTS, validate_discovery, validate_population
from trace_ovss.configs import BEST_CONFIGS, VISION_OUTPUTS
from trace_ovss.data.datasets import SegmentationPatchDataset
from trace_ovss.method.trace.constants import EPS
from trace_ovss.method.trace.indexing import pack_region_indices
from trace_ovss.method.two_view import TwoViewTRACE


ARMS = ("min", "T", "C", "Uniform", "Confidence")
MAX_IMAGES = 128
GPU_LOCK = Path("/home/wwww/hzl/results/reviewer_followup_gpu.lock")
TIMING_SCOPE = (
    "Synchronized CUDA wallclock per independent arm: episodic reset, initial readout, "
    "candidate construction/selection, requested diagnostic views, LayerNorm optimization, "
    "final inference and reconstruction/CPU prediction transfer. Includes accounting hooks. "
    "Excludes model setup/loading, RGB decode/H2D/geometry preparation common to arms, "
    "post-run provenance checks, prediction/receipt disk writes, and offline scoring. "
    "Exclude only first complete episode (episode 0) per arm from timing aggregates; "
    "retain every prediction, count and accuracy observation including warmup."
)
CONFIG_KEYS = {
    "dataset", "data_subdir", "data_dir", "resize", "patch_size", "patch_stride",
    "batch_size", "lr", "steps", "trace_max_regions", "class_extensions",
    "trace_view_batch_size",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def validate_config(config, dataset):
    require(dataset in EVALUATION_COUNTS and dataset in BEST_CONFIGS, "Unsupported evaluation dataset")
    require(set(config) == CONFIG_KEYS, "Provide exactly the resolved config keys; no implicit defaults")
    require(config["dataset"] == BEST_CONFIGS[dataset].dataset, "Dataset/config mismatch")
    require(type(config["class_extensions"]) is bool, "class_extensions must be boolean")
    require(config["class_extensions"] == BEST_CONFIGS[dataset].class_extensions,
            "Keep the dataset's query-extension convention")
    for key in ("batch_size", "steps", "trace_max_regions", "trace_view_batch_size", "patch_stride"):
        require(type(config[key]) is int and config[key] > 0, f"Invalid {key}")
    require(type(config["lr"]) in (int, float) and math.isfinite(config["lr"])
            and config["lr"] > 0, "Invalid lr")
    for key in ("resize", "patch_size"):
        require(isinstance(config[key], list) and len(config[key]) == 2
                and all(type(v) is int and v > 0 for v in config[key]), f"Invalid {key}")
    require(isinstance(config["data_dir"], str) and Path(config["data_dir"]).is_absolute(),
            "data_dir must be an explicit absolute path")
    require(isinstance(config["data_subdir"], str), "Invalid data_subdir")
    require(min(config["resize"]) >= max(config["patch_size"]), "Resize smaller than patch")


def validate_request(dataset, config, seed, selection_seed, max_images, arms, order):
    validate_config(config, dataset)
    require(type(seed) is int and seed >= 0 and type(selection_seed) is int
            and selection_seed >= 0, "Explicit nonnegative seeds required")
    require(type(max_images) is int and 1 <= max_images <= MAX_IMAGES, "Bound is 1..128 images")
    require(bool(arms) and len(arms) == len(set(arms)) and set(arms) <= set(ARMS), "Invalid arms")
    require(order in ("fixed", "alternating"), "Predeclare fixed or alternating order")


def view_schedule(arm, candidates):
    require(arm in ARMS, "Unknown arm")
    kinds = {"min": ("target", "counterfactual"), "T": ("target",),
             "C": ("counterfactual",), "Uniform": (), "Confidence": ()}[arm]
    return [(i, kind) for i in range(candidates) for kind in kinds]


def allocation(arm, target, context, confidence):
    required = {"min": (target, context), "T": (target,), "C": (context,),
                "Uniform": (1.0,), "Confidence": (confidence,)}[arm]
    require(all(v is not None and math.isfinite(v) and 0 <= v <= 1 for v in required),
            "Required support missing or invalid")
    return min(required)


def normalized_weights(values):
    require(all(v is not None and math.isfinite(v) and 0 <= v <= 1 for v in values),
            "Cannot normalize missing/invalid allocation")
    effective = [v if v > EPS else 0.0 for v in values]
    total = sum(effective)
    return [v / total for v in effective] if total > 0 else None


def candidate_identity(rows):
    """No measured view support enters admission identity."""
    digest = hashlib.sha256()
    for row in rows:
        identity = {key: row[key] for key in
                    ("batch_idx", "image_idx", "class_idx", "confidence", "area", "component_idx")}
        digest.update(json.dumps(identity, sort_keys=True, allow_nan=False).encode())
        mask = np.asarray(row["region"], dtype=np.bool_)
        digest.update(str(mask.shape).encode())
        digest.update(mask.tobytes())
    return digest.hexdigest()


class StandaloneTRACE(TwoViewTRACE):
    """Local experimental subclass; inherited admission, LN optimizer and loss."""

    arm = "min"

    def _forward_trace_view_batches(self, keys, build_views):
        # No OOM fallback: a failure invalidates this fresh run, never changes config.
        for start in range(0, len(keys), self.trace_view_batch_size):
            chunk_keys = keys[start:start + self.trace_view_batch_size]
            self._diagnostic = True
            try:
                logits = self.forward_eval_logits(build_views(chunk_keys), interpolate=False,
                                                  save_weights=False).detach()
            finally:
                self._diagnostic = False
            logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            yield chunk_keys, logits
            del logits
        self.last_trace_view_batch_size = self.trace_view_batch_size

    @torch.no_grad()
    def evaluate_trace_views(self, images, original_probs, candidates):
        if self.arm == "min":
            return super().evaluate_trace_views(images, original_probs, candidates)
        self._two_view_counts = dict(diagnostic_views=0, diagnostic_calls=0)
        if not candidates:
            return []
        indices = pack_region_indices([row["region"] for row in candidates], images.device)
        keys = view_schedule(self.arm, len(candidates))
        contexts, supports = {}, {}

        def build_views(batch_keys):
            ids = tuple(dict.fromkeys(i for i, _ in batch_keys))
            masks = F.interpolate(torch.from_numpy(np.stack([candidates[i]["region"] for i in ids]))
                .to(device=images.device, dtype=images.dtype).unsqueeze(1),
                size=images.shape[-2:], mode="nearest")
            lookup = {i: masks[j:j + 1] for j, i in enumerate(ids)}
            views = []
            for i, kind in batch_keys:
                patch = candidates[i]["batch_idx"]
                image, mask = images[patch:patch + 1], lookup[i]
                if patch not in contexts:
                    helper = (self.make_trace_weak_image if kind == "target"
                              else self.make_trace_counterfactual_background)
                    contexts[patch] = helper(image)
                views.append(image * mask + contexts[patch] * (1.0 - mask))
            return torch.cat(views)

        for batch_keys, logits in self._forward_trace_view_batches(keys, build_views):
            values = [item.softmax(0)[candidates[i]["class_idx"]].flatten()
                      .index_select(0, indices[i]).mean()
                      for (i, _), item in zip(batch_keys, logits)]
            supports.update(zip(batch_keys, torch.stack(values).cpu().tolist()))
            self._two_view_counts["diagnostic_views"] += len(batch_keys)
            self._two_view_counts["diagnostic_calls"] += 1
            del logits, values
        rows = []
        for i, row in enumerate(candidates):
            target, context = supports.get((i, "target")), supports.get((i, "counterfactual"))
            confidence = float(row["confidence"])
            rho = allocation(self.arm, target, context, confidence)
            rows.append(dict(row, original_support=confidence, target_support=target,
                counterfactual_support=context, residual_support=None, dominance=None,
                quality=rho, reliability=rho, diagnostic_algorithm=self.arm, omitted=False))
        return rows

    def forward_eval_logits(self, images, *args, **kwargs):
        if hasattr(self, "work"):
            phase = "diagnostic" if getattr(self, "_diagnostic", False) else "eval"
            self._count(phase + "_forward", len(images))
        return super().forward_eval_logits(images, *args, **kwargs)

    def forward_adapt_logits(self, images):
        if hasattr(self, "work"):
            self._count("adapt_forward", len(images))
        return super().forward_adapt_logits(images)

    def _count(self, phase, inputs):
        self.work[phase + "_calls"] += 1
        self.work[phase + "_inputs"] += inputs

    @contextmanager
    def count_work(self):
        phases = ("diagnostic_forward", "eval_forward", "adapt_forward", "encoder", "diagnostic_encoder")
        self.work = {phase + suffix: 0 for phase in phases for suffix in ("_calls", "_inputs")}

        def stem(module, args):
            self._count("encoder", int(args[0].shape[0]))
            if getattr(self, "_diagnostic", False):
                self._count("diagnostic_encoder", int(args[0].shape[0]))

        handle = self.model.visual.conv1.register_forward_pre_hook(stem)
        try:
            yield self.work
        finally:
            handle.remove()
            del self.work


def arm_order(arms, episode, order):
    return list(reversed(arms)) if order == "alternating" and episode % 2 else list(arms)


def timing_summary(rows, arms):
    result = {}
    for arm in arms:
        measured = [row for row in rows if row["arm"] == arm and row["episode"] != 0]
        seconds = sum(row["seconds"] for row in measured)
        images = sum(row["images"] for row in measured)
        result[arm] = dict(measured_episodes=len(measured), measured_images=images,
            seconds=seconds if measured else None,
            seconds_per_image=seconds / images if images else None,
            images_per_second=images / seconds if images and seconds > 0 else None,
            warmup_episode=0, accuracy_includes_warmup=True)
    return result


def checked_sources():
    for module in (shared, sys.modules[TwoViewTRACE.__module__], sys.modules[SegmentationPatchDataset.__module__]):
        require(Path(inspect.getfile(module)).resolve().is_relative_to(ROOT), "Import outside worker source root")
    for name, module in list(sys.modules.items()):
        if name in ("trace_ovss", "tools.evolution", "tools.suite_inputs") or name.startswith(
                ("trace_ovss.", "tools.evolution.")):
            path = getattr(module, "__file__", None)
            if path:
                require(Path(path).resolve().is_relative_to(ROOT), f"Wrong import: {name}")
    sources = shared.source_hashes(ROOT)
    for path in (Path(__file__).resolve(), ROOT / "tests/test_standalone_probes.py"):
        sources[str(path)] = shared.sha256(path)
    return sources


def selected_indices(available, selection_seed, max_images, episode_order_seed=None):
    selected = sorted(np.random.default_rng(selection_seed).permutation(available)[:max_images].tolist())
    if episode_order_seed is None:
        return selected
    require(type(episode_order_seed) is int and episode_order_seed >= 0, "Invalid episode order seed")
    return np.random.default_rng(episode_order_seed).permutation(selected).tolist()


def build_panel(dataset, config, selection_seed, max_images, episode_order_seed=None):
    data = shared.image_only_dataset_class(SegmentationPatchDataset)(config["dataset"], config["data_dir"],
        resize=config["resize"], patch_size=config["patch_size"], patch_stride=config["patch_stride"],
        include_auxiliary=False)
    validate_discovery(data)
    validate_population(dataset, "evaluation", max_images, len(data))
    indices = selected_indices(len(data), selection_seed, max_images, episode_order_seed)
    image_root = data.data_dir / data.spec.image_dir
    inputs = []
    for index in indices:
        image, label = data.samples[index]
        inputs.append(dict(dataset_index=index, official_id=image.relative_to(image_root).as_posix(),
            image=str(image), image_sha256=shared.sha256(image), label=str(label),
            label_metadata=[label.stat().st_size, label.stat().st_mtime_ns]))
    queries = list(data.vocabulary.extensions if config["class_extensions"] else data.vocabulary.classes)
    mapping = list(data.vocabulary.extension_to_class) if config["class_extensions"] else None
    panel = dict(inputs=inputs, ordered_ids=[row["official_id"] for row in inputs],
        selected_dataset_indices=sorted(indices), available_images=len(data), full_split=False,
        classes=list(data.vocabulary.classes), queries=queries, query_to_class=mapping,
        reduce_zero_label=data.spec.reduce_zero_label,
        vocabulary_sha256=canonical_hash([queries, mapping]))
    data.expected_image_hashes = {row["image"]: row["image_sha256"] for row in inputs}
    return data, indices, panel


def prepare(args):
    config = read_json(args.config)
    validate_request(args.dataset, config, args.seed, args.selection_seed, args.max_images,
                     args.arms, args.arm_order)
    require(args.exposure_note.strip(), "Declare prior evaluation exposure/transfer scope")
    _, _, panel = build_panel(args.dataset, config, args.selection_seed, args.max_images, args.episode_order_seed)
    manifest = dict(format_version=1, worker="standalone_probes", dataset=args.dataset,
        config=config, config_path=str(args.config.resolve()), config_sha256=shared.sha256(args.config),
        seed=args.seed, selection_seed=args.selection_seed, episode_order_seed=args.episode_order_seed,
        max_images=args.max_images,
        arms=args.arms, arm_order=args.arm_order, panel=panel, source_sha256=checked_sources(),
        checkpoint=shared.checkpoint_identity(args.checkpoint), timing_scope=TIMING_SCOPE,
        exposure_note=args.exposure_note, claim_boundary="Bounded descriptive comparison; not untouched validation")
    write_json(args.manifest, manifest)
    print(json.dumps(dict(manifest=str(args.manifest.resolve()), sha256=shared.sha256(args.manifest))))


def validate_manifest(manifest):
    require(manifest["format_version"] == 1 and manifest["worker"] == "standalone_probes", "Wrong manifest")
    validate_request(manifest["dataset"], manifest["config"], manifest["seed"], manifest["selection_seed"],
                     manifest["max_images"], manifest["arms"], manifest["arm_order"])
    panel = manifest["panel"]
    require(panel["full_split"] is False and len(panel["inputs"]) == manifest["max_images"], "Not bounded subset")
    require(manifest["timing_scope"] == TIMING_SCOPE, "Timing boundary changed")
    require(bool(manifest["exposure_note"].strip()), "Missing exposure declaration")
    require(panel["available_images"] == EVALUATION_COUNTS[manifest["dataset"]], "Wrong official population")
    ids = [row["dataset_index"] for row in panel["inputs"]]
    expected = selected_indices(panel["available_images"], manifest["selection_seed"],
                                manifest["max_images"], manifest["episode_order_seed"])
    require(ids == expected, "Subset identities do not match explicit selection seed")
    require(panel["selected_dataset_indices"] == sorted(ids), "Subset membership differs")
    require(panel["ordered_ids"] == [row["official_id"] for row in panel["inputs"]], "Ordered IDs differ")
    require(len({row["official_id"] for row in panel["inputs"]}) == len(ids), "Duplicate official ID")
    require(panel["vocabulary_sha256"] == canonical_hash([panel["queries"], panel["query_to_class"]]),
            "Vocabulary hash mismatch")


@contextmanager
def gpu_lock():
    # Use exactly the scheduler's global inode; do not offer a private lock override.
    with GPU_LOCK.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield dict(path=str(GPU_LOCK), inode=os.fstat(stream.fileno()).st_ino,
                       pid=os.getpid(), exclusive=True)
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def verify_manifest_file(path, expected_hash):
    require(shared.sha256(path) == expected_hash, "Manifest hash mismatch")
    manifest = read_json(path)
    validate_manifest(manifest)
    return manifest


def execute_predictions(method, data, indices, manifest, output):
    from trace_ovss.cli import build_patch_image_ids, build_patch_valid_mask
    config, arms = manifest["config"], manifest["arms"]
    for arm in arms:
        (output / "predictions" / arm).mkdir(parents=True)
    (output / "episodes").mkdir()
    method.reset()
    initial = shared.restored_state(method)
    frozen = [(p, p._version) for p in method.model.parameters() if not p.requires_grad]
    costs = []
    with (output / "cost.jsonl").open("x") as cost_stream:
        for episode, start in enumerate(range(0, len(indices), config["batch_size"])):
            chosen = indices[start:start + config["batch_size"]]
            batch = shared.image_collate([data[index] for index in chosen])
            images = batch["img_patches"].to(method.device)
            meta = batch["meta"]
            valid = build_patch_valid_mask(meta, config["patch_size"], config["patch_stride"], method.device)
            image_ids = build_patch_image_ids(meta, method.device)
            rng = shared.rng_state(method.device)
            identity = None
            episode_id = canonical_hash([manifest["panel"]["inputs"][i]
                                         for i in range(start, start + len(chosen))])
            for position, arm in enumerate(arm_order(arms, episode, manifest["arm_order"])):
                method.arm = arm
                shared.restore_rng(rng, method.device)
                with method.count_work() as counts:
                    shared.synchronize(method.device)
                    tick = time.perf_counter()
                    losses = method.adapt(images, valid_mask=valid, image_ids=image_ids)
                    predictions = shared.label_predictions(method, images, meta, config,
                                                           manifest["panel"]["query_to_class"])
                    shared.synchronize(method.device)
                    seconds = time.perf_counter() - tick
                require(all(math.isfinite(v) for v in losses), "Nonfinite loss")
                stats = method.last_candidate_stats
                digest = candidate_identity(stats)
                identity = digest if identity is None else identity
                require(digest == identity, "Candidates differ across standalone arms")
                expected_views = len(view_schedule(arm, len(stats)))
                require(counts["diagnostic_forward_inputs"] == expected_views
                        == counts["diagnostic_encoder_inputs"], "Unexpected diagnostic schedule")
                require(not any(p._version != version for p, version in frozen), "Frozen parameters changed")
                weights = normalized_weights([row["reliability"] for row in stats])
                row = dict(episode=episode, episode_sha256=episode_id, arm=arm, order_position=position,
                    images=len(chosen), patches=len(images), seconds=seconds, warmup=episode == 0,
                    steps=len(losses), candidate_count=len(stats), candidate_sha256=digest,
                    active_candidates=sum(r["reliability"] > EPS for r in stats), **counts)
                row["total_forward_calls"] = sum(counts[k + "_forward_calls"] for k in ("eval", "diagnostic", "adapt"))
                row["total_forward_inputs"] = sum(counts[k + "_forward_inputs"] for k in ("eval", "diagnostic", "adapt"))
                row["total_eval_forward_calls"] = counts["eval_forward_calls"] + counts["diagnostic_forward_calls"]
                row["total_eval_forward_inputs"] = counts["eval_forward_inputs"] + counts["diagnostic_forward_inputs"]
                cost_stream.write(json.dumps(row, allow_nan=False) + "\n")
                cost_stream.flush()
                costs.append(row)
                for offset, prediction in enumerate(predictions):
                    with (output / "predictions" / arm / f"{start + offset:06d}.npy").open("xb") as stream:
                        np.save(stream, prediction, allow_pickle=False)
                receipt = dict(**row, dataset_indices=chosen, losses=losses,
                    candidates=[dict(batch_idx=r["batch_idx"], image_idx=r["image_idx"], class_idx=r["class_idx"],
                        confidence=r["confidence"], T=r["target_support"], C=r["counterfactual_support"],
                        rho=r["reliability"], normalized_weight=weights[i] if weights is not None else None)
                        for i, r in enumerate(stats)])
                write_json(output / "episodes" / f"{episode:06d}.{arm}.json", receipt)
                method.reset()
                require(shared.restored_state(method) == initial, "Episode reset changed initial state")
                del predictions, stats, receipt
    return timing_summary(costs, arms)


def prediction_artifacts(manifest):
    images = manifest["max_images"]
    episodes = (images + manifest["config"]["batch_size"] - 1) // manifest["config"]["batch_size"]
    return ({f"predictions/{arm}/{i:06d}.npy" for arm in manifest["arms"] for i in range(images)}
            | {f"episodes/{e:06d}.{arm}.json" for arm in manifest["arms"] for e in range(episodes)}
            | {"prediction_protocol.json", "initial_state.json", "prediction_result.json", "cost.jsonl"})


def seal_predictions(output, protocol):
    required = prediction_artifacts(protocol["manifest"])
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
    require(actual == required, "Incomplete or unexpected prediction artifacts")
    write_json(output / "PREDICTIONS_SEALED.json", dict(complete=True, gt_contents_opened=False,
        images=protocol["manifest"]["max_images"], arms=protocol["manifest"]["arms"],
        artifact_sha256={name: shared.sha256(output / name) for name in sorted(required)}))


def verify_prediction_seal(output):
    seal = read_json(output / "PREDICTIONS_SEALED.json")
    require(seal.get("complete") is True and seal.get("gt_contents_opened") is False, "Unsealed predictions")
    for name, digest in seal["artifact_sha256"].items():
        require((output / name).resolve().is_relative_to(output.resolve()), "Seal path escape")
        require(shared.sha256(output / name) == digest, f"Seal mismatch: {name}")
    protocol = read_json(output / "prediction_protocol.json")
    manifest = protocol["manifest"]
    validate_manifest(manifest)
    for key, expected in (("selection_seed", manifest["selection_seed"]),
                          ("episode_order_seed", manifest["episode_order_seed"]),
                          ("ordered_ids", manifest["panel"]["ordered_ids"])):
        require(protocol[key] == expected, f"Protocol/manifest mismatch: {key}")
    require(prediction_artifacts(manifest) == set(seal["artifact_sha256"]), "Seal population incomplete")
    actual = {str(p.relative_to(output)) for folder in ("predictions", "episodes")
              for p in (output / folder).rglob("*") if p.is_file()}
    require(actual == {p for p in prediction_artifacts(manifest) if p.startswith(("predictions/", "episodes/"))},
            "Unexpected prediction/episode population")
    require(seal["images"] == manifest["max_images"] and seal["arms"] == manifest["arms"], "Seal scope differs")
    return protocol


def predict(args):
    manifest = verify_manifest_file(args.manifest, args.manifest_sha256)
    require(checked_sources() == manifest["source_sha256"], "Frozen source changed")
    require(shared.sha256(manifest["config_path"]) == manifest["config_sha256"]
            and read_json(manifest["config_path"]) == manifest["config"], "Frozen config changed")
    data, indices, panel = build_panel(manifest["dataset"], manifest["config"],
                                      manifest["selection_seed"], manifest["max_images"], manifest["episode_order_seed"])
    require(panel == manifest["panel"], "Frozen RGB-only panel changed")
    checkpoint = shared.checkpoint_identity(manifest["checkpoint"]["path"])
    require(checkpoint == manifest["checkpoint"], "Checkpoint changed")
    # A pre-existing output is never resumed, even if empty or partially written.
    args.output.mkdir(parents=True, exist_ok=False)
    with gpu_lock() as lock:
        runtime = shared.configure_deterministic_runtime()
        protocol = dict(manifest=manifest, manifest_path=str(args.manifest.resolve()),
            manifest_sha256=args.manifest_sha256, gpu_lock=lock, deterministic_runtime=runtime,
            selection_seed=manifest["selection_seed"], episode_order_seed=manifest["episode_order_seed"],
            ordered_ids=panel["ordered_ids"])
        write_json(args.output / "prediction_protocol.json", protocol)
        random.seed(manifest["seed"])
        np.random.seed(manifest["seed"])
        torch.manual_seed(manifest["seed"])
        config = manifest["config"]
        method = StandaloneTRACE(classes=panel["queries"], lr=config["lr"], steps=config["steps"],
            trace_max_regions=config["trace_max_regions"], trace_view_batch_size=config["trace_view_batch_size"],
            vision_outputs=VISION_OUTPUTS, device="cuda", backbone=checkpoint["path"])
        require(shared.checkpoint_identity(checkpoint["path"]) == checkpoint, "Checkpoint changed on loading")
        write_json(args.output / "initial_state.json", dict(model_sha256=state_digest(method.model.state_dict()),
            optimizer_sha256=state_digest(method.optimizer_state), adapted_parameter_names=method.adapted_parameter_names,
            environment=dict(python=sys.version, platform=platform.platform(), torch=str(torch.__version__),
                numpy=np.__version__, pillow=Image.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(method.device),
                gpu_properties=str(torch.cuda.get_device_properties(method.device)),
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision())))
        timings = execute_predictions(method, data, indices, manifest, args.output)
        require(checked_sources() == manifest["source_sha256"], "Source changed during prediction")
        require(shared.deterministic_runtime_state() == runtime, "Deterministic runtime changed")
        require(shared.sha256(manifest["config_path"]) == manifest["config_sha256"], "Config changed during prediction")
        require(all(shared.sha256(r["image"]) == r["image_sha256"] for r in panel["inputs"]), "RGB changed")
        verify_manifest_file(args.manifest, args.manifest_sha256)
        write_json(args.output / "prediction_result.json", dict(status="predicted", full_split=False,
            images=manifest["max_images"], timings=timings, timing_scope=TIMING_SCOPE,
            accuracy_includes_warmup=True, measurements="Fresh independent standalone arms; not shared-arm costs"))
        seal_predictions(args.output, protocol)


def score(output):
    output = Path(output)
    protocol = verify_prediction_seal(output)  # All arms verified before the first label read.
    manifest = protocol["manifest"]
    require(checked_sources() == manifest["source_sha256"], "Frozen scoring source changed")
    owned = ("SCORING_STARTED.json", "scores_by_episode.jsonl", "score_inputs.json", "result.json", "SCORED_SEALED.json")
    require(not any((output / name).exists() for name in owned), "No scoring retry/overwrite")
    write_json(output / "SCORING_STARTED.json", dict(prediction_seal_sha256=shared.sha256(output / "PREDICTIONS_SEALED.json")))
    arms, panel = manifest["arms"], manifest["panel"]
    classes = len(panel["classes"])
    totals = {arm: np.zeros((2, classes), dtype=np.int64) for arm in arms}
    label_hashes = {}
    batch = manifest["config"]["batch_size"]
    with (output / "scores_by_episode.jsonl").open("x") as records:
        for start in range(0, manifest["max_images"], batch):
            counts = {arm: np.zeros((2, classes), dtype=np.int64) for arm in arms}
            for index in range(start, min(start + batch, manifest["max_images"])):
                row = panel["inputs"][index]
                path = Path(row["label"])
                require([path.stat().st_size, path.stat().st_mtime_ns] == row["label_metadata"], "GT metadata changed")
                content = path.read_bytes()
                label_hashes[str(path)] = hashlib.sha256(content).hexdigest()
                first = np.load(output / "predictions" / arms[0] / f"{index:06d}.npy", allow_pickle=False)
                with Image.open(io.BytesIO(content)) as source:
                    gt = np.asarray(source.resize((first.shape[1], first.shape[0]), Image.Resampling.NEAREST))
                if gt.ndim == 3:
                    gt = gt[..., 0]
                gt = gt.astype(np.int64, copy=True)
                if panel["reduce_zero_label"]:
                    background, valid = gt == 0, gt != 255
                    gt[valid] -= 1
                    gt[background] = 255
                for arm in arms:
                    prediction = first if arm == arms[0] else np.load(
                        output / "predictions" / arm / f"{index:06d}.npy", allow_pickle=False)
                    counts[arm] += np.asarray(shared.streaming_counts(prediction, gt, classes))
            for arm in arms:
                totals[arm] += counts[arm]
            records.write(json.dumps(dict(episode=start // batch, warmup=start == 0,
                arms={arm: dict(intersection=v[0].tolist(), union=v[1].tolist()) for arm, v in counts.items()})) + "\n")
    results = {}
    for arm, (intersection, union) in totals.items():
        valid = union > 0
        results[arm] = dict(miou_unrounded=float(100 * np.mean(intersection[valid] / union[valid])) if valid.any() else None,
                           intersection=intersection.tolist(), union=union.tolist())
    result = dict(status="completed", dataset=manifest["dataset"], seed=manifest["seed"],
        images=manifest["max_images"], full_split=False, arm_results=results,
        timings=read_json(output / "prediction_result.json")["timings"], timing_scope=TIMING_SCOPE,
        prediction_seal_sha256=shared.sha256(output / "PREDICTIONS_SEALED.json"),
        label_contents_first_opened_after_all_requested_arms_sealed=True,
        exposure_note=manifest["exposure_note"], claim_boundary=manifest["claim_boundary"])
    write_json(output / "result.json", result)
    write_json(output / "score_inputs.json", dict(label_sha256=label_hashes))
    write_json(output / "SCORED_SEALED.json", dict(prediction_seal_sha256=result["prediction_seal_sha256"],
        artifact_sha256={name: shared.sha256(output / name) for name in owned[:-1]}))
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="phase", required=True)
    prepare_parser = sub.add_parser("prepare", help="CPU-only input/config/source/checkpoint lock")
    prepare_parser.add_argument("--config", type=Path, required=True)
    prepare_parser.add_argument("--dataset", choices=tuple(EVALUATION_COUNTS), required=True)
    prepare_parser.add_argument("--seed", type=int, required=True)
    prepare_parser.add_argument("--selection-seed", type=int, required=True)
    prepare_parser.add_argument("--episode-order-seed", type=int,
                                help="Permute only selected IDs; default is official sorted order")
    prepare_parser.add_argument("--max-images", type=int, required=True)
    prepare_parser.add_argument("--arms", nargs="+", choices=ARMS, required=True)
    prepare_parser.add_argument("--arm-order", choices=("fixed", "alternating"), required=True)
    prepare_parser.add_argument("--checkpoint", type=Path, required=True)
    prepare_parser.add_argument("--exposure-note", required=True)
    prepare_parser.add_argument("--manifest", type=Path, required=True)
    predict_parser = sub.add_parser("predict", help="GPU phase; requires reviewed manifest digest and global lock")
    predict_parser.add_argument("--manifest", type=Path, required=True)
    predict_parser.add_argument("--manifest-sha256", required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    score_parser = sub.add_parser("score", help="CPU-only; requires complete all-arm prediction seal")
    score_parser.add_argument("--output", type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.phase == "prepare":
        prepare(args)
    elif args.phase == "predict":
        predict(args)
    else:
        print(json.dumps(score(args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
