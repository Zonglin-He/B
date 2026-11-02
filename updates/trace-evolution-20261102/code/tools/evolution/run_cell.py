"""Fixed-candidate two-view controls, with a strict predict/seal/score boundary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import random
import platform
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.evolution.policies import (POLICIES, candidate_digest, expand_policies,
                                     make_fixed_candidate_class, state_digest)


CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def deterministic_runtime_state():
    return dict(
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        deterministic_debug_mode=torch.get_deterministic_debug_mode(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        sdpa_enabled={name: getattr(torch.backends.cuda, name + "_sdp_enabled")()
                      for name in ("flash", "mem_efficient", "math", "cudnn")},
    )


def configure_deterministic_runtime():
    """Fail closed on conflicting startup state; do not select an SDPA backend."""
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured is not None and configured != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be exactly :4096:8")
    if torch.cuda.is_initialized():
        raise RuntimeError("Deterministic runtime must be configured before CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return deterministic_runtime_state()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def image_only_dataset_class(base):
    class ImageOnlyPatchDataset(base):
        def __getitem__(self, index):
            from trace_ovss.data.datasets import CLIP_MEAN, CLIP_STD
            image_path, mask_path = self.samples[index]
            content = image_path.read_bytes()
            expected = getattr(self, "expected_image_hashes", {}).get(str(image_path))
            if expected is not None and hashlib.sha256(content).hexdigest() != expected:
                raise RuntimeError("Image bytes differ from frozen inputs")
            with Image.open(io.BytesIO(content)) as source:
                image = source.convert("RGB")
            original_shape = (image.height, image.width)
            image = image.resize(self._target_size(original_shape), Image.Resampling.BILINEAR)
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()
            pixels = (pixels - CLIP_MEAN) / CLIP_STD
            patches, grid_shape = self._patchify(pixels)
            return dict(img_patches=patches, meta=dict(
                img_path=str(image_path), seg_map_path=str(mask_path), ori_shape=original_shape,
                img_shape=pixels.shape[-2:], patch_shape=self.patch_size, patch_grid_shape=grid_shape))

    return ImageOnlyPatchDataset


def image_collate(batch):
    if any(set(item) != {"img_patches", "meta"} for item in batch):
        raise ValueError("Image-only collation rejects labels and unknown fields")
    return dict(img_patches=torch.cat([item["img_patches"] for item in batch]),
                meta={key: [item["meta"][key] for item in batch] for key in batch[0]["meta"]})


def source_hashes(root):
    files = [path for path in sorted((root / "trace_ovss").rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo")]
    files += [Path(__file__).resolve().parents[1] / name for name in
              ("trace_algorithm_variants.py", "trace_execution_audit.py", "suite_inputs.py")]
    files += [Path(__file__), Path(__file__).with_name("policies.py")]
    return {str(path.resolve()): sha256(path) for path in files}


def checkpoint_identity(path=None):
    from trace_ovss.models._vendor_clip.clip import _MODELS
    url = _MODELS["ViT-L/14"]
    path = Path(path or Path.home() / ".cache/clip" / url.rsplit("/", 1)[1]).expanduser().resolve()
    before = path.stat()
    actual = sha256(path)
    expected = url.split("/")[-2]
    after = path.stat()
    if actual != expected or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Checkpoint differs from official ViT-L/14 identity or changed while hashing")
    return dict(path=str(path), sha256=actual, bytes=after.st_size, mtime_ns=after.st_mtime_ns,
                official_url=url, backbone="ViT-L/14")


def restored_state(method):
    actual = {name: p for name, p in method.model.named_parameters() if p.requires_grad}
    optimizer = [p for group in method.optimizer.param_groups for p in group["params"]]
    if (set(actual) != set(method.model_state) or len({id(p) for p in optimizer}) != len(optimizer)
            or {id(p) for p in optimizer} != {id(p) for p in actual.values()}):
        raise RuntimeError("Actual trainable/optimizer/snapshot parameter membership differs")
    if any(p.grad is not None and torch.count_nonzero(p.grad).item() for p in actual.values()):
        raise RuntimeError("Policy reset left nonzero parameter gradients")
    record = dict(trainable_sha256=state_digest(actual),
                  optimizer_sha256=state_digest(method.optimizer.state_dict()))
    if (record["trainable_sha256"] != state_digest(method.model_state)
            or record["optimizer_sha256"] != state_digest(method.optimizer_state)):
        raise RuntimeError("Trainable weights or optimizer did not restore exactly")
    return record


def rng_state(device):
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def restore_rng(state, device):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state(state[3], device)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def label_predictions(method, images, meta, config, mapping):
    from trace_ovss.cli import collapse_class_extensions
    from trace_ovss.evaluation import aggregate_patch_logits
    logits = method.evaluate(images)
    outputs = aggregate_patch_logits(logits, meta["patch_grid_shape"], meta["img_shape"],
                                     config["patch_size"], config["patch_stride"])
    predictions = []
    for output in outputs:
        probabilities = output.softmax(0)
        if mapping is not None:
            probabilities = collapse_class_extensions(probabilities, mapping)
        predictions.append(probabilities.argmax(0).cpu().numpy().astype(np.uint16))
    return predictions


def verify_reference(method, images, valid, image_ids, meta, config, mapping, *, atol, max_label_fraction):
    """Compare complete adaptation on the actual episode, never a scored GT panel."""
    from trace_ovss.method.trace import TRACE
    from trace_ovss.method.two_view import TwoViewTRACE
    from tools.trace_algorithm_variants import make_algorithm_class
    fixed_type = type(method)
    state = rng_state(method.device)
    outcomes = []
    try:
        for cls in (make_algorithm_class(TRACE, "quality_only"), TwoViewTRACE):
            method.__class__ = cls
            method.reset()
            restored_state(method)
            restore_rng(state, method.device)
            losses = method.adapt(images, valid_mask=valid, image_ids=image_ids)
            stats = copy.deepcopy(method.last_candidate_stats)
            predictions = label_predictions(method, images, meta, config, mapping)
            outcomes.append((losses, stats, predictions))
    finally:
        method.__class__ = fixed_type
        method.reset()
        restore_rng(state, method.device)
    a, b = outcomes
    masks_equal = (len(a[1]) == len(b[1]) and all(
        all(x[key] == y[key] for key in ("batch_idx", "image_idx", "class_idx"))
        and np.array_equal(x["region"], y["region"]) for x, y in zip(a[1], b[1])))
    support_diff = max((abs(x[key] - y[key]) for x, y in zip(a[1], b[1])
                        for key in ("target_support", "counterfactual_support", "reliability")), default=0.0)
    same_steps = len(a[0]) == len(b[0])
    loss_diff = max((abs(x - y) for x, y in zip(a[0], b[0])), default=0.0)
    pixels = sum(x.size for x in a[2])
    changed = sum(np.count_nonzero(x != y) for x, y in zip(a[2], b[2]))
    report = dict(candidate_masks_equal=masks_equal, same_steps=same_steps,
                  max_support_abs_difference=support_diff, max_loss_abs_difference=loss_diff,
                  changed_label_pixels=int(changed), total_pixels=pixels,
                  changed_label_fraction=changed / pixels, atol=atol,
                  max_label_fraction=max_label_fraction, device=str(method.device))
    if method.device.type == "cpu":
        atol = max_label_fraction = 0.0
    report["passed"] = bool(masks_equal and same_steps and support_diff <= atol
                            and loss_diff <= atol and changed / pixels <= max_label_fraction)
    return report


def execute_predictions(method, data, indices, config, mapping, variants, output, *, verify=False,
                        reference_atol=1e-5, reference_max_label_fraction=0.0):
    from trace_ovss.cli import build_patch_image_ids, build_patch_valid_mask
    from tools.trace_execution_audit import ExecutionAudit
    output = Path(output)
    for name in ["pre", *(entry[0] for entry in variants)]:
        (output / "predictions" / name).mkdir(parents=True)
    (output / "episodes").mkdir()
    method.reset()
    initial = restored_state(method)
    frozen = [(p, p._version) for p in method.model.parameters() if not p.requires_grad]
    costs = []
    with ExecutionAudit(method) as audit, (output / "cost.jsonl").open("x") as cost_file:
        for episode, start in enumerate(range(0, len(indices), config["batch_size"])):
            chosen = indices[start:start + config["batch_size"]]
            batch = image_collate([data[index] for index in chosen])
            images = batch["img_patches"].to(method.device)
            meta = batch["meta"]
            valid = build_patch_valid_mask(meta, config["patch_size"], config["patch_stride"], method.device)
            image_ids = build_patch_image_ids(meta, method.device)
            if verify:
                # Remove captured bound methods before switching the reference class.
                audit.close()
                try:
                    report = verify_reference(method, images, valid, image_ids, meta, config, mapping,
                                              atol=reference_atol, max_label_fraction=reference_max_label_fraction)
                finally:
                    audit.__init__(method)
                write_json(output / "episodes" / f"{episode:06d}.reference.json", report)
                if not report["passed"]:
                    raise RuntimeError("Reference equivalence gate failed; actual differences preserved")
            method.reset()
            audit.reset()
            synchronize(method.device)
            tick = time.perf_counter()
            pre = label_predictions(method, images, meta, config, mapping)
            synchronize(method.device)
            pre_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            stats = method.diagnose(images, valid, image_ids)
            synchronize(method.device)
            diagnostic_seconds = time.perf_counter() - tick
            shared_counts = audit.snapshot()
            random_state = rng_state(method.device)
            for offset, prediction in enumerate(pre):
                np.save(output / "predictions" / "pre" / f"{start + offset:06d}.npy", prediction, allow_pickle=False)
            weights, donors, resets = [], [], {}
            for label, policy, permutation_seed in variants:
                method.reset()
                reset = restored_state(method)
                if reset != initial or any(p._version != version for p, version in frozen):
                    raise RuntimeError("Policy source model or optimizer initialization changed")
                resets[label] = reset
                restore_rng(random_state, method.device)
                method.select_policy(policy, permutation_seed=permutation_seed or 0, episode=episode)
                weights.append(method.fixed_weights.copy())
                donors.append(method.fixed_donors.copy())
                audit.reset()
                if method.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(method.device)
                synchronize(method.device)
                tick = time.perf_counter()
                losses = method.adapt(images, valid_mask=valid, image_ids=image_ids)
                synchronize(method.device)
                adapted = time.perf_counter()
                post = label_predictions(method, images, meta, config, mapping)
                synchronize(method.device)
                evaluated = time.perf_counter()
                if not np.isfinite(losses).all():
                    raise RuntimeError("Nonfinite adaptation loss")
                for offset, prediction in enumerate(post):
                    np.save(output / "predictions" / label / f"{start + offset:06d}.npy", prediction, allow_pickle=False)
                cost = dict(episode=episode, policy=policy, policy_label=label, permutation_seed=permutation_seed,
                            images=len(chosen), steps=len(losses), adaptation_seconds=adapted - tick,
                            evaluation_seconds=evaluated - adapted,
                            peak_allocated_mib=(torch.cuda.max_memory_allocated(method.device) / 2**20
                                                if method.device.type == "cuda" else None),
                            **audit.snapshot())
                cost_file.write(json.dumps(cost, allow_nan=False) + "\n")
                cost_file.flush()
                costs.append(cost)
                write_json(output / "progress.json", dict(status="predicting", episode=episode,
                           policy=label, images_completed=start, total_images=len(indices)))
            if candidate_digest(stats) != method.fixed_digest:
                raise RuntimeError("Candidate cache changed during policy evaluation")
            if any(p._version != version for p, version in frozen):
                raise RuntimeError("Frozen model parameters changed during adaptation")
            masks = np.stack([row["region"] for row in stats]) if stats else np.empty((0, 0, 0), dtype=bool)
            np.savez_compressed(output / "episodes" / f"{episode:06d}.npz", masks=masks,
                image_index=np.array([start + row["image_idx"] for row in stats], dtype=np.int64),
                patch_index=np.array([row["batch_idx"] for row in stats], dtype=np.int64),
                query_index=np.array([row["class_idx"] for row in stats], dtype=np.int64),
                confidence=np.array([row["confidence"] for row in stats]),
                T=np.array([row["target_support"] for row in stats]),
                C=np.array([row["counterfactual_support"] for row in stats]),
                q=np.array([min(row["target_support"], row["counterfactual_support"]) for row in stats]),
                rho=np.stack(weights), donor_candidate_index=np.stack(donors))
            write_json(output / "episodes" / f"{episode:06d}.json", dict(
                episode=episode, image_indices=list(range(start, start + len(chosen))),
                dataset_indices=chosen, meta=meta, candidate_sha256=method.fixed_digest,
                candidate_count=len(stats), candidate_budget=config["trace_max_regions"],
                policy_labels=[entry[0] for entry in variants], reset_assertions=resets,
                shared_pre_seconds=pre_seconds, shared_diagnostic_seconds=diagnostic_seconds,
                shared_work=shared_counts,
                timing_scope="Shared pre/diagnosis once; policy adaptation/evaluation separately. Not standalone method cost."))
            write_json(output / "progress.json", dict(status="predicting", images_completed=start + len(chosen),
                       total_images=len(indices), episodes_completed=episode + 1))
    return costs


def seal_predictions(output, protocol):
    output = Path(output)
    if (output / "PREDICTIONS_SEALED.json").exists():
        raise FileExistsError("Prediction seal already exists")
    expected = {f"predictions/{policy}/{index:06d}.npy" for index in range(len(protocol["inputs"]))
                for policy in ["pre", *(entry[0] for entry in protocol["variants"]) ]}
    predictions = {str(path.relative_to(output)) for path in (output / "predictions").rglob("*.npy")}
    if predictions != expected:
        raise RuntimeError("Cannot seal incomplete predictions")
    for episode in range((len(protocol["inputs"]) + protocol["config"]["batch_size"] - 1) // protocol["config"]["batch_size"]):
        for suffix in ("json", "npz"):
            if not (output / "episodes" / f"{episode:06d}.{suffix}").is_file():
                raise RuntimeError("Cannot seal incomplete candidate episodes")
    files = [output / "prediction_protocol.json", output / "cost.jsonl", output / "prediction_result.json",
             output / "initial_state.json", output / "checkpoint_identity.json"]
    if (output / "expected_panel.json").exists():
        files.append(output / "expected_panel.json")
    files += sorted((output / "predictions").rglob("*.npy"))
    files += sorted((output / "episodes").iterdir())
    write_json(output / "PREDICTIONS_SEALED.json", dict(
        format_version=1, images=len(protocol["inputs"]),
        artifact_sha256={str(path.relative_to(output)): sha256(path) for path in files},
        gt_contents_opened=False, complete=True))


def verify_prediction_seal(output):
    output = Path(output).resolve()
    seal = read_json(output / "PREDICTIONS_SEALED.json")
    for relative, digest in seal["artifact_sha256"].items():
        path = output / relative
        if not path.resolve().is_relative_to(output) or sha256(path) != digest:
            raise RuntimeError(f"Prediction seal mismatch: {relative}")
    protocol = read_json(output / "prediction_protocol.json")
    parent_hash = protocol.get("expected_panel_sha256")
    if parent_hash is not None and sha256(output / "expected_panel.json") != parent_hash:
        raise RuntimeError("Parent panel binding changed")
    images = len(protocol["inputs"])
    expected = {f"predictions/{policy}/{index:06d}.npy" for index in range(images)
                for policy in ["pre", *(entry[0] for entry in protocol["variants"]) ]}
    actual = {str(path.relative_to(output)) for path in (output / "predictions").rglob("*.npy")}
    if (not seal.get("complete") or seal["images"] != images or actual != expected
            or not expected.issubset(seal["artifact_sha256"])):
        raise RuntimeError("Prediction population or seal coverage is incomplete")
    required = {"prediction_protocol.json", "cost.jsonl", "initial_state.json", "prediction_result.json",
                "checkpoint_identity.json"}
    if not required.issubset(seal["artifact_sha256"]):
        raise RuntimeError("Prediction seal lacks protocol/state coverage")
    return protocol


def streaming_counts(prediction, gt, classes):
    if prediction.shape != gt.shape or prediction.ndim != 2:
        raise ValueError("Prediction/GT shapes differ")
    if not np.issubdtype(prediction.dtype, np.integer) or np.any((prediction < 0) | (prediction >= classes)):
        raise ValueError("Invalid predicted labels")
    valid = gt != 255
    if np.any((gt[valid] < 0) | (gt[valid] >= classes)):
        raise ValueError("Invalid GT labels")
    pairs = gt[valid].astype(np.int64) * classes + prediction[valid].astype(np.int64)
    matrix = np.bincount(pairs, minlength=classes * classes).reshape(classes, classes)
    intersection = matrix.diagonal().copy()
    union = matrix.sum(0) + matrix.sum(1) - intersection
    return intersection, union


def score(output, *, dataset=None, seed=None, data_root=None):
    output = Path(output)
    protocol = verify_prediction_seal(output)
    for key, value in (("dataset", dataset), ("seed", seed), ("data_root", str(Path(data_root).resolve()) if data_root else None)):
        if value is not None and protocol[key] != value:
            raise RuntimeError(f"Score request differs from sealed {key}")
    if any((output / name).exists() for name in ("result.json", "SCORED_SEALED.json", "scores_by_episode.jsonl")):
        raise FileExistsError("Scoring is not automatically retried or overwritten")
    classes = len(protocol["classes"])
    names = ["pre", *(entry[0] for entry in protocol["variants"])]
    totals = {name: [np.zeros(classes, dtype=np.int64), np.zeros(classes, dtype=np.int64)] for name in names}
    label_hashes = {}
    batch_size = protocol["config"]["batch_size"]
    with (output / "scores_by_episode.jsonl").open("x") as records:
        for start in range(0, len(protocol["inputs"]), batch_size):
            episode_counts = {name: [np.zeros(classes, dtype=np.int64), np.zeros(classes, dtype=np.int64)] for name in names}
            for index in range(start, min(start + batch_size, len(protocol["inputs"]))):
                row = protocol["inputs"][index]
                path = Path(row["label"])
                if [path.stat().st_size, path.stat().st_mtime_ns] != row["label_metadata"]:
                    raise RuntimeError("GT identity changed since prediction enumeration")
                content = path.read_bytes()
                label_hashes[str(path)] = hashlib.sha256(content).hexdigest()
                pre = np.load(output / "predictions" / "pre" / f"{index:06d}.npy", allow_pickle=False)
                with Image.open(io.BytesIO(content)) as source:
                    gt = np.asarray(source.resize((pre.shape[1], pre.shape[0]), Image.Resampling.NEAREST))
                if gt.ndim == 3:
                    gt = gt[..., 0]
                gt = gt.astype(np.int64, copy=True)
                if protocol["reduce_zero_label"]:
                    background, valid = gt == 0, gt != 255
                    gt[valid] -= 1
                    gt[background] = 255
                for name in names:
                    prediction = pre if name == "pre" else np.load(
                        output / "predictions" / name / f"{index:06d}.npy", allow_pickle=False)
                    counts = streaming_counts(prediction, gt, classes)
                    for j in range(2):
                        totals[name][j] += counts[j]
                        episode_counts[name][j] += counts[j]
            records.write(json.dumps(dict(episode=start // batch_size,
                image_indices=list(range(start, min(start + batch_size, len(protocol["inputs"])))),
                policies={name: dict(intersection=i.tolist(), union=u.tolist())
                          for name, (i, u) in episode_counts.items()})) + "\n")
            write_json(output / "progress.json", dict(status="scoring", images_completed=min(start + batch_size, len(protocol["inputs"]))))
    policy_results = []
    for label, policy, permutation_seed in [("pre", "pre", None), *protocol["variants"]]:
        intersection, union = totals[label]
        valid = union > 0
        miou = float(100 * np.mean(intersection[valid] / union[valid])) if valid.any() else None
        policy_results.append(dict(policy=policy, policy_label=label, permutation_seed=permutation_seed,
                                   miou_unrounded=miou, intersection=intersection.tolist(), union=union.tolist()))
    result = dict(status="completed", dataset=protocol["dataset"], seed=protocol["seed"],
                  images=len(protocol["inputs"]), full_split=protocol["full_split"],
                  policy_results=[row for row in policy_results if row["policy"] != "pre"],
                  pre_result=policy_results[0], prediction_seal_sha256=sha256(output / "PREDICTIONS_SEALED.json"),
                  label_contents_first_opened_after_prediction_seal=True, timing_scope=protocol["timing_scope"])
    write_json(output / "result.json", result)
    write_json(output / "score_inputs.json", dict(label_sha256=label_hashes))
    write_json(output / "SCORED_SEALED.json", dict(prediction_seal_sha256=result["prediction_seal_sha256"],
        artifact_sha256={name: sha256(output / name) for name in
                        ("result.json", "score_inputs.json", "scores_by_episode.jsonl")}))
    write_json(output / "progress.json", dict(status="completed", images_completed=result["images"]))
    return result


def bind_expected_panel(output, actual, *, max_images):
    path = Path(output) / "expected_panel.json"
    if not path.exists():
        if max_images is None:
            raise RuntimeError("Full runs require controller expected_panel.json; standalone smoke needs --max-images")
        return None
    expected = read_json(path)
    for key in ("dataset", "split", "seed", "selection_seed", "image_ids", "label_ids",
                "file_metadata", "vocabulary_sha256", "config", "available_images", "image_sha256"):
        if expected.get(key) != actual[key]:
            raise RuntimeError(f"Parent expected_panel differs: {key}")
    return sha256(path)


def predict(args):
    runtime_state = configure_deterministic_runtime()
    root = args.package_root.resolve()
    sys.path.insert(0, str(root))
    from trace_ovss.configs import VISION_OUTPUTS, best_config
    from trace_ovss.data.datasets import SegmentationPatchDataset
    from trace_ovss.method.two_view import TwoViewTRACE
    from tools.suite_inputs import validate_discovery, validate_population
    if not Path(inspect.getfile(TwoViewTRACE)).resolve().is_relative_to(root):
        raise RuntimeError("TwoViewTRACE import is outside --package-root")
    config = best_config(args.dataset, args.data_root)
    if args.data_dir:
        config["data_dir"] = args.data_dir.resolve()
    data = image_only_dataset_class(SegmentationPatchDataset)(config["dataset"], config["data_dir"],
        resize=config["resize"], patch_size=config["patch_size"], patch_stride=config["patch_stride"], include_auxiliary=False)
    validate_discovery(data)
    validate_population(args.dataset, "evaluation", args.max_images, len(data))
    indices = (sorted(np.random.default_rng(args.selection_seed).permutation(len(data))[:args.max_images].tolist())
               if args.max_images is not None else list(range(len(data))))
    variants = expand_policies(args.policies, args.permutation_seeds)
    sources = source_hashes(root)
    inputs = [dict(image=str(data.samples[index][0]), label=str(data.samples[index][1]),
                   image_sha256=sha256(data.samples[index][0]),
                   label_metadata=[data.samples[index][1].stat().st_size, data.samples[index][1].stat().st_mtime_ns])
              for index in indices]
    data.expected_image_hashes = {row["image"]: row["image_sha256"] for row in inputs}
    vocabulary = data.vocabulary
    classes = vocabulary.extensions if config["class_extensions"] else vocabulary.classes
    mapping = vocabulary.extension_to_class if config["class_extensions"] else None
    panel = dict(dataset=args.dataset, split="evaluation", seed=args.seed, selection_seed=args.selection_seed,
        image_ids=[row["image"] for row in inputs], label_ids=[row["label"] for row in inputs],
        file_metadata=[[path.stat().st_size, path.stat().st_mtime_ns] for index in indices for path in data.samples[index]],
        vocabulary_sha256=hashlib.sha256(json.dumps([classes, mapping]).encode()).hexdigest(),
        config=json.loads(json.dumps({**config, "data_dir": str(config["data_dir"])})), available_images=len(data),
        image_sha256={row["image"]: row["image_sha256"] for row in inputs})
    parent_hash = bind_expected_panel(args.output, panel, max_images=args.max_images)
    if parent_hash is not None:
        data.expected_image_hashes = read_json(args.output / "expected_panel.json")["image_sha256"]
    checkpoint = checkpoint_identity(args.checkpoint)
    protocol = dict(dataset=args.dataset, seed=args.seed, images=len(indices), full_split=len(indices) == len(data),
        available_images=len(data), selection="suite_inputs random subset sorted by official sample index; no GT contents", inputs=inputs,
        data_root=str(args.data_root.resolve()), package_root=str(root),
        config={**config, "data_dir": str(config["data_dir"])}, classes=list(vocabulary.classes),
        queries=list(classes), query_to_class=mapping, reduce_zero_label=data.spec.reduce_zero_label,
        variants=variants, source_sha256=sources, reference_verification=args.verify_reference,
        panel=panel, expected_panel_sha256=parent_hash, checkpoint=checkpoint,
        deterministic_runtime=runtime_state,
        timing_scope="Shared pre/diagnosis once; separate policy adaptation/evaluation. Not standalone formal cost.")
    args.output.mkdir(parents=True, exist_ok=True)
    owned = ("prediction_protocol.json", "predictions", "episodes", "cost.jsonl", "initial_state.json",
             "prediction_result.json", "result.json", "PREDICTIONS_SEALED.json", "SCORED_SEALED.json",
             "score_inputs.json", "scores_by_episode.jsonl", "checkpoint_identity.json")
    if any((args.output / name).exists() for name in owned):
        raise FileExistsError("Runner-owned output already exists; no automatic overwrite/retry")
    write_json(args.output / "prediction_protocol.json", protocol)
    write_json(args.output / "checkpoint_identity.json", checkpoint)
    write_json(args.output / "progress.json", dict(status="initializing", images_completed=0, total_images=len(indices)))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cls = make_fixed_candidate_class(TwoViewTRACE)
    method = cls(classes=classes, lr=config["lr"], steps=config["steps"],
                 trace_max_regions=config["trace_max_regions"], trace_view_batch_size=config["trace_view_batch_size"],
                 vision_outputs=VISION_OUTPUTS, device="cuda", backbone=checkpoint["path"])
    checkpoint_stat = Path(checkpoint["path"]).stat()
    if [checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns] != [checkpoint["bytes"], checkpoint["mtime_ns"]]:
        raise RuntimeError("Checkpoint changed during model initialization")
    write_json(args.output / "initial_state.json", dict(model_sha256=state_digest(method.model.state_dict()),
        optimizer_sha256=state_digest(method.optimizer_state), adapted_parameter_names=method.adapted_parameter_names,
        checkpoint=checkpoint,
        environment=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                         pillow=Image.__version__, torch=str(torch.__version__), cuda=torch.version.cuda,
                         cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(method.device),
                         gpu_inventory=subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version",
                                                               "--format=csv,noheader"], text=True).strip(),
                         matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                         cudnn_tf32=torch.backends.cudnn.allow_tf32,
                         float32_matmul_precision=torch.get_float32_matmul_precision(),
                         **deterministic_runtime_state())))
    execute_predictions(method, data, indices, config, mapping, variants, args.output, verify=args.verify_reference,
                        reference_atol=args.reference_atol, reference_max_label_fraction=args.reference_max_label_fraction)
    if source_hashes(root) != sources or any(sha256(row["image"]) != row["image_sha256"] for row in inputs):
        raise RuntimeError("Source or input images changed during prediction")
    if parent_hash is not None and sha256(args.output / "expected_panel.json") != parent_hash:
        raise RuntimeError("Parent expected_panel changed during prediction")
    if deterministic_runtime_state() != runtime_state:
        raise RuntimeError("Deterministic runtime settings changed during prediction")
    write_json(args.output / "prediction_result.json", dict(status="predicted", images=len(indices),
        dataset=args.dataset, seed=args.seed, full_split=protocol["full_split"],
        policy_labels=[entry[0] for entry in variants], timing_scope=protocol["timing_scope"]))
    seal_predictions(args.output, protocol)
    write_json(args.output / "progress.json", dict(status="predictions_sealed", images_completed=len(indices)))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("package-root", "data-root", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--data-dir", type=Path)
    result.add_argument("--checkpoint", type=Path, help="Existing official ViT-L/14 file; never downloads")
    result.add_argument("--dataset", choices=("coco_cd", "pc60", "voc20", "voc21", "coco_object", "coco_stuff"), required=True)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--selection-seed", type=int, default=20261024)
    result.add_argument("--max-images", type=int)
    result.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    result.add_argument("--permutation-seeds", nargs="+", type=int, default=[0, 1, 2])
    result.add_argument("--phase", choices=("predict", "score", "run"), default="run")
    result.add_argument("--verify-reference", action="store_true")
    result.add_argument("--reference-atol", type=float, default=1e-5)
    result.add_argument("--reference-max-label-fraction", type=float, default=0.0)
    return result


def main(argv=None):
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    if args.seed < 0 or args.selection_seed < 0 or (args.max_images is not None and args.max_images < 1):
        arg_parser.error("Seed must be nonnegative and max-images positive")
    if (not np.isfinite(args.reference_atol) or args.reference_atol < 0
            or not 0 <= args.reference_max_label_fraction <= 1):
        arg_parser.error("Invalid reference tolerances")
    expand_policies(args.policies, args.permutation_seeds)
    if args.phase in ("predict", "run"):
        predict(args)
    if args.phase in ("score", "run"):
        score(args.output, dataset=args.dataset, seed=args.seed, data_root=args.data_root)


if __name__ == "__main__":
    main()
