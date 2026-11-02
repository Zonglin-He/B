"""Image-budget experiment with explicit file lists and stage-wide GT barriers."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evolution import run_cell as shared
from tools.evolution_followup import standalone as prior
from trace_ovss.configs import BEST_CONFIGS, VISION_OUTPUTS
from trace_ovss.data.datasets import SegmentationPatchDataset
from trace_ovss.method.image_balanced import ImageBalancedTRACE
from trace_ovss.method.trace.constants import EPS
from trace_ovss.method.trace.objective import TraceObjectiveMixin

ARMS = ("episode_min", "image_uniform", "image_confidence", "image_T", "image_C", "image_min")
COUNTS = {"calibration": 128, "confirmation": 256}
DATASETS = ("pc60", "coco_object")
TIMING_SCOPE = prior.TIMING_SCOPE
require, read_json, write_json = prior.require, prior.read_json, prior.write_json
sha256, canonical_hash = shared.sha256, prior.canonical_hash


class ExperimentTRACE(ImageBalancedTRACE):
    """One resettable model; the arm selects a declared objective and probe."""

    @property
    def arm(self):
        return getattr(self, "_arm", "image_min")

    @arm.setter
    def arm(self, value):
        require(value in ARMS, "Unknown confirmation arm")
        self._arm = value
        self.view_mode = value.split("_", 1)[1] if value in ("image_T", "image_C") else "min"

    @torch.no_grad()
    def evaluate_trace_views(self, images, original_probs, candidates):
        if self.arm in ("image_uniform", "image_confidence"):
            # Only the zero-view branches of the prior helper are used.
            previous = self._arm
            self._arm = "Uniform" if previous == "image_uniform" else "Confidence"
            try:
                return prior.StandaloneTRACE.evaluate_trace_views(self, images, original_probs, candidates)
            finally:
                self._arm = previous
        return super().evaluate_trace_views(images, original_probs, candidates)

    def prepare_trace_objective(self, entropy, terms):
        if self.arm == "episode_min":
            return TraceObjectiveMixin.prepare_trace_objective(entropy, terms)
        return super().prepare_trace_objective(entropy, terms)

    def trace_region_normalized_entropy(self, entropy, terms, *, prepared=None):
        if self.arm == "episode_min":
            return TraceObjectiveMixin.trace_region_normalized_entropy(entropy, terms, prepared=prepared)
        return super().trace_region_normalized_entropy(entropy, terms, prepared=prepared)

    _count = prior.StandaloneTRACE._count
    count_work = prior.StandaloneTRACE.count_work
    _forward_trace_view_batches = prior.StandaloneTRACE._forward_trace_view_batches

    def forward_eval_logits(self, images, *args, **kwargs):
        if hasattr(self, "work"):
            phase = "diagnostic" if getattr(self, "_diagnostic", False) else "eval"
            self._count(phase + "_forward", len(images))
        return super().forward_eval_logits(images, *args, **kwargs)

    def forward_adapt_logits(self, images):
        if hasattr(self, "work"):
            self._count("adapt_forward", len(images))
        return super().forward_adapt_logits(images)


class NativeGateReference(prior.TwoViewTRACE):
    """Unchanged native mathematics, with the declared fail-closed OOM policy."""

    _forward_trace_view_batches = prior.StandaloneTRACE._forward_trace_view_batches


def expected_views(arm, candidates):
    require(arm in ARMS and type(candidates) is int and candidates >= 0, "Invalid view-count request")
    return candidates * (0 if arm in ("image_uniform", "image_confidence")
                         else 1 if arm in ("image_T", "image_C") else 2)


def objective_coefficients(stats, arm):
    """Receipt coefficients, not independent per-candidate effect estimates."""
    active = [r["reliability"] > EPS for r in stats]
    groups = {int(r["image_idx"]) for r, keep in zip(stats, active) if keep}
    if not groups:
        return [0.0] * len(stats)
    if arm == "episode_min":
        total = sum(float(r["reliability"]) for r, keep in zip(stats, active) if keep)
        return [float(r["reliability"]) / total if keep else 0.0 for r, keep in zip(stats, active)]
    totals = {image: sum(float(r["reliability"]) for r, keep in zip(stats, active)
                         if keep and int(r["image_idx"]) == image) for image in groups}
    return [float(r["reliability"]) / totals[int(r["image_idx"])] / len(groups) if keep else 0.0
            for r, keep in zip(stats, active)]


def checked_sources():
    sources = shared.source_hashes(ROOT)
    for tree in (ROOT / "tools/evolution_confirmation", ROOT / "tools/evolution_followup"):
        for path in sorted(tree.glob("*.py")):
            sources[str(path.resolve())] = sha256(path)
    for path in sorted((ROOT / "tests").glob("test_*confirmation*.py")):
        sources[str(path.resolve())] = sha256(path)
    for path in (ROOT / "tests/test_image_balanced_trace.py",):
        require(path.is_file(), "Missing new-method tests")
        sources[str(path.resolve())] = sha256(path)
    require(Path(inspect.getfile(ImageBalancedTRACE)).resolve().is_relative_to(ROOT), "Wrong method import")
    return sources


def verify_label_metadata(row):
    """Stat only; annotation bytes remain unopened until the score phase."""
    status = Path(row["label"]).stat()
    expected = row["label_metadata"]
    if isinstance(expected, dict):
        actual = dict(size_bytes=status.st_size, mtime_ns=status.st_mtime_ns,
                      device=status.st_dev, inode=status.st_ino, contents_read=False)
    else:
        actual = [status.st_size, status.st_mtime_ns]
    require(actual == expected, "Label metadata changed")


def stage_rows(bundle, dataset, stage):
    require(dataset in DATASETS and stage in COUNTS, "Unknown dataset/stage")
    value = bundle["datasets"][dataset]
    rows = value[stage]
    require(len(rows) == COUNTS[stage], "Wrong cohort population")
    required = {"official_id", "image", "label", "image_sha256", "label_metadata"}
    require(all(required <= set(row) for row in rows), "Incomplete explicit cohort row")
    require(len({str(Path(row["image"]).resolve()) for row in rows}) == len(rows), "Duplicate cohort image")
    require(len({row["official_id"] for row in rows}) == len(rows), "Duplicate official cohort identity")
    other = value["confirmation" if stage == "calibration" else "calibration"]
    require(not ({str(Path(r["image"]).resolve()) for r in rows}
                 & {str(Path(r["image"]).resolve()) for r in other}), "Calibration/confirmation overlap")
    for row in rows:
        for key in ("image", "label"):
            require(Path(row[key]).is_absolute() and Path(row[key]).is_file(), "Missing explicit input")
        verify_label_metadata(row)
    return rows


def reordered(rows, seed):
    if seed is None:
        return list(rows)
    require(type(seed) is int and seed in (20261030, 20261031), "Order is not prelocked")
    return [rows[i] for i in np.random.default_rng(seed).permutation(len(rows)).tolist()]


def make_data(config, rows):
    image_base = shared.image_only_dataset_class(SegmentationPatchDataset)

    class ExplicitImageDataset(image_base):
        def _discover_samples(self):
            return [(Path(row["image"]), Path(row["label"])) for row in rows]

    data = ExplicitImageDataset(config["dataset"], config["data_dir"], resize=config["resize"],
        patch_size=config["patch_size"], patch_stride=config["patch_stride"], include_auxiliary=False)
    data.expected_image_hashes = {row["image"]: row["image_sha256"] for row in rows}
    return data


def panel(config, rows):
    data = make_data(config, rows)
    queries = list(data.vocabulary.extensions if config["class_extensions"] else data.vocabulary.classes)
    mapping = list(data.vocabulary.extension_to_class) if config["class_extensions"] else None
    return dict(inputs=rows, ordered_ids=[r["official_id"] for r in rows], full_split=False,
        classes=list(data.vocabulary.classes), queries=queries, query_to_class=mapping,
        reduce_zero_label=data.spec.reduce_zero_label, vocabulary_sha256=canonical_hash([queries, mapping]))


def prepare(args):
    bundle, config = read_json(args.cohorts), read_json(args.config)
    prior.validate_config(config, args.dataset)
    require(args.stage != "calibration" or args.order_seed is None, "Calibration has one fixed order")
    rows = reordered(stage_rows(bundle, args.dataset, args.stage), args.order_seed)
    manifest = dict(format_version=1, worker="image_budget_confirmation", dataset=args.dataset,
        stage=args.stage, max_images=len(rows), seed=0, arms=list(ARMS), arm_order="alternating",
        config=config, config_path=str(args.config.resolve()), config_sha256=sha256(args.config),
        cohorts_path=str(args.cohorts.resolve()), cohort_sha256=sha256(args.cohorts),
        suite_root=str(args.manifest.resolve().parent.parent), order_seed=args.order_seed,
        panel=panel(config, rows), source_sha256=checked_sources(),
        checkpoint=shared.checkpoint_identity(args.checkpoint), timing_scope=TIMING_SCOPE,
        exposure_scope=bundle.get("exposure_scope", "Prospective relative to documented exclusions, not exhaustive never-seen certification"))
    validate_manifest(manifest)
    write_json(args.manifest, manifest)
    print(canonical_hash(manifest))


def validate_manifest(manifest):
    require(manifest["format_version"] == 1 and manifest["worker"] == "image_budget_confirmation", "Wrong worker")
    require(manifest["dataset"] in DATASETS and manifest["stage"] in COUNTS, "Invalid stage")
    require(manifest["seed"] == 0 and manifest["arms"] == list(ARMS)
            and manifest["arm_order"] == "alternating", "Changed arm/seed protocol")
    require(manifest["max_images"] == COUNTS[manifest["stage"]]
            and manifest["panel"]["full_split"] is False, "Not fixed cohort")
    prior.validate_config(manifest["config"], manifest["dataset"])
    require(manifest["timing_scope"] == TIMING_SCOPE, "Changed timing scope")
    require(sha256(manifest["cohorts_path"]) == manifest["cohort_sha256"], "Cohort manifest changed")
    bundle = read_json(manifest["cohorts_path"])
    expected = reordered(stage_rows(bundle, manifest["dataset"], manifest["stage"]), manifest["order_seed"])
    require(panel(manifest["config"], expected) == manifest["panel"], "Explicit panel/vocabulary changed")
    require(manifest["stage"] != "calibration" or manifest["order_seed"] is None, "Calibration order changed")


def verify_manifest(path, expected_hash):
    require(sha256(path) == expected_hash, "Manifest SHA mismatch")
    manifest = read_json(path)
    validate_manifest(manifest)
    return manifest


def verify_selection_record(root, selection):
    """Recompute the global calibration-only choice before confirmation."""
    from tools.evolution_confirmation.queue import select_mode

    rule = read_json(root / "frozen_plan.json")["selection"]
    require(selection["rule"] == rule, "Selection rule differs from frozen plan")
    require(selection["suite_prepared_sha256"] == sha256(root / "SUITE_PREPARED.json"), "Selection suite binding differs")
    require(selection["calibration_barrier_sha256"]
            == sha256(root / "ALL_CALIBRATION_PREDICTIONS_SEALED.json"), "Selection calibration barrier differs")
    results = {}
    require(len(selection["calibration_score_sha256"]) == 2, "Selection needs two calibration scores")
    for relative, digest in selection["calibration_score_sha256"].items():
        cell = root / relative
        require(cell.resolve().is_relative_to(root.resolve()), "Selection score path escapes suite")
        require(sha256(cell / "SCORED_SEALED.json") == digest, "Calibration score seal drift")
        seal = read_json(cell / "SCORED_SEALED.json")
        for name, expected in seal["artifact_sha256"].items():
            path = cell / name
            require(path.resolve().is_relative_to(cell.resolve()) and sha256(path) == expected,
                    "Calibration scored artifact drift")
        result = read_json(cell / "result.json")
        require(result["stage"] == "calibration" and result["dataset"] in DATASETS
                and result["dataset"] not in results, "Selection score dataset/stage differs")
        results[result["dataset"]] = result
    mode, candidates = select_mode(results, rule)
    require(mode == selection["selected_mode"] and candidates == selection["candidates"],
            "Global mode is not the locked calibration-only choice")


def selection_before_confirmation(manifest):
    if manifest["stage"] != "confirmation":
        return None
    root = Path(manifest["suite_root"])
    path = root / "SELECTION_SEALED.json"
    selection = read_json(path)
    require(selection["complete"] is True and selection["selected_mode"] in ("T", "C", "min")
            and selection["cohort_sha256"] == manifest["cohort_sha256"], "Unsealed/wrong global selection")
    verify_selection_record(root, selection)
    return dict(path=str(path), sha256=sha256(path), selected_mode=selection["selected_mode"])


def artifacts(manifest):
    n, batch = manifest["max_images"], manifest["config"]["batch_size"]
    return ({f"predictions/{a}/{i:06d}.npy" for a in ARMS for i in range(n)}
            | {f"episodes/{e:06d}.{a}.json" for a in ARMS for e in range((n + batch - 1) // batch)}
            | {"prediction_protocol.json", "initial_state.json", "prediction_result.json", "cost.jsonl"})


def execute(method, data, manifest, output):
    from trace_ovss.cli import build_patch_image_ids, build_patch_valid_mask
    config = manifest["config"]
    for arm in ARMS:
        (output / "predictions" / arm).mkdir(parents=True)
    (output / "episodes").mkdir()
    method.reset()
    initial = shared.restored_state(method)
    frozen = [(p, p._version) for p in method.model.parameters() if not p.requires_grad]
    costs = []
    with (output / "cost.jsonl").open("x") as stream:
        for episode, start in enumerate(range(0, len(data), config["batch_size"])):
            indices = range(start, min(start + config["batch_size"], len(data)))
            batch = shared.image_collate([data[i] for i in indices])
            images, meta = batch["img_patches"].to(method.device), batch["meta"]
            valid = build_patch_valid_mask(meta, config["patch_size"], config["patch_stride"], method.device)
            image_ids = build_patch_image_ids(meta, method.device)
            rng, identity = shared.rng_state(method.device), None
            order = prior.arm_order(ARMS, episode, "alternating")
            for position, arm in enumerate(order):
                method.arm = arm
                shared.restore_rng(rng, method.device)
                with method.count_work() as counts:
                    shared.synchronize(method.device)
                    tick = time.perf_counter()
                    losses = method.adapt(images, valid_mask=valid, image_ids=image_ids)
                    predictions = shared.label_predictions(method, images, meta, config, manifest["panel"]["query_to_class"])
                    shared.synchronize(method.device)
                    seconds = time.perf_counter() - tick
                stats = method.last_candidate_stats
                digest = prior.candidate_identity(stats)
                identity = digest if identity is None else identity
                require(digest == identity, "Candidate admission changed across arms")
                require(counts["diagnostic_forward_inputs"] == counts["diagnostic_encoder_inputs"]
                        == expected_views(arm, len(stats)), "View schedule differs")
                require(all(math.isfinite(v) for v in losses), "Nonfinite loss")
                require(not any(p._version != version for p, version in frozen), "Frozen model changed")
                coefficients = objective_coefficients(stats, arm)
                row = dict(episode=episode, arm=arm, order_position=position, images=len(indices),
                    patches=len(images), seconds=seconds, warmup=episode == 0, steps=len(losses),
                    candidate_count=len(stats), candidate_sha256=digest, **counts)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                costs.append(row)
                for offset, prediction in enumerate(predictions):
                    require(prediction.ndim == 2 and np.isfinite(prediction).all(), "Invalid reconstructed prediction")
                    with (output / "predictions" / arm / f"{start + offset:06d}.npy").open("xb") as destination:
                        np.save(destination, prediction, allow_pickle=False)
                write_json(output / "episodes" / f"{episode:06d}.{arm}.json", dict(**row, losses=losses,
                    ordered_ids=manifest["panel"]["ordered_ids"][start:start + len(indices)],
                    candidates=[dict(batch_idx=r["batch_idx"], image_idx=r["image_idx"], class_idx=r["class_idx"],
                        confidence=r["confidence"], T=r["target_support"], C=r["counterfactual_support"],
                        rho=r["reliability"], objective_coefficient=coefficients[i]) for i, r in enumerate(stats)]))
                method.reset()
                require(shared.restored_state(method) == initial, "Model/optimizer reset differs")
                del predictions, stats
    aggregate = {}
    for arm in ARMS:
        rows = [r for r in costs if r["arm"] == arm]
        aggregate[arm] = {key: sum(r[key] for r in rows) for key in rows[0] if key.endswith(("_inputs", "_calls"))}
    return prior.timing_summary(costs, ARMS), aggregate


def predict(args):
    manifest = verify_manifest(args.manifest, args.manifest_sha256)
    require(checked_sources() == manifest["source_sha256"], "Source drift")
    require(sha256(manifest["config_path"]) == manifest["config_sha256"]
            and read_json(manifest["config_path"]) == manifest["config"], "Config drift")
    selected = selection_before_confirmation(manifest)
    checkpoint = shared.checkpoint_identity(manifest["checkpoint"]["path"])
    require(checkpoint == manifest["checkpoint"], "Checkpoint drift")
    rows = manifest["panel"]["inputs"]
    data = make_data(manifest["config"], rows)
    args.output.mkdir(parents=True, exist_ok=False)
    with prior.gpu_lock() as lock:
        runtime = shared.configure_deterministic_runtime()
        protocol = dict(manifest=manifest, manifest_sha256=args.manifest_sha256,
            manifest_path=str(args.manifest.resolve()), gpu_lock=lock, deterministic_runtime=runtime,
            selection=selected)
        write_json(args.output / "prediction_protocol.json", protocol)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        cfg = manifest["config"]
        method = ExperimentTRACE(classes=manifest["panel"]["queries"], lr=cfg["lr"], steps=cfg["steps"],
            trace_max_regions=cfg["trace_max_regions"], trace_view_batch_size=cfg["trace_view_batch_size"],
            vision_outputs=VISION_OUTPUTS, device="cuda", backbone=checkpoint["path"], view_mode="min")
        require(shared.checkpoint_identity(checkpoint["path"]) == checkpoint, "Checkpoint changed loading")
        write_json(args.output / "initial_state.json", dict(model_sha256=prior.state_digest(method.model.state_dict()),
            adapted_parameter_names=method.adapted_parameter_names,
            environment=dict(python=sys.version, torch=str(torch.__version__), platform=platform.platform(),
                             gpu=torch.cuda.get_device_name(method.device))))
        timings, costs = execute(method, data, manifest, args.output)
        require(checked_sources() == manifest["source_sha256"]
                and shared.deterministic_runtime_state() == runtime, "Runtime/source changed")
        require(all(sha256(r["image"]) == r["image_sha256"] for r in rows), "Input image changed")
        verify_manifest(args.manifest, args.manifest_sha256)
        require(selection_before_confirmation(manifest) == selected, "Selection changed during confirmation")
        write_json(args.output / "prediction_result.json", dict(status="predicted", images=len(rows),
            full_split=False, timings=timings, costs=costs, timing_scope=TIMING_SCOPE))
        required = artifacts(manifest)
        actual = {str(p.relative_to(args.output)) for p in args.output.rglob("*") if p.is_file()}
        require(actual == required, "Prediction inventory incomplete/unexpected")
        write_json(args.output / "PREDICTIONS_SEALED.json", dict(complete=True, gt_contents_opened=False,
            images=len(rows), arms=list(ARMS), artifact_sha256={n: sha256(args.output / n) for n in sorted(required)}))


def verify_prediction_seal(output):
    output = Path(output)
    seal = read_json(output / "PREDICTIONS_SEALED.json")
    protocol = read_json(output / "prediction_protocol.json")
    manifest = protocol["manifest"]
    validate_manifest(manifest)
    require(seal["complete"] is True and seal["gt_contents_opened"] is False
            and seal["images"] == manifest["max_images"] and seal["arms"] == list(ARMS), "Prediction scope differs")
    require(set(seal["artifact_sha256"]) == artifacts(manifest), "Prediction inventory differs")
    for name, digest in seal["artifact_sha256"].items():
        path = output / name
        require(path.resolve().is_relative_to(output.resolve()) and not path.is_symlink(), "Prediction path escape")
        require(sha256(path) == digest, f"Prediction hash differs: {name}")
    return protocol


def stage_barrier(output, manifest):
    from tools.evolution_confirmation.queue import cells

    root = Path(manifest["suite_root"])
    filename = "ALL_" + manifest["stage"].upper() + "_PREDICTIONS_SEALED.json"
    barrier = read_json(root / filename)
    require(barrier["complete"] is True and barrier["stage"] == manifest["stage"], "Stage barrier incomplete")
    expected = barrier["prediction_seal_sha256"]
    entries = {e["output"]: e for e in cells() if e["stage"] == manifest["stage"]}
    require(set(expected) == set(entries), "Stage barrier is not the exact dataset/order inventory")
    require(expected.get(str(output.resolve().relative_to(root.resolve()))) == sha256(output / "PREDICTIONS_SEALED.json"),
            "Current prediction absent from stage barrier")
    for relative, digest in expected.items():
        path = root / relative
        require(path.resolve().is_relative_to(root.resolve()) and path.absolute() == path.resolve(),
                "Barrier path escapes suite or aliases another cell")
        require(sha256(path / "PREDICTIONS_SEALED.json") == digest, "Stage prediction seal changed")
        protocol = verify_prediction_seal(path)
        require(protocol["manifest"]["stage"] == manifest["stage"]
                and protocol["manifest"]["cohort_sha256"] == manifest["cohort_sha256"], "Cross-cohort barrier")
        entry = entries[relative]
        frozen_manifest = root / entry["manifest"]
        require(protocol["manifest"] == read_json(frozen_manifest)
                and protocol["manifest_sha256"] == sha256(frozen_manifest)
                and protocol["manifest_path"] == str(frozen_manifest), "Stage cell manifest binding differs")
        require(all(protocol["manifest"][key] == entry[key] for key in ("stage", "dataset", "order_seed"))
                and protocol["manifest"]["max_images"] == entry["images"], "Stage cell dataset/order differs")
    return barrier


def score(output):
    output = Path(output)
    protocol = verify_prediction_seal(output)
    manifest = protocol["manifest"]
    require(checked_sources() == manifest["source_sha256"], "Scoring source drift")
    barrier = stage_barrier(output, manifest)
    selected = selection_before_confirmation(manifest)
    require(protocol["selection"] == selected, "Confirmation selected mode drift")
    names = ("SCORING_STARTED.json", "scores_by_episode.jsonl", "score_inputs.json", "result.json", "SCORED_SEALED.json")
    require(not any((output / n).exists() for n in names), "No scoring overwrite/retry")
    seal_sha = sha256(output / "PREDICTIONS_SEALED.json")
    write_json(output / names[0], dict(prediction_seal_sha256=seal_sha, stage_barrier=barrier, selection=selected))
    classes, cfg = len(manifest["panel"]["classes"]), manifest["config"]
    totals = {a: np.zeros((2, classes), dtype=np.int64) for a in ARMS}
    hashes = {}
    with (output / names[1]).open("x") as stream:
        for start in range(0, manifest["max_images"], cfg["batch_size"]):
            counts = {a: np.zeros((2, classes), dtype=np.int64) for a in ARMS}
            for index in range(start, min(start + cfg["batch_size"], manifest["max_images"])):
                row = manifest["panel"]["inputs"][index]
                label = Path(row["label"])
                verify_label_metadata(row)
                content = label.read_bytes()
                hashes[str(label)] = hashlib.sha256(content).hexdigest()
                first = np.load(output / "predictions" / ARMS[0] / f"{index:06d}.npy", allow_pickle=False)
                with Image.open(io.BytesIO(content)) as source:
                    gt = np.asarray(source.resize((first.shape[1], first.shape[0]), Image.Resampling.NEAREST))
                if gt.ndim == 3:
                    gt = gt[..., 0]
                gt = gt.astype(np.int64, copy=True)
                if manifest["panel"]["reduce_zero_label"]:
                    bg, valid = gt == 0, gt != 255
                    gt[valid] -= 1
                    gt[bg] = 255
                for arm in ARMS:
                    prediction = first if arm == ARMS[0] else np.load(
                        output / "predictions" / arm / f"{index:06d}.npy", allow_pickle=False)
                    require(prediction.shape == first.shape, "Prediction geometry differs across arms")
                    counts[arm] += np.asarray(shared.streaming_counts(prediction, gt, classes))
            for arm in ARMS:
                totals[arm] += counts[arm]
            stream.write(json.dumps(dict(episode=start // cfg["batch_size"],
                arms={a: dict(intersection=v[0].tolist(), union=v[1].tolist()) for a, v in counts.items()})) + "\n")
    result = dict(status="completed", dataset=manifest["dataset"], stage=manifest["stage"],
        order_seed=manifest["order_seed"], images=manifest["max_images"], full_split=False,
        arm_results={a: dict(miou_unrounded=float(100 * np.mean(i[u > 0] / u[u > 0])) if (u > 0).any() else None,
                            intersection=i.tolist(), union=u.tolist()) for a, (i, u) in totals.items()},
        **{k: read_json(output / "prediction_result.json")[k] for k in ("timings", "costs", "timing_scope")},
        prediction_seal_sha256=seal_sha, cohort_sha256=manifest["cohort_sha256"],
        label_contents_first_opened_after_all_stage_predictions_sealed=True, selection=selected,
        exposure_scope=manifest["exposure_scope"])
    write_json(output / "result.json", result)
    write_json(output / "score_inputs.json", dict(label_sha256=hashes))
    write_json(output / "SCORED_SEALED.json", dict(prediction_seal_sha256=seal_sha,
        artifact_sha256={n: sha256(output / n) for n in names[:-1]}))
    return result


def claim_gate_attempt(output, manifest_sha256):
    """An interruption never makes the same gate target appear unused again."""
    output = Path(output)
    claim = output.with_name(output.name + ".STARTED.json")
    require(not output.exists() and not claim.exists(), "Gate target already attempted; no retry")
    write_json(claim, dict(status="started", manifest_sha256=manifest_sha256))
    return claim


def gate(args):
    """Real-checkpoint single-source objective identity; no annotation reads."""
    import copy
    from tools.trace_execution_audit import ExecutionAudit
    from tools.evolution_followup.smoke import final_predictions
    from trace_ovss.cli import build_patch_image_ids, build_patch_valid_mask

    manifest = verify_manifest(args.manifest, args.manifest_sha256)
    require(manifest["stage"] == "calibration" and not args.output.exists(), "Fresh calibration gate only")
    require(checked_sources() == manifest["source_sha256"], "Gate source drift")
    checkpoint = shared.checkpoint_identity(manifest["checkpoint"]["path"])
    require(checkpoint == manifest["checkpoint"], "Gate checkpoint drift")
    claim = claim_gate_attempt(args.output, args.manifest_sha256)
    cfg = dict(manifest["config"], batch_size=1)
    data = make_data(cfg, manifest["panel"]["inputs"][:4])
    report = dict(passed=False, images=4, dataset=manifest["dataset"], gt_contents_opened=False,
                  manifest_sha256=args.manifest_sha256, attempt_sha256=sha256(claim), comparisons=[])
    try:
        with prior.gpu_lock() as lock:
            runtime = shared.configure_deterministic_runtime()
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            method = ExperimentTRACE(classes=manifest["panel"]["queries"], lr=cfg["lr"], steps=cfg["steps"],
                trace_max_regions=cfg["trace_max_regions"], trace_view_batch_size=cfg["trace_view_batch_size"],
                vision_outputs=VISION_OUTPUTS, device="cuda", backbone=checkpoint["path"], view_mode="min")
            method.reset()
            initial = shared.restored_state(method)
            for index in range(4):
                batch = shared.image_collate([data[index]])
                images, meta = batch["img_patches"].to(method.device), batch["meta"]
                valid = build_patch_valid_mask(meta, cfg["patch_size"], cfg["patch_stride"], method.device)
                ids = build_patch_image_ids(meta, method.device)
                rng, outcomes = shared.rng_state(method.device), []
                for cls, arm in ((NativeGateReference, None), (ExperimentTRACE, "image_min")):
                    method.__class__ = cls
                    if arm:
                        method.arm = arm
                    method.reset()
                    require(shared.restored_state(method) == initial, "Gate reset state differs")
                    shared.restore_rng(rng, method.device)
                    with ExecutionAudit(method) as audit:
                        losses = method.adapt(images, valid_mask=valid, image_ids=ids)
                        predictions = final_predictions(method, images, meta, cfg, manifest["panel"]["query_to_class"])
                        counts = audit.snapshot()
                    stats = copy.deepcopy(method.last_candidate_stats)
                    require(stats and len(losses) == cfg["steps"] and all(math.isfinite(v) for v in losses),
                            "Gate incomplete adaptation")
                    require(counts["inputs"] == (cfg["steps"] + 2) * len(images) + 2 * len(stats)
                            and counts["failed_forward_attempts"] == 0, "Gate two-view actual schedule differs")
                    view_calls = (2 * len(stats) + cfg["trace_view_batch_size"] - 1) // cfg["trace_view_batch_size"]
                    require(counts["diagnostic_calls"] == view_calls
                            and method.last_trace_view_batch_size == cfg["trace_view_batch_size"],
                            "Gate changed diagnostic batching")
                    outcomes.append((losses, predictions, stats))
                a, b = outcomes
                support_diff = max(abs(x[k] - y[k]) for x, y in zip(a[2], b[2])
                                   for k in ("target_support", "counterfactual_support", "reliability"))
                loss_diff = max(abs(x - y) for x, y in zip(a[0], b[0]))
                require(len(a[1]) == len(b[1]) and all(x.shape == y.shape for x, y in zip(a[1], b[1])),
                        "Gate prediction shapes differ")
                changed = sum(int(np.count_nonzero(x != y)) for x, y in zip(a[1], b[1]))
                same = prior.candidate_identity(a[2]) == prior.candidate_identity(b[2])
                require(same and support_diff <= 1e-5 and loss_diff <= 1e-5 and changed == 0,
                        "Single-source new/native equivalence failed")
                schedule_counts = {}
                for arm in ("image_T", "image_C", "image_uniform", "image_confidence"):
                    method.arm = arm
                    shared.restore_rng(rng, method.device)
                    with method.count_work() as counts:
                        losses = method.adapt(images, valid_mask=valid, image_ids=ids)
                        final_predictions(method, images, meta, cfg, manifest["panel"]["query_to_class"])
                    require(all(math.isfinite(v) for v in losses), "Single-view gate nonfinite loss")
                    require(counts["diagnostic_encoder_inputs"] == expected_views(arm, len(method.last_candidate_stats))
                            == counts["diagnostic_forward_inputs"], "Single/zero-view gate count differs")
                    require(prior.candidate_identity(method.last_candidate_stats) == prior.candidate_identity(a[2]),
                            "Gate candidate schedule changed")
                    schedule_counts[arm] = counts
                report["comparisons"].append(dict(candidate_masks_equal=same, candidates=len(a[2]),
                    max_support_abs_difference=support_diff, max_loss_abs_difference=loss_diff,
                    changed_label_pixels=changed, actual_single_zero_view_counts=schedule_counts))
                method.reset()
                require(shared.restored_state(method) == initial, "Final gate reset differs")
            require(shared.deterministic_runtime_state() == runtime and checked_sources() == manifest["source_sha256"],
                    "Gate runtime/source changed")
            require(shared.checkpoint_identity(checkpoint["path"]) == checkpoint, "Gate checkpoint changed loading")
            require(all(sha256(r["image"]) == r["image_sha256"] for r in manifest["panel"]["inputs"][:4]),
                    "Gate RGB changed")
            report.update(passed=True, gpu_lock=lock, deterministic_runtime=runtime,
                          checkpoint=checkpoint, source_sha256=manifest["source_sha256"])
        write_json(args.output, report)
    except BaseException as error:
        if not args.output.exists():
            report.update(error=f"{type(error).__name__}: {error}"[:2000])
            write_json(args.output, report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    prep = sub.add_parser("prepare")
    for name in ("cohorts", "config", "checkpoint", "manifest"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--dataset", choices=DATASETS, required=True)
    prep.add_argument("--stage", choices=tuple(COUNTS), required=True)
    prep.add_argument("--order-seed", type=int)
    pred = sub.add_parser("predict")
    pred.add_argument("--manifest", type=Path, required=True)
    pred.add_argument("--manifest-sha256", required=True)
    pred.add_argument("--output", type=Path, required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--output", type=Path, required=True)
    smoke = sub.add_parser("gate")
    smoke.add_argument("--manifest", type=Path, required=True)
    smoke.add_argument("--manifest-sha256", required=True)
    smoke.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == "prepare":
        prepare(args)
    elif args.phase == "predict":
        predict(args)
    elif args.phase == "gate":
        gate(args)
    else:
        print(json.dumps(score(args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
