"""Independent full-split protocol around the unchanged ExperimentTRACE math.

All annotation reads are confined to score(), after the six-cell barrier.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
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

from tools.evolution_confirmation import run as confirmation
from tools.evolution_followup import standalone as prior
from tools.evolution_image_full import queue
from trace_ovss.data.datasets import SegmentationPatchDataset
from trace_ovss.method.image_balanced import ImageBalancedTRACE
from trace_ovss.method.trace.constants import EPS

ROOT = Path(__file__).resolve().parents[2]
ARMS, DATASETS, POPULATIONS = queue.ARMS, queue.DATASETS, queue.POPULATIONS
TIMING_SCOPE = prior.TIMING_SCOPE
require, sha256, read_json, write_json = queue.require, queue.sha256, queue.read_json, queue.write_json
shared = confirmation.shared
# Do not subclass or rewrite the experiment's objective, estimator or adaptation.
ExperimentTRACE = confirmation.ExperimentTRACE
make_data, objective_coefficients = confirmation.make_data, confirmation.objective_coefficients
PREUPDATE_STATUS = "unavailable: no additional encoder forward and no independently gated reconstruction of initial logits"


def checked_sources():
    require(confirmation.ROOT == prior.ROOT == Path(shared.__file__).resolve().parents[2] == queue.ROOT == ROOT,
            "Cross-root runtime imports")
    for name, module in list(sys.modules.items()):
        if name == "trace_ovss" or name.startswith(("trace_ovss.", "tools.evolution", "tools.suite_inputs", "tools.trace_")):
            path = getattr(module, "__file__", None)
            if path:
                require(Path(path).resolve().is_relative_to(ROOT), "Imported source outside frozen root")
    return {str(ROOT / key): value for key, value in queue.source_inventory(ROOT).items()}


def validate_config(config):
    # Schema validation only: today's BEST_CONFIGS must never supply values.
    require(set(config) == set(prior.CONFIG_KEYS), "Exactly the inherited resolved config keys required")
    for key in ("batch_size", "steps", "trace_max_regions", "trace_view_batch_size", "patch_stride"):
        require(type(config[key]) is int and config[key] > 0, "Invalid inherited integer config")
    require(type(config["lr"]) in (int, float) and math.isfinite(config["lr"]) and config["lr"] > 0, "Invalid inherited lr")
    require(config["resize"] == config["patch_size"] == [224, 224]
            and type(config["class_extensions"]) is bool and Path(config["data_dir"]).is_absolute(), "Inherited preprocessing/schema differs")


def native_panel(config):
    # Discovery checks file names/split text and stats, never label bytes.
    data = shared.image_only_dataset_class(SegmentationPatchDataset)(config["dataset"], config["data_dir"],
        resize=config["resize"], patch_size=config["patch_size"], patch_stride=config["patch_stride"], include_auxiliary=False)
    queries = list(data.vocabulary.extensions if config["class_extensions"] else data.vocabulary.classes)
    mapping = list(data.vocabulary.extension_to_class) if config["class_extensions"] else None
    return data, dict(classes=list(data.vocabulary.classes), queries=queries, query_to_class=mapping,
                      reduce_zero_label=data.spec.reduce_zero_label)


def verify_inputs(manifest):
    data, vocabulary = native_panel(manifest["config"])
    rows = manifest["panel"]["inputs"]
    require([(str(a), str(b)) for a, b in data.samples] == [(r["image"], r["label"]) for r in rows],
            "Official full membership/order differs from parent")
    require(all(manifest["panel"][k] == v for k, v in vocabulary.items()), "Current vocabulary differs from frozen parent")
    for row in rows:
        require(sha256(row["image"]) == row["image_sha256"], "RGB content drift")
        confirmation.verify_label_metadata(row)


def prepare(args):
    suite = args.suite.absolute()
    require(ROOT == suite / "runtime" and Path.cwd().resolve() == ROOT, "Manifest prepare in new frozen runtime only")
    require(sha256(suite / "frozen_plan.json") == queue.PLAN_SHA256, "Full plan drift")
    queue.validate_plan(read_json(suite / "frozen_plan.json"))
    queue.selection_proof_files(suite / "selection")
    queue.validate_source_continuity(queue.source_inventory(ROOT), suite / "selection")
    dataset = args.dataset
    parent_path = suite / f"parents/{dataset}/prediction_protocol.json"
    parent = read_json(parent_path)
    require(sha256(parent_path) == queue.PARENT_HASHES[dataset][0], "Parent protocol drift")
    queue.validate_parent(parent, dataset)
    cfg = read_json(suite / f"configs/{dataset}.json")
    require(cfg == parent["config"], "Config not inherited verbatim")
    validate_config(cfg)
    panel = dict(inputs=parent["inputs"], ordered_ids=[r["image"] for r in parent["inputs"]], full_split=True,
                 **{k: parent[k] for k in ("classes", "queries", "query_to_class", "reduce_zero_label")})
    manifest = dict(format_version=1, worker="image_full", dataset=dataset, stage="full", seed=0,
        order_seed=None, max_images=POPULATIONS[dataset], arms=ARMS.copy(), arm_order="alternating",
        suite_root=str(suite), config=cfg, config_path=str(suite / f"configs/{dataset}.json"),
        config_sha256=sha256(suite / f"configs/{dataset}.json"), panel=panel,
        checkpoint=parent["checkpoint"], source_sha256=checked_sources(), timing_scope=TIMING_SCOPE,
        selection_sha256=queue.SELECTION_SHA256, parent_protocol_sha256=sha256(parent_path),
        parent_seal_sha256=queue.PARENT_HASHES[dataset][1], plan_sha256=queue.PLAN_SHA256,
        exposure_scope=queue.EXPOSURE, preupdate_prediction_status=PREUPDATE_STATUS)
    validate_manifest(manifest)
    verify_inputs(manifest)
    require(shared.checkpoint_identity(parent["checkpoint"]["path"]) == parent["checkpoint"], "Parent checkpoint changed")
    write_json(suite / f"manifests/{dataset}.json", manifest)


def validate_manifest(manifest):
    require(manifest["format_version"] == 1 and manifest["worker"] == "image_full"
            and manifest["stage"] == "full" and manifest["dataset"] in DATASETS, "Wrong full worker/scope")
    require(type(manifest["seed"]) is int and manifest["seed"] == 0 and manifest["order_seed"] is None
            and manifest["arms"] == ARMS and manifest["arm_order"] == "alternating"
            and manifest["max_images"] == POPULATIONS[manifest["dataset"]]
            and manifest["timing_scope"] == TIMING_SCOPE and manifest["plan_sha256"] == queue.PLAN_SHA256
            and manifest["selection_sha256"] == queue.SELECTION_SHA256
            and manifest["preupdate_prediction_status"] == PREUPDATE_STATUS, "Full admission/timing/selection differs")
    validate_config(manifest["config"])
    panel, n = manifest["panel"], manifest["max_images"]
    require(panel["full_split"] is True and len(panel["inputs"]) == len(panel["ordered_ids"]) == n
            and panel["ordered_ids"] == [r["image"] for r in panel["inputs"]]
            and panel["ordered_ids"] == sorted(set(panel["ordered_ids"])), "Full input order/population differs")
    require(panel["classes"] and panel["queries"] and type(panel["reduce_zero_label"]) is bool, "Missing vocabulary")


def admission(args):
    suite = args.suite.absolute()
    prepared = queue.verify_prepared(suite, args.suite_sha256, frozen=True)
    entry = next(e for e in prepared["cells"] if e["dataset"] == args.dataset)
    manifest = read_json(suite / entry["manifest"])
    require(checked_sources() == manifest["source_sha256"], "Worker source drift")
    return suite, entry, manifest


def expected_views(arm, count):
    require(arm in ARMS, "Non-full arm rejected")
    return confirmation.expected_views(arm, count)


def candidate_receipts(stats, arm, mapping):
    coefficients = objective_coefficients(stats, arm)
    records = []
    for index, row in enumerate(stats):
        region = np.asarray(row["region"])
        require(region.dtype == np.bool_ and region.ndim == 2 and region.size > 0, "Candidate region must be native boolean HW")
        scalars = {k: v.item() if isinstance(v, np.generic) else v for k, v in row.items() if k != "region"}
        # Hex packbits is lossless, bounded and dependency-free.
        records.append(dict(scalars, region=dict(shape=list(region.shape), encoding="packbits-little-hex",
            data=np.packbits(region.reshape(-1), bitorder="little").tobytes().hex(), true_pixels=int(region.sum())),
            objective_coefficient=coefficients[index],
            preupdate_candidate_category_idx=int(row["class_idx"] if mapping is None else mapping[int(row["class_idx"])])))
    json.dumps(records, allow_nan=False)
    return records


COUNTER_PHASES = ("diagnostic_forward", "eval_forward", "adapt_forward", "encoder", "diagnostic_encoder")
COUNTERS = tuple(p + s for p in COUNTER_PHASES for s in ("_calls", "_inputs"))


def validate_cost(row, arm, cfg):
    require(row["arm"] == arm and arm in ARMS
            and type(row["images"]) is int and row["images"] > 0
            and type(row["patches"]) is int and row["patches"] == row["images"]
            and type(row["candidate_count"]) is int and 0 <= row["candidate_count"] <= row["images"] * cfg["trace_max_regions"]
            and type(row["active_candidates"]) is int and 0 <= row["active_candidates"] <= row["candidate_count"]
            and type(row["steps"]) is int
            and type(row["seconds"]) in (float, int) and math.isfinite(row["seconds"]) and row["seconds"] >= 0,
            "Invalid finite cost/scope")
    require(all(type(row[k]) is int and row[k] >= 0 for k in COUNTERS), "Noninteger/negative actual counter")
    steps = cfg["steps"] if row["active_candidates"] else 0
    views = expected_views(arm, row["candidate_count"])
    calls = (views + cfg["trace_view_batch_size"] - 1) // cfg["trace_view_batch_size"]
    require(row["steps"] == steps and row["eval_forward_calls"] == 2
            and row["eval_forward_inputs"] == 2 * row["patches"]
            and row["adapt_forward_calls"] == steps and row["adapt_forward_inputs"] == steps * row["patches"]
            and row["diagnostic_forward_calls"] == row["diagnostic_encoder_calls"] == calls
            and row["diagnostic_forward_inputs"] == row["diagnostic_encoder_inputs"] == views
            and row["encoder_calls"] == 2 + steps + calls
            and row["encoder_inputs"] == (2 + steps) * row["patches"] + views, "Actual 0/0/1 schedule differs")


def artifacts(manifest):
    n, batch = manifest["max_images"], manifest["config"]["batch_size"]
    return ({f"predictions/{arm}/{i:06d}.npy" for arm in ARMS for i in range(n)}
            | {f"episodes/{e:06d}.{a}.json" for a in ARMS for e in range((n + batch - 1) // batch)}
            | {"prediction_protocol.json", "initial_state.json", "prediction_result.json", "cost.jsonl"})


def require_finite(tensors, phase):
    values = [value.detach() for value in tensors if value is not None]
    require(bool(values), f"Missing tensors in finite guard: {phase}")
    groups = {}
    for value in values:
        groups.setdefault(value.device, []).append(value)
    # Group by actual tensor devices, not an unresolved requested 'cuda'/'cpu:0'.
    # One host synchronization per actual device, never one per parameter.
    require(all(torch.stack([torch.isfinite(value).all() for value in group]).all().item()
                for group in groups.values()), f"Nonfinite {phase}; observation-only abort")


@contextmanager
def finite_observation(method):
    """No numeric replacement, clipping, extra forwards or gradient alteration.

    The model hook runs before adapter/TRACE sanitization or final argmax. Step
    hooks inspect all trainable parameters/gradients (and post-step Adam state).
    Checks execute inside the timed adapt/evaluate calls, not after the timer.
    """
    handles = []

    def raw_forward(model, args, kwargs, output):
        phase = ("adaptation" if kwargs.get("vision_out_type") == "mean" else
                 "diagnostic" if getattr(method, "_diagnostic", False) else
                 "final evaluation" if kwargs.get("interpolate") else "initial readout")
        require(isinstance(output, tuple) and torch.is_tensor(output[0]), "Unexpected raw model output")
        require_finite([v for v in output if torch.is_tensor(v)], "raw " + phase)

    def before_step(optimizer, args, kwargs):
        parameters = [p for g in optimizer.param_groups for p in g["params"]]
        require_finite(parameters + [p.grad for p in parameters], "pre-update parameters/gradients")

    def after_step(optimizer, args, kwargs):
        parameters = [p for g in optimizer.param_groups for p in g["params"]]
        state = [v for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)]
        require_finite(parameters + state, "post-update parameters/optimizer")

    try:
        handles.append(method.model.register_forward_hook(raw_forward, with_kwargs=True))
        handles.append(method.optimizer.register_step_pre_hook(before_step))
        handles.append(method.optimizer.register_step_post_hook(after_step))
        yield
    finally:
        for handle in handles:
            handle.remove()


def execute(method, data, manifest, output):
    try:
        return execute_predictions(method, data, manifest, output)
    except BaseException as error:
        path = output / "PREDICTION_FAILED.json"
        if not path.exists():
            write_json(path, dict(status="failed", error=f"{type(error).__name__}: {error}"[:2000], retry_allowed=False))
        raise


def execute_predictions(method, data, manifest, output):
    from trace_ovss.cli import build_patch_image_ids, build_patch_valid_mask
    cfg, panel = manifest["config"], manifest["panel"]
    for arm in ARMS:
        (output / "predictions" / arm).mkdir(parents=True)
    (output / "episodes").mkdir()
    method.reset()
    initial = shared.restored_state(method)
    frozen = [(p, p._version) for p in method.model.parameters() if not p.requires_grad]
    costs = []
    with (output / "cost.jsonl").open("x") as stream:
        for episode, start in enumerate(range(0, len(data), cfg["batch_size"])):
            indices = range(start, min(start + cfg["batch_size"], len(data)))
            batch = shared.image_collate([data[i] for i in indices])
            images, meta = batch["img_patches"].to(method.device), batch["meta"]
            valid = build_patch_valid_mask(meta, cfg["patch_size"], cfg["patch_stride"], method.device)
            ids = build_patch_image_ids(meta, method.device)
            rng, identity = shared.rng_state(method.device), None
            for position, arm in enumerate(prior.arm_order(ARMS, episode, "alternating")):
                method.arm = arm
                shared.restore_rng(rng, method.device)
                with finite_observation(method), method.count_work() as counts:
                    shared.synchronize(method.device)
                    tick = time.perf_counter()
                    losses = method.adapt(images, valid_mask=valid, image_ids=ids)
                    predictions = shared.label_predictions(method, images, meta, cfg, panel["query_to_class"])
                    shared.synchronize(method.device)
                    seconds = time.perf_counter() - tick
                stats = method.last_candidate_stats
                digest = prior.candidate_identity(stats)
                identity = digest if identity is None else identity
                require(digest == identity, "Candidate admission differs across arms")
                require(all(math.isfinite(v) for v in losses), "Nonfinite adaptation loss")
                require(all(p._version == version for p, version in frozen), "Frozen weights changed")
                row = dict(episode=episode, arm=arm, order_position=position, images=len(indices), patches=len(images),
                    seconds=seconds, warmup=episode == 0, steps=len(losses), candidate_count=len(stats),
                    active_candidates=sum(r["reliability"] > EPS for r in stats), candidate_sha256=digest, **counts)
                validate_cost(row, arm, cfg)
                # All serialization, masks and disk writes are outside the model-side timer.
                candidates = candidate_receipts(stats, arm, panel["query_to_class"])
                for offset, prediction in enumerate(predictions):
                    require(prediction.ndim == 2 and np.issubdtype(prediction.dtype, np.integer)
                            and (prediction >= 0).all() and (prediction < len(panel["classes"])).all(), "Invalid prediction")
                    with (output / "predictions" / arm / f"{start + offset:06d}.npy").open("xb") as destination:
                        np.save(destination, prediction, allow_pickle=False)
                write_json(output / "episodes" / f"{episode:06d}.{arm}.json", dict(**row, losses=losses,
                    ordered_ids=panel["ordered_ids"][start:start + len(indices)], candidates=candidates,
                    preupdate_prediction_status=PREUPDATE_STATUS))
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                costs.append(row)
                method.reset()
                require(shared.restored_state(method) == initial, "Model/optimizer reset differs")
                del predictions, stats, candidates
    totals = {arm: {k: sum(r[k] for r in costs if r["arm"] == arm) for k in COUNTERS} for arm in ARMS}
    return prior.timing_summary(costs, ARMS), totals


def build_method(manifest):
    cfg = manifest["config"]
    require(shared.checkpoint_identity(manifest["checkpoint"]["path"]) == manifest["checkpoint"], "Checkpoint drift")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    return ExperimentTRACE(classes=manifest["panel"]["queries"], lr=cfg["lr"], steps=cfg["steps"],
        trace_max_regions=cfg["trace_max_regions"], trace_view_batch_size=cfg["trace_view_batch_size"],
        vision_outputs=confirmation.VISION_OUTPUTS, device="cuda", backbone=manifest["checkpoint"]["path"], view_mode="C")


def predict(args):
    suite, entry, manifest = admission(args)
    queue.no_scoring(suite)
    require(not (suite / queue.BARRIER).exists(), "Prediction after scoring barrier rejected")
    # Every real gate must pass before even the first full cell is attempted.
    gate_hashes = {e["dataset"]: queue.validate_gate(suite, e) for e in queue.cells_with_hashes(suite)}
    verify_inputs(manifest)
    output = suite / entry["output"]
    output.mkdir(parents=True, exist_ok=False)
    with prior.gpu_lock() as lock:
        runtime = shared.configure_deterministic_runtime()
        write_json(output / "prediction_protocol.json", dict(manifest=manifest, manifest_path=str(suite / entry["manifest"]),
            manifest_sha256=entry["manifest_sha256"] , suite_prepared_sha256=args.suite_sha256,
            selection_sha256=queue.SELECTION_SHA256, gpu_lock=lock, deterministic_runtime=runtime, gate_sha256=gate_hashes))
        method = build_method(manifest)
        require(shared.checkpoint_identity(manifest["checkpoint"]["path"]) == manifest["checkpoint"], "Checkpoint changed loading")
        write_json(output / "initial_state.json", dict(model_sha256=prior.state_digest(method.model.state_dict()),
            adapted_parameter_names=method.adapted_parameter_names,
            environment=dict(python=sys.version, torch=str(torch.__version__), platform=platform.platform(),
                             gpu=torch.cuda.get_device_name(method.device))))
        timings, costs = execute(method, make_data(manifest["config"], manifest["panel"]["inputs"]), manifest, output)
        require(shared.deterministic_runtime_state() == runtime and checked_sources() == manifest["source_sha256"], "Runtime/source drift")
        queue.verify_prepared(suite, args.suite_sha256, frozen=True)
        verify_inputs(manifest)
        require({e["dataset"]: queue.validate_gate(suite, e) for e in queue.cells_with_hashes(suite)} == gate_hashes, "Gate drift")
        write_json(output / "prediction_result.json", dict(status="predicted", images=manifest["max_images"],
            full_split=True, timings=timings, costs=costs, timing_scope=TIMING_SCOPE))
        required = artifacts(manifest)
        require({str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()} == required, "Prediction inventory differs")
        write_json(output / "PREDICTIONS_SEALED.json", dict(complete=True, gt_contents_opened=False,
            images=manifest["max_images"], arms=ARMS.copy(), suite_prepared_sha256=args.suite_sha256,
            artifact_sha256={n: sha256(output / n) for n in sorted(required)}))


def verify_prediction_seal(output):
    output = Path(output).absolute()
    require(output == output.resolve(), "Prediction output alias rejected")
    protocol, seal = read_json(output / "prediction_protocol.json"), read_json(output / "PREDICTIONS_SEALED.json")
    manifest = protocol["manifest"]
    validate_manifest(manifest)
    suite = Path(manifest["suite_root"])
    entry = next(e for e in queue.cells() if e["dataset"] == manifest["dataset"])
    require(output == suite / entry["output"] and protocol["manifest_path"] == str(suite / entry["manifest"])
            and protocol["manifest_sha256"] == sha256(suite / entry["manifest"])
            and manifest == read_json(suite / entry["manifest"])
            and protocol["selection_sha256"] == queue.SELECTION_SHA256
            and protocol["suite_prepared_sha256"] == seal["suite_prepared_sha256"] == sha256(suite / "SUITE_PREPARED.json"),
            "Prediction root/manifest/prepared binding differs")
    require(seal["complete"] is True and seal["gt_contents_opened"] is False
            and seal["images"] == manifest["max_images"] and seal["arms"] == ARMS
            and set(seal["artifact_sha256"]) == artifacts(manifest), "Prediction seal scope differs")
    queue.check_hashes(output, seal["artifact_sha256"])
    require({str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
            <= artifacts(manifest) | {"PREDICTIONS_SEALED.json", *queue.SCORING_FILES}, "Unexpected prediction/scoring file")
    return protocol


def bounded_json_lines(path, maximum):
    with Path(path).open() as stream:
        for index in range(maximum + 1):
            line = stream.readline(64 * 1024 + 1)
            if not line:
                return
            require(index < maximum and len(line) <= 64 * 1024 and line.endswith("\n"), "Unbounded JSONL stream")
            value = json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
            yield value


def validate_cost_stream(output, manifest):
    cfg, n = manifest["config"], manifest["max_images"]
    episodes = (n + cfg["batch_size"] - 1) // cfg["batch_size"]
    rows = list(bounded_json_lines(output / "cost.jsonl", episodes * len(ARMS)))
    require(len(rows) == episodes * len(ARMS), "Incomplete actual cost stream")
    for episode in range(episodes):
        group = rows[episode * len(ARMS):(episode + 1) * len(ARMS)]
        order = prior.arm_order(ARMS, episode, "alternating")
        require([r["arm"] for r in group] == order and len({r["candidate_sha256"] for r in group}) == 1, "Cost arm/candidate order differs")
        for position, row in enumerate(group):
            require(row["episode"] == episode and row["order_position"] == position
                    and row["warmup"] is (episode == 0)
                    and row["images"] == min(cfg["batch_size"], n - episode * cfg["batch_size"]), "Actual episode scope differs")
            validate_cost(row, row["arm"], cfg)
    predicted = read_json(output / "prediction_result.json")
    totals = {arm: {k: sum(r[k] for r in rows if r["arm"] == arm) for k in COUNTERS} for arm in ARMS}
    require(predicted["status"] == "predicted" and predicted["images"] == n and predicted["full_split"] is True
            and predicted["timing_scope"] == TIMING_SCOPE and predicted["costs"] == totals
            and predicted["timings"] == prior.timing_summary(rows, ARMS), "Actual aggregate costs/timing differs")
    return predicted


def stage_barrier(suite, prepared_sha):
    prepared = queue.verify_prepared(suite, prepared_sha, frozen=True)
    barrier = read_json(suite / queue.BARRIER)
    require(barrier["complete"] is True and barrier["stage"] == "full" and barrier["gt_contents_opened"] is False
            and barrier["suite_prepared_sha256"] == prepared_sha
            and set(barrier["prediction_seal_sha256"]) == {e["output"] for e in prepared["cells"]}, "Exact six-cell barrier missing")
    for entry in prepared["cells"]:
        require(queue.validate_prediction(suite, entry) == barrier["prediction_seal_sha256"][entry["output"]], "Global prediction drift")
    return sha256(suite / queue.BARRIER)


def score(args):
    suite, entry, manifest = admission(args)
    barrier_sha = stage_barrier(suite, args.suite_sha256)
    output = suite / entry["output"]
    predicted = validate_cost_stream(output, manifest)
    require(not any((output / n).exists() for n in queue.SCORING_FILES), "No score retry/overwrite")
    prediction_sha = sha256(output / "PREDICTIONS_SEALED.json")
    write_json(output / "SCORING_STARTED.json", dict(prediction_seal_sha256=prediction_sha, barrier_sha256=barrier_sha,
                                                    suite_prepared_sha256=args.suite_sha256))
    cfg, panel = manifest["config"], manifest["panel"]
    classes = len(panel["classes"])
    totals = {a: np.zeros((2, classes), dtype=np.int64) for a in ARMS}
    hashes = {}
    with (output / "scores_by_episode.jsonl").open("x") as stream:
        for start in range(0, manifest["max_images"], cfg["batch_size"]):
            counts = {a: np.zeros((2, classes), dtype=np.int64) for a in ARMS}
            for index in range(start, min(start + cfg["batch_size"], manifest["max_images"])):
                row = panel["inputs"][index]
                confirmation.verify_label_metadata(row)
                content = Path(row["label"]).read_bytes()
                hashes[row["label"]] = hashlib.sha256(content).hexdigest()
                first = np.load(output / f"predictions/{ARMS[0]}/{index:06d}.npy", allow_pickle=False)
                with Image.open(io.BytesIO(content)) as source:
                    gt = np.asarray(source.resize((first.shape[1], first.shape[0]), Image.Resampling.NEAREST))
                if gt.ndim == 3:
                    gt = gt[..., 0]
                gt = gt.astype(np.int64, copy=True)
                if panel["reduce_zero_label"]:
                    background, valid = gt == 0, gt != 255
                    gt[valid] -= 1
                    gt[background] = 255
                for arm in ARMS:
                    prediction = first if arm == ARMS[0] else np.load(output / f"predictions/{arm}/{index:06d}.npy", allow_pickle=False)
                    require(prediction.shape == first.shape and prediction.ndim == 2
                            and np.issubdtype(prediction.dtype, np.integer) and (prediction < classes).all()
                            and (prediction >= 0).all(), "Invalid final prediction geometry/category")
                    counts[arm] += np.asarray(shared.streaming_counts(prediction, gt, classes))
            for arm in ARMS:
                totals[arm] += counts[arm]
            stream.write(json.dumps(dict(episode=start // cfg["batch_size"], images=min(cfg["batch_size"], manifest["max_images"] - start),
                arms={a: dict(intersection=v[0].tolist(), union=v[1].tolist()) for a, v in counts.items()})) + "\n")
    result = dict(status="completed", dataset=entry["dataset"], stage="full", seed=0, images=entry["images"], full_split=True,
        arm_results={a: dict(miou_unrounded=100 * math.fsum(int(x) / int(y) for x, y in zip(i, u) if y) / int((u > 0).sum())
                            if (u > 0).any() else None, intersection=i.tolist(), union=u.tolist()) for a, (i, u) in totals.items()},
        **{k: predicted[k] for k in ("timings", "costs", "timing_scope")}, prediction_seal_sha256=prediction_sha,
        barrier_sha256=barrier_sha, selection_sha256=queue.SELECTION_SHA256,
        label_contents_first_opened_after_all_full_predictions_sealed=True, exposure_scope=queue.EXPOSURE)
    write_json(output / "result.json", result)
    write_json(output / "score_inputs.json", dict(label_sha256=hashes))
    validate_score_stream(output, manifest, result)
    require(stage_barrier(suite, args.suite_sha256) == barrier_sha, "Barrier changed during scoring")
    write_json(output / "SCORED_SEALED.json", dict(prediction_seal_sha256=prediction_sha,
        artifact_sha256={n: sha256(output / n) for n in queue.SCORING_FILES[:-1]}))


def validate_score_stream(output, manifest, result):
    cfg, n = manifest["config"], manifest["max_images"]
    classes = len(manifest["panel"]["classes"])
    expected = (n + cfg["batch_size"] - 1) // cfg["batch_size"]
    totals = {a: [[0] * classes, [0] * classes] for a in ARMS}
    count = 0
    for episode, row in enumerate(bounded_json_lines(output / "scores_by_episode.jsonl", expected)):
        require(row["episode"] == episode and row["images"] == min(cfg["batch_size"], n - episode * cfg["batch_size"])
                and set(row["arms"]) == set(ARMS), "Score stream scope differs")
        for arm in ARMS:
            i, u = row["arms"][arm]["intersection"], row["arms"][arm]["union"]
            require(len(i) == len(u) == classes and all(type(x) is int and type(y) is int and 0 <= x <= y for x, y in zip(i, u)),
                    "Invalid streaming pooled counts")
            totals[arm] = [[x + y for x, y in zip(a, b)] for a, b in zip(totals[arm], (i, u))]
        count += 1
    require(count == expected, "Score stream omits episodes (including warmup)")
    for arm in ARMS:
        require(totals[arm] == [result["arm_results"][arm]["intersection"], result["arm_results"][arm]["union"]], "Nonpooled score totals")
        queue.pooled_miou(result["arm_results"][arm])
    labels = read_json(output / "score_inputs.json")["label_sha256"]
    require(set(labels) == {r["label"] for r in manifest["panel"]["inputs"]}
            and all(isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h) for h in labels.values()),
            "Score input hash coverage differs")


class GateReference(ImageBalancedTRACE):
    """Direct C objective with the same fail-closed batching and observation only."""

    _forward_trace_view_batches = prior.StandaloneTRACE._forward_trace_view_batches
    _count = prior.StandaloneTRACE._count
    count_work = prior.StandaloneTRACE.count_work

    def forward_eval_logits(self, images, *args, **kwargs):
        if hasattr(self, "work"):
            self._count(("diagnostic" if getattr(self, "_diagnostic", False) else "eval") + "_forward", len(images))
        return super().forward_eval_logits(images, *args, **kwargs)

    def forward_adapt_logits(self, images):
        if hasattr(self, "work"):
            self._count("adapt_forward", len(images))
        return super().forward_adapt_logits(images)


@contextmanager
def gradient_receipt(method):
    gradients, handles = {}, []
    try:
        for name, parameter in method.model.named_parameters():
            if parameter.requires_grad:
                gradients[name] = []

                def collect(value, key=name):
                    gradients[key].append(value.detach().clone())

                handles.append(parameter.register_hook(collect))
        yield gradients
    finally:
        for handle in handles:
            handle.remove()


def compare_gate_episode(method, reference, images, valid, ids, meta, cfg, mapping):
    """CPU-testable gate core; no class mutation, extra forward, or annotations."""
    method.reset()
    initial = shared.restored_state(method)
    rng, outcomes, counts_by_arm = shared.rng_state(method.device), [], {}
    for current, arm in ((reference, "image_C"), (method, "image_C")):
        current.reset()
        require(shared.restored_state(current) == initial, "Gate initial reset differs")
        if current is method:
            current.arm = arm
        shared.restore_rng(rng, current.device)
        with finite_observation(current), gradient_receipt(current) as gradients, current.count_work() as counts:
            losses = current.adapt(images, valid_mask=valid, image_ids=ids)
            predictions = shared.label_predictions(current, images, meta, cfg, mapping)
        stats = copy.deepcopy(current.last_candidate_stats)
        row = dict(arm=arm, images=len(meta["img_path"]), patches=len(images), seconds=0., steps=len(losses),
            candidate_count=len(stats), active_candidates=sum(r["reliability"] > EPS for r in stats), **counts)
        validate_cost(row, arm, cfg)
        require(all(math.isfinite(v) for v in losses), "Gate nonfinite loss")
        outcomes.append((losses, predictions, stats, gradients))
        counts_by_arm[arm] = row
        current.reset()
        require(shared.restored_state(current) == initial, "Gate final reset differs")
    a, b = outcomes
    same = prior.candidate_identity(a[2]) == prior.candidate_identity(b[2])
    require(same, "Gate candidate masks differ")
    support_diff = max((abs(x[k] - y[k]) for x, y in zip(a[2], b[2], strict=True)
                        for k in ("confidence", "counterfactual_support", "reliability")), default=0.)
    require(all(r["target_support"] is None for r in a[2] + b[2]), "Gate fabricated unmeasured T")
    coeff_diff = max((abs(x - y) for x, y in zip(objective_coefficients(a[2], "image_C"), objective_coefficients(b[2], "image_C"), strict=True)), default=0.)
    loss_diff = max((abs(x - y) for x, y in zip(a[0], b[0], strict=True)), default=0.)
    require(set(a[3]) == set(b[3]), "Gradient parameter membership differs")
    gradient_pairs = []
    for name in a[3]:
        require(len(a[3][name]) == len(a[0]) and len(b[3][name]) == len(b[0]), "Gate missing step gradients")
        gradient_pairs.extend(zip(a[3][name], b[3][name], strict=True))
    if gradient_pairs:
        require_finite([g for pair in gradient_pairs for g in pair], "gate gradient receipt")
    grad_diff = float(torch.stack([(x - y).abs().max() for x, y in gradient_pairs]).max().item()) if gradient_pairs else 0.
    require(len(a[1]) == len(b[1]) and all(x.shape == y.shape for x, y in zip(a[1], b[1])), "Gate label geometry differs")
    changed = sum(int(np.count_nonzero(x != y)) for x, y in zip(a[1], b[1]))
    require(all(math.isfinite(x) and x <= 1e-5 for x in (support_diff, coeff_diff, loss_diff, grad_diff)) and changed == 0,
            "Direct ImageBalancedTRACE C equivalence failed")
    for arm in ARMS[:2]:
        method.arm = arm
        shared.restore_rng(rng, method.device)
        with finite_observation(method), method.count_work() as counts:
            losses = method.adapt(images, valid_mask=valid, image_ids=ids)
            shared.label_predictions(method, images, meta, cfg, mapping)
        stats = method.last_candidate_stats
        require(prior.candidate_identity(stats) == prior.candidate_identity(a[2])
                and all(math.isfinite(v) for v in losses), "Gate control candidate/loss differs")
        row = dict(arm=arm, images=len(meta["img_path"]), patches=len(images), seconds=0., steps=len(losses),
            candidate_count=len(stats), active_candidates=sum(r["reliability"] > EPS for r in stats), **counts)
        validate_cost(row, arm, cfg)
        counts_by_arm[arm] = row
        method.reset()
        require(shared.restored_state(method) == initial, "Gate control reset differs")
    return dict(candidate_masks_equal=same, candidates=len(a[2]), reset_equal=True,
        max_support_abs_difference=support_diff, max_coefficient_abs_difference=coeff_diff,
        max_loss_abs_difference=loss_diff, max_gradient_abs_difference=grad_diff,
        changed_label_pixels=changed, actual_arm_counts=counts_by_arm)


def gate(args):
    from trace_ovss.cli import build_patch_image_ids, build_patch_valid_mask
    suite, entry, manifest = admission(args)
    queue.no_scoring(suite)
    require(not (suite / "cells").exists() and not (suite / queue.BARRIER).exists(), "Gates must precede all full prediction")
    output = suite / f"gates/{entry['dataset']}.json"
    claim = confirmation.claim_gate_attempt(output, entry["manifest_sha256"])
    cfg, comparisons, batches = manifest["config"], [], []
    report = dict(passed=False, dataset=entry["dataset"], images=4, gt_contents_opened=False,
        manifest_sha256=entry["manifest_sha256"], attempt_sha256=sha256(claim), configured_batch_size=cfg["batch_size"],
        suite_prepared_sha256=args.suite_sha256, ordered_ids=manifest["panel"]["ordered_ids"][:4])
    try:
        verify_inputs(manifest)
        with prior.gpu_lock() as lock:
            runtime = shared.configure_deterministic_runtime()
            method = build_method(manifest)
            method.reset()
            # Share frozen model storage, but use independent optimizer/snapshot state.
            reference = GateReference(classes=method.classes, lr=method.lr, steps=method.steps,
                trace_max_regions=method.trace_max_regions, trace_view_batch_size=method.trace_view_batch_size,
                vision_outputs=method.vision_outputs, prompts=method.prompt_templates, model=method.model,
                tokenize_fn=method.tokenize, device=method.device, view_mode="C")
            require(torch.equal(method.text_x, reference.text_x), "Gate text embeddings differ")
            data = make_data(cfg, manifest["panel"]["inputs"][:4])
            for start in range(0, 4, cfg["batch_size"]):
                batch = shared.image_collate([data[i] for i in range(start, min(start + cfg["batch_size"], 4))])
                images, meta = batch["img_patches"].to(method.device), batch["meta"]
                valid = build_patch_valid_mask(meta, cfg["patch_size"], cfg["patch_stride"], method.device)
                ids = build_patch_image_ids(meta, method.device)
                batches.append(len(meta["img_path"]))
                comparisons.append(compare_gate_episode(method, reference, images, valid, ids, meta, cfg, manifest["panel"]["query_to_class"]))
            require(sum(r["candidates"] for r in comparisons) > 0, "Vacuous gate without candidates")
            require(shared.deterministic_runtime_state() == runtime and checked_sources() == manifest["source_sha256"], "Gate runtime/source drift")
            require(shared.checkpoint_identity(manifest["checkpoint"]["path"]) == manifest["checkpoint"], "Gate checkpoint drift")
            verify_inputs(manifest)
            queue.verify_prepared(suite, args.suite_sha256, frozen=True)
            report.update(passed=True, comparisons=comparisons, batch_images=batches, gpu_lock=lock,
                          deterministic_runtime=runtime, checkpoint=manifest["checkpoint"], source_sha256=manifest["source_sha256"])
        write_json(output, report)
    except BaseException as error:
        if not output.exists():
            write_json(output, dict(report, error=f"{type(error).__name__}: {error}"[:2000]))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    for phase in ("prepare", "gate", "predict", "score"):
        command = sub.add_parser(phase)
        command.add_argument("--suite", type=Path, required=True)
        command.add_argument("--dataset", choices=DATASETS, required=True)
        if phase != "prepare":
            command.add_argument("--suite-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        dict(prepare=prepare, gate=gate, predict=predict, score=score)[args.phase](args)
    except BaseException as error:
        path = args.suite / f"logs/worker-{args.phase}-{args.dataset}.FAILED.json"
        if path.parent.is_dir() and not path.exists():
            write_json(path, dict(status="failed", phase=args.phase, dataset=args.dataset,
                error=f"{type(error).__name__}: {error}"[:2000], retry_allowed=False))
        raise


if __name__ == "__main__":
    main()
