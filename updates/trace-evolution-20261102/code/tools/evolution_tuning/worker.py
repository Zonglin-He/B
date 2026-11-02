"""Bounded, fail-closed single-probe worker. Annotation bytes open only in score.

Run this module from the frozen suite/runtime, never from a development checkout.
The controller owns search, selection, scheduling and stage-barrier creation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch

from tools.evolution import run_cell as shared
from tools.evolution_confirmation import run as confirmation
from tools.evolution_image_full import run as full
from tools.evolution_tuning.common import DATASETS, FULL_ARMS, canonical_hash, read_json, write_json, require, sha256

ROOT = Path(__file__).resolve().parents[2]
ARMS = FULL_ARMS
DATASET_CLASSES = dict(zip(DATASETS, ("PascalVOC20Dataset", "PascalVOC21Dataset", "PascalContext59Dataset",
    "PascalContext60Dataset", "COCOObjectDataset", "COCOStuffDataset", "COCOCDDataset", "WaterbirdsSegDataset")))
ExperimentTRACE = confirmation.ExperimentTRACE
make_data = confirmation.make_data
finite_observation = full.finite_observation
candidate_receipts = full.candidate_receipts
GateReference = full.GateReference
EPS, COUNTERS = full.EPS, full.COUNTERS
TIMING_SCOPE = (
    "Synchronized model-side wallclock: exact reset/state check, candidate admission, "
    "diagnostic views, optimization, final inference/reconstruction/CPU transfer and observation hooks. "
    "NoAdapt is reset plus evaluation only. Excludes setup, RGB decode/H2D/geometry, disk writes "
    "and offline scoring. Episode 0 is warmup, retained in all counts and scores."
)
SCORE_FILES = {"SCORING_STARTED.json", "scores_by_episode.jsonl", "score_inputs.json", "result.json", "SCORED_SEALED.json"}


def digest_valid(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def absolute_path(value):
    path = Path(value)
    require(path.is_absolute() and path == path.resolve(), "Noncanonical/aliased absolute path")
    return path


def validate_manifest(manifest):
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
            and manifest["algorithm"] == "ImageBalancedTRACE:C", "Wrong worker schema/algorithm")
    require(manifest["dataset"] in DATASETS and manifest["stage"] in ("calibration", "full", "gate"), "Unknown dataset/stage")
    require(type(manifest["seed"]) is int and manifest["seed"] == 0, "Only seed 0 admitted")
    require(manifest["arms"] == (ARMS if manifest["stage"] == "full" else ["image_C"]), "Wrong stage arms/order")
    suite, runtime = absolute_path(manifest["suite_root"]), absolute_path(manifest["runtime_root"])
    require(runtime == suite / "runtime", "Runtime must be suite/runtime")
    cfg = manifest["config"]
    full.validate_config(cfg)  # Schema only, NOT old six-dataset admission.
    require(cfg["dataset"] == DATASET_CLASSES[manifest["dataset"]]
            and cfg["class_extensions"] is (manifest["dataset"] not in ("coco_cd", "waterbirds_seg")), "Dataset/query-extension config binding differs")
    require(cfg["trace_view_batch_size"] == 16 and cfg["patch_stride"] == 112, "Frozen geometry/view batching changed")
    require(1 <= cfg["batch_size"] <= 32 and 1 <= cfg["steps"] <= 64 and 1 <= cfg["trace_max_regions"] <= 64
            and 1e-6 <= cfg["lr"] <= .04, "Unbounded tuning configuration")
    panel, rows = manifest["panel"], manifest["panel"]["inputs"]
    require(isinstance(rows, list) and rows and len(rows) <= 100000, "Missing/unbounded explicit panel")
    require(panel["ordered_ids"] == [r["official_id"] for r in rows]
            and len(set(panel["ordered_ids"])) == len(rows), "Ordered official IDs differ/duplicate")
    require(len({r["image"] for r in rows}) == len(rows), "Duplicate image")
    require(panel["classes"] and panel["queries"] and type(panel["reduce_zero_label"]) is bool, "Missing vocabulary")
    mapping = panel["query_to_class"]
    require((mapping is None and len(panel["queries"]) == len(panel["classes"]))
            or (isinstance(mapping, list) and len(mapping) == len(panel["queries"])
                and all(type(v) is int and 0 <= v < len(panel["classes"]) for v in mapping)), "Invalid query mapping")
    if "full_split" in panel:
        require(panel["full_split"] is (manifest["stage"] == "full"), "Panel stage indicator differs")
    if manifest["stage"] == "calibration":
        require(len(rows) == 256, "Calibration requires the frozen 256-image panel")
    if manifest["stage"] == "gate":
        require(len(rows) >= 4, "Gate requires four actual RGB inputs")
    for row in rows:
        require(isinstance(row["official_id"], str) and row["official_id"], "Invalid official identity")
        absolute_path(row["image"])
        absolute_path(row["label"])
        require(digest_valid(row["image_sha256"]), "Missing RGB hash")
        metadata = row["label_metadata"]
        require(isinstance(metadata, list) and len(metadata) == 2
                and all(type(v) is int and v >= 0 for v in metadata), "Expected label stat [size,mtime_ns]")
    sources = manifest["source_sha256"]
    require(isinstance(sources, dict) and sources, "Missing relative frozen sources")
    for key, value in sources.items():
        p = Path(key)
        require(not p.is_absolute() and ".." not in p.parts and str(p) == key and digest_valid(value), "Invalid relative source hash")
    require(digest_valid(manifest["checkpoint"]["sha256"]), "Missing checkpoint identity")
    absolute_path(manifest["checkpoint"]["path"])
    if "suite_prepared_sha256" in manifest:
        require(digest_valid(manifest["suite_prepared_sha256"]), "Invalid prepared hash")


def verify_sources(manifest):
    require(ROOT == absolute_path(manifest["runtime_root"]), "Worker not executing frozen runtime")
    sources = manifest["source_sha256"]
    for relative, expected in sources.items():
        path = ROOT / relative
        require(path.resolve() == path and path.is_file() and sha256(path) == expected, "Frozen source drift: " + relative)
    # This catches an already-imported module from another checkout as well as
    # partial inventories that omit a live implementation dependency.
    for name, module in list(sys.modules.items()):
        if name == "trace_ovss" or name.startswith(("trace_ovss.", "tools.evolution", "tools.suite_inputs", "tools.trace_")):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                require(path.is_relative_to(ROOT), "Cross-root runtime import: " + name)
                require(str(path.relative_to(ROOT)) in sources, "Imported source missing from snapshot: " + name)


def guard_label_aliases(manifest, *, runtime=False):
    """Stat-only guard: an RGB/provenance/checkpoint alias cannot open GT."""
    labels = set()
    for row in manifest["panel"]["inputs"]:
        status = Path(row["label"]).stat()
        labels.add((status.st_dev, status.st_ino))
    paths = [r["image"] for r in manifest["panel"]["inputs"]]
    paths += list(manifest["panel"].get("source_hashes", {}))
    paths += [manifest[k] for k in ("config_path", "inputs_path") if k in manifest]
    if runtime:
        paths += [manifest["checkpoint"]["path"]]
        paths += [Path(manifest["runtime_root"]) / name for name in manifest["source_sha256"]]
    for filename in paths:
        status = Path(filename).stat()
        require((status.st_dev, status.st_ino) not in labels, "Label isolation failure: non-label input aliases annotation")


def verify_inputs(manifest):
    guard_label_aliases(manifest)
    for row in manifest["panel"]["inputs"]:
        require(sha256(row["image"]) == row["image_sha256"], "RGB input drift")
        confirmation.verify_label_metadata(row)
        if "label_metadata_details" in row:
            expected = row["label_metadata_details"]
            status = Path(row["label"]).stat()
            require(all(expected[k] == v for k, v in dict(size_bytes=status.st_size,
                mtime_ns=status.st_mtime_ns, device=status.st_dev, inode=status.st_ino).items()), "Label physical identity drift")
    for filename, identity in manifest["panel"].get("source_hashes", {}).items():
        expected = identity["sha256"] if isinstance(identity, dict) else identity
        require(digest_valid(expected) and sha256(absolute_path(filename)) == expected, "Input provenance source drift")
    for path_key, hash_key in (("config_path", "config_sha256"), ("inputs_path", "inputs_sha256")):
        if path_key in manifest or hash_key in manifest:
            require(path_key in manifest and hash_key in manifest and sha256(manifest[path_key]) == manifest[hash_key], "Bound input file drift")
    data = make_data(manifest["config"], manifest["panel"]["inputs"])
    vocabulary = data.vocabulary
    cfg, panel = manifest["config"], manifest["panel"]
    queries = list(vocabulary.extensions if cfg["class_extensions"] else vocabulary.classes)
    mapping = list(vocabulary.extension_to_class) if cfg["class_extensions"] else None
    require(list(vocabulary.classes) == panel["classes"] and queries == panel["queries"]
            and mapping == panel["query_to_class"] and data.spec.reduce_zero_label == panel["reduce_zero_label"], "Frozen vocabulary/config differs")
    return data


def prepared_scope(manifest):
    """Bind dynamic manifests to the immutable prepared inputs and settings."""
    if "suite_prepared_sha256" not in manifest:
        return None
    suite = Path(manifest["suite_root"])
    require(sha256(suite / "SUITE_PREPARED.json") == manifest["suite_prepared_sha256"], "Suite prepared drift")
    prepared = read_json(suite / "SUITE_PREPARED.json")
    runnable = prepared["runnable"]
    require(runnable and len(set(runnable)) == len(runnable) and set(runnable) <= set(DATASETS)
            and set(runnable) | set(prepared["blocked"]) == set(DATASETS)
            and not set(runnable) & set(prepared["blocked"]), "Invalid prepared runnable/blocked scope")
    require(manifest["dataset"] in runnable and prepared["runtime_hashes"] == manifest["source_sha256"], "Manifest not in prepared runtime/scope")
    require(sha256(suite / "inputs.json") == prepared["inputs_sha256"]
            and sha256(suite / "frozen_contract.md") == prepared["contract_sha256"], "Prepared input/contract drift")
    entry = read_json(suite / "inputs.json")["datasets"][manifest["dataset"]]
    initial, cfg = prepared["initial_configs"][manifest["dataset"]], manifest["config"]
    require(set(initial) == set(cfg) and all(initial[k] == cfg[k] for k in cfg
        if k not in ("batch_size", "lr", "steps", "trace_max_regions")), "Frozen non-search setting drift")
    expected_panel = entry["evaluation" if manifest["stage"] == "full" else "calibration"]
    if manifest["stage"] == "gate":
        expected_panel = copy.deepcopy(expected_panel)
        expected_panel["inputs"] = expected_panel["inputs"][:4]
        expected_panel["ordered_ids"] = expected_panel["ordered_ids"][:4]
        expected_panel["full_split"] = False
    require(manifest["panel"] == expected_panel and manifest["checkpoint"] == prepared["checkpoint"], "Prepared panel/checkpoint binding differs")
    if manifest["stage"] == "full":
        require(not prepared["blocked"] and set(runnable) == set(DATASETS),
                "Full evaluation waits for all eight calibration selections")
        path = absolute_path(manifest["selected_config_path"])
        require(path == suite / "selections" / manifest["dataset"] / "SELECTED_CONFIG_SEALED.json"
                and sha256(path) == manifest["selected_config_sha256"], "Selected config seal drift")
        selected = read_json(path)
        require(selected["config"] == cfg and selected["calibration_only"] is True
                and selected["full_evaluation_used_for_selection"] is False, "Full config not calibration-selected")
        all_path = suite / "ALL_CONFIGS_SELECTED.json"
        require(sha256(all_path) == manifest["all_selections_sha256"], "Global selected configs drift")
        all_selected = read_json(all_path)
        require(all_selected["complete"] is True and set(all_selected["selections"]) == set(runnable), "Incomplete all-config selection scope")
        for dataset, expected in all_selected["selections"].items():
            require(sha256(suite / "selections" / dataset / "SELECTED_CONFIG_SEALED.json") == expected, "Selected peer config drift")
    return prepared


def admission(args):
    path, output = absolute_path(str(args.manifest)), absolute_path(str(args.output))
    require(digest_valid(args.manifest_sha256) and sha256(path) == args.manifest_sha256, "Manifest hash drift")
    manifest = read_json(path)
    validate_manifest(manifest)
    suite = Path(manifest["suite_root"])
    require(path.is_relative_to(suite) and output.is_relative_to(suite)
            and not output.is_relative_to(Path(manifest["runtime_root"])) and output != suite, "Manifest/output outside suite")
    if "output" in manifest:
        require(str(output) == manifest["output"], "Manifest output/CLI mismatch")
    guard_label_aliases(manifest, runtime=True)
    verify_sources(manifest)
    require(shared.checkpoint_identity(manifest["checkpoint"]["path"]) == manifest["checkpoint"], "Checkpoint drift")
    prepared_scope(manifest)
    return manifest, output


def scope(manifest):
    return dict(dataset=manifest["dataset"], stage=manifest["stage"], seed=manifest["seed"],
                calibration=manifest["stage"] == "calibration", full_split=manifest["stage"] == "full")


def validate_cost(row, arm, cfg):
    if arm != "no_adapt":
        full.validate_cost(row, arm, cfg)
        return
    require(row["arm"] == arm and type(row["images"]) is int and row["images"] > 0
            and all(type(row[k]) is int for k in ("patches", "candidate_count", "active_candidates", "steps"))
            and row["patches"] == row["images"] and row["candidate_count"] == row["active_candidates"] == row["steps"] == 0
            and type(row["seconds"]) in (int, float) and math.isfinite(row["seconds"]) and row["seconds"] >= 0, "Invalid NoAdapt cost")
    require(all(type(row[k]) is int and row[k] >= 0 for k in COUNTERS), "Invalid actual NoAdapt counters")
    expected = {key: 0 for key in COUNTERS}
    expected.update(eval_forward_calls=1, eval_forward_inputs=row["patches"], encoder_calls=1, encoder_inputs=row["patches"])
    require(all(row[k] == v for k, v in expected.items()), "NoAdapt must be exact reset/evaluate only")


def artifact_names(manifest):
    n, batch = len(manifest["panel"]["inputs"]), manifest["config"]["batch_size"]
    return ({f"predictions/{a}/{i:06d}.npy" for a in manifest["arms"] for i in range(n)}
            | {f"episodes/{e:06d}.{a}.json" for a in manifest["arms"] for e in range((n + batch - 1) // batch)}
            | {"prediction_protocol.json", "initial_state.json", "prediction_result.json", "cost.jsonl"})


def build_method(manifest):
    cfg = manifest["config"]
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    return ExperimentTRACE(classes=manifest["panel"]["queries"], lr=cfg["lr"], steps=cfg["steps"],
        trace_max_regions=cfg["trace_max_regions"], trace_view_batch_size=cfg["trace_view_batch_size"],
        vision_outputs=confirmation.VISION_OUTPUTS, device="cuda", backbone=manifest["checkpoint"]["path"], view_mode="C")


def run_policy(method, arm, images, valid, ids, meta, cfg, mapping, initial):
    # Do not set ExperimentTRACE.arm to no_adapt (it is intentionally not one
    # of that class's policies), and never fabricate an adaptation call.
    if arm != "no_adapt":
        method.arm = arm
    tick, counts = time.perf_counter(), {}
    try:
        with finite_observation(method), method.count_work() as counts:
            shared.synchronize(method.device)
            tick = time.perf_counter()
            method.reset()
            require(shared.restored_state(method) == initial, "Pre-policy reset drift")
            losses = [] if arm == "no_adapt" else method.adapt(images, valid_mask=valid, image_ids=ids)
            predictions = shared.label_predictions(method, images, meta, cfg, mapping)
            shared.synchronize(method.device)
            seconds = time.perf_counter() - tick
        require(all(math.isfinite(v) for v in losses), "Nonfinite adaptation loss")
        stats = [] if arm == "no_adapt" else method.last_candidate_stats
        row = dict(arm=arm, images=len(meta["img_path"]), patches=len(images), seconds=seconds,
            steps=len(losses), candidate_count=len(stats), active_candidates=sum(r["reliability"] > EPS for r in stats), **counts)
        validate_cost(row, arm, cfg)
        return losses, predictions, stats, row
    except BaseException as error:
        error.worker_context = dict(arm=arm, images=len(meta["img_path"]), patches=len(images),
            seconds_until_failure=time.perf_counter() - tick, actual_partial_counters=dict(counts))
        raise


def execute_predictions(method, data, manifest, output):
    from trace_ovss.cli import build_patch_image_ids, build_patch_valid_mask
    cfg, panel, arms = manifest["config"], manifest["panel"], manifest["arms"]
    for arm in arms:
        (output / "predictions" / arm).mkdir(parents=True)
    (output / "episodes").mkdir()
    method.reset()
    initial = shared.restored_state(method)
    write_json(output / "initial_state.json", initial)
    frozen = [(p, p._version) for p in method.model.parameters() if not p.requires_grad]
    costs = []
    with (output / "cost.jsonl").open("x") as stream:
        for episode, start in enumerate(range(0, len(data), cfg["batch_size"])):
            stop = min(start + cfg["batch_size"], len(data))
            batch = shared.image_collate([data[i] for i in range(start, stop)])
            images, meta = batch["img_patches"].to(method.device), batch["meta"]
            valid = build_patch_valid_mask(meta, cfg["patch_size"], cfg["patch_stride"], method.device)
            ids = build_patch_image_ids(meta, method.device)
            rng, identity = shared.rng_state(method.device), None
            order = confirmation.prior.arm_order(arms, episode, "alternating")
            for position, arm in enumerate(order):
                shared.restore_rng(rng, method.device)
                try:
                    losses, predictions, stats, row = run_policy(method, arm, images, valid, ids, meta, cfg, panel["query_to_class"], initial)
                except BaseException as error:
                    error.worker_context = dict(getattr(error, "worker_context", {}), episode=episode,
                        order_position=position, ordered_ids=panel["ordered_ids"][start:stop], **scope(manifest))
                    raise
                require(all(p._version == version for p, version in frozen), "Frozen model weights changed")
                digest = confirmation.prior.candidate_identity(stats)
                if arm != "no_adapt":
                    identity = digest if identity is None else identity
                    require(digest == identity, "Candidate admission differs between adapt policies")
                row.update(episode=episode, order_position=position, warmup=episode == 0, candidate_sha256=digest, **scope(manifest))
                candidates = [] if arm == "no_adapt" else candidate_receipts(stats, arm, panel["query_to_class"])
                require(len(predictions) == stop - start, "Prediction count differs")
                for index, prediction in zip(range(start, stop), predictions, strict=True):
                    validate_prediction(prediction, len(panel["classes"]))
                    with (output / f"predictions/{arm}/{index:06d}.npy").open("xb") as destination:
                        np.save(destination, prediction, allow_pickle=False)
                write_json(output / f"episodes/{episode:06d}.{arm}.json", dict(row, losses=losses,
                    ordered_ids=panel["ordered_ids"][start:stop], candidates=candidates, gt_contents_opened=False))
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                costs.append(row)
                method.reset()
                require(shared.restored_state(method) == initial, "Post-policy reset drift")
                del predictions, stats, candidates
    totals = {arm: {k: sum(r[k] for r in costs if r["arm"] == arm) for k in COUNTERS} for arm in arms}
    return confirmation.prior.timing_summary(costs, arms), totals


def failure_receipt(error, phase, manifest_sha):
    oom = isinstance(error, torch.cuda.OutOfMemoryError)
    # Only observation-guard diagnostics, not incidental words in a path.
    text = str(error).lower()
    nonfinite = isinstance(error, RuntimeError) and text.startswith(("nonfinite ", "non-finite "))
    kind = "oom" if oom else "nonfinite" if nonfinite else "fatal_drift"
    return dict(status="failed", phase=phase, failure_kind=kind, ineligible_trial=kind in ("oom", "nonfinite"),
                fatal=kind == "fatal_drift", retry_allowed=False, manifest_sha256=manifest_sha,
                context=getattr(error, "worker_context", None), error=f"{type(error).__name__}: {error}"[:4000])


def predict(args):
    manifest, output = admission(args)
    require(manifest["stage"] in ("calibration", "full"), "Gate manifest cannot predict")
    data = verify_inputs(manifest)
    output.mkdir(parents=True, exist_ok=False)
    try:
        with confirmation.prior.gpu_lock() as lock:
            runtime = shared.configure_deterministic_runtime()
            write_json(output / "prediction_protocol.json", dict(manifest=manifest, manifest_path=str(args.manifest),
                manifest_sha256=args.manifest_sha256, output=str(output), gpu_lock=lock,
                deterministic_runtime=runtime, timing_scope=TIMING_SCOPE, gt_contents_opened=False))
            method = build_method(manifest)
            timings, costs = execute_predictions(method, data, manifest, output)
            admission(args)
            verify_inputs(manifest)
            require(shared.deterministic_runtime_state() == runtime, "Deterministic runtime drift")
            write_json(output / "prediction_result.json", dict(status="predicted", images=len(data), arms=manifest["arms"],
                config=manifest["config"], timings=timings, costs=costs, timing_scope=TIMING_SCOPE, **scope(manifest)))
            write_json(output / "PREDICTIONS_SEALED.json", dict(complete=True, gt_contents_opened=False, output=str(output),
                manifest_sha256=args.manifest_sha256, images=len(data), arms=manifest["arms"], **scope(manifest),
                suite_prepared_sha256=manifest.get("suite_prepared_sha256"),
                artifact_sha256={name: sha256(output / name) for name in sorted(artifact_names(manifest))}))
    except BaseException as error:
        write_json(output / "PREDICTION_FAILED.json", dict(failure_receipt(error, "predict", args.manifest_sha256),
            config=manifest["config"], checkpoint=manifest["checkpoint"], source_sha256=manifest["source_sha256"],
            output=str(output), **scope(manifest)))
        raise


def verify_prediction_seal(output, expected_manifest=None, expected_sha=None, *, hash_payload=True):
    output = absolute_path(str(output))
    protocol = read_json(output / "prediction_protocol.json")
    manifest, seal = protocol["manifest"], read_json(output / "PREDICTIONS_SEALED.json")
    validate_manifest(manifest)
    require(protocol["output"] == str(output) and seal["output"] == str(output), "Prediction output binding differs")
    path = absolute_path(protocol["manifest_path"])
    require(path.is_relative_to(Path(manifest["suite_root"])) and output.is_relative_to(Path(manifest["suite_root"]))
            and sha256(path) == protocol["manifest_sha256"] == seal["manifest_sha256"]
            and read_json(path) == manifest, "Prediction manifest binding differs")
    if "output" in manifest:
        require(manifest["output"] == str(output), "Manifest output differs")
    if expected_manifest is not None:
        require(manifest == expected_manifest and protocol["manifest_sha256"] == expected_sha, "CLI/sealed manifest differs")
    require(seal["complete"] is True and seal["gt_contents_opened"] is False and protocol["gt_contents_opened"] is False
            and seal["images"] == len(manifest["panel"]["inputs"]) and seal["arms"] == manifest["arms"]
            and all(seal[k] == v for k, v in scope(manifest).items())
            and seal.get("suite_prepared_sha256") == manifest.get("suite_prepared_sha256"), "Prediction scope differs")
    names = artifact_names(manifest)
    require(set(seal["artifact_sha256"]) == names, "Prediction payload set differs")
    require({str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()} <= names | SCORE_FILES | {"PREDICTIONS_SEALED.json"}, "Unexpected prediction payload")
    for name in names:
        path = output / name
        require(path.resolve() == path and path.is_file() and digest_valid(seal["artifact_sha256"][name]), "Aliased prediction payload")
        if hash_payload:
            require(sha256(path) == seal["artifact_sha256"][name], "Prediction payload drift: " + name)
    return manifest


def verify_barrier(path, manifest, output, manifest_sha):
    barrier = read_json(path)
    require(barrier["complete"] is True and barrier["stage"] == manifest["stage"]
            and barrier.get("gt_contents_opened", False) is False, "Wrong/incomplete stage barrier")
    if "suite_prepared_sha256" in manifest:
        require(barrier.get("suite_prepared_sha256") == manifest["suite_prepared_sha256"], "Barrier prepared binding differs")
    links = barrier["prediction_seal_sha256"]
    require(isinstance(links, dict) and str(output) in links, "Barrier missing own absolute output link")
    prepared = prepared_scope(manifest)
    expected_datasets = set(prepared["runnable"]) if prepared else set(DATASETS)
    if manifest["stage"] == "calibration":
        require(set(links) == {str(output)}, "Calibration barrier must cover exactly this trial")
    else:
        require(manifest["stage"] == "full" and len(links) == len(expected_datasets), "Full barrier requires exact prepared runnable datasets (eight without prepared scope)")
        if prepared:
            require(set(links) == {str(Path(manifest["suite_root"]) / "full" / d) for d in expected_datasets}, "Full barrier output scope differs from prepared suite")
    datasets = set()
    for directory, expected in links.items():
        peer_output = absolute_path(directory)
        require(digest_valid(expected) and sha256(peer_output / "PREDICTIONS_SEALED.json") == expected, "Barrier prediction seal drift")
        peer = verify_prediction_seal(peer_output, manifest if peer_output == output else None,
                                      manifest_sha if peer_output == output else None)
        require(peer["stage"] == manifest["stage"] and peer["suite_root"] == manifest["suite_root"]
                and peer["runtime_root"] == manifest["runtime_root"] and peer["source_sha256"] == manifest["source_sha256"]
                and peer.get("suite_prepared_sha256") == manifest.get("suite_prepared_sha256"), "Cross-stage/suite barrier scope")
        require(peer["dataset"] not in datasets, "Duplicate barrier dataset")
        prepared_scope(peer)
        datasets.add(peer["dataset"])
    require(datasets == (expected_datasets if manifest["stage"] == "full" else {manifest["dataset"]}), "Barrier dataset coverage differs")
    return sha256(path)


def validate_prediction(prediction, classes):
    require(prediction.ndim == 2 and prediction.size > 0 and np.issubdtype(prediction.dtype, np.integer)
            and (prediction >= 0).all() and (prediction < classes).all(), "Invalid prediction geometry/category")


def validate_cost_stream(output, manifest):
    cfg, arms, n = manifest["config"], manifest["arms"], len(manifest["panel"]["inputs"])
    episodes = (n + cfg["batch_size"] - 1) // cfg["batch_size"]
    rows = list(full.bounded_json_lines(output / "cost.jsonl", episodes * len(arms)))
    require(len(rows) == episodes * len(arms), "Incomplete actual costs")
    for episode in range(episodes):
        group = rows[episode * len(arms):(episode + 1) * len(arms)]
        require([r["arm"] for r in group] == confirmation.prior.arm_order(arms, episode, "alternating"), "Cost policy order differs")
        require(len({r["candidate_sha256"] for r in group if r["arm"] != "no_adapt"}) == 1, "Adapt candidate identities differ")
        for position, row in enumerate(group):
            require(row["episode"] == episode and row["order_position"] == position and row["warmup"] is (episode == 0)
                    and row["images"] == min(cfg["batch_size"], n - episode * cfg["batch_size"])
                    and all(row[k] == v for k, v in scope(manifest).items()), "Episode/stage cost scope differs")
            validate_cost(row, row["arm"], cfg)
    predicted = read_json(output / "prediction_result.json")
    costs = {a: {k: sum(r[k] for r in rows if r["arm"] == a) for k in COUNTERS} for a in arms}
    require(predicted["status"] == "predicted" and predicted["images"] == n and predicted["arms"] == arms
            and predicted["config"] == cfg and predicted["costs"] == costs and predicted["timing_scope"] == TIMING_SCOPE
            and predicted["timings"] == confirmation.prior.timing_summary(rows, arms)
            and all(predicted[k] == v for k, v in scope(manifest).items()), "Aggregate measured cost/timing differs")
    return predicted


def score(args):
    manifest, output = admission(args)
    require(manifest["stage"] in ("calibration", "full"), "Gate cannot score")
    require(not any((output / name).exists() for name in SCORE_FILES), "No score overwrite/resume")
    verify_inputs(manifest)
    barrier_sha = verify_barrier(args.barrier, manifest, output, args.manifest_sha256)
    label_reference, label_path = None, None
    if manifest["stage"] == "calibration":
        label_path = Path(manifest["suite_root"]) / "calibration_labels" / f"{manifest['dataset']}.json"
        if label_path.exists():
            require(manifest.get("calibration_label_seal_path") == str(label_path)
                    and manifest.get("calibration_label_seal_sha256") == sha256(label_path),
                    "Calibration label seal missing or changed")
            label_reference = read_json(label_path)
            require(label_reference["dataset"] == manifest["dataset"]
                    and label_reference["panel_sha256"] == canonical_hash(manifest["panel"])
                    and set(label_reference["label_sha256"]) == {r["label"] for r in manifest["panel"]["inputs"]},
                    "Calibration label reference scope differs")
        else:
            require("calibration_label_seal_path" not in manifest
                    and "calibration_label_seal_sha256" not in manifest, "Pinned calibration labels missing")
    # verify_barrier hashes every prediction payload exactly once, before GT.
    predicted = validate_cost_stream(output, manifest)
    prediction_sha = sha256(output / "PREDICTIONS_SEALED.json")
    write_json(output / "SCORING_STARTED.json", dict(manifest_sha256=args.manifest_sha256,
        prediction_seal_sha256=prediction_sha, barrier_sha256=barrier_sha, **scope(manifest)))
    cfg, panel, arms = manifest["config"], manifest["panel"], manifest["arms"]
    classes, n = len(panel["classes"]), len(panel["inputs"])
    totals = {a: np.zeros((2, classes), dtype=np.int64) for a in arms}
    hashes = {}
    with (output / "scores_by_episode.jsonl").open("x") as stream:
        for start in range(0, n, cfg["batch_size"]):
            counts = {a: np.zeros((2, classes), dtype=np.int64) for a in arms}
            for index in range(start, min(start + cfg["batch_size"], n)):
                row = panel["inputs"][index]
                confirmation.verify_label_metadata(row)
                content = Path(row["label"]).read_bytes()
                hashes[row["label"]] = hashlib.sha256(content).hexdigest()
                if label_reference is not None:
                    require(hashes[row["label"]] == label_reference["label_sha256"][row["label"]],
                            "Calibration label content changed between trials")
                first = np.load(output / f"predictions/{arms[0]}/{index:06d}.npy", allow_pickle=False)
                validate_prediction(first, classes)
                require(first.shape == tuple(cfg["resize"]), "Frozen prediction resolution differs")
                with Image.open(io.BytesIO(content)) as source:
                    gt = np.asarray(source.resize((first.shape[1], first.shape[0]), Image.Resampling.NEAREST))
                if gt.ndim == 3:
                    gt = gt[..., 0]
                gt = gt.astype(np.int64, copy=True)
                if panel["reduce_zero_label"]:
                    background, valid = gt == 0, gt != 255
                    gt[valid] -= 1
                    gt[background] = 255
                confirmation.verify_label_metadata(row)
                for arm in arms:
                    prediction = first if arm == arms[0] else np.load(output / f"predictions/{arm}/{index:06d}.npy", allow_pickle=False)
                    validate_prediction(prediction, classes)
                    require(prediction.shape == first.shape, "Prediction shape differs between policies")
                    counts[arm] += np.asarray(shared.streaming_counts(prediction, gt, classes))
            for arm in arms:
                totals[arm] += counts[arm]
            stream.write(json.dumps(dict(episode=start // cfg["batch_size"], images=min(cfg["batch_size"], n - start),
                arms={a: dict(intersection=v[0].tolist(), union=v[1].tolist()) for a, v in counts.items()})) + "\n")
    result = dict(status="completed", images=n, config=cfg, **scope(manifest),
        arm_results={a: dict(miou_unrounded=100 * math.fsum(int(x) / int(y) for x, y in zip(i, u) if y) / int((u > 0).sum())
                            if (u > 0).any() else None, intersection=i.tolist(), union=u.tolist()) for a, (i, u) in totals.items()},
        **{k: predicted[k] for k in ("timings", "costs", "timing_scope")}, manifest_sha256=args.manifest_sha256,
        prediction_seal_sha256=prediction_sha, barrier_sha256=barrier_sha,
        label_contents_first_opened_after_stage_predictions_sealed=True)
    require(sha256(args.barrier) == barrier_sha and sha256(output / "PREDICTIONS_SEALED.json") == prediction_sha, "Scoring binding drift")
    admission(args)
    verify_inputs(manifest)
    if label_path is not None:
        if label_reference is None:
            label_path.parent.mkdir(exist_ok=True)
            write_json(label_path, dict(dataset=manifest["dataset"], panel_sha256=canonical_hash(panel),
                label_sha256=hashes, first_manifest_sha256=args.manifest_sha256,
                first_prediction_seal_sha256=prediction_sha, first_barrier_sha256=barrier_sha,
                opened_only_after_prediction_seal=True))
        else:
            require(sha256(label_path) == manifest["calibration_label_seal_sha256"], "Calibration label seal drift")
        result["calibration_label_seal_sha256"] = sha256(label_path)
    write_json(output / "result.json", result)
    write_json(output / "score_inputs.json", dict(label_sha256=hashes, manifest_sha256=args.manifest_sha256, barrier_sha256=barrier_sha))
    write_json(output / "SCORED_SEALED.json", dict(complete=True, prediction_seal_sha256=prediction_sha, barrier_sha256=barrier_sha,
        artifact_sha256={name: sha256(output / name) for name in sorted(SCORE_FILES - {"SCORED_SEALED.json"})}))
    return result


def compare_gate_episode(method, reference, images, valid, ids, meta, cfg, mapping):
    method.reset()
    initial = shared.restored_state(method)
    rng, outcomes, actual = shared.rng_state(method.device), [], {}
    for current in (reference, method):
        current.reset()
        require(shared.restored_state(current) == initial, "Gate initial reset differs")
        if current is method:
            current.arm = "image_C"
        shared.restore_rng(rng, current.device)
        with full.gradient_receipt(current) as gradients:
            losses, predictions, stats, row = run_policy(current, "image_C", images, valid, ids, meta, cfg, mapping, initial)
        parameters = {name: p.detach().clone() for name, p in current.model.named_parameters() if p.requires_grad}
        outcomes.append((losses, predictions, copy.deepcopy(stats), gradients, parameters))
        actual["experiment" if current is method else "direct"] = row
        current.reset()
        require(shared.restored_state(current) == initial, "Gate final reset differs")
    a, b = outcomes
    require(confirmation.prior.candidate_identity(a[2]) == confirmation.prior.candidate_identity(b[2]), "Gate candidate masks differ")
    require(all(r["target_support"] is None for r in a[2] + b[2]), "Gate fabricated unmeasured T support")
    support = max((abs(x[k] - y[k]) for x, y in zip(a[2], b[2], strict=True)
                   for k in ("confidence", "counterfactual_support", "reliability")), default=0.)
    coefficient = max((abs(x - y) for x, y in zip(confirmation.objective_coefficients(a[2], "image_C"),
                       confirmation.objective_coefficients(b[2], "image_C"), strict=True)), default=0.)
    loss = max((abs(x - y) for x, y in zip(a[0], b[0], strict=True)), default=0.)
    require(set(a[3]) == set(b[3]) == set(a[4]) == set(b[4]), "Gate gradient/parameter membership differs")
    gradient_differences = []
    for name in a[3]:
        require(len(a[3][name]) == len(a[0]) and len(b[3][name]) == len(b[0]), "Gate missing actual step gradients")
        for x, y in zip(a[3][name], b[3][name], strict=True):
            full.require_finite([x, y], "gate gradient")
            gradient_differences.append(float((x - y).abs().max().item()))
    gradient = max(gradient_differences, default=0.)
    parameter = max((float((a[4][name] - b[4][name]).abs().max().item()) for name in a[4]), default=0.)
    require(len(a[1]) == len(b[1]) and all(x.shape == y.shape for x, y in zip(a[1], b[1])), "Gate final label shape differs")
    changed = sum(int(np.count_nonzero(x != y)) for x, y in zip(a[1], b[1]))
    require(all(math.isfinite(v) and v <= 1e-5 for v in (support, coefficient, loss, gradient, parameter)) and changed == 0,
            "Direct ImageBalancedTRACE C equivalence failed")
    return dict(candidates=len(a[2]), candidate_masks_equal=True, reset_equal=True,
        max_support_abs_difference=support, max_coefficient_abs_difference=coefficient,
        max_loss_abs_difference=loss, max_gradient_abs_difference=gradient, max_parameter_abs_difference=parameter,
        changed_label_pixels=changed, actual_arm_counts=actual, initial_state=initial,
        losses=dict(direct=a[0], experiment=b[0]),
        candidates_by_implementation=dict(direct=candidate_receipts(a[2], "image_C", mapping),
                                          experiment=candidate_receipts(b[2], "image_C", mapping)),
        final_parameter_sha256=dict(direct=shared.state_digest(a[4]), experiment=shared.state_digest(b[4])))


def gate(args):
    from trace_ovss.cli import build_patch_image_ids, build_patch_valid_mask
    manifest, output = admission(args)
    require(manifest["stage"] == "gate", "Gate stage manifest required")
    output.parent.mkdir(parents=True, exist_ok=True)
    claim = confirmation.claim_gate_attempt(output, args.manifest_sha256)
    output.mkdir(exist_ok=False)
    report = dict(passed=False, images=4, gt_contents_opened=False, manifest_sha256=args.manifest_sha256,
        attempt_sha256=sha256(claim), output=str(output), config=manifest["config"],
        manifest=manifest, manifest_path=str(args.manifest), suite_prepared_sha256=manifest.get("suite_prepared_sha256"),
        ordered_ids=manifest["panel"]["ordered_ids"][:4], **scope(manifest))
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
            comparisons, batches = [], []
            cfg = manifest["config"]
            for start in range(0, 4, cfg["batch_size"]):
                batch = shared.image_collate([data[i] for i in range(start, min(start + cfg["batch_size"], 4))])
                images, meta = batch["img_patches"].to(method.device), batch["meta"]
                valid = build_patch_valid_mask(meta, cfg["patch_size"], cfg["patch_stride"], method.device)
                ids = build_patch_image_ids(meta, method.device)
                comparisons.append(compare_gate_episode(method, reference, images, valid, ids, meta, cfg, manifest["panel"]["query_to_class"]))
                batches.append(len(meta["img_path"]))
            require(sum(r["candidates"] for r in comparisons) > 0, "Vacuous gate without candidates")
            admission(args)
            verify_inputs(manifest)
            require(shared.deterministic_runtime_state() == runtime, "Gate runtime drift")
            report.update(passed=True, comparisons=comparisons, batch_images=batches, gpu_lock=lock,
                deterministic_runtime=runtime, checkpoint=manifest["checkpoint"], source_sha256=manifest["source_sha256"])
        write_json(output / "gate.json", report)
        write_json(output / "GATE_PASSED.json", dict(report, gate_sha256=sha256(output / "gate.json")))
    except BaseException as error:
        target = output / ("failure.json" if (output / "gate.json").exists() else "gate.json")
        write_json(target, dict(report, failure=failure_receipt(error, "gate", args.manifest_sha256)))
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("predict", "score", "gate"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--barrier", type=Path)
    args = parser.parse_args(argv)
    if (args.phase == "score") != (args.barrier is not None):
        parser.error("--barrier is required only for --phase score")
    return {"predict": predict, "score": score, "gate": gate}[args.phase](args)


if __name__ == "__main__":
    main()
