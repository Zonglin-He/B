"""Fixed six-cell standalone follow-up. Preparation is CPU-only; run is explicit.

Usage (not an authorization to execute):
  python -B -m tools.evolution_followup.queue prepare --output /absolute/NEW_SUITE
  cd /absolute/NEW_SUITE/runtime
  /home/wwww/miniconda3/envs/mlmp/bin/python -B -m tools.evolution_followup.queue \
      run --suite /absolute/NEW_SUITE --suite-sha256 REVIEWED_SUITE_PREPARED_SHA256

Main owns systemd, CUDA compatibility and memory limits. No service, search,
historical-service dependency, retry, resume or result overwrite is implemented.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("pc60", "coco_object")
CONDITIONS = ("inherited_sorted", "common_sorted", "inherited_regrouped")
ARMS = ["Uniform", "Confidence", "T", "C", "min"]
PLAN_ARMS = ["uniform", "confidence", "target", "counterfactual", "trace"]
OVERRIDES = dict(batch_size=2, lr=.001, steps=10, trace_max_regions=16, trace_view_batch_size=16)
POPULATIONS = dict(pc60=5105, coco_object=5000)
PYTHON = Path("/home/wwww/miniconda3/envs/mlmp/bin/python")
PARENT = Path("/home/wwww/hzl/results/trace_self_evolution_20261028_1338")
TREES = ("trace_ovss", "tools", "tests")
LOG_BYTES = 256 * 1024
SNAPSHOT_BYTES = 64 * 1024 * 1024
EXPOSURE = ("Reused development-exposed evaluation images; fixed inherited/common transfer "
            "and dependent episode regrouping; descriptive, not untouched validation")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def worker():
    from tools.evolution_followup import standalone
    require(standalone.ROOT == ROOT, "Standalone imported outside this frozen source root")
    return standalone


def validate_plan(plan):
    require(plan["status"] == "design_locked_before_new_predictions", "Plan is not prelocked")
    require(plan["datasets"] == list(DATASETS) and plan["images_per_dataset"] == 128
            and plan["subset_selection_seed"] == 20261029 and plan["model_seed"] == 0,
            "Dataset/subset/model-seed scope changed")
    require(plan["arms"] == PLAN_ARMS, "Plan arms/order changed")
    require(plan["new_method"] is False and plan["parameter_search"] is False, "No method/search extension")
    require(plan["counts"] == dict(formal_cells=6, adaptation_arm_results=30, images_per_formal_cell=128),
            "Plan counts changed")
    expected = [dict(name="inherited_sorted", config="unchanged current dataset settings", episode_order_seed=None),
        dict(name="common_sorted", overrides=OVERRIDES, episode_order_seed=None),
        dict(name="inherited_regrouped", config="unchanged current dataset settings", episode_order_seed=20261030)]
    require(plan["conditions"] == expected, "Only the three prelocked conditions are allowed")


def read_parent(root, dataset):
    cell = Path(root) / "cells/mechanism" / dataset
    protocol_path, seal_path = cell / "prediction_protocol.json", cell / "PREDICTIONS_SEALED.json"
    seal = read_json(seal_path)
    require(seal.get("complete") is True and seal.get("gt_contents_opened") is False
            and seal["images"] == POPULATIONS[dataset], "Parent prediction seal incomplete")
    require(seal["artifact_sha256"]["prediction_protocol.json"] == sha256(protocol_path),
            "Parent protocol does not match PREDICTIONS_SEALED hash")
    protocol = read_json(protocol_path)
    require(protocol["dataset"] == dataset and protocol["seed"] == 0 and protocol["full_split"] is True
            and protocol["images"] == len(protocol["inputs"]) == POPULATIONS[dataset], "Wrong scalar parent scope")
    require(protocol["config"]["resize"] == protocol["config"]["patch_size"] == [224, 224],
            "This locked pilot requires inherited one-crop 224 preprocessing")
    worker().validate_config(protocol["config"], dataset)
    return protocol, dict(protocol_path=str(protocol_path.resolve()), protocol_sha256=sha256(protocol_path),
        seal_path=str(seal_path.resolve()), seal_sha256=sha256(seal_path))


def derive_config(parent, condition):
    require(condition in CONDITIONS, "Unknown condition")
    result = copy.deepcopy(parent["config"])
    if condition == "common_sorted":
        result.update(OVERRIDES)
    return result


def source_inventory(root):
    root = Path(root).resolve()
    inventory, total = {}, 0
    for tree in TREES:
        require((root / tree).is_dir(), f"Missing source tree: {tree}")
        for path in sorted((root / tree).rglob("*")):
            if any(part in ("__pycache__", ".pytest_cache", ".git") for part in path.parts):
                continue
            if path.suffix in (".pyc", ".pyo"):
                continue
            require(not path.is_symlink(), f"Source symlink is not snapshot-safe: {path}")
            if path.is_file():
                total += path.stat().st_size
                require(total <= SNAPSHOT_BYTES, "Source snapshot exceeds 64 MiB bound")
                inventory[str(path.relative_to(root))] = sha256(path)
    return inventory


def snapshot(source, destination):
    inventory = source_inventory(source)
    destination.mkdir()
    for relative in inventory:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    require(source_inventory(source) == inventory == source_inventory(destination), "Source changed during snapshot")
    return inventory


def check_hashes(root, inventory):
    root = Path(root).resolve()
    for relative, expected in inventory.items():
        path = root / relative
        require(path.resolve().is_relative_to(root) and not path.is_symlink(), "Inventory path escapes suite")
        require(sha256(path) == expected, f"Frozen artifact changed: {relative}")


def child(suite, runtime_hashes, python, arguments, label, *, cpu=False):
    """One foreground child, bounded log tail, no implicit retry or GPU lock."""
    require(source_inventory(suite / "runtime") == runtime_hashes, "Frozen runtime source changed before child")
    env = dict(os.environ, PYTHONPATH=str(suite / "runtime"), PYTHONDONTWRITEBYTECODE="1")
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    command = [str(python), "-B", "-m", *arguments]
    log = suite / "logs" / f"{label}.log"
    require(not log.exists(), "Child log already exists; no retry")
    tail, total = bytearray(), 0
    process = subprocess.Popen(command, cwd=suite / "runtime", env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        while True:
            block = process.stdout.read(64 * 1024)
            if not block:
                break
            total += len(block)
            tail.extend(block)
            if len(tail) > LOG_BYTES:
                del tail[:-LOG_BYTES]
        code = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        process.stdout.close()
        with log.open("xb") as stream:
            stream.write(tail)
    require(code == 0, f"Child failed ({code}): {label}; see bounded log")
    return dict(argv=command, log=str(log.relative_to(suite)), log_sha256=sha256(log),
                output_bytes=total, retained_bytes=len(tail), exit_code=code, cpu_only=cpu)


def validate_panels(suite, prepared):
    manifests, parents = {}, {}
    for dataset in DATASETS:
        parents[dataset] = read_json(suite / "parents" / dataset / "prediction_protocol.json")
    for entry in [*prepared["cells"], *prepared["smokes"]]:
        manifest = worker().verify_manifest_file(suite / entry["manifest"], entry["manifest_sha256"])
        dataset, parent = entry["dataset"], parents[entry["dataset"]]
        require(manifest["dataset"] == dataset and manifest["seed"] == 0
                and manifest["selection_seed"] == 20261029 and manifest["max_images"] == entry["images"]
                and manifest["episode_order_seed"] == entry["episode_order_seed"]
                and manifest["arms"] == entry["arms"] and manifest["arm_order"] == "alternating",
                "Manifest differs from locked cell")
        require(manifest["config"] == derive_config(parent, entry["condition"]), "Config not derived from sealed parent")
        require(manifest["checkpoint"] == parent["checkpoint"], "Checkpoint differs from parent")
        config_path = (suite / entry["config"]).resolve()
        require(Path(manifest["config_path"]) == config_path and manifest["config_sha256"] == sha256(config_path),
                "Manifest config path/hash mismatch")
        source_keys = [str(Path(p).relative_to(ROOT)) for p in worker().checked_sources()]
        expected_sources = {str((suite / "runtime" / key).resolve()): prepared["runtime_hashes"][key] for key in source_keys}
        require(manifest["source_sha256"] == expected_sources, "Manifest source binding differs from frozen runtime")
        panel = manifest["panel"]
        for key in ("classes", "queries", "query_to_class", "reduce_zero_label"):
            require(panel[key] == parent[key], f"Parent query/label convention changed: {key}")
        parent_inputs = {r["image"]: r for r in parent["inputs"]}
        for row in panel["inputs"]:
            require({k: row[k] for k in ("image", "label", "image_sha256", "label_metadata")}
                    == parent_inputs.get(row["image"]), "Selected input differs from sealed parent identity")
        manifests[entry["key"]] = manifest
    for dataset in DATASETS:
        sorted_panel = manifests[f"{dataset}/inherited_sorted"]["panel"]
        common = manifests[f"{dataset}/common_sorted"]["panel"]
        regrouped = manifests[f"{dataset}/inherited_regrouped"]["panel"]
        require(sorted_panel == common, "Common config changed image population/order/vocabulary")
        require(sorted_panel["ordered_ids"] != regrouped["ordered_ids"], "Regrouping did not change order")
        a, b = copy.deepcopy(sorted_panel), copy.deepcopy(regrouped)
        for panel in (a, b):
            panel["inputs"] = sorted(panel["inputs"], key=lambda r: r["dataset_index"])
            panel.pop("ordered_ids")
        require(a == b, "Regrouping changed more than order")
        smoke = manifests[f"smoke/{dataset}"]["panel"]
        require({r["image"] for r in smoke["inputs"]} <= {r["image"] for r in sorted_panel["inputs"]},
                "Smoke images are not in fixed formal subset")
    return manifests


def prepare(args):
    source, suite = args.source_root.resolve(), args.output.resolve()
    require(not any(suite.is_relative_to(source / tree) for tree in TREES), "Output cannot be inside source trees")
    plan = read_json(args.plan)
    validate_plan(plan)
    parents = {d: read_parent(args.parent_root, d) for d in DATASETS}
    require(parents[DATASETS[0]][0]["checkpoint"] == parents[DATASETS[1]][0]["checkpoint"], "Parent checkpoints differ")
    require(args.python.is_file(), "Existing CUDA environment Python required")
    suite.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "configs", "manifests", "parents"):
        (suite / name).mkdir()
    state(suite, "snapshot")
    shutil.copyfile(args.plan, suite / "frozen_plan.json")
    require(read_json(suite / "frozen_plan.json") == plan
            and sha256(args.plan) == sha256(suite / "frozen_plan.json"), "Plan changed while copying")
    runtime_hashes = snapshot(source, suite / "runtime")
    prepared = dict(format_version=1, worker="standalone_queue", plan_sha256=sha256(suite / "frozen_plan.json"),
        runtime_root=str(suite / "runtime"), runtime_hashes=runtime_hashes,
        python=str(args.python.absolute()), python_sha256=sha256(args.python), cells=[], smokes=[], parents={},
        arm_order="alternating", arms=ARMS, exposure_note=EXPOSURE, child_receipts=[])
    for dataset, (parent, binding) in parents.items():
        destination = suite / "parents" / dataset
        destination.mkdir()
        shutil.copyfile(binding["protocol_path"], destination / "prediction_protocol.json")
        shutil.copyfile(binding["seal_path"], destination / "PREDICTIONS_SEALED.json")
        require(sha256(destination / "prediction_protocol.json") == binding["protocol_sha256"]
                and sha256(destination / "PREDICTIONS_SEALED.json") == binding["seal_sha256"], "Parent changed during copy")
        require(read_parent_snapshot(suite, dataset)[0] == parent, "Copied parent differs from validated scalar protocol")
        prepared["parents"][dataset] = binding
        for condition in CONDITIONS:
            relative = f"configs/{dataset}.{condition}.json"
            write_json(suite / relative, derive_config(parent, condition))
            prepared["cells"].append(dict(key=f"{dataset}/{condition}", dataset=dataset, condition=condition,
                config=relative, manifest=f"manifests/{dataset}.{condition}.json", images=128, arms=ARMS,
                episode_order_seed=20261030 if condition == "inherited_regrouped" else None,
                output=f"cells/{dataset}/{condition}"))
        prepared["smokes"].append(dict(key=f"smoke/{dataset}", dataset=dataset, condition="inherited_sorted",
            config=f"configs/{dataset}.inherited_sorted.json", manifest=f"manifests/{dataset}.smoke.json",
            images=4, arms=["min"], episode_order_seed=None, output=f"smoke/{dataset}.json"))
    base_inventory = {str(p.relative_to(suite)): sha256(p) for folder in ("configs", "parents")
                      for p in (suite / folder).rglob("*") if p.is_file()}
    base_inventory["frozen_plan.json"] = prepared["plan_sha256"]
    for entry in [*prepared["cells"], *prepared["smokes"]]:
        state(suite, "prepare", entry["key"])
        check_hashes(suite, base_inventory)
        require(sha256(args.python) == prepared["python_sha256"], "Python binary changed")
        command = ["tools.evolution_followup.standalone", "prepare", "--dataset", entry["dataset"],
            "--config", str(suite / entry["config"]), "--seed", "0", "--selection-seed", "20261029",
            "--max-images", str(entry["images"]), "--arms", *entry["arms"], "--arm-order", "alternating",
            "--checkpoint", parents[entry["dataset"]][0]["checkpoint"]["path"],
            "--exposure-note", EXPOSURE, "--manifest", str(suite / entry["manifest"])]
        if entry["episode_order_seed"] is not None:
            command += ["--episode-order-seed", str(entry["episode_order_seed"])]
        receipt = child(suite, runtime_hashes, args.python, command, "prepare-" + entry["key"].replace("/", "-"), cpu=True)
        prepared["child_receipts"].append(receipt)
        entry["manifest_sha256"] = sha256(suite / entry["manifest"])
    validate_panels(suite, prepared)
    prepared["artifact_sha256"] = {**base_inventory,
        **{entry["manifest"]: entry["manifest_sha256"] for entry in [*prepared["cells"], *prepared["smokes"]]},
        **{r["log"]: r["log_sha256"] for r in prepared["child_receipts"]}}
    check_hashes(suite, prepared["artifact_sha256"])
    require(source_inventory(suite / "runtime") == runtime_hashes, "Snapshot changed during CPU prepare")
    write_json(suite / "SUITE_PREPARED.json", prepared)
    state(suite, "prepared")
    print(json.dumps(dict(suite=str(suite), suite_sha256=sha256(suite / "SUITE_PREPARED.json"))))


def verify_prepared(suite, expected_hash, *, frozen=False):
    require(sha256(suite / "SUITE_PREPARED.json") == expected_hash, "Prepared-suite hash mismatch")
    prepared = read_json(suite / "SUITE_PREPARED.json")
    require(prepared["format_version"] == 1 and prepared["worker"] == "standalone_queue", "Wrong prepared suite")
    require(Path(prepared["runtime_root"]) == suite / "runtime", "Snapshot relocated")
    if frozen:
        require(ROOT == suite / "runtime" and Path.cwd().resolve() == ROOT,
                "Run controller from the frozen snapshot, not live source")
    require(source_inventory(suite / "runtime") == prepared["runtime_hashes"], "Frozen runtime changed")
    check_hashes(suite, prepared["artifact_sha256"])
    require(sha256(prepared["python"]) == prepared["python_sha256"], "Frozen Python binary changed")
    plan = read_json(suite / "frozen_plan.json")
    validate_plan(plan)
    require(sha256(suite / "frozen_plan.json") == prepared["plan_sha256"], "Frozen plan changed")
    require([(e["dataset"], e["condition"]) for e in prepared["cells"]]
            == [(d, c) for d in DATASETS for c in CONDITIONS], "Six-cell population/order changed")
    require([e["dataset"] for e in prepared["smokes"]] == list(DATASETS), "Two smoke gates required")
    for entry in [*prepared["cells"], *prepared["smokes"]]:
        smoke = entry["key"].startswith("smoke/")
        expected_key = f"smoke/{entry['dataset']}" if smoke else f"{entry['dataset']}/{entry['condition']}"
        require(entry["key"] == expected_key and entry["images"] == (4 if smoke else 128)
                and entry["arms"] == (["min"] if smoke else ARMS)
                and entry["episode_order_seed"] == (20261030 if entry["condition"] == "inherited_regrouped" else None),
                "Cell scope changed")
        require((suite / entry["output"]).resolve().is_relative_to(suite), "Output path escape")
        stem = f"{entry['dataset']}.{'smoke' if smoke else entry['condition']}"
        require(entry["manifest"] == f"manifests/{stem}.json"
                and entry["config"] == f"configs/{entry['dataset']}.{entry['condition']}.json"
                and entry["output"] == (f"smoke/{entry['dataset']}.json" if smoke
                                        else f"cells/{entry['dataset']}/{entry['condition']}"), "Cell paths changed")
    for dataset in DATASETS:
        parent, _ = read_parent_snapshot(suite, dataset)
        require(parent["dataset"] == dataset, "Wrong frozen parent")
    validate_panels(suite, prepared)
    return prepared


def read_parent_snapshot(suite, dataset):
    # No live historical output or service is needed once these scalar files are frozen.
    root = suite / "parents" / dataset
    protocol, seal = read_json(root / "prediction_protocol.json"), read_json(root / "PREDICTIONS_SEALED.json")
    require(seal["complete"] is True and seal["gt_contents_opened"] is False
            and seal["artifact_sha256"]["prediction_protocol.json"] == sha256(root / "prediction_protocol.json"),
            "Frozen parent protocol linkage changed")
    return protocol, seal


def state(suite, phase, active=None, *, gates=0, predictions=0, scores=0, error=None):
    temporary = suite / "queue_state.tmp"
    write_json(temporary, dict(phase=phase, active_cell=active, passed_smokes=gates,
        sealed_predictions=predictions, scored_cells=scores, total_cells=6, updated_unix=time.time(), error=error))
    temporary.replace(suite / "queue_state.json")


@contextmanager
def queue_lock(suite):
    with (suite / "queue.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate_smoke(suite, entry):
    report = read_json(suite / entry["output"])
    manifest = read_json(suite / entry["manifest"])
    require(report["passed"] is True and report["gt_contents_opened"] is False
            and report["images"] == 4 and report["dataset"] == entry["dataset"]
            and report["manifest_sha256"] == entry["manifest_sha256"]
            and report["candidate_masks_equal"] is True and report["same_steps"] is True
            and report["max_support_abs_difference"] <= 1e-5 and report["max_loss_abs_difference"] <= 1e-5
            and report["changed_label_pixels"] == 0 and report["atol"] == 1e-5
            and report["finite"] is True and report["count_checks_passed"] is True,
            "GT-free smoke equivalence gate failed")
    require(report["suite_prepared_sha256"] == sha256(suite / "SUITE_PREPARED.json")
            and report["source_sha256"] == manifest["source_sha256"]
            and report["checkpoint"] == manifest["checkpoint"]
            and report["ordered_ids"] == manifest["panel"]["ordered_ids"]
            and report["input_image_sha256"] == {r["image"]: r["image_sha256"] for r in manifest["panel"]["inputs"]}
            and report["prediction_shapes"] == [[224, 224]] * 4
            and report["total_label_pixels"] == 4 * 224 * 224 and report["candidates"] > 0,
            "Smoke scope/source/input/output-shape binding changed")
    require(set(report["runs"]) == {"native", "standalone_min"}
            and all(r["steps"] == manifest["config"]["steps"] for r in report["runs"].values()),
            "Smoke incomplete native/standalone adaptation")
    strict_runtime(report["deterministic_runtime"])
    require(report["gpu_lock"]["path"] == str(worker().GPU_LOCK) and report["gpu_lock"]["exclusive"] is True,
            "Smoke did not own shared global GPU lock")
    return sha256(suite / entry["output"])


def strict_runtime(runtime):
    require(runtime["cublas_workspace_config"] == ":4096:8"
            and runtime["deterministic_algorithms"] is True and runtime["deterministic_warn_only"] is False
            and runtime["deterministic_debug_mode"] == 2 and runtime["cudnn_deterministic"] is True
            and runtime["cudnn_benchmark"] is False, "Runtime determinism contract changed")


def validate_prediction(suite, entry):
    output = suite / entry["output"]
    protocol = worker().verify_prediction_seal(output)
    require(protocol["manifest_sha256"] == entry["manifest_sha256"]
            and protocol["manifest"] == read_json(suite / entry["manifest"]), "Prediction not bound to exact frozen manifest")
    require(protocol["manifest_path"] == str((suite / entry["manifest"]).resolve()), "Prediction manifest path differs")
    smoke_report = read_json(suite / f"smoke/{entry['dataset']}.json")
    strict_runtime(protocol["deterministic_runtime"])
    require(protocol["deterministic_runtime"] == smoke_report["deterministic_runtime"],
            "Smoke/formal deterministic runtime or dispatch settings differ")
    require(protocol["gpu_lock"]["path"] == str(worker().GPU_LOCK) and protocol["gpu_lock"]["exclusive"] is True,
            "Prediction did not own shared global GPU lock")
    result = read_json(output / "prediction_result.json")
    require(result["status"] == "predicted" and result["images"] == 128 and result["full_split"] is False
            and result["timing_scope"] == worker().TIMING_SCOPE and result["accuracy_includes_warmup"] is True,
            "Formal prediction result incomplete")
    return sha256(output / "PREDICTIONS_SEALED.json")


def validate_score(suite, entry):
    output = suite / entry["output"]
    pred_hash = validate_prediction(suite, entry)
    scored = read_json(output / "SCORED_SEALED.json")
    require(scored["prediction_seal_sha256"] == pred_hash, "Score seal not linked to new prediction")
    require(set(scored["artifact_sha256"]) == {"SCORING_STARTED.json", "scores_by_episode.jsonl", "score_inputs.json", "result.json"},
            "Score artifact coverage changed")
    check_hashes(output, scored["artifact_sha256"])
    require(read_json(output / "SCORING_STARTED.json")["prediction_seal_sha256"] == pred_hash,
            "Scoring-start marker does not bind new predictions")
    result = read_json(output / "result.json")
    require(result["status"] == "completed" and result["dataset"] == entry["dataset"] and result["seed"] == 0
            and result["images"] == 128 and result["full_split"] is False and result["prediction_seal_sha256"] == pred_hash
            and result["label_contents_first_opened_after_all_requested_arms_sealed"] is True
            and result["timing_scope"] == worker().TIMING_SCOPE and result["exposure_note"] == EXPOSURE
            and result["claim_boundary"] == read_json(suite / entry["manifest"])["claim_boundary"]
            and set(result["arm_results"]) == set(ARMS) == set(result["timings"]), "Formal score scope incomplete")
    require(result["timings"] == read_json(output / "prediction_result.json")["timings"], "Scoring changed timing metadata")
    with (output / "cost.jsonl").open() as stream:
        costs = [json.loads(line) for line in stream]
    require(len(costs) <= 128 * 5 and worker().timing_summary(costs, ARMS) == result["timings"],
            "Timing denominators differ from actual new cost stream")
    for arm in ARMS:
        value = result["arm_results"][arm]["miou_unrounded"]
        require(value is None or math.isfinite(value) and 0 <= value <= 100, "Invalid mIoU")
    return result


def run(args):
    suite = args.suite.resolve()
    prepared = verify_prepared(suite, args.suite_sha256, frozen=True)
    gates = predictions = scores = 0
    with queue_lock(suite):
        require(not any((suite / p).exists() for p in ("EXECUTION_STARTED.json", "SUITE_SEALED.json", "cells", "smoke")),
                "No execution retry/resume/overwrite")
        write_json(suite / "EXECUTION_STARTED.json", dict(suite_sha256=args.suite_sha256, pid=os.getpid()))
        receipts = []
        def launch(arguments, label, cpu=False):
            verify_prepared(suite, args.suite_sha256, frozen=True)
            receipts.append(child(suite, prepared["runtime_hashes"], Path(prepared["python"]), arguments, label, cpu=cpu))
        try:
            (suite / "smoke").mkdir()
            smoke_hashes = {}
            for entry in prepared["smokes"]:
                state(suite, "smoke", entry["key"], gates=gates)
                launch(["tools.evolution_followup.smoke", "--suite", str(suite), "--suite-sha256", args.suite_sha256,
                        "--dataset", entry["dataset"]], "smoke-" + entry["dataset"])
                smoke_hashes[entry["key"]] = validate_smoke(suite, entry)
                gates += 1
            pred_hashes = {}
            for entry in prepared["cells"]:
                state(suite, "predict", entry["key"], gates=gates, predictions=predictions)
                launch(["tools.evolution_followup.standalone", "predict", "--manifest", str(suite / entry["manifest"]),
                        "--manifest-sha256", entry["manifest_sha256"], "--output", str(suite / entry["output"])],
                       "predict-" + entry["key"].replace("/", "-"))
                pred_hashes[entry["key"]] = validate_prediction(suite, entry)
                predictions += 1
            require({e["key"]: validate_prediction(suite, e) for e in prepared["cells"]} == pred_hashes,
                    "Prediction barrier changed")
            require(not any((suite / e["output"] / name).exists() for e in prepared["cells"]
                            for name in ("SCORING_STARTED.json", "result.json", "SCORED_SEALED.json")),
                    "A formal cell opened scoring before the six-cell prediction barrier")
            write_json(suite / "ALL_PREDICTIONS_SEALED.json", dict(suite_sha256=args.suite_sha256,
                smoke_sha256=smoke_hashes, prediction_seal_sha256=pred_hashes, labels_opened_by_queue=False))
            results = []
            for entry in prepared["cells"]:
                state(suite, "score", entry["key"], gates=gates, predictions=predictions, scores=scores)
                launch(["tools.evolution_followup.standalone", "score", "--output", str(suite / entry["output"])],
                       "score-" + entry["key"].replace("/", "-"), cpu=True)
                results.append((entry, validate_score(suite, entry)))
                scores += 1
            verify_prepared(suite, args.suite_sha256, frozen=True)
            require({e["key"]: validate_smoke(suite, e) for e in prepared["smokes"]} == smoke_hashes, "Smoke gate changed")
            require({e["key"]: validate_prediction(suite, e) for e in prepared["cells"]} == pred_hashes, "Final prediction changed")
            for entry in prepared["cells"]:
                validate_score(suite, entry)
            with (suite / "rows.csv").open("x", newline="") as stream:
                columns = ["dataset", "condition", "arm", "images", "full_split", "miou_unrounded",
                           "measured_images", "seconds", "seconds_per_image", "images_per_second"]
                table = csv.DictWriter(stream, fieldnames=columns)
                table.writeheader()
                for entry, result in results:
                    for arm in ARMS:
                        timing = result["timings"][arm]
                        table.writerow(dict(dataset=entry["dataset"], condition=entry["condition"], arm=arm,
                            images=128, full_split=False, miou_unrounded=result["arm_results"][arm]["miou_unrounded"],
                            **{k: timing[k] for k in columns[6:]}))
            write_json(suite / "execution_children.json", receipts)
            state(suite, "completed", gates=gates, predictions=predictions, scores=scores)
            inventory = dict(prepared["artifact_sha256"])
            for relative in ("SUITE_PREPARED.json", "EXECUTION_STARTED.json", "ALL_PREDICTIONS_SEALED.json",
                             "rows.csv", "execution_children.json", "queue_state.json"):
                inventory[relative] = sha256(suite / relative)
            for entry in prepared["smokes"]:
                inventory[entry["output"]] = sha256(suite / entry["output"])
            for entry in prepared["cells"]:
                for name in ("PREDICTIONS_SEALED.json", "SCORED_SEALED.json", "result.json"):
                    inventory[entry["output"] + "/" + name] = sha256(suite / entry["output"] / name)
            inventory.update({r["log"]: r["log_sha256"] for r in receipts})
            check_hashes(suite, inventory)
            write_json(suite / "SUITE_SEALED.json", dict(complete=True, formal_cells=6, arm_results=30,
                passed_gt_free_gates=2, full_split=False, suite_prepared_sha256=args.suite_sha256,
                runtime_hashes=prepared["runtime_hashes"], artifact_sha256=inventory,
                payload_verification="All cell prediction and score payloads rehashed via their linked seals",
                exposure_note=EXPOSURE))
        except BaseException as error:
            state(suite, "failed", read_json(suite / "queue_state.json").get("active_cell")
                  if (suite / "queue_state.json").exists() else None,
                  gates=gates, predictions=predictions, scores=scores, error=f"{type(error).__name__}: {error}"[:2000])
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = result.add_subparsers(dest="phase", required=True)
    prep = sub.add_parser("prepare", help="CPU-only; new source snapshot, configs and manifests")
    prep.add_argument("--source-root", type=Path, default=ROOT)
    prep.add_argument("--plan", type=Path, default=ROOT / "docs/trace-standalone-followup-plan.json")
    prep.add_argument("--parent-root", type=Path, default=PARENT)
    prep.add_argument("--python", type=Path, default=PYTHON)
    prep.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("run", help="Explicit GPU execution; frozen runtime cwd required")
    execute.add_argument("--suite", type=Path, required=True)
    execute.add_argument("--suite-sha256", required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    fresh = args.phase == "prepare" and not args.output.exists()
    try:
        if args.phase == "prepare":
            prepare(args)
        else:
            run(args)
    except BaseException as error:
        if fresh and (args.output / "queue_state.json").exists():
            state(args.output, "failed", read_json(args.output / "queue_state.json").get("active_cell"),
                  error=f"{type(error).__name__}: {error}"[:2000])
        raise


if __name__ == "__main__":
    main()
