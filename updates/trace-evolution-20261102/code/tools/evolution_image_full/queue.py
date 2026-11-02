"""Bounded six-dataset controller. Preparation/verification are CPU-only.

Main owns the independently audited launch and the 12 GiB systemd service.
No service creation, search, retry, resume or historical runtime edits occur here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import time

from tools.evolution_confirmation import queue as old
from tools.evolution_followup import queue as audited

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("pc60", "coco_object", "voc21", "voc20", "coco_cd", "coco_stuff")
POPULATIONS = dict(zip(DATASETS, (5105, 5000, 1449, 1449, 500, 5000)))
ARMS = ["image_uniform", "image_confidence", "image_C"]
WORKER = "tools.evolution_image_full.run"
PLAN_SHA256 = "598bc2715e959fc03421f28748ffe706522ec5c7757ca9fedb03aee959952dd8"
SELECTION_ROOT = Path("/home/wwww/hzl/results/trace_image_budget_20261029_0233")
SELECTION_SHA256 = "90fa3a20097698e9b83cafd416a4b05fe7f033dcd8aadb027c970a9788ac11e4"
SELECTION_PREPARED_SHA256 = "579372bf5a4208ad9e1f465698b0d82193ecd8d7c70162b3b1b3c67c84f056c5"
PARENT_ROOT = Path("/home/wwww/hzl/results/trace_self_evolution_20261028_1338")
VOC20_ROOT = Path("/home/wwww/hzl/results/trace_evolution_voc20_cli_recovery_20261028_2312")
PARENT_HASHES = {
    "pc60": ("9a7b1cd20e72c04d6e44db4f6a2f1f6520bcc11ceaeb93035cdeaf961802ef5e", "0b3b9c7dcf128e1229e281b1f9b54627244d7abf3311678203e31b9703c491ac"),
    "coco_object": ("397972602aec47b40d0f705a770583d5c2f45e0e2ff16eb1383c645fc8f4d743", "63e7b5e08963ee82a1a5f519cdf658b07509c68a5baef64f54f3c936e8d74e1e"),
    "voc21": ("9dcfb9d42a8a9ee463c4a6c2f4be805d2e44e6454deaa3f3aa53b7ca4d99afb0", "2f3155446fc285cce4e963e52c41611bae97f3eacdfdc32ee38c8ca2f053c0a1"),
    "voc20": ("709b0502e156e908e52069896892320a2925bb318cacc9e4c765eb3ce232b025", "e87fe99865d9d07478da842659c6e899a44809ba898daa76f264f820aecd480f"),
    "coco_cd": ("1b08b148344f139dc31e7b7285ab556e12e40fd3de3c1ae468da79f570b25b35", "9fe16315093b2af05744e4008ddb495fa7318ead34ef44bfbbf9384dfff22c34"),
    "coco_stuff": ("f3d82494be65465f44ce49645b8659bfeb601ea599cd4418277e3093265c12ec", "43c16da245e60a58a7ed9dcf3feb7483f284f1410da93d0a2f7d3cea08407842"),
}
SCORING_FILES = old.SCORING_FILES
INHERITED_TREES = ("trace_ovss/", "tools/evolution/", "tools/evolution_confirmation/", "tools/evolution_followup/")
INHERITED_FILES = ("tools/suite_inputs.py", "tools/trace_algorithm_variants.py", "tools/trace_execution_audit.py")
BARRIER = "ALL_FULL_PREDICTIONS_SEALED.json"
EXPOSURE = "Development-exposed official full splits and inherited per-dataset configurations; fixed C, not untouched or equal-tuning confirmation."
require, sha256, read_json, write_json = audited.require, audited.sha256, audited.read_json, audited.write_json
check_hashes, source_inventory = audited.check_hashes, audited.source_inventory
child, pooled_miou = old.child, old.pooled_miou


def worker():
    from tools.evolution_image_full import run
    require(run.ROOT == ROOT and old.ROOT == audited.ROOT == ROOT, "Cross-root worker/helper imports")
    return run


def cells():
    return [dict(key=f"full/{d}/seed0", dataset=d, images=POPULATIONS[d], arms=ARMS.copy(),
                 config=f"configs/{d}.json", manifest=f"manifests/{d}.json", output=f"cells/full/{d}/seed0")
            for d in DATASETS]


def cells_with_hashes(suite):
    return read_json(Path(suite) / "SUITE_PREPARED.json")["cells"]


def validate_plan(plan):
    require(plan["schema_version"] == 1 and plan["status"] == "locked_before_new_full_predictions"
            and plan["algorithm"] == "ImageBalancedTRACE" and plan["selected_global_mode"] == "C"
            and plan["datasets"] == list(DATASETS) and plan["populations"] == POPULATIONS
            and plan["arms"] == ARMS and plan["model_seed"] == 0
            and (plan["formal_cells"], plan["arm_results"], plan["gt_free_gates"], plan["gate_images_per_dataset"]) == (6, 18, 6, 4)
            and plan["selection_parent"] == str(SELECTION_ROOT)
            and plan["selection_sha256"] == SELECTION_SHA256
            and plan["selection_prepared_sha256"] == SELECTION_PREPARED_SHA256, "Full plan scope changed")


def parent_directory(dataset):
    require(dataset in DATASETS, "Unknown parent dataset")
    return (VOC20_ROOT if dataset == "voc20" else PARENT_ROOT) / "cells/mechanism" / dataset


def read_parent(dataset):
    directory = parent_directory(dataset)
    protocol_path, seal_path = directory / "prediction_protocol.json", directory / "PREDICTIONS_SEALED.json"
    protocol_sha, seal_sha = PARENT_HASHES[dataset]
    require(sha256(protocol_path) == protocol_sha and sha256(seal_path) == seal_sha, "Pinned parent drift")
    protocol, seal = read_json(protocol_path), read_json(seal_path)
    require(seal["complete"] is True and seal["gt_contents_opened"] is False
            and seal["images"] == POPULATIONS[dataset]
            and seal["artifact_sha256"]["prediction_protocol.json"] == protocol_sha, "Unbound parent protocol")
    validate_parent(protocol, dataset)
    return protocol, dict(protocol_path=str(protocol_path), protocol_sha256=protocol_sha,
                         seal_path=str(seal_path), seal_sha256=seal_sha,
                         complete=True, gt_contents_opened=False, images=seal["images"],
                         sealed_protocol_sha256=protocol_sha)


def validate_parent(protocol, dataset):
    require(protocol["dataset"] == dataset and type(protocol["seed"]) is int and protocol["seed"] == 0
            and protocol["full_split"] is True
            and protocol["images"] == protocol["available_images"] == len(protocol["inputs"]) == POPULATIONS[dataset],
            "Wrong inherited full scope")
    rows = protocol["inputs"]
    for key in ("image", "label"):
        require(len({r[key] for r in rows}) == len(rows)
                and all(Path(r[key]).is_absolute() for r in rows), "Duplicate/nonabsolute parent paths")
        physical = [(s.st_dev, s.st_ino) for r in rows for s in (Path(r[key]).stat(),)]
        require(len(set(physical)) == len(rows), "Duplicate physical parent input")
    require([r["image"] for r in rows] == sorted(r["image"] for r in rows), "Parent not in original official order")
    require(protocol["config"]["resize"] == protocol["config"]["patch_size"] == [224, 224], "Parent preprocessing changed")
    require(all(old.valid_label_metadata(r["label_metadata"]) and len(r["image_sha256"]) == 64 for r in rows),
            "Parent input identity incomplete")


def selection_proof_files(root):
    """Only the sealed calibration evidence is used to verify the old choice."""
    selection = read_json(root / "SELECTION_SEALED.json")
    require(sha256(root / "SELECTION_SEALED.json") == SELECTION_SHA256
            and sha256(root / "SUITE_PREPARED.json") == SELECTION_PREPARED_SHA256
            and selection["complete"] is True and selection["selected_mode"] == "C", "Pinned C selection differs")
    completed = read_json(root / "SUITE_SEALED.json")
    require(completed["complete"] is True and completed["selected_mode"] == "C"
            and completed["suite_prepared_sha256"] == SELECTION_PREPARED_SHA256
            and completed["selection_sha256"] == SELECTION_SHA256
            and completed["formal_cells"] == 8 and completed["arm_results"] == 48, "Selection parent not completed")
    expected = {f"cells/calibration/{d}/sorted" for d in ("pc60", "coco_object")}
    require(set(selection["calibration_score_sha256"]) == expected, "Non-calibration selection evidence")
    files = {"SELECTION_SEALED.json", "SUITE_PREPARED.json", "SUITE_SEALED.json", "frozen_plan.json",
             "ALL_CALIBRATION_PREDICTIONS_SEALED.json"}
    for relative in sorted(expected):
        seal = read_json(root / relative / "SCORED_SEALED.json")
        require(set(seal["artifact_sha256"]) == set(SCORING_FILES[:-1]), "Incomplete calibration score proof")
        files.add(relative + "/SCORED_SEALED.json")
        files.update(relative + "/" + n for n in SCORING_FILES[:-1])
    directly_sealed = {"SELECTION_SEALED.json", "SUITE_PREPARED.json", "frozen_plan.json", "ALL_CALIBRATION_PREDICTIONS_SEALED.json"}
    directly_sealed.update(relative + "/" + name for relative in expected for name in ("SCORED_SEALED.json", "result.json"))
    for relative in directly_sealed:
        require(completed["artifact_sha256"].get(relative) == sha256(root / relative), "Completed suite proof drift")
    from tools.evolution_confirmation.run import verify_selection_record
    verify_selection_record(root, selection)
    return files


def validate_source_continuity(inventory, selection_root):
    """Freeze every inherited math, data, resource and reconstruction dependency."""
    path = Path(selection_root) / "SUITE_PREPARED.json"
    require(sha256(path) == SELECTION_PREPARED_SHA256, "Selected source inventory pin differs")
    selected = read_json(path)["runtime_hashes"]

    def inherited(name):
        return name.startswith(INHERITED_TREES) or name in INHERITED_FILES

    expected = {k: v for k, v in selected.items() if inherited(k)}
    require(set(INHERITED_FILES) <= set(expected)
            and all(any(k.startswith(tree) for k in expected) for tree in INHERITED_TREES), "Missing selected dependencies")
    actual = {k: v for k, v in inventory.items() if inherited(k)}
    require(actual == expected, "Selected-C inherited source continuity failed")
    return expected


def inventory(suite, folders):
    return old.inventory(suite, folders)


def state(suite, phase, active=None, *, gates=0, predictions=0, scores=0, error=None):
    temporary = suite / "queue_state.tmp"
    write_json(temporary, dict(phase=phase, active_cell=active, passed_gates=gates,
        sealed_predictions=predictions, scored_cells=scores, total_cells=6, total_arm_results=18,
        updated_unix=time.time(), error=error))
    temporary.replace(suite / "queue_state.json")


def copy_checked(source, target):
    before = sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    require(sha256(source) == sha256(target) == before, "Input changed during snapshot")


def prepare(args):
    source, suite = args.source_root.resolve(), args.output.absolute()
    require(suite == suite.resolve() and not suite.exists(), "Fresh canonical output only; no retry/resume")
    require(source == ROOT and old.ROOT == audited.ROOT == ROOT, "Prepare from owned source root")
    require(not any(suite.is_relative_to(source / t) for t in audited.TREES), "Output inside source tree")
    require(sha256(args.plan) == PLAN_SHA256, "Locked full plan bytes changed")
    validate_plan(read_json(args.plan))
    proof_files = selection_proof_files(SELECTION_ROOT)
    validate_source_continuity(source_inventory(source), SELECTION_ROOT)
    parents = {d: read_parent(d) for d in DATASETS}
    require(len({json.dumps(p[0]["checkpoint"], sort_keys=True) for p in parents.values()}) == 1, "Parent checkpoints differ")
    require(args.python.is_file(), "Existing Python binary required")
    suite.mkdir(parents=True)
    for folder in ("logs", "configs", "parents", "manifests", "selection"):
        (suite / folder).mkdir()
    state(suite, "snapshot")
    copy_checked(args.plan, suite / "frozen_plan.json")
    hashes = audited.snapshot(source, suite / "runtime")
    validate_source_continuity(hashes, SELECTION_ROOT)
    for relative in sorted(proof_files):
        copy_checked(SELECTION_ROOT / relative, suite / "selection" / relative)
    selection_proof_files(suite / "selection")
    validate_source_continuity(hashes, suite / "selection")
    for dataset, (protocol, receipt) in parents.items():
        destination = suite / "parents" / dataset
        destination.mkdir()
        copy_checked(Path(receipt["protocol_path"]), destination / "prediction_protocol.json")
        write_json(destination / "parent_receipt.json", receipt)
        write_json(suite / f"configs/{dataset}.json", protocol["config"])
    prepared = dict(format_version=1, worker="image_full_queue", suite_root=str(suite),
        runtime_root=str(suite / "runtime"), runtime_hashes=hashes, plan_sha256=PLAN_SHA256,
        python=str(args.python.absolute()), python_sha256=sha256(args.python), cells=cells(), arms=ARMS.copy(),
        selected_mode="C", selection_sha256=SELECTION_SHA256, exposure_note=EXPOSURE, child_receipts=[])
    for entry in prepared["cells"]:
        state(suite, "prepare", entry["key"])
        prepared["child_receipts"].append(child(suite, hashes, args.python,
            [WORKER, "prepare", "--suite", str(suite), "--dataset", entry["dataset"]],
            "prepare-" + entry["dataset"], cpu=True))
        entry["manifest_sha256"] = sha256(suite / entry["manifest"])
    prepared["artifact_sha256"] = inventory(suite, ("configs", "parents", "selection", "manifests", "logs"))
    prepared["artifact_sha256"]["frozen_plan.json"] = sha256(suite / "frozen_plan.json")
    validate_panels(suite, prepared, verify_inputs=True)
    require(source_inventory(suite / "runtime") == hashes and sha256(args.python) == prepared["python_sha256"],
            "Runtime/Python drift during prepare")
    check_hashes(suite, prepared["artifact_sha256"])
    write_json(suite / "SUITE_PREPARED.json", prepared)
    state(suite, "prepared")
    print(json.dumps(dict(suite=str(suite), suite_sha256=sha256(suite / "SUITE_PREPARED.json"))))


def validate_panels(suite, prepared, *, verify_inputs=False):
    expected = cells()
    require(len(prepared["cells"]) == 6, "Exact six-cell inventory required")
    for entry, locked in zip(prepared["cells"], expected, strict=True):
        require(set(entry) == set(locked) | {"manifest_sha256"}
                and all(entry[k] == v for k, v in locked.items()), "Cell inventory/order/arms differs")
        path = suite / entry["manifest"]
        require(sha256(path) == entry["manifest_sha256"], "Manifest bytes changed")
        manifest = read_json(path)
        dataset = entry["dataset"]
        worker().validate_manifest(manifest)
        parent = read_json(suite / f"parents/{dataset}/prediction_protocol.json")
        validate_parent(parent, dataset)
        receipt = read_json(suite / f"parents/{dataset}/parent_receipt.json")
        protocol_sha, seal_sha = PARENT_HASHES[dataset]
        require(sha256(suite / f"parents/{dataset}/prediction_protocol.json") == protocol_sha
                and receipt == dict(protocol_path=str(parent_directory(dataset) / "prediction_protocol.json"),
                    protocol_sha256=protocol_sha, seal_path=str(parent_directory(dataset) / "PREDICTIONS_SEALED.json"),
                    seal_sha256=seal_sha, complete=True, gt_contents_opened=False, images=POPULATIONS[dataset],
                    sealed_protocol_sha256=protocol_sha), "Pinned parent receipt differs")
        require(manifest["dataset"] == dataset and manifest["suite_root"] == str(suite)
                and manifest["config_path"] == str(suite / entry["config"])
                and manifest["config_sha256"] == sha256(suite / entry["config"])
                and manifest["config"] == parent["config"] == read_json(suite / entry["config"])
                and manifest["checkpoint"] == parent["checkpoint"], "Inherited config/checkpoint/root differs")
        require(manifest["parent_protocol_sha256"] == protocol_sha
                and manifest["parent_seal_sha256"] == seal_sha
                and manifest["source_sha256"] == {str(suite / "runtime" / k): v for k, v in prepared["runtime_hashes"].items()},
                "Parent/source binding differs")
        require(manifest["panel"]["inputs"] == parent["inputs"], "Full membership/order changed")
        for key in ("classes", "queries", "query_to_class", "reduce_zero_label"):
            require(manifest["panel"][key] == parent[key], "Inherited vocabulary/label convention changed")
        if verify_inputs:
            worker().verify_inputs(manifest)


def verify_prepared(suite, expected_hash, *, frozen=False, verify_inputs=False):
    suite = Path(suite).absolute()
    require(suite == suite.resolve(), "Suite aliases/relocation rejected")
    require(sha256(suite / "SUITE_PREPARED.json") == expected_hash, "Reviewed prepared-suite SHA differs")
    prepared = read_json(suite / "SUITE_PREPARED.json")
    require(prepared["format_version"] == 1 and prepared["worker"] == "image_full_queue"
            and prepared["suite_root"] == str(suite) and prepared["runtime_root"] == str(suite / "runtime")
            and prepared["arms"] == ARMS and prepared["selected_mode"] == "C"
            and prepared["selection_sha256"] == SELECTION_SHA256, "Wrong prepared suite/root/selection")
    if frozen:
        require(ROOT == suite / "runtime" and Path.cwd().resolve() == ROOT, "Run only from frozen runtime cwd")
    require(source_inventory(suite / "runtime") == prepared["runtime_hashes"], "Frozen source drift")
    check_hashes(suite, prepared["artifact_sha256"])
    expected_inventory = inventory(suite, ("configs", "parents", "selection", "manifests"))
    require(all(prepared["artifact_sha256"].get(k) == v for k, v in expected_inventory.items()), "Unfrozen input file")
    require(sha256(prepared["python"]) == prepared["python_sha256"], "Python drift")
    require(sha256(suite / "frozen_plan.json") == prepared["plan_sha256"] == PLAN_SHA256, "Plan drift")
    validate_plan(read_json(suite / "frozen_plan.json"))
    selection_proof_files(suite / "selection")
    validate_source_continuity(prepared["runtime_hashes"], suite / "selection")
    validate_panels(suite, prepared, verify_inputs=verify_inputs)
    return prepared


def validate_gate(suite, entry):
    path = suite / f"gates/{entry['dataset']}.json"
    report, marker = read_json(path), path.with_name(path.name + ".STARTED.json")
    require(report["passed"] is True and report["images"] == 4 and report["gt_contents_opened"] is False
            and report["dataset"] == entry["dataset"] and report["manifest_sha256"] == entry["manifest_sha256"]
            and report["attempt_sha256"] == sha256(marker)
            and read_json(marker) == dict(status="started", manifest_sha256=entry["manifest_sha256"]), "GT-free gate/STARTED binding failed")
    cfg = read_json(suite / entry["config"])
    manifest = read_json(suite / entry["manifest"])
    require(report["suite_prepared_sha256"] == sha256(suite / "SUITE_PREPARED.json")
            and report["source_sha256"] == manifest["source_sha256"] and report["checkpoint"] == manifest["checkpoint"]
            and report["ordered_ids"] == manifest["panel"]["ordered_ids"][:4], "Gate source/input/prepared linkage differs")
    require(report["configured_batch_size"] == cfg["batch_size"]
            and report["batch_images"] == [min(cfg["batch_size"], 4 - i) for i in range(0, 4, cfg["batch_size"])],
            "Gate batching differs")
    require(len(report["comparisons"]) == len(report["batch_images"]), "Missing gate episodes")
    require(sum(c["candidates"] for c in report["comparisons"]) > 0, "Vacuous gate")
    for comparison, images in zip(report["comparisons"], report["batch_images"], strict=True):
        require(comparison["candidate_masks_equal"] is True and comparison["reset_equal"] is True
                and comparison["changed_label_pixels"] == 0
                and all(type(comparison[k]) in (float, int) and math.isfinite(comparison[k])
                        and 0 <= comparison[k] <= 1e-5 for k in (
                            "max_support_abs_difference", "max_coefficient_abs_difference", "max_loss_abs_difference", "max_gradient_abs_difference"))
                and set(comparison["actual_arm_counts"]) == set(ARMS), "Gate numerical identity failed")
        for arm in ARMS:
            require(comparison["actual_arm_counts"][arm]["images"] == images
                    and comparison["actual_arm_counts"][arm]["candidate_count"] == comparison["candidates"], "Gate count scope differs")
            worker().validate_cost(comparison["actual_arm_counts"][arm], arm, cfg)
    return sha256(path)


def validate_prediction(suite, entry):
    output = suite / entry["output"]
    protocol = worker().verify_prediction_seal(output)
    require(protocol["manifest"] == read_json(suite / entry["manifest"])
            and protocol["manifest_sha256"] == entry["manifest_sha256"]
            and protocol["manifest_path"] == str(suite / entry["manifest"])
            and protocol["suite_prepared_sha256"] == sha256(suite / "SUITE_PREPARED.json"), "Prediction manifest/root binding differs")
    worker().validate_cost_stream(output, protocol["manifest"])
    return sha256(output / "PREDICTIONS_SEALED.json")


def validate_score(suite, entry):
    output = suite / entry["output"]
    prediction_sha = validate_prediction(suite, entry)
    manifest = read_json(suite / entry["manifest"])
    seal = read_json(output / "SCORED_SEALED.json")
    require(seal["prediction_seal_sha256"] == prediction_sha
            and set(seal["artifact_sha256"]) == set(SCORING_FILES[:-1]), "Score seal inventory/binding differs")
    check_hashes(output, seal["artifact_sha256"])
    barrier_sha = sha256(suite / BARRIER)
    started = read_json(output / "SCORING_STARTED.json")
    require(started == dict(prediction_seal_sha256=prediction_sha, barrier_sha256=barrier_sha,
                           suite_prepared_sha256=sha256(suite / "SUITE_PREPARED.json")), "Score-start binding differs")
    result, predicted = read_json(output / "result.json"), read_json(output / "prediction_result.json")
    require(result["status"] == "completed" and result["dataset"] == entry["dataset"]
            and result["stage"] == "full" and result["seed"] == 0 and result["full_split"] is True
            and result["images"] == entry["images"] and result["prediction_seal_sha256"] == prediction_sha
            and result["barrier_sha256"] == barrier_sha and result["selection_sha256"] == SELECTION_SHA256
            and result["label_contents_first_opened_after_all_full_predictions_sealed"] is True
            and set(result["arm_results"]) == set(result["timings"]) == set(result["costs"]) == set(ARMS)
            and all(result[k] == predicted[k] for k in ("timings", "costs", "timing_scope")), "Scored scope/cost differs")
    worker().validate_score_stream(output, manifest, result)
    for arm in ARMS:
        pooled_miou(result["arm_results"][arm])
    return result


def no_scoring(suite):
    old.no_scoring(suite, cells())


def run(args):
    suite = args.suite.absolute()
    prepared = verify_prepared(suite, args.suite_sha256, frozen=True, verify_inputs=True)
    gates = predictions = scores = 0
    with audited.queue_lock(suite):
        require(not any((suite / p).exists() for p in ("EXECUTION_STARTED.json", "SUITE_SEALED.json", "cells", "gates", BARRIER)),
                "Fresh execution only; no retry/resume")
        write_json(suite / "EXECUTION_STARTED.json", dict(suite_prepared_sha256=args.suite_sha256, pid=os.getpid()))
        receipts, protected, prediction_hashes, results = [], {}, {}, []

        def progress(phase, active=None, error=None):
            state(suite, phase, active, gates=gates, predictions=predictions, scores=scores, error=error)

        def launch(arguments, label, cpu=False):
            verify_prepared(suite, args.suite_sha256, frozen=True)
            check_hashes(suite, protected)
            receipts.append(child(suite, prepared["runtime_hashes"], Path(prepared["python"]), arguments, label, cpu=cpu))

        try:
            (suite / "gates").mkdir()
            for entry in prepared["cells"]:
                progress("gate", entry["key"])
                no_scoring(suite)
                launch([WORKER, "gate", "--suite", str(suite), "--suite-sha256", args.suite_sha256,
                        "--dataset", entry["dataset"]], "gate-" + entry["dataset"])
                protected[f"gates/{entry['dataset']}.json"] = validate_gate(suite, entry)
                marker = f"gates/{entry['dataset']}.json.STARTED.json"
                protected[marker] = sha256(suite / marker)
                gates += 1
            for entry in prepared["cells"]:
                progress("predict", entry["key"])
                no_scoring(suite)
                launch([WORKER, "predict", "--suite", str(suite), "--suite-sha256", args.suite_sha256,
                        "--dataset", entry["dataset"]], "predict-" + entry["dataset"])
                prediction_hashes[entry["output"]] = validate_prediction(suite, entry)
                no_scoring(suite)
                predictions += 1
            current = {e["output"]: validate_prediction(suite, e) for e in prepared["cells"]}
            require(current == prediction_hashes, "Predictions changed before global barrier")
            no_scoring(suite)
            write_json(suite / BARRIER, dict(complete=True, stage="full", gt_contents_opened=False,
                suite_prepared_sha256=args.suite_sha256, prediction_seal_sha256=current))
            protected[BARRIER] = sha256(suite / BARRIER)
            for entry in prepared["cells"]:
                progress("score", entry["key"])
                launch([WORKER, "score", "--suite", str(suite), "--suite-sha256", args.suite_sha256,
                        "--dataset", entry["dataset"]], "score-" + entry["dataset"], cpu=True)
                results.append((entry, validate_score(suite, entry)))
                for name in SCORING_FILES:
                    relative = entry["output"] + "/" + name
                    protected[relative] = sha256(suite / relative)
                scores += 1
            verify_prepared(suite, args.suite_sha256, frozen=True)
            check_hashes(suite, protected)
            require({e["output"]: validate_prediction(suite, e) for e in prepared["cells"]} == current,
                    "Final prediction seals differ")
            for entry in prepared["cells"]:
                validate_score(suite, entry)
            with (suite / "rows.csv").open("x", newline="") as stream:
                columns = ("dataset", "seed", "arm", "images", "full_split", "miou_unrounded", "margin_vs_image_confidence_pp",
                           "diagnostic_encoder_inputs", "measured_images", "seconds", "seconds_per_image", "images_per_second",
                           "manifest_sha256", "selection_sha256", "prediction_seal_sha256", "score_seal_sha256", "exposure")
                table = csv.DictWriter(stream, fieldnames=columns)
                table.writeheader()
                for entry, result in results:
                    for arm in ARMS:
                        table.writerow(dict(dataset=entry["dataset"], seed=0, arm=arm, images=entry["images"], full_split=True,
                            miou_unrounded=pooled_miou(result["arm_results"][arm]),
                            margin_vs_image_confidence_pp=pooled_miou(result["arm_results"][arm]) - pooled_miou(result["arm_results"]["image_confidence"]),
                            diagnostic_encoder_inputs=result["costs"][arm]["diagnostic_encoder_inputs"],
                            **{k: result["timings"][arm][k] for k in ("measured_images", "seconds", "seconds_per_image", "images_per_second")},
                            manifest_sha256=entry["manifest_sha256"], selection_sha256=SELECTION_SHA256,
                            prediction_seal_sha256=current[entry["output"]],
                            score_seal_sha256=sha256(suite / entry["output"] / "SCORED_SEALED.json"), exposure=EXPOSURE))
            write_json(suite / "execution_children.json", receipts)
            progress("completed")
            artifacts = {**prepared["artifact_sha256"], **protected, **inventory(suite, ("logs",))}
            artifacts.update({p + "/PREDICTIONS_SEALED.json": h for p, h in current.items()})
            artifacts.update({n: sha256(suite / n) for n in ("SUITE_PREPARED.json", "EXECUTION_STARTED.json", "execution_children.json", "queue_state.json", "rows.csv")})
            check_hashes(suite, artifacts)
            write_json(suite / "SUITE_SEALED.json", dict(complete=True, full_split=True, formal_cells=6, arm_results=18,
                passed_gt_free_gates=6, selected_mode="C", selection_sha256=SELECTION_SHA256,
                suite_prepared_sha256=args.suite_sha256, runtime_hashes=prepared["runtime_hashes"], artifact_sha256=artifacts,
                exposure_note=EXPOSURE, service_success_verified=False))
        except BaseException as error:
            progress("failed", error=f"{type(error).__name__}: {error}"[:2000])
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="phase", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-root", type=Path, default=ROOT)
    prep.add_argument("--plan", type=Path, default=ROOT / "docs/image-full-plan.json")
    prep.add_argument("--python", type=Path, default=audited.PYTHON)
    prep.add_argument("--output", type=Path, required=True)
    for phase in ("verify", "run"):
        command = sub.add_parser(phase)
        command.add_argument("--suite", type=Path, required=True)
        command.add_argument("--suite-sha256", required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.phase == "prepare":
        fresh = not args.output.exists()
        try:
            prepare(args)
        except BaseException as error:
            if fresh and (args.output / "queue_state.json").exists():
                state(args.output, "failed", error=f"{type(error).__name__}: {error}"[:2000])
            raise
    elif args.phase == "verify":
        verify_prepared(args.suite, args.suite_sha256, verify_inputs=True)
        print(json.dumps(dict(verified=True, suite_sha256=args.suite_sha256)))
    else:
        run(args)


if __name__ == "__main__":
    main()
