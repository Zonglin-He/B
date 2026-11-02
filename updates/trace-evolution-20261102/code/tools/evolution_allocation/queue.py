"""Prepare and execute one immutable, no-search ten-outcome ablation suite."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType

from tools.evolution_allocation.common import (
    ALGORITHM, ARMS, DATASETS, EXPOSURE, PARENT, PARENT_SHA, PERMUTATION_SEED,
    POPULATIONS, RESOURCES, ROOT, canonical_hash, expected_manifest, manifest_path, read_json,
    require, sha256, verify_inventory, verify_prepared, writable_path, write_json,
)

PYTHON = Path("/home/wwww/miniconda3/envs/mlmp/bin/python")
WORKER = "tools.evolution_allocation.worker"
PARENT_SERVICE = "trace-waterbirds-full-20261029-0817.service"
OUTPUT_BASE = PARENT.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def state(suite, phase, **fields):
    writable_path(suite, Path(suite) / "queue_state.json")
    temporary = writable_path(suite, Path(suite) / "queue_state.json.tmp")
    require(not temporary.exists(), "Stale state temporary file; no resume")
    write_json(Path(suite) / "queue_state.json", dict(phase=phase, updated_at=now(), pid=os.getpid(), **fields), replace=True)


def service_terminal(name):
    result = subprocess.run(["systemctl", "--user", "show", name, "-p", "LoadState", "-p", "ActiveState",
        "-p", "SubState", "-p", "MainPID", "-p", "Result", "-p", "ExecMainStatus"],
        text=True, capture_output=True, timeout=30, check=True)
    actual = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    require(actual == dict(LoadState="loaded", ActiveState="active", SubState="exited", MainPID="0",
                          Result="success", ExecMainStatus="0"), "Parent not at retained successful terminal")
    return actual


def parent_report(source):
    path = source / "paper_evolution/tools/export_tuned_full.py"
    contents = path.read_bytes()
    require(len(contents) <= 1024 * 1024, "Oversized parent importer")
    before = hashlib.sha256(contents).hexdigest()
    module = ModuleType("allocation_parent_importer")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    # Execute the authenticated source itself; -B alone does not prevent
    # SourceFileLoader from reading an old timestamp-valid .pyc.
    exec(compile(contents, str(path), "exec"), module.__dict__)
    report = module.import_suite(PARENT)
    require(before == report["importer_sha256"] == sha256(path), "Importer changed while executing")
    return report


def validate_output_root(suite, source):
    require(suite.parent == OUTPUT_BASE and suite.name.startswith("trace_allocation_controls_")
            and not suite.is_relative_to(source) and not suite.is_relative_to(PARENT)
            and suite == suite.resolve() and not suite.exists(), "Output must be a new isolated allocation suite")


def expected_command(suite, prepared):
    return [prepared["python"], "-B", "-m", "tools.evolution_allocation.queue", "run",
            "--suite", str(suite), "--suite-sha256", prepared["_verified_sha256"]]


def execution_identity(suite, prepared):
    invocation = os.environ.get("INVOCATION_ID", "")
    require(len(invocation) == 32 and all(c in "0123456789abcdef" for c in invocation),
            "Run requires a real systemd invocation")
    command = Path("/proc/self/cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    require(command == expected_command(suite, prepared), "Actual controller command differs")
    groups = Path("/proc/self/cgroup").read_text().splitlines()
    group = next((line.split(":", 2)[2] for line in groups if line.startswith("0::")), None)
    require(group and group.endswith("/" + prepared["service_name"]), "Controller not inside declared retained service")
    fields = (*RESOURCES, "WorkingDirectory", "InvocationID", "ControlGroup", "ExecMainPID", "ExecStart")
    response = subprocess.run(["systemctl", "--user", "show", prepared["service_name"],
        "--property=" + ",".join(fields)], text=True, capture_output=True, check=True, timeout=30)
    resource_state = dict(line.split("=", 1) for line in response.stdout.splitlines() if "=" in line)
    require(all(resource_state.get(key) == value for key, value in RESOURCES.items())
            and resource_state.get("WorkingDirectory") == str(suite / "runtime")
            and resource_state.get("InvocationID") == invocation
            and resource_state.get("ControlGroup") == group and resource_state.get("ExecMainPID") == str(os.getpid())
            and "argv[]=" + " ".join(command) + " ;" in resource_state.get("ExecStart", ""),
            "Actual resource/root/invocation envelope not installed before GPU admission")
    return dict(started_at=now(), suite_prepared_sha256=prepared["_verified_sha256"], pid=os.getpid(),
        invocation_id=invocation, command=command, cwd=str(Path.cwd().resolve()), control_group=group,
        service=prepared["service_name"], resources=resource_state, retry_allowed=False)


def bound_service_terminal(suite, prepared, execution, service):
    require(service == prepared["service_name"] == execution["service"], "Unrelated retained service")
    fields = ("Id", "LoadState", "ActiveState", "SubState", "MainPID", "Result", "ExecMainCode",
        "ExecMainStatus", "ExecMainPID", "ExecStart", "WorkingDirectory", "ControlGroup", "InvocationID",
        "ExecMainStartTimestamp", "ExecMainExitTimestamp", *RESOURCES)
    response = subprocess.run(["systemctl", "--user", "show", service, "--property=" + ",".join(fields)],
        text=True, capture_output=True, check=True, timeout=30)
    actual = dict(line.split("=", 1) for line in response.stdout.splitlines() if "=" in line)
    require(set(fields) <= set(actual) and actual["Id"] == service and actual["LoadState"] == "loaded"
            and actual["ActiveState"] == "active" and actual["SubState"] == "exited"
            and actual["MainPID"] == "0" and actual["Result"] == "success"
            and actual["ExecMainCode"] == "1" and actual["ExecMainStatus"] == "0"
            and actual["ExecMainStartTimestamp"] and actual["ExecMainExitTimestamp"], "Service not at real successful terminal")
    require(type(execution["pid"]) is int and execution["pid"] > 0
            and actual["ExecMainPID"] == str(execution["pid"])
            and execution["command"] == expected_command(suite, prepared)
            and execution["suite_prepared_sha256"] == prepared["_verified_sha256"]
            and execution["retry_allowed"] is False, "Execution PID/root/hash/command differs")
    command = "argv[]=" + " ".join(expected_command(suite, prepared)) + " ;"
    started = execution["resources"]
    require(all(started.get(key) == value for key, value in RESOURCES.items())
            and started.get("ControlGroup") == execution["control_group"]
            and execution["control_group"].endswith("/" + service)
            and started.get("InvocationID") == execution["invocation_id"]
            and started.get("ExecMainPID") == str(execution["pid"])
            and started.get("WorkingDirectory") == execution["cwd"]
            and command in started.get("ExecStart", ""), "Startup resource/cgroup receipt differs")
    require(command in actual["ExecStart"] and actual["WorkingDirectory"] == str(suite / "runtime") == execution["cwd"]
            and actual["InvocationID"] == execution["invocation_id"]
            and len(execution["invocation_id"]) == 32
            and actual["ControlGroup"] in ("", execution["control_group"]), "Service invocation/source root binding differs")
    require(all(actual[key] == value for key, value in RESOURCES.items()), "Actual execution resource envelope differs")
    return actual


def validate_gate(suite, prepared, dataset):
    from tools.evolution_allocation import worker
    output = suite / "gates" / dataset
    path = manifest_path(suite, dataset, "gate")
    manifest = read_json(path)
    worker.guard_output_labels(output, manifest)
    gate = read_json(output / "gate.json")
    passed = read_json(output / "GATE_PASSED.json")
    require(passed == dict(gate, gate_sha256=sha256(output / "gate.json")), "Gate receipt hash/scope differs")
    require(gate["passed"] is True and gate["images"] == 4 and gate["gt_contents_opened"] is False
            and gate["manifest"] == manifest and gate["manifest_sha256"] == sha256(path)
            and gate["manifest_path"] == str(path) and gate["source_sha256"] == prepared["runtime_hashes"]
            and gate["suite_prepared_sha256"] == prepared["_verified_sha256"]
            and gate["output"] == str(output) and gate["checkpoint"] == prepared["checkpoint"]
            and gate["config"] == manifest["config"]
            and gate["ordered_ids"] == manifest["panel"]["ordered_ids"] and gate["arms"] == ARMS,
            "Gate not bound to this prepared real-model cell")
    claim = output.with_name(output.name + ".STARTED.json")
    attempt = read_json(output / "STARTED.json")
    require(gate["attempt_sha256"] == attempt["attempt_sha256"] == sha256(claim)
            and attempt["phase"] == "gate" and attempt["manifest_sha256"] == sha256(path)
            and attempt["retry_allowed"] is False and attempt["gt_contents_opened"] is False
            and read_json(claim)["manifest_sha256"] == sha256(path), "Gate persistent attempt binding differs")
    require(gate["comparisons"] and sum(gate["batch_images"]) == 4
            and len(gate["comparisons"]) == len(gate["batch_images"]), "Gate population/comparisons incomplete")
    for comparison, images in zip(gate["comparisons"], gate["batch_images"], strict=True):
        require(comparison["candidates"] > 0 and comparison["candidate_masks_equal"] is True
                and comparison["reset_equal"] is True and comparison["changed_label_pixels"] == 0,
                "Vacuous or non-equivalent native gate")
        for key in ("max_support_abs_difference", "max_coefficient_abs_difference", "max_loss_abs_difference",
                    "max_gradient_abs_difference", "max_parameter_abs_difference"):
            value = comparison[key]
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1e-5, "Native equivalence threshold failed")
        actual = comparison["actual_arm_counts"]
        require(set(actual) == {"direct_C", *ARMS}
                and len({row["candidate_sha256"] for row in actual.values()}) == 1,
                "Missing native/factorial/shuffle finite gate or original diagnostic mismatch")
        for arm, row in actual.items():
            worker.validate_cost(row, "image_region" if arm == "direct_C" else arm, manifest["config"])
            require(row["images"] == images and row["active_candidates"] > 0
                    and len(row["losses"]) == manifest["config"]["steps"]
                    and all(math.isfinite(x) for x in row["losses"]), "Missing real finite adaptation updates")
    return passed


def prepare(args):
    from tools.evolution_followup.queue import snapshot
    source, suite = args.source_root.resolve(strict=True), args.output.absolute()
    require(suite == suite.resolve() and not suite.exists() and source == ROOT, "New canonical suite/source required")
    validate_output_root(suite, source)
    require(sha256(PARENT / "SUITE_PREPARED.json") == PARENT_SHA, "Wrong tuned parent")
    terminal = service_terminal(PARENT_SERVICE)
    report = parent_report(source)  # Bounded metadata/aggregate validation, no model or GT reads.
    require(len(report["cells"]) == 8 and report["arm_results"] == 32, "Eight verified parent datasets required")
    parent_prepared = read_json(PARENT / "SUITE_PREPARED.json")
    suite.mkdir(parents=True)
    state(suite, "preparing")
    inventory = snapshot(source, suite / "runtime")
    require(all(inventory.get(key) == value for key, value in parent_prepared["runtime_hashes"].items()),
            "Inherited method/worker source changed; only additive ablation modules allowed")
    shutil.copy2(source / "docs/allocation-controls-contract.md", suite / "frozen_contract.md")
    parent_files, entries = {}, {}
    for name in ("SUITE_PREPARED.json", "SUITE_SEALED.json", "ALL_CONFIGS_SELECTED.json", "ALL_FULL_PREDICTIONS_SEALED.json"):
        target = suite / "parents" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(PARENT / name, target)
        require(sha256(target) == report["metadata_sha256"].get(str(PARENT / name)), "Global parent copy differs from verified read-set")
        parent_files[str(target.relative_to(suite))] = sha256(target)
    for dataset in DATASETS:
        directory = suite / "parents" / dataset
        directory.mkdir()
        sources = {"prediction_protocol.json": PARENT / "full" / dataset / "prediction_protocol.json",
            "PREDICTIONS_SEALED.json": PARENT / "full" / dataset / "PREDICTIONS_SEALED.json",
            "SCORED_SEALED.json": PARENT / "full" / dataset / "SCORED_SEALED.json",
            "result.json": PARENT / "full" / dataset / "result.json",
            "SELECTED_CONFIG_SEALED.json": PARENT / "selections" / dataset / "SELECTED_CONFIG_SEALED.json"}
        for name, original in sources.items():
            target = directory / name
            shutil.copy2(original, target)
            require(sha256(target) == sha256(original) == report["metadata_sha256"].get(str(original)),
                    "Parent artifact changed since verified read-set")
            parent_files[str(target.relative_to(suite))] = sha256(target)
        protocol = read_json(directory / "prediction_protocol.json")
        original_manifest = protocol["manifest"]
        panel = original_manifest["panel"]
        selected = read_json(directory / "SELECTED_CONFIG_SEALED.json")
        require(original_manifest["dataset"] == dataset and original_manifest["stage"] == "full"
                and original_manifest["seed"] == 0 and len(panel["inputs"]) == POPULATIONS[dataset]
                and selected["config"] == original_manifest["config"]
                and selected["full_evaluation_used_for_selection"] is False, "Parent full/config scope mismatch")
        require(read_json(directory / "PREDICTIONS_SEALED.json")["artifact_sha256"]["prediction_protocol.json"]
                == sha256(directory / "prediction_protocol.json"), "Parent protocol not sealed")
        write_json(directory / "panel.json", panel)
        parent_files[str((directory / "panel.json").relative_to(suite))] = sha256(directory / "panel.json")
        entries[dataset] = dict(panel_path=f"parents/{dataset}/panel.json", panel_sha256=sha256(directory / "panel.json"),
                               config=selected["config"])
    write_json(suite / "PARENT_IMPORT.json", dict(parent_suite=str(PARENT), parent_prepared_sha256=PARENT_SHA,
        service_status=terminal, verified_parent_populations=8, verified_parent_full_outcomes=32,
        parent_metadata_sha256=report["metadata_sha256"], source_continuity="Every parent runtime file byte-identical; new files additive",
        source_importer_path=str(source / "paper_evolution/tools/export_tuned_full.py"),
        source_importer_sha256=report["importer_sha256"],
        payload_scope="Parent frozen controller certifies original full payloads; preparation rechecks metadata and pooled counts only"))
    prepared = dict(schema_version=1, algorithm=ALGORITHM, datasets=list(DATASETS), arms=ARMS,
        permutation_seed=PERMUTATION_SEED, model_seed=0, parameter_search=False, retry=False,
        exposure=EXPOSURE, parent_suite=str(PARENT), parent_prepared_sha256=PARENT_SHA,
        parent_proof_sha256=sha256(suite / "PARENT_IMPORT.json"), parent_artifact_sha256=parent_files,
        runtime_hashes=inventory, contract_sha256=sha256(suite / "frozen_contract.md"),
        checkpoint=parent_prepared["checkpoint"], entries=entries, python=str(args.python.resolve(strict=True)),
        service_name=args.service, resources=RESOURCES,
        created_at=now(), scoring_boundary="Both five-arm full predictions sealed before any new GT score")
    write_json(suite / "SUITE_PREPARED.json", prepared)
    identity = sha256(suite / "SUITE_PREPARED.json")
    (suite / "manifests").mkdir()
    for dataset in DATASETS:
        for stage in ("gate", "full"):
            write_json(manifest_path(suite, dataset, stage), expected_manifest(suite, prepared, dataset, stage, identity))
    verify_prepared(suite, identity)
    state(suite, "prepared", prepared_sha256=identity, cells=4, full_arm_results=10, gpu_work_started=False)
    print(json.dumps(dict(suite=str(suite), prepared_sha256=identity, cells=4, full_arm_results=10), indent=2))


def child(suite, prepared, phase, dataset, stage, *, barrier=None):
    verify_prepared(suite, prepared["_verified_sha256"], frozen=True)
    path = manifest_path(suite, dataset, stage)
    manifest = read_json(path)
    output = Path(manifest["output"])
    logs = suite / "logs"
    writable_path(suite, logs)
    logs.mkdir(exist_ok=True)
    log, receipt = logs / f"{dataset}.{phase}.log", logs / f"{dataset}.{phase}.json"
    writable_path(suite, log)
    writable_path(suite, receipt)
    require(not log.exists() and not receipt.exists(), "No child retry allowed")
    command = [prepared["python"], "-B", "-m", WORKER, "--phase", phase,
        "--manifest", str(path), "--manifest-sha256", sha256(path), "--output", str(output)]
    if barrier is not None:
        command += ["--barrier", str(barrier)]
    env = dict(os.environ, PYTHONPATH=str(suite / "runtime"), PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if phase in ("score", "verify"):
        env["CUDA_VISIBLE_DEVICES"] = ""
    state(suite, phase, dataset=dataset, stage=stage, active_output=str(output), command=command)
    started = now()
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=suite / "runtime", env=env,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    write_json(receipt, dict(command=command, started_at=started, ended_at=now(), returncode=result.returncode,
        manifest_sha256=sha256(path), log_sha256=sha256(log), output=str(output), phase=phase))
    require(result.returncode == 0, f"Worker failed; no retry: {receipt}")
    verify_prepared(suite, prepared["_verified_sha256"], frozen=True)


def validate_results(suite, prepared):
    from tools.evolution_tuning.common import pooled_miou
    from tools.evolution_allocation import worker
    rows, results = [], {}
    barrier = sha256(suite / "ALL_FULL_PREDICTIONS_SEALED.json")
    for dataset in DATASETS:
        output = suite / "full" / dataset
        manifest_file = manifest_path(suite, dataset, "full")
        manifest = read_json(manifest_file)
        worker.guard_output_labels(output, manifest)
        seal = read_json(output / "SCORED_SEALED.json")
        require(seal["complete"] is True and seal["barrier_sha256"] == barrier
                and seal["prediction_seal_sha256"] == sha256(output / "PREDICTIONS_SEALED.json"), "Score seal binding differs")
        require(set(seal["artifact_sha256"]) == worker.SCORE_FILES - {"SCORED_SEALED.json"}, "Incomplete canonical score inventory")
        worker.verify_prediction_seal(output, manifest, sha256(manifest_file), hash_payload=True)
        worker.validate_cost_stream(output, manifest)
        worker.verify_score_seal(output, manifest, sha256(manifest_file), suite / "ALL_FULL_PREDICTIONS_SEALED.json")
        verify_inventory(output, seal["artifact_sha256"])
        result = read_json(output / "result.json")
        require(result["status"] == "completed" and result["dataset"] == dataset
                and result["images"] == POPULATIONS[dataset] and result["full_split"] is True
                and result["config"] == prepared["entries"][dataset]["config"]
                and set(result["arm_results"]) == set(ARMS)
                and result["manifest_sha256"] == sha256(manifest_path(suite, dataset, "full"))
                and result["barrier_sha256"] == barrier, "Canonical result differs")
        for arm in ARMS:
            counts = result["arm_results"][arm]
            require(len(counts["intersection"]) == len(counts["union"]) == len(manifest["panel"]["classes"]),
                    "Wrong category count dimension")
            metric = pooled_miou(counts["intersection"], counts["union"])
            require(abs(metric - counts["miou_unrounded"]) <= 1e-10 and math.isfinite(metric), "Invalid pooled score")
            require(type(result["timings"][arm]["seconds_per_image"]) in (int, float)
                    and math.isfinite(result["timings"][arm]["seconds_per_image"])
                    and result["timings"][arm]["seconds_per_image"] >= 0, "Invalid measured time")
            rows.append(dict(dataset=dataset, arm=arm, images=result["images"], mIoU=metric,
                ms_per_image=1000 * result["timings"][arm]["seconds_per_image"], full_split=True,
                exposure=EXPOSURE, result_path=str(output / "result.json")))
        results[dataset] = sha256(output / "SCORED_SEALED.json")
    require(len(rows) == 10, "Expected ten full outcomes")
    return rows, results


def run(args):
    suite = args.suite.absolute()
    prepared = verify_prepared(suite, args.suite_sha256, frozen=True)
    with writable_path(suite, suite / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (suite / "EXECUTION_STARTED.json").exists(), "No restart/resume allowed")
        execution = execution_identity(suite, prepared)
        write_json(writable_path(suite, suite / "EXECUTION_STARTED.json"), execution)
        try:
            for dataset in DATASETS:
                child(suite, prepared, "gate", dataset, "gate")
                validate_gate(suite, prepared, dataset)
            write_json(writable_path(suite, suite / "ALL_GATES_PASSED.json"), dict(complete=True, suite_prepared_sha256=args.suite_sha256,
                gates={d: sha256(suite / "gates" / d / "GATE_PASSED.json") for d in DATASETS}))
            for dataset in DATASETS:
                child(suite, prepared, "predict", dataset, "full")
            boundary = suite / "ALL_FULL_PREDICTIONS_SEALED.json"
            writable_path(suite, boundary)
            links = {}
            for dataset in DATASETS:
                output = suite / "full" / dataset
                from tools.evolution_allocation import worker
                path = manifest_path(suite, dataset, "full")
                worker.guard_output_labels(output, read_json(path))
                seal = read_json(output / "PREDICTIONS_SEALED.json")
                require(seal["complete"] is True and seal["gt_contents_opened"] is False
                        and seal["arms"] == ARMS and seal["images"] == POPULATIONS[dataset], "Incomplete full predictions")
                worker.verify_prediction_seal(output, read_json(path), sha256(path), hash_payload=True)
                worker.validate_cost_stream(output, read_json(path))
                links[str(output)] = sha256(output / "PREDICTIONS_SEALED.json")
            write_json(boundary, dict(complete=True, stage="full", gt_contents_opened=False,
                suite_prepared_sha256=args.suite_sha256, prediction_seal_sha256=links))
            for dataset in DATASETS:
                child(suite, prepared, "score", dataset, "full", barrier=boundary)
            for dataset in DATASETS:
                child(suite, prepared, "verify", dataset, "full")
            rows, results = validate_results(suite, prepared)
            with writable_path(suite, suite / "full_metrics.csv").open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            write_json(writable_path(suite, suite / "SUITE_SEALED.json"), dict(complete=True, suite_prepared_sha256=args.suite_sha256,
                datasets=list(DATASETS), arms=ARMS, full_outcomes=10, score_seals=results,
                full_metrics_sha256=sha256(suite / "full_metrics.csv"), barrier_sha256=sha256(boundary),
                execution_started_sha256=sha256(suite / "EXECUTION_STARTED.json"),
                service_exit_certified=False, finished_at=now()))
            state(suite, "completed", full_outcomes=10, score_seals=results)
        except BaseException as error:
            state(suite, "failed", error=f"{type(error).__name__}: {error}", retry_allowed=False)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    setup = sub.add_parser("prepare")
    setup.add_argument("--output", type=Path, required=True)
    setup.add_argument("--source-root", type=Path, default=ROOT)
    setup.add_argument("--python", type=Path, default=PYTHON)
    setup.add_argument("--service", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--suite", type=Path, required=True)
    execute.add_argument("--suite-sha256", required=True)
    args = parser.parse_args(argv)
    return prepare(args) if args.phase == "prepare" else run(args)


if __name__ == "__main__":
    main()
