"""Parameter-only accuracy/cost search, scheduled after VOC21 batch calibration."""

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from benchmark_simplification import execution_environment, source_hashes, with_budget, write_json
from finalize_removal_suite import active_workers
from run_removal_suite import completed_attempt, digest, next_attempt, utc_now, validate_artifact
from run_voc21_batch_tuning import (batch_panel, calibration_row, choose_batch,
                                   save_rows, validate_disjoint_panels)
from suite_inputs import expected_inputs, freeze_protocol


DATASETS = ("voc21", "pc60")
STAGES = ("trace_max_regions", "steps", "lr")
TOLERANCE = 0.02
TOOL_FILES = ("benchmark_simplification.py", "trace_ablation_variants.py", "suite_inputs.py",
              "run_removal_suite.py", "finalize_removal_suite.py", "run_voc21_batch_tuning.py",
              "run_accuracy_budget_search.py")
OWNED_OUTPUT = None


def values_for(stage, initial):
    if stage == "trace_max_regions":
        values = (4, 8, 16, 32, initial[stage])
    elif stage == "steps":
        values = (4, 8, 16, 24, initial[stage])
    elif stage == "lr":
        values = tuple(initial[stage] * factor for factor in (0.5, 1.0, 1.5))
    else:
        raise ValueError(stage)
    return sorted(set(values))


def select_accuracy_first(rows):
    if not rows or any(row["status"] not in {"completed", "oom"} for row in rows):
        raise RuntimeError("Incomplete stage or unresolved failure")
    valid = [row for row in rows if row["status"] == "completed"]
    if not valid or any(not np.isfinite(row["mIoU"]) or not np.isfinite(row["ms_per_image"])
                        or row["ms_per_image"] <= 0 for row in valid):
        raise RuntimeError("No valid accuracy/cost evidence")
    best = max(row["mIoU"] for row in valid)
    return min((row for row in valid if row["mIoU"] >= best - TOLERANCE),
               key=lambda row: (row["ms_per_image"], row["trace_max_regions"], row["steps"], row["lr"]))


def pareto_flags(rows):
    """Keep both signs; do not call a faster but less accurate run dominant."""
    return [not any(b["mIoU"] >= a["mIoU"] and b["ms_per_image"] <= a["ms_per_image"]
                    and (b["mIoU"] > a["mIoU"] or b["ms_per_image"] < a["ms_per_image"])
                    for b in rows) for a in rows]


def configured(panel, config):
    result = copy.deepcopy(panel)
    result["config"] = copy.deepcopy(config)
    return result


def config_key(config):
    import hashlib
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def audit_dependency(dependency, parent, environment):
    """Bind the new search to verified upstream trials, not status summaries."""
    upstream = json.loads((dependency / "suite_protocol.json").read_text())
    experiment = {name: upstream["tool_hashes"][name] for name in
                  ("benchmark_simplification.py", "trace_ablation_variants.py")}
    rows, seals = [], {}
    for batch in range(1, 17):
        selected_path = dependency / "calibration" / f"batch_{batch:02d}" / "selected_result.json"
        if not selected_path.exists():
            raise RuntimeError(f"Missing completed upstream batch {batch}; unresolved OOMs are not inherited")
        saved = json.loads(selected_path.read_text())
        directory = Path(saved["result_path"]).parent
        if saved["result_sha256"] != digest(directory / "result.json"):
            raise RuntimeError("Upstream calibration result changed")
        result = validate_artifact(directory, batch_panel(upstream["calibration"], batch),
            upstream["source_hashes"], "full", experiment, upstream["environment"])
        rows.append(calibration_row(batch, directory, result))
        for name in ("result.json", "protocol.json", "per_image_counts.json"):
            seals[str(directory / name)] = digest(directory / name)
    winner = choose_batch(rows)
    selection = json.loads((dependency / "selection.json").read_text())
    if selection["selected_trial"] != winner or selection["validation_used_for_selection"] is not False:
        raise RuntimeError("Upstream winner cannot be reproduced from all sixteen training trials")
    verification = json.loads((dependency / "verification.json").read_text())
    batch = winner["batch_size"]
    if verification["batch_size"] != batch:
        raise RuntimeError("Upstream selected/verified batch mismatch")
    directory = Path(verification["result_path"]).parent
    expected = batch_panel(upstream["evaluation"], batch)
    if verification["reused_existing_full"]:
        baseline = Path(upstream["baseline_result"])
        if directory / "result.json" != baseline or digest(baseline) != upstream["baseline_sha256"]:
            raise RuntimeError("Reused full validation does not match the sealed baseline")
        original = json.loads((parent / "suite_protocol.json").read_text())
        experiment = {name: original["tool_hashes"][name] for name in experiment}
        minimal = {key: environment[key] for key in ("python", "torch", "cuda", "gpu")}
        full = validate_artifact(directory, expected, upstream["source_hashes"], "full",
            experiment, minimal, companion_protocol=parent / "suite_protocol.json")
    else:
        full_seal = json.loads((dependency / "selected_full_validation/selected_result.json").read_text())
        if (full_seal["result_path"] != verification["result_path"] or
                full_seal["result_sha256"] != digest(directory / "result.json")):
            raise RuntimeError("Upstream selected full-validation seal changed")
        full = validate_artifact(directory, expected, upstream["source_hashes"], "full",
                                 experiment, upstream["environment"])
    if full["images"] != 1449 or not full["full_split"] or abs(full["miou_unrounded"] - verification["mIoU"]) > 1e-9:
        raise RuntimeError("Upstream full verification is incomplete or differs from its artifact")
    for name in ("result.json", "protocol.json", "per_image_counts.json"):
        seals[str(directory / name)] = digest(directory / name)
    return dict(batch_size=batch, verified_full_mIoU=full["miou_unrounded"], evidence_sha256=seals,
                all_sixteen_calibration_trials_validated=True)


def worker_command(root, expected, directory):
    config = expected["config"]
    return [sys.executable, str(root / "tools/benchmark_simplification.py"),
        "--package-root", str(root), "--data-root", str(Path(config["data_dir"]).parent),
        "--data-dir", config["data_dir"], "--dataset", expected["dataset"],
        "--variant", "full", "--split", expected["split"], "--seed", "0",
        "--selection-seed", str(expected["selection_seed"]),
        "--max-images", str(len(expected["image_ids"])), "--name", directory.parent.name,
        "--output", str(directory), "--batch-size", str(config["batch_size"]),
        "--max-regions", str(config["trace_max_regions"]), "--steps", str(config["steps"]),
        "--lr", str(config["lr"]), "--view-batch-size", str(config["trace_view_batch_size"])]


def run_cell(output, runtime, expected, manifest, locks, stage):
    cell = output / expected["split"] / expected["dataset"] / config_key(expected["config"])
    cell.mkdir(parents=True, exist_ok=True)
    freeze_protocol(cell / "expected.json", expected)
    directory = completed_attempt(cell)
    if directory is None and (cell / "oom.json").exists():
        raise RuntimeError(f"Unresolved OOM evidence requires inspection: {cell}")
    if directory is None:
        directory = next_attempt(cell)
        directory.mkdir()
        command = worker_command(runtime, expected, directory)
        write_json(output / "queue_state.json", dict(status="running", pid=os.getpid(),
            dataset=expected["dataset"], stage=stage, current_output=str(directory), command=command))
        with (directory / "stdout.log").open("w") as handle:
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                pass_fds=tuple(lock.fileno() for lock in locks),
                env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        if completed.returncode:
            failure_path = directory / "failure.json"
            failure = json.loads(failure_path.read_text()) if failure_path.exists() else {}
            if expected["split"] == "calibration" and failure.get("error") == "OutOfMemoryError":
                write_json(cell / "oom.json", dict(attempt=str(directory), failure=failure,
                    failure_sha256=digest(failure_path), protocol_sha256=digest(directory / "protocol.json"),
                    status="unresolved", reason="Capacity and temporary GPU contention are not yet distinguished"))
                raise RuntimeError(f"Unresolved OOM; do not silently remove a search candidate: {directory}")
            raise RuntimeError(f"Worker failure; inspect {directory}")
    result = validate_artifact(directory, expected, manifest["source_hashes"], "full",
        {name: manifest["tool_hashes"][name] for name in ("benchmark_simplification.py", "trace_ablation_variants.py")},
        manifest["environment"])
    selection_path = cell / "selected_result.json"
    selection = dict(result_path=str(directory / "result.json"), result_sha256=digest(directory / "result.json"))
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise RuntimeError("Selected evidence identity changed")
    write_json(selection_path, selection)
    return row_from(expected, directory, result)


def row_from(expected, directory, result):
    config = expected["config"]
    return dict(dataset=expected["dataset"], split=expected["split"], images=len(expected["image_ids"]),
        batch_size=config["batch_size"], trace_max_regions=config["trace_max_regions"], steps=config["steps"],
        lr=config["lr"], status="completed" if result else "oom",
        mIoU=result["miou_unrounded"] if result else None,
        ms_per_image=result["ms_per_image"] if result else None,
        diagnostic_views_per_image=result["diagnostic_views_per_image"] if result else None,
        result_path=str(directory / "result.json") if result else str(directory))


def discover_panels(runtime, data_root, data_dirs):
    result = expected_inputs(runtime, data_root,
        {f"{d}/calibration": (d, "calibration", 256) for d in DATASETS}, data_dirs, selection_seed=2027)
    result.update(expected_inputs(runtime, data_root,
        {f"{d}/evaluation": (d, "evaluation", 512) for d in DATASETS}, data_dirs))
    for dataset in DATASETS:
        validate_disjoint_panels(result[f"{dataset}/calibration"], result[f"{dataset}/evaluation"])
    return result


def main():
    global OWNED_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "package-root", "data-root", "batch-tuning", "parent-suite", "coco-object-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    own_lock = (output / "queue.lock").open("a")
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OWNED_OUTPUT = output
    dependency, parent = args.batch_tuning.resolve(), args.parent_suite.resolve()
    tools = Path(__file__).resolve().parent
    request = dict(source_hashes=source_hashes(args.package_root / "trace_ovss"),
        tool_hashes={name: digest(tools / name) for name in TOOL_FILES}, environment=execution_environment(),
        data_root=str(args.data_root.resolve()), data_dirs={"coco_object": str(args.coco_object_dir.resolve())},
        dependency=str(dependency), dependency_protocol_sha256=digest(dependency / "suite_protocol.json"),
        parent_suite=str(parent), datasets=list(DATASETS), calibration_images=256, evaluation_images=512,
        stage_order=list(STAGES), K_values="4, 8, 16, 32, inherited K", steps_values="4, 8, 16, 24, inherited steps",
        lr_factors=[0.5, 1.0, 1.5], tolerance=TOLERANCE,
        selection="Each coordinate uses its training-calibration maximum; final choice uses ALL visited configurations including baseline. Lower runtime only within 0.02 of that global maximum.",
        fixed="Algorithm, candidates policy, three views, reliability, normalization, lr/steps/K except scheduled coordinate; no guards or CLS.",
        scope="Single-seed development search and one paired 512-image verification; not full or three-seed main results.",
        deferred={"coco_object": "The audited converted object-mask training split is absent locally; no validation-label substitution.",
                  "coco_cd": "Only four candidate slots and five steps; retain current completed control while investigating the expensive settings first."})
    inherited = json.loads((dependency / "suite_protocol.json").read_text())
    if request["source_hashes"] != inherited["source_hashes"]:
        raise RuntimeError("Production algorithm differs from the running VOC21 study")
    freeze_protocol(output / "requested_protocol.json", request)
    runtime = output / "runtime"
    if not runtime.exists():
        temporary = output / "runtime.preparing"
        shutil.copytree(args.package_root / "trace_ovss", temporary / "trace_ovss",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (temporary / "tools").mkdir()
        for name in TOOL_FILES:
            shutil.copy2(tools / name, temporary / "tools" / name)
        temporary.rename(runtime)
    check_snapshot(runtime, request)
    inputs = discover_panels(runtime, args.data_root, request["data_dirs"])
    freeze_protocol(output / "input_protocol.json", inputs)
    if args.prepare_only:
        if not (output / "queue_state.json").exists():
            write_json(output / "queue_state.json", dict(status="prepared", dependency=str(dependency)))
        return
    write_json(output / "queue_state.json", dict(status="waiting_for_voc21_batch_search", pid=os.getpid()))
    locks = [own_lock]
    # Dependency -> audit -> primary is the same acquisition order as the tuner.
    for lock_path in (dependency / "queue.lock", parent / "final_audit/audit.lock", parent / "queue.lock"):
        lock = lock_path.open("a")
        fcntl.flock(lock, fcntl.LOCK_EX)
        locks.append(lock)
    for directory in (dependency, parent / "final_audit", parent):
        if json.loads((directory / "queue_state.json").read_text())["status"] != "completed":
            raise RuntimeError(f"Unfinished dependency: {directory}")
    if active_workers(dependency) or active_workers(parent) or active_workers(output):
        raise RuntimeError("An earlier GPU worker remains; do not overlap it")
    if request["environment"] != execution_environment():
        raise RuntimeError("Environment changed while waiting")
    if digest(dependency / "suite_protocol.json") != request["dependency_protocol_sha256"]:
        raise RuntimeError("Dependency protocol changed")
    if inputs != discover_panels(runtime, args.data_root, request["data_dirs"]):
        raise RuntimeError("Inputs changed while waiting")
    dependency_audit = audit_dependency(dependency, parent, request["environment"])
    selected_batch = dependency_audit["batch_size"]
    starting = {d: copy.deepcopy(inputs[f"{d}/calibration"]["config"]) for d in DATASETS}
    starting["voc21"]["batch_size"] = selected_batch
    manifest = {**request, "inputs": inputs, "starting_configs": starting,
                "dependency_audit": dependency_audit,
                "dependency_selection_sha256": digest(dependency / "selection.json"),
                "dependency_verification_sha256": digest(dependency / "verification.json")}
    freeze_protocol(output / "suite_protocol.json", manifest)
    rows, comparisons = [], []
    for dataset in DATASETS:
        initial, current = starting[dataset], copy.deepcopy(starting[dataset])
        calibration = inputs[f"{dataset}/calibration"]
        baseline_row = run_cell(output, runtime, configured(calibration, initial), manifest, locks, "baseline")
        if baseline_row["status"] != "completed":
            raise RuntimeError("Baseline must succeed before optimizing its budget")
        rows.append({**baseline_row, "stage": "baseline"})
        save_rows(output / "trials.csv", rows)
        for stage in STAGES:
            stage_rows = []
            values = values_for(stage, initial)
            for value in values:
                candidate = {**current, stage: value}
                row = run_cell(output, runtime, configured(calibration, candidate), manifest, locks, stage)
                stage_rows.append(row)
                rows.append({**row, "stage": stage})
                save_rows(output / "trials.csv", rows)
            winner = select_accuracy_first(stage_rows)
            current[stage] = winner[stage]
            freeze_protocol(output / f"{dataset}_{stage}_selection.json",
                            dict(values=values, selected=winner, config=current, selection_split="train"))
        # Compare all visited configurations against one global maximum; small
        # per-stage timing ties must not accumulate into an accuracy concession.
        final_winner = select_accuracy_first([row for row in rows if row["dataset"] == dataset])
        current = {**initial, **{key: final_winner[key] for key in STAGES}}
        freeze_protocol(output / f"{dataset}_selected_config.json",
                        dict(config=current, selected_trial=final_winner,
                             production_defaults_modified=False, validation_used_for_selection=False))
        evaluation = inputs[f"{dataset}/evaluation"]
        before = run_cell(output, runtime, configured(evaluation, initial), manifest, locks, "validation_baseline")
        after = run_cell(output, runtime, configured(evaluation, current), manifest, locks, "validation_selected")
        comparison = dict(dataset=dataset, images=512, full_split=False, seed=0,
            baseline_mIoU=before["mIoU"], selected_mIoU=after["mIoU"], delta_mIoU=after["mIoU"]-before["mIoU"],
            baseline_ms_per_image=before["ms_per_image"], selected_ms_per_image=after["ms_per_image"],
            speedup=before["ms_per_image"] / after["ms_per_image"],
            baseline_views=before["diagnostic_views_per_image"], selected_views=after["diagnostic_views_per_image"],
            baseline_result=before["result_path"], selected_result=after["result_path"])
        comparisons.append(comparison)
        save_rows(output / "verification.csv", comparisons)
        valid = list({row["result_path"]: row for row in rows
                      if row["dataset"] == dataset and row["status"] == "completed"}.values())
        save_rows(output / f"{dataset}_pareto.csv", [dict(**row, pareto=flag)
                  for row, flag in zip(valid, pareto_flags(valid))])
    check_snapshot(runtime, request)
    if inputs != discover_panels(runtime, args.data_root, request["data_dirs"]):
        raise RuntimeError("Dataset files changed during search")
    write_json(output / "queue_state.json", dict(status="completed", datasets=list(DATASETS),
               finished_at=utc_now(), production_defaults_modified=False))


def check_snapshot(runtime, request):
    if request["source_hashes"] != source_hashes(runtime / "trace_ovss") or request["tool_hashes"] != {
            name: digest(runtime / "tools" / name) for name in TOOL_FILES}:
        raise RuntimeError("Frozen source or experimental tools changed")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if OWNED_OUTPUT is not None:
            path = OWNED_OUTPUT / "queue_state.json"
            state = json.loads(path.read_text()) if path.exists() else {}
            write_json(path, {**state, "status": "failed", "error": type(error).__name__, "message": str(error)})
        raise
