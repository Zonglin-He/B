"""Exhaustive VOC21 batch-size calibration after the existing GPU queues."""

import argparse
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from benchmark_simplification import execution_environment, source_hashes, with_batch_size, write_json
from finalize_removal_suite import active_workers
from run_removal_suite import completed_attempt, digest, next_attempt, utc_now, validate_artifact
from suite_inputs import expected_inputs, freeze_protocol


BATCH_SIZES = tuple(range(1, 17))
TIE_TOLERANCE = 0.02
TOOL_FILES = ("benchmark_simplification.py", "trace_ablation_variants.py", "suite_inputs.py",
              "run_removal_suite.py", "finalize_removal_suite.py", "run_voc21_batch_tuning.py",
              "analyze_voc21_tuning.py")
OWNED_OUTPUT = None


def batch_panel(panel, batch_size):
    expected = copy.deepcopy(panel)
    expected["config"] = with_batch_size(expected["config"], batch_size)
    return expected


def choose_batch(rows):
    if sorted(row["batch_size"] for row in rows) != list(BATCH_SIZES):
        raise RuntimeError("Complete all sixteen declared batch-size trials before selection")
    if any(row["status"] not in {"completed", "oom"} for row in rows):
        raise RuntimeError("Unresolved trial failure cannot be silently excluded")
    completed = [row for row in rows if row["status"] == "completed"]
    if not completed:
        raise RuntimeError("No successful calibration trial")
    for row in completed:
        if not (np.isfinite(row["mIoU"]) and np.isfinite(row["ms_per_image"]) and row["ms_per_image"] > 0):
            raise RuntimeError("Invalid calibration score or timing")
    best_score = max(row["mIoU"] for row in completed)
    eligible = [row for row in completed if row["mIoU"] >= best_score - TIE_TOLERANCE]
    return min(eligible, key=lambda row: (row["ms_per_image"], row["batch_size"]))


def validate_disjoint_panels(calibration, evaluation):
    if set(calibration["image_ids"]) & set(evaluation["image_ids"]):
        raise RuntimeError("Training calibration overlaps formal validation")
    if set(calibration["label_ids"]) & set(evaluation["label_ids"]):
        raise RuntimeError("Calibration and validation share annotation files")


def calibration_row(batch_size, directory, result):
    if result is None:
        return dict(batch_size=batch_size, status="oom", mIoU=None, mAcc=None,
                    ms_per_image=None, peak_allocated_mib=None, result_path=str(directory))
    return dict(batch_size=batch_size, status="completed", mIoU=result["miou_unrounded"],
                mAcc=result["metrics"]["mAcc"], ms_per_image=result["ms_per_image"],
                peak_allocated_mib=result["peak_allocated_mib"], result_path=str(directory / "result.json"))


def save_rows(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_cell(output, root, expected, split, locks, manifest, name, queue_output):
    output.mkdir(parents=True, exist_ok=True)
    selected = output / "selected_result.json"
    if selected.exists():
        saved = json.loads(selected.read_text())
        directory = Path(saved["result_path"]).parent
        if saved["result_sha256"] != digest(directory / "result.json"):
            raise RuntimeError("Selected tuning evidence changed")
    elif (output / "oom.json").exists():
        return output, None
    else:
        directory = completed_attempt(output)
        if directory is None:
            directory = next_attempt(output)
            directory.mkdir()
            command = [sys.executable, str(root / "tools/benchmark_simplification.py"),
                "--package-root", str(root), "--data-root", manifest["data_root"],
                "--dataset", "voc21", "--variant", "full", "--split", split,
                "--batch-size", str(expected["config"]["batch_size"]),
                "--seed", "0", "--selection-seed", str(expected["selection_seed"]),
                "--name", name, "--output", str(directory)]
            if split == "calibration":
                command += ["--max-images", str(len(expected["image_ids"]))]
            write_json(queue_output / "queue_state.json",
                       dict(status="running", stage=split, batch_size=expected["config"]["batch_size"],
                            current_output=str(directory), command=command, pid=os.getpid()))
            with (directory / "stdout.log").open("w") as handle:
                process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                    pass_fds=tuple(lock.fileno() for lock in locks),
                    env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
            if process.returncode:
                failure_path = directory / "failure.json"
                failure = json.loads(failure_path.read_text()) if failure_path.exists() else {}
                if split == "calibration" and failure.get("error") == "OutOfMemoryError":
                    write_json(output / "oom.json", dict(attempt=str(directory), failure=failure))
                    return directory, None
                raise RuntimeError(f"Tuning worker failed: {directory}")
    result = validate_artifact(directory, expected, manifest["source_hashes"], "full",
        {name: manifest["tool_hashes"][name] for name in ("benchmark_simplification.py", "trace_ablation_variants.py")},
        manifest["environment"])
    write_json(selected, dict(result_path=str(directory / "result.json"),
                             result_sha256=digest(directory / "result.json")))
    return directory, result


def plot_trials(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10})
    complete = [row for row in rows if row["status"] == "completed"]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    x = [row["batch_size"] for row in complete]
    axes[0].plot(x, [row["mIoU"] for row in complete], "o-", color="#176b64")
    axes[0].set_ylabel("Training-calibration mIoU (%)")
    axes[1].plot(x, [row["ms_per_image"] for row in complete], "s-", color="#984723")
    axes[1].set_ylabel("Time per image (ms)")
    for axis in axes:
        axis.set_xlabel("Adaptation batch size")
        axis.set_xticks(BATCH_SIZES)
        axis.grid(alpha=.2)
    figure.savefig(output / "batch_size_sensitivity.pdf")
    figure.savefig(output / "batch_size_sensitivity.png", dpi=300)
    plt.close(figure)


def main():
    global OWNED_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--parent-suite", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output, parent = args.output.resolve(), args.parent_suite.resolve()
    output.mkdir(parents=True, exist_ok=True)
    own_lock = (output / "queue.lock").open("a")
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OWNED_OUTPUT = output
    full_directory = parent / "full_validation/voc21/full/attempt_001"
    baseline = json.loads((full_directory / "result.json").read_text())
    if baseline["status"] != "completed" or baseline["images"] != 1449 or not baseline["full_split"]:
        raise RuntimeError("Finish the complete VOC21 validation before tuning")
    analysis = output / "analysis"
    if not (analysis / "summary.json").exists() or not (analysis / "report.md").exists():
        raise RuntimeError("Complete and save the class-level diagnosis before preparing tuning")
    json.loads((analysis / "summary.json").read_text())
    tools = Path(__file__).resolve().parent
    package = args.package_root.resolve()
    hashes = source_hashes(package / "trace_ovss")
    parent_manifest = json.loads((parent / "suite_protocol.json").read_text())
    if hashes != parent_manifest["source_hashes"]:
        raise RuntimeError("Production algorithm changed; this is not a parameter-only experiment")
    request = dict(source_hashes=hashes, data_root=str(args.data_root.resolve()),
        tool_hashes={name: digest(tools / name) for name in TOOL_FILES},
        analysis_hashes={p.name: digest(p) for p in sorted(analysis.iterdir()) if p.is_file()},
        baseline_result=str(full_directory / "result.json"), baseline_sha256=digest(full_directory / "result.json"),
        parent_protocol_sha256=digest(parent / "suite_protocol.json"), environment=execution_environment(),
        batch_sizes=list(BATCH_SIZES), tie_tolerance_miou=TIE_TOLERANCE,
        objective="Maximum training-calibration mIoU; prefer lower measured latency within 0.02 points.",
        fixed="K, learning rate, steps, view batch size, interventions, formula, candidate policy, optimizer and prompts",
        parameter_scope="Only adaptation batch_size varies. Validation is not used to choose a trial.")
    freeze_protocol(output / "requested_protocol.json", request)
    root = output / "runtime"
    if not root.exists():
        temporary = output / "runtime.preparing"
        shutil.copytree(package / "trace_ovss", temporary / "trace_ovss",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (temporary / "tools").mkdir()
        for name in TOOL_FILES:
            shutil.copy2(tools / name, temporary / "tools" / name)
        temporary.rename(root)
    if hashes != source_hashes(root / "trace_ovss") or request["tool_hashes"] != {
            name: digest(root / "tools" / name) for name in TOOL_FILES}:
        raise RuntimeError("Frozen tuning snapshot changed")
    calibration = expected_inputs(root, args.data_root,
        {"calibration": ("voc21", "calibration", 512)}, selection_seed=2027)["calibration"]
    evaluation = expected_inputs(root, args.data_root,
        {"evaluation": ("voc21", "evaluation", None)})["evaluation"]
    validate_disjoint_panels(calibration, evaluation)
    if calibration["available_images"] != 1464:
        raise RuntimeError("Expected the complete 1,464-image VOC2012 segmentation training split")
    validate_artifact(full_directory, evaluation, hashes, "full",
                      companion_protocol=parent / "suite_protocol.json")
    manifest = {**request, "calibration": calibration, "evaluation": evaluation}
    freeze_protocol(output / "suite_protocol.json", manifest)
    if args.prepare_only:
        if not (output / "queue_state.json").exists():
            write_json(output / "queue_state.json", dict(status="prepared", trials=16))
        return
    write_json(output / "queue_state.json", dict(status="waiting_for_existing_queues", pid=os.getpid()))
    # The final audit already owns its lock while waiting for the primary queue.
    # Acquire that lock FIRST, or two waiting coordinators could deadlock.
    audit_lock = (parent / "final_audit/audit.lock").open("a")
    fcntl.flock(audit_lock, fcntl.LOCK_EX)
    state = json.loads((parent / "final_audit/queue_state.json").read_text())
    if state["status"] != "completed":
        raise RuntimeError("The existing final audit did not complete; leave the other experiments untouched")
    parent_lock = (parent / "queue.lock").open("a")
    fcntl.flock(parent_lock, fcntl.LOCK_EX)
    if active_workers(parent) or json.loads((parent / "queue_state.json").read_text())["status"] != "completed":
        raise RuntimeError("An earlier GPU worker or unfinished queue remains")
    if request["environment"] != execution_environment():
        raise RuntimeError("Execution environment changed while waiting")
    current_calibration = expected_inputs(root, args.data_root,
        {"calibration": ("voc21", "calibration", 512)}, selection_seed=2027)["calibration"]
    if current_calibration != calibration:
        raise RuntimeError("Calibration inputs changed while waiting")
    rows = []
    for batch_size in BATCH_SIZES:
        directory, result = run_cell(output / "calibration" / f"batch_{batch_size:02d}", root,
            batch_panel(calibration, batch_size), "calibration", (own_lock, audit_lock, parent_lock),
            manifest, f"voc21_calibration_batch_{batch_size:02d}", output)
        rows.append(calibration_row(batch_size, directory, result))
        save_rows(output / "trials.csv", rows)
        print(json.dumps(rows[-1]), flush=True)
    winner = choose_batch(rows)
    selected = batch_panel(evaluation, winner["batch_size"])
    write_json(output / "selection.json", dict(selected_trial=winner,
        highest_calibration_miou=max(row["mIoU"] for row in rows if row["status"] == "completed"),
        validation_used_for_selection=False))
    write_json(output / "selected_config.json", dict(config=selected["config"],
        selection_split="VOC2012 train", selection_seed=2027, calibration_images=512,
        production_defaults_modified=False))
    plot_trials(output, rows)
    reused = False
    if winner["batch_size"] == evaluation["config"]["batch_size"]:
        saved_protocol = json.loads((full_directory / "protocol.json").read_text())
        minimal = {key: request["environment"][key] for key in ("python", "torch", "cuda", "gpu")}
        if saved_protocol["environment"] != minimal:
            raise RuntimeError("Cannot reuse the baseline with a changed minimal execution environment")
        verified = validate_artifact(full_directory, selected, hashes, "full",
                                    companion_protocol=parent / "suite_protocol.json")
        verified_directory, reused = full_directory, True
    else:
        verified_directory, verified = run_cell(output / "selected_full_validation", root,
            selected, "evaluation", (own_lock, audit_lock, parent_lock), manifest, "voc21_calibration_selected_full", output)
    original_iou = 100 * np.asarray(baseline["intersection"]) / np.maximum(baseline["union"], 1)
    selected_iou = 100 * np.asarray(verified["intersection"]) / np.maximum(verified["union"], 1)
    save_rows(output / "full_class_comparison.csv", [dict(class_idx=index,
        baseline_IoU=float(a), selected_IoU=float(b), delta_IoU=float(b-a))
        for index, (a, b) in enumerate(zip(original_iou, selected_iou))])
    if hashes != source_hashes(root / "trace_ovss"):
        raise RuntimeError("Algorithm changed during tuning")
    if request["tool_hashes"] != {name: digest(root / "tools" / name) for name in TOOL_FILES}:
        raise RuntimeError("Frozen experimental tools changed during tuning")
    if calibration != expected_inputs(root, args.data_root,
            {"calibration": ("voc21", "calibration", 512)}, selection_seed=2027)["calibration"]:
        raise RuntimeError("Calibration inputs changed during tuning")
    if evaluation != expected_inputs(root, args.data_root,
            {"evaluation": ("voc21", "evaluation", None)})["evaluation"]:
        raise RuntimeError("Validation inputs changed during tuning")
    write_json(output / "verification.json", dict(batch_size=winner["batch_size"],
        mIoU=verified["miou_unrounded"], baseline_mIoU=baseline["miou_unrounded"],
        delta_mIoU=verified["miou_unrounded"] - baseline["miou_unrounded"],
        full_validation_images=verified["images"], reused_existing_full=reused,
        result_path=str(verified_directory / "result.json"), production_defaults_modified=False))
    write_json(output / "queue_state.json", dict(status="completed", trials=len(rows),
        selected_batch_size=winner["batch_size"], full_mIoU=verified["miou_unrounded"], finished_at=utc_now()))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if OWNED_OUTPUT is not None:
            path = OWNED_OUTPUT / "queue_state.json"
            state = json.loads(path.read_text()) if path.exists() else {}
            write_json(path, {**state, "status": "failed", "error": type(error).__name__, "message": str(error)})
        raise
