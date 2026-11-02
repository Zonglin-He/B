"""Frozen single-GPU TRACE validation and Method-aligned removal experiments."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from benchmark_simplification import execution_environment, source_hashes, write_json
from suite_inputs import expected_inputs, freeze_protocol, validate_cell_inputs
from trace_ablation_variants import DEFINITIONS, VARIANTS


TOOL_FILES = ("benchmark_simplification.py", "trace_ablation_variants.py",
              "suite_inputs.py", "run_removal_suite.py")
OWNED_OUTPUT = None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def completed_attempt(cell_root):
    completed = sorted(path for path in cell_root.glob("attempt_*")
                       if (path / "result.json").exists())
    return completed[0] if completed else None


def next_attempt(cell_root):
    numbers = [int(path.name.removeprefix("attempt_")) for path in cell_root.glob("attempt_*")
               if path.name.removeprefix("attempt_").isdigit()]
    return cell_root / f"attempt_{max(numbers, default=0) + 1:03d}"


def experiment_plan():
    cells = []
    for variant in VARIANTS:
        cells.append(dict(stage="smoke", dataset="coco_cd", limit=56, variant=variant))
    # The cheap complete split provides early end-to-end evidence for every removal.
    for variant in VARIANTS:
        cells.append(dict(stage="ablation", dataset="coco_cd", limit=None, variant=variant))
    for dataset in ("voc21", "pc60", "coco_object"):
        for variant in VARIANTS:
            cells.append(dict(stage="ablation", dataset=dataset, limit=512, variant=variant))
    for dataset in ("coco_cd", "voc20", "voc21", "pc60"):
        cells.append(dict(stage="full_validation", dataset=dataset, limit=None, variant="full"))
    for cell in cells:
        cell["key"] = f'{cell["stage"]}/{cell["dataset"]}/{cell["variant"]}'
        cell["panel"] = f'{cell["dataset"]}/{cell["limit"]}'
    return cells


def validate_artifact(directory, expected, hashes, variant, experiment_hashes=None,
                      environment=None, companion_protocol=None):
    result_path, protocol_path = directory / "result.json", directory / "protocol.json"
    result = json.loads(result_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    validate_cell_inputs(protocol, expected)
    if "file_metadata" in protocol and protocol["file_metadata"] != expected.get("file_metadata"):
        raise RuntimeError("Cached input-file identity changed")
    if "file_metadata" not in protocol:
        if companion_protocol is None or not directory.resolve().is_relative_to(companion_protocol.parent.resolve()):
            raise RuntimeError("Cached input-file identity is unavailable or changed")
        companion = json.loads(companion_protocol.read_text())
        if expected not in companion["input_panels"].values() or companion_protocol.stat().st_mtime_ns > protocol_path.stat().st_mtime_ns:
            raise RuntimeError("Companion input seal was not recorded before this cell")
    if environment is not None and protocol.get("environment") != environment:
        raise RuntimeError("Execution environment differs or was not recorded")
    if result["status"] != "completed" or result["protocol_sha256"] != digest(protocol_path):
        raise RuntimeError(f"Incomplete or unsealed cell: {directory}")
    if protocol["config"] != expected["config"]:
        raise RuntimeError("View batching or optimization configuration changed")
    if protocol["source_hashes"] != hashes or protocol.get("variant", "full") != variant:
        raise RuntimeError("Algorithm identity does not match the frozen experiment")
    if experiment_hashes is not None and protocol.get("experiment_hashes") != experiment_hashes:
        raise RuntimeError("Ablation implementation does not match the frozen experiment")
    predictions = json.loads((directory / "per_image_counts.json").read_text())
    if [row["image_id"] for row in predictions] != expected["image_ids"]:
        raise RuntimeError("Per-image records are incomplete or out of order")
    if result["images"] != len(predictions) or result["images"] != len(expected["image_ids"]):
        raise RuntimeError("Result count differs from its frozen population")
    full = len(predictions) == expected["available_images"]
    if result["full_split"] != full or protocol["full_split"] != full:
        raise RuntimeError("Full/subset evidence label is incorrect")
    intersection = np.array([row["intersection"] for row in predictions]).sum(0)
    union = np.array([row["union"] for row in predictions]).sum(0)
    if not (np.array_equal(intersection, result["intersection"]) and np.array_equal(union, result["union"])):
        raise RuntimeError("Dataset metrics disagree with saved per-image counts")
    miou = 100 * np.mean(intersection[union > 0] / union[union > 0])
    if not np.isfinite(miou) or abs(miou - result["miou_unrounded"]) > 1e-9:
        raise RuntimeError("mIoU cannot be reproduced from saved counts")
    return result


def save_tables(output, rows):
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    full_rows = {(row["dataset"], row["images"]): row for row in rows
                 if row["stage"] == "ablation" and row["variant"] == "full"}
    ablations = []
    for row in rows:
        if row["stage"] != "ablation":
            continue
        full = full_rows.get((row["dataset"], row["images"]))
        ablations.append({**row, "delta_vs_full": row["mIoU"] - full["mIoU"] if full else None})
    if ablations:
        with (output / "ablation_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ablations[0].keys())
            writer.writeheader()
            writer.writerows(ablations)


def main():
    global OWNED_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--coco-object-dir", type=Path, required=True)
    parser.add_argument("--reuse-coco-cd", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OWNED_OUTPUT = output
    package_root = args.package_root.resolve()
    data_dirs = dict(coco_object=str(args.coco_object_dir.resolve()))
    runtime = output / "runtime"
    frozen_tools = runtime / "tools"
    requested_hashes = source_hashes(package_root / "trace_ovss")
    tools_root = Path(__file__).resolve().parent
    tool_hashes = {name: digest(tools_root / name) for name in TOOL_FILES}
    request = dict(data_root=str(args.data_root.resolve()), data_dirs=data_dirs,
        source_hashes=requested_hashes, tool_hashes=tool_hashes,
        execution_environment=execution_environment(),
        reuse_coco_cd=str(args.reuse_coco_cd.resolve()) if args.reuse_coco_cd else None)
    freeze_protocol(output / "requested_protocol.json", request)
    if not runtime.exists():
        temporary = output / "runtime.preparing"
        if temporary.exists():
            raise RuntimeError("Incomplete snapshot exists; inspect it before retrying")
        shutil.copytree(package_root / "trace_ovss", temporary / "trace_ovss",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (temporary / "tools").mkdir()
        for name in TOOL_FILES:
            shutil.copy2(tools_root / name, temporary / "tools" / name)
        temporary.rename(runtime)
    if requested_hashes != source_hashes(runtime / "trace_ovss") or tool_hashes != {
            name: digest(frozen_tools / name) for name in TOOL_FILES}:
        raise RuntimeError("Frozen runtime differs from the requested code")
    cells = experiment_plan()
    panels = {cell["panel"]: (cell["dataset"], "evaluation", cell["limit"]) for cell in cells}
    inputs = expected_inputs(runtime, args.data_root, panels, data_dirs=data_dirs)
    manifest = dict(**request, cells=cells, input_panels=inputs,
        seed=0, selection_seed=20261024,
        variants=DEFINITIONS,
        adaptation="Episodic, visual LayerNorm only; inherited dataset configurations; no retuning.",
        scope="Single-seed full validation plus fixed-512 representative ablations (COCO-CD full 500).",
        deferred=["pc59 full", "coco_object full", "coco_stuff full", "cityscapes",
                  "waterbirds_seg", "second_backbone", "ACDC"],
        interpretation="Subset ablations are not full-dataset main-table evidence. All signs retained.")
    freeze_protocol(output / "suite_protocol.json", manifest)
    if args.prepare_only:
        write_json(output / "queue_state.json", dict(status="prepared", cells_total=len(cells)))
        print(f"Prepared {len(cells)} cells with frozen source, configs and image IDs", flush=True)
        return
    frozen_experiment_hashes = {name: tool_hashes[name] for name in
                               ("benchmark_simplification.py", "trace_ablation_variants.py")}
    rows = []
    started = time.perf_counter()
    for cell in cells:
        expected = inputs[cell["panel"]]
        cell_root = output / cell["key"]
        cell_root.mkdir(parents=True, exist_ok=True)
        selection = cell_root / "selected_result.json"
        reused = False
        if selection.exists():
            selected = json.loads(selection.read_text())
            directory = Path(selected["result_path"]).parent
            if selected["result_sha256"] != digest(directory / "result.json"):
                raise RuntimeError("Selected evidence was modified")
            reused = selected["external_reuse"]
        elif args.reuse_coco_cd and cell["dataset"] == "coco_cd" and cell["limit"] is None and cell["variant"] == "full":
            directory = args.reuse_coco_cd.resolve()
            reused = True
        elif cell["stage"] == "full_validation" and cell["dataset"] == "coco_cd":
            selected = json.loads((output / "ablation/coco_cd/full/selected_result.json").read_text())
            directory = Path(selected["result_path"]).parent
            reused = selected["external_reuse"]
        else:
            directory = completed_attempt(cell_root)
            if directory is None:
                directory = next_attempt(cell_root)
                directory.mkdir()
                command = [sys.executable, str(frozen_tools / "benchmark_simplification.py"),
                    "--package-root", str(runtime), "--data-root", str(args.data_root),
                    "--dataset", cell["dataset"], "--variant", cell["variant"],
                    "--name", cell["key"], "--seed", "0", "--selection-seed", "20261024",
                    "--output", str(directory)]
                if cell["limit"]:
                    command += ["--max-images", str(cell["limit"])]
                if cell["dataset"] in data_dirs:
                    command += ["--data-dir", data_dirs[cell["dataset"]]]
                write_json(output / "queue_state.json", dict(status="running", pid=os.getpid(),
                    started_at=utc_now(), completed=len(rows), cells_total=len(cells),
                    current=cell, current_output=str(directory), command=command))
                print(f'Running {len(rows) + 1}/{len(cells)}: {cell["key"]}', flush=True)
                with (directory / "stdout.log").open("w") as handle:
                    # Keep the lock alive in the worker if the coordinator dies.
                    process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                        pass_fds=(lock.fileno(),),
                        env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
                if process.returncode:
                    write_json(output / "queue_state.json", dict(status="failed", completed=len(rows),
                        current=cell, current_output=str(directory), returncode=process.returncode))
                    raise RuntimeError(f"Failed cell; earlier evidence retained: {directory}")
        result = validate_artifact(directory, expected, requested_hashes, cell["variant"],
                                   None if reused else frozen_experiment_hashes,
                                   request["execution_environment"])
        write_json(selection, dict(result_path=str(directory / "result.json"),
            result_sha256=digest(directory / "result.json"), external_reuse=reused))
        rows.append(dict(stage=cell["stage"], dataset=cell["dataset"], variant=cell["variant"],
            label=VARIANTS[cell["variant"]], images=result["images"], full_split=result["full_split"],
            mIoU=result["miou_unrounded"], mAcc=result["metrics"]["mAcc"],
            ms_per_image=result["ms_per_image"], peak_allocated_mib=result["peak_allocated_mib"],
            diagnostic_views_per_image=result["diagnostic_views_per_image"],
            external_reuse=reused, result_path=str(directory / "result.json")))
        save_tables(output, rows)
    if inputs != expected_inputs(runtime, args.data_root, panels, data_dirs=data_dirs):
        raise RuntimeError("Input files changed during the experiment suite")
    if requested_hashes != source_hashes(runtime / "trace_ovss") or tool_hashes != {
            name: digest(frozen_tools / name) for name in TOOL_FILES}:
        raise RuntimeError("Frozen implementation changed during the experiment suite")
    write_json(output / "queue_state.json", dict(status="completed", completed=len(rows),
        cells_total=len(cells), finished_at=utc_now(), wall_seconds=time.perf_counter() - started))
    print(f"Completed {len(rows)} cells; see summary.csv and ablation_summary.csv", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if OWNED_OUTPUT is not None:
            path = OWNED_OUTPUT / "queue_state.json"
            state = json.loads(path.read_text()) if path.exists() else {}
            write_json(path, {**state, "status": "failed", "error": type(error).__name__,
                              "message": str(error)})
        raise
