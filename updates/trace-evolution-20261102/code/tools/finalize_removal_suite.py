"""Refresh the legacy Full control after the frozen GPU queue, without interrupting it."""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from benchmark_simplification import execution_environment, source_hashes, write_json
from run_removal_suite import (TOOL_FILES, completed_attempt, digest, next_attempt,
                              save_tables, utc_now, validate_artifact)
from suite_inputs import expected_inputs, freeze_protocol
from trace_ablation_variants import VARIANTS


OWNED_OUTPUT = None


def control_attempt(output):
    root = output / "fresh_full_coco_cd"
    root.mkdir(exist_ok=True)
    completed = completed_attempt(root)
    if completed is not None:
        return completed, False
    directory = next_attempt(root)
    directory.mkdir()
    return directory, True


def active_workers(root):
    workers = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().decode().split("\0")
            if any(arg.endswith("benchmark_simplification.py") for arg in args) and "--output" in args:
                output = Path(args[args.index("--output") + 1]).resolve()
                if output.is_relative_to(root):
                    workers.append(int(process.name))
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    return workers


def main():
    global OWNED_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    suite = args.suite.resolve()
    output = suite / "final_audit"
    output.mkdir(exist_ok=True)
    audit_lock = (output / "audit.lock").open("a")
    fcntl.flock(audit_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OWNED_OUTPUT = output
    parent_protocol = suite / "suite_protocol.json"
    parent = json.loads(parent_protocol.read_text())
    runtime = output / "runtime"
    tools_root = Path(__file__).resolve().parent
    files = (*TOOL_FILES, "finalize_removal_suite.py")
    hashes = {name: digest(tools_root / name) for name in files}
    environment = execution_environment()
    if source_hashes(args.package_root / "trace_ovss") != parent["source_hashes"]:
        raise RuntimeError("Final control must use the unchanged production algorithm")
    request = dict(parent_protocol_sha256=digest(parent_protocol), tools=hashes,
        environment=environment, source_hashes=parent["source_hashes"],
        purpose="Refresh Full CD once; no method, configuration, cohort or outcome-based selection change.")
    freeze_protocol(output / "requested_protocol.json", request)
    if not runtime.exists():
        temporary = output / "runtime.preparing"
        shutil.copytree(args.package_root / "trace_ovss", temporary / "trace_ovss",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (temporary / "tools").mkdir()
        for name in files:
            shutil.copy2(tools_root / name, temporary / "tools" / name)
        temporary.rename(runtime)
    if parent["source_hashes"] != source_hashes(runtime / "trace_ovss") or hashes != {
            name: digest(runtime / "tools" / name) for name in files}:
        raise RuntimeError("Final-audit snapshot changed")
    panels = {cell["panel"]: (cell["dataset"], "evaluation", cell["limit"]) for cell in parent["cells"]}
    inputs = expected_inputs(runtime, Path(parent["data_root"]), panels, data_dirs=parent["data_dirs"])
    if inputs != parent["input_panels"]:
        raise RuntimeError("Input file identity changed since the pre-run suite seal")
    freeze_protocol(output / "input_protocol.json", dict(input_panels=inputs, environment=environment))
    if args.prepare_only:
        write_json(output / "queue_state.json", dict(status="prepared"))
        return
    write_json(output / "queue_state.json", dict(status="waiting_for_primary_queue", pid=os.getpid()))
    # No GPU access before acquiring the original queue's lock. Also check orphan
    # workers because the first coordinator version did not pass its lock fd.
    queue_lock = (suite / "queue.lock").open("a")
    fcntl.flock(queue_lock, fcntl.LOCK_EX)
    while active_workers(suite):
        time.sleep(5)
    state = json.loads((suite / "queue_state.json").read_text())
    if state["status"] != "completed":
        write_json(output / "queue_state.json", dict(status="failed_dependency", parent_status=state))
        raise RuntimeError("Primary suite did not complete; no automatic retry")
    if inputs != expected_inputs(runtime, Path(parent["data_root"]), panels, data_dirs=parent["data_dirs"]):
        raise RuntimeError("Input file identity changed while waiting")
    directory, needs_run = control_attempt(output)
    experiment_hashes = {name: hashes[name] for name in
                         ("benchmark_simplification.py", "trace_ablation_variants.py")}
    if needs_run:
        command = [sys.executable, str(runtime / "tools/benchmark_simplification.py"),
            "--package-root", str(runtime), "--data-root", parent["data_root"],
            "--dataset", "coco_cd", "--variant", "full", "--name", "audited_full_reference",
            "--output", str(directory)]
        write_json(output / "queue_state.json", dict(status="running_fresh_full_reference", command=command))
        with (directory / "stdout.log").open("a") as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                pass_fds=(queue_lock.fileno(), audit_lock.fileno()),
                env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        if process.returncode:
            write_json(output / "queue_state.json", dict(status="failed", returncode=process.returncode))
            raise RuntimeError("Fresh Full reference failed")
    fresh = validate_artifact(directory, inputs["coco_cd/None"], parent["source_hashes"],
                              "full", experiment_hashes, environment)
    rows, distributions = [], []
    parent_hashes = {name: parent["tool_hashes"][name] for name in experiment_hashes}
    for cell in parent["cells"]:
        if cell["stage"] == "smoke":
            continue
        replacement = cell["dataset"] == "coco_cd" and cell["variant"] == "full"
        if replacement:
            artifact, result = directory, fresh
            extended_environment = True
        else:
            selected = json.loads((suite / cell["key"] / "selected_result.json").read_text())
            artifact = Path(selected["result_path"]).parent
            if selected["result_sha256"] != digest(artifact / "result.json"):
                raise RuntimeError("Selected primary evidence changed")
            result = validate_artifact(artifact, inputs[cell["panel"]], parent["source_hashes"],
                cell["variant"], parent_hashes, companion_protocol=parent_protocol)
            saved = json.loads((artifact / "protocol.json").read_text())
            if saved["environment"] != {key: environment[key] for key in ("python", "torch", "cuda", "gpu")}:
                raise RuntimeError("Primary cell has incompatible minimal environment identity")
            extended_environment = saved["environment"] == environment
        rows.append(dict(stage=cell["stage"], dataset=cell["dataset"], variant=cell["variant"],
            label=VARIANTS[cell["variant"]],
            images=result["images"], full_split=result["full_split"], mIoU=result["miou_unrounded"],
            mAcc=result["metrics"]["mAcc"], ms_per_image=result["ms_per_image"],
            peak_allocated_mib=result["peak_allocated_mib"],
            diagnostic_views_per_image=result["diagnostic_views_per_image"],
            extended_environment_identity=extended_environment,
            cost_comparable_to_refreshed_full=extended_environment,
            result_path=str(artifact / "result.json")))
        if cell["stage"] == "ablation":
            with (artifact / "candidate_diagnostics.csv").open() as handle:
                candidates = list(csv.DictReader(handle))
            areas = [int(row["area"]) for row in candidates]
            distributions.append(dict(dataset=cell["dataset"] , variant=cell["variant"],
                candidates_per_image=len(areas) / result["images"],
                selected_area_tokens_per_image=sum(areas) / result["images"],
                median_candidate_area_tokens=float(np.median(areas)) if areas else None))
    save_tables(output, rows)
    with (output / "candidate_policy_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=distributions[0].keys())
        writer.writeheader()
        writer.writerows(distributions)
    if inputs != expected_inputs(runtime, Path(parent["data_root"]), panels, data_dirs=parent["data_dirs"]):
        raise RuntimeError("Input files changed during the final audit")
    write_json(output / "queue_state.json", dict(status="completed", cells=len(rows), finished_at=utc_now(),
        note="Old external Full CD excluded. Extended environment unknown for primary costs; treat these as exploratory profiles."))


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
