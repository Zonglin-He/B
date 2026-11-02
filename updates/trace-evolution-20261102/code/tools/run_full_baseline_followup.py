"""Wait for the two-view full queue, then run its matched full baselines serially."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from run_matched_baseline_suite import publish, sha, validate_cell, verify_suite, write_json


CONTROLLER = "run_full_baseline_followup.py"
FILES = (CONTROLLER, "run_matched_baseline_suite.py", "benchmark_matched_baselines.py",
         "analyze_two_view_full.py")
DATASETS = ("pc60", "voc21", "coco_cd", "coco_object")
METHODS = ("mlmp", "tent", "tpt", "noadapt")


def plan():
    return [dict(dataset=d, method=m, panel=f"{d}/full") for d in DATASETS for m in METHODS]


def verify_code(output):
    frozen = json.loads((output / "code_seal.json").read_text())
    if {name: sha(output / "code" / name) for name in FILES} != frozen:
        raise RuntimeError("Frozen baseline followup code changed")


def accepted_hashes(directory):
    return {n: sha(directory / n) for n in ("result.json", "protocol.json", "cost.csv", "per_image_counts.json")}


def check_accepted(directory, frozen):
    if accepted_hashes(directory) != frozen:
        raise RuntimeError(f"Previously accepted baseline evidence changed: {directory}")


def run_analysis(output, suite):
    command = [sys.executable, str(output / "code/analyze_two_view_full.py"),
        "--trace-root", str(suite), "--baseline-root", str(output), "--output", str(output / "analysis")]
    try:
        with (output / "analysis.log").open("w") as log:
            analysis = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        status = dict(returncode=analysis.returncode, status="completed" if analysis.returncode == 0 else "failed", command=command)
    except Exception as error:
        status = dict(status="failed", error=type(error).__name__, message=str(error), command=command)
    try:
        write_json(output / "analysis_status.json", status)
    except Exception as error:
        # Post-analysis failures must never mutate the already sealed experiment.
        print(f"Could not persist post-analysis status: {error}; {status}", flush=True)


def execute(args, own_lock, stack):
    output, suite, baseline = args.output, args.suite_root, args.baseline_root
    verify_code(output)
    write_json(output / "queue_state.json", dict(status="waiting_for_two_view_full",
        dependency=str(suite), pid=os.getpid(), completed=0, total=len(plan())))
    dependency_lock = stack.enter_context((suite / "queue.lock").open("a"))
    fcntl.flock(dependency_lock, fcntl.LOCK_EX)
    verify_suite(suite)
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                     "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError(f"Existing GPU processes will not be interrupted: {active}")
    parent = json.loads((suite / "suite_protocol.json").read_text())
    worker = output / "code/benchmark_matched_baselines.py"
    spec = importlib.util.spec_from_file_location("full_baseline_adapter", worker)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    controls = {suite / "suite_protocol.json", suite / "SUITE_SEALED.json",
                suite / "fixed_configs.json", suite / "input_hashes.json",
                Path.home() / ".cache/clip/ViT-L-14.pt"}
    controls.update(suite / "runtime/tools" / name for name in parent["tool_hashes"])
    for cell in plan():
        _, files = adapter.archived_settings(baseline, cell["dataset"], cell["method"],
                                            parent["input_panels"][cell["panel"]]["config"])
        controls.update(files)
    controls = sorted(controls)
    sources = adapter.source_snapshot(suite / "runtime", baseline, controls, worker)
    parent_inputs = json.loads((suite / "input_hashes.json").read_text())
    write_json(output / "manifest.json", dict(created_at=datetime.now(timezone.utc).isoformat(),
        suite_root=str(suite), cells=plan(), source_fingerprint=sources,
        baseline_root=str(baseline), seed=0, automatic_promotion=False,
        automatic_retry=False, parameter_search=False,
        scope="Full split; matching image order, vocabulary, preprocessing and episode batches. "
              "Baselines use archived native settings; this is not equal optimizer-budget tuning."))
    algorithm_rows = list(csv.DictReader((suite / "algorithm_summary.csv").open()))
    rows, accepted = [], {}
    for index, cell in enumerate(plan()):
        dataset, method = cell["dataset"], cell["method"]
        expected = parent["input_panels"][cell["panel"]]
        directory = output / dataset / method
        directory.parent.mkdir(parents=True, exist_ok=True)
        verify_code(output)
        adapter.check_snapshot(sources, adapter.source_snapshot(suite / "runtime", baseline, controls, worker))
        command = [sys.executable, str(worker), "--suite-root", str(suite), "--package-root",
            str(suite / "runtime"), "--baseline-root", str(baseline), "--dataset", dataset,
            "--method", method, "--panel-key", cell["panel"], "--output", str(directory), "--workers", "2"]
        write_json(output / "queue_state.json", dict(status="running", pid=os.getpid(),
            completed=index, total=len(plan()), current=cell, command=command))
        with (directory.parent / f"{method}.launcher.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                cwd=suite / "runtime", pass_fds=(own_lock.fileno(), dependency_lock.fileno()),
                env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        row = dict(dataset=dataset, method=method, images=len(expected["image_ids"]),
            status="failed", mIoU=None, ms_per_image=None, visual_inputs_per_image=None,
            visual_calls_per_image=None, result_path=str(directory / "result.json"),
            returncode=completed.returncode)
        if completed.returncode == 0:
            result = validate_cell(directory, expected, method, sources,
                expected_input_hashes=parent_inputs, expected_environment=parent["environment"])
            accepted[str(directory)] = accepted_hashes(directory)
            row.update(status="completed", mIoU=result["miou_unrounded"],
                ms_per_image=result["ms_per_image"], visual_inputs_per_image=result["visual_inputs_per_image"],
                visual_calls_per_image=result["visual_calls_per_image"])
        rows.append(row)
        publish(output, rows, algorithm_rows)
        print(json.dumps(dict(completed=index + 1, total=len(plan()), **row)), flush=True)
    verify_code(output)
    verify_suite(suite)
    adapter.check_snapshot(sources, adapter.source_snapshot(suite / "runtime", baseline, controls, worker))
    for row in rows:
        if row["status"] == "completed":
            directory = Path(row["result_path"]).parent
            check_accepted(directory, accepted[str(directory)])
            validate_cell(directory, parent["input_panels"][row["dataset"] + "/full"],
                row["method"], sources, expected_input_hashes=parent_inputs,
                expected_environment=parent["environment"])
    publish(output, rows, algorithm_rows)
    write_json(output / "acceptance_hashes.json", accepted)
    failures = sum(r["status"] != "completed" for r in rows)
    write_json(output / "queue_state.json", dict(status="completed_with_failures" if failures else "completed",
        total=len(plan()), completed=len(rows), failures=failures, finished_at=datetime.now(timezone.utc).isoformat()))
    names = ["manifest.json", "code_seal.json", "baselines.csv", "comparison.csv", "queue_state.json", "acceptance_hashes.json"]
    names += [str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() and
              p.name in {"protocol.json", "result.json", "cost.csv", "per_image_counts.json", "failure.json"}]
    write_json(output / "SUITE_SEALED.json", dict(artifact_sha256={n: sha(output / n) for n in names},
                                                failures=failures, automatic_promotion=False))
    if not failures:
        run_analysis(output, suite)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("suite-root", "baseline-root", "output"):
        parser.add_argument("--" + key, type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args()
    output = args.output
    with ExitStack() as stack:
        if args.lock_fd is None:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError("Use a fresh followup directory")
            (output / "code").mkdir(parents=True, exist_ok=True)
            for name in FILES:
                shutil.copy2(Path(__file__).with_name(name), output / "code" / name)
            write_json(output / "code_seal.json", {name: sha(output / "code" / name) for name in FILES})
            lock = stack.enter_context((output / "queue.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            command = [sys.executable, str(output / "code" / CONTROLLER), "--suite-root", str(args.suite_root),
                "--baseline-root", str(args.baseline_root), "--output", str(output), "--lock-fd", str(lock.fileno())]
            options = dict(pass_fds=(lock.fileno(),), cwd=output / "code")
            if args.detach:
                with (output / "queue.log").open("a") as log:
                    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True, **options)
                write_json(output / "launch.json", dict(pid=child.pid, command=command))
                print(f"Queued matched full baseline followup {child.pid}: {output}", flush=True)
            else:
                subprocess.run(command, check=True, **options)
        else:
            if Path(__file__).resolve() != output / "code" / CONTROLLER:
                raise RuntimeError("Followup must execute frozen code")
            lock = stack.enter_context(os.fdopen(args.lock_fd, "a"))
            if not os.path.samestat(os.fstat(lock.fileno()), (output / "queue.lock").stat()):
                raise RuntimeError("Inherited queue lock mismatch")
            try:
                execute(args, lock, stack)
            except BaseException as error:
                write_json(output / "queue_state.json", dict(status="failed", error=type(error).__name__,
                    message=str(error), automatic_retry=False))
                raise


if __name__ == "__main__":
    main()
