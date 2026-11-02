"""Frozen, sequential two-view full validation, with PC60 first."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmark_simplification import execution_environment, source_hashes, write_json
from run_algorithm_efficiency_suite import (
    acquire_lock, run_cell, save_csv, snapshot_runtime, tool_dependencies,
    verify_snapshot, verify_suite_seal,
)
from run_removal_suite import digest, utc_now
from suite_inputs import EVALUATION_COUNTS, expected_inputs, freeze_protocol


CONTROLLER = "run_two_view_full_suite.py"
DATASETS = ("pc60", "voc21", "coco_cd", "coco_object")


def experiment_plan():
    return [dict(stage="full_validation", dataset=d, limit=EVALUATION_COUNTS[d],
                 algorithm="quality_only", variant="full", panel=f"{d}/full",
                 key=f"full_validation/{d}/quality_only") for d in DATASETS]


def input_hashes(panels):
    hashes = {}
    for panel in panels.values():
        for name in (*panel["image_ids"], *panel["label_ids"]):
            if name in hashes:
                continue
            path = Path(name)
            before = path.stat()
            h = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Input changed while hashing: {name}")
            hashes[name] = h.hexdigest()
    return hashes


def discover(runtime, manifest):
    return expected_inputs(runtime, Path(manifest["data_root"]),
        {c["panel"]: (c["dataset"], "evaluation", None) for c in manifest["cells"]},
        data_dirs=manifest["data_dirs"], selection_seed=20261024)


def prepare(args):
    if not verify_suite_seal(args.pilot):
        raise RuntimeError("The pilot must be sealed")
    pilot = json.loads((args.pilot / "suite_protocol.json").read_text())
    tools = Path(__file__).resolve().parent
    source = source_hashes(args.package_root / "trace_ovss")
    if source != pilot["source_hashes"]:
        raise RuntimeError("Production algorithm changed since the pilot")
    verify_inherited_algorithm(tools, pilot)
    request = dict(created_at=utc_now(), source_hashes=source,
        tool_hashes={name: digest(tools / name) for name in tool_dependencies(tools, (CONTROLLER,))},
        environment=execution_environment(), data_root=pilot["data_root"],
        data_dirs=pilot["data_dirs"], cells=experiment_plan(), seed=0,
        selection_seed=20261024, config_fixed=True, parameter_search=False,
        automatic_promotion=False, automatic_retry=False,
        pilot=str(args.pilot), pilot_seal_sha256=digest(args.pilot / "SUITE_SEALED.json"),
        objective_epsilon=pilot["objective_epsilon"],
        scope="Complete evaluation splits, seed 0. No outcomes used for parameter selection.",
        acceptance="Compare against matched baselines; old TRACE is not an acceptance threshold.")
    runtime = snapshot_runtime(args.output, args.package_root, tools, request)
    panels = discover(runtime, request)
    for dataset in DATASETS:
        if panels[f"{dataset}/full"]["config"] != pilot["input_panels"][f"{dataset}/128"]["config"]:
            raise RuntimeError(f"Configuration changed since pilot: {dataset}")
    manifest = dict(**request, input_panels=panels)
    freeze_protocol(args.output / "suite_protocol.json", manifest)
    freeze_protocol(args.output / "fixed_configs.json", {
        d: panels[f"{d}/full"]["config"] for d in DATASETS})
    freeze_protocol(args.output / "input_hashes.json", input_hashes(panels))
    write_json(args.output / "queue_state.json", dict(status="prepared", total=4, completed=0))
    return manifest


def verify_inherited_algorithm(tools, pilot):
    for name in ("benchmark_simplification.py", "trace_algorithm_variants.py", "trace_ablation_variants.py"):
        if digest(tools / name) != pilot["tool_hashes"][name]:
            raise RuntimeError(f"Algorithm implementation changed since pilot: {name}")


def verify_inputs(output, manifest):
    verify_snapshot(output / "runtime", manifest)
    if execution_environment() != manifest["environment"]:
        raise RuntimeError("Execution environment changed")
    if discover(output / "runtime", manifest) != manifest["input_panels"]:
        raise RuntimeError("Full input population/configuration changed")
    if input_hashes(manifest["input_panels"]) != json.loads((output / "input_hashes.json").read_text()):
        raise RuntimeError("Input content changed")
    pilot = Path(manifest["pilot"])
    if digest(pilot / "SUITE_SEALED.json") != manifest["pilot_seal_sha256"]:
        raise RuntimeError("Pilot seal changed")


def execute(output, manifest, lock):
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                     "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError(f"Existing GPU workers will not be interrupted: {active}")
    verify_inputs(output, manifest)
    rows, episodes = [], []
    for cell in manifest["cells"]:
        verify_snapshot(output / "runtime", manifest)
        row, costs = run_cell(output, output / "runtime", manifest, cell, (lock,))
        rows.append(row)
        episodes.extend(costs)
        save_csv(output / "summary.csv", rows)
        save_csv(output / "algorithm_summary.csv", rows)
        save_csv(output / "all_costs.csv", episodes)
        print(json.dumps(dict(completed=len(rows), total=4, **row)), flush=True)
    verify_inputs(output, manifest)
    for cell in manifest["cells"]:
        run_cell(output, output / "runtime", manifest, cell, (lock,), allow_execution=False)
    write_json(output / "queue_state.json", dict(status="completed", total=4, completed=4,
                                                finished_at=utc_now()))
    names = ["suite_protocol.json", "fixed_configs.json", "input_hashes.json", "queue_state.json",
             "summary.csv", "algorithm_summary.csv", "all_costs.csv"]
    for cell in manifest["cells"]:
        names.append(cell["key"] + "/selected_result.json")
        names += [str(p.relative_to(output)) for p in (output / cell["key"]).rglob("*")
                  if p.is_file() and p.suffix in {".json", ".csv"}]
    freeze_protocol(output / "SUITE_SEALED.json", dict(
        artifact_sha256={name: digest(output / name) for name in sorted(set(names))},
        full_validation_cells=4, seed=0, automatic_promotion=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--package-root", type=lambda p: Path(p).resolve())
    parser.add_argument("--pilot", type=lambda p: Path(p).resolve())
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args()
    output = args.output
    with ExitStack() as stack:
        if args.lock_fd is None:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError("A fresh output directory is required")
            if not args.package_root or not args.pilot:
                parser.error("Preparation needs --package-root and --pilot")
            output.mkdir(parents=True, exist_ok=True)
            lock = acquire_lock(stack, output / "queue.lock", nonblocking=True)
            prepare(args)
            if args.prepare_only:
                return
            command = [sys.executable, str(output / "runtime/tools" / CONTROLLER),
                       "--output", str(output), "--lock-fd", str(lock.fileno())]
            options = dict(pass_fds=(lock.fileno(),), cwd=output / "runtime")
            if args.detach:
                with (output / "queue.log").open("a") as log:
                    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True, **options)
                write_json(output / "launch.json", dict(pid=child.pid, command=command, launched_at=utc_now()))
                print(f"Started sequential frozen controller {child.pid}: {output}", flush=True)
            else:
                subprocess.run(command, check=True, **options)
        else:
            if Path(__file__).resolve() != output / "runtime/tools" / CONTROLLER:
                raise RuntimeError("Execution must use the frozen controller")
            lock = stack.enter_context(os.fdopen(args.lock_fd, "a"))
            if not os.path.samestat(os.fstat(lock.fileno()), (output / "queue.lock").stat()):
                raise RuntimeError("Inherited queue lock mismatch")
            manifest = json.loads((output / "suite_protocol.json").read_text())
            try:
                execute(output, manifest, lock)
            except BaseException as error:
                old = json.loads((output / "queue_state.json").read_text())
                write_json(output / "queue_state.json", dict(old, status="failed",
                    error=type(error).__name__, message=str(error), automatic_retry=False))
                raise


if __name__ == "__main__":
    main()
