"""Fixed-configuration lazy two-view experiment after the full baseline queue."""

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import csv

from benchmark_simplification import execution_environment, source_hashes, write_json
from run_algorithm_efficiency_suite import (
    acquire_lock, run_cell, save_csv, snapshot_runtime, tool_dependencies, verify_snapshot,
)
from run_matched_baseline_suite import publish, verify_suite
from run_removal_suite import digest, utc_now
from run_two_view_full_suite import input_hashes
from suite_inputs import expected_inputs, freeze_protocol


CONTROLLER = "run_lazy_quality_suite.py"
ALGORITHMS = ("quality_only", "quality_staged", "quality_lazy_01", "quality_lazy_class_01")
DATASETS = ("pc60", "voc21", "coco_cd", "coco_object")
PATHS = ("output", "package_root", "pilot", "baseline_pilot", "wait_for")


def verify_baseline_binding(pilot, baselines):
    record = json.loads((baselines / "manifest.json").read_text())
    if (Path(record["suite_root"]).resolve() != pilot.resolve()
            or record["suite_seal_sha256"] != digest(pilot / "SUITE_SEALED.json")
            or Path(record["package_root"]).resolve() != (pilot / "runtime").resolve()):
        raise RuntimeError("Baseline pilot is not bound to this algorithm pilot and seal")
    parent = json.loads((pilot / "suite_protocol.json").read_text())
    for row in csv.DictReader((baselines / "baselines.csv").open()):
        panel = parent["input_panels"].get(row["dataset"] + "/128")
        if panel is None or int(row["images"]) != len(panel["image_ids"]):
            raise RuntimeError("Baseline and algorithm populations differ")


def plan():
    cells = [dict(stage="smoke", dataset="pc60", limit=4, algorithm=a) for a in ALGORITHMS]
    for i, dataset in enumerate(DATASETS):
        order = ALGORITHMS[i:] + ALGORITHMS[:i]
        cells += [dict(stage="evaluation", dataset=dataset, limit=128, algorithm=a) for a in order]
    for c in cells:
        c.update(variant="full", panel=f'{c["dataset"]}/{c["limit"]}',
                 key=f'{c["stage"]}/{c["dataset"]}/{c["algorithm"]}')
    return cells


def discover(runtime, manifest):
    panels = {c["panel"]: (c["dataset"], "evaluation", c["limit"]) for c in manifest["cells"]}
    return expected_inputs(runtime, Path(manifest["data_root"]), panels,
                           data_dirs=manifest["data_dirs"], selection_seed=20261024)


def prepare(args):
    verify_suite(args.pilot)
    verify_suite(args.baseline_pilot)
    verify_baseline_binding(args.pilot, args.baseline_pilot)
    parent = json.loads((args.pilot / "suite_protocol.json").read_text())
    source = source_hashes(args.package_root / "trace_ovss")
    if source != parent["source_hashes"]:
        raise RuntimeError("Production source changed since fixed-config pilot")
    tools = Path(__file__).parent
    requested = dict(**{p: str(getattr(args, p)) for p in PATHS},
        created_at=utc_now(), source_hashes=source,
        tool_hashes={n: digest(tools / n) for n in tool_dependencies(tools, (CONTROLLER,))},
        environment=execution_environment(), data_root=parent["data_root"], data_dirs=parent["data_dirs"],
        cells=plan(), seed=0, selection_seed=20261024, config_fixed=True,
        objective_epsilon=parent["objective_epsilon"], parameter_search=False,
        automatic_promotion=False, automatic_retry=False,
        dependency_code_sha256=digest(args.wait_for / "code_seal.json"),
        pilot_seal_sha256=digest(args.pilot / "SUITE_SEALED.json"),
        baseline_pilot_seal_sha256=digest(args.baseline_pilot / "SUITE_SEALED.json"),
        scope="Four CPU-tested algorithms, four 4-image smoke cells and sixteen fixed 128-image cells. "
              "No outcomes used to select parameters. Development evidence, not full-split results.",
        certificate="Conditional bound on normalized loss coefficients, not gradients, Adam updates or mIoU.")
    runtime = snapshot_runtime(args.output, args.package_root, tools, requested)
    panels = discover(runtime, requested)
    for dataset in DATASETS:
        if panels[dataset + "/128"] != parent["input_panels"][dataset + "/128"]:
            raise RuntimeError(f"Pilot inputs/configuration changed: {dataset}")
    from trace_lazy_quality import LAZY_DEFINITIONS
    manifest = dict(**requested, input_panels=panels,
                    algorithm_definitions={"quality_only": "All paired T/C views, min(T,C)", **LAZY_DEFINITIONS})
    freeze_protocol(args.output / "suite_protocol.json", manifest)
    freeze_protocol(args.output / "input_hashes.json", input_hashes(panels))
    write_json(args.output / "queue_state.json", dict(status="prepared", total=20, completed=0))


def verify_inputs(output, manifest):
    verify_snapshot(output / "runtime", manifest)
    if execution_environment() != manifest["environment"] or discover(output / "runtime", manifest) != manifest["input_panels"]:
        raise RuntimeError("Environment, configurations or populations changed")
    if input_hashes(manifest["input_panels"]) != json.loads((output / "input_hashes.json").read_text()):
        raise RuntimeError("Input content changed")
    for key in ("pilot", "baseline_pilot"):
        root = Path(manifest[key])
        verify_suite(root)
        if digest(root / "SUITE_SEALED.json") != manifest[key + "_seal_sha256"]:
            raise RuntimeError("Upstream evidence seal changed")
    verify_baseline_binding(Path(manifest["pilot"]), Path(manifest["baseline_pilot"]))


def execute(output, manifest, lock, stack):
    dependency = Path(manifest["wait_for"])
    write_json(output / "queue_state.json", dict(status="waiting_for_full_baselines", pid=os.getpid(),
        dependency=str(dependency), total=20, completed=0))
    gate = stack.enter_context((dependency / "queue.lock").open("a"))
    fcntl.flock(gate, fcntl.LOCK_EX)
    verify_suite(dependency)
    if digest(dependency / "code_seal.json") != manifest["dependency_code_sha256"]:
        raise RuntimeError("Dependency implementation changed after preparation")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError(f"Existing GPU workers will not be interrupted: {active}")
    verify_inputs(output, manifest)
    rows, costs = [], []
    for cell in manifest["cells"]:
        verify_snapshot(output / "runtime", manifest)
        row, episodes = run_cell(output, output / "runtime", manifest, cell, (lock, gate))
        rows.append(row)
        costs.extend(episodes)
        save_csv(output / "summary.csv", rows)
        save_csv(output / "algorithm_summary.csv", [r for r in rows if r["stage"] == "evaluation"])
        save_csv(output / "all_costs.csv", costs)
        print(json.dumps(dict(completed=len(rows), total=20, **row)), flush=True)
    verify_inputs(output, manifest)
    for cell in manifest["cells"]:
        run_cell(output, output / "runtime", manifest, cell, (lock, gate), allow_execution=False)
    base_rows = list(csv.DictReader((Path(manifest["baseline_pilot"]) / "baselines.csv").open()))
    for row in base_rows:
        for key in ("mIoU", "ms_per_image"):
            row[key] = float(row[key]) if row[key] else None
    publish(output, base_rows, [r for r in rows if r["stage"] == "evaluation"])
    write_json(output / "queue_state.json", dict(status="completed", total=20, completed=20, finished_at=utc_now()))
    names = ["suite_protocol.json", "input_hashes.json", "summary.csv", "algorithm_summary.csv",
             "all_costs.csv", "baselines.csv", "comparison.csv", "queue_state.json"]
    for cell in manifest["cells"]:
        names += [str(p.relative_to(output)) for p in (output / cell["key"]).rglob("*")
                  if p.is_file() and p.suffix in {".csv", ".json"}]
    freeze_protocol(output / "SUITE_SEALED.json", dict(artifact_sha256={n: digest(output / n) for n in names},
        automatic_promotion=False, approximation_tolerance=.01, full_split=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in PATHS:
        parser.add_argument("--" + path.replace("_", "-"), type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args()
    with ExitStack() as stack:
        if args.lock_fd is None:
            if args.output.exists() and any(args.output.iterdir()):
                raise FileExistsError("Use a fresh algorithm-study directory")
            args.output.mkdir(parents=True, exist_ok=True)
            lock = acquire_lock(stack, args.output / "queue.lock", nonblocking=True)
            prepare(args)
            command = [sys.executable, str(args.output / "runtime/tools" / CONTROLLER), "--lock-fd", str(lock.fileno())]
            for p in PATHS:
                command += ["--" + p.replace("_", "-"), str(getattr(args, p))]
            options = dict(pass_fds=(lock.fileno(),), cwd=args.output / "runtime")
            if args.detach:
                with (args.output / "queue.log").open("a") as log:
                    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True, **options)
                write_json(args.output / "launch.json", dict(pid=child.pid, command=command))
                print(f"Queued lazy two-view study {child.pid}: {args.output}", flush=True)
            else:
                subprocess.run(command, check=True, **options)
        else:
            if Path(__file__).resolve() != args.output / "runtime/tools" / CONTROLLER:
                raise RuntimeError("Execution requires the frozen controller")
            lock = stack.enter_context(os.fdopen(args.lock_fd, "a"))
            if not os.path.samestat(os.fstat(lock.fileno()), (args.output / "queue.lock").stat()):
                raise RuntimeError("Inherited lock mismatch")
            manifest = json.loads((args.output / "suite_protocol.json").read_text())
            if any(str(getattr(args, p)) != manifest[p] for p in PATHS):
                raise RuntimeError("Prepared request paths changed")
            try:
                execute(args.output, manifest, lock, stack)
            except BaseException as error:
                write_json(args.output / "queue_state.json", dict(status="failed", error=type(error).__name__,
                    message=str(error), automatic_retry=False))
                raise


if __name__ == "__main__":
    main()
