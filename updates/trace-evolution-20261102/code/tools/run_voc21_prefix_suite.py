"""Bounded VOC21 forward reduction: lazy follow-up and cached-prefix study."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from benchmark_simplification import execution_environment, source_hashes, write_json
from run_algorithm_efficiency_suite import (
    acquire_lock, check_close, run_cell, save_csv, snapshot_runtime,
    tool_dependencies, validate_algorithm_artifact, validate_parameter_record, verify_snapshot,
)
from run_matched_baseline_suite import validate_cell, verify_suite
from run_removal_suite import digest, utc_now
from run_two_view_full_suite import input_hashes
from suite_inputs import EVALUATION_COUNTS, expected_inputs, freeze_protocol


CONTROLLER = "run_voc21_prefix_suite.py"
ALGORITHMS = ("quality_tail6", "quality_tail12", "quality_tail0", "quality_tail6_uncached")
CALIBRATION_ALGORITHMS = (*ALGORITHMS, "quality_only")
LAZY_ALGORITHMS = ("quality_lazy_01", "quality_lazy_class_01")
PREFIX_FULL_ALGORITHMS = ("quality_tail12", "quality_tail6")
FULL_ALGORITHMS = (*LAZY_ALGORITHMS, *PREFIX_FULL_ALGORITHMS)
PARAMETER_ALGORITHMS = (*CALIBRATION_ALGORITHMS, *LAZY_ALGORITHMS)
TOTAL_CELLS = 13
BASELINES = ("noadapt", "tent", "tpt", "mlmp")
PATHS = ("output", "package_root", "parent_suite", "baseline_suite", "wait_for")
RESULTS = Path("/home/wwww/hzl/results")
DEFAULT_PATHS = dict(parent_suite=RESULTS / "trace_two_view_full_20261025",
                     baseline_suite=RESULTS / "trace_two_view_full_baselines_20261025",
                     wait_for=RESULTS / "trace_lazy_quality_gpu_20261025")
FULL_PANEL = "voc21/full"
SMOKE_PANEL = "voc21/calibration/4"
CALIBRATION_PANEL = "voc21/calibration/128"
SELECTION_SEED = 20261024
ENCODER_BLOCKS = 24
PREFIX_DEPTHS = dict(zip(ALGORITHMS, (6, 12, 0, 6)))


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plan():
    cells = []
    for stage, split, limit, algorithms, panel in (
        ("full_validation", "evaluation", None, LAZY_ALGORITHMS, FULL_PANEL),
        ("smoke", "calibration", 4, ALGORITHMS, SMOKE_PANEL),
        ("calibration", "calibration", 128, CALIBRATION_ALGORITHMS, CALIBRATION_PANEL),
        ("full_validation", "evaluation", None, PREFIX_FULL_ALGORITHMS, FULL_PANEL),
    ):
        cells.extend(dict(stage=stage, split=split, limit=limit, algorithm=algorithm,
                          dataset="voc21", variant="full", panel=panel,
                          key=f"{stage}/voc21/{algorithm}") for algorithm in algorithms)
    return cells


def parameter_names(algorithm):
    """Names for the inherited ViT-L/14 reduced encoder, without loading weights."""
    tail = PREFIX_DEPTHS.get(algorithm, ENCODER_BLOCKS)
    if algorithm not in PARAMETER_ALGORITHMS:
        raise ValueError(algorithm)
    names = [f"visual.ln_post.{p}" for p in ("weight", "bias")]
    if algorithm not in PREFIX_DEPTHS:
        names += [f"visual.ln_pre.{p}" for p in ("weight", "bias")]
    for index in range(ENCODER_BLOCKS - tail, ENCODER_BLOCKS):
        norms = ("ln_1", "ln_2") if index < ENCODER_BLOCKS - 1 else ("ln_1",)
        names += [f"visual.transformer.resblocks.{index}.{norm}.{p}"
                  for norm in norms for p in ("weight", "bias")]
    return sorted(names)


def study_policy():
    return dict(cells=plan(), seed=0, selection_seed=SELECTION_SEED,
                config_fixed=True, parameter_search=False, validation_selection=False,
                automatic_promotion=False, automatic_retry=False,
                full_algorithms=list(FULL_ALGORITHMS),
                lazy_full_followup=list(LAZY_ALGORITHMS),
                lazy_followup_scope="Full follow-up of both existing predeclared lazy variants; "
                                    "not test-metric threshold tuning or winner selection.",
                full_runs_independent_of_calibration_scores=True,
                calibration_role="algorithm_development_not_validation_selection",
                optimizer_parameter_names={a: parameter_names(a) for a in PARAMETER_ALGORITHMS},
                encoder_blocks=ENCODER_BLOCKS,
                scope="Both existing lazy variants on full VOC21 first, four train-split prefix "
                      "smoke cells, five fixed train calibration cells, then both predeclared "
                      "prefix full VOC21 validation cells. Retain every outcome; "
                      "no tuning, score replacement, winner selection or promotion.")


def validate_policy(manifest):
    for key, value in study_policy().items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Frozen study policy changed: {key}")


def discover(runtime, manifest):
    panels = {c["panel"]: (c["dataset"], c["split"], c["limit"]) for c in manifest["cells"]}
    return expected_inputs(runtime, Path(manifest["data_root"]), panels,
                           data_dirs=manifest["data_dirs"], selection_seed=SELECTION_SEED)


def validate_panels(panels, parent):
    if set(panels) != {SMOKE_PANEL, CALIBRATION_PANEL, FULL_PANEL}:
        raise RuntimeError("Unexpected study panels")
    full = panels[FULL_PANEL]
    if full != parent["input_panels"][FULL_PANEL]:
        raise RuntimeError("Full VOC21 inputs/configuration differ from the parent")
    if (full["split"] != "evaluation" or full["available_images"] != EVALUATION_COUNTS["voc21"]
            or len(full["image_ids"]) != full["available_images"]):
        raise RuntimeError("Full VOC21 population is incomplete")
    for key, count in ((SMOKE_PANEL, 4), (CALIBRATION_PANEL, 128), (FULL_PANEL, full["available_images"])):
        panel = panels[key]
        if (panel["dataset"] != "voc21" or panel["seed"] != 0
                or panel["selection_seed"] != SELECTION_SEED
                or len(panel["image_ids"]) != count or len(panel["label_ids"]) != count
                or len(set(panel["image_ids"])) != count or len(set(panel["label_ids"])) != count
                or len(panel["file_metadata"]) != 2 * count
                or panel["available_images"] < count):
            raise RuntimeError("Incomplete or duplicate fixed panel")
        if panel["config"] != full["config"] or panel["vocabulary_sha256"] != full["vocabulary_sha256"]:
            raise RuntimeError("Calibration must inherit the full configuration and vocabulary")
        if key == FULL_PANEL:
            continue
        if panel["split"] != "calibration":
            raise RuntimeError("Development panels must use the calibration split")
        for field in ("image_ids", "label_ids"):
            # Resolve aliases as well as names; VOC IDs identify the same image across roots.
            ids = {Path(p).resolve() for p in panel[field]}
            validation = {Path(p).resolve() for p in full[field]}
            if ids & validation or {p.stem for p in ids} & {p.stem for p in validation}:
                raise RuntimeError("Calibration and evaluation images/labels are not disjoint")
    for field in ("image_ids", "label_ids"):
        if not set(panels[SMOKE_PANEL][field]).issubset(panels[CALIBRATION_PANEL][field]):
            raise RuntimeError("Smoke must be nested within calibration")


def require_owned_artifact(path, root):
    path = Path(path).resolve()
    root = root.resolve()
    if not path.is_relative_to(root):
        raise RuntimeError("Comparison artifact is outside its sealed owner")
    relative = path.relative_to(root).as_posix()
    if read_json(root / "SUITE_SEALED.json")["artifact_sha256"].get(relative) != digest(path):
        raise RuntimeError("Comparison artifact is not bound to its owning seal")
    return path


def verify_baseline_binding(parent_root, baseline_root, parent):
    record = read_json(baseline_root / "manifest.json")
    controls = record["source_fingerprint"]["controls"]
    if (Path(record["suite_root"]).resolve() != parent_root.resolve()
            or record["source_fingerprint"]["native"] != parent["source_hashes"]):
        raise RuntimeError("Full baselines are not bound to the source queue")
    for name in ("SUITE_SEALED.json", "suite_protocol.json", "input_hashes.json", "fixed_configs.json"):
        path = parent_root / name
        if controls.get(str(path)) != digest(path):
            raise RuntimeError(f"Baseline parent seal/control changed: {name}")
    rows = read_csv(baseline_root / "baselines.csv")
    expected = {(d, m) for d in ("pc60", "voc21", "coco_cd", "coco_object") for m in BASELINES}
    keys = [(r["dataset"], r["method"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise RuntimeError("Full baseline suite must contain all four methods for all four datasets")
    for row in rows:
        count = len(parent["input_panels"][row["dataset"] + "/full"]["image_ids"])
        if row["status"] != "completed" or int(row["images"]) != count:
            raise RuntimeError("Full baseline suite contains failed or unmatched populations")
    return record, rows


def reference_rows(manifest):
    parent_root, baseline_root = Path(manifest["parent_suite"]), Path(manifest["baseline_suite"])
    parent = read_json(parent_root / "suite_protocol.json")
    record, rows = verify_baseline_binding(parent_root, baseline_root, parent)
    expected = parent["input_panels"][FULL_PANEL]
    hashes = read_json(parent_root / "input_hashes.json")
    references = []
    for row in rows:
        if row["dataset"] != "voc21":
            continue
        path = require_owned_artifact(row["result_path"], baseline_root)
        for name in ("protocol.json", "cost.csv", "per_image_counts.json"):
            require_owned_artifact(path.with_name(name), baseline_root)
        result = validate_cell(path.parent, expected, row["method"], record["source_fingerprint"],
                               expected_input_hashes=hashes, expected_environment=parent["environment"])
        for key, value in (("mIoU", result["miou_unrounded"]), ("ms_per_image", result["ms_per_image"])):
            check_close(row[key], value, key)
        references.append(dict(dataset="voc21", method=row["method"], images=result["images"],
                               mIoU=result["miou_unrounded"], ms_per_image=result["ms_per_image"],
                               peak_allocated_mib=result["peak_allocated_mib"], result_path=str(path)))
    source = [r for r in read_csv(parent_root / "algorithm_summary.csv")
              if r["dataset"] == "voc21" and r["algorithm"] == "quality_only"]
    if len(source) != 1:
        raise RuntimeError("Full source queue must have exactly one VOC21 quality_only result")
    path = require_owned_artifact(source[0]["result_path"], parent_root)
    result, costs, _ = validate_algorithm_artifact(path.parent, expected, parent, "quality_only")
    for key, value in (("mIoU", result["miou_unrounded"]), ("ms_per_image", costs["ms_per_image"])):
        check_close(source[0][key], value, key)
    references.append(dict(dataset="voc21", method="quality_only", images=result["images"],
                           mIoU=result["miou_unrounded"], ms_per_image=costs["ms_per_image"],
                           peak_allocated_mib=costs["peak_allocated_mib"], result_path=str(path)))
    return references


def checkpoint_identity():
    path = Path.home() / ".cache/clip/ViT-L-14.pt"
    before = path.stat()
    value = dict(path=str(path), bytes=before.st_size, mtime_ns=before.st_mtime_ns, sha256=digest(path))
    after = path.stat()
    if (after.st_size, after.st_mtime_ns, after.st_ino) != (before.st_size, before.st_mtime_ns, before.st_ino):
        raise RuntimeError("Checkpoint changed while hashing")
    return value


def verify_upstream(manifest):
    for key in ("parent_suite", "baseline_suite"):
        root = Path(manifest[key])
        verify_suite(root)
        if digest(root / "SUITE_SEALED.json") != manifest[key + "_seal_sha256"]:
            raise RuntimeError("Upstream completion seal changed")
    if digest(Path(manifest["wait_for"]) / "suite_protocol.json") != manifest["dependency_protocol_sha256"]:
        raise RuntimeError("Pending dependency protocol changed")


def prepare(args):
    parent_root, baseline_root = args.parent_suite, args.baseline_suite
    verify_suite(parent_root)
    verify_suite(baseline_root)
    parent = read_json(parent_root / "suite_protocol.json")
    baseline, _ = verify_baseline_binding(parent_root, baseline_root, parent)
    pending = read_json(args.wait_for / "suite_protocol.json")
    source = source_hashes(args.package_root / "trace_ovss")
    environment = execution_environment()
    if not source or source != parent["source_hashes"] or source != pending["source_hashes"]:
        raise RuntimeError("Production source changed from the parent or pending dependency")
    if environment != parent["environment"] or environment != pending["environment"]:
        raise RuntimeError("Inherited execution environment changed")
    if (Path(pending["wait_for"]).resolve() != baseline_root
            or pending["dependency_code_sha256"] != digest(baseline_root / "code_seal.json")):
        raise RuntimeError("Pending queue does not follow the frozen baseline suite")
    lazy_cells = [c["algorithm"] for c in pending["cells"] if c["dataset"] == "voc21"
                  and c["stage"] == "evaluation" and c["limit"] == 128]
    if (not set(LAZY_ALGORITHMS).issubset(lazy_cells)
            or pending["input_panels"]["voc21/128"]["config"] != parent["input_panels"][FULL_PANEL]["config"]):
        raise RuntimeError("Lazy follow-up must inherit both predeclared variants and the fixed configuration")
    checkpoint = checkpoint_identity()
    if baseline["source_fingerprint"]["controls"].get(checkpoint["path"]) != checkpoint["sha256"]:
        raise RuntimeError("Checkpoint differs from the full baselines")
    tools = Path(__file__).resolve().parent
    for name in ("trace_lazy_quality.py", "trace_algorithm_variants.py", "trace_ablation_variants.py"):
        if digest(tools / name) != pending["tool_hashes"][name]:
            raise RuntimeError(f"Existing lazy algorithm changed before full follow-up: {name}")
    dependencies = tool_dependencies(tools, (CONTROLLER, "trace_cached_prefix.py"))
    request = dict(**{key: str(getattr(args, key)) for key in PATHS}, **study_policy(),
                   created_at=utc_now(), source_hashes=source,
                   tool_hashes={name: digest(tools / name) for name in dependencies},
                   environment=environment, checkpoint=checkpoint, data_root=parent["data_root"],
                   data_dirs=parent["data_dirs"], objective_epsilon=parent["objective_epsilon"],
                   parent_suite_seal_sha256=digest(parent_root / "SUITE_SEALED.json"),
                   baseline_suite_seal_sha256=digest(baseline_root / "SUITE_SEALED.json"),
                   dependency_protocol_sha256=digest(args.wait_for / "suite_protocol.json"))
    freeze_protocol(args.output / "requested_protocol.json", request)
    runtime = snapshot_runtime(args.output, args.package_root, tools, request)
    panels = discover(runtime, request)
    validate_panels(panels, parent)
    hashes = input_hashes(panels)
    parent_hashes = read_json(parent_root / "input_hashes.json")
    full = panels[FULL_PANEL]
    if any(hashes[name] != parent_hashes.get(name) for name in (*full["image_ids"], *full["label_ids"])):
        raise RuntimeError("Full VOC21 input bytes differ from the sealed parent")
    manifest = dict(**request, input_panels=panels)
    freeze_protocol(args.output / "suite_protocol.json", manifest)
    freeze_protocol(args.output / "input_hashes.json", hashes)
    freeze_protocol(args.output / "fixed_configs.json", dict(voc21=full["config"]))
    names = ("requested_protocol.json", "suite_protocol.json", "input_hashes.json", "fixed_configs.json")
    freeze_protocol(args.output / "code_seal.json", dict(artifact_sha256={n: digest(args.output / n) for n in names}))
    verify_upstream(manifest)
    verify_snapshot(runtime, manifest)
    write_json(args.output / "queue_state.json", dict(status="prepared", total=TOTAL_CELLS, completed=0,
                                                      automatic_retry=False))
    return manifest


def verify_inputs(output, manifest):
    validate_policy(manifest)
    for name, value in read_json(output / "code_seal.json")["artifact_sha256"].items():
        if digest(output / name) != value:
            raise RuntimeError(f"Prepared input/protocol seal changed: {name}")
    verify_snapshot(output / "runtime", manifest)
    verify_upstream(manifest)
    parent = read_json(Path(manifest["parent_suite"]) / "suite_protocol.json")
    validate_panels(manifest["input_panels"], parent)
    if execution_environment() != manifest["environment"] or checkpoint_identity() != manifest["checkpoint"]:
        raise RuntimeError("Execution environment or checkpoint changed")
    if discover(output / "runtime", manifest) != manifest["input_panels"]:
        raise RuntimeError("Input metadata, population or configuration changed")
    if input_hashes(manifest["input_panels"]) != read_json(output / "input_hashes.json"):
        raise RuntimeError("Frozen input bytes changed")


def ensure_idle(proc_root=Path("/proc")):
    workers = []
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process / "cmdline").read_bytes().decode().split("\0")
            if any(Path(arg).name in {"benchmark_simplification.py", "benchmark_matched_baselines.py"}
                   for arg in command) and "--output" in command:
                workers.append(int(process.name))
        except (OSError, UnicodeError):
            continue
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                     text=True).strip()
    if workers or active:
        raise RuntimeError(f"Existing workers will not be interrupted: benchmark={workers}, GPU={active}")


def wait_for_dependencies(output, manifest, stack):
    write_json(output / "queue_state.json", dict(status="waiting_for_lazy_quality", pid=os.getpid(),
               dependency=manifest["wait_for"], total=TOTAL_CELLS, completed=0, automatic_retry=False))
    locks = []
    # Follow the existing chain downstream to upstream; never hold a parent while waiting on lazy.
    for key in ("wait_for", "baseline_suite", "parent_suite"):
        root = Path(manifest[key])
        lock = stack.enter_context((root / "queue.lock").open("r"))
        fcntl.flock(lock, fcntl.LOCK_EX)
        locks.append(lock)
        verify_suite(root)
    verify_upstream(manifest)
    freeze_protocol(output / "dependency_checks.json", dict(
        dependency_protocol_sha256=manifest["dependency_protocol_sha256"],
        completion_seals={key: digest(Path(manifest[key]) / "SUITE_SEALED.json")
                          for key in ("wait_for", "baseline_suite", "parent_suite")},
        completed=True, lock_order=["wait_for", "baseline_suite", "parent_suite"]))
    return tuple(locks)


def verify_completion_dependencies(output, manifest):
    verify_upstream(manifest)
    checks = read_json(output / "dependency_checks.json")
    for key, value in checks["completion_seals"].items():
        root = Path(manifest[key])
        verify_suite(root)
        if digest(root / "SUITE_SEALED.json") != value:
            raise RuntimeError("Dependency completion seal changed after waiting")


def validate_parameter_metadata(directory, manifest, algorithm):
    record = read_json(directory / "adaptation_parameters.json")
    if (record["algorithm"] != algorithm or record["encoder_blocks"] != manifest["encoder_blocks"]
            or sorted(record["names"]) != manifest["optimizer_parameter_names"][algorithm]):
        raise RuntimeError("Optimizer parameter names differ from the frozen study")
    validate_parameter_record(record, manifest["optimizer_parameter_names"][algorithm])


def publish(output, rows, episodes, references):
    save_csv(output / "summary.csv", rows)
    save_csv(output / "calibration_summary.csv", [r for r in rows if r["stage"] == "calibration"])
    full = [r for r in rows if r["stage"] == "full_validation"]
    save_csv(output / "algorithm_summary.csv", full)
    save_csv(output / "all_costs.csv", episodes)
    save_csv(output / "references.csv", references)
    comparisons = []
    for row in full:
        for reference in references:
            if row["images"] != reference["images"] or row["full_split"] is not True:
                raise RuntimeError("Only matched full-split results may enter the baseline comparison")
            comparisons.append(dict(dataset="voc21", algorithm=row["algorithm"], reference=reference["method"],
                images=row["images"], mIoU=row["mIoU"], reference_mIoU=reference["mIoU"],
                delta_mIoU=row["mIoU"] - reference["mIoU"], ms_per_image=row["ms_per_image"],
                reference_ms_per_image=reference["ms_per_image"],
                runtime_ratio=row["ms_per_image"] / reference["ms_per_image"],
                peak_allocated_mib=row["peak_allocated_mib"],
                reference_peak_allocated_mib=reference["peak_allocated_mib"],
                result_path=row["result_path"], reference_result_path=reference["result_path"],
                scope="same full VOC21 IDs/config/vocabulary; baselines retain native optimizer settings"))
    save_csv(output / "comparison.csv", comparisons)


def publish_class_report(output, rows, references):
    baselines = [r for r in references if r["method"] in BASELINES]
    if len(baselines) != len(BASELINES) or {r["method"] for r in baselines} != set(BASELINES):
        raise RuntimeError("Class comparison requires all four full baselines")
    strongest = max(baselines, key=lambda row: row["mIoU"])

    def class_ious(result):
        intersection, union = result["intersection"], result["union"]
        if len(intersection) != 21 or len(union) != 21:
            raise RuntimeError("Full VOC21 class axis must contain all 21 classes")
        values = []
        for i, u in zip(intersection, union):
            if not math.isfinite(i) or not math.isfinite(u) or not 0 <= i <= u:
                raise RuntimeError("Invalid class segmentation counts")
            values.append(100 * i / u if u else None)
        return values

    baseline = class_ious(read_json(Path(strongest["result_path"])))
    records = []
    for row in rows:
        if row["stage"] != "full_validation":
            continue
        values = class_ious(read_json(Path(row["result_path"])))
        for index, (trace_iou, baseline_iou) in enumerate(zip(values, baseline)):
            records.append(dict(algorithm=row["algorithm"], class_index=index,
                TRACEIoU=trace_iou, baselineIoU=baseline_iou,
                delta=trace_iou - baseline_iou if trace_iou is not None and baseline_iou is not None else None,
                baseline=strongest["method"], result_path=row["result_path"],
                baseline_result_path=strongest["result_path"]))
    save_csv(output / "full_class_comparison.csv", records)


def execute(output, manifest, own_lock, stack):
    validate_policy(manifest)
    locks = (own_lock, *wait_for_dependencies(output, manifest, stack))
    ensure_idle()
    verify_inputs(output, manifest)
    references = reference_rows(manifest)
    rows, episodes = [], []
    for cell in manifest["cells"]:
        verify_snapshot(output / "runtime", manifest)
        ensure_idle()
        row, costs = run_cell(output, output / "runtime", manifest, cell, locks)
        validate_parameter_metadata(Path(row["result_path"]).parent, manifest, cell["algorithm"])
        rows.append(dict(row, split=cell["split"], panel=cell["panel"]))
        episodes.extend(dict(cost, split=cell["split"], panel=cell["panel"]) for cost in costs)
        publish(output, rows, episodes, references)
        write_json(output / "queue_state.json", dict(status="running", pid=os.getpid(), total=TOTAL_CELLS,
                   completed=len(rows), last_completed=cell, automatic_retry=False))
        print(json.dumps(dict(completed=len(rows), total=TOTAL_CELLS, **rows[-1])), flush=True)
    verify_inputs(output, manifest)
    verify_completion_dependencies(output, manifest)
    if reference_rows(manifest) != references:
        raise RuntimeError("Full comparison references changed")
    for cell in manifest["cells"]:
        row, _ = run_cell(output, output / "runtime", manifest, cell, locks, allow_execution=False)
        validate_parameter_metadata(Path(row["result_path"]).parent, manifest, cell["algorithm"])
    publish_class_report(output, rows, references)
    write_json(output / "queue_state.json", dict(status="completed", total=TOTAL_CELLS, completed=TOTAL_CELLS,
               finished_at=utc_now(), automatic_retry=False, automatic_promotion=False))
    names = ["requested_protocol.json", "suite_protocol.json", "code_seal.json", "fixed_configs.json",
             "input_hashes.json", "dependency_checks.json", "summary.csv", "calibration_summary.csv",
             "algorithm_summary.csv", "all_costs.csv", "references.csv", "comparison.csv",
             "full_class_comparison.csv", "queue_state.json"]
    for cell in manifest["cells"]:
        names += [str(p.relative_to(output)) for p in (output / cell["key"]).rglob("*")
                  if p.is_file() and p.suffix in {".json", ".csv"}]
    freeze_protocol(output / "SUITE_SEALED.json", dict(
        artifact_sha256={n: digest(output / n) for n in sorted(set(names))},
        smoke_cells=4, calibration_cells=5, full_validation_cells=4,
        lazy_full_validation_cells=2, prefix_full_validation_cells=2,
        full_algorithms=list(FULL_ALGORITHMS), automatic_promotion=False, automatic_retry=False,
        calibration_role=manifest["calibration_role"]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PATHS:
        parser.add_argument("--" + name.replace("_", "-"), type=lambda p: Path(p).resolve(),
                            required=name not in DEFAULT_PATHS, default=DEFAULT_PATHS.get(name))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--detach", action="store_true")
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def launch_controller(args, lock):
    command = [sys.executable, str(args.output / "runtime/tools" / CONTROLLER), "--lock-fd", str(lock.fileno())]
    for name in PATHS:
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    options = dict(pass_fds=(lock.fileno(),), cwd=args.output / "runtime")
    if args.detach:
        with (args.output / "queue.log").open("a") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, **options)
        write_json(args.output / "launch.json", dict(pid=child.pid, command=command, launched_at=utc_now()))
        print(f"Queued frozen VOC21 forward-reduction study {child.pid}: {args.output}", flush=True)
    else:
        subprocess.run(command, check=True, **options)


def main(argv=None):
    args = parse_args(argv)
    with ExitStack() as stack:
        if args.lock_fd is None:
            for key in ("parent_suite", "baseline_suite", "wait_for", "package_root"):
                root = getattr(args, key)
                if args.output.is_relative_to(root) or root.is_relative_to(args.output):
                    raise RuntimeError("Output must be separate from source and existing queues")
            if args.output.exists() and any(args.output.iterdir()):
                raise FileExistsError("Use a fresh study directory; no automatic resume or retry")
            args.output.mkdir(parents=True, exist_ok=True)
            lock = acquire_lock(stack, args.output / "queue.lock", nonblocking=True)
        else:
            if (Path(__file__).resolve() != args.output / "runtime/tools" / CONTROLLER
                    or args.prepare_only or args.detach):
                raise RuntimeError("Execution requires the frozen controller and inherited lock")
            lock = stack.enter_context(os.fdopen(args.lock_fd, "a"))
            if not os.path.samestat(os.fstat(lock.fileno()), (args.output / "queue.lock").stat()):
                raise RuntimeError("Inherited queue lock mismatch")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if read_json(args.output / "queue_state.json").get("status") != "prepared":
                raise RuntimeError("Only a freshly prepared queue may execute; no automatic retry")
        try:
            if args.lock_fd is None:
                prepare(args)
                if not args.prepare_only:
                    launch_controller(args, lock)
            else:
                manifest = read_json(args.output / "suite_protocol.json")
                if any(manifest[key] != str(getattr(args, key)) for key in PATHS):
                    raise RuntimeError("Prepared request paths changed")
                execute(args.output, manifest, lock, stack)
        except BaseException as error:
            state_path = args.output / "queue_state.json"
            state = read_json(state_path) if state_path.exists() else {}
            write_json(state_path, {**state, "status": "failed", "error": type(error).__name__,
                                   "message": str(error), "automatic_retry": False})
            raise


if __name__ == "__main__":
    main()
