"""Fixed-config, fewer-forward comparisons after the existing GPU queues."""

import argparse
import ast
from contextlib import ExitStack
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from benchmark_simplification import execution_environment, source_hashes, write_json
from finalize_removal_suite import active_workers
from run_removal_suite import completed_attempt, digest, next_attempt, utc_now, validate_artifact
from suite_inputs import expected_inputs, freeze_protocol


ALGORITHMS = ("standard", "staged_full", "quality_only", "selective_01", "selective_05")
DATASETS = ("voc21", "pc60", "coco_object", "coco_cd")
SELECTION_SEED = 20261024
CONTROLLER = "run_algorithm_efficiency_suite.py"
TOOL_FILES = (CONTROLLER, "benchmark_simplification.py", "trace_algorithm_variants.py")
PATH_ARGUMENTS = ("output", "package_root", "data_root", "batch_tuning", "parent_suite", "coco_object_dir")
TIMING_SCOPE = "GPU-synchronized adaptation and final inference; excludes loading/scoring/model setup"
PREFIX_DEPTHS = {"quality_tail6": 6, "quality_tail12": 12, "quality_tail0": 0,
                 "quality_tail6_uncached": 6}


def experiment_plan():
    cells = [dict(stage="smoke", dataset="voc21", limit=4, algorithm=a) for a in ALGORITHMS]
    for index, dataset in enumerate(DATASETS):
        order = ALGORITHMS[index:] + ALGORITHMS[:index]
        cells.extend(dict(stage="evaluation", dataset=dataset, limit=128, algorithm=a) for a in order)
    for cell in cells:
        cell.update(variant="full", key=f'{cell["stage"]}/{cell["dataset"]}/{cell["algorithm"]}',
                    panel=f'{cell["dataset"]}/{cell["limit"]}')
    return cells


def tool_dependencies(root, entrypoints=TOOL_FILES):
    """Follow local Python imports, including imports inside helper functions."""
    pending, visited = list(entrypoints), set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        path = root / name
        tree = ast.parse(path.read_text(), filename=str(path))
        visited.add(name)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = list(Path(name).parent.parts)
                if node.level:
                    base = base[:len(base) - node.level + 1]
                else:
                    base = []
                module = ".".join([*base, *([node.module] if node.module else [])])
                modules = [module] + [f"{module}.{alias.name}".strip(".") for alias in node.names]
            for module in modules:
                parts = module.split(".")
                candidates = [Path(*parts).with_suffix(".py")]
                candidates += [Path(*parts[:i]) / "__init__.py" for i in range(1, len(parts) + 1)]
                for candidate in candidates:
                    if (root / candidate).is_file() and candidate.as_posix() not in visited:
                        pending.append(candidate.as_posix())
    return tuple(sorted(visited))


def verify_snapshot(runtime, manifest):
    if source_hashes(runtime / "trace_ovss") != manifest["source_hashes"]:
        raise RuntimeError("Frozen production source changed")
    if source_hashes(runtime / "tools") != manifest["tool_hashes"]:
        raise RuntimeError("Frozen runtime tools or transitive helpers changed")


def snapshot_runtime(output, package, tools, request):
    runtime = output / "runtime"
    if not runtime.exists():
        temporary = output / "runtime.preparing"
        if temporary.exists():
            raise RuntimeError("Incomplete snapshot exists; preserve it for inspection")
        shutil.copytree(package / "trace_ovss", temporary / "trace_ovss",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in request["tool_hashes"]:
            destination = temporary / "tools" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tools / name, destination)
        verify_snapshot(temporary, request)
        temporary.rename(runtime)
    verify_snapshot(runtime, request)
    return runtime


def discover_panels(runtime, manifest):
    plan = {c["panel"]: (c["dataset"], "evaluation", c["limit"]) for c in manifest["cells"]}
    return expected_inputs(runtime, Path(manifest["data_root"]), plan,
                           data_dirs=manifest["data_dirs"], selection_seed=SELECTION_SEED)


def verify_dependency_protocols(manifest):
    for key, path_key in (("dependency_protocol_sha256", "batch_tuning"),
                          ("parent_protocol_sha256", "parent_suite")):
        if digest(Path(manifest[path_key]) / "suite_protocol.json") != manifest[key]:
            raise RuntimeError("Upstream protocol changed since preparation")


def prepare_suite(args):
    output, package = args.output, args.package_root
    tools = Path(__file__).resolve().parent
    hashes = source_hashes(package / "trace_ovss")
    if not hashes:
        raise RuntimeError("Missing production source")
    for dependency in (args.batch_tuning, args.parent_suite):
        upstream = json.loads((dependency / "suite_protocol.json").read_text())
        if upstream["source_hashes"] != hashes:
            raise RuntimeError("Production source differs from the original suite")
    request = dict(
        **{name: str(getattr(args, name)) for name in PATH_ARGUMENTS},
        source_hashes=hashes,
        tool_hashes={name: digest(tools / name) for name in tool_dependencies(tools)},
        environment=execution_environment(),
        data_dirs={"coco_object": str(args.coco_object_dir)},
        dependency_protocol_sha256=digest(args.batch_tuning / "suite_protocol.json"),
        parent_protocol_sha256=digest(args.parent_suite / "suite_protocol.json"),
        cells=experiment_plan(), algorithms=list(ALGORITHMS), seed=0, selection_seed=SELECTION_SEED,
        config_fixed=True, inherit_batch_winner=False, parameter_search=False,
        validation_selection=False, automatic_promotion=False,
        fixed="Inherited best_config per dataset: optimizer, K, batch, view batch, steps and learning rate.",
        scope="Five 4-image VOC21 smoke cells, then twenty fixed 128-image evaluation cells; not full-split evidence.",
        interpretation="All accuracy signs retained. No accuracy gate, parameter search, winner selection or promotion.")
    freeze_protocol(output / "requested_protocol.json", request)
    runtime = snapshot_runtime(output, package, tools, request)
    panels = discover_panels(runtime, request)
    # Input discovery installs the frozen package root before algorithm imports.
    from trace_algorithm_variants import ALGORITHMS as available, ALGORITHM_DEFINITIONS
    from trace_ovss.method.trace.constants import EPS
    if tuple(available) != ALGORITHMS:
        raise RuntimeError("Algorithm registry differs from the declared five comparisons")
    for cell in request["cells"]:
        panel = panels[cell["panel"]]
        if len(panel["image_ids"]) != cell["limit"]:
            raise RuntimeError("Incomplete fixed evaluation panel")
    if panels["voc21/4"]["config"] != panels["voc21/128"]["config"]:
        raise RuntimeError("Smoke and evaluation must use identical configuration")
    manifest = dict(**request, input_panels=panels, algorithm_definitions=ALGORITHM_DEFINITIONS,
                    objective_epsilon=EPS)
    freeze_protocol(output / "suite_protocol.json", manifest)
    freeze_protocol(output / "fixed_configs.json", {
        dataset: panels[f"{dataset}/128"]["config"] for dataset in DATASETS})
    verify_snapshot(runtime, manifest)
    return manifest


def acquire_lock(stack, path, nonblocking=False):
    lock = stack.enter_context(path.open("a"))
    fcntl.flock(lock, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
    return lock


def require_completed(root):
    state = json.loads((root / "queue_state.json").read_text())
    if state.get("status") != "completed":
        raise RuntimeError(f"Dependency did not complete: {root}: {state.get('status')}")
    return state


def wait_for_workers(output, root):
    while workers := active_workers(root):
        write_json(output / "queue_state.json", dict(status="waiting_for_existing_workers",
                   pid=os.getpid(), workers=workers, scope=str(root)))
        time.sleep(5)


def wait_for_dependencies(output, manifest, stack):
    dependency, parent = Path(manifest["batch_tuning"]), Path(manifest["parent_suite"])
    write_json(output / "queue_state.json", dict(status="waiting_for_batch_tuning", pid=os.getpid()))
    batch_lock = acquire_lock(stack, dependency / "queue.lock")
    batch_state = require_completed(dependency)
    wait_for_workers(output, dependency)
    write_json(output / "queue_state.json", dict(status="waiting_for_parent_locks", pid=os.getpid()))
    # Match the batch coordinator: audit before primary queue avoids deadlock.
    audit_lock = acquire_lock(stack, parent / "final_audit/audit.lock")
    audit_state = require_completed(parent / "final_audit")
    parent_lock = acquire_lock(stack, parent / "queue.lock")
    parent_state = require_completed(parent)
    wait_for_workers(output, Path("/"))
    verify_dependency_protocols(manifest)
    freeze_protocol(output / "dependency_checks.json", dict(
        batch_tuning=batch_state, parent_audit=audit_state, parent_suite=parent_state,
        no_active_benchmark_workers=True, inherited_config_or_outcomes=False))
    return (batch_lock, audit_lock, parent_lock)


def experiment_hashes(manifest, algorithm):
    lazy = ("quality_staged", "quality_lazy_01", "quality_lazy_class_01")
    if algorithm not in (*ALGORITHMS, *lazy, *PREFIX_DEPTHS):
        raise ValueError(algorithm)
    names = ["benchmark_simplification.py", "trace_ablation_variants.py"]
    if "trace_execution_audit.py" in manifest["tool_hashes"]:
        names.append("trace_execution_audit.py")
    if algorithm != "standard":
        names.append("trace_algorithm_variants.py")
    if algorithm in lazy:
        names.append("trace_lazy_quality.py")
    if algorithm in PREFIX_DEPTHS:
        names.append("trace_cached_prefix.py")
    return {name: manifest["tool_hashes"][name] for name in names}


def worker_command(runtime, manifest, cell, directory):
    command = [sys.executable, str(runtime / "tools/benchmark_simplification.py"),
        "--package-root", str(runtime), "--data-root", manifest["data_root"],
        "--dataset", cell["dataset"], "--variant", "full", "--algorithm", cell["algorithm"],
        "--split", cell.get("split", "evaluation"), "--seed", "0", "--selection-seed", str(SELECTION_SEED),
        "--name", cell["key"], "--output", str(directory)]
    if cell["limit"] is not None:
        command += ["--max-images", str(cell["limit"])]
    if cell["dataset"] in manifest["data_dirs"]:
        command += ["--data-dir", manifest["data_dirs"][cell["dataset"]]]
    return command


def numeric(value, name, *, integer=False, positive=False):
    number = float(value)
    if (not math.isfinite(number) or number < 0 or (positive and number <= 0)
            or (integer and int(number) != number)):
        raise RuntimeError(f"Invalid cost/metric field: {name}")
    return int(number) if integer else number


def check_close(actual, expected, name):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError(f"Saved cost disagrees with cost.csv: {name}")


def validate_costs(directory, result, expected, algorithm, objective_epsilon):
    with (directory / "cost.csv").open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise RuntimeError("Empty cost.csv")
    diagnostics = {}
    diagnostic_path = directory / "candidate_diagnostics.csv"
    if diagnostic_path.exists():
        with diagnostic_path.open(newline="") as handle:
            for candidate in csv.DictReader(handle):
                episode = numeric(candidate["episode"], "diagnostic episode", integer=True)
                if episode >= len(raw_rows):
                    raise RuntimeError("Candidate diagnostic belongs to an unknown episode")
                diagnostics.setdefault(episode, []).append(candidate)
    integer_fields = ("episode", "images", "update_steps", "candidates", "diagnostic_views",
                      "fully_traced_candidates", "effective_view_batch_size", "calls", "inputs")
    float_fields = ("adaptation_seconds", "inference_seconds", "total_seconds", "peak_allocated_mib",
                    "omitted_mass_bound", "omitted_upper_mass")
    rows = []
    for index, raw in enumerate(raw_rows):
        row = {key: numeric(raw[key], key, integer=True) for key in integer_fields}
        row.update({key: numeric(raw[key], key) for key in float_fields})
        batch_size = expected["config"]["batch_size"]
        episode_images = expected["image_ids"][index * batch_size:(index + 1) * batch_size]
        if row["episode"] != index or row["images"] != len(episode_images) or not episode_images:
            raise RuntimeError("Cost episodes or batch population are inconsistent")
        candidates = diagnostics.get(index, [])
        if len(candidates) != row["candidates"]:
            raise RuntimeError("Candidate diagnostics do not audit the complete episode")
        reliabilities = []
        for candidate in candidates:
            rho = numeric(candidate["reliability"], "candidate reliability")
            if rho > 1 or candidate["image_id"] not in episode_images:
                raise RuntimeError("Invalid candidate reliability or image identity")
            reliabilities.append(rho)
        required_steps = expected["config"]["steps"] if any(r > objective_epsilon for r in reliabilities) else 0
        if row["update_steps"] != required_steps:
            raise RuntimeError("Cost update steps differ from the fixed configuration or audited empty objective")
        if (row["calls"] <= 0 or row["inputs"] < row["calls"]
                or row["diagnostic_views"] > row["inputs"]
                or row["fully_traced_candidates"] > row["candidates"]
                or row["omitted_mass_bound"] > 1
                or row["total_seconds"] <= 0 or row["peak_allocated_mib"] <= 0):
            raise RuntimeError("Invalid observed visual cost or memory")
        count, traced = row["candidates"], row["fully_traced_candidates"]
        if algorithm in {"standard", "staged_full"}:
            views, required_traced, bound = 3 * count, count, 0.0
        elif algorithm == "quality_only" or algorithm in PREFIX_DEPTHS:
            views, required_traced, bound = 2 * count, 0, 0.0
            if algorithm in PREFIX_DEPTHS:
                validate_lazy_candidates(candidates, row, "quality_staged", objective_epsilon)
        elif algorithm in {"selective_01", "selective_05"}:
            views, required_traced = count + 2 * traced, traced
            bound = {"selective_01": .01, "selective_05": .05}[algorithm]
        elif algorithm in {"quality_staged", "quality_lazy_01", "quality_lazy_class_01"}:
            measured_cf = sum(c["counterfactual_support"] != "" for c in candidates)
            views, required_traced = count + measured_cf, 0
            bound = 0. if algorithm == "quality_staged" else .01
            validate_lazy_candidates(candidates, row, algorithm, objective_epsilon)
        else:
            raise ValueError(algorithm)
        if row["diagnostic_views"] != views or traced != required_traced:
            raise RuntimeError("Observed diagnostic views disagree with the algorithm")
        if row["omitted_mass_bound"] > bound + 1e-12:
            raise RuntimeError("Observed omitted mass bound exceeds the fixed algorithm tolerance")
        if bound == 0 and row["omitted_upper_mass"] != 0:
            raise RuntimeError("Nonselective algorithm reports omitted mass")
        if algorithm != "standard":
            if any(c.get("fully_traced") not in {"True", "False"} for c in candidates):
                raise RuntimeError("Missing fully-traced candidate audit")
            if sum(c["fully_traced"] == "True" for c in candidates) != traced:
                raise RuntimeError("Fully-traced count disagrees with candidate diagnostics")
        check_close(row["total_seconds"], row["adaptation_seconds"] + row["inference_seconds"], "total_seconds")
        if algorithm in PREFIX_DEPTHS or "block_inputs_by_layer" in raw:
            row.update(validate_block_costs(raw, row, algorithm))
        rows.append(row)
    if sum(row["images"] for row in rows) != result["images"]:
        raise RuntimeError("Cost population differs from per-image evidence")
    warmup = min(2, len(rows) - 1)
    if result["warmup_episodes"] != warmup:
        raise RuntimeError("Warmup exclusion differs from the benchmark protocol")
    measured = rows[warmup:]
    count = sum(row["images"] for row in measured)
    summary = dict(
        measured_images=count, warmup_episodes=warmup,
        ms_per_image=1000 * sum(row["total_seconds"] for row in measured) / count,
        peak_allocated_mib=max(row["peak_allocated_mib"] for row in measured),
        diagnostic_views_per_image=sum(row["diagnostic_views"] for row in measured) / count,
        visual_inputs_per_image=sum(row["inputs"] for row in measured) / count,
        visual_calls_per_image=sum(row["calls"] for row in measured) / count,
        all_images=result["images"],
        all_diagnostic_views=sum(row["diagnostic_views"] for row in rows),
        all_visual_inputs=sum(row["inputs"] for row in rows),
        all_visual_calls=sum(row["calls"] for row in rows),
        all_peak_allocated_mib=max(row["peak_allocated_mib"] for row in rows),
        all_fully_traced_candidates=sum(row["fully_traced_candidates"] for row in rows),
        max_omitted_mass_bound=max(row["omitted_mass_bound"] for row in rows),
        total_omitted_upper_mass=sum(row["omitted_upper_mass"] for row in rows))
    for key in ("ms_per_image", "peak_allocated_mib", "diagnostic_views_per_image"):
        check_close(result[key], summary[key], key)
    if "block_inputs" in rows[0]:
        summary.update(block_inputs_per_image=sum(r["block_inputs"] for r in measured) / count,
                       block_calls_per_image=sum(r["block_calls"] for r in measured) / count,
                       all_block_inputs=sum(r["block_inputs"] for r in rows),
                       all_failed_forward_attempts=sum(r["failed_forward_attempts"] for r in rows),
                       all_failed_visual_inputs=sum(r["failed_inputs"] for r in rows),
                       encoder_blocks=rows[0]["encoder_blocks"])
        for key in ("block_inputs_per_image", "block_calls_per_image", "encoder_blocks"):
            check_close(result[key], summary[key], key)
    return summary, raw_rows


def validate_block_costs(raw, row, algorithm):
    """Count full and cached-tail forwards independently; no nominal FLOP claims."""
    depth = numeric(raw["encoder_blocks"], "encoder blocks", integer=True, positive=True)
    patches = numeric(raw["patches"], "patches", integer=True, positive=True)
    vectors, failed = {}, {}
    for key in ("block_inputs", "block_calls"):
        vector = json.loads(raw[key + "_by_layer"])
        if not isinstance(vector, list) or len(vector) != depth:
            raise RuntimeError("Block cost vector has the wrong depth")
        values = [numeric(x, key, integer=True) for x in vector]
        if sum(values) != numeric(raw[key], key, integer=True):
            raise RuntimeError("Block cost sum disagrees with measured layer counters")
        vectors[key] = values
        failed_values = json.loads(raw.get("failed_" + key + "_by_layer", json.dumps([0] * depth)))
        if not isinstance(failed_values, list) or len(failed_values) != depth:
            raise RuntimeError("Failed block cost vector has the wrong depth")
        failed[key] = [numeric(x, "failed " + key, integer=True) for x in failed_values]
        if (sum(failed[key]) != numeric(raw.get("failed_" + key, 0), "failed " + key, integer=True)
                or any(f > v for f, v in zip(failed[key], values))):
            raise RuntimeError("Failed block work is inconsistent with total work")
    if any(i < c for i, c in zip(vectors["block_inputs"], vectors["block_calls"])):
        raise RuntimeError("More block calls than block image inputs")
    failed_calls = numeric(raw.get("failed_calls", 0), "failed stem calls", integer=True)
    failed_inputs = numeric(raw.get("failed_inputs", 0), "failed stem inputs", integer=True)
    failed_attempts = numeric(raw.get("failed_forward_attempts", 0), "failed forwards", integer=True)
    if (failed_calls > row["calls"] or failed_inputs > row["inputs"] or failed_inputs < failed_calls
            or failed_calls > failed_attempts
            or any(i < c for i, c in zip(failed["block_inputs"], failed["block_calls"]))
            or any(v > failed_inputs for v in failed["block_inputs"])
            or any(v > failed_calls for v in failed["block_calls"])):
        raise RuntimeError("Invalid failed-forward work accounting")
    for key, first in (("block_inputs", failed_inputs), ("block_calls", failed_calls)):
        values = [first, *failed[key]]
        if any(b > a for a, b in zip(values, values[1:])):
            raise RuntimeError("Failed diagnostic work cannot grow in later blocks")
    successful = {key: [v - f for v, f in zip(vectors[key], failed[key])] for key in vectors}
    stem_inputs, stem_calls = row["inputs"] - failed_inputs, row["calls"] - failed_calls
    for key, values in successful.items():
        if "successful_" + key in raw:
            if (numeric(raw["successful_" + key], "successful " + key, integer=True) != sum(values)
                    or json.loads(raw["successful_" + key + "_by_layer"]) != values):
                raise RuntimeError("Successful block counters differ from total minus failed")
    for key, value in (("inputs", stem_inputs), ("calls", stem_calls)):
        if "successful_" + key in raw and numeric(raw["successful_" + key], key, integer=True) != value:
            raise RuntimeError("Successful stem counter differs from total minus failed")
    diagnostic_calls = numeric(raw["diagnostic_calls"], "successful diagnostic calls", integer=True)
    if (row["diagnostic_views"] == 0) != (diagnostic_calls == 0) or diagnostic_calls > row["diagnostic_views"]:
        raise RuntimeError("Diagnostic calls cannot explain the successful views")
    full_inputs = row["diagnostic_views"] + patches * (row["update_steps"] + 2)
    full_calls = diagnostic_calls + row["update_steps"] + 2
    tail = PREFIX_DEPTHS.get(algorithm, depth)
    cached = algorithm in PREFIX_DEPTHS and algorithm != "quality_tail6_uncached"
    if cached:
        prefix = depth - tail
        if prefix <= 0:
            raise RuntimeError("Cached prefix must contain at least one block")
        # A cache can be released before final evaluation, requiring one fresh prefix.
        allowed = {row["diagnostic_views"] + patches, row["diagnostic_views"] + 2 * patches}
        if stem_inputs not in allowed or any(v != stem_inputs for v in successful["block_inputs"][:prefix]):
            raise RuntimeError("Frozen prefix work does not match measured stem executions")
        original_prefix_calls = (stem_inputs - row["diagnostic_views"]) // patches
        if (stem_calls != diagnostic_calls + original_prefix_calls
                or any(v != stem_calls for v in successful["block_calls"][:prefix])):
            raise RuntimeError("Frozen prefix calls differ from successful stem executions")
        if (any(v != full_inputs for v in successful["block_inputs"][prefix:])
                or any(v != full_calls for v in successful["block_calls"][prefix:])):
            raise RuntimeError("Cached tail did not execute every configured update")
    elif (stem_inputs != full_inputs or stem_calls != full_calls
          or any(v != full_inputs for v in successful["block_inputs"])
          or any(v != full_calls for v in successful["block_calls"])):
        raise RuntimeError("Full encoder work differs from diagnostic plus all configured updates")
    return dict(block_inputs=sum(vectors["block_inputs"]), block_calls=sum(vectors["block_calls"]),
                encoder_blocks=depth, patches=patches, failed_forward_attempts=failed_attempts,
                failed_inputs=failed_inputs, failed_calls=failed_calls)


def validate_parameter_record(record, expected_names):
    names = set(expected_names)
    for key in ("names", "actual_names", "optimizer_names"):
        values = record[key]
        if not isinstance(values, list) or len(values) != len(names) or set(values) != names:
            raise RuntimeError("Declared, actual and optimizer parameter names differ")
    counts = record["parameter_numel"]
    if set(counts) != names:
        raise RuntimeError("Parameter size inventory differs from trained names")
    total = sum(numeric(n, "parameter numel", integer=True, positive=True) for n in counts.values())
    if total != numeric(record["trainable_parameters"], "trainable parameters", integer=True, positive=True):
        raise RuntimeError("Actual trainable parameter total is inconsistent")


def validate_lazy_candidates(candidates, cost, algorithm, eps):
    """Independently reconstruct the missing-probe certificate from saved scores."""
    groups = {}
    for candidate in candidates:
        target = numeric(candidate["target_support"], "target support")
        rho = numeric(candidate["reliability"], "quality reliability")
        if target > 1 or rho > target or candidate["residual_support"] != "" or candidate["dominance"] != "":
            raise RuntimeError("Invalid two-view signal or invented residual measurement")
        group = groups.setdefault((candidate["image_id"], candidate["class_idx"]), [[], []])
        if candidate["counterfactual_support"] == "":
            if (candidate["omitted"] != "True" or candidate["quality"] != "" or rho != 0
                    or algorithm == "quality_staged"):
                raise RuntimeError("Missing probe values must remain missing with zero update weight")
            if target > eps:
                group[1].append(target)
        else:
            cf = numeric(candidate["counterfactual_support"], "counterfactual support")
            if cf > 1 or candidate["omitted"] != "False":
                raise RuntimeError("Invalid measured counterfactual support")
            check_close(rho, min(target, cf), "lazy quality")
            check_close(candidate["quality"], rho, "lazy saved quality")
            if rho > eps:
                group[0].append(rho)
    observed = math.fsum(x for a, _ in groups.values() for x in a)
    unseen = math.fsum(x for _, u in groups.values() for x in u)
    bound = unseen / (observed + unseen) if observed + unseen else 0.
    check_close(cost["omitted_mass_bound"], bound, "omitted mass certificate")
    check_close(cost["omitted_upper_mass"], unseen, "omitted upper mass")
    if algorithm == "quality_lazy_class_01":
        for a, u in groups.values():
            a, u = math.fsum(a), math.fsum(u)
            if a + u and u / (a + u) > .01 + 1e-12:
                raise RuntimeError("Per-image/query-class omitted mass exceeds 1 percent")


def artifact_hashes(directory):
    names = ["result.json", "protocol.json", "per_image_counts.json", "cost.csv"]
    if (directory / "candidate_diagnostics.csv").exists():
        names.append("candidate_diagnostics.csv")
    if (directory / "adaptation_parameters.json").exists():
        names.append("adaptation_parameters.json")
    return {name: digest(directory / name) for name in names}


def validate_algorithm_artifact(directory, expected, manifest, algorithm):
    protocol = json.loads((directory / "protocol.json").read_text())
    if protocol.get("algorithm") != algorithm:
        raise RuntimeError("Algorithm identity differs from the frozen cell")
    if protocol.get("configuration_overrides") != {} or protocol.get("force_uniform_policy") is not False:
        raise RuntimeError("Fixed-config algorithm cells cannot override parameters or policy")
    if protocol.get("timing_scope") != TIMING_SCOPE:
        raise RuntimeError("Unknown synchronized timing scope")
    result = validate_artifact(directory, expected, manifest["source_hashes"], "full",
                               experiment_hashes(manifest, algorithm), manifest["environment"])
    for key, value in (("algorithm", algorithm), ("variant", "full"),
                       ("dataset", expected["dataset"]), ("split", expected["split"])):
        if result.get(key) != value:
            raise RuntimeError(f"Result identity mismatch: {key}")
    for row in json.loads((directory / "per_image_counts.json").read_text()):
        intersection, union = np.asarray(row["intersection"]), np.asarray(row["union"])
        if (intersection.ndim != 1 or intersection.shape != union.shape
                or not np.isfinite(intersection).all() or not np.isfinite(union).all()
                or (intersection < 0).any() or (union < intersection).any()
                or (intersection != np.floor(intersection)).any() or (union != np.floor(union)).any()):
            raise RuntimeError("Invalid per-image segmentation counts")
        prediction_hash = row.get("prediction_sha256", "")
        if len(prediction_hash) != 64 or any(c not in "0123456789abcdef" for c in prediction_hash):
            raise RuntimeError("Missing per-image prediction hash")
    for key in ("mIoU", "mAcc"):
        value = numeric(result["metrics"][key], key)
        if value > 100:
            raise RuntimeError("Invalid aggregate metric")
    costs, episodes = validate_costs(directory, result, expected, algorithm, manifest["objective_epsilon"])
    if algorithm in PREFIX_DEPTHS:
        parameters = json.loads((directory / "adaptation_parameters.json").read_text())
        depth = costs["encoder_blocks"]
        names = {f"visual.ln_post.{p}" for p in ("weight", "bias")}
        for i in range(depth - PREFIX_DEPTHS[algorithm], depth):
            norms = ("ln_1", "ln_2") if i < depth - 1 else ("ln_1",)
            names.update(f"visual.transformer.resblocks.{i}.{norm}.{p}"
                         for norm in norms for p in ("weight", "bias"))
        if (parameters["algorithm"] != algorithm or parameters["encoder_blocks"] != depth
                or len(parameters["names"]) != len(names) or set(parameters["names"]) != names):
            raise RuntimeError("Adapted LayerNorm scope differs from declared frozen prefix")
        validate_parameter_record(parameters, names)
    return result, costs, episodes


def run_cell(output, runtime, manifest, cell, locks, *, allow_execution=True):
    root = output / cell["key"]
    root.mkdir(parents=True, exist_ok=True)
    expected = manifest["input_panels"][cell["panel"]]
    selection = root / "selected_result.json"
    attempts = sorted(root.glob("attempt_*"))
    if len(attempts) > 1:
        raise RuntimeError("Multiple attempts require explicit inspection; no outcome-based selection")
    directory = completed_attempt(root)
    if directory is None:
        if attempts or selection.exists() or not allow_execution:
            raise RuntimeError(f"Unfinished attempt preserved; no automatic retry: {root}")
        directory = next_attempt(root)
        directory.mkdir()
        command = worker_command(runtime, manifest, cell, directory)
        write_json(output / "queue_state.json", dict(status="running", pid=os.getpid(),
                   current=cell, current_output=str(directory), command=command))
        with (directory / "stdout.log").open("w") as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                pass_fds=tuple(lock.fileno() for lock in locks),
                env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        if process.returncode:
            raise RuntimeError(f"Algorithm worker failed ({process.returncode}); evidence retained: {directory}")
    if selection.exists():
        selected = json.loads(selection.read_text())
        if (selected["result_path"] != str(directory / "result.json")
                or selected["artifact_sha256"] != artifact_hashes(directory)
                or selected["checks_sha256"] != digest(directory / "checks.json")):
            raise RuntimeError("Selected evidence or cost seal changed")
    result, costs, episodes = validate_algorithm_artifact(directory, expected, manifest, cell["algorithm"])
    checks = dict(algorithm=cell["algorithm"], config_fixed=True, source_verified=True,
                  environment_verified=True, data_cohort_verified=True, per_image_metrics_verified=True,
                  synchronized_costs_verified=True, artifact_sha256=artifact_hashes(directory))
    freeze_protocol(directory / "checks.json", checks)
    freeze_protocol(selection, dict(result_path=str(directory / "result.json"),
        result_sha256=digest(directory / "result.json"), artifact_sha256=checks["artifact_sha256"],
        checks_sha256=digest(directory / "checks.json")))
    row = dict(stage=cell["stage"], dataset=cell["dataset"], algorithm=cell["algorithm"],
               variant="full", images=result["images"], full_split=result["full_split"],
               config_fixed=True, mIoU=result["miou_unrounded"], mAcc=result["metrics"]["mAcc"],
               **costs, result_path=str(directory / "result.json"), cost_path=str(directory / "cost.csv"))
    return row, [dict(stage=cell["stage"], dataset=cell["dataset"], algorithm=cell["algorithm"],
                      cost_path=str(directory / "cost.csv"), **episode) for episode in episodes]


def save_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix(".tmp")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_tables(output, rows, episodes):
    references = {(row["stage"], row["dataset"], row["algorithm"]): row for row in rows}
    comparisons = []
    for row in rows:
        compared = dict(row)
        for algorithm in ("standard", "staged_full"):
            reference = references.get((row["stage"], row["dataset"], algorithm))
            compared[f"delta_mIoU_vs_{algorithm}"] = row["mIoU"] - reference["mIoU"] if reference else None
        comparisons.append(compared)
    save_csv(output / "summary.csv", comparisons)
    save_csv(output / "algorithm_summary.csv", [r for r in comparisons if r["stage"] == "evaluation"])
    save_csv(output / "all_costs.csv", episodes)


def verify_execution_inputs(runtime, manifest):
    verify_snapshot(runtime, manifest)
    verify_dependency_protocols(manifest)
    if manifest["environment"] != execution_environment():
        raise RuntimeError("Execution environment changed since preparation")
    if manifest["input_panels"] != discover_panels(runtime, manifest):
        raise RuntimeError("Frozen data cohort or configuration changed")


def verify_suite_seal(output):
    path = output / "SUITE_SEALED.json"
    if not path.exists():
        return False
    seal = json.loads(path.read_text())
    for name, expected in seal["artifact_sha256"].items():
        artifact = output / name
        if not artifact.is_file() or digest(artifact) != expected:
            raise RuntimeError(f"Completed suite evidence changed: {name}")
    return True


def execute_suite(output, manifest, own_lock, stack):
    runtime = output / "runtime"
    sealed = verify_suite_seal(output)
    locks = (own_lock, *wait_for_dependencies(output, manifest, stack))
    verify_execution_inputs(runtime, manifest)
    rows, episodes = [], []
    for cell in manifest["cells"]:
        verify_snapshot(runtime, manifest)
        row, costs = run_cell(output, runtime, manifest, cell, locks, allow_execution=not sealed)
        rows.append(row)
        episodes.extend(costs)
        save_tables(output, rows, episodes)
        print(json.dumps(dict(completed=len(rows), cells_total=len(manifest["cells"]), **row)), flush=True)
    verify_execution_inputs(runtime, manifest)
    # Revalidate every accepted cell before sealing the suite, including early smoke evidence.
    for cell in manifest["cells"]:
        run_cell(output, runtime, manifest, cell, locks, allow_execution=False)
    names = ["suite_protocol.json", "requested_protocol.json", "fixed_configs.json", "dependency_checks.json",
             "summary.csv", "algorithm_summary.csv", "all_costs.csv"]
    names += [f'{cell["key"]}/selected_result.json' for cell in manifest["cells"]]
    freeze_protocol(output / "SUITE_SEALED.json", dict(
        artifact_sha256={name: digest(output / name) for name in names},
        smoke_cells=5, evaluation_cells=20, config_fixed=True, automatic_promotion=False))
    write_json(output / "queue_state.json", dict(status="completed", completed=len(rows),
               smoke_cells=5, evaluation_cells=20, cells_total=25, finished_at=utc_now()))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PATH_ARGUMENTS:
        parser.add_argument("--" + name.replace("_", "-"), type=lambda value: Path(value).resolve(), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--detach", action="store_true")
    parser.add_argument("--execute-prepared", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--queue-lock-fd", type=int, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def launch_controller(args, lock):
    command = [sys.executable, str(args.output / "runtime/tools" / CONTROLLER),
               "--execute-prepared", "--queue-lock-fd", str(lock.fileno())]
    for name in PATH_ARGUMENTS:
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    options = dict(pass_fds=(lock.fileno(),), cwd=str(args.output / "runtime"))
    if args.detach:
        with (args.output / "queue.log").open("a") as handle:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True, **options)
        write_json(args.output / "launch.json", dict(pid=child.pid, command=command, launched_at=utc_now()))
        print(f"Detached frozen controller PID {child.pid}; {args.output / 'queue.log'}", flush=True)
    else:
        completed = subprocess.run(command, **options)
        if completed.returncode:
            raise RuntimeError(f"Frozen controller failed ({completed.returncode}); evidence retained")


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if args.execute_prepared:
            if (Path(__file__).resolve() != args.output / "runtime/tools" / CONTROLLER
                    or args.queue_lock_fd is None or args.prepare_only or args.detach):
                raise RuntimeError("Prepared execution requires the frozen controller and inherited lock")
            own_lock = stack.enter_context(os.fdopen(args.queue_lock_fd, "a"))
            if not os.path.samestat(os.fstat(own_lock.fileno()), (args.output / "queue.lock").stat()):
                raise RuntimeError("Inherited queue lock does not belong to this output")
            fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            own_lock = acquire_lock(stack, args.output / "queue.lock", nonblocking=True)
        state_path = args.output / "queue_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get("status", "").startswith("failed"):
            raise RuntimeError("Failed queue preserved; inspect it and use a fresh output, no automatic retry")
        try:
            if state.get("status") == "completed" and not verify_suite_seal(args.output):
                raise RuntimeError("Completed queue has no suite seal; preserve evidence for inspection")
            if args.execute_prepared:
                manifest = json.loads((args.output / "suite_protocol.json").read_text())
                for name in PATH_ARGUMENTS:
                    if manifest[name] != str(getattr(args, name)):
                        raise RuntimeError(f"Prepared request path changed: {name}")
                execute_suite(args.output, manifest, own_lock, stack)
            else:
                prepare_suite(args)
                if not state_path.exists():
                    write_json(state_path, dict(status="prepared", smoke_cells=5, evaluation_cells=20, cells_total=25))
                if not args.prepare_only:
                    launch_controller(args, own_lock)
        except BaseException as error:
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            write_json(state_path, {**state, "status": "failed", "error": type(error).__name__, "message": str(error)})
            raise


if __name__ == "__main__":
    main()
