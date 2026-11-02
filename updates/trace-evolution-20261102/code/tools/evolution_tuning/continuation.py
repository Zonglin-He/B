"""New immutable Waterbirds search and eight-dataset full stage, after seven searches."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from tools.evolution_tuning import queue
from tools.evolution_tuning.common import (
    ALGORITHM, COORDINATES, DATASETS, FULL_ARMS, MAX_TRIALS, ORDER, ROOT,
    canonical_hash, choose, read_json, require, sha256, write_json,
)

PARENT_DATASETS = tuple(d for d in ORDER if d != "waterbirds_seg")
PANEL_KEYS = ("inputs", "ordered_ids", "classes", "queries", "query_to_class",
              "reduce_zero_label", "full_split", "images", "input_identity_sha256")


def panel_identity(panel):
    return {key: panel[key] for key in PANEL_KEYS}


def unchanged_method(parent, current):
    fixed = {name: digest for name, digest in parent["runtime_hashes"].items()
             if name.startswith("trace_ovss/") or (name.startswith("tools/") and name.endswith(".py")
             and name != "tools/evolution_tuning/inputs.py")}
    require(fixed and all(current["runtime_hashes"].get(name) == digest for name, digest in fixed.items()),
            "Parent method/worker/search executable differs")
    require(parent["checkpoint"] == current["checkpoint"]
            and parent["initial_configs"] == current["initial_configs"], "Parent checkpoint/settings differ")


def normalize_paths(args):
    for name in ("parent_suite", "python", "source_root", "data_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve(strict=True))
    args.output = args.output.expanduser().absolute()
    return args


def prepare(args):
    from tools.evolution_tuning.inputs import prepare_inputs
    from tools.evolution.run_cell import checkpoint_identity

    args = normalize_paths(args)
    parent = queue.verify(args.parent_suite, args.parent_sha256)
    require(parent["runnable"] == list(PARENT_DATASETS) and set(parent["blocked"]) == {"waterbirds_seg"},
            "Expected the frozen seven-search upstream suite")
    suite = args.output.absolute()
    suite.mkdir(parents=True, exist_ok=False)
    require(suite == suite.resolve(), "Suite aliases rejected")
    queue.state(suite, "preparing")
    bundle = prepare_inputs(args.source_root, args.data_root, suite, calibration_size=256, seed=2027,
                            allow_waterbirds_development_overlap=True)
    entries = bundle["datasets"]
    require(set(entries) == set(DATASETS) and all(e["calibration"] for e in entries.values()), "Eight calibrations required")
    original = read_json(args.parent_suite / "inputs.json")["datasets"]
    for dataset in DATASETS:
        require(panel_identity(entries[dataset]["evaluation"]) == panel_identity(original[dataset]["evaluation"]),
                "Original full population/protocol changed: " + dataset)
        if dataset != "waterbirds_seg":
            require(panel_identity(entries[dataset]["calibration"]) == panel_identity(original[dataset]["calibration"]),
                    "Parent calibration population/protocol changed: " + dataset)
    runtime = queue.snapshot(args.source_root, suite / "runtime")
    shutil.copy2(args.source_root / "docs/single-probe-tuning-contract.md", suite / "frozen_contract.md")
    mapping = queue.rebase_panel_sources(entries, args.source_root, suite, runtime)
    write_json(suite / "inputs.json", dict(datasets=entries, frozen_source_path_mapping=mapping,
        preparation_metadata={k: v for k, v in bundle.items() if k != "datasets"}))
    prepared = dict(schema_version=1, algorithm=ALGORITHM, method_changes=False,
        datasets=list(DATASETS), order=list(ORDER), runnable=list(ORDER), blocked={},
        calibration_images=256, calibration_seed=2027, model_seed=0,
        max_unique_trials_per_dataset=MAX_TRIALS, coordinates=list(COORDINATES), full_arms=FULL_ARMS,
        python=str(args.python), runtime_hashes=runtime, inputs_sha256=sha256(suite / "inputs.json"),
        initial_configs={d: entries[d]["config"] for d in DATASETS},
        contract_sha256=sha256(suite / "frozen_contract.md"), checkpoint=checkpoint_identity(),
        retry=False, created_at=queue.now(),
        selection="Reuse seven sealed calibration selections; only Waterbirds searches the fixed grid.",
        full_label_barrier="All eight configurations and full predictions seal before full scoring.",
        historical_exposure="Development-exposed protocols; Waterbirds calibration is a subset of full.",
        continuation=dict(parent_suite=str(args.parent_suite), parent_prepared_sha256=args.parent_sha256,
            parent_service=args.parent_service, reused_datasets=list(PARENT_DATASETS),
            searched_datasets=["waterbirds_seg"], waterbirds_evaluation_scope="development_overlap",
            authorization="User explicitly chose existing train tuning and disclosure of overlap on 2026-10-29."))
    unchanged_method(parent, prepared)
    write_json(suite / "SUITE_PREPARED.json", prepared)
    queue.state(suite, "prepared_waiting_for_parent", parent_suite=str(args.parent_suite))
    print(json.dumps(dict(suite=str(suite), prepared_sha256=sha256(suite / "SUITE_PREPARED.json")), indent=2))


def service_status(name):
    require(name.endswith(".service") and "/" not in name and not name.startswith("-"), "Unsafe parent service")
    proc = subprocess.run(["systemctl", "--user", "show", name, "-p", "LoadState", "-p", "ActiveState",
        "-p", "SubState", "-p", "MainPID", "-p", "Result", "-p", "ExecMainStatus"],
        capture_output=True, text=True, timeout=30)
    require(proc.returncode == 0, "Parent service status unavailable")
    return dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)


def parent_ready(status, state):
    require(status.get("LoadState") == "loaded", "Parent service identity lost")
    require(state.get("phase") != "failed" and status.get("ActiveState") != "failed", "Parent failed; no recovery")
    if status.get("SubState") == "running" and status.get("ActiveState") == "active":
        require(int(status.get("MainPID", "0")) > 0, "Running parent without process")
        return False
    require(status.get("ActiveState") == "active" and status.get("SubState") == "exited"
            and status.get("MainPID") == "0" and status.get("Result") == "success"
            and status.get("ExecMainStatus") == "0", "Parent did not complete successfully")
    require(state.get("phase") == "awaiting_all_eight_calibrations", "Unexpected parent terminal phase")
    return True


def wait_for_parent(suite, prepared):
    dependency = prepared["continuation"]
    parent = Path(dependency["parent_suite"])
    while True:
        status = service_status(dependency["parent_service"])
        state = read_json(parent / "queue_state.json")
        if parent_ready(status, state):
            return status
        queue.state(suite, "waiting_for_seven_calibrations", parent_suite=str(parent),
                    parent_phase=state["phase"], gpu_work_started=False)
        time.sleep(120)


def read_parent_selections(suite, prepared, entries, status):
    dependency = prepared["continuation"]
    parent = Path(dependency["parent_suite"])
    upstream = queue.verify(parent, dependency["parent_prepared_sha256"])
    unchanged_method(upstream, prepared)
    terminal = read_json(parent / "CALIBRATION_SELECTIONS_SEALED.json")
    require(terminal["complete"] is False and terminal["full_evaluation_ready"] is False
            and terminal["suite_prepared_sha256"] == dependency["parent_prepared_sha256"]
            and set(terminal["blocked"]) == {"waterbirds_seg"}
            and set(terminal["selections"]) == set(PARENT_DATASETS), "Incomplete upstream selection seal")
    require(not (parent / "ALL_CONFIGS_SELECTED.json").exists()
            and not (parent / "ALL_FULL_PREDICTIONS_SEALED.json").exists(), "Unexpected parent full stage")
    gates = read_json(parent / "ALL_GATES_PASSED.json")
    require(gates["complete"] is True and set(gates["gates"]) == set(PARENT_DATASETS), "Incomplete upstream gates")
    for dataset, expected in gates["gates"].items():
        require(sha256(parent / "gates" / dataset / "GATE_PASSED.json") == expected, "Upstream gate drift")
    with (parent / "trials.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(rows and set(r["dataset"] for r in rows) == set(PARENT_DATASETS), "Incomplete upstream trials")
    selected = {}
    old_entries = read_json(parent / "inputs.json")["datasets"]
    for dataset in PARENT_DATASETS:
        path = parent / "selections" / dataset / "SELECTED_CONFIG_SEALED.json"
        require(sha256(path) == terminal["selections"][dataset], "Upstream selected config drift")
        original = read_json(path)
        require(original["dataset"] == dataset and original["calibration_only"] is True
                and original["full_evaluation_used_for_selection"] is False
                and original["suite_prepared_sha256"] == dependency["parent_prepared_sha256"], "Upstream selection scope")
        dataset_rows = [r for r in rows if r["dataset"] == dataset]
        require(len(dataset_rows) == original["unique_trials"] <= MAX_TRIALS, "Upstream trial count mismatch")
        candidates, keys = [], set()
        inventory = {r["key"]: r for r in original["all_trials"]}
        require(len(inventory) == len(dataset_rows), "Upstream trial inventory differs")
        for row in dataset_rows:
            cfg = json.loads(row["config"])
            key = queue.config_key(cfg)
            require(key not in keys and key in inventory, "Duplicate/unknown upstream trial")
            keys.add(key)
            require(inventory[key]["status"] == row["status"]
                    and inventory[key]["result_path"] == row["result_path"], "Upstream trial binding differs")
            output = Path(row["result_path"])
            output = output.parent if row["status"] == "completed" else output
            require(output == parent / "trials" / dataset / f"{int(row['trial']):03d}_{key}", "Upstream trial path differs")
            manifest_path = output.parent / (output.name + ".manifest.json")
            manifest = read_json(manifest_path)
            require(manifest["config"] == cfg and manifest["dataset"] == dataset
                    and manifest["stage"] == "calibration" and manifest["suite_root"] == str(parent)
                    and manifest["suite_prepared_sha256"] == dependency["parent_prepared_sha256"]
                    and manifest["source_sha256"] == upstream["runtime_hashes"]
                    and manifest["panel"] == old_entries[dataset]["calibration"], "Upstream trial manifest differs")
            if row["status"] == "ineligible":
                queue.ineligible_failure(output, manifest_path)
                continue
            require(row["status"] == "completed", "Unknown upstream trial status")
            result = queue.validate_result(output, dataset, old_entries[dataset]["calibration"], ["image_C"], full=False)
            require(result["config"] == cfg and result["manifest_sha256"] == sha256(manifest_path)
                    and sha256(output / "SCORED_SEALED.json") == row["scored_seal_sha256"]
                    and result["prediction_seal_sha256"] == sha256(output / "PREDICTIONS_SEALED.json"), "Upstream result evidence differs")
            score = result["arm_results"]["image_C"]["miou_unrounded"]
            duration = 1000 * result["timings"]["image_C"]["seconds_per_image"]
            require(score == float(row["mIoU"]) and duration == float(row["ms_per_image"]), "Upstream CSV metric differs")
            candidates.append(dict(status="completed", config=cfg, mIoU=score, ms_per_image=duration, trial=int(row["trial"])))
        winner = choose(candidates)
        require(winner["config"] == original["config"] and winner["trial"] == original["selected_trial"]["trial"],
                "Upstream winner differs from fixed calibration rule")
        record = copy.deepcopy(original)
        record.update(suite_prepared_sha256=prepared["_verified_sha256"],
            calibration_panel_sha256=canonical_hash(entries[dataset]["calibration"]),
            selection_reused_without_search=True, origin=dict(source_suite=str(parent),
                source_prepared_sha256=dependency["parent_prepared_sha256"], selection_path=str(path),
                selection_sha256=sha256(path), original_panel_sha256=original["calibration_panel_sha256"],
                trial_csv_sha256=sha256(parent / "trials.csv")))
        directory = suite / "selections" / dataset
        directory.mkdir(parents=True)
        write_json(directory / "SELECTED_CONFIG_SEALED.json", record)
        selected[dataset] = record
    write_json(suite / "PARENT_IMPORT_SEALED.json", dict(complete=True, parent_service_status=status,
        parent_prepared_sha256=dependency["parent_prepared_sha256"],
        parent_selection_seal_sha256=sha256(parent / "CALIBRATION_SELECTIONS_SEALED.json"),
        imported_selections={d: sha256(suite / "selections" / d / "SELECTED_CONFIG_SEALED.json") for d in selected}))
    return selected


def full_stage(suite, prepared, entries, selected):
    require(set(selected) == set(DATASETS), "All eight selections must freeze before full")
    write_json(suite / "ALL_CONFIGS_SELECTED.json", dict(complete=True, full_evaluation_used_for_selection=False,
        selections={d: sha256(suite / "selections" / d / "SELECTED_CONFIG_SEALED.json") for d in selected},
        waterbirds_evaluation_scope="development_overlap"))
    cells = []
    for dataset in ORDER:
        output = suite / "full" / dataset
        selection = suite / "selections" / dataset / "SELECTED_CONFIG_SEALED.json"
        manifest = queue.make_manifest(suite, prepared, dataset, "full", selected[dataset]["config"],
            entries[dataset]["evaluation"], FULL_ARMS, output, selected_config_path=str(selection),
            selected_config_sha256=sha256(selection), all_selections_sha256=sha256(suite / "ALL_CONFIGS_SELECTED.json"))
        code, receipt = queue.child(suite, prepared, "predict", manifest, output)
        require(code == 0, f"Full prediction failed: {receipt}")
        cells.append((dataset, output, manifest))
    boundary = suite / "ALL_FULL_PREDICTIONS_SEALED.json"
    queue.barrier(suite, "full", [o for _, o, _ in cells], boundary)
    rows = []
    for dataset, output, manifest in cells:
        code, receipt = queue.child(suite, prepared, "score", manifest, output, barrier=boundary)
        require(code == 0, f"Full score failed: {receipt}")
        result = queue.validate_result(output, dataset, entries[dataset]["evaluation"], FULL_ARMS, full=True)
        for arm, metrics in result["arm_results"].items():
            rows.append(dict(dataset=dataset, arm=arm, mIoU=metrics["miou_unrounded"], images=result["images"],
                full_split=True, seed=0, config=selected[dataset]["config"], result_path=str(output / "result.json"),
                ms_per_image=1000 * result["timings"][arm]["seconds_per_image"],
                evaluation_scope="development_overlap" if dataset == "waterbirds_seg" else "full_development_exposed",
                calibration_full_source_id_overlap=256 if dataset == "waterbirds_seg" else 0,
                independent_test_claim=False))
        queue.save_csv(suite / "full_metrics.csv", rows)
    require(len(rows) == 32, "Missing eight-dataset full policies")
    return cells, rows


def run(args):
    suite = args.suite.absolute()
    prepared = queue.verify(suite, args.suite_sha256, frozen=True)
    require("continuation" in prepared and not prepared["blocked"], "Not a complete continuation")
    with queue.exclusive(suite):
        require(not (suite / "EXECUTION_STARTED.json").exists(), "Started continuation cannot retry")
        write_json(suite / "EXECUTION_STARTED.json", dict(pid=os.getpid(), started_at=queue.now(), retry_allowed=False,
                   suite_prepared_sha256=args.suite_sha256))
        try:
            entries = read_json(suite / "inputs.json")["datasets"]
            status = wait_for_parent(suite, prepared)
            selected = read_parent_selections(suite, prepared, entries, status)
            panel = copy.deepcopy(entries["waterbirds_seg"]["calibration"])
            panel["inputs"], panel["ordered_ids"] = panel["inputs"][:4], panel["ordered_ids"][:4]
            panel["full_split"] = False
            output = suite / "gates" / "waterbirds_seg"
            manifest = queue.make_manifest(suite, prepared, "waterbirds_seg", "gate", entries["waterbirds_seg"]["config"],
                                            panel, ["image_C"], output)
            code, receipt = queue.child(suite, prepared, "gate", manifest, output)
            require(code == 0 and read_json(output / "GATE_PASSED.json")["passed"] is True, f"Waterbirds gate failed: {receipt}")
            rows = []
            selected["waterbirds_seg"] = queue.tune(suite, prepared, "waterbirds_seg", entries["waterbirds_seg"], rows)
            cells, full_rows = full_stage(suite, prepared, entries, selected)
            queue.verify(suite, args.suite_sha256, frozen=True)
            write_json(suite / "SUITE_SEALED.json", dict(complete=True, datasets=list(DATASETS), arm_results=len(full_rows),
                suite_prepared_sha256=args.suite_sha256, parent_import_sha256=sha256(suite / "PARENT_IMPORT_SEALED.json"),
                waterbirds_gate_sha256=sha256(output / "GATE_PASSED.json"),
                selected_config_sha256=sha256(suite / "ALL_CONFIGS_SELECTED.json"),
                all_full_prediction_barrier_sha256=sha256(suite / "ALL_FULL_PREDICTIONS_SEALED.json"),
                scored_seals={d: sha256(o / "SCORED_SEALED.json") for d, o, _ in cells},
                metrics_sha256=sha256(suite / "full_metrics.csv"), trials_sha256=sha256(suite / "trials.csv"),
                waterbirds_evaluation_scope="development_overlap", service_exit_certified=False))
            queue.state(suite, "completed", arm_results=32, completed_datasets=list(DATASETS))
        except BaseException as error:
            queue.state(suite, "failed", error=f"{type(error).__name__}: {error}"[:2000])
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-root", type=Path, default=ROOT)
    prep.add_argument("--data-root", type=Path, default=Path("/home/wwww/hzl/data"))
    prep.add_argument("--python", type=Path, default=queue.PYTHON)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--parent-suite", type=Path, required=True)
    prep.add_argument("--parent-sha256", required=True)
    prep.add_argument("--parent-service", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--suite", type=Path, required=True)
    execute.add_argument("--suite-sha256", required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.phase](args)


if __name__ == "__main__":
    main()
