"""Fresh bounded per-dataset calibration search, then sealed full evaluation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from tools.evolution_tuning.common import (
    ALGORITHM, COORDINATES, DATASETS, FULL_ARMS, MAX_TRIALS, ORDER, ROOT,
    canonical_hash, choose, coarse_values, config_key, fine_values, pooled_miou,
    read_json, require, sha256, verify_hashes, write_json,
)


WORKER = "tools.evolution_tuning.worker"
PYTHON = Path("/home/wwww/miniconda3/envs/mlmp/bin/python")


def now():
    return datetime.now(timezone.utc).isoformat()


def state(suite, phase, **fields):
    write_json(suite / "queue_state.json", dict(phase=phase, updated_at=now(), pid=os.getpid(), **fields), replace=True)


def snapshot(source, target):
    # Same reviewed source boundary as prior suites, including the new package.
    from tools.evolution_followup.queue import snapshot as copy_snapshot
    return copy_snapshot(source, target)


def rebase_panel_sources(datasets, source, suite, inventory):
    """Bind code/config provenance to copied bytes, not the mutable checkout."""
    source = Path(source).resolve()
    mapping = {}
    for entry in datasets.values():
        for panel_name in ("calibration", "evaluation"):
            panel = entry.get(panel_name)
            if panel is None:
                continue
            rebound = {}
            for filename, record in panel.get("source_hashes", {}).items():
                original = Path(filename)
                if original.is_relative_to(source):
                    relative = str(original.relative_to(source))
                    if relative in inventory:
                        target = suite / "runtime" / relative
                    elif relative == "docs/single-probe-tuning-contract.md":
                        target = suite / "frozen_contract.md"
                    else:
                        raise RuntimeError(f"Uncopied managed input source: {relative}")
                    expected = record["sha256"] if isinstance(record, dict) else record
                    require(sha256(target) == expected, "Input source changed before snapshot rebinding")
                    mapping[filename] = str(target)
                    rebound[str(target)] = record
                else:
                    rebound[filename] = record
            panel["source_hashes"] = rebound
    return mapping


def prepare(args):
    from tools.evolution_tuning.inputs import prepare_inputs
    from tools.evolution.run_cell import checkpoint_identity

    suite = args.output.absolute()
    suite.mkdir(parents=True, exist_ok=False)
    require(suite == suite.resolve(), "Suite directory aliases rejected")
    state(suite, "preparing")
    try:
        bundle = prepare_inputs(args.source_root.absolute(), args.data_root.absolute(), suite,
                                calibration_size=256, seed=2027)
        datasets = bundle.get("datasets", bundle)
        require(set(datasets) == set(DATASETS), "All eight original datasets must be accounted for")
        runnable, blocked = [], {}
        for dataset in ORDER:
            entry = datasets[dataset]
            if not entry.get("calibration"):
                blocked[dataset] = entry.get("blocked_reason", entry.get("reason", "No source-disjoint calibration"))
            else:
                require(len(entry["calibration"]["inputs"]) == 256, "Calibration must contain256 images")
                runnable.append(dataset)
            require(entry.get("evaluation") and entry.get("config"), "Missing full dataset/config")
        require(runnable, "No runnable training calibration dataset")
        runtime_hashes = snapshot(args.source_root.absolute(), suite / "runtime")
        copy_contract = args.source_root / "docs/single-probe-tuning-contract.md"
        shutil.copy2(copy_contract, suite / "frozen_contract.md")
        mapping = rebase_panel_sources(datasets, args.source_root.absolute(), suite, runtime_hashes)
        write_json(suite / "inputs.json", dict(datasets=datasets, frozen_source_path_mapping=mapping,
            preparation_metadata={k: v for k, v in bundle.items() if k != "datasets"} if "datasets" in bundle else {}))
        prepared = dict(schema_version=1, algorithm=ALGORITHM, method_changes=False,
            datasets=list(DATASETS), order=list(ORDER), runnable=runnable, blocked=blocked,
            calibration_images=256, calibration_seed=2027, model_seed=0,
            max_unique_trials_per_dataset=MAX_TRIALS, coordinates=list(COORDINATES),
            full_arms=FULL_ARMS, python=str(args.python.absolute()),
            runtime_hashes=runtime_hashes, inputs_sha256=sha256(suite / "inputs.json"),
            initial_configs={d: datasets[d]["config"] for d in DATASETS},
            contract_sha256=sha256(suite / "frozen_contract.md"), checkpoint=checkpoint_identity(),
            selection="Calibration mIoU maximum; within0.02pp lower measured model time then config tuple. No test selection.",
            full_label_barrier="All eight datasets/config selections and all eight full arm predictions seal before full scoring.",
            historical_exposure="Development-exposed full protocols and inherited starts; no globally untouched claim.",
            retry=False, created_at=now())
        write_json(suite / "SUITE_PREPARED.json", prepared)
        state(suite, "prepared", runnable=runnable, blocked=blocked, total_datasets=8)
        print(json.dumps(dict(suite=str(suite), prepared_sha256=sha256(suite / "SUITE_PREPARED.json"),
                              runnable=runnable, blocked=blocked), indent=2))
    except BaseException as error:
        state(suite, "prepare_failed", error=f"{type(error).__name__}: {error}")
        raise


def verify(suite, expected, *, frozen=False):
    suite = Path(suite).absolute()
    require(suite == suite.resolve(), "Suite aliases rejected")
    require(sha256(suite / "SUITE_PREPARED.json") == expected, "Prepared identity drift")
    prepared = read_json(suite / "SUITE_PREPARED.json")
    require(prepared["algorithm"] == ALGORITHM and prepared["datasets"] == list(DATASETS)
            and prepared["order"] == list(ORDER) and prepared["full_arms"] == FULL_ARMS
            and prepared["max_unique_trials_per_dataset"] == MAX_TRIALS
            and prepared["coordinates"] == list(COORDINATES), "Scope/search plan differs")
    require(sha256(suite / "inputs.json") == prepared["inputs_sha256"]
            and sha256(suite / "frozen_contract.md") == prepared["contract_sha256"], "Frozen inputs/contract drift")
    verify_hashes(suite / "runtime", prepared["runtime_hashes"])
    from tools.evolution_followup.queue import source_inventory
    require(source_inventory(suite / "runtime") == prepared["runtime_hashes"], "Frozen runtime inventory additions/removals")
    actual = {}
    for path in (suite / "runtime").rglob("*"):
        require(not path.is_symlink() and path.name not in ("__pycache__", ".pytest_cache")
                and path.suffix not in (".pyc", ".pyo"), "Unrecorded executable cache/symlink")
        if path.is_file():
            actual[str(path.relative_to(suite / "runtime"))] = sha256(path)
    require(actual == prepared["runtime_hashes"], "Unrecorded runtime-root/startup file")
    if frozen:
        require(ROOT == suite / "runtime" and Path.cwd().resolve() == ROOT, "Run only from frozen runtime")
    prepared["_verified_sha256"] = expected
    return prepared


def make_manifest(suite, prepared, dataset, stage, config, panel, arms, output, **extra):
    initial = prepared["initial_configs"][dataset]
    require(set(initial) == set(config), "Config schema changed")
    require(all(initial[k] == config[k] for k in initial if k not in COORDINATES), "Non-search method/protocol setting changed")
    require(1 <= config["batch_size"] <= 32 and 1 <= config["steps"] <= 64
            and 1 <= config["trace_max_regions"] <= 64 and 1e-6 <= config["lr"] <= .04,
            "Config exceeds fixed search bounds")
    manifest = dict(schema_version=1, algorithm=ALGORITHM, suite_root=str(suite),
        suite_prepared_sha256=prepared["_verified_sha256"], runtime_root=str(suite / "runtime"),
        dataset=dataset, stage=stage, seed=0, config=config, checkpoint=prepared["checkpoint"],
        panel=panel, arms=arms, source_sha256=prepared["runtime_hashes"], output=str(output), **extra)
    label_seal = suite / "calibration_labels" / f"{dataset}.json"
    if stage == "calibration" and label_seal.exists():
        manifest.update(calibration_label_seal_path=str(label_seal), calibration_label_seal_sha256=sha256(label_seal))
    path = output.parent / (output.name + ".manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest)
    return path


def child(suite, prepared, phase, manifest, output, *, barrier=None):
    verify(suite, prepared["_verified_sha256"], frozen=True)
    command = [prepared["python"], "-B", "-m", WORKER, "--phase", phase,
               "--manifest", str(manifest), "--manifest-sha256", sha256(manifest), "--output", str(output)]
    if barrier is not None:
        command += ["--barrier", str(barrier)]
    logs = suite / "logs"
    logs.mkdir(exist_ok=True)
    label = canonical_hash([str(output), phase])[:24]
    receipt, log = logs / (label + ".json"), logs / (label + ".log")
    require(not receipt.exists() and not log.exists(), "Child retry rejected")
    state(suite, phase, active_output=str(output), manifest=str(manifest), command=command)
    environment = dict(os.environ, PYTHONPATH=str(suite / "runtime"), PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if phase == "score":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    started = now()
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=suite / "runtime", env=environment,
                                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    write_json(receipt, dict(command=command, phase=phase, output=str(output), manifest_sha256=sha256(manifest),
        started_at=started, ended_at=now(), returncode=result.returncode, log_sha256=sha256(log)))
    verify(suite, prepared["_verified_sha256"], frozen=True)
    return result.returncode, receipt


def prediction_seal(output):
    path = output / "PREDICTIONS_SEALED.json"
    seal = read_json(path)
    require(seal["complete"] is True and seal["gt_contents_opened"] is False,
            "Missing valid prediction seal")
    verify_hashes(output, seal["artifact_sha256"])
    return sha256(path)


def barrier(suite, stage, outputs, path):
    require(outputs and not path.exists(), "Empty or repeated scoring barrier")
    write_json(path, dict(complete=True, stage=stage, gt_contents_opened=False,
        suite_prepared_sha256=sha256(suite / "SUITE_PREPARED.json"),
        prediction_seal_sha256={str(output): prediction_seal(output) for output in outputs}))


def validate_result(output, dataset, panel, arms, *, full):
    seal = read_json(output / "SCORED_SEALED.json")
    verify_hashes(output, seal["artifact_sha256"])
    result = read_json(output / "result.json")
    require(result["dataset"] == dataset and result["images"] == len(panel["inputs"])
            and result["full_split"] is full and set(result["arm_results"]) == set(arms), "Result scope mismatch")
    for row in result["arm_results"].values():
        metric = pooled_miou(row["intersection"], row["union"])
        require(abs(metric - row["miou_unrounded"]) <= 1e-10, "Result is not pooled-count mIoU")
    if not full:
        protocol = read_json(output / "prediction_protocol.json")["manifest"]
        labels = Path(protocol["suite_root"]) / "calibration_labels" / f"{dataset}.json"
        require(result["calibration_label_seal_sha256"] == sha256(labels)
                and read_json(output / "score_inputs.json")["label_sha256"] == read_json(labels)["label_sha256"],
                "Calibration label content changed between trials")
    return result


def save_csv(path, rows):
    fields = sorted({k for row in rows for k in row})
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)
    temporary.replace(path)


def ineligible_failure(output, manifest_path):
    """Only an exactly bound, explicitly nonfatal numerical receipt can skip."""
    path = output / "PREDICTION_FAILED.json"
    require(path.is_file() and not (output / "failure.json").exists(), "Missing/conflicting worker failure receipts")
    failure, manifest = read_json(path), read_json(manifest_path)
    require(failure.get("fatal") is False and failure.get("ineligible_trial") is True
            and failure.get("retry_allowed") is False
            and failure.get("failure_kind") in ("oom", "nonfinite")
            and failure.get("status") == "failed" and failure.get("phase") == "predict",
            "Fatal/malformed worker failure; stop instead of skipping")
    require(failure.get("manifest_sha256") == sha256(manifest_path)
            and failure.get("config") == manifest["config"]
            and failure.get("source_sha256") == manifest["source_sha256"]
            and failure.get("checkpoint") == manifest["checkpoint"]
            and failure.get("output") == str(output)
            and failure.get("dataset") == manifest["dataset"]
            and failure.get("stage") == "calibration" and failure.get("seed") == 0
            and failure.get("calibration") is True and failure.get("full_split") is False,
            "Numerical failure identity/provenance differs")
    return failure


def trial(suite, prepared, dataset, entry, config, stage, rows, cache):
    key = config_key(config)
    if key in cache:
        return cache[key]
    require(len(cache) < MAX_TRIALS, "Bounded trial inventory exhausted")
    number = len(cache)
    output = suite / "trials" / dataset / f"{number:03d}_{key}"
    panel = entry["calibration"]
    manifest = make_manifest(suite, prepared, dataset, "calibration", config, panel, ["image_C"], output,
                             search_coordinate=stage, trial_index=number)
    code, receipt = child(suite, prepared, "predict", manifest, output)
    if code:
        failure = ineligible_failure(output, manifest)
        row = dict(dataset=dataset, stage=stage, trial=number, status="ineligible", config=config,
                   mIoU=None, ms_per_image=None, failure=failure, result_path=str(output), receipt=str(receipt),
                   failure_sha256=sha256(output / "PREDICTION_FAILED.json"))
    else:
        bound = output.parent / (output.name + ".barrier.json")
        barrier(suite, "calibration", [output], bound)
        code, receipt = child(suite, prepared, "score", manifest, output, barrier=bound)
        require(code == 0, f"Scoring failed; inspect {receipt}")
        result = validate_result(output, dataset, panel, ["image_C"], full=False)
        timing = result["timings"]["image_C"]
        milliseconds = 1000 * timing["seconds_per_image"]
        row = dict(dataset=dataset, stage=stage, trial=number, status="completed", config=config,
            mIoU=result["arm_results"]["image_C"]["miou_unrounded"], ms_per_image=milliseconds,
            images=result["images"], full_split=False, result_path=str(output / "result.json"),
            scored_seal_sha256=sha256(output / "SCORED_SEALED.json"))
    cache[key] = row
    rows.append(row)
    save_csv(suite / "trials.csv", rows)
    state(suite, "trial_completed", dataset=dataset, unique_trials=len(cache), last_trial=row,
          best_trial=choose(list(cache.values())) if any(r["status"] == "completed" for r in cache.values()) else None)
    return row


def tune(suite, prepared, dataset, entry, rows):
    initial = copy.deepcopy(entry["config"])
    current, cache = copy.deepcopy(initial), {}
    trial(suite, prepared, dataset, entry, current, "inherited", rows, cache)
    selections = suite / "selections" / dataset
    selections.mkdir(parents=True)
    for coordinate in COORDINATES:
        for resolution in ("coarse", "fine"):
            values = coarse_values(coordinate, initial) if resolution == "coarse" else fine_values(coordinate, current)
            trials = []
            for value in values:
                candidate = copy.deepcopy(current)
                candidate[coordinate] = value
                trials.append(trial(suite, prepared, dataset, entry, candidate,
                                    f"{coordinate}/{resolution}", rows, cache))
            winner = choose(trials)
            current = copy.deepcopy(winner["config"])
            write_json(selections / f"{coordinate}_{resolution}.json", dict(
                coordinate=coordinate, resolution=resolution, values=values,
                selected_trial=winner, config=current, validation_used_for_selection=False))
    winner = choose(list(cache.values()))
    selected = dict(dataset=dataset, config=winner["config"], selected_trial=winner,
        unique_trials=len(cache), all_trials=[dict(key=k, status=v["status"], result_path=v["result_path"])
                                           for k, v in cache.items()],
        calibration_only=True, full_evaluation_used_for_selection=False,
        calibration_panel_sha256=canonical_hash(entry["calibration"]),
        suite_prepared_sha256=sha256(suite / "SUITE_PREPARED.json"))
    write_json(selections / "SELECTED_CONFIG_SEALED.json", selected)
    return selected


@contextmanager
def exclusive(suite):
    with (suite / "queue.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(args):
    suite = args.suite.absolute()
    prepared = verify(suite, args.suite_sha256, frozen=True)
    with exclusive(suite):
        require(not (suite / "EXECUTION_STARTED.json").exists(), "No restart/retry of a started suite")
        write_json(suite / "EXECUTION_STARTED.json", dict(pid=os.getpid(), started_at=now(),
            suite_prepared_sha256=args.suite_sha256, retry_allowed=False))
        entries = read_json(suite / "inputs.json")["datasets"]
        trials, selected, gates, full = [], {}, {}, []
        try:
            for dataset in prepared["runnable"]:
                entry = entries[dataset]
                panel = copy.deepcopy(entry["calibration"])
                panel["inputs"] = panel["inputs"][:4]
                panel["ordered_ids"] = panel["ordered_ids"][:4]
                panel["full_split"] = False
                output = suite / "gates" / dataset
                manifest = make_manifest(suite, prepared, dataset, "gate", entry["config"], panel,
                                         ["image_C"], output)
                code, receipt = child(suite, prepared, "gate", manifest, output)
                require(code == 0, f"Real gate failed; inspect {receipt}")
                gate = read_json(output / "GATE_PASSED.json")
                require(gate.get("passed", gate.get("status") == "passed") is True, "Gate receipt missing pass")
                gates[dataset] = sha256(output / "GATE_PASSED.json")
            write_json(suite / "ALL_GATES_PASSED.json", dict(complete=True, gates=gates))
            for dataset in prepared["runnable"]:
                selected[dataset] = tune(suite, prepared, dataset, entries[dataset], trials)
            if prepared["blocked"]:
                write_json(suite / "CALIBRATION_SELECTIONS_SEALED.json", dict(
                    complete=False, full_evaluation_ready=False, blocked=prepared["blocked"],
                    selections={d: sha256(suite / "selections" / d / "SELECTED_CONFIG_SEALED.json") for d in selected},
                    suite_prepared_sha256=args.suite_sha256, full_evaluation_used_for_selection=False))
                state(suite, "awaiting_all_eight_calibrations", completed_selections=list(selected),
                      blocked=prepared["blocked"], unique_trials=len(trials), full_scoring_started=False)
                return
            require(set(selected) == set(DATASETS), "All eight configs must seal before full evaluation")
            write_json(suite / "ALL_CONFIGS_SELECTED.json", dict(complete=True,
                selections={d: sha256(suite / "selections" / d / "SELECTED_CONFIG_SEALED.json") for d in selected},
                full_evaluation_used_for_selection=False))
            for dataset in prepared["runnable"]:
                output = suite / "full" / dataset
                path = suite / "selections" / dataset / "SELECTED_CONFIG_SEALED.json"
                manifest = make_manifest(suite, prepared, dataset, "full", selected[dataset]["config"],
                    entries[dataset]["evaluation"], FULL_ARMS, output, selected_config_path=str(path),
                    selected_config_sha256=sha256(path), all_selections_sha256=sha256(suite / "ALL_CONFIGS_SELECTED.json"))
                code, receipt = child(suite, prepared, "predict", manifest, output)
                require(code == 0, f"Full prediction failed; inspect {receipt}")
                full.append((dataset, output, manifest))
            bound = suite / "ALL_FULL_PREDICTIONS_SEALED.json"
            barrier(suite, "full", [o for _, o, _ in full], bound)
            results = []
            for dataset, output, manifest in full:
                code, receipt = child(suite, prepared, "score", manifest, output, barrier=bound)
                require(code == 0, f"Full score failed; inspect {receipt}")
                result = validate_result(output, dataset, entries[dataset]["evaluation"], FULL_ARMS, full=True)
                for arm, metric in result["arm_results"].items():
                    results.append(dict(dataset=dataset, arm=arm, mIoU=metric["miou_unrounded"],
                        images=result["images"], full_split=True, seed=0, config=selected[dataset]["config"],
                        ms_per_image=1000 * result["timings"][arm]["seconds_per_image"],
                        result_path=str(output / "result.json")))
                save_csv(suite / "full_metrics.csv", results)
            verify(suite, args.suite_sha256, frozen=True)
            complete = not prepared["blocked"]
            write_json(suite / ("SUITE_SEALED.json" if complete else "PARTIAL_SUITE_SEALED.json"), dict(
                complete=complete, completed_datasets=prepared["runnable"], blocked=prepared["blocked"],
                arm_results=len(results), unique_trials=len(trials), selection_only_calibration=True,
                suite_prepared_sha256=args.suite_sha256,
                scored_seals={d: sha256(o / "SCORED_SEALED.json") for d, o, _ in full},
                results_sha256=sha256(suite / "full_metrics.csv"), trials_sha256=sha256(suite / "trials.csv"),
                service_exit_certified=False))
            state(suite, "completed" if complete else "partial_completed_blocked", completed_datasets=prepared["runnable"],
                  blocked=prepared["blocked"], arm_results=len(results), unique_trials=len(trials), error=None)
        except BaseException as error:
            state(suite, "failed", error=f"{type(error).__name__}: {error}"[:2000],
                  completed_selections=list(selected), unique_trials=len(trials))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-root", type=Path, default=ROOT)
    prep.add_argument("--data-root", type=Path, default=Path("/home/wwww/hzl/data"))
    prep.add_argument("--python", type=Path, default=PYTHON)
    prep.add_argument("--output", type=Path, required=True)
    for phase in ("verify", "run"):
        command = sub.add_parser(phase)
        command.add_argument("--suite", type=Path, required=True)
        command.add_argument("--suite-sha256", required=True)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args)
    elif args.phase == "verify":
        prepared = verify(args.suite, args.suite_sha256)
        print(json.dumps(dict(verified=True, runnable=prepared["runnable"], blocked=prepared["blocked"])))
    else:
        run(args)


if __name__ == "__main__":
    main()
