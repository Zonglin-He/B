"""Freeze and execute the first sequential, label-isolated evolution suite."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


DATASETS = ("coco_cd", "pc60", "voc21", "coco_object")
POLICIES = ("trace", "uniform", "confidence", "target", "counterfactual", "mean", "permuted")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}}


def plan(datasets=None, trace_only=False):
    selected = tuple(datasets) if datasets is not None else DATASETS
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select distinct, nonempty datasets")
    policies = ["trace"] if trace_only else list(POLICIES)
    cells = [dict(key=f"smoke/{d}", dataset=d, seed=0, limit=4, policies=list(policies))
             for d in selected]
    cells += [dict(key=f"mechanism/{d}", dataset=d, seed=0, limit=None, policies=list(policies))
              for d in selected]
    cells += [dict(key=f"repetition/{d}/seed{s}", dataset=d, seed=s, limit=None, policies=["trace"])
              for d in selected for s in (1, 2)]
    if datasets is None:
        cells += [dict(key="smoke/coco_stuff", dataset="coco_stuff", seed=0, limit=4,
                   policies=list(POLICIES)),
              dict(key="mechanism/coco_stuff", dataset="coco_stuff", seed=0,
                   limit=None, policies=list(POLICIES))]
    return cells


def freeze(args):
    if args.output.exists():
        raise FileExistsError("Use a new suite directory; never overwrite a prior attempt")
    args.output.mkdir(parents=True)
    runtime = args.output / "runtime"
    for name in ("trace_ovss", "tools"):
        shutil.copytree(args.package_root / name, runtime / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("self-evolution-contract.md", "self-evolution-initial-audit.md", "self-evolution-runtime-audit.md"):
        shutil.copy2(args.package_root / "docs" / name, args.output / name)
    originals = args.output / "original_paper"
    originals.mkdir()
    for path in args.originals:
        shutil.copy2(path, originals / path.name)
    sys.path.insert(0, str(runtime / "tools"))
    from suite_inputs import expected_inputs
    cells = plan(getattr(args, "datasets", None), getattr(args, "trace_only", False))
    panels = expected_inputs(runtime, args.data_root,
        {c["key"]: (c["dataset"], "evaluation", c["limit"]) for c in cells},
        data_dirs={"coco_object": "/home/wwww/hzl/data/coco_mlmp"}, selection_seed=20261024)
    images = sorted({p for panel in panels.values() for p in panel["image_ids"]})
    manifest = dict(schema=1, created_at=now(), cells=cells, input_panels=panels,
        package_root=str(args.package_root), data_root=str(args.data_root),
        runtime_hashes=source_hashes(runtime),
        image_sha256={p: digest(p) for p in images},
        original_paper_sha256={p.name: digest(p) for p in originals.iterdir()},
        method="trace_ovss.method.two_view.TwoViewTRACE", rho="min(T,C)",
        selection_seed=20261024, permutation_seeds=[0, 1, 2],
        parameter_search=False, automatic_retry=False, diagnostic_sharing=True,
        inference_label_pixels=False,
        interpretation="Fixed-candidate mechanism controls, not standalone end-to-end runtime measurements.",
        historical_results_are_development_evidence=True)
    dependencies = getattr(args, "dependency", None) or []
    services = getattr(args, "dependency_service", None) or []
    if len(dependencies) != len(services):
        raise ValueError("Each dependency needs exactly one service identity")
    manifest["dependencies"] = []
    receipt_paths = getattr(args, "dependency_completion_receipt", None) or []
    sys.path.insert(0, str(runtime / "tools"))
    from evolution_completion import read, receipt_spec, validate_receipt
    receipt_by_root = {}
    for path in receipt_paths:
        root = str(Path(read(path)["root"]).resolve())
        if root in receipt_by_root or root not in {str(p) for p in dependencies}:
            raise ValueError("Completion receipts must map uniquely to requested dependencies")
        receipt_by_root[root] = receipt_spec(path)
    for parent, service in zip(dependencies, services):
        parent_manifest = json.loads((parent / "manifest.json").read_text())
        expected_total = len(parent_manifest.get("cells", parent_manifest.get("datasets", [])))
        if not expected_total:
            raise ValueError("Dependency has no declared cells")
        manifest["dependencies"].append(dict(path=str(parent), service=service,
            manifest_sha256=digest(parent / "manifest.json"), total=expected_total))
        if str(parent) in receipt_by_root:
            # Preparation binds evidence only. Execution rechecks quiescence
            # after all live dependency services, including cache users, exit.
            validate_receipt(receipt_by_root[str(parent)], parent, require_quiescent=False)
            manifest["dependencies"][-1]["completion_receipt"] = receipt_by_root[str(parent)]
    if args.supersedes:
        previous = json.loads((args.supersedes / "queue_state.json").read_text())
        if previous.get("status") != "failed":
            raise RuntimeError("Supersession requires an explicitly failed earlier stage")
        manifest["supersedes"] = dict(path=str(args.supersedes),
            manifest_sha256=digest(args.supersedes / "manifest.json"),
            failure_state_sha256=digest(args.supersedes / "queue_state.json"),
            reason=getattr(args, "supersession_reason", None) or
                "Runtime determinism amendment from GT-free A/A and A/B replay; no formula, configuration or threshold change.")
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "MANIFEST_SEALED.json", dict(
        manifest_sha256=digest(args.output / "manifest.json"), created_at=now()))
    write_json(args.output / "queue_state.json", dict(status="prepared", completed=0,
        total=len(cells), updated_at=now()))
    print(json.dumps(dict(prepared=str(args.output), cells=len(cells))), flush=True)


def verify_runtime(output, manifest):
    seal = json.loads((output / "MANIFEST_SEALED.json").read_text())
    if digest(output / "manifest.json") != seal["manifest_sha256"]:
        raise RuntimeError("Manifest changed after preparation")
    if source_hashes(output / "runtime") != manifest["runtime_hashes"]:
        raise RuntimeError("Frozen runtime changed")


def write_summary(output, rows):
    if not rows:
        return
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)


def verify_artifacts(directory, seal):
    entries = seal.get("artifact_sha256")
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError("Seal has no artifact hashes")
    for relative, expected in entries.items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()) or digest(path) != expected:
            raise RuntimeError(f"Artifact seal mismatch: {path}")
    return set(entries)


def verify_cell(directory, cell, panel, manifest):
    prediction = json.loads((directory / "PREDICTIONS_SEALED.json").read_text())
    scored = json.loads((directory / "SCORED_SEALED.json").read_text())
    covered = verify_artifacts(directory, prediction)
    scored_covered = verify_artifacts(directory, scored)
    prediction_sha = digest(directory / "PREDICTIONS_SEALED.json")
    if (scored.get("prediction_seal_sha256") != prediction_sha
            or not {"result.json", "score_inputs.json", "scores_by_episode.jsonl"} <= scored_covered
            or not prediction.get("complete") or prediction.get("gt_contents_opened") is not False
            or prediction.get("images") != len(panel["image_ids"])):
        raise RuntimeError("Incomplete or inconsistent prediction/scoring seal chain")
    required = {"prediction_protocol.json", "initial_state.json", "cost.jsonl",
                "prediction_result.json", "expected_panel.json"}
    if not required <= covered:
        raise RuntimeError("Prediction seal lacks protocol/state/parent coverage")
    protocol = json.loads((directory / "prediction_protocol.json").read_text())
    runtime_state = protocol.get("deterministic_runtime", {})
    required_runtime = dict(cublas_workspace_config=":4096:8", deterministic_algorithms=True,
                            deterministic_warn_only=False, cudnn_deterministic=True, cudnn_benchmark=False)
    if any(runtime_state.get(k) != v for k, v in required_runtime.items()):
        raise RuntimeError("Worker did not use the frozen deterministic runtime")
    if protocol.get("expected_panel_sha256") != digest(directory / "expected_panel.json"):
        raise RuntimeError("Cell was not bound to the frozen parent panel")
    if json.loads((directory / "expected_panel.json").read_text()) != panel:
        raise RuntimeError("Parent panel content changed")
    actual_panel = protocol.get("panel", {})
    for key in ("dataset", "split", "seed", "selection_seed", "image_ids", "label_ids",
                "file_metadata", "vocabulary_sha256", "config", "available_images"):
        if actual_panel.get(key) != panel[key]:
            raise RuntimeError(f"Prediction protocol differs from frozen panel: {key}")
    inputs = protocol.get("inputs", [])
    if ([r["image"] for r in inputs] != panel["image_ids"]
            or [r["label"] for r in inputs] != panel["label_ids"]
            or any(r["image_sha256"] != manifest["image_sha256"][r["image"]] for r in inputs)):
        raise RuntimeError("Prediction images/content differ from prepared inputs")
    runtime = directory.parents[len(Path(cell["key"]).parts)] / "runtime"
    # The controller verifies the complete runtime; bind each worker's declared
    # source subset as well, rather than trusting the worker's path strings.
    for path, expected in protocol.get("source_sha256", {}).items():
        relative = str(Path(path).resolve().relative_to(runtime.resolve()))
        if manifest["runtime_hashes"].get(relative) != expected:
            raise RuntimeError(f"Worker source differs from frozen runtime: {path}")
    if not protocol.get("source_sha256"):
        raise RuntimeError("Worker did not record source identity")
    expected_variants = [(f"permuted_{s}", p, s) if p == "permuted" else (p, p, None)
                         for p in cell["policies"]
                         for s in (manifest["permutation_seeds"] if p == "permuted" else [None])]
    if protocol.get("variants") != [list(v) for v in expected_variants]:
        raise RuntimeError("Requested policy/permutation variants are not complete")
    names = ["pre", *[v[0] for v in expected_variants]]
    expected_payload = {f"predictions/{name}/{i:06d}.npy"
                        for name in names for i in range(len(inputs))}
    actual_payload = {str(p.relative_to(directory)) for p in (directory / "predictions").rglob("*.npy")}
    if actual_payload != expected_payload or not expected_payload <= covered:
        raise RuntimeError("Prediction payload coverage differs from requested inputs/policies")
    result = json.loads((directory / "result.json").read_text())
    if (result.get("status") != "completed" or result.get("prediction_seal_sha256") != prediction_sha
            or result.get("label_contents_first_opened_after_prediction_seal") is not True
            or result["images"] != len(inputs) or result["dataset"] != cell["dataset"]
            or result["seed"] != cell["seed"] or bool(result["full_split"]) != (cell["limit"] is None)):
        raise RuntimeError("Cell result identity/completion or label-isolation claim differs")
    variants = [(r["policy"], r.get("permutation_seed")) for r in result["policy_results"]]
    if len(variants) != len(set(variants)) or set(variants) != {(v[1], v[2]) for v in expected_variants}:
        raise RuntimeError("Missing, duplicate or unrequested policy results")
    expected_labels = {(policy, seed): label for label, policy, seed in expected_variants}
    if any(r.get("policy_label") != expected_labels[r["policy"], r.get("permutation_seed")]
           for r in result["policy_results"]):
        raise RuntimeError("Policy labels do not identify their requested metrics")
    pre = result["pre_result"]
    if pre.get("policy") != "pre" or pre.get("policy_label") != "pre" or pre.get("permutation_seed") is not None:
        raise RuntimeError("Pre-adaptation metrics were assigned to another policy")
    scored_inputs = json.loads((directory / "score_inputs.json").read_text())
    if set(scored_inputs.get("label_sha256", {})) != set(panel["label_ids"]):
        raise RuntimeError("Offline scoring label population differs")
    totals = {name: None for name in names}
    batch = panel["config"]["batch_size"]
    with (directory / "scores_by_episode.jsonl").open() as handle:
        records = 0
        for episode, line in enumerate(handle):
            row = json.loads(line)
            if (row["episode"] != episode or row["image_indices"] != list(range(
                    episode * batch, min((episode + 1) * batch, len(inputs))))
                    or set(row["policies"]) != set(names)):
                raise RuntimeError("Scored episode coverage differs from frozen boundaries")
            for name, counts in row["policies"].items():
                values = [counts["intersection"], counts["union"]]
                if (len(values[0]) != len(protocol["classes"]) or len(values[1]) != len(values[0])
                        or any(type(v) is not int or v < 0 for a in values for v in a)
                        or any(i > u for i, u in zip(*values))):
                    raise RuntimeError("Invalid segmentation counts")
                if totals[name] is None:
                    totals[name] = values
                else:
                    totals[name] = [[a + b for a, b in zip(x, y)] for x, y in zip(totals[name], values)]
            records += 1
    if records != (len(inputs) + batch - 1) // batch:
        raise RuntimeError("Missing scored episodes")
    for row in [result["pre_result"], *result["policy_results"]]:
        i, u = totals[row["policy_label"]]
        if row["intersection"] != i or row["union"] != u:
            raise RuntimeError("Result does not aggregate sealed episode counts")
        ratios = [a / b for a, b in zip(i, u) if b]
        expected = 100 * math.fsum(ratios) / len(ratios) if ratios else None
        if expected is None or not math.isfinite(row["miou_unrounded"]) or abs(expected - row["miou_unrounded"]) > 1e-10:
            raise RuntimeError("Reported mIoU differs from complete dataset counts")
    return result


def wait_dependencies(output, manifest):
    dependencies = manifest.get("dependencies", [])
    if not dependencies:
        return
    sys.path.insert(0, str(output / "runtime"))
    from tools.evolution_replay.controller import service_state
    from tools.evolution_completion import validate_receipt

    while True:
        receipts = []
        # Cached-control workers legitimately load the completed parent's
        # runtime. Wait for live downstream services before checking whether
        # any process still references a receipt-backed runtime.
        ordered = sorted(dependencies, key=lambda d: bool(d.get("completion_receipt")))
        for dependency in ordered:
            parent = Path(dependency["path"])
            if digest(parent / "manifest.json") != dependency["manifest_sha256"]:
                raise RuntimeError("Dependency manifest changed")
            if dependency.get("completion_receipt"):
                receipts.append(validate_receipt(dependency["completion_receipt"], parent))
                continue
            status = json.loads((parent / "queue_state.json").read_text())
            service = service_state(dependency["service"])
            terminal = service["MainPID"] == "0"
            if status["status"] == "failed" or (terminal and service["Result"] != "success"):
                raise RuntimeError("Dependency failed; no automatic recovery or GPU launch")
            if terminal and status["status"] != "completed":
                raise RuntimeError("Dependency exited without completing its frozen plan")
            if status["status"] != "completed" or not terminal:
                break
            if (status.get("completed") != dependency["total"]
                    or status.get("total") != dependency["total"]
                    or (service["ActiveState"], service["SubState"]) not in {
                        ("inactive", "dead"), ("active", "exited")}):
                raise RuntimeError("Dependency completion count or terminal state differs")
            seal = parent / "SUITE_SEALED.json"
            verify_artifacts(parent, json.loads(seal.read_text()))
            receipts.append(dict(path=str(parent), seal_sha256=digest(seal), service=service))
        if len(receipts) == len(dependencies):
            write_json(output / "dependencies.json", receipts)
            return
        write_json(output / "queue_state.json", dict(status="waiting_for_dependencies",
            completed=0, total=len(manifest["cells"]), updated_at=now(), pid=os.getpid()))
        time.sleep(60)


def execute(output):
    manifest = json.loads((output / "manifest.json").read_text())
    if Path(__file__).resolve() != output / "runtime/tools/self_evolution_controller.py":
        raise RuntimeError("Execute the frozen controller inside the suite runtime")
    verify_runtime(output, manifest)
    lock_path = Path("/home/wwww/hzl/results/reviewer_followup_gpu.lock")
    with (output / "queue.lock").open("a") as local, lock_path.open("a") as gpu:
        fcntl.flock(local, fcntl.LOCK_EX | fcntl.LOCK_NB)
        wait_dependencies(output, manifest)
        verify_runtime(output, manifest)
        write_json(output / "queue_state.json", dict(status="waiting_for_gpu_lock", completed=0,
            total=len(manifest["cells"]), updated_at=now(), pid=os.getpid()))
        fcntl.flock(gpu, fcntl.LOCK_EX)
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                           "--format=csv,noheader"], text=True).strip()
        if active:
            raise RuntimeError(f"Unmanaged active GPU workers are not interrupted: {active}")
        rows = []
        for index, cell in enumerate(manifest["cells"]):
            verify_runtime(output, manifest)
            directory = output / "cells" / cell["key"]
            if directory.exists():
                raise FileExistsError(f"No automatic retries or reuse of partial cells: {directory}")
            directory.mkdir(parents=True)
            panel = dict(manifest["input_panels"][cell["key"]], seed=cell["seed"],
                         image_sha256={p: manifest["image_sha256"][p]
                                       for p in manifest["input_panels"][cell["key"]]["image_ids"]})
            write_json(directory / "expected_panel.json", panel)
            command = [sys.executable, str(output / "runtime/tools/evolution/run_cell.py"),
                "--package-root", str(output / "runtime"), "--data-root", manifest["data_root"],
                "--dataset", cell["dataset"], "--output", str(directory),
                "--seed", str(cell["seed"]), "--phase", "run",
                "--policies", *cell["policies"], "--permutation-seeds", "0", "1", "2"]
            if cell["dataset"] == "coco_object":
                command += ["--data-dir", "/home/wwww/hzl/data/coco_mlmp"]
            if cell["limit"]:
                command += ["--max-images", str(cell["limit"]), "--verify-reference"]
            write_json(output / "queue_state.json", dict(status="running", completed=index,
                total=len(manifest["cells"]), active_cell=cell["key"],
                updated_at=now(), pid=os.getpid(), command=command))
            started = time.monotonic()
            with (directory / "run.log").open("x") as log:
                subprocess.run(command, cwd=output / "runtime", stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            verify_runtime(output, manifest)
            result = verify_cell(directory, cell, panel, manifest)
            for row in result["policy_results"]:
                rows.append(dict(cell=cell["key"], dataset=cell["dataset"], seed=cell["seed"],
                    images=result["images"], full_split=result["full_split"],
                    policy=row["policy"], permutation_seed=row.get("permutation_seed"),
                    miou_unrounded=row["miou_unrounded"], result_path=str(directory / "result.json")))
            write_summary(output, rows)
            write_json(output / "queue_state.json", dict(status="running", completed=index + 1,
                total=len(manifest["cells"]), last_cell=cell["key"],
                last_cell_seconds=time.monotonic() - started, updated_at=now(), pid=os.getpid()))
            print(json.dumps(dict(completed=index + 1, cell=cell["key"], result=result)), flush=True)
        # Re-read images after the queue to detect external changes during execution.
        for path, expected in manifest["image_sha256"].items():
            if digest(path) != expected:
                raise RuntimeError(f"Input changed during execution: {path}")
        for cell in manifest["cells"]:
            directory = output / "cells" / cell["key"]
            panel = json.loads((directory / "expected_panel.json").read_text())
            verify_cell(directory, cell, panel, manifest)
        write_json(output / "queue_state.json", dict(status="completed", completed=len(manifest["cells"]),
            total=len(manifest["cells"]), updated_at=now(), pid=os.getpid()))
        artifacts = [output / "manifest.json", output / "summary.csv", output / "queue_state.json"]
        if manifest.get("dependencies"):
            artifacts.append(output / "dependencies.json")
        artifacts += [output / "cells" / c["key"] / n for c in manifest["cells"]
                      for n in ("result.json", "PREDICTIONS_SEALED.json", "SCORED_SEALED.json")]
        write_json(output / "SUITE_SEALED.json", dict(created_at=now(),
            artifact_sha256={str(p.relative_to(output)): digest(p) for p in artifacts},
            total_cells=len(manifest["cells"]), final_paper_review_complete=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument("--package-root", type=lambda x: Path(x).resolve())
    parser.add_argument("--data-root", type=lambda x: Path(x).resolve(), default=Path("/home/wwww/hzl/data"))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--originals", type=Path, nargs="*", default=[])
    parser.add_argument("--supersedes", type=lambda x: Path(x).resolve())
    parser.add_argument("--supersession-reason")
    parser.add_argument("--datasets", nargs="+", choices=[*DATASETS, "voc20", "coco_stuff"])
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--dependency", type=lambda x: Path(x).resolve(), action="append")
    parser.add_argument("--dependency-service", action="append")
    parser.add_argument("--dependency-completion-receipt", type=lambda x: Path(x).resolve(), action="append")
    args = parser.parse_args()
    if args.prepare:
        if not args.package_root:
            parser.error("Preparation needs --package-root")
        if len(args.dependency or []) != len(args.dependency_service or []):
            parser.error("Each --dependency needs one --dependency-service")
        freeze(args)
        return
    try:
        execute(args.output)
    except BaseException as error:
        old = json.loads((args.output / "queue_state.json").read_text())
        write_json(args.output / "queue_state.json", dict(old, status="failed", updated_at=now(),
            error=repr(error), automatic_retry=False))
        raise


if __name__ == "__main__":
    main()
