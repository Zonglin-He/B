"""Wait for the frozen parent to finish, then run two predeclared cached controls."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


DATASETS = {"pc60": 5105, "coco_object": 5000}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def hashes(path):
    return {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}}


def state(output, status, **kwargs):
    write(output / "queue_state.json", dict(status=status, updated_at=datetime.now(timezone.utc).isoformat(),
        total=2, **kwargs))


def verify_seal(directory, seal):
    if not seal.get("artifact_sha256"):
        raise RuntimeError("Empty artifact seal")
    for relative, expected in seal["artifact_sha256"].items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()) or digest(path) != expected:
            raise RuntimeError(f"Changed sealed artifact: {path}")


def service_state(name):
    result = subprocess.run(["systemctl", "--user", "show", name,
        "--property=LoadState,MainPID,ActiveState,SubState,Result"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Could not verify parent service: {result.stderr.strip()}")
    state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if not {"LoadState", "MainPID", "ActiveState", "SubState", "Result"} <= set(state):
        raise RuntimeError("Parent service query omitted required terminal-state fields")
    if state["LoadState"] != "loaded" or not state["MainPID"].isdigit():
        raise RuntimeError("Parent service identity/state is unavailable")
    return state


def wait_parent(output, manifest):
    parent = Path(manifest["parent"])
    if manifest.get("parent_completion_receipt"):
        if digest(parent / "manifest.json") != manifest["parent_manifest_sha256"]:
            raise RuntimeError("Parent manifest changed")
        from evolution_completion import validate_receipt
        evidence = validate_receipt(manifest["parent_completion_receipt"], parent)
        write(output / "dependency.json", evidence)
        return
    while True:
        if digest(parent / "manifest.json") != manifest["parent_manifest_sha256"]:
            raise RuntimeError("Parent manifest changed")
        status = read(parent / "queue_state.json")
        if status["status"] == "failed":
            raise RuntimeError("Parent failed; dependent controls are not automatically retried")
        if status["status"] == "completed" and (parent / "SUITE_SEALED.json").is_file():
            actual_service = service_state(manifest["parent_service"])
            if actual_service["MainPID"] == "0":
                if actual_service["Result"] != "success":
                    raise RuntimeError("Parent service exited abnormally")
                if (actual_service["ActiveState"], actual_service["SubState"]) not in {
                        ("inactive", "dead"), ("active", "exited")}:
                    raise RuntimeError("Parent service is not in a verified successful terminal state")
                verify_seal(parent, read(parent / "SUITE_SEALED.json"))
                write(output / "dependency.json", dict(parent_seal_sha256=digest(parent / "SUITE_SEALED.json"),
                    parent_queue_state_sha256=digest(parent / "queue_state.json"), service=actual_service))
                return
        state(output, "waiting_for_parent", completed=0)
        time.sleep(60)


def execute(output):
    manifest = read(output / "manifest.json")
    if digest(output / "manifest.json") != read(output / "MANIFEST_SEALED.json")["manifest_sha256"]:
        raise RuntimeError("Replay manifest changed after preparation")
    if Path(__file__).resolve() != output / "runtime/controller.py":
        raise RuntimeError("Use the frozen controller")
    if hashes(output / "runtime") != manifest["source_sha256"]:
        raise RuntimeError("Frozen replay sources changed")
    with (output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        wait_parent(output, manifest)
        results = {}
        parent = Path(manifest["parent"])
        for index, (dataset, images) in enumerate(DATASETS.items()):
            if hashes(output / "runtime") != manifest["source_sha256"]:
                raise RuntimeError("Frozen replay sources changed")
            cell = output / "cells" / dataset
            if cell.exists():
                raise FileExistsError("Existing control outputs are not resumed or overwritten")
            command = [sys.executable, str(output / "runtime/run_cell.py"),
                "--parent-cell", str(parent / "cells/mechanism" / dataset),
                "--package-root", str(parent / "runtime"), "--output", str(cell), "--phase", "run"]
            state(output, "running", completed=index, active_cell=dataset, command=command)
            # The worker owns the nonblocking shared GPU lock. A race with another
            # job fails visibly rather than interfering with that job or retrying.
            with (output / f"{dataset}.log").open("x") as log:
                subprocess.run(command, cwd=output / "runtime", stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            prediction = read(cell / "PREDICTIONS_SEALED.json")
            scored = read(cell / "SCORED_SEALED.json")
            verify_seal(cell, prediction)
            verify_seal(cell, scored)
            result = read(cell / "result.json")
            gate = read(cell / "replay_gate.json")
            if (gate.get("passed") is not True or gate.get("changed_label_pixels") != 0
                    or gate.get("tolerance") != 0 or gate.get("gt_contents_opened") is not False
                    or "replay_gate.json" not in prediction["artifact_sha256"]):
                raise RuntimeError("Cached TRACE equivalence gate is not a sealed exact-label pass")
            if (result["status"] != "completed" or result["images"] != images
                    or not result["full_split"] or result["dataset"] != dataset or result["seed"] != 0
                    or {(r["policy"], r.get("permutation_seed")) for r in result["policy_results"]}
                       != {("product", None), ("pixel_normalized", None)}
                    or len(result["policy_results"]) != 2
                    or scored["prediction_seal_sha256"] != digest(cell / "PREDICTIONS_SEALED.json")
                    or result["prediction_seal_sha256"] != scored["prediction_seal_sha256"]):
                raise RuntimeError("Cached control result is incomplete or not bound to its prediction seal")
            results[dataset] = result
            write(output / "results.json", results)
            state(output, "running", completed=index + 1)
        state(output, "completed", completed=2)
        files = ["manifest.json", "dependency.json", "results.json", "queue_state.json"]
        files += [f"cells/{d}/{name}" for d in DATASETS for name in
                  ("result.json", "replay_gate.json", "PREDICTIONS_SEALED.json", "SCORED_SEALED.json")]
        write(output / "SUITE_SEALED.json", dict(artifact_sha256={name: digest(output / name) for name in files}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=lambda s: Path(s).resolve(), required=True)
    parser.add_argument("--parent", type=lambda s: Path(s).resolve())
    parser.add_argument("--parent-service", default="trace-self-evolution-20261028-1338.service")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--parent-completion-receipt", type=lambda s: Path(s).resolve())
    parser.add_argument("--supersedes", type=lambda s: Path(s).resolve())
    args = parser.parse_args()
    if args.prepare:
        if not args.parent or args.output.exists():
            raise ValueError("Preparation requires a parent and a fresh output directory")
        args.output.mkdir(parents=True)
        shutil.copytree(Path(__file__).parent, args.output / "runtime",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(Path(__file__).parent.parent / "evolution_completion.py",
                     args.output / "runtime/evolution_completion.py")
        manifest = dict(parent=str(args.parent), parent_service=args.parent_service,
            parent_manifest_sha256=digest(args.parent / "manifest.json"),
            source_sha256=hashes(args.output / "runtime"), datasets=DATASETS,
            controls=["product", "pixel_normalized"], parameter_search=False,
            automatic_retry=False, candidate_cache_reuse=True,
            scope="Matched source-tracing aggregation and region-normalization controls, not standalone cost.")
        if args.parent_completion_receipt:
            sys.path.insert(0, str(args.output / "runtime"))
            from evolution_completion import receipt_spec, validate_receipt
            manifest["parent_completion_receipt"] = receipt_spec(args.parent_completion_receipt)
            validate_receipt(manifest["parent_completion_receipt"], args.parent)
        if args.supersedes:
            if read(args.supersedes / "queue_state.json")["status"] != "failed" or (args.supersedes / "cells").exists():
                raise RuntimeError("Recovery requires a failed controller with no started experiment cells")
            if digest(args.supersedes / "runtime/run_cell.py") != digest(args.output / "runtime/run_cell.py"):
                raise RuntimeError("Recovery changed the experimental worker")
            manifest["supersedes"] = dict(path=str(args.supersedes),
                manifest_sha256=digest(args.supersedes / "manifest.json"),
                failure_state_sha256=digest(args.supersedes / "queue_state.json"),
                reason="New controller admits an explicitly audited complete artifact suite after unit reclamation; worker unchanged.")
        write(args.output / "manifest.json", manifest)
        write(args.output / "MANIFEST_SEALED.json", dict(manifest_sha256=digest(args.output / "manifest.json")))
        state(args.output, "prepared", completed=0)
    else:
        try:
            execute(args.output)
        except BaseException as error:
            state(args.output, "failed", error=repr(error), automatic_retry=False)
            raise


if __name__ == "__main__":
    main()
