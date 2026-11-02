"""Explicit artifact-completion audit for a reclaimed transient service.

A missing systemd unit cannot certify exit status. This opt-in receipt instead
records verified scientific completion and absence of runtime processes. It is
never an automatic fallback for an unavailable or failed service query.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_index(root):
    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    if digest(root / "manifest.json") != read(root / "MANIFEST_SEALED.json")["manifest_sha256"]:
        raise RuntimeError("Completed dependency manifest changed")
    entries = read(root / "SUITE_SEALED.json").get("artifact_sha256", {})
    if not {"manifest.json", "queue_state.json"} <= set(entries):
        raise RuntimeError("Dependency suite seal does not cover its manifest and status")
    for relative, expected in entries.items():
        path = root / relative
        if not path.resolve().is_relative_to(root) or digest(path) != expected:
            raise RuntimeError(f"Completed dependency seal changed: {relative}")
    total = len(manifest.get("cells", manifest.get("datasets", [])))
    status = read(root / "queue_state.json")
    if not total or status.get("status") != "completed" or status.get("completed") != total or status.get("total") != total:
        raise RuntimeError("Dependency did not complete its entire declared plan")
    return manifest, total


def verify_sources(root, manifest):
    expected = manifest.get("runtime_hashes", manifest.get("source_sha256", {}))
    if not expected:
        raise RuntimeError("Dependency has no frozen source identity")
    runtime = Path(root) / "runtime"
    actual = {str(path.relative_to(runtime)): digest(path) for path in runtime.rglob("*")
              if path.is_file() and "__pycache__" not in path.parts
              and path.suffix not in {".pyc", ".pyo"}}
    if actual != expected:
        raise RuntimeError("Completed dependency sources changed after audit")


def runtime_processes(root):
    runtime = str((Path(root) / "runtime").resolve())
    found = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            args = (path / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        # Permission failure must not silently certify quiescence.
        if any(arg == runtime or arg.startswith(runtime + "/") for arg in args):
            found.append(int(path.name))
            continue
        relative_python = bool(args and args[0] and (
            Path(args[0]).name.startswith("python") or any(
                arg.endswith(".py") and not Path(arg).is_absolute() for arg in args if arg)))
        try:
            if relative_python and path.stat().st_uid == os.getuid():
                cwd = (path / "cwd").resolve(strict=True)
                if cwd.is_relative_to(Path(runtime)):
                    found.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def receipt_spec(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path))


def validate_receipt(specification, root, *, require_quiescent=True):
    root = Path(root).resolve()
    path = Path(specification["path"])
    if digest(path) != specification["sha256"]:
        raise RuntimeError("Completion receipt changed after preparation")
    receipt = read(path)
    manifest, total = verify_index(root)
    verify_sources(root, manifest)
    if (receipt.get("schema") != 1 or receipt.get("root") != str(root)
            or receipt.get("basis") != "complete_artifacts_and_no_runtime_process"
            or receipt.get("process_exit_status_certified") is not False
            or receipt.get("all_cells_verified") != total
            or receipt.get("manifest_sha256") != digest(root / "manifest.json")
            or receipt.get("suite_seal_sha256") != digest(root / "SUITE_SEALED.json")
            or receipt.get("queue_state_sha256") != digest(root / "queue_state.json")):
        raise RuntimeError("Completion receipt does not bind this completed dependency")
    if require_quiescent and runtime_processes(root):
        raise RuntimeError("Completed dependency still has runtime processes")
    return dict(root=str(root), basis=receipt["basis"], receipt_sha256=digest(path),
                suite_seal_sha256=receipt["suite_seal_sha256"],
                process_exit_status_certified=False, quiescence_checked=require_quiescent)


def audit(root, output, service):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Never overwrite an existing recovery audit")
    manifest, total = verify_index(root)
    if runtime_processes(root):
        raise RuntimeError("Cannot audit while original runtime processes remain")
    spec = importlib.util.spec_from_file_location("sealed_main_controller",
        root / "runtime/tools/self_evolution_controller.py")
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    controller.verify_runtime(root, manifest)
    verified = []
    for cell in manifest["cells"]:
        panel = dict(manifest["input_panels"][cell["key"]], seed=cell["seed"],
            image_sha256={p: manifest["image_sha256"][p]
                          for p in manifest["input_panels"][cell["key"]]["image_ids"]})
        result = controller.verify_cell(root / "cells" / cell["key"], cell, panel, manifest)
        verified.append(dict(key=cell["key"], dataset=result["dataset"], images=result["images"],
                             full_split=result["full_split"], seed=result["seed"]))
    for image, expected in manifest["image_sha256"].items():
        if digest(image) != expected:
            raise RuntimeError("Original input image changed")
    verify_index(root)
    if runtime_processes(root):
        raise RuntimeError("Original runtime restarted during completion audit")
    query = subprocess.run(["systemctl", "--user", "show", service,
        "--property=LoadState,MainPID,ActiveState,SubState,Result"], capture_output=True, text=True)
    observed = dict(line.split("=", 1) for line in query.stdout.splitlines() if "=" in line)
    if query.returncode or observed.get("LoadState") != "not-found" or observed.get("MainPID") != "0":
        raise RuntimeError("This explicit audit is only for a reclaimed, absent service")
    journal = subprocess.run(["journalctl", "--user", "-u", service, "-n", "20",
        "--no-pager", "-o", "json"], capture_output=True, text=True, check=True)
    receipt = dict(schema=1, created_at=datetime.now(timezone.utc).isoformat(), root=str(root),
        basis="complete_artifacts_and_no_runtime_process", process_exit_status_certified=False,
        note="Unit was reclaimed. Its displayed default Result=success is not exit evidence.",
        manifest_sha256=digest(root / "manifest.json"),
        suite_seal_sha256=digest(root / "SUITE_SEALED.json"),
        queue_state_sha256=digest(root / "queue_state.json"), all_cells_verified=total,
        cells=verified, service_name=service, service_observation=observed,
        journal=[json.loads(line) for line in journal.stdout.splitlines() if line.strip()])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(receipt=str(output), all_cells_verified=total,
                         process_exit_status_certified=False)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service", required=True)
    args = parser.parse_args()
    audit(args.root, args.output, args.service)


if __name__ == "__main__":
    main()
