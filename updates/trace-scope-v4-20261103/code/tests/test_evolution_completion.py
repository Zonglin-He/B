"""CPU-only recovery admission tests using synthetic suites, not real payloads."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import evolution_completion as completion
from tools import self_evolution_controller as controller
from tools.evolution_replay import controller as replay


def write(path, value):
    path.write_text(json.dumps(value) + "\n")


def seal_index(root):
    write(root / "MANIFEST_SEALED.json", {
        "manifest_sha256": completion.digest(root / "manifest.json")})
    write(root / "SUITE_SEALED.json", {"artifact_sha256": {
        name: completion.digest(root / name)
        for name in ("manifest.json", "queue_state.json")}})


def make_receipt(root, path):
    write(path, dict(schema=1, root=str(root),
        basis="complete_artifacts_and_no_runtime_process",
        process_exit_status_certified=False, all_cells_verified=2,
        manifest_sha256=completion.digest(root / "manifest.json"),
        suite_seal_sha256=completion.digest(root / "SUITE_SEALED.json"),
        queue_state_sha256=completion.digest(root / "queue_state.json")))
    return completion.receipt_spec(path)


@pytest.fixture(autouse=True)
def isolated_imports_and_no_services(monkeypatch):
    # Both controller layouts must use the same mocked admission module.
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "evolution_completion", completion)

    def unexpected_service(*args, **kwargs):
        pytest.fail("Unexpected real subprocess/service access")

    monkeypatch.setattr(completion.subprocess, "run", unexpected_service)


@pytest.fixture
def completed(tmp_path, monkeypatch):
    root = tmp_path / "parent"
    runtime = root / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "worker.py").write_text("# synthetic frozen source\n")
    manifest = dict(cells=[dict(key=key, seed=0) for key in ("first", "second")],
        input_panels={key: {"image_ids": []} for key in ("first", "second")},
        image_sha256={}, runtime_hashes={
            "worker.py": completion.digest(runtime / "worker.py")})
    write(root / "manifest.json", manifest)
    write(root / "queue_state.json", dict(status="completed", completed=2, total=2))
    seal_index(root)
    path = tmp_path / "completion.json"
    specification = make_receipt(root, path)
    monkeypatch.setattr(completion, "runtime_processes", lambda _: [])
    return SimpleNamespace(root=root, path=path, spec=specification, manifest=manifest)


def test_valid_receipt_never_certifies_process_exit(completed):
    evidence = completion.validate_receipt(completed.spec, completed.root)
    assert evidence["process_exit_status_certified"] is False
    assert evidence["quiescence_checked"] is True
    assert evidence["basis"] == "complete_artifacts_and_no_runtime_process"
    assert evidence["receipt_sha256"] == completed.spec["sha256"]


def test_changed_receipt_bytes_rejected(completed):
    completed.path.write_text(completed.path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="receipt changed"):
        completion.validate_receipt(completed.spec, completed.root)


@pytest.mark.parametrize("field,value", [
    ("schema", 2), ("root", "/different/parent"), ("basis", "service_success"),
    ("process_exit_status_certified", True), ("process_exit_status_certified", 0),
    ("process_exit_status_certified", None), ("process_exit_status_certified", "false"),
    ("all_cells_verified", 1), ("manifest_sha256", "0" * 64),
    ("suite_seal_sha256", "0" * 64), ("queue_state_sha256", "0" * 64),
])
def test_rehashed_but_misbound_receipt_rejected(completed, field, value):
    receipt = completion.read(completed.path)
    receipt[field] = value
    write(completed.path, receipt)
    with pytest.raises(RuntimeError, match="does not bind"):
        completion.validate_receipt(completion.receipt_spec(completed.path), completed.root)


def test_receipt_requires_explicit_false_exit_certification(completed):
    receipt = completion.read(completed.path)
    del receipt["process_exit_status_certified"]
    write(completed.path, receipt)
    with pytest.raises(RuntimeError, match="does not bind"):
        completion.validate_receipt(completion.receipt_spec(completed.path), completed.root)


@pytest.mark.parametrize("name", ["manifest.json", "queue_state.json", "SUITE_SEALED.json"])
def test_changed_sealed_metadata_rejected(completed, name):
    path = completed.root / name
    path.write_text(path.read_text() + "\n")
    with pytest.raises(RuntimeError):
        completion.validate_receipt(completed.spec, completed.root)


@pytest.mark.parametrize("status,done,total", [
    ("failed", 2, 2), ("running", 2, 2), ("completed", 1, 2), ("completed", 2, 3),
])
def test_even_resealed_incomplete_queue_is_rejected(completed, status, done, total):
    write(completed.root / "queue_state.json", dict(status=status, completed=done, total=total))
    seal_index(completed.root)
    spec = make_receipt(completed.root, completed.path)
    with pytest.raises(RuntimeError, match="entire declared plan"):
        completion.validate_receipt(spec, completed.root)


def test_empty_declared_plan_rejected(completed):
    completed.manifest["cells"] = []
    write(completed.root / "manifest.json", completed.manifest)
    write(completed.root / "queue_state.json", dict(status="completed", completed=0, total=0))
    seal_index(completed.root)
    with pytest.raises(RuntimeError, match="entire declared plan"):
        completion.verify_index(completed.root)


@pytest.mark.parametrize("missing", ["manifest.json", "queue_state.json"])
def test_suite_seal_must_cover_manifest_and_queue(completed, missing):
    seal = completion.read(completed.root / "SUITE_SEALED.json")
    del seal["artifact_sha256"][missing]
    write(completed.root / "SUITE_SEALED.json", seal)
    with pytest.raises(RuntimeError, match="does not cover"):
        completion.verify_index(completed.root)


def test_suite_seal_cannot_escape_root(completed, tmp_path):
    outside = tmp_path / "outside.json"
    write(outside, {})
    seal = completion.read(completed.root / "SUITE_SEALED.json")
    seal["artifact_sha256"]["../outside.json"] = completion.digest(outside)
    write(completed.root / "SUITE_SEALED.json", seal)
    with pytest.raises(RuntimeError, match="seal changed"):
        completion.verify_index(completed.root)


@pytest.mark.parametrize("field", ["runtime_hashes", "source_sha256"])
@pytest.mark.parametrize("mutation", ["changed", "added", "deleted"])
def test_receipt_rechecks_frozen_sources(completed, field, mutation):
    if field == "source_sha256":
        completed.manifest[field] = completed.manifest.pop("runtime_hashes")
        write(completed.root / "manifest.json", completed.manifest)
        seal_index(completed.root)
        completed.spec = make_receipt(completed.root, completed.path)
    runtime = completed.root / "runtime"
    if mutation == "changed":
        (runtime / "worker.py").write_text("# changed after audit\n")
    elif mutation == "added":
        (runtime / "untracked.py").write_text("# added after audit\n")
    else:
        (runtime / "worker.py").unlink()
    with pytest.raises(RuntimeError, match="sources changed after audit"):
        completion.validate_receipt(completed.spec, completed.root)


@pytest.mark.parametrize("field", ["runtime_hashes", "source_sha256"])
def test_source_manifest_layout_and_bytecode_exclusions(completed, field):
    if field == "source_sha256":
        completed.manifest[field] = completed.manifest.pop("runtime_hashes")
        write(completed.root / "manifest.json", completed.manifest)
        seal_index(completed.root)
        completed.spec = make_receipt(completed.root, completed.path)
    runtime = completed.root / "runtime"
    cache = runtime / "__pycache__"
    cache.mkdir()
    (cache / "worker.cpython.pyc").write_bytes(b"bytecode")
    (runtime / "worker.pyc").write_bytes(b"bytecode")
    (runtime / "worker.pyo").write_bytes(b"bytecode")
    assert completion.validate_receipt(completed.spec, completed.root)


def test_missing_source_identity_rejected(completed):
    del completed.manifest["runtime_hashes"]
    write(completed.root / "manifest.json", completed.manifest)
    seal_index(completed.root)
    spec = make_receipt(completed.root, completed.path)
    with pytest.raises(RuntimeError, match="no frozen source identity"):
        completion.validate_receipt(spec, completed.root)


def test_receipt_rejects_active_upstream_process(completed, monkeypatch):
    monkeypatch.setattr(completion, "runtime_processes", lambda _: [81001])
    with pytest.raises(RuntimeError, match="still has runtime processes"):
        completion.validate_receipt(completed.spec, completed.root)


@pytest.fixture
def audit_stubs(completed, monkeypatch):
    fake = SimpleNamespace(verify_runtime=Mock(), verify_cell=Mock(return_value=dict(
        dataset="synthetic", images=0, full_split=True, seed=0)))
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=Mock()))
    monkeypatch.setattr(completion, "importlib", SimpleNamespace(util=SimpleNamespace(
        spec_from_file_location=lambda *args: spec, module_from_spec=lambda _: fake)))
    observed = dict(LoadState="not-found", MainPID="0", ActiveState="inactive",
                    SubState="dead", Result="success")

    def service_query(command, **kwargs):
        assert command[0] in {"systemctl", "journalctl"}
        assert "--user" in command and "parent.service" in command
        stdout = ("\n".join(f"{key}={value}" for key, value in observed.items())
                  if command[0] == "systemctl" else '{"MESSAGE":"synthetic journal"}\n')
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(completion.subprocess, "run", service_query)
    return fake, observed


@pytest.mark.parametrize("result", ["success", "exit-code"])
def test_audit_records_absence_not_service_exit(completed, audit_stubs, tmp_path, result):
    fake, observed = audit_stubs
    observed["Result"] = result
    output = tmp_path / "new_completion.json"
    completion.audit(completed.root, output, "parent.service")
    receipt = completion.read(output)
    assert receipt["process_exit_status_certified"] is False
    assert receipt["service_observation"] == observed
    assert "not exit evidence" in receipt["note"]
    assert receipt["all_cells_verified"] == len(receipt["cells"]) == 2
    fake.verify_runtime.assert_called_once_with(completed.root, completed.manifest)
    assert fake.verify_cell.call_count == 2
    assert completion.validate_receipt(completion.receipt_spec(output), completed.root)


@pytest.mark.parametrize("when", ["before", "during"])
def test_audit_rejects_active_or_restarted_upstream(completed, audit_stubs, tmp_path, monkeypatch, when):
    processes = iter([[81001]] if when == "before" else [[], [81001]])
    monkeypatch.setattr(completion, "runtime_processes", lambda _: next(processes))
    output = tmp_path / "new_completion.json"
    with pytest.raises(RuntimeError, match="runtime.*processes|runtime restarted"):
        completion.audit(completed.root, output, "parent.service")
    assert not output.exists()


@pytest.mark.parametrize("field,value", [("LoadState", "loaded"), ("MainPID", "81001")])
def test_audit_requires_reclaimed_absent_service(completed, audit_stubs, tmp_path, field, value):
    audit_stubs[1][field] = value
    output = tmp_path / "new_completion.json"
    with pytest.raises(RuntimeError, match="reclaimed, absent service"):
        completion.audit(completed.root, output, "parent.service")
    assert not output.exists()


def test_audit_never_overwrites_existing_receipt(completed):
    before = completed.path.read_bytes()
    with pytest.raises(FileExistsError, match="Never overwrite"):
        completion.audit(completed.root, completed.path, "parent.service")
    assert completed.path.read_bytes() == before


@pytest.fixture
def fake_proc(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    root = tmp_path / "suite"
    (root / "runtime").mkdir(parents=True)
    monkeypatch.setattr(completion, "Path", lambda value: proc if value == "/proc" else Path(value))
    return root, proc


def add_process(proc, pid, args):
    path = proc / str(pid)
    path.mkdir()
    (path / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in args) + b"\0")
    return path


@pytest.mark.parametrize("command", ["python", "/bin/sh"])
def test_process_detection_matches_runtime_not_sibling_prefix(fake_proc, command):
    root, proc = fake_proc
    runtime = str(root / "runtime")
    add_process(proc, 81001, [command, runtime + "/tools/worker.py"])
    add_process(proc, 81002, [command, "--package-root", runtime])
    add_process(proc, 81003, [command, runtime + "-other/worker.py"])
    (proc / "81004").mkdir()  # Process disappeared before cmdline could be read.
    (proc / "not-a-pid").mkdir()
    assert set(completion.runtime_processes(root)) == {81001, 81002}


def test_process_permission_errors_do_not_certify_quiescence(fake_proc, monkeypatch):
    root, proc = fake_proc
    process = add_process(proc, 81001, ["python"])
    original = Path.read_bytes

    def denied(path):
        if path == process / "cmdline":
            raise PermissionError("synthetic inaccessible process")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", denied)
    with pytest.raises(PermissionError):
        completion.runtime_processes(root)


@pytest.mark.parametrize("directory,expected", [
    ("runtime", [81001]), ("runtime/tools", [81001]), ("runtime-other", []),
])
def test_process_detection_rejects_relative_launch_from_runtime(fake_proc, directory, expected):
    root, proc = fake_proc
    process = add_process(proc, 81001, ["python", "tools/self_evolution_controller.py"])
    cwd = root / directory
    cwd.mkdir(parents=True, exist_ok=True)
    (process / "cwd").symlink_to(cwd, target_is_directory=True)
    assert completion.runtime_processes(root) == expected


@pytest.mark.parametrize("args", [
    ["python", "tools/self_evolution_controller.py"],
    ["/usr/bin/python3.12", "-m", "tools.self_evolution_controller"],
    ["pypy3", "tools/self_evolution_controller.py"],
])
def test_process_cwd_permission_errors_do_not_certify_quiescence(fake_proc, monkeypatch, args):
    root, proc = fake_proc
    process = add_process(proc, 81001, args)
    original = Path.resolve

    def denied(path, *args, **kwargs):
        if path == process / "cwd":
            raise PermissionError("synthetic inaccessible process cwd")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", denied)
    with pytest.raises(PermissionError):
        completion.runtime_processes(root)


def test_unrelated_nondumpable_daemon_does_not_require_cwd_access(fake_proc, monkeypatch):
    root, proc = fake_proc
    process = add_process(proc, 81001, ["/usr/bin/dbus-daemon", "--session"])
    original = Path.resolve

    def denied(path, *args, **kwargs):
        if path == process / "cwd":
            raise PermissionError("synthetic nondumpable unrelated daemon")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", denied)
    assert completion.runtime_processes(root) == []


@pytest.fixture(params=["replay", "suite"])
def gate(request, completed, tmp_path):
    output = tmp_path / "downstream"
    output.mkdir()
    if request.param == "replay":
        manifest = dict(parent=str(completed.root), parent_service="parent.service",
            parent_manifest_sha256=completion.digest(completed.root / "manifest.json"),
            parent_completion_receipt=completed.spec)
        return SimpleNamespace(wait=replay.wait_parent, output=output, manifest=manifest,
            binding=manifest, hash_key="parent_manifest_sha256",
            receipt_key="parent_completion_receipt", evidence=output / "dependency.json")
    dependency = dict(path=str(completed.root), service="parent.service", total=2,
        manifest_sha256=completion.digest(completed.root / "manifest.json"),
        completion_receipt=completed.spec)
    return SimpleNamespace(wait=controller.wait_dependencies, output=output,
        manifest=dict(cells=[{}], dependencies=[dependency]), binding=dependency,
        hash_key="manifest_sha256", receipt_key="completion_receipt",
        evidence=output / "dependencies.json")


def test_explicit_receipt_admission_preserves_non_exit_evidence(gate, monkeypatch):
    monkeypatch.setattr(replay, "service_state", lambda _: pytest.fail("Receipt must be explicit"))
    gate.wait(gate.output, gate.manifest)
    evidence = completion.read(gate.evidence)
    if isinstance(evidence, list):
        evidence = evidence[0]
    assert evidence["process_exit_status_certified"] is False


def test_missing_service_never_automatically_uses_available_receipt(gate, monkeypatch):
    del gate.binding[gate.receipt_key]
    text = "LoadState=not-found\nMainPID=0\nActiveState=inactive\nSubState=dead\nResult=success\n"
    monkeypatch.setattr(completion.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout=text, stderr=""))
    monkeypatch.setattr(completion, "validate_receipt", lambda *args:
        pytest.fail("No automatic receipt fallback is permitted"))
    with pytest.raises(RuntimeError, match="identity/state is unavailable"):
        gate.wait(gate.output, gate.manifest)
    assert not gate.evidence.exists()


def test_gate_rejects_receipt_changed_after_preparation(gate, completed):
    completed.path.write_text(completed.path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="receipt changed"):
        gate.wait(gate.output, gate.manifest)
    assert not gate.evidence.exists()


def test_receipt_gate_enforces_downstream_frozen_manifest_binding(gate):
    gate.binding[gate.hash_key] = "0" * 64
    with pytest.raises(RuntimeError, match="manifest changed"):
        gate.wait(gate.output, gate.manifest)
    assert not gate.evidence.exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_replay_preparation_freezes_only_explicit_receipt(completed, tmp_path, monkeypatch, explicit):
    output = tmp_path / "prepared_replay"
    args = ["controller", "--prepare", "--output", str(output), "--parent", str(completed.root)]
    if explicit:
        args += ["--parent-completion-receipt", str(completed.path)]
    monkeypatch.setattr(sys, "argv", args)
    replay.main()
    manifest = completion.read(output / "manifest.json")
    assert ("parent_completion_receipt" in manifest) is explicit
    if explicit:
        assert manifest["parent_completion_receipt"] == completed.spec
    assert manifest["parent_manifest_sha256"] == completion.digest(completed.root / "manifest.json")
    assert manifest["source_sha256"]["evolution_completion.py"] == completion.digest(
        output / "runtime/evolution_completion.py")
    assert completion.read(output / "MANIFEST_SEALED.json")["manifest_sha256"] == completion.digest(
        output / "manifest.json")
    assert manifest["automatic_retry"] is False
    assert not (output / "cells").exists()


def test_replay_recovery_retains_old_failed_directory(completed, tmp_path, monkeypatch):
    previous = tmp_path / "failed_replay"
    (previous / "runtime").mkdir(parents=True)
    write(previous / "manifest.json", {"original": True})
    write(previous / "queue_state.json", {"status": "failed"})
    (previous / "runtime/run_cell.py").write_bytes(
        Path(replay.__file__).with_name("run_cell.py").read_bytes())
    before = {p: p.read_bytes() for p in previous.rglob("*") if p.is_file()}
    output = tmp_path / "recovery"
    monkeypatch.setattr(sys, "argv", ["controller", "--prepare", "--output", str(output),
        "--parent", str(completed.root), "--parent-completion-receipt", str(completed.path),
        "--supersedes", str(previous)])
    replay.main()
    assert {p: p.read_bytes() for p in previous.rglob("*") if p.is_file()} == before
    supersedes = completion.read(output / "manifest.json")["supersedes"]
    assert supersedes["path"] == str(previous)
    assert supersedes["failure_state_sha256"] == completion.digest(previous / "queue_state.json")


@pytest.fixture
def freeze_args(completed, tmp_path, monkeypatch):
    package = tmp_path / "synthetic_package"
    for name in ("trace_ovss", "tools", "docs"):
        (package / name).mkdir(parents=True)
    (package / "tools/evolution_completion.py").write_text("# synthetic source identity\n")
    for name in ("self-evolution-contract.md", "self-evolution-initial-audit.md",
                 "self-evolution-runtime-audit.md"):
        (package / "docs" / name).write_text("synthetic fixture\n")
    monkeypatch.setitem(sys.modules, "suite_inputs", SimpleNamespace(expected_inputs=
        lambda runtime, data, cells, **kwargs: {key: {"image_ids": []} for key in cells}))
    return SimpleNamespace(output=tmp_path / "prepared_suite", package_root=package,
        data_root=tmp_path / "unused_data", originals=[], datasets=["voc20"], trace_only=True,
        dependency=[completed.root], dependency_service=["parent.service"],
        dependency_completion_receipt=[completed.path], supersedes=None)


@pytest.mark.parametrize("explicit", [False, True])
def test_suite_preparation_freezes_only_explicit_receipt(freeze_args, completed, explicit):
    if not explicit:
        freeze_args.dependency_completion_receipt = []
    controller.freeze(freeze_args)
    manifest = completion.read(freeze_args.output / "manifest.json")
    dependency = manifest["dependencies"][0]
    assert ("completion_receipt" in dependency) is explicit
    if explicit:
        assert dependency["completion_receipt"] == completed.spec
    assert dependency["manifest_sha256"] == completion.digest(completed.root / "manifest.json")
    assert manifest["runtime_hashes"]["tools/evolution_completion.py"] == completion.digest(
        freeze_args.output / "runtime/tools/evolution_completion.py")
    assert completion.read(freeze_args.output / "MANIFEST_SEALED.json")["manifest_sha256"] == completion.digest(
        freeze_args.output / "manifest.json")
    assert manifest["automatic_retry"] is False


@pytest.mark.parametrize("damage", ["duplicate", "unrequested"])
def test_suite_preparation_receipts_bind_uniquely_to_requested_roots(freeze_args, completed, damage):
    if damage == "duplicate":
        freeze_args.dependency_completion_receipt.append(completed.path)
    else:
        receipt = completion.read(completed.path)
        receipt["root"] = str(completed.root.parent / "unrequested")
        write(completed.path, receipt)
    with pytest.raises(ValueError, match="map uniquely"):
        controller.freeze(freeze_args)
    assert not (freeze_args.output / "MANIFEST_SEALED.json").exists()


def test_preparation_defers_quiescence_but_execution_rechecks_it(freeze_args, completed, monkeypatch):
    processes = Mock(return_value=[81001])
    monkeypatch.setattr(completion, "runtime_processes", processes)
    evidence = completion.validate_receipt(completed.spec, completed.root, require_quiescent=False)
    assert evidence["quiescence_checked"] is False
    assert evidence["process_exit_status_certified"] is False
    processes.assert_not_called()

    validate = Mock(wraps=completion.validate_receipt)
    monkeypatch.setattr(completion, "validate_receipt", validate)
    controller.freeze(freeze_args)
    validate.assert_called_once_with(completed.spec, completed.root, require_quiescent=False)
    processes.assert_not_called()
    output = freeze_args.output
    manifest = completion.read(output / "manifest.json")
    assert manifest["dependencies"][0]["completion_receipt"] == completed.spec
    assert not (output / "dependencies.json").exists()
    assert not (output / "cells").exists()

    with pytest.raises(RuntimeError, match="still has runtime processes"):
        controller.wait_dependencies(output, manifest)
    assert validate.call_args.kwargs == {}
    assert not (output / "dependencies.json").exists()
    assert not (output / "cells").exists()

    processes.return_value = []
    controller.wait_dependencies(output, manifest)
    admitted = completion.read(output / "dependencies.json")[0]
    assert admitted["quiescence_checked"] is True
    assert admitted["process_exit_status_certified"] is False


@pytest.mark.parametrize("damage", ["receipt", "source", "queue"])
def test_deferred_preparation_still_checks_evidence(freeze_args, completed, monkeypatch, damage):
    processes = Mock(return_value=[81001])
    monkeypatch.setattr(completion, "runtime_processes", processes)
    if damage == "receipt":
        receipt = completion.read(completed.path)
        receipt["manifest_sha256"] = "0" * 64
        write(completed.path, receipt)
    elif damage == "source":
        (completed.root / "runtime/worker.py").write_text("# changed after audit\n")
    else:
        write(completed.root / "queue_state.json", dict(status="completed", completed=1, total=2))
        seal_index(completed.root)
    with pytest.raises(RuntimeError):
        controller.freeze(freeze_args)
    processes.assert_not_called()
    assert not (freeze_args.output / "MANIFEST_SEALED.json").exists()


def test_replay_preparation_does_not_defer_quiescence(completed, tmp_path, monkeypatch):
    output = tmp_path / "strict_replay"
    monkeypatch.setattr(completion, "runtime_processes", lambda _: [81001])
    monkeypatch.setattr(sys, "argv", ["controller", "--prepare", "--output", str(output),
        "--parent", str(completed.root), "--parent-completion-receipt", str(completed.path)])
    with pytest.raises(RuntimeError, match="still has runtime processes"):
        replay.main()
    assert not (output / "MANIFEST_SEALED.json").exists()
    assert not (output / "cells").exists()


@pytest.mark.parametrize("receipt_position", [0, 1, 2])
@pytest.mark.parametrize("lingering_process", [False, True])
def test_all_live_consumers_finish_before_strict_receipt_admission(
        completed, tmp_path, monkeypatch, receipt_position, lingering_process):
    output = tmp_path / "dependent"
    output.mkdir()
    dependencies = []
    for index in range(2):
        consumer = tmp_path / f"consumer{index}"
        consumer.mkdir()
        write(consumer / "manifest.json", {"cells": [{}, {}]})
        write(consumer / "queue_state.json", dict(status="completed", completed=2, total=2))
        seal_index(consumer)
        dependencies.append(dict(path=str(consumer), service=f"consumer{index}.service", total=2,
            manifest_sha256=completion.digest(consumer / "manifest.json")))
    dependencies.insert(receipt_position, dict(path=str(completed.root), service="parent.service",
        total=2, manifest_sha256=completion.digest(completed.root / "manifest.json"),
        completion_receipt=completed.spec))
    manifest = dict(cells=[{}], dependencies=dependencies)
    phase = 0
    processes = Mock(side_effect=lambda _: [81001] if phase < 2 or lingering_process else [])
    monkeypatch.setattr(completion, "runtime_processes", processes)

    def service_state(name):
        assert name in {"consumer0.service", "consumer1.service"}
        active = phase == 0 or (phase == 1 and name == "consumer1.service")
        return dict(MainPID="81001" if active else "0", Result="success",
            ActiveState="active" if active else "inactive", SubState="running" if active else "dead")

    def finish_one_consumer(_):
        nonlocal phase
        processes.assert_not_called()
        assert not (output / "dependencies.json").exists()
        assert completion.read(output / "queue_state.json")["status"] == "waiting_for_dependencies"
        phase += 1
        assert phase <= 2

    monkeypatch.setattr(replay, "service_state", service_state)
    monkeypatch.setattr(controller.time, "sleep", finish_one_consumer)
    if lingering_process:
        with pytest.raises(RuntimeError, match="still has runtime processes"):
            controller.wait_dependencies(output, manifest)
        assert not (output / "dependencies.json").exists()
    else:
        controller.wait_dependencies(output, manifest)
        admitted = completion.read(output / "dependencies.json")
        assert len(admitted) == 3
        receipt_evidence = next(row for row in admitted if "receipt_sha256" in row)
        assert receipt_evidence["quiescence_checked"] is True
        assert receipt_evidence["process_exit_status_certified"] is False
    assert phase == 2
    processes.assert_called_once_with(completed.root)
    assert not (output / "cells").exists()
