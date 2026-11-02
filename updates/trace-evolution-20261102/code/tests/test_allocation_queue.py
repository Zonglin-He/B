"""Synthetic admission/controller checks; no models, data payloads or GPU."""

import copy
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.evolution_allocation import common as c
from tools.evolution_allocation import queue as q
from tools.evolution_allocation.analyze import contrasts, coverage_ratios


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    suite = tmp_path / "suite"
    runtime = suite / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "native.py").write_text("VALUE = 1\n")
    sources = {"native.py": c.sha256(runtime / "native.py")}
    parent = tmp_path / "parent"
    parent.mkdir()
    monkeypatch.setattr(c, "PARENT", parent)
    monkeypatch.setattr(c, "POPULATIONS", dict(coco_object=4, pc60=4))
    monkeypatch.setattr(q, "POPULATIONS", c.POPULATIONS)
    (suite / "parents").mkdir()
    c.write_json(suite / "parents/SUITE_PREPARED.json", dict(runtime_hashes=sources))
    monkeypatch.setattr(c, "PARENT_SHA", c.sha256(suite / "parents/SUITE_PREPARED.json"))
    artifacts = {"parents/SUITE_PREPARED.json": c.PARENT_SHA}
    for name in ("SUITE_SEALED.json", "ALL_CONFIGS_SELECTED.json", "ALL_FULL_PREDICTIONS_SEALED.json"):
        c.write_json(suite / "parents" / name, dict(complete=True))
        artifacts[f"parents/{name}"] = c.sha256(suite / "parents" / name)
    entries = {}
    checkpoint = dict(path=str(tmp_path / "checkpoint"), sha256="a" * 64)
    Path(checkpoint["path"]).write_text("synthetic checkpoint: stat only")
    for dataset in c.DATASETS:
        directory = suite / "parents" / dataset
        directory.mkdir()
        cfg = dict(dataset="COCOObjectDataset" if dataset == "coco_object" else "PascalContext60Dataset",
            resize=[224, 224], patch_size=[224, 224], patch_stride=112, class_extensions=True,
            batch_size=2, steps=3, trace_max_regions=4, trace_view_batch_size=16, lr=.001)
        inputs = [dict(image=str(tmp_path / f"{dataset}{i}.jpg"), label=str(tmp_path / f"{dataset}{i}.png"),
                       official_id=str(i), image_sha256="b" * 64, label_metadata=[10, 20]) for i in range(4)]
        for row in inputs:
            Path(row["image"]).write_text("synthetic RGB")
            Path(row["label"]).write_text("synthetic label: stat only in admission tests")
        panel = dict(inputs=inputs, ordered_ids=[str(i) for i in range(4)], classes=["a", "b"],
            queries=["a", "b"], query_to_class=[0, 1], reduce_zero_label=False, full_split=True, images=4)
        c.write_json(directory / "panel.json", panel)
        c.write_json(directory / "prediction_protocol.json", dict(manifest=dict(config=cfg, checkpoint=checkpoint, panel=panel)))
        c.write_json(directory / "SELECTED_CONFIG_SEALED.json", dict(config=cfg))
        for name in ("PREDICTIONS_SEALED.json", "SCORED_SEALED.json", "result.json"):
            c.write_json(directory / name, dict(complete=True))
        for path in directory.iterdir():
            artifacts[str(path.relative_to(suite))] = c.sha256(path)
        entries[dataset] = dict(config=cfg, panel_path=f"parents/{dataset}/panel.json",
                                panel_sha256=c.sha256(directory / "panel.json"))
    (suite / "frozen_contract.md").write_text("Locked scientific protocol\n")
    originals = c.parent_sources()
    for local, original in originals.items():
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(suite / local, original)
    c.write_json(suite / "PARENT_IMPORT.json", dict(parent_prepared_sha256=c.PARENT_SHA,
        verified_parent_full_outcomes=32, verified_parent_populations=8, source_importer_sha256="a" * 64,
        parent_metadata_sha256={str(original): c.sha256(original) for original in originals.values()}))
    prepared = dict(algorithm=c.ALGORITHM, datasets=list(c.DATASETS), arms=c.ARMS.copy(), permutation_seed=c.PERMUTATION_SEED,
        model_seed=0, parameter_search=False, retry=False, exposure=c.EXPOSURE, parent_suite=str(parent),
        parent_prepared_sha256=c.PARENT_SHA, contract_sha256=c.sha256(suite / "frozen_contract.md"),
        parent_proof_sha256=c.sha256(suite / "PARENT_IMPORT.json"), parent_artifact_sha256=artifacts,
        runtime_hashes=sources, entries=entries, checkpoint=checkpoint, python="/python",
        service_name="trace-allocation-controls-test.service", resources=c.RESOURCES)
    c.write_json(suite / "SUITE_PREPARED.json", prepared)
    identity = c.sha256(suite / "SUITE_PREPARED.json")
    (suite / "manifests").mkdir()
    for dataset in c.DATASETS:
        for stage in ("gate", "full"):
            c.write_json(c.manifest_path(suite, dataset, stage), c.expected_manifest(suite, prepared, dataset, stage, identity))
    return suite, prepared, identity


def test_exact_frozen_admission(frozen):
    suite, _, identity = frozen
    assert c.verify_prepared(suite, identity)["_verified_sha256"] == identity


@pytest.mark.parametrize("field,value", [("seed", True), ("seed", 1), ("permutation_seed", 123),
    ("arms", ["image_region"]), ("algorithm", "ImageBalancedTRACE:C"), ("stage", "calibration")])
def test_scope_changes_rejected(frozen, field, value):
    suite, _, _ = frozen
    manifest = c.read_json(c.manifest_path(suite, "pc60", "full"))
    manifest[field] = value
    with pytest.raises(RuntimeError):
        c.validate_manifest(manifest)


@pytest.mark.parametrize("key,value", [("lr", .002), ("steps", 1), ("batch_size", 1), ("trace_max_regions", 2)])
def test_no_parameter_change_even_in_valid_bounds(frozen, key, value):
    suite, _, identity = frozen
    path = c.manifest_path(suite, "coco_object", "full")
    manifest = c.read_json(path)
    manifest["config"][key] = value
    c.write_json(path, manifest, replace=True)
    with pytest.raises(RuntimeError, match="manifest differs"):
        c.verify_prepared(suite, identity)


@pytest.mark.parametrize("filename", ["startup.py", "__pycache__/native.pyc", "tools/extra.py"])
def test_runtime_extras_and_caches_rejected(frozen, filename):
    suite, _, identity = frozen
    path = suite / "runtime" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("unexpected")
    with pytest.raises(RuntimeError):
        c.verify_prepared(suite, identity)


def test_parent_panel_drift_rejected(frozen):
    suite, _, identity = frozen
    (suite / "parents/pc60/panel.json").write_text("{}")
    with pytest.raises(RuntimeError):
        c.verify_prepared(suite, identity)


def test_source_drift_rejected(frozen):
    suite, _, identity = frozen
    (suite / "runtime/native.py").write_text("VALUE = 2")
    with pytest.raises(RuntimeError):
        c.verify_prepared(suite, identity)


def test_gate_has_exact_four_rgb_rows(frozen):
    suite, _, _ = frozen
    gate = c.read_json(c.manifest_path(suite, "pc60", "gate"))
    assert len(gate["panel"]["inputs"]) == 4
    assert gate["panel"]["full_split"] is False
    assert gate["arms"] == c.ARMS
    assert gate["panel"]["images"] == 4
    c.validate_manifest(gate)


def test_foreign_runtime_rejected(frozen):
    suite, _, identity = frozen
    with pytest.raises(RuntimeError, match="frozen runtime"):
        c.verify_prepared(suite, identity, frozen=True)


def test_gate_cannot_request_score_barrier(frozen):
    suite, _, identity = frozen
    manifest = c.read_json(c.manifest_path(suite, "pc60", "gate"))
    with pytest.raises(RuntimeError, match="barrier path/stage"):
        c.verify_barrier(suite / "ALL_FULL_PREDICTIONS_SEALED.json", manifest, Path(manifest["output"]), identity)


def test_partial_barrier_rejected_before_worker_import(frozen):
    suite, _, identity = frozen
    manifest = c.read_json(c.manifest_path(suite, "pc60", "full"))
    path = suite / "ALL_FULL_PREDICTIONS_SEALED.json"
    c.write_json(path, dict(complete=True, stage="full", gt_contents_opened=False,
        suite_prepared_sha256=identity, prediction_seal_sha256={str(suite / "full/pc60"): "a" * 64}))
    with pytest.raises(RuntimeError, match="both exact full"):
        c.verify_barrier(path, manifest, Path(manifest["output"]), c.sha256(c.manifest_path(suite, "pc60", "full")))


def test_restart_rejected_without_child(frozen, monkeypatch):
    suite, prepared, identity = frozen
    prepared = dict(prepared, _verified_sha256=identity)
    monkeypatch.setattr(q, "verify_prepared", lambda *args, **kwargs: prepared)
    c.write_json(suite / "EXECUTION_STARTED.json", dict(started=True))
    monkeypatch.setattr(q, "child", lambda *args, **kwargs: pytest.fail("must not run a child"))
    with pytest.raises(RuntimeError, match="restart/resume"):
        q.run(SimpleNamespace(suite=suite, suite_sha256=identity))


def test_child_sets_cpu_for_score_and_disallows_retry(frozen, monkeypatch):
    suite, prepared, identity = frozen
    prepared = dict(prepared, _verified_sha256=identity)
    monkeypatch.setattr(q, "verify_prepared", lambda *args, **kwargs: prepared)
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(q.subprocess, "run", fake_run)
    q.child(suite, prepared, "score", "pc60", "full", barrier=suite / "boundary")
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert calls[0][1]["cwd"] == suite / "runtime"
    assert "--barrier" in calls[0][0]
    with pytest.raises(RuntimeError, match="retry"):
        q.child(suite, prepared, "score", "pc60", "full")


def test_failed_child_marks_suite_failed_and_never_scores(frozen, monkeypatch):
    suite, prepared, identity = frozen
    prepared = dict(prepared, _verified_sha256=identity)
    monkeypatch.setattr(q, "verify_prepared", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(q, "execution_identity", lambda *args: dict(pid=1))
    phases = []
    def failure(*args, **kwargs):
        phases.append(args[2])
        raise RuntimeError("gate failure")
    monkeypatch.setattr(q, "child", failure)
    with pytest.raises(RuntimeError, match="gate failure"):
        q.run(SimpleNamespace(suite=suite, suite_sha256=identity))
    assert phases == ["gate"]
    assert c.read_json(suite / "queue_state.json")["phase"] == "failed"
    assert not (suite / "ALL_FULL_PREDICTIONS_SEALED.json").exists()


def test_signed_contrasts_do_not_choose_winner():
    result = contrasts(dict(zip(c.ARMS, [10., 12., 8., 9., 11.])))
    assert result == dict(image_minus_episode_region=-2., image_minus_episode_token=-1.,
        region_minus_token_image=2., region_minus_token_episode=3.,
        normalization_by_pooling_interaction=-1., native_minus_shuffled=-1.)


def test_missing_negative_control_cannot_be_omitted():
    with pytest.raises(RuntimeError):
        contrasts({"image_region": 10.})


def test_permutation_coverage_not_fixed_by_force():
    counts = dict(active_candidates=10, active_sources=5, eligible_candidates=7, eligible_sources=2,
        moved_candidates=4, moved_sources=1, changed_score_candidates=2, changed_score_sources=1)
    actual = coverage_ratios(counts)
    assert actual["eligible_source_fraction"] == .4
    assert actual["changed_score_fraction_of_eligible"] == 2 / 7
    bad = dict(counts, changed_score_candidates=8)
    with pytest.raises(RuntimeError):
        coverage_ratios(bad)


def repack(suite, prepared):
    c.write_json(suite / "SUITE_PREPARED.json", prepared, replace=True)
    identity = c.sha256(suite / "SUITE_PREPARED.json")
    for dataset in c.DATASETS:
        for stage in ("gate", "full"):
            c.write_json(c.manifest_path(suite, dataset, stage),
                         c.expected_manifest(suite, prepared, dataset, stage, identity), replace=True)
    return identity


def test_coherent_parent_repack_cannot_change_selected_lr(frozen):
    suite, prepared, _ = frozen
    prepared["entries"]["pc60"]["config"]["lr"] = .003
    for name in ("prediction_protocol.json", "SELECTED_CONFIG_SEALED.json"):
        path = suite / "parents/pc60" / name
        record = c.read_json(path)
        (record["manifest"] if name.startswith("prediction") else record)["config"]["lr"] = .003
        c.write_json(path, record, replace=True)
        prepared["parent_artifact_sha256"][str(path.relative_to(suite))] = c.sha256(path)
    identity = repack(suite, prepared)
    with pytest.raises(RuntimeError, match="fixed parent read-set"):
        c.verify_prepared(suite, identity)


def test_missing_parent_chain_artifact_rejected_even_with_rehashed_prepare(frozen):
    suite, prepared, _ = frozen
    del prepared["parent_artifact_sha256"]["parents/ALL_CONFIGS_SELECTED.json"]
    identity = repack(suite, prepared)
    with pytest.raises(RuntimeError, match="parent chain"):
        c.verify_prepared(suite, identity)


def test_symlinked_logs_do_not_write_outside_suite(frozen, tmp_path, monkeypatch):
    suite, prepared, identity = frozen
    prepared = dict(prepared, _verified_sha256=identity)
    external = tmp_path / "protected"
    external.mkdir()
    (suite / "logs").symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(q, "verify_prepared", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(q.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not start child"))
    with pytest.raises(RuntimeError, match="symlink"):
        q.child(suite, prepared, "gate", "pc60", "gate")
    assert list(external.iterdir()) == []


def test_state_temporary_alias_rejected(frozen, tmp_path):
    suite, _, _ = frozen
    protected = tmp_path / "original"
    protected.write_text("preserve")
    (suite / "queue_state.json.tmp").symlink_to(protected)
    with pytest.raises(RuntimeError):
        q.state(suite, "running")
    assert protected.read_text() == "preserve"


def test_output_must_be_isolated_direct_child(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "OUTPUT_BASE", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    q.validate_output_root(tmp_path / "trace_allocation_controls_new", source)
    with pytest.raises(RuntimeError):
        q.validate_output_root(source / "trace_allocation_controls_nested", source)
    with pytest.raises(RuntimeError):
        q.validate_output_root(tmp_path / "wrong_name", source)


def service_fixture(suite, prepared):
    prepared = dict(prepared, _verified_sha256=c.sha256(suite / "SUITE_PREPARED.json"))
    service = prepared["service_name"]
    execution = dict(pid=1234, command=q.expected_command(suite, prepared),
        suite_prepared_sha256=prepared["_verified_sha256"], retry_allowed=False, service=service,
        cwd=str(suite / "runtime"), invocation_id="b" * 32, control_group="/user/app/" + service)
    actual = dict(Id=service, LoadState="loaded", ActiveState="active", SubState="exited", MainPID="0",
        Result="success", ExecMainCode="1", ExecMainStatus="0", ExecMainPID="1234",
        ExecStart="{ argv[]=" + " ".join(execution["command"]) + " ; ignore_errors=no; }",
        WorkingDirectory=execution["cwd"], ControlGroup=execution["control_group"], InvocationID=execution["invocation_id"],
        ExecMainStartTimestamp="start", ExecMainExitTimestamp="end", **c.RESOURCES)
    execution["resources"] = {key: actual[key] for key in
        (*c.RESOURCES, "ControlGroup", "InvocationID", "ExecMainPID", "WorkingDirectory", "ExecStart")}
    return prepared, execution, actual


@pytest.mark.parametrize("field,value", [("ExecMainPID", "999"), ("InvocationID", "c" * 32),
    ("WorkingDirectory", "/elsewhere"), ("ExecStart", "unrelated"), ("ControlGroup", "/other"),
    ("MainPID", "1234"), ("SubState", "running"), ("MemoryMax", "999"), ("Id", "other.service"),
    ("ExecMainStatus", "1"), ("ExecMainCode", "2")])
def test_successful_but_unrelated_service_rejected(frozen, monkeypatch, field, value):
    suite, prepared, _ = frozen
    prepared, execution, actual = service_fixture(suite, prepared)
    actual[field] = value
    monkeypatch.setattr(q.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(stdout="\n".join(f"{k}={v}" for k, v in actual.items())))
    with pytest.raises(RuntimeError):
        q.bound_service_terminal(suite, prepared, execution, prepared["service_name"])


def test_retained_success_service_matches_exact_invocation(frozen, monkeypatch):
    suite, prepared, _ = frozen
    prepared, execution, actual = service_fixture(suite, prepared)
    monkeypatch.setattr(q.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(stdout="\n".join(f"{k}={v}" for k, v in actual.items())))
    assert q.bound_service_terminal(suite, prepared, execution, prepared["service_name"]) == actual
    with pytest.raises(RuntimeError, match="Unrelated"):
        q.bound_service_terminal(suite, prepared, execution, "unrelated-success.service")


def test_retired_empty_cgroup_requires_same_invocation_and_startup_proof(frozen, monkeypatch):
    suite, prepared, _ = frozen
    prepared, execution, actual = service_fixture(suite, prepared)
    actual["ControlGroup"] = ""
    monkeypatch.setattr(q.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(stdout="\n".join(f"{k}={v}" for k, v in actual.items())))
    assert q.bound_service_terminal(suite, prepared, execution, prepared["service_name"])["ControlGroup"] == ""
    execution["resources"]["ControlGroup"] = "/forged"
    with pytest.raises(RuntimeError, match="Startup"):
        q.bound_service_terminal(suite, prepared, execution, prepared["service_name"])


def test_bare_gate_boolean_is_not_equivalence_evidence(frozen):
    suite, prepared, identity = frozen
    prepared = dict(prepared, _verified_sha256=identity)
    output = suite / "gates/pc60"
    output.mkdir(parents=True)
    c.write_json(output / "gate.json", dict(passed=True))
    c.write_json(output / "GATE_PASSED.json", dict(passed=True, gate_sha256=c.sha256(output / "gate.json")))
    with pytest.raises((KeyError, RuntimeError)):
        q.validate_gate(suite, prepared, "pc60")


def test_unlisted_canonical_result_rejected_before_reading(frozen):
    suite, prepared, identity = frozen
    prepared = dict(prepared, _verified_sha256=identity)
    c.write_json(suite / "ALL_FULL_PREDICTIONS_SEALED.json", dict(complete=True))
    output = suite / "full/coco_object"
    output.mkdir(parents=True)
    c.write_json(output / "PREDICTIONS_SEALED.json", dict(complete=True))
    c.write_json(output / "SCORED_SEALED.json", dict(complete=True,
        barrier_sha256=c.sha256(suite / "ALL_FULL_PREDICTIONS_SEALED.json"),
        prediction_seal_sha256=c.sha256(output / "PREDICTIONS_SEALED.json"), artifact_sha256={"placeholder": "a" * 64}))
    with pytest.raises(RuntimeError, match="canonical score inventory"):
        q.validate_results(suite, prepared)


def test_importer_executes_authenticated_source_not_timestamp_cache(tmp_path):
    import os
    import py_compile
    source = tmp_path / "paper_evolution/tools/export_tuned_full.py"
    source.parent.mkdir(parents=True)
    text = ('from pathlib import Path\nimport hashlib\ndef import_suite(root):\n'
        '    return dict(marker="STALE", importer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())\n')
    source.write_text(text)
    status = source.stat()
    py_compile.compile(str(source), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    source.write_text(text.replace("STALE", "FRESH"))
    os.utime(source, ns=(status.st_atime_ns, status.st_mtime_ns))
    assert q.parent_report(tmp_path)["marker"] == "FRESH"


@pytest.mark.parametrize("aliased", ["barrier", "peer_seal"])
def test_common_barrier_rejects_annotation_before_hash(frozen, monkeypatch, aliased):
    suite, _, identity = frozen
    manifest_path = c.manifest_path(suite, "pc60", "full")
    manifest = c.read_json(manifest_path)
    for dataset in c.DATASETS:
        output = suite / "full" / dataset
        output.mkdir(parents=True)
        c.write_json(output / "PREDICTIONS_SEALED.json", dict(complete=True))
    boundary = suite / "ALL_FULL_PREDICTIONS_SEALED.json"
    c.write_json(boundary, dict(complete=True, stage="full", gt_contents_opened=False,
        suite_prepared_sha256=identity, prediction_seal_sha256={str(suite / "full" / d): "a" * 64 for d in c.DATASETS}))
    victim = boundary if aliased == "barrier" else suite / "full/coco_object/PREDICTIONS_SEALED.json"
    victim.unlink()
    victim.hardlink_to(manifest["panel"]["inputs"][0]["label"])
    original_hash = c.sha256
    gt = Path(manifest["panel"]["inputs"][0]["label"]).stat()
    def no_label_hash(path):
        status = Path(path).stat()
        assert (status.st_dev, status.st_ino) != (gt.st_dev, gt.st_ino), "GT must not reach hashing"
        return original_hash(path)
    monkeypatch.setattr(c, "sha256", no_label_hash)
    with pytest.raises(RuntimeError, match="aliases annotation"):
        c.verify_barrier(boundary, manifest, Path(manifest["output"]), original_hash(manifest_path))


@pytest.mark.parametrize("fault", [None, "invocation", "parent_pid", "cgroup", "memory"])
def test_gpu_child_requires_same_live_controller_envelope(frozen, monkeypatch, fault):
    suite, prepared, _ = frozen
    prepared, execution, actual = service_fixture(suite, prepared)
    actual.update(ActiveState="active", SubState="running", MainPID=str(execution["pid"]))
    c.write_json(suite / "EXECUTION_STARTED.json", execution)
    manifest = c.read_json(c.manifest_path(suite, "pc60", "full"))
    monkeypatch.setenv("INVOCATION_ID", "c" * 32 if fault == "invocation" else execution["invocation_id"])
    monkeypatch.setattr(c.os, "getppid", lambda: 999 if fault == "parent_pid" else execution["pid"])
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: suite / "runtime"))
    original_read = Path.read_text
    def read(path, *args, **kwargs):
        if str(path) == "/proc/self/cgroup":
            return "0::" + ("/outside" if fault == "cgroup" else execution["control_group"]) + "\n"
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    if fault == "memory":
        actual["MemoryMax"] = "infinity"
    monkeypatch.setattr(c.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(stdout="\n".join(f"{k}={v}" for k, v in actual.items())))
    if fault is None:
        assert c.require_gpu_envelope(manifest) == actual
    else:
        with pytest.raises(RuntimeError):
            c.require_gpu_envelope(manifest)
