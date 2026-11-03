from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.evolution_fifty import legacy_worker as worker
from tools.evolution_fifty.intake import sha256, write_new
from test_generalization_worker import TinyData
from test_generalization_runtime import method as base_method
from test_legacy_runtime import config


def make_method(spec):
    base = base_method()
    result = worker.legacy_runtime.LegacyExperimentTRACE(dataset_name=spec["config"]["dataset"],
        benchmark_classes=base.classes, query_to_class=None, classes=base.classes, lr=base.lr, steps=base.steps,
        trace_max_regions=base.trace_max_regions, trace_view_batch_size=base.trace_view_batch_size,
        vision_outputs=base.vision_outputs, prompts=base.prompt_templates, model=base.model,
        tokenize_fn=base.tokenize, device="cpu", view_mode="C")
    result.model.eval()
    return result


def fixture(tmp_path, monkeypatch):
    suite = tmp_path/"suite"
    suite.mkdir()
    runtime = suite/"runtime"
    runtime.mkdir()
    (runtime/"fixture.py").write_text("CPU_fixture=True\n")
    rows, specs, annotations = {}, [], {}
    for dataset, taxonomy in (("coco_object","COCOObjectDataset"), ("pc60","PascalContext60Dataset")):
        items = []
        for i in range(5):
            label, rgb = tmp_path/f"{dataset}.GT{i}", tmp_path/f"{dataset}.RGB{i}"
            label.write_bytes(b"Synthetic GT forbidden content")
            rgb.write_bytes(b"Synthetic RGB never decoded")
            state = label.stat()
            details = dict(device=state.st_dev, inode=state.st_ino, size_bytes=state.st_size,
                           mtime_ns=state.st_mtime_ns, contents_read=False)
            items.append(dict(official_id=f"source{i}", image=str(rgb), label=str(label),
                              label_metadata=[state.st_size,state.st_mtime_ns], label_metadata_details=details))
            annotations[str(label)] = details
        cfg = config()
        cfg["dataset"] = taxonomy
        rows[dataset] = items
        specs.append(dict(id="M03."+dataset, dataset=dataset, images=5, config=cfg,
            modes=list(worker.legacy_runtime.MODES), preprocessing="native_PIL_bilinear", backbone="ViT-L/14",
            model_type="naclip", checkpoint=dict(CPU_fixture=True),
            vocabulary=dict(classes=["an object","a surface"], queries=["an object","a surface"],
                            query_to_class=None, reduce_zero_label=False)))
        monkeypatch.setitem(worker.source.POPULATIONS, dataset, 5)
    components = worker.legacy_rho.load_components()
    source = dict(rows=rows, specs=specs, annotation_metadata=annotations, G_condition_ids=[],
        GT_opened=False, old_predictions_reused=False,
        prospective_analysis_sha256=worker.ANALYSIS_SHA,
        legacy_archive=dict(path=str(worker.legacy_rho.ARCHIVE), sha256=components["archive_sha256"],
            member_sha256=components["member_sha256"], episode_objective_sha256=worker.legacy_runtime.OBJECTIVE_SHA,
            old_engine_loaded=False, old_optimizer_loaded=False, old_OOM_iterator_selected=False))
    write_new(suite/"INPUTS.json", source)
    write_new(suite/"PROTOCOL.json", dict(CPU_fixture=True))
    prepared = dict(root=str(suite), schema_version=1, runtime_hashes={"fixture.py":sha256(runtime/"fixture.py")},
        inputs_sha256=sha256(suite/"INPUTS.json"), protocol_sha256=sha256(suite/"PROTOCOL.json"), parent_native_hashes={})
    write_new(suite/"SUITE_PREPARED.json", prepared)
    digest = sha256(suite/"SUITE_PREPARED.json")
    report = suite/"INDEPENDENT_REPORT.md"
    report.write_text("Synthetic CPU fixture, not a real admission\n")
    audit = dict(decision="ADMITTED", suite_prepared_sha256=digest, independent_reviewer="CPU fixture",
        M03_predictions_allowed=True, GT_scoring_allowed=False, report_path=str(report), report_sha256=sha256(report))
    write_new(suite/"ADMITTED.json", audit)
    monkeypatch.setattr(worker, "__file__", str(runtime/"fixture.py"))
    monkeypatch.setattr(worker, "source_inputs", lambda:copy.deepcopy(source))
    monkeypatch.setattr(worker.general, "runtime_imports", lambda *_:None)
    monkeypatch.setattr(worker, "native_identity", lambda:{})
    barriers = []
    monkeypatch.setattr(worker, "forbid_annotations", lambda paths:barriers.append(paths))
    def checkpoint(_): assert barriers
    monkeypatch.setattr(worker, "checked_checkpoint", checkpoint)
    monkeypatch.setattr(worker, "build_method", make_method)
    monkeypatch.setattr(worker.confirmation, "make_data", TinyData)
    monkeypatch.setattr(worker, "deterministic_runtime", lambda:torch.manual_seed(17))
    monkeypatch.setattr(worker, "prior_GPU_terminal", lambda:dict(CPU_fixture=True))
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda _:SimpleNamespace(free=256*1024**3))
    held = dict(active=False)
    @contextmanager
    def lock():
        assert not held["active"]
        held["active"] = True
        try: yield dict(CPU_fixture=True)
        finally: held["active"] = False
    monkeypatch.setattr(worker, "gpu_lock", lock)
    original = worker.publish_seal
    def publish(path, value):
        assert not held["active"], "M03 seal must follow shared-lock release"
        original(path, value)
    monkeypatch.setattr(worker, "publish_seal", publish)
    return suite, digest, source, audit, held


def test_entire_tiny_controller_actual_gates_all_predictions_then_seal_without_GT(tmp_path, monkeypatch):
    suite, digest, source, _, held = fixture(tmp_path, monkeypatch)
    original, forbidden = Path.open, set(source["annotation_metadata"])
    def guarded(path, *args, **kwargs):
        assert str(path) not in forbidden
        if "predictions" in path.parts: assert (suite/"ALL_GATES_PASSED.json").exists()
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    worker.run(suite, digest)
    seal = json.loads((suite/"ALL_PREDICTIONS_SEALED.json").read_text())
    assert seal["complete"] and seal["cells"] == 8 and seal["image_method_evaluations"] == 40
    assert not seal["GT_opened"] and not seal["scientific_completion"] and not held["active"]
    for dataset, child_hash in seal["dataset_seals"].items():
        child = suite/dataset/"PREDICTIONS_SEALED.json"
        assert sha256(child) == child_hash
        for name, digest_value in json.loads(child.read_text())["artifact_sha256"].items():
            assert sha256(suite/name) == digest_value
    assert not torch.cuda.is_initialized()
    with pytest.raises(ValueError, match="No M03 resume"):
        worker.run(suite, digest)


def test_running_prior_rejects_before_admission_or_STARTED(tmp_path, monkeypatch):
    suite, digest, *_ = fixture(tmp_path, monkeypatch)
    def blocked(): raise ValueError("Prior GPU active")
    monkeypatch.setattr(worker, "prior_GPU_terminal", blocked)
    monkeypatch.setattr(worker, "admission", lambda *_:pytest.fail("Admission after failed prerequisite"))
    with pytest.raises(ValueError, match="Prior GPU"):
        worker.run(suite, digest)
    assert not (suite/"STARTED.json").exists()


@pytest.mark.parametrize("fault", ["gate", "lock_release", "final_progress", "full_update"])
def test_failure_discards_path_without_recovery_or_global_seal(tmp_path, monkeypatch, fault):
    suite, digest, *_ = fixture(tmp_path, monkeypatch)
    def fail(*args, **kwargs): raise RuntimeError("synthetic M03 failure")
    if fault == "gate": monkeypatch.setattr(worker.legacy_gate, "compare_gate", fail)
    elif fault == "lock_release":
        @contextmanager
        def lock():
            yield dict(CPU_fixture=True)
            fail()
        monkeypatch.setattr(worker, "gpu_lock", lock)
    elif fault == "final_progress":
        original = worker.write_json
        def write(path, value, **kwargs):
            if value["phase"] == "M03_payloads_complete_awaiting_global_seal_and_exit": fail()
            return original(path, value, **kwargs)
        monkeypatch.setattr(worker, "write_json", write)
    else:
        methods = []
        original = worker.build_method
        def build(spec):
            method = original(spec)
            methods.append(method)
            return method
        monkeypatch.setattr(worker, "build_method", build)
        def update_then_fail(method, data, rows, spec, suite, budget, progress):
            images, meta, valid, ids = worker.general.episode_batch(data, 0, 2, method, spec["config"])
            worker.legacy_runtime.run_arm(method, "C.image", images, valid, ids, meta, spec["config"], None)
            monkeypatch.setattr(method, "reset", lambda:pytest.fail("Failed method reset/reused by outer controller"))
            fail()
        monkeypatch.setattr(worker.legacy_prediction, "execute_dataset", update_then_fail)
    with pytest.raises(RuntimeError, match="synthetic M03 failure"):
        worker.run(suite, digest)
    assert (suite/"FAILED.json").exists() and not (suite/"ALL_PREDICTIONS_SEALED.json").exists()
    with pytest.raises(ValueError, match="No M03 resume"):
        worker.run(suite, digest)


@pytest.mark.parametrize("fault", ["wrong_permission", "report_GT_alias", "prepared_GT_alias", "extra_runtime_GT_alias"])
def test_exact_distinct_admission_and_no_metadata_source_GT_alias(tmp_path, monkeypatch, fault):
    suite, digest, source, audit, _ = fixture(tmp_path, monkeypatch)
    if fault == "wrong_permission":
        audit.pop("M03_predictions_allowed")
        audit["M02_predictions_allowed"] = True
        (suite/"ADMITTED.json").write_text(json.dumps(audit))
    else:
        name = {"report_GT_alias":"INDEPENDENT_REPORT.md", "prepared_GT_alias":"SUITE_PREPARED.json",
                "extra_runtime_GT_alias":"runtime/extra.py"}[fault]
        path = suite/name
        if path.exists(): path.unlink()
        os.link(next(iter(source["annotation_metadata"])), path)
        original = Path.open
        def no_read(candidate, *args, **kwargs):
            assert candidate != path
            return original(candidate, *args, **kwargs)
        monkeypatch.setattr(Path, "open", no_read)
    with pytest.raises(ValueError): worker.admission(suite, digest)


def test_named_active_worker_is_not_an_idle_prerequisite(monkeypatch):
    monkeypatch.setattr(worker.source, "prior_GPU_terminal", lambda:dict(G_complete=True))
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **k:SimpleNamespace(
        stdout="LoadState=loaded\nActiveState=active\nSubState=running\nMainPID=17\n"))
    # Call the real function, not the synthetic-fixture replacement.
    with pytest.raises(ValueError, match="not idle"): worker.prior_GPU_terminal()


@pytest.mark.parametrize("kind", ["exact_archive", "corrupt_archive"])
def test_archive_annotation_hardlink_rejected_before_loader_hash_or_read(tmp_path, monkeypatch, kind):
    actual_source_inputs = worker.source_inputs
    suite, _, inputs, _, _ = fixture(tmp_path, monkeypatch)
    content = worker.legacy_rho.ARCHIVE.read_bytes() if kind == "exact_archive" else b"corrupt archive"
    archive = tmp_path/"aliased_source.tar.gz"
    label = tmp_path/"archive_annotation"
    archive.write_bytes(content)
    os.link(archive, label)
    state = label.stat()
    inputs["annotation_metadata"][str(label)] = dict(device=state.st_dev, inode=state.st_ino,
        size_bytes=state.st_size, mtime_ns=state.st_mtime_ns, contents_read=False)
    monkeypatch.setattr(worker.source, "source_inputs", lambda:copy.deepcopy(inputs))
    monkeypatch.setattr(worker.legacy_rho, "ARCHIVE", archive)
    barriers = []
    monkeypatch.setattr(worker, "forbid_annotations", lambda paths:barriers.append(paths))
    def forbidden(*args, **kwargs): pytest.fail("Archive/annotation reached loader or contents")
    monkeypatch.setattr(worker.legacy_rho, "load_components", forbidden)
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        if path in (archive, label): forbidden()
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="before read"):
        actual_source_inputs()
    assert barriers and str(label) in barriers[-1]
