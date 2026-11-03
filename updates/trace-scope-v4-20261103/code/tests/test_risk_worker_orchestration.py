from contextlib import contextmanager
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.evolution_fifty import risk_worker as worker
from tools.evolution_fifty.intake import sha256, write_new
from test_unit_risk import AuditMethod, candidate


class CPUFixtureMethod(AuditMethod):
    @contextmanager
    def count_work(self):
        yield {"CPU_fixture_actual_optimizer": True}


def setup_cpu_prediction(tmp_path, monkeypatch):
    prepared = dict(forbidden_labels=[], datasets={})
    panels = tmp_path/"panels"
    panels.mkdir()
    gates = {}
    for dataset in ("coco_object", "pc60"):
        examples = [dict(source_id="source", candidate_id=f"id{i}", episode=0, candidate=candidate()) for i in range(2)]
        write_new(panels/f"{dataset}.panel.json", dict(examples=examples, candidate_count=2))
        rows = [dict(image="fixtureRGB0"), dict(image="fixtureRGB1")]
        write_new(panels/f"{dataset}.episodes.json", {"0": rows})
        prepared["datasets"][dataset] = dict(candidate_count=2, native_config=dict(patch_size=[4,4],patch_stride=4),
                                             query_to_class=None)
        gate = tmp_path/f"{dataset}.GATE_PASSED.json"
        write_new(gate, dict(passed=True, CPU_fixture_only=True))
        gates[dataset] = sha256(gate)
    expected = "f"*64
    write_new(tmp_path/"ALL_GATES_PASSED.json", dict(suite_prepared_sha256=expected, gates=gates))
    monkeypatch.setattr(worker, "admission", lambda *_: (tmp_path, prepared))
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda _: SimpleNamespace(free=32*1024**3))
    monkeypatch.setattr(worker, "forbid_annotations", lambda *_: None)
    @contextmanager
    def fixture_lock():
        yield dict(path="CPU_fixture_only", pid=0)
    monkeypatch.setattr(worker, "gpu_lock", fixture_lock)
    monkeypatch.setattr(worker, "deterministic_runtime", lambda: torch.manual_seed(7))
    monkeypatch.setattr(worker, "build_method", lambda _: CPUFixtureMethod(steps=2))
    monkeypatch.setattr(worker.confirmation, "make_data", lambda *_: [0, 1])
    image = torch.randn((2,3,4,4), generator=torch.Generator().manual_seed(3))
    meta = dict(img_path=["fixtureRGB0", "fixtureRGB1"], img_shape=[(4,4),(4,4)], patch_grid_shape=[(1,1),(1,1)])
    monkeypatch.setattr(worker.shared, "image_collate", lambda _: dict(img_patches=image, meta=meta))
    monkeypatch.setattr(worker, "verify_cached_candidates", lambda *_: None)
    monkeypatch.setattr(worker, "source_prediction", lambda m, x, *args:
                        m.forward_adapt_logits(x)[0, 1].argmax(0).detach().numpy().astype(np.uint16))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return expected


def test_two_dataset_prediction_executes_real_filesystem_progress_overwrite_and_barrier(tmp_path, monkeypatch):
    expected = setup_cpu_prediction(tmp_path, monkeypatch)
    worker.predict(tmp_path, expected)
    state = json.loads((tmp_path/"queue_state.json").read_text())
    assert state["phase"] == "predictions_sealed_awaiting_offline_support_replay"
    assert state["scientific_completion"] is False and state["labels_read"] is False
    barrier = json.loads((tmp_path/"ALL_RISK_PREDICTIONS_SEALED.json").read_text())
    assert set(barrier["dataset_seals"]) == {"coco_object", "pc60"}
    for dataset in barrier["dataset_seals"]:
        seal_path = tmp_path/dataset/"PREDICTIONS_SEALED.json"
        assert sha256(seal_path) == barrier["dataset_seals"][dataset]
        seal = json.loads(seal_path.read_text())
        assert seal["candidates"] == 2 and seal["gt_contents_opened"] is False
        for name, digest in seal["artifact_sha256"].items():
            assert sha256(tmp_path/dataset/name) == digest
        for index in range(2):
            case = tmp_path/dataset/f"case_{index:06d}"
            record = json.loads((case/"trajectory.json").read_text())
            assert record["actual_steps"] == 2 and record["update_weight"] == 1
            assert len(list(case.glob("*.labels.npy"))) == 3
            assert len(list(case.glob("*.visual_ln.npz"))) == 3
    assert not (tmp_path/"FAILED.json").exists()
    assert torch.cuda.is_initialized() is False
    with pytest.raises(ValueError, match="No risk resume/retry"):
        worker.predict(tmp_path, expected)


def test_progress_failure_is_preserved_and_never_creates_prediction_barrier(tmp_path, monkeypatch):
    expected = setup_cpu_prediction(tmp_path, monkeypatch)
    def broken_writer(*args, **kwargs):
        raise TypeError("injected progress interface failure")
    monkeypatch.setattr(worker, "write_json", broken_writer)
    with pytest.raises(TypeError, match="progress interface"):
        worker.predict(tmp_path, expected)
    failed = json.loads((tmp_path/"FAILED.json").read_text())
    assert failed["retry_allowed"] is False
    assert not (tmp_path/"ALL_RISK_PREDICTIONS_SEALED.json").exists()
    assert (tmp_path/"coco_object"/"case_000000"/"trajectory.json").is_file()
    with pytest.raises(ValueError, match="No risk resume/retry"):
        worker.predict(tmp_path, expected)


def test_insufficient_disk_rejected_before_start_lock_or_model(tmp_path, monkeypatch):
    expected = setup_cpu_prediction(tmp_path, monkeypatch)
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda _: SimpleNamespace(free=16*1024**3-1))
    def unexpected_call(*args, **kwargs):
        pytest.fail("Disk preflight must reject before GPU lock or model construction")
    monkeypatch.setattr(worker, "gpu_lock", unexpected_call)
    monkeypatch.setattr(worker, "build_method", unexpected_call)
    with pytest.raises(ValueError, match="Insufficient risk checkpoint disk budget"):
        worker.predict(tmp_path, expected)
    for name in ("STARTED.json", "FAILED.json", "GPU_LOCK_ACQUIRED.json", "ALL_RISK_PREDICTIONS_SEALED.json"):
        assert not (tmp_path/name).exists()
    assert not (tmp_path/"coco_object").exists()
    assert torch.cuda.is_initialized() is False
