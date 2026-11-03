from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.evolution_fifty import factorial_worker as worker
from tools.evolution_fifty.intake import sha256
from test_spatial_worker import fixture as spatial_fixture
from test_factorial_runtime import method


def fixture(tmp_path,monkeypatch):
    suite,_,_,_,held = spatial_fixture(tmp_path,monkeypatch)
    source = worker.source_inputs()
    (suite/"INPUTS.json").write_text(json.dumps(source))
    prepared = json.loads((suite/"SUITE_PREPARED.json").read_text())
    prepared["inputs_sha256"] = sha256(suite/"INPUTS.json")
    (suite/"SUITE_PREPARED.json").write_text(json.dumps(prepared))
    digest = sha256(suite/"SUITE_PREPARED.json")
    audit = json.loads((suite/"ADMITTED.json").read_text())
    audit.pop("M02_predictions_allowed")
    audit.update(suite_prepared_sha256=digest,M04_predictions_allowed=True)
    (suite/"ADMITTED.json").write_text(json.dumps(audit))
    monkeypatch.setattr(worker,"__file__",str(suite/"runtime/fixture.py"))
    monkeypatch.setattr(worker,"native_identity",lambda:{})
    monkeypatch.setattr(worker,"forbid_annotations",worker.source_worker.forbid_annotations)
    monkeypatch.setattr(worker,"checked_checkpoint",worker.source_worker.checked_checkpoint)
    monkeypatch.setattr(worker,"build_method",lambda _:method())
    monkeypatch.setattr(worker,"gpu_lock",worker.source_worker.gpu_lock)
    monkeypatch.setattr(worker,"publish_seal",worker.source_worker.publish_seal)
    monkeypatch.setattr(worker,"deterministic_runtime",worker.source_worker.deterministic_runtime)
    monkeypatch.setattr(worker,"require_named_GPU_idle",lambda:dict(CPU_fixture=True))
    return suite,digest,source,audit,held


def test_whole_factorial_controller_real_tiny_gates_and_all_predictions_without_GT(tmp_path,monkeypatch):
    suite,digest,source,_,held = fixture(tmp_path,monkeypatch)
    labels = set(source["annotation_metadata"])
    original = Path.open
    def guard(path,*args,**kwargs):
        assert str(path) not in labels
        if "predictions" in path.parts:
            assert (suite/"ALL_GATES_PASSED.json").exists()
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    worker.run(suite,digest)
    result = json.loads((suite/"ALL_PREDICTIONS_SEALED.json").read_text())
    assert result["cells"] == 16 and result["image_method_evaluations"] == 80
    assert not result["GT_opened"] and not result["scientific_completion"]
    for dataset in worker.DATASETS:
        path = suite/dataset/"PREDICTIONS_SEALED.json"
        assert sha256(path) == result["dataset_seals"][dataset]
        child = json.loads(path.read_text())
        assert child["images"] == 5 and child["image_method_evaluations"] == 40
        for name,digest_value in child["artifact_sha256"].items():
            assert sha256(suite/name) == digest_value
    assert not held["active"] and not torch.cuda.is_initialized()
    with pytest.raises(ValueError,match="No M04 resume"):
        worker.run(suite,digest)


@pytest.mark.parametrize("block",["G","M02"])
def test_no_second_GPU_before_prior_and_named_worker_idle_checks(tmp_path,monkeypatch,block):
    suite,digest,_,_,_ = fixture(tmp_path,monkeypatch)
    def fail():
        raise ValueError("prior GPU still running")
    if block == "G":
        monkeypatch.setattr(worker.source_worker,"prior_GPU_terminal",fail)
    else:
        monkeypatch.setattr(worker,"require_named_GPU_idle",fail)
    monkeypatch.setattr(worker,"admission",lambda *_:pytest.fail("prior must precede admission"))
    with pytest.raises(ValueError,match="still running"):
        worker.run(suite,digest)
    assert not (suite/"STARTED.json").exists()


@pytest.mark.parametrize("fault",["gate","lock_release","final_progress"])
def test_factorial_controller_failure_preserves_partial_and_blocks_seal(tmp_path,monkeypatch,fault):
    suite,digest,_,_,held = fixture(tmp_path,monkeypatch)
    def fail(*_,**__):
        raise RuntimeError("synthetic M04 controller failure")
    if fault == "gate":
        monkeypatch.setattr(worker.factorial_gate,"compare_gate",fail)
    elif fault == "lock_release":
        @contextmanager
        def lock():
            yield dict(CPU_fixture=True)
            fail()
        monkeypatch.setattr(worker,"gpu_lock",lock)
    else:
        original = worker.write_json
        def write(path,value,**kwargs):
            if value["phase"] == "M04_payloads_complete_awaiting_global_seal_and_exit":
                fail()
            return original(path,value,**kwargs)
        monkeypatch.setattr(worker,"write_json",write)
    with pytest.raises(RuntimeError,match="M04 controller failure"):
        worker.run(suite,digest)
    assert (suite/"FAILED.json").exists() and not (suite/"ALL_PREDICTIONS_SEALED.json").exists()
    assert not held["active"]


@pytest.mark.parametrize("fault",["permission","report_alias","runtime_alias"])
def test_factorial_requires_distinct_admission_and_protected_metadata(tmp_path,monkeypatch,fault):
    suite,digest,source,audit,_ = fixture(tmp_path,monkeypatch)
    if fault == "permission":
        audit.pop("M04_predictions_allowed")
        audit["M02_predictions_allowed"] = True
        (suite/"ADMITTED.json").write_text(json.dumps(audit))
    else:
        import os
        path = suite/("INDEPENDENT_REPORT.md" if fault == "report_alias" else "runtime/extra.py")
        if path.exists():
            path.unlink()
        os.link(next(iter(source["annotation_metadata"])),path)
        original = Path.open
        def guard(candidate,*args,**kwargs):
            assert candidate != path, "GT alias must fail before reading"
            return original(candidate,*args,**kwargs)
        monkeypatch.setattr(Path,"open",guard)
    with pytest.raises(ValueError):
        worker.admission(suite,digest)


@pytest.mark.parametrize("state,accepted",[
    (dict(LoadState="not-found",ActiveState="inactive",SubState="dead",MainPID="0"),True),
    (dict(LoadState="loaded",ActiveState="active",SubState="exited",MainPID="0"),True),
    (dict(LoadState="loaded",ActiveState="active",SubState="running",MainPID="900"),False),
    (dict(LoadState="loaded",ActiveState="activating",SubState="start",MainPID="0"),False),
    ({},False)])
def test_named_GPU_idle_is_not_a_scientific_completion_claim(monkeypatch,state,accepted):
    monkeypatch.setattr(worker.subprocess,"run",lambda *args,**kwargs:SimpleNamespace(returncode=0,
        stdout="\n".join(f"{k}={v}" for k,v in state.items())))
    if accepted:
        assert worker.require_named_GPU_idle() == state
    else:
        with pytest.raises(ValueError,match="not confirmed idle"):
            worker.require_named_GPU_idle()
