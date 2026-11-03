from contextlib import contextmanager
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.evolution_fifty import spatial_worker as worker
from tools.evolution_fifty.intake import sha256,write_new
from test_generalization_worker import TinyData
from test_image_full_worker import config
from test_spatial_runtime import method


def fixture(tmp_path,monkeypatch):
    suite = tmp_path/"suite"
    suite.mkdir()
    runtime = suite/"runtime"
    runtime.mkdir()
    (runtime/"fixture.py").write_text("CPU_fixture=True\n")
    rows,specs,annotations = {},[],{}
    for dataset in worker.DATASETS:
        items = []
        for i in range(5):
            label = tmp_path/f"{dataset}.GT{i}"
            label.write_bytes(b"synthetic GT must never be read")
            rgb = tmp_path/f"{dataset}.RGB{i}"
            rgb.write_bytes(b"synthetic RGB supplied as tensor, never decoded")
            state = label.stat()
            details = dict(device=state.st_dev,inode=state.st_ino,size_bytes=state.st_size,
                mtime_ns=state.st_mtime_ns,contents_read=False)
            items.append(dict(official_id=f"source{i}",image=str(rgb),label=str(label),
                label_metadata=[state.st_size,state.st_mtime_ns],label_metadata_details=details))
            annotations[str(label)] = details
        rows[dataset] = items
        specs.append(dict(id="M02."+dataset,dataset=dataset,images=5,config=config(),
            modes=list(worker.spatial_runtime.MODES),preprocessing="native_PIL_bilinear",backbone="ViT-L/14",
            model_type="naclip",checkpoint=dict(CPU_fixture=True),
            vocabulary=dict(classes=["an object","a surface"],queries=["an object","a surface"],
                query_to_class=None,reduce_zero_label=False)))
        monkeypatch.setitem(worker.POPULATIONS,dataset,5)
    source = dict(rows=rows,specs=specs,annotation_metadata=annotations,G_condition_ids=[],GT_opened=False)
    write_new(suite/"INPUTS.json",source)
    write_new(suite/"PROTOCOL.json",dict(CPU_fixture=True))
    prepared = dict(root=str(suite),schema_version=1,runtime_hashes={"fixture.py":sha256(runtime/"fixture.py")},
        inputs_sha256=sha256(suite/"INPUTS.json"),protocol_sha256=sha256(suite/"PROTOCOL.json"),parent_native_hashes={})
    write_new(suite/"SUITE_PREPARED.json",prepared)
    digest = sha256(suite/"SUITE_PREPARED.json")
    report = suite/"INDEPENDENT_REPORT.md"
    report.write_text("Synthetic CPU audit, not actual artifact admission\n")
    audit = dict(decision="ADMITTED",suite_prepared_sha256=digest,independent_reviewer="CPU fixture",
        M02_predictions_allowed=True,GT_scoring_allowed=False,report_path=str(report),report_sha256=sha256(report))
    write_new(suite/"ADMITTED.json",audit)
    monkeypatch.setattr(worker,"__file__",str(runtime/"fixture.py"))
    monkeypatch.setattr(worker,"source_inputs",lambda:copy.deepcopy(source))
    monkeypatch.setattr(worker,"validate_inputs",lambda *_:None)
    monkeypatch.setattr(worker.general,"runtime_imports",lambda *_:None)
    monkeypatch.setattr(worker,"native_identity",lambda:{})
    barrier = []
    monkeypatch.setattr(worker,"forbid_annotations",lambda paths:barrier.append(paths))
    def checkpoint(_):
        assert barrier, "Annotation barrier must precede checkpoint validation"
    monkeypatch.setattr(worker,"checked_checkpoint",checkpoint)
    monkeypatch.setattr(worker,"build_method",lambda _:method())
    monkeypatch.setattr(worker.confirmation,"make_data",TinyData)
    monkeypatch.setattr(worker,"deterministic_runtime",lambda:torch.manual_seed(17))
    monkeypatch.setattr(worker,"prior_GPU_terminal",lambda:dict(CPU_fixture=True))
    monkeypatch.setattr(worker.shutil,"disk_usage",lambda _:SimpleNamespace(free=256*1024**3))
    held = dict(active=False)
    @contextmanager
    def lock():
        assert not held["active"]
        held["active"] = True
        try:
            yield dict(CPU_fixture=True)
        finally:
            held["active"] = False
    monkeypatch.setattr(worker,"gpu_lock",lock)
    real_publish = worker.publish_seal
    def publish(path,value):
        assert not held["active"], "M02 must release the shared lock before publishing seals"
        real_publish(path,value)
    monkeypatch.setattr(worker,"publish_seal",publish)
    return suite,digest,source,audit,held


def test_entire_CPU_controller_real_tiny_gates_then_full_predictions_without_GT(tmp_path,monkeypatch):
    suite,digest,source,_,held = fixture(tmp_path,monkeypatch)
    original = Path.open
    labels = set(source["annotation_metadata"])
    def guard(path,*args,**kwargs):
        assert str(path) not in labels, "GT content opened in M02 prediction"
        if "predictions" in path.parts:
            assert (suite/"ALL_GATES_PASSED.json").exists()
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    worker.run(suite,digest)
    global_seal = json.loads((suite/"ALL_PREDICTIONS_SEALED.json").read_text())
    assert global_seal["complete"] and global_seal["cells"] == 8
    assert global_seal["image_method_evaluations"] == 40
    assert global_seal["GT_opened"] is False and global_seal["scientific_completion"] is False
    for dataset in worker.DATASETS:
        path = suite/dataset/"PREDICTIONS_SEALED.json"
        assert sha256(path) == global_seal["dataset_seals"][dataset]
        child = json.loads(path.read_text())
        assert child["complete"] and child["images"] == 5 and child["image_method_evaluations"] == 20
        for name,digest_value in child["artifact_sha256"].items():
            assert sha256(suite/name) == digest_value
    assert not held["active"] and not torch.cuda.is_initialized()
    with pytest.raises(ValueError,match="No M02 resume"):
        worker.run(suite,digest)


def test_run_refuses_running_prior_GPU_before_admission_or_markers(tmp_path,monkeypatch):
    suite,digest,_,_,_ = fixture(tmp_path,monkeypatch)
    def blocked():
        raise ValueError("G1954 still running")
    monkeypatch.setattr(worker,"prior_GPU_terminal",blocked)
    monkeypatch.setattr(worker,"admission",lambda *_:pytest.fail("Cannot admit/launch another GPU while prior running"))
    with pytest.raises(ValueError,match="still running"):
        worker.run(suite,digest)
    assert not (suite/"STARTED.json").exists()


@pytest.mark.parametrize("fault",["gate","lock_release","final_progress"])
def test_controller_failures_preserve_evidence_without_global_seal(tmp_path,monkeypatch,fault):
    suite,digest,_,_,held = fixture(tmp_path,monkeypatch)
    def fail(*args,**kwargs):
        raise RuntimeError("synthetic controller failure")
    if fault == "gate":
        monkeypatch.setattr(worker.spatial_gate,"compare_gate",fail)
    elif fault == "lock_release":
        @contextmanager
        def lock():
            yield dict(CPU_fixture=True)
            fail()
        monkeypatch.setattr(worker,"gpu_lock",lock)
    else:
        real_write = worker.write_json
        def write(path,value,**kwargs):
            if value["phase"] == "M02_payloads_complete_awaiting_global_seal_and_exit":
                fail()
            real_write(path,value,**kwargs)
        monkeypatch.setattr(worker,"write_json",write)
    with pytest.raises(RuntimeError,match="synthetic controller"):
        worker.run(suite,digest)
    assert (suite/"FAILED.json").exists() and not (suite/"ALL_PREDICTIONS_SEALED.json").exists()
    assert not held["active"]


@pytest.mark.parametrize("fault",["wrong_permission","report_GT_alias","prepared_GT_alias","extra_runtime_GT_alias"])
def test_M02_requires_own_admission_and_rejects_metadata_source_GT_aliases(tmp_path,monkeypatch,fault):
    suite,digest,source,audit,_ = fixture(tmp_path,monkeypatch)
    if fault == "wrong_permission":
        audit.pop("M02_predictions_allowed")
        audit["G_primary_predictions_allowed"] = True
        (suite/"ADMITTED.json").write_text(json.dumps(audit))
    else:
        import os
        path = suite/({"report_GT_alias":"INDEPENDENT_REPORT.md", "prepared_GT_alias":"SUITE_PREPARED.json",
                       "extra_runtime_GT_alias":"runtime/extra.py"}[fault])
        if path.exists():
            path.unlink()
        os.link(next(iter(source["annotation_metadata"])),path)
        original = Path.open
        def no_alias_read(candidate,*args,**kwargs):
            assert candidate != path, "M02 GT alias reached read"
            return original(candidate,*args,**kwargs)
        monkeypatch.setattr(Path,"open",no_alias_read)
    with pytest.raises(ValueError):
        worker.admission(suite,digest)
