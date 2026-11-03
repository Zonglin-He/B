import copy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.evolution_fifty import generalization_worker as worker
from tools.evolution_fifty.intake import sha256, write_new
from tools.evolution_fifty.generalization import visual_layers
from trace_ovss.models._vendor_clip.clip import _MODELS
from test_generalization_runtime import method
from test_image_full_worker import config, batch_meta


def metadata():
    clean = {}
    for dataset,name in (("coco_object","COCOObjectDataset"),("pc60","PascalContext60Dataset"),("voc20","PascalVOC20Dataset")):
        cfg = config()
        cfg.update(dataset=name,resize=[224,224],patch_size=[224,224])
        clean[dataset] = dict(config=cfg,panel=dict(classes=["a","b"],queries=["a","b"],query_to_class=None,reduce_zero_label=dataset == "voc20"))
    checkpoints = {b:dict(path="/fixture/"+_MODELS[b].rsplit("/",1)[1],sha256=_MODELS[b].split("/")[-2],
        bytes=1,mtime_ns=1,official_url=_MODELS[b],backbone=b,vision_outputs=visual_layers(b)) for b in worker.BACKBONES}
    return clean,checkpoints


def test_primary_condition_scope_is_exactly57cells_not_all132_or_partial_panels():
    specs = worker.condition_specs(*metadata())
    worker.validate_conditions(specs)
    assert len(specs) == len({s["id"] for s in specs}) == 19
    assert sum(s["images"]*3 for s in specs) == 104214
    assert sum(s["model_type"] == "sclip" for s in specs) == 1
    assert sum(s["corruption"] != "original" for s in specs) == 15
    assert set(tuple(s["arms"]) for s in specs) == {worker.ARMS}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "severity", "model", "tuning", "vocabulary", "layers", "checkpoint", "input", "population"])
def test_condition_tampering_is_rejected_before_model_or_annotation_work(mutation):
    specs = worker.condition_specs(*metadata())
    if mutation == "missing":
        specs.pop()
    elif mutation == "duplicate":
        specs[-1] = copy.deepcopy(specs[-2])
    elif mutation == "severity":
        specs[-1]["severity"] = 1
    elif mutation == "model":
        specs[-1]["model_type"] = "clip"
    elif mutation == "tuning":
        specs[-1]["config"]["lr"] *= 2
    elif mutation == "vocabulary":
        specs[-1]["vocabulary"]["queries"].append("extra")
    elif mutation == "layers":
        specs[0]["vision_outputs"] = list(range(-1,-19,-1))
    elif mutation == "checkpoint":
        specs[0]["checkpoint"]["sha256"] = "f"*64
    elif mutation == "input":
        specs[-1]["input_key"] = "clean.voc20"
    elif mutation == "population":
        specs[0]["images"] = 128
    with pytest.raises(ValueError):
        worker.validate_conditions(specs)


class TinyData(list):
    def __init__(self,cfg,rows):
        self.rows = rows
        meta = batch_meta(len(rows))
        meta["img_path"] = [r["image"] for r in rows]
        meta["seg_map_path"] = [r["label"] for r in rows]
        images = torch.randn(len(rows),3,32,32,generator=torch.Generator().manual_seed(31))
        super().__init__([dict(img_patches=images[i:i+1],meta={k:v[i] for k,v in meta.items()}) for i in range(len(rows))])


def fixture(tmp_path,monkeypatch):
    suite = tmp_path/"suite"
    suite.mkdir()
    (suite/"runtime").mkdir()
    specs,inputs = [],{}
    for i,backend in enumerate(("naclip","sclip")):
        rows = []
        for j in range(5):
            gt = tmp_path/f"GT_{i}_{j}.never_open"
            gt.write_bytes(b"Annotations must not be read by G prediction")
            rows.append(dict(official_id=f"source{j}",image=f"/synthetic/RGB_{i}_{j}",label=str(gt)))
        name = f"condition{i}"
        inputs[name] = rows
        specs.append(dict(id=name,input_key=name,model_type=backend,backbone="ViT-B/16",config=config(),images=5,
            checkpoint=metadata()[1]["ViT-B/16"],
            vocabulary=dict(classes=["an object","a surface"],queries=["an object","a surface"],query_to_class=None,reduce_zero_label=False)))
    write_new(suite/"INPUTS.json",inputs)
    write_new(suite/"CONDITIONS.json",specs)
    write_new(suite/"PROTOCOL.json",dict(CPU_fixture_only=True))
    prepared = dict(root=str(suite),runtime_hashes={},inputs_sha256=sha256(suite/"INPUTS.json"),
        conditions_sha256=sha256(suite/"CONDITIONS.json"),protocol_sha256=sha256(suite/"PROTOCOL.json"))
    write_new(suite/"SUITE_PREPARED.json",prepared)
    digest = sha256(suite/"SUITE_PREPARED.json")
    report = suite/"INDEPENDENT_REPORT.md"
    report.write_text("SYNTHETIC CPU fixture, not actual artifact admission\n")
    write_new(suite/"ADMITTED.json",dict(CPU_fixture_only=True))
    audit = dict(admission_path=str(suite/"ADMITTED.json"),admission_sha256=sha256(suite/"ADMITTED.json"),
        report_path=str(report),report_sha256=sha256(report),independent_reviewer="CPU fixture")
    for key,path in (("admission_identity",suite/"ADMITTED.json"),("report_identity",report)):
        s = path.stat()
        audit[key] = [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
    monkeypatch.setattr(worker,"admission",lambda *_: (suite,prepared,specs,inputs,audit))
    monkeypatch.setattr(worker,"runtime_imports",lambda *_:None)
    monkeypatch.setattr(worker.runtime,"build_method",lambda spec: method(spec["model_type"]))
    monkeypatch.setattr(worker,"GeneralizationRGBDataset",TinyData)
    monkeypatch.setattr(worker,"deterministic_runtime",lambda: torch.manual_seed(0))
    monkeypatch.setattr(worker.shutil,"disk_usage",lambda _:SimpleNamespace(free=256*1024**3))
    @contextmanager
    def lock():
        yield dict(CPU_fixture=True)
    monkeypatch.setattr(worker,"gpu_lock",lock)
    monkeypatch.setattr(worker,"EXPECTED_CONDITIONS",2)
    monkeypatch.setattr(worker,"EXPECTED_CELLS",6)
    monkeypatch.setattr(worker,"EXPECTED_EVALUATIONS",30)
    (suite/"reference").mkdir()
    from tools.evolution_fifty.sclip_reference import SOURCE
    (suite/"reference/sclip_model.py").write_bytes(SOURCE.read_bytes())
    return suite,digest,prepared,specs,inputs,audit


def test_CPU_filesystem_all_gates_before_all_full_predictions_and_global_barrier(tmp_path,monkeypatch):
    suite,digest,prepared,specs,inputs,audit = fixture(tmp_path,monkeypatch)
    original_open = Path.open
    denied = {r["label"] for rows in inputs.values() for r in rows}
    def guard(path,*args,**kwargs):
        assert str(path) not in denied, "GT content reached G prediction"
        if "predictions" in path.parts:
            assert (suite/"ALL_GATES_PASSED.json").exists()
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    worker.run(suite,digest)
    barrier = json.loads((suite/"ALL_G_PREDICTIONS_SEALED.json").read_text())
    assert barrier["complete"] and barrier["cells"] == 6 and barrier["image_method_evaluations"] == 30
    assert not barrier["GT_opened"] and not barrier["scientific_completion"]
    assert barrier["audit_binding"] == audit and barrier["started_sha256"] == sha256(suite/"STARTED.json")
    gates = json.loads((suite/"ALL_GATES_PASSED.json").read_text())
    assert set(gates["conditions"]) == set(barrier["condition_seals"]) == {s["id"] for s in specs}
    for spec in specs:
        output = suite/"conditions"/spec["id"]
        seal = json.loads((output/"PREDICTIONS_SEALED.json").read_text())
        assert seal["image_method_evaluations"] == 15
        assert sha256(output/"PREDICTIONS_SEALED.json") == barrier["condition_seals"][spec["id"]]
        for path,expected in seal["artifact_sha256"].items():
            assert sha256(suite/path) == expected
        result = json.loads((output/"prediction_result.json").read_text())
        for arm in worker.ARMS:
            assert len(list((output/"predictions"/arm).glob("*.npy"))) == 5
            assert result["warmup_adjusted_throughput"][arm]["measured_images"] == 3
            assert result["warmup_adjusted_throughput"][arm]["accuracy_includes_warmup"]
    assert not torch.cuda.is_initialized()
    with pytest.raises(ValueError,match="No G resume/retry"):
        worker.run(suite,digest)


def test_failed_gate_prevents_any_full_predictions_and_retry(tmp_path,monkeypatch):
    suite,digest,*_ = fixture(tmp_path,monkeypatch)
    def fail(*args,**kwargs):
        raise ValueError("injected actual numerical gate failure")
    monkeypatch.setattr(worker.runtime,"compare_gate",fail)
    with pytest.raises(ValueError,match="numerical gate"):
        worker.run(suite,digest)
    assert (suite/"FAILED.json").exists() and not (suite/"conditions").exists()
    assert not (suite/"ALL_G_PREDICTIONS_SEALED.json").exists()
    with pytest.raises(ValueError,match="No G resume/retry"):
        worker.run(suite,digest)


def test_final_progress_failure_preserves_condition_seals_but_no_global_barrier(tmp_path,monkeypatch):
    suite,digest,*_ = fixture(tmp_path,monkeypatch)
    native = worker.write_json
    def fail(path,value,**kwargs):
        if value["phase"] == "all_G_payloads_complete_awaiting_seal_and_exit":
            raise OSError("final G progress failed")
        return native(path,value,**kwargs)
    monkeypatch.setattr(worker,"write_json",fail)
    with pytest.raises(OSError,match="final G progress"):
        worker.run(suite,digest)
    assert len(list((suite/"conditions").glob("*/PREDICTIONS_SEALED.json"))) == 2
    assert not (suite/"ALL_G_PREDICTIONS_SEALED.json").exists() and (suite/"FAILED.json").exists()


def test_lock_teardown_failure_cannot_publish_condition_or_global_seal(tmp_path,monkeypatch):
    suite,digest,*_ = fixture(tmp_path,monkeypatch)
    calls = 0
    @contextmanager
    def lock():
        nonlocal calls
        calls += 1
        yield dict(CPU_fixture=True)
        if calls == 3:
            raise RuntimeError("prediction lock teardown failed")
    monkeypatch.setattr(worker,"gpu_lock",lock)
    with pytest.raises(RuntimeError,match="lock teardown"):
        worker.run(suite,digest)
    assert list((suite/"conditions").glob("*/predictions/*/*.npy"))
    assert not list((suite/"conditions").glob("*/PREDICTIONS_SEALED.json"))
    assert not (suite/"ALL_G_PREDICTIONS_SEALED.json").exists()


def test_disk_budget_preflight_prevents_gate_or_model_work(tmp_path,monkeypatch):
    suite,digest,*_ = fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(worker.shutil,"disk_usage",lambda _:SimpleNamespace(free=worker.RESERVE))
    monkeypatch.setattr(worker,"gates",lambda *_:pytest.fail("Gates before disk preflight"))
    with pytest.raises(ValueError,match="disk headroom"):
        worker.run(suite,digest)
    assert not (suite/"STARTED.json").exists()


def test_payload_write_uses_exact_serialized_bytes_and_keeps_shared_reserve(tmp_path,monkeypatch):
    inventory,cost = {},dict(bytes=0)
    value = dict(rows=[dict(label="a",values=list(range(20))) for _ in range(100)])
    actual = (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    monkeypatch.setattr(worker.shutil,"disk_usage",lambda _:SimpleNamespace(free=worker.RESERVE+len(actual)-1))
    path = tmp_path/"never_written.json"
    with pytest.raises(ValueError,match="disk reserve"):
        worker.write_payload(path,value,inventory,tmp_path,cost)
    assert not path.exists() and cost["bytes"] == 0


def test_source_input_drift_prevents_global_seal(tmp_path,monkeypatch):
    suite,digest,*_ = fixture(tmp_path,monkeypatch)
    original = worker.execute_condition
    def drift(*args,**kwargs):
        result = original(*args,**kwargs)
        (suite/"INPUTS.json").write_text("{}")
        return result
    monkeypatch.setattr(worker,"execute_condition",drift)
    with pytest.raises(ValueError,match="source/input drift"):
        worker.run(suite,digest)
    assert (suite/"FAILED.json").exists() and not (suite/"ALL_G_PREDICTIONS_SEALED.json").exists()


def test_gate_window_startup_drift_cannot_become_new_baseline(tmp_path,monkeypatch):
    suite,digest,*_ = fixture(tmp_path,monkeypatch)
    original = worker.gates
    def drift(*args,**kwargs):
        result = original(*args,**kwargs)
        path = suite/"STARTED.json"
        value = json.loads(path.read_text())
        value["pid"] = -123
        path.write_text(json.dumps(value))
        return result
    monkeypatch.setattr(worker,"gates",drift)
    with pytest.raises(ValueError,match="startup binding changed during gates"):
        worker.run(suite,digest)
    assert not (suite/"conditions").exists()
    assert (suite/"FAILED.json").exists() and not (suite/"ALL_G_PREDICTIONS_SEALED.json").exists()


@pytest.mark.parametrize("target",["child_prediction","gate_barrier","gate_child"])
def test_completed_child_and_gate_drift_block_global_barrier(tmp_path,monkeypatch,target):
    suite,digest,prepared,specs,*_ = fixture(tmp_path,monkeypatch)
    original = worker.execute_condition
    calls = 0
    def drift(*args,**kwargs):
        nonlocal calls
        result = original(*args,**kwargs)
        calls += 1
        if calls == 2:
            path = {"child_prediction":suite/"conditions"/specs[0]["id"]/"PREDICTIONS_SEALED.json",
                "gate_barrier":suite/"ALL_GATES_PASSED.json",
                "gate_child":suite/"gates"/(specs[0]["id"]+".json")}[target]
            value = json.loads(path.read_text())
            value["injected_drift"] = True
            path.write_text(json.dumps(value))
        return result
    monkeypatch.setattr(worker,"execute_condition",drift)
    with pytest.raises(ValueError):
        worker.run(suite,digest)
    assert (suite/"FAILED.json").exists() and not (suite/"ALL_G_PREDICTIONS_SEALED.json").exists()


def test_gate_barrier_window_drift_is_not_recaptured(tmp_path,monkeypatch):
    suite,digest,*_ = fixture(tmp_path,monkeypatch)
    original = worker.gates
    def drift(*args,**kwargs):
        binding = original(*args,**kwargs)
        path = suite/"ALL_GATES_PASSED.json"
        value = json.loads(path.read_text())
        value["injected_drift"] = True
        path.write_text(json.dumps(value))
        return binding
    monkeypatch.setattr(worker,"gates",drift)
    with pytest.raises(ValueError):
        worker.run(suite,digest)
    assert not (suite/"conditions").exists() and not (suite/"ALL_G_PREDICTIONS_SEALED.json").exists()


def test_real_admission_requires_new_G_scope_and_exact_runtime(tmp_path,monkeypatch):
    actual_admission = worker.admission
    suite,digest,prepared,specs,inputs,audit = fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(worker,"__file__",str(suite/"runtime/tools/evolution_fifty/generalization_worker.py"))
    monkeypatch.setattr(worker,"native_identity",lambda:{})
    monkeypatch.setattr(worker,"validate_inputs",lambda *_:[])
    monkeypatch.setattr(worker,"forbid_annotations",lambda *_:None)
    monkeypatch.setattr(worker,"checked_checkpoint",lambda *_:None)
    monkeypatch.setattr(worker,"condition_specs",lambda *_:specs)
    cleanroot,genroot = tmp_path/"old_inputs_metadata",tmp_path/"fresh_inputs_metadata"
    cleanroot.mkdir()
    genroot.mkdir()
    monkeypatch.setattr(worker,"CLEAN_ROOT",cleanroot)
    monkeypatch.setattr(worker,"GEN_ROOT",genroot)
    terminal = tmp_path/"GEN_TERMINAL.json"
    monkeypatch.setattr(worker,"GEN_TERMINAL",terminal)
    write_new(cleanroot/"CLEAN_INPUTS.json",{})
    write_new(cleanroot/"CHECKPOINTS.json",{})
    write_new(cleanroot/"INPUTS_PREPARED.json",dict(artifact_sha256={n:sha256(cleanroot/n) for n in ("CLEAN_INPUTS.json","CHECKPOINTS.json")}))
    corruptions = {c:[] for c in worker.CORRUPTIONS}
    for c,rows in corruptions.items():
        inputs[f"corruption.{c}"] = rows
    (suite/"INPUTS.json").write_text(json.dumps(inputs))
    write_new(genroot/"VOC20_C_S5_INPUTS.json",corruptions)
    write_new(genroot/"INPUTS_SEALED.json",dict(artifact_sha256={"VOC20_C_S5_INPUTS.json":sha256(genroot/"VOC20_C_S5_INPUTS.json")}))
    monkeypatch.setattr(worker,"GEN_SEAL_SHA",sha256(genroot/"INPUTS_SEALED.json"))
    write_new(terminal,dict(successful_fixture=True))
    prepared.update(parent_native_hashes={},schema_version=1,forbidden_labels=[],
        inputs_sha256=sha256(suite/"INPUTS.json"),
        source_clean_receipt_sha256=sha256(cleanroot/"INPUTS_PREPARED.json"),
        fresh_RGB_seal_sha256=worker.GEN_SEAL_SHA,fresh_RGB_terminal_receipt_sha256=sha256(terminal))
    (suite/"SUITE_PREPARED.json").write_text(json.dumps(prepared))
    digest = sha256(suite/"SUITE_PREPARED.json")
    admitted = dict(decision="ADMITTED",suite_prepared_sha256=digest,independent_reviewer="synthetic reviewer",
        label_free_support_replay_allowed=True,report_path=str(suite/"INDEPENDENT_REPORT.md"),report_sha256=sha256(suite/"INDEPENDENT_REPORT.md"))
    (suite/"ADMITTED.json").write_text(json.dumps(admitted))
    with pytest.raises(ValueError,match="exact G prediction artifact admission"):
        actual_admission(suite,digest)
    admitted.update(G_primary_predictions_allowed=True,GT_scoring_allowed=False)
    (suite/"ADMITTED.json").write_text(json.dumps(admitted))
    result = actual_admission(suite,digest)
    assert result[:4] == (suite,prepared,specs,inputs)
    assert result[4]["admission_sha256"] == sha256(suite/"ADMITTED.json")
    (suite/"runtime/unapproved_namespace.py").write_text("# not frozen\n")
    with pytest.raises(ValueError,match="changed G runtime"):
        actual_admission(suite,digest)


def test_cross_root_import_is_rejected_even_if_worker_source_is_frozen(tmp_path,monkeypatch):
    worker.runtime_imports(worker.ROOT)
    untrusted = tmp_path/"external_helper.py"
    untrusted.write_text("# not frozen\n")
    monkeypatch.setitem(worker.sys.modules,"tools.evolution_fifty.untrusted_shadow",SimpleNamespace(__file__=str(untrusted)))
    with pytest.raises(ValueError,match="outside exact runtime"):
        worker.runtime_imports(worker.ROOT)
