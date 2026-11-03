import copy
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.evolution_fifty import curve_prediction_worker as worker
from tools.evolution_fifty.intake import sha256,write_new
from test_generalization_worker import TinyData,metadata
from test_generalization_runtime import method
from test_image_full_worker import config


def pins():
    return dict(pid=17,invocation="a"*32,global_sha256="b"*64)


def test_actual_fixed_ten_condition_plan_has_no_S5_or_new_models():
    clean,checkpoints=metadata()
    anchors=worker.general.condition_specs(clean,checkpoints)
    anchor=next(s for s in anchors if s["id"] == "naclip.ViT-L-14.voc20.gaussian_noise.s5")
    specs=[]
    for row in worker.curve.conditions():
        spec=copy.deepcopy(anchor)
        spec.update(id="naclip.ViT-L-14.voc20."+row["id"],input_key=row["id"],
                    corruption=row["corruption"],severity=row["severity"])
        specs.append(spec)
    inputs={s["input_key"]:[{} for _ in range(1449)] for s in specs}
    worker.validate_specs(specs,inputs,anchor)
    assert len(specs) == 10 and sum(s["images"]*3 for s in specs) == 43470
    assert {s["severity"] for s in specs} == {1,3}
    assert all(s["checkpoint"] == anchor["checkpoint"] and s["config"] == anchor["config"] for s in specs)
    for field,value in (("severity",True),("severity",1.),("model_type","sclip"),("images",128),
                        ("input_key","clean.voc20"),("vision_outputs",[-1]),("arms",["image_C"])):
        changed=copy.deepcopy(specs);changed[0][field]=value
        with pytest.raises(ValueError):worker.validate_specs(changed,inputs,anchor)


@pytest.mark.parametrize("change",[dict(pid=True),dict(pid=0),dict(pid=2**31),dict(invocation="g"*32),
    dict(invocation="a"*31),dict(global_sha256="b"*63),dict(extra=1)])
def test_operator_pins_strict_and_no_implicit_future_identity(change):
    value=pins();value.update(change)
    with pytest.raises(ValueError):worker.operator_pins(value)


def fixture(tmp_path,monkeypatch):
    suite=tmp_path/"suite"
    source=tmp_path/"source"
    source.mkdir()
    for tree in ("tools","tests","trace_ovss"):
        (source/tree).mkdir();(source/tree/"fixture.py").write_text("synthetic=True\n")
    rows=[];registry={}
    for i in range(5):
        rgb,gt=tmp_path/f"RGB{i}",tmp_path/f"GT{i}"
        rgb.write_bytes(b"Synthetic tensor input, not decoded");gt.write_bytes(b"GT must not open")
        st=gt.stat();details=dict(device=st.st_dev,inode=st.st_ino,size_bytes=st.st_size,mtime_ns=st.st_mtime_ns,contents_read=False)
        rows.append(dict(official_id=f"source{i}",image=str(rgb),label=str(gt),label_metadata=[st.st_size,st.st_mtime_ns],label_metadata_details=details))
        registry[str(gt)]=details
    specs=[]
    for i in range(2):
        specs.append(dict(id=f"fixture{i}",input_key=f"fixture{i}",dataset="voc20",images=5,
            backbone="ViT-B/16",model_type="naclip",config=config(),
            vocabulary=dict(classes=["an object","a surface"],queries=["an object","a surface"],query_to_class=None),
            checkpoint=dict(CPU_fixture=True),arms=list(worker.general.ARMS)))
    inputs=dict(native=dict(rows=dict(toy=rows),annotation_metadata=registry),rows={s["input_key"]:rows for s in specs},
                specs=specs,anchor=specs[0],GT_opened=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setattr(worker,"ROOT",source)
    monkeypatch.setattr(worker,"source_inputs",lambda:copy.deepcopy(inputs))
    monkeypatch.setattr(worker,"data_exclusions",lambda _: {(Path(r[k]).stat().st_dev,Path(r[k]).stat().st_ino) for r in rows for k in ("image","label")})
    monkeypatch.setattr(worker,"prerequisite_status",lambda _: dict(CPU_fixture=True))
    monkeypatch.setattr(worker,"prerequisite_bindings",lambda *_:dict(CPU_fixture=True))
    monkeypatch.setattr(worker,"forbid_annotations",lambda *_:None)
    monkeypatch.setattr(worker.general,"checked_checkpoint",lambda *_:None)
    monkeypatch.setattr(worker.general,"runtime_imports",lambda *_:None)
    monkeypatch.setattr(worker.shutil,"disk_usage",lambda _:SimpleNamespace(free=256*1024**3))
    monkeypatch.setattr(worker,"CONDITIONS",2)
    monkeypatch.setattr(worker,"CELLS",6)
    monkeypatch.setattr(worker,"EVALUATIONS",30)
    digest=worker.prepare(suite,pins())
    monkeypatch.setattr(worker,"__file__",str(suite/"runtime/tools/fixture.py"))
    report=suite/"INDEPENDENT_REPORT.md";report.write_text("Synthetic independent CPU fixture only\n")
    write_new(suite/"ADMITTED.json",dict(decision="ADMITTED",suite_prepared_sha256=digest,
        independent_reviewer="CPU_fixture",G03_predictions_allowed=True,GT_scoring_allowed=False,
        report_path=str(report),report_sha256=sha256(report)))
    monkeypatch.setattr(worker.general.runtime,"build_method",lambda _:method())
    monkeypatch.setattr(worker.general,"GeneralizationRGBDataset",TinyData)
    monkeypatch.setattr(worker.general,"deterministic_runtime",lambda:torch.manual_seed(0))
    monkeypatch.setattr(worker,"deterministic_runtime",lambda:torch.manual_seed(0))
    held=dict(active=False)
    @contextmanager
    def lock():
        assert not held["active"];held["active"]=True
        try:yield dict(CPU_fixture=True)
        finally:held["active"]=False
    monkeypatch.setattr(worker.general,"gpu_lock",lock)
    return suite,digest,inputs


def test_toy_full_prepare_admit_real_updates_gates_before_full_and_global_last(tmp_path,monkeypatch):
    suite,digest,inputs=fixture(tmp_path,monkeypatch)
    actual=Path.open
    gt=set(inputs["native"]["annotation_metadata"])
    def guard(path,*a,**k):
        assert str(path) not in gt
        if "predictions" in path.parts:assert (suite/"ALL_GATES_PASSED.json").exists()
        return actual(path,*a,**k)
    monkeypatch.setattr(Path,"open",guard)
    worker.run(suite,digest)
    seal=json.loads((suite/"ALL_G_PREDICTIONS_SEALED.json").read_bytes())
    assert seal["complete"] and seal["cells"] == 6 and seal["image_method_evaluations"] == 30
    assert len(seal["condition_seals"]) == 2 and not seal["GT_opened"]
    for name,d in seal["condition_seals"].items():
        path=suite/"conditions"/name/"PREDICTIONS_SEALED.json"
        assert sha256(path)==d
        child=json.loads(path.read_bytes())
        assert child["images"] == 5 and child["image_method_evaluations"] == 15
        for n,h in child["artifact_sha256"].items():assert sha256(suite/n)==h
    with pytest.raises(ValueError,match="resume/retry"):worker.run(suite,digest)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("fault",["permission","GT_permission","report_alias","input_alias","runtime_alias","protocol"])
def test_admission_faults_precede_STARTED_and_model_work(tmp_path,monkeypatch,fault):
    suite,digest,inputs=fixture(tmp_path,monkeypatch)
    if fault in ("permission","GT_permission"):
        p=suite/"ADMITTED.json";v=json.loads(p.read_bytes())
        v["G03_predictions_allowed" if fault == "permission" else "GT_scoring_allowed"]=fault == "GT_permission"
        p.unlink();write_new(p,v)
    elif fault == "protocol":
        p=suite/"PROTOCOL.json";p.write_text("{}")
    else:
        p=suite/{"report_alias":"INDEPENDENT_REPORT.md","input_alias":"INPUTS.json","runtime_alias":"runtime/tools/fixture.py"}[fault]
        p.unlink();p.hardlink_to(next(iter(inputs["native"]["annotation_metadata"])))
    monkeypatch.setattr(worker.general.runtime,"build_method",lambda *_:pytest.fail("Model reached invalid admission"))
    with pytest.raises(ValueError):worker.run(suite,digest)
    assert not (suite/"STARTED.json").exists()


@pytest.mark.parametrize("fault",["STARTED_fsync","gates","full","final_source","final_prerequisite","final_output"])
def test_owned_failure_no_global_no_reuse_and_partial_preserved(tmp_path,monkeypatch,fault):
    suite,digest,inputs=fixture(tmp_path,monkeypatch)
    if fault == "STARTED_fsync":
        actual=worker.generated.os.fsync
        def fsync(fd):
            if worker.os.readlink(f"/proc/self/fd/{fd}") == str(suite/"STARTED.json"):raise OSError("Synthetic fsync failure")
            return actual(fd)
        monkeypatch.setattr(worker.generated.os,"fsync",fsync)
    elif fault == "gates":monkeypatch.setattr(worker.general,"gates",lambda *_:(_ for _ in ()).throw(RuntimeError("Synthetic gate failure")))
    elif fault == "full":monkeypatch.setattr(worker.core,"execute_condition",lambda *_:(_ for _ in ()).throw(RuntimeError("Synthetic full failure")))
    elif fault == "final_source":
        actual=worker.source_inputs;calls=[]
        def source():
            calls.append(True);value=actual()
            if len(calls)>1:value["GT_opened"]=True
            return value
        monkeypatch.setattr(worker,"source_inputs",source)
    elif fault == "final_prerequisite":
        actual=worker.prerequisite_bindings;calls=[]
        def prior(*a):
            calls.append(True)
            return actual(*a) if len(calls)==1 else dict(drift=True)
        monkeypatch.setattr(worker,"prerequisite_bindings",prior)
    else:
        actual=worker.publish_seal
        def seal(path,value):
            if path.name == "ALL_G_PREDICTIONS_SEALED.json":raise OSError("Synthetic final publication failure")
            return actual(path,value)
        monkeypatch.setattr(worker,"publish_seal",seal)
    with pytest.raises((ValueError,OSError,RuntimeError)):worker.run(suite,digest)
    assert (suite/"STARTED.json").exists() and (suite/"FAILED.json").exists()
    assert not (suite/"ALL_G_PREDICTIONS_SEALED.json").exists()
    with pytest.raises((ValueError,OSError,RuntimeError)):worker.run(suite,digest)


def test_foreign_racing_STARTED_not_claimed_or_overwritten(tmp_path,monkeypatch):
    suite,digest,_=fixture(tmp_path,monkeypatch)
    actual=worker.os.open
    def opened(name,flags,*a,**k):
        if name == "STARTED.json" and flags & worker.os.O_CREAT:(suite/"STARTED.json").write_bytes(b"foreign")
        return actual(name,flags,*a,**k)
    monkeypatch.setattr(worker.os,"open",opened)
    with pytest.raises(FileExistsError):worker.run(suite,digest)
    assert (suite/"STARTED.json").read_bytes() == b"foreign" and not (suite/"FAILED.json").exists()


def test_prepare_rejects_unfinished_named_prerequisite_before_source_or_mkdir(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setattr(worker,"prerequisite_status",lambda *_:(_ for _ in ()).throw(ValueError("still running")))
    monkeypatch.setattr(worker,"source_inputs",lambda:pytest.fail("Source read before prior terminal"))
    with pytest.raises(ValueError,match="still running"):worker.prepare(tmp_path/"new",pins())
    assert not (tmp_path/"new").exists()


@pytest.mark.parametrize("which",["temp_regular","temp_symlink","temp_hardlink","dest_symlink","dest_hardlink"])
def test_progress_alias_or_existing_staging_never_changes_victim(tmp_path,monkeypatch,which):
    suite,digest,_=fixture(tmp_path,monkeypatch)
    victim=tmp_path/"synthetic_frozen_RGB";victim.write_bytes(b"unchanged synthetic RGB bytes")
    destination=suite/("queue_state.json.tmp" if which.startswith("temp") else "queue_state.json")
    if which.endswith("symlink"):destination.symlink_to(victim)
    elif which.endswith("hardlink"):destination.hardlink_to(victim)
    else:destination.write_bytes(b"existing staging evidence")
    before=victim.stat();identity=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
    with pytest.raises((ValueError,FileExistsError)):worker.run(suite,digest)
    after=victim.stat()
    assert victim.read_bytes() == b"unchanged synthetic RGB bytes"
    assert (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) == identity
    assert (suite/"FAILED.json").exists() and not (suite/"ALL_G_PREDICTIONS_SEALED.json").exists()
    if which == "temp_regular":assert destination.read_bytes() == b"existing staging evidence"


def test_new_IO_core_retains_old_loss_candidates_predictions_and_work(tmp_path,monkeypatch):
    suite,_,inputs=fixture(tmp_path,monkeypatch)
    spec=inputs["specs"][0]
    data=TinyData(spec["config"],inputs["rows"][spec["input_key"]])
    torch.manual_seed(101);left=method()
    torch.manual_seed(101);right=method()
    rng=torch.get_rng_state()
    old_progress=dict(completed_conditions=0,completed_source_episodes=0)
    old=worker.general.execute_condition(left,data,spec,suite/"old",suite,dict(bytes=0),old_progress)
    torch.set_rng_state(rng)
    new_progress=dict(completed_conditions=0,completed_source_episodes=0)
    new=worker.core.execute_condition(right,data,spec,suite/"new",suite,dict(bytes=0),new_progress,worker.shared.write_json)
    assert old_progress == new_progress and old["images"] == new["images"] == 5
    for name in old["artifact_sha256"]:
        suffix=Path(name).relative_to("old")
        oldpath,newpath=suite/name,suite/"new"/suffix
        if suffix.suffix == ".npy" or suffix.name == "initial_state.json":
            assert sha256(oldpath) == sha256(newpath)
        elif suffix.parent.name == "episodes":
            a,b=json.loads(oldpath.read_bytes()),json.loads(newpath.read_bytes())
            a.pop("seconds");b.pop("seconds")
            assert a == b
    a=json.loads((suite/"old/prediction_result.json").read_bytes())
    b=json.loads((suite/"new/prediction_result.json").read_bytes())
    for key in ("actual_work","legal_visual_LN","frozen_parameter_sha256"):
        assert a[key] == b[key]
