import copy
import json
from pathlib import Path
import shutil

import pytest
import torch

from test_legacy_score_components import fixture as archive_fixture
from test_legacy_score_payload import fixture as payload_fixture
from tools.evolution_fifty import legacy_score_worker as worker
from tools.evolution_fifty.intake import sha256,write_new


def pins():
    return dict(pid=777,invocation="a"*32,start_sha256="b"*64,gate_sha256="c"*64,barrier_sha256="d"*64)


def fixture(tmp_path,monkeypatch):
    parent,suite = tmp_path/"parent",tmp_path/"score"
    parent.mkdir()
    suite.mkdir()
    source = dict(rows={},specs=[],annotation_metadata={},GT_opened=False,old_predictions_reused=False)
    inventories,children = {},{}
    def save(root,name,value):
        path = root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        write_new(path,value)
        return sha256(path)
    for dataset in worker.DATASETS:
        container = tmp_path/(dataset+"_fixture")
        container.mkdir()
        _,bound,spec,_,_ = payload_fixture(container,monkeypatch)
        shutil.copytree(Path(bound["parent"])/"pc60",parent/dataset)
        summary_path = parent/dataset/"prediction_result.json"
        summary = json.loads(summary_path.read_bytes())
        summary["dataset"] = dataset
        summary_path.unlink()
        write_new(summary_path,summary)
        inventories[dataset] = {str(p.relative_to(parent)):sha256(p) for p in (parent/dataset).rglob("*") if p.is_file()}
        spec = copy.deepcopy(spec)
        spec.update(dataset=dataset,id="M03."+dataset,modes=list(worker.MODES))
        spec["config"]["dataset"] = dict(pc60="PascalContext60Dataset",coco_object="COCOObjectDataset")[dataset]
        source["specs"].append(spec)
        source["rows"][dataset] = bound["source"]["rows"]["pc60"]
        source["annotation_metadata"].update(bound["source"]["annotation_metadata"])
        monkeypatch.setitem(worker.POPULATIONS,dataset,2)
    content,archive_spec = archive_fixture(monkeypatch)
    archive_path = tmp_path/"pinned.synthetic.tar.gz"
    archive_path.write_bytes(content)
    archive_spec["path"] = str(archive_path)
    source["legacy_archive"] = archive_spec
    monkeypatch.setattr(worker.prediction.legacy_rho,"ARCHIVE",archive_path)
    monkeypatch.setattr(worker.prediction.legacy_rho,"ARCHIVE_SHA",archive_spec["sha256"])
    for root in (parent,suite):
        (root/"runtime").mkdir()
        (root/"runtime"/"fixture.py").write_text("Synthetic_CPU_only=True\n")
    inventory = {"fixture.py":sha256(parent/"runtime"/"fixture.py")}
    report = parent/"INDEPENDENT_REPORT.md"
    report.write_text("Synthetic parent report, not actual M03 prediction permission\n")
    report_sha = sha256(report)
    analysis_path = tmp_path/"ANALYSIS.json"
    contrasts = {k:{m:v for m,v in zip(worker.MODES,w) if v} for k,w in worker.analysis.CONTRASTS.items()}
    save(tmp_path,"ANALYSIS.json",dict(datasets={d:2 for d in worker.DATASETS},conditions=list(worker.MODES),
        full_cells=8,image_method_evaluations=16,primary_contrast=dict(name=worker.analysis.PRIMARY,coefficients=contrasts.pop(worker.analysis.PRIMARY)),
        secondary_contrasts=[dict(name=k,coefficients=v) for k,v in contrasts.items()],
        bootstrap=dict(replicates=10000,root_seed=20261102)))
    source["prospective_analysis_sha256"] = sha256(analysis_path)
    inputs_sha = save(parent,"INPUTS.json",source)
    protocol_sha = save(parent,"PROTOCOL.json",dict(synthetic_CPU=True))
    parent_sha = save(parent,"SUITE_PREPARED.json",dict(schema_version=1,root=str(parent),
        inputs_sha256=inputs_sha,protocol_sha256=protocol_sha,parent_native_hashes={},runtime_hashes=inventory))
    admission_sha = save(parent,"ADMITTED.json",dict(decision="ADMITTED",suite_prepared_sha256=parent_sha,
        M03_predictions_allowed=True,GT_scoring_allowed=False))
    def identity(path):
        st = path.stat()
        return [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns]
    audit = dict(admission_path=str(parent/"ADMITTED.json"),admission_sha256=admission_sha,
        admission_identity=identity(parent/"ADMITTED.json"),report_path=str(report),report_sha256=report_sha,
        report_identity=identity(report),independent_reviewer="SyntheticCPU")
    current = pins()
    current["start_sha256"] = save(parent,"STARTED.json",dict(suite_prepared_sha256=parent_sha,audit=audit,pid=current["pid"],GT_opened=False))
    gates = {d:save(parent,f"gates/{d}.json",dict(passed=True,dataset=d,actual_RGB_sources=4,GT_opened=False)) for d in worker.DATASETS}
    current["gate_sha256"] = save(parent,"ALL_GATES_PASSED.json",dict(complete=True,suite_prepared_sha256=parent_sha,datasets=gates,GT_opened=False))
    for spec in source["specs"]:
        d = spec["dataset"]
        children[d] = save(parent,f"{d}/PREDICTIONS_SEALED.json",dict(complete=True,GT_opened=False,dataset=d,
            suite_prepared_sha256=parent_sha,images=2,modes=list(worker.MODES),image_method_evaluations=8,
            artifact_sha256=inventories[d],resource_observation=dict(synthetic=True)))
    current["barrier_sha256"] = save(parent,"ALL_PREDICTIONS_SEALED.json",dict(complete=True,GT_opened=False,
        suite_prepared_sha256=parent_sha,cells=8,image_method_evaluations=16,dataset_seals=children,audit_binding=audit,
        started_sha256=current["start_sha256"],gate_binding=dict(barrier_sha256=current["gate_sha256"],datasets=gates),legacy_archive=archive_spec))
    for key,value in dict(PARENT=parent,PARENT_SHA=parent_sha,PARENT_ADMISSION_SHA=admission_sha,
        PARENT_REPORT_SHA=report_sha,ANALYSIS=analysis_path,ANALYSIS_SHA=sha256(analysis_path)).items():monkeypatch.setattr(worker,key,value)
    base = copy.deepcopy(source)
    del base["legacy_archive"],base["prospective_analysis_sha256"]
    for spec in base["specs"]:spec.update(id="M02."+spec["dataset"],modes=["native","translated","global","confidence"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setattr(worker,"parent_terminal",lambda p:dict(ExecMainPID=str(p["pid"]),InvocationID=p["invocation"]))
    monkeypatch.setattr(worker,"preliminary_terminal",lambda:dict(ExecMainPID=str(current["pid"]),InvocationID=current["invocation"]))
    monkeypatch.setattr(worker.prediction.source,"source_inputs",lambda:copy.deepcopy(base))
    # Tiny synthetic populations cannot satisfy the full benchmark validator.
    monkeypatch.setattr(worker.prediction,"validate_inputs",lambda _:None)
    monkeypatch.setattr(worker.shared,"native_binding",lambda *_:None)
    monkeypatch.setattr(worker.prediction.general,"runtime_imports",lambda *_:None)
    monkeypatch.setattr(worker,"__file__",str(suite/"runtime"/"fixture.py"))
    bound = worker.bind_parent(current)
    inputs_sha = save(suite,"SCORE_INPUTS.json",bound)
    protocol_sha = save(suite,"PROTOCOL.json",dict(experiment="M03_offline_CPU_score",parent_prepared_sha256=parent_sha,
        parent_pins=current,prospective_analysis_sha256=worker.ANALYSIS_SHA,new_GPU=False))
    score_sha = save(suite,"SUITE_PREPARED.json",dict(schema_version=1,root=str(suite),runtime_hashes=inventory,
        inputs_sha256=inputs_sha,protocol_sha256=protocol_sha,parent_prepared_sha256=parent_sha,parent_pins=current))
    report2 = suite/"INDEPENDENT_REPORT.md"
    report2.write_text("Synthetic CPU score audit, not actual M03 GT permission\n")
    save(suite,"ADMITTED.json",dict(decision="ADMITTED",suite_prepared_sha256=score_sha,independent_reviewer="SyntheticCPU",
        CPU_GT_scoring_allowed=True,M03_GT_scoring_allowed=True,report_path=str(report2),report_sha256=sha256(report2)))
    original = worker.analysis.paired_summary
    monkeypatch.setattr(worker.analysis,"paired_summary",lambda v,d:original(v,d,replicates=20))
    return suite,score_sha,bound


def test_actual_synthetic_controller_streams_only_after_new_admission_and_seals_last(tmp_path,monkeypatch):
    suite,digest,bound = fixture(tmp_path,monkeypatch)
    original = Path.open
    forbidden = {r["image"] for rows in bound["source"]["rows"].values() for r in rows}
    def guard(path,*args,**kwargs):
        assert str(path) not in forbidden
        if str(path) in bound["source"]["annotation_metadata"]:
            assert (suite/"STARTED.json").exists() and not (suite/"ANALYSIS_SEALED.json").exists()
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    worker.run(suite,digest)
    seal = json.loads((suite/"ANALYSIS_SEALED.json").read_bytes())
    assert seal["complete"] and seal["cells"] == 8 and seal["image_method_evaluations"] == 16
    assert len(seal["artifact_sha256"]) == 4 and not torch.cuda.is_initialized()
    results = json.loads((suite/"DATASET_RESULTS.json").read_bytes())
    assert all(v["contrasts"]["single_C_vs_legacy_image"]["delta_pp"] == 100. for v in results.values())
    with pytest.raises(ValueError,match="resume/retry"):worker.run(suite,digest)


def test_running_parent_rejects_prepare_before_source_archive_or_directory(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    def blocked(_):raise ValueError("parent still running")
    monkeypatch.setattr(worker,"parent_terminal",blocked)
    monkeypatch.setattr(worker,"bind_parent",lambda *_:pytest.fail("Cannot bind a running parent"))
    with pytest.raises(ValueError,match="still running"):worker.prepare(tmp_path/"score",pins())
    assert not (tmp_path/"score").exists()


@pytest.mark.parametrize("entry",["admission","run"])
def test_running_parent_prevents_admission_and_run_before_source_or_metadata(tmp_path,monkeypatch,entry):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    def blocked():raise ValueError("parent still running")
    monkeypatch.setattr(worker,"preliminary_terminal",blocked)
    monkeypatch.setattr(worker.prediction.source,"source_inputs",lambda:pytest.fail("Running parent reached source"))
    monkeypatch.setattr(worker.protected,"metadata",lambda *_a,**_k:pytest.fail("Running parent reached metadata"))
    with pytest.raises(ValueError,match="still running"):getattr(worker,entry)(tmp_path/"score","a"*64)
    assert not (tmp_path/"score").exists()


@pytest.mark.parametrize("field,value",[("pid",True),("pid",0),("invocation","g"*32),("invocation","a"*31),
    ("start_sha256",None),("gate_sha256","a"*63),("barrier_sha256","g"*64)])
def test_bad_operator_pins_reject_before_systemctl(field,value,monkeypatch):
    current = pins()
    current[field] = value
    monkeypatch.setattr(worker.subprocess,"run",lambda *_a,**_k:pytest.fail("Invalid pins queried service"))
    with pytest.raises(ValueError):worker.parent_terminal(current)


@pytest.mark.parametrize("field,value",[("ActiveState","failed"),("SubState","running"),("MainPID","777"),
    ("ExecMainPID","778"),("ExecMainStatus","1"),("Result","exit-code"),("InvocationID","b"*32)])
def test_exact_retained_terminal_identity_not_idle_or_running_success(monkeypatch,field,value):
    current = pins()
    expected = dict(LoadState="loaded",ActiveState="active",SubState="exited",MainPID="0",ExecMainPID="777",
        ExecMainCode="1",ExecMainStatus="0",Result="success",InvocationID="a"*32)
    actual = dict(expected,**{field:value})
    monkeypatch.setattr(worker.subprocess,"run",lambda *a,**k:type("R",(),dict(stdout="\n".join(f"{k}={v}" for k,v in actual.items())))())
    with pytest.raises(ValueError,match="successfully exited"):worker.parent_terminal(current)
    actual.update(expected)
    assert worker.parent_terminal(current) == expected


@pytest.mark.parametrize("fault",["global_pin","parent_prepared","parent_admission","parent_report","start_pin","gate_pin","protocol","failure"])
def test_parent_graph_faults_precede_GT(tmp_path,monkeypatch,fault):
    suite,_,bound = fixture(tmp_path,monkeypatch)
    current = copy.deepcopy(bound["parent_pins"])
    if fault == "global_pin":current["barrier_sha256"] = "0"*64
    elif fault == "start_pin":current["start_sha256"] = "0"*64
    elif fault == "gate_pin":current["gate_sha256"] = "0"*64
    elif fault == "failure":write_new(worker.PARENT/"FAILED.json",dict(synthetic=True))
    else:monkeypatch.setattr(worker,dict(parent_prepared="PARENT_SHA",parent_admission="PARENT_ADMISSION_SHA",parent_report="PARENT_REPORT_SHA",protocol="ANALYSIS_SHA")[fault],"0"*64)
    monkeypatch.setattr(worker.payload,"annotation_bytes",lambda *_:pytest.fail("Parent fault read GT"))
    with pytest.raises(ValueError):worker.bind_parent(current)
    assert not (suite/"STARTED.json").exists()


@pytest.mark.parametrize("fault",["CPU_permission","M03_permission","report_alias","prepared_alias","runtime_alias","input_alias","archive_GT","archive_RGB","archive_source"])
def test_distinct_admission_and_role_aliases_reject_before_data_content(tmp_path,monkeypatch,fault):
    suite,digest,bound = fixture(tmp_path,monkeypatch)
    if fault.endswith("permission"):
        path = suite/"ADMITTED.json"
        value = json.loads(path.read_bytes())
        value["CPU_GT_scoring_allowed" if fault == "CPU_permission" else "M03_GT_scoring_allowed"] = False
        path.unlink()
        write_new(path,value)
    else:
        targets = dict(report_alias=suite/"INDEPENDENT_REPORT.md",prepared_alias=suite/"SUITE_PREPARED.json",
            runtime_alias=suite/"runtime"/"fixture.py",input_alias=suite/"SCORE_INPUTS.json")
        if fault.startswith("archive_"):
            path = worker.prediction.legacy_rho.ARCHIVE
            victim = Path(bound["source"]["rows"]["pc60"][0]["label" if fault == "archive_GT" else "image"]) if fault != "archive_source" else worker.PARENT/"runtime"/"fixture.py"
        else:path,victim = targets[fault],Path(bound["source"]["rows"]["pc60"][0]["label"])
        path.unlink()
        path.hardlink_to(victim)
    blocked = {r[k] for rows in bound["source"]["rows"].values() for r in rows for k in ("image","label")}
    original = Path.open
    def guard(path,*args,**kwargs):
        assert str(path) not in blocked
        if fault.startswith("archive_"):assert path != worker.prediction.legacy_rho.ARCHIVE
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    with pytest.raises(ValueError):worker.run(suite,digest)
    assert not (suite/"STARTED.json").exists()


@pytest.mark.parametrize("fault",["missing","GT_alias","RGB_alias","oversize"])
def test_late_second_dataset_payload_fault_precedes_any_GT(tmp_path,monkeypatch,fault):
    suite,digest,bound = fixture(tmp_path,monkeypatch)
    path = worker.PARENT/"pc60/predictions/legacy.episode/000001.npy"
    if fault == "oversize":
        with path.open("wb") as stream:stream.truncate(2*1024**2+1)
    else:
        path.unlink()
        if fault != "missing":path.hardlink_to(bound["source"]["rows"]["pc60"][0]["label" if fault == "GT_alias" else "image"])
    monkeypatch.setattr(worker.payload,"annotation_bytes",lambda *_:pytest.fail("Bad graph read any GT"))
    with pytest.raises((ValueError,FileNotFoundError)):worker.admission(suite,digest)
    assert not (suite/"STARTED.json").exists()


@pytest.mark.parametrize("fault",["execute","final_parent","output","progress","publish","audit_drift","archive_drift"])
def test_failures_have_no_global_and_no_automatic_resume(tmp_path,monkeypatch,fault):
    suite,digest,_ = fixture(tmp_path,monkeypatch)
    def fail(*_a,**_k):raise RuntimeError("synthetic failure")
    if fault == "execute":monkeypatch.setattr(worker,"execute",fail)
    elif fault == "final_parent":
        original,calls = worker.bind_parent,[]
        def bind(current):
            calls.append(1)
            if len(calls)>1:raise RuntimeError("synthetic failure")
            return original(current)
        monkeypatch.setattr(worker,"bind_parent",bind)
    elif fault == "output":monkeypatch.setattr(worker,"OUTPUT_CAP",1)
    elif fault == "progress":monkeypatch.setattr(worker.shared,"write_json",fail)
    elif fault == "publish":monkeypatch.setattr(worker,"publish_seal",fail)
    else:
        original = worker.execute
        def execute(*args):
            result = original(*args)
            path = worker.prediction.legacy_rho.ARCHIVE if fault == "archive_drift" else suite/"INDEPENDENT_REPORT.md"
            with path.open("ab") as stream:stream.write(b"drift")
            return result
        monkeypatch.setattr(worker,"execute",execute)
    with pytest.raises((ValueError,RuntimeError)):worker.run(suite,digest)
    assert (suite/"FAILED.json").exists() and not (suite/"ANALYSIS_SEALED.json").exists()


def test_post_STARTED_stat_failure_is_owned_failed_receipt_not_unmarked_partial(tmp_path,monkeypatch):
    suite,digest,bound = fixture(tmp_path,monkeypatch)
    original = worker.os.fsync
    def fsync(fd):
        result = original(fd)
        if worker.os.readlink(f"/proc/self/fd/{fd}") == str(suite/"STARTED.json"):
            with Path(bound["source"]["rows"]["pc60"][-1]["label"]).open("ab") as stream:stream.write(b"changed")
        return result
    monkeypatch.setattr(worker.os,"fsync",fsync)
    with pytest.raises(ValueError,match="annotation"):worker.run(suite,digest)
    assert (suite/"STARTED.json").exists() and (suite/"FAILED.json").exists()
    assert not (suite/"ANALYSIS_SEALED.json").exists()
