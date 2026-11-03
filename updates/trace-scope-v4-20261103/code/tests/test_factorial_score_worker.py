import copy
import json
from pathlib import Path
import shutil

import pytest
import torch

from test_factorial_score_payload import fixture as payload_fixture
from tools.evolution_fifty import factorial_score_worker as worker
from tools.evolution_fifty.intake import sha256,write_new


def fixture(tmp_path,monkeypatch):
    parent,suite = tmp_path/"parent",tmp_path/"score"
    parent.mkdir()
    suite.mkdir()
    source = dict(rows={},specs=[],annotation_metadata={})
    inventories,children = {},{}
    def save(root,name,value):
        path = root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        write_new(path,value)
        return sha256(path)
    for dataset in worker.DATASETS:
        container = tmp_path/(dataset+"_fixture")
        container.mkdir()
        _,bound,spec,_ = payload_fixture(container)
        original = Path(bound["parent"])/"coco_object"
        shutil.copytree(original,parent/dataset)
        summary_path = parent/dataset/"prediction_result.json"
        summary = json.loads(summary_path.read_bytes())
        summary["dataset"] = dataset
        summary_path.unlink()
        write_new(summary_path,summary)
        inventories[dataset] = {str(p.relative_to(parent)):sha256(p) for p in (parent/dataset).rglob("*") if p.is_file()}
        spec = dict(spec,dataset=dataset)
        source["specs"].append(spec)
        source["rows"][dataset] = bound["source"]["rows"]["coco_object"]
        source["annotation_metadata"].update(bound["source"]["annotation_metadata"])
        monkeypatch.setitem(worker.POPULATIONS,dataset,2)
    for root in (parent,suite):
        (root/"runtime").mkdir()
        (root/"runtime"/"fixture.py").write_text("Synthetic_CPU_only=True\n")
    runtime_inventory = {"fixture.py":sha256(parent/"runtime"/"fixture.py")}
    inputs_sha = save(parent,"INPUTS.json",source)
    protocol_sha = save(parent,"PROTOCOL.json",dict(synthetic_CPU=True))
    parent_sha = save(parent,"SUITE_PREPARED.json",dict(schema_version=1,root=str(parent),
        inputs_sha256=inputs_sha,protocol_sha256=protocol_sha,parent_native_hashes={},runtime_hashes=runtime_inventory))
    report = parent/"INDEPENDENT_REPORT.md"
    report.write_text("Synthetic parent audit, not actual GPU permission\n")
    report_sha = sha256(report)
    admission_sha = save(parent,"ADMITTED.json",dict(decision="ADMITTED",suite_prepared_sha256=parent_sha,
        M04_predictions_allowed=True,GT_scoring_allowed=False))
    def identity(path):
        s = path.stat()
        return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
    audit = dict(admission_path=str(parent/"ADMITTED.json"),admission_sha256=admission_sha,
        admission_identity=identity(parent/"ADMITTED.json"),report_path=str(report),report_sha256=report_sha,
        report_identity=identity(report),independent_reviewer="SyntheticCPU")
    start_sha = save(parent,"STARTED.json",dict(suite_prepared_sha256=parent_sha,audit=audit,pid=2121324,GT_opened=False))
    gates = {d:save(parent,f"gates/{d}.json",dict(passed=True,dataset=d,actual_RGB_sources=4,GT_opened=False))
             for d in worker.DATASETS}
    gate_sha = save(parent,"ALL_GATES_PASSED.json",dict(complete=True,suite_prepared_sha256=parent_sha,datasets=gates,GT_opened=False))
    for spec in source["specs"]:
        d = spec["dataset"]
        children[d] = save(parent,f"{d}/PREDICTIONS_SEALED.json",dict(complete=True,GT_opened=False,
            suite_prepared_sha256=parent_sha,dataset=d,images=2,studies=list(worker.core.STUDIES),
            factorial_arms=list(worker.core.runtime.ARMS),image_method_evaluations=16,
            artifact_sha256=inventories[d],resource_observation=dict(synthetic=True)))
    barrier_sha = save(parent,"ALL_PREDICTIONS_SEALED.json",dict(complete=True,GT_opened=False,
        suite_prepared_sha256=parent_sha,cells=16,image_method_evaluations=32,dataset_seals=children,
        audit_binding=audit,started_sha256=start_sha,gate_binding=dict(barrier_sha256=gate_sha,datasets=gates)))
    analysis = tmp_path/"ANALYSIS.json"
    write_new(analysis,dict(datasets={d:2 for d in worker.DATASETS},studies=list(worker.core.STUDIES),
        arms=list(worker.core.runtime.ARMS),contrast_weights_in_arm_order={k:list(v) for k,v in worker.analysis.CONTRASTS.items()},
        uncertainty=dict(replicates=10000,root_seed=20261102)))
    changes = dict(PARENT=parent,PARENT_SHA=parent_sha,PARENT_START_SHA=start_sha,
        PARENT_ADMISSION_SHA=admission_sha,PARENT_REPORT_SHA=report_sha,PARENT_GATE_SHA=gate_sha,
        ANALYSIS=analysis,ANALYSIS_SHA=sha256(analysis))
    for name,value in changes.items():monkeypatch.setattr(worker,name,value)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setattr(worker,"parent_terminal",lambda:dict(ExecMainPID="2121324",InvocationID="synthetic"))
    monkeypatch.setattr(worker.prediction,"source_inputs",lambda:copy.deepcopy(source))
    monkeypatch.setattr(worker.shared,"native_binding",lambda *_:None)
    monkeypatch.setattr(worker.prediction.general,"runtime_imports",lambda *_:None)
    monkeypatch.setattr(worker,"__file__",str(suite/"runtime"/"fixture.py"))
    bound = worker.bind_parent(barrier_sha)
    inputs_sha = save(suite,"SCORE_INPUTS.json",bound)
    protocol_sha = save(suite,"PROTOCOL.json",dict(synthetic_CPU=True,
        receipt_arithmetic=dict(torch_version=worker.payload.receipts.CUDA_DIVISION_VERSION,
            torch_git=worker.payload.receipts.CUDA_DIVISION_GIT,source=worker.payload.receipts.CUDA_DIVISION_SOURCE,
            coefficient="Pinned CUDA tensor division followed by opmath CPU-scalar reciprocal multiplication and native cast; exact equality, no tolerance")))
    score_sha = save(suite,"SUITE_PREPARED.json",dict(schema_version=1,root=str(suite),
        inputs_sha256=inputs_sha,protocol_sha256=protocol_sha,runtime_hashes=runtime_inventory,
        parent_prepared_sha256=parent_sha,parent_barrier_sha256=barrier_sha))
    report2 = suite/"INDEPENDENT_REPORT.md"
    report2.write_text("Synthetic score audit, not actual M04 GT permission\n")
    save(suite,"ADMITTED.json",dict(decision="ADMITTED",suite_prepared_sha256=score_sha,
        independent_reviewer="SyntheticCPU",CPU_GT_scoring_allowed=True,M04_GT_scoring_allowed=True,
        report_path=str(report2),report_sha256=sha256(report2)))
    original = worker.analysis.paired_summary
    monkeypatch.setattr(worker.analysis,"paired_summary",lambda v,d:original(v,d,replicates=20))
    return suite,score_sha,bound


def test_synthetic_full_streaming_admission_run_and_final_global(tmp_path,monkeypatch):
    suite,digest,bound = fixture(tmp_path,monkeypatch)
    original = Path.open
    def guard(path,*args,**kwargs):
        if str(path) in bound["source"]["annotation_metadata"]:
            assert (suite/"STARTED.json").exists() and not (suite/"ANALYSIS_SEALED.json").exists()
        assert str(path) not in {r["image"] for rows in bound["source"]["rows"].values() for r in rows}
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    worker.run(suite,digest)
    seal = json.loads((suite/"ANALYSIS_SEALED.json").read_bytes())
    assert seal["complete"] and seal["cells"] == 16 and seal["image_method_evaluations"] == 32
    assert len(seal["artifact_sha256"]) == 4
    assert not torch.cuda.is_initialized()
    result = json.loads((suite/"DATASET_RESULTS.json").read_bytes())
    assert all(v["studies"]["natural"]["contrasts"]["native_C_minus_Confidence"]["delta_pp"] == 100. for v in result.values())
    with pytest.raises(ValueError,match="resume/retry"):worker.run(suite,digest)


def test_running_parent_prevents_prepare_before_any_source_or_output(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    def blocked():raise ValueError("parent still running")
    monkeypatch.setattr(worker,"parent_terminal",blocked)
    monkeypatch.setattr(worker,"bind_parent",lambda *_:pytest.fail("No binding while running"))
    with pytest.raises(ValueError,match="still running"):worker.prepare(tmp_path/"new","a"*64)
    assert not (tmp_path/"new").exists()


@pytest.mark.parametrize("fault",["global_pin","parent_prepared","parent_admission","parent_start","gate_pin","protocol","parent_failure"])
def test_immutable_parent_graph_faults_reject_before_GT(tmp_path,monkeypatch,fault):
    suite,digest,bound = fixture(tmp_path,monkeypatch)
    if fault == "global_pin":bound["barrier_sha256"] = "a"*64
    elif fault == "parent_prepared":monkeypatch.setattr(worker,"PARENT_SHA","a"*64)
    elif fault == "parent_admission":monkeypatch.setattr(worker,"PARENT_ADMISSION_SHA","a"*64)
    elif fault == "parent_start":monkeypatch.setattr(worker,"PARENT_START_SHA","a"*64)
    elif fault == "gate_pin":monkeypatch.setattr(worker,"PARENT_GATE_SHA","a"*64)
    elif fault == "protocol":monkeypatch.setattr(worker,"ANALYSIS_SHA","a"*64)
    elif fault == "parent_failure":write_new(worker.PARENT/"FAILED.json",dict(synthetic=True))
    monkeypatch.setattr(worker.payload,"annotation_bytes",lambda *_:pytest.fail("Parent fault read GT"))
    with pytest.raises(ValueError):worker.bind_parent(bound["barrier_sha256"])
    assert not (suite/"STARTED.json").exists()


@pytest.mark.parametrize("fault",["CPU_permission","M04_permission","report_alias","prepared_alias","runtime_alias","input_alias"])
def test_distinct_exact_admission_and_metadata_aliases_precede_GT(tmp_path,monkeypatch,fault):
    suite,digest,bound = fixture(tmp_path,monkeypatch)
    if fault.endswith("permission"):
        path = suite/"ADMITTED.json"
        value = json.loads(path.read_bytes())
        value["CPU_GT_scoring_allowed" if fault == "CPU_permission" else "M04_GT_scoring_allowed"] = False
        path.unlink()
        write_new(path,value)
    else:
        paths = dict(report_alias=suite/"INDEPENDENT_REPORT.md",prepared_alias=suite/"SUITE_PREPARED.json",
            runtime_alias=suite/"runtime"/"fixture.py",input_alias=suite/"SCORE_INPUTS.json")
        path = paths[fault]
        path.unlink()
        path.hardlink_to(bound["source"]["rows"]["coco_object"][0]["label"])
    monkeypatch.setattr(worker.payload,"annotation_bytes",lambda *_:pytest.fail("Admission fault read GT"))
    with pytest.raises(ValueError):worker.run(suite,digest)
    assert not (suite/"STARTED.json").exists()


@pytest.mark.parametrize("fault",["execute","final_parent","output","progress","publish","audit_drift"])
def test_failures_never_leave_global_or_resume(tmp_path,monkeypatch,fault):
    suite,digest,bound = fixture(tmp_path,monkeypatch)
    def fail(*_args,**_kwargs):raise RuntimeError("synthetic final failure")
    if fault == "execute":monkeypatch.setattr(worker,"execute",fail)
    elif fault == "final_parent":
        original = worker.bind_parent
        calls = []
        def bind(pin):
            calls.append(1)
            if len(calls)>1:raise RuntimeError("synthetic final failure")
            return original(pin)
        monkeypatch.setattr(worker,"bind_parent",bind)
    elif fault == "output":monkeypatch.setattr(worker,"OUTPUT_CAP",1)
    elif fault == "progress":monkeypatch.setattr(worker.shared,"write_json",fail)
    elif fault == "publish":monkeypatch.setattr(worker,"publish_seal",fail)
    elif fault == "audit_drift":
        original = worker.execute
        def execute(*args):
            result = original(*args)
            with (suite/"INDEPENDENT_REPORT.md").open("ab") as stream:stream.write(b"drift")
            return result
        monkeypatch.setattr(worker,"execute",execute)
    with pytest.raises((ValueError,RuntimeError)):worker.run(suite,digest)
    assert (suite/"FAILED.json").exists() and not (suite/"ANALYSIS_SEALED.json").exists()


@pytest.mark.parametrize("fault",["missing","GT_alias","RGB_alias","source_alias","oversize"])
def test_late_second_dataset_prediction_graph_fault_rejects_admission_before_ANY_GT(tmp_path,monkeypatch,fault):
    suite,digest,bound = fixture(tmp_path,monkeypatch)
    parent = Path(bound["parent"])
    path = parent/"pc60/predictions/common_active.Dk_Oc/000001.npy"
    if fault == "oversize":
        with path.open("wb") as stream:stream.truncate(2*1024**2+1)
    else:
        path.unlink()
        if fault != "missing":
            victim = dict(GT_alias=Path(bound["source"]["rows"]["coco_object"][0]["label"]),
                RGB_alias=Path(bound["source"]["rows"]["coco_object"][0]["image"]),
                source_alias=parent/"runtime"/"fixture.py")[fault]
            path.hardlink_to(victim)
    monkeypatch.setattr(worker.payload,"annotation_bytes",lambda *_:pytest.fail("Graph fault read any GT"))
    original = Path.open
    forbidden = {r[k] for rows in bound["source"]["rows"].values() for r in rows for k in ("image","label")}
    def guard(p,*args,**kwargs):
        assert str(p) not in forbidden and "/predictions/" not in str(p)
        return original(p,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    with pytest.raises((ValueError,FileNotFoundError)):
        worker.admission(suite,digest)
    assert not (suite/"STARTED.json").exists() and not (suite/"ANALYSIS_SEALED.json").exists()
