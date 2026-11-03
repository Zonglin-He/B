import copy
import json
from pathlib import Path
import shutil

import pytest
import torch

from test_value_score_receipts import fixture as receipt_fixture
import numpy as np
from PIL import Image
from tools.evolution_fifty import value_score_payload as payload
from tools.evolution_fifty import value_score_worker as worker
from tools.evolution_fifty.intake import sha256,write_new



def stream_fixture(tmp_path,dataset,*,images=2):
    parent,suite = tmp_path/"parent",tmp_path/"score"
    parent.mkdir()
    suite.mkdir()
    points = payload.prediction.runtime.points(dataset)
    _,spec,_,_ = receipt_fixture(dataset,points[0])
    spec["images"] = images
    spec["vocabulary"]["reduce_zero_label"] = False
    rows,inventory = [],{}
    def save(name,value):
        path = parent/name
        path.parent.mkdir(parents=True,exist_ok=True)
        if isinstance(value,np.ndarray):np.save(path,value,allow_pickle=False)
        elif isinstance(value,bytes):path.write_bytes(value)
        else:write_new(path,value)
        inventory[name] = sha256(path)
    for i in range(images):
        gt,rgb = tmp_path/f"GT{i}.png",tmp_path/f"RGB{i}.rgb"
        image = Image.fromarray(np.array([[0,1,2],[2,0,1]],dtype=np.uint8)).convert("P")
        image.putpalette([255,0,0,0,255,0,0,0,255]+[0]*759)
        image.save(gt)
        rgb.write_bytes(b"No RGB content should open during scoring")
        st = gt.stat()
        details = dict(device=st.st_dev,inode=st.st_ino,size_bytes=st.st_size,mtime_ns=st.st_mtime_ns,contents_read=False)
        rows.append(dict(official_id=f"s{i}",image=str(rgb),label=str(gt),label_metadata=[st.st_size,st.st_mtime_ns],label_metadata_details=details))
    costs = []
    by_name = {p["id"]:p for p in points}
    names = tuple(by_name)
    for episode,start in enumerate(range(0,images,2)):
        current = rows[start:start+2]
        for position,name in enumerate(payload.prediction.prediction.runtime.prior.arm_order(names,episode,"alternating")):
            point = by_name[name]
            saved,_,_,cost = receipt_fixture(dataset,point,empty=len(current) == 1)
            extra = dict(episode=episode,order_position=position,warmup=episode == 0)
            if len(current) == 1:extra.update(images=1,patches=1,eval_forward_inputs=2,encoder_inputs=2)
            saved.update(extra,ordered_ids=[r["official_id"] for r in current])
            cost.update(extra)
            save(f"{dataset}/episodes/{episode:06d}.{name}.json",saved)
            costs.append(cost)
            for i,row in enumerate(current,start):
                labels = payload.decode_annotation(Path(row["label"]).read_bytes(),(224,224),3,False)
                prediction = labels if point == points[1] and len(current) > 1 else (labels+1)%3
                save(f"{dataset}/predictions/{name}/{i:06d}.npy",prediction.astype(np.uint16))
    save(f"{dataset}/cost.jsonl","".join(json.dumps(r)+"\n" for r in costs).encode())
    legal = dict(parameter_tensors=1,parameter_elements=1,
        trainable_names=["backbone.visual.ln_pre.weight"],visual_LN_names=["visual.ln_pre.weight"])
    initial = dict(legal_visual_LN=legal,frozen_parameter_sha256="a"*64,reset=dict(trainable_sha256="b"*64,optimizer_sha256="c"*64))
    save(f"{dataset}/initial_state.json",initial)
    counters = (*payload.prediction.runtime.native.COUNTERS,"optimizer_steps")
    summary = dict(dataset=dataset,images=images,points=list(points),GT_opened=False,
        time_matching_done=False,accuracy_or_score_selected_points=False,legal_visual_LN=legal,
        frozen_parameter_sha256=initial["frozen_parameter_sha256"],
        actual_work={name:{k:sum(r[k] for r in costs if r["arm"] == name) for k in counters} for name in names},
        raw_model_seconds={name:sum(r["seconds"] for r in costs if r["arm"] == name) for name in names},
        warmup_adjusted_throughput=payload.prediction.prediction.runtime.prior.timing_summary(costs,names))
    save(f"{dataset}/prediction_result.json",summary)
    source = dict(rows={dataset:rows},specs=[spec],annotation_metadata={r["label"]:r["label_metadata_details"] for r in rows})
    bound = dict(parent=str(parent),source=source,seals={dataset:dict(artifact_sha256=inventory)})
    bound["payload_metadata"] = payload.preflight_payloads(parent,[spec],bound["seals"],source)
    return suite,bound,spec,save


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
        _,bound,spec,_ = stream_fixture(container,dataset)
        shutil.copytree(Path(bound["parent"])/dataset,parent/dataset)
        summary_path = parent/dataset/"prediction_result.json"
        summary = json.loads(summary_path.read_bytes())
        summary["dataset"] = dataset
        summary_path.unlink()
        write_new(summary_path,summary)
        inventories[dataset] = {str(p.relative_to(parent)):sha256(p) for p in (parent/dataset).rglob("*") if p.is_file()}
        spec = copy.deepcopy(spec)
        spec.update(dataset=dataset,id="V01."+dataset)
        spec["config"]["dataset"] = dict(pc60="PascalContext60Dataset",coco_object="COCOObjectDataset")[dataset]
        source["specs"].append(spec)
        source["rows"][dataset] = bound["source"]["rows"][dataset]
        source["annotation_metadata"].update(bound["source"]["annotation_metadata"])
        monkeypatch.setitem(worker.POPULATIONS,dataset,2)
    for root in (parent,suite):
        (root/"runtime").mkdir()
        (root/"runtime"/"fixture.py").write_text("Synthetic_CPU_only=True\n")
    inventory = {"fixture.py":sha256(parent/"runtime"/"fixture.py")}
    report = parent/"INDEPENDENT_REPORT.md"
    report.write_text("Synthetic parent report, not actual V01 prediction permission\n")
    report_sha = sha256(report)
    analysis_path = tmp_path/"ANALYSIS.json"
    value_path = tmp_path/"VALUE.json"
    save(tmp_path,"VALUE.json",dict(V01=dict(frozen_grid_conditions=10,image_method_evaluations_if_all_run=50525)))
    monkeypatch.setattr(worker.prediction,"VALUE_PROTOCOL",value_path)
    monkeypatch.setattr(worker.prediction,"VALUE_PROTOCOL_SHA",sha256(value_path))
    source.update(value_protocol_path=str(value_path),value_protocol_sha256=sha256(value_path),
                  prospective_value=json.loads(value_path.read_bytes())["V01"])
    save(tmp_path,"ANALYSIS.json",dict(parent_value_protocol_sha256=sha256(value_path),datasets=list(worker.DATASETS),
        points={d:{arm:[p["steps"] for p in worker.value_runtime.points(d) if p["arm"] == arm]
                   for arm in ("image_C","image_confidence")} for d in worker.DATASETS},
        uncertainty=dict(replicates=10000,root_seed=20261102)))
    inputs_sha = save(parent,"INPUTS.json",source)
    protocol_sha = save(parent,"PROTOCOL.json",dict(synthetic_CPU=True))
    parent_sha = save(parent,"SUITE_PREPARED.json",dict(schema_version=1,root=str(parent),
        inputs_sha256=inputs_sha,protocol_sha256=protocol_sha,parent_native_hashes={},runtime_hashes=inventory))
    admission_sha = save(parent,"ADMITTED.json",dict(decision="ADMITTED",suite_prepared_sha256=parent_sha,
        V01_predictions_allowed=True,GT_scoring_allowed=False))
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
            suite_prepared_sha256=parent_sha,images=2,points=list(worker.value_runtime.points(d)),image_method_evaluations=10,
            artifact_sha256=inventories[d],resource_observation=dict(synthetic=True)))
    current["barrier_sha256"] = save(parent,"ALL_PREDICTIONS_SEALED.json",dict(complete=True,GT_opened=False,
        suite_prepared_sha256=parent_sha,cells=10,image_method_evaluations=20,dataset_seals=children,audit_binding=audit,
        started_sha256=current["start_sha256"],gate_binding=dict(barrier_sha256=current["gate_sha256"],datasets=gates)))
    for key,value in dict(PARENT=parent,PARENT_SHA=parent_sha,PARENT_ADMISSION_SHA=admission_sha,
        PARENT_REPORT_SHA=report_sha,ANALYSIS=analysis_path,ANALYSIS_SHA=sha256(analysis_path)).items():monkeypatch.setattr(worker,key,value)
    base = copy.deepcopy(source)
    del base["value_protocol_path"],base["value_protocol_sha256"],base["prospective_value"]
    for spec in base["specs"]:
        spec.update(id="M02."+spec["dataset"],modes=["native","translated","global","confidence"])
        del spec["points"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setattr(worker,"parent_terminal",lambda p:dict(ExecMainPID=str(p["pid"]),InvocationID=p["invocation"]))
    monkeypatch.setattr(worker,"preliminary_terminal",lambda:dict(ExecMainPID=str(current["pid"]),InvocationID=current["invocation"]))
    monkeypatch.setattr(worker.prediction.source,"source_inputs",lambda:copy.deepcopy(base))
    # Tiny synthetic populations cannot satisfy the full benchmark validator.
    monkeypatch.setattr(worker.prediction,"validate_inputs",lambda *_:None)
    monkeypatch.setattr(worker.shared,"native_binding",lambda *_:None)
    monkeypatch.setattr(worker.prediction.general,"runtime_imports",lambda *_:None)
    monkeypatch.setattr(worker,"__file__",str(suite/"runtime"/"fixture.py"))
    bound = worker.bind_parent(current)
    inputs_sha = save(suite,"SCORE_INPUTS.json",bound)
    protocol_sha = save(suite,"PROTOCOL.json",dict(experiment="V01_offline_CPU_score",parent_prepared_sha256=parent_sha,
        parent_pins=current,prospective_analysis_sha256=worker.ANALYSIS_SHA,new_GPU=False))
    score_sha = save(suite,"SUITE_PREPARED.json",dict(schema_version=1,root=str(suite),runtime_hashes=inventory,
        inputs_sha256=inputs_sha,protocol_sha256=protocol_sha,parent_prepared_sha256=parent_sha,parent_pins=current))
    report2 = suite/"INDEPENDENT_REPORT.md"
    report2.write_text("Synthetic CPU score audit, not actual V01 GT permission\n")
    save(suite,"ADMITTED.json",dict(decision="ADMITTED",suite_prepared_sha256=score_sha,independent_reviewer="SyntheticCPU",
        CPU_GT_scoring_allowed=True,V01_GT_scoring_allowed=True,report_path=str(report2),report_sha256=sha256(report2)))
    original = worker.analysis.summarize
    monkeypatch.setattr(worker.analysis,"summarize",lambda d,v,c:original(d,v,c,replicates=20))
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
    assert seal["complete"] and seal["cells"] == 10 and seal["image_method_evaluations"] == 20
    assert len(seal["artifact_sha256"]) == 4 and not torch.cuda.is_initialized()
    results = json.loads((suite/"DATASET_RESULTS.json").read_bytes())
    assert all(v["native_C_minus_same_steps_Confidence"]["difference_pp"] == 100. for v in results.values())
    with pytest.raises(ValueError,match="resume/retry"):worker.run(suite,digest)


def test_running_parent_rejects_prepare_before_source_or_directory(tmp_path,monkeypatch):
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


@pytest.mark.parametrize("fault",["CPU_permission","V01_permission","report_alias","prepared_alias","runtime_alias","input_alias"])
def test_distinct_admission_and_role_aliases_reject_before_data_content(tmp_path,monkeypatch,fault):
    suite,digest,bound = fixture(tmp_path,monkeypatch)
    if fault.endswith("permission"):
        path = suite/"ADMITTED.json"
        value = json.loads(path.read_bytes())
        value["CPU_GT_scoring_allowed" if fault == "CPU_permission" else "V01_GT_scoring_allowed"] = False
        path.unlink()
        write_new(path,value)
    else:
        targets = dict(report_alias=suite/"INDEPENDENT_REPORT.md",prepared_alias=suite/"SUITE_PREPARED.json",
            runtime_alias=suite/"runtime"/"fixture.py",input_alias=suite/"SCORE_INPUTS.json")
        path,victim = targets[fault],Path(bound["source"]["rows"]["pc60"][0]["label"])
        path.unlink()
        path.hardlink_to(victim)
    blocked = {r[k] for rows in bound["source"]["rows"].values() for r in rows for k in ("image","label")}
    original = Path.open
    def guard(path,*args,**kwargs):
        assert str(path) not in blocked
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    with pytest.raises(ValueError):worker.run(suite,digest)
    assert not (suite/"STARTED.json").exists()


@pytest.mark.parametrize("fault",["missing","GT_alias","RGB_alias","oversize"])
def test_late_second_dataset_payload_fault_precedes_any_GT(tmp_path,monkeypatch,fault):
    suite,digest,bound = fixture(tmp_path,monkeypatch)
    path = worker.PARENT/f"pc60/predictions/{worker.value_runtime.points('pc60')[-1]['id']}/000001.npy"
    if fault == "oversize":
        with path.open("wb") as stream:stream.truncate(2*1024**2+1)
    else:
        path.unlink()
        if fault != "missing":path.hardlink_to(bound["source"]["rows"]["pc60"][0]["label" if fault == "GT_alias" else "image"])
    monkeypatch.setattr(worker.payload,"annotation_bytes",lambda *_:pytest.fail("Bad graph read any GT"))
    with pytest.raises((ValueError,FileNotFoundError)):worker.admission(suite,digest)
    assert not (suite/"STARTED.json").exists()


@pytest.mark.parametrize("fault",["execute","final_parent","output","progress","publish","audit_drift"])
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
            path = suite/"INDEPENDENT_REPORT.md"
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


@pytest.mark.parametrize("fault",["fsync","close","close_mutation","fstat","identity"])
def test_actual_exclusive_STARTED_creation_fault_is_owned_without_GT(tmp_path,monkeypatch,fault):
    suite,digest,_ = fixture(tmp_path,monkeypatch)
    path = suite/"STARTED.json"
    original_open,original_sync,original_stat = Path.open,worker.os.fsync,worker.os.fstat
    def deny(*_a,**_k):pytest.fail("STARTED failure reached execute or GT")
    monkeypatch.setattr(worker,"execute",deny)
    monkeypatch.setattr(worker.payload,"annotation_bytes",deny)
    def target(fd):return worker.os.readlink(f"/proc/self/fd/{fd}") == str(path)
    def fsync(fd):
        if target(fd):
            if fault == "fsync":raise OSError("STARTED fsync injected")
            if fault == "identity":
                path.unlink()
                with original_open(path,"xb") as out:out.write(b"different marker")
        return original_sync(fd)
    def fstat(fd):
        if fault == "fstat" and target(fd):raise OSError("STARTED fstat injected")
        return original_stat(fd)
    class CloseFault:
        def __init__(self,stream):self.stream=stream
        def __getattr__(self,name):return getattr(self.stream,name)
        def __enter__(self):return self
        def __exit__(self,*args):
            result=self.stream.__exit__(*args)
            if fault == "close":raise OSError("STARTED close injected")
            if fault == "close_mutation":
                with original_open(path,"ab") as out:out.write(b"tamper")
            return result
    def opened(p,*args,**kwargs):
        stream=original_open(p,*args,**kwargs)
        return CloseFault(stream) if p == path and args == ("xb",) and fault.startswith("close") else stream
    monkeypatch.setattr(Path,"open",opened)
    monkeypatch.setattr(worker.os,"fsync",fsync)
    monkeypatch.setattr(worker.os,"fstat",fstat)
    with pytest.raises((OSError,ValueError)):worker.run(suite,digest)
    assert path.exists() and (suite/"FAILED.json").exists() and not (suite/"ANALYSIS_SEALED.json").exists()


def test_racing_foreign_STARTED_is_not_claimed_or_modified(tmp_path,monkeypatch):
    suite,digest,_ = fixture(tmp_path,monkeypatch)
    path = suite/"STARTED.json"
    original = Path.open
    def opened(p,*args,**kwargs):
        if p == path and args == ("xb",):
            with original(path,"xb") as stream:stream.write(b"foreign run")
        return original(p,*args,**kwargs)
    monkeypatch.setattr(Path,"open",opened)
    with pytest.raises(FileExistsError):worker.run(suite,digest)
    assert path.read_bytes() == b"foreign run"
    assert not (suite/"FAILED.json").exists() and not (suite/"ANALYSIS_SEALED.json").exists()


def test_prepare_snapshot_admission_run_with_full_bootstrap_and_no_prediction_hook(tmp_path,monkeypatch):
    _,_,bound = fixture(tmp_path,monkeypatch)
    original_analysis = __import__("importlib").reload(worker.analysis).summarize
    monkeypatch.setattr(worker.analysis,"summarize",original_analysis)
    source_root = tmp_path/"synthetic_sources"
    for name in ("trace_ovss","tools","tests"):
        (source_root/name).mkdir(parents=True)
        (source_root/name/"__init__.py").write_text("Synthetic_CPU_only=True\n")
    monkeypatch.setattr(worker,"ROOT",source_root)
    def no_hook(*_a,**_k):pytest.fail("Scorer called prediction-only source/hook")
    monkeypatch.setattr(worker.prediction,"source_inputs",no_hook)
    monkeypatch.setattr(worker.prediction,"forbid_annotations",no_hook)
    fresh = tmp_path/"fresh_score"
    digest = worker.prepare(fresh,bound["parent_pins"])
    prepared = json.loads((fresh/"SUITE_PREPARED.json").read_bytes())
    assert prepared["GT_opened"] is False and prepared["CPU_scoring_started"] is False
    assert len(prepared["runtime_hashes"]) == 3
    report = fresh/"INDEPENDENT_REPORT.md"
    report.write_text("Synthetic new permission only\n")
    write_new(fresh/"ADMITTED.json",dict(decision="ADMITTED",suite_prepared_sha256=digest,
        independent_reviewer="SyntheticCPU",CPU_GT_scoring_allowed=True,V01_GT_scoring_allowed=True,
        report_path=str(report),report_sha256=sha256(report)))
    monkeypatch.setattr(worker,"__file__",str(fresh/"runtime"/"tools"/"__init__.py"))
    worker.run(fresh,digest)
    result = json.loads((fresh/"DATASET_RESULTS.json").read_bytes())
    for dataset in worker.DATASETS:
        assert result[dataset]["bootstrap"]["replicates"] == 10000
        assert len(result[dataset]["points"]) == 5
        assert result[dataset]["time_matching"]["near_time"] is True
        assert all(p["model_seconds"] == .2 for p in result[dataset]["points"].values())
    assert (fresh/"ANALYSIS_SEALED.json").exists() and not torch.cuda.is_initialized()


@pytest.mark.parametrize("target",["VALUE_PROTOCOL","ANALYSIS"])
@pytest.mark.parametrize("role",["GT","RGB","source"])
def test_prospective_protocol_aliases_reject_before_content_open(tmp_path,monkeypatch,target,role):
    suite,digest,bound = fixture(tmp_path,monkeypatch)
    path = worker.prediction.VALUE_PROTOCOL if target == "VALUE_PROTOCOL" else worker.ANALYSIS
    victim = worker.PARENT/"runtime"/"fixture.py" if role == "source" else Path(bound["source"]["rows"]["pc60"][0]["label" if role == "GT" else "image"])
    path.unlink()
    path.hardlink_to(victim)
    original = Path.open
    def guard(p,*args,**kwargs):
        # A pinned runtime source may be hashed in its own legitimate role;
        # the aliased protocol path must never be opened as metadata.
        assert p != path and (role == "source" or p != victim)
        return original(p,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    with pytest.raises(ValueError):worker.run(suite,digest)
    assert not (suite/"STARTED.json").exists()

