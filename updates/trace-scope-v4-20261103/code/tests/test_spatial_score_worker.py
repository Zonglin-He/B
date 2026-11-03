import copy
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from tools.evolution_fifty import spatial_score_worker as worker
from tools.evolution_fifty.intake import sha256,write_new


def fixture(tmp_path,monkeypatch):
    parent,suite = tmp_path/"parent",tmp_path/"score"
    parent.mkdir()
    suite.mkdir()
    source = dict(rows={},specs=[],annotation_metadata={})
    inventories,children = {},{}
    legal = dict(parameter_tensors=1,parameter_elements=1,trainable_names=["backbone.visual.ln_pre.weight"],
        visual_LN_names=["visual.ln_pre.weight"])
    initial = dict(legal_visual_LN=legal,frozen_parameter_sha256="a"*64,
        reset=dict(trainable_sha256="b"*64,optimizer_sha256="c"*64))
    def save(root,relative,value,inventory=None):
        path = root/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        if isinstance(value,np.ndarray): np.save(path,value,allow_pickle=False)
        else: write_new(path,value)
        if inventory is not None: inventory[relative] = sha256(path)
        return sha256(path)
    for dataset in worker.DATASETS:
        rows,inventory = [],{}
        monkeypatch.setitem(worker.POPULATIONS,dataset,3)
        for i in range(3):
            label,rgb = tmp_path/f"{dataset}.GT{i}.png",tmp_path/f"{dataset}.RGB{i}"
            image = Image.fromarray(np.array([[0,1,0],[1,0,1]],np.uint8)).convert("P")
            image.putpalette([255,0,0,0,255,0]+[0]*762)
            image.save(label)
            rgb.write_bytes(b"Synthetic RGB forbidden during CPU scoring")
            state = label.stat()
            details = dict(device=state.st_dev,inode=state.st_ino,size_bytes=state.st_size,mtime_ns=state.st_mtime_ns,contents_read=False)
            rows.append(dict(official_id=f"{dataset}{i}",image=str(rgb),label=str(label),
                label_metadata=[state.st_size,state.st_mtime_ns],label_metadata_details=details))
            source["annotation_metadata"][str(label)] = details
        cfg = dict(batch_size=2,steps=30 if dataset == "coco_object" else 7,trace_max_regions=35,trace_view_batch_size=4)
        spec = dict(id="M02."+dataset,dataset=dataset,images=3,config=cfg,modes=list(worker.MODES),
            vocabulary=dict(classes=["a","b"],queries=["a","b"],query_to_class=None,reduce_zero_label=False))
        source["rows"][dataset] = rows
        source["specs"].append(spec)
        costs = []
        for episode,start in enumerate(range(0,3,2)):
            images = min(2,3-start)
            order = worker.prediction.general.runtime.prior.arm_order(worker.MODES,episode,"alternating")
            for position,mode in enumerate(order):
                row = dict(arm=mode,mode=mode,images=images,patches=images,steps=0,optimizer_steps=0,
                    seconds=.1+position*.2+episode*.3,episode=episode,order_position=position,warmup=episode == 0,
                    candidate_count=0,active_candidates=0,candidate_sha256=hashlib.sha256(b"").hexdigest(),
                    **{k:0 for k in worker.runtime.image_worker.COUNTERS})
                row.update(eval_forward_calls=2,eval_forward_inputs=2*images,encoder_calls=2,encoder_inputs=2*images)
                worker.runtime.validate_cost(row,[],cfg)
                costs.append(row)
                save(parent,f"{dataset}/episodes/{episode:06d}.{mode}.json",dict(**row,GT_opened=False,
                    scientific_identity="M02_spatial_"+mode,losses=[],candidates=[],preservation_geometry=[],
                    ordered_ids=[r["official_id"] for r in rows[start:start+2]]),inventory)
        for i,row in enumerate(rows):
            target = worker.decode_annotation(Path(row["label"]).read_bytes(),(224,224),2,False)
            for mode in worker.MODES:
                values = 1-target if mode in ("translated","confidence") else target
                save(parent,f"{dataset}/predictions/{mode}/{i:06d}.npy",values.astype(np.uint16),inventory)
        costpath = parent/dataset/"cost.jsonl"
        costpath.write_text("".join(json.dumps(row)+"\n" for row in costs))
        inventory[str(costpath.relative_to(parent))] = sha256(costpath)
        save(parent,f"{dataset}/initial_state.json",initial,inventory)
        summary = dict(dataset=dataset,images=3,modes=list(worker.MODES),GT_opened=False,
            actual_work={m:{k:sum(r[k] for r in costs if r["mode"] == m)
                for k in (*worker.runtime.image_worker.COUNTERS,"optimizer_steps")} for m in worker.MODES},
            raw_model_seconds={m:sum(r["seconds"] for r in costs if r["mode"] == m) for m in worker.MODES},
            warmup_adjusted_throughput=worker.prediction.general.runtime.prior.timing_summary(costs,worker.MODES),
            legal_visual_LN=legal,frozen_parameter_sha256=initial["frozen_parameter_sha256"])
        save(parent,f"{dataset}/prediction_result.json",summary,inventory)
        inventories[dataset] = inventory
    for root in (parent,suite):
        (root/"runtime").mkdir()
        (root/"runtime"/"fixture.py").write_text("CPU_fixture=True\n")
    parent_inputs = save(parent,"INPUTS.json",source)
    parent_protocol = save(parent,"PROTOCOL.json",dict(CPU_fixture=True))
    parent_sha = save(parent,"SUITE_PREPARED.json",dict(schema_version=1,root=str(parent),inputs_sha256=parent_inputs,
        protocol_sha256=parent_protocol,parent_native_hashes={},runtime_hashes={"fixture.py":sha256(parent/"runtime"/"fixture.py")}))
    report = parent/"INDEPENDENT_REPORT.md"
    report.write_text("Synthetic parent report, NOT actual admission\n")
    report_sha = sha256(report)
    audit_sha = save(parent,"ADMITTED.json",dict(decision="ADMITTED",suite_prepared_sha256=parent_sha,
        M02_predictions_allowed=True,GT_scoring_allowed=False))
    audit = dict(admission_path=str(parent/"ADMITTED.json"),admission_sha256=audit_sha,
        admission_identity=worker.protected.Path(parent/"ADMITTED.json").stat(),report_path=str(report))
    def identity(path):
        s=path.stat()
        return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
    audit.update(admission_identity=identity(parent/"ADMITTED.json"),report_identity=identity(report),
        report_sha256=report_sha,independent_reviewer="SyntheticCPU")
    start_sha = save(parent,"STARTED.json",dict(suite_prepared_sha256=parent_sha,audit=audit,pid=2086569,GT_opened=False))
    gatechildren = {}
    for dataset in worker.DATASETS:
        gatechildren[dataset] = save(parent,f"gates/{dataset}.json",dict(passed=True,dataset=dataset,actual_RGB_sources=4,GT_opened=False))
    gatesha = save(parent,"ALL_GATES_PASSED.json",dict(complete=True,suite_prepared_sha256=parent_sha,datasets=gatechildren,GT_opened=False))
    for spec in source["specs"]:
        d = spec["dataset"]
        children[d] = save(parent,f"{d}/PREDICTIONS_SEALED.json",dict(complete=True,GT_opened=False,
            suite_prepared_sha256=parent_sha,dataset=d,images=3,modes=list(worker.MODES),image_method_evaluations=12,
            artifact_sha256=inventories[d],resource_observation=dict(synthetic=True)))
    global_sha = save(parent,"ALL_PREDICTIONS_SEALED.json",dict(complete=True,GT_opened=False,suite_prepared_sha256=parent_sha,
        cells=8,image_method_evaluations=24,dataset_seals=children,audit_binding=audit,started_sha256=start_sha,
        gate_binding=dict(barrier_sha256=gatesha,datasets=gatechildren)))
    analysis = tmp_path/"ANALYSIS.json"
    write_new(analysis,dict(prediction_prepared_sha256=parent_sha,arm_order=list(worker.MODES),
        fixed_contrasts={k:list(v) for k,v in worker.analysis.CONTRASTS.items()},uncertainty=dict(replicates=10000,root_seed=20261102)))
    for k,v in dict(PARENT=parent,PARENT_SHA=parent_sha,PARENT_BARRIER_SHA=global_sha,PARENT_START_SHA=start_sha,
                    PARENT_ADMISSION_SHA=audit_sha,PARENT_REPORT_SHA=report_sha,ANALYSIS=analysis,ANALYSIS_SHA=sha256(analysis)).items():
        monkeypatch.setattr(worker,k,v)
    monkeypatch.setattr(worker,"parent_terminal",lambda:dict(ExecMainPID="2086569",InvocationID="synthetic"))
    monkeypatch.setattr(worker.prediction,"source_inputs",lambda:copy.deepcopy(source))
    monkeypatch.setattr(worker.prediction,"validate_inputs",lambda *_:None)
    monkeypatch.setattr(worker,"native_binding",lambda *_:None)
    monkeypatch.setattr(worker.prediction.general,"runtime_imports",lambda *_:None)
    monkeypatch.setattr(worker,"__file__",str(suite/"runtime"/"fixture.py"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    bound = worker.bind_parent()
    inputs_sha = save(suite,"SCORE_INPUTS.json",bound)
    protocol_sha = save(suite,"PROTOCOL.json",dict(CPU_fixture=True))
    score_sha = save(suite,"SUITE_PREPARED.json",dict(root=str(suite),schema_version=1,parent_prepared_sha256=parent_sha,
        inputs_sha256=inputs_sha,protocol_sha256=protocol_sha,runtime_hashes={"fixture.py":sha256(suite/"runtime"/"fixture.py")}))
    report2 = suite/"INDEPENDENT_REPORT.md"
    report2.write_text("Synthetic exact score report; NOT actual GT permission\n")
    save(suite,"ADMITTED.json",dict(decision="ADMITTED",suite_prepared_sha256=score_sha,independent_reviewer="SyntheticCPU",
        CPU_GT_scoring_allowed=True,M02_GT_scoring_allowed=True,report_path=str(report2),report_sha256=sha256(report2)))
    original = worker.analysis.paired_summary
    monkeypatch.setattr(worker.analysis,"paired_summary",lambda v,d:original(v,d,replicates=20))
    return suite,score_sha,bound,source


def test_actual_synthetic_admission_streaming_run_and_global_last(tmp_path,monkeypatch):
    suite,digest,bound,source = fixture(tmp_path,monkeypatch)
    original = Path.open
    def guard(path,*args,**kwargs):
        if str(path) in {r["image"] for rows in source["rows"].values() for r in rows}:
            pytest.fail("Scoring opened RGB")
        if str(path) in source["annotation_metadata"]:
            assert (suite/"STARTED.json").exists() and not (suite/"ANALYSIS_SEALED.json").exists()
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    worker.run(suite,digest)
    seal = json.loads((suite/"ANALYSIS_SEALED.json").read_text())
    assert seal["complete"] and seal["cells"] == 8 and seal["image_method_evaluations"] == 24
    results = json.loads((suite/"DATASET_RESULTS.json").read_text())
    for value in results.values():
        assert value["contrasts"]["native_minus_translated"]["delta_pp"] == 100.
        assert value["contrasts"]["native_minus_translated"]["CI95_pp"] == [100.,100.]
        assert value["contrasts"]["native_minus_global"]["delta_pp"] == 0.
    assert not torch.cuda.is_initialized()
    with pytest.raises(ValueError,match="resume/retry"): worker.run(suite,digest)


def test_prepare_running_parent_rejects_before_any_output(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    def blocked():raise ValueError("parent still running")
    monkeypatch.setattr(worker,"parent_terminal",blocked)
    monkeypatch.setattr(worker,"bind_parent",lambda:pytest.fail("Cannot bind running parent"))
    with pytest.raises(ValueError,match="still running"):worker.prepare(tmp_path/"new")
    assert not (tmp_path/"new").exists()


@pytest.mark.parametrize("fault",["permission","report_alias","prepared_alias","runtime_alias","source_drift"])
def test_score_admission_faults_reject_without_GT(tmp_path,monkeypatch,fault):
    suite,digest,bound,source = fixture(tmp_path,monkeypatch)
    if fault == "permission":
        path = suite/"ADMITTED.json"
        value = json.loads(path.read_text())
        value.pop("M02_GT_scoring_allowed")
        value["M02_predictions_allowed"] = True
        path.write_text(json.dumps(value))
    elif fault == "source_drift":
        target = copy.deepcopy(bound)
        target["source"]["specs"][0]["images"] = 999
        monkeypatch.setattr(worker,"bind_parent",lambda:target)
    else:
        target = suite/{"report_alias":"INDEPENDENT_REPORT.md","prepared_alias":"SUITE_PREPARED.json","runtime_alias":"runtime/extra.py"}[fault]
        if target.exists():target.unlink()
        os.link(next(iter(source["annotation_metadata"])),target)
    original = Path.open
    forbidden = worker.prediction.annotation_inodes(source)
    def guard(path,*args,**kwargs):
        state = path.stat()
        assert (state.st_dev,state.st_ino) not in forbidden, "GT content opened before score admission"
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    with pytest.raises(ValueError):worker.run(suite,digest)
    assert not (suite/"STARTED.json").exists() and not (suite/"ANALYSIS_SEALED.json").exists()


@pytest.mark.parametrize("fault",["prediction_hash","prediction_dtype","prediction_shape","prediction_GT_alias","episode_order","cost_hash","summary_cost"])
def test_streaming_faults_fail_closed_no_global(tmp_path,monkeypatch,fault):
    suite,digest,bound,source = fixture(tmp_path,monkeypatch)
    spec = source["specs"][0]
    dataset = spec["dataset"]
    relative = f"{dataset}/predictions/native/000000.npy"
    if fault == "episode_order":relative = f"{dataset}/episodes/000000.native.json"
    elif fault == "cost_hash":relative = f"{dataset}/cost.jsonl"
    elif fault == "summary_cost":relative = f"{dataset}/prediction_result.json"
    path = worker.PARENT/relative
    inventory = bound["seals"][dataset]["artifact_sha256"]
    if fault == "prediction_GT_alias":
        path.unlink()
        os.link(source["rows"][dataset][0]["label"],path)
    elif fault == "prediction_dtype":np.save(path,np.zeros((224,224),np.uint8),allow_pickle=False)
    elif fault == "prediction_shape":np.save(path,np.zeros((1,2),np.uint16),allow_pickle=False)
    elif fault in ("episode_order","summary_cost"):
        value = json.loads(path.read_text())
        if fault == "episode_order":value["ordered_ids"].reverse()
        else:value["raw_model_seconds"] = {}
        path.write_text(json.dumps(value))
    else:path.write_bytes(b"corrupt")
    if fault in ("prediction_dtype","prediction_shape","episode_order","summary_cost"):inventory[relative] = sha256(path)
    with pytest.raises((ValueError,KeyError)):
        worker.score_dataset(suite,bound,spec,dict(bytes=0),{})
    assert not (suite/"ANALYSIS_SEALED.json").exists()


@pytest.mark.parametrize("fault",["execute","final_progress","final_parent","final_publish"])
def test_controller_failures_retain_failed_without_final_seal(tmp_path,monkeypatch,fault):
    suite,digest,bound,_ = fixture(tmp_path,monkeypatch)
    def fail(*args,**kwargs):raise RuntimeError("synthetic scoring failure")
    if fault == "execute":monkeypatch.setattr(worker,"execute",fail)
    elif fault == "final_progress":
        real = worker.write_json
        def write(path,value,**kwargs):
            if value["phase"] == "M02_analysis_payloads_complete_awaiting_seal_and_exit":fail()
            return real(path,value,**kwargs)
        monkeypatch.setattr(worker,"write_json",write)
    elif fault == "final_parent":
        real = worker.bind_parent
        seen = []
        def bind():
            seen.append(1)
            if len(seen)>1:fail()
            return real()
        monkeypatch.setattr(worker,"bind_parent",bind)
    else:monkeypatch.setattr(worker,"publish_seal",fail)
    with pytest.raises(RuntimeError,match="synthetic scoring"):worker.run(suite,digest)
    assert (suite/"FAILED.json").exists() and not (suite/"ANALYSIS_SEALED.json").exists()


@pytest.mark.parametrize("fault",[None,"mask","candidate_identity","area","geometry","support",
    "out_of_frame","false_fallback","wrong_overlap","nonpolicy_legal"])
def test_nonempty_native_candidates_and_translated_geometry_are_validated(tmp_path,monkeypatch,fault):
    suite,_,bound,source = fixture(tmp_path,monkeypatch)
    spec = source["specs"][0]
    dataset,cfg = spec["dataset"],spec["config"]
    inventory = bound["seals"][dataset]["artifact_sha256"]
    _,raw = worker.cost_binding(worker.PARENT,spec,inventory,worker.prediction.annotation_inodes(source))
    mask = np.zeros((16,16),bool)
    mask[:2,:2] = True
    candidates = [dict(batch_idx=i,image_idx=i,class_idx=0,confidence=.5,reliability=.5,area=4,component_idx=0,
        region=dict(shape=[16,16],encoding="packbits-little-hex",data=np.packbits(mask.reshape(-1),bitorder="little").tobytes().hex(),true_pixels=4)) for i in range(2)]
    identity = worker.runtime.image_worker.prior.candidate_identity([dict(c,region=mask) for c in candidates])
    for row in raw[:4]:
        mode = row["mode"]
        views = 0 if mode == "confidence" else 2
        calls,steps = int(views>0),cfg["steps"]
        row.update(steps=steps,optimizer_steps=steps,candidate_count=2,active_candidates=2,candidate_sha256=identity,
            adapt_forward_calls=steps,adapt_forward_inputs=steps*2,diagnostic_forward_calls=calls,
            diagnostic_forward_inputs=views,diagnostic_encoder_calls=calls,diagnostic_encoder_inputs=views,
            encoder_calls=2+steps+calls,encoder_inputs=(2+steps)*2+views)
        geometry = [dict(candidate_index=i,batch_idx=i,mode=mode,readout_and_loss_remain_original=True) for i in range(2)]
        if mode == "translated":
            _,expected = worker.translated_candidate(np.repeat(np.repeat(mask,14,axis=0),14,axis=1))
            for g in geometry:g.update(expected)
        value = dict(**row,GT_opened=False,scientific_identity="M02_spatial_"+mode,losses=[.1]*steps,
            candidates=copy.deepcopy(candidates),preservation_geometry=geometry,ordered_ids=[r["official_id"] for r in source["rows"][dataset][:2]])
        if mode == "translated":
            if fault == "mask":value["candidates"][0]["region"]["data"] = "00"
            elif fault == "candidate_identity":value["candidates"][0]["component_idx"] = 99
            elif fault == "area":value["preservation_geometry"][0]["shifted_area"] = 783
            elif fault == "geometry":value["preservation_geometry"][0]["readout_and_loss_remain_original"] = False
            elif fault == "support":value["candidates"][0]["reliability"] = float("nan")
            elif fault == "out_of_frame":value["preservation_geometry"][0].update(dy=1000000,dx=-1000000)
            elif fault == "false_fallback":value["preservation_geometry"][0].update(dy=0,dx=0,shiftable=False,
                intersection=784,overlap_of_candidate=1.,iou=1.)
            elif fault == "wrong_overlap":value["preservation_geometry"][0].update(intersection=784,overlap_of_candidate=1.,iou=1.)
            elif fault == "nonpolicy_legal":value["preservation_geometry"][0].update(dy=0,dx=28)
        relative = f"{dataset}/episodes/000000.{mode}.json"
        path = worker.PARENT/relative
        path.write_text(json.dumps(value))
        inventory[relative] = sha256(path)
    coverage = {m:dict(candidates=0,EPS_active_candidates=0,sources=0) for m in worker.MODES}
    coverage.update(translated_shiftable=0,translated_unshiftable=0,translated_overlap_sum=0.)
    args = (worker.PARENT,spec,inventory,worker.prediction.annotation_inodes(source),0,raw,coverage)
    if fault:
        with pytest.raises(ValueError):worker.episode_metadata(*args)
    else:
        result = worker.episode_metadata(*args)
        assert coverage["translated_shiftable"] == 2 and coverage["translated_unshiftable"] == 0
        assert all(v["candidates"] == 2 for k,v in coverage.items() if k in worker.MODES)
        assert all(v == [r["official_id"] for r in source["rows"][dataset][:2]] for v in result.values())


@pytest.mark.parametrize("fault",["ActiveState","SubState","MainPID","ExecMainPID","ExecMainCode","ExecMainStatus","InvocationID"])
def test_true_parent_terminal_requires_exact_retained_identity(monkeypatch,fault):
    values = dict(LoadState="loaded",ActiveState="active",SubState="exited",MainPID="0",ExecMainPID="2086569",
        ExecMainCode="1",ExecMainStatus="0",Result="success",InvocationID="775d3e1d206d42e0801641992d4b4ed4")
    values[fault] = "wrong"
    monkeypatch.setattr(worker.subprocess,"run",lambda *a,**k:SimpleNamespace(stdout="\n".join(f"{k}={v}" for k,v in values.items())))
    with pytest.raises(ValueError,match="successfully exited"):worker.parent_terminal()


@pytest.mark.parametrize("relative",["SUITE_PREPARED.json","INPUTS.json","PROTOCOL.json","STARTED.json",
    "ALL_PREDICTIONS_SEALED.json","INDEPENDENT_REPORT.md","coco_object/initial_state.json","coco_object/cost.jsonl"])
def test_parent_metadata_alias_is_rejected_before_GT_read(tmp_path,monkeypatch,relative):
    _,_,_,source = fixture(tmp_path,monkeypatch)
    target = worker.PARENT/relative
    target.unlink()
    os.link(next(iter(source["annotation_metadata"])),target)
    original = Path.open
    def guard(path,*args,**kwargs):
        assert path != target, "Parent metadata annotation alias opened"
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    with pytest.raises(ValueError):worker.bind_parent()


@pytest.mark.parametrize("kind",["hardlink","symlink"])
@pytest.mark.parametrize("name",["queue_state.json.tmp","queue_state.json"])
def test_progress_alias_rejects_without_modifying_synthetic_GT(tmp_path,monkeypatch,kind,name):
    suite,digest,_,source = fixture(tmp_path,monkeypatch)
    victim = Path(source["rows"][worker.DATASETS[0]][0]["label"])
    alias = suite/name
    if kind == "hardlink":os.link(victim,alias)
    else:alias.symlink_to(victim)
    before,identity = victim.read_bytes(),victim.stat()
    with pytest.raises((ValueError,FileExistsError)):worker.run(suite,digest)
    after = victim.stat()
    assert victim.read_bytes() == before
    assert (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_nlink) == (
        identity.st_dev,identity.st_ino,identity.st_size,identity.st_mtime_ns,identity.st_ctime_ns,identity.st_nlink)
    assert (suite/"FAILED.json").exists() and not (suite/"ANALYSIS_SEALED.json").exists()


@pytest.mark.parametrize("aliased",[False,True])
def test_native_source_alias_stat_exclusion_precedes_all_native_hash_reads(tmp_path,monkeypatch,aliased):
    paths = ("tools/evolution_fifty/risk_worker.py","tools/evolution_fifty/unit_risk.py","trace_ovss/a.py")
    expected = {}
    for relative in paths:
        path = tmp_path/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("CPU_source=True\n")
        expected[relative] = sha256(path)
    (tmp_path/"tests").mkdir()
    monkeypatch.setattr(worker,"ROOT",tmp_path)
    if not aliased:
        worker.native_binding(expected,set())
        return
    victim = tmp_path/"synthetic_GT"
    victim.write_bytes(b"native hash must never read this")
    target = tmp_path/paths[-1]
    target.unlink()
    os.link(victim,target)
    state = victim.stat()
    monkeypatch.setattr(worker.protected,"source_content",lambda *a,**k:pytest.fail("Must enumerate and exclude all source GT aliases before any native content"))
    with pytest.raises(ValueError):worker.native_binding(expected,{(state.st_dev,state.st_ino)})
