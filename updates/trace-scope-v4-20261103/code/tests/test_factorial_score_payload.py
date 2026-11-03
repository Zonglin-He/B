import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from test_factorial_score_receipts import fixture as episode_fixture
from tools.evolution_fifty import factorial_analysis as analysis
from tools.evolution_fifty import factorial_score_payload as payload
from tools.evolution_fifty.intake import sha256,write_new


def fixture(tmp_path):
    parent,suite = tmp_path/"parent",tmp_path/"score"
    parent.mkdir()
    suite.mkdir()
    _,spec,ids,_ = episode_fixture()
    spec.update(dataset="coco_object",images=2)
    spec["vocabulary"]["reduce_zero_label"] = False
    rows,inventory = [],{}
    def save(name,value):
        path = parent/name
        path.parent.mkdir(parents=True,exist_ok=True)
        if isinstance(value,np.ndarray): np.save(path,value,allow_pickle=False)
        elif isinstance(value,bytes):path.write_bytes(value)
        else:write_new(path,value)
        inventory[name] = sha256(path)
    for i in range(2):
        gt,rgb = tmp_path/f"GT{i}.png",tmp_path/f"RGB{i}.rgb"
        image = Image.fromarray(np.array([[0,1,0],[1,0,1]],dtype=np.uint8)).convert("P")
        image.putpalette([255,0,0,0,255,0]+[0]*762)
        image.save(gt)
        rgb.write_bytes(b"RGB must never be opened by scoring")
        st = gt.stat()
        details = dict(device=st.st_dev,inode=st.st_ino,size_bytes=st.st_size,mtime_ns=st.st_mtime_ns,contents_read=False)
        rows.append(dict(official_id=ids[i],image=str(rgb),label=str(gt),
            label_metadata=[st.st_size,st.st_mtime_ns],label_metadata_details=details))
    costs = []
    order = payload.prediction.prediction.runtime.prior.arm_order(payload.MODES,0,"alternating")
    for position,mode in enumerate(order):
        saved,_,_,cost = episode_fixture(mode)
        extra = dict(episode=0,order_position=position,warmup=True)
        saved.update(extra)
        cost.update(extra)
        save(f"coco_object/episodes/000000.{mode}.json",saved)
        costs.append(cost)
        for i,row in enumerate(rows):
            labels = payload.decode_annotation(Path(row["label"]).read_bytes(),(224,224),2,False)
            values = labels if mode.endswith("Dc_Oc") else 1-labels
            save(f"coco_object/predictions/{mode}/{i:06d}.npy",values.astype(np.uint16))
    save("coco_object/cost.jsonl","".join(json.dumps(r)+"\n" for r in costs).encode())
    legal = dict(parameter_tensors=1,parameter_elements=1,
        trainable_names=["backbone.visual.ln_pre.weight"],visual_LN_names=["visual.ln_pre.weight"])
    initial = dict(legal_visual_LN=legal,frozen_parameter_sha256="a"*64,
        reset=dict(trainable_sha256="b"*64,optimizer_sha256="c"*64))
    save("coco_object/initial_state.json",initial)
    counters = (*payload.prediction.runtime.native.COUNTERS,"optimizer_steps")
    summary = dict(dataset="coco_object",images=2,modes=list(payload.MODES),
        studies=list(payload.prediction.STUDIES),factorial_arms=list(payload.prediction.runtime.ARMS),GT_opened=False,
        legal_visual_LN=legal,frozen_parameter_sha256=initial["frozen_parameter_sha256"],
        actual_work={m:{k:sum(r[k] for r in costs if r["arm"] == m) for k in counters} for m in payload.MODES},
        raw_model_seconds={m:sum(r["seconds"] for r in costs if r["arm"] == m) for m in payload.MODES},
        warmup_adjusted_throughput=payload.prediction.prediction.runtime.prior.timing_summary(costs,payload.MODES))
    save("coco_object/prediction_result.json",summary)
    source = dict(rows=dict(coco_object=rows),specs=[spec],annotation_metadata={r["label"]:r["label_metadata_details"] for r in rows})
    bound = dict(parent=str(parent),source=source,seals=dict(coco_object=dict(artifact_sha256=inventory)))
    bound["payload_metadata"] = payload.preflight_payloads(parent,[spec],bound["seals"],source)
    return suite,bound,spec,save


def test_real_synthetic_stream_reads_each_annotation_once_eight_serial_arms_native_PIL(tmp_path,monkeypatch):
    suite,bound,spec,_ = fixture(tmp_path)
    rows = bound["source"]["rows"]["coco_object"]
    reads = {r["label"]:0 for r in rows}
    rgb = {r["image"] for r in rows}
    original = Path.open
    def guard(path,*args,**kwargs):
        assert str(path) not in rgb
        if str(path) in reads: reads[str(path)] += 1
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    budget,labels,progress = dict(bytes=0),{},[]
    counts,ordered,cost,coverage,path,digest = payload.score_dataset(suite,bound,spec,budget,labels,progress.append)
    assert counts.shape == (1,2,4,2,2) and counts.dtype == np.int64
    assert ordered == [[r["official_id"] for r in rows]]
    assert set(reads.values()) == {1} and len(labels) == 2
    assert digest == sha256(path) and budget["bytes"] == path.stat().st_size
    assert progress[-1]["completed_sources"] == 2 and progress[-1]["GT_opened"] is True
    assert coverage["natural.Dk_Ok"]["original_candidates"] == 3
    assert coverage["common_active.Dk_Ok"]["retained_candidates"] == 2
    result,_ = analysis.paired_summary(counts,"coco_object",replicates=20)
    assert result["studies"]["natural"]["contrasts"]["native_C_minus_Confidence"]["delta_pp"] == 100.
    assert not list(suite.glob("*SEALED*"))
    with pytest.raises(FileExistsError):
        payload.score_dataset(suite,bound,spec,dict(bytes=0),{},lambda _:None)


@pytest.mark.parametrize("fault",["order","counter","time","warmup","summary","state","study","scope","steps"])
def test_hash_consistent_cost_faults_reject_before_any_GT_or_arrays(tmp_path,monkeypatch,fault):
    suite,bound,spec,save = fixture(tmp_path)
    parent = Path(bound["parent"])
    relative = "coco_object/cost.jsonl"
    rows = [json.loads(r) for r in (parent/relative).read_bytes().splitlines()]
    if fault == "order": rows[0]["order_position"] = 1
    elif fault == "counter": rows[0]["encoder_inputs"] += 1
    elif fault == "time": rows[0]["seconds"] = float("nan")
    elif fault == "warmup": rows[0]["warmup"] = False
    elif fault == "summary":
        name = "coco_object/prediction_result.json"
        value = json.loads((parent/name).read_bytes())
        value["raw_model_seconds"][payload.MODES[0]] += 1
        (parent/name).unlink()
        save(name,value)
    elif fault == "state":
        name = "coco_object/initial_state.json"
        value = json.loads((parent/name).read_bytes())
        value["reset"]["optimizer_sha256"] = "g"*64
        (parent/name).unlink()
        save(name,value)
    elif fault == "study": rows[0]["study"] = "other"
    elif fault == "scope": rows[0]["production_Confidence_timing"] = True
    elif fault == "steps": rows[0]["steps"] = 0
    (parent/relative).unlink()
    save(relative,"".join(json.dumps(r)+"\n" for r in rows).encode())
    monkeypatch.setattr(payload,"annotation_bytes",lambda *_:pytest.fail("Fault must reject before GT"))
    monkeypatch.setattr(payload,"hard_array",lambda *_:pytest.fail("Fault must reject before arrays"))
    with pytest.raises(ValueError): payload.score_dataset(suite,bound,spec,dict(bytes=0),{},lambda _:None)
    assert not list(suite.iterdir())


@pytest.mark.parametrize("fault",["episode_identity","mask","assignment","different_C"])
def test_episode_faults_precede_GT_for_every_arm(tmp_path,monkeypatch,fault):
    suite,bound,spec,save = fixture(tmp_path)
    name = "coco_object/episodes/000000.common_active.Dk_Oc.json"
    path = Path(bound["parent"])/name
    value = json.loads(path.read_bytes())
    if fault == "episode_identity": value["ordered_ids"].reverse()
    elif fault == "mask": value["candidates"][0]["region"]["data"] = "0"*64
    elif fault == "assignment": value["assignment"]["candidate_count"] += 1
    elif fault == "different_C": value["candidates"][0]["original_C"] += .1
    path.unlink()
    save(name,value)
    monkeypatch.setattr(payload,"annotation_bytes",lambda *_:pytest.fail("Episode fault read GT"))
    with pytest.raises(ValueError): payload.score_dataset(suite,bound,spec,dict(bytes=0),{},lambda _:None)
    assert not list(suite.glob("*SEALED*"))


@pytest.mark.parametrize("fault",["payload_digest","GT_alias","output_cap","disk_reserve","progress_failure"])
def test_streaming_faults_propagate_without_seal_or_retry(tmp_path,monkeypatch,fault):
    suite,bound,spec,save = fixture(tmp_path)
    inv = bound["seals"]["coco_object"]["artifact_sha256"]
    name = "coco_object/predictions/natural.Dk_Ok/000000.npy"
    path = Path(bound["parent"])/name
    if fault == "payload_digest":inv[name] = "0"*64
    elif fault == "GT_alias":
        path.unlink()
        path.hardlink_to(bound["source"]["rows"]["coco_object"][0]["label"])
        inv[name] = sha256(path)
    elif fault == "output_cap": monkeypatch.setattr(payload,"OUTPUT_CAP",1)
    elif fault == "disk_reserve": monkeypatch.setattr(payload.shutil,"disk_usage",lambda _:type("D",(),dict(free=0))())
    def progress(_):
        if fault == "progress_failure": raise RuntimeError("synthetic advisory failure")
    with pytest.raises((ValueError,RuntimeError)):payload.score_dataset(suite,bound,spec,dict(bytes=0),{},progress)
    assert not list(suite.glob("*SEALED*"))


def test_partial_single_source_episode_with_empty_candidate_pool_is_retained(tmp_path):
    suite,bound,spec,save = fixture(tmp_path)
    parent = Path(bound["parent"])
    spec["images"] = 1
    bound["source"]["rows"]["coco_object"] = bound["source"]["rows"]["coco_object"][:1]
    costs = []
    for position,mode in enumerate(payload.prediction.prediction.runtime.prior.arm_order(payload.MODES,0,"alternating")):
        saved,_,_,cost = episode_fixture(mode,empty=True)
        updates = dict(images=1,patches=1,eval_forward_inputs=2,encoder_inputs=2,
            episode=0,order_position=position,warmup=True)
        cost.update(updates)
        saved.update(updates,ordered_ids=["synthetic0"])
        name = f"coco_object/episodes/000000.{mode}.json"
        (parent/name).unlink()
        save(name,saved)
        costs.append(cost)
    name = "coco_object/cost.jsonl"
    (parent/name).unlink()
    save(name,"".join(json.dumps(r)+"\n" for r in costs).encode())
    name = "coco_object/prediction_result.json"
    summary = json.loads((parent/name).read_bytes())
    counters = (*payload.prediction.runtime.native.COUNTERS,"optimizer_steps")
    summary.update(images=1,actual_work={m:{k:sum(r[k] for r in costs if r["arm"] == m) for k in counters} for m in payload.MODES},
        raw_model_seconds={m:sum(r["seconds"] for r in costs if r["arm"] == m) for m in payload.MODES},
        warmup_adjusted_throughput=payload.prediction.prediction.runtime.prior.timing_summary(costs,payload.MODES))
    (parent/name).unlink()
    save(name,summary)
    inventory = bound["seals"]["coco_object"]["artifact_sha256"]
    for mode in payload.MODES:
        unused = f"coco_object/predictions/{mode}/000001.npy"
        (parent/unused).unlink()
        del inventory[unused]
    bound["payload_metadata"] = payload.preflight_payloads(parent,[spec],bound["seals"],bound["source"])
    labels = {}
    counts,ordered,_,coverage,_,_ = payload.score_dataset(suite,bound,spec,dict(bytes=0),labels,lambda _:None)
    assert counts.shape == (1,2,4,2,2) and ordered == [["synthetic0"]] and len(labels) == 1
    assert all(v["sources"] == 1 and v["original_candidates"] == v["EPS_active_candidates"] == 0 for v in coverage.values())


@pytest.mark.parametrize("fault",["missing","GT_alias","RGB_alias","source_alias","oversize","symlink","directory"])
def test_complete_stat_preflight_rejects_before_any_GT_prediction_or_source_open(tmp_path,monkeypatch,fault):
    suite,bound,spec,_ = fixture(tmp_path)
    parent = Path(bound["parent"])
    path = parent/"coco_object/predictions/common_active.Dk_Oc/000001.npy"
    victim = None
    if fault == "oversize":
        with path.open("wb") as stream: stream.truncate(2*1024**2+1)
    else:
        path.unlink()
        if fault == "GT_alias": victim = Path(bound["source"]["rows"]["coco_object"][0]["label"])
        elif fault == "RGB_alias": victim = Path(bound["source"]["rows"]["coco_object"][0]["image"])
        elif fault == "source_alias":
            (parent/"runtime").mkdir()
            victim = parent/"runtime"/"fixture.py"
            victim.write_bytes(b"Synthetic source must never be opened as a prediction\n")
        elif fault == "directory":path.mkdir()
        elif fault == "symlink":
            target = tmp_path/"target.npy"
            target.write_bytes(b"Synthetic external payload")
            path.symlink_to(target)
        if victim is not None:path.hardlink_to(victim)
    protected_paths = {r[k] for r in bound["source"]["rows"]["coco_object"] for k in ("image","label")}
    original = Path.open
    def guard(p,*args,**kwargs):
        assert str(p) not in protected_paths and "/predictions/" not in str(p)
        if victim is not None:assert p != victim
        return original(p,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    monkeypatch.setattr(payload,"annotation_bytes",lambda *_:pytest.fail("Preflight fault opened GT"))
    with pytest.raises((ValueError,FileNotFoundError)):
        payload.score_dataset(suite,bound,spec,dict(bytes=0),{},lambda _:None)
    assert not list(suite.iterdir())


def test_preflight_pins_prediction_identity_for_streaming_without_reopening_RGB(tmp_path,monkeypatch):
    suite,bound,spec,_ = fixture(tmp_path)
    original = payload.annotation_bytes
    path = Path(bound["parent"])/"coco_object/predictions/natural.Dk_Ok/000000.npy"
    victim = Path(bound["source"]["rows"]["coco_object"][0]["image"])
    def swap(row):
        result = original(row)
        path.unlink()
        path.hardlink_to(victim)
        return result
    monkeypatch.setattr(payload,"annotation_bytes",swap)
    raw_open = Path.open
    def guard(p,*args,**kwargs):
        assert p != path and p != victim
        return raw_open(p,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    with pytest.raises(ValueError,match="protected input|identity differs"):
        payload.score_dataset(suite,bound,spec,dict(bytes=0),{},lambda _:None)
    assert not list(suite.glob("*SEALED*"))
