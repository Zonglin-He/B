import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from test_value_score_receipts import fixture as receipt_fixture,rebind
from tools.evolution_fifty import value_analysis as analysis
from tools.evolution_fifty import value_score_payload as payload
from tools.evolution_fifty.intake import sha256,write_new


def fixture(tmp_path,*,images=2):
    parent,suite = tmp_path/"parent",tmp_path/"score"
    parent.mkdir()
    suite.mkdir()
    points = payload.prediction.runtime.points("pc60")
    _,spec,_,_ = receipt_fixture("pc60",points[0])
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
            saved,_,_,cost = receipt_fixture("pc60",point,empty=len(current) == 1)
            extra = dict(episode=episode,order_position=position,warmup=episode == 0)
            if len(current) == 1:extra.update(images=1,patches=1,eval_forward_inputs=2,encoder_inputs=2)
            saved.update(extra,ordered_ids=[r["official_id"] for r in current])
            cost.update(extra)
            save(f"pc60/episodes/{episode:06d}.{name}.json",saved)
            costs.append(cost)
            for i,row in enumerate(current,start):
                labels = payload.decode_annotation(Path(row["label"]).read_bytes(),(224,224),3,False)
                prediction = labels if point == points[1] and len(current) > 1 else (labels+1)%3
                save(f"pc60/predictions/{name}/{i:06d}.npy",prediction.astype(np.uint16))
    save("pc60/cost.jsonl","".join(json.dumps(r)+"\n" for r in costs).encode())
    legal = dict(parameter_tensors=1,parameter_elements=1,
        trainable_names=["backbone.visual.ln_pre.weight"],visual_LN_names=["visual.ln_pre.weight"])
    initial = dict(legal_visual_LN=legal,frozen_parameter_sha256="a"*64,reset=dict(trainable_sha256="b"*64,optimizer_sha256="c"*64))
    save("pc60/initial_state.json",initial)
    counters = (*payload.prediction.runtime.native.COUNTERS,"optimizer_steps")
    summary = dict(dataset="pc60",images=images,points=list(points),GT_opened=False,
        time_matching_done=False,accuracy_or_score_selected_points=False,legal_visual_LN=legal,
        frozen_parameter_sha256=initial["frozen_parameter_sha256"],
        actual_work={name:{k:sum(r[k] for r in costs if r["arm"] == name) for k in counters} for name in names},
        raw_model_seconds={name:sum(r["seconds"] for r in costs if r["arm"] == name) for name in names},
        warmup_adjusted_throughput=payload.prediction.prediction.runtime.prior.timing_summary(costs,names))
    save("pc60/prediction_result.json",summary)
    source = dict(rows=dict(pc60=rows),specs=[spec],annotation_metadata={r["label"]:r["label_metadata_details"] for r in rows})
    bound = dict(parent=str(parent),source=source,seals=dict(pc60=dict(artifact_sha256=inventory)))
    bound["payload_metadata"] = payload.preflight_payloads(parent,[spec],bound["seals"],source)
    return suite,bound,spec,save


@pytest.mark.parametrize("images",[1,2,5])
def test_all_five_points_serial_native_GT_once_including_empty_tail(tmp_path,monkeypatch,images):
    suite,bound,spec,_ = fixture(tmp_path,images=images)
    rows = bound["source"]["rows"]["pc60"]
    reads = {r["label"]:0 for r in rows}
    rgb = {r["image"] for r in rows}
    original = Path.open
    def guard(path,*args,**kwargs):
        assert str(path) not in rgb
        if str(path) in reads:reads[str(path)] += 1
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    budget,labels,progress = dict(bytes=0),{},[]
    counts,ordered,cost,coverage,path,digest = payload.score_dataset(suite,bound,spec,budget,labels,progress.append)
    assert counts.shape == ((images+1)//2,5,2,3) and counts.dtype == np.int64
    assert ordered == [[r["official_id"] for r in rows[i:i+2]] for i in range(0,images,2)]
    assert set(reads.values()) == {1} and len(labels) == images
    assert digest == sha256(path) and budget["bytes"] == path.stat().st_size
    names = [p["id"] for p in payload.prediction.runtime.points("pc60")]
    assert coverage[names[0]]["candidates"] == 3*(images//2)
    assert coverage[names[0]]["diagnostic_views"] == 3*(images//2)
    assert coverage[names[-1]]["diagnostic_views"] == 0
    summary = analysis.summarize("pc60",counts,cost["raw_model_seconds"],replicates=20)
    if images == 2:assert summary["native_C_minus_same_steps_Confidence"]["difference_pp"] == 100.
    assert len(summary["points"]) == 5 and not list(suite.glob("*SEALED*"))
    assert progress[-1]["completed_sources"] == images
    with pytest.raises(FileExistsError):payload.score_dataset(suite,bound,spec,dict(bytes=0),{},lambda _:None)


@pytest.mark.parametrize("fault",["order","budget","scope","aux","counter","time","summary","state"])
def test_repinned_cost_faults_fail_before_any_GT(tmp_path,monkeypatch,fault):
    suite,bound,spec,save = fixture(tmp_path)
    parent = Path(bound["parent"])
    name = "pc60/cost.jsonl"
    costs = [json.loads(r) for r in (parent/name).read_bytes().splitlines()]
    if fault == "order":costs[0]["order_position"] += 1
    elif fault == "budget":costs[0]["requested_steps"] += 1
    elif fault == "scope":costs[0]["actual_step_observation_in_model_time"] = False
    elif fault == "aux":costs[0]["auxiliary_checkpoint_readouts"] = 1
    elif fault == "counter":costs[0]["encoder_inputs"] += 1
    elif fault == "time":costs[0]["seconds"] = float("nan")
    else:
        other = "pc60/prediction_result.json" if fault == "summary" else "pc60/initial_state.json"
        value = json.loads((parent/other).read_bytes())
        if fault == "summary":value["time_matching_done"] = True
        else:value["reset"]["optimizer_sha256"] = "g"*64
        (parent/other).unlink()
        save(other,value)
    (parent/name).unlink()
    save(name,"".join(json.dumps(r)+"\n" for r in costs).encode())
    bound["payload_metadata"] = payload.preflight_payloads(parent,[spec],bound["seals"],bound["source"])
    monkeypatch.setattr(payload,"annotation_bytes",lambda *_:pytest.fail("Cost fault read GT"))
    with pytest.raises(ValueError):payload.score_dataset(suite,bound,spec,dict(bytes=0),{},lambda _:None)
    assert not list(suite.iterdir())


@pytest.mark.parametrize("fault",["missing","GT_alias","RGB_alias","oversize","lateGT","gridfloat"])
def test_complete_stat_graph_and_registry_fail_before_any_contents(tmp_path,monkeypatch,fault):
    suite,bound,spec,_ = fixture(tmp_path)
    last = payload.prediction.runtime.points("pc60")[-1]["id"]
    path = Path(bound["parent"])/f"pc60/predictions/{last}/000001.npy"
    rows = bound["source"]["rows"]["pc60"]
    if fault == "lateGT":
        with Path(rows[-1]["label"]).open("ab") as stream:stream.write(b"drift")
    elif fault == "gridfloat":spec["points"][0]["steps"] = float(spec["points"][0]["steps"])
    elif fault == "oversize":
        with path.open("wb") as stream:stream.truncate(2*1024**2+1)
    else:
        path.unlink()
        if fault != "missing":path.hardlink_to(rows[0]["label" if fault == "GT_alias" else "image"])
    original = Path.open
    forbidden = {r[k] for r in rows for k in ("label","image")}
    def guard(p,*args,**kwargs):
        assert str(p) not in forbidden and "/predictions/" not in str(p) and "/episodes/" not in str(p)
        return original(p,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    with pytest.raises((ValueError,FileNotFoundError)):payload.score_dataset(suite,bound,spec,dict(bytes=0),{},lambda _:None)
    assert not list(suite.iterdir())


def test_valid_but_budget_dependent_C_diagnostic_rejects_before_GT(tmp_path,monkeypatch):
    suite,bound,spec,save = fixture(tmp_path)
    name = f"pc60/episodes/000000.{spec['points'][0]['id']}.json"
    path = Path(bound["parent"])/name
    value = json.loads(path.read_bytes())
    for field in ("counterfactual_support","quality","reliability"):value["candidates"][0][field] += .01
    rebind(value)
    path.unlink()
    save(name,value)
    bound["payload_metadata"] = payload.preflight_payloads(Path(bound["parent"]),[spec],bound["seals"],bound["source"])
    monkeypatch.setattr(payload,"annotation_bytes",lambda *_:pytest.fail("Changed initial diagnostic read GT"))
    with pytest.raises(ValueError,match="initial diagnostic"):payload.score_dataset(suite,bound,spec,dict(bytes=0),{},lambda _:None)


@pytest.mark.parametrize("fault",["progress","same_size","cap","reserve","fsync"])
def test_no_seal_or_success_on_output_faults(tmp_path,monkeypatch,fault):
    suite,bound,spec,_ = fixture(tmp_path,images=5)
    if fault == "cap":monkeypatch.setattr(payload,"OUTPUT_CAP",1)
    elif fault == "reserve":monkeypatch.setattr(payload.shutil,"disk_usage",lambda _:type("D",(),dict(free=0))())
    elif fault == "fsync":monkeypatch.setattr(payload.os,"fsync",lambda *_:(_ for _ in ()).throw(RuntimeError("fsync failed")))
    def progress(row):
        if fault == "progress":raise RuntimeError("advisory failed")
        if fault == "same_size" and row["completed_sources"] == 2:
            with (suite/"pc60.episode_counts.jsonl").open("r+b") as stream:stream.write(b"[")
    with pytest.raises((ValueError,RuntimeError)):payload.score_dataset(suite,bound,spec,dict(bytes=0),{},progress)
    assert not list(suite.glob("*SEALED*"))
