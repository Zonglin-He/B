import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from test_legacy_score_receipts import fixture as episode_fixture
from tools.evolution_fifty import legacy_analysis as analysis
from tools.evolution_fifty import legacy_score_payload as payload
from tools.evolution_fifty.intake import sha256,write_new


def fixture(tmp_path,monkeypatch,*,images=2,empty=False):
    monkeypatch.setattr("time.time",lambda:1700000000.)
    parent,suite = tmp_path/"parent",tmp_path/"score"
    parent.mkdir()
    suite.mkdir()
    _,spec,ids,_,components = episode_fixture(monkeypatch)
    spec.update(images=images)
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
        rgb.write_bytes(b"RGB must never open during scoring")
        st = gt.stat()
        details = dict(device=st.st_dev,inode=st.st_ino,size_bytes=st.st_size,mtime_ns=st.st_mtime_ns,contents_read=False)
        rows.append(dict(official_id=f"s{i}",image=str(rgb),label=str(gt),
            label_metadata=[st.st_size,st.st_mtime_ns],label_metadata_details=details))
    archive = tmp_path/"synthetic.archive"
    archive.write_bytes(b"Bytes-only synthetic components already passed separately")
    costs = []
    for episode,start in enumerate(range(0,images,2)):
        current = rows[start:start+2]
        order = payload.prediction.prediction.runtime.prior.arm_order(payload.MODES,episode,"alternating")
        for position,mode in enumerate(order):
            saved,_,_,cost,_ = episode_fixture(monkeypatch,mode,empty=empty or len(current) == 1)
            extra = dict(episode=episode,order_position=position,warmup=episode == 0)
            if len(current) == 1:
                extra.update(images=1,patches=1,eval_forward_inputs=2,encoder_inputs=2)
            saved.update(extra,ordered_ids=[r["official_id"] for r in current])
            cost.update(extra)
            save(f"pc60/episodes/{episode:06d}.{mode}.json",saved)
            costs.append(cost)
            for i,row in enumerate(current,start):
                labels = payload.decode_annotation(Path(row["label"]).read_bytes(),(224,224),3,False)
                save(f"pc60/predictions/{mode}/{i:06d}.npy",(labels if mode.startswith("C.") else (labels+1)%3).astype(np.uint16))
    save("pc60/cost.jsonl","".join(json.dumps(r)+"\n" for r in costs).encode())
    legal = dict(parameter_tensors=1,parameter_elements=1,
        trainable_names=["backbone.visual.ln_pre.weight"],visual_LN_names=["visual.ln_pre.weight"])
    scope = dict(legacy_archive_sha256=components["archive_sha256"],legacy_source_members=components["member_sha256"],
        episode_objective_sha256=payload.prediction.runtime.OBJECTIVE_SHA)
    initial = dict(legal_visual_LN=legal,frozen_parameter_sha256="a"*64,
        reset=dict(trainable_sha256="b"*64,optimizer_sha256="c"*64),**scope)
    save("pc60/initial_state.json",initial)
    counters = (*payload.prediction.runtime.native.COUNTERS,"optimizer_steps")
    summary = dict(dataset="pc60",images=images,modes=list(payload.MODES),GT_opened=False,**scope,
        legal_visual_LN=legal,frozen_parameter_sha256=initial["frozen_parameter_sha256"],
        actual_work={m:{k:sum(r[k] for r in costs if r["arm"] == m) for k in counters} for m in payload.MODES},
        raw_model_seconds={m:sum(r["seconds"] for r in costs if r["arm"] == m) for m in payload.MODES},
        warmup_adjusted_throughput=payload.prediction.prediction.runtime.prior.timing_summary(costs,payload.MODES))
    save("pc60/prediction_result.json",summary)
    source = dict(rows=dict(pc60=rows),specs=[spec],legacy_archive=dict(path=str(archive)),
        annotation_metadata={r["label"]:r["label_metadata_details"] for r in rows})
    bound = dict(parent=str(parent),source=source,seals=dict(pc60=dict(artifact_sha256=inventory)))
    bound["payload_metadata"] = payload.preflight_payloads(parent,[spec],bound["seals"],source)
    return suite,bound,spec,components,save


@pytest.mark.parametrize("empty,images",[(False,2),(True,1),(False,5)])
def test_streams_native_PIL_GT_once_per_source_four_serial_arms_and_empty_tail(tmp_path,monkeypatch,empty,images):
    suite,bound,spec,components,_ = fixture(tmp_path,monkeypatch,empty=empty,images=images)
    rows = bound["source"]["rows"]["pc60"]
    reads = {r["label"]:0 for r in rows}
    blocked = {r["image"] for r in rows}|{bound["source"]["legacy_archive"]["path"]}
    original = Path.open
    def guard(path,*args,**kwargs):
        assert str(path) not in blocked
        if str(path) in reads:reads[str(path)] += 1
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    budget,labels,progress = dict(bytes=0),{},[]
    counts,ordered,cost,coverage,path,digest = payload.score_dataset(suite,bound,spec,budget,labels,progress.append,components)
    assert counts.shape == ((images+1)//2,4,2,3) and counts.dtype == np.int64
    assert ordered == [[r["official_id"] for r in rows[i:i+2]] for i in range(0,images,2)] and set(reads.values()) == {1}
    assert digest == sha256(path) and budget["bytes"] == path.stat().st_size
    assert coverage["C.image"]["candidates"] == (0 if empty else 3*(images//2))
    assert coverage["legacy.image"]["diagnostic_views"] == (0 if empty else 7*(images//2))
    assert progress[-1]["completed_sources"] == images and progress[-1]["GT_opened"] is True
    result = analysis.paired_summary(counts,"pc60",replicates=20)
    assert result["contrasts"]["single_C_vs_legacy_image"]["delta_pp"] == 100.
    assert not list(suite.glob("*SEALED*"))
    with pytest.raises(FileExistsError):
        payload.score_dataset(suite,bound,spec,dict(bytes=0),{},lambda _:None,components)


@pytest.mark.parametrize("fault",["order","counter","time","warmup","summary","state","archive","scope","steps"])
def test_hash_consistent_cost_drift_rejects_before_GT_or_arrays(tmp_path,monkeypatch,fault):
    suite,bound,spec,components,save = fixture(tmp_path,monkeypatch)
    parent = Path(bound["parent"])
    name = "pc60/cost.jsonl"
    costs = [json.loads(r) for r in (parent/name).read_bytes().splitlines()]
    if fault == "order":costs[0]["order_position"] += 1
    elif fault == "counter":costs[0]["encoder_inputs"] += 1
    elif fault == "time":costs[0]["seconds"] = float("nan")
    elif fault == "warmup":costs[0]["warmup"] = False
    elif fault in ("summary","state","archive"):
        other = "pc60/initial_state.json" if fault == "state" else "pc60/prediction_result.json"
        value = json.loads((parent/other).read_bytes())
        if fault == "summary":value["raw_model_seconds"][payload.MODES[0]] += 1
        elif fault == "state":value["reset"]["optimizer_sha256"] = "g"*64
        else:value["legacy_archive_sha256"] = "0"*64
        (parent/other).unlink()
        save(other,value)
    elif fault == "scope":costs[0]["old_engine_loaded"] = True
    elif fault == "steps":costs[0]["steps"] = 0
    (parent/name).unlink()
    save(name,"".join(json.dumps(r)+"\n" for r in costs).encode())
    bound["payload_metadata"] = payload.preflight_payloads(parent,[spec],bound["seals"],bound["source"])
    monkeypatch.setattr(payload,"annotation_bytes",lambda *_:pytest.fail("Cost fault read GT"))
    monkeypatch.setattr(payload,"hard_array",lambda *_:pytest.fail("Cost fault read array"))
    with pytest.raises(ValueError):payload.score_dataset(suite,bound,spec,dict(bytes=0),{},lambda _:None,components)
    assert not list(suite.iterdir())


@pytest.mark.parametrize("fault",["ids","formula","role","objective","different_C"])
def test_all_four_episode_receipts_validate_before_GT(tmp_path,monkeypatch,fault):
    suite,bound,spec,components,save = fixture(tmp_path,monkeypatch)
    name = "pc60/episodes/000000.legacy.episode.json" if fault != "different_C" else "pc60/episodes/000000.C.episode.json"
    path = Path(bound["parent"])/name
    value = json.loads(path.read_bytes())
    if fault == "ids":value["ordered_ids"].reverse()
    elif fault == "formula":value["candidates"][0]["reliability"] += .01
    elif fault == "role":value["candidates"][0]["role"] = "stuff"
    elif fault == "objective":value["candidates"][0]["objective_coefficient"] += .01
    else:
        for key in ("quality","reliability","counterfactual_support"):value["candidates"][0][key] += .01
    path.unlink()
    save(name,value)
    bound["payload_metadata"] = payload.preflight_payloads(Path(bound["parent"]),[spec],bound["seals"],bound["source"])
    monkeypatch.setattr(payload,"annotation_bytes",lambda *_:pytest.fail("Receipt fault read GT"))
    with pytest.raises(ValueError):payload.score_dataset(suite,bound,spec,dict(bytes=0),{},lambda _:None,components)
    assert not list(suite.glob("*SEALED*"))


@pytest.mark.parametrize("fault",["missing","GT_alias","RGB_alias","archive_alias","oversize","symlink","directory"])
def test_complete_graph_stat_faults_precede_all_data_opens(tmp_path,monkeypatch,fault):
    suite,bound,spec,components,_ = fixture(tmp_path,monkeypatch)
    path = Path(bound["parent"])/"pc60/predictions/legacy.episode/000001.npy"
    rows = bound["source"]["rows"]["pc60"]
    if fault == "oversize":
        with path.open("wb") as stream:stream.truncate(2*1024**2+1)
    else:
        path.unlink()
        if fault == "GT_alias":path.hardlink_to(rows[0]["label"])
        elif fault == "RGB_alias":path.hardlink_to(rows[0]["image"])
        elif fault == "archive_alias":path.hardlink_to(bound["source"]["legacy_archive"]["path"])
        elif fault == "directory":path.mkdir()
        elif fault == "symlink":path.symlink_to(rows[0]["image"])
    original = Path.open
    protected = {r[k] for r in rows for k in ("image","label")}|{bound["source"]["legacy_archive"]["path"]}
    def guard(p,*args,**kwargs):
        assert str(p) not in protected and "/predictions/" not in str(p)
        return original(p,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    with pytest.raises((ValueError,FileNotFoundError)):
        payload.score_dataset(suite,bound,spec,dict(bytes=0),{},lambda _:None,components)
    assert not list(suite.iterdir())


@pytest.mark.parametrize("fault",["digest","output_cap","disk_reserve","progress"])
def test_failure_has_no_seal_retry_or_silent_drop(tmp_path,monkeypatch,fault):
    suite,bound,spec,components,_ = fixture(tmp_path,monkeypatch)
    if fault == "digest":bound["seals"]["pc60"]["artifact_sha256"]["pc60/predictions/C.image/000000.npy"] = "0"*64
    elif fault == "output_cap":monkeypatch.setattr(payload,"OUTPUT_CAP",1)
    elif fault == "disk_reserve":monkeypatch.setattr(payload.shutil,"disk_usage",lambda _:type("D",(),dict(free=0))())
    def progress(_):
        if fault == "progress":raise RuntimeError("synthetic progress failure")
    with pytest.raises((ValueError,RuntimeError)):
        payload.score_dataset(suite,bound,spec,dict(bytes=0),{},progress,components)
    assert not list(suite.glob("*SEALED*"))


@pytest.mark.parametrize("fault",["same_size","append","path_replace","symlink","hardlink"])
def test_progress_boundary_output_tampering_rejects_without_success_or_seal(tmp_path,monkeypatch,fault):
    suite,bound,spec,components,_ = fixture(tmp_path,monkeypatch,images=5)
    output = suite/"pc60.episode_counts.jsonl"
    def progress(value):
        if value["completed_sources"] != 2:return
        if fault == "same_size":
            raw = output.read_bytes()
            offset = raw.index(b'"I_U_by_mode"')
            i = next(i for i in range(offset,len(raw)) if raw[i:i+1].isdigit())
            with output.open("r+b") as stream:stream.write(raw[:i]+(b"7" if raw[i:i+1] != b"7" else b"8")+raw[i+1:])
        elif fault == "append":
            with output.open("ab") as stream:stream.write(b"x")
        elif fault == "hardlink":(suite/"alias").hardlink_to(output)
        else:
            copy = suite/"replacement"
            copy.write_bytes(output.read_bytes())
            output.unlink()
            if fault == "symlink":output.symlink_to(copy)
            else:copy.rename(output)
    with pytest.raises(ValueError,match="counts"):
        payload.score_dataset(suite,bound,spec,dict(bytes=0),{},progress,components)
    assert not list(suite.glob("*SEALED*"))


def test_final_descriptor_rehash_rejects_same_extent_corruption_at_fsync(tmp_path,monkeypatch):
    suite,bound,spec,components,_ = fixture(tmp_path,monkeypatch)
    output = suite/"pc60.episode_counts.jsonl"
    real_fsync = payload.os.fsync
    def fsync(fd):
        real_fsync(fd)
        with output.open("r+b") as stream:stream.write(b"[")
    monkeypatch.setattr(payload.os,"fsync",fsync)
    with pytest.raises(ValueError,match="counts"):
        payload.score_dataset(suite,bound,spec,dict(bytes=0),{},lambda _:None,components)
    assert not list(suite.glob("*SEALED*"))


@pytest.mark.parametrize("fault",["size","mtime","replace","symlink","registry","row"])
def test_all_GT_stat_pins_preflight_before_any_GT_or_payload_contents(tmp_path,monkeypatch,fault):
    suite,bound,spec,components,_ = fixture(tmp_path,monkeypatch)
    row = bound["source"]["rows"]["pc60"][-1]
    path = Path(row["label"])
    if fault == "size":
        with path.open("ab") as stream:stream.write(b"x")
    elif fault == "mtime":
        state = path.stat()
        payload.os.utime(path,ns=(state.st_atime_ns,state.st_mtime_ns+1000000000))
    elif fault in ("replace","symlink"):
        other = tmp_path/"replacedGT.png"
        other.write_bytes(path.read_bytes())
        path.unlink()
        if fault == "replace":other.rename(path)
        else:path.symlink_to(other)
    elif fault == "registry":bound["source"]["annotation_metadata"][str(path)] = dict(row["label_metadata_details"],inode=-1)
    else:row["label_metadata"] = [0,0]
    original = Path.open
    forbidden = {r[k] for r in bound["source"]["rows"]["pc60"] for k in ("label","image")}
    def guard(p,*args,**kwargs):
        assert str(p) not in forbidden and "/predictions/" not in str(p) and "/episodes/" not in str(p)
        return original(p,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    with pytest.raises(ValueError,match="annotation"):
        payload.score_dataset(suite,bound,spec,dict(bytes=0),{},lambda _:None,components)
    assert not list(suite.iterdir())
