import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools.evolution_fifty import curve_score_payload as core
from tools.evolution_fifty.intake import sha256, write_new


def fixture(tmp_path, images=5):
    parent, suite = tmp_path/"pred", tmp_path/"score"
    parent.mkdir(); suite.mkdir()
    cfg = dict(batch_size=2, resize=[224,224], steps=1, trace_max_regions=2, trace_view_batch_size=2)
    spec = dict(id="naclip.ViT-L-14.voc20.fog.s1", dataset="voc20", arms=list(core.ARMS), images=images,
                config=cfg, input_key="fog.s1", vocabulary=dict(classes=list(range(20)), reduce_zero_label=True))
    rows, inventory = [], {}
    def save(relative, value):
        path = parent/relative
        path.parent.mkdir(exist_ok=True, parents=True)
        if isinstance(value,np.ndarray): np.save(path,value,allow_pickle=False)
        elif isinstance(value,bytes): path.write_bytes(value)
        else: write_new(path,value)
        inventory[relative] = sha256(path)
    for i in range(images):
        gt, rgb = tmp_path/f"gt{i}.png", tmp_path/f"rgb{i}.png"
        im = Image.fromarray(np.array([[0,1,2],[2,1,255]],dtype=np.uint8)).convert("P")
        im.putpalette([c for i in range(256) for c in (i,255-i,i//2)])
        im.save(gt); rgb.write_bytes(b"RGB bytes may not open in scorer")
        s = gt.stat()
        details = dict(device=s.st_dev,inode=s.st_ino,size_bytes=s.st_size,mtime_ns=s.st_mtime_ns,contents_read=False)
        rows.append(dict(official_id=f"s{i}", image=str(rgb), label=str(gt),
                         label_metadata=[s.st_size,s.st_mtime_ns],label_metadata_details=details))
    costs = []
    for episode,start in enumerate(range(0,images,2)):
        n = min(2,images-start)
        order = core.protected.prediction.runtime.prior.arm_order(core.ARMS,episode,"alternating")
        for position,arm in enumerate(order):
            row = dict(arm=arm,images=n,patches=n,steps=0,seconds=0.1,candidate_count=0,active_candidates=0,
                       candidate_sha256=None if arm == "no_adapt" else "a"*64,
                       episode=episode,order_position=position,warmup=episode == 0,
                       **{k:0 for k in core.protected.prediction.runtime.COUNTERS})
            calls = 1 if arm == "no_adapt" else 2
            row.update(eval_forward_calls=calls,eval_forward_inputs=calls*n,encoder_calls=calls,encoder_inputs=calls*n)
            costs.append(row)
            save(f"conditions/{spec['id']}/episodes/{episode:06d}.{arm}.json",
                 dict(**row,ordered_ids=[r["official_id"] for r in rows[start:start+n]],GT_opened=False,losses=[],candidates=[]))
            for i in range(start,start+n):
                labels = core.metric.decode_annotation(Path(rows[i]["label"]).read_bytes(),20,True)
                prediction = np.where(labels == 255,0,labels)
                if arm == "image_confidence": prediction = (prediction+1)%20
                save(f"conditions/{spec['id']}/predictions/{arm}/{i:06d}.npy",prediction.astype(np.uint16))
    prefix = f"conditions/{spec['id']}/"
    save(prefix+"cost.jsonl",("".join(json.dumps(r)+"\n" for r in costs)).encode())
    legal = dict(parameter_tensors=1,parameter_elements=1,trainable_names=["backbone.visual.ln_pre.weight"],visual_LN_names=["visual.ln_pre.weight"])
    initial = dict(legal_visual_LN=legal,frozen_parameter_sha256="b"*64,reset=dict(trainable_sha256="c"*64,optimizer_sha256="d"*64))
    save(prefix+"initial_state.json",initial)
    save(prefix+"prediction_result.json",dict(condition=spec["id"],images=images,arms=list(core.ARMS),GT_opened=False,
         legal_visual_LN=legal,frozen_parameter_sha256=initial["frozen_parameter_sha256"],
         actual_work={a:{k:sum(r[k] for r in costs if r["arm"] == a) for k in core.protected.prediction.runtime.COUNTERS} for a in core.ARMS},
         raw_model_seconds={a:sum(r["seconds"] for r in costs if r["arm"] == a) for a in core.ARMS},
         warmup_adjusted_throughput=core.protected.prediction.runtime.prior.timing_summary(costs,core.ARMS)))
    source = dict(rows={spec["input_key"]:rows},native=dict(annotation_metadata={r["label"]:r["label_metadata_details"] for r in rows}))
    bound = dict(parent=str(parent),source=source,specs=[spec],seals={spec["id"]:dict(artifact_sha256=inventory)})
    bound["payload_metadata"] = core.preflight_payloads(parent,[spec],bound["seals"],source)
    return suite,bound,spec,save


@pytest.mark.parametrize("images",[1,2,5])
def test_serial_G_geometry_and_reused_counts(tmp_path,monkeypatch,images):
    suite,bound,spec,_ = fixture(tmp_path,images)
    rows = bound["source"]["rows"][spec["input_key"]]
    rgb, reads = {r["image"] for r in rows}, {r["label"]:0 for r in rows}
    original = Path.open
    def guarded(path,*args,**kwargs):
        assert str(path) not in rgb
        if str(path) in reads: reads[str(path)] += 1
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    budget,labels,progress = dict(bytes=0),{},[]
    counts,ordered,cost,path,digest = core.score_condition(suite,bound,spec,budget,labels,progress.append)
    assert counts.shape == ((images+1)//2,3,2,20)
    assert set(reads.values()) == {1} and len(labels) == images
    assert digest == sha256(path) and budget["bytes"] == path.stat().st_size
    assert core.metric.miou(counts.sum(0))[2] == 100.0
    reused = core.load_reused_counts(path,digest,core.file_identity(path,set(),core.REUSED_CAP),ordered)
    np.testing.assert_array_equal(reused,counts)
    assert progress[-1]["completed_sources"] == images and not list(suite.glob("*SEALED*"))
    with pytest.raises(FileExistsError): core.score_condition(suite,bound,spec,dict(bytes=0),{},lambda _:None)


@pytest.mark.parametrize("fault",["missing","GTalias","RGBalias","symlink","oversize","state","cost","lateGT"])
def test_complete_graph_fault_before_GT(tmp_path,monkeypatch,fault):
    suite,bound,spec,_ = fixture(tmp_path)
    relative = f"conditions/{spec['id']}/predictions/{core.ARMS[-1]}/000004.npy"
    path = Path(bound["parent"])/relative
    rows = bound["source"]["rows"][spec["input_key"]]
    if fault in ("GTalias","RGBalias","symlink"):
        path.unlink()
        victim = rows[0]["label" if fault == "GTalias" else "image"]
        if fault == "symlink": path.symlink_to(victim)
        else: os.link(victim,path)
    elif fault == "missing":path.unlink()
    elif fault == "oversize":
        with path.open("r+b") as stream:stream.truncate(2*1024**2+1)
    elif fault == "lateGT":Path(rows[-1]["label"]).write_bytes(b"changed")
    else:
        other = Path(bound["parent"])/f"conditions/{spec['id']}/{'initial_state.json' if fault == 'state' else 'cost.jsonl'}"
        other.write_bytes(b"changed")
    original = Path.open
    def guard(p,*args,**kwargs):
        assert str(p) not in {r["label"] for r in rows}
        return original(p,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    with pytest.raises((ValueError,FileNotFoundError)):core.score_condition(suite,bound,spec,dict(bytes=0),{},lambda _:None)
    assert not list(suite.iterdir())


@pytest.mark.parametrize("fault",["float","bool","negative","too_large","I_gt_U","episode","ids","extra","hash","line"])
def test_streamed_S5_rejects_rebound_malformed_counts(tmp_path,fault):
    path = tmp_path/"s5.episode_counts.jsonl"
    counts = np.zeros((3,2,20),dtype=np.int64).tolist()
    row = dict(episode=0,ordered_ids=["one"],labels=[dict(source_id="one",label="/GT",sha256="a"*64)],I_U_by_arm=counts)
    if fault in ("float","bool","negative","too_large","I_gt_U"):
        counts[0][0][0] = {"float":0.0,"bool":False,"negative":-1,"too_large":224**2+1,"I_gt_U":1}[fault]
    if fault == "episode":row["episode"] = True
    if fault == "ids":row["ordered_ids"] = ["other"]
    data = (json.dumps(row)+"\n").encode()
    if fault == "extra":data += data
    if fault == "line":data = b" "*(core.LINE_CAP+1)+data
    path.write_bytes(data)
    digest = "0"*64 if fault == "hash" else sha256(path)
    with pytest.raises(ValueError):core.load_reused_counts(path,digest,core.file_identity(path,set(),core.REUSED_CAP),[["one"]])


@pytest.mark.parametrize("fault",["same_size","same_inode","symlink"])
def test_counts_progress_mutation_not_reported_as_success(tmp_path,fault):
    path = tmp_path/"counts.jsonl"
    def mutate():
        if fault == "same_size":path.write_bytes(b"x"*path.stat().st_size)
        elif fault == "same_inode":
            with path.open("r+b") as stream:stream.write(b"x")
        else:
            other = tmp_path/"other"; other.write_bytes(b"foreign")
            path.unlink();path.symlink_to(other)
    with pytest.raises(ValueError):
        with core.output_lines(path,dict(bytes=0)) as append:append(dict(x=1),mutate)


def test_stream_S5_does_not_use_unbounded_read(tmp_path,monkeypatch):
    path = tmp_path/"s5.jsonl"
    row = dict(episode=0,ordered_ids=["one"],labels=[dict(source_id="one")],I_U_by_arm=np.zeros((3,2,20),dtype=np.int64).tolist())
    path.write_text(json.dumps(row)+"\n")
    original = Path.open
    class Reader:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):return self
        def __exit__(self,*a):self.stream.close()
        def fileno(self):return self.stream.fileno()
        def readline(self,n):assert 0 < n <= core.LINE_CAP+1; return self.stream.readline(n)
        def read(self,n):assert n == 1; return self.stream.read(n)
    monkeypatch.setattr(Path,"open",lambda p,*a,**kw:Reader(original(p,*a,**kw)))
    result = core.load_reused_counts(path,hashlib.sha256((json.dumps(row)+"\n").encode()).hexdigest(),core.file_identity(path,set(),core.REUSED_CAP),[["one"]])
    assert result.sum() == 0


@pytest.mark.parametrize("fault",["bool_episode","candidate","pair","nonfinite","losses","ids"])
def test_rehashed_cost_and_episode_faults_before_GT(tmp_path,monkeypatch,fault):
    suite,bound,spec,save = fixture(tmp_path,2)
    prefix = f"conditions/{spec['id']}/"
    parent = Path(bound["parent"])
    relative = prefix+"cost.jsonl"
    costs = [json.loads(line) for line in (parent/relative).read_bytes().splitlines()]
    if fault in ("losses","ids"):
        target = prefix+f"episodes/000000.{core.ARMS[0]}.json"
        row = json.loads((parent/target).read_bytes())
        if fault == "losses":row["losses"] = [0.]
        else:row["ordered_ids"] = ["wrong","wrong"]
        (parent/target).unlink()
        save(target,row)
    else:
        if fault == "bool_episode":costs[0]["episode"] = False
        elif fault == "nonfinite":costs[0]["seconds"] = float("nan")
        elif fault == "candidate":
            for row in costs:
                if row["arm"] != "no_adapt":row["candidate_sha256"] = None
        else:
            next(row for row in costs if row["arm"] == "image_C")["candidate_sha256"] = "b"*64
        # Test cost faults with a self-consistent content digest and fresh stat.
        save(relative,("".join(json.dumps(r)+"\n" for r in costs)).encode())
    bound["payload_metadata"] = core.preflight_payloads(parent,[spec],bound["seals"],bound["source"])
    monkeypatch.setattr(core,"annotation_bytes",lambda _:pytest.fail("Malformed metadata reached GT"))
    with pytest.raises(ValueError):core.score_condition(suite,bound,spec,dict(bytes=0),{},lambda _:None)
    assert not list(suite.glob("*SEALED*"))


@pytest.mark.parametrize("fault",["fsync","close"])
def test_owned_counts_IO_failure_does_not_return_digest(tmp_path,monkeypatch,fault):
    path = tmp_path/"counts.jsonl"
    if fault == "fsync":monkeypatch.setattr(core.os,"fsync",lambda _:(_ for _ in ()).throw(OSError("synthetic fsync")))
    else:
        original = Path.open
        class Stream:
            def __init__(self,fd):self.fd=fd
            def __getattr__(self,k):return getattr(self.fd,k)
            def __enter__(self):return self
            def __exit__(self,*args):self.fd.close();raise OSError("synthetic close")
        monkeypatch.setattr(Path,"open",lambda p,*a,**kw:Stream(original(p,*a,**kw)))
    with pytest.raises(OSError):
        with core.output_lines(path,dict(bytes=0)) as append:append(dict(x=1),lambda:None)
    assert path.exists() and not hasattr(append,"digest")


def test_S5_identical_bytes_replacement_after_initial_pin_check_rejected(tmp_path,monkeypatch):
    path,replacement = tmp_path/"s5.jsonl",tmp_path/"new.jsonl"
    row = dict(episode=0,ordered_ids=["one"],labels=[dict(source_id="one")],I_U_by_arm=np.zeros((3,2,20),dtype=np.int64).tolist())
    raw = (json.dumps(row)+"\n").encode()
    path.write_bytes(raw);replacement.write_bytes(raw)
    expected = core.file_identity(path,set(),core.REUSED_CAP)
    original = core.file_identity
    def swap(p,*args):
        pin = original(p,*args)
        replacement.replace(path)
        return pin
    monkeypatch.setattr(core,"file_identity",swap)
    with pytest.raises(ValueError,match="between pin validation"):
        core.load_reused_counts(path,hashlib.sha256(raw).hexdigest(),expected,[["one"]])


@pytest.mark.parametrize("name",["initial_state.json","prediction_result.json","cost.jsonl"])
def test_reopened_cost_metadata_replacement_rejected_before_GT(tmp_path,monkeypatch,name):
    suite,bound,spec,_ = fixture(tmp_path)
    path = Path(bound["parent"])/f"conditions/{spec['id']}/{name}"
    replacement = tmp_path/"same.json"
    replacement.write_bytes(path.read_bytes())
    original = core.protected.condition_cost
    def replace_then_validate(*args):
        replacement.replace(path)
        return original(*args)
    monkeypatch.setattr(core.protected,"condition_cost",replace_then_validate)
    monkeypatch.setattr(core,"annotation_bytes",lambda _:pytest.fail("Prepared cost identity drift reached GT"))
    with pytest.raises(ValueError):core.score_condition(suite,bound,spec,dict(bytes=0),{},lambda _:None)
    assert not list(suite.iterdir())


@pytest.mark.parametrize("key,value",[("episode",False),("order_position",0.0),("steps",False),("images",2.0),("warmup",1)])
def test_episode_cost_link_exact_JSON_types_before_GT(tmp_path,monkeypatch,key,value):
    suite,bound,spec,save = fixture(tmp_path,2)
    relative = f"conditions/{spec['id']}/episodes/000000.no_adapt.json"
    path = Path(bound["parent"])/relative
    row = json.loads(path.read_bytes())
    row[key] = value
    path.unlink();save(relative,row)
    bound["payload_metadata"] = core.preflight_payloads(Path(bound["parent"]),[spec],bound["seals"],bound["source"])
    monkeypatch.setattr(core,"annotation_bytes",lambda _:pytest.fail("Episode JSON type drift reached GT"))
    with pytest.raises(ValueError,match="episode/cost/source identity"):
        core.score_condition(suite,bound,spec,dict(bytes=0),{},lambda _:None)
    assert not list(suite.glob("*SEALED*"))
