"""V01 five-point native-PIL streaming core, no admission or seal.

Real inputs require the completed exact prediction parent, independently
admitted CPU_GT controller and externally capped execution. Tests are synthetic.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat

import numpy as np

from . import factorial_score_payload as bounded
from . import generalization_score as metric
from . import generalization_score_worker as protected
from . import legacy_score_payload as native_inputs
from . import value_prediction as prediction
from . import value_score_receipts as receipts
from .bootstrap import POPULATIONS,require
from .risk_score_inputs import decode_annotation
from .risk_score_payload import annotation_bytes,hard_array
from .risk_score_worker import canonical_child

RESERVE,OUTPUT_CAP = 16*1024**3,256*1024**2
ROOT = Path(__file__).resolve().parents[2]


def preflight_payloads(parent,specs,seals,source):
    parent = Path(parent)
    native_inputs.preflight_annotations(source)
    forbidden = bounded.data_exclusions(source)
    roots = [(ROOT,True)]
    if (parent/"runtime").exists():roots.append((parent/"runtime",False))
    for root,snapshot in roots:
        members = protected.source_members(root,forbidden,snapshot_source=snapshot)
        forbidden.update((s[0],s[1]) for s in members.values())
    identities,seen = {},set()
    for spec in specs:
        dataset,n,batch = spec["dataset"],spec["images"],spec["config"]["batch_size"]
        require(dataset in prediction.runtime.NATIVE_STEPS and type(n) is int and 0 < n <= POPULATIONS[dataset]
                and type(batch) is int and 0 < batch <= 512 and type(spec["points"]) is list,
                "V01 payload population/grid resource bound differs")
        for point in spec["points"]:prediction.runtime.validate_point(dataset,point)
        require(spec["points"] == list(prediction.runtime.points(dataset)), "V01 serialized grid differs")
        names = [p["id"] for p in spec["points"]]
        expected = {f"{dataset}/predictions/{name}/{i:06d}.npy":2*1024**2 for name in names for i in range(n)}
        expected.update({f"{dataset}/episodes/{e:06d}.{name}.json":8*1024**2
                         for name in names for e in range((n+batch-1)//batch)})
        expected.update({f"{dataset}/{name}":limit for name,limit in
                         (("cost.jsonl",16*1024**2),("initial_state.json",2*1024**2),("prediction_result.json",2*1024**2))})
        inventory = seals[dataset]["artifact_sha256"]
        require(set(inventory) == set(expected), "V01 complete payload membership differs")
        for relative,limit in expected.items():
            digest,path = inventory[relative],parent/relative
            require(type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                    "V01 payload digest declaration differs")
            current = path.lstat()
            identity = (current.st_dev,current.st_ino)
            require(stat.S_ISREG(current.st_mode) and current.st_nlink == 1 and path.resolve(strict=True) == path
                    and identity not in forbidden and identity not in seen and 0 < current.st_size <= limit,
                    "V01 payload preflight path/inode/extent differs")
            seen.add(identity)
            identities[relative] = [*identity,current.st_size,current.st_mtime_ns]
    return identities


def cost_binding(parent,spec,inventory,forbidden,identities):
    dataset,cfg = spec["dataset"],spec["config"]
    points = prediction.runtime.points(dataset)
    names,by_name = tuple(p["id"] for p in points),{p["id"]:p for p in points}
    require(type(cfg["steps"]) is int and cfg["steps"] == prediction.runtime.NATIVE_STEPS[dataset],
            "V01 native step configuration differs")
    def saved(name):
        relative = dataset+"/"+name
        return protected.metadata(canonical_child(parent,relative),inventory[relative],forbidden,limit=2*1024**2,
                                  expected_identity=tuple(identities[relative]))[0]
    summary,initial = saved("prediction_result.json"),saved("initial_state.json")
    relative = dataset+"/cost.jsonl"
    raw,_ = bounded.pinned_content(parent,relative,inventory[relative],forbidden,16*1024**2,identities)
    lines = raw.splitlines()
    require(all(0 < len(line) <= 64*1024 for line in lines), "V01 raw cost line bound exceeded")
    rows = [json.loads(line) for line in lines]
    episodes = (spec["images"]+cfg["batch_size"]-1)//cfg["batch_size"]
    require(len(rows) == episodes*5, "V01 full cost population differs")
    for episode in range(episodes):
        order = prediction.prediction.runtime.prior.arm_order(names,episode,"alternating")
        images = min(cfg["batch_size"],spec["images"]-episode*cfg["batch_size"])
        for position,row in enumerate(rows[episode*5:(episode+1)*5]):
            point = by_name[order[position]]
            require(type(row["episode"]) is int and row["episode"] == episode and row["arm"] == row["point_id"] == point["id"]
                    and type(row["order_position"]) is int and row["order_position"] == position and row["warmup"] is (episode == 0)
                    and type(row["images"]) is int and row["images"] == images == row["patches"]
                    and row["dataset"] == dataset and row["weighting_arm"] == point["arm"]
                    and type(row["native_steps"]) is int and row["native_steps"] == cfg["steps"]
                    and type(row["requested_steps"]) is int and row["requested_steps"] == point["steps"]
                    and row["actual_step_observation_in_model_time"] is True and row["time_scope"] == receipts.TIME_SCOPE
                    and type(row["auxiliary_checkpoint_readouts"]) is int and row["auxiliary_checkpoint_readouts"] == 0
                    and row["GT_opened"] is False, "V01 raw cost order/point/scope differs")
            counters = (*prediction.runtime.native.COUNTERS,"steps","optimizer_steps","candidate_count","active_candidates")
            require(all(type(row[k]) is int and row[k] >= 0 for k in counters)
                    and row["active_candidates"] <= row["candidate_count"] <= images*cfg["trace_max_regions"]
                    and type(row["seconds"]) in (float,int) and math.isfinite(row["seconds"]) and row["seconds"] >= 0,
                    "V01 invalid raw time/counter population")
            steps = point["steps"] if row["active_candidates"] else 0
            views = row["candidate_count"] if point["arm"] == "image_C" else 0
            calls = (views+cfg["trace_view_batch_size"]-1)//cfg["trace_view_batch_size"]
            require(row["steps"] == row["optimizer_steps"] == steps
                    and row["eval_forward_calls"] == 2 and row["eval_forward_inputs"] == 2*images
                    and row["adapt_forward_calls"] == steps and row["adapt_forward_inputs"] == steps*images
                    and row["diagnostic_forward_calls"] == row["diagnostic_encoder_calls"] == calls
                    and row["diagnostic_forward_inputs"] == row["diagnostic_encoder_inputs"] == views
                    and row["encoder_calls"] == 2+steps+calls and row["encoder_inputs"] == (2+steps)*images+views,
                    "V01 measured actual optimizer/forward schedule differs")
    counters = (*prediction.runtime.native.COUNTERS,"optimizer_steps")
    require(summary["dataset"] == dataset and summary["images"] == spec["images"] and summary["points"] == list(points)
            and summary["GT_opened"] is False and summary["time_matching_done"] is False and summary["accuracy_or_score_selected_points"] is False
            and summary["actual_work"] == {name:{k:sum(r[k] for r in rows if r["arm"] == name) for k in counters} for name in names}
            and summary["raw_model_seconds"] == {name:sum(r["seconds"] for r in rows if r["arm"] == name) for name in names}
            and summary["warmup_adjusted_throughput"] == prediction.prediction.runtime.prior.timing_summary(rows,names),
            "V01 measured cost summary/grid differs")
    protected.validate_visual_LN_names(initial["legal_visual_LN"])
    require(summary["legal_visual_LN"] == initial["legal_visual_LN"]
            and summary["frozen_parameter_sha256"] == initial["frozen_parameter_sha256"]
            and set(initial["reset"]) == {"trainable_sha256","optimizer_sha256"}, "V01 initial state differs")
    for digest in (*initial["reset"].values(),initial["frozen_parameter_sha256"]):
        require(type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                "V01 invalid initial state digest")
    return summary,rows


def score_dataset(suite,bound,spec,budget,label_hashes,progress):
    parent,dataset = Path(bound["parent"]),spec["dataset"]
    rows,cfg = bound["source"]["rows"][dataset],spec["config"]
    require(type(budget["bytes"]) is int and 0 <= budget["bytes"] <= OUTPUT_CAP
            and dataset in prediction.runtime.NATIVE_STEPS and 0 < len(rows) == spec["images"] <= POPULATIONS[dataset],
            "V01 score population/budget differs")
    classes,batch = len(spec["vocabulary"]["classes"]),cfg["batch_size"]
    require(0 < classes <= 81 and type(batch) is int and 0 < batch <= 512, "V01 class/batch resource bound differs")
    names = tuple(p["id"] for p in prediction.runtime.points(dataset))
    forbidden = bounded.data_exclusions(bound["source"])
    inventory = bound["seals"][dataset]["artifact_sha256"]
    identities = preflight_payloads(parent,[spec],bound["seals"],bound["source"])
    require(identities == {n:v for n,v in bound["payload_metadata"].items() if n.startswith(dataset+"/")},
            "V01 payload identities drifted after score preparation")
    cost,raw_rows = cost_binding(parent,spec,inventory,forbidden,identities)
    counts = np.zeros(((len(rows)+batch-1)//batch,5,2,classes),dtype=np.int64)
    coverage = {n:dict(candidates=0,EPS_active_candidates=0,active_sources=0,diagnostic_views=0,sources=0) for n in names}
    ordered,path,hasher = [],suite/(dataset+".episode_counts.jsonl"),hashlib.sha256()
    with path.open("xb+") as stream:
        created,written = os.fstat(stream.fileno()),0
        def identity(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink,s.st_mode)
        def observed():
            current,on_path = os.fstat(stream.fileno()),path.lstat()
            require(identity(current) == identity(on_path) and stat.S_ISREG(current.st_mode) and current.st_nlink == 1
                    and path.resolve(strict=True) == path and (created.st_dev,created.st_ino) == (current.st_dev,current.st_ino)
                    and current.st_size == written, "V01 counts output identity/extent differs")
            return identity(current)
        previous = observed()
        for episode,start in enumerate(range(0,len(rows),batch)):
            current,checked = rows[start:start+batch],{}
            ids = [r["official_id"] for r in current]
            by_mode = {r["arm"]:r for r in raw_rows[episode*5:(episode+1)*5]}
            require(set(by_mode) == set(names), "V01 incomplete episode cost grid")
            for name in names:
                relative = f"{dataset}/episodes/{episode:06d}.{name}.json"
                saved,_ = protected.metadata(canonical_child(parent,relative),inventory[relative],forbidden,limit=8*1024**2,
                                            expected_identity=tuple(identities[relative]))
                checked[name] = receipts.validate_episode(saved,spec,ids,by_mode[name])
            receipts.validate_shared(dataset,checked)
            for name,value in checked.items():
                for key in ("candidates","EPS_active_candidates","active_sources","diagnostic_views"):coverage[name][key] += value[key]
                coverage[name]["sources"] += len(current)
            ordered.append(ids)
            gt_receipts = []
            for index,row in enumerate(current,start):
                content,digest = annotation_bytes(row)
                labels = decode_annotation(content,(224,224),classes,spec["vocabulary"]["reduce_zero_label"])
                require(row["label"] not in label_hashes or label_hashes[row["label"]] == digest, "Repeated V01 GT content drift")
                label_hashes[row["label"]] = digest
                gt_receipts.append(dict(source_id=row["official_id"],label=row["label"],sha256=digest))
                for a,name in enumerate(names):
                    relative = f"{dataset}/predictions/{name}/{index:06d}.npy"
                    raw,_ = bounded.pinned_content(parent,relative,inventory[relative],forbidden,2*1024**2,identities)
                    counts[episode,a] += metric.counts(hard_array(raw,classes,(224,224)),labels,classes)
            line = (json.dumps(dict(episode=episode,ordered_ids=ids,labels=gt_receipts,I_U_by_point=counts[episode].tolist()),allow_nan=False)+"\n").encode()
            require(shutil.disk_usage(suite).free >= RESERVE+len(line) and budget["bytes"]+len(line) <= OUTPUT_CAP,
                    "V01 score output budget exceeded")
            require(observed() == previous, "V01 completed counts changed before write")
            require(stream.write(line) == len(line), "Short V01 counts write")
            stream.flush()
            written += len(line)
            budget["bytes"] += len(line)
            hasher.update(line)
            previous = observed()
            progress(dict(phase="CPU_V01_score",dataset=dataset,completed_sources=min(start+batch,len(rows)),total_sources=len(rows),GT_opened=True,pid=os.getpid()))
            require(observed() == previous, "V01 completed counts changed during progress")
        stream.flush()
        os.fsync(stream.fileno())
        require(observed() == previous, "V01 completed counts changed before final read")
        stream.seek(0)
        persisted,total = hashlib.sha256(),0
        while chunk := stream.read(min(64*1024,written-total+1)):
            total += len(chunk)
            require(total <= written <= OUTPUT_CAP, "V01 persisted counts exceed output bound")
            persisted.update(chunk)
        require(total == written and persisted.hexdigest() == hasher.hexdigest() and observed() == previous,
                "V01 persisted output content/identity differs")
    require(identity(path.lstat()) == previous and path.resolve(strict=True) == path, "V01 counts changed during close")
    return counts,ordered,cost,coverage,path,hasher.hexdigest()
