"""Bounded native-PIL M03 scoring core, without admission or sealing.

The future controller must verify the complete successful parent and obtain
new exact M03 CPU_GT permission before using this on real payloads.
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
from . import legacy_prediction as prediction
from . import legacy_score_receipts as receipts
from .bootstrap import POPULATIONS,require
from .risk_score_inputs import decode_annotation
from .risk_score_payload import annotation_bytes,hard_array
from .risk_score_worker import canonical_child

MODES = receipts.MODES
RESERVE,OUTPUT_CAP = 16*1024**3,256*1024**2
ROOT = Path(__file__).resolve().parents[2]


def preflight_annotations(source):
    """Validate every pinned GT stat, without opening annotation contents."""
    registry = source["annotation_metadata"]
    require(type(registry) is dict and registry, "M03 annotation registry is missing")
    for name,expected in registry.items():
        path = Path(name)
        state = path.lstat()
        observed = dict(device=state.st_dev,inode=state.st_ino,size_bytes=state.st_size,
            mtime_ns=state.st_mtime_ns,contents_read=False)
        require(path.is_absolute() and path.resolve(strict=True) == path and stat.S_ISREG(state.st_mode)
                and 0 < state.st_size <= 16*1024**2 and expected == observed,
                "M03 complete annotation stat identity differs")
    for rows in source["rows"].values():
        for row in rows:
            require(row["label"] in registry and row["label_metadata_details"] == registry[row["label"]]
                    and row["label_metadata"] == [registry[row["label"]]["size_bytes"],registry[row["label"]]["mtime_ns"]],
                    "M03 row annotation registry binding differs")


def preflight_payloads(parent,specs,seals,source):
    """Pin the complete four-condition graph with stat only, before any GT."""
    parent = Path(parent)
    preflight_annotations(source)
    forbidden = bounded.data_exclusions(source)
    roots = [(ROOT,True)]
    if (parent/"runtime").exists():
        roots.append((parent/"runtime",False))
    for root,snapshot in roots:
        members = protected.source_members(root,forbidden,snapshot_source=snapshot)
        forbidden.update((s[0],s[1]) for s in members.values())
    # The archive is a source input, never a prediction or episode receipt.
    archive = Path(source["legacy_archive"]["path"])
    state = archive.lstat()
    require(archive.is_absolute() and archive.resolve(strict=True) == archive and stat.S_ISREG(state.st_mode)
            and (state.st_dev,state.st_ino) not in forbidden, "M03 archive source role differs")
    forbidden.add((state.st_dev,state.st_ino))
    identities,seen = {},set()
    for spec in specs:
        dataset,images,batch = spec["dataset"],spec["images"],spec["config"]["batch_size"]
        require(dataset in ("coco_object","pc60") and type(images) is int and 0 < images <= POPULATIONS[dataset]
                and type(batch) is int and 0 < batch <= 512, "M03 payload preflight population differs")
        expected = {f"{dataset}/predictions/{m}/{i:06d}.npy":2*1024**2 for m in MODES for i in range(images)}
        expected.update({f"{dataset}/episodes/{e:06d}.{m}.json":8*1024**2
                         for m in MODES for e in range((images+batch-1)//batch)})
        expected.update({f"{dataset}/{name}":limit for name,limit in
                         (("cost.jsonl",16*1024**2),("initial_state.json",2*1024**2),("prediction_result.json",2*1024**2))})
        inventory = seals[dataset]["artifact_sha256"]
        require(set(inventory) == set(expected), "M03 preflight complete payload membership differs")
        for relative,limit in expected.items():
            digest,path = inventory[relative],parent/relative
            require(type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                    "M03 declared payload digest differs")
            current = path.lstat()
            identity = (current.st_dev,current.st_ino)
            require(stat.S_ISREG(current.st_mode) and current.st_nlink == 1 and path.resolve(strict=True) == path
                    and identity not in forbidden and identity not in seen and 0 < current.st_size <= limit,
                    "M03 payload preflight path/inode/extent differs")
            seen.add(identity)
            identities[relative] = [*identity,current.st_size,current.st_mtime_ns]
    return identities


def cost_binding(parent,spec,inventory,forbidden,identities,components):
    """Bind raw order, initial state and measured work before source scoring."""
    dataset,cfg = spec["dataset"],spec["config"]
    def saved(name):
        relative = dataset+"/"+name
        return protected.metadata(canonical_child(parent,relative),inventory[relative],forbidden,limit=2*1024**2,
                                  expected_identity=tuple(identities[relative]))[0]
    summary,initial = saved("prediction_result.json"),saved("initial_state.json")
    relative = dataset+"/cost.jsonl"
    raw,_ = bounded.pinned_content(parent,relative,inventory[relative],forbidden,16*1024**2,identities)
    lines = raw.splitlines()
    require(all(0 < len(line) <= 64*1024 for line in lines), "M03 cost line bound exceeded")
    rows = [json.loads(line) for line in lines]
    episodes = (spec["images"]+cfg["batch_size"]-1)//cfg["batch_size"]
    require(len(rows) == episodes*len(MODES), "M03 raw cost population differs")
    archive_fields = dict(legacy_archive_sha256=components["archive_sha256"],
        legacy_source_members=components["member_sha256"],episode_objective_sha256=prediction.runtime.OBJECTIVE_SHA)
    for value in (summary,initial):
        require(all(value[k] == v for k,v in archive_fields.items()), "M03 cost initial/archive binding differs")
    for episode in range(episodes):
        order = prediction.prediction.runtime.prior.arm_order(MODES,episode,"alternating")
        images = min(cfg["batch_size"],spec["images"]-episode*cfg["batch_size"])
        for position,row in enumerate(rows[episode*4:(episode+1)*4]):
            require(type(row["episode"]) is int and row["episode"] == episode and row["arm"] == order[position]
                    and type(row["order_position"]) is int and row["order_position"] == position
                    and row["warmup"] is (episode == 0) and row["GT_opened"] is False
                    and type(row["images"]) is int and row["images"] == images == row["patches"]
                    and all(row[k] == v for k,v in archive_fields.items())
                    and all(row[k] is False for k in ("old_engine_loaded","old_optimizer_loaded","old_OOM_iterator_selected")),
                    "M03 raw cost order/images/archive/scope differs")
            counters = (*prediction.runtime.native.COUNTERS,"steps","optimizer_steps","candidate_count","active_candidates")
            require(all(type(row[k]) is int and row[k] >= 0 for k in counters)
                    and row["active_candidates"] <= row["candidate_count"] <= images*cfg["trace_max_regions"]
                    and type(row["seconds"]) in (float,int) and math.isfinite(row["seconds"]) and row["seconds"] >= 0,
                    "M03 invalid raw time/counters")
            steps = cfg["steps"] if row["active_candidates"] else 0
            views = row["diagnostic_forward_inputs"]
            require((views == row["candidate_count"] if row["arm"].startswith("C.") else
                     2*row["candidate_count"] <= views <= 3*row["candidate_count"]),
                    "M03 diagnostic view-count bound differs")
            calls = (views+cfg["trace_view_batch_size"]-1)//cfg["trace_view_batch_size"]
            require(row["steps"] == row["optimizer_steps"] == steps
                    and row["eval_forward_calls"] == 2 and row["eval_forward_inputs"] == 2*images
                    and row["adapt_forward_calls"] == steps and row["adapt_forward_inputs"] == steps*images
                    and row["diagnostic_forward_calls"] == row["diagnostic_encoder_calls"] == calls
                    and row["diagnostic_encoder_inputs"] == views
                    and row["encoder_calls"] == 2+steps+calls and row["encoder_inputs"] == (2+steps)*images+views,
                    "M03 measured update/forward schedule differs")
    counters = (*prediction.runtime.native.COUNTERS,"optimizer_steps")
    require(summary["dataset"] == dataset and summary["images"] == spec["images"]
            and summary["modes"] == list(MODES) and summary["GT_opened"] is False
            and summary["actual_work"] == {m:{k:sum(r[k] for r in rows if r["arm"] == m) for k in counters} for m in MODES}
            and summary["raw_model_seconds"] == {m:sum(r["seconds"] for r in rows if r["arm"] == m) for m in MODES}
            and summary["warmup_adjusted_throughput"] == prediction.prediction.runtime.prior.timing_summary(rows,MODES),
            "M03 cost summary differs from complete raw records")
    protected.validate_visual_LN_names(initial["legal_visual_LN"])
    require(summary["legal_visual_LN"] == initial["legal_visual_LN"]
            and summary["frozen_parameter_sha256"] == initial["frozen_parameter_sha256"]
            and set(initial["reset"]) == {"trainable_sha256","optimizer_sha256"}, "M03 initial state differs")
    for digest in (*initial["reset"].values(),initial["frozen_parameter_sha256"]):
        require(type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                "M03 invalid initial state digest")
    return summary,rows


def episode_metadata(parent,spec,inventory,forbidden,episode,raw_rows,source_ids,coverage,identities,components):
    by_mode = {r["arm"]:r for r in raw_rows[episode*4:(episode+1)*4]}
    require(set(by_mode) == set(MODES), "M03 incomplete episode cost modes")
    checked = {}
    for mode in MODES:
        relative = f"{spec['dataset']}/episodes/{episode:06d}.{mode}.json"
        saved,_ = protected.metadata(canonical_child(parent,relative),inventory[relative],forbidden,limit=8*1024**2,
                                    expected_identity=tuple(identities[relative]))
        checked[mode] = receipts.validate_episode(saved,spec,source_ids,by_mode[mode],components)
    receipts.validate_shared(checked)
    for mode,result in checked.items():
        for key in ("candidates","EPS_active_candidates","active_sources","diagnostic_views"):
            coverage[mode][key] += result[key]
        coverage[mode]["sources"] += len(source_ids)


def score_dataset(suite,bound,spec,budget,label_hashes,progress,components):
    """Stream each source GT once across four arms; caller owns GT admission."""
    parent,dataset = Path(bound["parent"]),spec["dataset"]
    rows,cfg = bound["source"]["rows"][dataset],spec["config"]
    require(type(budget["bytes"]) is int and 0 <= budget["bytes"] <= OUTPUT_CAP
            and dataset in ("coco_object","pc60") and 0 < len(rows) == spec["images"] <= POPULATIONS[dataset],
            "M03 scoring population/budget differs")
    classes,batch = len(spec["vocabulary"]["classes"]),cfg["batch_size"]
    require(0 < classes <= 256 and type(batch) is int and 0 < batch <= 512, "M03 scoring class/batch bound differs")
    forbidden = bounded.data_exclusions(bound["source"])
    archive = Path(bound["source"]["legacy_archive"]["path"]).stat()
    forbidden.add((archive.st_dev,archive.st_ino))
    inventory = bound["seals"][dataset]["artifact_sha256"]
    identities = preflight_payloads(parent,[spec],bound["seals"],bound["source"])
    expected = {name:value for name,value in bound["payload_metadata"].items() if name.startswith(dataset+"/")}
    require(identities == expected, "M03 payload identities changed since score preparation")
    cost,raw_rows = cost_binding(parent,spec,inventory,forbidden,identities,components)
    counts = np.zeros(((len(rows)+batch-1)//batch,4,2,classes),dtype=np.int64)
    coverage = {m:dict(candidates=0,EPS_active_candidates=0,active_sources=0,diagnostic_views=0,sources=0) for m in MODES}
    ordered,path = [],suite/(dataset+".episode_counts.jsonl")
    hasher = hashlib.sha256()
    with path.open("xb+") as stream:
        created = os.fstat(stream.fileno())
        written = 0
        def identity(state):
            return (state.st_dev,state.st_ino,state.st_size,state.st_mtime_ns,state.st_ctime_ns,state.st_nlink,state.st_mode)
        def observed():
            current,on_path = os.fstat(stream.fileno()),path.lstat()
            require(identity(current) == identity(on_path) and stat.S_ISREG(current.st_mode) and current.st_nlink == 1
                    and path.resolve(strict=True) == path
                    and (created.st_dev,created.st_ino) == (current.st_dev,current.st_ino)
                    and current.st_size == written, "M03 counts output identity/extent changed")
            return identity(current)
        previous = observed()
        for episode,start in enumerate(range(0,len(rows),batch)):
            current = rows[start:start+batch]
            ids = [r["official_id"] for r in current]
            episode_metadata(parent,spec,inventory,forbidden,episode,raw_rows,ids,coverage,identities,components)
            ordered.append(ids)
            gt_receipts = []
            for index,row in enumerate(current,start):
                content,digest = annotation_bytes(row)
                labels = decode_annotation(content,(224,224),classes,spec["vocabulary"]["reduce_zero_label"])
                key = row["label"]
                require(key not in label_hashes or label_hashes[key] == digest, "Repeated M03 GT bytes changed")
                label_hashes[key] = digest
                gt_receipts.append(dict(source_id=row["official_id"],label=key,sha256=digest))
                for a,mode in enumerate(MODES):
                    relative = f"{dataset}/predictions/{mode}/{index:06d}.npy"
                    raw,_ = bounded.pinned_content(parent,relative,inventory[relative],forbidden,2*1024**2,identities)
                    counts[episode,a] += metric.counts(hard_array(raw,classes,(224,224)),labels,classes)
            line = (json.dumps(dict(episode=episode,ordered_ids=ids,labels=gt_receipts,
                I_U_by_mode=counts[episode].tolist()),allow_nan=False)+"\n").encode()
            require(shutil.disk_usage(suite).free >= RESERVE+len(line) and budget["bytes"]+len(line) <= OUTPUT_CAP,
                    "M03 score output budget exceeded")
            require(observed() == previous, "M03 completed counts changed before write")
            require(stream.write(line) == len(line), "Short M03 episode-count write")
            stream.flush()
            written += len(line)
            hasher.update(line)
            budget["bytes"] += len(line)
            previous = observed()
            progress(dict(phase="CPU_M03_score",dataset=dataset,completed_sources=min(start+batch,len(rows)),
                total_sources=len(rows),GT_opened=True,pid=os.getpid()))
            require(observed() == previous, "M03 completed counts changed during progress")
        stream.flush()
        os.fsync(stream.fileno())
        require(observed() == previous, "M03 completed counts changed before final read")
        stream.seek(0)
        persisted,total = hashlib.sha256(),0
        while chunk := stream.read(min(64*1024,written-total+1)):
            total += len(chunk)
            require(total <= written <= OUTPUT_CAP, "M03 persisted counts exceed output bound")
            persisted.update(chunk)
        require(total == written and persisted.hexdigest() == hasher.hexdigest() and observed() == previous,
                "M03 persisted counts digest/identity differs")
    require(identity(path.lstat()) == previous and path.resolve(strict=True) == path,
            "M03 completed counts changed during close")
    return counts,ordered,cost,coverage,path,hasher.hexdigest()
