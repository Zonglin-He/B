"""Bounded native-PIL M04 scoring core for a future admitted CPU controller.

There is no CLI, admission, launch, prepare or seal here. A controller must bind
all16 sealed predictions and the exact successful parent exit, then obtain NEW
independent M04 CPU_GT permission before calling this with real data.
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

from . import factorial_prediction as prediction
from . import factorial_score_receipts as receipts
from . import generalization_score as metric
from . import generalization_score_worker as protected
from . import spatial_score_worker as shared
from .bootstrap import POPULATIONS,require
from .risk_score_inputs import decode_annotation
from .risk_score_payload import annotation_bytes,hard_array,stable_bytes
from .risk_score_worker import canonical_child

MODES = prediction.MODES
RESERVE,OUTPUT_CAP = 16*1024**3,256*1024**2
ROOT = Path(__file__).resolve().parents[2]


def data_exclusions(source):
    """Stat-only GT/RGB role separation, before metadata or source reads."""
    forbidden = set(shared.prediction.annotation_inodes(source))
    annotation_ids = set(forbidden)
    for rows in source["rows"].values():
        for row in rows:
            path = Path(row["image"])
            state = path.lstat()
            require(path.is_absolute() and path.resolve(strict=True) == path and stat.S_ISREG(state.st_mode)
                    and (state.st_dev,state.st_ino) not in annotation_ids, "M04 RGB path/annotation alias differs")
            forbidden.add((state.st_dev,state.st_ino))
    return forbidden


def preflight_payloads(parent,specs,seals,source):
    """Stat ALL declared payloads before any GT, without reading their bytes."""
    parent = Path(parent)
    forbidden = data_exclusions(source)
    roots = [(ROOT,True)]
    if (parent/"runtime").exists():
        roots.append((parent/"runtime",False))
    for root,snapshot in roots:
        members = protected.source_members(root,forbidden,snapshot_source=snapshot)
        forbidden.update((s[0],s[1]) for s in members.values())
    identities,seen = {},set()
    for spec in specs:
        dataset,images,batch = spec["dataset"],spec["images"],spec["config"]["batch_size"]
        require(dataset in ("coco_object","pc60") and type(images) is int and 0 < images <= POPULATIONS[dataset]
                and type(batch) is int and 0 < batch <= 512, "M04 payload preflight population differs")
        expected = {f"{dataset}/predictions/{m}/{i:06d}.npy":2*1024**2 for m in MODES for i in range(images)}
        expected.update({f"{dataset}/episodes/{e:06d}.{m}.json":8*1024**2
                         for m in MODES for e in range((images+batch-1)//batch)})
        expected.update({f"{dataset}/{name}":limit for name,limit in
                         (("cost.jsonl",16*1024**2),("initial_state.json",2*1024**2),("prediction_result.json",2*1024**2))})
        inventory = seals[dataset]["artifact_sha256"]
        require(set(inventory) == set(expected), "M04 preflight complete payload membership differs")
        for relative,limit in expected.items():
            digest = inventory[relative]
            require(type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                    "M04 declared payload digest differs")
            path = parent/relative
            state = path.lstat()
            identity = (state.st_dev,state.st_ino)
            require(stat.S_ISREG(state.st_mode) and state.st_nlink == 1 and path.resolve(strict=True) == path
                    and identity not in forbidden and identity not in seen and 0 < state.st_size <= limit,
                    "M04 payload preflight path/inode/extent differs")
            seen.add(identity)
            identities[relative] = [*identity,state.st_size,state.st_mtime_ns]
    return identities


def pinned_content(parent,relative,digest,forbidden,limit,identities):
    path = canonical_child(parent,relative)
    state = path.stat()
    require((state.st_dev,state.st_ino) not in forbidden, "M04 payload aliases a protected input")
    return stable_bytes(path,limit,digest,expected_identity=tuple(identities[relative]))


def cost_binding(parent,spec,inventory,forbidden,identities):
    """Metadata-only full schedule/state binding; no GT or prediction arrays."""
    dataset,cfg = spec["dataset"],spec["config"]
    def saved(name):
        relative = dataset+"/"+name
        return protected.metadata(canonical_child(parent,relative),inventory[relative],forbidden,limit=2*1024**2,
                                  expected_identity=tuple(identities[relative]))[0]
    summary,initial = saved("prediction_result.json"),saved("initial_state.json")
    relative = dataset+"/cost.jsonl"
    raw,_ = pinned_content(parent,relative,inventory[relative],forbidden,16*1024**2,identities)
    lines = raw.splitlines()
    require(all(0 < len(line) <= 64*1024 for line in lines), "M04 cost line bound exceeded")
    rows = [json.loads(line) for line in lines]
    episodes = (spec["images"]+cfg["batch_size"]-1)//cfg["batch_size"]
    require(len(rows) == episodes*len(MODES), "M04 raw cost population differs")
    for episode in range(episodes):
        order = prediction.prediction.runtime.prior.arm_order(MODES,episode,"alternating")
        images = min(cfg["batch_size"],spec["images"]-episode*cfg["batch_size"])
        for position,row in enumerate(rows[episode*8:(episode+1)*8]):
            mode = order[position]
            study,arm = mode.split(".")
            require(type(row["episode"]) is int and row["episode"] == episode and row["arm"] == mode
                    and type(row["order_position"]) is int and row["order_position"] == position
                    and row["warmup"] is (episode == 0) and row["study"] == study and row["factorial_arm"] == arm
                    and row["common_active"] is (study == "common_active")
                    and row["all_arms_measure_C"] is True and row["production_Confidence_timing"] is False
                    and row["coefficient_observation_in_model_time"] is True
                    and type(row["images"]) is int and row["images"] == images == row["patches"],
                    "M04 raw cost order/images/scope differs")
            counters = (*prediction.runtime.native.COUNTERS,"steps","optimizer_steps","candidate_count","active_candidates")
            require(all(type(row[k]) is int and row[k] >= 0 for k in counters)
                    and 0 <= row["active_candidates"] <= row["candidate_count"] <= images*cfg["trace_max_regions"]
                    and type(row["seconds"]) in (float,int) and math.isfinite(row["seconds"]) and row["seconds"] >= 0,
                    "M04 invalid raw time/counters")
            steps = cfg["steps"] if row["active_candidates"] else 0
            views = row["candidate_count"]
            calls = (views+cfg["trace_view_batch_size"]-1)//cfg["trace_view_batch_size"]
            require(row["steps"] == row["optimizer_steps"] == steps
                    and row["eval_forward_calls"] == 2 and row["eval_forward_inputs"] == 2*images
                    and row["adapt_forward_calls"] == steps and row["adapt_forward_inputs"] == steps*images
                    and row["diagnostic_forward_calls"] == row["diagnostic_encoder_calls"] == calls
                    and row["diagnostic_forward_inputs"] == row["diagnostic_encoder_inputs"] == views
                    and row["encoder_calls"] == 2+steps+calls and row["encoder_inputs"] == (2+steps)*images+views,
                    "M04 measured update/forward schedule differs")
    counters = (*prediction.runtime.native.COUNTERS,"optimizer_steps")
    require(summary["dataset"] == dataset and summary["images"] == spec["images"]
            and summary["modes"] == list(MODES) and summary["studies"] == list(prediction.STUDIES)
            and summary["factorial_arms"] == list(prediction.runtime.ARMS) and summary["GT_opened"] is False
            and summary["actual_work"] == {m:{k:sum(r[k] for r in rows if r["arm"] == m) for k in counters} for m in MODES}
            and summary["raw_model_seconds"] == {m:sum(r["seconds"] for r in rows if r["arm"] == m) for m in MODES}
            and summary["warmup_adjusted_throughput"] == prediction.prediction.runtime.prior.timing_summary(rows,MODES),
            "M04 cost summary differs from complete raw records")
    protected.validate_visual_LN_names(initial["legal_visual_LN"])
    require(summary["legal_visual_LN"] == initial["legal_visual_LN"]
            and summary["frozen_parameter_sha256"] == initial["frozen_parameter_sha256"]
            and set(initial["reset"]) == {"trainable_sha256","optimizer_sha256"}, "M04 initial state differs")
    for digest in (*initial["reset"].values(),initial["frozen_parameter_sha256"]):
        require(type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                "M04 invalid initial state digest")
    return summary,rows


def episode_metadata(parent,spec,inventory,forbidden,episode,raw_rows,source_ids,coverage,identities):
    by_mode = {r["arm"]:r for r in raw_rows[episode*8:(episode+1)*8]}
    require(set(by_mode) == set(MODES), "M04 incomplete episode cost modes")
    checked = {}
    for mode in MODES:
        relative = f"{spec['dataset']}/episodes/{episode:06d}.{mode}.json"
        saved,_ = protected.metadata(canonical_child(parent,relative),inventory[relative],forbidden,limit=8*1024**2,
                                    expected_identity=tuple(identities[relative]))
        checked[mode] = receipts.validate_episode(saved,spec,source_ids,by_mode[mode])
    receipts.validate_shared(checked)
    for mode,result in checked.items():
        for key,value in (("original_candidates",len(result["original_candidates"])),
                          ("retained_candidates",result["retained_candidates"]),
                          ("EPS_active_candidates",result["EPS_active_candidates"]),
                          ("active_sources",result["active_sources"]),("sources",len(source_ids))):
            coverage[mode][key] += value


def score_dataset(suite,bound,spec,budget,label_hashes,progress):
    """Stream each GT once across eight arms; caller owns exact GT admission."""
    parent,dataset = Path(bound["parent"]),spec["dataset"]
    rows,cfg = bound["source"]["rows"][dataset],spec["config"]
    require(type(budget["bytes"]) is int and 0 <= budget["bytes"] <= OUTPUT_CAP
            and dataset in ("coco_object","pc60") and 0 < len(rows) == spec["images"] <= POPULATIONS[dataset],
            "M04 scoring population/budget differs")
    classes,batch = len(spec["vocabulary"]["classes"]),cfg["batch_size"]
    require(0 < classes <= 256 and type(batch) is int and 0 < batch <= 512,
            "M04 scoring class/batch resource bound differs")
    forbidden = data_exclusions(bound["source"])
    inventory = bound["seals"][dataset]["artifact_sha256"]
    identities = preflight_payloads(parent,[spec],bound["seals"],bound["source"])
    expected = {name:value for name,value in bound["payload_metadata"].items() if name.startswith(dataset+"/")}
    require(identities == expected, "M04 payload identities changed since score preparation")
    cost,raw_rows = cost_binding(parent,spec,inventory,forbidden,identities)
    counts = np.zeros(((len(rows)+batch-1)//batch,2,4,2,classes),dtype=np.int64)
    coverage = {m:dict(original_candidates=0,retained_candidates=0,EPS_active_candidates=0,active_sources=0,sources=0) for m in MODES}
    ordered,path = [],suite/(dataset+".episode_counts.jsonl")
    hasher = hashlib.sha256()
    with path.open("xb") as stream:
        created = os.fstat(stream.fileno())
        for episode,start in enumerate(range(0,len(rows),batch)):
            current = rows[start:start+batch]
            ids = [r["official_id"] for r in current]
            episode_metadata(parent,spec,inventory,forbidden,episode,raw_rows,ids,coverage,identities)
            ordered.append(ids)
            gt_receipts = []
            for index,row in enumerate(current,start):
                content,digest = annotation_bytes(row)
                labels = decode_annotation(content,(224,224),classes,spec["vocabulary"]["reduce_zero_label"])
                key = row["label"]
                require(key not in label_hashes or label_hashes[key] == digest, "Repeated M04 GT bytes changed")
                label_hashes[key] = digest
                gt_receipts.append(dict(source_id=row["official_id"],label=key,sha256=digest))
                for a,mode in enumerate(MODES):
                    relative = f"{dataset}/predictions/{mode}/{index:06d}.npy"
                    raw,_ = pinned_content(parent,relative,inventory[relative],forbidden,2*1024**2,identities)
                    counts[episode,a//4,a%4] += metric.counts(hard_array(raw,classes,(224,224)),labels,classes)
            line = (json.dumps(dict(episode=episode,ordered_ids=ids,labels=gt_receipts,
                I_U_by_study_arm=counts[episode].tolist()),allow_nan=False)+"\n").encode()
            require(shutil.disk_usage(suite).free >= RESERVE+len(line) and budget["bytes"]+len(line) <= OUTPUT_CAP,
                    "M04 score output budget exceeded")
            require(stream.write(line) == len(line), "Short M04 episode-count write")
            stream.flush()
            hasher.update(line)
            budget["bytes"] += len(line)
            progress(dict(phase="CPU_M04_score",dataset=dataset,completed_sources=min(start+batch,len(rows)),
                total_sources=len(rows),GT_opened=True,pid=os.getpid()))
        stream.flush()
        os.fsync(stream.fileno())
        after,path_state = os.fstat(stream.fileno()),path.stat()
        require((created.st_dev,created.st_ino) == (after.st_dev,after.st_ino) == (path_state.st_dev,path_state.st_ino)
                and after.st_size == path_state.st_size, "M04 counts output identity changed")
    return counts,ordered,cost,coverage,path,hasher.hexdigest()
