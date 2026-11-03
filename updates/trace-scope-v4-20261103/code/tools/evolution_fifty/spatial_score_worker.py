"""M02-only bounded CPU scoring, separately admitted after prediction exit.

Preparation/binding reads metadata, never annotation or prediction contents.
Actual GT and hard-label reads exist only in execute(), behind new admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

import numpy as np
import torch

from trace_ovss.method.probe_geometry import translated_candidate
from . import generalization_score as metric
from . import generalization_score_worker as protected
from . import spatial_analysis as analysis
from . import spatial_runtime as runtime
from . import spatial_worker as prediction
from .bootstrap import POPULATIONS, require
from .intake import sha256
from .risk_score_inputs import decode_annotation
from .risk_score_payload import annotation_bytes, hard_array, stable_bytes
from .risk_score_worker import canonical_child, publish_seal, verify_audit_binding

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/home/wwww/hzl/results/trace_mechanism_20261102")
PARENT = BASE/"M02_spatial_20261102_2156"
PARENT_SHA = "f3eaa8494d4bdd07224b0c640af13985a92dd5ea3391761d13a91c1459ef2e69"
PARENT_BARRIER_SHA = "10ea6d4e43f9e4dd2829de66dab52310af2d4c80290cf5f3f1a695d24132dc73"
PARENT_START_SHA = "2546b33bbc5f5f08bf5ec82eae4c14d262ecde251d2fbd7e92c90eecc8d6fb3a"
PARENT_ADMISSION_SHA = "7a4265370647faacebb1a8d1697b1ea1d4f641f2e8b02667d249f88928590be8"
PARENT_REPORT_SHA = "d7ef4629d0c57ac30dca4a62972e7ca07b492c14a2c9a1d792f674bd7b2b127f"
ANALYSIS = BASE/"M02_ANALYSIS_PROTOCOL_v1.json"
ANALYSIS_SHA = "84df1e6183124027de039ef342bed1eb1d1238b8809dc9fc109c67d00ea9a21c"
RESERVE, OUTPUT_CAP = 16*1024**3, 256*1024**2
MODES, DATASETS = runtime.MODES, prediction.DATASETS


def write_json(path,value,*,replace=False):
    """Advisory progress only: never truncate a pre-existing temporary inode."""
    path = Path(path)
    require(replace is True and path.name == "queue_state.json" and path.is_absolute()
            and path.parent.resolve(strict=True) == path.parent, "M02 progress path differs")
    require(not path.is_symlink() and (not path.exists() or path.stat().st_nlink == 1),
            "M02 progress destination is aliased")
    content = (json.dumps(value,indent=2,allow_nan=False)+"\n").encode()
    require(len(content) <= 64*1024, "M02 progress exceeds byte bound")
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("xb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "M02 progress staging inode differs")
        require(stream.write(content) == len(content), "Short M02 progress write")
        stream.flush()
        os.fsync(stream.fileno())
        actual,opened = temporary.stat(),os.fstat(stream.fileno())
        require(not temporary.is_symlink() and actual.st_nlink == opened.st_nlink == 1
                and (actual.st_dev,actual.st_ino,actual.st_size,actual.st_mtime_ns,actual.st_ctime_ns)
                == (opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns,opened.st_ctime_ns)
                and (opened.st_dev,opened.st_ino) == (before.st_dev,before.st_ino)
                and opened.st_size == len(content), "M02 progress staging pathname changed")
    temporary.replace(path)


def cpu_only():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "M02 CPU scoring requires hidden, uninitialized CUDA")


def parent_terminal():
    keys = "LoadState,ActiveState,SubState,MainPID,ExecMainPID,ExecMainCode,ExecMainStatus,Result,InvocationID"
    result = subprocess.run(["systemctl","--user","show","trace-spatial-20261102-2156.service",
        "--property="+keys],check=True,text=True,capture_output=True,timeout=30)
    actual = dict(line.split("=",1) for line in result.stdout.splitlines() if "=" in line)
    expected = dict(LoadState="loaded",ActiveState="active",SubState="exited",MainPID="0",ExecMainPID="2086569",
        ExecMainCode="1",ExecMainStatus="0",Result="success",InvocationID="775d3e1d206d42e0801641992d4b4ed4")
    require(all(actual.get(k) == v for k,v in expected.items()), "Exact M02 parent has not successfully exited")
    return actual


def bounded_content(path,digest,forbidden,limit):
    path = Path(path)
    state = path.stat()
    require((state.st_dev,state.st_ino) not in forbidden, "M02 score payload aliases GT")
    return stable_bytes(path,limit,digest,expected_identity=(state.st_dev,state.st_ino,state.st_size,state.st_mtime_ns))


def native_binding(expected,forbidden):
    """The pinned M02 preparation supplies native hashes; stat all before reads."""
    critical = ("tools/evolution_fifty/risk_worker.py","tools/evolution_fifty/unit_risk.py")
    prefixes = ("trace_ovss/","tools/evolution/","tools/evolution_confirmation/",
                "tools/evolution_followup/","tools/evolution_image_full/")
    require(type(expected) is dict and all(n in expected for n in critical)
            and any(n.startswith("trace_ovss/") for n in expected)
            and all(n.startswith(prefixes) or n in critical for n in expected), "M02 native membership differs")
    members = protected.source_members(ROOT,forbidden,snapshot_source=True)
    require(set(expected) <= set(members), "M02 native source member missing")
    for name,digest in expected.items():
        protected.source_content(ROOT/name,members[name],digest)


def cost_binding(parent,spec,inventory,forbidden):
    dataset,cfg = spec["dataset"],spec["config"]
    def saved(name):
        relative = dataset+"/"+name
        return protected.metadata(canonical_child(parent,relative),inventory[relative],forbidden,limit=2*1024**2)[0]
    summary,initial = saved("prediction_result.json"),saved("initial_state.json")
    relative = dataset+"/cost.jsonl"
    raw,_ = bounded_content(canonical_child(parent,relative),inventory[relative],forbidden,16*1024**2)
    lines = raw.splitlines()
    require(all(0 < len(line) <= 64*1024 for line in lines), "M02 cost line bound exceeded")
    rows = [json.loads(line) for line in lines]
    episodes = (spec["images"]+cfg["batch_size"]-1)//cfg["batch_size"]
    require(len(rows) == episodes*4, "M02 raw cost population differs")
    for episode in range(episodes):
        order = prediction.general.runtime.prior.arm_order(MODES,episode,"alternating")
        images = min(cfg["batch_size"],spec["images"]-episode*cfg["batch_size"])
        for position,row in enumerate(rows[episode*4:(episode+1)*4]):
            require(row["episode"] == episode and row["arm"] == row["mode"] == order[position]
                    and row["order_position"] == position and row["warmup"] is (episode == 0)
                    and type(row["images"]) is int and row["images"] == images == row["patches"],
                    "M02 raw cost order/images differs")
            require(all(type(row[k]) is int and row[k] >= 0 for k in
                        (*runtime.image_worker.COUNTERS,"steps","optimizer_steps","candidate_count","active_candidates"))
                    and 0 <= row["active_candidates"] <= row["candidate_count"] <= images*cfg["trace_max_regions"]
                    and type(row["seconds"]) in (float,int) and math.isfinite(row["seconds"])
                    and row["seconds"] >= 0, "Invalid M02 raw time/counters")
            steps = cfg["steps"] if row["active_candidates"] else 0
            views = row["diagnostic_forward_inputs"]
            expected_views = 0 if row["mode"] == "confidence" else row["candidate_count"]
            require((views == expected_views if row["mode"] != "global" else
                     0 <= views <= min(images,row["candidate_count"]))
                    and (views > 0 if row["candidate_count"] and row["mode"] == "global" else True),
                    "M02 diagnostic view scope differs")
            calls = (views+cfg["trace_view_batch_size"]-1)//cfg["trace_view_batch_size"]
            require(row["steps"] == row["optimizer_steps"] == steps
                    and row["eval_forward_calls"] == 2 and row["eval_forward_inputs"] == 2*images
                    and row["adapt_forward_calls"] == steps and row["adapt_forward_inputs"] == steps*images
                    and row["diagnostic_forward_calls"] == row["diagnostic_encoder_calls"] == calls
                    and row["diagnostic_encoder_inputs"] == views
                    and row["encoder_calls"] == 2+steps+calls and row["encoder_inputs"] == (2+steps)*images+views,
                    "M02 actual raw update/forward schedule differs")
    counters = (*runtime.image_worker.COUNTERS,"optimizer_steps")
    require(summary["dataset"] == dataset and summary["images"] == spec["images"]
            and summary["modes"] == list(MODES) and summary["GT_opened"] is False
            and summary["actual_work"] == {m:{k:sum(r[k] for r in rows if r["mode"] == m) for k in counters} for m in MODES}
            and summary["raw_model_seconds"] == {m:sum(r["seconds"] for r in rows if r["mode"] == m) for m in MODES}
            and summary["warmup_adjusted_throughput"] == prediction.general.runtime.prior.timing_summary(rows,MODES),
            "M02 cost/state summary mismatch")
    legal = initial["legal_visual_LN"]
    protected.validate_visual_LN_names(legal)
    require(summary["legal_visual_LN"] == legal and summary["frozen_parameter_sha256"] == initial["frozen_parameter_sha256"]
            and set(initial["reset"]) == {"trainable_sha256","optimizer_sha256"}, "M02 initial state differs")
    for digest in (*initial["reset"].values(),initial["frozen_parameter_sha256"]):
        require(isinstance(digest,str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                "Invalid M02 initial state digest")
    return summary,rows


def bind_parent():
    terminal = parent_terminal()
    source = prediction.source_inputs()
    forbidden = prediction.annotation_inodes(source)
    require(PARENT.is_absolute() and PARENT.resolve(strict=True) == PARENT
            and not any((PARENT/n).exists() for n in ("FAILED.json","CANCELLED.json")), "Unsuccessful M02 parent")
    prepared,_ = protected.metadata(PARENT/"SUITE_PREPARED.json",PARENT_SHA,forbidden)
    inputs,_ = protected.metadata(PARENT/"INPUTS.json",prepared["inputs_sha256"],forbidden)
    require(prepared["root"] == str(PARENT) and prepared["schema_version"] == 1 and inputs == source,
            "M02 immutable source differs")
    native_binding(prepared["parent_native_hashes"],forbidden)
    prediction.validate_inputs(inputs["rows"],inputs["specs"])
    protected.metadata(PARENT/"PROTOCOL.json",prepared["protocol_sha256"],forbidden)
    protected.guarded_inventory(PARENT/"runtime",prepared["runtime_hashes"],forbidden)
    barrier,barrier_sha = protected.metadata(PARENT/"ALL_PREDICTIONS_SEALED.json",PARENT_BARRIER_SHA,forbidden)
    require(barrier["complete"] is True and barrier["GT_opened"] is False and barrier["suite_prepared_sha256"] == PARENT_SHA
            and barrier["cells"] == 8 and barrier["image_method_evaluations"] == sum(POPULATIONS[d]*4 for d in DATASETS)
            and set(barrier["dataset_seals"]) == set(DATASETS) and barrier["started_sha256"] == PARENT_START_SHA,
            "M02 complete prediction barrier differs")
    audit = barrier["audit_binding"]
    require(audit["admission_path"] == str(PARENT/"ADMITTED.json") and audit["report_path"] == str(PARENT/"INDEPENDENT_REPORT.md")
            and audit["admission_sha256"] == PARENT_ADMISSION_SHA and audit["report_sha256"] == PARENT_REPORT_SHA,
            "Wrong M02 prediction audit binding")
    admitted,_ = protected.metadata(PARENT/"ADMITTED.json",PARENT_ADMISSION_SHA,forbidden)
    report_state = (PARENT/"INDEPENDENT_REPORT.md").stat()
    require((report_state.st_dev,report_state.st_ino) not in forbidden, "M02 parent report aliases GT")
    verify_audit_binding(audit)
    start,_ = protected.metadata(PARENT/"STARTED.json",PARENT_START_SHA,forbidden)
    require(admitted["decision"] == "ADMITTED" and admitted["suite_prepared_sha256"] == PARENT_SHA
            and admitted["M02_predictions_allowed"] is True and admitted["GT_scoring_allowed"] is False
            and start["suite_prepared_sha256"] == PARENT_SHA and type(start["pid"]) is int
            and str(start["pid"]) == terminal["ExecMainPID"] and start["audit"] == audit and start["GT_opened"] is False,
            "M02 prediction permission/startup differs")
    prediction.verify_gates(PARENT,PARENT_SHA,barrier["gate_binding"],forbidden)
    seals = {}
    for spec in inputs["specs"]:
        dataset,images,batch = spec["dataset"],spec["images"],spec["config"]["batch_size"]
        seal,_ = protected.metadata(PARENT/dataset/"PREDICTIONS_SEALED.json",barrier["dataset_seals"][dataset],forbidden)
        require(seal["complete"] is True and seal["GT_opened"] is False and seal["dataset"] == dataset
                and seal["suite_prepared_sha256"] == PARENT_SHA and seal["images"] == images == POPULATIONS[dataset]
                and seal["modes"] == list(MODES) and seal["image_method_evaluations"] == images*4,
                "M02 dataset prediction population differs")
        expected = {f"{dataset}/predictions/{m}/{i:06d}.npy" for m in MODES for i in range(images)}
        expected |= {f"{dataset}/episodes/{e:06d}.{m}.json" for m in MODES for e in range((images+batch-1)//batch)}
        expected |= {f"{dataset}/{n}" for n in ("initial_state.json","cost.jsonl","prediction_result.json")}
        require(set(seal["artifact_sha256"]) == expected, "M02 full prediction inventory differs")
        cost_binding(PARENT,spec,seal["artifact_sha256"],forbidden)
        seals[dataset] = seal
    protocol,_ = protected.metadata(ANALYSIS,ANALYSIS_SHA,forbidden)
    require(protocol["prediction_prepared_sha256"] == PARENT_SHA and protocol["arm_order"] == list(MODES)
            and protocol["fixed_contrasts"] == {k:list(v) for k,v in analysis.CONTRASTS.items()}
            and protocol["uncertainty"]["replicates"] == 10000 and protocol["uncertainty"]["root_seed"] == 20261102,
            "M02 prospective analysis differs")
    return dict(parent=str(PARENT),prepared_sha256=PARENT_SHA,barrier_sha256=barrier_sha,
                terminal=terminal,source=inputs,seals=seals,analysis_sha256=ANALYSIS_SHA)


def prepare(output):
    cpu_only()
    parent_terminal()
    bound = bind_parent()
    output = Path(output).absolute()
    require(output.resolve() == output and not output.exists(), "Fresh canonical M02 score suite required")
    output.mkdir()
    inputs_sha = protected.write_prepared_json(output/"SCORE_INPUTS.json",bound)
    protocol_sha = protected.write_prepared_json(output/"PROTOCOL.json",dict(experiment="M02_offline_CPU_score",
        parent_prepared_sha256=PARENT_SHA,prospective_analysis_sha256=ANALYSIS_SHA,new_GPU=False,
        boundary="All8 predictions sealed AND actual success exit AND new independent exact CPU_GT and M02_GT permission",
        annotation_geometry="Historical native PIL NEAREST224, palette indices preserved; native reduce-zero/ignore mapping; NOT G OpenCV nearest",
        estimates="Pooled per-class I/U for all4modes, three fixed contrasts;10000 shared original-episode PCG64 draws, pointwise95/undefined retained/nullCI",
        coverage="Original candidate identity across4modes, EPS-active counts, translated preserved-area/shift/overlap/fallback coverage; not candidate causality",
        time="Retain all raw model time/work and episode0; offline overhead separate, no equal-time claim",
        failures="Failclosed/noresume/automaticretry; atomic global seal last",
        limits=dict(output_cap_bytes=OUTPUT_CAP,shared_disk_reserve_bytes=RESERVE,GPU=False)))
    inventory = protected.guarded_snapshot(ROOT,output/"runtime",prediction.annotation_inodes(bound["source"]))
    return protected.write_prepared_json(output/"SUITE_PREPARED.json",dict(schema_version=1,root=str(output),
        runtime_hashes=inventory,inputs_sha256=inputs_sha,protocol_sha256=protocol_sha,
        parent_prepared_sha256=PARENT_SHA,GT_opened=False,CPU_scoring_started=False))


def admission(suite,expected_sha):
    cpu_only()
    bound = bind_parent()
    forbidden = prediction.annotation_inodes(bound["source"])
    suite = Path(suite).resolve(strict=True)
    prepared,_ = protected.metadata(suite/"SUITE_PREPARED.json",expected_sha,forbidden)
    require(prepared["root"] == str(suite) and prepared["schema_version"] == 1
            and prepared["parent_prepared_sha256"] == PARENT_SHA and Path(__file__).resolve().is_relative_to(suite/"runtime"),
            "Wrong exact M02 score runtime")
    protected.guarded_inventory(suite/"runtime",prepared["runtime_hashes"],forbidden)
    prediction.general.runtime_imports(suite/"runtime")
    inputs,_ = protected.metadata(suite/"SCORE_INPUTS.json",prepared["inputs_sha256"],forbidden)
    protected.metadata(suite/"PROTOCOL.json",prepared["protocol_sha256"],forbidden)
    require(inputs == bound, "M02 scoring parent binding changed")
    audit,audit_sha = protected.metadata(suite/"ADMITTED.json",forbidden=forbidden,limit=4*1024**2)
    require(audit["decision"] == "ADMITTED" and audit["suite_prepared_sha256"] == expected_sha
            and audit.get("independent_reviewer") and audit.get("CPU_GT_scoring_allowed") is True
            and audit.get("M02_GT_scoring_allowed") is True, "New exact M02 CPU_GT admission required")
    report = suite/"INDEPENDENT_REPORT.md"
    require(audit["report_path"] == str(report), "Wrong M02 score audit report path")
    identities = []
    for path in (suite/"ADMITTED.json",report):
        state = path.stat()
        require((state.st_dev,state.st_ino) not in forbidden, "M02 score audit/report aliases GT")
        identities.append([state.st_dev,state.st_ino,state.st_size,state.st_mtime_ns])
    binding = dict(admission_path=str(suite/"ADMITTED.json"),admission_sha256=audit_sha,admission_identity=identities[0],
        report_path=str(report),report_sha256=audit["report_sha256"],report_identity=identities[1],independent_reviewer=audit["independent_reviewer"])
    verify_audit_binding(binding)
    return suite,prepared,inputs,binding


def episode_metadata(parent,spec,inventory,forbidden,episode,raw_rows,coverage):
    dataset,cfg = spec["dataset"],spec["config"]
    by_mode = {row["mode"]:row for row in raw_rows[episode*4:(episode+1)*4]}
    shared_identity,ordered = None,{}
    for mode in MODES:
        relative = f"{dataset}/episodes/{episode:06d}.{mode}.json"
        saved,_ = protected.metadata(canonical_child(parent,relative),inventory[relative],forbidden,limit=8*1024**2)
        require(saved["GT_opened"] is False and saved["scientific_identity"] == "M02_spatial_"+mode
                and all(saved[k] == v for k,v in by_mode[mode].items())
                and len(saved["losses"]) == saved["optimizer_steps"]
                and all(type(v) in (float,int) and math.isfinite(v) for v in saved["losses"]),
                "M02 episode cost/loss identity differs")
        candidates,geometry = saved["candidates"],saved["preservation_geometry"]
        require(len(candidates) == saved["candidate_count"] == len(geometry), "M02 candidate/geometry membership differs")
        decoded = []
        for i,(candidate,geom) in enumerate(zip(candidates,geometry,strict=True)):
            region = candidate["region"]
            require(region["shape"] == [16,16] and region["encoding"] == "packbits-little-hex"
                    and type(region["data"]) is str and len(region["data"]) == 64,
                    "M02 native L14 candidate mask differs")
            mask = np.unpackbits(np.frombuffer(bytes.fromhex(region["data"]),dtype=np.uint8),bitorder="little").astype(bool).reshape(16,16)
            require(int(mask.sum()) == region["true_pixels"] == candidate["area"] > 0
                    and type(candidate["batch_idx"]) is int and candidate["image_idx"] == candidate["batch_idx"]
                    and 0 <= candidate["batch_idx"] < saved["images"] and type(candidate["class_idx"]) is int
                    and 0 <= candidate["class_idx"] < len(spec["vocabulary"]["queries"])
                    and all(type(candidate[k]) in (float,int) and math.isfinite(candidate[k]) and 0 <= candidate[k] <= 1
                            for k in ("confidence","reliability")), "M02 original candidate identity/support differs")
            decoded.append(dict(candidate,region=mask))
            require(geom["candidate_index"] == i and geom["batch_idx"] == candidate["batch_idx"]
                    and geom["mode"] == mode and geom["readout_and_loss_remain_original"] is True,
                    "M02 geometry changed readout/loss or membership")
            if mode == "translated":
                area = candidate["area"]*196
                _,expected_geometry = translated_candidate(np.repeat(np.repeat(mask,14,axis=0),14,axis=1))
                require(all(type(geom[k]) is int for k in ("dy","dx","original_area","shifted_area","intersection"))
                        and geom["original_area"] == geom["shifted_area"] == area
                        and 0 <= geom["intersection"] <= area and type(geom["shiftable"]) is bool
                        and geom["shiftable"] == ((geom["dy"],geom["dx"]) != (0,0))
                        and all(math.isfinite(geom[k]) for k in ("overlap_of_candidate","iou"))
                        and geom["overlap_of_candidate"] == geom["intersection"]/area
                        and geom["iou"] == geom["intersection"]/(2*area-geom["intersection"]),
                        "M02 translated area/overlap/fallback receipt differs")
                require(all(geom[k] == value for k,value in expected_geometry.items()),
                        "M02 translated geometry differs from original-mask deterministic policy")
                coverage["translated_shiftable"] += int(geom["shiftable"])
                coverage["translated_unshiftable"] += int(not geom["shiftable"])
                coverage["translated_overlap_sum"] += geom["overlap_of_candidate"]
        runtime.validate_cost(saved,decoded,cfg)
        identity = runtime.image_worker.prior.candidate_identity(decoded)
        require(saved["candidate_sha256"] == identity and (shared_identity is None or identity == shared_identity),
                "M02 original candidates differ across modes")
        shared_identity = identity
        ordered[mode] = saved["ordered_ids"]
        coverage[mode]["candidates"] += len(candidates)
        coverage[mode]["EPS_active_candidates"] += saved["active_candidates"]
        coverage[mode]["sources"] += saved["images"]
    return ordered


def score_dataset(suite,bound,spec,budget,label_hashes):
    parent,dataset = Path(bound["parent"]),spec["dataset"]
    rows,cfg = bound["source"]["rows"][dataset],spec["config"]
    classes,batch = len(spec["vocabulary"]["classes"]),cfg["batch_size"]
    forbidden = prediction.annotation_inodes(bound["source"])
    inventory = bound["seals"][dataset]["artifact_sha256"]
    cost,raw_rows = cost_binding(parent,spec,inventory,forbidden)
    counts = np.zeros(((len(rows)+batch-1)//batch,4,2,classes),dtype=np.int64)
    coverage = {m:dict(candidates=0,EPS_active_candidates=0,sources=0) for m in MODES}
    coverage.update(translated_shiftable=0,translated_unshiftable=0,translated_overlap_sum=0.)
    ordered,path = [],suite/(dataset+".episode_counts.jsonl")
    hasher = hashlib.sha256()
    with path.open("xb") as stream:
        created = os.fstat(stream.fileno())
        for episode,start in enumerate(range(0,len(rows),batch)):
            current = rows[start:start+batch]
            ids = [row["official_id"] for row in current]
            actual_ids = episode_metadata(parent,spec,inventory,forbidden,episode,raw_rows,coverage)
            require(all(v == ids for v in actual_ids.values()), "M02 source episode order differs")
            ordered.append(ids)
            gt_receipts = []
            for index,row in enumerate(current,start):
                content,digest = annotation_bytes(row)
                labels = decode_annotation(content,(224,224),classes,spec["vocabulary"]["reduce_zero_label"])
                key = row["label"]
                require(key not in label_hashes or label_hashes[key] == digest, "Repeated M02 GT bytes changed")
                label_hashes[key] = digest
                gt_receipts.append(dict(source_id=row["official_id"],label=key,sha256=digest))
                for a,mode in enumerate(MODES):
                    relative = f"{dataset}/predictions/{mode}/{index:06d}.npy"
                    raw,_ = bounded_content(canonical_child(parent,relative),inventory[relative],forbidden,2*1024**2)
                    counts[episode,a] += metric.counts(hard_array(raw,classes,(224,224)),labels,classes)
            line = (json.dumps(dict(episode=episode,ordered_ids=ids,labels=gt_receipts,I_U_by_mode=counts[episode].tolist()),allow_nan=False)+"\n").encode()
            require(shutil.disk_usage(suite).free >= RESERVE+len(line) and budget["bytes"]+len(line) <= OUTPUT_CAP,
                    "M02 score output budget exceeded")
            require(stream.write(line) == len(line), "Short M02 episode-count write")
            stream.flush()
            hasher.update(line)
            budget["bytes"] += len(line)
            write_json(suite/"queue_state.json",dict(phase="CPU_M02_score",dataset=dataset,
                completed_sources=min(start+batch,len(rows)),total_sources=len(rows),GT_opened=True,pid=os.getpid()),replace=True)
        stream.flush()
        os.fsync(stream.fileno())
        after,path_state = os.fstat(stream.fileno()),path.stat()
        require((created.st_dev,created.st_ino) == (after.st_dev,after.st_ino) == (path_state.st_dev,path_state.st_ino)
                and after.st_size == path_state.st_size, "M02 counts output identity changed")
    return counts,ordered,cost,coverage,path,hasher.hexdigest()


def execute(suite,bound):
    budget,labels,results,artifacts = dict(bytes=0),{},{},{}
    for spec in bound["source"]["specs"]:
        values,ordered,cost,coverage,path,digest = score_dataset(suite,bound,spec,budget,labels)
        result = analysis.paired_summary(values,spec["dataset"])
        result.update(images=spec["images"],ordered_source_episodes=ordered,cost=cost,
            preservation_coverage=coverage,source="L_local",resources=bound["seals"][spec["dataset"]]["resource_observation"])
        results[spec["dataset"]] = result
        artifacts[path.name] = digest
    for name,value in (("DATASET_RESULTS.json",results),("ANNOTATIONS_READ.json",labels)):
        encoded = (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
        require(shutil.disk_usage(suite).free >= RESERVE+len(encoded) and budget["bytes"]+len(encoded) <= OUTPUT_CAP,
                "M02 score summary exceeds prewrite budget")
        digest = protected.write_prepared_json(suite/name,value)
        budget["bytes"] += len(encoded)
        artifacts[name] = digest
    return dict(complete=True,datasets=len(results),cells=4*len(results),
        image_method_evaluations=sum(v["images"]*4 for v in results.values()),
        artifact_sha256=artifacts,GT_opened=True,new_GPU=False,output_bytes=budget["bytes"])


def run(suite,expected_sha):
    suite,prepared,inputs,audit = admission(suite,expected_sha)
    require(not any((suite/n).exists() for n in ("STARTED.json","FAILED.json","ANALYSIS_SEALED.json")), "No M02 score resume/retry")
    require(shutil.disk_usage(suite).free >= RESERVE+OUTPUT_CAP, "M02 score disk headroom insufficient")
    verify_audit_binding(audit)
    started_sha = protected.write_prepared_json(suite/"STARTED.json",dict(suite_prepared_sha256=expected_sha,audit=audit,
        pid=os.getpid(),GT_opened=False,automatic_retry=False,parent_barrier_sha256=inputs["barrier_sha256"]))
    begin = time.perf_counter()
    forbidden = prediction.annotation_inodes(inputs["source"])
    try:
        result = execute(suite,inputs)
        require(result["datasets"] == 2 and result["cells"] == 8
                and result["image_method_evaluations"] == sum(POPULATIONS[d]*4 for d in DATASETS), "Incomplete M02 scoring population")
        require(bind_parent() == inputs, "M02 prediction parent changed before final score seal")
        protected.guarded_inventory(suite/"runtime",prepared["runtime_hashes"],forbidden)
        for name,digest in (("SCORE_INPUTS.json",prepared["inputs_sha256"]),("PROTOCOL.json",prepared["protocol_sha256"]),
                            ("SUITE_PREPARED.json",expected_sha),("STARTED.json",started_sha)):
            protected.metadata(suite/name,digest,forbidden)
        prediction.general.runtime_imports(suite/"runtime")
        verify_audit_binding(audit)
        require(not torch.cuda.is_initialized(), "M02 CPU score initialized CUDA")
        for relative,digest in result["artifact_sha256"].items():
            bounded_content(canonical_child(suite,relative),digest,forbidden,16*1024**2)
        write_json(suite/"queue_state.json",dict(phase="M02_analysis_payloads_complete_awaiting_seal_and_exit",
            GT_opened=True,pid=os.getpid()),replace=True)
        publish_seal(suite/"ANALYSIS_SEALED.json",dict(**result,suite_prepared_sha256=expected_sha,
            parent_barrier_sha256=inputs["barrier_sha256"],prospective_analysis_sha256=ANALYSIS_SHA,
            audit_binding=audit,started_sha256=started_sha,elapsed_seconds=time.perf_counter()-begin))
    except BaseException as error:
        protected.write_prepared_json(suite/"FAILED.json",dict(error=f"{type(error).__name__}: {error}",automatic_retry=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","validate-admission","run"))
    parser.add_argument("--suite",required=True,type=Path)
    parser.add_argument("--suite-sha256")
    args = parser.parse_args()
    if args.phase == "prepare":
        print(dict(prepared_sha256=prepare(args.suite),GT_opened=False,GPU_started=False))
    else:
        require(args.suite_sha256,"Exact M02 scoring prepared SHA required")
        if args.phase == "validate-admission":
            admission(args.suite,args.suite_sha256)
            print(dict(admission_valid=True,CUDA_initialized=torch.cuda.is_initialized(),GT_opened=False,GPU_started=False))
        else:
            run(args.suite,args.suite_sha256)
