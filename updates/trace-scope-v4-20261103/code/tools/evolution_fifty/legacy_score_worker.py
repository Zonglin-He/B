"""M03-only CPU scorer, requiring a successful exact prediction parent.

The operator pins the future parent's actual PID/invocation and startup,
gate/global hashes only after verifying its retained successful exit. No model,
GPU launcher, adaptation, automatic retry or inference-time GT access exists.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

import torch

from . import generalization_score_worker as protected
from . import legacy_analysis as analysis
from . import legacy_score_components as loader
from . import legacy_score_payload as payload
from . import legacy_worker as prediction
from . import spatial_score_worker as shared
from .bootstrap import POPULATIONS,require
from .risk_score_payload import stable_bytes
from .risk_score_worker import canonical_child,publish_seal,verify_audit_binding

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/home/wwww/hzl/results/trace_mechanism_20261102")
PARENT = BASE/"M03_legacy_20261103_0017"
PARENT_SHA = "514b4b16531005b0ceea5c4fb9884387a637986f407e83bad0f9209708f7580b"
PARENT_UNIT = "trace-legacy-20261103-0017.service"
PARENT_ADMISSION_SHA = "a3d1901f106309c9e40785fef0f086b97f12ebc7101af4679aef27ccab64ef7e"
PARENT_REPORT_SHA = "5f59a96b503088d197f68d1ba86d098f0ce6bf827ab443d3d9cc4b1d27a0939b"
ANALYSIS = BASE/"M03_ANALYSIS_PROTOCOL_v1.json"
ANALYSIS_SHA = "d302adb2178d79188e2eced086ffc24965d1e2f0738467435f6950870dab4fc3"
MODES,DATASETS = analysis.MODES,prediction.DATASETS
RESERVE,OUTPUT_CAP = payload.RESERVE,payload.OUTPUT_CAP


def cpu_only():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "M03 CPU scoring requires hidden uninitialized CUDA")


def validate_pins(pins):
    require(type(pins) is dict and set(pins) == {"pid","invocation","start_sha256","gate_sha256","barrier_sha256"},
            "Exact operator M03 parent pins required")
    require(type(pins["pid"]) is int and pins["pid"] > 0, "M03 parent PID pin differs")
    for key,size in (("invocation",32),("start_sha256",64),("gate_sha256",64),("barrier_sha256",64)):
        require(type(pins[key]) is str and len(pins[key]) == size
                and all(c in "0123456789abcdef" for c in pins[key]), "M03 parent identity/hash pin differs")


def preliminary_terminal():
    keys = "LoadState,ActiveState,SubState,MainPID,ExecMainPID,ExecMainCode,ExecMainStatus,Result,InvocationID"
    result = subprocess.run(["systemctl","--user","show",PARENT_UNIT,"--property="+keys],
        check=True,text=True,capture_output=True,timeout=30)
    actual = dict(line.split("=",1) for line in result.stdout.splitlines() if "=" in line)
    expected = dict(LoadState="loaded",ActiveState="active",SubState="exited",MainPID="0",
        ExecMainCode="1",ExecMainStatus="0",Result="success")
    pid,invocation = actual.get("ExecMainPID",""),actual.get("InvocationID","")
    require(all(actual.get(k) == v for k,v in expected.items()) and pid.isdecimal() and int(pid) > 0
            and len(invocation) == 32 and all(c in "0123456789abcdef" for c in invocation),
            "M03 prediction service has not successfully exited")
    return {k:actual[k] for k in (*expected,"ExecMainPID","InvocationID")}


def parent_terminal(pins):
    validate_pins(pins)
    actual = preliminary_terminal()
    require(actual["ExecMainPID"] == str(pins["pid"]) and actual["InvocationID"] == pins["invocation"],
            "Exact M03 prediction parent has not successfully exited")
    return actual


def source_exclusions(source,*,archive=True,include_sources=True):
    payload.preflight_annotations(source)
    forbidden = payload.bounded.data_exclusions(source)
    data = set(forbidden)
    for root,snapshot in ((ROOT,True),(PARENT/"runtime",False)):
        members = protected.source_members(root,forbidden,snapshot_source=snapshot)
        forbidden.update((s[0],s[1]) for s in members.values())
    if archive:
        path = prediction.legacy_rho.ARCHIVE
        current = path.lstat()
        require(path.is_absolute() and path.resolve(strict=True) == path and stat.S_ISREG(current.st_mode)
                and current.st_nlink == 1 and 0 < current.st_size <= 1024**2
                and (current.st_dev,current.st_ino) not in forbidden, "M03 archive source path/role differs")
        forbidden.add((current.st_dev,current.st_ino))
        data.add((current.st_dev,current.st_ino))
    return forbidden if include_sources else data


def archive_components(source,spec,expected_identity=None):
    forbidden = source_exclusions(source,archive=False)
    path = prediction.legacy_rho.ARCHIVE
    state = path.lstat()
    require(spec["path"] == str(path) and path.is_absolute() and path.resolve(strict=True) == path
            and stat.S_ISREG(state.st_mode) and state.st_nlink == 1
            and (state.st_dev,state.st_ino) not in forbidden and 0 < state.st_size <= 1024**2,
            "M03 archive source aliases protected data/source or differs")
    identity = [state.st_dev,state.st_ino,state.st_size,state.st_mtime_ns]
    require(expected_identity is None or identity == expected_identity, "M03 archive file identity changed")
    raw,_ = stable_bytes(path,1024**2,prediction.legacy_rho.ARCHIVE_SHA,expected_identity=tuple(identity))
    return loader.load_components_bytes(raw,spec),identity


def reconstructed_source(base,archive_spec):
    source = copy.deepcopy(base)
    for spec in source["specs"]:
        spec.update(id="M03."+spec["dataset"],modes=list(MODES))
    source["prospective_analysis_sha256"] = ANALYSIS_SHA
    source["legacy_archive"] = copy.deepcopy(archive_spec)
    prediction.validate_inputs(source)
    return source


def bind_parent(pins):
    terminal = parent_terminal(pins)
    # The existing native source bootstrap pins G metadata before first read;
    # do not use legacy_worker.source_inputs(), which opens an archive pathname.
    base = prediction.source.source_inputs()
    forbidden = source_exclusions(base)
    require(PARENT.is_absolute() and PARENT.resolve(strict=True) == PARENT
            and not any((PARENT/n).exists() for n in ("FAILED.json","CANCELLED.json")), "Unsuccessful M03 prediction parent")
    prepared,_ = protected.metadata(PARENT/"SUITE_PREPARED.json",PARENT_SHA,forbidden)
    inputs,_ = protected.metadata(PARENT/"INPUTS.json",prepared["inputs_sha256"],forbidden)
    source = reconstructed_source(base,inputs["legacy_archive"])
    require(prepared["schema_version"] == 1 and prepared["root"] == str(PARENT) and source == inputs,
            "M03 original source input identity differs")
    data_forbidden = source_exclusions(base,include_sources=False)
    shared.native_binding(prepared["parent_native_hashes"],data_forbidden)
    protected.metadata(PARENT/"PROTOCOL.json",prepared["protocol_sha256"],forbidden)
    protected.guarded_inventory(PARENT/"runtime",prepared["runtime_hashes"],data_forbidden)
    barrier,_ = protected.metadata(PARENT/"ALL_PREDICTIONS_SEALED.json",pins["barrier_sha256"],forbidden)
    require(barrier["complete"] is True and barrier["GT_opened"] is False and barrier["suite_prepared_sha256"] == PARENT_SHA
            and barrier["cells"] == 8 and barrier["image_method_evaluations"] == sum(POPULATIONS[d]*4 for d in DATASETS)
            and set(barrier["dataset_seals"]) == set(DATASETS) and barrier["started_sha256"] == pins["start_sha256"]
            and barrier["gate_binding"]["barrier_sha256"] == pins["gate_sha256"]
            and barrier["legacy_archive"] == inputs["legacy_archive"], "M03 complete prediction barrier differs")
    audit = barrier["audit_binding"]
    require(audit["admission_path"] == str(PARENT/"ADMITTED.json") and audit["report_path"] == str(PARENT/"INDEPENDENT_REPORT.md")
            and audit["admission_sha256"] == PARENT_ADMISSION_SHA and audit["report_sha256"] == PARENT_REPORT_SHA,
            "Wrong M03 prediction audit binding")
    admitted,_ = protected.metadata(PARENT/"ADMITTED.json",PARENT_ADMISSION_SHA,forbidden)
    shared.bounded_content(PARENT/"INDEPENDENT_REPORT.md",PARENT_REPORT_SHA,forbidden,4*1024**2)
    verify_audit_binding(audit)
    start,_ = protected.metadata(PARENT/"STARTED.json",pins["start_sha256"],forbidden)
    require(admitted["decision"] == "ADMITTED" and admitted["suite_prepared_sha256"] == PARENT_SHA
            and admitted["M03_predictions_allowed"] is True and admitted["GT_scoring_allowed"] is False
            and start["suite_prepared_sha256"] == PARENT_SHA and type(start["pid"]) is int
            and start["pid"] == pins["pid"] and start["audit"] == audit and start["GT_opened"] is False,
            "M03 prediction permission/startup differs")
    prediction.source.verify_gates(PARENT,PARENT_SHA,barrier["gate_binding"],forbidden)
    seals = {}
    for spec in inputs["specs"]:
        d,n = spec["dataset"],spec["images"]
        seal,_ = protected.metadata(PARENT/d/"PREDICTIONS_SEALED.json",barrier["dataset_seals"][d],forbidden)
        require(seal["complete"] is True and seal["GT_opened"] is False and seal["dataset"] == d
                and seal["suite_prepared_sha256"] == PARENT_SHA and seal["images"] == n == POPULATIONS[d]
                and seal["modes"] == list(MODES) and seal["image_method_evaluations"] == n*4,
                "M03 full dataset population differs")
        seals[d] = seal
    payload_metadata = payload.preflight_payloads(PARENT,inputs["specs"],seals,inputs)
    components,archive_identity = archive_components(inputs,inputs["legacy_archive"])
    for spec in inputs["specs"]:
        payload.cost_binding(PARENT,spec,seals[spec["dataset"]]["artifact_sha256"],forbidden,payload_metadata,components)
    protocol,_ = protected.metadata(ANALYSIS,ANALYSIS_SHA,forbidden)
    expected_contrasts = {row["name"]:row["coefficients"] for row in [protocol["primary_contrast"],*protocol["secondary_contrasts"]]}
    require(protocol["datasets"] == {d:POPULATIONS[d] for d in DATASETS} and protocol["conditions"] == list(MODES)
            and protocol["full_cells"] == 8 and protocol["image_method_evaluations"] == sum(POPULATIONS[d]*4 for d in DATASETS)
            and expected_contrasts == {k:{m:v for m,v in zip(MODES,weights) if v} for k,weights in analysis.CONTRASTS.items()}
            and protocol["primary_contrast"]["name"] == analysis.PRIMARY
            and protocol["bootstrap"]["replicates"] == 10000 and protocol["bootstrap"]["root_seed"] == 20261102,
            "M03 prospective analysis protocol differs")
    return dict(parent=str(PARENT),prepared_sha256=PARENT_SHA,parent_pins=copy.deepcopy(pins),terminal=terminal,
        source=inputs,seals=seals,payload_metadata=payload_metadata,archive_identity=archive_identity,analysis_sha256=ANALYSIS_SHA)


def prepare(output,pins):
    cpu_only()
    parent_terminal(pins)
    bound = bind_parent(pins)
    output = Path(output).absolute()
    require(output.resolve() == output and not output.exists(), "Fresh canonical M03 scoring suite required")
    output.mkdir()
    inputs_sha = protected.write_prepared_json(output/"SCORE_INPUTS.json",bound)
    protocol_sha = protected.write_prepared_json(output/"PROTOCOL.json",dict(experiment="M03_offline_CPU_score",
        parent_prepared_sha256=PARENT_SHA,parent_pins=pins,prospective_analysis_sha256=ANALYSIS_SHA,new_GPU=False,
        boundary="All8 full predictions and exact retained successful exit, new independent CPU_GT AND M03_GT permission",
        annotation_geometry="Native indexed PIL NEAREST224, not G OpenCV nearest",
        estimates="C.image-legacy.image primary;5fixedcontrasts/pooledI-U/10000wholeoriginalepisodePCG64/pointwise95/undefinednullretained",
        cost="All raw episodes including0, role-dependent legacy views, not equal compute or whole historical reproduction",
        failures="No resume/retry, protected exclusive progress/outputs, final atomic global seal last",
        limits=dict(output_cap_bytes=OUTPUT_CAP,shared_disk_reserve_bytes=RESERVE,GPU=False)))
    inventory = protected.guarded_snapshot(ROOT,output/"runtime",source_exclusions(bound["source"],include_sources=False))
    return protected.write_prepared_json(output/"SUITE_PREPARED.json",dict(schema_version=1,root=str(output),
        runtime_hashes=inventory,inputs_sha256=inputs_sha,protocol_sha256=protocol_sha,parent_prepared_sha256=PARENT_SHA,
        parent_pins=pins,GT_opened=False,CPU_scoring_started=False))


def admission(suite,expected_sha):
    cpu_only()
    preliminary_terminal()
    forbidden = source_exclusions(prediction.source.source_inputs())
    suite = Path(suite).resolve(strict=True)
    prepared,_ = protected.metadata(suite/"SUITE_PREPARED.json",expected_sha,forbidden)
    require(prepared["schema_version"] == 1 and prepared["root"] == str(suite) and prepared["parent_prepared_sha256"] == PARENT_SHA
            and Path(__file__).resolve().is_relative_to(suite/"runtime"), "Wrong exact M03 scoring runtime")
    bound = bind_parent(prepared["parent_pins"])
    protected.guarded_inventory(suite/"runtime",prepared["runtime_hashes"],source_exclusions(bound["source"],include_sources=False))
    prediction.general.runtime_imports(suite/"runtime")
    inputs,_ = protected.metadata(suite/"SCORE_INPUTS.json",prepared["inputs_sha256"],forbidden)
    protocol,_ = protected.metadata(suite/"PROTOCOL.json",prepared["protocol_sha256"],forbidden)
    require(inputs == bound and protocol["experiment"] == "M03_offline_CPU_score"
            and protocol["parent_prepared_sha256"] == PARENT_SHA and protocol["parent_pins"] == prepared["parent_pins"]
            and protocol["prospective_analysis_sha256"] == ANALYSIS_SHA and protocol["new_GPU"] is False,
            "M03 parent score binding/protocol drift")
    audit,audit_sha = protected.metadata(suite/"ADMITTED.json",forbidden=forbidden,limit=4*1024**2)
    require(audit["decision"] == "ADMITTED" and audit["suite_prepared_sha256"] == expected_sha
            and audit.get("independent_reviewer") and audit.get("CPU_GT_scoring_allowed") is True
            and audit.get("M03_GT_scoring_allowed") is True, "New exact independent M03 CPU_GT admission required")
    report = suite/"INDEPENDENT_REPORT.md"
    require(audit["report_path"] == str(report), "Wrong M03 score audit report path")
    shared.bounded_content(report,audit["report_sha256"],forbidden,4*1024**2)
    identities = []
    for path in (suite/"ADMITTED.json",report):
        st = path.stat()
        require((st.st_dev,st.st_ino) not in forbidden, "M03 score audit/report aliases protected input")
        identities.append([st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns])
    binding = dict(admission_path=str(suite/"ADMITTED.json"),admission_sha256=audit_sha,admission_identity=identities[0],
        report_path=str(report),report_sha256=audit["report_sha256"],report_identity=identities[1],independent_reviewer=audit["independent_reviewer"])
    verify_audit_binding(binding)
    return suite,prepared,inputs,binding


def execute(suite,bound):
    require(payload.preflight_payloads(Path(bound["parent"]),bound["source"]["specs"],bound["seals"],bound["source"])
            == bound["payload_metadata"], "M03 complete payload graph changed before first GT")
    components,_ = archive_components(bound["source"],bound["source"]["legacy_archive"],bound["archive_identity"])
    budget,labels,results,artifacts = dict(bytes=0),{},{},{}
    for spec in bound["source"]["specs"]:
        values,ordered,cost,coverage,path,digest = payload.score_dataset(suite,bound,spec,budget,labels,
            lambda row:shared.write_json(suite/"queue_state.json",row,replace=True),components)
        result = analysis.paired_summary(values,spec["dataset"])
        result.update(images=spec["images"],ordered_source_episodes=ordered,cost=cost,candidate_coverage=coverage,
            source="L_local",resources=bound["seals"][spec["dataset"]]["resource_observation"])
        results[spec["dataset"]] = result
        artifacts[path.name] = digest
    for name,value in (("DATASET_RESULTS.json",results),("ANNOTATIONS_READ.json",labels)):
        encoded = (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
        require(shutil.disk_usage(suite).free >= RESERVE+len(encoded) and budget["bytes"]+len(encoded) <= OUTPUT_CAP,
                "M03 score summary exceeds prewrite budget")
        artifacts[name] = protected.write_prepared_json(suite/name,value)
        budget["bytes"] += len(encoded)
    return dict(complete=True,datasets=len(results),cells=4*len(results),GT_opened=True,new_GPU=False,
        image_method_evaluations=sum(v["images"]*4 for v in results.values()),artifact_sha256=artifacts,output_bytes=budget["bytes"])


def run(suite,expected_sha):
    suite,prepared,inputs,audit = admission(suite,expected_sha)
    require(not any((suite/n).exists() for n in ("STARTED.json","FAILED.json","ANALYSIS_SEALED.json")), "No M03 score resume/retry")
    require(shutil.disk_usage(suite).free >= RESERVE+OUTPUT_CAP, "M03 score disk headroom insufficient")
    verify_audit_binding(audit)
    started = dict(suite_prepared_sha256=expected_sha,audit=audit,pid=os.getpid(),GT_opened=False,
                   automatic_retry=False,parent_pins=inputs["parent_pins"])
    content = (json.dumps(started,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    require(len(content) <= 64*1024, "M03 STARTED metadata exceeds bound")
    owned = False
    try:
        # Own failures after exclusive creation, but never claim another run's
        # marker if a competing creator won the pathname before this open.
        path = suite/"STARTED.json"
        with path.open("xb") as stream:
            owned = True
            first = os.fstat(stream.fileno())
            require(stat.S_ISREG(first.st_mode) and first.st_nlink == 1, "M03 STARTED inode differs")
            require(stream.write(content) == len(content), "Short M03 STARTED write")
            stream.flush()
            os.fsync(stream.fileno())
            state,actual = os.fstat(stream.fileno()),path.lstat()
            identity = lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink,s.st_mode)
            require(path.resolve(strict=True) == path and identity(state) == identity(actual)
                    and (state.st_dev,state.st_ino) == (first.st_dev,first.st_ino)
                    and state.st_nlink == 1 and state.st_size == len(content), "M03 STARTED pathname changed")
        started_sha = hashlib.sha256(content).hexdigest()
        begin = time.perf_counter()
        forbidden = source_exclusions(inputs["source"])
        protected.metadata(path,started_sha,forbidden,limit=64*1024)
        result = execute(suite,inputs)
        require(result["datasets"] == 2 and result["cells"] == 8
                and result["image_method_evaluations"] == sum(POPULATIONS[d]*4 for d in DATASETS), "Incomplete M03 scoring population")
        require(bind_parent(inputs["parent_pins"]) == inputs, "M03 prediction parent drift before final seal")
        protected.guarded_inventory(suite/"runtime",prepared["runtime_hashes"],source_exclusions(inputs["source"],include_sources=False))
        for name,digest in (("SCORE_INPUTS.json",prepared["inputs_sha256"]),("PROTOCOL.json",prepared["protocol_sha256"]),
                            ("SUITE_PREPARED.json",expected_sha),("STARTED.json",started_sha)):
            protected.metadata(suite/name,digest,forbidden)
        prediction.general.runtime_imports(suite/"runtime")
        verify_audit_binding(audit)
        require(not torch.cuda.is_initialized(), "M03 CPU scoring initialized CUDA")
        for relative,digest in result["artifact_sha256"].items():
            shared.bounded_content(canonical_child(suite,relative),digest,forbidden,16*1024**2)
        shared.write_json(suite/"queue_state.json",dict(phase="M03_analysis_payloads_complete_awaiting_seal_and_exit",
            GT_opened=True,pid=os.getpid()),replace=True)
        publish_seal(suite/"ANALYSIS_SEALED.json",dict(**result,suite_prepared_sha256=expected_sha,
            parent_pins=inputs["parent_pins"],prospective_analysis_sha256=ANALYSIS_SHA,
            audit_binding=audit,started_sha256=started_sha,elapsed_seconds=time.perf_counter()-begin))
    except BaseException as error:
        if owned:
            protected.write_prepared_json(suite/"FAILED.json",dict(error=f"{type(error).__name__}: {error}",automatic_retry=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","validate-admission","run"))
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--suite-sha256")
    parser.add_argument("--parent-pid",type=int)
    parser.add_argument("--parent-invocation")
    parser.add_argument("--parent-start-sha256")
    parser.add_argument("--parent-gate-sha256")
    parser.add_argument("--parent-barrier-sha256")
    args = parser.parse_args()
    if args.phase == "prepare":
        pins = dict(pid=args.parent_pid,invocation=args.parent_invocation,start_sha256=args.parent_start_sha256,
            gate_sha256=args.parent_gate_sha256,barrier_sha256=args.parent_barrier_sha256)
        print(dict(prepared_sha256=prepare(args.suite,pins),GT_opened=False,GPU_started=False))
    else:
        require(args.suite_sha256,"Exact M03 score prepared SHA required")
        if args.phase == "validate-admission":
            admission(args.suite,args.suite_sha256)
            print(dict(admission_valid=True,CUDA_initialized=torch.cuda.is_initialized(),GT_opened=False,GPU_started=False))
        else:run(args.suite,args.suite_sha256)
