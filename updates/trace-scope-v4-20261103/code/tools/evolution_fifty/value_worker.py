"""V01-only freeze/admit/gate/predict controller. No GT scoring entry point.

Preparation is CPU-only and not execution permission. Execution is manual,
single-attempt, after exact G1954/M02/M04/M03 successful exits and new independent artifact
admission. Source-input reuse never reuses G's differently resized predictions.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

import torch

from tools.evolution import run_cell as shared
from tools.evolution_confirmation import run as confirmation
from tools.evolution_followup.standalone import gpu_lock
from tools.evolution_tuning.common import write_json
from . import generalization_runtime as native
from . import generalization_worker as general
from . import generalization_score_worker as protected
from . import value_gate
from . import value_prediction
from . import value_runtime
from . import spatial_worker as source
from .bootstrap import POPULATIONS, require
from .generalization import visual_layers
from .intake import sha256
from .risk_replay_worker import native_identity
from .risk_score_worker import publish_seal, verify_audit_binding
from .risk_worker import checked_checkpoint, deterministic_runtime, forbid_annotations

ROOT = Path(__file__).resolve().parents[2]
VALUE_PROTOCOL = Path("/home/wwww/hzl/results/trace_mechanism_20261102/VALUE_EXTENSION_PROTOCOL_v1.json")
VALUE_PROTOCOL_SHA = "44a2c1ab3f48a37a603c6ed6f1f31c6abd0985952ca04df729f4ca448d0cf95f"
DATASETS = ("coco_object", "pc60")


def source_inputs():
    inputs = source.source_inputs()
    forbidden = source.annotation_inodes(inputs)
    forbid_annotations(sorted(inputs["annotation_metadata"]))
    prospective, _ = protected.metadata(VALUE_PROTOCOL, VALUE_PROTOCOL_SHA, forbidden)
    require(prospective["V01"]["frozen_grid_conditions"] == 10
            and prospective["V01"]["image_method_evaluations_if_all_run"] == 50525,
            "Pinned V01 prospective value protocol differs")
    for spec in inputs["specs"]:
        spec["id"] = "V01."+spec["dataset"]
        spec["points"] = list(value_runtime.points(spec["dataset"]))
        spec.pop("modes")
    inputs.update(value_protocol_path=str(VALUE_PROTOCOL),value_protocol_sha256=VALUE_PROTOCOL_SHA,
                  prospective_value=prospective["V01"])
    validate_inputs(inputs["rows"],inputs["specs"])
    return inputs


def annotation_inodes(source):
    return {(s["device"],s["inode"]) for s in source["annotation_metadata"].values()}


def validate_inputs(rows,specs):
    require(set(rows) == set(DATASETS) and [s["dataset"] for s in specs] == list(DATASETS), "V01 dataset membership differs")
    forbidden = protected.label_inodes(rows)
    for spec in specs:
        dataset, cfg = spec["dataset"],spec["config"]
        native.image_worker.validate_config(cfg)
        require(type(spec["points"]) is list, "V01 serialized grid must be a list")
        for point in spec["points"]:
            value_runtime.validate_point(dataset,point)
        require(spec["id"] == "V01."+dataset and spec["points"] == list(value_runtime.points(dataset))
                and type(cfg["steps"]) is int and cfg["steps"] == value_runtime.NATIVE_STEPS[dataset]
                and spec["backbone"] == "ViT-L/14"
                and spec["model_type"] == "naclip" and spec["preprocessing"] == "native_PIL_bilinear"
                and spec["vision_outputs"] == visual_layers("ViT-L/14")
                and cfg["resize"] == cfg["patch_size"] == [224,224], "V01 actual model/config/geometry differs")
        require(spec["images"] == len(rows[dataset]) == POPULATIONS[dataset]
                and len({r["official_id"] for r in rows[dataset]}) == len(rows[dataset]), "V01 full population differs")
        for row in rows[dataset]:
            general.label_identity(row)
            path = Path(row["image"])
            require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "V01 RGB path is not canonical")
            state = path.stat()
            require((state.st_dev,state.st_ino) not in forbidden and 0 < state.st_size <= 32*1024**2,
                    "V01 RGB aliases GT or exceeds size bound")
    return forbidden


def prepare(output):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "CPU-only V01 preparation required")
    source = source_inputs()
    forbid_annotations(sorted(source["annotation_metadata"]))
    output = Path(output).absolute()
    require(not output.exists() and output.resolve() == output, "Fresh canonical V01 suite required")
    output.mkdir()
    inputs_sha = protected.write_prepared_json(output/"INPUTS.json",source)
    protocol_sha = protected.write_prepared_json(output/"PROTOCOL.json",dict(experiment="V01_fixed_compute_grid",
        datasets=list(DATASETS),points={d:list(value_runtime.points(d)) for d in DATASETS},
        images=10105,image_method_evaluations=50525,seed=0,
        value_protocol_sha256=VALUE_PROTOCOL_SHA,prospective_value=source["prospective_value"],
        algorithm="Unchanged production C/Confidence; only the ten predeclared integer update budgets vary",
        source="Exact clean input IDs/config/vocabulary; native PILbilinear RGB decoding, no differently resized G prediction reuse",
        gates="Both datasets first4RGB and all5points, every actual step finite loss/gradient/parameter/Adam/final/support <=1e-5, zerochangedlabels",
        reference="Explicit masked mean sharing original candidate/probe; not independent whole algorithm",
        reset="Native theta0/Adam/RNG restored between every point; identical source/candidate identities across budgets",
        timing="Plain run_point full core: synchronized native reset insideadapt to final readout plus common finite/Adam observer; decode/H2D/IO/GT/preflight/finalcleanup excluded",
        auxiliary="Gate-only reset snapshots/step receipts are NOT eligible formal accuracy-time observations; no added full-run checkpoint forwards",
        selection="No time/score selection in prediction. Future offline matching uses nearest abs logtime, tie fewer steps; near-time only ratio[0.9,1.1], no interpolated mIoU",
        barriers="All10full predictions plus actual successful retained exit before new independent CPU_GT scoring permission",
        resources=dict(single_shared_GPU=True,RAM_bytes=12*1024**3,swap_bytes=1024**3,
            shared_disk_reserve_bytes=general.RESERVE,prediction_artifact_bytes=general.MAX_PAYLOAD_BYTES,
            OOMPolicy="stop",Restart="no",RemainAfterExit=True),
        launch="Manual once only, after exact G1954/M02/M04/M03 successful retained exits and complete prediction seals",
        failures="Failclosed; discard method; no resume/automatic retry; retain all partial/negative/null results"))

    runtime = protected.guarded_snapshot(ROOT,output/"runtime",annotation_inodes(source))
    return protected.write_prepared_json(output/"SUITE_PREPARED.json",dict(schema_version=1,root=str(output),
        runtime_hashes=runtime,inputs_sha256=inputs_sha,protocol_sha256=protocol_sha,
        parent_native_hashes=native_identity(),GT_opened=False,GPU_started=False))


def admission(suite,expected_sha):
    require(not torch.cuda.is_initialized(), "V01 admission must precede CUDA initialization")
    source = source_inputs()
    validate_inputs(source["rows"],source["specs"])
    forbidden = annotation_inodes(source)
    forbid_annotations(sorted(source["annotation_metadata"]))
    suite = Path(suite).resolve(strict=True)
    prepared, _ = protected.metadata(suite/"SUITE_PREPARED.json",expected_sha,forbidden)
    require(prepared["root"] == str(suite) and prepared["schema_version"] == 1
            and Path(__file__).resolve().is_relative_to(suite/"runtime"), "Wrong V01 frozen runtime")
    protected.guarded_inventory(suite/"runtime",prepared["runtime_hashes"],forbidden)
    general.runtime_imports(suite/"runtime")
    inputs, _ = protected.metadata(suite/"INPUTS.json",prepared["inputs_sha256"],forbidden)
    protected.metadata(suite/"PROTOCOL.json",prepared["protocol_sha256"],forbidden)
    require(inputs == source and native_identity() == prepared["parent_native_hashes"], "V01 source/native metadata drift")
    audit, audit_sha = protected.metadata(suite/"ADMITTED.json",forbidden=forbidden,limit=4*1024**2)
    require(audit["decision"] == "ADMITTED" and audit["suite_prepared_sha256"] == expected_sha
            and audit.get("independent_reviewer") and audit.get("V01_predictions_allowed") is True
            and audit.get("GT_scoring_allowed") is False, "New independent exact V01 prediction admission required")
    report = suite/"INDEPENDENT_REPORT.md"
    require(audit["report_path"] == str(report), "V01 independent report path differs")
    identities = []
    for path in (suite/"ADMITTED.json",report):
        state = path.stat()
        require((state.st_dev,state.st_ino) not in forbidden, "V01 admission/report aliases GT")
        identities.append([state.st_dev,state.st_ino,state.st_size,state.st_mtime_ns])
    binding = dict(admission_path=str(suite/"ADMITTED.json"),admission_sha256=audit_sha,admission_identity=identities[0],
        report_path=str(report),report_sha256=audit["report_sha256"],report_identity=identities[1],
        independent_reviewer=audit["independent_reviewer"])
    verify_audit_binding(binding)
    for spec in inputs["specs"]:
        checked_checkpoint(spec["checkpoint"])
    return suite,prepared,inputs,binding


def build_method(spec):
    method = native.build_method(spec)
    method.model.eval()
    require(type(method) is native.ExperimentTRACE, "Unchanged V01 production method required")
    native.legal_parameters(method)
    return method


def gates(suite,expected_sha,inputs):
    children = {}
    (suite/"gates").mkdir()
    for spec in inputs["specs"]:
        cfg,rows = spec["config"],inputs["rows"][spec["dataset"]][:4]
        with gpu_lock() as lock:
            deterministic_runtime()
            method = None
            try:
                method = build_method(spec)
                data = confirmation.make_data(cfg,rows)
                reports = []
                for start in range(0,len(data),cfg["batch_size"]):
                    images,meta,valid,ids = general.episode_batch(data,start,min(start+cfg["batch_size"],len(data)),method,cfg)
                    reports.append(value_gate.compare_gate(method,spec["dataset"],images,valid,ids,meta,cfg,spec["vocabulary"]["query_to_class"]))
                require(len(data) == 4 and bool(reports), "Both V01 datasets need4 actual RGB gate sources")
                record = dict(passed=True,dataset=spec["dataset"],suite_prepared_sha256=expected_sha,
                    actual_RGB_sources=4,ordered_ids=[r["official_id"] for r in rows],comparisons=reports,
                    gpu_lock=lock,GT_opened=False)
            finally:
                del method
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
        path = suite/"gates"/(spec["dataset"]+".json")
        publish_seal(path,record)
        children[spec["dataset"]] = sha256(path)
    path = suite/"ALL_GATES_PASSED.json"
    publish_seal(path,dict(complete=True,suite_prepared_sha256=expected_sha,datasets=children,GT_opened=False))
    return dict(barrier_sha256=sha256(path),datasets=children)


def verify_gates(suite,expected_sha,binding,forbidden):
    barrier, _ = protected.metadata(suite/"ALL_GATES_PASSED.json",binding["barrier_sha256"],forbidden)
    require(barrier["complete"] is True and barrier["GT_opened"] is False
            and barrier["suite_prepared_sha256"] == expected_sha and barrier["datasets"] == binding["datasets"]
            and set(barrier["datasets"]) == set(DATASETS), "Both V01 gates required")
    for dataset,digest in barrier["datasets"].items():
        record, _ = protected.metadata(suite/"gates"/(dataset+".json"),digest,forbidden)
        require(record["passed"] is True and record["dataset"] == dataset
                and record["actual_RGB_sources"] == 4 and record["GT_opened"] is False, "Invalid V01 gate child")


PRIOR_SUITES = (
    ("M02_spatial_20261102_2156","f3eaa8494d4bdd07224b0c640af13985a92dd5ea3391761d13a91c1459ef2e69",
     "trace-spatial-20261102-2156.service",8,40420),
    ("M04_factorial_20261102_2243","02d9e169ae5bf5f7fd723f896ae5cb208c2278a1c85d7a2bd8f04b83d907f201",
     "trace-factorial-20261102-2243.service",16,80840),
    ("M03_legacy_20261103_0017","514b4b16531005b0ceea5c4fb9884387a637986f407e83bad0f9209708f7580b",
     "trace-legacy-20261103-0017.service",8,40420),
)
PRIOR_ROOT = Path("/home/wwww/hzl/results/trace_mechanism_20261102")


def prior_GPU_terminal():
    previous = source.prior_GPU_terminal()
    inputs = source_inputs()
    forbidden = annotation_inodes(inputs)
    G_start,G_start_sha = protected.metadata(protected.PARENT/"STARTED.json",forbidden=forbidden)
    G_barrier,_ = protected.metadata(protected.PARENT/"ALL_G_PREDICTIONS_SEALED.json",
        previous["barrier_sha256"],forbidden)
    require(G_start["suite_prepared_sha256"] == protected.PARENT_SHA and G_start["GT_opened"] is False
            and type(G_start["pid"]) is int and str(G_start["pid"]) == previous["terminal"]["ExecMainPID"]
            and G_barrier["started_sha256"] == G_start_sha,
            "V01 G STARTED identity or global receipt binding differs")
    previous = dict(**previous,started_sha256=G_start_sha)
    result = {}
    keys = ("LoadState","ActiveState","SubState","MainPID","ExecMainPID","ExecMainCode",
            "ExecMainStatus","Result","InvocationID")
    for name,digest,unit,cells,evaluations in PRIOR_SUITES:
        status = subprocess.run(["systemctl","--user","show",unit,"--property="+",".join(keys)],
            check=True,text=True,capture_output=True,timeout=30)
        actual = dict(line.split("=",1) for line in status.stdout.splitlines() if "=" in line)
        expected = dict(LoadState="loaded",ActiveState="active",SubState="exited",MainPID="0",
                        ExecMainCode="1",ExecMainStatus="0",Result="success")
        require(all(actual.get(k) == v for k,v in expected.items())
                and len(actual.get("InvocationID","")) == 32, "V01 prior named GPU has not successfully exited")
        parent = PRIOR_ROOT/name
        prepared,_ = protected.metadata(parent/"SUITE_PREPARED.json",digest,forbidden)
        require(prepared["root"] == str(parent) and not (parent/"FAILED.json").exists(),
                "V01 prior prepared root or failure differs")
        start,start_sha = protected.metadata(parent/"STARTED.json",forbidden=forbidden)
        require(start["suite_prepared_sha256"] == digest and start["GT_opened"] is False
                and type(start["pid"]) is int and str(start["pid"]) == actual["ExecMainPID"],
                "V01 prior service and STARTED identities differ")
        barrier,barrier_sha = protected.metadata(parent/"ALL_PREDICTIONS_SEALED.json",forbidden=forbidden)
        require(barrier["complete"] is True and barrier["GT_opened"] is False
                and barrier["suite_prepared_sha256"] == digest and barrier["started_sha256"] == start_sha
                and barrier["cells"] == cells and barrier["image_method_evaluations"] == evaluations
                and set(barrier["dataset_seals"]) == set(DATASETS), "V01 prior complete prediction barrier differs")
        for dataset,child_sha in barrier["dataset_seals"].items():
            child,_ = protected.metadata(parent/dataset/"PREDICTIONS_SEALED.json",child_sha,forbidden)
            require(child["complete"] is True and child["GT_opened"] is False and child["dataset"] == dataset
                    and child["suite_prepared_sha256"] == digest and child["images"] == POPULATIONS[dataset]
                    and child["image_method_evaluations"] == POPULATIONS[dataset]*(cells//2),
                    "V01 prior full-population child differs")
        result[name] = dict(terminal=actual,barrier_sha256=barrier_sha,started_sha256=start_sha)
    return dict(G_completion=previous,named_completed_workers=result)


def run(suite,expected_sha):
    prior = prior_GPU_terminal()
    suite,prepared,inputs,audit = admission(suite,expected_sha)
    require(not any((suite/n).exists() for n in ("STARTED.json","FAILED.json","ALL_PREDICTIONS_SEALED.json")), "No V01 resume/retry")
    require(shutil.disk_usage(suite).free >= general.RESERVE+general.MAX_PAYLOAD_BYTES, "V01 shared disk headroom insufficient")
    verify_audit_binding(audit)
    protected.write_prepared_json(suite/"STARTED.json",dict(suite_prepared_sha256=expected_sha,audit=audit,
        prior_GPU=prior,pid=os.getpid(),GT_opened=False,automatic_retry=False))
    started_sha = sha256(suite/"STARTED.json")
    forbidden = annotation_inodes(inputs)
    try:
        gate_binding = gates(suite,expected_sha,inputs)
        verify_gates(suite,expected_sha,gate_binding,forbidden)
        budget,seals = dict(bytes=0),{}
        for spec in inputs["specs"]:
            with gpu_lock() as lock:
                deterministic_runtime()
                method = None
                try:
                    method = build_method(spec)
                    if method.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(method.device)
                    data = confirmation.make_data(spec["config"],inputs["rows"][spec["dataset"]])
                    begin = time.perf_counter()
                    result = value_prediction.execute_dataset(method,data,inputs["rows"][spec["dataset"]],spec,suite,budget,
                        lambda row:write_json(suite/"queue_state.json",dict(phase="V01_paired_full_prediction",pid=os.getpid(),**row),replace=True))
                    require(result["payloads_complete"] is True and result["GT_opened"] is False
                            and result["dataset"] == spec["dataset"] and result["images"] == spec["images"]
                            and result["image_method_evaluations"] == spec["images"]*5
                            and result["points"] == list(value_runtime.points(spec["dataset"])), "Incomplete V01 dataset result")
                    result.update(complete=True,suite_prepared_sha256=expected_sha,gpu_lock=lock,
                        resource_observation=dict(dataset_model_and_IO_wall_seconds=time.perf_counter()-begin,
                            process_peak_RSS_bytes_cumulative=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                            GPU_peak_allocated_bytes=None if method.device.type != "cuda" else torch.cuda.max_memory_allocated(method.device)))
                finally:
                    del method
                    if torch.cuda.is_initialized():
                        torch.cuda.empty_cache()
            path = suite/spec["dataset"]/"PREDICTIONS_SEALED.json"
            publish_seal(path,result)
            seals[spec["dataset"]] = sha256(path)
        require(set(seals) == set(DATASETS) and source_inputs() == inputs
                and native_identity() == prepared["parent_native_hashes"], "V01 source/native drift or missing dataset")
        protected.guarded_inventory(suite/"runtime",prepared["runtime_hashes"],forbidden)
        protected.metadata(suite/"SUITE_PREPARED.json",expected_sha,forbidden)
        protected.metadata(suite/"INPUTS.json",prepared["inputs_sha256"],forbidden)
        protected.metadata(suite/"PROTOCOL.json",prepared["protocol_sha256"],forbidden)
        general.runtime_imports(suite/"runtime")
        verify_audit_binding(audit)
        require(prior_GPU_terminal() == prior, "V01 prior completion bindings changed")
        require(sha256(suite/"STARTED.json") == started_sha, "V01 start receipt changed")
        verify_gates(suite,expected_sha,gate_binding,forbidden)
        for dataset,digest in seals.items():
            protected.metadata(suite/dataset/"PREDICTIONS_SEALED.json",digest,forbidden)
        write_json(suite/"queue_state.json",dict(phase="V01_payloads_complete_awaiting_global_seal_and_exit",
            datasets_complete=2,GT_opened=False,pid=os.getpid()),replace=True)
        publish_seal(suite/"ALL_PREDICTIONS_SEALED.json",dict(complete=True,suite_prepared_sha256=expected_sha,
            dataset_seals=seals,cells=len(inputs["specs"])*5,
            image_method_evaluations=sum(s["images"]*5 for s in inputs["specs"]),artifact_bytes=budget["bytes"],
            gate_binding=gate_binding,audit_binding=audit,started_sha256=started_sha,GT_opened=False,scientific_completion=False))
    except BaseException as error:
        protected.write_prepared_json(suite/"FAILED.json",dict(error=f"{type(error).__name__}: {error}",
            GT_opened=False,automatic_retry=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","validate-admission","run"))
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--suite-sha256")
    args = parser.parse_args()
    if args.phase == "prepare":
        print(dict(prepared_sha256=prepare(args.suite),GPU_started=False,GT_opened=False))
    else:
        require(args.suite_sha256,"Exact V01 prepared SHA required")
        if args.phase == "validate-admission":
            admission(args.suite,args.suite_sha256)
            print(dict(admission_valid=True,CUDA_initialized=torch.cuda.is_initialized(),GPU_started=False,GT_opened=False))
        else:
            run(args.suite,args.suite_sha256)

