"""M04 freeze/admit/gate/full prediction controller; no GT entry point.

Manual single-attempt execution requires G1954 true success, other named GPU
work idle, new exact-artifact admission and externally enforced service caps.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

import torch

from tools.evolution_confirmation import run as confirmation
from tools.evolution_followup.standalone import gpu_lock
from tools.evolution_tuning.common import write_json
from . import factorial_gate
from . import factorial_prediction
from . import factorial_runtime
from . import generalization_runtime as native
from . import generalization_worker as general
from . import generalization_score_worker as protected
from . import spatial_worker as source_worker
from .bootstrap import require
from .intake import sha256
from .risk_replay_worker import native_identity
from .risk_score_worker import publish_seal,verify_audit_binding
from .risk_worker import checked_checkpoint,deterministic_runtime,forbid_annotations

ROOT = Path(__file__).resolve().parents[2]
DATASETS = source_worker.DATASETS


def source_inputs():
    source = source_worker.source_inputs()
    for spec in source["specs"]:
        spec["id"] = "M04."+spec["dataset"]
        spec.pop("modes")
        spec.update(studies=list(factorial_prediction.STUDIES),factorial_arms=list(factorial_runtime.ARMS))
    return source


annotation_inodes = source_worker.annotation_inodes


def prepare(output):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "CPU-only M04 prepare required")
    source = source_inputs()
    forbid_annotations(sorted(source["annotation_metadata"]))
    output = Path(output).absolute()
    require(not output.exists() and output.resolve() == output, "Fresh canonical M04 suite required")
    output.mkdir()
    inputs_sha = protected.write_prepared_json(output/"INPUTS.json",source)
    protocol_sha = protected.write_prepared_json(output/"PROTOCOL.json",dict(experiment="M04_rank_distribution_full",
        datasets=list(DATASETS),studies=list(factorial_prediction.STUDIES),factorial_arms=list(factorial_runtime.ARMS),
        images=10105,cells=16,image_method_evaluations=80840,seed=0,
        assignment="Within original source, raw confidence/C multisets on confidence/C order; lexical crop/query/component tie keys; original candidates fixed",
        admission="Natural each-arm EPS support and common-active rawmin(conf,C)>EPS conditioning are separate, never silently equated",
        source="Pinned original clean G input IDs/config/vocabulary; official L14 NACLIP last18; native PIL bilinear, no G predictions or candidate reuse",
        objective="Unchanged current image-balanced original-region entropy, detached assigned scores; production C not changed",
        diagnostics="All arms measure same production C, including Dk_Ok; no claim native zero-view Confidence timing",
        coefficients="Actual native dtype prepared rho/normalizer/source count plus original source/candidate identity; float64 ideal receipt separate, not causal or independent equivalence gate",
        gates="Both first4 RGB panels, all8study/arm combinations actualfinite steps/loss/gradient/parameter/Adam/final<=1e-5/0labels; naturaluncrossed production parity; shared probe/assignment reference limitation disclosed",
        reset="theta0/Adam and shared original episode RNG restored for every condition; explicit positional integer single-crop IDs before forward",
        timing="Synchronized reset/candidates/Cdiagnostic/adaptation/readout plus common finite/Adam/budget observation; exclude decode/H2D/serialization/GT; all episode0 predictions retained",
        barriers="All16full prediction cells sealed AND real successful service exit before separate new independent CPU_GT artifact admission",
        resources=dict(single_shared_GPU=True,RAM_bytes=12*1024**3,swap_bytes=1024**3,
            shared_disk_reserve_bytes=general.RESERVE,prediction_artifact_bytes=general.MAX_PAYLOAD_BYTES,
            OOMPolicy="stop",Restart="no",RemainAfterExit=True),
        launch="Manual once only after exact G1954 successful retained exit, named M02 service idle, sharedlock/operator service checks; no auto launch/dependency queue",
        failures="Keep partial/negative/null; failclosed/dispose method, no resume or automatic retry"))
    runtime = protected.guarded_snapshot(ROOT,output/"runtime",annotation_inodes(source))
    return protected.write_prepared_json(output/"SUITE_PREPARED.json",dict(schema_version=1,root=str(output),
        runtime_hashes=runtime,inputs_sha256=inputs_sha,protocol_sha256=protocol_sha,
        parent_native_hashes=native_identity(),GT_opened=False,GPU_started=False))


def admission(suite,expected_sha):
    require(not torch.cuda.is_initialized(), "M04 admission must precede CUDA")
    source = source_inputs()
    forbidden = annotation_inodes(source)
    forbid_annotations(sorted(source["annotation_metadata"]))
    suite = Path(suite).resolve(strict=True)
    prepared,_ = protected.metadata(suite/"SUITE_PREPARED.json",expected_sha,forbidden)
    require(prepared["root"] == str(suite) and prepared["schema_version"] == 1
            and Path(__file__).resolve().is_relative_to(suite/"runtime"), "Wrong M04 frozen runtime")
    protected.guarded_inventory(suite/"runtime",prepared["runtime_hashes"],forbidden)
    general.runtime_imports(suite/"runtime")
    inputs,_ = protected.metadata(suite/"INPUTS.json",prepared["inputs_sha256"],forbidden)
    protected.metadata(suite/"PROTOCOL.json",prepared["protocol_sha256"],forbidden)
    require(inputs == source and native_identity() == prepared["parent_native_hashes"], "M04 input/native drift")
    audit,audit_sha = protected.metadata(suite/"ADMITTED.json",forbidden=forbidden,limit=4*1024**2)
    require(audit["decision"] == "ADMITTED" and audit["suite_prepared_sha256"] == expected_sha
            and audit.get("independent_reviewer") and audit.get("M04_predictions_allowed") is True
            and audit.get("GT_scoring_allowed") is False, "New independent exact M04 admission required")
    report = suite/"INDEPENDENT_REPORT.md"
    require(audit["report_path"] == str(report), "Wrong M04 audit report path")
    identities = []
    for path in (suite/"ADMITTED.json",report):
        state = path.stat()
        require((state.st_dev,state.st_ino) not in forbidden, "M04 audit/report aliases GT")
        identities.append([state.st_dev,state.st_ino,state.st_size,state.st_mtime_ns])
    binding = dict(admission_path=str(suite/"ADMITTED.json"),admission_sha256=audit_sha,admission_identity=identities[0],
        report_path=str(report),report_sha256=audit["report_sha256"],report_identity=identities[1],
        independent_reviewer=audit["independent_reviewer"])
    verify_audit_binding(binding)
    for spec in inputs["specs"]:
        checked_checkpoint(spec["checkpoint"])
    return suite,prepared,inputs,binding


def build_method(spec):
    base = native.build_method(spec)
    result = factorial_runtime.FactorialExperimentTRACE(classes=base.classes,lr=base.lr,steps=base.steps,
        trace_max_regions=base.trace_max_regions,trace_view_batch_size=base.trace_view_batch_size,
        vision_outputs=base.vision_outputs,prompts=base.prompt_templates,model=base.model,
        tokenize_fn=base.tokenize,device=base.device,view_mode="C")
    result.model.eval()
    native.legal_parameters(result)
    return result


def require_named_GPU_idle():
    keys = ("LoadState","ActiveState","SubState","MainPID")
    result = subprocess.run(["systemctl","--user","show","trace-spatial-20261102-2156.service",
        "--property="+",".join(keys)],check=False,capture_output=True,text=True,timeout=30)
    state = dict(line.split("=",1) for line in result.stdout.splitlines() if "=" in line)
    require(result.returncode in (0,4) and state.get("MainPID") == "0"
            and ((state.get("LoadState") == "not-found" and state.get("ActiveState") == "inactive"
                  and state.get("SubState") == "dead")
                 or (state.get("LoadState") == "loaded" and state.get("ActiveState") in ("active","inactive","failed")
                     and state.get("SubState") in ("exited","dead","failed"))),
            "Named M02 GPU worker not confirmed idle; no second GPU process")
    return state


def prior_GPU_terminal():
    prior = source_worker.prior_GPU_terminal()
    return dict(G1954=prior,M02_idle_only_not_scientific_completion=require_named_GPU_idle())


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
                    stop = min(start+cfg["batch_size"],len(data))
                    images,meta,valid,ids = general.episode_batch(data,start,stop,method,cfg)
                    reports.append(factorial_gate.compare_gate(method,images,valid,ids,meta,cfg,
                        spec["vocabulary"]["query_to_class"],[r["official_id"] for r in rows[start:stop]]))
                require(len(data) == 4 and bool(reports), "M04 requires both4RGB gates")
                record = dict(passed=True,dataset=spec["dataset"],suite_prepared_sha256=expected_sha,
                    actual_RGB_sources=4,ordered_ids=[r["official_id"] for r in rows],comparisons=reports,gpu_lock=lock,GT_opened=False)
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


verify_gates = source_worker.verify_gates


def run(suite,expected_sha):
    prior = prior_GPU_terminal()
    suite,prepared,inputs,audit = admission(suite,expected_sha)
    require(not any((suite/n).exists() for n in ("STARTED.json","FAILED.json","ALL_PREDICTIONS_SEALED.json")), "No M04 resume/retry")
    require(shutil.disk_usage(suite).free >= general.RESERVE+general.MAX_PAYLOAD_BYTES, "M04 shared disk headroom insufficient")
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
                    result = factorial_prediction.execute_dataset(method,data,inputs["rows"][spec["dataset"]],spec,suite,budget,
                        lambda row:write_json(suite/"queue_state.json",dict(phase="M04_paired_full_prediction",pid=os.getpid(),**row),replace=True))
                    require(result["payloads_complete"] is True and result["GT_opened"] is False
                            and result["dataset"] == spec["dataset"] and result["images"] == spec["images"]
                            and result["studies"] == list(factorial_prediction.STUDIES)
                            and result["factorial_arms"] == list(factorial_runtime.ARMS)
                            and result["image_method_evaluations"] == spec["images"]*8, "Incomplete M04 dataset")
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
                and native_identity() == prepared["parent_native_hashes"], "M04 source/native drift or missing dataset")
        protected.guarded_inventory(suite/"runtime",prepared["runtime_hashes"],forbidden)
        protected.metadata(suite/"SUITE_PREPARED.json",expected_sha,forbidden)
        protected.metadata(suite/"INPUTS.json",prepared["inputs_sha256"],forbidden)
        protected.metadata(suite/"PROTOCOL.json",prepared["protocol_sha256"],forbidden)
        general.runtime_imports(suite/"runtime")
        verify_audit_binding(audit)
        require(sha256(suite/"STARTED.json") == started_sha, "M04 start receipt drift")
        verify_gates(suite,expected_sha,gate_binding,forbidden)
        for dataset,digest in seals.items():
            protected.metadata(suite/dataset/"PREDICTIONS_SEALED.json",digest,forbidden)
        write_json(suite/"queue_state.json",dict(phase="M04_payloads_complete_awaiting_global_seal_and_exit",
            datasets_complete=2,GT_opened=False,pid=os.getpid()),replace=True)
        publish_seal(suite/"ALL_PREDICTIONS_SEALED.json",dict(complete=True,suite_prepared_sha256=expected_sha,
            dataset_seals=seals,cells=len(inputs["specs"])*8,
            image_method_evaluations=sum(s["images"]*8 for s in inputs["specs"]),artifact_bytes=budget["bytes"],
            gate_binding=gate_binding,audit_binding=audit,started_sha256=started_sha,GT_opened=False,scientific_completion=False))
    except BaseException as error:
        protected.write_prepared_json(suite/"FAILED.json",dict(error=f"{type(error).__name__}: {error}",GT_opened=False,automatic_retry=False))
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
        require(args.suite_sha256,"Exact M04 prepared SHA required")
        if args.phase == "validate-admission":
            admission(args.suite,args.suite_sha256)
            print(dict(admission_valid=True,CUDA_initialized=torch.cuda.is_initialized(),GPU_started=False,GT_opened=False))
        else:
            run(args.suite,args.suite_sha256)
