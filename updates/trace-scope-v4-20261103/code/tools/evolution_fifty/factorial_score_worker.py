"""M04-only CPU scoring; no preparation before exact successful prediction exit.

The operator supplies the freshly verified prediction-global hash at prepare.
This pin is frozen into a NEW scoring suite. No previous study's permission
admits M04 GT, and no model, GPU launch, adaptation or resume exists here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

import torch

from . import factorial_analysis as analysis
from . import factorial_prediction as core
from . import factorial_score_payload as payload
from . import factorial_worker as prediction
from . import generalization_score_worker as protected
from . import spatial_score_worker as shared
from .bootstrap import POPULATIONS,require
from .risk_score_worker import canonical_child,publish_seal,verify_audit_binding

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/home/wwww/hzl/results/trace_mechanism_20261102")
PARENT = BASE/"M04_factorial_20261102_2243"
PARENT_SHA = "02d9e169ae5bf5f7fd723f896ae5cb208c2278a1c85d7a2bd8f04b83d907f201"
PARENT_START_SHA = "f6e1749b763d8027baeaf8a7eb29117bbbfca67e8403815d9489f54cfd3e743c"
PARENT_ADMISSION_SHA = "ceba28ed1fc114e47e420cc14563fc4ce9d7e991fe67db72c8021b640b4f8f46"
PARENT_REPORT_SHA = "d900e39a40f606354ef00061b0bdcac7fecefa030b7222fda5df0ff39cd11c3e"
PARENT_GATE_SHA = "913365e51a67490815e9e2e9b58a7ced912635cc3aedabc0e14219bce4888a98"
ANALYSIS = BASE/"M04_ANALYSIS_PROTOCOL_v1.json"
ANALYSIS_SHA = "cf08822ef2e406b4508f2ac8f1b2c3093edd92248fc322aa161107c2aae63867"
MODES,DATASETS = core.MODES,prediction.DATASETS
RESERVE,OUTPUT_CAP = payload.RESERVE,payload.OUTPUT_CAP


def cpu_only():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "M04 CPU scoring requires hidden uninitialized CUDA")


def parent_terminal():
    keys = "LoadState,ActiveState,SubState,MainPID,ExecMainPID,ExecMainCode,ExecMainStatus,Result,InvocationID"
    result = subprocess.run(["systemctl","--user","show","trace-factorial-20261102-2243.service",
        "--property="+keys],check=True,text=True,capture_output=True,timeout=30)
    actual = dict(line.split("=",1) for line in result.stdout.splitlines() if "=" in line)
    expected = dict(LoadState="loaded",ActiveState="active",SubState="exited",MainPID="0",ExecMainPID="2121324",
        ExecMainCode="1",ExecMainStatus="0",Result="success",InvocationID="d1e5a7f8cff34dbb979545716d78bdd0")
    require(all(actual.get(k) == v for k,v in expected.items()), "Exact M04 prediction parent has not successfully exited")
    return actual


def bind_parent(barrier_sha):
    require(type(barrier_sha) is str and len(barrier_sha) == 64 and all(c in "0123456789abcdef" for c in barrier_sha),
            "Operator-verified M04 prediction global SHA required")
    terminal = parent_terminal()
    source = prediction.source_inputs()
    forbidden = payload.data_exclusions(source)
    require(PARENT.is_absolute() and PARENT.resolve(strict=True) == PARENT
            and not any((PARENT/n).exists() for n in ("FAILED.json","CANCELLED.json")), "Unsuccessful M04 prediction parent")
    prepared,_ = protected.metadata(PARENT/"SUITE_PREPARED.json",PARENT_SHA,forbidden)
    inputs,_ = protected.metadata(PARENT/"INPUTS.json",prepared["inputs_sha256"],forbidden)
    require(prepared["root"] == str(PARENT) and prepared["schema_version"] == 1 and inputs == source,
            "M04 original input identity differs")
    shared.native_binding(prepared["parent_native_hashes"],forbidden)
    protected.metadata(PARENT/"PROTOCOL.json",prepared["protocol_sha256"],forbidden)
    protected.guarded_inventory(PARENT/"runtime",prepared["runtime_hashes"],forbidden)
    barrier,_ = protected.metadata(PARENT/"ALL_PREDICTIONS_SEALED.json",barrier_sha,forbidden)
    require(barrier["complete"] is True and barrier["GT_opened"] is False and barrier["suite_prepared_sha256"] == PARENT_SHA
            and barrier["cells"] == 16 and barrier["image_method_evaluations"] == sum(POPULATIONS[d]*8 for d in DATASETS)
            and set(barrier["dataset_seals"]) == set(DATASETS) and barrier["started_sha256"] == PARENT_START_SHA
            and barrier["gate_binding"]["barrier_sha256"] == PARENT_GATE_SHA, "M04 complete prediction barrier differs")
    audit = barrier["audit_binding"]
    require(audit["admission_path"] == str(PARENT/"ADMITTED.json") and audit["report_path"] == str(PARENT/"INDEPENDENT_REPORT.md")
            and audit["admission_sha256"] == PARENT_ADMISSION_SHA and audit["report_sha256"] == PARENT_REPORT_SHA,
            "Wrong M04 prediction audit binding")
    admitted,_ = protected.metadata(PARENT/"ADMITTED.json",PARENT_ADMISSION_SHA,forbidden)
    report_state = (PARENT/"INDEPENDENT_REPORT.md").stat()
    require((report_state.st_dev,report_state.st_ino) not in forbidden, "M04 parent report aliases GT")
    verify_audit_binding(audit)
    start,_ = protected.metadata(PARENT/"STARTED.json",PARENT_START_SHA,forbidden)
    require(admitted["decision"] == "ADMITTED" and admitted["suite_prepared_sha256"] == PARENT_SHA
            and admitted["M04_predictions_allowed"] is True and admitted["GT_scoring_allowed"] is False
            and start["suite_prepared_sha256"] == PARENT_SHA and type(start["pid"]) is int
            and str(start["pid"]) == terminal["ExecMainPID"] and start["audit"] == audit and start["GT_opened"] is False,
            "M04 prediction permission/startup differs")
    prediction.verify_gates(PARENT,PARENT_SHA,barrier["gate_binding"],forbidden)
    seals = {}
    for spec in inputs["specs"]:
        d,n,batch = spec["dataset"],spec["images"],spec["config"]["batch_size"]
        seal,_ = protected.metadata(PARENT/d/"PREDICTIONS_SEALED.json",barrier["dataset_seals"][d],forbidden)
        require(seal["complete"] is True and seal["GT_opened"] is False and seal["dataset"] == d
                and seal["suite_prepared_sha256"] == PARENT_SHA and seal["images"] == n == POPULATIONS[d]
                and seal["studies"] == list(core.STUDIES) and seal["factorial_arms"] == list(core.runtime.ARMS)
                and seal["image_method_evaluations"] == n*8, "M04 full dataset population differs")
        expected = {f"{d}/predictions/{m}/{i:06d}.npy" for m in MODES for i in range(n)}
        expected |= {f"{d}/episodes/{e:06d}.{m}.json" for m in MODES for e in range((n+batch-1)//batch)}
        expected |= {f"{d}/{name}" for name in ("initial_state.json","cost.jsonl","prediction_result.json")}
        require(set(seal["artifact_sha256"]) == expected, "M04 complete prediction inventory differs")
        seals[d] = seal
    payload_metadata = payload.preflight_payloads(PARENT,inputs["specs"],seals,inputs)
    for spec in inputs["specs"]:
        payload.cost_binding(PARENT,spec,seals[spec["dataset"]]["artifact_sha256"],forbidden,payload_metadata)
    protocol,_ = protected.metadata(ANALYSIS,ANALYSIS_SHA,forbidden)
    require(protocol["datasets"] == {d:POPULATIONS[d] for d in DATASETS}
            and protocol["studies"] == list(core.STUDIES) and protocol["arms"] == list(core.runtime.ARMS)
            and protocol["contrast_weights_in_arm_order"] == {k:list(v) for k,v in analysis.CONTRASTS.items()}
            and protocol["uncertainty"]["replicates"] == 10000 and protocol["uncertainty"]["root_seed"] == 20261102,
            "M04 prospective analysis protocol differs")
    return dict(parent=str(PARENT),prepared_sha256=PARENT_SHA,barrier_sha256=barrier_sha,
        terminal=terminal,source=inputs,seals=seals,payload_metadata=payload_metadata,analysis_sha256=ANALYSIS_SHA)


def prepare(output,barrier_sha):
    cpu_only()
    parent_terminal()
    bound = bind_parent(barrier_sha)
    output = Path(output).absolute()
    require(output.resolve() == output and not output.exists(), "Fresh canonical M04 scoring suite required")
    output.mkdir()
    inputs_sha = protected.write_prepared_json(output/"SCORE_INPUTS.json",bound)
    protocol_sha = protected.write_prepared_json(output/"PROTOCOL.json",dict(experiment="M04_offline_CPU_score",
        parent_prepared_sha256=PARENT_SHA,parent_prediction_global_sha256=barrier_sha,prospective_analysis_sha256=ANALYSIS_SHA,
        new_GPU=False,boundary="All16 sealed predictions plus exact retained successful parent exit, new independent CPU_GT AND M04_GT permission",
        annotation_geometry="Historical native PIL NEAREST224 preserving palette/native mapping, NOT G OpenCV nearest",
        estimates="Natural/common-active separate,6 fixed contrasts, pooled class I/U then mIoU,10000 shared whole-original-episode PCG64 draws/pointwise95/nullundefined retained",
        coverage="Original/retained/EPSactive candidate/source counts, recomputed assignment/native-dtype budgets; not candidate causality",
        receipt_arithmetic=dict(torch_version=payload.receipts.CUDA_DIVISION_VERSION,
            torch_git=payload.receipts.CUDA_DIVISION_GIT,source=payload.receipts.CUDA_DIVISION_SOURCE,
            coefficient="Pinned CUDA tensor division followed by opmath CPU-scalar reciprocal multiplication and native cast; exact equality, no tolerance"),
        time="All arms measure C and all raw episodes including0 retained; no zero-diagnostic Confidence or equal-time claim",
        failures="No resume/automaticretry; exclusive progress/outputs, final atomic global seal last",
        limits=dict(output_cap_bytes=OUTPUT_CAP,shared_disk_reserve_bytes=RESERVE,GPU=False)))
    inventory = protected.guarded_snapshot(ROOT,output/"runtime",prediction.annotation_inodes(bound["source"]))
    return protected.write_prepared_json(output/"SUITE_PREPARED.json",dict(schema_version=1,root=str(output),
        runtime_hashes=inventory,inputs_sha256=inputs_sha,protocol_sha256=protocol_sha,
        parent_prepared_sha256=PARENT_SHA,parent_barrier_sha256=barrier_sha,GT_opened=False,CPU_scoring_started=False))


def admission(suite,expected_sha):
    cpu_only()
    forbidden = payload.data_exclusions(prediction.source_inputs())
    suite = Path(suite).resolve(strict=True)
    prepared,_ = protected.metadata(suite/"SUITE_PREPARED.json",expected_sha,forbidden)
    require(prepared["root"] == str(suite) and prepared["schema_version"] == 1
            and prepared["parent_prepared_sha256"] == PARENT_SHA and Path(__file__).resolve().is_relative_to(suite/"runtime"),
            "Wrong exact M04 scoring runtime")
    bound = bind_parent(prepared["parent_barrier_sha256"])
    protected.guarded_inventory(suite/"runtime",prepared["runtime_hashes"],forbidden)
    prediction.general.runtime_imports(suite/"runtime")
    inputs,_ = protected.metadata(suite/"SCORE_INPUTS.json",prepared["inputs_sha256"],forbidden)
    protocol,_ = protected.metadata(suite/"PROTOCOL.json",prepared["protocol_sha256"],forbidden)
    require(protocol["receipt_arithmetic"] == dict(torch_version=payload.receipts.CUDA_DIVISION_VERSION,
        torch_git=payload.receipts.CUDA_DIVISION_GIT,source=payload.receipts.CUDA_DIVISION_SOURCE,
        coefficient="Pinned CUDA tensor division followed by opmath CPU-scalar reciprocal multiplication and native cast; exact equality, no tolerance"),
        "M04 pinned CUDA receipt arithmetic protocol differs")
    require(inputs == bound, "M04 parent score binding drift")
    audit,audit_sha = protected.metadata(suite/"ADMITTED.json",forbidden=forbidden,limit=4*1024**2)
    require(audit["decision"] == "ADMITTED" and audit["suite_prepared_sha256"] == expected_sha
            and audit.get("independent_reviewer") and audit.get("CPU_GT_scoring_allowed") is True
            and audit.get("M04_GT_scoring_allowed") is True, "New exact independent M04 CPU_GT admission required")
    report = suite/"INDEPENDENT_REPORT.md"
    require(audit["report_path"] == str(report), "Wrong M04 score audit report path")
    identities = []
    for path in (suite/"ADMITTED.json",report):
        st = path.stat()
        require((st.st_dev,st.st_ino) not in forbidden, "M04 score audit/report aliases GT")
        identities.append([st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns])
    binding = dict(admission_path=str(suite/"ADMITTED.json"),admission_sha256=audit_sha,admission_identity=identities[0],
        report_path=str(report),report_sha256=audit["report_sha256"],report_identity=identities[1],independent_reviewer=audit["independent_reviewer"])
    verify_audit_binding(binding)
    return suite,prepared,inputs,binding


def execute(suite,bound):
    require(payload.preflight_payloads(Path(bound["parent"]),bound["source"]["specs"],bound["seals"],bound["source"])
            == bound["payload_metadata"], "M04 complete payload graph changed before first GT")
    budget,labels,results,artifacts = dict(bytes=0),{},{},{}
    for spec in bound["source"]["specs"]:
        values,ordered,cost,coverage,path,digest = payload.score_dataset(suite,bound,spec,budget,labels,
            lambda row:shared.write_json(suite/"queue_state.json",row,replace=True))
        result,_ = analysis.paired_summary(values,spec["dataset"])
        result.update(images=spec["images"],ordered_source_episodes=ordered,cost=cost,allocation_coverage=coverage,
            source="L_local",resources=bound["seals"][spec["dataset"]]["resource_observation"])
        results[spec["dataset"]] = result
        artifacts[path.name] = digest
    for name,value in (("DATASET_RESULTS.json",results),("ANNOTATIONS_READ.json",labels)):
        encoded = (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
        require(shutil.disk_usage(suite).free >= RESERVE+len(encoded) and budget["bytes"]+len(encoded) <= OUTPUT_CAP,
                "M04 score summary exceeds prewrite budget")
        artifacts[name] = protected.write_prepared_json(suite/name,value)
        budget["bytes"] += len(encoded)
    return dict(complete=True,datasets=len(results),cells=8*len(results),GT_opened=True,new_GPU=False,
        image_method_evaluations=sum(v["images"]*8 for v in results.values()),artifact_sha256=artifacts,output_bytes=budget["bytes"])


def run(suite,expected_sha):
    suite,prepared,inputs,audit = admission(suite,expected_sha)
    require(not any((suite/n).exists() for n in ("STARTED.json","FAILED.json","ANALYSIS_SEALED.json")), "No M04 score resume/retry")
    require(shutil.disk_usage(suite).free >= RESERVE+OUTPUT_CAP, "M04 score disk headroom insufficient")
    verify_audit_binding(audit)
    started = dict(suite_prepared_sha256=expected_sha,audit=audit,pid=os.getpid(),GT_opened=False,
                   automatic_retry=False,parent_barrier_sha256=inputs["barrier_sha256"])
    content = (json.dumps(started,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    require(len(content) <= 64*1024, "M04 STARTED metadata exceeds bound")
    owned = False
    try:
        # Own failures after exclusive creation, but never claim another run's
        # marker if a competing creator won the pathname before this open.
        path = suite/"STARTED.json"
        with path.open("xb") as stream:
            owned = True
            first = os.fstat(stream.fileno())
            require(stat.S_ISREG(first.st_mode) and first.st_nlink == 1, "M04 STARTED inode differs")
            require(stream.write(content) == len(content), "Short M04 STARTED write")
            stream.flush()
            os.fsync(stream.fileno())
            state,actual = os.fstat(stream.fileno()),path.lstat()
            identity = lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink,s.st_mode)
            require(path.resolve(strict=True) == path and identity(state) == identity(actual)
                    and (state.st_dev,state.st_ino) == (first.st_dev,first.st_ino)
                    and state.st_nlink == 1 and state.st_size == len(content), "M04 STARTED pathname changed")
        started_sha = hashlib.sha256(content).hexdigest()
        begin = time.perf_counter()
        forbidden = payload.data_exclusions(inputs["source"])
        protected.metadata(path,started_sha,forbidden,limit=64*1024)
        result = execute(suite,inputs)
        require(result["datasets"] == 2 and result["cells"] == 16
                and result["image_method_evaluations"] == sum(POPULATIONS[d]*8 for d in DATASETS), "Incomplete M04 scoring population")
        require(bind_parent(inputs["barrier_sha256"]) == inputs, "M04 prediction parent drift before final seal")
        protected.guarded_inventory(suite/"runtime",prepared["runtime_hashes"],forbidden)
        for name,digest in (("SCORE_INPUTS.json",prepared["inputs_sha256"]),("PROTOCOL.json",prepared["protocol_sha256"]),
                            ("SUITE_PREPARED.json",expected_sha),("STARTED.json",started_sha)):
            protected.metadata(suite/name,digest,forbidden)
        prediction.general.runtime_imports(suite/"runtime")
        verify_audit_binding(audit)
        require(not torch.cuda.is_initialized(), "M04 CPU scoring initialized CUDA")
        for relative,digest in result["artifact_sha256"].items():
            shared.bounded_content(canonical_child(suite,relative),digest,forbidden,16*1024**2)
        shared.write_json(suite/"queue_state.json",dict(phase="M04_analysis_payloads_complete_awaiting_seal_and_exit",
            GT_opened=True,pid=os.getpid()),replace=True)
        publish_seal(suite/"ANALYSIS_SEALED.json",dict(**result,suite_prepared_sha256=expected_sha,
            parent_barrier_sha256=inputs["barrier_sha256"],prospective_analysis_sha256=ANALYSIS_SHA,
            audit_binding=audit,started_sha256=started_sha,elapsed_seconds=time.perf_counter()-begin))
    except BaseException as error:
        if owned:
            protected.write_prepared_json(suite/"FAILED.json",dict(error=f"{type(error).__name__}: {error}",automatic_retry=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","validate-admission","run"))
    parser.add_argument("--suite",required=True,type=Path)
    parser.add_argument("--suite-sha256")
    parser.add_argument("--parent-barrier-sha256")
    args = parser.parse_args()
    if args.phase == "prepare":
        print(dict(prepared_sha256=prepare(args.suite,args.parent_barrier_sha256),GT_opened=False,GPU_started=False))
    else:
        require(args.suite_sha256,"Exact M04 scoring prepared SHA required")
        if args.phase == "validate-admission":
            admission(args.suite,args.suite_sha256)
            print(dict(admission_valid=True,CUDA_initialized=torch.cuda.is_initialized(),GT_opened=False,GPU_started=False))
        else:
            run(args.suite,args.suite_sha256)
