"""M02-only freeze/admit/gate/predict controller. No GT scoring entry point.

Preparation is CPU-only and not execution permission. Execution is manual,
single-attempt, after exact G1954 successful exit and new independent artifact
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
import time

import torch

from tools.evolution import run_cell as shared
from tools.evolution_confirmation import run as confirmation
from tools.evolution_followup.standalone import gpu_lock
from tools.evolution_tuning.common import write_json
from . import generalization_runtime as native
from . import generalization_worker as general
from . import generalization_score_worker as protected
from . import spatial_gate
from . import spatial_prediction
from . import spatial_runtime
from .bootstrap import POPULATIONS, require
from .generalization import visual_layers
from .intake import sha256
from .risk_replay_worker import native_identity
from .risk_score_worker import publish_seal, verify_audit_binding
from .risk_worker import checked_checkpoint, deterministic_runtime, forbid_annotations

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("coco_object", "pc60")


def source_inputs():
    parent = protected.PARENT
    prepared, _ = protected.metadata(parent/"SUITE_PREPARED.json", protected.PARENT_SHA,
        expected_identity=protected.PARENT_METADATA_IDENTITIES["SUITE_PREPARED.json"])
    inputs, _ = protected.metadata(parent/"INPUTS.json", prepared["inputs_sha256"],
        expected_identity=protected.PARENT_METADATA_IDENTITIES["INPUTS.json"])
    forbidden = protected.label_inodes(inputs)
    source_specs, _ = protected.metadata(parent/"CONDITIONS.json", prepared["conditions_sha256"], forbidden)
    general.validate_conditions(source_specs)
    annotations = {}
    for key,value in inputs.items():
        if key.startswith("clean."):
            for row in value:
                general.label_identity(row)
                annotations[row["label"]] = copy.deepcopy(row["label_metadata_details"])
    checkpoint = next(s["checkpoint"] for s in source_specs if s["backbone"] == "ViT-L/14")
    rows = {d:inputs["clean."+d] for d in DATASETS}
    specs = []
    for dataset in DATASETS:
        original = next(s for s in source_specs if s["dataset"] == dataset and s["corruption"] == "original")
        spec = copy.deepcopy(original)
        spec.update(id="M02."+dataset,backbone="ViT-L/14",model_type="naclip",checkpoint=copy.deepcopy(checkpoint),
            vision_outputs=visual_layers("ViT-L/14"),modes=list(spatial_runtime.MODES),preprocessing="native_PIL_bilinear")
        spec.pop("arms")
        specs.append(spec)
    validate_inputs(rows,specs)
    return dict(parent_input_suite=str(parent),parent_prepared_sha256=protected.PARENT_SHA,
        parent_inputs_sha256=prepared["inputs_sha256"],parent_conditions_sha256=prepared["conditions_sha256"],
        rows=rows,specs=specs,annotation_metadata=annotations,
        G_condition_ids=[s["id"] for s in source_specs],GT_opened=False,old_predictions_reused=False)


def annotation_inodes(source):
    return {(s["device"],s["inode"]) for s in source["annotation_metadata"].values()}


def validate_inputs(rows,specs):
    require(set(rows) == set(DATASETS) and [s["dataset"] for s in specs] == list(DATASETS), "M02 dataset membership differs")
    forbidden = protected.label_inodes(rows)
    for spec in specs:
        dataset, cfg = spec["dataset"],spec["config"]
        native.image_worker.validate_config(cfg)
        require(spec["modes"] == list(spatial_runtime.MODES) and spec["backbone"] == "ViT-L/14"
                and spec["model_type"] == "naclip" and spec["preprocessing"] == "native_PIL_bilinear"
                and spec["vision_outputs"] == visual_layers("ViT-L/14")
                and cfg["resize"] == cfg["patch_size"] == [224,224], "M02 actual model/config/geometry differs")
        require(spec["images"] == len(rows[dataset]) == POPULATIONS[dataset]
                and len({r["official_id"] for r in rows[dataset]}) == len(rows[dataset]), "M02 full population differs")
        for row in rows[dataset]:
            general.label_identity(row)
            path = Path(row["image"])
            require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "M02 RGB path is not canonical")
            state = path.stat()
            require((state.st_dev,state.st_ino) not in forbidden and 0 < state.st_size <= 32*1024**2,
                    "M02 RGB aliases GT or exceeds size bound")
    return forbidden


def prepare(output):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "CPU-only M02 preparation required")
    source = source_inputs()
    forbid_annotations(sorted(source["annotation_metadata"]))
    output = Path(output).absolute()
    require(not output.exists() and output.resolve() == output, "Fresh canonical M02 suite required")
    output.mkdir()
    inputs_sha = protected.write_prepared_json(output/"INPUTS.json",source)
    protocol_sha = protected.write_prepared_json(output/"PROTOCOL.json",dict(experiment="M02_spatial_full",
        datasets=list(DATASETS),modes=list(spatial_runtime.MODES),images=10105,image_method_evaluations=40420,seed=0,
        algorithm="Production ImageBalancedTRACE:C unchanged; only diagnostic preservation changes in named controls",
        source="Exact frozen clean G input IDs/config/vocabulary reused; source RGB redecoded using native PILbilinear, no G model prediction or candidate reuse",
        spatial="Native candidate preservation, legal equal-shape/area no-wrap translated preservation, whole-image context, originalconfidence; original query/readout/loss fixed",
        gates="Both datasets first4RGB, all4modes, finite actual optimizer steps/loss/grad/param/Adam/final within1e-5 and0changedlabels; native/conf production parity",
        reference="Direct masked mean arithmetic, shared probe/geometry; not independent wholealgorithm implementation",
        reset="Native theta0 and Adam/RNG per episode and mode; same original candidates across all4 modes",
        timing="Synchronized reset through final readout includes finite/actual-step observation; decode/H2D/serialization/GT excluded; raw totals and episode0 retained",
        barriers="All8fullcells prediction sealed and actual successful exit before new separate independent CPU GT scoring admission",
        resources=dict(single_shared_GPU=True,RAM_bytes=12*1024**3,swap_bytes=1024**3,
            shared_disk_reserve_bytes=general.RESERVE,prediction_artifact_bytes=general.MAX_PAYLOAD_BYTES,
            OOMPolicy="stop",Restart="no",RemainAfterExit=True),
        launch="No automatic dependency queue; manual single attempt only after exact G1954 successful retained exit",
        failures="Failclosed/dispose method; no resume/automatic retry; retain failed/negative/null results"))
    runtime = protected.guarded_snapshot(ROOT,output/"runtime",annotation_inodes(source))
    return protected.write_prepared_json(output/"SUITE_PREPARED.json",dict(schema_version=1,root=str(output),
        runtime_hashes=runtime,inputs_sha256=inputs_sha,protocol_sha256=protocol_sha,
        parent_native_hashes=native_identity(),GT_opened=False,GPU_started=False))


def admission(suite,expected_sha):
    require(not torch.cuda.is_initialized(), "M02 admission must precede CUDA initialization")
    source = source_inputs()
    validate_inputs(source["rows"],source["specs"])
    forbidden = annotation_inodes(source)
    forbid_annotations(sorted(source["annotation_metadata"]))
    suite = Path(suite).resolve(strict=True)
    prepared, _ = protected.metadata(suite/"SUITE_PREPARED.json",expected_sha,forbidden)
    require(prepared["root"] == str(suite) and prepared["schema_version"] == 1
            and Path(__file__).resolve().is_relative_to(suite/"runtime"), "Wrong M02 frozen runtime")
    protected.guarded_inventory(suite/"runtime",prepared["runtime_hashes"],forbidden)
    general.runtime_imports(suite/"runtime")
    inputs, _ = protected.metadata(suite/"INPUTS.json",prepared["inputs_sha256"],forbidden)
    protected.metadata(suite/"PROTOCOL.json",prepared["protocol_sha256"],forbidden)
    require(inputs == source and native_identity() == prepared["parent_native_hashes"], "M02 source/native metadata drift")
    audit, audit_sha = protected.metadata(suite/"ADMITTED.json",forbidden=forbidden,limit=4*1024**2)
    require(audit["decision"] == "ADMITTED" and audit["suite_prepared_sha256"] == expected_sha
            and audit.get("independent_reviewer") and audit.get("M02_predictions_allowed") is True
            and audit.get("GT_scoring_allowed") is False, "New independent exact M02 prediction admission required")
    report = suite/"INDEPENDENT_REPORT.md"
    require(audit["report_path"] == str(report), "M02 independent report path differs")
    identities = []
    for path in (suite/"ADMITTED.json",report):
        state = path.stat()
        require((state.st_dev,state.st_ino) not in forbidden, "M02 admission/report aliases GT")
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
    result = spatial_runtime.SpatialExperimentTRACE(classes=base.classes,lr=base.lr,steps=base.steps,
        trace_max_regions=base.trace_max_regions,trace_view_batch_size=base.trace_view_batch_size,
        vision_outputs=base.vision_outputs,prompts=base.prompt_templates,model=base.model,
        tokenize_fn=base.tokenize,device=base.device,view_mode="C")
    result.model.eval()
    spatial_runtime.legal_parameters(result)
    return result


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
                    reports.append(spatial_gate.compare_gate(method,images,valid,ids,meta,cfg,spec["vocabulary"]["query_to_class"]))
                require(len(data) == 4 and bool(reports), "Both M02 datasets need4 actual RGB gate sources")
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
            and set(barrier["datasets"]) == set(DATASETS), "Both M02 gates required")
    for dataset,digest in barrier["datasets"].items():
        record, _ = protected.metadata(suite/"gates"/(dataset+".json"),digest,forbidden)
        require(record["passed"] is True and record["dataset"] == dataset
                and record["actual_RGB_sources"] == 4 and record["GT_opened"] is False, "Invalid M02 gate child")


def prior_GPU_terminal():
    terminal = protected.parent_terminal()
    prepared, _ = protected.metadata(protected.PARENT/"SUITE_PREPARED.json",protected.PARENT_SHA,
        expected_identity=protected.PARENT_METADATA_IDENTITIES["SUITE_PREPARED.json"])
    source = source_inputs()
    barrier, digest = protected.metadata(protected.PARENT/"ALL_G_PREDICTIONS_SEALED.json",
        forbidden=annotation_inodes(source))
    require(barrier["complete"] is True and barrier["suite_prepared_sha256"] == protected.PARENT_SHA
            and barrier["GT_opened"] is False and barrier["cells"] == 57
            and barrier["image_method_evaluations"] == 104214 and prepared["root"] == str(protected.PARENT)
            and set(barrier["condition_seals"]) == set(source["G_condition_ids"])
            and len(barrier["condition_seals"]) == 19,
            "G1954 completion barrier required before next manual GPU worker")
    for name,child_sha in barrier["condition_seals"].items():
        child, _ = protected.metadata(protected.PARENT/"conditions"/name/"PREDICTIONS_SEALED.json",
            child_sha,annotation_inodes(source))
        require(child["complete"] is True and child["GT_opened"] is False
                and child["condition"]["id"] == name and child["suite_prepared_sha256"] == protected.PARENT_SHA,
                "G prior worker child barrier differs")
    return dict(terminal=terminal,barrier_sha256=digest)


def run(suite,expected_sha):
    prior = prior_GPU_terminal()
    suite,prepared,inputs,audit = admission(suite,expected_sha)
    require(not any((suite/n).exists() for n in ("STARTED.json","FAILED.json","ALL_PREDICTIONS_SEALED.json")), "No M02 resume/retry")
    require(shutil.disk_usage(suite).free >= general.RESERVE+general.MAX_PAYLOAD_BYTES, "M02 shared disk headroom insufficient")
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
                    result = spatial_prediction.execute_dataset(method,data,inputs["rows"][spec["dataset"]],spec,suite,budget,
                        lambda row:write_json(suite/"queue_state.json",dict(phase="M02_paired_full_prediction",pid=os.getpid(),**row),replace=True))
                    require(result["payloads_complete"] is True and result["GT_opened"] is False
                            and result["dataset"] == spec["dataset"] and result["images"] == spec["images"]
                            and result["image_method_evaluations"] == spec["images"]*4
                            and result["modes"] == list(spatial_runtime.MODES), "Incomplete M02 dataset result")
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
                and native_identity() == prepared["parent_native_hashes"], "M02 source/native drift or missing dataset")
        protected.guarded_inventory(suite/"runtime",prepared["runtime_hashes"],forbidden)
        protected.metadata(suite/"SUITE_PREPARED.json",expected_sha,forbidden)
        protected.metadata(suite/"INPUTS.json",prepared["inputs_sha256"],forbidden)
        protected.metadata(suite/"PROTOCOL.json",prepared["protocol_sha256"],forbidden)
        general.runtime_imports(suite/"runtime")
        verify_audit_binding(audit)
        require(sha256(suite/"STARTED.json") == started_sha, "M02 start receipt changed")
        verify_gates(suite,expected_sha,gate_binding,forbidden)
        for dataset,digest in seals.items():
            protected.metadata(suite/dataset/"PREDICTIONS_SEALED.json",digest,forbidden)
        write_json(suite/"queue_state.json",dict(phase="M02_payloads_complete_awaiting_global_seal_and_exit",
            datasets_complete=2,GT_opened=False,pid=os.getpid()),replace=True)
        publish_seal(suite/"ALL_PREDICTIONS_SEALED.json",dict(complete=True,suite_prepared_sha256=expected_sha,
            dataset_seals=seals,cells=len(inputs["specs"])*4,
            image_method_evaluations=sum(s["images"]*4 for s in inputs["specs"]),artifact_bytes=budget["bytes"],
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
        require(args.suite_sha256,"Exact M02 prepared SHA required")
        if args.phase == "validate-admission":
            admission(args.suite,args.suite_sha256)
            print(dict(admission_valid=True,CUDA_initialized=torch.cuda.is_initialized(),GPU_started=False,GT_opened=False))
        else:
            run(args.suite,args.suite_sha256)
