"""M03-only freeze/admit/gate/full prediction controller, never GT scoring.

New exact artifact admission and manual capped execution are required. Sharing
input-validation helpers does not transfer any M02/G execution permission.
"""

from __future__ import annotations

import argparse
import copy
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
from . import generalization_runtime as native
from . import generalization_worker as general
from . import generalization_score_worker as protected
from . import legacy_gate, legacy_prediction, legacy_runtime, legacy_rho
from . import spatial_worker as source
from .bootstrap import require
from .intake import sha256
from .risk_replay_worker import native_identity
from .risk_score_worker import publish_seal, verify_audit_binding
from .risk_worker import checked_checkpoint, deterministic_runtime, forbid_annotations

ROOT = Path(__file__).resolve().parents[2]
DATASETS = source.DATASETS
ANALYSIS = general.BASE/"M03_ANALYSIS_PROTOCOL_v1.json"
ANALYSIS_SHA = "d302adb2178d79188e2eced086ffc24965d1e2f0738467435f6950870dab4fc3"


def source_inputs():
    inputs = source.source_inputs()
    forbidden = source.annotation_inodes(inputs)
    forbid_annotations(sorted(inputs["annotation_metadata"]))
    archive = legacy_rho.ARCHIVE
    require(archive.resolve(strict=True) == archive.absolute() and archive.is_file()
            and not archive.is_symlink(), "M03 archive source is not canonical")
    before = archive.stat()
    require((before.st_dev, before.st_ino) not in forbidden, "M03 archive aliases an annotation before read")
    for spec in inputs["specs"]:
        spec.update(id="M03."+spec["dataset"], modes=list(legacy_runtime.MODES))
    components = legacy_rho.load_components()
    after = archive.stat()
    identity = lambda state:(state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns)
    require(identity(before) == identity(after), "M03 archive identity changed during source loading")
    analysis, _ = protected.metadata(ANALYSIS, ANALYSIS_SHA, forbidden)
    require(analysis["conditions"] == list(legacy_runtime.MODES) and analysis["full_cells"] == 8
            and analysis["image_method_evaluations"] == 40420
            and analysis["bootstrap"]["replicates"] == 10000, "M03 prospective analysis differs")
    inputs["prospective_analysis_sha256"] = ANALYSIS_SHA
    inputs["legacy_archive"] = dict(path=str(legacy_rho.ARCHIVE),
        sha256=components["archive_sha256"], member_sha256=components["member_sha256"],
        episode_objective_sha256=legacy_runtime.OBJECTIVE_SHA,
        old_engine_loaded=False, old_optimizer_loaded=False, old_OOM_iterator_selected=False)
    validate_inputs(inputs)
    return inputs


def validate_inputs(inputs):
    require(inputs["GT_opened"] is False and inputs["old_predictions_reused"] is False,
            "M03 source isolation/reuse identity differs")
    require(inputs["prospective_analysis_sha256"] == ANALYSIS_SHA, "M03 prospective analysis binding differs")
    require([s["dataset"] for s in inputs["specs"]] == list(DATASETS), "M03 dataset membership differs")
    taxonomy = dict(coco_object="COCOObjectDataset", pc60="PascalContext60Dataset")
    original = copy.deepcopy(inputs["specs"])
    for spec, mirror in zip(inputs["specs"], original, strict=True):
        require(spec["id"] == "M03."+spec["dataset"] and spec["modes"] == list(legacy_runtime.MODES)
                and spec["config"]["dataset"] == taxonomy[spec["dataset"]], "M03 condition/taxonomy differs")
        mirror.update(id="M02."+spec["dataset"], modes=list(source.spatial_runtime.MODES))
    source.validate_inputs(inputs["rows"], original)
    archive = inputs["legacy_archive"]
    require(archive["path"] == str(legacy_rho.ARCHIVE)
            and archive["sha256"] == legacy_rho.ARCHIVE_SHA
            and set(archive["member_sha256"]) == set(legacy_rho.MEMBERS)
            and archive["episode_objective_sha256"] == legacy_runtime.OBJECTIVE_SHA
            and all(archive[k] is False for k in ("old_engine_loaded", "old_optimizer_loaded", "old_OOM_iterator_selected")),
            "M03 pinned archive scope differs")


def prepare(output):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "CPU-only M03 preparation required")
    inputs = source_inputs()
    forbid_annotations(sorted(inputs["annotation_metadata"]))
    output = Path(output).absolute()
    require(not output.exists() and output.resolve() == output, "Fresh canonical M03 suite required")
    output.mkdir()
    inputs_sha = protected.write_prepared_json(output/"INPUTS.json", inputs)
    protocol_sha = protected.write_prepared_json(output/"PROTOCOL.json", dict(experiment="M03_pinned_diagnostic_objective_full",
        datasets=list(DATASETS), modes=list(legacy_runtime.MODES), images=10105, cells=8,
        image_method_evaluations=40420, seed=0,
        algorithm="Current confidence TopK candidates, engine/Adam/LN/readout unchanged; C vs pinned legacy role/view/rho crossed with image vs episode normalization",
        legacy_scope=inputs["legacy_archive"],
        prospective_analysis_sha256=ANALYSIS_SHA,
        source="Exact clean source IDs/config/vocabulary from G metadata; native PILbilinear/L14last18, no old candidate or prediction reuse",
        reset="Original episode RNG restored per condition, exact theta0/Adam reset; no recovery of failed method",
        gates="Both datasets first4RGB/all4conditions, every actual finite step loss/gradient/parameters/Adam/final <=1e-5 and0changedlabels; C.image extra unchanged production parity",
        reference="Explicit masked mean arithmetic sharing pinned diagnostic/candidates, not a complete historical reproduction",
        timing="Synchronized native reset/candidates/probe/adaptation/finalreadout and common finite/actualAdam observation; decode/H2D/IO/GT excluded",
        barriers="All8 full predictions/global seals AND actual successful retained exit before new separately admitted CPU_GT scoring",
        resources=dict(single_shared_GPU=True, RAM_bytes=12*1024**3, swap_bytes=1024**3,
            shared_disk_reserve_bytes=general.RESERVE, prediction_artifact_bytes=general.MAX_PAYLOAD_BYTES,
            OOMPolicy="stop", Restart="no", RemainAfterExit=True),
        launch="No autolaunch. Exact G success plus named M02/M04 idle required in code; operator must verify actual M02/M04 completion before manually starting M03",
        failures="Single attempt; propagate all errors, discard failed method, retain partials without seal; no automatic retry/resume"))
    inventory = protected.guarded_snapshot(ROOT, output/"runtime", source.annotation_inodes(inputs))
    return protected.write_prepared_json(output/"SUITE_PREPARED.json", dict(schema_version=1, root=str(output),
        runtime_hashes=inventory, inputs_sha256=inputs_sha, protocol_sha256=protocol_sha,
        parent_native_hashes=native_identity(), GT_opened=False, GPU_started=False))


def admission(suite, expected_sha):
    require(not torch.cuda.is_initialized(), "M03 admission must precede CUDA initialization")
    current = source_inputs()
    forbidden = source.annotation_inodes(current)
    forbid_annotations(sorted(current["annotation_metadata"]))
    suite = Path(suite).resolve(strict=True)
    prepared, _ = protected.metadata(suite/"SUITE_PREPARED.json", expected_sha, forbidden)
    require(prepared["root"] == str(suite) and prepared["schema_version"] == 1
            and Path(__file__).resolve().is_relative_to(suite/"runtime"), "Wrong M03 exact runtime")
    protected.guarded_inventory(suite/"runtime", prepared["runtime_hashes"], forbidden)
    general.runtime_imports(suite/"runtime")
    inputs, _ = protected.metadata(suite/"INPUTS.json", prepared["inputs_sha256"], forbidden)
    protected.metadata(suite/"PROTOCOL.json", prepared["protocol_sha256"], forbidden)
    require(inputs == current and native_identity() == prepared["parent_native_hashes"], "M03 source/native drift")
    audit, audit_sha = protected.metadata(suite/"ADMITTED.json", forbidden=forbidden, limit=4*1024**2)
    require(audit["decision"] == "ADMITTED" and audit["suite_prepared_sha256"] == expected_sha
            and audit.get("independent_reviewer") and audit.get("M03_predictions_allowed") is True
            and audit.get("GT_scoring_allowed") is False, "Distinct exact M03 prediction admission required")
    report = suite/"INDEPENDENT_REPORT.md"
    require(audit["report_path"] == str(report), "M03 report path differs")
    identities = []
    for path in (suite/"ADMITTED.json", report):
        state = path.stat()
        require((state.st_dev, state.st_ino) not in forbidden, "M03 audit/report aliases GT")
        identities.append([state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns])
    binding = dict(admission_path=str(suite/"ADMITTED.json"), admission_sha256=audit_sha, admission_identity=identities[0],
        report_path=str(report), report_sha256=audit["report_sha256"], report_identity=identities[1],
        independent_reviewer=audit["independent_reviewer"])
    verify_audit_binding(binding)
    for spec in inputs["specs"]:
        checked_checkpoint(spec["checkpoint"])
    return suite, prepared, inputs, binding


def prior_GPU_terminal():
    previous = source.prior_GPU_terminal()
    observations = {}
    for unit in ("trace-spatial-20261102-2156.service", "trace-factorial-20261102-2243.service"):
        result = subprocess.run(["systemctl", "--user", "show", unit,
            "--property=LoadState,ActiveState,SubState,MainPID"], check=True, text=True, capture_output=True, timeout=30)
        actual = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        require(actual.get("MainPID") == "0" and actual.get("ActiveState") in ("active", "inactive", "failed")
                and actual.get("SubState") in ("exited", "dead", "failed"), "Another named GPU worker is not idle")
        observations[unit] = actual
    return dict(G_completion=previous, named_worker_idle=observations,
                idle_is_not_scientific_completion=True)


def build_method(spec):
    base = native.build_method(spec)
    result = legacy_runtime.LegacyExperimentTRACE(dataset_name=spec["config"]["dataset"],
        benchmark_classes=spec["vocabulary"]["classes"], query_to_class=spec["vocabulary"]["query_to_class"],
        classes=base.classes, lr=base.lr, steps=base.steps, trace_max_regions=base.trace_max_regions,
        trace_view_batch_size=base.trace_view_batch_size, vision_outputs=base.vision_outputs,
        prompts=base.prompt_templates, model=base.model, tokenize_fn=base.tokenize, device=base.device, view_mode="C")
    result.model.eval()
    native.legal_parameters(result)
    return result


def gates(suite, expected_sha, inputs):
    children = {}
    (suite/"gates").mkdir()
    for spec in inputs["specs"]:
        rows, cfg = inputs["rows"][spec["dataset"]][:4], spec["config"]
        with gpu_lock() as lock:
            deterministic_runtime()
            method = None
            try:
                method = build_method(spec)
                data, reports = confirmation.make_data(cfg, rows), []
                for start in range(0, len(data), cfg["batch_size"]):
                    images, meta, valid, ids = general.episode_batch(data, start, min(start+cfg["batch_size"], len(data)), method, cfg)
                    reports.append(legacy_gate.compare_gate(method, images, valid, ids, meta, cfg,
                                                            spec["vocabulary"]["query_to_class"]))
                require(len(data) == 4 and bool(reports), "M03 needs actual4RGB per dataset")
                record = dict(passed=True, dataset=spec["dataset"], suite_prepared_sha256=expected_sha,
                    actual_RGB_sources=4, ordered_ids=[r["official_id"] for r in rows], comparisons=reports,
                    legacy_archive=inputs["legacy_archive"], gpu_lock=lock, GT_opened=False)
            finally:
                del method
                if torch.cuda.is_initialized(): torch.cuda.empty_cache()
        path = suite/"gates"/(spec["dataset"]+".json")
        publish_seal(path, record)
        children[spec["dataset"]] = sha256(path)
    path = suite/"ALL_GATES_PASSED.json"
    publish_seal(path, dict(complete=True, suite_prepared_sha256=expected_sha, datasets=children, GT_opened=False))
    return dict(barrier_sha256=sha256(path), datasets=children)


def run(suite, expected_sha):
    prior = prior_GPU_terminal()
    suite, prepared, inputs, audit = admission(suite, expected_sha)
    require(not any((suite/n).exists() for n in ("STARTED.json", "FAILED.json", "ALL_PREDICTIONS_SEALED.json")), "No M03 resume/retry")
    require(shutil.disk_usage(suite).free >= general.RESERVE+general.MAX_PAYLOAD_BYTES, "M03 disk headroom insufficient")
    verify_audit_binding(audit)
    protected.write_prepared_json(suite/"STARTED.json", dict(suite_prepared_sha256=expected_sha, audit=audit,
        prior_GPU=prior, pid=os.getpid(), GT_opened=False, automatic_retry=False))
    started_sha, forbidden = sha256(suite/"STARTED.json"), source.annotation_inodes(inputs)
    try:
        gate_binding = gates(suite, expected_sha, inputs)
        source.verify_gates(suite, expected_sha, gate_binding, forbidden)
        budget, seals = dict(bytes=0), {}
        for spec in inputs["specs"]:
            with gpu_lock() as lock:
                deterministic_runtime()
                method = None
                try:
                    method = build_method(spec)
                    if method.device.type == "cuda": torch.cuda.reset_peak_memory_stats(method.device)
                    data = confirmation.make_data(spec["config"], inputs["rows"][spec["dataset"]])
                    begin = time.perf_counter()
                    result = legacy_prediction.execute_dataset(method, data, inputs["rows"][spec["dataset"]], spec, suite, budget,
                        lambda row:write_json(suite/"queue_state.json", dict(phase="M03_paired_full_prediction", pid=os.getpid(), **row), replace=True))
                    require(result["payloads_complete"] is True and result["GT_opened"] is False
                            and result["dataset"] == spec["dataset"] and result["images"] == spec["images"]
                            and result["image_method_evaluations"] == spec["images"]*4
                            and result["modes"] == list(legacy_runtime.MODES), "Incomplete M03 full result")
                    result.update(complete=True, suite_prepared_sha256=expected_sha, gpu_lock=lock,
                        resource_observation=dict(dataset_model_and_IO_wall_seconds=time.perf_counter()-begin,
                            process_peak_RSS_bytes_cumulative=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                            GPU_peak_allocated_bytes=None if method.device.type != "cuda" else torch.cuda.max_memory_allocated(method.device)))
                finally:
                    del method
                    if torch.cuda.is_initialized(): torch.cuda.empty_cache()
            path = suite/spec["dataset"]/"PREDICTIONS_SEALED.json"
            publish_seal(path, result)
            seals[spec["dataset"]] = sha256(path)
        require(set(seals) == set(DATASETS) and source_inputs() == inputs
                and native_identity() == prepared["parent_native_hashes"], "M03 source/native drift or missing dataset")
        protected.guarded_inventory(suite/"runtime", prepared["runtime_hashes"], forbidden)
        protected.metadata(suite/"SUITE_PREPARED.json", expected_sha, forbidden)
        protected.metadata(suite/"INPUTS.json", prepared["inputs_sha256"], forbidden)
        protected.metadata(suite/"PROTOCOL.json", prepared["protocol_sha256"], forbidden)
        general.runtime_imports(suite/"runtime")
        verify_audit_binding(audit)
        require(sha256(suite/"STARTED.json") == started_sha, "M03 startup changed")
        source.verify_gates(suite, expected_sha, gate_binding, forbidden)
        for dataset, digest in seals.items(): protected.metadata(suite/dataset/"PREDICTIONS_SEALED.json", digest, forbidden)
        write_json(suite/"queue_state.json", dict(phase="M03_payloads_complete_awaiting_global_seal_and_exit",
            datasets_complete=2, GT_opened=False, pid=os.getpid()), replace=True)
        publish_seal(suite/"ALL_PREDICTIONS_SEALED.json", dict(complete=True, suite_prepared_sha256=expected_sha,
            dataset_seals=seals, cells=len(inputs["specs"])*4,
            image_method_evaluations=sum(s["images"]*4 for s in inputs["specs"]), artifact_bytes=budget["bytes"],
            gate_binding=gate_binding, audit_binding=audit, started_sha256=started_sha,
            legacy_archive=inputs["legacy_archive"], GT_opened=False, scientific_completion=False))
    except BaseException as error:
        protected.write_prepared_json(suite/"FAILED.json", dict(error=f"{type(error).__name__}: {error}",
            GT_opened=False, automatic_retry=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "validate-admission", "run"))
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--suite-sha256")
    args = parser.parse_args()
    if args.phase == "prepare":
        print(dict(prepared_sha256=prepare(args.suite), GPU_started=False, GT_opened=False))
    else:
        require(args.suite_sha256, "Exact M03 prepared SHA required")
        if args.phase == "validate-admission":
            admission(args.suite, args.suite_sha256)
            print(dict(admission_valid=True, CUDA_initialized=torch.cuda.is_initialized(), GPU_started=False, GT_opened=False))
        else:
            run(args.suite, args.suite_sha256)
