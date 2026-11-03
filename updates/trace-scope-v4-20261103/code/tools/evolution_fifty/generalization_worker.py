"""Fresh primary G01/G02/G04 prediction stage, never a GT scoring stage.

CPU preparation is not admission. Actual GPU execution requires independent
exact-artifact admission and the existing shared single-worker lock.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
import torch

from tools.evolution import run_cell as shared
from tools.evolution_confirmation import run as confirmation
from tools.evolution_followup.queue import snapshot
from tools.evolution_followup.standalone import gpu_lock
from tools.evolution_tuning.common import write_json
from . import generalization_runtime as runtime
from .bootstrap import TUNED, POPULATIONS, require
from .generalization import ARMS, CORRUPTIONS, BACKBONES, cell_matrix, visual_layers, protocol as general_protocol
from .generalization_data import GeneralizationRGBDataset
from .intake import sha256, write_new
from .risk_replay_manifest import bound_json
from .risk_replay_worker import native_identity
from .risk_score_payload import stable_bytes
from .risk_score_worker import publish_seal, verify_audit_binding
from .risk_worker import checked_checkpoint, deterministic_runtime, forbid_annotations, strict_runtime_inventory
from .sclip_reference import SOURCE as SCLIP_SOURCE, SOURCE_SHA as SCLIP_SHA

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/home/wwww/hzl/results/trace_mechanism_20261102")
CLEAN_ROOT = BASE/"G01_G02_inputs_v1"
GEN_ROOT = BASE/"G02_fresh_RGB_20261102_1607"
GEN_SEAL_SHA = "0b271e92ec1ebf97e62bb7a2828676da1854b1e5283084b9088b40b147d763ef"
GEN_TERMINAL = BASE/"G02_CPU_generation_terminal_001.json"
MAX_PAYLOAD_BYTES, RESERVE = 32*1024**3, 16*1024**3
EXPECTED_CONDITIONS, EXPECTED_CELLS, EXPECTED_EVALUATIONS = 19,57,104214


def vocabulary(panel):
    return {k:copy.deepcopy(panel[k]) for k in ("classes", "queries", "query_to_class", "reduce_zero_label")}


def runtime_imports(root):
    root = Path(root).resolve(strict=True)
    for name,module in list(sys.modules.items()):
        if name == "trace_ovss" or name.startswith(("trace_ovss.","tools.evolution","tools.trace_","tools.suite_inputs")):
            file = getattr(module,"__file__",None)
            if file:
                require(Path(file).resolve().is_relative_to(root), "G imported executable source outside exact runtime")


def condition_specs(clean, checkpoints):
    """Group the fixed57arm-cells into19paired-input conditions, no subsets."""
    result = []
    for row in cell_matrix():
        if row["stage"] != "primary" or row["arm"] != ARMS[0]:
            continue
        dataset, backbone, model = row["dataset"], row["backbone"], row["model_type"]
        key = f"{model}.{backbone.replace('/', '-')}.{dataset}.{row['corruption']}.s{row['severity']}"
        result.append(dict(id=key, dataset=dataset, backbone=backbone, model_type=model,
            corruption=row["corruption"], severity=row["severity"], images=row["images"], arms=list(ARMS),
            input_key=f"clean.{dataset}" if row["corruption"] == "original" else f"corruption.{row['corruption']}",
            config=copy.deepcopy(clean[dataset]["config"]), vocabulary=vocabulary(clean[dataset]["panel"]),
            vision_outputs=visual_layers(backbone), checkpoint=copy.deepcopy(checkpoints[backbone])))
    require(len(result) == EXPECTED_CONDITIONS and sum(r["images"]*len(ARMS) for r in result) == EXPECTED_EVALUATIONS,
            "Primary G matrix differs")
    return result


def validate_conditions(specs):
    from trace_ovss.models._vendor_clip.clip import _MODELS
    expected = [r for r in cell_matrix() if r["stage"] == "primary" and r["arm"] == ARMS[0]]
    require(len(specs) == len(expected) == EXPECTED_CONDITIONS
            and len({s["id"] for s in specs}) == len(specs), "G condition membership differs")
    checkpoints, configs = {},{}
    for spec,row in zip(specs,expected,strict=True):
        for key in ("dataset","backbone","model_type","corruption","severity","images","vision_outputs"):
            require(spec[key] == row[key], "G primary condition definition differs")
        require(spec["arms"] == list(ARMS) and spec["checkpoint"]["backbone"] == spec["backbone"],
                "G internal arms/checkpoint backbone differ")
        url = _MODELS[spec["backbone"]]
        require(spec["checkpoint"]["sha256"] == url.split("/")[-2]
                and spec["checkpoint"]["official_url"] == url
                and Path(spec["checkpoint"]["path"]).name == url.rsplit("/",1)[1],
                "G checkpoint is not the pinned official pretrained model")
        canonical = f"{spec['model_type']}.{spec['backbone'].replace('/', '-')}.{spec['dataset']}.{spec['corruption']}.s{spec['severity']}"
        input_key = f"clean.{spec['dataset']}" if spec["corruption"] == "original" else f"corruption.{spec['corruption']}"
        require(spec["id"] == canonical and spec["input_key"] == input_key, "G condition/input identity differs")
        runtime.image_worker.validate_config(spec["config"])
        dataset = spec["dataset"]
        anchor = (spec["config"],spec["vocabulary"])
        require(dataset not in configs or configs[dataset] == anchor,
                "G condition-specific settings or vocabulary tuning forbidden")
        configs[dataset] = anchor
        backbone = spec["backbone"]
        require(backbone not in checkpoints or checkpoints[backbone] == spec["checkpoint"], "G checkpoint varies bycondition")
        checkpoints[backbone] = spec["checkpoint"]
    require(sum(s["images"]*len(ARMS) for s in specs) == EXPECTED_EVALUATIONS
            and len(specs)*len(ARMS) == EXPECTED_CELLS, "G primary population/evaluation count differs")


def label_identity(row):
    path = Path(row["label"])
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "Canonical G annotation metadata required")
    status, details = path.stat(), row["label_metadata_details"]
    require([status.st_size, status.st_mtime_ns] == row["label_metadata"]
            and details == dict(size_bytes=status.st_size, mtime_ns=status.st_mtime_ns,
                device=status.st_dev, inode=status.st_ino, contents_read=False), "Frozen G label metadata differs")
    return status.st_dev, status.st_ino


def validate_inputs(inputs, conditions):
    validate_conditions(conditions)
    require(set(inputs) == {"clean.coco_object", "clean.pc60", "clean.voc20"}
            | {f"corruption.{c}" for c in CORRUPTIONS}, "G input-set membership differs")
    forbidden = {label_identity(r) for rows in inputs.values() for r in rows}
    for spec in conditions:
        cfg, rows = spec["config"], inputs[spec["input_key"]]
        require(spec["images"] == len(rows) == POPULATIONS[spec["dataset"]]
                and len({r["official_id"] for r in rows}) == len(rows)
                and len({r["image"] for r in rows}) == len(rows), "Exact G full population required")
        require(cfg["resize"] == cfg["patch_size"] == [224,224] and spec["arms"] == list(ARMS),
                "G geometry/arms differ")
        for row in rows:
            path = Path(row["image"])
            require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "Canonical G RGB path required")
            status = path.stat()
            require((status.st_dev,status.st_ino) not in forbidden and 0 < status.st_size <= 32*1024**2,
                    "G RGB aliases GT or exceeds bound")
        if spec["corruption"] == "original":
            require(all("corruption" not in r for r in rows), "Clean G inputs silently corrupted")
        else:
            original = inputs["clean.voc20"]
            for index,(clean,row) in enumerate(zip(original,rows,strict=True)):
                require(row["official_id"] == clean["official_id"] and row["label"] == clean["label"]
                        and row["label_metadata_details"] == clean["label_metadata_details"]
                        and row["clean_image"] == clean["image"] and row["clean_image_sha256"] == clean["image_sha256"]
                        and row["corruption"] == spec["corruption"] and row["severity"] == 5
                        and row["original_sample_idx"] == index
                        and row["preprocessing"] == "already_corrupted_and_cv2_resized_no_second_resize",
                        "Fresh corrupted observation/source/order binding differs")
    return sorted({r["label"] for rows in inputs.values() for r in rows})


def prepare(output):
    output = Path(output).absolute()
    require(not output.exists() and output.resolve() == output, "Fresh canonical G suite required")
    receipt = json.loads(stable_bytes(CLEAN_ROOT/"INPUTS_PREPARED.json", 4*1024**2)[0])
    clean = bound_json(CLEAN_ROOT/"CLEAN_INPUTS.json", receipt["artifact_sha256"]["CLEAN_INPUTS.json"])
    checkpoints = bound_json(CLEAN_ROOT/"CHECKPOINTS.json", receipt["artifact_sha256"]["CHECKPOINTS.json"])
    for dataset in ("coco_object", "pc60", "voc20"):
        parent = TUNED/"full"/dataset
        seal = json.loads(stable_bytes(parent/"PREDICTIONS_SEALED.json", 16*1024**2)[0])
        original = bound_json(parent/"prediction_protocol.json", seal["artifact_sha256"]["prediction_protocol.json"])["manifest"]
        require(seal["complete"] is True and seal["gt_contents_opened"] is False
                and original["config"] == clean[dataset]["config"] and original["panel"] == clean[dataset]["panel"],
                "Clean settings/population not inherited from sealed parent")
    generated = bound_json(GEN_ROOT/"INPUTS_SEALED.json", GEN_SEAL_SHA)
    terminal = json.loads(stable_bytes(GEN_TERMINAL, 1024**2)[0])
    require(generated["complete"] is True and generated["generated_RGBs"] == 21735
            and generated["labels_read"] is False and generated["GPU_used"] is False
            and terminal["input_seal_sha256"] == GEN_SEAL_SHA
            and terminal["CPU_input_generation_successfully_completed"] is True
            and all(terminal["service"][k] == v for k,v in dict(LoadState="loaded",ActiveState="active",SubState="exited",
                MainPID=0, ExecMainPID=1832234, ExecMainCode=1, ExecMainStatus=0, Result="success",
                InvocationID="f4710c4ba93b48b5ac3d9bc5e8e8a91f").items()), "Fresh input-generation terminal receipt differs")
    corrupted = bound_json(GEN_ROOT/"VOC20_C_S5_INPUTS.json", generated["artifact_sha256"]["VOC20_C_S5_INPUTS.json"])
    inputs = {f"clean.{d}":v["panel"]["inputs"] for d,v in clean.items()}
    inputs.update({f"corruption.{c}":rows for c,rows in corrupted.items()})
    specs = condition_specs(clean,checkpoints)
    labels = validate_inputs(inputs,specs)
    core = native_identity()
    for checkpoint in checkpoints.values():
        checked_checkpoint(checkpoint)
    require(sha256(SCLIP_SOURCE) == SCLIP_SHA, "Pinned official SCLIP reference changed")
    output.mkdir()
    (output/"reference").mkdir()
    shutil.copyfile(SCLIP_SOURCE, output/"reference/sclip_model.py")
    write_new(output/"INPUTS.json", inputs)
    write_new(output/"CONDITIONS.json", specs)
    write_new(output/"PROTOCOL.json", dict(experiment="G01_G02_G04_primary", prospective_design=general_protocol(),
        executable_prediction_stage=True,actual_GPU_started=False,scientific_completion=False,
        conditions=19, cells=57, image_method_evaluations=104214, ordered_arms=list(ARMS), seed=0,
        transfer="Exact historical per-dataset config; no search or test selection. Last9/12 B16,18/24 L14",
        observations="Corrupted input itself generates theta0 candidates/confidence/C, is adapted, and is predicted; no clean guidance",
        gates="All19conditions each4realRGB sources: NoAdapt and paired finite Confidence/C loss/gradient/parameter/Adam/finalparams<=1e-5 and0changedlabels, actual14/16token geometry. No vacuous adaptation gate",
        official_SCLIP="Pinned official final-layer features/input-LN gradients/querylabels gate on same actual weights/RGB; distinct from TRACE9layer readout, never claim published wholepipeline equality",
        timing="Synchronized reset-through-final-readout model time includes finite observation; excludes decode/H2D/receipts/GT. Report all raw totals and exclude episode0 only from warmup-adjusted descriptive throughput",
        outputs="One uint16 hard prediction per arm/source, packed original candidates per adapted arm/episode, raw actual work/timing; noGT arrays",
        GT="No GT bytes in prepare/gates/predict. All57fullcells sealed AND actual successful exit then separate independently admitted offlineCPUscore; VOC20cv2nearest/backgroundignore unchanged",
        reuse="Reuse only bound source inputs/config and completed generation receipt; no old model predictions/candidates, no historical corruption cache",
        limits=dict(prediction_artifact_bytes=MAX_PAYLOAD_BYTES,shared_disk_reserve_bytes=RESERVE,
            RAM_bytes=12*1024**3,swap_bytes=1024**3,Restart="no",OOMPolicy="stop",single_shared_GPU_lock=True),
        failures="No resume/retry/fallback batch reduction; all failed/negative/null evidence retained; no scoring entrypoint"))
    inventory = snapshot(ROOT,output/"runtime")
    write_new(output/"SUITE_PREPARED.json", dict(schema_version=1, root=str(output),runtime_hashes=inventory,
        inputs_sha256=sha256(output/"INPUTS.json"),conditions_sha256=sha256(output/"CONDITIONS.json"),
        protocol_sha256=sha256(output/"PROTOCOL.json"),reference_sha256=SCLIP_SHA,
        parent_native_hashes=core, source_clean_receipt_sha256=sha256(CLEAN_ROOT/"INPUTS_PREPARED.json"),
        fresh_RGB_seal_sha256=GEN_SEAL_SHA, fresh_RGB_terminal_receipt_sha256=sha256(GEN_TERMINAL),
        forbidden_labels=labels, GPU_started=False, GT_opened=False))
    return sha256(output/"SUITE_PREPARED.json")


def admission(suite, expected_sha):
    suite = Path(suite).resolve(strict=True)
    prepared = bound_json(suite/"SUITE_PREPARED.json",expected_sha)
    require(prepared["root"] == str(suite) and prepared["schema_version"] == 1
            and Path(__file__).resolve().is_relative_to(suite/"runtime")
            and strict_runtime_inventory(suite/"runtime") == prepared["runtime_hashes"], "Wrong or changed G runtime")
    runtime_imports(suite/"runtime")
    bound_json(suite/"PROTOCOL.json",prepared["protocol_sha256"])
    specs = bound_json(suite/"CONDITIONS.json",prepared["conditions_sha256"])
    inputs = bound_json(suite/"INPUTS.json",prepared["inputs_sha256"])
    require(native_identity() == prepared["parent_native_hashes"] and sha256(suite/"reference/sclip_model.py") == SCLIP_SHA,
            "G native or official SCLIP source changed")
    labels = validate_inputs(inputs,specs)
    require(labels == prepared["forbidden_labels"], "G annotation exclusion set changed")
    forbid_annotations(labels)
    receipt = bound_json(CLEAN_ROOT/"INPUTS_PREPARED.json",prepared["source_clean_receipt_sha256"])
    clean = bound_json(CLEAN_ROOT/"CLEAN_INPUTS.json",receipt["artifact_sha256"]["CLEAN_INPUTS.json"])
    checkpoints = bound_json(CLEAN_ROOT/"CHECKPOINTS.json",receipt["artifact_sha256"]["CHECKPOINTS.json"])
    require(specs == condition_specs(clean,checkpoints)
            and all(inputs[f"clean.{d}"] == clean[d]["panel"]["inputs"] for d in clean),
            "G config/input no longer equals its historical clean anchor")
    require(prepared["fresh_RGB_seal_sha256"] == GEN_SEAL_SHA, "Wrong fresh G RGB seal identity")
    generated = bound_json(GEN_ROOT/"INPUTS_SEALED.json",GEN_SEAL_SHA)
    corrupted = bound_json(GEN_ROOT/"VOC20_C_S5_INPUTS.json",generated["artifact_sha256"]["VOC20_C_S5_INPUTS.json"])
    require(set(corrupted) == set(CORRUPTIONS)
            and all(inputs[f"corruption.{c}"] == corrupted[c] for c in CORRUPTIONS),
            "G corrupted inputs differ from the actual completed fresh generator manifest")
    bound_json(GEN_TERMINAL,prepared["fresh_RGB_terminal_receipt_sha256"])
    audit_path = suite/"ADMITTED.json"
    state = audit_path.stat()
    audit_identity = (state.st_dev,state.st_ino,state.st_size,state.st_mtime_ns)
    audit_raw,audit_sha = stable_bytes(audit_path,4*1024**2,expected_identity=audit_identity)
    audit = json.loads(audit_raw)
    require(audit["decision"] == "ADMITTED" and audit["suite_prepared_sha256"] == expected_sha
            and audit.get("independent_reviewer") and audit.get("G_primary_predictions_allowed") is True
            and audit.get("GT_scoring_allowed") is False, "Independent exact G prediction artifact admission required")
    report = Path(audit["report_path"])
    require(report == suite/"INDEPENDENT_REPORT.md", "Wrong G independent report path")
    state = report.stat()
    report_identity = (state.st_dev,state.st_ino,state.st_size,state.st_mtime_ns)
    stable_bytes(report,4*1024**2,audit["report_sha256"],expected_identity=report_identity)
    for checkpoint in {s["backbone"]:s["checkpoint"] for s in specs}.values():
        checked_checkpoint(checkpoint)
    return suite,prepared,specs,inputs,dict(admission_path=str(audit_path),admission_sha256=audit_sha,
        admission_identity=list(audit_identity),report_path=str(report),report_identity=list(report_identity),
        report_sha256=audit["report_sha256"],independent_reviewer=audit["independent_reviewer"])


def episode_batch(data,start,stop,method,cfg):
    from trace_ovss.cli import build_patch_image_ids, build_patch_valid_mask
    batch = shared.image_collate([data[i] for i in range(start,stop)])
    images,meta = batch["img_patches"].to(method.device),batch["meta"]
    valid = build_patch_valid_mask(meta,cfg["patch_size"],cfg["patch_stride"],method.device)
    return images,meta,valid,build_patch_image_ids(meta,method.device)


def gates(suite,expected_sha,specs,inputs):
    seals = {}
    (suite/"gates").mkdir()
    for spec in specs:
        rows,cfg = inputs[spec["input_key"]][:4],spec["config"]
        reports, official = [],None
        with gpu_lock() as lock:
            deterministic_runtime()
            method = runtime.build_method(spec)
            reference = runtime.make_reference(method)
            try:
                data = GeneralizationRGBDataset(cfg,rows)
                patch = BACKBONES[spec["backbone"]][1]
                for start in range(0,len(data),cfg["batch_size"]):
                    images,meta,valid,ids = episode_batch(data,start,min(start+cfg["batch_size"],len(data)),method,cfg)
                    if spec["model_type"] == "sclip" and official is None:
                        official = runtime.official_sclip_gate(method,images,suite/"reference/sclip_model.py")
                    reports.append(runtime.compare_gate(method,reference,images,valid,ids,meta,cfg,
                        spec["vocabulary"]["query_to_class"],tuple(n//patch for n in cfg["patch_size"])))
                require(len(data) == 4, "Four actual RGB sources required in every G condition gate")
                record = dict(passed=True,condition=spec["id"],suite_prepared_sha256=expected_sha,
                    actual_RGB_sources=4, ordered_ids=[r["official_id"] for r in rows],
                    batch_images=[min(cfg["batch_size"],4-i) for i in range(0,4,cfg["batch_size"])],
                    comparisons=reports, official_SCLIP_component=official,
                    legal_visual_LN=runtime.legal_parameters(method),gpu_lock=lock,GT_opened=False)
            finally:
                del reference,method
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        path = suite/"gates"/(spec["id"]+".json")
        publish_seal(path,record)
        seals[spec["id"]] = sha256(path)
    barrier = suite/"ALL_GATES_PASSED.json"
    publish_seal(barrier,dict(complete=True,suite_prepared_sha256=expected_sha,
        conditions=seals,GT_opened=False))
    return dict(barrier_sha256=sha256(barrier),conditions=seals)


def write_payload(path,value,inventory,root,cost):
    if isinstance(value,np.ndarray):
        stream = io.BytesIO()
        np.save(stream,value,allow_pickle=False)
        content = stream.getvalue()
    else:
        content = (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    require(shutil.disk_usage(root).free >= RESERVE+len(content), "G write would consume shared disk reserve")
    require(cost["bytes"]+len(content) <= MAX_PAYLOAD_BYTES, "G prediction artifacts exceed stage cap")
    with path.open("xb") as stream:
        require(stream.write(content) == len(content), "Short G payload write")
    cost["bytes"] += path.stat().st_size
    require(cost["bytes"] <= MAX_PAYLOAD_BYTES, "Actual G artifact bytes exceed cap")
    inventory[str(path.relative_to(root))] = sha256(path)


def execute_condition(method,data,spec,output,suite,cost,progress):
    cfg,rows,inventory = spec["config"],data.rows,{}
    output.mkdir()
    for arm in ARMS:
        (output/"predictions"/arm).mkdir(parents=True)
    (output/"episodes").mkdir()
    method.reset()
    initial = shared.restored_state(method)
    frozen = [(p,p._version) for p in method.model.parameters() if not p.requires_grad]
    frozen_digest = shared.state_digest({n:p for n,p in method.model.named_parameters() if not p.requires_grad})
    legal = runtime.legal_parameters(method)
    write_payload(output/"initial_state.json",dict(reset=initial,legal_visual_LN=legal,
        frozen_parameter_sha256=frozen_digest),inventory,suite,cost)
    all_costs = []
    with (output/"cost.jsonl").open("x") as stream:
        for episode,start in enumerate(range(0,len(data),cfg["batch_size"])):
            stop = min(start+cfg["batch_size"],len(data))
            images,meta,valid,ids = episode_batch(data,start,stop,method,cfg)
            require(list(meta["img_path"]) == [r["image"] for r in rows[start:stop]], "G observed RGB order differs")
            rng,identity = shared.rng_state(method.device),None
            for position,arm in enumerate(runtime.prior.arm_order(ARMS,episode,"alternating")):
                shared.restore_rng(rng,method.device)
                predictions,stats,losses,row = runtime.run_arm(method,arm,images,valid,ids,meta,cfg,
                    spec["vocabulary"]["query_to_class"])
                row.update(episode=episode,order_position=position,warmup=episode == 0)
                if arm != "no_adapt":
                    identity = row["candidate_sha256"] if identity is None else identity
                    require(row["candidate_sha256"] == identity, "Full G C/Confidence candidate population differs")
                require(all(p._version == version for p,version in frozen), "Frozen G weights changed during adaptation")
                require(len(predictions) == stop-start, "Missing G source prediction")
                classes = len(spec["vocabulary"]["classes"])
                for i,prediction in enumerate(predictions,start):
                    require(prediction.shape == tuple(cfg["resize"]) and prediction.dtype == np.uint16
                            and bool((prediction < classes).all()), "Invalid mapped G hard prediction")
                    write_payload(output/"predictions"/arm/f"{i:06d}.npy",prediction,inventory,suite,cost)
                candidates = [] if arm == "no_adapt" else runtime.image_worker.candidate_receipts(stats,arm,
                    spec["vocabulary"]["query_to_class"])
                write_payload(output/"episodes"/f"{episode:06d}.{arm}.json",dict(**row,losses=losses,
                    ordered_ids=[r["official_id"] for r in rows[start:stop]],candidates=candidates,
                    no_adapt_candidate_status="not computed, not shared from adapted arms" if arm == "no_adapt" else None,
                    GT_opened=False),inventory,suite,cost)
                line = json.dumps(row,allow_nan=False)+"\n"
                size = len(line.encode())
                require(shutil.disk_usage(suite).free >= RESERVE+size and cost["bytes"]+size <= MAX_PAYLOAD_BYTES,
                        "G cost stream exceeds shared storage budget")
                stream.write(line)
                stream.flush()
                cost["bytes"] += size
                all_costs.append(row)
                method.reset()
                require(shared.restored_state(method) == initial, "G episode/arm reset differs")
            progress["completed_source_episodes"] += 1
            write_json(suite/"queue_state.json",dict(phase="paired_full_G_prediction",condition=spec["id"],
                sources_in_condition=stop,total_sources_in_condition=len(data),stored_bytes=cost["bytes"],
                **progress,pid=os.getpid(),GT_opened=False),replace=True)
            del images,meta,valid,ids
    require(shared.state_digest({n:p for n,p in method.model.named_parameters() if not p.requires_grad}) == frozen_digest,
            "G full-condition frozen parameter bytes changed")
    costpath = output/"cost.jsonl"
    inventory[str(costpath.relative_to(suite))] = sha256(costpath)
    write_payload(output/"prediction_result.json",dict(condition=spec["id"],images=len(data),arms=list(ARMS),
        actual_work={arm:{k:sum(r[k] for r in all_costs if r["arm"] == arm) for k in runtime.COUNTERS} for arm in ARMS},
        raw_model_seconds={arm:sum(r["seconds"] for r in all_costs if r["arm"] == arm) for arm in ARMS},
        warmup_adjusted_throughput=runtime.prior.timing_summary(all_costs,ARMS),
        legal_visual_LN=legal,frozen_parameter_sha256=frozen_digest,GT_opened=False,
        metric_status="No GT or score; all source predictions including episode0 retained"),inventory,suite,cost)
    expected_episodes=(len(data)+cfg["batch_size"]-1)//cfg["batch_size"]
    require(len(all_costs) == expected_episodes*len(ARMS)
            and all(sum(r["images"] for r in all_costs if r["arm"] == arm) == spec["images"] for arm in ARMS),
            "Incomplete per-arm G full population")
    prefix = str(output.relative_to(suite))+"/"
    expected_names = {prefix+f"predictions/{a}/{i:06d}.npy" for a in ARMS for i in range(len(data))}
    expected_names |= {prefix+f"episodes/{e:06d}.{a}.json" for a in ARMS for e in range(expected_episodes)}
    expected_names |= {prefix+n for n in ("initial_state.json","cost.jsonl","prediction_result.json")}
    require(set(inventory) == expected_names, "G prediction inventory missing or unexpected artifacts")
    return dict(complete=True,condition=spec,images=len(data),image_method_evaluations=len(data)*len(ARMS),
        artifact_sha256=inventory,GT_opened=False)


def verify_gate_binding(suite,expected_sha,specs,binding):
    gate = bound_json(suite/"ALL_GATES_PASSED.json",binding["barrier_sha256"])
    require(gate["complete"] is True and gate["suite_prepared_sha256"] == expected_sha
            and gate["conditions"] == binding["conditions"]
            and set(gate["conditions"]) == {s["id"] for s in specs}, "Every G condition gate required before full prediction")
    for name,digest in gate["conditions"].items():
        bound_json(suite/"gates"/(name+".json"),digest)


def predict(suite,expected_sha,prepared,specs,inputs,audit,started_sha,gate_binding):
    require(sha256(suite/"STARTED.json") == started_sha, "G startup binding changed during gates")
    verify_gate_binding(suite,expected_sha,specs,gate_binding)
    (suite/"conditions").mkdir()
    cost,progress,seals = dict(bytes=0),dict(completed_conditions=0,completed_source_episodes=0),{}
    for spec in specs:
        condition_started = time.perf_counter()
        with gpu_lock() as lock:
            deterministic_runtime()
            method = runtime.build_method(spec)
            try:
                if method.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(method.device)
                data = GeneralizationRGBDataset(spec["config"],inputs[spec["input_key"]])
                result = execute_condition(method,data,spec,suite/"conditions"/spec["id"],suite,cost,progress)
                result.update(suite_prepared_sha256=expected_sha,gpu_lock=lock)
                result["resource_observation"] = dict(
                    condition_wall_seconds_including_lock_wait=time.perf_counter()-condition_started,
                    process_peak_RSS_bytes_cumulative_since_start=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                    GPU_peak_allocated_bytes=None if method.device.type != "cuda" else torch.cuda.max_memory_allocated(method.device),
                    GPU_peak_reserved_bytes=None if method.device.type != "cuda" else torch.cuda.max_memory_reserved(method.device),
                    GPU_peak_scope="Full condition after construction, including resident model; construction peak excluded; CPU RSS is cumulative")
            finally:
                del method
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        seal = suite/"conditions"/spec["id"]/"PREDICTIONS_SEALED.json"
        publish_seal(seal,result)
        seals[spec["id"]] = sha256(seal)
        progress["completed_conditions"] += 1
    require(progress["completed_conditions"] == len(specs) == EXPECTED_CONDITIONS, "Incomplete primary G stage")
    runtime_imports(suite/"runtime")
    require(sha256(suite/"SUITE_PREPARED.json") == expected_sha
            and strict_runtime_inventory(suite/"runtime") == prepared["runtime_hashes"]
            and sha256(suite/"INPUTS.json") == prepared["inputs_sha256"]
            and sha256(suite/"CONDITIONS.json") == prepared["conditions_sha256"]
            and sha256(suite/"PROTOCOL.json") == prepared["protocol_sha256"], "G source/input drift before sealing")
    verify_audit_binding(audit)
    require(sha256(suite/"STARTED.json") == started_sha, "G startup binding changed")
    write_json(suite/"queue_state.json",dict(phase="all_G_payloads_complete_awaiting_seal_and_exit",**progress,
        GT_opened=False,scientific_completion=False,pid=os.getpid()),replace=True)
    verify_gate_binding(suite,expected_sha,specs,gate_binding)
    for name,digest in seals.items():
        bound_json(suite/"conditions"/name/"PREDICTIONS_SEALED.json",digest)
    publish_seal(suite/"ALL_G_PREDICTIONS_SEALED.json",dict(complete=True,suite_prepared_sha256=expected_sha,
        condition_seals=seals,cells=EXPECTED_CELLS,image_method_evaluations=EXPECTED_EVALUATIONS,
        total_prediction_artifact_bytes=cost["bytes"],audit_binding=audit,started_sha256=started_sha,
        gate_binding=gate_binding,GT_opened=False,scientific_completion=False))


def run(suite,expected_sha):
    suite,prepared,specs,inputs,audit = admission(suite,expected_sha)
    require(not any((suite/n).exists() for n in ("STARTED.json","FAILED.json","ALL_G_PREDICTIONS_SEALED.json")),
            "No G resume/retry")
    require(shutil.disk_usage(suite).free >= MAX_PAYLOAD_BYTES+RESERVE, "Insufficient shared disk headroom for G")
    verify_audit_binding(audit)
    write_new(suite/"STARTED.json",dict(suite_prepared_sha256=expected_sha,audit=audit,pid=os.getpid(),
        GT_opened=False,automatic_retry=False))
    started_sha = sha256(suite/"STARTED.json")
    try:
        gate_binding = gates(suite,expected_sha,specs,inputs)
        predict(suite,expected_sha,prepared,specs,inputs,audit,started_sha,gate_binding)
    except BaseException as error:
        write_new(suite/"FAILED.json",dict(error=f"{type(error).__name__}: {error}",GT_opened=False,automatic_retry=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","run"))
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--suite-sha256")
    args = parser.parse_args()
    if args.phase == "prepare":
        print(dict(prepared_sha256=prepare(args.suite),GPU_started=False,GT_opened=False))
    else:
        require(args.suite_sha256,"Expected G suite SHA required")
        run(args.suite,args.suite_sha256)
