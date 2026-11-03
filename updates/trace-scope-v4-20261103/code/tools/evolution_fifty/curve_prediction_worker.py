"""G03-only ten-condition prediction controller; never a GT scorer.

No automatic launch or retry. CPU freeze additionally requires the operator's
actual completed V01 identity, so this stage cannot overtake the fixed queue.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

import torch

from . import corruption_curve as curve
from . import corruption_curve_worker as generated
from . import curve_prediction as core
from . import generalization_score_worker as protected
from . import generalization_worker as general
from . import spatial_score_worker as shared
from .bootstrap import require
from .risk_score_worker import publish_seal, verify_audit_binding
from .risk_worker import deterministic_runtime, forbid_annotations

ROOT = Path(__file__).resolve().parents[2]
BASE = protected.BASE
INPUT_ROOT = BASE/"G03_CPU_inputs_20261103_0701"
INPUT_SHA = "7ac8ba9b2610f212cfd031167098cbc22e8afae375c9e636345e7bd409093e55"
INPUT_SEAL = "fecf5943d0c2476616fcd1a51eba916505dd907aceb2dfe5fd3bf54ff9486689"
INPUT_RECEIPT = BASE/"G03_CPU_generation_terminal_001.json"
INPUT_RECEIPT_SHA = "7dc5d3e59b166f44321a5bd2d3f939aab91b8ed8935a1fcfcadbcc397fc11d80"
INPUT_ADMISSION = "2c764181a4d0174a73ab19061c6bdf53d053343361af1a6b03dc509ac7c082d1"
INPUT_REPORT = "6da3710a24f4538304b92067bb7a4438d8074d50a0614c32a4f893051771cc28"
INPUT_START = "1c9ac29f522f2a8fc386918b97dd660743c796b68057ab5d59bb493987b217f8"
V01 = BASE/"V01_value_20261103_0111"
V01_SHA = "f7632c77f32c288bc5bc414eddb22e9527b4bd0628975d57b0ac1124b3ec3eb4"
RESERVE, OUTPUT_CAP = general.RESERVE, general.MAX_PAYLOAD_BYTES
CONDITIONS, CELLS, EVALUATIONS = 10, 30, 43470
RESOURCES = dict(MemoryMax=12*1024**3,MemorySwapMax=1024**3,OOMPolicy="stop",
                 Restart="no",RemainAfterExit=True,KillMode="control-group")


def service(unit,pid,invocation):
    keys="LoadState,ActiveState,SubState,MainPID,ExecMainPID,ExecMainCode,ExecMainStatus,Result,InvocationID"
    result=subprocess.run(["systemctl","--user","show",unit,"--property="+keys],
                          check=True,capture_output=True,text=True,timeout=30)
    actual=dict(row.split("=",1) for row in result.stdout.splitlines() if "=" in row)
    expected=dict(LoadState="loaded",ActiveState="active",SubState="exited",MainPID="0",
        ExecMainPID=str(pid),ExecMainCode="1",ExecMainStatus="0",Result="success",InvocationID=invocation)
    require(actual == expected,"G03 prerequisite has not exited successfully with exact identity")
    return actual


def operator_pins(value):
    require(type(value) is dict and set(value) == {"pid","invocation","global_sha256"}
            and type(value["pid"]) is int and 0 < value["pid"] < 2**31
            and type(value["invocation"]) is str and len(value["invocation"]) == 32
            and all(c in "0123456789abcdef" for c in value["invocation"])
            and curve.digest(value["global_sha256"]),"Fresh operator-verified V01 PID/invocation/global pins required")
    return copy.deepcopy(value)


def prerequisite_status(pins):
    pins=operator_pins(pins)
    statuses={}
    for unit,pid,inv in (
        ("trace-generalization-primary-20261102-1954.service",2043872,"2f14c1b37a0741f39500d48dae7ff353"),
        ("trace-spatial-20261102-2156.service",2086569,"775d3e1d206d42e0801641992d4b4ed4"),
        ("trace-factorial-20261102-2243.service",2121324,"d1e5a7f8cff34dbb979545716d78bdd0"),
        ("trace-legacy-20261103-0017.service",2190696,"f32b507b0a0c46759dd6e52527c83a8f"),
        ("trace-value-20261103-0111.service",pins["pid"],pins["invocation"])):
        statuses[unit]=service(unit,pid,inv)
    return statuses


def source_inputs():
    service("trace-corruption-curve-inputs-20261103-0701.service",2207453,"7b63dd877a1f40a1a8d8d7c793a46ae1")
    original=generated.source_inputs()
    native=original["native"]
    data_forbidden=generated.exclusions(native,sources=False)
    forbidden=generated.exclusions(native)
    for root in (INPUT_ROOT/"runtime",INPUT_ROOT/"reference",protected.PARENT/"runtime"):
        forbidden.update(v[:2] for v in protected.source_members(root,data_forbidden).values())
    require(INPUT_ROOT.resolve(strict=True) == INPUT_ROOT and not (INPUT_ROOT/"FAILED.json").exists(),
            "G03 input parent failed or aliased")
    prepared,_=protected.metadata(INPUT_ROOT/"CPU_PREPARED.json",INPUT_SHA,forbidden)
    saved,_=protected.metadata(INPUT_ROOT/"SOURCE_INPUTS.json",prepared["inputs_sha256"],forbidden)
    require(prepared["root"] == str(INPUT_ROOT) and curve.same_provenance(saved,original),
            "G03 input source binding differs")
    protected.metadata(INPUT_ROOT/"PROTOCOL.json",prepared["protocol_sha256"],forbidden)
    protected.guarded_inventory(INPUT_ROOT/"runtime",prepared["runtime_hashes"],data_forbidden)
    protected.guarded_inventory(INPUT_ROOT/"reference",prepared["reference_sha256"],data_forbidden)
    seal,_=protected.metadata(INPUT_ROOT/"INPUTS_SEALED.json",INPUT_SEAL,forbidden)
    receipt,_=protected.metadata(INPUT_RECEIPT,INPUT_RECEIPT_SHA,forbidden)
    require(seal["complete"] is True and seal["prepared_sha256"] == INPUT_SHA
            and seal["generated_RGBs"] == curve.SOURCES*10 and seal["GT_opened"] is False
            and seal["GPU_used"] is False and receipt["input_seal_sha256"] == INPUT_SEAL
            and receipt["CPU_input_generation_successfully_completed"] is True,
            "G03 all-input seal or operator terminal receipt differs")
    audit=seal["audit_binding"]
    require(audit["admission_path"] == str(INPUT_ROOT/"ADMITTED.json")
            and audit["report_path"] == str(INPUT_ROOT/"INDEPENDENT_REPORT.md")
            and audit["admission_sha256"] == INPUT_ADMISSION and audit["report_sha256"] == INPUT_REPORT
            and seal["started_sha256"] == INPUT_START,"Wrong G03 input admission/start binding")
    verify_audit_binding(audit)
    admitted,_=protected.metadata(INPUT_ROOT/"ADMITTED.json",INPUT_ADMISSION,forbidden)
    start,_=protected.metadata(INPUT_ROOT/"STARTED.json",INPUT_START,forbidden)
    require(admitted["G03_CPU_inputs_allowed"] is True and admitted["GPU_allowed"] is False
            and admitted["GT_scoring_allowed"] is False and start["pid"] == 2207453
            and start["prepared_sha256"] == INPUT_SHA and start["audit"] == audit
            and start["GT_opened"] is False and start["GPU_used"] is False,"G03 input startup permission differs")
    pins=seal["artifact_identities"]
    require(set(pins) == set(seal["artifact_sha256"]) and len(pins) == curve.SOURCES*10+2,
            "G03 complete input inventory differs")
    for name,current in pins.items():
        require(generated.file_pin(INPUT_ROOT/name,forbidden,64*1024**2,single=True) == current,
                "G03 generated payload identity differs")
    require(len({tuple(v[:2]) for v in pins.values()}) == len(pins)
            and sum(v[2] for v in pins.values()) == seal["output_bytes"] <= generated.OUTPUT_CAP,
            "G03 input inode uniqueness or bytes differ")
    metadata_forbidden=forbidden|{tuple(v[:2]) for n,v in pins.items() if n.endswith(".png")}
    manifest,_=protected.metadata(INPUT_ROOT/"VOC20_C_CURVE_INPUTS.json",
        seal["artifact_sha256"]["VOC20_C_CURVE_INPUTS.json"],metadata_forbidden)
    curve.validate_manifest(manifest,original["rows"],str(INPUT_ROOT))
    names={str(Path(r["image"]).relative_to(INPUT_ROOT)) for rows in manifest.values() for r in rows}
    require(set(pins) == names|{"VOC20_C_CURVE_INPUTS.json","cost.json"},"G03 input membership mismatch")
    gp,_=protected.metadata(protected.PARENT/"SUITE_PREPARED.json",protected.PARENT_SHA,forbidden,
        expected_identity=protected.PARENT_METADATA_IDENTITIES["SUITE_PREPARED.json"])
    conditions,_=protected.metadata(protected.PARENT/"CONDITIONS.json",gp["conditions_sha256"],forbidden)
    general.validate_conditions(conditions)
    anchors={s["corruption"]:s for s in conditions if s["backbone"] == "ViT-L/14"
             and s["model_type"] == "naclip" and s["dataset"] == "voc20" and s["severity"] == 5}
    require(set(curve.CURVE_TYPES) <= set(anchors),"Missing fixed L14 S5 anchors")
    first=anchors[curve.CURVE_TYPES[0]]
    fields=("config","vocabulary","checkpoint","vision_outputs")
    require(all(all(curve.same_provenance(s[k],first[k]) for k in fields) for s in anchors.values()),
            "G03 anchor settings differ by corruption")
    shared.native_binding(gp["parent_native_hashes"],data_forbidden)
    specs=[]
    for row in curve.conditions():
        spec=copy.deepcopy(anchors[row["corruption"]])
        spec.update(id=f"naclip.ViT-L-14.voc20.{row['id']}",input_key=row["id"],severity=row["severity"])
        specs.append(spec)
    validate_specs(specs,manifest,first)
    return dict(native=native,rows=manifest,specs=specs,anchor=first,parent_native_hashes=gp["parent_native_hashes"],
        generated_input_identities=pins,input_prepared_sha256=INPUT_SHA,input_seal_sha256=INPUT_SEAL,
        input_terminal_receipt_sha256=INPUT_RECEIPT_SHA,G_prepared_sha256=protected.PARENT_SHA,GT_opened=False)


def validate_specs(specs,inputs,anchor):
    expected=[]
    for row in curve.conditions():
        spec=copy.deepcopy(anchor)
        spec.update(id=f"naclip.ViT-L-14.voc20.{row['id']}",input_key=row["id"],
                    corruption=row["corruption"],severity=row["severity"])
        expected.append(spec)
    require(curve.same_provenance(specs,expected) and set(inputs) == {r["id"] for r in curve.conditions()}
            and len(specs) == CONDITIONS and sum(s["images"]*len(s["arms"]) for s in specs) == EVALUATIONS,
            "G03 fixed ten-condition/30-cell matrix differs")
    require(anchor["dataset"] == "voc20" and anchor["model_type"] == "naclip" and anchor["backbone"] == "ViT-L/14"
            and anchor["images"] == curve.SOURCES and anchor["arms"] == list(general.ARMS)
            and all(len(inputs[s["input_key"]]) == curve.SOURCES for s in specs),"G03 full population differs")


def data_exclusions(inputs):
    return generated.exclusions(inputs["native"],sources=False)


def prerequisite_bindings(pins,forbidden):
    statuses=prerequisite_status(pins)
    members=protected.source_members(V01/"runtime",forbidden)
    data_forbidden=set(forbidden)
    forbidden=forbidden|{v[:2] for v in members.values()}
    prepared,_=protected.metadata(V01/"SUITE_PREPARED.json",V01_SHA,forbidden)
    start,start_sha=protected.metadata(V01/"STARTED.json",forbidden=forbidden)
    seal,_=protected.metadata(V01/"ALL_PREDICTIONS_SEALED.json",pins["global_sha256"],forbidden)
    require(prepared["root"] == str(V01) and not (V01/"FAILED.json").exists()
            and start["suite_prepared_sha256"] == V01_SHA and type(start["pid"]) is int and start["pid"] == pins["pid"]
            and start["GT_opened"] is False and seal["complete"] is True and seal["GT_opened"] is False
            and seal["suite_prepared_sha256"] == V01_SHA and seal["started_sha256"] == start_sha
            and seal["cells"] == 10 and seal["image_method_evaluations"] == 50525
            and set(seal["dataset_seals"]) == {"coco_object","pc60"},"G03 V01 full barrier differs")
    protected.guarded_inventory(V01/"runtime",prepared["runtime_hashes"],data_forbidden)
    binding=seal["audit_binding"]
    require(binding["admission_path"] == str(V01/"ADMITTED.json")
            and binding["report_path"] == str(V01/"INDEPENDENT_REPORT.md"),"Wrong G03 V01 audit paths")
    admitted,_=protected.metadata(V01/"ADMITTED.json",binding["admission_sha256"],forbidden)
    generated.file_pin(V01/"INDEPENDENT_REPORT.md",forbidden,4*1024**2)
    require(admitted["decision"] == "ADMITTED" and admitted["suite_prepared_sha256"] == V01_SHA
            and admitted["V01_predictions_allowed"] is True and admitted["GT_scoring_allowed"] is False,
            "G03 V01 prediction permission differs")
    verify_audit_binding(seal["audit_binding"])
    require(seal["audit_binding"] == start["audit"],"G03 V01 startup audit differs")
    gate,_=protected.metadata(V01/"ALL_GATES_PASSED.json",seal["gate_binding"]["barrier_sha256"],forbidden)
    require(gate["complete"] is True and gate["GT_opened"] is False and gate["suite_prepared_sha256"] == V01_SHA
            and gate["datasets"] == seal["gate_binding"]["datasets"]
            and set(gate["datasets"]) == {"coco_object","pc60"},"G03 V01 gate barrier differs")
    for d,digest in gate["datasets"].items():
        child,_=protected.metadata(V01/"gates"/(d+".json"),digest,forbidden)
        require(child["passed"] is True and child["GT_opened"] is False and child["dataset"] == d
                and child["actual_RGB_sources"] == 4 and child["suite_prepared_sha256"] == V01_SHA,
                "G03 V01 actual gate child differs")
    for d,n in (("coco_object",5000),("pc60",5105)):
        child,_=protected.metadata(V01/d/"PREDICTIONS_SEALED.json",seal["dataset_seals"][d],forbidden)
        require(child["complete"] is True and child["GT_opened"] is False and child["dataset"] == d
                and child["suite_prepared_sha256"] == V01_SHA and child["images"] == n
                and child["image_method_evaluations"] == n*5,"G03 V01 full child differs")
    return dict(operator=operator_pins(pins),statuses=statuses,V01_started_sha256=start_sha)


def protocol(inputs,pins):
    return dict(experiment="G03_L14_fixed_severity_curves_prediction_only",conditions=CONDITIONS,cells=CELLS,
        evaluations=EVALUATIONS,input_seal_sha256=INPUT_SEAL,prior_V01=operator_pins(pins),
        same_original_config_and_candidate_algorithm=True,arms=list(general.ARMS),
        S5="Reuse sealed G1954 only during separately admitted analysis; never predicted in this suite",
        GT_allowed=False,NoAdapt="Zero update same TRACE readout, not literature plain NACLIP",
        resources=copy.deepcopy(RESOURCES),output_cap_bytes=OUTPUT_CAP,reserve_bytes=RESERVE,
        gates="Every10condition first4RGB, real finite updates/1e-5 and0changedlabels before any full prediction",
        failures="One externally capped manual service; own STARTED failures retained; discard failed models; no resume")


def prepare(output,prior_v01):
    generated.cpu_only()
    prerequisite_status(prior_v01)
    inputs=source_inputs()
    forbidden=data_exclusions(inputs)
    prior=prerequisite_bindings(prior_v01,forbidden)
    forbid_annotations(sorted(inputs["native"]["annotation_metadata"]))
    general.checked_checkpoint(inputs["anchor"]["checkpoint"])
    output=Path(output).absolute()
    require(output.resolve() == output and not output.exists(),"Fresh canonical G03 model suite required")
    require(shutil.disk_usage(output.parent).free >= RESERVE+OUTPUT_CAP,"G03 disk headroom insufficient")
    output.mkdir()
    ins=protected.write_prepared_json(output/"INPUTS.json",inputs)
    prot=protected.write_prepared_json(output/"PROTOCOL.json",protocol(inputs,prior_v01))
    inventory=protected.guarded_snapshot(ROOT,output/"runtime",forbidden)
    return protected.write_prepared_json(output/"SUITE_PREPARED.json",dict(schema_version=1,root=str(output),
        inputs_sha256=ins,protocol_sha256=prot,runtime_hashes=inventory,prior_binding=prior,
        prior_V01=operator_pins(prior_v01),GT_opened=False,GPU_started=False))


def admission(suite,expected_sha):
    require(not torch.cuda.is_initialized(),"G03 admission must precede CUDA initialization")
    inputs=source_inputs()
    forbidden=data_exclusions(inputs)
    suite=Path(suite).resolve(strict=True)
    members=protected.source_members(suite/"runtime",forbidden)
    metadata_forbidden=forbidden|{v[:2] for v in members.values()}
    prepared,_=protected.metadata(suite/"SUITE_PREPARED.json",expected_sha,metadata_forbidden)
    require(prepared["root"] == str(suite) and prepared["schema_version"] == 1
            and Path(__file__).resolve().is_relative_to(suite/"runtime"),"Wrong exact G03 runtime")
    require(prerequisite_bindings(prepared["prior_V01"],metadata_forbidden) == prepared["prior_binding"],
            "G03 exact prerequisite drift")
    saved,_=protected.metadata(suite/"INPUTS.json",prepared["inputs_sha256"],metadata_forbidden)
    plan,_=protected.metadata(suite/"PROTOCOL.json",prepared["protocol_sha256"],metadata_forbidden)
    require(curve.same_provenance(saved,inputs) and curve.same_provenance(plan,protocol(inputs,prepared["prior_V01"])),
            "G03 source or protocol drift")
    protected.guarded_inventory(suite/"runtime",prepared["runtime_hashes"],forbidden)
    general.runtime_imports(suite/"runtime")
    forbid_annotations(sorted(inputs["native"]["annotation_metadata"]))
    general.checked_checkpoint(inputs["anchor"]["checkpoint"])
    audit,audit_sha=protected.metadata(suite/"ADMITTED.json",forbidden=metadata_forbidden,limit=4*1024**2)
    require(audit["decision"] == "ADMITTED" and audit["suite_prepared_sha256"] == expected_sha
            and audit.get("independent_reviewer") and audit.get("G03_predictions_allowed") is True
            and audit.get("GT_scoring_allowed") is False,"New independent G03 model artifact admission required")
    report=suite/"INDEPENDENT_REPORT.md"
    require(audit["report_path"] == str(report),"Wrong G03 model report path")
    shared.bounded_content(report,audit["report_sha256"],metadata_forbidden,4*1024**2)
    ids=[generated.file_pin(p,metadata_forbidden,4*1024**2)[:4] for p in (suite/"ADMITTED.json",report)]
    binding=dict(admission_path=str(suite/"ADMITTED.json"),admission_sha256=audit_sha,admission_identity=ids[0],
        report_path=str(report),report_sha256=audit["report_sha256"],report_identity=ids[1],independent_reviewer=audit["independent_reviewer"])
    verify_audit_binding(binding)
    return suite,prepared,inputs,binding


def verify_gates(suite,expected_sha,inputs,binding):
    forbidden=data_exclusions(inputs)
    gate,_=protected.metadata(suite/"ALL_GATES_PASSED.json",binding["barrier_sha256"],forbidden)
    require(gate["complete"] is True and gate["GT_opened"] is False
            and gate["suite_prepared_sha256"] == expected_sha and gate["conditions"] == binding["conditions"]
            and set(gate["conditions"]) == {s["id"] for s in inputs["specs"]},"G03 all10 gate barrier required")
    for spec in inputs["specs"]:
        record,_=protected.metadata(suite/"gates"/(spec["id"]+".json"),binding["conditions"][spec["id"]],forbidden)
        require(record["passed"] is True and record["GT_opened"] is False and record["condition"] == spec["id"]
                and record["suite_prepared_sha256"] == expected_sha and record["actual_RGB_sources"] == 4
                and record["ordered_ids"] == [r["official_id"] for r in inputs["rows"][spec["input_key"]][:4]],
                "G03 actual four-RGB gate identity differs")
        require(len(record["comparisons"]) == (4+spec["config"]["batch_size"]-1)//spec["config"]["batch_size"],
                "G03 gate episode coverage differs")
        for episode in record["comparisons"]:
            require(set(episode) == set(general.ARMS),"G03 gate arm coverage differs")
            for arm,row in episode.items():
                require(row["passed"] is True and row["GT_opened"] is False and row["changed_labels"] == 0
                        and set(row["max_abs"]) == {"loss","final_parameter","gradient","parameter","adam","support"}
                        and all(type(v) in (int,float) and math.isfinite(v) and 0 <= v <= 1e-5 for v in row["max_abs"].values()),
                        "G03 numerical gate threshold differs")
                general.runtime.validate_cost(row["actual_counts"],arm,spec["config"])


def run(suite,expected_sha):
    suite,prepared,inputs,audit=admission(suite,expected_sha)
    require(not any((suite/n).exists() for n in ("STARTED.json","FAILED.json","ALL_G_PREDICTIONS_SEALED.json")),
            "No G03 model resume/retry")
    require(shutil.disk_usage(suite).free >= RESERVE+OUTPUT_CAP,"G03 execution disk headroom insufficient")
    owned=[]
    forbidden=data_exclusions(inputs)
    try:
        verify_audit_binding(audit)
        started,_=generated.write_output(suite,"STARTED.json",generated.encoded(dict(suite_prepared_sha256=expected_sha,
            audit=audit,pid=os.getpid(),GT_opened=False,automatic_retry=False,prior=prepared["prior_binding"])),
            dict(bytes=0),lambda:owned.append(True))
        gates=general.gates(suite,expected_sha,inputs["specs"],inputs["rows"])
        verify_gates(suite,expected_sha,inputs,gates)
        (suite/"conditions").mkdir()
        budget,progress,seals=dict(bytes=0),dict(completed_conditions=0,completed_source_episodes=0),{}
        for spec in inputs["specs"]:
            method=None
            begin=time.perf_counter()
            with general.gpu_lock() as lock:
                deterministic_runtime()
                try:
                    method=general.runtime.build_method(spec)
                    if method.device.type == "cuda":torch.cuda.reset_peak_memory_stats(method.device)
                    data=general.GeneralizationRGBDataset(spec["config"],inputs["rows"][spec["input_key"]])
                    result=core.execute_condition(method,data,spec,suite/"conditions"/spec["id"],suite,budget,progress,shared.write_json)
                    require(result["complete"] is True and result["GT_opened"] is False
                            and result["condition"] == spec and result["images"] == spec["images"]
                            and result["image_method_evaluations"] == spec["images"]*3,"Incomplete G03 full condition")
                    result.update(suite_prepared_sha256=expected_sha,gpu_lock=lock,resource_observation=dict(
                        condition_wall_seconds_including_lock_wait=time.perf_counter()-begin,
                        process_peak_RSS_bytes_cumulative_since_start=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                        GPU_peak_allocated_bytes=None if method.device.type != "cuda" else torch.cuda.max_memory_allocated(method.device)))
                finally:
                    del method
                    if torch.cuda.is_initialized():torch.cuda.empty_cache()
            path=suite/"conditions"/spec["id"]/"PREDICTIONS_SEALED.json"
            publish_seal(path,result)
            seals[spec["id"]]=general.sha256(path)
            progress["completed_conditions"]+=1
        shared.write_json(suite/"queue_state.json",dict(phase="G03_all_payloads_awaiting_seal_and_exit",**progress,
            GT_opened=False,pid=os.getpid()),replace=True)
        require(len(seals) == CONDITIONS and curve.same_provenance(source_inputs(),inputs),"G03 source drift or incomplete population")
        require(prerequisite_bindings(prepared["prior_V01"],forbidden) == prepared["prior_binding"],"G03 prerequisite drift before seal")
        protected.guarded_inventory(suite/"runtime",prepared["runtime_hashes"],forbidden)
        for name,digest in (("SUITE_PREPARED.json",expected_sha),("INPUTS.json",prepared["inputs_sha256"]),
                            ("PROTOCOL.json",prepared["protocol_sha256"]),("STARTED.json",started)):
            protected.metadata(suite/name,digest,forbidden)
        verify_audit_binding(audit)
        general.runtime_imports(suite/"runtime")
        verify_gates(suite,expected_sha,inputs,gates)
        for name,digest in seals.items():protected.metadata(suite/"conditions"/name/"PREDICTIONS_SEALED.json",digest,forbidden)
        publish_seal(suite/"ALL_G_PREDICTIONS_SEALED.json",dict(complete=True,suite_prepared_sha256=expected_sha,
            condition_seals=seals,cells=CELLS,image_method_evaluations=EVALUATIONS,total_prediction_artifact_bytes=budget["bytes"],
            audit_binding=audit,started_sha256=started,gate_binding=gates,input_seal_sha256=INPUT_SEAL,
            prior_binding=prepared["prior_binding"],GT_opened=False,scientific_completion=False))
    except BaseException as error:
        if owned:protected.write_prepared_json(suite/"FAILED.json",dict(error=f"{type(error).__name__}: {error}",GT_opened=False,automatic_retry=False))
        raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","validate-admission","run"))
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--suite-sha256")
    parser.add_argument("--prior-v01-pins",help="JSON operator-verified pid/invocation/global_sha256; only after actual V01 completion")
    args=parser.parse_args()
    if args.phase == "prepare":
        require(args.prior_v01_pins,"Exact completed V01 operator pins required")
        print(dict(prepared_sha256=prepare(args.suite,json.loads(args.prior_v01_pins)),GPU_started=False,GT_opened=False))
    else:
        require(args.suite_sha256,"Exact G03 model prepared SHA required")
        if args.phase == "run":run(args.suite,args.suite_sha256)
        else:
            admission(args.suite,args.suite_sha256)
            print(dict(admission_valid=True,CUDA_initialized=torch.cuda.is_initialized(),GT_opened=False,GPU_started=False))
