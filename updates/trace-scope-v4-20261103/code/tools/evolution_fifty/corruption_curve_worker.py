"""G03-only protected CPU input generator, no model or scoring entry point.

Each actual run needs a fresh frozen artifact and distinct independent CPU
input-generation permission. The old S5 inputs are references, never rewritten.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import threading
import time
import types

import cv2
import numpy as np
from PIL import Image
import torch

from . import corruption_curve as curve
from . import corruption_materialize as prior
from . import generalization_score_worker as protected
from . import generalization_worker as general
from . import legacy_score_payload as annotations
from . import spatial_score_worker as shared
from .bootstrap import require
from .corruption_parity import UPSTREAM_COMMIT
from .risk_score_payload import stable_bytes
from .risk_score_worker import publish_seal, verify_audit_binding
from .risk_worker import forbid_annotations

ROOT = Path(__file__).resolve().parents[2]
BASE = protected.BASE
PARENT = BASE/"G02_fresh_RGB_20261102_1607"
PARENT_SHA = "6c382852116f0f15a4390cfae779d9ed61a0305ffc8d892027a4fa4bf6a7bc18"
PARENT_SEAL = "0b271e92ec1ebf97e62bb7a2828676da1854b1e5283084b9088b40b147d763ef"
RECEIPT = BASE/"G02_CPU_generation_terminal_001.json"
RECEIPT_SHA = "e3a85521922b67dec382c9c9bd319d35f8958ff90d461c77863a4669072ae02c"
REFERENCE_NAMES = {"__init__.py","corruptions.py",*(f"frost/frost{i}.png" for i in (1,2,3)),
                   *(f"frost/frost{i}.jpg" for i in (4,5,6))}
RESERVE,OUTPUT_CAP = 16*1024**3,8*1024**3
EXECUTION_RESOURCES=dict(MemoryMax=12*1024**3,MemorySwapMax=1024**3,CPUQuota_percent=100,
    Nice=10,IO="idle",threads=[1,1,1],OOMPolicy="stop",Restart="no",RemainAfterExit=True,KillMode="control-group")


def cpu_only():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized()
            and threading.active_count() == 1, "G03 CPU generator requires hidden CUDA and serialized Python execution")


def identity(state):
    return [state.st_dev,state.st_ino,state.st_size,state.st_mtime_ns,state.st_ctime_ns,state.st_nlink,state.st_mode]


def file_pin(path,forbidden,limit,*,single=False):
    path=Path(path)
    state=path.lstat()
    require(path.is_absolute() and path.resolve(strict=True) == path and stat.S_ISREG(state.st_mode)
            and (state.st_dev,state.st_ino) not in forbidden and 0 < state.st_size <= limit
            and (not single or state.st_nlink == 1), "G03 file path/role/extent differs")
    return identity(state)


def data_roles(native):
    annotations.preflight_annotations(native)
    labels={(v["device"],v["inode"]) for v in native["annotation_metadata"].values()}
    roles=set(labels)
    for rows in native["rows"].values():
        for row in rows:
            current=file_pin(row["image"],labels,32*1024**2)
            roles.add(tuple(current[:2]))
    return labels,roles


def exclusions(native,*,sources=True,references=True):
    _,roles=data_roles(native)
    if sources:
        for root,snapshot in ((ROOT,True),(PARENT/"runtime",False)):
            members=protected.source_members(root,roles,snapshot_source=snapshot)
            roles.update(v[:2] for v in members.values())
    if references:
        members=protected.source_members(PARENT/"reference",roles)
        roles.update(v[:2] for v in members.values())
    return roles


def reference_pins(root,expected,forbidden):
    require(set(expected) == REFERENCE_NAMES and root.resolve(strict=True) == root,
            "G03 exact upstream reference membership differs")
    members=protected.source_members(root,forbidden)
    require(set(members) == REFERENCE_NAMES, "G03 extra reference source/resource")
    pins={name:file_pin(root/name,forbidden,8*1024**2,single=True) for name in sorted(expected)}
    require(len({tuple(v[:2]) for v in pins.values()}) == 8 and sum(v[2] for v in pins.values()) <= 32*1024**2
            and all(curve.digest(v) for v in expected.values()), "G03 reference identity/resource bound differs")
    return pins


def source_inputs():
    # These two independently operator-pinned G files precede registry parsing.
    g=protected.PARENT
    prepared,_=protected.metadata(g/"SUITE_PREPARED.json",protected.PARENT_SHA,
        expected_identity=protected.PARENT_METADATA_IDENTITIES["SUITE_PREPARED.json"])
    original,_=protected.metadata(g/"INPUTS.json",prepared["inputs_sha256"],
        expected_identity=protected.PARENT_METADATA_IDENTITIES["INPUTS.json"])
    rows={k:v for k,v in original.items() if k.startswith("clean.")}
    require(set(rows) == {"clean.coco_object","clean.pc60","clean.voc20"}, "G03 trusted clean membership differs")
    registry={}
    for values in original.values():
        for row in values:
            general.label_identity(row)
            details=row["label_metadata_details"]
            require(row["label"] not in registry or curve.same_provenance(registry[row["label"]],details),
                    "G03 annotation registry conflicts")
            registry[row["label"]]=copy.deepcopy(details)
    native=dict(rows=copy.deepcopy(original),annotation_metadata=registry)
    forbidden=exclusions(native)
    require(PARENT.resolve(strict=True) == PARENT
            and not any((PARENT/n).exists() for n in ("FAILED.json","CANCELLED.json")), "G03 failed S5 parent")
    cp,_=protected.metadata(PARENT/"CPU_PREPARED.json",PARENT_SHA,forbidden)
    seal,_=protected.metadata(PARENT/"INPUTS_SEALED.json",PARENT_SEAL,forbidden)
    receipt,_=protected.metadata(RECEIPT,RECEIPT_SHA,forbidden)
    expected_service=dict(unit="trace-corruption-inputs-20261102-1607.service",LoadState="loaded",ActiveState="active",
        SubState="exited",MainPID=0,ExecMainPID=1832234,ExecMainCode=1,ExecMainStatus=0,Result="success",
        InvocationID="f4710c4ba93b48b5ac3d9bc5e8e8a91f")
    require(curve.same_provenance(receipt["service"],expected_service) and receipt["prepared_sha256"] == PARENT_SHA
            and receipt["input_seal_sha256"] == PARENT_SEAL and receipt["CPU_input_generation_successfully_completed"] is True
            and cp["root"] == str(PARENT) and cp["GPU_allowed"] is False
            and seal["complete"] is True and seal["labels_read"] is False and seal["GPU_used"] is False
            and seal["prepared_sha256"] == PARENT_SHA and type(seal["generated_RGBs"]) is int
            and seal["generated_RGBs"] == curve.SOURCES*15,
            "G03 S5 parent completion identity differs")
    clean,_=protected.metadata(PARENT/"SOURCE_INPUTS.json",cp["source_inputs_sha256"],forbidden)
    protocol,_=protected.metadata(PARENT/"PROTOCOL.json",cp["protocol_sha256"],forbidden)
    s5,_=protected.metadata(PARENT/"VOC20_C_S5_INPUTS.json",seal["artifact_sha256"]["VOC20_C_S5_INPUTS.json"],forbidden)
    require(set(s5) == set(curve.CORRUPTIONS) and curve.same_provenance(clean,rows["clean.voc20"])
            and protocol["official_source_commit"] == UPSTREAM_COMMIT
            and protocol["versions"] == prior.versions()
            and protocol["reference_sha256"] == cp["reference_sha256"] == seal["reference_sha256"]
            and seal["artifact_sha256"]["SOURCE_INPUTS.json"] == cp["source_inputs_sha256"]
            and seal["artifact_sha256"]["PROTOCOL.json"] == cp["protocol_sha256"], "G03 clean/reference provenance differs")
    selected={k:s5[k] for k in curve.CURVE_TYPES}
    curve.validate_sources(clean)
    curve.validate_S5_reuse(selected,clean)
    native["rows"].update({"S5."+k:v for k,v in selected.items()})
    forbidden=exclusions(native)
    refs={str(Path(k).relative_to("reference")):v for k,v in cp["reference_sha256"].items()}
    rp=reference_pins(PARENT/"reference",refs,exclusions(native,references=False))
    labels,_=data_roles(native)
    rgb={r["image"]:file_pin(r["image"],labels,32*1024**2) for r in clean}
    s5_pins={r["image"]:file_pin(r["image"],labels,curve.MAX_PNG_BYTES) for values in selected.values() for r in values}
    return dict(rows=clean,native=native,S5_selected=selected,reference_sha256=refs,reference_identities=rp,
        source_RGB_identities=rgb,S5_RGB_identities=s5_pins,parent_prepared_sha256=PARENT_SHA,parent_seal_sha256=PARENT_SEAL,
        parent_terminal_receipt_sha256=RECEIPT_SHA,G_inputs_sha256=prepared["inputs_sha256"],GT_opened=False)


def reference_copy(output,inputs):
    forbidden=exclusions(inputs["native"],references=False)
    pins=reference_pins(PARENT/"reference",inputs["reference_sha256"],forbidden)
    require(pins == inputs["reference_identities"], "G03 reference changed before copy")
    target=output/"reference"
    target.mkdir()
    budget=dict(bytes=0)
    for name,current in pins.items():
        raw,_=stable_bytes(PARENT/"reference"/name,8*1024**2,inputs["reference_sha256"][name],tuple(current[:4]))
        write_output(output,"reference/"+name,raw,budget)
    protected.guarded_inventory(target,inputs["reference_sha256"],forbidden)


def prepare(output):
    cpu_only()
    inputs=source_inputs()
    output=Path(output).absolute()
    require(output.resolve() == output and not output.exists(), "Fresh canonical G03 CPU suite required")
    require(shutil.disk_usage(output.parent).free >= RESERVE+256*1024**2, "G03 preparation headroom insufficient")
    output.mkdir()
    inputs_sha=protected.write_prepared_json(output/"SOURCE_INPUTS.json",inputs)
    protocol_sha=protected.write_prepared_json(output/"PROTOCOL.json",dict(experiment="G03_CPU_RGB_inputs_only",
        fixed_plan=curve.plan(),versions=prior.versions(),upstream_commit=UPSTREAM_COMMIT,
        limits=dict(output_cap_bytes=OUTPUT_CAP,shared_reserve_bytes=RESERVE),GT_allowed=False,GPU_allowed=False,
        execution_resources=copy.deepcopy(EXECUTION_RESOURCES),
        launch="New independent exact CPU input artifact admission; manual externally capped one attempt; no resume",
        completion="Inputs only; no model gate, prediction, metric or scientific result"))
    reference_copy(output,inputs)
    inventory=protected.guarded_snapshot(ROOT,output/"runtime",exclusions(inputs["native"],sources=False))
    return protected.write_prepared_json(output/"CPU_PREPARED.json",dict(schema_version=1,root=str(output),
        inputs_sha256=inputs_sha,protocol_sha256=protocol_sha,runtime_hashes=inventory,
        reference_sha256=inputs["reference_sha256"],GT_opened=False,GPU_allowed=False))


def admission(suite,expected_sha):
    cpu_only()
    inputs=source_inputs()
    forbidden=exclusions(inputs["native"])
    suite=Path(suite).resolve(strict=True)
    metadata_forbidden=set(forbidden)
    for root in (suite/"runtime",suite/"reference"):
        # The active frozen runtime is a source role, not a forbidden data role.
        member_forbidden=exclusions(inputs["native"],sources=False) if root == suite/"runtime" else forbidden
        metadata_forbidden.update(v[:2] for v in protected.source_members(root,member_forbidden).values())
    prepared,_=protected.metadata(suite/"CPU_PREPARED.json",expected_sha,metadata_forbidden)
    require(prepared["schema_version"] == 1 and prepared["root"] == str(suite)
            and Path(__file__).resolve().is_relative_to(suite/"runtime"), "Wrong exact G03 CPU runtime")
    saved,_=protected.metadata(suite/"SOURCE_INPUTS.json",prepared["inputs_sha256"],metadata_forbidden)
    protocol,_=protected.metadata(suite/"PROTOCOL.json",prepared["protocol_sha256"],metadata_forbidden)
    require(curve.same_provenance(saved,inputs) and protocol["experiment"] == "G03_CPU_RGB_inputs_only"
            and curve.same_provenance(protocol["fixed_plan"],curve.plan()) and protocol["versions"] == prior.versions()
            and protocol["limits"] == dict(output_cap_bytes=OUTPUT_CAP,shared_reserve_bytes=RESERVE)
            and curve.same_provenance(protocol["execution_resources"],EXECUTION_RESOURCES)
            and protocol["GT_allowed"] is False and protocol["GPU_allowed"] is False
            and prepared["reference_sha256"] == inputs["reference_sha256"], "G03 input/protocol/version drift")
    protected.guarded_inventory(suite/"runtime",prepared["runtime_hashes"],exclusions(inputs["native"],sources=False))
    reference_pins(suite/"reference",prepared["reference_sha256"],forbidden)
    protected.guarded_inventory(suite/"reference",prepared["reference_sha256"],forbidden)
    general.runtime_imports(suite/"runtime")
    audit,audit_sha=protected.metadata(suite/"ADMITTED.json",forbidden=metadata_forbidden,limit=4*1024**2)
    require(audit["decision"] == "ADMITTED" and audit["suite_prepared_sha256"] == expected_sha
            and audit.get("independent_reviewer") and audit.get("G03_CPU_inputs_allowed") is True
            and audit.get("GPU_allowed") is False and audit.get("GT_scoring_allowed") is False,
            "New exact independent G03 CPU input admission required")
    report=suite/"INDEPENDENT_REPORT.md"
    require(audit["report_path"] == str(report), "Wrong G03 independent report")
    shared.bounded_content(report,audit["report_sha256"],metadata_forbidden,4*1024**2)
    ids=[file_pin(p,metadata_forbidden,4*1024**2)[:4] for p in (suite/"ADMITTED.json",report)]
    binding=dict(admission_path=str(suite/"ADMITTED.json"),admission_sha256=audit_sha,admission_identity=ids[0],
        report_path=str(report),report_sha256=audit["report_sha256"],report_identity=ids[1],independent_reviewer=audit["independent_reviewer"])
    verify_audit_binding(binding)
    return suite,prepared,inputs,binding


def write_output(suite,relative,content,budget,on_create=lambda:None):
    require(type(relative) is str and PurePosixPath(relative).parts and relative == str(PurePosixPath(relative)) and not PurePosixPath(relative).is_absolute()
            and ".." not in PurePosixPath(relative).parts and type(content) is bytes and 0 < len(content) <= 64*1024**2,
            "Invalid G03 output path/bytes")
    require(type(budget) is dict and type(budget.get("bytes")) is int and 0 <= budget["bytes"] <= OUTPUT_CAP
            and budget["bytes"]+len(content) <= OUTPUT_CAP and shutil.disk_usage(suite).free >= RESERVE+len(content),
            "G03 output/reserve budget exceeded before write")
    directory=os.open(suite,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    fd=None
    try:
        for part in PurePosixPath(relative).parts[:-1]:
            try:os.mkdir(part,dir_fd=directory)
            except FileExistsError:pass
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory)
            os.close(directory)
            directory=child
        fd=os.open(PurePosixPath(relative).name,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory)
        on_create()
        stream=os.fdopen(fd,"w+b")
        fd=None
        with stream:
            first=os.fstat(stream.fileno())
            require(stat.S_ISREG(first.st_mode) and first.st_nlink == 1, "G03 output inode differs")
            require(stream.write(content) == len(content), "Short G03 output write")
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            hasher=hashlib.sha256()
            total=0
            while True:
                chunk=stream.read(min(65536,len(content)+1-total))
                if not chunk:break
                total+=len(chunk)
                require(total <= len(content), "G03 persisted output grew beyond original extent")
                hasher.update(chunk)
            current=identity(os.fstat(stream.fileno()))
            require(current[:2] == identity(first)[:2] and current[2] == len(content) and current[5] == 1
                    and hasher.hexdigest() == hashlib.sha256(content).hexdigest(), "G03 persisted output differs")
        require(file_pin(suite/relative,set(),64*1024**2,single=True) == current, "G03 output changed on close")
        budget["bytes"]+=len(content)
        return hasher.hexdigest(),current
    finally:
        if fd is not None:os.close(fd)
        os.close(directory)


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()


def authenticated_reference(root,expected,forbidden):
    name="_trace_pinned_upstream_corruptions"
    require(not any(k == name or k.startswith(name+".") for k in sys.modules), "Reference namespace already loaded")
    pins=reference_pins(root,expected,forbidden)
    contents={}
    for member,current in pins.items():
        contents[member],_=stable_bytes(root/member,8*1024**2,expected[member],tuple(current[:4]))
        require(file_pin(root/member,forbidden,8*1024**2,single=True) == current,
                "G03 reference changed during authenticated read")
    require(all(file_pin(root/member,forbidden,8*1024**2,single=True) == current
                for member,current in pins.items()), "G03 reference drift before code execution")
    # Both Python modules execute only authenticated in-memory bytes. An empty
    # package search path prevents a later relative import from reopening disk.
    package=types.ModuleType(name)
    package.__file__,package.__package__,package.__path__=str(root/"__init__.py"),name,[]
    child=types.ModuleType(name+".corruptions")
    child.__file__,child.__package__=str(root/"corruptions.py"),name
    sys.modules[name],sys.modules[child.__name__]=package,child
    exec(compile(contents["corruptions.py"],child.__file__,"exec"),child.__dict__)
    package.corruptions=child
    exec(compile(contents["__init__.py"],package.__file__,"exec"),package.__dict__)
    require(all(file_pin(root/member,forbidden,8*1024**2,single=True) == current
                for member,current in pins.items()), "G03 reference drift before first RGB")
    return package


def execute(suite,inputs):
    require(curve.same_provenance(source_inputs(),inputs), "G03 source changed before first RGB")
    forbid_annotations(list(inputs["native"]["annotation_metadata"]))
    labels,_=data_roles(inputs["native"])
    generated={c["id"]:[] for c in curve.conditions()}
    budget,inventory,pins=dict(bytes=0),{},{}
    costs=[]
    rgb_bytes=0
    namespace_before={name for name in sys.modules if name == "_trace_pinned_upstream_corruptions"
                      or name.startswith("_trace_pinned_upstream_corruptions.")}
    try:
        module=authenticated_reference(suite/"reference",inputs["reference_sha256"],exclusions(inputs["native"]))
        for index,row in enumerate(inputs["rows"]):
            cpu_only()
            expected=inputs["source_RGB_identities"][row["image"]]
            require(file_pin(row["image"],labels,32*1024**2) == expected, "G03 original RGB stat drift")
            raw,_=stable_bytes(row["image"],32*1024**2,row["image_sha256"],tuple(expected[:4]))
            with Image.open(io.BytesIO(raw)) as header:
                require(0 < header.width*header.height*3 <= curve.MAX_SOURCE_BYTES, "G03 original decoded geometry exceeds bound")
            bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            require(bgr is not None, "G03 original RGB decode failed")
            tick=time.perf_counter()
            for condition in curve.conditions():
                image=curve.generate_image(module,bgr,index,condition["corruption"],condition["severity"])
                name=f"rgb/{condition['corruption']}/severity_{condition['severity']}/{index:06d}.png"
                digest,current=write_output(suite,name,image,budget)
                rgb_bytes+=len(image)
                inventory[name],pins[name]=digest,current
                item=copy.deepcopy(row)
                item.update(image=str(suite/name),image_sha256=digest,clean_image=row["image"],clean_image_sha256=row["image_sha256"],
                    corruption=condition["corruption"],severity=condition["severity"],original_sample_idx=index,preprocessing=curve.PREPROCESSING)
                generated[condition["id"]].append(item)
            costs.append(dict(source_index=index,generated_RGBs=10,seconds=time.perf_counter()-tick,GT_opened=False,GPU_used=False))
            shared.write_json(suite/"queue_state.json",dict(phase="G03_generating_inputs_only",completed_sources=index+1,
                total_sources=len(inputs["rows"]),completed_RGBs=(index+1)*10,bytes=budget["bytes"],GT_opened=False),replace=True)
        curve.validate_manifest(generated,inputs["rows"],str(suite))
        for name,value in (("VOC20_C_CURVE_INPUTS.json",generated),("cost.json",costs)):
            inventory[name],pins[name]=write_output(suite,name,encoded(value),budget)
        return dict(complete=True,generated_RGBs=len(inputs["rows"])*10,generated_bytes=rgb_bytes,output_bytes=budget["bytes"],
            artifact_sha256=inventory,artifact_identities=pins,GT_opened=False,GPU_used=False,scientific_completion=False)
    finally:
        for name in list(sys.modules):
            if name not in namespace_before and (name == "_trace_pinned_upstream_corruptions"
                                                or name.startswith("_trace_pinned_upstream_corruptions.")):del sys.modules[name]


def run(suite,expected_sha):
    suite,prepared,inputs,audit=admission(suite,expected_sha)
    require(not any((suite/n).exists() for n in ("STARTED.json","FAILED.json","INPUTS_SEALED.json")), "No G03 CPU resume/retry")
    require(shutil.disk_usage(suite).free >= RESERVE+OUTPUT_CAP, "G03 CPU generation headroom insufficient")
    owned=[]
    try:
        verify_audit_binding(audit)
        started,_=write_output(suite,"STARTED.json",encoded(dict(pid=os.getpid(),prepared_sha256=expected_sha,
            audit=audit,GT_opened=False,GPU_used=False,automatic_retry=False)),dict(bytes=0),lambda:owned.append(True))
        begin=time.perf_counter()
        forbidden=exclusions(inputs["native"])
        protected.metadata(suite/"STARTED.json",started,forbidden)
        result=execute(suite,inputs)
        require(result["generated_RGBs"] == curve.SOURCES*10, "Incomplete G03 input population")
        shared.write_json(suite/"queue_state.json",dict(phase="G03_inputs_complete_awaiting_seal_and_exit",
            completed_RGBs=result["generated_RGBs"],GT_opened=False,GPU_used=False),replace=True)
        require(curve.same_provenance(source_inputs(),inputs), "G03 source drift before input seal")
        protected.guarded_inventory(suite/"runtime",prepared["runtime_hashes"],exclusions(inputs["native"],sources=False))
        protected.guarded_inventory(suite/"reference",prepared["reference_sha256"],forbidden)
        for name,digest in (("CPU_PREPARED.json",expected_sha),("SOURCE_INPUTS.json",prepared["inputs_sha256"]),
                            ("PROTOCOL.json",prepared["protocol_sha256"]),("STARTED.json",started)):
            value,_=protected.metadata(suite/name,digest,forbidden)
            if name == "PROTOCOL.json":require(value["versions"] == prior.versions(), "G03 final package version drift")
        for name,current in result["artifact_identities"].items():
            require(file_pin(suite/name,forbidden,64*1024**2,single=True) == current, "G03 output drift before seal")
        verify_audit_binding(audit)
        general.runtime_imports(suite/"runtime")
        cpu_only()
        publish_seal(suite/"INPUTS_SEALED.json",dict(**result,prepared_sha256=expected_sha,audit_binding=audit,
            started_sha256=started,elapsed_seconds=time.perf_counter()-begin,parent_S5_seal_sha256=PARENT_SEAL))
    except BaseException as error:
        if owned:protected.write_prepared_json(suite/"FAILED.json",dict(error=f"{type(error).__name__}: {error}",automatic_retry=False))
        raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","validate-admission","run"))
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--suite-sha256")
    args=parser.parse_args()
    if args.phase == "prepare":print(dict(prepared_sha256=prepare(args.suite),GT_opened=False,GPU_started=False))
    else:
        require(args.suite_sha256,"Exact G03 CPU prepared SHA required")
        if args.phase == "validate-admission":
            admission(args.suite,args.suite_sha256)
            print(dict(admission_valid=True,CUDA_initialized=torch.cuda.is_initialized(),GT_opened=False,GPU_started=False))
        else:run(args.suite,args.suite_sha256)
