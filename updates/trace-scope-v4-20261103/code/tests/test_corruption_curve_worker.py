import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tools.evolution_fifty import corruption_curve_worker as worker
from tools.evolution_fifty.intake import sha256,write_new


def fixture(tmp_path,monkeypatch):
    parent,g,root,suite=(tmp_path/n for n in ("S5","G","sources","G03"))
    for p in (parent,g,root):p.mkdir()
    (parent/"runtime").mkdir()
    (parent/"runtime"/"fake.py").write_text("synthetic_CPU=True\n")
    for name in ("trace_ovss","tools","tests"):
        (root/name).mkdir()
        (root/name/"fake.py").write_text("synthetic_CPU=True\n")
    (parent/"reference").mkdir()
    for name in worker.REFERENCE_NAMES:
        (parent/"reference"/name).parent.mkdir(parents=True,exist_ok=True)
        content=("import numpy as np\ndef corrupt(image,**kwargs):\n"
                 " return np.clip(image.astype(int)+np.random.randint(-9,10,image.shape),0,255).astype(np.uint8)\n") if name == "__init__.py" else "# synthetic reference resource\n"
        (parent/"reference"/name).write_text(content)
    refs={"reference/"+n:sha256(parent/"reference"/n) for n in worker.REFERENCE_NAMES}
    clean=[]
    for i in range(2):
        rgb,label=tmp_path/f"clean{i}.png",tmp_path/f"label{i}.invalid"
        assert cv2.imwrite(str(rgb),np.full((37,53,3),70+i,np.uint8))
        label.write_bytes(b"Never read GT contents")
        s=label.stat()
        details=dict(device=s.st_dev,inode=s.st_ino,size_bytes=s.st_size,mtime_ns=s.st_mtime_ns,contents_read=False)
        clean.append(dict(official_id=f"s{i}",image=str(rgb),image_sha256=sha256(rgb),label=str(label),
            label_metadata=[s.st_size,s.st_mtime_ns],label_metadata_details=details))
    s5={}
    for kind in worker.curve.CORRUPTIONS:
        s5[kind]=[]
        for i,source in enumerate(clean):
            path=tmp_path/f"S5.{kind}.{i}.png"
            assert cv2.imwrite(str(path),np.full((224,224,3),100,np.uint8))
            row=copy.deepcopy(source)
            row.update(image=str(path),image_sha256=sha256(path),clean_image=source["image"],
                clean_image_sha256=source["image_sha256"],corruption=kind,severity=5,original_sample_idx=i,
                preprocessing=worker.curve.PREPROCESSING)
            s5[kind].append(row)
    def save(path,value):
        write_new(path,value)
        return sha256(path)
    original={"clean."+d:copy.deepcopy(clean) for d in ("coco_object","pc60","voc20")}
    original.update({"corrupt."+k:copy.deepcopy(v) for k,v in s5.items()})
    inputs_sha=save(g/"INPUTS.json",original)
    gp_sha=save(g/"SUITE_PREPARED.json",dict(inputs_sha256=inputs_sha))
    source_sha=save(parent/"SOURCE_INPUTS.json",clean)
    protocol_sha=save(parent/"PROTOCOL.json",dict(official_source_commit=worker.UPSTREAM_COMMIT,reference_sha256=refs,
        versions=worker.prior.versions()))
    s5_sha=save(parent/"VOC20_C_S5_INPUTS.json",s5)
    cp_sha=save(parent/"CPU_PREPARED.json",dict(root=str(parent),source_inputs_sha256=source_sha,
        protocol_sha256=protocol_sha,reference_sha256=refs,GPU_allowed=False))
    seal_sha=save(parent/"INPUTS_SEALED.json",dict(complete=True,labels_read=False,GPU_used=False,
        prepared_sha256=cp_sha,generated_RGBs=30,reference_sha256=refs,
        artifact_sha256={"SOURCE_INPUTS.json":source_sha,"PROTOCOL.json":protocol_sha,"VOC20_C_S5_INPUTS.json":s5_sha}))
    receipt=tmp_path/"CPU_terminal.json"
    receipt_sha=save(receipt,dict(prepared_sha256=cp_sha,input_seal_sha256=seal_sha,
        CPU_input_generation_successfully_completed=True,
        service=dict(unit="trace-corruption-inputs-20261102-1607.service",LoadState="loaded",ActiveState="active",
            SubState="exited",MainPID=0,ExecMainPID=1832234,ExecMainCode=1,ExecMainStatus=0,Result="success",
            InvocationID="f4710c4ba93b48b5ac3d9bc5e8e8a91f")))
    for key,value in dict(ROOT=root,PARENT=parent,PARENT_SHA=cp_sha,PARENT_SEAL=seal_sha,RECEIPT=receipt,RECEIPT_SHA=receipt_sha).items():
        monkeypatch.setattr(worker,key,value)
    monkeypatch.setattr(worker.curve,"SOURCES",2)
    monkeypatch.setattr(worker.protected,"PARENT",g)
    monkeypatch.setattr(worker.protected,"PARENT_SHA",gp_sha)
    def ident(path):
        s=path.stat()
        return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    monkeypatch.setattr(worker.protected,"PARENT_METADATA_IDENTITIES",{n:ident(g/n) for n in ("SUITE_PREPARED.json","INPUTS.json")})
    monkeypatch.setattr(worker.general,"runtime_imports",lambda *_:None)
    monkeypatch.setattr(worker,"forbid_annotations",lambda *_:None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    digest=worker.prepare(suite)
    monkeypatch.setattr(worker,"__file__",str(suite/"runtime"/"tools"/"fake.py"))
    report=suite/"INDEPENDENT_REPORT.md"
    report.write_text("Synthetic independent CPU input permission only\n")
    save(suite/"ADMITTED.json",dict(decision="ADMITTED",suite_prepared_sha256=digest,
        independent_reviewer="SyntheticCPU",G03_CPU_inputs_allowed=True,GPU_allowed=False,GT_scoring_allowed=False,
        report_path=str(report),report_sha256=sha256(report)))
    return suite,digest,clean,s5


def test_complete_toy_prepare_admit_generate_inputs_only_and_global_last(tmp_path,monkeypatch):
    suite,digest,clean,s5=fixture(tmp_path,monkeypatch)
    original=Path.open
    labels={r["label"] for r in clean}
    old_images={r["image"] for rows in s5.values() for r in rows}
    counts={r["image"]:0 for r in clean}
    def opened(path,*a,**k):
        assert str(path) not in labels and str(path) not in old_images
        if str(path) in counts:counts[str(path)]+=1
        return original(path,*a,**k)
    monkeypatch.setattr(Path,"open",opened)
    worker.run(suite,digest)
    seal=json.loads((suite/"INPUTS_SEALED.json").read_bytes())
    assert seal["complete"] and seal["generated_RGBs"] == 20 and not seal["scientific_completion"]
    assert not seal["GT_opened"] and not seal["GPU_used"] and set(counts.values()) == {1}
    for name,expected in seal["artifact_sha256"].items():assert sha256(suite/name) == expected
    generated=json.loads((suite/"VOC20_C_CURVE_INPUTS.json").read_bytes())
    assert worker.curve.validate_manifest(generated,clean,str(suite))["new_RGBs"] == 20
    with pytest.raises(ValueError,match="resume/retry"):worker.run(suite,digest)
    assert not torch.cuda.is_initialized()
    assert not any(k.startswith("_trace_pinned_upstream_corruptions") for k in worker.sys.modules)


@pytest.mark.parametrize("metadata_alias",[False,True])
def test_frozen_root_source_role_not_self_forbidden_but_metadata_alias_rejected(tmp_path,monkeypatch,metadata_alias):
    suite,digest,_,_=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(worker,"ROOT",suite/"runtime")
    if metadata_alias:
        path=suite/"CPU_PREPARED.json"
        victim=suite/"runtime"/"tools"/"fake.py"
        path.unlink();path.hardlink_to(victim)
        original=Path.open
        def guard(p,*a,**k):
            assert p != path and p != victim
            return original(p,*a,**k)
        monkeypatch.setattr(Path,"open",guard)
        with pytest.raises(ValueError):worker.admission(suite,digest)
    else:
        admitted,_,_,_=worker.admission(suite,digest)
        assert admitted == suite
    assert not (suite/"STARTED.json").exists()


@pytest.mark.parametrize("fault",["CPU_permission","GT_permission","GPU_permission","report_alias","source_alias","runtime_alias","reference_alias","reference_extra"])
def test_admission_faults_precede_RGB_and_GT(tmp_path,monkeypatch,fault):
    suite,digest,clean,_=fixture(tmp_path,monkeypatch)
    if fault.endswith("permission"):
        path=suite/"ADMITTED.json"
        value=json.loads(path.read_bytes())
        value[{"CPU_permission":"G03_CPU_inputs_allowed","GT_permission":"GT_scoring_allowed","GPU_permission":"GPU_allowed"}[fault]]=fault != "CPU_permission"
        path.unlink();write_new(path,value)
    elif fault == "reference_extra":(suite/"reference"/"extra.py").write_text("unlisted=True\n")
    else:
        path={"report_alias":suite/"INDEPENDENT_REPORT.md","source_alias":suite/"SOURCE_INPUTS.json",
            "runtime_alias":suite/"runtime"/"tools"/"fake.py","reference_alias":suite/"reference"/"corruptions.py"}[fault]
        path.unlink();path.hardlink_to(clean[0]["label"])
    blocked={r[k] for r in clean for k in ("image","label")}
    actual=Path.open
    def guard(path,*a,**k):
        assert str(path) not in blocked
        return actual(path,*a,**k)
    monkeypatch.setattr(Path,"open",guard)
    with pytest.raises(ValueError):worker.run(suite,digest)
    assert not (suite/"STARTED.json").exists() and not (suite/"INPUTS_SEALED.json").exists()


@pytest.mark.parametrize("fault",["metadata_GT","metadata_RGB","metadata_source","parent_seal","receipt","source_stat","S5_stat"])
def test_real_source_binding_faults_reject_before_source_payload(tmp_path,monkeypatch,fault):
    suite,digest,clean,s5=fixture(tmp_path,monkeypatch)
    if fault.startswith("metadata_"):
        path=worker.PARENT/"CPU_PREPARED.json"
        victim=Path(clean[0]["label" if fault == "metadata_GT" else "image"]) if fault != "metadata_source" else worker.ROOT/"tools"/"fake.py"
        path.unlink();path.hardlink_to(victim)
    elif fault == "parent_seal":monkeypatch.setattr(worker,"PARENT_SEAL","0"*64)
    elif fault == "receipt":monkeypatch.setattr(worker,"RECEIPT_SHA","0"*64)
    else:
        path=Path(clean[0]["image"] if fault == "source_stat" else s5[worker.curve.CURVE_TYPES[0]][0]["image"])
        path.unlink();path.hardlink_to(clean[0]["label"])
    actual=Path.open
    blocked={r[k] for r in clean for k in ("image","label")}
    def guard(path,*a,**k):
        assert str(path) not in blocked
        return actual(path,*a,**k)
    monkeypatch.setattr(Path,"open",guard)
    with pytest.raises(ValueError):worker.run(suite,digest)
    assert not (suite/"STARTED.json").exists()


@pytest.mark.parametrize("fault",["fsync","output_cap","progress","final_output","final_audit","final_reference"])
def test_owned_failure_no_global_no_retry_or_GT(tmp_path,monkeypatch,fault):
    suite,digest,clean,_=fixture(tmp_path,monkeypatch)
    if fault == "fsync":
        actual=worker.os.fsync
        def fsync(fd):
            if worker.os.readlink(f"/proc/self/fd/{fd}") == str(suite/"STARTED.json"):raise OSError("STARTED fsync failure")
            return actual(fd)
        monkeypatch.setattr(worker.os,"fsync",fsync)
    elif fault == "output_cap":
        actual=worker.execute
        def execute(*args):
            monkeypatch.setattr(worker,"OUTPUT_CAP",1)
            return actual(*args)
        monkeypatch.setattr(worker,"execute",execute)
    else:
        actual=worker.shared.write_json
        def progress(path,value,**kwargs):
            if fault == "progress":raise RuntimeError("progress failed")
            actual(path,value,**kwargs)
            if value["phase"] == "G03_inputs_complete_awaiting_seal_and_exit":
                if fault == "final_output":target=next((suite/"rgb").rglob("*.png"))
                elif fault == "final_audit":target=suite/"INDEPENDENT_REPORT.md"
                else:target=suite/"reference"/"corruptions.py"
                with target.open("ab") as stream:stream.write(b"drift")
        monkeypatch.setattr(worker.shared,"write_json",progress)
    with pytest.raises((ValueError,OSError,RuntimeError)):worker.run(suite,digest)
    assert (suite/"STARTED.json").exists() and (suite/"FAILED.json").exists()
    assert not (suite/"INPUTS_SEALED.json").exists()
    with pytest.raises((ValueError,OSError)):worker.run(suite,digest)


@pytest.mark.parametrize("relative",["../victim","/absolute","x/../victim","x//out"])
def test_output_path_injection_rejects_before_creation(tmp_path,relative):
    with pytest.raises(ValueError):worker.write_output(tmp_path,relative,b"payload",dict(bytes=0))
    assert not list(tmp_path.iterdir())


def test_output_symlink_parent_and_foreign_marker_are_never_followed(tmp_path,monkeypatch):
    suite,digest,_,_=fixture(tmp_path,monkeypatch)
    outside=tmp_path/"outside";outside.mkdir()
    (suite/"alias").symlink_to(outside,target_is_directory=True)
    with pytest.raises(OSError):worker.write_output(suite,"alias/victim",b"payload",dict(bytes=0))
    assert not list(outside.iterdir())
    actual=worker.os.open
    def opened(path,flags,*a,**k):
        if path == "STARTED.json" and flags & worker.os.O_CREAT:
            (suite/"STARTED.json").write_bytes(b"foreign marker")
        return actual(path,flags,*a,**k)
    monkeypatch.setattr(worker.os,"open",opened)
    with pytest.raises(FileExistsError):worker.run(suite,digest)
    assert (suite/"STARTED.json").read_bytes() == b"foreign marker"
    assert not (suite/"FAILED.json").exists()


@pytest.mark.parametrize("budget",[dict(bytes=-1),dict(bytes=True),dict(bytes=0.),{},dict(bytes=worker.OUTPUT_CAP+1)])
def test_output_budget_types_and_bounds_reject_before_file_creation(tmp_path,budget):
    with pytest.raises(ValueError):worker.write_output(tmp_path,"file.bin",b"x",budget)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("victim_kind",["GT","RGB"])
def test_operator_pinned_G_bootstrap_rejects_alias_before_content(tmp_path,monkeypatch,victim_kind):
    suite,digest,clean,_=fixture(tmp_path,monkeypatch)
    path=worker.protected.PARENT/"INPUTS.json"
    path.unlink();path.hardlink_to(clean[0]["label" if victim_kind == "GT" else "image"])
    actual=Path.open
    def guard(p,*a,**k):
        assert p != path and str(p) != clean[0]["label"] and str(p) != clean[0]["image"]
        return actual(p,*a,**k)
    monkeypatch.setattr(Path,"open",guard)
    with pytest.raises(ValueError,match="Bootstrap metadata identity"):worker.run(suite,digest)


def test_recorded_S5_package_version_drift_rejects_before_RGB(tmp_path,monkeypatch):
    suite,digest,_,_=fixture(tmp_path,monkeypatch)
    before=worker.prior.versions()
    monkeypatch.setattr(worker.prior,"versions",lambda:dict(before,numpy="changed"))
    with pytest.raises(ValueError,match="provenance"):worker.run(suite,digest)
    assert not (suite/"STARTED.json").exists()


def test_own_reference_cannot_be_opened_as_prepared_metadata(tmp_path,monkeypatch):
    suite,digest,_,_=fixture(tmp_path,monkeypatch)
    path=suite/"CPU_PREPARED.json"
    victim=suite/"reference"/"__init__.py"
    path.unlink();path.hardlink_to(victim)
    actual=Path.open
    def guard(p,*a,**k):
        assert p != path and p != victim
        return actual(p,*a,**k)
    monkeypatch.setattr(Path,"open",guard)
    with pytest.raises(ValueError):worker.run(suite,digest)


@pytest.mark.parametrize("member",["__init__.py","corruptions.py","frost/frost1.png"])
@pytest.mark.parametrize("timing",["post_STARTED","descriptor_open"])
def test_reference_mutations_never_execute_changed_bytes_or_read_RGB(tmp_path,monkeypatch,member,timing):
    import builtins
    suite,digest,clean,_=fixture(tmp_path,monkeypatch)
    target=suite/"reference"/member
    builtins.G03_changed_reference_executions=0
    original_execute,original_open=worker.execute,Path.open
    changed=[]
    def mutate():
        target.write_bytes(b"import builtins\nbuiltins.G03_changed_reference_executions += 1\n")
        changed.append(True)
    def execute(*a):
        assert (suite/"STARTED.json").exists()
        if timing == "post_STARTED":mutate()
        return original_execute(*a)
    def opened(path,*a,**k):
        assert str(path) not in {r[key] for r in clean for key in ("image","label")}
        if timing == "descriptor_open" and path == target and a and a[0] == "rb" and (suite/"STARTED.json").exists() and not changed:
            mutate()
        return original_open(path,*a,**k)
    monkeypatch.setattr(worker,"execute",execute)
    monkeypatch.setattr(Path,"open",opened)
    try:
        with pytest.raises(ValueError):worker.run(suite,digest)
        assert changed and builtins.G03_changed_reference_executions == 0
        assert (suite/"FAILED.json").exists() and not (suite/"INPUTS_SEALED.json").exists()
        assert not (suite/"rgb").exists()
        assert not any(k.startswith("_trace_pinned_upstream_corruptions") for k in worker.sys.modules)
    finally:del builtins.G03_changed_reference_executions


def test_authenticated_reference_relative_import_uses_pinned_bytes_without_reopen(tmp_path,monkeypatch):
    suite,_,_,_=fixture(tmp_path,monkeypatch)
    root=suite/"reference"
    (root/"corruptions.py").write_bytes(b"VALUE = 37\n")
    (root/"__init__.py").write_bytes(b"from .corruptions import VALUE\n")
    expected={n:sha256(root/n) for n in worker.REFERENCE_NAMES}
    original=Path.open
    reads=[]
    def opened(path,*a,**k):
        if path.is_relative_to(root):
            assert a and a[0] == "rb"
            reads.append(str(path.relative_to(root)))
        return original(path,*a,**k)
    monkeypatch.setattr(Path,"open",opened)
    try:
        module=worker.authenticated_reference(root,expected,set())
        assert module.VALUE == 37 and module.__path__ == []
        assert sorted(reads) == sorted(worker.REFERENCE_NAMES)
    finally:
        for name in list(worker.sys.modules):
            if name.startswith("_trace_pinned_upstream_corruptions"):del worker.sys.modules[name]


def test_preexisting_foreign_reference_namespace_is_not_claimed_or_removed(tmp_path,monkeypatch):
    suite,digest,_,_=fixture(tmp_path,monkeypatch)
    name="_trace_pinned_upstream_corruptions"
    foreign=worker.types.ModuleType(name)
    monkeypatch.setitem(worker.sys.modules,name,foreign)
    with pytest.raises(ValueError,match="namespace already loaded"):worker.run(suite,digest)
    assert worker.sys.modules[name] is foreign
    assert (suite/"FAILED.json").exists() and not (suite/"INPUTS_SEALED.json").exists()


def test_reference_layout_keeps_the_original_frost_subdirectory(tmp_path,monkeypatch):
    expected={"__init__.py","corruptions.py","frost/frost1.png","frost/frost2.png","frost/frost3.png",
              "frost/frost4.jpg","frost/frost5.jpg","frost/frost6.jpg"}
    assert worker.REFERENCE_NAMES == expected
    suite,digest,_,_=fixture(tmp_path,monkeypatch)
    assert {str(p.relative_to(suite/"reference")) for p in (suite/"reference").rglob("*") if p.is_file()} == expected
    prepared=json.loads((suite/"CPU_PREPARED.json").read_bytes())
    assert set(prepared["reference_sha256"]) == expected
    assert all(sha256(suite/"reference"/name) == h for name,h in prepared["reference_sha256"].items())
    worker.admission(suite,digest)
