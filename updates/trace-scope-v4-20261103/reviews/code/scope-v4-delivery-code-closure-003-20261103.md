# Scope-v4 Delivery C4 Independent Narrow CODE Review

## Findings First

**C4-F1, P2, confirmed; C4 NOT CLOSED on code004.**
The new pre-content opt-in gate authenticates the external v002 manifest only
against the caller's supplied manifest hash, not yet against the fixed trusted
v002 archive. A self-consistently repinned external manifest, exporter registry,
seal and opt-in can pass that check and open the newly exception-authorized
seal before the original archive rejects the counterfeit manifest.

Exact source: tools/evolution_fifty/delivery_packager.py:629 reads the declared
external manifest; :630-632 compare its registry member pin to the same
specification; :638 opens opted-in seal content; only :650 calls paper_check,
which binds manifest members to the actual archive. The archive's fixed
Policy SHA remains enabled, correct and unchanged in the reproduction.

Measured ordinary synthetic result:
```text
COUNTEREXAMPLE {"seal_content_opens": 1, "error": "Paper tar member hash mismatch", "trusted_archive_unchanged": true, "permission_granted": false}
AssertionError: Exception-authorized seal opened before trusted v002 archive authenticates its external manifest
assert not ['ANALYSIS_SEALED.json']
1 failed, 28 passed in 0.23s
```

This does not produce a completed invalid packet or override the final archive
hash check. It requires an invalid, self-consistently repinned input spec/plan,
not alteration of the supplied code004 bytes. The defect is specifically the
new guarantee that exception content is not opened until its registry is
bound to the reviewed v002 paper. Eventual failure does not undo the earlier
content read. It is not a claim about actual seal contents, plan002, paper
science, or a scientific reference execution.

A narrowly sufficient repair is to authenticate the external manifest/registry
association to the already fixed v002 archive before the first opted-in read
(for example, perform its bounded archive/manifest check early and reuse the
authenticated bytes/result). Merely comparing two caller-supplied pin
dictionaries earlier is insufficient. A legal control and the invalid-spec
case must both keep the fixed synthetic archive Policy hash enabled.

## Scope And Preservation

Same-reviewer C4 naming/readonly-hardlink CODE delta only, after actual plan002
DENIED. No reopening of C1/C2/C3, paper/science, or real input-plan admission.
No actual seal bytes/stats, real source inventory, SPEC/plan retry, real
materialize/build, service/GPU/Git/network/upload, datasets, RGB/GT/NPY,
models/weights, countstreams, metrics or bootstrap were accessed.
Synthetic fixture materialize/build/verify in private /tmp are test operations,
not an actual delivery build or permission. Six scientific reference-mutation
demonstrations remain NOT RUN.

Whole original code004 worker/tests are preserved in the authenticated
three-entry archive, and both versions were privately extracted only after
archive SHA verification. Exact SHA/size, regular-member type, no link targets,
exact three-member names and identical embedded/external CODE_FREEZE were
checked. No implementation, tests, frozen archives, earlier report, sealed
metadata, SPEC or plan was edited by this reviewer.

Scratch: /tmp/delivery-c4-independent-o4gm5C
Report is the only new persistent worktree output. A repair, if supplied,
must append a separately pinned narrow closure without erasing this finding.

## Exact Start And End Pins

| Object | SHA256 | Bytes |
|---|---|---:|
| code004 CODE_FREEZE.json | 131bb85dc85ad921e3024525afb5866fc246b4e6aebd9deec547484e71e282c4 | 900 |
| code004 code_v004.tar.gz | a2e6b06e4579eadaf23e5f16754627ec0e87ce572f6e113ce3195dd314fba731 | 24073 |
| code004 worker | 4e41bb3496edc2c2cd177d3bab6ea726966fd268cd4e717a50882bb2a10498f7 | 52088 |
| code004 tests | 67dffe406205991f5745f0473294fbeb1f10c7240e62521f5b7eb322d3370a1d | 42997 |
| authenticated code003 CODE_FREEZE.json | 0947dfa26644ab6c8842dc9bd31f509f5bebce838c3b1e9c6dce186da5e024f7 | 769 |
| authenticated code003 archive | 37c34fe0ab72269bd4caa956a1840f9d96badec8bc4de12d32897102918f2de0 | 19901 |
| code003 worker | d66504d3b23a4adb73b78c26e8c5dab599c31b6fde254dcc319df60d3d449ff3 | 46190 |
| code003 tests | d3e4ac4cd5adba2611e0c96afbd07d4a996e78652213e199440a1aae55563970 | 27395 |
| prior exact-plan DENIED raw | 102c08af2388d8b399c4fa019b14413ecf1cfd93b1ef74b762cec68d60a85b7a | - |
| prior exact-plan DENIED JSON | aa234850443bcc7f5353304f7e2ee8537d9d019613d1e7738cc0e4386e9efb7f | - |

Codefreeze paths are the user's exact
/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_code_003 and _004
directories. Worker/test archive members are tools/evolution_fifty/
delivery_packager.py and test_delivery_packager.py. Initial and final
worktree worker/test SHA values matched code004. Prior own DENIED report/JSON
hashes remained unchanged. The nine literal paths in READONLY_REGISTRY_SEALS
were compared solely with those recorded in the prior own DENIED JSON; they
match exactly. None of those real paths was opened or statted.

## Delta Review

Only two archived files differ. Worker unified diff has 209 lines; test diff
372 lines. Existing fixture seal geometry was changed from a synthetic suffix
to the exact allowed M01 literal path under a temporary results root.

AST comparison found seven modified existing functions: make_plan, materialize,
observe, plan_records, read, rebind, validate_spec. Six functions added:
_observe_readonly_seal, _observe_regular, _read_readonly_seal,
_read_with_observer, _readonly_record_pin, _readonly_seal_pins. None removed.
The read function's original FD-bound body is factored into
_read_with_observer; public read/observe/pin_read retain nlink1.

Limits, Policy, ACK/SCOPE, metadata_path, registry_metadata, pin validators,
directory/close handling, code inventory, paper_check, pack/verify_packet,
OwnedOutput, validate/build/create_plan/main/verify_delivery are AST-unchanged.
This is delta evidence and regression checking, not a renewed old CODE audit.

The new spec field enforces exact entry keys, exact integer nlink=2, positive
integer extent, lowercase SHA, canonical absolute paths and the nine literal
metadata geometries. It rejects duplicate, unused and registry-mismatched
entries and restricts opted records to sealed_summary_metadata. Record caps,
complete preflight inode uniqueness and final seven-field identity equality
remain. No default nlink1 fallback exists for an explicit nlink2 opt-in; absent
or empty opt-ins retain the prior single-link policy.

Confirmed unresolved limitation is C4-F1 above; no other blocker was confirmed
in this bounded delta.

## Actual Test Results

All execution imported only the authenticated private code004 copy:
/tmp/delivery-c4-independent-o4gm5C/004/tools/evolution_fifty/delivery_packager.py

Environment for each command:
```sh
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
 /home/wwww/miniconda3/envs/mlmp/bin/python -B <runner>
```
Each runner explicitly set cv2, torch intra and inter threads to 1 and observed
[1,1,1], CUDA_initialized=false before/after. Pytest used no cacheprovider,
private basetemp and no plugin autoload. Audit hooks denied actual results
content/enumeration and process/network operations; fixture Policy roots were
private /tmp. Target code has no research/model/scoring imports.

| Command Suffix | Result |
|---|---|
| run_frozen.py | 114 passed in 1.06s; process exit0 |
| run_independent.py | 28 passed, 1 failed in 0.23s; process exit1; failure is C4-F1 |
| run_remaining.py | 7 passed in 0.14s; process exit0 |

Total: 150 cases, 149 passed and 1 genuine boundary failure. There was no
separate failed test setup or model-capacity failure in this C4 review.
Author/main counts are not substituted for these independent observations.

Independent 36 probes cover: legal trusted-archive readonly validate/build/
verify; invalid self-consistent manifest/registry before-content ordering;
nine exact typed-schema mutations; six full-graph failures (including a later
non-seal input hardlink) with zero exception content reads; nine ctime/count/
same-bytes-inode changes in FD-open, actual-read and final-rebind windows;
three public strict readers; all nine distinct seals with a complete
36-entry synthetic registry; five non-seal role hardlinks; unchanged no-opt-in
single-link success.

The all-nine/36 positive control reported:
```json
{"registry_count":36,"distinct_optins":9,"validate_opted_reads":9,"build_opted_reads":9,"readonly_identities_unchanged":true}
```
It kept an exact fixed synthetic v002 archive SHA enabled, used genuine distinct
nlink2 inodes, and required real synthetic validate/build/verify. All original
seal seven-field identities were unchanged. No production seal or archive was
used. Output hardlink/foreign-role, alias collision, hash corruption and late
build failure controls also passed in the 114 frozen regression cases.

## Reproduction And Harness Pins

Single-finding replay using unchanged initial harness:
```sh
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
 /home/wwww/miniconda3/envs/mlmp/bin/python -B \
 /tmp/delivery-c4-independent-o4gm5C/run_independent.py \
 -k counterfeit_manifest
```
The initial 29-case run is retained as a failure, not counted as a passing
rejection test. Full harness bodies below preserve the counterexample and
legal controls independently of temporary fixture artifacts.

| Scratch File | SHA256 |
|---|---|
| independent_c4.py | bcc0925246d3c25ab7f7b9a1fd104f8af8f02b832089bcc1f4fed0592ab82395 |
| independent_remaining.py | 7e08722248f965441bb3f19b6f7e4e86f809beb1bd1cda261543887edca577f8 |
| run_frozen.py | 2a153b26ede333b3dba1a0e3df82cef8616df13dfb385843f73aa0a8dff1ea5c |
| run_independent.py | 14ad92a1c06cbde21cd8a4ccd6fd1a9d27f4c4c242b6fe709054b2414f7a7c9a |
| run_remaining.py | 090ab462dfd3abcbef3b2132302fdbb800d0e0dfb48a6a09a2663f1cf32c8db5 |
| authenticate.py | 39dd6936fc4cd37feaec2323b5035bcbacad84a70dd7de4494ebb8af586466ba |
| ast_delta.py | 616f9fa1f1eaeb6b2a29ef3a4c5812992b02ec60d26c2eab5b2569f4a438aab2 |
| delivery_packager.py.diff | 311438415eb2fd2cd297e445cec38086c2e1a1c72dc66af037aff388561d366c |
| test_delivery_packager.py.diff | 1349aed5dc5e81e40b4fa69c063c8b0014e1157e0d8c176c91cc55b9a7c4d622 |

## Disposition

Initial C4 review completed with one unresolved P2 boundary defect.
**C4 NOT CLOSED on worker4e41bb34/tests67dffe40/code004.**
Plan002 remains DENIED; this report issues no plan admission, build,
scientific, GT/model/GPU, network/Git/upload or acceptance permission.
New SPEC003/plan003 require separate future exact admission after actual
CODE closure. No supporting evidence was modified to obtain a pass.

## Preserved independent_c4.py

```python
"""C4 independent ordinary synthetic probes. No actual result paths."""
import copy,os,json
from dataclasses import replace
from pathlib import Path
import pytest
from tools.evolution_fifty import delivery_packager as p
from tools.evolution_fifty.test_delivery_packager import fixture,save,c4_spec,c4_seal,c4_change,destination

def trusted(f,spec):
    paper=next(x for x in spec["papers"] if x["label"]=="draft_v002")
    return replace(f["policy"],v002_archive_sha256=paper["archive"]["sha256"])

def watch_seal(f,monkeypatch):
    counts=[]
    orig=p.os.open
    def opening(path,*a,**kw):
        if os.fsdecode(path)=="ANALYSIS_SEALED.json":
            counts.append(os.fsdecode(path))
        return orig(path,*a,**kw)
    monkeypatch.setattr(p.os,"open",opening)
    return counts

def test_control_fixed_archive_and_readonly_seal(fixture):
    f=fixture; spec=c4_spec(f); policy=trusted(f,spec)
    before=p.identity(Path(c4_seal(f)["path"]).stat())
    pin=f["plan_pin"](spec)
    result=p.validate(pin,policy,f["limits"])
    assert result["valid"] and result["writes_performed"] is False
    result=p.build(pin,destination(f),policy,f["limits"])
    assert result["complete"] is True and result["publication_actions_performed"] is False
    assert p.identity(Path(c4_seal(f)["path"]).stat())==before
    assert p.verify_delivery(destination(f),result["local_receipt_pin"],policy,f["limits"])["complete"]
    print("CONTROL fixed trusted archive; validate/build/verify synthetic complete")

def test_counterfeit_manifest_and_registry_do_not_open_exception_before_archive_auth(fixture,monkeypatch):
    f=fixture;spec=c4_spec(f);policy=trusted(f,spec)
    # Keep the sole trusted v002 archive and its fixed Policy hash unchanged.
    paper=next(x for x in spec["papers"] if x["label"]=="draft_v002")
    original_archive=Path(paper["archive"]["path"]).read_bytes()
    # Only synthetic external registry/manifest and exact-literal-path seal change.
    seal=c4_seal(f)
    new_seal=save(Path(seal["path"]),b'{"unreviewed_synthetic_seal":true}\n')
    registry_path=Path(spec["exporter_provenance"]["path"])
    registry=json.loads(registry_path.read_bytes())
    rel=str(Path(seal["path"]).relative_to(policy.metadata))
    registry["source_pins"][rel].update(sha256=new_seal["sha256"],bytes=new_seal["bytes"])
    spec["exporter_provenance"]=save(registry_path,registry)
    spec["readonly_registry_seal_hardlinks"]=[dict(new_seal,nlink=2)]
    manifest_path=Path(paper["manifest"]["path"])
    manifest=json.loads(manifest_path.read_bytes())
    manifest["members"]["generated/source_provenance.json"]={k:spec["exporter_provenance"][k] for k in ("sha256","bytes")}
    paper["manifest"]=save(manifest_path,manifest)
    plan=p.make_plan(spec,policy,f["limits"])
    pin=save(f["tmp"]/"self-consistent-invalid-manifest-plan.json",plan)
    opened=watch_seal(f,monkeypatch)
    with pytest.raises(ValueError) as error:
        p.validate(pin,policy,f["limits"])
    assert Path(paper["archive"]["path"]).read_bytes()==original_archive
    print("COUNTEREXAMPLE",json.dumps(dict(seal_content_opens=len(opened),error=str(error.value),trusted_archive_unchanged=True,permission_granted=False)))
    assert not opened,"Exception-authorized seal opened before trusted v002 archive authenticates its external manifest"

@pytest.mark.parametrize("entryfield,value",[
    ("bytes",True),("bytes",1.0),("bytes",-1),("bytes",0),("sha256","G"*64),
    ("sha256",0),("path",None),("nlink",2.0),("nlink",True)])
def test_typed_optin_rejects_without_seal_open(fixture,monkeypatch,entryfield,value):
    f=fixture;spec=c4_spec(f);spec["readonly_registry_seal_hardlinks"][0][entryfield]=value
    opened=watch_seal(f,monkeypatch)
    with pytest.raises((ValueError,TypeError)):
        p.make_plan(spec,f["policy"],f["limits"])
    assert not opened

@pytest.mark.parametrize("mutation",["singlelink","thirdlink","oversize","symlink","directory","late_other_role_hardlink"])
def test_fullgraph_barrier_precedes_any_exception_open(fixture,monkeypatch,mutation):
    f=fixture;spec=c4_spec(f);pin=f["plan_pin"](spec);seal=Path(c4_seal(f)["path"])
    if mutation=="singlelink":(f["tmp"]/"synthetic-readonly-seal-alias").unlink()
    elif mutation=="thirdlink":os.link(seal,f["tmp"]/"third")
    elif mutation=="oversize":seal.write_bytes(b"x"*(f["limits"].metadata+1))
    elif mutation=="symlink":
        seal.unlink();seal.symlink_to(f["tmp"]/"synthetic-readonly-seal-alias")
    elif mutation=="directory":seal.unlink();seal.mkdir()
    else:
        # Last code role, not a seal; graph must complete before seal content.
        last=Path(p.parsed(Path(pin["path"]).read_bytes())["code"][-1]["path"])
        os.link(last,f["tmp"]/"foreign-code-alias")
    opened=watch_seal(f,monkeypatch)
    with pytest.raises(ValueError):
        p.validate(pin,f["policy"],f["limits"])
    assert not opened

@pytest.mark.parametrize("stage",["fd_open","during_read","final_rebind"])
@pytest.mark.parametrize("change",["ctime","count","same_bytes_inode"])
def test_original_identity_bound_in_each_window(fixture,monkeypatch,stage,change):
    f=fixture;spec=c4_spec(f);pin=f["plan_pin"](spec);seal=c4_seal(f)
    inode=p.identity(Path(seal["path"]).stat())[:2];done=[]
    if stage=="fd_open":
        orig=p.os.open
        def op(path,*a,**kw):
            fd=orig(path,*a,**kw)
            if os.fsdecode(path)=="ANALYSIS_SEALED.json" and not done:
                c4_change(f,change);done.append(True)
            return fd
        monkeypatch.setattr(p.os,"open",op)
    elif stage=="during_read":
        orig=p.os.read
        def rd(fd,n):
            b=orig(fd,n)
            if p.identity(os.fstat(fd))[:2]==inode and not done:
                c4_change(f,change);done.append(True)
            return b
        monkeypatch.setattr(p.os,"read",rd)
    else:
        orig=p.rebind
        def rb(*a,**kw):
            c4_change(f,change);done.append(True)
            return orig(*a,**kw)
        monkeypatch.setattr(p,"rebind",rb)
    with pytest.raises(ValueError):
        p.validate(pin,f["policy"],f["limits"])
    assert done

@pytest.mark.parametrize("method",["observe","read","pin_read"])
def test_public_readers_never_accept_grant(fixture,method):
    f=fixture;c4_spec(f);pin=c4_seal(f)
    with pytest.raises(ValueError,match="single-link"):
        if method=="pin_read":p.pin_read(pin,1024)
        else:getattr(p,method)(pin["path"],1024)


```

## Preserved independent_remaining.py

```python
import copy,os,json
from dataclasses import replace
from pathlib import Path
import pytest
from tools.evolution_fifty import delivery_packager as p
from tools.evolution_fifty.test_delivery_packager import fixture,save,archive_bytes,unpack,destination,c4_spec,c4_seal

def test_nine_distinct_hardlinked_seals_with_complete_36_pin_registry(fixture,monkeypatch):
    f=fixture;spec=copy.deepcopy(f["spec"]);meta={};grants=[];initial={}
    for i,rel in enumerate(sorted(p.READONLY_REGISTRY_SEALS)):
        path=f["policy"].metadata/rel
        pin=save(path,p.encoded(dict(synthetic_seal=i)))
        os.link(path,f["tmp"]/("seal-alias-"+str(i)))
        grants.append(dict(pin,nlink=2));meta[rel]={k:pin[k] for k in ("sha256","bytes")}
        initial[str(path)]=p.identity(path.stat())
    for i in range(27):
        rel="M01_offline_score_synthetic_"+str(i)+"/DATASET_RESULTS.json"
        pin=save(f["policy"].metadata/rel,p.encoded(dict(synthetic_summary=i,delta=-i,CI=None)))
        meta[rel]={k:pin[k] for k in ("sha256","bytes")}
    assert len(meta)==36
    spec["readonly_registry_seal_hardlinks"]=grants
    spec["exporter_provenance"]=save(Path(spec["exporter_provenance"]["path"]),dict(source_pins=meta,results_root=str(f["policy"].metadata)))
    for paper in spec["papers"]:
        contents=unpack(Path(paper["archive"]["path"]).read_bytes())
        contents["generated/source_provenance.json"]=Path(spec["exporter_provenance"]["path"]).read_bytes()
        manifest=dict(label=paper["label"],members={n:dict(sha256=p.sha(b),bytes=len(b)) for n,b in contents.items()})
        paper["manifest"]=save(Path(paper["manifest"]["path"]),manifest)
        paper["archive"]=save(Path(paper["archive"]["path"]),archive_bytes(list(contents.items())))
    v002=next(x for x in spec["papers"] if x["label"]=="draft_v002")
    policy=replace(f["policy"],metadata_count=36,v002_archive_sha256=v002["archive"]["sha256"])
    plan=p.make_plan(spec,policy,f["limits"]);pin=save(f["tmp"]/"synthetic36-plan.json",plan)
    reads=[];orig=p._read_readonly_seal
    def reading(record,*a,**kw):
        reads.append(record["pin"]["path"]);return orig(record,*a,**kw)
    monkeypatch.setattr(p,"_read_readonly_seal",reading)
    assert p.validate(pin,policy,f["limits"])["valid"]
    assert len(reads)==len(set(reads))==9
    assert all(p.identity(Path(path).stat())==identity for path,identity in initial.items())
    result=p.build(pin,destination(f),policy,f["limits"])
    assert result["complete"] and result["publication_actions_performed"] is False
    assert all(p.identity(Path(path).stat())==identity for path,identity in initial.items())
    assert len(reads)==18
    print("NINE_ALL36",json.dumps(dict(registry_count=36,distinct_optins=9,validate_opted_reads=9,build_opted_reads=9,readonly_identities_unchanged=True)))

@pytest.mark.parametrize("target",["registry","manifest","plan","review","code"])
def test_nonseal_input_roles_stay_singlelink_before_exception_read(fixture,monkeypatch,target):
    f=fixture;spec=c4_spec(f);pin=f["plan_pin"](spec)
    paths=dict(registry=Path(spec["exporter_provenance"]["path"]),
        manifest=Path(next(x for x in spec["papers"] if x["label"]=="draft_v002")["manifest"]["path"]),
        plan=Path(pin["path"]),review=Path(spec["reviews"]["v001"]["path"]),code=f["work"]/"tests/synthetic.py")
    os.link(paths[target],f["tmp"]/"unexpected-role-link")
    reads=[];orig=p._read_readonly_seal
    def reading(*a,**kw):
        reads.append(True);return orig(*a,**kw)
    monkeypatch.setattr(p,"_read_readonly_seal",reading)
    with pytest.raises(ValueError,match="single-link"):
        p.validate(pin,f["policy"],f["limits"])
    assert not reads

def test_absent_optin_normal_singlelink_spec_remains_accepted(fixture):
    f=fixture
    assert p.validate(f["plan_pin"](),f["policy"],f["limits"])["valid"]


```

## Preserved run_independent.py

```python
import os,sys,pathlib,json
os.environ.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
import torch,cv2
torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
ROOT=pathlib.Path("/tmp/delivery-c4-independent-o4gm5C");sys.path.insert(0,str(ROOT/"004"))
from tools.evolution_fifty import delivery_packager as p
assert pathlib.Path(p.__file__)==ROOT/"004/tools/evolution_fifty/delivery_packager.py"
def guard(event,args):
 if event in ("subprocess.Popen","socket.connect","socket.bind","os.system"):raise AssertionError("No process/network")
 if event in ("open","os.scandir","os.listdir") and isinstance(args[0],(str,bytes,os.PathLike)):
  assert not os.fsdecode(args[0]).startswith("/home/wwww/hzl/results/"),"No actual result access"
sys.addaudithook(guard)
import pytest
print(json.dumps(dict(origin=p.__file__,threads=[cv2.getNumThreads(),torch.get_num_threads(),torch.get_num_interop_threads()],CUDA_initialized=torch.cuda.is_initialized())))
rc=pytest.main(["-q","-s","-p","no:cacheprovider","--confcutdir="+str(ROOT),"--basetemp="+str(ROOT/"independent-tmp"),str(ROOT/"independent_c4.py"),*sys.argv[1:]])
assert not torch.cuda.is_initialized()
raise SystemExit(rc)


```

## Preserved run_frozen.py

```python
import os,sys,json,pathlib,importlib
os.environ.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
import cv2,torch
cv2.setNumThreads(1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
def forbidden(*a,**kw):raise AssertionError("No model or GPU allowed")
torch.load=forbidden;torch.jit.load=forbidden;torch.cuda.init=forbidden
ROOT=pathlib.Path("/tmp/delivery-c4-independent-o4gm5C")
sys.path.insert(0,str(ROOT/"004"))
from tools.evolution_fifty import delivery_packager as p
assert pathlib.Path(p.__file__)==ROOT/"004/tools/evolution_fifty/delivery_packager.py"
def audit(event,args):
 if event in ("subprocess.Popen","os.system","socket.connect","socket.bind"):raise AssertionError("No process/network")
 if event in ("open","os.scandir","os.listdir"):
  a=args[0]
  if isinstance(a,(str,bytes,os.PathLike)):
   path=os.fsdecode(a)
   if path.startswith("/home/wwww/hzl/results/"):raise AssertionError("No actual results access: "+path)
sys.addaudithook(audit)
import pytest
print(json.dumps(dict(frozen_origin=p.__file__,CUDA_initialized=torch.cuda.is_initialized(),threads=[cv2.getNumThreads(),torch.get_num_threads(),torch.get_num_interop_threads()])))
rc=pytest.main(["-q","-p","no:cacheprovider","--confcutdir="+str(ROOT),"--basetemp="+str(ROOT/"pytest-tmp"),str(ROOT/"004/tools/evolution_fifty/test_delivery_packager.py"),*sys.argv[1:]])
assert not torch.cuda.is_initialized()
raise SystemExit(rc)


```

## Preserved run_remaining.py

```python
import os,sys,pathlib,json
os.environ.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
import torch,cv2
torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
ROOT=pathlib.Path("/tmp/delivery-c4-independent-o4gm5C");sys.path.insert(0,str(ROOT/"004"))
from tools.evolution_fifty import delivery_packager as p
assert pathlib.Path(p.__file__)==ROOT/"004/tools/evolution_fifty/delivery_packager.py"
def guard(event,args):
 if event in ("subprocess.Popen","socket.connect","socket.bind","os.system"):raise AssertionError("No process/network")
 if event in ("open","os.scandir","os.listdir") and isinstance(args[0],(str,bytes,os.PathLike)):
  assert not os.fsdecode(args[0]).startswith("/home/wwww/hzl/results/"),"No actual result access"
sys.addaudithook(guard)
import pytest
print(json.dumps(dict(origin=p.__file__,threads=[cv2.getNumThreads(),torch.get_num_threads(),torch.get_num_interop_threads()],CUDA_initialized=torch.cuda.is_initialized())))
rc=pytest.main(["-q","-s","-p","no:cacheprovider","--confcutdir="+str(ROOT),"--basetemp="+str(ROOT/"remaining-tmp"),str(ROOT/"independent_remaining.py")])
assert not torch.cuda.is_initialized()
raise SystemExit(rc)


```

## Preserved authenticate.py

```python
import pathlib,hashlib,json,tarfile,io,difflib,os,stat
BASE=pathlib.Path("/home/wwww/hzl/results")
DEST=pathlib.Path("/tmp/delivery-c4-independent-o4gm5C")
pins={"003":("0947dfa26644ab6c8842dc9bd31f509f5bebce838c3b1e9c6dce186da5e024f7","37c34fe0ab72269bd4caa956a1840f9d96badec8bc4de12d32897102918f2de0"),"004":("131bb85dc85ad921e3024525afb5866fc246b4e6aebd9deec547484e71e282c4","a2e6b06e4579eadaf23e5f16754627ec0e87ce572f6e113ce3195dd314fba731")}
texts={}
for v,(mh,ah) in pins.items():
 root=BASE/("trace_scope_v4_delivery_20261103_code_"+v)
 def auth(p,h):
  assert p.resolve(strict=True)==p
  fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
  try:
   s=os.fstat(fd);assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_size<1000000
   b=os.read(fd,1000001);assert len(b)==s.st_size and hashlib.sha256(b).hexdigest()==h
   def ident(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink,s.st_mode)
   assert ident(s)==ident(os.fstat(fd))==ident(p.lstat())
   return b
  finally:os.close(fd)
 m=auth(root/"CODE_FREEZE.json",mh);a=auth(root/("code_v"+v+".tar.gz"),ah)
 manifest=json.loads(m);t=tarfile.open(fileobj=io.BytesIO(a),mode="r:gz"); members=t.getmembers()
 assert len(members)==3 and set(x.name for x in members)==set(manifest["members"])|{"CODE_FREEZE.json"}
 for x in members:
  assert x.isfile() and not x.linkname and x.size<1000000
  b=t.extractfile(x).read(1000001)
  if x.name=="CODE_FREEZE.json":assert b==m
  else:
   p=manifest["members"][x.name];assert len(b)==p["bytes"] and hashlib.sha256(b).hexdigest()==p["sha256"]
  out=DEST/v/x.name;out.parent.mkdir(parents=True,exist_ok=True)
  with out.open("xb") as f:f.write(b)
  texts[v,x.name]=b.decode()
for name in pins and ("tools/evolution_fifty/delivery_packager.py","tools/evolution_fifty/test_delivery_packager.py"):
 d="".join(difflib.unified_diff(texts["003",name].splitlines(True),texts["004",name].splitlines(True),fromfile="003/"+name,tofile="004/"+name))
 out=DEST/(pathlib.Path(name).name+".diff")
 with out.open("x") as f:f.write(d)
 print(name,"delta_lines",len(d.splitlines()),"diff",out)
print("Authenticated both archives and exact three-member manifests; extracted only private scratch")


```

## Preserved ast_delta.py

```python
import ast,pathlib,hashlib,json
root=pathlib.Path("/tmp/delivery-c4-independent-o4gm5C")
def top(ver):
 tree=ast.parse((root/ver/"tools/evolution_fifty/delivery_packager.py").read_bytes())
 return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
a,b=top("003"),top("004")
print(json.dumps(dict(unchanged=sorted(k for k in a.keys()&b.keys() if a[k]==b[k]),modified=sorted(k for k in a.keys()&b.keys() if a[k]!=b[k]),added=sorted(b.keys()-a.keys()),removed=sorted(a.keys()-b.keys()))))


```

