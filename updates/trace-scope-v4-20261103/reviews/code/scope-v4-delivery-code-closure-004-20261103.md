# Scope-v4 Delivery C4-F1 Independent Narrow Closure

## Findings And Disposition

**C4-F1 CLOSED; C4 narrow CODE scope CLOSED on code005.**
No remaining blocker was confirmed in the requested repair boundary.
This is not an admission of SPEC003/plan003 or any actual delivery build.

The exact original independent counterexample was rerun unchanged against
authenticated code005. It retains a fixed, enabled trusted synthetic v002
archive SHA, keeps the archive bytes unchanged, and self-consistently repins
only the synthetic external manifest, registry, seal and opt-in.
The prior code004 result was one exception-authorized seal open followed by
late Paper tar member hash mismatch. The new result is the same rejection
with **zero seal content opens**.

```text
COUNTEREXAMPLE {"seal_content_opens": 0, "error": "Paper tar member hash mismatch", "trusted_archive_unchanged": true, "permission_granted": false}
1 passed, 28 deselected in 0.04s
```

The code004 NOT CLOSED report and its genuine failure remain unchanged:
docs/scope-v4-delivery-code-closure-003-20261103.md
SHA256 6c971b78154e22b1b3ff0eb15aea1e934a63297282b7f1cba9aa74c74ee84251.
Whole original code004 worker/test bytes remain preserved in its authenticated
code_v004.tar.gz. This report does not rewrite or relabel the old failure.

## Exact Authentication

All archive hashes were checked before parsing/extracting source into private
scratch. The bounded authentication script opened only the supplied code004/
code005 archive and CODE_FREEZE paths. Each was canonical, nofollow,
single-link regular, below the one-MiB authentication cap, exact hash/extent
matched and stable in all seven stat identity fields across the FD read.
Each tar contained exactly the two declared regular worker/test members and
CODE_FREEZE.json, no member link targets, with the embedded manifest
byte-identical to the external one. Every worker/test extent and SHA matched
that manifest. No real paper or source input archive was read.

| Object | SHA256 | Bytes |
|---|---|---:|
| code005 CODE_FREEZE.json | ae6ccdf68d73c9627d0143dce547467d228705f110a81e9d583a64513acea430 | - |
| code005 code_v005.tar.gz | 65b06477dacf153dd43b6e442e5793d1d144a996b56e1f7f9a7ef7e65645e1ef | 24493 |
| code005 worker | eb87a9acca0e9fc89844169c93329cac29f6370e851dd1803411ed5d708320e9 | 52459 |
| code005 tests | aaae80239f4a778c1452ccffa2f69d45618e6bc8e5cb06d02f92ffb6ea85c92e | 45735 |
| code004 CODE_FREEZE.json | 131bb85dc85ad921e3024525afb5866fc246b4e6aebd9deec547484e71e282c4 | 900 |
| code004 code_v004.tar.gz | a2e6b06e4579eadaf23e5f16754627ec0e87ce572f6e113ce3195dd314fba731 | 24073 |
| code004 worker | 4e41bb3496edc2c2cd177d3bab6ea726966fd268cd4e717a50882bb2a10498f7 | 52088 |
| code004 tests | 67dffe406205991f5745f0473294fbeb1f10c7240e62521f5b7eb322d3370a1d | 42997 |

Exact codefreeze directories:
 /home/wwww/hzl/results/trace_scope_v4_delivery_20261103_code_004
 /home/wwww/hzl/results/trace_scope_v4_delivery_20261103_code_005
Archive worker/test members:
 tools/evolution_fifty/delivery_packager.py
 tools/evolution_fifty/test_delivery_packager.py

Private scratch: /tmp/delivery-c4f1-closure-qvDT7S.
Initial and final live worktree worker/test hashes matched the supplied
code005 pins. Final private frozen worker/test hashes also matched.
Archive005/manifest005 pins and prior raw report hash were unchanged.

## Repair Boundary Inspection

Compared authenticated code004 to code005. Worker unified diff: 35 lines;
test unified diff: 59 lines (three appended test cases, including two actions).
AST comparison: only materialize changed; 40 other top-level functions/classes
are unchanged; no functions/classes added or removed.

In code005 tools/evolution_fifty/delivery_packager.py:
- :612 complete graph stat/size/alias preflight remains before content.
- :623 registry bytes are bound to original graph identity and supplied SHA.
- :630 external manifest bytes are similarly authenticated and :632 bind the
  registry pin to its declared member.
- :637 now reads the fixed-policy-pinned v002 archive using the same original
  graph identity and SHA.
- :638 runs the existing bounded paper_check before :644 can read a seal.
- :639 caches the authenticated archive bytes; :655 avoids a second archive
  parse while retaining the result. Final rebind still checks all observed
  seven-field source identities.

The trusted fixed v002 archive Policy constraint, paper_check's member/extent/
hash/decompression limits, and every generic input/output, schema, role,
nlink, FD, directory and final-rebind helper are unchanged. The fix authenticates
the manifest to the actual trusted archive before granting the new exception;
it is not another comparison of self-consistent caller pin dictionaries.

Legal opt-in behavior, default single-link policy, no fallback, caps and failure
semantics remain as in the prior C4 boundary. C1/C2/C3 and paper/science were
not reopened. This AST/diff check and the requested regressions do not claim
an independent new full review of all old code.

## Actual Independent Execution

No mutable worktree module was imported by the test processes. Module origin:
/tmp/delivery-c4f1-closure-qvDT7S/005/tools/evolution_fifty/delivery_packager.py

For every runner:
```sh
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
 /home/wwww/miniconda3/envs/mlmp/bin/python -B <runner>
```
cv2.setNumThreads(1), torch.set_num_threads(1), and
torch.set_num_interop_threads(1) were explicitly called. Observed [1,1,1];
CUDA_initialized=false before/after. Pytest disabled cacheprovider, used private
basetemp and disabled plugin autoload. Runner audit hooks denied actual result
path opens/enumeration and process/network calls. All fixture roots and
synthetic build/validate/materialize/verify activity were ordinary /tmp data.
No real source inventory or real spec/plan operation was performed.

| Command | Observed Result | Exit |
|---|---|---:|
| run_current.py exact | original failed probe now 1 passed, 28 deselected in 0.04s | 0 |
| run_current.py all | 153 passed in 1.59s | 0 |
| run_cache.py | 6 passed in 0.12s | 0 |

The 153-case run is exactly 117 frozen code005 tests plus the unchanged prior
29-case and 7-case independent harnesses. Therefore the original failing probe
and the other 35 prior probes all passed on code005. The isolated exact run is
a repeated confirmation, not an extra unique case. Unique total is **159
passed** (153 + 6). No test setup/harness failure occurred in this closure.

Actual combined tail:
```text
CONTROL fixed trusted archive; validate/build/verify synthetic complete
COUNTEREXAMPLE {"seal_content_opens": 0, "error": "Paper tar member hash mismatch", "trusted_archive_unchanged": true, "permission_granted": false}
NINE_ALL36 {"registry_count": 36, "distinct_optins": 9, "validate_opted_reads": 9, "build_opted_reads": 9, "readonly_identities_unchanged": true}
153 passed in 1.59s
```

Independent additional six controls, each exercised for validate and build:
1. Successful fixed-archive check occurs before the seal reader. Archive read
   and successful v002 paper_check occur exactly once; cached bytes/result
   are reused without a second archive read/check.
2. A same-bytes archive inode replacement after its successful early check
   is rejected by final identity rebind; no LOCAL_VERIFY receipt is produced.
3. Corrupted pinned archive bytes are rejected by SHA before any seal read
   and before the synthetic output destination exists.

The supplied 117-test suite additionally independently executed the new
self-consistent invalid-spec regressions for both validate and build. Each
requires zero actual seal opens, unchanged trusted archive bytes, and no
synthetic output directory. The legal fixed-archive control completes synthetic
validate/build/verify without changing the input seal's seven-field identity.

## Preserved Harness Identity

The original independent files were not edited or repinned:
 /tmp/delivery-c4-independent-o4gm5C/independent_c4.py
 bcc0925246d3c25ab7f7b9a1fd104f8af8f02b832089bcc1f4fed0592ab82395
 /tmp/delivery-c4-independent-o4gm5C/independent_remaining.py
 7e08722248f965441bb3f19b6f7e4e86f809beb1bd1cda261543887edca577f8
Their full bodies and old failing evidence are preserved in the prior raw
6c971b78... report. Only runtime import location changed via the new runner;
all original test bodies and fixture imports use authenticated code005.

New scratch files:
| File | SHA256 |
|---|---|
| authenticate.py | 9b2b9fad475edb4824fb8db9bda9f1926bf384ea2b64e86380d3682d201789bf |
| run_current.py | 65d7145ceec52ff7a77d3a5bccd7a74063f0c5927ee457d966f1e7515bebcdb2 |
| run_cache.py | 7afeb7586ac4db459ab67a6f8c0491cc8535b41587036127137442458fa6aa52 |
| independent_cache_controls.py | 7787c7e13f8c6e207cc1423dacebcd3ce01ac525d0305ff6b9eac9afdd610bce |
| delivery_packager.py.diff | 7dd53a07d07ee2c7d2f1a432805e594a70594a53530e1db075c91d02cf472ef0 |
| test_delivery_packager.py.diff | 5496c1fac0bffef1a483380625416cd2ff06dea3ca07d982a02bbbefe2fc43a1 |

## Final Limits

Only this new worktree raw report and private /tmp harness/tests were written.
No implementation/test/frozen source/evidence/plan/metadata was modified.
No real seal bytes or stat, actual RGB/GT/NPY/weights/countstreams, paper input
archive, service/GPU/lock/Git/network/upload, estimator or bootstrap was used.
Six scientific reference-mutation demonstrations remain NOT RUN.

Prior exact-plan DENIED report remains
102c08af2388d8b399c4fa019b14413ecf1cfd93b1ef74b762cec68d60a85b7a
and decision JSON remains
aa234850443bcc7f5353304f7e2ee8537d9d019613d1e7738cc0e4386e9efb7f.
Their own hashes alone were rechecked; plan002 and its actual source input
paths were not accessed or reconsidered. **Plan002 remains DENIED.**

**C4-F1/C4 CODE closure is granted only for the exact code005 worker/tests
above. No actual input-plan admission, local build, GT/model/GPU, paper
acceptance or publication permission is granted.** Any future SPEC003/plan003
requires its own exact-input admission before a real bounded CPU build.

## Preserved authenticate.py

```python
import pathlib,hashlib,json,tarfile,io,difflib,os,stat,ast
BASE=pathlib.Path("/home/wwww/hzl/results");DEST=pathlib.Path("/tmp/delivery-c4f1-closure-qvDT7S")
pins={"004":("131bb85dc85ad921e3024525afb5866fc246b4e6aebd9deec547484e71e282c4","a2e6b06e4579eadaf23e5f16754627ec0e87ce572f6e113ce3195dd314fba731"),"005":("ae6ccdf68d73c9627d0143dce547467d228705f110a81e9d583a64513acea430","65b06477dacf153dd43b6e442e5793d1d144a996b56e1f7f9a7ef7e65645e1ef")}
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
 manifest=json.loads(m);t=tarfile.open(fileobj=io.BytesIO(a),mode="r:gz");members=t.getmembers()
 assert len(members)==3 and set(x.name for x in members)==set(manifest["members"])|{"CODE_FREEZE.json"}
 print(v,"archive_bytes",len(a),"manifest",manifest)
 for x in members:
  assert x.isfile() and not x.linkname and x.size<1000000
  b=t.extractfile(x).read(1000001)
  if x.name=="CODE_FREEZE.json":assert b==m
  else:
   pin=manifest["members"][x.name];assert len(b)==pin["bytes"] and hashlib.sha256(b).hexdigest()==pin["sha256"]
  out=DEST/v/x.name;out.parent.mkdir(parents=True,exist_ok=True)
  with out.open("xb") as f:f.write(b)
  texts[v,x.name]=b.decode()
for name in ("tools/evolution_fifty/delivery_packager.py","tools/evolution_fifty/test_delivery_packager.py"):
 delta="".join(difflib.unified_diff(texts["004",name].splitlines(True),texts["005",name].splitlines(True),fromfile="004/"+name,tofile="005/"+name))
 with (DEST/(pathlib.Path(name).name+".diff")).open("x") as f:f.write(delta)
 print("DELTA",name,len(delta.splitlines()))
def top(v):
 return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(texts[v,"tools/evolution_fifty/delivery_packager.py"]).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
a,b=top("004"),top("005")
print("AST",json.dumps(dict(modified=sorted(k for k in a.keys()&b.keys() if a[k]!=b[k]),added=sorted(b.keys()-a.keys()),removed=sorted(a.keys()-b.keys()),unchanged_count=sum(a[k]==b[k] for k in a.keys()&b.keys()))))


```

## Preserved run_current.py

```python
import os,sys,pathlib,json
os.environ.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
import torch,cv2
torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
ROOT=pathlib.Path("/tmp/delivery-c4f1-closure-qvDT7S");OLD=pathlib.Path("/tmp/delivery-c4-independent-o4gm5C")
sys.path.insert(0,str(ROOT/"005"))
from tools.evolution_fifty import delivery_packager as p
assert pathlib.Path(p.__file__)==ROOT/"005/tools/evolution_fifty/delivery_packager.py"
def guard(event,args):
 if event in ("subprocess.Popen","socket.connect","socket.bind","os.system"):raise AssertionError("No process/network")
 if event in ("open","os.scandir","os.listdir") and isinstance(args[0],(str,bytes,os.PathLike)):
  assert not os.fsdecode(args[0]).startswith("/home/wwww/hzl/results/"),"No actual result access"
sys.addaudithook(guard)
import pytest
print(json.dumps(dict(origin=p.__file__,threads=[cv2.getNumThreads(),torch.get_num_threads(),torch.get_num_interop_threads()],CUDA_initialized=torch.cuda.is_initialized())))
which=sys.argv[1]
targets=[str(OLD/"independent_c4.py")] if which=="exact" else [str(ROOT/"005/tools/evolution_fifty/test_delivery_packager.py"),str(OLD/"independent_c4.py"),str(OLD/"independent_remaining.py")]
extra=["-k","counterfeit_manifest"] if which=="exact" else []
rc=pytest.main(["-q","-s","-p","no:cacheprovider","--confcutdir="+str(ROOT),"--basetemp="+str(ROOT/(which+"-tmp")),*targets,*extra])
assert not torch.cuda.is_initialized()
raise SystemExit(rc)


```

## Preserved run_cache.py

```python
import os,sys,pathlib,json
os.environ.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
import torch,cv2
torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
ROOT=pathlib.Path("/tmp/delivery-c4f1-closure-qvDT7S");sys.path.insert(0,str(ROOT/"005"))
from tools.evolution_fifty import delivery_packager as p
assert pathlib.Path(p.__file__)==ROOT/"005/tools/evolution_fifty/delivery_packager.py"
def guard(event,args):
 if event in ("subprocess.Popen","socket.connect","socket.bind","os.system"):raise AssertionError("No process/network")
 if event in ("open","os.scandir","os.listdir") and isinstance(args[0],(str,bytes,os.PathLike)):
  assert not os.fsdecode(args[0]).startswith("/home/wwww/hzl/results/"),"No actual result access"
sys.addaudithook(guard)
import pytest
print(json.dumps(dict(origin=p.__file__,threads=[cv2.getNumThreads(),torch.get_num_threads(),torch.get_num_interop_threads()],CUDA_initialized=torch.cuda.is_initialized())))
rc=pytest.main(["-q","-s","-p","no:cacheprovider","--confcutdir="+str(ROOT),"--basetemp="+str(ROOT/"cache-tmp"),str(ROOT/"independent_cache_controls.py")])
assert not torch.cuda.is_initialized()
raise SystemExit(rc)


```

## Preserved independent_cache_controls.py

```python
"""C4-F1 cache/order/final-binding controls on ordinary synthetic archives only."""
import os,copy
from pathlib import Path
from dataclasses import replace
import pytest
from tools.evolution_fifty import delivery_packager as p
from tools.evolution_fifty.test_delivery_packager import fixture,c4_spec,c4_seal,destination

def setup(f):
    spec=c4_spec(f)
    paper=next(x for x in spec["papers"] if x["label"]=="draft_v002")
    policy=replace(f["policy"],v002_archive_sha256=paper["archive"]["sha256"])
    return spec,paper,policy,f["plan_pin"](spec)

@pytest.mark.parametrize("action",["validate","build"])
def test_successful_trusted_check_and_single_cached_archive_read_precede_seal(fixture,monkeypatch,action):
    f=fixture;spec,paper,policy,pin=setup(f)
    event=[];checked=[]
    original_read,original_check,original_seal=p.pin_read,p.paper_check,p._read_readonly_seal
    def reading(v,*a,**kw):
        b=original_read(v,*a,**kw)
        if v["path"]==paper["archive"]["path"]:event.append("archive_read")
        return b
    def checking(raw,manifest,current,limits):
        out=original_check(raw,manifest,current,limits)
        if current["label"]=="draft_v002":
            checked.append(True);event.append("trusted_check_passed")
        return out
    def sealed(*a,**kw):
        assert checked and event==["archive_read","trusted_check_passed"]
        event.append("seal_read")
        return original_seal(*a,**kw)
    monkeypatch.setattr(p,"pin_read",reading);monkeypatch.setattr(p,"paper_check",checking);monkeypatch.setattr(p,"_read_readonly_seal",sealed)
    if action=="validate":assert p.validate(pin,policy,f["limits"])["valid"]
    else:assert p.build(pin,destination(f),policy,f["limits"])["complete"]
    assert event==["archive_read","trusted_check_passed","seal_read"] and len(checked)==1

@pytest.mark.parametrize("action",["validate","build"])
def test_cached_archive_samebytes_inode_swap_fails_final_rebind(fixture,monkeypatch,action):
    f=fixture;spec,paper,policy,pin=setup(f);archive=Path(paper["archive"]["path"])
    changed=[];original=p._read_readonly_seal
    def sealed(*a,**kw):
        b=original(*a,**kw)
        replacement=f["tmp"]/"samebytes-archive"
        replacement.write_bytes(archive.read_bytes());replacement.replace(archive);changed.append(True)
        return b
    monkeypatch.setattr(p,"_read_readonly_seal",sealed)
    with pytest.raises(ValueError,match="Final input identity drift"):
        if action=="validate":p.validate(pin,policy,f["limits"])
        else:p.build(pin,destination(f),policy,f["limits"])
    assert changed and not (destination(f)/"LOCAL_VERIFY.json").exists()

@pytest.mark.parametrize("action",["validate","build"])
def test_archive_byte_corruption_rejects_before_seal_read(fixture,monkeypatch,action):
    f=fixture;spec,paper,policy,pin=setup(f);archive=Path(paper["archive"]["path"])
    b=archive.read_bytes();archive.write_bytes(bytes([b[0]^1])+b[1:])
    opened=[]
    def no_seal(*a,**kw):
        opened.append(True);raise AssertionError("Archive authentication must precede seal content")
    monkeypatch.setattr(p,"_read_readonly_seal",no_seal)
    with pytest.raises(ValueError,match="Input hash mismatch before parse"):
        if action=="validate":p.validate(pin,policy,f["limits"])
        else:p.build(pin,destination(f),policy,f["limits"])
    assert not opened and not destination(f).exists()


```

## Preserved delivery_packager.py.diff

```diff
--- 004/tools/evolution_fifty/delivery_packager.py
+++ 005/tools/evolution_fifty/delivery_packager.py
@@ -607,6 +607,7 @@
     spec = plan["specification"]
     grants = _readonly_seal_pins(spec, policy, plan["source_metadata"])
     observations, inodes, payloads = {}, set(), {"PIN_PLAN.json": plan_raw}
+    papers = {}
     # Every leaf is type/size/inode checked before any input-content open.
     for r in records:
         pin = r["pin"]
@@ -631,6 +632,11 @@
         require(frozen_registry == {k:spec["exporter_provenance"][k] for k in ("sha256", "bytes")},
                 "Exporter registry not bound to exact reviewed v002 manifest")
         payloads[r["member"]] = raw
+        manifest_raw = raw
+        r = next(r for r in records if r["role"] == "original_paper_archive" and r["pin"] == paper["archive"])
+        raw = pin_read(r["pin"], r["cap"], observations[r["pin"]["path"]])
+        papers["draft_v002"] = paper_check(raw, manifest_raw, paper, limits)
+        payloads[r["member"]] = raw
     for r in records:
         if r["member"] in payloads:
             continue
@@ -644,10 +650,10 @@
     registry = parsed(payloads["evidence/exporter_source_provenance.json"])
     require(encoded(registry_metadata(registry, policy)) == encoded(plan["source_metadata"]),
             "Exporter allowlist/pin membership mismatch")
-    papers = {}
     for paper in spec["papers"]:
         base = "paper/"+paper["label"]+"/"
-        papers[paper["label"]] = paper_check(payloads[base+"original.tar.gz"], payloads[base+"FREEZE.json"], paper, limits)
+        if paper["label"] not in papers:
+            papers[paper["label"]] = paper_check(payloads[base+"original.tar.gz"], payloads[base+"FREEZE.json"], paper, limits)
         if paper["label"] == "draft_v002":
             frozen_registry = parsed(payloads[base+"FREEZE.json"])["members"].get("generated/source_provenance.json")
             require(frozen_registry == {k:spec["exporter_provenance"][k] for k in ("sha256", "bytes")},

```

## Preserved test_delivery_packager.py.diff

```diff
--- 004/tools/evolution_fifty/test_delivery_packager.py
+++ 005/tools/evolution_fifty/test_delivery_packager.py
@@ -895,3 +895,56 @@
     metadata = [{k:entry[k] for k in ('path', 'sha256', 'bytes')} for entry in entries]
     assert len(p._readonly_seal_pins(spec, f['policy'], metadata)) == 9
     p.validate_spec(spec, f['policy'])
+
+
+@pytest.mark.parametrize('action', ['validate', 'build'])
+def test_c4_fixed_trusted_archive_authenticates_external_manifest_before_seal_read(fixture, monkeypatch, action):
+    f = fixture
+    spec = c4_spec(f)
+    paper = next(paper for paper in spec['papers'] if paper['label'] == 'draft_v002')
+    original_archive = Path(paper['archive']['path']).read_bytes()
+    policy = replace(f['policy'], v002_archive_sha256=paper['archive']['sha256'])
+    seal = c4_seal(f)
+    changed_seal = save(Path(seal['path']), b'{"unreviewed_synthetic_seal":true}\n')
+    registry_path = Path(spec['exporter_provenance']['path'])
+    registry = json.loads(registry_path.read_bytes())
+    relative = str(Path(seal['path']).relative_to(policy.metadata))
+    registry['source_pins'][relative].update(sha256=changed_seal['sha256'], bytes=changed_seal['bytes'])
+    spec['exporter_provenance'] = save(registry_path, registry)
+    spec['readonly_registry_seal_hardlinks'] = [dict(changed_seal, nlink=2)]
+    manifest_path = Path(paper['manifest']['path'])
+    manifest = json.loads(manifest_path.read_bytes())
+    manifest['members']['generated/source_provenance.json'] = {
+        k:spec['exporter_provenance'][k] for k in ('sha256', 'bytes')}
+    paper['manifest'] = save(manifest_path, manifest)
+    plan = p.make_plan(spec, policy, f['limits'])
+    pin = save(f['tmp']/'self-consistent-invalid-manifest-plan.json', plan)
+    opened, real = [], p.os.open
+    def opening(path, *args, **kwargs):
+        if os.fsdecode(path) == 'ANALYSIS_SEALED.json':
+            opened.append(path)
+        return real(path, *args, **kwargs)
+    monkeypatch.setattr(p.os, 'open', opening)
+    with pytest.raises(ValueError, match='Paper tar member hash mismatch'):
+        if action == 'validate':
+            p.validate(pin, policy, f['limits'])
+        else:
+            p.build(pin, destination(f), policy, f['limits'])
+    assert not opened
+    assert Path(paper['archive']['path']).read_bytes() == original_archive
+    assert not destination(f).exists()
+
+
+def test_c4_fixed_trusted_archive_accepts_original_registry_seal(fixture):
+    f = fixture
+    spec = c4_spec(f)
+    paper = next(paper for paper in spec['papers'] if paper['label'] == 'draft_v002')
+    policy = replace(f['policy'], v002_archive_sha256=paper['archive']['sha256'])
+    seal = Path(c4_seal(f)['path'])
+    before = p.identity(seal.stat())
+    pin = f['plan_pin'](spec)
+    assert p.validate(pin, policy, f['limits'])['valid'] is True
+    result = p.build(pin, destination(f), policy, f['limits'])
+    assert result['complete'] is True
+    assert p.verify_delivery(destination(f), result['local_receipt_pin'], policy, f['limits'])['complete'] is True
+    assert p.identity(seal.stat()) == before

```

