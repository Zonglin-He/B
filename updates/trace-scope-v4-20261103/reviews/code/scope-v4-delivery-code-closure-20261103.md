# SAME Reviewer Delivery Packager C1/C2 Narrow Closure

Requested scope: only the C1 final-verifier and C2 directory-FD ownership
repairs, plus affected synthetic regression/invariants. Not a new full code,
paper, real-input, pin-plan, artifact, scientific or publication audit. Only
this new report is a persistent reviewer output; source/tests and the original
raw report remain unchanged. No real builder/materialization/source inventory,
GPU/service/Git/network/data/GT/RGB/NPY/model/countstream/metric/bootstrap work.
Six reference-mutation demonstrations remain NOT RUN.

## Authenticated Initial Pins

```text
delivery_packager.py
9f0a53e489b5c9fa0de364e9c6f156336d544cd091aa04d87e51f7e3a4661748 46186 bytes
test_delivery_packager.py
5d7e9876657e882fb410a55caa196943087dbe0ab01bf91f1c134e66a52a671f 26735 bytes
/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_code_002/code_v002.tar.gz
2d234c34cb94790782c28ed9c941df94bd0958b287bc154bf9b1b52d615a22d0 19726 bytes
/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_code_002/CODE_FREEZE.json
dd74afa35dcb6d8533eef5dcd39037f89671a017ceb7664a72f82a9709000d05
Original raw report (not edited)
8c8da4b54b441a5523164f7403e94e7c792fa7c58c4f8065d03784eaf64e5674
Original v001 whole-code archive
e62fa728902de586134dc9fb209e3117d612fc844e81bb921197d4d3dca4bd5e
```

The v002 archive SHA and external CODE_FREEZE SHA were independently computed
before decompression/extraction. Bounded standard-library decode (1 MiB maximum;
81920 actual expanded tar bytes), exact three unique regular no-link entries,
each source size/hash and embedded/external CODE_FREEZE equality were verified.
Exclusive extraction used new private0700
`/tmp/delivery-code-closure-w7zh85k1/`. Execution below uses those resident bytes,
not imports from evolving main. The complete two-file diff against the retained
v001 resident source was independently inspected without Git.

Implementation delta observed:
- directory():195-197 registers the child in the cleanup list immediately after
  open and before its initial fstat.
- verify_delivery():691 onward captures directory state; authenticated read()
  returns/binds each of LOCAL_VERIFY.json, MANIFEST.json and delivery.tar.gz.
- verify_delivery():750-755 rechecks exact membership, each original seven-field
  leaf identity, membership again and full directory state before returning.
  Read-only directory state includes mtime/ctime, unlike the ancestry-only
  directory_id guard. No source-policy, scope, estimator or publication action
  was changed. Twelve new tests cover ten action/change combinations, stat
  cleanup, and namespace change during the final leaf pass.

## Protected Synthetic Setup

```python setup
import os,sys,json,hashlib,tempfile,importlib.util,types,traceback,io,contextlib,copy,gzip,tarfile,errno
from pathlib import Path
RESIDENT=Path('/tmp/delivery-code-closure-w7zh85k1')
WORK=Path('/home/wwww/.codex/worktrees/trace-simplification/hzl')
PINS={'delivery_packager.py':'9f0a53e489b5c9fa0de364e9c6f156336d544cd091aa04d87e51f7e3a4661748',
 'test_delivery_packager.py':'5d7e9876657e882fb410a55caa196943087dbe0ab01bf91f1c134e66a52a671f'}
ORIGINAL_RAW=WORK/'docs/scope-v4-delivery-code-independent-20261103.md'
def hashfile(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def source_hashes():return {name:hashfile(WORK/'tools/evolution_fifty'/name) for name in PINS}
assert source_hashes()==PINS and hashfile(ORIGINAL_RAW)=='8c8da4b54b441a5523164f7403e94e7c792fa7c58c4f8065d03784eaf64e5674'
for name,sha in PINS.items():assert hashfile(RESIDENT/'tools/evolution_fifty'/name)==sha
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and sys.dont_write_bytecode
import torch,cv2,pytest
cv2.setNumThreads(1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
def denied(*a,**k):raise AssertionError('No actual model/GPU/service/network')
torch.load=denied;torch.jit.load=denied;torch.cuda.init=denied
def guard_path(value,dir_fd=None):
 if isinstance(value,int):path=Path(os.readlink('/proc/self/fd/'+str(value)))
 else:
  path=Path(os.fsdecode(value))
  if not path.is_absolute():path=(Path.cwd() if dir_fd is None else Path(os.readlink('/proc/self/fd/'+str(dir_fd))))/path
 path=path.resolve()
 for forbidden in ('/home/wwww/hzl/results','/home/wwww/hzl/data',str(WORK/'results'),str(WORK/'data'),str(WORK/'paper_evolution'),'/home/wwww/.cache/clip'):
  assert not path.is_relative_to(forbidden),'No actual paper/experiment/payload read/scan'
original_os_open=os.open
def guarded_open(path,flags,mode=0o777,*,dir_fd=None):
 guard_path(path,dir_fd);return original_os_open(path,flags,mode,dir_fd=dir_fd)
os.open=guarded_open
def audit(event,args):
 if event in ('subprocess.Popen','socket.connect','socket.bind'):raise AssertionError('No subprocess/network')
 if event in ('os.scandir','os.listdir'):guard_path(args[0])
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  if args[1] is not None or Path(os.fsdecode(args[0])).is_absolute():guard_path(args[0])
sys.addaudithook(audit)
for name,path in [('tools',RESIDENT/'tools'),('tools.evolution_fifty',RESIDENT/'tools/evolution_fifty')]:
 package=types.ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value
p=module('tools.evolution_fifty.delivery_packager',RESIDENT/'tools/evolution_fifty/delivery_packager.py')
sys.modules['tools.evolution_fifty'].delivery_packager=p
t=module('independent_delivery_closure_fixtures',RESIDENT/'tools/evolution_fifty/test_delivery_packager.py')
RESULTS=[]
def check(name,fn):
 try:row=dict(name=name,passed=True,detail=fn())
 except BaseException as error:row=dict(name=name,passed=False,error=repr(error),traceback=traceback.format_exc(limit=8))
 RESULTS.append(row);print(json.dumps(row),flush=True)
@contextlib.contextmanager
def fixture():
 with tempfile.TemporaryDirectory(prefix='delivery-narrow-synthetic-') as tmp,pytest.MonkeyPatch.context() as mp:
  f=t.fixture.__wrapped__(Path(tmp),mp)
  assert f['policy'].worktree.is_relative_to('/tmp') and f['policy'].results.is_relative_to('/tmp')
  yield f,mp
def refused(fn):
 try:fn()
 except (ValueError,OSError) as error:return repr(error)
 raise AssertionError('Expected rejection')
print(json.dumps(dict(initial_hashes=source_hashes(),original_raw_sha256=hashfile(ORIGINAL_RAW),resident=str(RESIDENT),CUDA_initialized=torch.cuda.is_initialized(),threads=[torch.get_num_threads(),torch.get_num_interop_threads(),cv2.getNumThreads()])),flush=True)
```

## Replays And Focused Regressions

Original C1 timing and mutations are retained: after the final verifier's
bounded archive read, add FAILED or replace its already-read manifest. The
only necessary hook adaptation is pin_read -> read plus explicit verifier-phase
tracking, because repaired bound() now calls read directly and build also has
earlier legitimate readbacks. No production helper is bypassed or stubbed.
Both full synthetic build and portable verify are exercised.

```python narrow_checks
from pathlib import Path
report=Path('docs/scope-v4-delivery-code-closure-20261103.md').read_text()
exec(compile(report.split('```python setup\n',1)[1].split('\n```',1)[0],'delivery-narrow-setup','exec'))
def C1(action,kind):
 with fixture() as (f,mp):
  plan=f['plan_pin']();target=t.destination(f)
  if action=='verify':original_result=p.build(plan,target,f['policy'],f['limits'])
  original_read,original_verify=p.read,p.verify_delivery;active=[];changed=[]
  def verifying(*a,**kw):
   active.append(True)
   try:return original_verify(*a,**kw)
   finally:active.pop()
  def read(path,*a,**kw):
   result=original_read(path,*a,**kw)
   if active and Path(path)==target/'delivery.tar.gz' and not changed:
    changed.append(True)
    if kind=='late_FAILED':t.save(target/'FAILED.json',dict(complete=False,no_resume=True))
    else:
     name={'manifest_changed':'MANIFEST.json','manifest_same':'MANIFEST.json','receipt_same':'LOCAL_VERIFY.json','archive_same':'delivery.tar.gz'}[kind]
     leaf=target/name;raw=b'{"replacement":true}\n' if kind=='manifest_changed' else leaf.read_bytes()
     replacement=target/'temporary';replacement.write_bytes(raw);replacement.replace(leaf)
   return result
  mp.setattr(p,'read',read);mp.setattr(p,'verify_delivery',verifying)
  if action=='build':fn=lambda:p.build(plan,target,f['policy'],f['limits'])
  else:fn=lambda:p.verify_delivery(target,original_result['local_receipt_pin'],f['policy'],f['limits'])
  error=refused(fn);assert changed
  if action=='build':assert (target/'FAILED.json').exists()
  return dict(error=error,completion_returned=False,owned_FAILED=(target/'FAILED.json').exists(),members=sorted(x.name for x in target.iterdir()))
for action in ('build','verify'):
 for kind in ('late_FAILED','manifest_changed','manifest_same','receipt_same','archive_same'):
  check('C1/'+action+'/'+kind,lambda action=action,kind=kind:C1(action,kind))
def C2():
 with fixture() as (f,mp):
  leaf=f['tmp']/'ordinary.json';leaf.write_bytes(b'{}');opened=set();faults=[]
  original_open,original_close,original_fstat=p.os.open,p.os.close,p.os.fstat
  def open_spy(*a,**kw):fd=original_open(*a,**kw);opened.add(fd);return fd
  def close_spy(fd):original_close(fd);opened.discard(fd)
  def fstat(fd):
   if fd in opened and not faults and os.readlink('/proc/self/fd/'+str(fd))==str(f['tmp']):
    faults.append(fd);raise OSError(errno.EIO,'Synthetic directory fstat error')
   return original_fstat(fd)
  mp.setattr(p.os,'open',open_spy);mp.setattr(p.os,'close',close_spy);mp.setattr(p.os,'fstat',fstat)
  error=refused(lambda:p.read(leaf,100));assert faults and not opened
  return dict(error=error,leaked_directory_fds=len(opened),directory_context_never_yielded=True)
check('C2/original_fstat_exception',C2)
def legal_readonly():
 with fixture() as (f,mp):
  result=p.build(f['plan_pin'](),t.destination(f),f['policy'],f['limits']);target=t.destination(f)
  before={name:(p.identity((target/name).lstat()),(target/name).read_bytes()) for name in ('LOCAL_VERIFY.json','MANIFEST.json','delivery.tar.gz')}
  before_directory=p.identity(target.lstat())
  actual=p.verify_delivery(target,result['local_receipt_pin'],f['policy'],f['limits'])
  assert actual['complete'] is True and p.identity(target.lstat())==before_directory
  assert {name:(p.identity((target/name).lstat()),(target/name).read_bytes()) for name in before}==before
  return dict(complete=True,all_three_identities_and_bytes_unchanged=True,directory_metadata_unchanged=True)
check('legal_readonly_verification',legal_readonly)
def final_namespace():
 with fixture() as (f,mp):
  result=p.build(f['plan_pin'](),t.destination(f),f['policy'],f['limits']);target=t.destination(f)
  original_read,original_observe=p.read,p.observe;archive_read=[];changed=[]
  def read(path,*a,**kw):
   value=original_read(path,*a,**kw)
   if Path(path)==target/'delivery.tar.gz':archive_read.append(True)
   return value
  def observe(path,*a,**kw):
   value=original_observe(path,*a,**kw)
   if archive_read and Path(path)==target/'delivery.tar.gz' and not changed:
    changed.append(True);leaf=target/'MANIFEST.json';replacement=target/'temporary';replacement.write_bytes(leaf.read_bytes());replacement.replace(leaf)
   return value
  mp.setattr(p,'read',read);mp.setattr(p,'observe',observe)
  error=refused(lambda:p.verify_delivery(target,result['local_receipt_pin'],f['policy'],f['limits']))
  assert changed and 'directory membership drift' in error
  return dict(error=error,completion_returned=False,samebytes_prior_leaf_swap_at_last_rebind=True)
check('final_leaf_pass_namespace_guard',final_namespace)
print(json.dumps(dict(total=len(RESULTS),passed=sum(r['passed'] for r in RESULTS),final_hashes=source_hashes(),original_raw_sha256=hashfile(ORIGINAL_RAW),CUDA_initialized=torch.cuda.is_initialized(),threads=[torch.get_num_threads(),torch.get_num_interop_threads(),cv2.getNumThreads()])),flush=True)
assert all(r['passed'] for r in RESULTS) and source_hashes()==PINS and not torch.cuda.is_initialized()
```

```python regression
from pathlib import Path
report=Path('docs/scope-v4-delivery-code-closure-20261103.md').read_text()
exec(compile(report.split('```python setup\n',1)[1].split('\n```',1)[0],'delivery-narrow-setup','exec'))
with tempfile.TemporaryDirectory(prefix='delivery-closure-pytest-') as tmp:
 code=pytest.main(['-q','-c','/dev/null','--rootdir',str(RESIDENT),'--import-mode=importlib','-p','no:cacheprovider','--basetemp',str(Path(tmp)/'pytest'),str(RESIDENT/'tools/evolution_fifty/test_delivery_packager.py')])
print(json.dumps(dict(pytest_exit=int(code),final_hashes=source_hashes(),original_raw_sha256=hashfile(ORIGINAL_RAW),CUDA_initialized=torch.cuda.is_initialized(),threads=[torch.get_num_threads(),torch.get_num_interop_threads(),cv2.getNumThreads()])),flush=True)
assert code==0 and source_hashes()==PINS and not torch.cuda.is_initialized()
```

Replay from the worktree, replacing `narrow_checks` by `regression` for that run:

```bash
awk '/^```python narrow_checks$/{active=1;next} active && /^```$/{exit} active{print}' docs/scope-v4-delivery-code-closure-20261103.md | env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/wwww/miniconda3/envs/mlmp/bin/python -B -
```

Disposition pending observed execution; no prospective pass is claimed.

## Actual Narrow Results

Both documented execution blocks completed with exit 0 against the authenticated
v002 resident source. **13/13 independent narrow checks passed. The unmodified
supplied module ran 60 tests, all passed in 2.55s** (pytest exit 0). Main's
reported 0.67s remains an attributed author timing, not this independent result.
No new setup/harness failure occurred during this closure run. The old v001
guard failure and original findings remain solely in the unchanged original raw.

| Check | Actual outcome |
| --- | --- |
| Original C1 late FAILED, full synthetic build | ValueError: Failed/foreign output members; no completion returned; FAILED retained |
| Original C1 replaced already-read manifest, full synthetic build | ValueError: Final output leaf identity drift; no completion returned; owned FAILED retained |
| Same two mutations during standalone portable verification | Both rejected; no completion returned |
| Same-byte/new-inode substitution of manifest, receipt or archive, build | All three rejected on original leaf identity; owned FAILED retained |
| Same three substitutions during standalone portable verification | All three rejected on original leaf identity |
| Original C2 initial directory-fstat error | Original OSError propagated; zero open directory FDs remained; directory context never yielded |
| Legal build followed by read-only verification | Success; all three file identities and bytes, and directory metadata, unchanged |
| Namespace substitution during last final-leaf observation | Same-byte replacement of earlier manifest rejected by final directory metadata guard |

The `owned_FAILED` output key in the generic harness records marker existence.
For build faults, the existing owned failure handler creates/preserves it. For
standalone verification's late-FAILED case, the marker is the synthetic injected
fault, not a write performed by the verifier. Standalone verification of other
mutations remains read-only and does not create FAILED. A receipt may already
exist when final verification rejects; it does not override failure or count as
a successful returned completion. The normal read-only control validates this
distinction without suppressing an exception or changing the output contract.

All runtime observations were CPU-only: CUDA initialized false, torch intra-op
1, torch inter-op 1, cv2 threads 1. Bytecode, pytest cache and plugin autoload
were disabled. Protected fixtures use /tmp roots, small synthetic archives,
reviews/metadata/source and an explicit synthetic policy; none is an actual
publication input, real source snapshot, or artifact-verification substitute.

## Disposition And Limits

**SAME-reviewer narrow review COMPLETE. C1 CLOSED and C2 CLOSED on source
9f0a53e4... / tests 5d7e9876... . No unresolved finding was identified in this
repair scope.**

C1's original two failure modes and the affected same-byte/namespace regressions
now reject before completion return. The verifier no longer adopts later
leaf inodes as its original verified leaves and checks final exact membership.
C2 now registers ownership immediately after acquiring the child directory FD,
so the original fallible fstat boundary is covered by cleanup. The two-function
delta does not change the original byte-preservation, negative/null, R1 PARTIAL,
current-code-not-all-frozen-runtime, excluded-payload or no-network/publication
scope contracts. Running the requested 60-test regression is not a new paper,
science or full historical CODE review.

This is bounded verification of the repaired checks and tested fault windows,
not an atomic immutable snapshot against arbitrary concurrent writers or a
promise that files cannot change after checks/return. No filesystem immutability
or scientific admission is claimed. Actual operator specification, exact pin
plan and eventual local package still require their distinct reviews/validation.
This CODE closure grants no GT/data/GPU/network/upload/publication permission.

No implementation, test, old raw report, frozen code archive or real artifact
was edited. Original v001 whole bytes and findings remain preserved. New final
source hashes observed inside both completed harnesses matched their initial
pins, and the original raw report remained exactly:

```text
8c8da4b54b441a5523164f7403e94e7c792fa7c58c4f8065d03784eaf64e5674
```

Exact closed source/test pins:

```text
9f0a53e489b5c9fa0de364e9c6f156336d544cd091aa04d87e51f7e3a4661748 delivery_packager.py
5d7e9876657e882fb410a55caa196943087dbe0ab01bf91f1c134e66a52a671f test_delivery_packager.py
```
