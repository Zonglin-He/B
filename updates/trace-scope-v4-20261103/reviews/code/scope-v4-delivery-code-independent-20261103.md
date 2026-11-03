# Independent Scope-v4 Delivery Packager CODE Review

New distinct packager CODE scope only. Not a repeated paper/exporter/scorer
review, real pin-plan validation, artifact admission or publication permission.
No actual operator specification/plan exists for this review. All packaging
execution below is against ordinary synthetic fixtures under /tmp, never real
paper/evidence/source input inventory. Six reference-mutation demonstrations
remain NOT RUN. No Git, network, services, GPU/model/data/GT/RGB/NPY/countstream
access, metric/bootstrap execution or actual package materialization is allowed.

## Authenticated Initial Bytes

```text
tools/evolution_fifty/delivery_packager.py
607991a2fed6a7f6341e8e9c3d3634c05ad10f01005d6c4c62b08ddc90565cad
45568 bytes
tools/evolution_fifty/test_delivery_packager.py
081668380a9272a8bba7a4a0a2ba3aa4c4cab9dbca946c9acaa967a70b3a880c
22097 bytes
/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_code_001/code_v001.tar.gz
e62fa728902de586134dc9fb209e3117d612fc844e81bb921197d4d3dca4bd5e
18398 bytes, 71680 expanded tar bytes
/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_code_001/CODE_FREEZE.json
d0bac9e9ebe8cceee32c2fb7055c0a773cbab4a14d8d65a1930c26cb8d6a74f6
488 bytes
```

Archive SHA verified before decompression/extraction. Bounded standard-library
gzip decode (1 MiB expanded maximum), exact three unique regular no-link member
names, member sizes and hashes, and equality of embedded/external CODE_FREEZE
were checked before exclusive extraction into new private0700
`/tmp/delivery-code-independent-it94hdg9/`. Only these preserved source bytes
are used for execution. No actual paper archive or manuscript review was opened.
All 842 implementation lines and 422 test lines were read independently.

One setup pathname error is preserved separately: an initial hash command tried
`tests/test_delivery_packager.py`, which does not exist. The supplied test lives
at `tools/evolution_fifty/test_delivery_packager.py`; correcting only that lookup
matched its exact supplied pin. This is not an implementation/artifact finding.

## Protected Synthetic Harness

```python setup
import os,sys,json,hashlib,tempfile,importlib.util,types,traceback,io,contextlib,copy,gzip,tarfile,errno,stat
from pathlib import Path
from dataclasses import replace
RESIDENT=Path('/tmp/delivery-code-independent-it94hdg9')
WORK=Path('/home/wwww/.codex/worktrees/trace-simplification/hzl')
PINS={'delivery_packager.py':'607991a2fed6a7f6341e8e9c3d3634c05ad10f01005d6c4c62b08ddc90565cad',
 'test_delivery_packager.py':'081668380a9272a8bba7a4a0a2ba3aa4c4cab9dbca946c9acaa967a70b3a880c'}
def hashfile(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def source_hashes():return {n:hashfile(WORK/'tools/evolution_fifty'/n) for n in PINS}
assert source_hashes()==PINS
for n,sha in PINS.items():assert hashfile(RESIDENT/'tools/evolution_fifty'/n)==sha
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and sys.dont_write_bytecode
import torch,cv2,pytest
cv2.setNumThreads(1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
def denied(*a,**k):raise AssertionError('No actual model/GPU/service/network access')
torch.load=denied;torch.jit.load=denied;torch.cuda.init=denied
def audit(event,args):
 if event in ('subprocess.Popen','socket.connect','socket.bind'):raise AssertionError('No subprocess/network')
 if event in ('open','os.scandir','os.listdir') and isinstance(args[0],(str,bytes,os.PathLike)):
  path=Path(os.fsdecode(args[0])).resolve()
  for forbidden in ('/home/wwww/hzl/results','/home/wwww/hzl/data',str(WORK/'results'),str(WORK/'data'),str(WORK/'paper_evolution'),'/home/wwww/.cache/clip'):
   assert not path.is_relative_to(forbidden),'No actual experiment/paper/payload scan/read'
sys.addaudithook(audit)
for name,path in [('tools',RESIDENT/'tools'),('tools.evolution_fifty',RESIDENT/'tools/evolution_fifty')]:
 package=types.ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value
p=module('tools.evolution_fifty.delivery_packager',RESIDENT/'tools/evolution_fifty/delivery_packager.py')
sys.modules['tools.evolution_fifty'].delivery_packager=p
t=module('independent_delivery_fixtures',RESIDENT/'tools/evolution_fifty/test_delivery_packager.py')
assert Path(p.__file__).is_relative_to(RESIDENT)
RESULTS=[]
def check(name,fn):
 try:row=dict(name=name,passed=True,detail=fn())
 except BaseException as error:row=dict(name=name,passed=False,error=repr(error),traceback=traceback.format_exc(limit=8))
 RESULTS.append(row);print(json.dumps(row),flush=True)
@contextlib.contextmanager
def fixture():
 with tempfile.TemporaryDirectory(prefix='delivery-synthetic-') as tmp,pytest.MonkeyPatch.context() as mp:
  f=t.fixture.__wrapped__(Path(tmp),mp)
  assert f['policy'].worktree.is_relative_to('/tmp') and f['policy'].results.is_relative_to('/tmp')
  yield f,mp
print(json.dumps(dict(source_hashes=source_hashes(),resident=str(RESIDENT),CUDA_initialized=torch.cuda.is_initialized(),threads=[torch.get_num_threads(),torch.get_num_interop_threads(),cv2.getNumThreads()])),flush=True)
```

```python baseline
from pathlib import Path
report=Path('docs/scope-v4-delivery-code-independent-20261103.md').read_text()
exec(compile(report.split('```python setup\n',1)[1].split('\n```',1)[0],'delivery-independent-setup','exec'))
with tempfile.TemporaryDirectory(prefix='delivery-pytest-') as tmp:
 code=pytest.main(['-q','-c','/dev/null','--rootdir',str(RESIDENT),'--import-mode=importlib','-p','no:cacheprovider','--basetemp',str(Path(tmp)/'pytest'),str(RESIDENT/'tools/evolution_fifty/test_delivery_packager.py')])
print(json.dumps(dict(pytest_exit=int(code),source_hashes=source_hashes(),CUDA_initialized=torch.cuda.is_initialized(),threads=[torch.get_num_threads(),torch.get_num_interop_threads(),cv2.getNumThreads()])),flush=True)
assert code==0 and source_hashes()==PINS and not torch.cuda.is_initialized()
```

Replay any named block with the same environment, e.g.:

```bash
awk '/^```python baseline$/{active=1;next} active && /^```$/{exit} active{print}' docs/scope-v4-delivery-code-independent-20261103.md | env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/wwww/miniconda3/envs/mlmp/bin/python -B -
```

Disposition pending actual independent probes. No assumed verdict or grade.

## Confirmed Finding C1 P2: Final Verifier Accepts A Failed Or Changed Packet

Exact original implementation: **delivery_packager.py:690-744**, called at
**787-790** immediately before build returns successful completion.
The verifier enumerates exactly three output names only at 691-697, reads the
receipt and manifest, then reads/verifies the archive. It never rechecks final
membership or rebinds the receipt/manifest/archive leaves before returning.
The directory context checks directory identity, not child membership/content.

Both independent ordinary synthetic reproductions in `finalization_probes`
below executed the full original synthetic build (no builder stubs). The only
instrumentation performs an ordinary /tmp filesystem change after the final
archive's bounded read returns, before tar verification and successful return:

1. Add FAILED.json, complete=false. Build returns complete=true with
   FAILED.json, LOCAL_VERIFY.json, MANIFEST.json, delivery.tar.gz all present.
   This contradicts the exact three-file/any-FAILED-invalidates-completion
   contract at SCOPE:81. The FAILED marker is not dropped; it is simply missed.
2. Replace the already-read manifest leaf with `{"replacement":true}\n`.
   Build returns complete=true and publishes a local receipt whose manifest
   hash no longer matches the actual named manifest file.

Actual second reproduction: expected manifest SHA
`c2c6274715a03777d50a3195319efa5020b171f71153003f050a999e3ff47a09`,
actual SHA `11ac1e1db6ea7dd24ef55c6dd5d83d687eda72205e21061b841ccbc7ffddf4c4`.
The first used manifest SHA
`7c4813fe2a6edb5222b81a9c8b70e1bb2d38ceb89a1f3db18f5b562910a121ac`
unchanged while the fourth FAILED member appeared. Paths vary by synthetic
fixture and naturally change manifest hashes; the mismatch/membership result
is deterministic. Both source hashes remained the exact original pins.

This is a finite final-verification window reproduced before return, not a
claim that arbitrary modifications after a completed return can be prevented.
The existing early write/read binding tests do not cover it. Required repair
boundary: final exact membership and all final output leaf bindings, including
failure-state detection, must still hold when verification declares completion.
No real package exists or was touched. Original scope remains in progress;
this is an early confirmed finding, not the final review verdict.

## Initial Harness Failure (Not A Product Finding)

The first baseline command produced **24 passed / 24 failed in 1.41s**, then
its temporary-directory cleanup hit the same assertion. The reviewer audit
hook resolved an `os.open('paper_evolution', ..., dir_fd=<synthetic parent>)`
audit event relative to cwd instead of its directory FD. It consequently
misclassified a synthetic /tmp directory as the real worktree's paper tree.
Representative stack: delivery_packager.py:195 -> audit event
`('open', ('paper_evolution', None, 720896))` -> reviewer
`AssertionError('No actual experiment/paper/payload scan/read')`.
No actual paper/data was read. The original setup block and failure count are
retained above; no source or supplied test changed. Corrected setup below
authenticates actual os.open relative-directory context rather than guessing.

```python setup_v2
from pathlib import Path
report=Path('docs/scope-v4-delivery-code-independent-20261103.md').read_text()
setup=report.split('```python setup\n',1)[1].split('\n```',1)[0]
begin=setup.index('def audit(event,args):');end=setup.index('sys.addaudithook(audit)',begin)
corrected='''def guard_path(value, dir_fd=None):
 if isinstance(value,int):path=Path(os.readlink('/proc/self/fd/'+str(value)))
 else:
  path=Path(os.fsdecode(value))
  if not path.is_absolute():
   parent=Path.cwd() if dir_fd is None else Path(os.readlink('/proc/self/fd/'+str(dir_fd)))
   path=parent/path
 path=path.resolve()
 for forbidden in ('/home/wwww/hzl/results','/home/wwww/hzl/data',str(WORK/'results'),str(WORK/'data'),str(WORK/'paper_evolution'),'/home/wwww/.cache/clip'):
  assert not path.is_relative_to(forbidden),'No actual experiment/paper/payload scan/read'
original_os_open=os.open
def guarded_open(path,flags,mode=0o777,*,dir_fd=None):
 guard_path(path,dir_fd)
 return original_os_open(path,flags,mode,dir_fd=dir_fd)
os.open=guarded_open
def audit(event,args):
 if event in ('subprocess.Popen','socket.connect','socket.bind'):raise AssertionError('No subprocess/network')
 if event in ('os.scandir','os.listdir'):guard_path(args[0])
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  # os.open has mode=None; its real dir_fd context is checked by guarded_open.
  if args[1] is not None or Path(os.fsdecode(args[0])).is_absolute():guard_path(args[0])
'''
exec(compile(setup[:begin]+corrected+setup[end:],'delivery-independent-setup-v2','exec'))
```

```python baseline_v2
from pathlib import Path
report=Path('docs/scope-v4-delivery-code-independent-20261103.md').read_text()
exec(compile(report.split('```python setup_v2\n',1)[1].split('\n```',1)[0],'delivery-protected-v2','exec'))
with tempfile.TemporaryDirectory(prefix='delivery-pytest-v2-') as tmp:
 code=pytest.main(['-q','-c','/dev/null','--rootdir',str(RESIDENT),'--import-mode=importlib','-p','no:cacheprovider','--basetemp',str(Path(tmp)/'pytest'),str(RESIDENT/'tools/evolution_fifty/test_delivery_packager.py')])
print(json.dumps(dict(pytest_exit=int(code),source_hashes=source_hashes(),CUDA_initialized=torch.cuda.is_initialized(),threads=[torch.get_num_threads(),torch.get_num_interop_threads(),cv2.getNumThreads()])),flush=True)
assert code==0 and source_hashes()==PINS and not torch.cuda.is_initialized()
```

```python finalization_probes
from pathlib import Path
report=Path('docs/scope-v4-delivery-code-independent-20261103.md').read_text()
exec(compile(report.split('```python setup_v2\n',1)[1].split('\n```',1)[0],'delivery-protected-v2','exec'))
def finalization(kind):
 with fixture() as (f,mp):
  plan=f['plan_pin']();target=t.destination(f);original=p.pin_read;changes=[]
  def read(pin,*a,**kw):
   content=original(pin,*a,**kw)
   if Path(pin['path'])==target/'delivery.tar.gz' and not changes:
    if kind=='late_FAILED':t.save(target/'FAILED.json',dict(complete=False,no_resume=True))
    else:
     replacement=target/'temporary';replacement.write_bytes(b'{"replacement":true}\n');replacement.replace(target/'MANIFEST.json')
    changes.append(True)
   return content
  mp.setattr(p,'pin_read',read)
  result=p.build(plan,target,f['policy'],f['limits'])
  assert changes and result['complete'] is True
  detail=dict(returned_complete=result['complete'],late_failed=(target/'FAILED.json').exists(),output_members=sorted(x.name for x in target.iterdir()),expected_manifest_sha=result['manifest']['sha256'],actual_manifest_sha=p.sha((target/'MANIFEST.json').read_bytes()))
  if kind=='late_FAILED':assert detail['late_failed'] and len(detail['output_members'])==4
  else:assert detail['expected_manifest_sha']!=detail['actual_manifest_sha']
  return detail
for kind in ('late_FAILED','late_manifest_replacement'):check(kind,lambda kind=kind:finalization(kind))
print(json.dumps(dict(confirmed_reproductions=sum(r['passed'] for r in RESULTS),source_hashes=source_hashes(),CUDA_initialized=torch.cuda.is_initialized())),flush=True)
assert all(r['passed'] for r in RESULTS) and source_hashes()==PINS
```

## Remaining Original-Scope Controls

```python remaining_controls
from pathlib import Path
report=Path('docs/scope-v4-delivery-code-independent-20261103.md').read_text()
exec(compile(report.split('```python setup_v2\n',1)[1].split('\n```',1)[0],'delivery-protected-v2','exec'))
def refused(fn):
 try:fn()
 except (ValueError,OSError,EOFError,tarfile.TarError) as error:return repr(error)
 raise AssertionError('Expected rejection')
def oracle():
 with fixture() as (f,mp):
  plan=f['plan_pin']();out=t.destination(f);result=p.build(plan,out,f['policy'],f['limits'])
  blobs=t.unpack((out/'delivery.tar.gz').read_bytes());manifest=json.loads(blobs['MANIFEST.json'])
  for r in manifest['source_records']:assert blobs[r['member']]==Path(r['pin']['path']).read_bytes()
  assert p.verify_delivery(out,result['local_receipt_pin'],f['policy'],f['limits'])['complete'] is True
  assert b'"delta":-2,"CI":null' in blobs['evidence/M01_offline_score_synthetic/DATASET_RESULTS.json']
  scope=blobs['DELIVERY_SCOPE.md'].decode();assert all(v in scope for v in ('PARTIAL','not a claim of frozen-runtime identity','NOT RUN','No Git, network','not turnkey'))
  return dict(all_pinned_source_bytes_preserved=len(manifest['source_records']),negative_and_null_preserved=True,R1_PARTIAL=True,portable_verify=True)
check('independent_byte_and_scope_oracle',oracle)
def metadata_roles():
 with fixture() as (f,mp):
  denied=[]
  for name in ('x/predictions/RESULT.json','x/episodes/RESULT.json','x/runtime/RESULT.json','x/reference/RESULT.json','x/labels/RESULT.json','x/annotations/RESULT.json','x/states/RESULT.json','x/cost.jsonl','x/a.npy'):
   denied.append(refused(lambda n=name:p.metadata_path(dict(path=str(f['policy'].metadata/n)),f['policy'])))
  return dict(refused_roles=len(denied))
check('metadata_role_allowlist',metadata_roles)
def preflight():
 with fixture() as (f,mp):
  plan_pin=f['plan_pin']();plan=json.loads(Path(plan_pin['path']).read_bytes());leaf=Path(plan['code'][-1]['path']);leaf.unlink();os.mkfifo(leaf)
  original=p.pin_read;opened=[]
  def read(pin,*a,**k):opened.append(pin['path']);return original(pin,*a,**k)
  mp.setattr(p,'pin_read',read);error=refused(lambda:p.build(plan_pin,t.destination(f),f['policy'],f['limits']))
  assert opened==[plan_pin['path']] and not t.destination(f).exists()
  return dict(error=error,only_plan_opened=True,no_output=True)
check('late_graph_FIFO_zero_source_contents',preflight)
def wronghash():
 with fixture() as (f,mp):
  plan=f['plan_pin']();bad=dict(plan,sha256='0'*64);calls=[]
  def parser(*a,**k):calls.append(1);raise AssertionError('Invalid hash reached parser')
  mp.setattr(p,'parsed',parser);error=refused(lambda:p.validate(bad,f['policy'],f['limits']))
  assert not calls;return dict(error=error,parser_calls=0)
check('plan_hash_before_parse',wronghash)
def parent_swap():
 with fixture() as (f,mp):
  folder=f['tmp']/'ordinary';folder.mkdir();leaf=folder/'pin.json';leaf.write_bytes(b'{}');original=p.os.read;changes=[]
  def read(fd,size):
   raw=original(fd,size)
   if raw and not changes and os.readlink('/proc/self/fd/'+str(fd))==str(leaf):
    folder.rename(f['tmp']/'held');folder.mkdir();leaf.write_bytes(b'{}');changes.append(True)
   return raw
  mp.setattr(p.os,'read',read);error=refused(lambda:p.read(leaf,100,p.sha(b'{}'),2))
  assert changes;return dict(error=error,same_bytes_parent_swap_refused=True)
check('directory_FD_parent_replacement',parent_swap)
def foreign_mkdir():
 with fixture() as (f,mp):
  plan=f['plan_pin']();target=t.destination(f);original=p.os.mkdir;changed=[]
  def mkdir(path,*a,**k):
   if path==target.name and not changed:
    changed.append(True);original(path,*a,**k);(target/'sentinel').write_bytes(b'foreign')
   return original(path,*a,**k)
  mp.setattr(p.os,'mkdir',mkdir);error=refused(lambda:p.build(plan,target,f['policy'],f['limits']))
  assert changed and sorted(x.name for x in target.iterdir())==['sentinel'] and (target/'sentinel').read_bytes()==b'foreign'
  return dict(error=error,foreign_output_untouched=True,no_owned_FAILED=True)
check('exclusive_mkdir_foreign_race',foreign_mkdir)
def paper_format(kind):
 with fixture() as (f,mp):
  paper=f['spec']['papers'][0];manifest=Path(paper['manifest']['path']).read_bytes();content=list(t.unpack(Path(paper['archive']['path']).read_bytes()).items())
  if kind=='duplicate':raw=t.archive_bytes(content+[content[0]])
  elif kind=='bad_digest':raw=t.archive_bytes([(name,b'modified' if i==0 else data) for i,(name,data) in enumerate(content)])
  elif kind=='hardlink':
   stream=io.BytesIO()
   with tarfile.open(fileobj=stream,mode='w') as archive:
    h=tarfile.TarInfo(content[0][0]);h.type=tarfile.LNKTYPE;h.linkname='foreign';archive.addfile(h)
   raw=gzip.compress(stream.getvalue())
  elif kind=='special':
   stream=io.BytesIO()
   with tarfile.open(fileobj=stream,mode='w') as archive:
    h=tarfile.TarInfo(content[0][0]);h.type=tarfile.FIFOTYPE;archive.addfile(h)
   raw=gzip.compress(stream.getvalue())
  else:raise AssertionError(kind)
  return dict(error=refused(lambda:p.paper_check(raw,manifest,paper,f['limits'])),extraction=False)
for kind in ('duplicate','bad_digest','hardlink','special'):check('paper_'+kind,lambda kind=kind:paper_format(kind))
def compressed_cap(kind):
 with fixture() as (f,mp):
  raw=gzip.compress(b'\0'*8192);parsed=[]
  def parser(*a,**k):parsed.append(True);raise AssertionError('Expanded cap reached tar parser')
  if kind=='paper':
   paper=f['spec']['papers'][0];manifest=Path(paper['manifest']['path']).read_bytes();limits=replace(f['limits'],paper_expanded=4096)
   fn=lambda:p.paper_check(raw,manifest,paper,limits)
  else:
   limits=replace(f['limits'],package=4096);fn=lambda:p.verify_packet(raw,{'x':b'x'},limits)
  mp.setattr(p.tarfile,'open',parser);error=refused(fn);assert not parsed
  return dict(error=error,tar_parser_calls=0)
for kind in ('paper','packet'):check(kind+'_expanded_cap_before_tar_parse',lambda kind=kind:compressed_cap(kind))
def create_plan_failure():
 with fixture() as (f,mp):
  pin=t.save(f['tmp']/'operator.json',f['spec']);out=t.destination(f,'plan');original=p.os.fsync;calls=[]
  def fsync(fd):
   if os.readlink('/proc/self/fd/'+str(fd))==str(out/'PIN_PLAN.json'):
    calls.append(True);raise OSError(errno.EIO,'Synthetic owned plan fsync error')
   return original(fd)
  mp.setattr(p.os,'fsync',fsync);error=refused(lambda:p.create_plan(pin,out,f['policy'],f['limits']))
  assert calls and not (out/'LOCAL_VERIFY.json').exists()
  retry=refused(lambda:p.create_plan(pin,out,f['policy'],f['limits']))
  return dict(error=error,retry=retry,no_completion_claim=True)
check('plan_fsync_failure_no_success_no_reuse',create_plan_failure)
def directory_fstat_failure():
 with fixture() as (f,mp):
  leaf=f['tmp']/'ordinary.json';leaf.write_bytes(b'{}');opened=set();faults=[]
  original_open=p.os.open;original_close=p.os.close;original_fstat=p.os.fstat
  def open_spy(*a,**k):fd=original_open(*a,**k);opened.add(fd);return fd
  def close_spy(fd):original_close(fd);opened.discard(fd)
  def fstat(fd):
   if fd in opened and not faults and os.readlink('/proc/self/fd/'+str(fd))==str(f['tmp']):
    faults.append(fd);raise OSError(errno.EIO,'Synthetic directory fstat error')
   return original_fstat(fd)
  mp.setattr(p.os,'open',open_spy);mp.setattr(p.os,'close',close_spy);mp.setattr(p.os,'fstat',fstat)
  error=refused(lambda:p.read(leaf,100));leaked=sorted(opened)
  detail=dict(error=error,leaked_directory_fds=len(leaked),leaked_paths=[os.readlink('/proc/self/fd/'+str(fd)) for fd in leaked])
  for fd in leaked:close_spy(fd)
  assert faults and len(leaked)==1
  return detail
check('directory_fstat_exception_cleanup_reproduction',directory_fstat_failure)
print(json.dumps(dict(total=len(RESULTS),passed=sum(r['passed'] for r in RESULTS),source_hashes=source_hashes(),CUDA_initialized=torch.cuda.is_initialized(),threads=[torch.get_num_threads(),torch.get_num_interop_threads(),cv2.getNumThreads()])),flush=True)
assert all(r['passed'] for r in RESULTS) and not torch.cuda.is_initialized()
```

## Confirmed Finding C2 P3: Directory-FD Registration Follows A Fallible Stat

Exact location: **delivery_packager.py:195-197** and cleanup at **204-205**.
`child = os.open(...)` succeeds, but the child is appended to the owned `fds`
list only after `os.fstat(child)` is evaluated for the directory chain. If that
stat raises, the finally block closes the preceding directory FDs, not this
new child FD. The public read/validate/create/build helpers can propagate that
error while leaving the child open in a long-lived calling process.

Actual synthetic reproduction (`directory_fstat_exception_cleanup_reproduction`
above) injected a single OSError(EIO) from the real newly opened directory FD's
fstat. Read raised as expected but the open/close tracker found **one still-open
directory FD**, `/tmp/delivery-synthetic-qvioun__` in this run. The harness
confirmed the live FD through /proc/self/fd, then explicitly closed it so the
review itself left no leaked descriptor. No code/state mutation was required.

Severity is **P3**, not a scientific or packet-integrity blocker: a one-shot
failing CLI process exits and the OS reclaims descriptors. It is nevertheless
an exception-ownership gap in reusable functions and does not meet the stated
cleanup intent. Register an acquired FD before fallible work on it; the existing
read-error/close-error test does not exercise this earlier stat boundary.

## Actual Results And Scope Assessment

After the reviewer-only dir_fd guard correction, the unmodified frozen module
ran **48 tests, all passed in 1.58s**, pytest exit 0. Main's .41s/.39s timings
are attributed author results, not substituted for this independent timing.
The first reviewer setup result (24 passed, 24 guard failures in 1.41s, plus
cleanup guard assertion) remains retained and is not counted as product failure.

Independent supplemental execution comprised:

- **13 expected-behavior controls**, all passed: exact source/paper/raw-review
  byte oracle with negative/null preservation and portable verifier; nine
  metadata role exclusions; late graph FIFO before source-content opens; hash
  before parse; same-byte parent-directory swap; raced foreign mkdir untouched;
  duplicate/hash-size/hardlink/special tar members; paper and packet expanded
  caps before tar parsing; plan fsync failure without successful return/reuse.
- **Two C1 defect reproductions**, both confirmed: completion with a late
  FAILED member and completion with a replaced already-read manifest.
- **One C2 exception-cleanup reproduction**, confirmed: one unclosed child
  directory FD after its initial fstat fails, then cleaned up by the harness.

The harness's `passed=true` on defect reproductions means the specified faulty
outcome was reproduced, not that those implementation behaviors are correct.
All three execution blocks retained above are replayable with the documented
hidden-CUDA, -B, no-plugin/no-cache, single-thread environment. Actual torch
intra/inter-op and cv2 counts were 1/1/1; CUDA initialized remained false.

### Trust Boundaries Inspected

1. **Input policy and schema.** The production policy fixes the v002 archive,
   both raw manuscript review hashes and disposition hash; supplied independent
   CODE report path/hash/size, exact exporter registry and remaining operator
   pins are external inputs. ACK must exactly match typed JSON with R1 PARTIAL,
   unrecovered local construction false, no acceptance or publication action.
   Actual production pins were inspected as source constants only. No real
   paper/review/archive/result evidence was independently re-audited here.
2. **Code snapshot.** Explicit first-party roots, notices and text configs are
   bounded/enumerated without importing research code. Source/config bytes are
   UTF-8 and NUL-checked. Hidden/known-sensitive/payload/vendor roots are excluded;
   symlink/special input and single-link/role collisions are refused. These are
   path/byte rules, not semantic credential/content certification. The package
   truthfully labels code as current first-party source, not all historical
   frozen runtimes, all dependencies or turnkey reproduction.
3. **Metadata and original papers.** Metadata roles are derived from the pinned
   exporter registry, bounded and later bound to the exact reviewed v002 manifest.
   All materialization leaves are stat-checked before their contents are opened;
   hashes precede parsing. Complete original archives are retained as bytes, with
   exact permitted paper member and four derived-composite verification, no
   extraction. Tar decompression is bounded before parsing; links, unexpected
   members, duplicates and non-padding trailing bytes are refused. Source total,
   packet/member, expansion and disk reserve checks are present.
4. **Publication and science.** No metric, bootstrap, positivity filter, result
   rewriting, Git/network/subprocess/imported-research execution exists in this
   worker. The independent oracle preserved every synthetic pinned source byte,
   including `{"delta":-2,"CI":null}` and both raw review byte strings. SCOPE
   retains R1 PARTIAL, exclusions, NOT RUN demonstrations and separate later
   main-author B publication. This is not a judgment that any scientific
   finding is accepted, reproducible or positive.
5. **Plan-stage limit.** `create_plan` requires a well-formed explicit CODE
   report pin in its spec; it does not semantically grade Markdown or prove a
   completed/positive review. It snapshots code and reads the registry before
   later full `validate`/`build` source materialization authenticates every
   report/document/paper pin. A successfully created plan is therefore only a
   candidate plan, never implicit independent exact-plan admission. The user's
   required separate exact-plan validation and operator decision remain external.
6. **Output lifecycle.** Exclusive new destinations, O_EXCL leaves, bounded
   FD/path writes, persisted readback, fixed output names, no retry/resume and
   owned FAILED attempts cover tested early faults. Failed plan creation cannot
   return a success pin and its partial root cannot be reused, though it has no
   complete-packet receipt. C1 is the distinct missed final revalidation window;
   C2 is distinct from normal-path and close-error descriptor cleanup.

## Original-Scope Disposition

**Original CODE review COMPLETE with C1 P2 OPEN and C2 P3 OPEN.** C1 prevents
unqualified completion/verification closure on these bytes. C2 is a lower-priority
library cleanup issue, not evidence of scientific alteration. No third material
finding was confirmed in this bounded review. Findings are not a desired grade
or verdict and do not concern old closed scopes.

Both original whole source files remain preserved in the authenticated supplied
code archive and the private resident extraction. No implementation/test edits
or real plan/package creation were made by this reviewer. This report is the
sole persistent new output. A later repair must preserve this original evidence
and receive a separate same-reviewer exact-hash narrow closure; no future repair
or actual publication package is approved by this original review.

Final execution-time source hashes matched the initial exact pins:

```text
607991a2fed6a7f6341e8e9c3d3634c05ad10f01005d6c4c62b08ddc90565cad delivery_packager.py
081668380a9272a8bba7a4a0a2ba3aa4c4cab9dbca946c9acaa967a70b3a880c test_delivery_packager.py
```
