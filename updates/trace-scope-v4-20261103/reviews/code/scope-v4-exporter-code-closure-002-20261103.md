# SAME Reviewer Residual F2 Publication Repair Review

This scope is only the residual F2 anonymous-staging/publication delta, not a
new exporter/scientific/manuscript review. F1 remains CLOSED and is not reopened.
No actual paper freeze, archive, experiment content, GT/RGB/NPY/model read,
GPU/service query, bootstrap or publication is authorized or performed here.
Only this report and ordinary synthetic scratch fixtures are reviewer writes.

Start pins independently observed on 2026-11-03:

```text
freeze_snapshot.py      49a427ad2b55a203749d5a30a98a8b27e23d1524aabcb42faf61007519bdd059
test_export_scope_v4.py 6d6f3ef1ba7835ff9bb23449bfa0e5fdda6fd88bd598cc60e7e09cf82f664aea
export_scope_v4.py      f6bad26234d37ee77f8750c7e57f46d40c5d7237d75fd9a3f576051455ba56d2
```

All three whole files copied unedited into /tmp/exporter-f2-closure-DWaiXY/
and authenticated before use. Prior raw reports remain immutable:

```text
exporter_code_review_001.md  02361eae6e7b4242b5e1edbc84b991209e129bc800cf2a451f135ae2ef3639d9
exporter_code_closure_001.md 619bf48d58be42fbfd1e361f0e26e152b658533402cf710458fa451611af2d9e
```

Source diff against the retained 4263b1c6 freezer: named pending creation,
closed-path link and unlink are removed. Lines 108-121 create and authenticate
an O_TMPFILE RDWR FD, nlink 0, and keep it open through final member validation.
Lines 123-133 link the owned /proc/self/fd entry using linkat semantics, then
bind the named target to the same FD, require nlink 1, and reread exact raw
bytes. Root binding and fsync precede success output. No pathname fallback or
pending cleanup exists. MANIFEST_ONLY remains explicit; this is not a durable
whole-byte archive or experiment admission.

The original residual script asserted the old pending pathname existed because
4263 created it. With no named staging this assertion is no longer a valid
fixture condition. The probes below retain the original second-inventory
replacement timing and bytes in two legal ways: create the old basename there,
or start with an unrelated foreign file and rename/replace it at that same
boundary. Correct publication and foreign-file preservation, rather than
mandatory refusal, are the expected results.

## Executable Synthetic Harness

```python narrow_harness
import os,sys,json,hashlib,tempfile,importlib.util,traceback,io,contextlib,errno,stat
from pathlib import Path
PAPER=Path('/home/wwww/.codex/worktrees/trace-simplification/hzl/paper_evolution/scope_v4_20261103')
RESIDENT=Path('/tmp/exporter-f2-closure-DWaiXY')
EXPECTED={'freeze_snapshot.py':'49a427ad2b55a203749d5a30a98a8b27e23d1524aabcb42faf61007519bdd059',
 'test_export_scope_v4.py':'6d6f3ef1ba7835ff9bb23449bfa0e5fdda6fd88bd598cc60e7e09cf82f664aea',
 'export_scope_v4.py':'f6bad26234d37ee77f8750c7e57f46d40c5d7237d75fd9a3f576051455ba56d2'}
PRIOR={'exporter_code_review_001.md':'02361eae6e7b4242b5e1edbc84b991209e129bc800cf2a451f135ae2ef3639d9',
 'exporter_code_closure_001.md':'619bf48d58be42fbfd1e361f0e26e152b658533402cf710458fa451611af2d9e'}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes():return {name:digest(PAPER/'tools'/name) for name in EXPECTED}
assert hashes()==EXPECTED
for name,sha in EXPECTED.items():assert digest(RESIDENT/name)==sha
for name,sha in PRIOR.items():assert digest(PAPER/'reviews'/name)==sha
assert sys.dont_write_bytecode and os.environ['CUDA_VISIBLE_DEVICES']==''
def audit(event,args):
 if event=='subprocess.Popen':assert Path(args[0]).name in ('fc-list','fc-match','lscpu'),'No service/GPU commands'
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).resolve()
  for root in ('/home/wwww/hzl/results','/home/wwww/hzl/data','/home/wwww/.cache/clip'):
   assert not p.is_relative_to(root),'No actual experiment or payload read'
sys.addaudithook(audit)
fonts=tempfile.TemporaryDirectory(prefix='f2-closure-fonts-');os.environ['MPLCONFIGDIR']=fonts.name
import cv2,torch,pytest
cv2.setNumThreads(1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
def denied(*a,**k):raise AssertionError('No model/GPU/assets')
torch.load=denied;torch.jit.load=denied;torch.cuda.init=denied
spec=importlib.util.spec_from_file_location('independent_anonymous_freezer',RESIDENT/'freeze_snapshot.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
RESULTS=[]
def check(name,fn):
 try:item=dict(name=name,passed=True,detail=fn())
 except BaseException as error:item=dict(name=name,passed=False,error=repr(error),traceback=traceback.format_exc(limit=8))
 RESULTS.append(item);print(json.dumps(item),flush=True)
def fixture(root):
 for directory in ('tools','generated','figures','build'):(root/directory).mkdir()
 (root/'tools/tool.py').write_bytes(b'original source');(root/'empty.tex').write_bytes(b'')
 for name in ('main.pdf','supplement.pdf','render_qa.json'):(root/'build'/name).write_bytes(b'fixture only')
def probe(kind):
 with tempfile.TemporaryDirectory(prefix='f2-anonymous-') as tmp,pytest.MonkeyPatch.context() as mp:
  root=Path(tmp);fixture(root);mp.setattr(F,'ROOT',root);mp.setattr(sys,'argv',['freeze_snapshot.py','--label','independent_probe'])
  target=root/'FREEZE_independent_probe.json';pending=root/'.FREEZE_independent_probe.pending'
  original_inventory=F.inventory;original_open=F.os.open;original_link=F.os.link;original_read=F.read_member
  original_unlink=F.os.unlink;original_fsync=F.os.fsync;original_write=F.os.write
  stages=[];links=[];unlinks=[];calls=[];changes=[];foreign=b'{"replacement":true}\n';output=io.StringIO()
  if kind=='old_pending_replace':pending.write_bytes(b'foreign preexisting pending')
  def opened(path,flags,*a,**k):
   if flags & os.O_TMPFILE == os.O_TMPFILE:
    if kind=='unsupported_tmpfile':raise OSError(errno.EOPNOTSUPP,'synthetic unsupported anonymous staging')
    fd=original_open(path,flags,*a,**k);stages.append(fd);assert os.fstat(fd).st_nlink==0;return fd
   return original_open(path,flags,*a,**k)
  mp.setattr(F.os,'open',opened)
  def unlink(*a,**k):unlinks.append(str(a[0]));return original_unlink(*a,**k)
  mp.setattr(F.os,'unlink',unlink)
  def inventory():
   value=original_inventory();calls.append(1)
   if len(calls)==2:
    if kind in ('old_pending_create','old_pending_replace'):
     if pending.exists():pending.rename(root/'.held_original_pending')
     pending.write_bytes(foreign)
    elif kind=='new_member':(root/'generated/late.json').write_bytes(b'{}');return original_inventory()
    elif kind=='anonymous_tamper':os.pwrite(stages[-1],b'X',0)
   return value
  mp.setattr(F,'inventory',inventory)
  def read_member(rootfd,rootid,name,**kwargs):
   value=original_read(rootfd,rootid,name,**kwargs)
   if not kwargs and name=='tools/tool.py' and not changes:
    if kind=='earlier_member':(root/'build/main.pdf').write_bytes(b'changed earlier member')
    elif kind=='parent':
     (root/'tools').rename(root/'held_tools');(root/'tools').mkdir();(root/'tools/tool.py').write_bytes(b'original source')
    elif kind=='root':
     root.rename(root.with_name(root.name+'-held'));root.mkdir()
    changes.append(1)
   return value
  mp.setattr(F,'read_member',read_member)
  def link(source,destination,**kwargs):
   assert source=='/proc/self/fd/'+str(stages[-1]) and kwargs['follow_symlinks'] is True
   assert os.fstat(stages[-1]).st_nlink==0 and isinstance(kwargs['dst_dir_fd'],int)
   stage_bytes=os.pread(stages[-1],os.fstat(stages[-1]).st_size+1,0)
   assert json.loads(stage_bytes)['label']=='independent_probe'
   links.append(hashlib.sha256(stage_bytes).hexdigest())
   if kind=='unsupported_link':raise OSError(errno.EOPNOTSUPP,'synthetic unsupported linkat')
   if kind=='foreign_target':target.write_bytes(b'foreign target')
   if kind=='foreign_symlink':target.symlink_to(root/'empty.tex')
   result=original_link(source,destination,**kwargs)
   if kind in ('published_samebytes_swap','published_changed_swap'):
    replacement=root/'replacement';replacement.write_bytes(stage_bytes if kind=='published_samebytes_swap' else foreign);replacement.replace(target)
   return result
  mp.setattr(F.os,'link',link)
  if kind=='member_cap':mp.setattr(F,'MAX_MEMBER',8)
  if kind=='total_cap':mp.setattr(F,'MAX_TOTAL',20)
  if kind=='short_write':
   def short_write(fd,data):return original_write(fd,data[:-1])
   mp.setattr(F.os,'write',short_write)
  if kind=='stage_fsync':
   def fsync(fd):
    if stages and fd==stages[-1]:raise OSError(errno.EIO,'synthetic staging fsync failure')
    return original_fsync(fd)
   mp.setattr(F.os,'fsync',fsync)
  error=None
  try:
   with contextlib.redirect_stdout(output):F.main()
  except (ValueError,OSError) as exc:error=repr(exc)
  assert not unlinks,'Freezer must not consume any named foreign pending or target'
  for fd in stages:
   try:os.fstat(fd)
   except OSError as exc:assert exc.errno==errno.EBADF
   else:raise AssertionError('Anonymous staging FD leaked')
  success=kind in ('normal','old_pending_create','old_pending_replace')
  if success:
   assert error is None and len(links)==1
   data=target.read_bytes();record=json.loads(data);sha=hashlib.sha256(data).hexdigest()
   assert sha==links[0]==output.getvalue().splitlines()[-1]
   assert record['label']=='independent_probe' and record['archive_status'].startswith('MANIFEST_ONLY')
   for name,pin in record['members'].items():assert pin==dict(bytes=(root/name).stat().st_size,sha256=digest(root/name))
   assert target.stat().st_nlink==1
   if kind!='normal':assert pending.read_bytes()==foreign
   if kind=='old_pending_replace':assert (root/'.held_original_pending').read_bytes()==b'foreign preexisting pending'
   if kind=='normal':
    assert not pending.exists()
    try:F.main()
    except ValueError:pass
    else:raise AssertionError('Existing final label overwritten')
    assert target.read_bytes()==data
   return dict(published_owned_manifest=True,published_sha=sha,printed_sha=sha,foreign_pending_preserved=kind!='normal',unlink_calls=0,anonymous_fd_closed=True)
  assert error is not None and not output.getvalue(), 'Fault must not print a successful freeze result'
  if kind=='foreign_target':assert target.read_bytes()==b'foreign target'
  elif kind=='foreign_symlink':assert target.is_symlink() and (root/'empty.tex').read_bytes()==b''
  elif kind.startswith('published_'):assert target.exists()
  else:assert not target.exists()
  return dict(error=error,success_output=False,target_present=target.exists(),unlink_calls=0,anonymous_fd_closed=True)
for kind in ('normal','old_pending_create','old_pending_replace','foreign_target','foreign_symlink',
 'unsupported_tmpfile','unsupported_link','anonymous_tamper','published_samebytes_swap','published_changed_swap',
 'parent','root','new_member','earlier_member','member_cap','total_cap','short_write','stage_fsync'):
 check('F2/'+kind,lambda kind=kind:probe(kind))
with tempfile.TemporaryDirectory(prefix='f2-regression-') as tmp:
 code=pytest.main(['-q','-c','/dev/null','--rootdir',str(RESIDENT),'--import-mode=importlib','-p','no:cacheprovider','--basetemp',str(Path(tmp)/'pytest'),str(RESIDENT/'test_export_scope_v4.py')])
print(json.dumps(dict(independent_controls=len(RESULTS),passed=sum(x['passed'] for x in RESULTS),pytest_exit=int(code),final_hashes=hashes(),prior_report_hashes={name:digest(PAPER/'reviews'/name) for name in PRIOR},CUDA_initialized=torch.cuda.is_initialized(),threads=[torch.get_num_threads(),torch.get_num_interop_threads(),cv2.getNumThreads()])),flush=True)
assert all(x['passed'] for x in RESULTS) and code==0 and hashes()==EXPECTED
assert all(digest(PAPER/'reviews'/name)==sha for name,sha in PRIOR.items()) and not torch.cuda.is_initialized()
```

Replay from the worktree:

```bash
awk '/^```python narrow_harness$/{active=1;next} active && /^```$/{exit} active{print}' paper_evolution/scope_v4_20261103/reviews/exporter_code_closure_002.md | env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/wwww/miniconda3/envs/mlmp/bin/python -B -
```

Disposition awaits actual execution below. No anticipated outcome is a result.

## Actual Results

The exact replay command above completed with exit 0. **18/18 independent
synthetic controls passed; 48 regression tests passed in 1.97s** (pytest exit
0). These are this reviewer's actual observations, distinct from main's
reported 48-test timing. No harness/setup failure occurred in this run.

Normal freeze and both old-pending interference cases each published the same
correct owned-manifest SHA and printed that exact SHA:

```text
21e155cdd106c212cf8c7a0730dbcc8ded9137b97e6a37de32dbb06fce2c9d88
```

Both old-pending variants preserved the foreign replacement JSON byte-for-byte.
The rename/replace variant additionally preserved the initial foreign pending
bytes in `.held_original_pending`. Every independent probe observed **zero
unlink calls by the freezer**. Every created anonymous staging FD was closed
on return or error. At the actual publication call the source was exactly
`/proc/self/fd/<owned_fd>`, the FD had nlink 0, the destination was relative to
the held root directory FD, and follow_symlinks was True. After success the
published manifest was a single-link regular file with correctly hashed members.

| Independent control | Actual observed outcome |
| --- | --- |
| Normal legal manifest, empty member, same-label retry | Correct bytes and printed hash; no pending file; retry refused without overwrite |
| Create old pending basename after second inventory | Correct owned manifest published; foreign pending preserved |
| Rename/replace preexisting foreign old pending at original repro boundary | Correct owned manifest published; both foreign files preserved |
| Race foreign regular final target | FileExistsError; foreign target preserved; no success output |
| Race foreign symlink final target | FileExistsError; symlink and referenced file preserved; no success output |
| Unsupported O_TMPFILE | OSError before publication; no named staging fallback |
| Unsupported descriptor linkat | OSError before publication; no named staging fallback |
| Mutate anonymous FD during final inventory | Rejected `Freeze staging FD drift`; no target |
| Replace published target with identical bytes/new inode before target rebind | Rejected `Published manifest is not the owned FD`; no success output or deletion |
| Replace published target with different bytes/new inode before target rebind | Same rejection; no success output or deletion |
| Replace parent directory after initial member read | Rejected original member identity drift; no target |
| Replace root after initial member read | FileNotFoundError while revalidating; no target or success output |
| Add member at final inventory | Rejected membership change; no target |
| Change already-read earlier member | Rejected final-pass identity drift; no target |
| Per-member size bound | Rejected before publication |
| Aggregate size bound | Rejected before publication |
| Short staging write | Rejected before publication; FD closed |
| Staging fsync error | OSError before publication; FD closed |

For the two deliberately substituted post-link targets, a named file exists
after rejection and is not deleted. The independent assertion is **no successful
return/output and no consumption of the foreign file**, not automatic rollback
of externally replaced paths. This is distinct from the resolved residual,
which previously returned success and printed a hash for different bytes.

Final harness output confirmed current source pins exactly matched start pins,
both prior report hashes remained unchanged, CUDA initialized was false, and
torch intra/inter-op and cv2 threads were 1/1/1. Bytecode, pytest cache and
plugin autoload were disabled. Actual O_TMPFILE plus owned-FD link publication
succeeded on ordinary `/tmp` fixtures; no actual paper-root freeze was performed
to test or claim execution on its filesystem.

The retained first-repair whole files were also separately hashed without
modification and matched their prior pins:

```text
reviews/exporter_F2_first_repair_original_v002/freeze_snapshot.py
4263b1c63ebc19b641baa87f72d8aca2599537a5079fdc44bb9f24dae955ad1f
reviews/exporter_F2_first_repair_original_v002/test_export_scope_v4.py
73351d8221e0f9d7eb83b1e6876f383773cd7095dfd9df27ec3fa57632ff9b2a
```

## Final Disposition

**Residual F2 CLOSED on freezer 49a427ad... / tests 6d6f3ef1... . Narrow
review COMPLETE; no unresolved finding in this repair scope.**

The original unsafe closed-path staging publication no longer exists. The
kernel links the still-owned anonymous FD rather than any mutable pending
basename, and successful return requires exact published FD/path/content
binding. Original interference is now unrelated and preserved, not silently
consumed. No source/test edits were made by this reviewer. F1's prior closure
is retained without a new full review; running the existing full small test
module as regression is not a repeat scientific/provenance audit.

This conclusion is exact-source CODE closure only. It does not establish an
immutable whole-manuscript snapshot against arbitrary concurrent edits after
checks, validate a future real freeze/archive, certify manuscript science or
visual layout, or grant publication/experiment/GT permission. The manifest-only
limitation remains accurate and the separately required verified whole-byte
archive remains unperformed here. All original evidence and the prior failed
repair remain preserved in their existing reports.
