# SAME Reviewer Exporter F1/F2 Narrow Repair Review

Original scope completed with two P2 findings; original raw
exporter_code_review_001.md SHA02361eae6e7b4242b5e1edbc84b991209e129bc800cf2a451f135ae2ef3639d9
is unchanged and not overwritten. This is only the requested repair delta and
affected synthetic integration, not old scientific/full-manuscript review,
publication, artifact/GT permission, bootstrap/service/GPU or metric rerun.

Initial repaired exact hashes observed:

```text
export_scope_v4.py f6bad26234d37ee77f8750c7e57f46d40c5d7237d75fd9a3f576051455ba56d2
test_export_scope_v4.py 73351d8221e0f9d7eb83b1e6876f383773cd7095dfd9df27ec3fa57632ff9b2a
freeze_snapshot.py 4263b1c63ebc19b641baa87f72d8aca2599537a5079fdc44bb9f24dae955ad1f
build.py a75aeb3bf7074a8c2a29fb8f6f7560095d4ba833aa9b2185b0bd75b869bd30a7
render_qa.py 6b109c4d29079197af5f30f0fff8be065ac70f229d26dfa4a93dd1b5c2edddeb
draw_pipeline.py ccd0edcc5c1e4a94bf374c060f172e18276e17c1c827ecdf0ee680cd54a3822b
```

Three repaired files copied unedited to /tmp/paper-export-closure-sCeROC/
before probing. Main's46-test result is attributed, not an independent count.
Current freezer explicitly MANIFEST_ONLY; no durable byte archive or actual
manuscript freeze is performed or claimed in this review.

## Protected Setup And Original Counterexamples

```python setup
import os,sys,json,hashlib,tempfile,importlib.util,traceback,copy,io,contextlib
from pathlib import Path
PAPER=Path('/home/wwww/.codex/worktrees/trace-simplification/hzl/paper_evolution/scope_v4_20261103')
RESIDENT=Path('/tmp/paper-export-closure-sCeROC')
EXPECTED={'export_scope_v4.py':'f6bad26234d37ee77f8750c7e57f46d40c5d7237d75fd9a3f576051455ba56d2',
 'test_export_scope_v4.py':'73351d8221e0f9d7eb83b1e6876f383773cd7095dfd9df27ec3fa57632ff9b2a',
 'freeze_snapshot.py':'4263b1c63ebc19b641baa87f72d8aca2599537a5079fdc44bb9f24dae955ad1f',
 'build.py':'a75aeb3bf7074a8c2a29fb8f6f7560095d4ba833aa9b2185b0bd75b869bd30a7',
 'render_qa.py':'6b109c4d29079197af5f30f0fff8be065ac70f229d26dfa4a93dd1b5c2edddeb',
 'draw_pipeline.py':'ccd0edcc5c1e4a94bf374c060f172e18276e17c1c827ecdf0ee680cd54a3822b'}
def hashes():return {n:hashlib.sha256((PAPER/'tools'/n).read_bytes()).hexdigest() for n in EXPECTED}
snapshot=hashes();assert snapshot==EXPECTED
assert hashlib.sha256((PAPER/'reviews/exporter_code_review_001.md').read_bytes()).hexdigest()=='02361eae6e7b4242b5e1edbc84b991209e129bc800cf2a451f135ae2ef3639d9'
for name in ('export_scope_v4.py','test_export_scope_v4.py','freeze_snapshot.py'):assert hashlib.sha256((RESIDENT/name).read_bytes()).hexdigest()==EXPECTED[name]
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and sys.dont_write_bytecode
def audit(event,args):
 if event=='subprocess.Popen':assert Path(args[0]).name in ('fc-list','fc-match','lscpu'),'No services/status/GPU commands'
 if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).resolve()
 for root in ('/home/wwww/hzl/results','/home/wwww/hzl/data','/home/wwww/.cache/clip'):assert not p.is_relative_to(root),'No actual experiment/payload/model/bootstrap'
sys.addaudithook(audit)
scratch=tempfile.TemporaryDirectory(prefix='paper-closure-fonts-');os.environ['MPLCONFIGDIR']=scratch.name
import torch,cv2,pytest
cv2.setNumThreads(1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
def denied(*a,**k):raise AssertionError('No model/GPU/assets')
torch.load=denied;torch.jit.load=denied;torch.cuda.init=denied
def module(name,file):
 spec=importlib.util.spec_from_file_location(name,RESIDENT/file);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
E=module('independent_repaired_export','export_scope_v4.py');F=module('independent_repaired_freezer','freeze_snapshot.py')
RESULTS=[]
def check(name,fn):
 try:item=dict(name=name,passed=True,detail=fn())
 except BaseException as error:item=dict(name=name,passed=False,error=repr(error),traceback=traceback.format_exc(limit=7))
 RESULTS.append(item);print(json.dumps(item),flush=True)
def rejected(fn):
 try:fn()
 except (ValueError,OSError) as error:return repr(error)
 raise AssertionError('Expected rejection')
def freeze_fixture(root):
 for directory in ('tools','generated','figures','build'):(root/directory).mkdir()
 (root/'tools/tool.py').write_bytes(b'original source')
 for name in ('main.pdf','supplement.pdf','render_qa.json'):(root/'build'/name).write_bytes(b'fixture only')
print(json.dumps(dict(initial_hashes=snapshot,CUDA_initialized=torch.cuda.is_initialized(),threads=[torch.get_num_threads(),torch.get_num_interop_threads(),cv2.getNumThreads()])),flush=True)
```

```python original_counterexamples
from pathlib import Path
raw=Path('paper_evolution/scope_v4_20261103/reviews/exporter_code_closure_001.md').read_text()
exec(compile(raw.split('```python setup\n',1)[1].split('\n```',1)[0],'paper-closure-setup','exec'))
def reader_repro():
 with tempfile.TemporaryDirectory(prefix='export-reader-race-') as tmp,pytest.MonkeyPatch.context() as mp:
  root=Path(tmp);folder=root/'metadata';folder.mkdir();path=folder/'summary.json';path.write_bytes(b'{"value":1}')
  expected=hashlib.sha256(path.read_bytes()).hexdigest();loads=E.json.loads;swaps=[]
  def parse(*a,**k):
   result=loads(*a,**k)
   if not swaps:
    folder.rename(root/'held');folder.mkdir();path.write_bytes(b'{"value":2}');swaps.append(True)
   return result
  mp.setattr(E.json,'loads',parse);reader=E.Reader(root,{'metadata/summary.json'})
  error=rejected(lambda:reader.read('metadata/summary.json',expected))
  assert swaps and reader.provenance=={}
  return dict(error=error,provenance_records=0)
def freezer_repro():
 with tempfile.TemporaryDirectory(prefix='paper-freeze-race-') as tmp,pytest.MonkeyPatch.context() as mp:
  root=Path(tmp);freeze_fixture(root);target=root/'tools/tool.py'
  mp.setattr(F,'ROOT',root);mp.setattr(sys,'argv',['freeze_snapshot.py','--label','independent_probe'])
  original_read=F.os.read;swaps=[]
  def read(fd,count):
   content=original_read(fd,count)
   if content and not swaps and os.readlink('/proc/self/fd/'+str(fd))==str(target):
    (root/'tools').rename(root/'held_tools');(root/'tools').mkdir();target.write_bytes(b'replacement source');swaps.append(True)
   return content
  mp.setattr(F.os,'read',read);error=rejected(F.main)
  assert swaps and not (root/'FREEZE_independent_probe.json').exists()
  return dict(error=error,published_manifest=False)
check('F1/original_counterexample',reader_repro);check('F2/original_counterexample',freezer_repro)
print(json.dumps(dict(total=len(RESULTS),passed=sum(r['passed'] for r in RESULTS),final_disk_hashes=hashes())),flush=True)
assert all(r['passed'] for r in RESULTS) and hashes()==snapshot
```

## New Pending Publication Boundary Probe

```python pending_repro
from pathlib import Path
raw=Path('paper_evolution/scope_v4_20261103/reviews/exporter_code_closure_001.md').read_text()
exec(compile(raw.split('```python setup\n',1)[1].split('\n```',1)[0],'paper-pending-setup','exec'))
with tempfile.TemporaryDirectory(prefix='paper-freeze-pending-') as tmp,pytest.MonkeyPatch.context() as mp:
 root=Path(tmp);freeze_fixture(root);mp.setattr(F,'ROOT',root);mp.setattr(sys,'argv',['freeze_snapshot.py','--label','independent_probe'])
 pending=root/'.FREEZE_independent_probe.pending';published=root/'FREEZE_independent_probe.json'
 original_inventory=F.inventory;calls=[]
 def inventory():
  value=original_inventory();calls.append(1)
  if len(calls)==2:
   assert pending.exists();pending.rename(root/'.held_original_pending');pending.write_bytes(b'{"replacement":true}\n')
  return value
 mp.setattr(F,'inventory',inventory);output=io.StringIO()
 with contextlib.redirect_stdout(output):F.main()
 actual=hashlib.sha256(published.read_bytes()).hexdigest();printed=output.getvalue().splitlines()[-1]
 print(json.dumps(dict(repro='pending_replaced_before_publication',published=True,published_bytes=published.read_text(),published_sha=actual,printed_sha=printed,pending_removed=not pending.exists())),flush=True)
 assert actual!=printed and published.read_bytes()==b'{"replacement":true}\n' and not pending.exists()
print(json.dumps(dict(final_disk_hashes=hashes(),CUDA_initialized=torch.cuda.is_initialized(),threads=[torch.get_num_threads(),torch.get_num_interop_threads(),cv2.getNumThreads()])),flush=True)
assert hashes()==snapshot and not torch.cuda.is_initialized()
```

```sh
LABEL=original_counterexamples
awk -v label="$LABEL" '$0 == "```python " label {active=1;next} active && $0 == "```" {exit} active {print}' paper_evolution/scope_v4_20261103/reviews/exporter_code_closure_001.md | env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/wwww/miniconda3/envs/mlmp/bin/python -B -
```

Disposition pending actual narrow probes/regression.

## Actual Original Repro Results And Confirmed Residual F2

Original counterexamples both rejected on the new exact bytes:
F1 ValueError('Metadata directory changed during parse'),0 provenance records;
F2 ValueError('Manuscript parent directory changed while freezing'),no published
manifest. Source hashes remained the new supplied pins.

Residual F2 P2 at freeze_snapshot.py:108-121: staging FD/path is checked only
inside the write context before close. The final member pass can take place
after its pathname has been replaced; os.link publishes that unbound pathname
without a new staging identity/content check, then removes it. Actual ordinary
synthetic reproduction output:

```json
{"repro":"pending_replaced_before_publication","published":true,"published_bytes":"{\"replacement\":true}\n","published_sha":"11ac1e1db6ea7dd24ef55c6dd5d83d687eda72205e21061b841ccbc7ffddf4c4","printed_sha":"ca792898ae0723fd5424c0411565b0b2d5d3f0ea77faa3d34e50542a9934f4e6","pending_removed":true}
```

This is not the old parent-swap again: new pending publication itself accepts
bytes that are not its computed manifest. No actual paper freeze/source was
modified. Pending original was renamed to .held_original_pending and replaced
with an ordinary generated JSON file at the second inventory call; membership
and every manuscript member remain unchanged. The freezer returns successfully.
Whole repaired bytes remain available at /tmp/paper-export-closure-sCeROC/.
F1 closure supported; F2 remains OPEN pending this publication-boundary repair.

## Remaining Narrow Controls

```python controls_and_tests
from pathlib import Path
raw=Path('paper_evolution/scope_v4_20261103/reviews/exporter_code_closure_001.md').read_text()
exec(compile(raw.split('```python setup\n',1)[1].split('\n```',1)[0],'paper-control-setup','exec'))
def reader_control(variant):
 with tempfile.TemporaryDirectory(prefix='reader-control-') as tmp,pytest.MonkeyPatch.context() as mp:
  parent=Path(tmp);root=parent/'root';root.mkdir();folder=root/'metadata';folder.mkdir();path=folder/'summary.json';path.write_bytes(b'{"value":1}')
  digest=hashlib.sha256(path.read_bytes()).hexdigest();reader=E.Reader(root,{'metadata/summary.json'})
  if variant=='legal_hardlink':
   (folder/'retained-link.json').hardlink_to(path);assert reader.read('metadata/summary.json',digest)=={'value':1}
   assert reader.provenance['metadata/summary.json']['identity'][5]==2
   return 'legal exact historical hardlink accepted'
  original=E.json.loads;swapped=[]
  def loads(*a,**k):
   result=original(*a,**k)
   if not swapped:
    if variant=='samebytes_parent':folder.rename(root/'held');folder.mkdir();path.write_bytes(b'{"value":1}')
    else:root.rename(parent/'held');folder.mkdir(parents=True);path.write_bytes(b'{"value":1}')
    swapped.append(True)
   return result
  mp.setattr(E.json,'loads',loads);error=rejected(lambda:reader.read('metadata/summary.json',digest))
  assert swapped and not reader.provenance;return dict(error=error,provenance_records=0)
for variant in ('legal_hardlink','samebytes_parent','root_replacement'):check('F1/'+variant,lambda v=variant:reader_control(v))
def freeze_control(variant):
 with tempfile.TemporaryDirectory(prefix='freeze-control-') as tmp,pytest.MonkeyPatch.context() as mp:
  root=Path(tmp);freeze_fixture(root);(root/'empty.tex').write_bytes(b'')
  mp.setattr(F,'ROOT',root);mp.setattr(sys,'argv',['freeze_snapshot.py','--label','probe'])
  target=root/'FREEZE_probe.json';pending=root/'.FREEZE_probe.pending'
  if variant=='normal':
   F.main();raw=target.read_bytes();manifest=json.loads(raw)
   assert manifest['archive_status']=='MANIFEST_ONLY_REQUIRE_VERIFIED_WHOLE_BYTE_ARCHIVE_BEFORE_EDITS'
   for name,record in manifest['members'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==record['sha256']
   assert not pending.exists();rejected(F.main);assert target.read_bytes()==raw
   return dict(legal_manifest=True,empty_member=True,label_not_overwritten=True,archive_claim=False)
  if variant=='foreign_pending':pending.write_bytes(b'foreign pending')
  elif variant=='foreign_target_race':
   original=F.os.link
   def link(*a,**k):
    target.write_bytes(b'foreign target');return original(*a,**k)
   mp.setattr(F.os,'link',link)
  elif variant=='new_member':
   original=F.inventory;calls=[]
   def inventory():
    calls.append(1)
    if len(calls)==2:(root/'generated/late.json').write_bytes(b'{}')
    return original()
   mp.setattr(F,'inventory',inventory)
  elif variant=='earlier_member':
   original=F.read_member;changed=[]
   def read(rootfd,rootid,name,**kwargs):
    value=original(rootfd,rootid,name,**kwargs)
    if name=='tools/tool.py' and not kwargs and not changed:
     (root/'build/main.pdf').write_bytes(b'changed earlier member');changed.append(True)
    return value
   mp.setattr(F,'read_member',read)
  elif variant=='member_cap':mp.setattr(F,'MAX_MEMBER',8)
  elif variant=='total_cap':mp.setattr(F,'MAX_TOTAL',20)
  error=rejected(F.main)
  if variant=='foreign_target_race':assert target.read_bytes()==b'foreign target' and pending.exists()
  else:assert not target.exists()
  if variant=='foreign_pending':assert pending.read_bytes()==b'foreign pending'
  if variant in ('new_member','earlier_member'):assert pending.exists()
  return dict(error=error,no_successful_publication=True,foreign_paths_preserved=True)
for variant in ('normal','foreign_pending','foreign_target_race','new_member','earlier_member','member_cap','total_cap'):
 check('F2/'+variant,lambda v=variant:freeze_control(v))
with tempfile.TemporaryDirectory(prefix='export-repaired-regression-') as tmp:
 code=pytest.main(['-q','-c','/dev/null','--rootdir',str(RESIDENT),'--import-mode=importlib','-p','no:cacheprovider','--basetemp',str(Path(tmp)/'pytest'),str(RESIDENT/'test_export_scope_v4.py')])
print(json.dumps(dict(supplemental=len(RESULTS),passed=sum(r['passed'] for r in RESULTS),pytest_exit=int(code),final_disk_hashes=hashes(),CUDA_initialized=torch.cuda.is_initialized(),threads=[torch.get_num_threads(),torch.get_num_interop_threads(),cv2.getNumThreads()])),flush=True)
assert all(r['passed'] for r in RESULTS) and code==0 and hashes()==snapshot and not torch.cuda.is_initialized()
```

## Final Observed Narrow Results

The `controls_and_tests` block above was executed against the authenticated
resident copies with this command from the worktree:

```bash
awk '/^```python controls_and_tests$/{active=1;next} active && /^```$/{exit} active{print}' paper_evolution/scope_v4_20261103/reviews/exporter_code_closure_001.md | env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/wwww/miniconda3/envs/mlmp/bin/python -B -
```

Actual results: **46 regression tests passed in 1.96s**, pytest exit 0;
**10/10 supplemental controls passed**. These counts are independent of the
main author's reported timings. Supplemental cases and observed outcomes:

| Control | Observed outcome |
| --- | --- |
| F1 legal exact historical hardlink | Accepted correct pinned JSON, nlink 2 retained in provenance |
| F1 same-byte parent substitution during parse | Rejected directory change; zero provenance records |
| F1 root substitution during parse | Rejected root/path change; zero provenance records |
| F2 legal normal freeze and same-label retry | Correct member hashes, empty member accepted, explicit manifest-only status; retry refused without overwriting target |
| F2 foreign pending file | FileExistsError; foreign pending bytes preserved; no target |
| F2 foreign target raced at link | FileExistsError; foreign target bytes and own pending preserved |
| F2 new member before final inventory | Rejected membership change; no target; pending preserved |
| F2 earlier member drift after its initial read | Rejected final-pass identity drift; no target; pending preserved |
| F2 per-member bound | Rejected before successful publication |
| F2 aggregate bound | Rejected before successful publication |

The original two exact counterexamples also rejected, as recorded above.
The separate residual staging-substitution repro returned success and
published the replacement JSON with a SHA differing from the printed SHA.
It remains a confirmed failure, not included among passing controls.

Final source checks inside the completed harness matched all six start hashes:

```text
export_scope_v4.py      f6bad26234d37ee77f8750c7e57f46d40c5d7237d75fd9a3f576051455ba56d2
test_export_scope_v4.py 73351d8221e0f9d7eb83b1e6876f383773cd7095dfd9df27ec3fa57632ff9b2a
freeze_snapshot.py      4263b1c63ebc19b641baa87f72d8aca2599537a5079fdc44bb9f24dae955ad1f
build.py                a75aeb3bf7074a8c2a29fb8f6f7560095d4ba833aa9b2185b0bd75b869bd30a7
render_qa.py            6b109c4d29079197af5f30f0fff8be065ac70f229d26dfa4a93dd1b5c2edddeb
draw_pipeline.py        ccd0edcc5c1e4a94bf374c060f172e18276e17c1c827ecdf0ee680cd54a3822b
```

Observed CUDA initialized: false. Torch intra/inter-op and cv2 thread counts:
1/1/1. Bytecode, pytest cache and plugin autoload disabled. Ordinary synthetic
fixtures and resident source copies only; no actual paper freeze/publication,
scientific recomputation, GPU/service access, or actual GT/RGB/NPY/model reads.

## Disposition

**Narrow review execution COMPLETE; F1 CLOSED; F2 OPEN (P2).**

F1's original postparse provenance boundary and the additional same-byte/root
variants now reject without provenance. F2's original manuscript-parent swap
is fixed, but the replacement staging pathname can still be successfully
published after its verified FD is closed. Therefore these freezer bytes do
not support F2 closure or a verified freeze-publication claim. The explicit
MANIFEST_ONLY statement is accurate and does not itself remedy publication of
bytes different from the manifest that was computed and printed.

All initial failures, replay code, repaired start/end pins and the residual
failure are retained here. Original full-scope raw report remains separate and
unchanged. No original 343-leaf scientific/projection review was repeated and
no implementation/test edits were made by this reviewer. A subsequent repair
requires a new exact-hash narrow publication-boundary check; no approval of a
proposed but untested future implementation is implied.
