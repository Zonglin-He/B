# Scope V4 Exporter And QA Tools Independent CODE Review

## Scope And Initial Pins

Distinct NEW exporter/QA CODE scope. Not a manuscript scientific review or a
repeat of prior scorer/prediction reviews. The prior dispatch capacity error
ran no review and supplies no finding or verdict. No old raw reviews/grades/
targets, payloads/count streams/bootstrap, services/status/GPU, GT/RGB/NPY or
weights are read. Ordinary synthetic metadata/derived paper fixtures only.
Actual later reads, if used, are bounded exact pinned summary metadata and
the explicitly listed seven retained assets only. No estimator recomputation.

Initial observed hashes:

```text
export_scope_v4.py bdfdca15a5bffd95f50199bffbb193c5b51afa2a59eabdd82b275f76de4571a4
test_export_scope_v4.py a19d2c4cc2221b80ad98409af1116242e16b85130ad9a7f6821c48a15deba766
build.py a75aeb3bf7074a8c2a29fb8f6f7560095d4ba833aa9b2185b0bd75b869bd30a7
render_qa.py 6b109c4d29079197af5f30f0fff8be065ac70f229d26dfa4a93dd1b5c2edddeb
freeze_snapshot.py b3122525affa8c87937a300bf55c4221ce55486c8632ef6fe1081061e4bcf4cc
draw_pipeline.py ccd0edcc5c1e4a94bf374c060f172e18276e17c1c827ecdf0ee680cd54a3822b
```

Earlier render_qa hasha458be662a2cef615f1ceacb5401882e404afe30218da3bdbc26c646011033df
was observed before implementation inspection, then superseded by the declared
PDF-digest-folder repair. Only current6b109... is reviewed. All six current
source files were copied unmodified to /tmp/paper-export-independent-GT1GQ7/
before reproduction/main repairs. Main author38 tests/projection/build assertions
are attributed, not independent observations yet. User supplies frozen evidence
eda4079f5c01b41b1b8e3f7538449ba2ddf06272af453956778c8e3df33fb845 and
provenance52926388ae88a8e728010e21bc0b9d5f0bc780b537ffa5dd020d3b790a02cbfc;
these require verification before being claimed current observations.

## CPU Synthetic Setup

```python setup
import os,sys,json,hashlib,tempfile,importlib.util,traceback,copy
from pathlib import Path
PAPER=Path('/home/wwww/.codex/worktrees/trace-simplification/hzl/paper_evolution/scope_v4_20261103')
ORIGINAL=Path('/tmp/paper-export-independent-GT1GQ7')
NAMES=('export_scope_v4.py','test_export_scope_v4.py','build.py','render_qa.py','freeze_snapshot.py','draw_pipeline.py')
EXPECTED=('bdfdca15a5bffd95f50199bffbb193c5b51afa2a59eabdd82b275f76de4571a4','a19d2c4cc2221b80ad98409af1116242e16b85130ad9a7f6821c48a15deba766','a75aeb3bf7074a8c2a29fb8f6f7560095d4ba833aa9b2185b0bd75b869bd30a7','6b109c4d29079197af5f30f0fff8be065ac70f229d26dfa4a93dd1b5c2edddeb','b3122525affa8c87937a300bf55c4221ce55486c8632ef6fe1081061e4bcf4cc','ccd0edcc5c1e4a94bf374c060f172e18276e17c1c827ecdf0ee680cd54a3822b')
def hashes():return {n:hashlib.sha256((PAPER/'tools'/n).read_bytes()).hexdigest() for n in NAMES}
snapshot=hashes();assert list(snapshot.values())==list(EXPECTED)
for name,digest in zip(NAMES,EXPECTED):assert hashlib.sha256((ORIGINAL/name).read_bytes()).hexdigest()==digest
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and sys.dont_write_bytecode
def audit(event,args):
 if event=='subprocess.Popen':
  assert Path(args[0]).name in ('fc-list','fc-match','lscpu'),'No services/status/GPU or unrelated commands'
 if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).resolve()
 for root in ('/home/wwww/hzl/results','/home/wwww/hzl/data','/home/wwww/.cache/clip'):
  assert not p.is_relative_to(root),'No actual experiments/payloads/model assets in synthetic harness'
sys.addaudithook(audit)
scratch=tempfile.TemporaryDirectory(prefix='paper-export-review-');os.environ['MPLCONFIGDIR']=scratch.name
import cv2,torch,pytest
cv2.setNumThreads(1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
def denied(*a,**k):raise AssertionError('No model/assets/GPU')
torch.load=denied;torch.jit.load=denied;torch.cuda.init=denied
def module(name,filename):
 spec=importlib.util.spec_from_file_location(name,ORIGINAL/filename);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
E=module('independent_original_export','export_scope_v4.py')
F=module('independent_original_freeze','freeze_snapshot.py')
Q=module('independent_current_renderqa','render_qa.py')
print(json.dumps(dict(initial_hashes=snapshot,threads=[torch.get_num_threads(),torch.get_num_interop_threads(),cv2.getNumThreads()])),flush=True)
```

## Read/Freeze Identity Hypotheses

```python identity_repros
from pathlib import Path
raw=(Path('paper_evolution/scope_v4_20261103/reviews/exporter_code_review_001.md')).read_text()
exec(compile(raw.split('```python setup\n',1)[1].split('\n```',1)[0],'export-review-setup','exec'))
with tempfile.TemporaryDirectory(prefix='export-reader-race-') as tmp,pytest.MonkeyPatch.context() as mp:
 root=Path(tmp);folder=root/'metadata';folder.mkdir();path=folder/'summary.json';path.write_bytes(b'{"value":1}')
 expected=hashlib.sha256(path.read_bytes()).hexdigest();loads=E.json.loads;swaps=[]
 def parse(*a,**k):
  result=loads(*a,**k)
  if not swaps:
   folder.rename(root/'held');folder.mkdir();path.write_bytes(b'{"value":2}');swaps.append(True)
  return result
 mp.setattr(E.json,'loads',parse)
 reader=E.Reader(root,{'metadata/summary.json'});result=reader.read('metadata/summary.json',expected)
 current=hashlib.sha256(path.read_bytes()).hexdigest()
 print(json.dumps(dict(repro='Reader_parent_swap_during_parse',accepted=result,expected_sha=expected,current_path_sha=current,provenance=reader.provenance)),flush=True)
 assert result['value']==1 and current!=expected and swaps
with tempfile.TemporaryDirectory(prefix='paper-freeze-race-') as tmp,pytest.MonkeyPatch.context() as mp:
 root=Path(tmp)
 for directory in ('tools','generated','figures','build'):(root/directory).mkdir()
 target=root/'tools'/'tool.py';target.write_bytes(b'original source')
 for name in ('main.pdf','supplement.pdf','render_qa.json'):(root/'build'/name).write_bytes(b'fixture only')
 mp.setattr(F,'ROOT',root);mp.setattr(sys,'argv',['freeze_snapshot.py','--label','independent_probe'])
 original_read=F.os.read;swaps=[]
 def read(fd,count):
  content=original_read(fd,count)
  if content and not swaps and os.readlink('/proc/self/fd/'+str(fd))==str(target):
   (root/'tools').rename(root/'held_tools');(root/'tools').mkdir();target.write_bytes(b'replacement source');swaps.append(True)
  return content
 mp.setattr(F.os,'read',read);F.main()
 manifest=json.loads((root/'FREEZE_independent_probe.json').read_bytes());pin=manifest['members']['tools/tool.py']
 current=hashlib.sha256(target.read_bytes()).hexdigest()
 print(json.dumps(dict(repro='freeze_parent_swap_during_read',manifest_pin=pin,current_path_sha=current,accepted=True)),flush=True)
 assert swaps and current!=pin['sha256']
print(json.dumps(dict(final_disk_hashes=hashes(),CUDA_initialized=torch.cuda.is_initialized())),flush=True)
assert hashes()==snapshot and not torch.cuda.is_initialized()
```

```sh
LABEL=identity_repros
awk -v label="$LABEL" '$0 == "```python " label {active=1;next} active && $0 == "```" {exit} active {print}' paper_evolution/scope_v4_20261103/reviews/exporter_code_review_001.md | env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/wwww/miniconda3/envs/mlmp/bin/python -B -
```

Disposition pending actual reproductions and remaining review.

## Confirmed Findings On Initial Bytes

F1 P2, export_scope_v4.py:148-150: the reader verifies directory/path binding
before JSON parsing but after parsing checks only the original FD. Replacing
the parent directory during parse preserves that file FD's identity while the
published path resolves to different bytes. Reproduction accepted value1 and
recorded SHA48208f9428d64634bd8e28ff345bf0eab60d53c18fa2fbdb0b9bc1e84df2b5f6,
but the current named path SHA was
49c987621f206f09e5fbe23b516b55a36f838cb14867961f1d84a554d3a35b6b.
This is stale pathname provenance before read returns, NOT acceptance of an
unhashed scientific value: the returned value remains the correctly pinned
old value1. Original hashes stayed unchanged at process completion.

F2 P2, freeze_snapshot.py:27-42 and:46: only FD-before/after fields are checked,
not current pathname/root/directory binding or final full membership before
publication. Parent tools-directory replacement during read accepted a freeze
whose tools/tool.py manifest SHA
1beb6dbff81f8e9fd943092c8be12611c6dc42cca13eb1d61edbc37cfc0540a3
differs from named current file SHA
7e579fb0c17d6cc5a211a0d2162dfba1463b48ddee17fc0aa2796e8e8106ae6a.
It printed a successful FREEZE path/hash rather than rejecting. This is an
inconsistent freeze manifest, not a paper-value/scientific finding. No real
snapshot or experiment bytes were modified by the repro.

Both ordinary synthetic reproductions exited0 with expected acceptance evidence;
observed CUDA uninitialized and threads1/1/1. Initial sources/test remained exact
hashes above. Fixes belong to exporter/QA helpers only; reviewer edits none.
Remaining original-scope review active; F1/F2 unresolved.

## Remaining Original CPU Checks

```python original_tests
from pathlib import Path
raw=Path('paper_evolution/scope_v4_20261103/reviews/exporter_code_review_001.md').read_text()
setup=raw.split('```python setup\n',1)[1].split('\n```',1)[0]
setup=setup.replace('assert list(snapshot.values())==list(EXPECTED)','assert all(snapshot[n]==d for n,d in zip(NAMES,EXPECTED) if n not in ("export_scope_v4.py","test_export_scope_v4.py","freeze_snapshot.py"))')
exec(compile(setup,'paper-original-test-setup','exec'))
with tempfile.TemporaryDirectory(prefix='paper-export-original-tests-') as tmp:
 code=pytest.main(['-q','-c','/dev/null','--rootdir',str(ORIGINAL),'--import-mode=importlib','-p','no:cacheprovider','--basetemp',str(Path(tmp)/'pytest'),str(ORIGINAL/'test_export_scope_v4.py')])
print(json.dumps(dict(original_exporter_sha=hashlib.sha256((ORIGINAL/'export_scope_v4.py').read_bytes()).hexdigest(),original_tests_sha=hashlib.sha256((ORIGINAL/'test_export_scope_v4.py').read_bytes()).hexdigest(),pytest_exit=int(code),current_disk_hashes=hashes(),CUDA_initialized=torch.cuda.is_initialized(),threads=[torch.get_num_threads(),torch.get_num_interop_threads(),cv2.getNumThreads()])),flush=True)
assert code==0 and not torch.cuda.is_initialized()
```

```python pinned_projection
from pathlib import Path
raw=Path('paper_evolution/scope_v4_20261103/reviews/exporter_code_review_001.md').read_text()
setup=raw.split('```python setup\n',1)[1].split('\n```',1)[0]
setup=setup.replace('assert list(snapshot.values())==list(EXPECTED)','assert all(snapshot[n]==d for n,d in zip(NAMES,EXPECTED) if n not in ("export_scope_v4.py","test_export_scope_v4.py","freeze_snapshot.py"))')
ACTUAL_ALLOWED=set();STATIC_ALLOWED=set()
setup=setup.replace("assert not p.is_relative_to(root),'No actual experiments/payloads/model assets in synthetic harness'", "assert not p.is_relative_to(root) or p in ACTUAL_ALLOWED,'No unallowlisted actual experiments/payloads/model assets'")
exec(compile(setup,'paper-pinned-projection-setup','exec'))
for key,spec in E.STUDIES.items():
 suite=spec[0];ACTUAL_ALLOWED.update(E.BASE/n for n in (E.terminal_name(key),suite+'/ANALYSIS_SEALED.json',suite+'/STARTED.json'))
 ACTUAL_ALLOWED.update(E.BASE/(suite+'/'+name) for name in E.SUMMARY_PINS.get(key,{}))
 if key=='M01':ACTUAL_ALLOWED.update(E.BASE/(suite+'/'+d+'/'+n) for d in E.DATASETS for n in ('ANALYSIS_SEALED.json','result.json'))
for suite,_,name,_ in E.CACHED.values():ACTUAL_ALLOWED.update((E.BASE/(suite+'/ANALYSIS_SEALED.json'),E.BASE/(suite+'/'+name)))
STATIC_ALLOWED.update(PAPER.parent/value[0] for value in E.RETAINED_ASSETS.values())
def old_paper_guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  path=Path(os.fsdecode(args[0])).resolve()
  if path.is_relative_to(PAPER.parent) and not path.is_relative_to(PAPER):assert path in STATIC_ALLOWED,'No old raw reviews or other old paper content'
sys.addaudithook(old_paper_guard)
def stable(path,expected,cap=16*1024**2):
 path=Path(path);assert path.resolve(strict=True)==path
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  first=os.fstat(fd);assert 0<first.st_size<=cap
  chunks=[];total=0
  while True:
   block=os.read(fd,min(65536,first.st_size+1-total))
   if not block:break
   total+=len(block);assert total<=first.st_size;chunks.append(block)
  content=b''.join(chunks)
  assert E.identity(os.fstat(fd))==E.identity(first)==E.identity(path.lstat()) and total==first.st_size
  assert hashlib.sha256(content).hexdigest()==expected
  return content
 finally:os.close(fd)
documents,bindings,pins=E.load_evidence();rows=E.normalize(documents);tables=E.tex_tables(rows)
assets=E.retained_assets(PAPER)
evidence_raw=stable(PAPER/'generated/evidence.json','eda4079f5c01b41b1b8e3f7538449ba2ddf06272af453956778c8e3df33fb845')
provenance_raw=stable(PAPER/'generated/source_provenance.json','52926388ae88a8e728010e21bc0b9d5f0bc780b537ffa5dd020d3b790a02cbfc')
evidence=json.loads(evidence_raw);provenance=json.loads(provenance_raw)
assert evidence['tables']==rows and evidence['sealed_summaries']==documents and evidence['completion_bindings']==bindings
assert provenance['source_pins']==pins and provenance['retained_assets']==assets
assert provenance['exporter_sha256']==EXPECTED[0] and provenance['table_rows']=={k:len(v) for k,v in rows.items()}
assert len(pins)==36 and len(assets)==7 and len(provenance['output_sha256'])==31
for name,content in tables.items():assert content==stable(PAPER/name,provenance['output_sha256'][name])
for name,digest in provenance['output_sha256'].items():stable(PAPER/name,digest)
for key in E.CACHED:assert 'cached true-complete seal only' in bindings[key]['completion_basis']
for key in E.STUDIES:assert 'true-success terminal receipt' in bindings[key]['completion_basis']
summary=dict(metadata_files=len(pins),metadata_total_bytes=sum(v['bytes'] for v in pins.values()),static_assets=len(assets),output_hash_members=len(provenance['output_sha256']),
 evidence_bytes=len(evidence_raw),evidence_tables_equal_fresh_projection=True,all_declared_outputs_hash_match=True,
 tex_tables_compared=len(tables),row_counts={k:len(v) for k,v in rows.items()},cached_completion_not_terminal_claim=True,
 actual_services_or_payloads_read=False,estimator_computed=False,current_disk_hashes=hashes(),CUDA_initialized=torch.cuda.is_initialized())
print(json.dumps(summary,sort_keys=True),flush=True)
assert not torch.cuda.is_initialized()
```

First pinned_projection attempt stopped in HARNESS before any metadata parse:
the audit hook allowlisted exact files but omitted Reader's O_DIRECTORY handle
on the results root. Preserve this as setup failure, not artifact/code evidence.
V2 adds only canonical directory ancestors of the already exact file allowlist;
the reader's own path allowlist and nofollow chain remain unchanged. It also
allows the directory handles for the seven explicit old static assets, not
old raw reviews or other historical content.

```python pinned_projection_v2
from pathlib import Path
raw=Path('paper_evolution/scope_v4_20261103/reviews/exporter_code_review_001.md').read_text()
script=raw.split('```python pinned_projection\n',1)[1].split('\n```',1)[0]
needle='documents,bindings,pins=E.load_evidence();'
insert="ACTUAL_ALLOWED.update(p for f in list(ACTUAL_ALLOWED) for p in f.parents if p.is_relative_to(E.BASE))\nSTATIC_ALLOWED.update(p for f in list(STATIC_ALLOWED) for p in f.parents if p.is_relative_to(PAPER.parent))\n"
assert needle in script
exec(compile(script.replace(needle,insert+needle),'paper-pinned-projection-v2','exec'))
```

## QA Helper Synthetic Controls And Diagram Asset

```python qa_helpers
from pathlib import Path
raw=Path('paper_evolution/scope_v4_20261103/reviews/exporter_code_review_001.md').read_text()
setup=raw.split('```python setup\n',1)[1].split('\n```',1)[0]
setup=setup.replace('assert list(snapshot.values())==list(EXPECTED)','assert all(snapshot[n]==d for n,d in zip(NAMES,EXPECTED) if n not in ("export_scope_v4.py","test_export_scope_v4.py","freeze_snapshot.py"))')
setup=setup.replace("('fc-list','fc-match','lscpu')","('fc-list','fc-match','lscpu','pdfinfo','pdffonts','pdftotext','pdftoppm')")
exec(compile(setup,'paper-QA-setup','exec'))
from PIL import Image
B=module('independent_original_build','build.py');D=module('independent_original_diagram','draw_pipeline.py')
RESULTS=[]
def check(name,fn):
 try:item=dict(name=name,passed=True,detail=fn())
 except BaseException as e:item=dict(name=name,passed=False,error=repr(e),traceback=traceback.format_exc(limit=7))
 RESULTS.append(item);print(json.dumps(item),flush=True)
def reject(fn):
 try:fn()
 except (ValueError,RuntimeError,OSError):return
 raise AssertionError('Expected failure')
def build_control(token):
 with tempfile.TemporaryDirectory(prefix='paper-build-control-') as tmp,pytest.MonkeyPatch.context() as mp:
  root=Path(tmp);(root/'tools').mkdir();calls=[]
  mp.setattr(B,'__file__',str(root/'tools'/'build.py'))
  def run(command,**kwargs):
   calls.append(command)
   assert kwargs['check'] is True and kwargs['cwd']==root/'build'
   assert kwargs['env']['TEXINPUTS']=='.:..:'
   if command[0]=='pdflatex':assert '-no-shell-escape' in command
   for name in ('main.log','supplement.log'):(root/'build'/name).write_text(token or 'Clean synthetic final log')
  mp.setattr(B.subprocess,'run',run)
  if token:reject(B.main)
  else:B.main()
  assert len(calls)==11
  return dict(commands=11,rejected=bool(token))
for token in ('','Overfull','Float too large','undefined references','undefined citations','multiply defined','Missing character'):
 check('build/'+(token or 'clean'),lambda t=token:build_control(t))
def qa_control(fault):
 with tempfile.TemporaryDirectory(prefix='paper-render-control-') as tmp,pytest.MonkeyPatch.context() as mp:
  root=Path(tmp);(root/'build').mkdir();pdf=root/'build'/'main.pdf';pdf.write_bytes(b'First synthetic PDF fixture')
  mp.setattr(Q,'ROOT',root);calls=[]
  def command(args):
   calls.append(args[0])
   if args[0]=='pdfinfo':return 'Pages: '+('2' if fault=='page_count' else '1')+'\n'
   if args[0]=='pdffonts':
    return 'name type encoding emb sub uni object ID\n----\nFont '+('Type 3' if fault=='type3' else 'TrueType')+' WinAnsi '+('no' if fault=='nonembedded' else 'yes')+' yes yes 1 0\n'
   if args[0]=='pdftotext':return 'Unresolved ??' if fault=='unresolved' else 'Clean synthetic text'
   if args[0]=='pdftoppm':
    image=Image.new('RGB',(20,20),'white');image.putpixel((10,10),(0,0,0));image.save(str(args[-1])+'-1.png')
    if fault=='PDF_drift':pdf.write_bytes(b'Changed synthetic PDF fixture')
    return ''
   raise AssertionError(args)
  mp.setattr(Q,'command',command)
  if fault:reject(lambda:Q.inspect('main'));return 'refused, no success record returned'
  first=Q.inspect('main');old=root/'build'/'render'/'main'/first['sha256'][:16]
  (old/'page-9.png').write_bytes(b'Old diagnostic remains untouched')
  pdf.write_bytes(b'New synthetic PDF fixture');second=Q.inspect('main')
  assert first['sha256']!=second['sha256'] and (old/'page-9.png').exists()
  assert second['rendered_pages']==1 and second['visual_inspection_status']=='REQUIRES_ACTUAL_IMAGE_REVIEW'
  assert first['contact_sheets'][0]!=second['contact_sheets'][0]
  return dict(new_digest_folder=True,old_diagnostics_preserved=True,visual_pass_not_claimed=True)
for fault in ('','page_count','type3','nonembedded','unresolved','PDF_drift'):
 check('render/'+(fault or 'digest_folders'),lambda f=fault:qa_control(f))
def diagram():
 with tempfile.TemporaryDirectory(prefix='paper-diagram-') as tmp,pytest.MonkeyPatch.context() as mp:
  root=Path(tmp)
  for name in ('tools','figures','build'):(root/name).mkdir()
  mp.setattr(D,'__file__',str(root/'tools'/'draw_pipeline.py'));D.main()
  data=(root/'figures'/'trace_pipeline.pdf').read_bytes();assert data.startswith(b'%PDF-') and len(data)>5000
  (root/'build'/'main.pdf').write_bytes(data);mp.setattr(Q,'ROOT',root)
  result=Q.inspect('main');assert result['pages']==1 and result['all_fonts_embedded'] and not result['type3_fonts']
  page=next((root/'build'/'render'/'main'/result['sha256'][:16]).glob('page-*.png'))
  with Image.open(page) as image:
   import numpy as np
   pixels=np.asarray(image.convert('RGB'));assert int((pixels.min(axis=-1)<200).sum())>1000
  return dict(actual_mathtext_PDF=True,actual_poppler_render=True,embedded_fonts=True,nonblank_pixels=True,visual_status=result['visual_inspection_status'])
check('draw_pipeline/real_scratch_PDF',diagram)
print(json.dumps(dict(total=len(RESULTS),passed=sum(r['passed'] for r in RESULTS),current_disk_hashes=hashes(),CUDA_initialized=torch.cuda.is_initialized(),threads=[torch.get_num_threads(),torch.get_num_interop_threads(),cv2.getNumThreads()])),flush=True)
assert all(r['passed'] for r in RESULTS) and not torch.cuda.is_initialized()
```

## Direct Sealed-Leaf Projection Oracle

```python projection_leaf_oracle
from pathlib import Path
raw=Path('paper_evolution/scope_v4_20261103/reviews/exporter_code_review_001.md').read_text()
script=raw.split('```python pinned_projection\n',1)[1].split('\n```',1)[0]
needle='documents,bindings,pins=E.load_evidence();'
insert="ACTUAL_ALLOWED.update(p for f in list(ACTUAL_ALLOWED) for p in f.parents if p.is_relative_to(E.BASE))\nSTATIC_ALLOWED.update(p for f in list(STATIC_ALLOWED) for p in f.parents if p.is_relative_to(PAPER.parent))\n"
script=script.replace(needle,insert+needle)
oracle="""checked=0
for key,table in (('M02','m02_spatial'),('M03','m03_legacy')):
 for r in rows[table]:
  assert r['metric']==documents[key][r['dataset']][r['kind']][r['name']];checked+=1
for table,kind in (('m04_arms','arms'),('m04_contrasts','contrasts')):
 for r in rows[table]:
  assert r['metric']==documents['M04'][r['dataset']]['studies'][r['study']][kind][r['name']];checked+=1
for table in ('g_clean','g_s5'):
 for r in rows[table]:assert r['record']==documents['G'][r['name']];checked+=1
for r in rows['g03_arms']:
 assert r['metric']==documents['G03']['curves'][r['corruption']][str(r['severity'])]['arms'][r['arm']];checked+=1
for r in rows['g03_contrasts']:
 source=documents['G03']['curves'][r['corruption']][str(r['severity'])]
 assert r['point']==source['C_minus_Confidence_pp'] and r['metric']==source['paired_difference'];checked+=1
for r in rows['v01_grid']:assert r['record']==documents['V01'][r['dataset']]['points'][r['name']];checked+=1
for r in rows['v01_matching']:
 assert r['metric']==documents['V01'][r['dataset']][r['name']];checked+=1
for table,field in (('a01_groups','source_groups'),('a01_subsets','class_subset_IoU')):
 for r in rows[table]:
  assert r['record'] in documents['A01']['datasets'][r['dataset']][field];checked+=1
for r in rows['m05_transitions']:
 if r['kind']=='contrast':assert r['metric']==documents['M05']['datasets'][r['dataset']]['C_minus_Confidence'][r['endpoint']]
 else:
  source=next(v for v in documents['M05']['rows'] if v['dataset']==r['dataset'] and v['arm']==r['name'])
  assert r['metric']=={'value':source[r['endpoint']]}
 checked+=1
for r in rows['m01_risk']:
 source=documents['M01'][r['dataset']]['checkpoints'][r['checkpoint']][r['region']]
 if r['comparison'] in ('C Q5 - Q1','Confidence Q5 - Q1'):
  name='C_Q5_minus_C_Q1' if r['comparison'].startswith('C Q') else 'confidence_Q5_minus_confidence_Q1'
  assert r['metric']==source['score_quintiles']['contrasts'][name][r['endpoint']]
 else:
  name='initial_scores' if r['comparison']=='Score matched' else 'initial_scores_and_GT_quality'
  assert r['metric']==source['matched'][name]['contrasts'][r['endpoint']]
 checked+=1
assert E.label('single_C_vs_legacy_episode')=='C episode - legacy episode'
v01_tex=tables['generated/v01_matching.tex'].decode()
assert all('\\\\texttt{n/a}' in line for line in v01_tex.splitlines() if 'Same-step C - Conf.' in line)
print(json.dumps(dict(direct_sealed_leaf_checks=checked,projection_values_and_nulls_preserved=True,M03_label_checked=True,V01_same_step_ratio_na=True)),flush=True)
"""
assert 'summary=dict(' in script
exec(compile(script.replace('summary=dict(',oracle+'summary=dict('),'paper-direct-leaf-oracle','exec'))
```

## Completed Original Scope: Results And Disposition

**Original CODE review COMPLETE with two confirmed unresolved P2 findings.
Not CLOSED/approved; F1/F2 repair and SAME-reviewer narrow closure required.**
No third concrete blocker found in this bounded exporter/QA scope. No further
original-scope tests are pending. This is not a full-manuscript scientific or
visual review, experiment/artifact/GT permission, publication approval, rating
or whole-project completion.

Actual independent observations on initial exact bytes:

-38 original exporter tests passed in2.20s, using authenticated original
  source/test copies and CPU-only synthetic fixtures. Terminal identity/global/
  STARTED membership, M01 child-seal membership, cached-completion distinction,
  nofollow/bounds/hash-before-parse, nonfinite/duplicate JSON and all-row/null/
  label/ownership cases passed. Passing tests do not cover or negate F1/F2.
-36 actual exact allowlisted metadata files/5692241 bytes were read, no result
  directory traversal or count stream/payload/status/bootstrap access. Seven
  explicit retained static assets matched both original source and new copy;
  no other old paper/review/grade content was opened.
-Actual evidence SHAeda4079f5c01b41b1b8e3f7538449ba2ddf06272af453956778c8e3df33fb845,
 11320457 bytes, and actual provenance SHA
 52926388ae88a8e728010e21bc0b9d5f0bc780b537ffa5dd020d3b790a02cbfc matched
 supplied pins. All31 declared output hashes matched, all21 generated TeX
 tables were byte-equal to a fresh metadata-only projection, every table row
 and full preserved source field matched generated/evidence.json. No estimator,
 matching, bootstrap, model or experiment recomputation occurred.
-343 direct sealed-leaf association checks additionally matched source metric
 objects/points/nulls for M01/M02/M03/M04/G/V01/G03/A01/M05 rows. Five G grouped
 records and eight repeated main-mechanism presentation rows were checked via
 complete fresh-projection/evidence equality and source inspection, not counted
 as343 independent leaf checks. M03 secondary label is C episode - legacy
 episode; V01 same-step ratio is n/a, nearest descriptive ratio remains separate.
 Natural/common-active and cached source/class/candidate-local scopes remain
 separate; original coverage/undefined/class-role fields remain in evidence.
-14/14 QA/helper controls passed:clean11-command build flow and all six flagged
 log-token refusals; digest-specific render folders preserving old diagnostics;
 page count/font embedding/Type3/unresolved-text/end-PDF-drift refusal; no visual
 approval inferred. Actual scratch draw_pipeline mathtext PDF built and rendered
 through Poppler, with embedded non-Type3 fonts and nonblank pixels. All actual
 generated/read diagram/PDF/image fixtures lived under /tmp, not the manuscript.
 Build control-flow cases stubbed compiler commands, not an independent full
 actual manuscript compile. Actual27-page visual QA remains main's separate work.
-F1 and F2 repros both demonstrated original acceptance as specified. No real
 paper source/evidence/experiment was changed. Main states actual freezer has
 not been run and publication has not occurred; those are main-attributed facts.

Row counts independently verified:

```json
{"m01_risk":96,"m02_spatial":14,"m03_legacy":18,"m04_arms":16,"m04_contrasts":24,"main_mechanism":8,"g_clean":4,"g_s5":15,"g_groups":5,"v01_grid":10,"v01_matching":6,"g03_arms":45,"g03_contrasts":15,"a01_groups":32,"a01_subsets":8,"m05_transitions":40}
```

### Required Repair/Repro Contract

F1 P2:at export_scope_v4.py:148-150 repeat the authenticated root/directory/
pathname binding after parsing and before recording successful provenance.
The reproduction fixture is one exact allowlisted metadata/summary.json with
bytes {"value":1}, expected SHA
48208f9428d64634bd8e28ff345bf0eab60d53c18fa2fbdb0b9bc1e84df2b5f6.
Inside json.loads' wrapper after real parsing, rename metadata to held, create
new metadata/summary.json containing {"value":2}. Original reader accepts and
publishes stale path provenance. Repair must refuse; unchanged legal/historical
hardlinked sealed files should still pass their exact hash/stable binding.

F2 P2:freeze_snapshot.py:27-42/:46 must not publish a freeze whose current
named members/parent-directory chain differ from authenticated read members.
The fixture contains tools/tool.py, generated/,figures/ and three required
build files. During os.read of tool.py, rename tools to held_tools and create
replacement tools/tool.py. Original writes FREEZE_independent_probe.json
successfully with a hash unlike the current path. Repair must refuse before
successful publication and preserve any foreign destination/owned partial
failure evidence according to the new flow. Repro paths are all generated
temporary names, not actual project mutations. Exact executable schemas and
all assertions are in identity_repros above.

### Whole-Byte Version Preservation Limit

Current freeze_snapshot.py writes a hash manifest only. It does NOT archive
the manuscript member bytes, so that JSON alone cannot restore a reviewed
version after later working-file edits. This is an explicit capability limit,
not a third reproduced identity defect or an experiment finding. To satisfy
main's newly requested durable whole-version workflow before future manuscript
editing, the repaired flow must preserve the actual authenticated member bytes
(or an equivalent exact recoverable immutable version), not just their hashes;
hash/copy/recovery consistency then belongs to narrow QA integration checks.
No additional scientific experiment is needed or authorized by this requirement.

### Original Byte Preservation And Limits

All six current source hashes still match initial pins at the end of every
completed check. Main's persistent originals at reviews/exporter_original_v001/
were independently hashed and match all six exact pins above; the original
provenance copy is main-reported pinned529263... and the actual live provenance
pin was independently verified. /tmp/paper-export-independent-GT1GQ7/ also keeps
the authenticated original whole source/test bytes for resident repros.
No implementation, test, generated evidence, manuscript, actual build or frozen
experiment source was edited. Only this raw report persisted from the reviewer.

The initial metadata-directory allowlist harness stop and corrected replay are
preserved; no harness failure is mislabeled as an artifact defect. All observed
processes used hidden uninitialized CUDA, cv2/torch intra/inter1/1/1, -B/no
bytecode, pytest plugin-autoload/cache off. No GPU/service/status query, GT/RGB/
NPY/weights/count stream/bootstrap or flagged reference mutation was accessed.
Ordinary CPU font/Poppler commands were used only for scratch derived diagrams.

Exact harness SHA256 including final newline:

```text
setup 1497d8192aed691495ffc30c3ad417632e829d2c0fd929c78d5827cb5fbb6536
identity_repros e9f06725e0d3e8de967160d6888b7361a3c446a7664e06a474126cf35b9db1a8
original_tests 3a8b9564cf55c368a6d498aa73c776bdc1fba1007fae0035353360292dc46560
pinned_projection 8b99f2d64e20e7e82ba631a4f85be61017d185ff1e4d9dfb8fa09a5ea86f51a1
pinned_projection_v2 b6bbad9117457af0f98dcac1798e8a763be10e42bf6b21cd7b4763e72b4dba97
qa_helpers 6140a3f96c2f72402ca5716500d612e588d1cd7e552f3dc44f84c669fc451825
projection_leaf_oracle f3a4b94d52d570ac2140cd4aab0a30b1d3342de3812843cb5eb6cf9a69c837ae
```
