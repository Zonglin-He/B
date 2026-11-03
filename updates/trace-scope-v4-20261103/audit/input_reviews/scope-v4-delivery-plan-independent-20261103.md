# Independent Exact Scope-v4 Delivery Input-Plan CPU Admission

Distinct exact-input-plan admission only, after completed delivery CODE reviews.
Not a new paper/scientific/C1-C3 CODE review or real package build/publication.
User expressly permits bounded RAM materialize() inside read-only validate(),
including exact enumerated input bytes and original paper tar members in memory.
No persistent packet/copy/extraction, create-plan, actual build, service/GPU/
network/Git/upload/GT/rawRGB/NPY/weights/countstream/metric/bootstrap work is
authorized. Embedded metadata/source/payload paths are never followed.
Six reference-mutation demonstrations remain NOT RUN.

Requested outputs: this raw report and exact-plan ADMITTED.json (or a denied
decision if checks fail). Intended decision location:
`/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_plan_002/ADMITTED.json`.
The decision is a manual operator gate; the closed builder does not implicitly
grant scientific/publication permission or parse the review's conclusions.

## Initial Authentication

```text
SPEC.json
/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_spec_002/SPEC.json
2b12d51f18c231293334086d1a72b8ad7c7eede32c06b04caa6d02c00365650e 6708 bytes
PIN_PLAN.json
/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_plan_002/PIN_PLAN.json
3801bba95c6f83ecd3eae6c95e9fbf042de60d2a06741b66b870facc339c02d7 74294 bytes
Resident runtime (outside first-party inventory)
/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_plan_002/runtime/delivery_packager.py
d66504d3b23a4adb73b78c26e8c5dab599c31b6fde254dcc319df60d3d449ff3 46190 bytes
Worktree source: same d66504d3... exact bytes
Tests d3e4ac4cd5adba2611e0c96afbd07d4a996e78652213e199440a1aae55563970
code_v003.tar.gz 37c34fe0ab72269bd4caa956a1840f9d96badec8bc4de12d32897102918f2de0
CODE_FREEZE.json 0947dfa26644ab6c8842dc9bd31f509f5bebce838c3b1e9c6dce186da5e024f7
```

Canonical nofollow, single-link regular-file bounded FD reads verified exact
hash/extent and stable dev/inode/size/mtime/ctime/nlink/mode for spec, plan and
resident runtime before parsing/execution. The canonical JSON encoding of the
spec and plan.specification agreed exactly, including typed ACK values. Source,
test and code-freeze hashes were independently checked. Initial inventory:
253 current code/config/notices files;36 pinned exporter metadata files; one
explicit extra M12 terminal; original whole v001/v002 papers and manifests;
both raw manuscript reviews, explicit CODE review/closure history, disposition
and publication documents. Full counts/caps and final disposition follow
actual validation below, not assumed from main's preparation result.

R1 ACK is PARTIAL; local mask/subset reconstruction recovered=false;
paper_acceptance_claimed=false; publication_actions_authorized=false. This
retains the historical limitation rather than fabricating scientific PASS.

Initial disposition was pending actual readonly validator and independent input checks.
Final disposition below is DENIED; no admission permission is issued.

## Independent Read-Only Input Check

The following checker reconstructs the exact records independently of the
packager, stats the complete graph before content, hashes all enumerated leaves,
checks unique roles/inodes/caps, authenticates registry membership and verifies
both complete paper archives in bounded RAM without filesystem extraction.
No embedded non-enumerated paths are followed and no paper table/metric is
recomputed. The resident validator's separate exact CLI run follows below.

```python independent_inputs
import os,stat,json,pathlib,hashlib,io,tarfile,zlib,collections,re,time
from pathlib import Path,PurePosixPath
START=time.monotonic()
W=Path('/home/wwww/.codex/worktrees/trace-simplification/hzl')
R=Path('/home/wwww/hzl/results')
S=R/'trace_scope_v4_delivery_20261103_spec_002/SPEC.json'
P=R/'trace_scope_v4_delivery_20261103_plan_002/PIN_PLAN.json'
RT=R/'trace_scope_v4_delivery_20261103_plan_002/runtime/delivery_packager.py'
def sha(b):return hashlib.sha256(b).hexdigest()
def identity(x):return [x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns,x.st_nlink,x.st_mode]
def canonical_obj(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def pairs(items):
 out={}
 for k,v in items:
  assert k not in out,'duplicate JSON key';out[k]=v
 return out
def parsed(raw):return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def ancestors(p):
 out=[]
 for d in reversed(p.parent.parents):
  st=d.lstat();assert stat.S_ISDIR(st.st_mode) and not d.is_symlink();out.append((d,(st.st_dev,st.st_ino,st.st_mode)))
 st=p.parent.lstat();assert stat.S_ISDIR(st.st_mode) and not p.parent.is_symlink();out.append((p.parent,(st.st_dev,st.st_ino,st.st_mode)))
 return out
def observed(path,cap,expected_size):
 p=Path(path);assert p.is_absolute() and p.resolve(strict=True)==p and not p.is_symlink()
 ancestors(p);st=p.lstat();assert stat.S_ISREG(st.st_mode) and st.st_nlink==1 and st.st_size==expected_size and 0<=st.st_size<=cap
 return identity(st)
def bounded(pin,cap,expected_identity=None):
 path=Path(pin['path']);parent_ids=ancestors(path);before=observed(path,cap,pin['bytes'])
 assert expected_identity is None or before==expected_identity
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 try:
  assert identity(os.fstat(fd))==before
  raw=bytearray()
  while len(raw)<=before[2]:
   b=os.read(fd,min(65536,before[2]+1-len(raw)))
   if not b:break
   raw.extend(b)
  assert len(raw)==before[2] and sha(raw)==pin['sha256'] and identity(os.fstat(fd))==before==identity(path.lstat())
  for parent,known in parent_ids:
   st=parent.lstat();assert (st.st_dev,st.st_ino,st.st_mode)==known
  assert path.resolve(strict=True)==path
  return bytes(raw),before
 finally:os.close(fd)
specpin=dict(path=str(S),sha256='2b12d51f18c231293334086d1a72b8ad7c7eede32c06b04caa6d02c00365650e',bytes=6708)
planpin=dict(path=str(P),sha256='3801bba95c6f83ecd3eae6c95e9fbf042de60d2a06741b66b870facc339c02d7',bytes=74294)
runtimepin=dict(path=str(RT),sha256='d66504d3b23a4adb73b78c26e8c5dab599c31b6fde254dcc319df60d3d449ff3',bytes=46190)
specraw,specid=bounded(specpin,1048576);planraw,planid=bounded(planpin,1048576);rtraw,rtid=bounded(runtimepin,1048576)
spec,plan=parsed(specraw),parsed(planraw)
assert canonical_obj(spec)==canonical_obj(plan['specification'])
assert spec['acknowledgements']==dict(R1_status='PARTIAL',all_referenced_reviews_completed=True,local_mask_subset_generation_recovered=False,local_packaging_authorized=True,paper_acceptance_claimed=False,publication_actions_authorized=False)
assert canonical_obj(spec['acknowledgements'])=='{"R1_status":"PARTIAL","all_referenced_reviews_completed":true,"local_mask_subset_generation_recovered":false,"local_packaging_authorized":true,"paper_acceptance_claimed":false,"publication_actions_authorized":false}'
L=plan['limits'];records=[]
def add(pin,member,role,cap):
 assert set(pin)=={'path','sha256','bytes'} and type(pin['bytes']) is int and pin['bytes']>0 and re.fullmatch('[a-f0-9]{64}',pin['sha256'])
 records.append(dict(pin=pin,member=member,role=role,cap=cap))
for paper in spec['papers']:
 assert paper['label'] in ('draft_v001','draft_v002')
 for kind in ('archive','manifest'):
  assert Path(paper[kind]['path']).is_relative_to(W/'paper_evolution/scope_v4_20261103')
 add(paper['archive'],'paper/'+paper['label']+'/original.tar.gz','original_paper_archive',L['paper_archive'])
 add(paper['manifest'],'paper/'+paper['label']+'/FREEZE.json','paper_manifest',L['metadata'])
for label,pin in spec['reviews'].items():add(pin,'reviews/'+Path(pin['path']).name,'raw_manuscript_review',L['review'])
add(spec['disposition'],'publication/'+str(Path(spec['disposition']['path']).relative_to(W)),'post_review_disposition',L['review'])
add(spec['code_review'],'reviews/code/'+Path(spec['code_review']['path']).name,'raw_delivery_code_review',L['review'])
for pin in spec['additional_code_reviews']:add(pin,'reviews/code/'+Path(pin['path']).name,'additional_new_raw_code_review',L['review'])
add(spec['exporter_provenance'],'evidence/exporter_source_provenance.json','exporter_registry',L['metadata'])
for pin in spec['documents']:add(pin,'publication/'+str(Path(pin['path']).relative_to(W)),'publication_document',L['review'])
for pin in spec['extra_metadata']:add(pin,'evidence/'+str(Path(pin['path']).relative_to(R/'trace_mechanism_20261102')),'extra_small_metadata',L['metadata'])
for pin in plan['source_metadata']:add(pin,'evidence/'+str(Path(pin['path']).relative_to(R/'trace_mechanism_20261102')),'sealed_summary_metadata',L['metadata'])
roots=('trace_ovss','tools/evolution','tools/evolution_fifty','tests')
notices={'README.md','pyproject.toml','trace_ovss/resources/NOTICE.md','trace_ovss/resources/licenses/NACLIP_LICENSE.txt'}
excluded={'external','vendor','third_party','dependencies','node_modules','site-packages','datasets','results','artifacts','outputs','checkpoints','weights','cache','__pycache__','fixtures','credentials','credentials.json','secrets','secrets.json','secrets.py','id_rsa','id_ed25519','service-account.json'}
for pin in plan['code']:
 name=str(Path(pin['path']).relative_to(W));p=PurePosixPath(name)
 assert name in notices or any(p.is_relative_to(PurePosixPath(root)) for root in roots)
 assert not any(part.startswith('.') or part.lower() in excluded for part in p.parts)
 assert not any(part=='data' for part in p.parts) or p.parts[:2]==('trace_ovss','data')
 assert p.suffix.lower() in {'.py','.txt','.yaml','.yml','.toml','.json','.md'}
 add(pin,'implementation/current_first_party/'+name,'current_code_not_frozen_runtime',L['code'])
assert len(plan['code'])==253 and len(plan['source_metadata'])==36
assert len(spec['extra_metadata'])==1 and spec['extra_metadata'][0]['path']==str(R/'trace_mechanism_20261102/M12_CPU_terminal_001.json')
assert len({r['pin']['path'] for r in records})==len(records)==len({r['member'] for r in records})
assert sum(r['pin']['bytes'] for r in records)<=L['total'] and len(records)+3<=L['members']
observations={};inodes=set()
for r in records:
 pin=r['pin'];state=observed(pin['path'],r['cap'],pin['bytes']);assert tuple(state[:2]) not in inodes
 observations[pin['path']]=state;inodes.add(tuple(state[:2]))
kept={};roles=collections.defaultdict(lambda:dict(files=0,bytes=0))
for r in records:
 raw,seen=bounded(r['pin'],r['cap'],observations[r['pin']['path']])
 roles[r['role']]['files']+=1;roles[r['role']]['bytes']+=len(raw)
 if r['role']=='current_code_not_frozen_runtime':assert b'\0' not in raw;raw.decode('utf-8')
 else:kept[r['member']]=raw
registry=parsed(kept['evidence/exporter_source_provenance.json'])
assert sha(kept['evidence/exporter_source_provenance.json'])=='096f91f464f98484e4253f95142947229154ca43ab30a49eefbf0c63eb5fb645'
assert registry['results_root']==str(R/'trace_mechanism_20261102') and len(registry['source_pins'])==36
expectedmeta=[dict(path=str(R/'trace_mechanism_20261102'/name),sha256=pin['sha256'],bytes=pin['bytes']) for name,pin in sorted(registry['source_pins'].items())]
assert canonical_obj(expectedmeta)==canonical_obj(plan['source_metadata'])
assert all(any(Path(pin['path']).name==n for pin in plan['code']) for n in ('README.md','pyproject.toml','NOTICE.md','NACLIP_LICENSE.txt'))
codepins={pin['path']:pin for pin in plan['code']}
assert codepins[str(W/'tools/evolution_fifty/delivery_packager.py')]['sha256']==runtimepin['sha256']
assert codepins[str(W/'tools/evolution_fifty/test_delivery_packager.py')]['sha256']=='d3e4ac4cd5adba2611e0c96afbd07d4a996e78652213e199440a1aae55563970'
papers=[];evidence=[]
def safe(name):
 p=PurePosixPath(name);assert name and str(p)==name and not p.is_absolute() and '..' not in p.parts and not any(x.startswith('.') for x in p.parts) and '\\' not in name and not any(ord(c)<32 for c in name)
for paper in spec['papers']:
 prefix='paper/'+paper['label']+'/'
 archive_raw=kept[prefix+'original.tar.gz'];manifest_raw=kept[prefix+'FREEZE.json'];manifest=parsed(manifest_raw)
 assert manifest['label']==paper['label'] and len(manifest['members'])==paper['members']
 expected=dict(manifest['members']);assert sum(p['bytes'] for p in expected.values())<=L['paper_expanded']
 if paper['manifest_member'] is not None:assert paper['manifest_member'] not in expected;expected[paper['manifest_member']]=dict(bytes=len(manifest_raw),sha256=sha(manifest_raw))
 z=zlib.decompressobj(16+zlib.MAX_WBITS);expanded=z.decompress(archive_raw,L['paper_expanded']+1);assert z.eof and not z.unused_data and len(expanded)<=L['paper_expanded']
 tar=tarfile.open(fileobj=io.BytesIO(expanded),mode='r:');seen=set();dirnames=set();derived=[];ev=None
 for i,m in enumerate(tar):
  assert i<1024
  n=m.name.rstrip('/') if m.isdir() else m.name;safe(n)
  if m.isdir():assert n not in dirnames;dirnames.add(n);continue
  assert m.isfile() and not m.linkname and 0<=m.size<=L['paper_member']
  if paper['tar_prefix']:assert n.startswith(paper['tar_prefix']+'/');n=n[len(paper['tar_prefix'])+1:]
  assert n in expected and n not in seen;seen.add(n)
  pin=expected[n];assert m.size==pin['bytes'] and re.fullmatch('[0-9a-f]{64}',pin['sha256'])
  content=tar.extractfile(m).read(L['paper_member']+1);assert len(content)==m.size and sha(content)==pin['sha256']
  if n.endswith('.png'):assert n in paper['derived_composites'];derived.append(n)
  else:assert PurePosixPath(n).suffix in {'.tex','.bib','.sty','.bst','.md','.py','.json','.pdf'}
  if n=='generated/evidence.json':ev=sha(content)
 assert seen==set(expected) and set(derived)==set(paper['derived_composites'])
 assert not any(expanded[tar.offset:])
 assert manifest['members']['generated/source_provenance.json']=={k:spec['exporter_provenance'][k] for k in ('bytes','sha256')}
 if ev:evidence.append(ev)
 papers.append(dict(label=paper['label'],archive_sha256=sha(archive_raw),archive_bytes=len(archive_raw),manifest_sha256=sha(manifest_raw),manifested_members=len(manifest['members']),archive_regular_members=len(seen),directory_members=len(dirnames),expanded_bytes=len(expanded),declared_member_bytes=sum(x['bytes'] for x in manifest['members'].values()),four_derived_composites=derived,evidence_sha256=ev))
assert len(evidence)==2 and len(set(evidence))==1 and evidence[0]=='eda4079f5c01b41b1b8e3f7538449ba2ddf06272af453956778c8e3df33fb845'
disposition=kept['publication/docs/scope-v4-paper-review-disposition-20261103.md'].decode()
r1lines=[line for line in disposition.splitlines() if 'R1' in line or 'PARTIAL' in line or 'unrecover' in line.lower()]
assert 'PARTIAL' in disposition and 'R1' in disposition
for r in records:assert observed(r['pin']['path'],r['cap'],r['pin']['bytes'])==observations[r['pin']['path']]
bounded(specpin,1048576,specid);bounded(planpin,1048576,planid);bounded(runtimepin,1048576,rtid)
audit_records=[dict(**r,observed_identity=observations[r['pin']['path']]) for r in records]
result=dict(independent_check_passed=True,spec_pin=dict(**specpin,identity=specid),plan_pin=dict(**planpin,identity=planid),runtime_pin=dict(**runtimepin,identity=rtid),input_files=len(records),input_bytes=sum(r['pin']['bytes'] for r in records),unique_input_inodes=len(inodes),roles=dict(roles),papers=papers,registry_names=sorted(registry['source_pins']),explicit_extra_metadata=spec['extra_metadata'],scope=plan['scope'],acknowledgements=spec['acknowledgements'],limits=L,R1_disposition_lines=r1lines,observed_input_graph_sha256=sha((canonical_obj(audit_records)+'\n').encode()),input_records=audit_records,seconds=time.monotonic()-START,persistent_writes=0,actual_raw_payload_reads=0,model_or_metric_execution=False)
print(json.dumps(result))

```

## Confirmed Admission Blocker: Single-Link Policy Incompatibility

**Final decision: DENIED. CPU_LOCAL_BOUNDED_BUILD=false.**

This is a blocking exact-plan input compatibility finding, not a reopening of
closed C1-C3 CODE reviews, a scientific failure, or evidence of content
corruption. The closed implementation explicitly requires nlink==1 in
`delivery_packager.py:212`. Its materialize stat preflight calls observe at
line529, validate calls materialize at line680, and main invokes validate at
line846. Exact plan002 includes nine sealed metadata inputs whose observed
link count is 2. The first rejection is G03 ANALYSIS_SEALED.json.

The independent checker preserved above stopped in its complete-graph
stat preflight with an AssertionError (observed regular/single-link/cap
condition). It never reached the subsequent content authentication, registry
parse, paper archive loops or disposition-text checks in that script.
Those lines are preserved intended checks, NOT claimed executed successes.
A subsequent reviewer orchestration attempt used JavaScript `s.index` instead
of `s.indexOf` and failed before a child command ran. That setup error is
separate from this independently confirmed input rejection.

### Complete Enumerated Stat Graph

312 exact enumerated input leaves; 16,188,345 pinned and observed bytes.
All 312 extents matched their pins and role caps. No role-inode collision,
symlink or nonregular input was observed. Nine link-count failures were found;
each has device 66306, regular-file mode 33204, role
sealed_summary_metadata, and role cap 8,388,608 bytes. This is stat-only
evidence, not content-hash or immutability proof.

Paths below are relative to
`/home/wwww/hzl/results/trace_mechanism_20261102/`.

| Exact Enumerated Path | Inode | Bytes | nlink |
|---|---:|---:|---:|
| `G03_offline_score_20261103_1455/ANALYSIS_SEALED.json` | 59638127 | 5738 | 2 |
| `G_offline_score_20261102_2321/ANALYSIS_SEALED.json` | 58237988 | 4186 | 2 |
| `M01_offline_score_20261102_1942/ANALYSIS_SEALED.json` | 58205233 | 1816 | 2 |
| `M01_offline_score_20261102_1942/coco_object/ANALYSIS_SEALED.json` | 58204956 | 654 | 2 |
| `M01_offline_score_20261102_1942/pc60/ANALYSIS_SEALED.json` | 58205236 | 654 | 2 |
| `M02_offline_score_20261103_0156/ANALYSIS_SEALED.json` | 58238105 | 1774 | 2 |
| `M03_offline_score_20261103_0856/ANALYSIS_SEALED.json` | 59376327 | 2018 | 2 |
| `M04_offline_score_20261103_0646/ANALYSIS_SEALED.json` | 58238398 | 1743 | 2 |
| `V01_offline_score_20261103_1258/ANALYSIS_SEALED.json` | 59404559 | 2020 | 2 |

No alternate hardlink names were sought. No seal was unlinked, rewritten,
copied, rehashed for workaround, or otherwise changed. No embedded metadata
payload path was followed. The graph check read no enumerated input contents
after preflight. The 253 current first-party code/config/notices entries,
36 registry metadata pins and explicit M12 terminal are plan-level inventory
observations, not fresh semantic authentication of all those contents.

### Actual Resident Validator Result

The exact authenticated resident runtime was executed through unchanged
main(), with its ordinary validate argv. A wrapper enforces enumerated reads,
denies filesystem mutation/process/network, and catches the exception solely
to report it. Its outer process exit0 means failure observation completed,
NOT that validation succeeded. Internal materialize is allowed in RAM, but
this invocation stopped before any enumerated payload content open.

Environment: CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE=1,
OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; mlmp Python3.10 with -B. Explicit
torch intra/inter and cv2 threads were observed [1,1,1]; CUDA not initialized.
No pytest regression or scientific computation was required or run for this
denied exact-input admission.

```json
{
  "validator_succeeded": false,
  "error_type": "ValueError",
  "error": "Not bounded single-link regular input",
  "traceback": "Traceback (most recent call last):\n  File \"<string>\", line 79, in <module>\n  File \"/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_plan_002/runtime/delivery_packager.py\", line 846, in main\n    result = validate(pin)\n  File \"/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_plan_002/runtime/delivery_packager.py\", line 680, in validate\n    payloads, observations, manifest = materialize(plan, raw, policy, limits)\n  File \"/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_plan_002/runtime/delivery_packager.py\", line 529, in materialize\n    observed = observe(pin[\"path\"], r[\"cap\"])\n  File \"/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_plan_002/runtime/delivery_packager.py\", line 212, in observe\n    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and 0 <= s.st_size <= cap,\n  File \"/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_plan_002/runtime/delivery_packager.py\", line 123, in require\n    raise ValueError(message)\nValueError: Not bounded single-link regular input\n",
  "exact_cli": [
    "/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_plan_002/runtime/delivery_packager.py",
    "validate",
    "--plan",
    "/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_plan_002/PIN_PLAN.json",
    "--sha256",
    "3801bba95c6f83ecd3eae6c95e9fbf042de60d2a06741b66b870facc339c02d7"
  ],
  "resident_source_sha256": "d66504d3b23a4adb73b78c26e8c5dab599c31b6fde254dcc319df60d3d449ff3",
  "spec_identity": [
    66306,
    59638144,
    6708,
    "1793702770143490492",
    "1793702770143490492",
    1,
    33204
  ],
  "plan_identity": [
    66306,
    59638146,
    74294,
    "1793702775038611027",
    "1793702775038611027",
    1,
    33152
  ],
  "runtime_identity": [
    66306,
    59638148,
    46190,
    "1793702838492276445",
    "1793702838492276445",
    1,
    33204
  ],
  "opened_leaf_paths": [
    "/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_plan_002/PIN_PLAN.json"
  ],
  "opened_enumerated_payload_paths": [],
  "directory_opens": 361,
  "persistent_write_attempts": [],
  "CUDA_initialized": false,
  "threads": [
    1,
    1,
    1
  ],
  "seconds": 0.050635794177651405,
  "peak_RSS_bytes": 7159103488
}
```

The reported peak RSS belongs to this instrumented Python/torch/cv2 wrapper,
not a measurement of pure packager working memory or proof of a memory cap.
No external memory cap was asserted. The failure occurred in about 0.051s.

Exact executable wrapper, preserved without converting rejection to success:

```python resident_validator
import os,sys,stat,json,hashlib,types,traceback,time,resource
from pathlib import Path
import torch,cv2
torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
def no_model(*a,**kw):raise AssertionError('No model/GPU')
torch.load=no_model;torch.jit.load=no_model;torch.cuda.init=no_model
START=time.monotonic()
W=Path('/home/wwww/.codex/worktrees/trace-simplification/hzl')
ROOT=Path('/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_plan_002')
SPEC=Path('/home/wwww/hzl/results/trace_scope_v4_delivery_20261103_spec_002/SPEC.json')
PLAN=ROOT/'PIN_PLAN.json';RUNTIME=ROOT/'runtime/delivery_packager.py'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def ident(s):return [s.st_dev,s.st_ino,s.st_size,str(s.st_mtime_ns),str(s.st_ctime_ns),s.st_nlink,s.st_mode]
def authenticate(path,expected,size):
 assert path.resolve(strict=True)==path
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW);before=os.fstat(fd)
 try:
  assert stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size==size
  raw=bytearray()
  while len(raw)<=size:
   b=os.read(fd,min(65536,size+1-len(raw)))
   if not b:break
   raw.extend(b)
  assert len(raw)==size and sha(raw)==expected and ident(os.fstat(fd))==ident(before)==ident(path.stat())
  return bytes(raw),ident(before)
 finally:os.close(fd)
spec_raw,spec_identity=authenticate(SPEC,'2b12d51f18c231293334086d1a72b8ad7c7eede32c06b04caa6d02c00365650e',6708)
plan_raw,plan_identity=authenticate(PLAN,'3801bba95c6f83ecd3eae6c95e9fbf042de60d2a06741b66b870facc339c02d7',74294)
worker,runtime_identity=authenticate(RUNTIME,'d66504d3b23a4adb73b78c26e8c5dab599c31b6fde254dcc319df60d3d449ff3',46190)
plan=json.loads(plan_raw);spec=json.loads(spec_raw);assert json.dumps(plan['specification'],sort_keys=True)==json.dumps(spec,sort_keys=True)
allowed={str(SPEC),str(PLAN),str(RUNTIME)}
def collect(x):
 if isinstance(x,dict):
  if set(x)=={'path','sha256','bytes'}:allowed.add(x['path'])
  else:
   for v in x.values():collect(v)
 elif isinstance(x,list):
  for v in x:collect(v)
collect(plan)
code_roots=[W/name for name in ('trace_ovss','tools/evolution','tools/evolution_fifty','tests')]
mod=types.ModuleType('delivery_plan_002_resident');mod.__file__=str(RUNTIME);sys.modules[mod.__name__]=mod
exec(compile(worker,str(RUNTIME),'exec'),mod.__dict__)
opened_files=[];directory_opens=0;writes=[]
def absolute(path,dir_fd=None):
 if isinstance(path,int):return Path(os.readlink('/proc/self/fd/'+str(path)))
 p=Path(os.fsdecode(path))
 return p if p.is_absolute() else (Path.cwd() if dir_fd is None else Path(os.readlink('/proc/self/fd/'+str(dir_fd))))/p
def system_path(path):return path.is_relative_to(sys.prefix) or path.is_relative_to('/usr/share/locale') or path.is_relative_to('/usr/lib/locale')
def leaf(path):
 p=absolute(path).resolve()
 assert str(p) in allowed or system_path(p),'Unlisted content open refused: '+str(p)
def directory_allowed(path):
 p=absolute(path).resolve()
 return system_path(p) or any(p.is_relative_to(r) for r in code_roots) or any(Path(n).is_relative_to(p) for n in allowed)
original_open=os.open
def opening(path,flags,mode=0o777,*,dir_fd=None):
 global directory_opens
 p=absolute(path,dir_fd)
 assert not flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND),'No persistent writes'
 if flags & os.O_DIRECTORY:
  assert directory_allowed(p),'Foreign directory open';directory_opens+=1
 else:
  leaf(p);opened_files.append(str(p))
 return original_open(path,flags,mode,dir_fd=dir_fd)
os.open=opening
def audit(event,args):
 if event in ('subprocess.Popen','socket.connect','socket.bind','os.system'):raise AssertionError('No process/network/service/Git')
 if event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.link','os.symlink','os.truncate','os.chmod','os.chown','os.utime','shutil.copyfile'):
  writes.append(event);raise AssertionError('No persistent outputs')
 if event in ('os.scandir','os.listdir'):
  p=absolute(args[0]).resolve()
  assert system_path(p) or any(p.is_relative_to(root) for root in code_roots),'No actual result/data tree enumeration'
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  assert not args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND),'No persistent write'
  if args[1] is not None:leaf(args[0])
sys.addaudithook(audit)
sys.argv=[str(RUNTIME),'validate','--plan',str(PLAN),'--sha256','3801bba95c6f83ecd3eae6c95e9fbf042de60d2a06741b66b870facc339c02d7']
try:
 mod.main();result=dict(validator_succeeded=True)
except BaseException as exc:
 result=dict(validator_succeeded=False,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
assert ident(SPEC.stat())==spec_identity and ident(PLAN.stat())==plan_identity and ident(RUNTIME.stat())==runtime_identity
result.update(exact_cli=sys.argv,resident_source_sha256=sha(worker),spec_identity=spec_identity,plan_identity=plan_identity,runtime_identity=runtime_identity,opened_leaf_paths=opened_files,opened_enumerated_payload_paths=[p for p in opened_files if p not in (str(SPEC),str(PLAN),str(RUNTIME))],directory_opens=directory_opens,persistent_write_attempts=writes,CUDA_initialized=torch.cuda.is_initialized(),threads=[torch.get_num_threads(),torch.get_num_interop_threads(),cv2.getNumThreads()],seconds=time.monotonic()-START,peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024)
print(json.dumps(result))

```

### Final Binding And Limits

Final explicit rehashes matched all initial pins: SPEC002 2b12d51f..., plan002
3801bba9..., resident runtime and worktree source d66504d3..., tests d3e4ac4c...,
code003 archive 37c34fe0..., and CODE_FREEZE 0947dfa2.... The actual wrapper
also checked exact before/after SPEC, plan and resident identities, recorded
above with nanosecond timestamps as strings. Prior raw CODE reports remain
8c8da4b5..., 69e40c43..., and 7ca53287... unchanged.

Because preflight rejected, this review does NOT claim completion of all-input
content hashing, whole-paper archive membership validation, terminal-success
semantic authentication, every review/disposition content binding, or
publication/paper acceptance. R1 PARTIAL and the unrecovered local mask/subset
limitation remain acknowledged, not upgraded to PASS.

The original proposed ADMITTED.json location above was a provisional success
path only. No such marker was created. At the user's final direction, the
DENIED decision is instead written to worktree
`docs/scope-v4-delivery-plan-admission-20261103.json`, binding this final report,
source, resident runtime, codefreeze, SPEC and plan. This keeps plan002 and all
sealed evidence unchanged.

All permissions are false, including CPU_LOCAL_BOUNDED_BUILD, GPU, GT scoring,
raw RGB/NPY/model access, network, Git, upload and paper acceptance. No packet,
build, plan retry, persistent extraction, service operation, or publication
was performed. No policy workaround is authorized. Any supported repair must
be preserved in a new codefreeze/SPEC/plan and independently reconsidered;
this exact plan002 remains DENIED. Review completed at this bounded result.

## Stat-Only Diagnostic Reproduction

This script reconstructs only explicit plan records and stats them. It does not
open their contents or follow embedded payload references. Its observed result
is the 312/9 table above. Timestamp fields from this diagnostic are deliberately
not used in the decision JSON; the exact resident-validator timestamps remain
strings in the preserved raw result.

```python stat_only_graph
import os,stat,json,pathlib,hashlib,io,tarfile,zlib,collections,re,time
from pathlib import Path,PurePosixPath
START=time.monotonic()
W=Path('/home/wwww/.codex/worktrees/trace-simplification/hzl')
R=Path('/home/wwww/hzl/results')
S=R/'trace_scope_v4_delivery_20261103_spec_002/SPEC.json'
P=R/'trace_scope_v4_delivery_20261103_plan_002/PIN_PLAN.json'
RT=R/'trace_scope_v4_delivery_20261103_plan_002/runtime/delivery_packager.py'
def sha(b):return hashlib.sha256(b).hexdigest()
def identity(x):return [x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns,x.st_nlink,x.st_mode]
def canonical_obj(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def pairs(items):
 out={}
 for k,v in items:
  assert k not in out,'duplicate JSON key';out[k]=v
 return out
def parsed(raw):return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def ancestors(p):
 out=[]
 for d in reversed(p.parent.parents):
  st=d.lstat();assert stat.S_ISDIR(st.st_mode) and not d.is_symlink();out.append((d,(st.st_dev,st.st_ino,st.st_mode)))
 st=p.parent.lstat();assert stat.S_ISDIR(st.st_mode) and not p.parent.is_symlink();out.append((p.parent,(st.st_dev,st.st_ino,st.st_mode)))
 return out
def observed(path,cap,expected_size):
 p=Path(path);assert p.is_absolute() and p.resolve(strict=True)==p and not p.is_symlink()
 ancestors(p);st=p.lstat();assert stat.S_ISREG(st.st_mode) and st.st_nlink==1 and st.st_size==expected_size and 0<=st.st_size<=cap
 return identity(st)
def bounded(pin,cap,expected_identity=None):
 path=Path(pin['path']);parent_ids=ancestors(path);before=observed(path,cap,pin['bytes'])
 assert expected_identity is None or before==expected_identity
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 try:
  assert identity(os.fstat(fd))==before
  raw=bytearray()
  while len(raw)<=before[2]:
   b=os.read(fd,min(65536,before[2]+1-len(raw)))
   if not b:break
   raw.extend(b)
  assert len(raw)==before[2] and sha(raw)==pin['sha256'] and identity(os.fstat(fd))==before==identity(path.lstat())
  for parent,known in parent_ids:
   st=parent.lstat();assert (st.st_dev,st.st_ino,st.st_mode)==known
  assert path.resolve(strict=True)==path
  return bytes(raw),before
 finally:os.close(fd)
specpin=dict(path=str(S),sha256='2b12d51f18c231293334086d1a72b8ad7c7eede32c06b04caa6d02c00365650e',bytes=6708)
planpin=dict(path=str(P),sha256='3801bba95c6f83ecd3eae6c95e9fbf042de60d2a06741b66b870facc339c02d7',bytes=74294)
runtimepin=dict(path=str(RT),sha256='d66504d3b23a4adb73b78c26e8c5dab599c31b6fde254dcc319df60d3d449ff3',bytes=46190)
specraw,specid=bounded(specpin,1048576);planraw,planid=bounded(planpin,1048576);rtraw,rtid=bounded(runtimepin,1048576)
spec,plan=parsed(specraw),parsed(planraw)
assert canonical_obj(spec)==canonical_obj(plan['specification'])
assert spec['acknowledgements']==dict(R1_status='PARTIAL',all_referenced_reviews_completed=True,local_mask_subset_generation_recovered=False,local_packaging_authorized=True,paper_acceptance_claimed=False,publication_actions_authorized=False)
assert canonical_obj(spec['acknowledgements'])=='{"R1_status":"PARTIAL","all_referenced_reviews_completed":true,"local_mask_subset_generation_recovered":false,"local_packaging_authorized":true,"paper_acceptance_claimed":false,"publication_actions_authorized":false}'
L=plan['limits'];records=[]
def add(pin,member,role,cap):
 assert set(pin)=={'path','sha256','bytes'} and type(pin['bytes']) is int and pin['bytes']>0 and re.fullmatch('[a-f0-9]{64}',pin['sha256'])
 records.append(dict(pin=pin,member=member,role=role,cap=cap))
for paper in spec['papers']:
 assert paper['label'] in ('draft_v001','draft_v002')
 for kind in ('archive','manifest'):
  assert Path(paper[kind]['path']).is_relative_to(W/'paper_evolution/scope_v4_20261103')
 add(paper['archive'],'paper/'+paper['label']+'/original.tar.gz','original_paper_archive',L['paper_archive'])
 add(paper['manifest'],'paper/'+paper['label']+'/FREEZE.json','paper_manifest',L['metadata'])
for label,pin in spec['reviews'].items():add(pin,'reviews/'+Path(pin['path']).name,'raw_manuscript_review',L['review'])
add(spec['disposition'],'publication/'+str(Path(spec['disposition']['path']).relative_to(W)),'post_review_disposition',L['review'])
add(spec['code_review'],'reviews/code/'+Path(spec['code_review']['path']).name,'raw_delivery_code_review',L['review'])
for pin in spec['additional_code_reviews']:add(pin,'reviews/code/'+Path(pin['path']).name,'additional_new_raw_code_review',L['review'])
add(spec['exporter_provenance'],'evidence/exporter_source_provenance.json','exporter_registry',L['metadata'])
for pin in spec['documents']:add(pin,'publication/'+str(Path(pin['path']).relative_to(W)),'publication_document',L['review'])
for pin in spec['extra_metadata']:add(pin,'evidence/'+str(Path(pin['path']).relative_to(R/'trace_mechanism_20261102')),'extra_small_metadata',L['metadata'])
for pin in plan['source_metadata']:add(pin,'evidence/'+str(Path(pin['path']).relative_to(R/'trace_mechanism_20261102')),'sealed_summary_metadata',L['metadata'])
roots=('trace_ovss','tools/evolution','tools/evolution_fifty','tests')
notices={'README.md','pyproject.toml','trace_ovss/resources/NOTICE.md','trace_ovss/resources/licenses/NACLIP_LICENSE.txt'}
excluded={'external','vendor','third_party','dependencies','node_modules','site-packages','datasets','results','artifacts','outputs','checkpoints','weights','cache','__pycache__','fixtures','credentials','credentials.json','secrets','secrets.json','secrets.py','id_rsa','id_ed25519','service-account.json'}
for pin in plan['code']:
 name=str(Path(pin['path']).relative_to(W));p=PurePosixPath(name)
 assert name in notices or any(p.is_relative_to(PurePosixPath(root)) for root in roots)
 assert not any(part.startswith('.') or part.lower() in excluded for part in p.parts)
 assert not any(part=='data' for part in p.parts) or p.parts[:2]==('trace_ovss','data')
 assert p.suffix.lower() in {'.py','.txt','.yaml','.yml','.toml','.json','.md'}
 add(pin,'implementation/current_first_party/'+name,'current_code_not_frozen_runtime',L['code'])
assert len(plan['code'])==253 and len(plan['source_metadata'])==36
assert len(spec['extra_metadata'])==1 and spec['extra_metadata'][0]['path']==str(R/'trace_mechanism_20261102/M12_CPU_terminal_001.json')
assert len({r['pin']['path'] for r in records})==len(records)==len({r['member'] for r in records})
assert sum(r['pin']['bytes'] for r in records)<=L['total'] and len(records)+3<=L['members']

bad=[];inode_paths=collections.defaultdict(list);extents=0
for r in records:
 path=Path(r['pin']['path']);st=path.lstat();state=identity(st);problems=[]
 if not stat.S_ISREG(st.st_mode):problems.append('not_regular')
 if st.st_nlink!=1:problems.append('link_count_not_1')
 if st.st_size!=r['pin']['bytes']:problems.append('size_differs_from_pin')
 if st.st_size>r['cap']:problems.append('role_cap_exceeded')
 if path.resolve(strict=True)!=path or path.is_symlink():problems.append('noncanonical_or_symlink')
 inode_paths[tuple(state[:2])].append(str(path));extents+=st.st_size
 if problems:bad.append(dict(path=str(path),role=r['role'],problems=problems,observed_identity=state,pinned_bytes=r['pin']['bytes'],role_cap=r['cap']))
print(json.dumps(dict(stat_files=len(records),pinned_input_bytes=sum(r['pin']['bytes'] for r in records),observed_input_bytes=extents,stat_failures=bad,role_inode_collisions=[v for v in inode_paths.values() if len(v)>1],input_contents_opened_after_preflight=0)))

```


