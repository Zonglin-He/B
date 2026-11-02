# Independent Allocation Controls Code Audit

## Scope and Status

Initial CPU-only review, 2026-10-29T11:20:36Z. Worktree:
`/home/wwww/.codex/worktrees/trace-simplification/hzl`; HEAD
`aba513ce6f12f1b8c134a8bb02343aa689b214cd`.

Reviewed `common.py`, `queue.py`, `tests/test_allocation_queue.py` against
`docs/allocation-controls-contract.md`, then included `analyze.py` when requested.
Read only the directly relevant inherited helper/importer functions to establish
their contracts. No paper review or rating. No worker/runtime implementation
review, model import, GPU job, suite preparation, live service query, original
dataset/prediction/GT access, old-suite modification, commit or push occurred.
Synthetic cases use disposable temporary directories and mocked subprocesses.
The only repository file written by this auditor is this document.

**Initial review is not an admission to freeze or execute GPU work.** Findings
below are open for the identified file versions. The final worker/runtime and
author fixes require a subsequent review by this same auditor. The actual parent
importer's reported 8 populations / 32 scores / 306 metadata checks are not
disputed or independently rerun here; F1 concerns the subsequent binding of those
checks to the copied artifacts and their re-admission.

## Inspected Identities

SHA256 identities, not just branch names:

| File | SHA256 |
| --- | --- |
| `docs/allocation-controls-contract.md` | `6f09f3b9606363236cae123f8fb08772f675531df02a7b79e81b9da3f40ff8ed` |
| `tools/evolution_allocation/common.py` | `8eb0b61d58782da9291e36e3387f236e9af5411947f1e3a578465312ac716396` |
| `tools/evolution_allocation/queue.py` | `d99cc7ca02e3c902606cd52f990447ba3fd5dbbaddbfd78c6533103b8a38ac64` |
| `tools/evolution_allocation/analyze.py` | `24471596745743eb33fc2cfb619acabf68b1c56a014dd8cf508dbfaaef428cd6` |
| `tests/test_allocation_queue.py`, initial 23-test version | `cf7b5738d1b35dcf7edbb929585fa9c3678a21810ee9022b318738ff18809ade` |
| `tests/test_allocation_queue.py`, extended 26-test version | `fdf68c78920e0f277f974ddec9f78d57bcc461112cbc8b179d6ecf833e593e19` |
| `tools/evolution_tuning/common.py`, helper portions | `aafc87fd02fb935631877f2ef96024021bc313a719bd0b36ca94f6cfabfa7007` |
| `tools/evolution_tuning/worker.py`, label/input guard portions only | `259fbb901b164eed6ca3c211db612947afdf96b7bd8259df47a3c35605767755` |
| `tools/evolution_followup/queue.py`, snapshot portions | `a298cf9f4052ae4b693d7746d85aa6f30e1769cecf83e9d25af5ad164eb462f5` |
| `paper_evolution/tools/export_tuned_full.py`, importer portions only | `58dc31d65b7651544e0623dca242dd7ae7365b5ec622b42103bdf185fb44c559` |

## Findings

### F1 [P1] Copied Parent Evidence Is Not Bound to the Verified Read-Set

Locations: `queue.py:62-63,72-108`; `common.py:134-150`.

Preparation obtains `report["metadata_sha256"]`, but records it without comparing
the subsequently copied parent files with that verified read-set. The copy-time
check compares a copied file only with the live original at copy time, not with
the bytes previously validated by the importer. Re-admission ignores that
read-set entirely. It checks parent artifacts against hashes supplied by the new
prepared file and compares selected config/panel with sibling local copies.
Only parent prepared bytes and inherited runtime hashes have a fixed anchor;
the copied selection, ordered population, vocabulary, checkpoint protocol and
parent prediction/score/barrier chain are not reconstructed from that anchor.
The proof verifier accepts just the fixed parent SHA plus numeric claims 8/32.

Reproduction R1: use the supplied `frozen` fixture, retain the pinned parent
prepared bytes, change PC60 `lr` from .001 to .003 in the entry and both copied
protocol/selection files, update their self-declared artifact hashes and derive
new manifests from a newly hashed prepared file. `verify_prepared` accepts it.
The fixture's proof has no metadata read-set or parent score/barrier seals at
all, yet is accepted by the real verifier. This does NOT demonstrate mutation
under an unchanged approved new prepared SHA; it demonstrates that the claimed
continuity to the fixed original parent is not authenticated when admitting a
new prepared identity. A live-parent change between validation and copying can
reach the same gap during normal preparation.

Required correction: authenticate each copied artifact against the exact
importer-verified identity; retain and recheck the necessary parent-chain links,
including original input/selection bindings, not just sibling equality. Record
the importer identity of the code actually executed, rather than hashing its
mutable path only afterward. That importer is outside the snapshot's three
trees (`trace_ovss`, `tools`, `tests`). Add a coherent-repack negative test.

### F2 [P1] Canonical Scores and Analysis Coverage Need Not Be Sealed

Locations: `queue.py:162-172`; `analyze.py:51-61`.

`validate_results` accepts any nonempty `artifact_sha256` map. It does not require
`result.json` to be a member before reading that file. A score seal can therefore
remain unchanged while canonical scores and timings change. Analysis also reads
`prediction_result.json` without requiring its membership in an authenticated
prediction inventory; comparing two mutable copies of coverage is not sealing.

Reproduction R2/R6: give `SCORED_SEALED.json` an inventory containing only a
hashed `placeholder.json`, write all canonical fields to an unlisted
`result.json`, and call `validate_results`. It returns ten rows. The complete
`analyze` path also accepts these unsealed scores and coverage, with invented
inventory-only native hash matches and the unrelated service in F3.

Required correction: enforce required canonical artifact membership, bind
manifest/source/prepared/prediction/barrier identities at score verification,
and verify coverage from its sealed canonical provenance. Rejection must happen
before CSV publication or analysis. Test omission, post-seal mutation and a
correct-looking but unrelated inventory independently.

### F3 [P1] Analysis Accepts an Unrelated Service as Execution Certification

Locations: `analyze.py:42-50,71,80-84`; `queue.py:37-44`.

The caller supplies any `--service`; validation checks only six generic status
properties. No unit command, working directory, suite/prepared hash, execution
PID, invocation identity, or cgroup relationship ties that successful service
to this experiment. `queue_state.json` similarly needs only `phase=completed`.
The queue's explicit `service_exit_certified=False` is truthful, but analysis
does not close that outstanding certification boundary. This matters especially
because native replay deliberately relies on historical inventories instead of
rehashing original payloads.

Reproduction R6: mock only the systemctl response to the six requested
successful terminal values; call actual `analyze(...,
"unrelated-success.service")` on the synthetic suite. It accepts the unrelated
service and returns all ten rows. Neither an execution receipt nor a
root/hash/PID-bound unit record is required. No live service was queried.

Required correction: bind the retained unit and actual invocation to this
suite/root/prepared identity and execution receipt, verify actual successful
termination, and recheck before returning the final interpretation. Capture
the execution resource envelope separately. Tests must reject a successful
service belonging to another suite, wrong PID/hash/root, and nonterminal states.

### F4 [P2] Controller Gates and Prediction Barrier Trust Incomplete Receipts

Locations: `queue.py:193-213`.

A gate with only `{"passed": true}` admits prediction scheduling. There is no
controller check of dataset/manifest/prepared/source identity, original native
comparison identity, required equivalence measures or finite-arm evidence.
Similarly, prediction barrier construction accepts declared arms/image count
plus an arbitrary nonempty artifact inventory; it does not require the full
prediction records/protocol or verify the peer manifest binding there.

Reproduction R4: stub child execution to produce bare gate booleans and full
prediction seals whose inventories contain only a placeholder. With prepared
verification isolated as in existing controller tests, actual `run` writes both
barriers and schedules both score children. For the synthetic terminal test
only, final result validation was stubbed too; the observed scheduling occurs
before that stub. This is a controller-validation reproduction, NOT evidence
that the unreviewed worker actually opens GT or that native equivalence fails.
`common.verify_barrier` delegates stronger payload validation to that worker,
which must still be reviewed.

Required correction: validate canonical gate schema and provenance before
admitting prediction, and validate exact complete peer seals/manifests before
creating the all-population barrier. Keep worker-side independent checks too.
Add positive and malformed-receipt end-to-end controller tests.

### F5 [P2] Pooled Score Arithmetic Does Not Establish Totals or Actual Cost

Locations: `queue.py:173-179`; `analyze.py:57-66`.

`pooled_miou` correctly validates paired nonnegative integer arrays and computes
pooled IoU, but the controller does not require the number of entries to equal
the vocabulary/readout class count, nor reconcile aggregate counts with sealed
per-image records and exact IDs. Declared image totals do not establish count
totals. Timing is multiplied by 1000 without a finite/nonnegative check or
operation-accounting reconciliation.

Reproduction R2/R6: on the two-class fixture, every arm can use
`intersection=[1], union=[1], miou_unrounded=100`, while
`seconds_per_image=-1`. Ten canonical rows with `mIoU=100` and
`ms_per_image=-1000` are accepted, including by `analyze`. F2 is independent:
adding `result.json` to the inventory does not repair the missing dimensional
and aggregate checks.

Required correction: check class dimensions, exact record population and IDs,
recomputed totals/readout conventions, finite actual timings, and real
operation/reset/diagnostic accounting against the sealed stream. The worker
may supply such checks, but they have not been inspected or certified here.

### F6 [P2] Writable Controller Paths Can Escape the Suite Through Symlinks

Locations: `queue.py:58-66,133-151,188`; `common.py:121-134`.

The runtime inventory rejects symlinks, but writable controller directories are
not checked. In particular `logs.mkdir(exist_ok=True)` follows an existing
directory symlink; log and receipt creation then write outside the suite. The
new-output check also permits a previously nonexistent child directory inside
an existing parent/old suite or source tree; canonicality alone is not isolation.

Reproduction R5: use the fixture and an external temporary directory representing
an old suite; set `suite/logs` to a symlink to that directory. Mock subprocess
execution and the frozen-runtime check exactly as the supplied `child` test
does. Actual `child(...,"gate","pc60","gate")` creates
`pc60.gate.log` and `pc60.gate.json` in the external directory. No real old suite
was touched. The log path checks and exclusive file opens do not prevent this.

Required correction: validate disjoint output roots before any creation and
reject symlink/alias escapes for all writable directories/files, including
locks, logs, state temporary files and manifests. Add negative tests for a
symlinked logs directory and outputs nested inside protected old/source roots.

## Verification and Reproduction

Executed, CPU-only, with bytecode and pytest cache disabled:

```bash
cd /home/wwww/.codex/worktrees/trace-simplification/hzl
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 /home/wwww/miniconda3/envs/mlmp/bin/python \
  -B -m pytest -q -p no:cacheprovider tests/test_allocation_queue.py
```

Initial output: `23 passed in 0.07s`. After analysis tests were added:
`26 passed in 0.07s`. These tests do not cover full preparation, authentic parent
chain admission, full successful controller scheduling, admission import-root
provenance, score validation or full `analyze` service certification.

Compact executable reproducer for R1/R2/F5 using only the provided synthetic
fixture and temporary directories (run via the same CPU-only Python command):

```python
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
from tools.evolution_allocation import common as c, queue as q

spec = importlib.util.spec_from_file_location("audit_fixture", "tests/test_allocation_queue.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
with TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp:
    suite, prepared, identity = t.frozen.__wrapped__(Path(tmp), mp)
    pinned = c.sha256(suite / "parents/SUITE_PREPARED.json")
    entry = prepared["entries"]["pc60"]
    entry["config"]["lr"] = .003
    for name in ("prediction_protocol.json", "SELECTED_CONFIG_SEALED.json"):
        p = suite / "parents/pc60" / name
        obj = c.read_json(p)
        (obj["manifest"] if "manifest" in obj else obj)["config"] = entry["config"]
        c.write_json(p, obj, replace=True)
        prepared["parent_artifact_sha256"][str(p.relative_to(suite))] = c.sha256(p)
    c.write_json(suite / "SUITE_PREPARED.json", prepared, replace=True)
    identity = c.sha256(suite / "SUITE_PREPARED.json")
    for d in c.DATASETS:
        for stage in ("gate", "full"):
            c.write_json(c.manifest_path(suite, d, stage),
                c.expected_manifest(suite, prepared, d, stage, identity), replace=True)
    c.verify_prepared(suite, identity)
    assert pinned == c.sha256(suite / "parents/SUITE_PREPARED.json")
    print("R1: changed inherited lr accepted with unchanged pinned parent bytes")

    c.write_json(suite / "ALL_FULL_PREDICTIONS_SEALED.json", {"not_a_barrier": True})
    barrier = c.sha256(suite / "ALL_FULL_PREDICTIONS_SEALED.json")
    for d in c.DATASETS:
        out = suite / "full" / d
        out.mkdir(parents=True)
        c.write_json(out / "PREDICTIONS_SEALED.json", {"not_predictions": True})
        c.write_json(out / "placeholder.json", {})
        c.write_json(out / "SCORED_SEALED.json", dict(complete=True,
            barrier_sha256=barrier,
            prediction_seal_sha256=c.sha256(out / "PREDICTIONS_SEALED.json"),
            artifact_sha256={"placeholder.json": c.sha256(out / "placeholder.json")}))
        c.write_json(out / "result.json", dict(status="completed", dataset=d,
            images=c.POPULATIONS[d], full_split=True, config=prepared["entries"][d]["config"],
            manifest_sha256=c.sha256(c.manifest_path(suite, d, "full")), barrier_sha256=barrier,
            arm_results={a: dict(intersection=[1], union=[1], miou_unrounded=100.) for a in c.ARMS},
            timings={a: dict(seconds_per_image=-1.) for a in c.ARMS}))
    rows, _ = q.validate_results(suite, prepared)
    assert len(rows) == 10 and all(r["ms_per_image"] == -1000 for r in rows)
    print("R2: ten unsealed one-class 100% scores and negative cost accepted")
```

Additional executed probes, with exact mocked boundaries described in F3/F4/F6:

```text
R1 ACCEPTED: changed inherited lr=.003; pinned parent bytes unchanged: True
R2 ACCEPTED: 10 unsealed one-class 100% rows; ms/image= -1000.0
R4 CONTROLLER REACHED SCORE from bare {passed:true} gates and non-prediction inventories:
   [('score', 'coco_object'), ('score', 'pc60')]
R5 escaped logs created: ['pc60.gate.json', 'pc60.gate.log']
R6 ANALYZE ACCEPTED unrelated service: unrelated-success.service
R6 report rows= 10 native inventory match= True
R6 current prediction payload files actually present: 0
```

R6 intentionally tested inventory trust, not payload decoding: the analysis
scope correctly says it does not rehash/decode original payloads. The defect is
missing authentication/certification of that evidence, not that bounded scope.

## Checks That Held and Remaining Boundaries

- The two named datasets, population sizes, seed, five-arm order and permutation
  seed are fixed in the controller. Manifest top-level keys are exact, and every
  manifest is compared against deterministic reconstruction from prepared bytes.
  No circular prepared/manifest hash dependency was found in this construction.
  F1 is a separate external parent-authentication gap.
- Runtime exact inventory checks reject extra files, caches, bytecode and
  symlinks; inherited runtime bytes must be retained. Loaded-module scanning
  checks selected prefixes, not an entire interpreter environment. It runs
  before the late inherited-worker import (`common.py:170-177`); ancestor
  namespace search paths and later imports are not independently certified by
  this review. Final worker admission must be checked before calling this closed.
- Restart sentinel, exclusive child log creation, failure propagation and the
  two-population loop ordering are present. Existing tests confirm a gate child
  failure stops before scoring. No retry/resume path was observed. No claim is
  made about recovery after external deletion/tampering with sentinels.
- Shared GPU locking and the required MemoryMax/swap/OOMPolicy/Restart/
  RemainAfterExit envelope are not established by these controller tests. The
  suite-local `queue.lock` is not the shared GPU lock. No freeze/service exists
  yet according to the author; external launch policy plus final worker locking
  must be supplied and reviewed, not inferred from these passing tests.
- `analyze.contrasts` has the six correct signed formulas, including interaction
  and native-minus-shuffled. It retains negative contrasts and does not select a
  winner. Coverage code checks nonnegative integers and within-category
  inequalities; actual counts still require sealed stream reconciliation.
- Native hash comparison follows both full population index ranges and labels
  its inventory-only scope accurately. F1/F2/F3 prevent treating its equality
  output as authenticated native replay evidence at this stage.
- GT isolation across every prediction, native real-RGB masks/queries/C/loss/
  gradient/parameter/output equivalence, finite device-native optimization,
  authentic candidate counts, measured cost, and final worker seal verification
  remain explicitly UNREVIEWED. No scientific result or GPU readiness follows
  from this audit.

## Handoff

Actionable blockers were surfaced during the audit. An attempted prompt to main
through the app messaging tool was rejected because main is this task's native
ancestor; no cross-thread message was delivered. This document and the auditor's
final return are the authoritative handoff. Fix only demonstrated issues, retain
these identities/results, then request recheck of the changed controller/common/
analysis plus final worker/runtime. Do not interpret this initial pass as GPU
admission.

## Recheck 1: In Progress, New Worker Included

The author subsequently authorized final worker/runtime review. Initial F1-F6
above are retained as historical findings on their stated hashes, not silently
replaced. Current recheck observations are appended below. No GPU execution or
prepared-suite admission has occurred.

### F7 [P1, OPEN] Peer-Population GT Can Be Hashed Before the Prediction Barrier

Locations on worker SHA256
`3862839759d5681eb1842ab2d727b0d39408e0c9e3f632b17156d41a45a3a922`:
`tools/evolution_allocation/worker.py:570-578,587,613-616`.

`guard_output_labels` constructs an inode set only from the current manifest's
dataset. It therefore does not recognize an output hardlink to the other fixed
population's annotation. `verify_prediction_seal` proceeds to hash that file
before rejecting its digest. Detecting drift after reading annotation bytes
does not satisfy the no-GT-before-all-predictions contract.

Independent CPU reproduction R7 used the actual worker verifier and its supplied
tiny fixture. Create a sealed COCO synthetic output and a PC60 synthetic input
manifest, with no PC60 predictions and no all-prediction barrier. Replace
`full/coco_object/predictions/image_region/000000.npy` with a hardlink to a PC60
synthetic label. Trap `worker.sha256` by the label's inode. The trap fires:

```text
R7 REPRODUCED: ATTEMPTED PEER-GT CONTENT HASH BEFORE ALL-PREDICTION BARRIER
```

No real labels or GPU were opened. The trap fires before any content read of
the synthetic label by the verifier. The fixture's small-population schema and
gate mocks isolate the output isolation check; the hash/guard path is real.

Required correction: collect and stat-check all labels in BOTH fixed full
manifests, not only current-cell labels, before reading any untrusted output
artifact, seal or input/runtime alias. Cover both alias directions and gate/full
verification; assert rejection before the annotation-content hashing routine is
called. The existing `test_hardlinked_output_never_hashes_gt` covers only the
same-dataset case. Full recheck remains in progress.

## Recheck 1 Verdict and Follow-Up Identities

Bounded recheck completed on the identities below. **Code admission is withheld:
F7b [P1], F9 [P1] and F8 [P2] remain necessary corrections.** This is also not
GPU admission. No actual suite preparation, freeze, model gate, CUDA execution
or retained-service inspection was performed. The subsequent actual prepared
artifact must be reviewed separately after code issues are closed.

The original F1-F6 reproducers are closed for their originally demonstrated
paths, as detailed below. Their original text and initial inspected hashes
above remain unchanged. F7's peer-payload branch was corrected during this
recheck; the separate early barrier-seal hash branch remains reproducible.

| Reviewed File | Latest Rechecked SHA256 |
| --- | --- |
| `docs/allocation-controls-contract.md` | `6f09f3b9606363236cae123f8fb08772f675531df02a7b79e81b9da3f40ff8ed` |
| `tools/evolution_allocation/common.py` | `e6a599420e633e0f4f05257db042af5d2d6c534b89a727b20adee669da0460fd` |
| `tools/evolution_allocation/queue.py` | `6abe10f086440e240f0c9fd7f826d5041f2785eb9e9c3141d6fcd20d9de051a6` |
| `tools/evolution_allocation/analyze.py` | `dbdf910ca205d9041de33c72b97b216f1f61c82ccfb4eb40cd5d8e16202d8097` |
| `tools/evolution_allocation/worker.py` | `13aaf060293cd2d73d4d3cc73cc3a67c23497705973cab8745cb487fcaab4b25` |
| `trace_ovss/method/allocation_runtime.py` | `36c62685e0fadf56863d1c62f0c3c004369f722d392129871d41251fa29e962e` |
| `tests/test_allocation_queue.py` | `556e330d90326f4f1b0c83819593904463610e660be781ada40a61f4a517e044` |
| `tests/test_allocation_worker.py` | `e382b6f3d8763a8c87c95524cfab520be9e748a200ee6b838c1a073c4998b9c4` |
| `tests/test_allocation_runtime.py` | `aaab8562e09f91e8cce660b1512043ab18bd57582274bb33d9c98705143090a2` |
| `tests/test_allocation_controls.py` | `3b25bbf7a3ca983d968b22ecc7e247fe003784b26310d895077b42f9d66cd162` |

Earlier recheck waypoints, not replacements for that final table: common
`e910de751ddefa3ac00cf4115d9525de9bed211f0bd24c7b70ec2a4bea6b7647`;
queue `eec713863a143e90f5abeaf7ca490e3c5707efa1552528f1b3a09ce4acc3060a`
then `3796e506c0a04247ae046afdd144ea839fbe92ec7b163d86a26f0234b542229f`;
worker `3862839759d5681eb1842ab2d727b0d39408e0c9e3f632b17156d41a45a3a922`
then `5bed9b05035dc8cef478eb48066e9c8556e69074ca4d6377054c5e130a4b2785`;
queue tests `5eed379a95e903b0a5e9052dbd0785a5ebe089d8b22591fada27becef5de1e5e`;
worker tests `a20023ca5f8121500412163bd2c133665e2fcc9c6f6942eef6097d8583ee3c79`,
`791a82d898aad50036c1fbeef3c3c4a00c88727224fb6546550c0425dfd134c2`,
then `ee61a53a234c7eb3453579bd0a212980d00c5b7d9ac9d96a44e92e6815d58240`.

### Original Finding Disposition

| Finding | Recheck Disposition and Evidence |
| --- | --- |
| F1 | Original sibling-only/copy-read-set defect resolved: exact parent copy inventory, comparison to the original verified read-set and fixed original live paths, and importer before/report/after SHA comparison are present. Coherent local repack and incomplete inventory regressions pass. Importer **executed-byte** authentication is not closed by these SHA comparisons; see F9. Live fixed-parent metadata remains an explicit trust/dependency boundary. |
| F2 | Resolved for original omission/unsealed-result reproducer: exact `SCORE_FILES` membership is checked, canonical prediction artifacts are hash-verified, and the full worker score verifier is called. Sealed `prediction_result.json` supplies coverage and the stream is reconciled. |
| F3 | Original unrelated-service reproducer resolved: declared unit, command/root/hash/PID, cgroup/invocation, resources and actual successful exit code are checked; analysis checks terminal state twice. The new pre-gate controller admission also checks the actual resource envelope. Direct worker CLI execution remains separate; see F8. |
| F4 | Original bare-boolean/placeholder barrier path resolved on the normal controller path: detailed real-gate schema/provenance, six arm/native cost rows, finite thresholds and exact peer prediction/cost validation are required. Independent synthetic full controller workflow R8 completed with the actual worker validators. |
| F5 | Original wrong class dimensions/negative time reproducer resolved: exact dimensions, finite nonnegative cost, per-image/episode/pooled totals and complete measured operation streams are checked. Verified actual native reduce-zero/ignore/readout helper reuse and count reconciliation. |
| F6 | Original symlinked logs, state temporary alias and nested output reproductions resolved by canonical writable-path checks and isolated direct-child output-root checks; corresponding regressions pass. No claim of race resistance against an actor replacing files between stat/open is made. |

### F7b [P1, OPEN] Barrier Hashes an Annotation-Aliased Seal Before the New Guard

Location on latest common hash: `common.py:248-264`, especially `:261`.

The eight new peer-alias tests exercise `worker.verify_prediction_seal`, whose
guard now covers both full populations. Those checks pass. The common barrier
function still hashes `peer/PREDICTIONS_SEALED.json` BEFORE calling that guarded
worker verifier. This is independent of the guard's corrected global label set.
Controller-side preguards added to queue gate/barrier creation do not protect
the subsequent direct `common.verify_barrier` path in scoring/reverification.

R7b was rerun after the global guard fixes: create both synthetic outputs,
replace COCO's `PREDICTIONS_SEALED.json` with a hardlink to its own synthetic GT,
and construct the synthetic barrier. Trap `common.sha256` on the GT inode, then
call the real `common.verify_barrier` from PC60. The trap still fires before
`worker.guard_output_labels` can run:

```text
F7b STILL REPRODUCES: peer-seal GT hash before guard
```

The trap prevents reading even the synthetic label content. Correction must
place the global stat-only guard before the first read/hash of each peer seal
(and before reading an aliased barrier itself), rather than only guarding the
later worker call. Add a regression against the common barrier entry, trapping
`common.sha256`, not only `worker.sha256`.

### F9 [P1, OPEN] Importer Source Hashes Do Not Identify Executed Bytecode

Location: `queue.py:51-59`, particularly `spec.loader.exec_module(module)`.

Python's source loader can execute an existing valid timestamp `.pyc`. `-B`
prevents writing bytecode, not reading it. This importer lives outside the exact
runtime inventory, and no check rejects its cache. Its reported importer SHA is
computed from `__file__` source bytes, so before/report/after can all match even
when different cached code actually ran. A same-size edit within timestamp
granularity, or a retained matching timestamp, is sufficient for this condition.

R11 is independent of actual parent evidence or GPU and passed against the
current loader. This temporary-file reproducer explicitly creates a timestamp
cache, then changes the same-length source while retaining its timestamp:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import py_compile
from tools.evolution_allocation import queue as q

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    p = root / "paper_evolution/tools/export_tuned_full.py"
    p.parent.mkdir(parents=True)
    source = ('from pathlib import Path\nimport hashlib\ndef import_suite(root):\n'
              '    return dict(marker="STALE", importer_sha256='
              'hashlib.sha256(Path(__file__).read_bytes()).hexdigest())\n')
    p.write_text(source)
    status = p.stat()
    py_compile.compile(str(p), doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    p.write_text(source.replace("STALE", "FRESH"))
    os.utime(p, ns=(status.st_atime_ns, status.st_mtime_ns))
    report = q.parent_report(root)
    assert report["marker"] == "STALE"
    print("R11: source=FRESH; executed=STALE; all three source SHA checks passed")
```

Correction: execute the exact source bytes whose digest was captured, or use a
loader that demonstrably does not consult cached bytecode and authenticate the
executed bytes. Do not rely on `PYTHONDONTWRITEBYTECODE`/`-B` as a read-cache
control. Retain a stale-cache negative/positive regression. No actual current
parent importer cache or parent evidence was altered during this audit.

### F8 [P2, OPEN] Direct GPU Worker Entrypoints Bypass the Resource Envelope

Locations: latest `common.py:225-245`; `worker.py:532-546,908-926`.

The controller's `execution_identity` now checks actual service resources before
the first gate, which fixes that issue on the supported queue path. However,
`worker --phase gate` and `worker --phase predict` are executable entrypoints and
their admission checks do not require the current systemd invocation/cgroup or
the controller's execution receipt. Direct gate invocation can proceed from a
valid frozen suite to GPU lock acquisition/model construction outside the
prepared 12 GiB/1 GiB/OOM-stop/no-restart service. Prediction similarly checks
prior gates, not its own actual execution envelope.

R10 used the real common manifest/prepared/parent/inode admission logic with the
synthetic fixture and no `INVOCATION_ID` or `EXECUTION_STARTED.json`; it accepted
the gate. The fixture-only import-origin scan and checkpoint identity were
isolated because the fixture is not a real frozen interpreter/checkpoint. This
is NOT a complete real-runtime launch or a claim that source verification can
be bypassed. Inspection of the actual subsequent gate/predict call path shows
no later service check before GPU work. No GPU entrypoint was executed.

Correction: for gate/predict, require the prepared controller invocation and
actual current cgroup/resources before acquiring the shared lock or configuring
CUDA/building a model, or reject standalone GPU worker use explicitly. CPU-only
offline verification can have a separately bounded policy. Test direct gate and
predict calls outside the retained unit and under a different invocation.

### Independent Recheck Execution

- Twice reran the first joined snapshot: `537 passed in 4.14s` on 147 oracle,
  260 runtime, 85 worker and 45 controller cases.
- During F7 edits, one intermediate run had `8 failed, 537 passed in 4.65s`.
  The alias assertions succeeded, but accidentally nested legacy RGB assertions
  then failed on a missing synthetic checkpoint. This test-only issue was fixed
  by main; its intermediate result is retained here rather than discarded.
- Latest identified snapshot: **`545 passed in 4.25s`**, same CPU-only command
  as above plus `tests/test_allocation_worker.py`, `tests/test_allocation_runtime.py`
  and `tests/test_allocation_controls.py`. No cache writes or CUDA initialization
  were requested. No background test session remains running.
- R8 independently executed actual `queue.run` scheduling and actual worker
  gate/prediction/cost/score/barrier verification on a tiny synthetic suite.
  Only preparation admission, live service admission and model-side child
  execution were replaced. Gate receipts were synthetic, not real-model
  equivalence evidence. Actual synthetic-label scoring and stream reconciliation
  ran on CPU. Observed phases were gate(COCO,PC60), predict(COCO,PC60),
  score(COCO,PC60), verify(COCO,PC60), ending with `full_outcomes=10`.
- R9 independently exercised the new real `execution_identity` logic with fake
  `/proc`/systemctl data: valid resource identity accepted; ten altered resource,
  root, invocation, cgroup, PID or command fields rejected before gate dispatch.
- R7/R7b, R10 and R11 are bounded adversarial probes, not scientific runs.
  Passing 545 authored tests does not supersede the independent failing probes.

### Runtime and Worker Review Boundaries

The device-native objective reuses native image/episode primitives, keeps loss
and entropy on their original device, and transfers only small detached
allocation receipts at preparation. Activity uses original host q before area
or permutation. Cache ownership/version checks and local engine preparation
prevent ordinary cross-episode reuse on the inspected call path. CPU tests
cover fp16/bfloat16/fp32/fp64, native EPS/empty behavior, shuffle fixed points,
source stability, gradients and owned metadata. CUDA behavior remains unrun.

The native image-region worker hook delegates directly to inherited C. Gate
code restores identical model/optimizer and RNG state across native plus five
arms, checks text embeddings and exact candidate/diagnostic identities, records
step gradients/parameters, and enforces 1e-5 differences and zero label changes.
Other arms require actual finite updates. Raw-output/optimizer observation hooks
run before native sanitization, with no OOM fallback or diagnostic batch retry.
The worker uses the inherited single shared GPU lock in gate and prediction.
No real-model threshold has been tested or loosened by this auditor.

Inference retains the inherited image-only decode, geometry, class-extension
collapse and final readout. Scoring uses nearest-neighbor label resize,
reduce-zero-label/ignore semantics and the inherited pooled-count implementation.
All exact per-image/episode counts, class dimensions, cost schedules, original
diagnostic equality and actual permutation-coverage receipts are reconciled.
The signed analysis preserves all ten rows and six contrasts; no winner,
bootstrap, significance or untouched-test claim is introduced.

Remaining launch requirements after code corrections: actual immutable prepared
artifact/input/source/parent identity review, independently observed real RGB
native/finite-arm gates, enforced real resources/shared lock, all-population
prediction sealing before scoring, and authentic retained-service termination.
No result or scientific success is certified by the CPU audit.

### Final Small-Tightening Recheck

After main's final metadata tightenings, independently reran the joined CPU
suite: **545 passed in 4.25s**. Inspected the additional pre-inventory global
stat guard in `verify_prepared`, explicit gate config equality, and identical
original candidate-diagnostic SHA requirement across all six native/arm rows.
These tightenings do not affect the three remaining findings.

Latest identities superseding only the corresponding rows of the previous
recheck table (all initial history remains valid):

| File | Final Rechecked SHA256 |
| --- | --- |
| `tools/evolution_allocation/common.py` | `bb1c6558643aaf5920b1a98ad5f13d78ece8a960ed48e25409470d99592390d8` |
| `tools/evolution_allocation/queue.py` | `a643a0fa8b5e7c0492d2fb50a7e90fa35b00c3567e3537948f14f8e7351c68f7` |
| `tests/test_allocation_queue.py` | `160f5fb808370d06447c103b90dd01518d71fa2fa4e924503c13ead5d8c20d19` |

Unchanged final worker SHA is
`13aaf060293cd2d73d4d3cc73cc3a67c23497705973cab8745cb487fcaab4b25`;
worker tests SHA is
`e382b6f3d8763a8c87c95524cfab520be9e748a200ee6b838c1a073c4998b9c4`.
Runtime, runtime tests, oracle tests, analysis and contract hashes are unchanged
from the preceding recheck table.

Both necessary P1 reproducers were rerun against this final snapshot:

```text
F7b still opens label-hash path before guard
F9 still executes STALE bytecode while source=FRESH; all SHA comparisons pass
```

F7b's final location is `common.py:266`: the peer seal is hashed before the
guarded `verify_prediction_seal` call at `:269`. F9 remains at
`queue.py:54-59`. F8's direct-worker call paths are unchanged. Thus the bounded
verdict is still **NOT ADMITTED: F7b P1, F9 P1, F8 P2**. Closing these requires
only targeted corrections/regressions to existing paths, not new artifact
formats or broader experiments. No source freeze, prepared artifact or GPU
execution is approved by this report.

## Recheck 2: Exact Open Paths Closed

2026-10-29T11:43:36Z. Bounded follow-up on F7b/F9/F8 and their targeted changes,
not a restart of the review. **No remaining necessary P1/P2 code blocker was
identified in this snapshot.** Code review is complete for these identities;
the next admissible stage is CPU preparation/freeze followed by independent
review of the actual prepared artifact. This statement does not approve GPU
execution, real-model gates, a particular prepared suite, or scientific claims.

All historical findings, failed probes, intermediate test failures and prior
withheld verdicts above are retained. This section supersedes their OPEN status
only for the exact corrected identities below.

### Closure Evidence

| Finding | Final Status | Independently Checked Correction |
| --- | --- | --- |
| F7 / F7b | CLOSED for demonstrated alias paths | Both full label populations are stat-indexed. `common.verify_barrier` checks the barrier inode before JSON read and calls the peer output guard before the peer seal hash. Queue gate/barrier/result reads and worker sealing are preguarded. Both cross-dataset directions and protocol/seal/prediction/STARTED paths are covered by the added tests. |
| F9 | CLOSED | `parent_report` captures source bytes, hashes those bytes, and executes `compile(contents, path, 'exec')` in a `ModuleType`; it no longer consults SourceFileLoader's bytecode cache. The original matching-timestamp STALE `.pyc` probe now executes FRESH source. |
| F8 | CLOSED | Both GPU entrypoints call `common.require_gpu_envelope` before STARTED, GPU locking or model work. It requires the recorded controller, matching PPID/invocation/cgroup/cwd, live declared unit, exact controller command/PID and actual resource limits. Direct entry without the controller receipt rejects before output creation. |

F1-F6 retain the resolved dispositions from Recheck 1. No new artifact format or
implementation change was made by the auditor.

### Final Code Identities

| File | SHA256 |
| --- | --- |
| `tools/evolution_allocation/common.py` | `6dcf8b7486765e64d1d472da56c3928336e9cf939ff3ce43caff1296ae3b0c66` |
| `tools/evolution_allocation/queue.py` | `95a6a4c78229d790cdc3fce22acc66f69fcc7349312269d03566595cbb4143f1` |
| `tools/evolution_allocation/analyze.py` | `dbdf910ca205d9041de33c72b97b216f1f61c82ccfb4eb40cd5d8e16202d8097` |
| `tools/evolution_allocation/worker.py` | `bc5bdd7590fad88cbb33023911e6b15be58869d5260cecf23f51499689294a64` |
| `trace_ovss/method/allocation_runtime.py` | `36c62685e0fadf56863d1c62f0c3c004369f722d392129871d41251fa29e962e` |
| `tests/test_allocation_queue.py` | `6bd5a912245da66eebd6c3670808ae5fe09fd8b0e79bd0d774f3d337c5c0a613` |
| `tests/test_allocation_worker.py` | `10191eea0069d910ce9159322b2413ae5424dde79d58a8e2ebf69ac06101da16` |
| `tests/test_allocation_runtime.py` | `aaab8562e09f91e8cce660b1512043ab18bd57582274bb33d9c98705143090a2` |
| `tests/test_allocation_controls.py` | `3b25bbf7a3ca983d968b22ecc7e247fe003784b26310d895077b42f9d66cd162` |
| `docs/allocation-controls-contract.md` | `6f09f3b9606363236cae123f8fb08772f675531df02a7b79e81b9da3f40ff8ed` |

### Final CPU Verification

Reran the joined four-file CPU command, bytecode/cache writes disabled and CUDA
hidden: **550 passed in 4.27s**. Independently reran the original exact P1 probes
and a positive/adversarial child-envelope probe, rather than relying solely on
new author tests:

```text
R7b CLOSED: peer-seal GT rejected before hash
R11 CLOSED: FRESH source executed despite matching STALE .pyc
R12 CLOSED: valid child envelope accepted; 14 altered unit fields and wrong PPID rejected
```

R12 uses synthetic `/proc`, process-parent, invocation and systemctl responses,
plus the real new envelope validator. It accepts a matching running unit and
rejects alterations of MemoryMax, MemorySwapMax, OOMPolicy, Restart,
RemainAfterExit, Id, ActiveState, SubState, WorkingDirectory, InvocationID,
ControlGroup, MainPID, ExecMainPID and ExecStart; wrong PPID also rejects. No live
unit query, GPU initialization or model construction occurred. The authored
direct-gate and direct-predict tests also pass. No test process remains running.

### Prepared Artifact Boundary

A read-only depth-one inventory of `/home/wwww/hzl/results` found no
`trace_allocation_controls_*` directory at this recheck. Therefore **no actual
prepared artifact has been audited or approved**. Do not substitute the code
hash table or synthetic fixtures for the forthcoming actual suite identity.
The artifact audit needs the canonical new suite path and its independently
reported `SUITE_PREPARED.json` SHA256, then must verify the actual frozen source,
four manifests, exact two full panels, inherited selection/checkpoint/protocol,
parent proof/copy read-set, new unit name/resources, no prior execution attempt,
and absence of stray caches/symlinks/unbound payloads. The auditor will not
prepare or launch it, because this audit's only repository write scope remains
this document.

Remaining boundaries are unchanged: CUDA and real-RGB/native/finite-arm gates
have not been executed by this auditor; actual runtime resource/service
identity and terminal success require live evidence at their respective
stages. No GPU admission and no scientific result is conveyed by this code
closure.

## Actual Prepared Artifact Audit: 20261029_1946

Completed 2026-10-29T11:52:31Z, read-only. This section supersedes only the prior
"no actual prepared artifact audited" state. All code-review history remains.

**ADMITTED for main to start the exact new retained service/controller below and
its mandatory GT-free real-model gates.** No necessary P1/P2 artifact blocker
was found. This admission is conditional on the exact frozen identity and live
unit/resource/argv envelope. It does not permit bypassing gates, loosening their
thresholds, retry/resume, changes to old suites, or calling an unbound GPU
worker. The reviewed controller may advance only when both original strict
gates pass. This audit does not claim CUDA/model gates have passed, certify any
new score, or guarantee completion/success of future full work.

### Exact Artifact Identity

- Suite: `/home/wwww/hzl/results/trace_allocation_controls_20261029_1946`.
- `SUITE_PREPARED.json` SHA256:
  `6f6cba9c41010586041ad03c34f4d174b00ec43efad8922db9b903c243534155`.
- `PARENT_IMPORT.json` SHA256:
  `ea482f978b77919804840ef9088842ed19ac38a6b4fc754e6ac7414f2bafc7b2`.
- Canonical runtime-inventory mapping SHA256:
  `b43fce08b00126172ec25b5c42d4261e049d7039be0230ab3dadefb14b8d2604`.
- Canonical 306-entry importer read-set mapping SHA256:
  `cabaf61a72c994f6acce957276bcab62d8a217efcfbbeb62f3c0c962c33263a1`.
- `queue_state.json` prelaunch SHA256:
  `ccdcc70df5feba65cb153012dc9563778c4a7d2b275e475f9803e1cf28b58637`.

Mapping digests use the existing canonical JSON convention: sorted keys,
compact separators, no NaN. Each mapped file was independently hashed too;
these mapping digests are not substitutes for those checks.

| Frozen Manifest | SHA256 |
| --- | --- |
| `manifests/coco_object.gate.json` | `185e1004d3b0c2664730d972eaaef8a5cebc9ae37123a6791f1934315384a476` |
| `manifests/coco_object.full.json` | `392fabe9a7bc7e077ce91e94b14f33067c5bc3a24533423fd17a2ca67ec6f6c5` |
| `manifests/pc60.gate.json` | `23dff14865c4715f328a7ba9ed06c5d41544b1210951420236e3a60cde3a55a7` |
| `manifests/pc60.full.json` | `09b524e51cb8e34a1baf7af30ae0605c0c8c988ab63e7ac75b53d327d8542f32` |

### Independent Inventory and Source Checks

The whole new artifact has exactly **198 files and 25 directories**, matching
the prepared metadata, four manifests, sixteen copied/derived parent files and
174 runtime files. There were no additional files/directories, symlinks,
hardlinks, bytecode, `__pycache__`, `.pytest_cache`, or aliases to the stat-indexed
input/annotation/checkpoint files. No gates/full/logs directories, queue lock,
EXECUTION_STARTED, worker attempt claims, gate/prediction/score barriers,
SUITE_SEALED or metrics file existed. The state was `prepared`, four cells,
ten PLANNED arm outcomes, `gpu_work_started=false`; none of those ten outcomes
has been executed or scored.

All **174** frozen runtime-file SHA256s matched the prepared inventory and the
current worktree. All **161** parent runtime files are present byte-identically;
the remaining thirteen files are additive. No native inherited file changed.
The additive set is the five allocation package files, the two allocation
method modules, four allocation test files, and the already inventoried
`tests/test_tuned_full_figures.py` / `tools/draw_tuned_full_figures.py`. The latter
were inventoried, not reviewed as paper content or run by this audit.

Core implementation hashes exactly match Recheck 2, including common
`6dcf8b7486765e64d1d472da56c3928336e9cf939ff3ce43caff1296ae3b0c66`,
queue `95a6a4c78229d790cdc3fce22acc66f69fcc7349312269d03566595cbb4143f1`,
worker `bc5bdd7590fad88cbb33023911e6b15be58869d5260cecf23f51499689294a64`
and runtime `36c62685e0fadf56863d1c62f0c3c004369f722d392129871d41251fa29e962e`.
The only subsequent test change is five same-controller unit-envelope cases:
`tests/test_allocation_queue.py` now hashes to
`1ea3f457891bd4cf2a7b2b48d569e60f67359b47bb16b4da1e61baa5559528ab`.
Their matching live-unit positive case and invocation/PPID/cgroup/memory
negative cases were inspected and included in the frozen CPU rerun.

The frozen contract matches the current worktree contract at
`1fc99cb05ce79471dc650073ffa70ea43ef3800a1ab7844aba4594716a045baf`.
Its text before the device-native status addendum, with the original terminal
newline, still hashes to the originally audited scientific protocol
`6f09f3b9606363236cae123f8fb08772f675531df02a7b79e81b9da3f40ff8ed`.
The added implementation-status text does not change the scientific protocol.

### Parent, Panel and Selection Continuity

Independently rehashed all **306** declared importer metadata/source entries;
every hash matched. Their allowed types were JSON, episode-score/cost JSONL,
CSV, Python, Markdown/text and the source-vendored tokenizer vocabulary gzip.
No prediction array, real RGB/GT or checkpoint payload was read. The importer
source identity is still
`58dc31d65b7651544e0623dca242dd7ae7365b5ec622b42103bdf185fb44c559`.
Its proof records eight parent populations and 32 parent scores; this audit
rechecked their metadata read-set, not their raw prediction/label payloads or a
fresh scientific re-evaluation.

All sixteen local parent inventory entries matched; every copied original was
also bound to its fixed parent path/read-set hash. The fixed parent prepared
identity remains
`4f3e54226f99230af48b2105372300d06f31651515324b3ab99dbe138ff3cfa3`.
Independently checked parent terminal links to ALL_CONFIGS_SELECTED and the
all-full-prediction barrier; both selected datasets' selection, protocol,
prediction-seal, score-seal and result links matched. Calibration-only selection
and `full_evaluation_used_for_selection=false` are retained. Checkpoint metadata
equals the original parent exactly.

Each new full panel equals BOTH the original parent's sealed full protocol
panel and its hash-bound `inputs.json` evaluation panel, including every input
row, order, split, vocabulary, query-to-class mapping and readout convention.
Each gate contains the exact first four full rows/IDs with `images=4` and
`full_split=false`. All four manifests passed exact reconstruction checks and
carry this prepared hash, runtime inventory, parent-proof hash and inherited
checkpoint/configuration. Model seed0, shuffle seed20261029, all five arms,
no-search/no-retry and development exposure remain fixed.

| Dataset | Full Images | Classes / Queries | B | lr | Steps | K |
| --- | --- | --- | --- | --- | --- | --- |
| COCO Object | 5000 | 81 / 112 | 13 | 0.00375 | 30 | 35 |
| PC60 | 5105 | 60 / 60 | 13 | 0.00375 | 7 | 20 |

Both use original `reduce_zero_label=false`, class extensions enabled,
view batch16, resize/patch224x224 and stride112. All ordered IDs, RGB paths and
label paths are unique within each population. All stored label size/mtime
metadata matched stat-only checks.

| Identity | COCO Object | PC60 |
| --- | --- | --- |
| Panel file SHA256 | `1761847a239c88c0cbbfb7d721cce9956bf7220dedc95170f725f24add6be062` | `92ec7f659c9e8c676792ce45fe6698a4a4cd6fa5626580f489680a44a8f7e01a` |
| Ordered-ID canonical SHA256 | `d7a6cf336dba697de84529c51615efc32a6b5bfb586c439c315441a176222a88` | `a2104bacd288f646547e37732633521919aab6dc887a3506d78e49569b801f94` |
| Vocabulary SHA256 | `f2afc2d64c9a945e53f052b9cd52400c5e2f9b62b8554a920daf1f7481a82fc1` | `54731e934f0d36402c8cb4d4a8e184c1e367c858fc25a5d8c7b0e84e4fc33b78` |
| Config canonical SHA256 | `86a400ffe4941ccf29e8b0d03d6d6c7eaf0a1731c42ca5471b6168c97123eeb1` | `ed812fb988e8e3e5bb76502c72113da46b81c5e83f65480a3b8bd2bbbfa156a0` |

Checkpoint path is `/home/wwww/.cache/clip/ViT-L-14.pt`, recorded SHA256
`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`.
Only its metadata and stat were checked: 932768134 bytes and recorded mtime
match. **Checkpoint contents were not opened or independently rehashed here.**
Actual worker admission retains its required checkpoint-content identity check.

### Frozen Runtime Verification and CPU Tests

Ran actual frozen `common.verify_prepared(..., frozen=True)` from this artifact's
runtime using its bound interpreter. Both common/queue module paths resolved
inside the frozen runtime. A Python open audit hook blocked every actual RGB,
GT and checkpoint path and any artifact write; verification passed with zero
forbidden payload opens. Torch was not imported by that metadata verification.

Separately ran only the four new allocation CPU test files FROM the frozen
runtime, with CUDA hidden, `-B`, `PYTHONDONTWRITEBYTECODE=1` and pytest cache
disabled: **555 passed in 4.31s**. Synthetic test payloads were confined to the
test temporary directories; no actual dataset/checkpoint/prediction payload was
used. No GPU/model job was launched. Final independent inventory/source hashing
and sentinel checks confirmed the artifact was unchanged after these commands;
prepared SHA remained the admitted value. No test process remains running.

### Bound Service and Launch Command

Exact new service: `trace-allocation-controls-20261029-1946.service`.
Read-only systemctl snapshots before and after this audit showed
`LoadState=not-found`, `ActiveState=inactive`, `SubState=dead`, `MainPID=0`.
It has NOT been started. No live installed resource envelope is claimed:
systemctl's not-found defaults are not the prepared resource specification.

The declared transient/retained unit MUST install these values before any gate:

| Property | Bound Value |
| --- | --- |
| WorkingDirectory | `/home/wwww/hzl/results/trace_allocation_controls_20261029_1946/runtime` |
| MemoryMax | `12884901888` (12 GiB) |
| MemorySwapMax | `1073741824` (1 GiB) |
| OOMPolicy | `stop` |
| Restart | `no` |
| RemainAfterExit | `yes` |

The exact controller argv verified against the frozen helper is below. This
block is a reviewed command specification, **not an executed launch**:

```bash
/home/wwww/miniconda3/envs/mlmp/bin/python3.10 -B -m tools.evolution_allocation.queue run \
  --suite /home/wwww/hzl/results/trace_allocation_controls_20261029_1946 \
  --suite-sha256 6f6cba9c41010586041ad03c34f4d174b00ec43efad8922db9b903c243534155
```

Use that resolved `python3.10` argv, not a differently spelled interpreter
alias. Run it as the declared unit's actual main process from the bound cwd,
without a shell/wrapper replacing the bound controller argv/PID. Supply the
frozen runtime import root and bytecode-disabled/thread environment as reviewed;
do not fabricate INVOCATION_ID. The controller and each GPU child will
independently check actual live unit resources, PID/root/command, invocation and
cgroup BEFORE model work. Retain the unit after exit; do not collect/remove its
terminal evidence. Any gate, finite, source, resource or identity failure must
stop the new attempt with no retry, fallback, threshold change or old-suite
restart.

**Final scope:** main may launch this exact gated service. Actual model/CUDA
behavior, native equivalence/finite-arm gate outcomes, later ten-arm prediction
seals, scores, cost evidence and true successful service terminal remain future
evidence, not facts established by this prepared-artifact audit. The auditor
modified only this document, never the artifact or implementation.

## Admission Suspension: Retained Terminal Cgroup Fix

The main agent independently suspended and superseded the 1946 admission after
deployment inspection showed that a real successfully exited retained systemd
unit can expose an empty `ControlGroup` while retaining its main execution PID
and InvocationID. This was NOT a new human/user pause request. The research goal
and automation remain active as reported by main; this auditor changed neither.
**All earlier launch
permission for `trace_allocation_controls_20261029_1946` is suspended and must
not be used. No GPU or 1946 launch is approved.** Its previous structural/hash
checks remain historical evidence only. The artifact is to remain immutable;
the terminal-verifier correction requires a distinct new CPU-prepared identity.

The original terminal check incorrectly required the still-live startup cgroup
to remain present after normal exit. This is an engineering terminal-acceptance
defect, not evidence of changed algorithms/numerics or scientific success.
The scoped correction is under review: preserve all strict live controller/
worker checks; allow retired empty terminal cgroup only with bound startup
proof, exact unit/root/argv/hash/PID/invocation/resources and true successful
exit, and seal the execution receipt. No old unit or artifact was modified.

## Terminal-Only Fix Clearance

Completed 2026-10-29T11:56:43Z. **The scoped terminal correction is code-cleared
for a fresh CPU preparation/freeze and subsequent artifact review. No new
necessary P1/P2 blocker was found in this change.** The superseded 1946 artifact
is NOT admitted for launch, and no GPU execution is approved by this section.
This is the same auditor continuing; a new actual prepared identity is still
required before any new launch decision.

### Scope and Inspected Identities

Compared current queue/analyze source directly against the immutable 1946
snapshot. Changes are confined to checking the authentic startup resource/
cgroup receipt at terminal verification, allowing a retired empty terminal
cgroup, writing the execution-receipt hash into the final suite seal, and
checking that hash in analysis. No algorithm/numerical/input/configuration/
parameter/native-gate threshold or live-admission helper changed.

| File | Cleared SHA256 |
| --- | --- |
| `tools/evolution_allocation/queue.py` | `3c3abf93efe724498303274f4a0987f1cfec111e81c988c37d4772654472b9c2` |
| `tools/evolution_allocation/analyze.py` | `3ad1c4de3faaf2d0f42f0b75c0dc0347f736aa7272227b0111ea8ba41ed7d478` |
| `tests/test_allocation_queue.py` | `485082fd6097e518ec71d4b88d439256b3e9e18cde6965db3c5c6ccd2a182598` |

Unchanged critical implementation identities: common
`6dcf8b7486765e64d1d472da56c3928336e9cf939ff3ce43caff1296ae3b0c66`,
worker `bc5bdd7590fad88cbb33023911e6b15be58869d5260cecf23f51499689294a64`,
runtime `36c62685e0fadf56863d1c62f0c3c004369f722d392129871d41251fa29e962e`.

### Engineering Defect and Closure

The previous `bound_service_terminal` would falsely reject a normal retained
successful exit when systemd retires the current cgroup. Main supplied the
real observation from the prior successful B unit: active/exited, retained
ExecMainPID3798232 and InvocationIDbe31778ba0b146c6a08368b6c3a851a5, but current
ControlGroup empty. This observation was supplied by main, not independently
queried by this auditor; the prior unit was not altered or restarted.

The corrected check accepts an empty CURRENT cgroup only after all required
successful-terminal properties hold: loaded, active/exited, MainPID0,
Resultsuccess, ExecMainCode1 and ExecMainStatus0, with execution start/exit
timestamps and the same declared unit/argv/root/prepared hash/ExecMainPID/
InvocationID/resources. Startup proof must still contain its matching nonempty
service cgroup, actual resource envelope, PID, invocation, cwd and argv. A
different nonempty current cgroup still rejects. Live controller and GPU-child
checks remain strict and nonempty; this exception is terminal-only.

Final `SUITE_SEALED.json` now includes `execution_started_sha256`; analysis
compares that hash with the actual receipt before accessing canonical results.
The final service check is still repeated after interpretation.

### Independent CPU Evidence

- Focused controller tests: **54 passed in 1.56s**.
- Joined allocation-only CPU tests: **556 passed in 4.28s**, with CUDA hidden,
  bytecode writes disabled and pytest cache disabled.
- Independent R13: matching nonempty and retired-empty terminal cgroups both
  accepted; **16** altered terminal variants rejected, including wrong nonempty
  cgroup, PID/invocation/root/argv, running/inactive/failure/exit states and all
  five resource properties.
- Independent R13: **6** forged startup-proof variants rejected with current
  terminal cgroup empty: startup cgroup, invocation, execution PID, cwd, argv
  and MemoryMax.
- Independent R14: actual `analyze` prefix rejected an execution-receipt seal
  hash mismatch after legitimate synthetic empty-cgroup terminal verification
  and BEFORE canonical result access. This was not a full-score/model test.

All probes used temporary fixtures and mocked systemctl responses, not GPU or
model jobs. No actual RGB/GT/checkpoint/prediction payload was read. No test
process remains running.

### 1946 Remains Superseded

Rechecked its prepared SHA, every frozen runtime hash and absence of execution,
gate/full/log/lock paths. Its original prepared identity is unchanged:
`6f6cba9c41010586041ad03c34f4d174b00ec43efad8922db9b903c243534155`.
It remains unexecuted and immutable, but contains the old terminal verifier.
**Do not launch it or modify it in place.** Only a new CPU-frozen suite carrying
the corrected identities may be submitted for the next artifact audit. Goal
and automation activity are not paused by this supersession or this audit.

## Corrected Actual Artifact Admission: 20261029_1958

Read-only bounded artifact recheck completed 2026-10-29T11:59:46Z against the
latest terminal-fix code clearance. **1958 is ADMITTED for main to start its
exact retained gated controller and mandatory real-model gates.** No necessary
P1/P2 artifact mismatch was found. This does not assert CUDA/model gates have
passed, authorize bypass/threshold relaxation/retry, or certify future full
predictions/scores/service success. The reviewed controller must satisfy its
actual live envelope and both strict gates before advancing. **1946 remains
superseded and NOT ADMITTED.**

### Identity and Bounded Verification

- Suite: `/home/wwww/hzl/results/trace_allocation_controls_20261029_1958`.
- Prepared SHA256:
  `669402e4022ce4f7baaa4d445160f3031b085217d8c712db85f095b452e790b2`.
- Runtime-inventory canonical SHA256:
  `ca7f68d51bd50646840139e73d04db312487edee801fe87d99258c09542a50cd`.
- Parent-proof file SHA256 is unchanged:
  `ea482f978b77919804840ef9088842ed19ac38a6b4fc754e6ac7414f2bafc7b2`.

Verified exact whole-artifact inventory: **198 files, 25 directories, 174
runtime files, four manifests**. No stray file/directory, cache/bytecode,
symlink/hardlink, execution sentinel, queue lock, log/gate/full output, prediction
barrier, final seal or metrics existed. State remains prepared with
`gpu_work_started=false`. The new service was independently queried read-only:
`LoadState=not-found`, `ActiveState=inactive`, `SubState=dead`, `MainPID=0`.
It has not been started; actual installed resources are not claimed yet.

All frozen runtime hashes match the latest cleared worktree. Compared with the
immutable 1946 snapshot, the changed runtime set is EXACTLY these three files:

| File | Reviewed Frozen SHA256 |
| --- | --- |
| `tools/evolution_allocation/queue.py` | `3c3abf93efe724498303274f4a0987f1cfec111e81c988c37d4772654472b9c2` |
| `tools/evolution_allocation/analyze.py` | `3ad1c4de3faaf2d0f42f0b75c0dc0347f736aa7272227b0111ea8ba41ed7d478` |
| `tests/test_allocation_queue.py` | `485082fd6097e518ec71d4b88d439256b3e9e18cde6965db3c5c6ccd2a182598` |

All **161** inherited parent runtime files remain byte-identical. All **306**
parent importer metadata/source read-set entries were independently rehashed
and matched again. Local parent copies and original fixed-path links passed the
actual frozen verifier. Contract, parent proof/inventory, checkpoint metadata,
selected configs, seeds, exposure, no-search/no-retry scope and all five arms
match 1946 exactly. No algorithm/parameter/native-file change was introduced.

Both full panels exactly equal their original sealed parent panels AND the
previously audited 1946 panels, including all ordered input rows, vocabulary,
query-to-class mapping and split/readout conventions. Metadata identities and
stat-only label/checkpoint checks pass. COCO Object remains5000, 81 classes/
112 queries, B13/lr0.00375/steps30/K35; PC60 remains5105, 60 classes/60 queries,
B13/lr0.00375/steps7/K20. Both use224x224, stride112, viewbatch16, extensions on,
reduce-zero false. Gate panels are the exact first four full rows. Ten outcomes
are PLANNED, not completed results.

| Manifest | SHA256 |
| --- | --- |
| `coco_object.gate.json` | `faead0c8b3a9faa70e0b9f36b9bc46398a9f88cb663f4fd4fbd9e7d5d0bd4bf0` |
| `coco_object.full.json` | `88b3cdc7f908b5fd2b6f7a8e053a90d0d38772f04fa6ceb7a5df46e599666f43` |
| `pc60.gate.json` | `46b28e251a1c586dbc310c6f19afea3e570c59bc32e52535f96c2a04fe169716` |
| `pc60.full.json` | `4868c46dd876d791d23dc1ff56fe3e6ff9f3517bce35a46785d3121bc6eb6250` |

All four manifests matched exact reconstruction with this prepared identity,
suite/runtime/output roots and inherited config/input/parent/checkpoint bindings.
Ran the actual frozen `common.verify_prepared(..., frozen=True)` from 1958's
runtime; common/queue imported from that frozen root, with no Torch import.
A Python open audit hook rejected actual RGB/GT/checkpoint paths and their
inode aliases and artifact writes: **zero actual payload opens, zero artifact
writes**. No model, GPU, prediction payload or actual label content was read.
The latest code's independent556-test result is recorded in the terminal-fix
clearance; this bounded artifact check did not repeat the full code audit.

### Exact Launch Binding

Service: `trace-allocation-controls-20261029-1958.service`.
WorkingDirectory/import root:
`/home/wwww/hzl/results/trace_allocation_controls_20261029_1958/runtime`.
Bound resources: MemoryMax12884901888 (12 GiB), MemorySwapMax1073741824 (1 GiB),
OOMPolicystop, Restartno, RemainAfterExityes. These must be actually installed
on the new retained unit; controller and GPU children verify them before model
work. Preserve the reviewed frozen-root/bytecode/thread environment and real
systemd invocation; no shell/PID wrapper or alternate interpreter spelling.

Reviewed exact controller argv, NOT executed by this auditor:

```bash
/home/wwww/miniconda3/envs/mlmp/bin/python3.10 -B -m tools.evolution_allocation.queue run \
  --suite /home/wwww/hzl/results/trace_allocation_controls_20261029_1958 \
  --suite-sha256 669402e4022ce4f7baaa4d445160f3031b085217d8c712db85f095b452e790b2
```

Final prepared hash still matched after all checks. Old1946 retains its original
prepared hash/source inventory and no attempt paths; it must remain immutable
and unlaunched. Only this audit document was modified. This admission changes
neither research goal nor automation state and is not evidence of a successful
GPU/model gate or any future scientific result.
