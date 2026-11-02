# Single-Probe Tuning: Independent CPU Code Audit

## Closure Recheck: 2026-10-29

**F2-R: CLOSED. F5: CLOSED.** No remaining demonstrated P1/P2 from these
two findings in the rechecked source versions. This was a bounded regression
recheck of F2-R/F5 only, not a renewed broad audit, launch authorization,
real-data validation or manuscript assessment. Earlier reports below are
archived evidence and their open statuses are superseded by this disposition.

### F2-R Closure Evidence

`tools/evolution_tuning/queue.py:137-143` now traverses the entire runtime,
rejects symlinks, cache directories and `.pyc`/`.pyo` files, hashes all remaining
files including runtime-root entries, and requires exact equality with the
frozen inventory. `child():170-192` uses this verifier before subprocess
startup and after exit. The previous root `sitecustomize.py` and ignored-cache
paths no longer pass this verification; rejection does not depend on whether
cached bytes are a valid Python code object.

The shadow-package test plus all three startup/cache parameterizations pass:
root `sitecustomize.py`, root `__pycache__/worker.cpython-311.pyc`, and
managed-tree `tools/worker.pyc`. Symlink rejection was checked in the explicit
whole-runtime guard; no additional symlink fixture was run in this recheck.

### F5 Closure Evidence

- `tools/evolution_tuning/worker.py:477-493` verifies inputs and the prediction
  barrier before consulting the calibration label reference. Existing
  references require the manifest's exact path/hash, matching dataset/panel
  hash, and complete label-path coverage; a pinned-but-missing reference fails.
- `worker.py:508-513` hashes each label only in score and compares with the
  pinned reference before prediction loading, label decoding or count updates
  for that image. Content mismatch aborts without a scored seal.
- `worker.py:542-554` creates the first reference exclusively only after the
  authorized scoring pass and final admission/input checks. It records panel
  hash, all label hashes, and first manifest/prediction/barrier identities.
  Later scores recheck the reference hash before writing their results.
- `tools/evolution_tuning/queue.py:161-163` pins an existing dataset label
  reference into subsequent calibration manifests. `queue.py:221-226` checks
  the result's reference hash and full scored-label hash table against it
  before accepting the calibration result.
- The new actual streaming regression scores the first 256-row fixture,
  creates a separately sealed second-LR output with the pinned reference,
  substitutes changed annotation bytes without changing file metadata, and
  verifies rejection before `SCORED_SEALED.json`. This previously demonstrated
  cross-trial content-drift path is closed.

Supplemental temporary CPU fixture: actual first `worker.score()` created the
reference and actual `queue.validate_result()` accepted the matching result;
actual `queue.make_manifest()` pinned its exact path/hash for the next LR;
mutating the reference then caused actual `queue.validate_result()` to reject
the earlier result. Worker admission/dataset verification were mocked only
for the synthetic score fixture. No real images, masks or preparation were used.

### Closure Verification

Focused command: **6 passed, 64 deselected in 1.99s**. Deselection was intentional
to honor the F2-R/F5-only recheck scope; this is not a rerun of the whole suite.

```text
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/wwww/miniconda3/envs/mlmp/bin/python -B -m pytest -q -p no:cacheprovider tests/test_evolution_tuning_queue.py tests/test_evolution_tuning_worker.py -k 'extra_runtime or unrecorded_startup or later_calibration_trial or bounded_streaming_score'
```

Source hashes were unchanged between inspection and completion:

```text
tools/evolution_tuning/queue.py
20e07e6263c3d47c20ae825c5ac0cde7984de5cdf923c31fdb6fc73ccd7cee1d
tools/evolution_tuning/worker.py
259fbb901b164eed6ca3c211db612947afdf96b7bd8259df47a3c35605767755
tests/test_evolution_tuning_queue.py
9582a0423099f96b787f375c1bf572a9a38ada7bde1745634219a18d18ac01f7
tests/test_evolution_tuning_worker.py
bcdeda797e280e34998b5143eaf8a11ded37c1f1d804b41fa2ba57e0b1feb034
```

Only this audit document was edited. No GPU, real preparation, results/history
scan, implementation/test edit, commit or agent delegation. `prepared0746`
was not inspected, changed or launched by this auditor; per the user's
disposition it remains a CPU-only superseded preflight, not post-fix evidence.

## Archived Follow-Up: 2026-10-29

Bounded follow-up completed across the final `common.py`, `queue.py`,
`worker.py`, `inputs.py`, their three focused test files, and the directly
reused loader, snapshot, reset/accounting, finite-guard and scoring helpers.
The first-pass report below is retained as historical evidence; this section
supersedes its open-finding status. No manuscript assessment or acceptance score.

**Remaining findings: two P2, no new P1 demonstrated.** F1, F3 and F4 are
verified fixed. F2's managed-source addition test is fixed, but its broader
executable-source integrity property remains incomplete as F2-R below.

### F2-R [P2] Uninventoried Startup Code And Cached Bytecode Still Execute

Locations: `tools/evolution_tuning/queue.py:134-136`, `:162-173`;
`tools/evolution_followup/queue.py:40`, `:125-139`;
`tools/evolution_tuning/worker.py:108-122`.

The new equality comparison is effective for included files under `trace_ovss`,
`tools`, and `tests`. The reused inventory excludes runtime-root files, all
`__pycache__` entries, and `.pyc`/`.pyo` files. Those exclusions include code
that the child interpreter can execute. `-B` and `PYTHONDONTWRITEBYTECODE=1`
disable cache writes, not reads of existing valid cached bytecode.

Two independent temporary-directory CPU reproductions called actual
`queue.verify()` with unchanged prepared hashes before launching harmless
Python children using the controller's `-B`, cwd, and PYTHONPATH conventions:

1. Adding `runtime/sitecustomize.py` passed verification; the child executed
   its harmless environment-marker assignment during interpreter startup.
   The file was absent from `source_inventory(runtime)`.
2. Adding a timestamp-valid `tools/__pycache__/frozen.*.pyc` also passed.
   Its code printed `unhashed cached code` when imported by `python -B`,
   although the recorded `.py` still matched its frozen hash. The fixture used
   `importlib`'s timestamp-pyc builder with the original source's size/mtime.

No real worker/model was launched and no runtime/source outside the temporary
fixtures was changed. These are demonstrated execution paths, not evidence of
tampering in an existing suite. Worker module-path checks do not bind executed
bytecode to the hashed `.py`; startup code also runs before worker admission.

Feasible fix: make the frozen runtime a closed allowlist over its entire tree,
including root-level entries. Reject unrecorded files and symlinks, and reject
all executable caches in a fresh `-B` runtime rather than ignoring them. If
non-code bookkeeping files are permitted, allowlist those exact paths
explicitly. Apply the check before child startup and after exit. Add both
startup-module and valid-cached-bytecode regression cases. The existing shadow
package regression alone does not cover these paths.

### F5 [P2] Fixed Calibration Labels Are Not Content-Bound Across Trials

Locations: `tools/evolution_tuning/inputs.py:256-260`;
`tools/evolution_tuning/worker.py:142-154`, `:493-495`, `:524-530`;
`tools/evolution_tuning/queue.py:202-210`, `:267-273`.

Prediction-time stat-only label checks respect the label barrier. After scoring
opens labels, however, each trial merely records its own label hashes in
`score_inputs.json`; neither scorer nor controller compares those hashes with
the preceding trials' calibration labels. Size/mtime/device/inode checks omit
ctime and cannot detect an in-place, same-size change with restored mtime.
Such a change can therefore alter the calibration objective during search
without invalidating the scores or selection evidence.

CPU reproduction: create two distinct-LR calibration fixtures (0.001 and 0.002)
with the same frozen 256-row panel, real prediction seals, and identical
all-zero synthetic predictions. Score the first. Rewrite the synthetic labels
from class 0 to class 1 in place, preserving size/inode and restoring mtime.
The real `verify_inputs()` accepted the second panel, including the physical
identity details, with only its vocabulary/dataset constructor substituted.
Both actual `worker.score()` calls and actual `queue.validate_result()` calls
succeeded; both scored seals were produced. Recorded calibration mIoU changed
from 100.0 to 0.0, and all 256 label hashes differed. Worker admission and the
score wrapper's dataset verification were mocked for these synthetic outputs;
no model, real inputs or real preparation was used.

The finding requires stat-preserving content drift; ordinary size/mtime/inode
drift already stops. It is not a claim that real labels changed, nor a finding
of inference-time label access. The missing property is a stable scored label
identity for the fixed calibration panel.

Feasible fix: after the first authorized calibration score, seal the dataset's
complete label-hash table and bind subsequent score-only processes to it.
Reject mismatches before completing another score or admitting it to selection.
Keep this evidence out of model/inference access. A read-only label snapshot
or additional ctime-based preparation/read guards can further protect the
pre-first-score interval without opening annotations during prediction. Test
the same-size, restored-mtime rewrite with two distinct configs.

### Verified First-Pass Fixes

- **F1 closed:** `queue.py:347-355` seals available calibration selections and
  returns `awaiting_all_eight_calibrations` before creating all-config/full
  artifacts when blocked. `worker.py:193-195` separately refuses full manifests
  unless all eight are runnable and none blocked. The new queue/worker
  regressions pass. A later authorized continuation still needs a defined
  separate stage; the current no-restart marker intentionally remains.
- **F2 original managed-file case closed:** exact inventory equality at
  `queue.py:135-136` rejects an added shadow package under a managed tree.
  The passing regression establishes that narrower case, not F2-R's exclusions.
- **F3 closed:** `queue.ineligible_failure():224-243` requires explicit nonfatal
  numerical status and exact manifest/config/source/checkpoint/output/stage
  bindings. `worker.failure_receipt():344-352` anchors numerical classification
  to guard-message prefixes instead of incidental filename words. Fatal and
  binding-mismatch regressions pass; no silent-retry path was found.
- **F4 closed:** `queue.tune():282-297` retains an inherited ineligible trial
  and continues batch coarse search. The regression confirms batch 28 fails,
  batch 1 is attempted and selected, and failed evidence remains retained.
- **Provenance rebasing verified:** `queue.rebase_panel_sources():43-70`
  checks copied hashes and rebinds managed panel provenance to runtime files
  or the frozen contract. Original paths remain in mapping/origin metadata;
  they are not substituted for the active frozen-path checks. Its regression
  passes. External historical/data metadata remain explicit external inputs.

### Integration Checks And Limits

- Input discovery mirrors declared loader ordering/counts and keeps complete
  full populations. Calibration is a deterministic seed-2027 training-source
  sample with own-evaluation source/path/inode overlap rejection. Waterbirds
  remains explicitly blocked, without shrinking its full population. This was
  checked against implementation and synthetic tests only; this auditor did
  not repeat the input author's real-data preparation or history inspection.
- The RGB-only dataset override does not load segmentation masks. Worker
  alias guards precede image/provenance/checkpoint hashing, and score verifies
  exact prediction payload sets and stage-barrier links before annotation
  reads. Full selected-config equality and all-eight selection/prediction
  coverage are checked by the worker, not just asserted by the controller.
- The worker reuses the declared method, no-OOM-fallback diagnostic batching,
  finite observation, reset/state comparison and actual counters. CPU tests
  exercise NoAdapt without adaptation, the 0/0/1 policy schedules, direct-C
  gate gradients/parameters/predictions, streaming pooled scoring, seals and
  no retry. No new method-math or validation-driven-selection bug was found.
- Global winner/dedup/tolerance/tie behavior remains as described in the first
  pass. Full manifests retain the selected four search parameters exactly.
- Controller-side scored-result checks remain narrower than worker checks;
  no additional end-to-end mismatch was demonstrated beyond F5. The stale
  `full_label_barrier` prose at `queue.py:111` and runnable-only wording in the
  worker's author audit do not describe the repaired all-eight admission rule;
  these are documentation residuals, not additional P1/P2 execution findings.
- No GPU/checkpoint execution, actual dataset preparation, full-run success,
  throughput, or real-data label stability is certified by this audit.

### Follow-Up Verification And Source Identity

Final joint focused run, after the main author's concurrent all-eight fixture
repair: **84 passed in 2.74s**, no deselection:

```text
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/wwww/miniconda3/envs/mlmp/bin/python -B -m pytest -q -p no:cacheprovider tests/test_evolution_tuning_queue.py tests/test_evolution_tuning_worker.py tests/test_evolution_tuning_inputs.py
```

Earlier runs caught the stale single-dataset full-scoring fixture. Its failure
was the intended new all-eight guard, not a runtime bug. The repaired fixture
constructs all eight selections/prediction outputs and now passes.

Executable primary sources were unchanged over the follow-up checks. Test
files changed concurrently under their owners and were rerun at the final
identities below. Only this audit document was edited by this auditor.

```text
tools/evolution_tuning/common.py
aafc87fd02fb935631877f2ef96024021bc313a719bd0b36ca94f6cfabfa7007
tools/evolution_tuning/queue.py
6bb8fa353c6be847038c2c48fab8da7f416bd0277f5fe4222b5cde13d8016a5b
tools/evolution_tuning/worker.py
483a708640866c378cf53c75349cc2de2a6010819c9a580f49aea082b0316f36
tools/evolution_tuning/inputs.py
b2d2107f034dcb955447d30891dbb6d22bea8010a9ad6e807bdb8e85515d4083
tests/test_evolution_tuning_queue.py
af51c2c609e656d58ad09ba8807974ffe8aecda531366e1fdb61f6697784b9c2
tests/test_evolution_tuning_worker.py
a77501cb37edc5f3496c7c9cadc8d9be58e85a7e1359ba4b323f4b8a634376a4
tests/test_evolution_tuning_inputs.py
07ba51debcdf52bc3647ffed1757ef7f5abdf412f7fe9b42cddff175886aaff7
```

## Archived First Pass

First pass, 2026-10-29. This is a controller/search code audit against
`docs/single-probe-tuning-contract.md`, not a manuscript review, acceptance
decision, or assessment of experimental outcomes. No acceptance score.

## Scope And Evidence

- Primary files: `tools/evolution_tuning/common.py`,
  `tools/evolution_tuning/queue.py`, `tests/test_evolution_tuning_queue.py`.
- Read the existing snapshot helper to resolve the controller's source boundary.
  Worker/input files were concurrently authored and present during this pass.
  Only relevant worker receipt, prepared-selection, and barrier interfaces were
  spot-checked; neither worker nor input preparation received a complete audit.
- No GPU, real preparation, real inference/scoring, results/history directory
  scan, commit, or agent delegation. Only this document was edited. Synthetic
  fixtures lived in automatically removed temporary directories outside the repo.
- Existing tests: `CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python -B -m
  pytest -q -p no:cacheprovider tests/test_evolution_tuning_queue.py`:
  **17 passed in 0.10s**.
- Additional inline CPU fixtures exercised actual controller functions with
  model subprocesses replaced by mocks. Their observations are stated below;
  they are not evidence of a real model or dataset run.

## Findings

### F1 [P1] Blocked Waterbirds Does Not Hold The All-Eight Full-Scoring Barrier

Location: `tools/evolution_tuning/queue.py:294-313`, especially `:296-298`;
the weakened prepared policy is also recorded at `:80`.

The contract requires selected configs for **all eight** to freeze before any
new full-evaluation scoring. `run()` selects only `prepared["runnable"]`, writes
`ALL_CONFIGS_SELECTED.json` with `complete=True` for that subset, and immediately
predicts and scores its full arms. Waterbirds being in `blocked` changes only
the terminal seal/status at `:322-331`; it does not stop full scoring. This
exposes full outcomes before the outstanding dataset's calibration decision or
selection has been resolved. Calling the resulting suite partial does not undo
that exposure.

CPU reproduction: supply seven runnable datasets and blocked Waterbirds; mock
gates/tuning/prediction and replace the barrier with a no-op; stop the fake child
on its first score dispatch. Actual `run()` wrote `complete=True` with seven
selection hashes and dispatched full score for `voc20`. The no-op barrier means
this fixture demonstrates controller dispatch, not an actual label read.
The current worker does not close the policy gap: `worker.py:203` validates
selection coverage against `runnable`, and `:418-424` also admits runnable-only
full barriers. This is a demonstrated controller policy mismatch, not merely a
missing worker feature.

Feasible fix: let runnable calibration/search finish and seal independently,
then persist an awaiting-all-eight-selection state and return without full
scoring while any dataset is blocked. Require exact `DATASETS` coverage before
the all-config complete marker and full scoring barrier. If a seven-dataset
exception is intended, obtain an explicit revised contract before scoring;
do not infer that exception from permission for other calibrations to progress.
Preserve the no-retry rule by defining any later full-only execution as a
separately authorized stage over existing immutable selection seals.

### F2 [P2] Frozen Source Verification Does Not Reject Added Executable Files

Location: `tools/evolution_tuning/queue.py:103`, `:127-149`;
`tools/evolution_tuning/common.py:63-68`.

`verify_hashes()` checks only keys in the recorded map. It never compares the
current runtime inventory to the frozen inventory. An added `.py` file or
package therefore passes the controller's pre-child and post-child checks even
though the executable source tree changed. The reused snapshot helper performs
an exact inventory comparison at snapshot creation, but that property is lost
in the subsequent controller verification path.

CPU reproduction: construct a minimal synthetic prepared suite with one hashed
runtime file; verify it, add an unlisted `unexpected.py`, and call actual
`queue.verify()` again. Both verifications succeed with the same prepared hash.
No real preparation was called.

Impact: an added package such as
`tools/evolution_tuning/worker/__init__.py` can shadow the existing `worker.py`
module when the next `python -m tools.evolution_tuning.worker` starts. More
generally, added importable files can change resolution or import-time behavior.
The worker's imported-module membership checks (`worker.py:108-122`) provide
some additional detection, but execute after imports and cannot be relied on
when the intended worker itself is shadowed. No actual source substitution or
child execution was performed in this audit.

Feasible fix: compare the complete current source inventory to
`prepared["runtime_hashes"]`, using the same trees, excluded caches, and symlink
rules as snapshot creation, before and after each child and at queue admission.
Add tests for extra files and an added shadowing package, not just byte drift,
path escape, or a pinned prepared digest.

### F3 [P2] Substring Failure Classification Can Skip Fatal Failures Or Abort Valid Numerical Failures

Location: `tools/evolution_tuning/queue.py:202-209`.

The controller stringifies the entire receipt and searches for
`OutOfMemoryError`, `CUDA out of memory`, or case-sensitive `Nonfinite `.
It ignores the worker's structured `failure_kind`, `fatal`,
`ineligible_trial`, `retry_allowed`, and manifest/config bindings.

Two actual `trial()` CPU fixtures, with only `child()` mocked, demonstrated:

1. A receipt with `failure_kind="fatal_drift"`, `fatal=True`, and
   `ineligible_trial=False`, whose error is
   `Input provenance source drift: /fixture/CUDA out of memory/source.json`,
   is recorded as an ineligible trial and search may continue. The path is a
   synthetic trigger; this is not a claim that such an input exists.
2. A valid structured numerical failure with `failure_kind="nonfinite"`,
   `fatal=False`, `ineligible_trial=True`, and error
   `RuntimeError: non-finite adaptation gradient` aborts as a fatal worker error.
   The current worker's `failure_receipt()` at `worker.py:342-348` explicitly
   recognizes both lowercase `nonfinite` and `non-finite`, making the controller
   and worker classifiers inconsistent.

Feasible fix: accept a single documented structured receipt with exact phase,
manifest SHA, config, source/checkpoint identity, and output bindings. Fatal
must take precedence. Admit only an explicit nonfatal `oom`/`nonfinite`
classification with `ineligible_trial=True` and `retry_allowed=False`; missing,
conflicting, or malformed evidence must stop. Do not classify by free-text
paths/messages. Add tests for both examples and conflicting receipts.

This finding concerns skip/abort classification, not observed silent retries:
the controller's child log/receipt guards and suite-start marker reject retries.
The worker's own numerical-error typing still needs a separate complete audit.

### F4 [P2] An Inherited OOM Prevents Searching Smaller Feasible Batches

Location: `tools/evolution_tuning/queue.py:233-234`.

Even when `trial()` correctly records the inherited configuration as an
ineligible CUDA OOM/numerical trial, `tune()` immediately requires it to have
completed. It never reaches the batch coarse search, including batch 1.
Consequently a large inherited batch can stop the dataset despite feasible
smaller configurations. The contract preserves the inherited candidate and
allows numerical/OOM trials to be ineligible; it does not require that
candidate to succeed before the declared search can begin.

CPU reproduction: use inherited batch 28 and mock `trial()` to return
`status="ineligible"`. Actual `tune()` makes exactly one trial call and raises
`Inherited starting trial must succeed`; no coarse batch is attempted.

Feasible fix: retain the inherited failed receipt/cache row, continue the
predeclared batch coarse candidates, and select from completed candidates.
Stop explicitly only if no valid coarse candidate exists or a fatal protocol
failure occurs. Test inherited OOM with a successful smaller batch, all-coarse
ineligible, and fatal-source failures separately.

## Checks With No Demonstrated Controller Bug

- Selection reads calibration result mIoU/time at `queue.py:215-219`; full
  metrics are produced only after tuning and are not fed back into `tune()`.
  This is a code-path observation, not proof that prepared panels are training
  source-disjoint. Full input provenance remains outside this first pass.
- `common.choose():99-107` uses the maximum of the supplied completed rows,
  one fixed 0.02 percentage-point window, measured time, then the deterministic
  `(batch_size, lr, steps, trace_max_regions)` tuple. Final `tune():251` supplies
  all cached rows, retaining the inherited candidate and earlier valid trials.
  No cumulative final tolerance or latest-coordinate-only winner bug found.
- Dedup keys bind the entire config (`common.py:71-72`); the cache is checked
  before execution and the 80-trial cap includes ineligible trials. One mocked
  complete search exercised actual `tune()` and `trial()` bookkeeping: **57
  unique prediction dispatches**, no duplicate config, inherited global winner
  retained when every proposal was worse, non-search fields unchanged.
- Full manifests use the exact selected config object, without reverting batch,
  LR, steps, or K (`queue.py:302-304`). Current worker prepared-scope checks also
  compare the complete config to its selected seal (`worker.py:194-205`). These
  are static integration observations, not a completed full-run validation.
- The controller writes a per-trial prediction barrier before calibration
  score and collects every runnable full prediction before full score. F1 is
  the coverage-policy failure; it is not an observation of score-before-own-
  prediction behavior.

## Test Gaps And Follow-Up Boundary

The existing 17 tests cover helpers, ranges, local selection examples, manifest
non-search-field guards, basic seal hashing, and prepared-digest pinning. They
do not exercise actual `run()` with blocked datasets, complete search execution,
failure receipt classification, inherited numerical failure recovery, or
source inventory additions. Passing them does not resolve F1-F4.

The controller result reader at `queue.py:169-178` relies on worker-produced
evidence: it checks listed scored hashes, dimensions by image count, arms, and
pooled mIoU, but does not independently require result/config/manifest/barrier
identity or an exact scored payload inventory. The present worker supplies
stronger checks. Treat end-to-end result/seal binding, complete gate evidence,
label isolation, dataset provenance/disjointness, and actual cost accounting as
pending worker/input integration audit items, not as demonstrated violations
solely because controller-side checks are narrower.

## Audited Source Identity

Primary file hashes were unchanged between initial inspection and fixture
completion. Findings apply to these snapshots; concurrent later edits require
reinspection.

```text
docs/single-probe-tuning-contract.md
74734afa67af6758882c320694baa53769118b82bd8cc94f8d5c0bb6c92f9bf0
tools/evolution_tuning/common.py
aafc87fd02fb935631877f2ef96024021bc313a719bd0b36ca94f6cfabfa7007
tools/evolution_tuning/queue.py
01518665db453562eb29cdb611aba41101ea920febe24db0ac42289165e9a62e
tests/test_evolution_tuning_queue.py
3d28ab63c2c4dbaa278390ffde7f0fe9a3bb8e5a4f01f79f6d5ca529429a2478
```
