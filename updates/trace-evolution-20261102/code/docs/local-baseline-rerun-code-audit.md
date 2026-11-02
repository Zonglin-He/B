# Independent Local Baseline Rerun Code Audit

## Decision

**ACTUAL PREPARED ARTIFACT ADMITTED for the exact retained service specified
below.** Independent CPU/source/artifact audit, 2026-10-29 through 2026-10-30
Asia/Shanghai. No remaining P1/P2 finding on the reviewed frozen artifact.
All LB-01 through LB-05 findings are closed. Admission is for the stated
single-attempt service only, with mandatory read-only deep dependency
certification and exclusive shared GPU lock before model/CUDA work. The auditor
has not launched the service or performed real GPU/gate verification. Only this
document is in the auditor's repository write set; tests use temporary fixtures.

Latest authorization is MLMP on all eight datasets, plus NoAdapt/TENT/TPT/WATT/
CLIPArTT on COCO-CD and Waterbirds-Seg only: 18 cells, 58,040 image-method
evaluations. PC59 is reported in primary Table 5 (31.95); only CD/WB are missing.
No extra model, seed, hyperparameter search, or paper review is requested.

## Finding Ledger

### LB-01 [P1] Native optimizer membership rejects valid archived methods

Location: `tools/evolution_local_baselines/worker.py:226-237`, especially line
236, called by `NativeState.__init__` before any native gate episode.

The archived TENT constructor freezes transformer, final text LN and token
embedding, and freezes the visual encoder except its LayerNorms. It does not
freeze the top-level `positional_embedding`, `text_projection`, or `logit_scale`.
These remain `requires_grad=True` but are intentionally absent from the native
Adam parameter groups. The worker incorrectly requires every requires-grad
parameter to belong to Adam. The same freeze/collect pattern occurs in archived
MLMP, WATT and CLIPArTT; NoAdapt uses the TENT constructor.

Independent executed CPU reproducer: load the actual archived `adapt/tent.py`
through `importlib.util.spec_from_file_location`, replace only its `ovss`
backend with a tiny CPU module exposing the native parameter layout, stub
printing/token features, instantiate native TENT with lr .001, then call
`worker.NativeState(method)`. No checkpoint, image, label or prediction is read.

Observed output:

```text
native_trainable_outside_Adam ['positional_embedding', 'text_projection', 'logit_scale']
REPRODUCED RuntimeError Native trainable membership drift
cuda_initialized False
```

Required correction: authenticate the optimizer's actual ordered parameter
membership against the native method's selection, not all requires-grad flags.
Keep these original non-optimized parameters unchanged; do not change native
freezing, loss, constructor, or backbone to satisfy the wrapper. Continue
finite checks on all parameters and gradients. Add a targeted CPU regression
using the actual archived constructor with a synthetic backend and reject
missing/extra/duplicate optimizer members. Re-run the reproducer independently.

Status: **CLOSED in source**, independently rechecked. Re-executed the actual
archived TENT constructor with the synthetic CPU backend: native selection
`['visual.1.weight', 'visual.1.bias']` passes; missing, extra, reordered and
duplicate Adam members are rejected. Original unused requires-grad parameters
are preserved. Author added actual-archived-constructor regression coverage.

### LB-02 [P1] MLMP native readout has twelve visual calls, not eleven

Location: `tools/evolution_local_baselines/common.py:47-50` and worker
`validate_cost`. The default eleven-call budget also applies to MLMP.

Archived `adapt/mlmp.py:136-138` requests `adaptive_weighted_mean`. Archived
`ovss/clip/model.py:594-633` calls `encode_image` once for per-layer features and
again for the weighted output. Ten adaptation steps therefore give twelve
visual traversals, not eleven. The frozen budget rejects native execution on
the first MLMP gate, even once LB-01 is corrected.

Executed independent CPU reproducer: AST-extract the actual archived
`CLIP.forward`, attach it to a tiny synthetic CPU module whose `encode_image`
returns finite features and increments a counter, invoke exactly the native
MLMP readout arguments, then validate ten adaptation calls plus the observed
readout calls using the current `common.settings(..., 'mlmp', ...)`.

```text
native_forward_visual_traversals 2 logit_shape (1, 2, 3, 4, 4)
REPRODUCED RuntimeError Actual native optimizer/loss/visual budget differs from frozen settings
cuda_initialized False
```

Required correction: freeze MLMP's truthful native count as twelve, ordinary
TENT/TPT as eleven, CLIPArTT as twenty-one, WATT as 141, NoAdapt as one. Amend
the contract's ordinary-method wording and tests before preparation. Do not
remove a native MLMP traversal or change native readout. The current worker
test deliberately rejecting twelve establishes the mismatch, not correctness.

Status: **CLOSED in source**. Independently confirmed `settings('mlmp')` now
requires twelve visual calls; contract explicitly preserves both native
readout passes. Worker regressions now reject eleven/thirteen calls and test
native MLMP-vs-TENT readout traversal counts. Combined tests after this change:
**130 passed** in 2.52 seconds. The archived math remains unchanged.

### LB-03 [P2] Gate validators authenticate hashes but not parity evidence

Locations: `common.verify_gates` and `queue.validate_gate`.

Executed fully in-memory fixture using the real validation functions, mocked
metadata reads/hash values and no file payloads. All eighteen reports had
`passed=True`, `images=4`, correct manifest hashes, self-consistent report/seal
hashes, but also `comparisons=[]`, `changed_labels=100`, `tolerance=100`.
Both validators accepted them:

```text
common_accepted_empty_comparisons_and_100_changed_labels 18
queue_accepted 6572581df2a0d6b2eb81b5a094251fde897d1f714e1b386ff1a6cd21917dbd82
```

Required correction: one shared semantic gate verifier, used by controller and
prediction admission, must validate exact manifest/source/settings/checkpoint
bindings; exactly two ordered batch-2 episodes; the declared alternating native
and observed orders; all finite deltas <= 1e-5; zero changed labels; reset proof;
method-specific actual costs; and exactly the expected finite update receipts
(zero for NoAdapt). Check the worker's actual field layout, not an invented
independent seed/backbone/test protocol. Add rejection tests for missing,
contradictory, nonfinite, wrong-method, and rehashed failing gate reports.

Status: **CLOSED in source and tests**. The new shared
`validate_gate_report` rejects the
original empty-comparison/changed-label fixture and checks parity trajectories,
manifest/source bindings and attempt markers. A subsequent fully in-memory
fixture with otherwise valid NoAdapt report/cost/attempt fields previously passed
with both episode `reset_state` fields absent; replacing one by
`{'ctx_sha256': 'INVALID'}` also passes. The report can omit `gpu_lock` and
`deterministic_runtime` entirely. Those residual cases are now corrected:
current code authenticates the eight-key reset schema/digests/name partition,
ctx only for TPT, equal episode reset states, shared GPU lock receipt, and fixed
deterministic default-attention state. Combined independent run: **144 passed**
in 2.70 seconds. Final LB-05 integration adds
`comparisons[*].parity_parameter_names`; current validator checks uniqueness,
native name-set membership, all optimized members, episode equality, and for
non-WATT exact equality with optimized membership. Independently reran a WATT
report mutation using `parity_parameter_names=['logit_scale']`: it is rejected
with `Native parity parameter membership differs`. Targeted regression tests
cover duplicate, missing and unknown parity members. Final local tests: 158
passed; broader regression: 753 passed.

### LB-04 [P1] Legitimate namespace package blocks the second gate model

Location: `worker.build_method`, original lines 197-202.

The pinned archive has no `utils/__init__.py`. `utils` is a legitimate Python
namespace package with `__file__ is None` and its search path under the archive.
Each gate builds both method and reference in the same process. After the first
build imports `utils.misc`, the second build rejects `utils` because the import
guard requires a non-null `__file__` for every native package.

Executed reproducer: import the actual archived `utils.misc` (no OVSS import,
model construction or checkpoint access), then call `build_method` with
`constructor_kwargs` mocked to avoid payload reads and model import trapped.

```text
real_archived_utils_file None namespace_paths ['.../baseline_source/utils']
REPRODUCED_SECOND_BUILD RuntimeError Contaminated native import: utils
cuda_initialized False
```

Required correction: authenticate all namespace `__path__` entries under the
frozen native root, while retaining strict file-origin validation for ordinary
modules. Do not silently delete/reload contaminated packages. Add positive
same-root namespace/repeated-build and negative external/mixed-path regressions.

Status: **CLOSED in source and tests**. Independently imported the real archived
namespace, called `verify_native_imports` twice successfully, and confirmed a
mixed same-root/external namespace is rejected. `build_method` invokes this
guard before imports, after class import, and after construction. Persistent
tests execute the archived import, exercise repeated construction admission,
and reject external/mixed/empty/fileless namespace and ordinary-module origins
without deleting or reloading the rejected module.

### LB-05 [P2] New invariant rejects native WATT averaging roundoff

Location: `NativeState.check_nonoptimized` and its call at the end of
`_run_episode`. Archived WATT collects all LayerNorm parameters into each of
seven snapshots, including frozen text LayerNorms, then averages and writes
them back via `load_state_dict(..., strict=False)` each cycle and at the end.
Seven identical FP32 values need not average to their original bit pattern.

Executed reproducer: AST-extract actual archived `WATT.weight_average`, apply it
to seven identical copies of a synthetic frozen text-LN vector [.1, .2, .3],
load its returned state as the native implementation does, then call the real
worker's `check_nonoptimized`:

```text
native_WATT_frozen_LN_roundoff_max 2.9802322387695312e-08
REPRODUCED RuntimeError Native reset tensor differs: nonoptimized.ln_final.weight
cuda_initialized False
```

This is a demonstrated valid-native CPU counterexample, not a claim about the
unopened checkpoint's exact values. Preserve native averaging; do not freeze
additional parameters or change arithmetic. Verify WATT's native-written LN
state with the direct-native gate/tolerance, retain strict guards for genuinely
untouched parameters, and validate exact reset before/after episodes. Include
non-Adam native-written state in parity evidence where needed.

Status: **CLOSED in source and tests**. `native_written_parameters` includes the
actual all-LayerNorm set only for WATT; both before/after-step and final parity
captures use that set. Original AST-extracted averaging reproducer now accepts
the native 2.98e-8 roundoff, rejects unrelated `logit_scale` drift, and verifies
exact reset. Reset receipt remains the same eight-key schema. Persistent tests
execute the actual archived WATT collection/average/write/reset code on tiny
CPU tensors, verify non-Adam LN parity capture and failure on drift/nonfinite
values, and demonstrate that the ordinary TENT untouched-parameter guard stays
strict. Common now authenticates parity membership in the gate certificate.

## Current Recheck

- FINAL independent run, with CUDA hidden and bytecode/pytest caches disabled:
  **753 passed in 6.02 seconds**, using exactly
  `tests/test_allocation_controls.py`, `tests/test_allocation_runtime.py`,
  `tests/test_allocation_worker.py`, `tests/test_allocation_queue.py`,
  `tests/test_local_baseline_queue.py`, `tests/test_local_baseline_worker.py`.
  The final two local-baseline files independently pass **158 tests in 2.76
  seconds**. These are CPU protocol/regression results, not real RGB gates.

- Re-read strengthened common/controller after the main author's update.
  Archive provenance is now copied byte-for-byte, hash-anchored, and native
  inventories equal the pinned provenance plus explicit extras. Parent and
  dependency runtime continuity and checkpoint identities are cross-bound.
- Independently ran `python -B -m pytest -q -p no:cacheprovider
  tests/test_local_baseline_queue.py`: **45 passed**, with CUDA hidden and
  bytecode/cache writes disabled. Only synthetic temporary fixtures are used.
- First worker-test run: **61 passed, 1 failed**. The failure was the TPT reset
  negative test's regex (`Native reset .* differs`) not matching the actual
  correct exception (`Native reset keys differ: optimizer.state`). This is a
  test assertion issue, not evidence of failed reset detection. Worker and
  tests were concurrently edited; re-run after the final revision is required.
- Combined independent CPU test run after the LB-01 correction: **127 passed**
  in 2.49 seconds. This supersedes the earlier regex-only test failure. It does
  not close LB-02, LB-03 residuals, LB-04 or LB-05; those cases were independently
  exercised outside that then-current test coverage.
- The initial parent-importer retention gap is corrected in current source:
  preparation copies the actual executed importer to `suite/parent_importer.py`,
  compares it to the execution receipt hash, and `verify_prepared` guards and
  authenticates those retained bytes against `proof['importer_sha256']`.
  Actual frozen artifact review is completed in the admission section below.
- Independently source-reviewed the added `queue.certify_completed` and CLI
  `certify`. An in-memory controller fixture validated 18 rows/58,040 image-
  method evaluations, CSV and canonical seals, with two terminal checks. A
  nonterminal-service exception is propagated before result validation, even
  with a completed suite seal. No new P1/P2 identified in this feature at this
  revision. This fixture mocks canonical worker verification, so it establishes
  controller composition only, not actual completed artifacts or service exit.

## Verified Checks And Boundaries

- Read the contract, common/controller/worker, inherited allocation admission
  and terminal checks, RGB-only loader/readout, and bounded archived native code.
- CPU import of the worker succeeds and does not initialize CUDA or import
  archived `adapt`, `ovss`, `utils`, or `mmengine` at import time.
- Native archive inventory verification succeeds for 32 files, pinned by
  provenance SHA256 `ab92729b308ccef321882b6736fb769153d51711359b68081ae56413bce68d1b`,
  commit `ba9f57bf47846ec727c616c5163233a98ca0bf94`, plus pinned extras.
- Observed settings preserve B2/lr .001/10 ordinary updates; WATT has native
  l2/m5/seven prompts, 70 Adam steps, five reported losses, 141 visual calls.
- This audit has not run prepare, any controller, real gate, prediction,
  scoring, or systemd intervention. No existing suite is edited or restarted.
- No checkpoint/model payload, dataset GT, RGB dataset payload, or prediction
  payload has been opened by the auditor. Synthetic CPU tensors are not real
  model or dataset verification.

## Execution Conditions

1. Source findings are closed. Freeze exactly the source revision recorded
   below; any subsequent substantive source edit invalidates that revision's
   clearance until rechecked.
2. COMPLETED: freeze and independently verify the actual new suite: runtime/native source,
   importer and parent proofs, all 36 manifests, settings/input identity/order/
   vocabulary/readout, archive provenance, attempt policy, and actual service
   configuration. A source-only review is insufficient.
3. Metadata/source/real-terminal verification COMPLETED in this audit; the
   controller must still perform its mandatory deep verification once after
   launch. Certify 2055 only from real loaded active/exited successful retained-service
   state, MainPID 0, ExecMainStatus 0, source-bound completed artifacts and the
   released shared GPU lock. Do not perturb it while it runs.
4. Only after admission may the retained service run all 18 real four-RGB parity
   gates. All 18 full predictions must seal before any new GT scoring.
5. Later completion requires all 18 canonical outcomes, exact populations,
   streamed pooled scores/costs, barrier and seals, and separately verified real
   successful service exit. No current claim of numerical parity, GPU safety,
   completed evaluation, independent test data, or equal tuning budget follows
   from this audit.

## Final Reviewed Source Identity

```text
tools/evolution_local_baselines/common.py
a42c9604d162cdfca0d4563a9781c97d73eb7a1a334ef19a0bb9e42fc4578264
tools/evolution_local_baselines/queue.py
be7478ed7adfbe4b03f8e67cccb272180706a32d44f15e6bfe2e669220e0899c
tools/evolution_local_baselines/worker.py
230f9cfb7f855cb9ba51a28e4b351b4e2ee612e2ede31129ff74a78ac1e0a7f3
tests/test_local_baseline_queue.py
409a7b5bb269a646f26487fa5e5076e22044ba1c1208181beba13fc8d32b5727
tests/test_local_baseline_worker.py
b0a1bf2a801fedf4c04715ad7d4f9c23be3f970a21add315f3ca62a09b1535c3
docs/local-baseline-rerun-contract.md
aaa5095e1f29c42011dd10dd1f4e977ffab230d54048e5b1c464030d1f4ca12f
```

The earlier source-only clearance has been superseded by the actual artifact
admission below. Numerical RGB parity, real GPU operation and completed new
evaluation remain unverified and are not implied by admission.

## Actual Frozen Artifact Admission

Admitted root:
`/home/wwww/hzl/results/trace_local_baselines_20261030_0008`

Prepared SHA256:
`11803f984bdd34fb0b78d925bbcf252f61485be3bc4d43f3eb3eb9fbad7d4515`

Service: `trace-local-baselines-20261030-0008.service`.

### Independent Executed Checks

- Ran the frozen `common.verify_prepared(..., frozen=True)` from the actual
  runtime. An audit read guard prohibited checkpoint, RGB, annotation and
  prediction payload opens, prohibited unlisted metadata/source reads and
  metadata-to-GT inode aliases, and capped each permitted file at 64 MiB.
  Verification opened 612 distinct metadata/source files and no prohibited
  payload. Inventories are exact: 180 runtime, 32 native, 52 parent copies,
  36 manifests, 18 planned cells, 58,040 image-method evaluations.
- Frozen common/controller/worker/tests and contract match every final reviewed
  source identity above. Exact inventory checks reject extra files, caches,
  symlinks and hardlinked frozen source. Native inventory is anchored to the
  recorded archive provenance SHA and commit, including pinned prompts/BPE.
- Parent proof and 306-file original metadata read-set rehashed successfully;
  its total is 106,792,653 bytes, largest file 23,457,388 bytes (`inputs.json`).
  The actual executed importer is retained as `parent_importer.py`, SHA256
  `58dc31d65b7651544e0623dca242dd7ae7365b5ec622b42103bdf185fb44c559`,
  authenticated by `PARENT_IMPORT.json` and the prepared proof binding.
- Checked all eight complete panels against original parent prediction
  protocols, exact IDs/order/config/vocabulary/query-to-class mapping and
  224-square/stride112 geometry. Full and gate manifests match canonical
  derivations, with exactly the first four RGB records for gates. Counts are
  VOC20/21 1,449 each; PC59/60 5,105 each; Object/Stuff 5,000 each; CD 500;
  Waterbirds 5,322. MLMP has eight cells; other methods only CD/WB.
- Stat-checked all 28,930 dataset rows, including label metadata, regular RGB
  existence, and no RGB-to-any-suite-label inode aliases. Loaded all eight
  vocabularies from frozen source resources and verified classes, queries and
  mapping. Checkpoint was metadata/stat only: recorded bytes/mtime agree; its
  contents were not opened or hashed by this auditor. Runtime worker must
  perform the existing actual input/checkpoint hash checks before use.
- Re-ran the same six test files FROM THIS FROZEN RUNTIME: **753 passed in
  5.98 seconds**, CUDA hidden, bytecode and pytest caches disabled.
- Actual gate and predict entrypoints, called without controller execution,
  were denied at missing `EXECUTION_STARTED.json`, before attempt/model/CUDA.
  Missing all18 gate and prediction barriers were denied. Direct controller
  identity check rejected the nonservice command/envelope. Model construction
  and CUDA initialization were trapped; prohibited payload reads were trapped.
  Before/after fingerprints of all 306 suite files matched exactly; no attempts
  or output directories were created.
- Independently verified dependency prepared/source/parent bindings and actual
  retained terminal through the frozen allocation terminal validator. Observed
  twice: loaded, active/exited, MainPID 0, ExecMainPID 620623, Result success,
  ExecMainCode 1, ExecMainStatus 0, InvocationID
  `9c01dab52725430a9b45feda83fda3a5`. Completion seal SHA256:
  `03f949e8e9fccce66b6ab1514188d90d4d7abc3027d7732959329c8a7d4a316a`.
  Verified ten canonical outcome metadata entries, score-seal/barrier/execution/
  CSV bindings, ten scored metadata-file hashes, and pooled metrics from stored
  class intersections/unions. No old prediction or episode-candidate payload
  was opened. The controller's deep payload certification is NOT replaced by
  this bounded metadata audit and must run before the first new GPU gate.

### Admitted Service Envelope

```text
ExecStart=/home/wwww/miniconda3/envs/mlmp/bin/python3.10 -B -m tools.evolution_local_baselines.queue run --suite /home/wwww/hzl/results/trace_local_baselines_20261030_0008 --suite-sha256 11803f984bdd34fb0b78d925bbcf252f61485be3bc4d43f3eb3eb9fbad7d4515
WorkingDirectory=/home/wwww/hzl/results/trace_local_baselines_20261030_0008/runtime
PYTHONPATH=/home/wwww/hzl/results/trace_local_baselines_20261030_0008/runtime
LD_LIBRARY_PATH=/home/wwww/hzl/results/driver_compat_58017302/nvidia_driver-linux-x86_64-580.173.02-archive/lib
CUDA_VISIBLE_DEVICES=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONDONTWRITEBYTECODE=1
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
MemoryMax=12884901888
MemorySwapMax=1073741824
OOMPolicy=stop
Restart=no
RemainAfterExit=yes
```

Python executable reports 3.10.13. The compatibility-library path and listed
CUDA/CUBLAS/thread environment match the actual 2055 retained-service metadata,
except PYTHONPATH correctly targets the new runtime. This is an admitted PLAN:
the new service is currently not-found/inactive/dead and has not been installed
or launched; no claim is made that a new actual cgroup envelope already exists.
Frozen controller/worker checks must reject a mismatched installed resource/
PID/invocation/command/cwd envelope before model/CUDA admission.

### Scope And Remaining Runtime Obligations

- Admission is limited to this root, prepared hash, frozen sources, service and
  envelope. No restart, retry, fallback, source mutation, model override or
  extra experiment is admitted. The auditor did not acquire the shared GPU lock;
  exclusive availability is enforced by the actual worker at runtime.
- Before any full prediction, all18 actual four-RGB direct-native parity gates
  must pass. Before any new GT scoring, all18 full prediction seals must exist
  and pass the global barrier. These real gates are not certified yet.
- All18 outcomes, streamed pooled scores/costs and seals, plus a separately
  verified successful retained service terminal, remain required for scientific
  completion. Reruns retain development exposure and unequal native tuning/
  backend/prompt budgets; admission does not change those claim boundaries.
- End-of-audit state: queue remains prepared; no `EXECUTION_STARTED.json`,
  `queue.lock`, gates/full/logs directories, stage barriers or suite completion
  seal. Prepared/source hashes match the initial artifact identities. No old
  runtime, score, study or review was edited.

## Initial Source Identity

These identify the first reproducer revision, not a frozen admission approval:

```text
common.py 160ecd3b2776467e6585f1a060182e373598c12ee8ab60f5b4204b1429c7b0ed
queue.py  fcd70863e358650a7ba30a7087ba0413c138741eede6c768f4a56e628d56df8c
worker.py 6963d65aaa703be9282b46c6442dd4d600bf6d47337e154f6efae992a991052b
contract  194fc87e38f734405896d42e66a6794771eaa336938035b1bd15c5cc8ebcd9e8
```
