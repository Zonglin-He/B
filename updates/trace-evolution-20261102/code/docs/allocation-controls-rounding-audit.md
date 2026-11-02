# Independent Allocation Accounting Rounding Audit

## Status and Scope

2026-10-29. Independent code audit, not implementation authorship or paper
review. The user authorized a minimal accounting repair and a distinct new
controlled freeze after the original 1958 gate failure, with one subsequent
gate attempt only after tests, source audit and actual-artifact audit.
This document does not authorize a GPU launch.

Worktree: `/home/wwww/.codex/worktrees/trace-simplification/hzl`.
Only this document is written by this auditor. No source/test edits, suite
edits, model construction, model forwards, GPU jobs, real dataset, checkpoint,
GT or prediction-payload reads occurred. CPU probes use synthetic tensors and
the existing synthetic test fixtures. Bytecode and pytest cache are disabled.
The original failed suite, old 32-full results, paper and reviews are untouched.

Read the failure report, protocol, prior independent code audit, worker,
allocation runtime, native image-balanced and trace objectives, and relevant
tests. The initial memory registry lookup returned no relevant entry; no
memory-derived claim is used here.

## Failure Interpretation

The failure report records `Actual token coefficient differs` in the original
1958 real gate. This audit reads that report, not actual model/prediction
payloads. The original two differing numbers were not retained. Neither a
measured one-ULP discrepancy nor correct completed native equivalence follows
from that report. Zero complete gate passes and no new scientific result are
established by the failed attempt.

The original code constructs an algebraic token receipt on the entropy device:
`(weight / normalizer / groups) / area / prompts`. The worker compared its
exported float with a CPU tensor reconstruction using exact equality. This
silently assumes identical arithmetic paths across devices. Candidate-level
`weight / normalizer / groups` has the same portability risk; passing that
comparison in the first partial episode does not settle later episodes.

These receipts describe algebraic allocation of a regional mean. They are NOT
measured autograd entries or a bitwise reconstruction of the executed loss.
The objective actually reduces candidate entropies, weights and sums them,
divides by the native normalizer, and averages active groups. Floating-point
reassociation need not reproduce that execution or its backward computation.
The actual complete native gate remains the loss/gradient/parameter evidence.

## Required Accounting Separation

The v2 design is appropriate within an accounting-only repair:

1. Preserve raw device candidate and token coefficients without replacing them.
2. Reconstruct candidate reference on the actual entropy device from exported
   weight/normalizer bits, in the original dtype and operation order:
   zero-dimensional tensor division, then Python-integer group division.
3. Reconstruct device token reference from RAW candidate coefficient bits,
   followed by separate Python-integer area and prompt divisions. Do not use
   portable candidate coefficient here, tensor divisors, or a combined divisor.
4. Require both raw/device-reference equalities EXACTLY at live preparation.
5. Compute CPU portable candidate coefficient from weight/normalizer/groups.
   Require exact equality to independently reconstructed CPU receipts. Compute
   portable token coefficient from that PORTABLE candidate coefficient.
6. Keep q, assigned q, weight, normalizer, activity, term/donor/source/crop,
   actual area, group count and shuffle coverage checks exact. Bind live device,
   dtype and layout to the actual entropy tensor, not merely a stored string.
7. Retain raw, reference, portable and signed deltas. Absolute deltas can be
   derived exactly from signed deltas; their absence alone is not an optimization
   correctness blocker. Do not introduce an epsilon or guessed ULP allowance.

Offline CPU verification checks portable calculations, recorded exact
raw/reference equality, delta arithmetic and scope. It does NOT reexecute CUDA.
Coherent alteration of raw AND reference with a matching delta can pass the
standalone semantic validator for a CUDA-labelled record. This is an explicit
trust boundary: live source-bound observation plus authenticated immutable
receipts, not a new independent CPU proof of CUDA behavior. A schema stamp is
not itself authentication. Existing source/manifest/attempt/seal bindings must
remain in force. CPU-labelled receipts must still equal portable values exactly.

## Source Review in Progress

The inspected v2 wrapper uses tiny detached scalar transfers only. The raw
runtime receipt computation and native objectives are unchanged. It reconstructs
candidate reference using `values[0] / values[1] / groups`, and token reference
using `values[2] / area / prompts`; indexing yields scalar tensors and divisors
retain their original forms. Portable candidate coefficient is checked against
`coefficient_receipts`, not substituted into the actual loss. No new model
forward or CPU loss offload is introduced.

The initial v1 source was worker SHA256
`549b9f89cbc71d4ab880caf58e47605dcc853a47a7951193644aa5a6c447128c`.
It was superseded during review. Its CPU evidence is not v2 clearance.

### F1: Post-Preparation Failure Context Is Incomplete

In the inspected v1 and first v2 versions, a same-device wrapper mismatch
attaches structured numeric context and `run_policy` now merges that context.
However, the later `validate_allocation_metadata` token branch still uses a
plain `require`, without `error.worker_context`, and reports the recomputed
portable value without the stored portable value that failed verification.

Independent reproduction: corrupt stored `portable_token_coefficient` by .125
in a valid synthetic CPU receipt and call the validator. It rejects, but
`worker_context` is absent, and the printed observed/reference/recomputed
portable values are all equal. This violates the proposed failure-forensics
requirement, not the optimization threshold. Before source clearance, attach
JSON-safe context containing raw/reference, stored and recomputed portable
values, candidate coefficient counterparts, term/area/prompts/groups/device/
dtype and deltas. Test propagation through `run_policy` and `failure_receipt`.

Nonfinite diagnostic values must be represented safely rather than causing
`allow_nan=False` receipt serialization to fail. Main added finite-safe string
encoding while this audit was running; the final version must be retested.

### Certificate Scope Boundary

`compare_gate_episode` already records per-arm allocation metadata in
`actual_arm_counts`, including successful raw/reference/portable deltas. The
gate certificate revalidator currently verifies authenticated gate bindings,
thresholds, counts and finite updates, but does not independently repeat all
new accounting-metadata checks. Adding such checks is useful receipt-schema
hardening, not an independent replay of CUDA. Under the authenticated live
observation trust model, this omission alone is not classified as a native
optimization-correctness blocker.

## Independent CPU Evidence So Far

- Allocation-only joined v1 rerun: **586 passed in 4.44 seconds**. CUDA hidden,
  `-B`, `PYTHONDONTWRITEBYTECODE=1`, pytest cache disabled.
- Independent synthetic v1 matrix: **40** five-arm/fp32-fp64/prompt-count
  1,3,7,11 cases preserved exact loss and gradient before/after receipt wrapping.
- Eight independent corruptions of coefficient/weight/normalizer/original q/
  assigned q/area/source/crop were rejected by the then-current v1 validator.
- Explicit coherent synthetic CUDA-labelled raw/reference alteration passed
  offline validation as expected under the stated trust boundary. No CUDA was
  executed and this is not empirical device-rounding evidence.
- The failure-context reproduction above succeeded. One exploratory probe
  incorrectly expected nonfinite JSON failure after main concurrently fixed
  it; that probe exited on its stale expectation, not on a model/runtime error.

Main reports a broader **731 passed / 7 failed** regression run. Main attributes
all seven to historical `image_budget0233` whole-method-tree continuity rejecting
the two additive allocation modules before injected mutations. This auditor has
not independently rerun that broader suite or verified its historical inventory.
Do not relax historical pins or relabel those seven as passing. The new freeze
must separately verify all 161 fixed B0817 parent runtime files byte-identically.

## Final Clearance Requirements

Final v2 review and independent rerun are pending; no source or artifact
clearance is issued by this intermediate record. Required CPU coverage includes
all five arms, fp32/fp64, 3D/4D entropy, multiple prompt/group counts, unequal
areas, overlap, singleton/shuffle/EPS/empty behavior, raw-versus-reference
corruption, portable-only corruption, stale v1 schema, wrong device/layout,
nonfinite strict-JSON diagnostics, and failure-context propagation. Confirm
receipt wrapping leaves loss, gradients and cache unchanged, including native
delegation. Preserve all strict native gate negative tests.

The new artifact must have distinct suite/service/prepared/runtime identities
and immutable source hashes, retain the original parent/config/input order,
and pass independent read-only artifact audit before the single new attempt.
No model override, automatic retry, fallback, old-suite restart or in-place
repair of 1958 is allowed. All prior source/resource/label-isolation/barrier
requirements remain. Both populations require their complete native real-RGB
GT-free gate and actual finite updates for every control. Keep absolute
loss/gradient/parameter thresholds at `1e-5` and changed output labels at zero.
CPU success is not CUDA or scientific success; another gate failure stops the
new attempt and preserves evidence without a second attempt.

## V2 Source Verdict: One Forensics Item Remains

Independent v2 review completed against these source identities:

| File | SHA256 |
| --- | --- |
| `tools/evolution_allocation/worker.py` | `1db1ece60652b63f315d7a9a54998f1eb619f0af9a1fb6b96a82056caeb97079` |
| `tests/test_allocation_worker.py` | `a0dc65dc8a4edcb970b60671eeb47883d2a0504c0944451344a9deb9a7316761` |
| `docs/allocation-controls-contract.md` | `96cb4841de7d1c2251fe8f716db01abc76310280d510fa3642f008190a9ed82e` |
| `trace_ovss/method/allocation_runtime.py` | `36c62685e0fadf56863d1c62f0c3c004369f722d392129871d41251fa29e962e` |
| `trace_ovss/method/image_balanced.py` | `fc45c3628723e2b70e2ce63c32db924bfc7ece29489ff2112a26ed24282a6aff` |
| `trace_ovss/method/trace/objective.py` | `d1dce2a27de03a371b8d00716609c49676b6a444c6d0f4e39b3c09963c8af589` |

The last three implementation identities match the pre-repair observation.
The current contract's repair addendum preserves native thresholds and the
single-new-attempt boundary; it is not permission to modify old freezes.

### Independent V2 Evidence

- Joined four allocation test files: **591 passed in 4.50 seconds**, CUDA
  hidden, no bytecode or pytest cache. This independently confirms the reported
  allocation count, not the broader historical regression result.
- **40** new v2 synthetic cases: all five arms, fp32/fp64, prompts 1/3/7/11,
  unequal and overlapping areas, multiple source groups. Receipt wrapping
  preserved exact loss and gradient and validated portable accounting.
- **15** independent corruptions rejected: weight, normalizer, original q,
  assigned q, area, source, crop, portable candidate coefficient, raw candidate
  coefficient, candidate reference/delta, raw token coefficient, token
  reference, portable token and token delta.
- Candidate and token NaN wrapper failures both produced strict-JSON-safe
  `failure_receipt` output. CUDA remained uninitialized.
- ASTs of `compare_gate_episode`, `gate`, `build_method`, `validate_cost` and
  `source_permutation` exactly match the original 1958 frozen worker. In
  particular, the real native loss/gradient/parameter gate was not changed.

### Sole Outstanding Clearance Item

F1 remains open for the worker hash above. Independently reproduced both
portable-candidate and portable-token corruptions: both reject, but neither
exception has `worker_context`. The candidate case exits through the generic
independent-accounting check. The token case prints identical raw/reference/
recomputed portable values while omitting the corrupted stored portable value.

Required scoped correction: route BOTH newly introduced portable-accounting
failures through JSON-safe structured context, including stored and recomputed
candidate/token values and term/area/prompts/groups/device/dtype/deltas. Preserve
all existing rejection conditions. Exercise the actual exceptions through
`run_policy` and `failure_receipt`, not only an artificial pretagged exception.

No additional necessary optimization/numerical blocker was found in v2. The
candidate-level extension is preferable to leaving a known second cross-device
division portability assumption in a single-attempt suite. This conclusion
does not certify actual CUDA arithmetic or the real gate. **Source clearance
for the new freeze is withheld only until F1 is closed and that exact final
source is independently retested.** Actual-artifact audit remains separate.

## Final V2 Source Clearance: F1 Closed

The following conclusion supersedes the pending source verdict above, not the
original failed gate or the requirement for a new artifact audit.

**F1 is CLOSED. The exact sources below are cleared for a distinct new CPU
preparation/freeze and independent actual-artifact review. No GPU launch is
admitted by this source clearance.** No further necessary P1/P2 source blocker
was found in the scoped accounting/forensics repair.

| File | Cleared SHA256 |
| --- | --- |
| `tools/evolution_allocation/worker.py` | `211e12ae019667557762d8faa07cf8a1a2e730ca4383e2e94a959462045baeae` |
| `tests/test_allocation_worker.py` | `cbefd2f7ba83acdf40813857b71414afdc56d2cea433fa58489710352c297fa3` |
| `docs/allocation-controls-contract.md` | `96cb4841de7d1c2251fe8f716db01abc76310280d510fa3642f008190a9ed82e` |

### Closure Review

`accounting_failure` now sanitizes nonfinite diagnostic floats to explicit
string representations before constructing both the error text and structured
`worker_context`. The wrapper uses this helper for raw/reference mismatches.
The validator moves portable candidate coefficient equality into the numeric
accounting block without deleting the condition or changing its exact equality.
Both candidate/token failures now include stored and recomputed portable
values, raw/reference values, stored/recomputed deltas and numeric scope.
The actual shape/device/dtype-bound wrapper and exact same-device comparisons
remain unchanged. No epsilon or ULP-based acceptance rule was added.

`run_policy` preserves the numeric context while adding arm/image/patch/time/
partial-operation context. `failure_receipt` retains that context and the
no-retry flag. The added four parametrized tests invoke the actual validator
through an `ExperimentTRACE` instance and `run_policy`, then serialize the
actual failure receipt. Model adaptation/prediction are synthetic stubs, not
real forwards; the numeric validator, exception helper and context merge are
the real code, not mocks returning pretagged exceptions.

### Final Independent Verification

- Actual joined allocation rerun: **595 passed in 4.58 seconds**. CUDA hidden,
  bytecode disabled, pytest cache disabled, only synthetic test artifacts.
- Independent extension beyond the submitted four cases: **8** actual
  validator -> `run_policy` -> `failure_receipt` cases, covering portable
  candidate/token corruption by finite .125, NaN, positive infinity and negative
  infinity. All rejected with stored/recomputed values, complete numeric scope,
  preserved arm and independent sentinel operation counters. All failure
  receipts serialized with `allow_nan=False` and retained `retry_allowed=false`.
- **8** independent wrapper failures for raw candidate/token finite/NaN/positive-
  infinity/negative-infinity corruption. Every case retained numeric scope and
  produced strict-JSON-safe failure receipts.
- Rechecked unchanged ASTs against 1958 for `compare_gate_episode`, `gate`,
  `build_method`, `validate_cost` and `source_permutation`. Native real gate,
  model configuration, operation schedule and shuffle protocol are unchanged.
- Runtime, image-balanced and trace-objective hashes remain the unchanged
  identities in the preceding table. CUDA remained uninitialized and no model
  was constructed. No test or command process remains running.

The earlier 40-case v2 exact loss/gradient invariance evidence remains relevant:
the closure only consolidates failure construction and moves an existing
portable equality into its diagnostic block. The joined final tests again
cover objective identity and strict native gate negative cases.

### Admission Boundary

The offline CUDA-labelled receipt trust boundary remains exactly as documented:
CPU validates portable arithmetic and authenticated recorded live equality, not
independent CUDA reexecution. Signed deltas remain inspectable measurements of
bookkeeping-path differences, not gradient error bounds or scientific outcomes.

Main may now create the distinct new CPU-frozen identity and submit it for
read-only artifact review. That review must bind these exact source hashes,
the fixed B0817 parent continuity, input/configuration/order, new suite/service/
prepared identity and unchanged resource/label/barrier/no-retry requirements.
Original 1958 must remain immutable. No actual artifact identity has yet been
submitted to this auditor, no actual CUDA gate has passed, and this clearance
does not start or authorize an additional automatic attempt. The audit remains
available for the requested artifact stage.

## Actual Artifact Admission: 20261029_2055

Read-only independent artifact audit completed 2026-10-29T12:58:24Z.
**2055 is ADMITTED for main to start its exact retained gated controller under
the user's single-new-attempt authorization.** No necessary P1/P2 artifact
mismatch was found. This admits the actual immutable prepared artifact, not a
different preparation, interpreter spelling, service, model override, retry or
gate bypass. This auditor did not start the service or run GPU/model work.

| Identity | Verified Value |
| --- | --- |
| Suite | `/home/wwww/hzl/results/trace_allocation_controls_20261029_2055` |
| Prepared SHA256 | `c3b9e5d1548655e6225e41f2b8def91007ecb6b4904343388958b14ba2f42dd4` |
| Runtime-inventory canonical SHA256 | `880c6d720f6f4e89cc383968912cf3f16ed4c26da6db5b1d49133a7e18ea0511` |
| Parent-proof SHA256 | `ea482f978b77919804840ef9088842ed19ac38a6b4fc754e6ac7414f2bafc7b2` |
| Fixed B0817 prepared SHA256 | `4f3e54226f99230af48b2105372300d06f31651515324b3ab99dbe138ff3cfa3` |
| Service | `trace-allocation-controls-20261029-2055.service` |
| Frozen runtime / required working directory | `/home/wwww/hzl/results/trace_allocation_controls_20261029_2055/runtime` |

### Actual Inventory and Source Identity

Verified the exact expected whole-artifact set: **198 regular files, 25
directories, 174 runtime files and four manifests**. No extra or missing file
or directory, symbolic link, hardlinked regular file, special file, pyc/pyo,
`__pycache__` or pytest cache exists. Prepared queue state records four planned
cells / ten planned full-arm outcomes and `gpu_work_started=false`. These are
planned counts, not completed scientific outcomes.

No execution receipt, gate/full/log directory, controller lock, gate barrier,
full-prediction barrier or final suite seal exists. The exact file inventory
also rules out stray attempt sentinels. Independent read-only systemctl queries
before and after verification showed the declared service `not-found`,
`inactive/dead`, MainPID0, ExecMainPID0 and no invocation/cgroup. It has not been
started. Not-found resource defaults such as infinity or RemainAfterExit=no are
NOT evidence of an installed resource envelope.

Every frozen runtime hash matches both its prepared inventory and current
worktree source. The frozen worker/test/contract exactly match final source
clearance above. Relative to the failed 1958 runtime, the changed runtime set
is EXACTLY `tools/evolution_allocation/worker.py` and
`tests/test_allocation_worker.py`; no native inherited objective changed.
The separately frozen contract carries the reviewed repair addendum.

All **161** fixed B0817 parent runtime files were rehashed and remain
byte-identical. The thirteen additive files are the allocation package/method/
test files and the already inventoried tuned-figure helper/test. This audit
does not review or change paper figures. All **306** declared importer
metadata/source read-set entries were independently rehashed and matched,
including the executed-source importer identity
`58dc31d65b7651544e0623dca242dd7ae7365b5ec622b42103bdf185fb44c559`.
Allowed entries are JSON/JSONL/CSV, source/docs/text and vendored tokenizer gzip;
no prediction array, actual RGB/GT or checkpoint payload was opened.

### Parent, Input, Selection and Checkpoint Binding

All sixteen local parent inventory entries match their declared identities;
original parent copies bind to the importer read-set through the actual frozen
verifier. Independently checked parent seal links to global selections,
all-full-prediction barrier and both selected datasets' prediction/score/result
seals. Both selected configurations still prohibit full evaluation for
selection. The parent's eight populations / 32 outcomes remain authenticated
metadata assertions and historical results, not a new payload reevaluation.

Both full panels exactly equal the original sealed parent protocol panel AND
the parent's hash-bound `inputs.json` evaluation panel, including all input
rows, order, vocabulary, query mapping and readout conventions. They also equal
the preserved 1958 panels. Gate panels are exactly the first four full rows,
with `images=4` and `full_split=false`. Ordered IDs, image paths and label paths
are unique within each population. Every label size/mtime matches its stored
metadata by STAT ONLY.

| Dataset | Full Images | Classes / Queries | B | lr | Steps | K |
| --- | --- | --- | --- | --- | --- | --- |
| COCO Object | 5000 | 81 / 112 | 13 | 0.00375 | 30 | 35 |
| PC60 | 5105 | 60 / 60 | 13 | 0.00375 | 7 | 20 |

Both retain 224x224 resize/patch, stride112, view batch16, class extensions on,
reduce-zero-label false, model seed0 and permutation seed20261029. Five arms,
no-search/no-retry and development-exposed interpretation remain fixed.

Checkpoint metadata exactly equals the fixed parent: ViT-L/14 at
`/home/wwww/.cache/clip/ViT-L-14.pt`, recorded SHA256
`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`,
932768134 bytes and matching recorded mtime. **Checkpoint bytes were NOT opened
or independently rehashed.** The actual worker's checkpoint-content identity
admission remains mandatory. Recorded RGB hashes similarly are not a fresh RGB
payload validation by this auditor.

| Manifest | Verified SHA256 |
| --- | --- |
| `coco_object.gate.json` | `250ec7e3804fddb81b2f09bd69bf37e376b4b3a00396580a0c2156f54714aabd` |
| `coco_object.full.json` | `f78373dc96213571b24bc8b9d7e491c3d47293fb787f7846a3eb8b86349fd9ef` |
| `pc60.gate.json` | `de0cf98956372fd0b4c7d989155cf6b3687260555973d10ddc6bc5c11af288f8` |
| `pc60.full.json` | `698fd90cc9a294d14eb951ced845ed3ccacb25b0d3e38bafadc737f06c69b78c` |

All four manifests pass exact reconstruction with this prepared SHA, runtime,
parent proof, config, checkpoint, source inventory and output roots.

### Safeguarded Frozen Verification

Ran the actual `common.verify_prepared(..., frozen=True)` with the bound
interpreter, cwd and imports inside 2055/runtime. A Python audit hook prohibited
all file writes/mutations and every actual image/label/checkpoint path, payload
extension or inode alias. The verification and independent inventory/read-set
hashing passed with **zero forbidden payload-open attempts and zero write
attempts**. Torch was not imported by that verification. No real prediction
payload was read.

Separately ran the four allocation CPU test files FROM the actual frozen
runtime: **595 passed in 4.63 seconds**, with CUDA hidden, `-B`,
`PYTHONDONTWRITEBYTECODE=1` and pytest cache disabled. The tests use synthetic
temporary artifacts, not actual experimental inputs or predictions. Main's
additional native116 result is reported by main and was not independently
rerun here. The seven historical continuity failures retain the separately
reported scope and are not relabelled as passing.

After tests, independently rechecked exact artifact file set, absence of
caches/sentinels, all frozen runtime/parent/contract/proof identities, prepared
state/hash and the declared service's unstarted status. No required process
remains running. Only this audit document was appended by this auditor.

### Exact Launch and Mandatory Live Envelope

Reviewed controller argv, NOT executed by this auditor:

```bash
/home/wwww/miniconda3/envs/mlmp/bin/python3.10 -B -m tools.evolution_allocation.queue run \
  --suite /home/wwww/hzl/results/trace_allocation_controls_20261029_2055 \
  --suite-sha256 c3b9e5d1548655e6225e41f2b8def91007ecb6b4904343388958b14ba2f42dd4
```

The declared retained service must actually install WorkingDirectory/import
root 2055/runtime, MemoryMax12884901888 (12GiB), MemorySwapMax1073741824 (1GiB),
OOMPolicystop, Restartno and RemainAfterExityes before model work. Use the exact
resolved python3.10 argv as the real controller main process, not an alias or
shell/PID wrapper. Preserve the frozen-root import path, disabled bytecode and
single-thread OMP/OPENBLAS/MKL environment. Do not fabricate InvocationID or
change the model. Controller and GPU children retain strict live unit/PID/argv/
cwd/prepared/invocation/cgroup/resource checks and the shared GPU lock.

Both populations' complete native real-RGB GT-free gates and all finite-arm
updates remain mandatory. Keep loss/gradient/parameter thresholds at 1e-5 and
changed labels at zero. A failure must stop this attempt without retry,
fallback, threshold change or another automatic preparation/launch. All ten
full-arm predictions must be sealed across both populations before any new GT
scoring; later scores, costs, seals and authentic terminal success still need
their own verification. No CUDA gate pass, full prediction, score or scientific
success is established by this artifact audit.

### Original Attempts Remain Preserved

Rechecked original prepared hashes and all 174 declared runtime hashes in
EACH old artifact. 1958 remains
`669402e4022ce4f7baaa4d445160f3031b085217d8c712db85f095b452e790b2`;
1946 remains
`6f6cba9c41010586041ad03c34f4d174b00ec43efad8922db9b903c243534155`.
Neither is repaired or restarted. The old failed evidence, old32full results,
paper and reviews were not written by this audit. **Only 2055 at the exact
admitted identity may proceed to the authorized new gated attempt.**
