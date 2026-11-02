# Independent Bounded Code Audit

Date: 2026-10-29. Workspace: `/home/wwww/.codex/worktrees/trace-simplification/hzl`.

Scope: `trace_ovss/method/image_balanced.py`, main-owned `tools/evolution_confirmation/run.py` including the actual `gate()` and latest `verify_selection_record()`, the two requested test modules, and directly relevant inherited helpers. The frozen `docs/confirmation-image-budget-plan.json` is the design contract, not experimental evidence. This is not a manuscript review, acceptance assessment, performance score, or approval to launch experiments.

## Findings

One P1 and two P2 findings. No P1/P2 finding in `image_balanced.py` itself within the inspected CPU contract. These are worker-local validation/execution gaps, not evidence that a real run has already violated the protocol. Queue/cohort implementation and their claimed test totals were not audited.

### P1: Stage barrier counts entries instead of the required distinct cells

Location: `tools/evolution_confirmation/run.py:405-415`; annotation-content read is downstream at `run.py:424-441`.

`stage_barrier()` requires two or six dictionary entries, checks their hashes, and compares only stage/cohort. It does not require unique resolved directories or the exact `(dataset, order_seed)` inventory. Consequently, one legitimate sealed PC60 cell can be listed under `cell` and `./cell` to satisfy calibration; six lexical aliases of that same cell satisfy confirmation. Every referenced seal and artifact can remain unchanged and hash-valid. Distinct directories containing repeated dataset/order cells have the same semantic problem.

CPU reproduction: with filesystem/hash/seal access mocked and the actual membership validator unchanged, both the two-entry calibration barrier and six-entry confirmation barrier accepted aliases resolving to ONE `pc60`, default-order directory. No other dataset/order cell existed in the synthetic scenario. This isolates membership validation; it is not a claim to have constructed or inspected real experiment artifacts.

Impact: `score()` can proceed to annotation bytes before both calibration datasets, or all six confirmation dataset/order cells, have predictions sealed. This contradicts the frozen plan's stage-wide label barrier (`docs/confirmation-image-budget-plan.json:25-26`). The strengthened selection verifier does not close the confirmation-stage membership gap and cannot undo an early calibration label read.

Smallest fix: reject duplicate resolved cell directories and duplicate semantic identities; require exact identity sets `{(pc60, None), (coco_object, None)}` for calibration and `DATASETS x {None, 20261030, 20261031}` for confirmation before returning the barrier. Keep the existing hash, containment, stage and cohort checks.

Missing regression: valid exact inventories must pass; path aliases, duplicate dataset/order cells, missing dataset/order combinations and extra identities must fail before `SCORING_STARTED.json` or any annotation-content access. The current missing-barrier test (`tests/test_confirmation_worker.py:104-113`) does not exercise a present but incomplete barrier.

### P2: Native gate can silently retry pixel-view OOM and still pass

Location: native class switch at `tools/evolution_confirmation/run.py:511-512`, insufficient schedule predicates at `run.py:525-526`. Relevant inherited behavior: `trace_ovss/method/trace/interventions.py:122-154`, especially `:130` and `:142-148`; audit limitation documented in `tools/trace_execution_audit.py:7-11` and implemented at `:77-99`.

Switching to native `TwoViewTRACE` restores its OOM fallback. If an OOM occurs while constructing pixel views, before `forward_eval_logits()`, the inherited iterator halves the view batch and retries. `ExecutionAudit.failed_forward_attempts` remains zero because its failure wrapper has not been entered. Total successful encoder inputs remain the expected value. Unlike the older smoke helper (`tools/evolution_followup/smoke.py:79-85`), this gate does not check diagnostic call count or the final effective view batch size.

CPU reproduction: injected one synthetic `torch.cuda.OutOfMemoryError` into native weak-view construction, with `torch.cuda.empty_cache` mocked so no CUDA operation occurred. The native engine retried at batch size 1 instead of configured 3. The current gate predicates still held: encoder inputs 6, failed forward attempts 0, identical candidate digest, maximum support/loss differences both 0, and changed predicted labels 0 versus the no-fallback image-min worker. This exercised synthetic CPU gate components, not the real-checkpoint GPU gate.

Impact: the gate can certify equivalence after an undeclared retry/batching change, despite the frozen no-retry runtime contract (`docs/confirmation-image-budget-plan.json:32`). Formal `ExperimentTRACE` prediction correctly uses the fail-closed helper; this finding concerns the native gate reference path.

Smallest fix: give the native gate reference a fail-closed diagnostic batch iterator as well, preserving its original objective and candidate semantics. Require configured batch size and expected diagnostic call count in the gate receipt. Checks alone detect a retry after the fact; disabling fallback is necessary to enforce no retry at execution time.

Missing regression: inject OOM during view construction and separately during a forward; require one attempt, propagated failure, and a failed gate receipt without a second batch or configuration change. This can remain CPU-only with synthetic exceptions and mocked CUDA cleanup.

### P2: Gate does not persist an attempt claim before execution

Location: freshness check at `tools/evolution_confirmation/run.py:485-486`; first report writes only at `run.py:563-567`, after model work at `:500-559`.

The gate checks that its output is absent but creates neither that output nor a durable started marker before model loading/execution. An uncatchable termination such as SIGKILL or an OOM-killer exit after work begins bypasses `except BaseException`, leaves no output, and allows the same worker invocation to start again as apparently fresh. Exclusive final JSON creation prevents overwriting an existing report, but does not prevent this unrecorded retry. The global GPU lock is not a persistent per-output attempt claim.

Impact: worker-level no-retry behavior after a hard stop is weaker than `predict()`, which claims a fresh output directory before loading its model (`run.py:350-363`), and `score()`, which writes `SCORING_STARTED.json` before annotations (`run.py:427-441`). Any additional scheduler-level prevention is outside this audit, not assumed absent.

Smallest fix: atomically create an exclusive, persistent gate-started marker or fresh gate directory before any model work; refuse an existing claim or report, and retain the claim on success, ordinary failure and interruption. Keep final report creation exclusive. No automatic cleanup/resume.

Missing regression: simulate termination after the claim and verify a second invocation is rejected before model construction. Also test two invocations sharing an output. This finding is source/control-flow evidence; no process was killed and no real gate was launched during the audit.

## Verified Contracts

- Image normalization: `image_balanced.py:18-50` groups by source `image_idx`, reuses detached old weighted region means, and averages only sources with nonempty active terms. Crop identity remains `batch_idx`; source identity is restored after active-candidate filtering at `:123-131`. Empty/inactive and EPS behavior remain inherited.
- Old control: `run.py:67-75` explicitly calls the original episode objective for `episode_min`, not the image-balanced hook. An additional multi-source CPU comparison found bitwise-equal losses, final logits and all parameters versus native `TwoViewTRACE`.
- Same state/admission: inherited `TRACE.adapt()` resets at `trace_ovss/method/trace/engine.py:135`; LN/Adam snapshot restoration is unchanged (`trace_ovss/method/adapter.py:78-101`). Worker candidate digests, actual diagnostic input counts and reset digests are checked at `run.py:306-331`. Twelve synthetic trajectories, all six arms in forward and reverse order, preserved native admission and exact initial LN/Adam state after every arm.
- Real view counts per candidate: T/C/min/Uniform/Confidence are 1/1/2/0/0; `episode_min` also uses 2. Counter hooks observe the visual encoder stem, not just a declared schedule. The actual worker overrides the native fallback with `prior.StandaloneTRACE._forward_trace_view_batches` (`run.py:77-79`; helper `tools/evolution_followup/standalone.py:159-172`).
- Image-only prediction: `run.py:166-176` uses the image-only dataset override; `tools/evolution/run_cell.py:77-104` reads image bytes only and rejects label tensors at collation. Dataset construction loads vocabulary/metadata, not annotation pixels. `verify_label_metadata()` (`run.py:128-137`) accepts the dict and legacy list stat schemas without opening contents. Pure-memory checks accepted both schemas and rejected changed size, inode or `contents_read` values.
- Latest selection binding: `run.py:229-267` compares the frozen selection rule, suite-prepared SHA, calibration-barrier SHA, both dataset score seals and their listed small artifacts, and recomputes the selected mode/candidate record. `predict()` invokes it before construction and again before sealing (`:345`, `:374`); `score()` compares the current selection receipt with the prediction-time receipt (`:425-426`). No stale finding about the earlier minimal selection record is retained.
- Prediction/score output writes use fresh directory creation or exclusive file modes (`run.py:323-324`, `:350`, `:427-435`; `tools/evolution_followup/standalone.py:67-70`). No overwrite/retry loop was found in those paths. The gate claim exception is finding P2 above.

## CPU Verification

Final requested-test snapshot:

```bash
timeout 180s env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 /home/wwww/miniconda3/envs/mlmp/bin/python -B -m pytest \
  -q -p no:cacheprovider tests/test_image_balanced_trace.py \
  tests/test_confirmation_worker.py \
  -k 'not explicit_image_loader_never_opens_label_pixels'
```

Result: **43 passed, 1 deselected in 1.68 s**. The deselected test (`tests/test_confirmation_worker.py:125-143`) creates and reads a synthetic RGB image; it was inspected but not executed to keep this audit free of image-file reads. No raw dataset image or annotation content was read.

Additional bounded, in-memory CPU probes: twelve synthetic arm trajectories; exact multi-source old-control comparison; dict/list stat-only metadata checks; barrier alias counterexamples for both stages; native view-construction OOM counterexample; and the latest selection verifier's valid case plus eight negative cases covering rule, suite, barrier, mode, candidate receipt, score population, score-seal hash and scored-artifact hash. The extra selection-negative probe stubbed only `queue.select_mode`; the final existing worker test also exercised the real selector using synthetic score JSON. Its implementation was not inspected. Temporary pytest metadata fixtures are not experiment results; no persistent source/result/paper file was edited.

No real checkpoint, production model inference, model download, CUDA workload, raw image/label read, scoring run, queue/cohort audit, paper edit or assistant model override. The real four-image gates were inspected, not executed. Their actual pass/fail, numerical behavior on production checkpoints and historical freeze timing are not established by this CPU audit. The only intended persistent write is this report.

## Snapshot Identity

SHA-256 values for the inspected final snapshot:

```text
fc45c3628723e2b70e2ce63c32db924bfc7ece29489ff2112a26ed24282a6aff  trace_ovss/method/image_balanced.py
d6b2da3c4785971869363ee951a8e01af452d946bacf9e695be4684b172ffa75  tools/evolution_confirmation/run.py
348dcf7d2bfa81de5d137d67ab2d2c40dc363547ea28c907a2ad2a013bbe107b  tests/test_image_balanced_trace.py
6dd141455eb9f715c45407e683ebc9a5dea9f3d3e9e3de3e229b7f3b21d2ba0d  tests/test_confirmation_worker.py
8569e1f9a51ee0647fe365eef3be9902b46697b017db2c4990acae1e0388ca03  docs/confirmation-image-budget-plan.json
```

All line references above apply to these hashes. Unrelated concurrent changes were left untouched. A later source revision requires checking the affected findings against that revision; this report does not authorize modifying a frozen experiment snapshot.

## Follow-Up: Exact Repair Verification

Date: 2026-10-29. This append-only follow-up supersedes the open status of the three findings above for the repaired hashes below. The original findings and their original source references remain historical evidence; they do not describe the repaired implementation.

**Verdict: all three original findings are closed on the inspected repaired snapshot. No additional P1/P2 finding was identified within these exact repairs.** This is a bounded code/CPU verdict, not certification of a production GPU gate, experiment results, queue orchestration, cohort construction or manuscript claims.

### Original P1: Stage Inventory Barrier -- Closed

Repair: `tools/evolution_confirmation/run.py:406-433`, especially `:414-415`, `:420-421` and `:426-432`.

The worker now requires the exact stage output-key set supplied by `queue.cells()`, rejects resolved-path aliases, and binds every prediction protocol to its expected frozen manifest contents, hash and path. It separately checks stage, dataset, order and population against the expected cell. A repeated seal or manifest cannot replace a required dataset/order cell while retaining a valid inventory.

The narrowly inspected inventory helper is `tools/evolution_confirmation/queue.py:62-72`; this was a dependency check for this repair, not a full queue audit. New tests at `tests/test_confirmation_worker.py:179-208` cover valid calibration/confirmation inventories, alias rejection and wrong dataset binding. Additional pure-memory probes accepted each correct stage inventory and rejected six mutations per stage: lexical aliases, a duplicate semantic identity under the expected distinct output names, a changed manifest hash, a changed manifest path, changed manifest contents and a missing output. The old alias counterexample no longer passes.

### Original P2: Native Gate OOM Retry -- Closed

Repair: `tools/evolution_confirmation/run.py:93-96`, native reference use at `:537`, and configured call/batch assertions at `:551-556`.

`NativeGateReference` overrides only the diagnostic iterator with the fail-closed helper at `tools/evolution_followup/standalone.py:159-172`. That helper has no OOM recovery around either view construction or forwarding, so both exceptions propagate without a smaller-batch attempt. Direct method-identity checks confirmed that native adaptation, reset, candidate collection, view evaluation, objective preparation and objective evaluation remain inherited unchanged from `TwoViewTRACE`.

Both CPU failure-injection tests at `tests/test_confirmation_worker.py:220-238` passed: view construction is attempted once; forward failure performs one view construction and one forward, with no retry. Gate checks now additionally require the expected number of diagnostic calls and the unchanged configured view batch size. No CUDA allocation or real OOM was used.

### Original P2: Persistent Gate Attempt -- Closed

Repair: `tools/evolution_confirmation/run.py:494-500`, claim creation at `:515`, report linkage at `:519`, and model construction only afterward at `:526-528`.

The gate creates a persistent `.STARTED.json` claim before dataset/model execution. Its writer uses exclusive creation (`tools/evolution_followup/standalone.py:67-70`), so a competing creation cannot overwrite an existing claim even if existence checks race. An interrupted attempt leaves its claim present and a repeated attempt is refused; the worker does not remove that marker on failure.

The persistence/retry regression at `tests/test_confirmation_worker.py:211-217` passed with the report still absent. The exact queue integration requested for this repair was also checked: `tools/evolution_confirmation/queue.py:347-357` requires the report's `attempt_sha256` to match the claim and requires the claim's status/manifest binding to match the gate entry. The synthetic gate fixture at `tests/test_confirmation_queue.py:129-136` creates that same claim/report link. Pure-memory checks accepted a valid linked receipt and rejected changed report claim hash, changed claim manifest hash and changed claim status. No full queue test suite or queue execution was run.

### Follow-Up Verification And Limits

The same bounded test command recorded above was rerun with the specified Python, `CUDA_VISIBLE_DEVICES=''`, bytecode/cache writing disabled and one CPU thread per numerical runtime. Result: **48 passed, 1 deselected in 1.68 s**. The image-file loader test remains deliberately deselected under the original image-read restriction; its source was not changed by this audit.

The supplementary probes used synthetic in-memory manifests/receipts and normal unit-test fixtures, not production prediction artifacts. No raw image/label content, checkpoint or production model was read or run. No GPU gate, queue orchestration, cohort audit, process-kill experiment or multi-process race experiment was performed. The exclusive-create and pre-model ordering conclusions are supported by the inspected control flow and passing marker-persistence regression. The only persistent edit made in this follow-up is this appended report section.

Repaired snapshot SHA-256 values:

```text
9c5ece8ce6fd34f5634424b53156effa93c79e483c970534eb2a1122fc1d910a  tools/evolution_confirmation/run.py
8e2a8ffc8456b5dfa21a5f49cb8d37bc1f80f46a80c7f965f4bd21f0dde33d34  tests/test_confirmation_worker.py
c25342f3f868fb4e485dd1784d3e8da061ba31b48a0ac01001bc1457163464dc  tools/evolution_confirmation/queue.py
13fbf8d7db9756b00ceae4b3290cd8968074cb19eb65bbf54dedca08ea7cf0b2  tests/test_confirmation_queue.py
```

Only the inventory/claim-validator and synthetic claim-fixture snippets of the latter two files were reviewed; their full-file hashes identify the snapshot, not full-file review coverage. `image_balanced.py`, `test_image_balanced_trace.py` and the frozen design plan retain their original hashes recorded above. This closure applies to the exact repairs at these hashes and does not authorize modifying frozen results or launching new predictions.
