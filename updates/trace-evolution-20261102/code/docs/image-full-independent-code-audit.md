# Independent CPU-Only Code Audit: Image Full

Status: CPU-ONLY CODE AUDIT PASS for the source hashes below, rechecked
2026-10-29 03:30 Asia/Shanghai. **No unresolved P1 or P2 findings.** All four
initial P1 findings and the optimizer-device residual in F3 are fixed and
independently rechecked. Main may proceed with its own CPU preparation review;
this is not evidence of completed real gates, full prediction or service success.
Scope: frozen full plan, new queue/worker and focused CPU tests. Not a manuscript
rating, not a repeat of the completed 48-result importer audit. Only this document
is owned by this reviewer. No GPU work, preparation, prediction, source edits,
historical-result writes or Git commits are authorized by this review.

## Findings And Rechecks

### F1 [P1, FIXED AND RECHECKED] Selection proof rejects the actual sealed parent

- Location: `tools/evolution_image_full/queue.py:123-129`.
- `selection_proof_files(SELECTION_ROOT)` raises `RuntimeError: Completed suite
  proof drift` against the pinned parent, without preparing or launching anything.
- The new validator adds every calibration score subartifact to `files`, then
  requires each to be directly indexed by the top-level `SUITE_SEALED.json`.
  The actual parent binds `SCORED_SEALED.json` and `result.json` there, while
  `SCORING_STARTED.json`, `score_inputs.json`, and `scores_by_episode.jsonl`
  are transitively bound by each `SCORED_SEALED.json`. All six missing direct
  entries have expected value `None`; this is not evidence of parent drift.
- Required fix: verify the existing transitive seal chain without altering any
  historical artifact. Retain exact selection/prepared pins and recomputation of
  the calibration-only C choice.
- Required CPU regression: the real pinned proof succeeds, while changing a
  transitively sealed score subartifact in a temporary copy fails.
- Recheck: author restricted top-level checks to actual directly sealed members
  while retaining `verify_selection_record()` for the transitive chain. Calling
  `selection_proof_files(SELECTION_ROOT)` now succeeds and returns 15 small proof
  files. The author's real-proof/transitive-tamper regression now passes in the
  combined suite; no historical artifact was rewritten for this fix.

### F2 [P1, FIXED AND RECHECKED] Canonical-string check rejects three inherited COCO protocols

- Location: `tools/evolution_image_full/queue.py:98-101`.
- Real `read_parent()` succeeds for PC60, VOC21 and VOC20, but rejects
  `coco_object`, `coco_cd`, and `coco_stuff` with
  `RuntimeError: Duplicate/noncanonical parent paths`.
- Their sealed RGB paths are unique absolute paths through dataset-directory
  symlinks. For example, `data/coco_mlmp/images/val2017/000000000139.jpg`
  resolves to `data/coco/val2017/000000000139.jpg`. All 5000 COCO Object,
  500 COCO-CD and 5000 COCO Stuff rows have this property. Requiring original
  strings to equal `resolve()` rejects the exact frozen protocols.
- Required fix: preserve the exact sealed path strings/order and allow these
  inherited symlink aliases; check physical uniqueness/identity separately where
  needed. Do not rewrite parent protocols or replace them with current discovery.
- Required CPU regression: all six actual pinned protocols pass; duplicate input,
  changed config/order/checkpoint/parent pin still fail.
- Recheck: author preserves sealed absolute strings and checks physical
  `(st_dev, st_ino)` uniqueness instead. Actual `read_parent()` now succeeds for
  all six datasets at their exact locked populations. Real six-parent,
  duplicate/order and pinned-parent mutation tests pass in the combined suite.

### F3 [P1, FIXED AND RECHECKED] Raw nonfinite forward values do not fail closed

- Location: `tools/evolution_image_full/run.py:215-238` and
  `tools/evolution_image_full/run.py:491-519`.
- The worker checks returned losses and integer predictions only. Inherited
  estimator/adaptation/diagnostic helpers sanitize raw logits with `nan_to_num`;
  final argmax also turns nonfinite logits into valid-looking integer labels.
- CPU reproduction used the real `execute()` and tiny DenseCLIP fixture, two
  synthetic RGB images, all three arms, no annotation reads. Injecting NaN into
  `model.forward` at each of initial readout, diagnostic readout, adaptation, and
  final evaluation separately was ACCEPTED in all four cases. Every execution
  returned aggregate costs/timing and wrote six prediction NPY files. The
  existing loss/counter/reset/candidate checks did not reject these failures.
- Required fix: observation-only finite guards before sanitization/argmax, for
  both worker and direct gate reference, plus adaptation gradients/parameters
  before they can contaminate an update. Preserve finite-input mathematics and
  do not add encoder forwards or hide guard cost from the stated timing scope.
- Required CPU regressions: inject NaN and Inf at every raw forward phase and
  nonfinite trainable gradients; assert failure before prediction/suite seals,
  no retry, and retained attempt/failure evidence. Finite-path identity and
  exact 0/0/1 measured schedules must remain unchanged.
- Partial fix independently rechecked: new observation-only model hooks reject
  all four raw-NaN injections before `execute()` returns, retaining
  `PREDICTION_FAILED.json`. No production math files were changed. Guards are
  inside timed calls, add no forward, and apply to both worker and reference.
- **Residual discovered in the first fix, now also fixed**: at the reviewed
  intermediate `tools/evolution_image_full/run.py:224-227`, Adam state was filtered with
  `v.device == method.device`. Production constructs the method with `"cuda"`,
  so `method.device` is `torch.device('cuda')`, whereas actual CUDA state tensors
  have indexed `torch.device('cuda:0')`; these compare unequal. The entire
  device-resident optimizer state can therefore be silently excluded.
- CPU reproduction needs no CUDA: use `method.device=torch.device('cpu:0')`
  while the real tensors are on `cpu`, register an earlier optimizer post-step
  hook that fills `exp_avg_sq` with Inf, then execute two synthetic inputs through
  all three arms. With `device='cpu'` the guard rejects the Inf; with `cpu:0`
  it ACCEPTS and returns normally. This is the same indexed/unindexed equality
  mismatch as the formal CUDA path. Finite parameters do not imply finite
  second-moment state.
- Required correction: inspect optimizer tensors using their actual parameter
  device or device-grouped finite reductions, not equality to an unresolved
  device spec. Include an indexed/unindexed CPU regression and Inf Adam state
  with finite parameters. Do not allocate CUDA tensors to test this.
- Final correction/recheck: `require_finite()` groups tensors by their actual
  devices; `after_step()` includes ALL tensor-valued optimizer state, including
  CPU scalar step state. Independent full `execute()` reproductions now reject
  Inf second moments for BOTH requested `cpu` and `cpu:0` and retain
  `PREDICTION_FAILED.json`. Six persisted tests cover both requested devices and
  `exp_avg`, `exp_avg_sq`, and `step`. Guarded/unguarded finite trajectories,
  parameter digests, losses, logits and actual work counters remain identical.

### F4 [P1, FIXED AND RECHECKED] Selected-C source continuity is not enforced

- Location: `tools/evolution_image_full/queue.py:112-135`,
  `tools/evolution_image_full/queue.py:174`, and
  `tools/evolution_image_full/queue.py:252`.
- Selection pins authenticate the old selection/prepared records, but the new
  runtime snapshot is only compared with its own newly recorded hashes. No
  check binds the executing `trace_ovss` math, `ExperimentTRACE`, candidate,
  preprocessing and reconstruction helpers to the selected suite's sealed
  `runtime_hashes`/source identities.
- A changed image-balanced objective or candidate implementation can therefore
  be freshly snapshotted and admitted under the old C selection. A direct gate
  against the same changed `ImageBalancedTRACE` does not detect this drift.
- Current relevant source hashes happen to match; this finding concerns missing
  enforcement, not evidence that an actual mathematical change already occurred.
- Required fix: compare the mathematical/data/reconstruction dependencies to
  the pinned selection-prepared inventory during prepare and admission. New
  orchestration and observation-only guards may be explicit, documented
  additions; do not broadly exempt the inherited method or silently replace the
  selected implementation.
- Required CPU regression: mutate one selected objective/candidate/data helper
  hash before snapshot/admission and reject it even if the new runtime's own
  self-hashes and direct-current-code gate agree. Allow only explicitly reviewed
  new orchestration/observation files.
- Recheck: `validate_source_continuity()` compares 61 inherited dependencies
  against the pinned selection-prepared inventory before snapshot, after snapshot
  and in prepared/worker admission. New full orchestration/observation code is
  separate; inherited production math and helpers remain byte-identical.
- Independently mutated inventory hashes for image-balanced objective, candidate
  collection, dataset loading, reconstruction worker, selected worker and prior
  helper: all six were rejected with `Selected-C inherited source continuity
  failed`. Current unchanged inventory passes.

## Evidence So Far

- Frozen plan SHA256 is
  `598bc2715e959fc03421f28748ffe706522ec5c7757ca9fedb03aee959952dd8`,
  matching the queue constant.
- Six populations are 5105/5000/1449/1449/500/5000 in locked order; VOC20 uses
  the explicit C4 recovery parent and the others use A18.
- Current `trace_ovss` files and selection worker/helper files match the hashes
  recorded by the selected-C suite's `SUITE_PREPARED.json` at this read.
  Enforced selection-source continuity remains under review.
- Reproductions used `CUDA_VISIBLE_DEVICES=''`, Python `-B` and
  `PYTHONDONTWRITEBYTECODE=1`. Reads were limited to protocols, metadata, proof
  files and source. No prediction payload directory scans were performed.
- The worker was initially still being authored. Its temporary absence was not
  a finding. Worker and completed risk tests have now been reviewed and run.
- Existing CPU suites `tests/test_confirmation_worker.py` and
  `tests/test_image_balanced_trace.py`: **49 passed** (1.70 seconds pytest time).
  They are inherited regression evidence, not a pass of the new full runtime.
- New `tests/test_image_full_worker.py`: **51 passed** (1.92 seconds pytest
  time). This does not cover the four-phase raw-nonfinite reproduction above.
- Interim author-test update (not the final acceptance run): combined full
  queue/worker plus inherited worker/objective tests returned **141 passed,
  12 failed**. The 12 new raw-nonfinite tests did not actually inject: at
  `tests/test_image_full_worker.py:263`, `logits.reshape(-1)[0] = nonfinite`
  mutates a copy for noncontiguous DenseCLIP logits. Independently checked
  initial/adapt/final strides, differing storage pointers and absence of any NaN
  in the returned tensor. Correct the test injection (e.g. direct multidimensional
  indexing or a contiguous copy) before using this suite for acceptance. The
  separate full-tensor model-output injection correctly exercises the guard.
- The author corrected injection to `logits[(0,) * logits.ndim] = nonfinite`.
  Re-run of the same four-file CPU suite: **153 passed in 4.93 seconds**. This
  includes raw NaN/+Inf/-Inf in four phases, worker/reference gradients and
  parameter guards, real selection/transitive tampering, real six parents,
  dependency drift, miniature six-gate/six-prediction/six-score queue, exact
  admission mutations and no-resume failures. That intermediate pass did not
  cover the then-open indexed-device omission and was not treated as clearance.
- Subsequent four-file CPU run: **172 passed in 5.04 seconds**, no failures or skips,
  including the corrected raw-injection tests, direct-reference phase injections,
  indexed/unindexed Adam-state tests and observation-only trajectory identity.
- FINAL five-file CPU run after author finishing edits: **237 passed in 22.76
  seconds**, no failures or skips. Added inherited confirmation-queue coverage.
  All six files in the reviewed-hash table below were hashed before and after
  that run and were byte-identical across the run. The final gate gradient
  completeness check and worker CLI failure receipts were read and reviewed.
- Read `docs/image-full-runtime-audit.md`; method, evidence availability,
  observation-only guards/timing, source continuity, symlink handling and
  execution boundaries agree with the reviewed implementation. Its CLI examples
  were not treated as authorization to prepare or launch a real suite.
- Independent CPU-injected OOM checks on both new worker and new direct C
  reference: view-construction OOM and encoder OOM each propagate on the first
  attempt, with no batch fallback and `_diagnostic` restored to false (4 cases).
- Native discovery order and vocabulary exactly equal all six pinned protocols,
  including the symlink paths rejected by F2. Only file names/split text and
  metadata were used for this additional check, not annotation contents.
- Independent `stage_barrier()` CPU mutation checks: exact valid six-cell map
  accepted; missing VOC20, aliased cell key, wrong prepared hash, GT-open marker
  and changed prediction seal each rejected (6 cases). Prepared admission and
  per-cell payload verification were mocked in these focused barrier tests;
  no historical prediction payload was read or rehashed.

## Raw-Nonfinite Reproduction

Run from the worktree with CUDA hidden and bytecode/cache writes disabled. This
uses the author's tiny CPU fixture and only a temporary synthetic output tree.
The initial reviewed implementation printed `ACCEPTED` for every phase below.
After repair, every phase must raise before `execute()` can return; a finite
control must continue to pass the actual cost and identity tests.

```python
import sys, tempfile
from pathlib import Path
sys.path.insert(0, 'tests')
from test_image_full_worker import method, config, batch_meta
from tools.evolution_image_full import run as w
import torch
for phase in ('initial', 'diagnostic', 'adapt', 'final'):
    m = method.__wrapped__()
    native = m.model.forward
    def injected(*args, **kw):
        out = native(*args, **kw)
        diagnostic = getattr(m, '_diagnostic', False)
        probe = kw.get('vision_out_type') == 'adaptive_weighted_mean'
        bad = ((phase == 'final' and kw.get('interpolate')) or
               (phase == 'diagnostic' and diagnostic) or
               (phase == 'initial' and probe and not diagnostic
                and not kw.get('interpolate')) or
               (phase == 'adapt' and kw.get('vision_out_type') == 'mean'))
        return (out[0] * float('nan'), *out[1:]) if bad else out
    m.model.forward = injected
    meta = batch_meta(2)
    data = [dict(img_patches=torch.randn(1, 3, 32, 32),
                 meta={k: v[i] for k, v in meta.items()}) for i in range(2)]
    manifest = dict(config=config(), max_images=2, panel=dict(
        query_to_class=None, classes=m.classes, ordered_ids=['0', '1']))
    with tempfile.TemporaryDirectory(prefix='image-full-cpu-audit-') as root:
        try:
            w.execute(m, data, manifest, Path(root))
            print(phase, 'ACCEPTED')
        except RuntimeError as error:
            print(phase, 'REJECTED', error)
```

## Reviewed Invariants

- Three-arm worker reuses selected `ExperimentTRACE` and source-image-normalized
  objective, not an episode-normalized or dual-min replacement. Worker/reference
  OOM behavior is an explicit fail-closed override; finite-input math is unchanged.
- Production loader is RGB-only and verifies each RGB hash at decode. Labels
  are only statted in preparation/gates/prediction. Score opens label contents
  only after `stage_barrier()` validates the exact six prediction cells.
- Gates precede all formal predictions. Gate marker uses exclusive creation;
  prediction output directory creation and scoring-start file preserve attempts.
  Queue execution marker/lock prevents resume; GPU lock is shared with prior
  queues and acquired without a waiting/retry loop.
- Exact manifests bind full parent membership/order/config/checkpoint/vocabulary,
  including the VOC20 C4 parent; current BEST_CONFIGS is not used as replacement.
- Actual encoder/forward counters enforce 0/0/1 schedules. Episode 0 remains in
  all predictions, pooled integer counts and work totals, and is excluded only
  from timing. No extra pre/T encoder forward is added; unmeasured T is null and
  initial full-map prediction is explicitly unavailable. Lossless native masks,
  source/crop/class identities, measured C and objective coefficients are saved.
- Pooled mIoU is recomputed from class intersections/unions and episode streams;
  score geometry/range is checked and per-label content hashes are retained.
- Model-side timer includes inherited reset/adapt/final reconstruction and CPU
  transfer; serialization, data loading and offline scoring are excluded by an
  explicit scope. Work reduction is not a blanket wall-time guarantee.
- `SUITE_SEALED` intentionally leaves `service_success_verified=False`; main
  must separately verify the genuine retained successful terminal service state.

## Final Verification

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wwww/miniconda3/envs/mlmp/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_image_full_worker.py tests/test_image_full_queue.py \
  tests/test_confirmation_worker.py tests/test_image_balanced_trace.py \
  tests/test_confirmation_queue.py
```

Reviewed hashes:

| File | SHA256 |
| --- | --- |
| `docs/image-full-plan.json` | `598bc2715e959fc03421f28748ffe706522ec5c7757ca9fedb03aee959952dd8` |
| `tools/evolution_image_full/run.py` | `6bacdc7e8eb1114867f83bfd35b40d2dfffb28767161316fa828b764a25c0e82` |
| `tools/evolution_image_full/queue.py` | `6cdf867cf94821634c186adf4e7b6a760af7a83cc274793614e8f56f60da12a9` |
| `tests/test_image_full_worker.py` | `8b3819ac13a93b8a907b28f44fe511b36828105f97c45ce153aad570216d345c` |
| `tests/test_image_full_queue.py` | `36c4be9b1e9f1edf18632d81c009061256da79181da1954bf3fe415d562e124b` |
| `docs/image-full-runtime-audit.md` | `e63471b2002628e5bfea482b1cbb3703ae74418b9566b625727980034298da4d` |

Real pinned selection proof (15 files), exact six parent protocols and 61
selected-source dependencies were reverified during the final recheck; real
selection/parent/continuity tests also pass in the final five-file run. No
manuscript score, new scientific result or historical-independence certification
is asserted. The completed-cohort importer audit remains separate and is not
duplicated here.

## Execution Boundary

This review made no changes outside this document, aside from disposable
synthetic CPU test outputs. No GPU, real-suite preparation, prediction, real GT
decode, process intervention, historical-result write or Git commit occurred.
Observation-only finite checks do not change production mathematics or add
forwards; their reductions/synchronization are inside model-side timing.

Main still owns: final main CPU pass, fresh prepared-runtime review/hash pin,
single worker/shared lock, retained systemd 12 GiB RAM / 1 GiB swap,
RemainAfterExit=yes, OOMPolicy=stop, Restart=no, six genuine four-RGB-image gates,
the complete 18-result global prediction-before-label barrier and genuine
successful retained terminal-state verification. CPU code clearance does not
substitute for any of these execution requirements. Changes to the reviewed
runtime or inherited-source inventory invalidate this source-specific clearance.
