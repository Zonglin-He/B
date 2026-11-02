# Image Full Runtime Implementation Audit

## Status And Scope

Implementation and CPU verification only. No real suite preparation, CUDA
initialization, checkpoint load, prediction, evaluation-label read, GPU job,
service launch or Git commit was performed by this implementation task. Main
owns the plan, independent audit, handoff, automation and subsequent launch.
Independent audit clearance is separate from passing these CPU tests.

The fixed plan is `docs/image-full-plan.json`, SHA256
`598bc2715e959fc03421f28748ffe706522ec5c7757ca9fedb03aee959952dd8`.
The only owned files are the new `tools/evolution_image_full/` package,
`tests/test_image_full_worker.py`, `tests/test_image_full_queue.py`, and this
document. Existing mathematical sources, old runtimes/results, main checkout,
importer files, manuscript files and the independent auditor's report are not
edited by this task.

## Fixed Admission

| Order | Dataset | Full Images | Inherited Batch | Config Parent |
| --- | --- | ---: | ---: | --- |
| 1 | pc60 | 5105 | 9 | A18 mechanism |
| 2 | coco_object | 5000 | 14 | A18 mechanism |
| 3 | voc21 | 1449 | 16 | A18 mechanism |
| 4 | voc20 | 1449 | 1 | final C4 recovery mechanism |
| 5 | coco_cd | 500 | 28 | A18 mechanism |
| 6 | coco_stuff | 5000 | 5 | A18 mechanism |

Every dataset executes `image_uniform`, `image_confidence`, and `image_C`:
six full cells, 18 arm results, seed0, unchanged original official order.
There is no calibration stage, reselection, T/min arm, new baseline, retry,
resume, OOM fallback, parameter/threshold/prompt change or service management.
Schema checking never obtains config values from today's `BEST_CONFIGS`.

Canonical parents are pinned by protocol and prediction-seal byte hashes in
the new queue. A18 is
`/home/wwww/hzl/results/trace_self_evolution_20261028_1338/cells/mechanism/{dataset}`;
VOC20 is
`/home/wwww/hzl/results/trace_evolution_voc20_cli_recovery_20261028_2312/cells/mechanism/voc20`.
There is no `full_seed0` parent. Original protocol JSON is frozen; its large
prediction seal/payload is not copied. A small parent receipt carries the
pinned old seal hash, complete/GT-free scope and sealed protocol hash.

The exact inherited input path strings are retained, including COCO directory
symlinks. Unique absolute names, unique physical `(device, inode)` identities
within each dataset, original sorted order, official discovery membership,
RGB hashes and inherited label-stat metadata are checked. No label bytes are
needed for these checks. Parent JSON includes all RGB hashes and label stats.

## Selection And Source Continuity

Selection parent:
`/home/wwww/hzl/results/trace_image_budget_20261029_0233`.
`SELECTION_SEALED.json` is pinned to
`90fa3a20097698e9b83cafd416a4b05fe7f033dcd8aadb027c970a9788ac11e4`;
its prepared parent is pinned to
`579372bf5a4208ad9e1f465698b0d82193ecd8d7c70162b3b1b3c67c84f056c5`.
The completed old suite must bind these records and the calibration seals.
The old `verify_selection_record` helper recomputes the locked choice using
only the two calibration results and the old queue's selection rule. No
confirmation result is opened for selection and no new selection is made.

Exactly 15 small proof files are frozen. Top-level completed-suite bindings
are verified where present. `SCORING_STARTED.json`, `score_inputs.json` and
`scores_by_episode.jsonl` are authenticated transitively through their
calibration `SCORED_SEALED.json`, not incorrectly required to appear directly
in the completed suite's inventory. Historical files are never modified.

Before snapshot and during new prepared-suite admission, all 61 selected
dependencies in `trace_ovss/`, `tools/evolution/`,
`tools/evolution_confirmation/`, `tools/evolution_followup/`, plus
`tools/suite_inputs.py`, `tools/trace_algorithm_variants.py` and
`tools/trace_execution_audit.py`, must match the pinned selection-prepared
runtime inventory. Extra files in those inherited trees are rejected too.
This covers mathematical code, candidates, data discovery/preprocessing,
vocabulary/prompts, model, objective, reconstruction and scoring. A new
self-consistent snapshot and a direct-current-code gate cannot excuse a
changed inherited dependency. Only the new full orchestration, tests and
observation hooks are additions; there is no inherited-math exemption.

## Worker, Gates And Finite Guards

`ExperimentTRACE` is an alias of the actual class in
`tools.evolution_confirmation.run`; its estimator, objective, adapt/reset and
reconstruction are not copied or rewritten. Proven image-only loader,
counter, alternating order, timing, GPU-lock and bounded child-log helpers
are reused without global or class monkeypatching.

There are six independent persistent GT-free gate attempts before any full
prediction. Each uses the first four inherited RGB inputs, grouped using the
unchanged config batch size (thus four singleton episodes for VOC20, a
four-image partial batch for the others). This is not a claim that four
images exercise a 28-image real GPU batch. CPU tests separately exercise
all six inherited full batch sizes with the real tiny model and all arms.

The C gate compares against direct `ImageBalancedTRACE(view_mode="C")`, with
only fail-closed diagnostic batching and observation counters added. The
two objects share model storage but have independent optimizer/snapshot
state; no `__class__` mutation is used. Every compared episode restores the
same initial weights/optimizer and RNG, then compares candidate masks and
identity, detached supports, objective coefficients, all losses, every
step's trainable gradients, final category labels and reset state.
Support/coefficient/loss/gradient tolerance is the fixed `1e-5`; changed
label pixels must be zero. Actual 0/0/1 diagnostic schedules are validated
for all controls/C. Empty-candidate gates cannot pass vacuously.

`finite_observation` uses temporary, removed-on-exit PyTorch hooks on both
worker and direct reference. A model forward hook checks the raw model
logits/features before adapter/helper `nan_to_num` or final argmax in initial,
diagnostic, adaptation and final-readout phases. Optimizer pre-step hooks
check all trainable parameters and gradients before an update; post-step
hooks check parameters and every tensor-valued Adam state, including step.
All state tensors are grouped by their actual tensor device, not compared
against the potentially unresolved requested `method.device`. Reductions
stay on device, followed by one `.item()` per actual device and phase,
never one per parameter. Gradient comparison is also device-aggregated.

These are observation-only abort guards: no replacement, clipping, alternate
math, extra encoder forward or fallback. They run inside timed
adapt/evaluate calls. Hook setup and disk writes remain outside model-side
timing. Finite guarded and unguarded CPU trajectories and work counts are
tested identical. Nonfinite failures retain `PREDICTION_FAILED.json` and/or
gate/worker failure evidence and cannot produce a prediction/suite seal.

## Evidence And Scoring

Native final predictions are retained for every arm/image, including episode0.
Every candidate receipt retains native scalar stats, confidence, measured C,
null unmeasured T, reliability, objective coefficient, image/crop/query IDs
and lossless boolean region. Regions use row-major `packbits`, little-bit
order, hex bytes, original shape and true-pixel count. Packing/serialization
and measured writes occur after the synchronized model timer.

The recorded pre-update *candidate* category is its native query ID mapped
through the inherited class-extension mapping. It is not a whole-image
pre-update prediction. `PREUPDATE_STATUS` explicitly marks full pre-update
category masks unavailable. No extra pre-category encoder forward or
unverified initial-logit reconstruction is introduced. This evidence supports
offline candidate-footprint analysis, not whole-pre-image outcome claims.

All six datasets/all three arms must seal their predictions before
`ALL_FULL_PREDICTIONS_SEALED.json` can be written. Scoring checks that exact
six-cell inventory, every immutable manifest/root/prepared binding and every
prediction seal before opening any evaluation annotation bytes. Score jobs
are separate CPU-only children. Intersection/union counts stream by episode
and pool by class; they are not averages of episode mIoUs. All annotation
byte hashes used by scoring are sealed separately.

Actual encoder/forward input and call counters are checked per episode and
recomputed from bounded JSONL. Only episode0 is omitted from timing, never
from predictions, scores or work counts. State remains a small finite phase
record; logs retain at most the inherited 256 KiB tail, source snapshot at
most the inherited 64 MiB, and JSONL lines at most 64 KiB. No old prediction
payloads are copied. A failed attempt is terminal and must not be resumed.

The new suite seal means six gates, six pooled full results and 18 CSV rows
were independently validated by the queue. It explicitly records
`service_success_verified=false`: main must separately verify the retained
successful terminal systemd state. The queue never substitutes artifact
completion for service success or scientific superiority.

## CPU Verification And CLI

Latest combined regression run: **237 passed in 22.60 seconds**, with
`CUDA_VISIBLE_DEVICES=''`, `-B` and `PYTHONDONTWRITEBYTECODE=1`. This is CPU
implementation evidence, not passage of the six real-checkpoint gates.

Run CPU regressions from the owned worktree:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/wwww/miniconda3/envs/mlmp/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_image_full_worker.py tests/test_image_full_queue.py \
  tests/test_confirmation_worker.py tests/test_image_balanced_trace.py \
  tests/test_confirmation_queue.py
```

Coverage includes the real six parent protocol/seal pins and real selection
proof (read-only, skipped only on hosts without these local artifacts),
transitive proof corruption, selected-source mutations, exact full versus
old cohort scope, three-arm full-batch work counts, direct C identity,
lossless mask retention, raw NaN/+Inf/-Inf injection in all four forward
phases for worker/reference, nonfinite gradients/parameters/Adam state,
indexed/unindexed requested devices, finite-trajectory identity, one-device
finite-flag synchronization, GT-free inference, missing/mutated barriers,
early scores, exact inventories, persistent markers, terminal child failures,
18-row synthetic completion and pooled score/cost/warmup validation.
Synthetic orchestration uses temporary fixtures and never calls real prepare.

Commands for main only after independent code-audit clearance; the following
were not executed here. `NEW_SUITE` must be a fresh canonical absolute path:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/wwww/miniconda3/envs/mlmp/bin/python -B -m tools.evolution_image_full.queue \
  prepare --output /absolute/NEW_SUITE

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/wwww/miniconda3/envs/mlmp/bin/python -B -m tools.evolution_image_full.queue \
  verify --suite /absolute/NEW_SUITE --suite-sha256 REVIEWED_PREPARED_SHA256

cd /absolute/NEW_SUITE/runtime
/home/wwww/miniconda3/envs/mlmp/bin/python -B -m tools.evolution_image_full.queue \
  run --suite /absolute/NEW_SUITE --suite-sha256 REVIEWED_PREPARED_SHA256
```

Main must launch the last command in the specified shared-GPU environment
and retained systemd service: 12 GiB RAM, 1 GiB swap, `RemainAfterExit=yes`,
`OOMPolicy=stop`, `Restart=no`. The worker uses the existing exclusive shared
GPU lock and deterministic runtime. No launch or clearance is implied by
this command listing or by CPU test success.
