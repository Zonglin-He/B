# Single-Probe Worker Audit

## Ownership And Invocation

Only `tools/evolution_tuning/worker.py`, its focused test and this audit are
worker-owned. No production method, old queue, paper, old results or commits
are changed. Development verification uses CPU tiny fixtures; no GPU runs.

```
python -m tools.evolution_tuning.worker --phase predict --manifest ABSOLUTE_JSON --manifest-sha256 SHA256 --output ABSOLUTE_DIRECTORY
python -m tools.evolution_tuning.worker --phase score --manifest ABSOLUTE_JSON --manifest-sha256 SHA256 --output SAME_DIRECTORY --barrier ABSOLUTE_JSON
python -m tools.evolution_tuning.worker --phase gate --manifest ABSOLUTE_JSON --manifest-sha256 SHA256 --output FRESH_GATE_DIRECTORY
```

The module must execute from the suite's frozen `runtime` copy. Admission
requires `runtime_root == suite_root/runtime == worker.ROOT`, relative source
hashes, canonical absolute manifest/output paths inside the suite, checkpoint
identity, explicit RGB hashes and annotation `[size, mtime_ns]` metadata.
Every imported managed source must be in the snapshot and under this root.
If manifest `output` is present it must equal the CLI directory exactly.
The protocol and seal always bind the exact CLI manifest hash/path and output.
Optional `config_path/config_sha256`, `inputs_path/inputs_sha256` and
`suite_prepared_sha256` bindings are checked when supplied.
Prepared manifests are also checked against the immutable input panels and
non-search settings. Full manifests bind their selected config and the complete
all-config selection inventory. RGB, checkpoint and provenance files cannot
alias annotation inodes; label metadata details are checked when supplied.

## Manifest And Barrier Integration

Schema version 1 and algorithm `ImageBalancedTRACE:C` follow the contract.
Dataset keys are `voc20`, `voc21`, `pc59`, `pc60`, `coco_object`, `coco_stuff`,
`coco_cd`, `waterbirds_seg`. Config is explicit, never read from current best
settings. Geometry is 224 square/stride 112 and view batch 16. Calibration
has exactly 256 rows and `[image_C]`; full uses exactly
`[no_adapt,image_uniform,image_confidence,image_C]`; gate has `[image_C]`
and at least four rows (only the first four RGB inputs are executed).
`ordered_ids` contains official IDs, in exactly the input-row order.

Barrier JSON has `complete: true`, matching `stage`, and
`prediction_seal_sha256: {ABSOLUTE_OUTPUT_DIRECTORY: SEAL_FILE_SHA256}`.
Calibration scope is exactly the current trial output. Full scope is exactly
the prepared runnable dataset outputs (all eight without a prepared scope) from this suite and runtime, with
the same source snapshot and optional prepared hash. Missing own output,
relative keys, duplicate datasets, cross-stage/root links, changed seals or
payloads fail before annotation bytes open. When manifests contain
`suite_prepared_sha256`, the barrier must contain that exact hash too.
All linked prediction payloads are hashed once, before scoring; the scorer
does not repeatedly rehash them per image or at the end.

## Execution Evidence

`ExperimentTRACE` is the existing confirmation class with C view mode.
RGB-only `make_data`, `label_predictions`, `finite_observation`, counting,
lossless `candidate_receipts`, and `streaming_counts` are reused. No global
ARMS/queue flags are mutated and no old six-dataset admission is called.
NoAdapt performs exact reset and one final evaluation, without `adapt`,
candidate selection or diagnostic calls. Adapt policies retain the existing
two evaluation calls, step count and 0/0/1 diagnostic view schedules.
Per-episode arm order alternates, with identical RNG states across policies,
exact reset verification between policies/episodes, and frozen-weight checks.

Prediction artifacts are `predictions/ARM/INDEX.npy`,
`episodes/EPISODE.ARM.json`, `cost.jsonl`, `prediction_protocol.json`,
`initial_state.json`, `prediction_result.json`, `PREDICTIONS_SEALED.json`.
Receipts retain all admitted candidate masks/scalars and objective weights.
Costs include actual calls/inputs and synchronized times, plus calibration
and full indicators. Timing includes reset and excludes the first episode
from timing aggregates only, never from counts or scores.

Scoring writes `SCORING_STARTED.json`, `scores_by_episode.jsonl`,
`score_inputs.json`, `result.json`, `SCORED_SEALED.json`. One annotation and
one policy prediction are scored at a time; pooled intersections/unions,
not episode mIoU averages, determine results. Label hashes are recorded only
after the prediction barrier, with metadata checks before and after reads.
No overwrite or resume is allowed, including after partial scoring.

OOM and nonfinite failures are separate ineligible-trial receipts with no
retry; other failures are fatal drift. Failure receipts retain manifest,
config, checkpoint, source provenance and available actual partial counters.

## Gate

`OUTPUT.STARTED.json` is an exclusive persistent attempt marker. Output is
a fresh directory containing `gate.json` and, only on success,
`GATE_PASSED.json` with `passed: true`; interruption or failure does not
permit another attempt at that output. Four actual RGB inputs use the
configured episode batch size. The experiment is compared against the
existing direct `ImageBalancedTRACE` reference, with observation/accounting
and fail-closed diagnostic batching only. Candidate masks must match exactly;
supports, objective coefficients, losses, every step gradient and final
trainable parameters must differ by at most 1e-5. Final label differences
must be zero. Actual one-view call/input counts and reset identities are
validated. No labels, accuracy, ground truth or selection scores enter gate.
The gate keeps both implementations' candidate receipts, loss trajectories,
initial-state identity and final parameter hashes alongside measured differences.

## Validation Boundary

CPU tests verify scheduling, reset, receipts, strict gate comparisons,
manifest/binding drift, exact barrier scope, streaming scoring and no retry.
Passing these tests is not a real-checkpoint GPU gate or an experimental
result. A future authorized real run must pass its own persistent gate.

Worker tests: 42 passed. Joint verification with new queue/input tests and
existing full-worker, confirmation-worker and image-balanced method CPU tests:
215 passed. The four-policy score fixture also passes the controller's actual
`validate_result` checks. No CUDA work or real-checkpoint run was launched.
