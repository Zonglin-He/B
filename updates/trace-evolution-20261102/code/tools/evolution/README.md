# Fixed-candidate two-view controls

Entrypoint: `tools/evolution/run_cell.py`. The base is the production
`trace_ovss.method.two_view.TwoViewTRACE`; production files are not patched.
The old `make_algorithm_class(TRACE, 'quality_only')` is only an optional
equivalence reference, not the policy execution base.

## CLI and controller contract

```bash
python -B tools/evolution/run_cell.py \
  --package-root /path/to/frozen/runtime \
  --data-root /home/wwww/hzl/data --dataset coco_cd \
  --output /path/to/fresh/cell --seed 0 \
  --policies trace uniform confidence target counterfactual mean permuted \
  --permutation-seeds 0 1 2 --max-images 4 --verify-reference --phase run
```

Optional `--data-dir` overrides discovery only, not adaptation settings.
COCO Object commonly needs `--data-dir /home/wwww/hzl/data/coco_mlmp`.
`best_config` is copied read-only; there are no optimization overrides.
The default `--selection-seed 20261024` matches `suite_inputs.expected_inputs`:
select a random subset, then sort its dataset indices. Complete runs omit
`--max-images` and require a controller-written `expected_panel.json`.
Without that parent file, only explicitly bounded standalone smoke runs are
permitted. The parent panel must include the selected `image_sha256` map.
Seed, cohort, labels, metadata, vocabulary, config and image hashes are strict
bindings, not fields the worker may silently correct.

Pre-created cell directories containing `run.log` and `expected_panel.json`
are supported. Existing worker-owned payloads/protocols/results are rejected.
No overwrite, resume, automatic retry, queue, GPU service or background work
is implemented. The controller owns GPU scheduling and locks.

`--phase predict` completes all predictions and writes
`PREDICTIONS_SEALED.json` before returning. `--phase score` verifies that seal
before reading any GT bytes. `--phase run` does those two steps in order.
Scoring emits `result.json` and `SCORED_SEALED.json`. The result contains
`images/full_split/seed/dataset`, `pre_result` (policy_label `pre`), and
`policy_results` with `policy/policy_label/permutation_seed/miou_unrounded`
plus intersection/union arrays. `permuted` expands to three named policies,
so the seven requested policy names produce nine results, not seven.

Prediction startup fixes `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA
initialization and enables deterministic algorithms in error mode, with
cuDNN deterministic enabled and benchmark disabled. A conflicting environment
or already-initialized CUDA fails rather than silently changing the protocol.
SDPA backend enablement and TF32 settings are not overridden. The actual
flags and cuBLAS environment are recorded in the protocol and initial-state
environment, and checked again before prediction sealing. Seed repeats under
this execution mode measure replay stability, not independent-sample uncertainty.

## Scientific invariants and files

Each episode diagnoses candidates and T/C once. Every policy independently
resets the same model and Adam state, restores the same RNG state, and changes
only rho. Actual trainable tensors, optimizer membership/state, frozen tensor
versions, candidate identities/masks/budget, and permutation weight multisets
are checked. The full selected set is fixed; EPS-active loss membership can
change as intended when weights change. Permutations are over the complete
episode, not semantic-role groups; their seeds are not training repeats.

The image-only dataset enumerates annotation paths but never opens masks.
It decodes image bytes bound to parent hashes and uses native resizing,
normalization and patchification. Offline scoring streams one image at a
time and stores per-episode intersection/union for clustered analysis.

- `predictions/pre/INDEX.npy`: shared reset-model labels.
- `predictions/POLICY_LABEL/INDEX.npy`: independently adapted labels.
- `episodes/EPISODE.npz`: token masks, global image index, episode patch
  index, query index, confidence, T/C/q, policy-by-candidate rho and donor IDs.
- `episodes/EPISODE.json`: image/patch metadata, policy order, reset assertions,
  candidate fingerprint and shared pre/diagnostic timing/work.
- `cost.jsonl`: per-policy adaptation/evaluation timing and ExecutionAudit
  counters. Shared diagnosis is not a standalone method-cost measurement.
- `checkpoint_identity.json`: official existing ViT-L/14 file identity,
  SHA256 computed with streaming reads once before loading. Loading uses the
  verified local path, so there is no download or extra vendor hash pass.
- `initial_state.json`: actual model/optimizer identity and runtime environment.
- `prediction_protocol.json`: source/resource hashes, frozen inputs and parent
  panel SHA, configuration, vocabulary, policies and checkpoint identity.
- `scores_by_episode.jsonl`: streaming per-episode, per-policy counts.

`--verify-reference` compares the actual complete episode against the old
quality_only wrapper and the new entry point, with independent resets.
It records candidate equality, maximum support/loss differences and changed
label pixels. CPU requires exact equality; GPU uses explicit
`--reference-atol` (default 1e-5) and `--reference-max-label-fraction`
(default 0, exact labels). A failed gate preserves the report and fails the
cell. Reference-check runtime is not included in mechanism-policy cost.

This is a joint fixed-candidate allocation experiment, not a singleton-weight
causal experiment. Implementation/CPU tests do not establish full-dataset
performance, statistical superiority or GPU equivalence.

## CPU tests

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -B -m pytest \
  tools/evolution/test_run_cell.py tests/test_two_view_entrypoint.py \
  tools/evolution/test_deterministic_runtime.py \
  tests/test_execution_audit.py tests/test_algorithm_variants.py \
  -q -p no:cacheprovider
```

Only existing torch, numpy, Pillow, pytest and repository code are needed.
Tiny models are injected; tests do not load public checkpoints or use CUDA.
