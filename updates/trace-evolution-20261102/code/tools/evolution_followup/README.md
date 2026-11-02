# Standalone Probe Comparison

`standalone.py` is a new bounded experimental worker, independent of the old
shared-diagnosis allocation measurements. It does not modify the scientific
engine, `probe_audit.py`, provenance documents, or manuscript. No run is implied
by installing or reviewing it. GPU prediction requires a separate explicit
invocation after the main reviewer freezes the manifest.

## Arms and Scientific Invariants

| Arm | Diagnostic views per candidate | Detached allocation |
| --- | ---: | --- |
| `min` | 2 | min(T, C), native `TwoViewTRACE` evaluation |
| `T` | 1 | preserved-only support T |
| `C` | 1 | context-perturbed-only support C |
| `Uniform` | 0 | one for every admitted candidate |
| `Confidence` | 0 | initial candidate mean class probability |

Each arm independently executes episodic reset, initial image readout, candidate
construction/selection, its requested views, adaptation and final inference.
No arm borrows another arm's diagnosis or weights. All arms use identical RGB
episode tensors, initial model/optimizer state, RNG state, query vocabulary,
valid geometry, and the inherited candidate collector and mean-region entropy
objective. Candidate identity/mask/confidence hashes must match exactly across
arms. Only visual LayerNorm affine parameters are adapted. There are no new
scientific parameters, floors, role rules, or tuning loops.

`min` delegates to native `TwoViewTRACE`; single-view arms use the exact native
weak-image/counterfactual-background helpers and mask/compositing expression.
The unused helper is not called. Zero-view arms build no diagnostic images.
Uncomputed T/C fields are JSON `null`, never zero or duplicated support. All
inactive allocation yields `normalized_weight: null`; an individual inactive
candidate in a nonempty allocation has normalized weight zero. The existing
engine's `EPS` admission threshold is unchanged. Batch packing differs across
view schedules and can produce small floating-point differences even with
identical pixels; matching single-view supports bit-for-bit is not guaranteed.

## Lock and Exposure Protocol

There are three separate commands, never an automatic predict-and-score mode:

1. `prepare`: CPU-only. Require a complete resolved config JSON, explicit model
   seed, subset selection seed, image bound, arm list/order, existing checkpoint
   and exposure declaration. Verify official evaluation population/discovery;
   hash RGB bytes, source/resources, config and official ViT-L/14 checkpoint.
   Read label paths/stat metadata only, not label contents. Write a new manifest
   exclusively. No checkpoint download or model initialization occurs.
2. Main reviews the manifest and freezes its SHA-256. `predict` requires this
   digest, revalidates the config/source/checkpoint/input panel, and obtains the
   shared nonblocking exclusive lock at
   `/home/wwww/hzl/results/reviewer_followup_gpu.lock` before initializing CUDA.
   A busy lock fails; the worker does not wait, retry, or create a private lock.
   Run outside a parent that already holds the same lock: inherited-lock mode
   is deliberately not implemented. Output must not already exist, even empty.
3. `score`: CPU-only. Rehash every prediction and episode receipt for **all**
   requested arms before the first label-content read. Stream independent
   intersection/union counts using the established `streaming_counts` helper.
   No old scores, timings, shared-arm predictions or diagnostics are loaded.

The hard operational bound is 1..128 images, always `full_split=false` for these
official populations. `selection_seed` chooses the fixed subset, sorted by
official sample index by default. Optional `--episode-order-seed` only permutes
those selected indices. The manifest/protocol record both seeds, the selected
membership and ordered official IDs; all arms receive the same order. Different
regroupings are not new splits or independent uncertainty replicates.

Arm order is explicitly `fixed` or `alternating`. Alternating reverses the
declared list on odd episodes, including the partial last episode. No order or
seed is selected from outcomes. Small panels and alternating order do not
eliminate all thermal/cache effects.

Every run records the caller's exposure note. The planned PC60/Object128
comparisons reuse evaluation images and may use descriptive regrouping. Neither
inherited-config timing nor common-config transfer is untouched validation.
No claim of significance, necessity or general superiority follows from this
worker. Any statistical resampling must use complete episodes, with regroupings
treated as dependent observations of the same images.

## Explicit Configuration

The JSON must contain **exactly** the resolved inherited config keys:

```text
dataset, data_subdir, data_dir, resize, patch_size, patch_stride,
batch_size, lr, steps, trace_max_regions, class_extensions, trace_view_batch_size
```

Use an absolute `data_dir`. `resize` and `patch_size` are two-element JSON arrays.
No `best_config` call supplies runtime values. The worker validates the dataset
identity and its existing extension convention, then uses only the supplied
config. The main reviewer must extract/freeze the intended inherited settings
from the authoritative prior protocol, not silently substitute today's defaults.

For the separately prelocked common-config transfer comparison set precisely
`batch_size=2`, `lr=0.001`, `steps=10`, `trace_max_regions=16`, and
`trace_view_batch_size=16`. Keep each dataset's own dataset ID, preprocessing,
vocabulary and extension convention. Freeze a separate config and manifest for
each dataset/config/order condition. Model seed 0 and the subset selection seed
are distinct; do not confuse seed 0 with selection or regrouping seeds.

## Usage After Main Review

Commands below are templates, not executed experiments. Use a fresh manifest
and output for every condition. Substitute the actual frozen paths, approved
selection seed and manifest digest. Do not search seeds.

```bash
cd /home/wwww/.codex/worktrees/trace-simplification/hzl

# CPU-only preparation. Repeat with coco_object and its own explicit config.
CUDA_VISIBLE_DEVICES='' python -B -m tools.evolution_followup.standalone prepare \
  --dataset pc60 --config /absolute/pc60-inherited.json \
  --seed 0 --selection-seed 20261024 --max-images 128 \
  --arms min T C Uniform Confidence --arm-order alternating \
  --checkpoint /absolute/existing/ViT-L-14.pt \
  --exposure-note 'Reused evaluation images; inherited-config descriptive cost comparison' \
  --manifest /absolute/new-pc60-inherited-manifest.json

# For a separately predeclared regrouping add --episode-order-seed APPROVED_SEED
# to prepare. It keeps precisely the same selected 128 images.

# GPU COMMAND: only after main review/freeze, never during implementation.
python -B -m tools.evolution_followup.standalone predict \
  --manifest /absolute/new-pc60-inherited-manifest.json \
  --manifest-sha256 REVIEWED_SHA256 --output /absolute/new-pc60-inherited-output

# CPU-only, separate process, only after every requested arm is sealed.
CUDA_VISIBLE_DEVICES='' python -B -m tools.evolution_followup.standalone score \
  --output /absolute/new-pc60-inherited-output
```

## Timing and Artifacts

Wallclock uses CUDA synchronization before and after each complete independent
arm. The interval includes reset, initial diagnosis, candidate selection, view
construction/forwards, optimization, final inference, reconstruction and CPU
prediction transfer, with accounting hooks active. It excludes model setup and
loading, shared RGB decoding/H2D/geometry preparation, post-run provenance checks,
prediction/receipt disk writes and offline scoring. It is model-side standalone
episode cost, **not** end-to-end data-pipeline latency.

Exclude only episode 0 per arm from timing aggregates, not one image, one step,
or the first observed diagnostic batch. Retain its raw timing/counters,
predictions and scoring counts. For a one-episode run all aggregate timing
rates are `null`. The seconds/image denominator is actual measured images,
including a partial last episode, not batch_size multiplied by episode count.

- `prediction_protocol.json`: reviewed manifest, external manifest digest,
  ordered IDs/seeds, runtime determinism and shared-lock receipt.
- `initial_state.json`: initial model/optimizer hashes, actual trainable names,
  runtime/device versions and numerical backend settings.
- `cost.jsonl`: one record per episode/arm, wallclock, candidates, active count,
  diagnostic encoder calls/inputs, total encoder calls/inputs, and disjoint
  non-diagnostic eval, diagnostic eval, adaptation forward calls/inputs. Explicit
  total eval counts include diagnostic eval; total forward counts include every
  initial/final eval, diagnostic eval and adaptation forward.
- `episodes/NNNNNN.ARM.json`: bounded per-episode receipt, candidate identity,
  support/null fields, allocations/normalized weights and optimization losses.
- `predictions/ARM/NNNNNN.npy`: streamed uint16 class predictions; one image at
  a time on disk. No large per-image JSON or accumulated prediction arrays.
- `PREDICTIONS_SEALED.json`: full cross-arm artifact inventory/hashes; no GT
  content has been opened by the worker at this boundary.
- `scores_by_episode.jsonl`, `score_inputs.json`, `result.json`,
  `SCORED_SEALED.json`: fresh offline scores, label hashes, timing summary and
  linkage to the all-arm prediction seal. Accuracy includes warmup.

Determinism uses the existing strict runtime helper, not a new SDPA policy.
The manifest binds source/resources and config; startup/end checks fail on
changes. Software determinism is not a promise of equality across devices,
library versions or different view batch shapes. OOM and other failures stop;
there is no adaptive batch reduction, retry, resume or overwrite. A failed run
can leave unsealed outputs or a scoring-start marker for inspection; choose a
new output for a separately authorized rerun. No subprocess, service or queue
is launched. The label boundary is enforced by the RGB-only loader and phase
design, not an OS sandbox against arbitrary unrelated code in the process.

## CPU Verification

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q \
  tests/test_standalone_probes.py
```

Tests inject a tiny CPU model: true zero/one/two-view forwards, exact diagnostic
pixels and native `min` equivalence, identical candidates, missing/inactive/empty
supports, normalized weights and inherited mean-region entropy, explicit protocol
validation, ordering-only regrouping, source and RGB integrity, label-admission
rejection, no OOM retry, warmup denominators, and streamed seal/score integration.
They do not validate GPU memory, full checkpoint execution, real-dataset accuracy,
thermal stability or meaningful speed differences. Those remain pending main
review and an explicitly authorized future run.
