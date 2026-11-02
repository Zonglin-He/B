# Image-Balanced TRACE

## Status and Scope

This is an explicit new algorithm, not an engineering-equivalent implementation
of episode-normalized TwoViewTRACE or StandaloneTRACE. Its scope is the objective
normalization and fixed diagnostic mode. No mIoU gain is assumed or promised.
Implementation and validation here are CPU-only, with no benchmark scoring.
The existing method, CLI, configurations, paper and manifests remain unchanged.

## Objective

For candidate R, let h_R be its mean token/prompt entropy, q_R its detached
reliability, and image_idx its source image identity. Define I+ as source images
with at least one nonempty candidate whose q_R > EPS. Then

```text
L = (1 / |I+|) sum_{j in I+}
      [sum_{R: image_idx=j, active} q_R h_R
       / sum_{R: image_idx=j, active} q_R]
```

Each h_R is the existing mean over the original candidate's tokens and, when
present, the prompt axis. There is no pooling of all candidate pixels, no area
multiplier, no equalization of candidate counts, and no extra tunable scalar.
Candidate count and relative q still determine allocation *within* each image.
The total normalized coefficient mass of each active image is 1 / |I+|.

The implementation reuses TraceObjectiveMixin separately within each image,
including float-converted detached weights, the Python q > EPS membership test,
empty-region skipping and the double-precision denominator > EPS comparison
with fallback normalizer one. EPS remains the existing 1e-8. As in the old
method, an empty objective returns a graph-connected zero. The inherited engine
retains its existing finite-logit and finite-entropy guards. No new safety gate
or weight sanitization is introduced. Single-source loss and entropy gradients
are bitwise equal to the old objective in the tested float32/float64 cases.

## Public Interface

```python
from trace_ovss.method.image_balanced import (
    ImageBalancedTRACE,
    image_balanced_region_entropy,
    prepare_image_balanced_objective,
)

method = ImageBalancedTRACE(..., view_mode="min")  # or "T" / "C"
losses = method.adapt(crops, image_ids=source_image_ids)
```

One constructor-selected mode applies to every candidate; there is no candidate-
specific mode selection. T uses target support, C uses counterfactual support,
and min uses their minimum. T/C/min schedule exactly 1/1/2 diagnostic inputs per
candidate, respectively, not two inputs with one ignored. Diagnostic model call
count follows the inherited view batching (and inherited OOM retry behavior),
so diagnostic inputs and batched calls are distinct quantities. T/C never
construct the unused view. The min evaluator is inherited directly.

Candidate stats retain quality, reliability (q/rho), original_support,
target_support, counterfactual_support, residual_support, dominance, omitted,
diagnostic_algorithm and the original candidate fields. Unmeasured support is
None, never a fabricated zero. Min retains diagnostic_algorithm="two_view";
single-view modes report "T"/"C". Counts use the existing _two_view_counts,
last_reliability_summary["diagnostic_views" / "diagnostic_calls"], and
last_trace_view_batch_size interfaces.

last_region_terms retains batch_idx, indices and rho, adding mandatory image_idx
from active candidate stats. batch_idx locates a crop's entropy; image_idx groups
crops from the same source. Noncontiguous/negative source IDs are valid. Without
image_ids, the inherited collector treats each crop as a separate source. Callers
with multiple crops per source must pass the correct mapping.

image_balanced_region_entropy is a stateless, tensor-only loss helper with no
model or label access. It accepts entropy shaped [prompt, crop, H, W] or
[crop, H, W], and region terms with source image_idx, crop batch_idx, detached
scalar rho and tensor indices (or the existing mask fallback). Missing source
identity fails rather than silently treating crops as independent images.
prepare_image_balanced_objective caches the per-source terms and normalizers;
cache reuse requires unchanged terms, device and dtype within an episode.

## Inherited Machinery and Controls

The subclass does not replace adapt, reset, candidate extraction/admission,
per-source top-K, connected-component geometry, eval/grad readouts, entropy,
selected visual LN parameters, Adam settings or reconstruction. It only selects
the diagnostic evaluator, preserves source identity through the estimator, and
supplies the loss preparation/evaluation hooks actually called by the engine.
There is no production dependency on tools or evolution_followup.

An experiment worker may bind StandaloneTRACE.evaluate_trace_views to an
ImageBalancedTRACE instance with arm="Uniform" or arm="Confidence". Those
controls reuse the worker's evaluator and zero diagnostic forwards while still
using this class's source identity retention and per-image loss. These are not
additional accepted constructor view_mode values. The old StandaloneTRACE min
baseline remains episode-normalized and is not changed here.

## Exact Boundaries

Multiplying all positive q in one source by a common positive factor preserves
the objective coefficients and entropy gradients, provided the active set is
unchanged and numerical guards/precision do not change behavior. One active
candidate cancels its own q only within that source. All-inactive and empty
sources contribute neither a numerator nor an image-denominator entry.

This removes explicit competition between source images through the episode's
sum-q denominator. It does not remove shared-parameter gradient interaction,
establish batch-optimization invariance, equate batched and separate Adam
updates, or isolate one image from changed diagnostics, candidates or gradients
of another. Changing the set of active images changes the outer 1 / |I+| factor.
Relative q within an image still matters. No ground-truth input is used.

Final global mode selection belongs to TRAIN calibration only. No held-out or
validation winner substitution is permitted; all held-out result signs must be
retained. This implementation performs no calibration, evaluation or scoring.

## CPU Verification

tests/test_image_balanced_trace.py covers exact single-source equivalence,
per-image rescaling invariance and an old-objective counterexample, token/prompt
means, detached q, inactive/empty sources, EPS guards, source/crop identities,
local singleton cancellation, actual diagnostic schedules and original masks,
unchanged optimizer/readout hooks, deterministic episodic toy-model integration,
and the worker's unbound Uniform/Confidence evaluator interface.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_image_balanced_trace.py
```
