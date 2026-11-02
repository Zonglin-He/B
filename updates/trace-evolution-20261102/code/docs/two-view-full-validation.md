# Two-view full validation and algorithmic forward reduction

## Frozen first stage

The user requested full validation of two-view shared quality, prioritizing
PC60, and further forward reduction through algorithm changes rather than
smaller K, fewer optimizer steps or timing-oriented batch tuning.

The frozen queue is
`/home/wwww/hzl/results/trace_two_view_full_20261025`.
It evaluates PC60 (5,105), VOC21 (1,449), COCO-CD (500), then COCO Object
(5,000), seed 0. All inherited dataset-specific parameters remain unchanged.
Every candidate has target-preserved and context-perturbed forwards;
reliability is `min(T,C)`. Residual support and dominance are unobserved,
not assigned artificial zero measurements. Production defaults and manuscript
claims are not changed.

The full matched baseline followup uses No Adapt, TENT, TPT and MLMP. WATT and
CLIPArTT stay excluded. Image order, vocabulary, preprocessing and adaptation
episode boundaries match TRACE. Baselines use their archived method-native
settings, including learning rate 0.001 and 10 steps, not artificially matched
TRACE optimizer settings. This is a matched-data comparison, not a claim of
equal tuning or optimizer budgets. The acceptance reference is the strongest
completed baseline for each dataset, not old TRACE or a filtered old result.

The detached followup is
`/home/wwww/hzl/results/trace_two_view_full_baselines_20261025` (16 cells).
It waits for the native queue lock and completion seal. The algorithm study
`/home/wwww/hzl/results/trace_lazy_quality_gpu_20261025` (four smoke cells,
16 fixed-128 cells) then waits for the full baseline followup. It compares
complete two-view evaluation, staged complete evaluation, global lazy CF and
image/query-class protected lazy CF. Failures stop dependent execution.

First completed full result: PC60, 5,105 images, seed 0, **30.584325 mIoU**,
118.187636 ms/image after two warmup episodes. Actual all-population counts:
223,028 diagnostic inputs, 289,393 total visual inputs, 21,488 visual calls.
These counts include warmup; timing excludes warmup, loading, scoring and
initialization. The matched full baselines were not complete at this checkpoint;
do not subtract a 128-image baseline score from this full result.

Four initial 128-image matched pilots favored two-view quality, but PC60's
margin was only 0.073 mIoU. This is a reason to validate, not evidence of
reliable superiority. A full-split seed-0 result is not three-seed evidence.

## Algorithm study

Exact view deduplication cannot provide a large improvement on the current
pilot: all four quality-only panels have zero whole-patch candidates and zero
duplicate masks within the same patch. They also have zero source images
without an active quality-weighted candidate. Skipping zero-weight image
updates therefore does not explain an efficiency improvement here.

The next bounded design uses `q=min(T,C)<=T`: measure all T, then calculate C
only while an unmeasured set could materially affect normalized coefficients.
With observed active weight A and unseen upper mass U, `U/(A+U)` bounds the
omitted normalized weight mass, conditional on the same support measurements.
It does not bound gradients, Adam trajectories, IoU, or rare-class harm.
Floating-point differences from staged batching require a staged all-view
control.

A companion rule applies the same bound separately to each source-image and
predicted-query-class group. It does not merge masks, infer roles, or inspect
GT. This prevents a high-mass class elsewhere in the episode from deciding
when a low-mass class has been sufficiently probed. The tolerance is an
explicit algorithmic approximation setting, not a parameter-free guarantee.
All variants retain identical K, learning rate, steps and episode batch size.
Count diagnostic image inputs, all visual image inputs, forward calls and
synchronized time separately. Batch packing can reduce calls without reducing
the number of visual inputs; neither alone guarantees end-to-end speedup.

CPU fixed-support replay is saved at
`/home/wwww/hzl/results/trace_lazy_quality_cache_20261025`. The protected
variant saves 7.70%, 8.83%, 0.20% and 12.89% of diagnostic inputs on the
PC60, VOC21, COCO-CD and COCO Object pilot caches. It drops no complete active
image/query-class group. The global rule drops 188, 237, 0 and 369 such groups,
respectively, despite satisfying the global coefficient bound. These are
query-decision replay results, not new segmentation scores or measured latency.

## Diagnose before parameter search

Only training/calibration samples may select a new parameter setting. Full
validation can reveal a failure mode but cannot be repeatedly searched for
the best reported score. No automatic search is triggered by a negative full
result. First compare the completed full candidate against all four baselines
and inspect its per-class intersection/union plus episode-level allocation.

| Parameter | Mechanism to inspect | Appropriate controlled experiment |
| --- | --- | --- |
| Episode batch size | Reset frequency, co-occurring classes and one episode-wide reliability denominator change together. It is not merely a throughput control. | Integer-by-integer calibration sweep around the inherited value; identical training IDs; report which classes gain/lose and timing. |
| K | Candidate admission and class coverage change; more candidates can add both evidence and confident fragments. | Measure selected candidate class/area coverage and marginal quality before considering a narrow K sweep. |
| Learning rate | Step magnitude may improve under-adapted classes or amplify a wrong entropy target. | Calibration before/after predictions and update trajectory, not an unsupported inference from one aggregate mIoU. |
| Steps | Repeated sharpening can saturate or worsen errors; equal lr-times-steps is not equivalent under Adam. | Record a fixed calibration trajectory at intermediate checkpoints and its cost. |

`analyze_two_view_full.py` writes class-level IoU and mean episode-normalized
coefficient shares after the full baseline followup. These coefficients are
not gradient attribution. Offline target-only coefficients are not adaptation
results. Negative deltas and small-support classes remain visible.

## Review and promotion boundary

Independent reviews cover the full runner, baseline adapter, mathematical
bound and new algorithm implementation. Fix concrete findings and re-test;
do not interpret a reviewer finding no additional issue as a proof of global
correctness. Failures retain their files and stop dependent queues; no silent
retry or selection among stochastic attempts. Followup GPU jobs wait on queue
locks and run one worker at a time. No automatic production or paper promotion.
