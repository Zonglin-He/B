# VOC21 parameter-only continuation review

The user requested class-level diagnosis followed by an exhaustive integer
batch-size search, without changing the main algorithm or interrupting the
existing experiments. The original primary queue and its reference-refresh
audit both completed before the new search obtained the GPU locks.

## Analysis

The bounded CPU analysis reconstructs dataset mIoU from summed intersections
and unions, verifies ordered input and vocabulary identity, and separates the
1,449-image full-validation result from the 512-image matched controls. It
matches all 16,629 candidate identities and areas across the weighting
controls, including expanded text-query to evaluation-class mapping.
Normalized loss coefficients are explicitly not measured entropy losses or
gradient contributions. Class deficits are not inferred from unmatched
cohorts. The main agent inspected the analysis implementation and independently
reproduced the aggregate scores, class changes, rank correlation and weight TV.

## Tuning Review

An independent static reviewer found no actionable P1/P2 defect within the new
tuner and supporting batch override. The review checked:

- Only copied `batch_size` changes; production algorithm files are unchanged.
- Every integer 1 through 16 must be represented before selecting a winner.
- Selection uses the global training-calibration maximum, with a 0.02-point
  latency tie-break; validation outcomes do not select the trial.
- The fixed 512-image calibration cohort comes from the complete 1,464-image
  VOC2012 train split with seed 2027, disjoint from 1,449 validation images.
- Final-audit lock precedes primary lock, avoiding a deadlock with the original
  finalizer; workers inherit the held lock descriptors.
- Completed attempts are validated and adopted; failed attempts are preserved.
  Recorded calibration OOMs remain infeasible trials, and other failures stop.
- The selected full configuration is evaluated once, or the identity-checked
  existing result is reused when batch size 16 wins. Production defaults and
  manuscript results are not automatically replaced.

The main agent executed all **132 tests**, including the new analysis and tuner
tests, successfully. Real input discovery confirmed the split sizes, seed,
fixed configuration and zero path overlap. The first GPU trial started after
the existing queues completed; this is execution-start verification, not a
completed batch search or evidence that a better configuration has been found.

The static review did not exercise every queued orchestration, retry or reuse
path. Those are remaining execution-verification boundaries, not diagnosed
defects. All parameter-trial and validation outcomes are retained, including
negative or null changes.
