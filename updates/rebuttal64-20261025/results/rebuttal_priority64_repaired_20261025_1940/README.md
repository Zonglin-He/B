# Priority 64-Image Rebuttal Experiments

## Scheduling

Explicit user request: finish the in-flight original COCO Object cell, insert
these small studies, then resume the remaining original queue, and only then
start the representative second-backbone suite.

The in-flight cell completed and was validated. Its result is retained in the
original output directory. This controller owns the original queue/GPU leases
during the insertion and continuation. Completed original cells are validated
and skipped, never overwritten or rerun. After priority results are ready,
PRIORITY_RESULTS_READY.json is written and the original queue resumes. Backbone
launch occurs only after the original suite is complete and its seal is saved.

## Fixed Panels and Comparisons

All formal studies use 64 fixed images selected with seed 20271025. Native
per-dataset K, lr, steps, batch size, source checkpoint and TRACE formulas are
unchanged. Predictions/diagnostic masks are sealed before GT scoring. No metric
threshold, sample or control is selected for a favorable result.

| Study | Datasets | Policies |
|---|---|---|
| Semantic-role rule | PC60, COCO-Stuff | role-aware; shared quality only; all-role dominance |
| Candidate granularity / internal mixing | COCO Object, PC60 | native CC; eroded intervention boundary; dilated intervention boundary; finer local diagnosis; within-region token redistribution |
| Initial-model bias / weight refresh | COCO-CD, PC60 under source-resolution Gaussian noise s3 | uniform; original confidence; fixed TRACE; midpoint-refresh TRACE |

The role worker and results are in the role_root named by manifest.json. Local
geometry/refresh cells are in cells/ under this directory. Four-image smoke
checks are explicitly separated from 64-image outcomes. tables/summary.csv
contains completed non-smoke outcomes only, not anticipated results.

## A: Candidate Geometry and Instances

COCO instance masks are used only after adaptation predictions are sealed.
instance_geometry.csv records same-class instance overlaps and their token
counts for each selected candidate. instance_fragments.csv records how many
selected candidates touch each covered instance/crop pair. Crowds are excluded
from instance counts, not silently removed from benchmark semantic scoring.

These are token-visible, crop-level, selected-candidate statistics. They are not
instance AP, do not imply all GT instances are covered, and must not count
repeated crop observations as distinct physical instances. Candidate spatial
endpoints additionally record GT purity and boundary/interior pre/post accuracy.

## B: Internal Reliability Mixing

The within-region control uses per-token supports from exactly the original
candidate interventions. Native masks, coverage and each parent rho budget are
fixed. It redistributes that parent's mass inside the candidate, not across
parents. Uniform local weights are numerically and gradient-equivalent to the
native objective in CPU tests.

The finer-region control partitions a parent into at most four disjoint bbox
quadrants, diagnoses those child masks independently, then redistributes the
same parent budget according to local support. Children do not independently
consume K slots or multiply the parent's optimization budget. No GT instance
mask defines a child or a weight.

quality_aggregation_gap records min(mean(T),mean(C))-mean(min(T,C)); it measures
the averaging effect, not a probability of harm. Internal controls isolate
within-parent allocation, not the optimal total coefficient across parents.
spatial_summary.csv retains whole/boundary/interior and all/mixed/majority-correct
cohorts with sample/event denominators. Label transitions are not described as
GT-probability changes or confidence increases.

## C: Diagnostic Boundary Sensitivity

Erode/dilate the intervention mask by one four-connected feature-grid token.
Candidate selection, original readout positions and original loss mask remain
unchanged. Empty eroded masks remain in the experiment and are counted per
episode. One token is a substantial perturbation for tiny candidates; it must
not be described as universally minor boundary noise.

## Verification and Repair History

12 role-policy CPU tests and 7 local-control CPU tests passed. The prior smoke
archive is retained at rebuttal_priority64_20261025_1934. Its profiling assertion
incorrectly equated a logical view with a visual-encoder input. The archived
adaptive-weighted readout uses two encoder passes per OVSS evaluation. The
repaired counter separately records logical views, OVSS calls, visual calls and
visual inputs; it does not change inference, reliability or adaptation. All four
repaired GPU smoke cells and all six formal 64-image cells have now completed.
The offline pixel-space COCO instance audit has also completed; it includes
uncovered GT instances and is separate from the token-visible audit above.
See RESULTS.md for exact outcomes, denominators and claim boundaries. The
original queue has resumed; queue_state.json and queue.log track its continuation.

No rounded difference, deterministic repeat, pixel count or candidate count is
treated as independent statistical significance. These fixed subsets answer
targeted mechanisms, not full-dataset main-result claims.
