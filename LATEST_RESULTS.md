# Latest Completed Results

This page separates submitted-method evidence from new algorithm variants.
The authoritative rows, populations, seeds, and paths remain in the linked CSVs.
This is a snapshot, not a live experiment dashboard.

## New: Completed 64-Image Reviewer Controls

The [separate verified update](updates/rebuttal64-20261025/README.md) contains all
completed region-boundary/internal-allocation, semantic-role, and noisy
fixed-versus-refresh controls, including unfavorable and null outcomes.
The original main queue remains separate and continues running.

COCO Object within-region allocation, holding each native parent budget fixed,
changes subset mIoU from 19.5348 to 19.9188 and actual whole-image correct-to-wrong
rate from 19.2164% to 13.0575%. PC60 mIoU changes from 23.7211 to 23.6625.
These are fixed 64-image diagnostic results, not full-validation replacements.
The offline instance audit includes uncovered instances in its denominator.

## Submitted TRACE: Full-Split Module Ablation

The COCO-CD and PC60 rows below have completed seeds 0/1/2; the values are
identical across seeds under this deterministic protocol. The archived Full
seed-0 result is reused after source/configuration/population/seal checks.

| Variant | COCO-CD mIoU | PC60 mIoU |
|---|---:|---:|
| Full TRACE | 75.1644 | 30.3682 |
| w/o Region Construction | 72.3842 | 30.3036 |
| w/o Confidence-Source Tracing | 67.2863 | 30.1343 |
| w/o Reliability-Weighted Adaptation | 66.6572 | 30.1185 |

[Raw ablation table](results/rebuttal_original_20261025_122642/tables/ablation.csv)
contains completed individual cells for other datasets as captured; it must not
be interpreted as three completed seeds for every dataset.

## Submitted TRACE: Fixed-Candidate Weighting Controls

Full splits, seed 0, fixed candidates and optimization settings:

| Weight policy | VOC21 | PC60 | COCO Object | COCO-CD |
|---|---:|---:|---:|---:|
| Uniform | 52.1639 | 30.1185 | 29.0653 | 66.6572 |
| Original confidence | 52.1397 | 30.1343 | 29.9398 | 67.2863 |
| Flip consistency | 52.3032 | 30.1604 | 29.2520 | 71.4934 |
| Shared multi-view quality | 52.1524 | 30.3876 | 31.9362 | 75.1475 |
| Full TRACE | 52.1589 | 30.3682 | 31.9972 | 75.1644 |

[Episode controls](results/reviewer_followup_20261010_123627/tables/episode_controls.csv).
Shared quality explains most of the observed improvement; these results do not
establish a uniformly large incremental benefit for dominance attenuation.

## Submitted TRACE: Refresh, Operators, and Segmentation Audit

- [Fixed versus midpoint refresh](results/rebuttal_original_20261025_122642/tables/refresh.csv):
  128 fixed images per condition, seed 0; midpoint refresh adds 0.113/0.200 mIoU
  on clean/noisy COCO-CD and 0.052/0.004 on clean/noisy PC60.
- [Alternative context operators](results/rebuttal_original_20261025_122642/tables/robustness.csv):
  128 fixed images per dataset, seeds 0/1/2; default/noise/masking/color-statistics
  mIoU is 75.939/76.245/76.236/76.556 on COCO-CD and
  29.398/29.512/29.431/29.553 on PC60.
- [Candidate selection](results/rebuttal_original_20261025_122642/tables/selection.csv):
  fixed-K controls are subset results, not full validation.
- [Actual segmentation-event audit](results/rebuttal_original_20261025_122642/offline/interpretation.md):
  preserve separate probability-drop, region-accuracy-loss, majority-crossing,
  and reconstructed pixel-transition endpoints. Conditional matched evidence
  is not a universal raw-risk detector or causal-identification claim.

## New Algorithms: Two-View Full Validation

These are **new algorithm variants**, not replacements for the submitted-method
rebuttal evidence. Full validation, one recorded seed, fixed configurations:

| Dataset | Two-view quality mIoU | Measured ms/image |
|---|---:|---:|
| VOC21 | 52.3965 | 292.00 |
| PC60 | 30.5843 | 118.19 |
| COCO Object | 32.6783 | 263.69 |
| COCO-CD | 75.1896 | 32.48 |

[Full result and profiling table](results/trace_two_view_full_20261025/algorithm_summary.csv).
Timing follows the recorded warm-up and measured-image protocol, not an average
of every image. Do not mix these numbers with differently instrumented runs.

## New VOC21 Forward-Reduction Variants

Full validation on 1,449 images, fixed variant configurations:

| Variant | mIoU | Measured ms/image |
|---|---:|---:|
| Lazy quality, omitted-mass bound 0.01 | 52.3801 | 235.86 |
| Class-aware lazy quality | 52.3700 | 264.99 |
| Cached prefix, last 12 blocks adapted | 52.5190 | 230.55 |
| Cached prefix, last 6 blocks adapted | 52.2684 | 200.20 |

[Full results](results/trace_voc21_forward_reduction_20261025/algorithm_summary.csv).
Restricted adaptation changes the parameter set and is an algorithm variant,
not merely an equivalent runtime optimization.

## Pending and Historical Evidence

The captured queue states are in [snapshot.json](snapshot.json). Planned backbone
jobs and partially completed suites do not imply completed validation. Earlier
paper versions, CFC guards/CLS, historical calibration trials, smoke runs, failures,
and cancelled work remain available with their original names and provenance.
