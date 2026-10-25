# Completed 64-Image Rebuttal Controls

All priority GPU cells completed on 2026-10-25. The controller wrote
`PRIORITY_RESULTS_READY.json` and resumed the original queue. The second
backbone remains scheduled after the original suite.

## Protocol

- 64 fixed images per dataset, selected before outcomes with seed 20271025.
  These are diagnostic subsets, not full-validation main-table results.
- Archived submitted TRACE and original dataset K, learning rate, steps and
  batch size; no outcome-driven tuning of samples, masks or event thresholds.
- Predictions/support records sealed before GT scoring. GT never creates a
  candidate, intervention or update weight.
- C-to-W means actual pre-correct/post-wrong top-1 labels, not a probability-drop
  event. Whole-image rates use initially correct valid pixels as the denominator.
- Fine controls keep each parent's native rho and the inter-parent objective
  normalizer. Internal coefficients sum to one. Children do not consume K slots
  or multiply the parent budget. Realized entropy and gradients can still change.
- No token count, candidate count or deterministic repeat is a seed/significance
  test. No formal method or paper result has been replaced by a new control.

## A. Offline Candidate Geometry Against COCO Instances

On the same 64 COCO Object images, selected token masks are upsampled to the
actual 224x224 intervention/output space; official instances are resized there.
Instance overlap means at least one pixel, with raw overlap sizes retained.

| Quantity | Count |
| --- | ---: |
| Selected thing candidates | 1,203 |
| Candidates touching at least two same-class GT instances | 75 |
| Candidates with both correct and incorrect class locations | 394 |
| Visible non-crowd GT instances | 445 |
| Instances not touched by a selected same-class candidate | 135 |
| Instances touched by multiple selected same-class candidates | 108 |

Uncovered instances remain in the denominator. Crowd and zero-area records are
explicit in all_gt_instances.csv. Coverage loss can result from prediction,
class selection and top-K, not just component boundaries. Touching multiple
correct same-class instances is not itself a semantic segmentation error.
These counts differ from the separate feature-token audit; do not mix them.

Sources: offline_pixel_geometry/result.json, candidate_pixel_geometry.csv,
all_gt_instances.csv and geometry_policy_summary.csv in that directory.

## B. Fixed-Parent-Budget Internal Allocation

Within-region support uses the exact native intervention outputs to compute
tokenwise minimum quality and thing-only linear dominance, normalized inside
the original parent. It needs no extra diagnostic views in this comparison.
Finer local diagnosis partitions a parent into up to four disjoint quadrants,
diagnoses the children and redistributes the same parent budget. Neither uses
GT instance boundaries.

| Allocation | COCO Object mIoU | PC60 mIoU |
| --- | ---: | ---: |
| Original region-uniform TRACE | 19.5348 | 23.7211 |
| Finer local diagnosis, matched parent budget | 19.4496 | 23.7526 |
| Within-region support, matched parent budget | 19.9188 | 23.6625 |

| Allocation | COCO Object whole-image C-to-W (%) | PC60 whole-image C-to-W (%) |
| --- | ---: | ---: |
| Original region-uniform TRACE | 19.2164 | 9.7200 |
| Finer local diagnosis | 17.7760 | 9.9328 |
| Within-region support | 13.0575 | 9.5840 |

Within-region support improves COCO Object mIoU by 0.3841 and lowers whole-image
C-to-W by 6.1589 percentage points. PC60 mIoU changes by -0.0586. This is direct
evidence that internal allocation can matter, not that averaging always
preserves local reliability.

On the 394 pixel-space semantically mixed thing candidates, C-to-W is 20.0516%
for native versus 10.3892% for within-region support. Both use 549,636 initially
correct candidate pixels. This pooled candidate-pixel rate is distinct from the
whole-image rate above.

The saved gap min(mean(T),mean(C))-mean(min(T,C)) measures the source-averaging
effect, not actual update utility or calibrated harm probability.

Sources: cells/regions/{coco_object,pc60}/result.json and their
candidate_spatial_endpoints.csv/spatial_summary.csv; the pixel-space mixed
cohort is in offline_pixel_geometry/geometry_policy_summary.csv.

## C. Intervention Boundary Sensitivity

Only the view-generation mask changes by one four-connected feature-grid token.
Candidate selection, original diagnostic readout and loss mask remain fixed.

| Intervention boundary | COCO Object mIoU | Delta | PC60 mIoU | Delta |
| --- | ---: | ---: | ---: | ---: |
| Original | 19.5348 | 0.0000 | 23.7211 | 0.0000 |
| Eroded by one token | 19.1664 | -0.3683 | 23.7017 | -0.0194 |
| Dilated by one token | 19.0235 | -0.5113 | 23.7695 | +0.0484 |

Erosion empties 1,944/2,137 COCO Object and 1,238/1,398 PC60 candidate masks.
All are retained and counted. One-token erosion is severe for these tiny
components; it is not minor subpixel noise. This control does not establish
accurate instance boundaries or universal boundary invariance.

## Other Completed Priority Controls

These controls were requested alongside A/B/C and completed in the same block.

| Clean role policy | PC60 mIoU | COCO-Stuff mIoU |
| --- | ---: | ---: |
| Role-aware TRACE | 23.7211 | 13.3341 |
| Shared quality only | 23.7382 | 13.4216 |
| Dominance for all roles | 23.9076 | 13.3017 |

This panel does not show a substantial, consistent role-aware mIoU advantage.
The semantic design rationale is distinct from performance evidence.

Fixed Gaussian-noise severity-3 panels, corruption before resize:

| Weights | COCO-CD mIoU | PC60 mIoU |
| --- | ---: | ---: |
| Uniform regional weights | 54.4978 | 23.2311 |
| Source confidence | 56.9789 | 23.1292 |
| Fixed initial TRACE | 74.5767 | 22.6776 |
| TRACE refreshed once at midpoint | 75.1181 | 22.5958 |

Fixed TRACE outperforms confidence-only on this COCO-CD panel; midpoint refresh
adds 0.5414. PC60 does not show that ordering. These results do not prove that
refreshing necessarily harms, or that initial fixed weights are always best.
Actual refresh times and score changes are retained separately.

## Rebuttal Boundaries and Verification

Old connected-components versus class-union tests spatial locality, not boundary
precision or internal mixture. A measures occurrence, B controls parent budgets
while testing internal allocation, and C tests diagnostic-mask sensitivity.
Do not claim that precise boundaries or internal mixture cannot matter.

12 role-policy and 7 local-control CPU tests passed; four four-image GPU smoke
cells passed. Native update equivalence and seals are saved per cell.
A prior smoke-only profiling assertion was repaired in fresh output roots:
the archived adaptive-weighted readout has two visual-encoder passes per logical
view. Logical views and encoder calls are now separate counters, with no change
to predictions, reliability or adaptation mathematics.

Unrounded outcomes and paths are in tables/summary.csv. The controller continues
the remaining frozen original commands, then launches the deferred backbone.
