# VOC21 Analysis

## Cohorts

- Full official validation: 1449 images, mIoU 52.35047870. Absolute class IoUs: `full_per_class_iou.csv`.
- Matched controls: 512 images in identical order and identical image-batch episodes; not full validation.
- IoU is recomputed after summing per-image intersections/unions, never by averaging image IoUs.

| 512-image comparison | Full mIoU | Control mIoU | Full - control (pp) |
| --- | ---: | ---: | ---: |
| no_source_tracing | 52.583139 | 52.116952 | +0.466187 |
| no_reliability_weighting | 52.583139 | 52.119351 | +0.463787 |
| no_region_normalization | 52.583139 | 51.120290 | +1.462849 |
| legacy_reference_panel | 52.583139 | 52.320321 | +0.262818 |

## Losing Classes First

Class deltas below are Full minus the stated control on matched 512 images.
- no_source_tracing: sofa -3.703 pp; pottedplant -2.989 pp; background -1.330 pp; horse -0.890 pp; person -0.735 pp; bicycle -0.685 pp; bird -0.672 pp; cow -0.416 pp.
- no_reliability_weighting: sofa -4.151 pp; pottedplant -3.978 pp; background -1.763 pp; horse -1.197 pp; person -1.096 pp; bicycle -0.875 pp; bird -0.393 pp; aeroplane -0.157 pp; cow -0.056 pp.
- no_region_normalization: pottedplant -2.893 pp; chair -2.487 pp; table -1.161 pp; sheep -0.517 pp; bottle -0.144 pp; dog -0.088 pp; cat -0.088 pp.
- legacy_reference_panel: aeroplane -1.270 pp; ship -0.938 pp; bird -0.313 pp; background -0.267 pp; cat -0.223 pp; bottle -0.157 pp; pottedplant -0.086 pp.

## Weight Diagnostics

- 16,629 candidate identities and areas match across all four controls; 32 episodes. Query aliases map to their VOC21 evaluation row, including background at row 0.
- Pooled rho/confidence Spearman: 0.164136; mean episode normalized-weight TV: 0.633467.
- Dominance D > T: 7,479 candidates (44.98%); mean d 0.058168, p95 0.385564.
- `matched512_loss_weight_changes.csv` reports per-class normalized loss-weight allocation, not measured entropy loss. Actual per-class objective losses are unavailable in these artifacts.
- `episode_weight_comparison.csv` includes TV, rank correlation, largest weight and effective candidate count. A common positive scale on all episode weights cancels.

## Identity And Boundaries

- The 52.32 legacy and 52.58 new panel scores use the same 512-image cohort, not full VOC21. View-forward batching differs (8 versus 16); kernel and reliability implementation also changed; image batch size stays 16. Their delta is not attributable to the reliability formula alone.
- Legacy full-VOC21 scalar 52.16 has no supplied per-class counts. No full-cohort class delta or subtraction against this unmatched 512 panel is performed.
- Intersections/unions do not identify ground-truth class pixel counts, false positives and false negatives separately, per-class accuracy, or measured per-class entropy loss; none are inferred.
- Full-validation records are not used as the matched Full512 control: changing episode membership changes episodic adaptation. Weight diagnostics describe the fixed 512 panel only.
- This is post-hoc evaluation-label analysis for future tuning, not an untouched final test or a causal explanation of class changes. No batch-size trial or selection is made here.
- Protocol checksums, recorded snapshot/tool hashes, current production hashes, ordered image/label IDs, vocabulary, seed, configuration, official pool size and count aggregation passed. All inspected evidence hashes are recorded in `summary.json` and rechecked before writing.
- Input size/mtime values match the pre-run suite manifest. They are metadata identities, not image/mask content hashes. Legacy reference has no embedded environment or file metadata; its metadata agreement is retrospective only.
- Analysis is read-only CPU work. It does not run or select batch_size 1..16 trials, modify the algorithm, or inspect/alter other running queue cells.
