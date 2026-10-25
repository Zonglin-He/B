# Corruption protocol audit (2026-10-01)

## Conclusion

The current panel is not a replication of the July corruption evaluation.
There is a verified transformation-order mismatch: the historical pipeline
corrupts the source-resolution image before resize; the current review panel
resizes to 224x224 before corruption. Do not attribute the observed score gap
solely to the simplified TRACE algorithm. No model evaluation was run for this
audit and no completed results were replaced.

## Historical evidence

- Saved run: `results/paper_evidence_suite_20260710/07_complete_corruptions/corruption_full/voc21/gaussian_noise/configurations.txt`.
- Its recorded commit is `ba9f57bf47846ec727c616c5163233a98ca0bf94`, accessible in
  `results/reviewer_response_20260929/baseline_source`.
- At that commit, `utils/segmentation_datasets.py` inserts `CorruptTransform`
  immediately after `LoadImageFromFile`, before `ResizeAndPatchify`.
- `utils/mm_transforms.py` confirms corruption on the loaded image, then cv2
  bilinear resize. The saved run has `corruption_cache_dir: None`.
- The historical pipeline processes BGR, then converts to RGB for normalization.
- Historical severity is 5; the current panel uses 1, 3 and 5. The six common
  operators have matching severity parameter lists in the two implementations.

## Current evidence

`tools/reviewer_response/corruptions.py:81-83` resizes RGB using PIL BILINEAR,
then applies upstream imagecorruptions 1.1.2. Its provenance explicitly records
this order. Thus it follows its own frozen protocol, but not the old protocol.

The current input-hash audit reports 420 completed cells, 76 matched condition
groups, and no cross-method input mismatches. This establishes matched inputs
within this panel, not equivalence to the historical evaluation.

## CPU order-only control

Run `tools/reviewer_response/audit_corruption_order.py` with the mlmp Python
environment. It selects 32 VOC21 images from the current sealed subset using
seed 20271001. Both paths use identical RGB decoding, upstream library,
severity 5, and PIL resize; only operation order differs. No GT or GPU is used.
This intentionally isolates order rather than reproducing all historical
BGR/cv2 details. MAE is relative to the common clean resized image (0..255).

| Operator | Corrupt then resize | Resize then corrupt | Ratio |
|---|---:|---:|---:|
| Gaussian noise | 25.230 | 59.835 | 2.372 |
| Shot noise | 26.219 | 63.675 | 2.429 |
| Defocus blur | 11.668 | 16.192 | 1.388 |
| Brightness | 76.561 | 78.264 | 1.022 |
| Contrast | 47.911 | 47.913 | 1.000 |
| JPEG compression | 7.633 | 10.498 | 1.375 |

MAE is an input-distortion measure, not an estimate of mIoU or relative method
performance. Noise is attenuated by downsampling in the first path; fixed
blur kernels also operate at different relative spatial scales.

## Historical/current results (not controlled comparisons)

Severity 5 Gaussian noise mIoU: VOC21 46.25 vs 33.69; PC60 24.64 vs 14.49;
COCO Object 26.75 vs 12.79. Sources are the July `summary.csv` and current
`tables/corruption_cells.csv`. In addition to preprocessing, these differ in
method version, frozen optimization configuration, image set (full vs 256),
and noise seed policy. Historical results use `cfctta`, including now-deleted
CLS/fragment/KL components. VOC21 historical batch size is 2 versus current 16.
The old 15-corruption mean must not be compared to the new six-operator,
three-severity mean as though they were the same metric.

## Required interpretation and follow-up

Retain the current panel as the resize-before-corruption evaluation. Any
legacy-aligned rerun must use a separate manifest/output directory and matched
inputs for TRACE and retained baselines. First compare both preprocessing
orders on fixed images with identical current model/configuration, then decide
the scope of full evaluation. Do not choose a protocol merely because it favors
TRACE, overwrite frozen artifacts, or promise that restoring the old order
will restore TRACE's lead over baselines.
