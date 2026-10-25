# CFC-TTA Paper Evidence Report

All numbers below are newly run artifacts from this queue unless a source file states otherwise.

## Gain Attribution Bridge

| Dataset | Stage | mIoU | Delta vs MLMP | Delta vs previous |
| --- | --- | --- | --- | --- |
| coco_cd | 01_mlmp_matched | 73.59 | 0.00 | -- |
| coco_cd | 02_uniform_regions | 74.07 | 0.48 | 0.48 |
| coco_cd | 03_target_support | 73.98 | 0.39 | -0.09 |
| coco_cd | 04_source_diagnosis | 74.74 | 1.15 | 0.76 |
| coco_cd | 05_counterfactual_stability | 74.73 | 1.14 | -0.01 |
| coco_cd | 06_fragment_guard | 74.69 | 1.10 | -0.04 |
| coco_cd | 07_full_gated_cls | 74.70 | 1.11 | 0.01 |
| pc60 | 01_mlmp_matched | 29.89 | 0.00 | -- |
| pc60 | 02_uniform_regions | 30.22 | 0.33 | 0.33 |
| pc60 | 03_target_support | 30.43 | 0.54 | 0.21 |
| pc60 | 04_source_diagnosis | 30.34 | 0.45 | -0.09 |
| pc60 | 05_counterfactual_stability | 30.38 | 0.49 | 0.04 |
| pc60 | 06_fragment_guard | 30.39 | 0.50 | 0.01 |
| pc60 | 07_full_gated_cls | 30.42 | 0.53 | 0.03 |
| voc20 | 01_mlmp_matched | 83.50 | 0.00 | -- |
| voc20 | 02_uniform_regions | 82.59 | -0.91 | -0.91 |
| voc20 | 03_target_support | 83.88 | 0.38 | 1.29 |
| voc20 | 04_source_diagnosis | 83.51 | 0.01 | -0.37 |
| voc20 | 05_counterfactual_stability | 84.25 | 0.75 | 0.74 |
| voc20 | 06_fragment_guard | 84.63 | 1.13 | 0.38 |
| voc20 | 07_full_gated_cls | 85.03 | 1.53 | 0.40 |
| waterbirds_seg | 01_mlmp_matched | 51.35 | 0.00 | -- |
| waterbirds_seg | 02_uniform_regions | 54.51 | 3.16 | 3.16 |
| waterbirds_seg | 03_target_support | 54.28 | 2.93 | -0.23 |
| waterbirds_seg | 04_source_diagnosis | 54.34 | 2.99 | 0.06 |
| waterbirds_seg | 05_counterfactual_stability | 54.31 | 2.96 | -0.03 |
| waterbirds_seg | 06_fragment_guard | 55.19 | 3.84 | 0.88 |
| waterbirds_seg | 07_full_gated_cls | 55.19 | 3.84 | 0.00 |

## Paired Bootstrap Confidence Intervals

| Dataset | Delta mIoU | 95% CI | P(delta>0) | Images |
| --- | --- | --- | --- | --- |
| voc20 | 1.534 | [0.387, 2.754] | 0.9949 | 1449 |
| pc60 | 0.527 | [0.362, 0.699] | 1.0000 | 5105 |
| coco_cd | 1.112 | [0.403, 1.801] | 0.9982 | 500 |
| waterbirds_seg | 3.834 | [3.429, 4.227] | 1.0000 | 5322 |

## Efficiency

Timing excludes the first 20 batches and mIoU is reported only on the 120-batch profile subset.

| Dataset | Variant | Subset mIoU | s/batch | Forwards/batch | Peak MB | Views |
| --- | --- | --- | --- | --- | --- | --- |
| voc20 | mlmp_official | 79.26 | 0.148 | 11.00 | 2377.1 | 0.00 |
| voc20 | cfctta_deployment | 82.68 | 0.301 | 20.98 | 2147.7 | 17.35 |
| voc20 | cfctta_controlled_r2 | 78.63 | 0.232 | 13.93 | 2369.5 | 12.85 |
| voc20 | cfctta_controlled_r4 | 79.77 | 0.276 | 15.41 | 2369.5 | 23.97 |
| voc20 | cfctta_controlled_r8 | 79.67 | 0.317 | 16.65 | 2369.5 | 33.77 |
| coco_object | mlmp_official | 24.73 | 0.150 | 11.00 | 3707.5 | 0.00 |
| coco_object | cfctta_deployment | 30.04 | 1.550 | 46.38 | 4157.9 | 111.17 |
| coco_object | cfctta_controlled_r2 | 23.34 | 0.238 | 14.00 | 3705.1 | 12.75 |
| coco_object | cfctta_controlled_r4 | 23.91 | 0.299 | 15.94 | 3705.1 | 27.83 |
| coco_object | cfctta_controlled_r8 | 24.33 | 0.423 | 19.86 | 3705.1 | 59.06 |
| cityscapes | mlmp_official | 33.49 | 2.829 | 11.00 | 20718.4 | 0.00 |
| cityscapes | cfctta_deployment | 33.32 | 16.816 | 451.67 | 11552.5 | 3465.99 |
| cityscapes | cfctta_controlled_r2 | 34.12 | 5.249 | 69.06 | 21303.5 | 453.41 |
| cityscapes | cfctta_controlled_r4 | 34.12 | 7.496 | 140.42 | 21303.6 | 1024.10 |
| cityscapes | cfctta_controlled_r8 | 34.17 | 12.056 | 284.92 | 21303.8 | 2179.63 |

## Complete Corruption Evaluation

No completed rows.

## Protocol Notes

- Corruptions use severity 5 and retain each dataset's clean-selected CFC-TTA parameters.
- The bridge uses matched optimizer and batch settings; it is not the quoted MLMP paper table.
- SCLIP uses its final-layer dense output; no SCLIP-specific CFC-TTA tuning is performed.
- Paired bootstrap resamples images and recomputes aggregate mIoU for every resample.
- Same-class analysis is candidate-level and pairs identical region hashes across policies.
