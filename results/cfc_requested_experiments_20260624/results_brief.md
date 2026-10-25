# Requested CFC-TTA Experiment Summary

This report separates completed model evaluations from analysis of existing artifacts.  It does not introduce new method changes.

## High-CFC Subset

Top-10% selector: `cfc_score_max`
| Method | mIoU | Harmful % | Flip % | Correct-to-harm % | Images | Regions |
| --- | --- | --- | --- | --- | --- | --- |
| NoAdapt | 6.68 | 0.00 | 0.00 | 0.00 | 145 | 407 |
| TENT | 6.31 | 62.41 | 9.00 | 58.33 | 145 | 407 |
| MLMP | 6.87 | 48.89 | 7.12 | 50.77 | 145 | 407 |
| CFC-TTA | 6.84 | 46.93 | 11.36 | 48.92 | 145 | 407 |

Top-10% selector: `background_support_max`
| Method | mIoU | Harmful % | Flip % | Correct-to-harm % | Images | Regions |
| --- | --- | --- | --- | --- | --- | --- |
| NoAdapt | 5.49 | 0.00 | 0.00 | 0.00 | 127 | 182 |
| TENT | 5.19 | 52.75 | 4.80 | 50.30 | 127 | 182 |
| MLMP | 5.52 | 51.10 | 4.38 | 51.50 | 127 | 182 |
| CFC-TTA | 5.41 | 42.31 | 6.59 | 43.11 | 127 | 182 |

## Initially Correct Predictions Harmed

| Method | Correct regions | Correct-to-harm % | Correct-to-worse % | Correct-to-wrong % | Mean acc delta |
| --- | --- | --- | --- | --- | --- |
| NoAdapt | 1863 | 0.00 | 0.00 | 0.00 | 0.0000 |
| TENT | 1842 | 56.68 | 40.07 | 3.42 | -0.0240 |
| MLMP | 1863 | 50.94 | 25.98 | 0.97 | -0.0009 |
| CFC-TTA | 1863 | 48.85 | 24.26 | 2.31 | -0.0091 |

## Context-Conflict Benchmark

Status: `formal_flip_metrics_available`.
| Method | Samples | Object mIoU | Harmful % | Flip % | Correct-to-harm % |
| --- | --- | --- | --- | --- | --- |
| NoAdapt | 240 | 44.97 | 0.00 | 0.00 | 0.00 |
| TENT | 240 | 42.84 | 68.33 | 17.08 | 63.78 |
| MLMP | 240 | 40.96 | 62.50 | 27.08 | 52.97 |
| CFC-TTA | 240 | 33.71 | 48.75 | 30.42 | 37.84 |

## Corruption Detail Summary

| Dataset | Avg mIoU | MLMP avg | Delta | Wins vs MLMP | Top-1 count | Avg rank | Worst |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PascalContext59-C | 26.64 | 27.03 | -0.39 | 3 | 3 | 1.80 | zoom_blur |
| PascalContext60-C | 24.64 | 24.07 | 0.57 | 13 | 13 | 1.13 | zoom_blur |
| VOC21-C | 46.92 | 46.25 | 0.67 | 14 | 14 | 1.07 | zoom_blur |

## Strong Contextual Shortcut Datasets

| Dataset | Method | mIoU | Top10 harm % | Interpretation |
| --- | --- | --- | --- | --- |
| COCO-CD | NoAdapt | 63.39 | 0.00 | baseline_reference |
| COCO-CD | TENT | 73.56 | 0.00 | baseline_reference |
| COCO-CD | MLMP | 72.10 | 4.84 | baseline_reference |
| COCO-CD | CFC-TTA | 29.39 | 88.71 | negative_for_current_cfctta |
| COCO-CD safe CFC-TTA variant | NoAdapt | 63.39 | 0.00 | baseline_reference |
| COCO-CD safe CFC-TTA variant | TENT | 73.56 | 3.85 | baseline_reference |
| COCO-CD safe CFC-TTA variant | MLMP | 72.10 | 0.00 | baseline_reference |
| COCO-CD safe CFC-TTA variant | CFC-TTA | 32.10 | 57.69 | negative_for_current_cfctta |
| Waterbirds-Seg | NoAdapt | 46.96 | 0.00 | available |
| Waterbirds-Seg | TENT | 55.97 | 1.89 | available |
| Waterbirds-Seg | MLMP | 52.18 | 4.10 | available |
| Waterbirds-Seg | CFC-TTA | 21.02 | 78.97 | available |

## Claim Guidance

- The strongest completed positive evidence remains VOC20 overall and synthetic context-conflict harm reduction.
- High-CFC VOC20 is mixed: CFC-TTA reduces harmful update rate versus MLMP in the top-10% CFC-score subset, but does not beat MLMP on mIoU/flip rate there.
- COCO-CD is a negative diagnostic for the current implementation and should be treated as a limitation or omitted from main positive claims.
- VOC21-C and PascalContext-C per-corruption CFC-TTA values are compared with MLMP paper Appendix Tables 21--23.

## Files

- `/home/wwww/hzl/results/cfc_requested_experiments_20260624/high_cfc_subset_formal.csv`
- `/home/wwww/hzl/results/cfc_requested_experiments_20260624/correct_prediction_harm.csv`
- `/home/wwww/hzl/results/cfc_requested_experiments_20260624/context_conflict_formal.csv`
- `/home/wwww/hzl/results/cfc_requested_experiments_20260624/corruption_detail_voc21_pc.csv`
- `/home/wwww/hzl/results/cfc_requested_experiments_20260624/corruption_summary_voc21_pc.csv`
- `/home/wwww/hzl/results/cfc_requested_experiments_20260624/shortcut_dataset_status.csv`
