# Matched baseline semantic-role diagnosis

## Protocol boundary

All methods are evaluated on the same deterministic clean-image prefix and the same frozen pre-adaptation CFC-TTA candidate masks. TENT/MLMP use public/default optimizer settings; CFC-TTA uses its canonical dataset-specific clean configuration. This is a mechanism audit, not a fully optimizer-matched leaderboard.

## Subset mIoU

| Dataset | NoAdapt | TENT | MLMP | CFC-TTA | CFC - MLMP |
|---|---:|---:|---:|---:|---:|
| voc20 | 75.79 | 70.89 | 81.96 | 83.49 | +1.53 |
| pc60 | 27.47 | 25.68 | 28.40 | 28.83 | +0.43 |
| coco_object | 27.67 | 25.10 | 28.72 | 28.96 | +0.24 |
| coco_stuff | 18.90 | 17.72 | 19.33 | 20.21 | +0.88 |
| coco_cd | 62.76 | 71.76 | 71.17 | 73.04 | +1.87 |
| waterbirds_seg | 39.96 | 45.36 | 42.59 | 49.26 | +6.67 |
| voc21 | 48.82 | 43.62 | 49.07 | 51.57 | +2.50 |
| pc59 | 30.07 | 28.29 | 31.25 | 31.54 | +0.29 |
| cityscapes | 32.46 | 29.20 | 32.86 | 34.05 | +1.19 |

## Dataset-role macro averages

Each row is an unweighted mean over available dataset-role strata with at least 20 initially-correct candidates (the `all` row is always retained); rates are percentages below. Every raw stratum remains in `matched_role_by_dataset.csv`.

| Method | Role | Strata | Region accuracy delta | Correct-to-harm | Mean flip | Flip incidence |
|---|---|---:|---:|---:|---:|---:|
| NoAdapt | thing | 9 | 0.00 | 0.00 | 0.00 | 0.00 |
| NoAdapt | stuff | 4 | 0.00 | 0.00 | 0.00 | 0.00 |
| NoAdapt | background | 4 | 0.00 | 0.00 | 0.00 | 0.00 |
| NoAdapt | all | 9 | 0.00 | 0.00 | 0.00 | 0.00 |
| TENT | thing | 9 | 2.89 | 11.31 | 14.23 | 56.29 |
| TENT | stuff | 4 | 0.25 | 23.54 | 18.72 | 66.60 |
| TENT | background | 4 | 1.23 | 8.79 | 9.39 | 43.31 |
| TENT | all | 9 | 1.30 | 15.02 | 14.40 | 54.89 |
| MLMP | thing | 9 | 2.20 | 12.54 | 13.19 | 52.35 |
| MLMP | stuff | 4 | 1.42 | 19.79 | 13.39 | 58.00 |
| MLMP | background | 4 | -0.03 | 5.52 | 4.18 | 17.31 |
| MLMP | all | 9 | 1.31 | 12.51 | 11.01 | 43.70 |
| CFC-TTA | thing | 9 | 6.26 | 16.03 | 20.79 | 69.32 |
| CFC-TTA | stuff | 4 | 2.71 | 19.85 | 20.20 | 68.37 |
| CFC-TTA | background | 4 | 1.28 | 4.43 | 5.42 | 24.87 |
| CFC-TTA | all | 9 | 3.92 | 13.49 | 16.53 | 55.58 |
