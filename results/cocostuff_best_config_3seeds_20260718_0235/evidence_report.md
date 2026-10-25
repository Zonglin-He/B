# COCO-Stuff Tuned-Configuration Three-Seed Follow-up

## Experiment Evidence

- Evidence type: `newly_run_full_clean_3seeds`.
- Configuration: the seed-0 Optuna winner from `cocostuff_optuna_coordinate_20260717_2252_fair240`.
- Seeds: 0, 1, and 2 on all 5,000 COCO-Stuff clean validation images.
- Seed 0 reuses the already completed full-clean artifact; seeds 1 and 2 were newly run with the identical configuration.

| Seed | mIoU | mDice | mAcc | Delta vs MLMP |
|---:|---:|---:|---:|---:|
| 0 | 21.66 | 32.37 | 42.21 | +0.41 |
| 1 | 21.53 | 32.21 | 42.22 | +0.28 |
| 2 | 21.53 | 32.21 | 42.04 | +0.28 |
| Mean +/- std | 21.57 +/- 0.08 | 32.26 +/- 0.09 | 42.16 +/- 0.10 | +0.32 |

## Decision

The tuned configuration remains above the MLMP reference of 21.25 mIoU, but it does not improve the previous canonical CFC-TTA result of 21.61 +/- 0.02. Its higher variance also indicates seed-0 selection bias. The project therefore keeps the previous three-seed configuration as the canonical COCO-Stuff setting.

## Artifacts

- Per-seed rows: `per_seed_summary.csv`
- Aggregate row: `mean_std_summary.csv`
