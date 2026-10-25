# Role/Fragment-Aware CFC-TTA Experiment Report

Mode: experiment-evidence-pass.

Implementation status:
- Added foreground fragment guard to rescale thing-region reliability before entropy adaptation.
- Added conditional background budget cap as an ablation switch.
- Kept the final spatial objective as one region-normalized entropy loss.
- Kept background cap disabled by default in the tested final setting because cap-only was harmful.

Main code artifacts:
- `/home/wwww/hzl/external/MLMP/adapt/cfctta.py`
- `/home/wwww/hzl/external/MLMP/main.py`
- `/home/wwww/hzl/experiments/analyze_cfc_loss_contribution.py`
- `/home/wwww/hzl/experiments/run_role_fragment_matrix.py`

## 1. Failure Mechanism Audit

Best aggressive baseline had a noisy foreground substitution problem:

| Dataset | Best baseline thing loss from 1-2 patch regions | GT correct in that bin |
|---|---:|---:|
| COCO-CD | 50.7% | 0.2% |
| Waterbirds-Seg | 46.6% | 3.0% |

After foreground fragment guard:

| Dataset | Guarded thing loss from 1-2 patch regions | GT correct in that bin |
|---|---:|---:|
| COCO-CD | 24.5% | 0.2% |
| Waterbirds-Seg | 23.8% | 3.0% |

Interpretation:
- The previous best configuration improved foreground budget, but much of that budget came from tiny noisy fragments.
- Foreground fragment guard directly reduces this failure mode without hard-filtering all small regions.

Artifacts:
- `/home/wwww/hzl/results/minimal_cfc_area_bucket_best_20260630`
- `/home/wwww/hzl/results/minimal_cfc_area_bucket_fg_guard_20260630`

## 2. COCO-CD Full Ablation

| Variant | mIoU | mDice | mAcc | Delta vs MLMP |
|---|---:|---:|---:|---:|
| old best baseline | 67.93 | 79.74 | 88.84 | -4.17 |
| foreground fragment guard | 73.51 | 84.06 | 88.66 | +1.41 |
| background cap only | 65.61 | 77.83 | 89.19 | -6.49 |
| foreground guard + background cap | 71.34 | 82.42 | 90.83 | -0.76 |
| no grouping + guard + cap | 71.34 | 82.42 | 90.83 | -0.76 |

Conclusion:
- COCO-CD is fixed by foreground fragment guard.
- Background cap alone is harmful.
- Adding cap to guard hurts compared with guard-only.
- Current lightweight local grouping had no measurable effect in this setting.

Artifact:
- `/home/wwww/hzl/results/role_fragment_matrix_coco_full_20260630/summary.csv`

## 3. Waterbirds-Seg Full Validation

| Variant | mIoU | mDice | mAcc | Delta vs MLMP |
|---|---:|---:|---:|---:|
| foreground fragment guard | 53.15 | 65.68 | 64.88 | +0.97 |

Reference:
- Previous old best full result: 50.97 mIoU.
- MLMP reference: 52.18 mIoU.

Conclusion:
- The same guard that fixes COCO-CD also moves Waterbirds-Seg above the MLMP reference.
- mAcc is lower than old aggressive baseline, so the gain is mainly in mIoU; this should be reported honestly.

Artifact:
- `/home/wwww/hzl/results/role_fragment_waterbirds_full_fg_guard_20260630/summary.csv`

## 4. Regular Dataset Full Checks

| Dataset | fg guard mIoU | MLMP reference | Delta vs MLMP |
|---|---:|---:|---:|
| VOC20 | 84.51 | 83.76 | +0.75 |
| VOC21 | 52.43 | 50.78 | +1.65 |
| COCO Object | 29.27 | 28.84 | +0.43 |

Compared with previous best CFC-TTA:
- VOC20: 84.30 -> 84.51.
- VOC21: 51.83 -> 52.43.
- COCO Object: 28.94 -> 29.27.

Artifacts:
- `/home/wwww/hzl/results/role_fragment_voc20_full_fg_guard_20260630/summary.csv`
- `/home/wwww/hzl/results/role_fragment_voc21_full_fg_guard_20260630/summary.csv`
- `/home/wwww/hzl/results/role_fragment_coco_object_full_fg_guard_20260630/summary.csv`

## 5. Regression Proxy

128-batch matched subset:

| Dataset | old baseline | fg guard | Delta |
|---|---:|---:|---:|
| VOC20 | 81.53 | 81.40 | -0.13 |
| VOC21 | 52.03 | 52.66 | +0.63 |
| COCO Object | 26.12 | 26.53 | +0.41 |

Cityscapes proxy was stopped because 128 batches were too slow and not central to the COCO-CD/Waterbirds failure analysis.

Artifacts:
- `/home/wwww/hzl/results/role_fragment_regression_proxy128_baseline_20260630/summary.csv`
- `/home/wwww/hzl/results/role_fragment_regression_proxy128_20260630/summary.csv`

## 6. Current Recommendation

Adopt foreground fragment guard as the next default CFC-TTA reliability rescaling:

```text
rho'_R = rho_R * q_fg(R), for thing regions
q_fg(R) = s_keep(R) * [eta + (1 - eta) * sigmoid((log |R| - log a0) / tau_a)]
```

Keep background cap as an ablation/analysis switch, not as the default method:

```text
rho'_R = gamma_bg rho_R, for background regions
```

Reason:
- The experiments show cap-only worsens both COCO-CD full and Waterbirds proxy.
- Guard-only is the only setting that consistently improves shortcut datasets and regular full datasets tested so far.

## 7. Remaining Protocol Risks

- Single-seed validation.
- Dataset-specific clean configurations are still being used.
- Cityscapes full regression with foreground guard has not been completed.
- PC59/PC60 full regression with foreground guard has not been completed.
- MLMP references are local/paper-aligned constants, not re-run in this exact matrix.

