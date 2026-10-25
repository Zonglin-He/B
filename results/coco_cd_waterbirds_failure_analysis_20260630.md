# COCO-CD / Waterbirds Failure Analysis for Minimal CFC-TTA

Mode: evidence-inventory-only / failure analysis from existing artifacts.

Evidence status:
- Full clean results: `newly_run` artifacts from 2026-06-29/30.
- Dataset profile and dataflow tables: `newly_run_analysis` artifacts from 2026-06-29.
- No new full validation was launched for this report.

## 1. Final Result Gap

| Dataset | Best CFC-TTA mIoU | MLMP mIoU | Delta |
|---|---:|---:|---:|
| COCO-CD | 67.93 | 72.10 | -4.17 |
| Waterbirds-Seg | 50.97 | 52.18 | -1.21 |

Artifacts:
- `/home/wwww/hzl/results/minimal_cfc_final_clean_best_summary_20260630.csv`
- `/home/wwww/hzl/results/minimal_cfc_full_shortcut_aggressive_refine2_20260630/coco_cd_lr003_steps5/coco_cd/minimal_cfc/results.txt`
- `/home/wwww/hzl/results/minimal_cfc_full_shortcut_aggressive_refine2_20260630/waterbirds_lr003_steps5/waterbirds_seg/minimal_cfc/results.txt`

## 2. Dataset-Level Diagnosis

| Dataset | Background Pixels | Thing Pixels | Largest Background Component | Mean Classes/Image |
|---|---:|---:|---:|---:|
| COCO-CD | 84.18% | 15.82% | 83.89% | 2.0 |
| Waterbirds-Seg | 86.66% | 13.34% | 86.36% | 2.0 |

These two datasets are not merely context-heavy. They are binary/ternary shortcut datasets where background is an explicit evaluated class and dominates the image. This makes them different from VOC-style object-centric splits and from Cityscapes/PascalContext-style dense stuff splits:

- The foreground object occupies only about 13-16% of pixels.
- The largest background component alone covers about 84-86% of the image.
- Each image has only background plus one foreground class, so update signals are extremely easy to collapse into background sharpening.

Artifact:
- `/home/wwww/hzl/results/minimal_cfc_dataset_profiles_all_20260629/dataset_profile.csv`

## 3. Dataflow Diagnosis

Correction after implementation audit:

The earlier `background update mass` values below are pixel-weighted weight-map
statistics, not the actual region-normalized spatial loss contribution.  The
current implementation does use region-normalized entropy:

```text
L_sp = sum_R rho_R mean_{i in R} H(p_i) / (sum_R rho_R + eps)
```

Therefore, the correct accounting must distinguish pixel mass, region
coefficient mass, and actual loss numerator.

| Dataset | Background Update Mass | Thing Update Mass | Background Reliability | Object-Evidence Thing Reliability |
|---|---:|---:|---:|---:|
| COCO-CD | 92.00% | 8.00% | 0.661 | 0.309 |
| Waterbirds-Seg | 94.06% | 5.94% | 0.813 | 0.506 |

The table above should now be read only as pixel-weighted mass.  A new audit of
the actual loss contribution gives:

| Config | Dataset | Background Pixel Mass | Background Region Coeff. | Background Loss Contribution | Thing Loss Contribution |
|---|---|---:|---:|---:|---:|
| default | COCO-CD | 94.17% | 69.14% | 76.21% | 23.79% |
| default | Waterbirds-Seg | 94.83% | 76.70% | 75.49% | 24.51% |
| best shortcut | COCO-CD | 94.16% | 48.44% | 41.20% | 58.80% |
| best shortcut | Waterbirds-Seg | 93.70% | 48.96% | 39.31% | 60.69% |

Thus, the default configuration is indeed background-dominated in the actual
spatial loss, but the best shortcut configuration is not.  In the best
configuration, increasing the candidate budget and lowering the confidence
threshold shifts the actual loss contribution toward foreground regions.

The remaining failure under the best configuration is therefore not simply
"background consumes all gradients."  It is more specifically:

- the default method under-allocates actual loss to foreground on shortcut-heavy
  datasets;
- the aggressive/best method reallocates loss toward foreground, but the newly
  included foreground candidates are mostly tiny and low-quality;
- this improves mIoU but is still below MLMP and can harm mAcc.

Best-configuration source summary confirms the foreground-quality issue:

| Dataset | Source Type | Mean Reliability | Mean Area | Low-res GT Correct Frac. |
|---|---|---:|---:|---:|
| COCO-CD | background | 0.503 | 52.3 | 83.6% |
| COCO-CD | context-dominant thing | 0.041 | 8.5 | 6.4% |
| COCO-CD | mixed thing | 0.193 | 7.0 | 7.5% |
| COCO-CD | object-evidence thing | 0.288 | 1.6 | 1.5% |
| Waterbirds-Seg | background | 0.398 | 29.1 | 81.1% |
| Waterbirds-Seg | context-dominant thing | 0.039 | 6.9 | 6.9% |
| Waterbirds-Seg | mixed thing | 0.235 | 7.4 | 10.6% |
| Waterbirds-Seg | object-evidence thing | 0.288 | 2.3 | 3.2% |

This shows that the aggressive configuration obtains more foreground loss
contribution by admitting many small foreground candidates, but those candidates
are poorly aligned with low-resolution GT.  The update is no longer dominated by
background; it is dominated by noisy foreground entropy.

In the default configuration, selected background regions dominate the effective
update budget. The reason is not only pixel area; it is also reliability
assignment:

- Background candidates have high `s_keep`, high `s_cf`, and `source_weight=1`.
- Thing candidates are frequently diagnosed as context-dominant or mixed, so their source weights are reduced.
- The current background guard only suppresses a background region when it is actually more confident as a foreground class. It does not prevent correctly classified background from consuming most of the TTA update.

Per-image pixel-weight concentration:

| Dataset | Images with Background Mass > 90% | Images with Thing Mass < 10% |
|---|---:|---:|
| COCO-CD | 78.1% | 78.1% |
| Waterbirds-Seg | 78.1% | 78.1% |

This is one failure mode for the default configuration.  For the best shortcut
configuration, the stronger diagnosis is that foreground reallocation succeeds
numerically but the foreground candidates are too noisy and tiny to beat MLMP.

Artifacts:
- `/home/wwww/hzl/results/minimal_cfc_dataflow_failed_20260629/coco_cd/source_summary.csv`
- `/home/wwww/hzl/results/minimal_cfc_dataflow_failed_20260629/coco_cd/weight_summary.csv`
- `/home/wwww/hzl/results/minimal_cfc_dataflow_failed_20260629/waterbirds_seg/source_summary.csv`
- `/home/wwww/hzl/results/minimal_cfc_dataflow_failed_20260629/waterbirds_seg/weight_summary.csv`
- `/home/wwww/hzl/results/minimal_cfc_dataflow_failed_20260629/coco_cd/weight_rows.csv`
- `/home/wwww/hzl/results/minimal_cfc_dataflow_failed_20260629/waterbirds_seg/weight_rows.csv`

## 4. Component-Level Diagnosis

### CLS Stream

Turning off CLS has almost no effect:

| Dataset | Default | alpha_cls=0 |
|---|---:|---:|
| COCO-CD | 62.87 | 62.97 |
| Waterbirds-Seg | 47.55 | 47.61 |

The current CLS gate already ignores pure background candidates. Therefore, the main bottleneck is not global CLS adaptation.

### Candidate Coverage

Increasing candidate coverage without stronger optimization does not help:

| Dataset | alpha_cls=0 | cover20/conf0.25 | cover32/conf0 |
|---|---:|---:|---:|
| COCO-CD | 62.97 | 62.91 | 62.89 |
| Waterbirds-Seg | 47.61 | 47.54 | 47.52 |

This means the issue is not simply missing candidate regions. Adding more regions mainly adds more background or low-quality foreground candidates.

### Optimization Strength

Aggressive update helps but does not solve the mechanism:

| Dataset | cover32/conf0 + lr=1e-3, steps=5 | lr=3e-3, steps=5 | MLMP |
|---|---:|---:|---:|
| COCO-CD | 66.14 | 67.93 | 72.10 |
| Waterbirds-Seg | 50.13 | 50.97 | 52.18 |

The improvement shows that CFC-TTA can still recover useful adaptation signal. However, the mAcc drops sharply in the best aggressive setting, especially on Waterbirds-Seg (75.36 to 68.13), indicating a tradeoff rather than a clean fix.

Artifacts:
- `/home/wwww/hzl/results/minimal_cfc_clean_full_all_20260629_latest/summary.csv`
- `/home/wwww/hzl/results/minimal_cfc_full_targeted_shortcut_refine_20260630/*/summary.csv`
- `/home/wwww/hzl/results/minimal_cfc_full_shortcut_aggressive_refine2_20260630/*/summary.csv`

## 5. Why COCO-CD and Waterbirds Are Hard for the Current Method

1. The current method solves target-vs-context reliability mainly for thing candidates.
   For background candidates, it uses `rho = sqrt(s_keep * s_cf) * guard_bg`, with source weight effectively equal to 1 unless foreground leakage is detected.

2. On COCO-CD and Waterbirds, background is not just nuisance context. It is a correct evaluated class and the dominant visual region.
   Therefore, the method assigns high reliability to background, and the spatial entropy objective mostly sharpens background predictions.

3. Foreground labels are small and shortcut-sensitive.
   On COCO-CD, object-evidence thing regions have low GT correctness in the sampled low-resolution data. On Waterbirds, context-dominant thing regions are often more GT-correct than object-evidence regions, because landbird/waterbird labels are strongly context-defined.

4. Region-normalized entropy prevents direct pixel-count domination.
   The previous pixel-mass statistic overstated the actual loss contribution.
   However, in the default configuration background still contributes about
   75% of the spatial-loss numerator.  In the best configuration, this is fixed
   by admitting more foreground candidates, but those candidates are noisy.

5. More candidate coverage does not fix this.
   Coverage sweeps did not improve mIoU, which suggests the failure is update allocation rather than proposal recall alone.

## 6. Most Likely Fix

The next method change should not be a blind background cap.  The audit shows
that a cap is plausible for the default configuration, but the best configuration
already has foreground-majority loss contribution.  The stronger target is:

1. prevent background domination when foreground candidates exist;
2. prevent noisy tiny foreground candidates from taking over after aggressive
   coverage expansion.

A minimal role-balanced reliability rescaling is still useful, but it should be
paired with a foreground quality floor or area-aware candidate quality check.
Otherwise, it may only move update mass from correct background to bad
foreground regions.

Possible minimal fix:

```text
L_sp =
  lambda_fg * mean_weighted_entropy(thing candidates)
  + lambda_bg * mean_weighted_entropy(background candidates)
```

with a simple constraint such as:

```text
background update mass <= cap_bg
```

or equivalently:

```text
rho_bg = rho_bg * min(1, kappa * thing_mass / (background_mass + eps))
```

This keeps the current simple method structure and addresses the default
background-budget failure.  For the best shortcut configuration, also consider:

```text
rho_thing = rho_thing * q_fg
```

where `q_fg` is a lightweight quality gate based on region area, initial region
confidence, or agreement with the original dense prediction.  This should avoid
using many tiny foreground fragments as the main adaptation signal.

For Waterbirds, an additional lightweight option is to treat `landbird`/`waterbird` as contextual thing labels: do not hard punish context support, but require foreground bird-region support before allowing the context label to dominate adaptation.

## 7. Claim Readiness

- Ready claim: The earlier 92%/94% number is pixel-weighted mass, not actual
  region-normalized loss contribution.
- Ready claim: Under the default configuration, actual spatial-loss contribution
  is still background-heavy, around 75%.
- Ready claim: Under the best shortcut configuration, actual spatial-loss
  contribution shifts to thing regions, but selected thing candidates are tiny
  and poorly GT-aligned in the sampled low-resolution audit.
- Ready claim: CLS is not the main failure source on these datasets.
- Ready claim: More candidate coverage alone is not enough.
- Weakened claim: CFC-TTA generally solves shortcut-heavy datasets. Current evidence does not support this for COCO-CD/Waterbirds.
- Next experiment needed: role-balanced reliability rescaling plus foreground
  quality control on COCO-CD and Waterbirds-Seg, with regression on VOC20,
  VOC21, COCO Object, and Cityscapes.

Additional artifacts:
- `/home/wwww/hzl/results/minimal_cfc_loss_contribution_audit_best_20260630/loss_contribution_summary.csv`
- `/home/wwww/hzl/results/minimal_cfc_dataflow_shortcut_best_20260630/coco_cd/source_summary.csv`
- `/home/wwww/hzl/results/minimal_cfc_dataflow_shortcut_best_20260630/waterbirds_seg/source_summary.csv`
