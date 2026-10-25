# Candidate Bottleneck and Semantic-Role Baseline Report

Date: 2026-07-17

This is an independent experiment report. It does not replace or merge into the existing four-dataset Table 4(c), whose purpose is confidence-controlled source-risk analysis.

## Experiment Evidence

### 1. Candidate-oracle bottleneck with complete-method anchors

Evidence type: `newly_run` and `newly_run_analysis_of_newly_run`.

The internal factorial disables CLS and crosses predicted versus GT-component candidates with uniform versus CFC reliability. NoAdapt, TENT, MLMP, and complete CFC-TTA are reference anchors on the same deterministic 240-image prefixes; they are not extra factorial cells.

| Dataset | NoAdapt | TENT | MLMP | CFC-TTA | Pred+uniform | Pred+CFC | Oracle+uniform | Oracle+CFC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VOC20 | 75.79 | 70.89 | 81.96 | 83.49 | 77.84 | 78.71 | 79.99 | 78.04 |
| PascalContext60 | 27.47 | 25.68 | 28.40 | 28.83 | 29.88 | 30.29 | 29.55 | 29.88 |
| COCO Object | 27.67 | 25.10 | 28.72 | 28.96 | 26.39 | 25.20 | 25.42 | 25.93 |
| COCO-Stuff | 18.90 | 17.72 | 19.33 | 20.21 | 17.73 | 17.85 | 16.67 | 17.79 |

GT candidates raise area-weighted candidate purity to 1.0 and component recall at IoU 0.5 from 0.117 to 0.836 on VOC20, 0.080 to 0.599 on PascalContext60, 0.060 to 0.544 on COCO Object, and 0.053 to 0.949 on COCO-Stuff. Downstream mIoU nevertheless changes non-monotonically: under uniform weights, oracle candidates change mIoU by +2.15, -0.33, -0.97, and -1.06 points, respectively. Better geometry is therefore not a performance upper bound; candidate composition and update allocation interact.

Artifacts:

- `candidate_oracle_baseline_comparison/candidate_oracle_with_baseline_anchors.{csv,md}`
- `candidate_oracle_4datasets_240/decomposition.csv`
- `candidate_oracle_quality_4datasets_240/candidate_quality_summary.csv`

### 2. Matched baseline comparison by semantic role

Evidence type: `newly_run` and `newly_run_analysis_of_newly_run`.

All methods use the same clean-image prefix, NaCLIP ViT-L/14 backbone, and frozen pre-adaptation CFC-TTA candidate masks. TENT and MLMP use public/default optimizer settings; CFC-TTA uses its canonical dataset-specific clean configuration.

| Dataset | NoAdapt | TENT | MLMP | CFC-TTA | CFC - MLMP |
|---|---:|---:|---:|---:|---:|
| VOC20 | 75.79 | 70.89 | 81.96 | 83.49 | +1.53 |
| PascalContext60 | 27.47 | 25.68 | 28.40 | 28.83 | +0.43 |
| COCO Object | 27.67 | 25.10 | 28.72 | 28.96 | +0.24 |
| COCO-Stuff | 18.90 | 17.72 | 19.33 | 20.21 | +0.88 |
| COCO-CD | 62.76 | 71.76 | 71.17 | 73.04 | +1.87 |
| Waterbirds-Seg | 39.96 | 45.36 | 42.59 | 49.26 | +6.67 |
| VOC21 | 48.82 | 43.62 | 49.07 | 51.57 | +2.50 |
| PascalContext59 | 30.07 | 28.29 | 31.25 | 31.54 | +0.29 |
| Cityscapes (60 images) | 32.46 | 29.20 | 32.86 | 34.05 | +1.19 |

Macro averages below are unweighted over available dataset-role strata with at least 20 initially-correct candidates. Region accuracy delta, correct-to-harm, and flip values are percentage points or percentages.

| Method | Role | Strata | Region gain | Correct-to-harm | Mean flip | Flip incidence |
|---|---|---:|---:|---:|---:|---:|
| TENT | thing | 9 | +2.89 | 11.31 | 14.23 | 56.29 |
| TENT | stuff | 4 | +0.25 | 23.54 | 18.72 | 66.60 |
| TENT | background | 4 | +1.23 | 8.79 | 9.39 | 43.31 |
| MLMP | thing | 9 | +2.20 | 12.54 | 13.19 | 52.35 |
| MLMP | stuff | 4 | +1.42 | 19.79 | 13.39 | 58.00 |
| MLMP | background | 4 | -0.03 | 5.52 | 4.18 | 17.31 |
| CFC-TTA | thing | 9 | +6.26 | 16.03 | 20.79 | 69.32 |
| CFC-TTA | stuff | 4 | +2.71 | 19.85 | 20.20 | 68.37 |
| CFC-TTA | background | 4 | +1.28 | 4.43 | 5.42 | 24.87 |

The result is not a universal harm-detector win. CFC-TTA produces much larger mean region gains, but its canonical clean configurations also induce more flips on several ordinary datasets. The strongest aligned result is Waterbirds-Seg: compared with MLMP, CFC-TTA raises thing-region accuracy gain from +2.01 to +16.85 points while reducing thing correct-to-harm from 35.48% to 20.16%. On Cityscapes, CFC-TTA raises thing/stuff region gain from +2.92/+1.73 to +7.50/+3.52 points and lowers correct-to-harm from 17.73%/25.04% to 11.84%/18.80%. Background has the lowest macro correct-to-harm under CFC-TTA (4.43%), but CFC is not uniformly safer for thing or stuff.

Artifacts:

- `matched_baseline_role_report/matched_baseline_role_report.md`
- `matched_baseline_role_report/matched_role_by_dataset.csv`
- `matched_baseline_role_report/matched_role_macro.csv`
- `matched_baseline_role_report/cfctta_vs_mlmp_by_dataset_role.csv`
- `matched_baseline_role_9datasets_240_city60/protocol.json`

## Protocol Risks

1. These are deterministic diagnostic subsets and a single seed, not full-set benchmark estimates.
2. Eight datasets use 240 images; Cityscapes uses 60 images because every image generates 36 patches. Dataset-role macro averaging prevents tiled datasets from dominating.
3. TENT/MLMP use their public default optimization while CFC-TTA uses canonical dataset-specific settings. The comparison matches data, backbone, and candidate masks, but not every optimizer hyperparameter.
4. Frozen candidates come from pre-adaptation CFC-TTA dense predictions. This is necessary for matched region comparison but should not be interpreted as each baseline's native proposal quality.
5. Candidate-oracle rows use GT information and are diagnostic only. They are neither deployable nor strict upper bounds.
6. Correct-to-harm is measured after a joint image/batch update. It is an association with a candidate's diagnostics, not a per-candidate causal intervention label.
7. NaN values in raw role CSVs occur only where a dataset has no candidate of that role or no initially-correct candidate. Aggregate mIoU files contain no NaNs.
8. The superseded absent-class-as-zero run is isolated under `matched_baseline_role_9datasets_240_invalid_absent_class_zero` and marked invalid; none of its subset mIoUs are used here.

## Results / Setup Draft

On matched deterministic clean subsets, CFC-TTA exceeds MLMP on all nine evaluated rows by 0.24--6.67 mIoU. Freezing the same pre-adaptation candidate masks further reveals role-dependent behavior. CFC-TTA yields larger macro region-accuracy gains than MLMP for thing (+6.26 versus +2.20 points), stuff (+2.71 versus +1.42), and background (+1.28 versus -0.03). This gain is not uniformly accompanied by fewer flips: the clearest joint improvement occurs on shortcut-heavy Waterbirds-Seg and on the Cityscapes subset, whereas several ordinary clean datasets exhibit more aggressive candidate changes. Together with the separate confidence-controlled Table 4(c), the result supports role-conditioned update allocation rather than a universal candidate-level harm detector.

## Paper-Derived Missing Experiments

The supplied papers were reviewed at the level of their experiment variables rather than only their method summaries.

1. **P0: Vocabulary-composition stress.** DenseVLM Table 4 changes category sets; FreeCP compares full and GT vocabularies and isolates redundancy/ambiguity; VocAlign Figures 5--6 control Top-K size against random classes. Run benchmark labels, labels plus mapped synonyms, equal-count related/co-occurring distractors, and equal-count unrelated distractors on VOC20, PascalContext60, COCO Object, and COCO-Stuff for NoAdapt/MLMP/CFC-TTA. Report base-class mIoU retention, added-label false-positive area, role-specific correct-to-harm, reliability shift, and best/worst classes.
2. **P0/P1: Semantic-correct but localization-wrong subset.** PACL first verifies patch-text alignment rather than assuming image-level recognition transfers locally. Select samples where the image-level vocabulary contains the GT class but the corresponding confident candidate has IoU below 0.3, then compare NoAdapt/MLMP/CFC-TTA region-IoU delta, correct-to-harm, and wrong-location confident area.
3. **Appendix: Hard-zero versus soft-blur interventions.** Mask-adapted CLIP Figure 1 identifies a masked-image domain gap; LaVG Table 4 compares grounding schemes including Gaussian blur. Compare hard zero, mean fill, Gaussian blur, and the current soft view while holding candidate masks and update settings fixed.
4. **Appendix: Candidate budget/cost Pareto.** LaVG reports cost and CLIPtrase jointly analyzes accuracy and FLOPs. Sweep only candidate budget, and report mIoU, runtime, peak memory, and diagnostic forwards.
5. **Optional: Prompt-bank robustness.** MLMP and Seg-TTO show that prompt design is an experimental variable. Compare one shared template with the shared seven-template bank for MLMP and CFC-TTA; keep this separate from the confidence-source contribution.

Existing evidence already covers clean three-seed evaluation, shortcut datasets, module and bridge ablations, source AUC/reliability bins, small-object regression, runtime, second backbone, same-class multi-object, five counterfactual perturbations, and paired bootstrap. Repeating full corruptions, adding a 22-domain suite, or introducing graph/SAM components would not address the remaining mechanism gap.

## Remaining Blockers

- Candidate-oracle and matched semantic-role baseline comparisons are complete.
- Vocabulary-composition stress has not yet been run and is the highest-value missing OVSS-specific experiment.
- Semantic-correct/localization-wrong is the next most direct mechanism diagnostic.
