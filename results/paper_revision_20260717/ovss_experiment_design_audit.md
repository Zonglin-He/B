# OVSS Experiment Design Audit

Date: 2026-07-17

## Scope

This audit maps experiment patterns from the seven supplied OVSS papers and the most directly related OVSS test-time adaptation literature to the current TRACE paper. It distinguishes evidence already present, evidence already computed but not yet emphasized, and genuinely missing tests. The goal is a reviewer-facing evidence chain, not a larger collection of loosely related benchmarks.

## Current Evidence Chain

The current draft already covers most of the core evaluation layers expected in a strong OVSS-TTA paper:

1. **Breadth:** seven standard clean OVSS benchmarks with three-seed reporting, plus COCO-CD and Waterbirds-Seg.
2. **Matched adaptation evidence:** MLMP comparisons under a controlled protocol and paired bootstrap confidence intervals.
3. **Mechanism evidence:** prediction utility versus update reliability, matched target/context regions, confidence-controlled harm-detection AUC, and dataset/role-controlled reliability quartiles.
4. **Method evidence:** leave-one-module-out ablation, small-object regression, CLS support analysis, and final segmentation examples.
5. **Deployment evidence:** measured runtime, forward calls, and memory on three datasets.

The revised Figure 5, Figure 7, and Table 4(c) close three presentation gaps: the fragment examples are now genuine scored components, the COCO-CD qualitative metric now matches the displayed target region, and the reliability analysis now controls both raw confidence and dataset/semantic-role composition.

## Transferable Patterns From The Supplied Papers

| Paper | Experiment pattern worth adopting | TRACE counterpart |
|---|---|---|
| Mask-Adapted CLIP, CVPR 2023 | Oracle decomposition identifies whether masks or region-language classification are the bottleneck. | Add a candidate-oracle decomposition to separate candidate construction from source scoring and adaptation. |
| PACL, CVPR 2023 | A proxy task and AUROC test the claimed local semantic property before downstream segmentation. | The expanded harm-detection AUC is the right analogue; report confidence-controlled AUC and clustered uncertainty. |
| CLIPSelf, ICLR 2024 | Region recognition is split across boxes, thing masks, and stuff masks; sanity controls rule out trivial explanations. | Report source-risk quality by thing/stuff/background and compare against confidence, entropy, and area-only controls. |
| DenseVLM, ICCV 2025 | Foreground/background alignment is analyzed category-wise, and thing/stuff contributions are separated. | Add a role-specific table for candidate coverage, harmful-update prevalence, and risk AUC. Do not pool roles without control. |
| CASS, CVPR 2025 | Broad datasets are complemented by a secondary metric, backbone transfer, and component-level qualitative evidence. | Keep mAcc beside mIoU where class imbalance matters; treat the second-backbone study as a compatibility test, not a guaranteed win. |
| LaVG | Mask discovery and grounding are evaluated separately, with scheme comparisons and cost. | Use the candidate-oracle experiment and candidate-budget/runtime Pareto to expose where TRACE gains and cost originate. |
| CLIPtrase | A mechanism proxy, module ablation, sensitivity, integration with another segmentation system, and efficiency form a coherent package. | Put counterfactual-view robustness and candidate-budget sensitivity in the appendix; retain one concise main-paper efficiency table. |

## Direct OVSS-TTA Literature

- **MLMP** is the closest matched setting: per-test-sample optimization, standard OVSS datasets, local and CLS entropy, multiple prompts/layers, conventional TTA baselines, and synthetic corruptions. It should remain the principal baseline.
- **Seg-TTO** evaluates specialized-domain adaptation across many tasks and multiple OVSS systems. Its strongest transferable pattern is cross-backbone integration and explicit accuracy-cost analysis, but its prompt/feature optimization protocol is not identical to TRACE.
- **SemLA** retrieves and fuses a pre-built LoRA library over multiple target domains. It is relevant related work, but its external adapter library and domain-adaptation protocol make it unsuitable as a drop-in main-table baseline unless that resource assumption is reproduced.

## Recommended Experiment Set

### P0: Add Before Submission

#### 1. Candidate-Oracle Bottleneck Decomposition

Run four conditions on VOC20, PC60, COCO Object, and COCO-Stuff:

| Candidate masks | Reliability/update policy | Question answered |
|---|---|---|
| Predicted | Uniform | Pixel/region baseline |
| Predicted | TRACE | Full practical method |
| GT connected components | Uniform | Candidate upper bound without source tracing |
| GT connected components | TRACE | Whether source tracing remains useful with oracle candidates |

Report mIoU, candidate recall, oracle mask coverage, initially-correct-region harm, and correct-to-harm. This is the highest-value missing experiment because it reveals whether failures come from candidate construction or confidence-source diagnosis.

#### 2. Role-Specific Source Diagnosis

For thing, stuff, and background candidates, report:

- candidate count and pixel/region coverage;
- initially correct count and harmful-update prevalence;
- source-risk AUC, entropy AUC, and inverse-confidence AUC;
- reliability calibration or quartile correct-to-harm.

Use dataset-stratified macro averaging or report each dataset separately. Background currently has few or no harmful positives in some diagnostic subsets, so mark AUC as undefined rather than pooling it into an apparently stronger result.

#### 3. Compact Accuracy-Cost Pareto

The profiler already contains candidate budgets of 2, 4, 8, and deployment mode. Convert it into one compact plot or table with mIoU, adaptation time, forward calls, and peak memory. Clearly distinguish bounded profiling mIoU from full-dataset benchmark mIoU. This turns the diagnostic-view overhead from an isolated limitation into a measurable trade-off.

### P1: Appendix Strengthening

#### 4. Counterfactual Perturbation Robustness

The five-view study already exists. Report no-CF, blur-shift, gray-blur, channel-shift, and local-mean on VOC20, PC60, COCO-CD, and Waterbirds-Seg. The result is stable on the latter three datasets, while VOC20 is sensitive to local-mean replacement. State this boundary explicitly instead of selecting only the best perturbation.

#### 5. Same-Class Multi-Object Diagnostic

Keep this as a focused diagnostic rather than a main gain table. Report sibling-aware versus self-only suppression on images containing multiple instances of the same class, with leakage, correct-to-harm, and target IoU. Existing evidence suggests a small safety benefit rather than a large aggregate mIoU gain.

#### 6. Prompt Robustness Under Fairness Control

Use the same prompt bank for MLMP and TRACE and vary only the shared bank, for example one canonical template, the current seven-template bank, and an ImageNet-style ensemble. Report the TRACE-minus-MLMP delta, not only absolute performance. This tests whether source reliability survives prompt choice without turning prompt engineering into part of the method claim.

#### 7. Failure Taxonomy

Add two or three examples for each of: missed candidate, wrong class-text alignment, context-defined label, unstable intervention, and over-fragmentation. Each example should show the exact candidate, reliability components, pre/post prediction, and target IoU. This is more informative than adding another page of selected successes.

## Evidence Already Available But Not Ready As A Positive Main Claim

The SCLIP second-backbone run used a frozen NaCLIP-oriented TRACE configuration. TRACE approximately matches MLMP on PC60 (22.03 versus 22.00 mIoU) but trails it on VOC20 and COCO Object. This supports a compatibility limitation, not backbone-independent superiority. A fair positive transfer claim would require a small protocol-level recalibration shared across datasets or a second backbone whose dense outputs satisfy the current candidate assumptions.

## Experiments Not Recommended For The Main Paper

- Completing every remaining synthetic corruption after the user stopped that suite. The clean, shortcut, mechanism, and paired evidence already serve the current story more directly.
- A unified hyperparameter configuration solely for aesthetics. Dataset-specific settings are already disclosed; forcing unification would test a different claim.
- Treating Seg-TTO or SemLA as directly matched baselines without reproducing their distinct resource and domain assumptions.
- More component toggles that do not correspond to named Method modules.

## Suggested Paper Order

1. Standard clean and shortcut-oriented main results.
2. Paired uncertainty.
3. Prediction utility versus update reliability.
4. Expanded controlled source-risk analysis.
5. Method-aligned module ablation and small-object regression.
6. Candidate-oracle bottleneck decomposition.
7. Accuracy-cost Pareto.
8. Qualitative successes and failure taxonomy.
9. Appendix: role-specific details, perturbation robustness, prompt robustness, and same-class multi-object analysis.

## Claim Boundary

The controlled AUC analysis establishes that TRACE reliability ranks harmful updates beyond entropy and raw confidence in the diagnostic split. It does not establish causal identification of the evidence source. Qualitative examples are selected illustrations, not frequency estimates. The candidate-oracle experiment is therefore the most important remaining bridge between the source-diagnosis mechanism and end-to-end OVSS performance.
