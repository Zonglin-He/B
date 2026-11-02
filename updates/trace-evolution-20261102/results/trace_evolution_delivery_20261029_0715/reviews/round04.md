# Independent Review: TRACE, Round 04

**Paper:** TRACE: Context-Probed Region Allocation for Test-Time Adaptation in Open-Vocabulary Semantic Segmentation  
**Overall rating:** 6/10 (weak accept)  
**Acceptance recommendation:** Weak accept  
**Confidence:** 4/5  
**Fatal validity issue evidenced:** No.

## Review Scope and Integrity

I read the actual 15-page combined `paper/build/main.pdf`, including Appendices A-F, and the eight-page standalone `paper/build/supplement.pdf`. I inspected rendered pages, including the candidate illustrations, pipeline, results tables, qualitative segmentations, and mathematical appendix. Table numbers below refer to the combined PDF; the standalone supplement renumbers its tables and equations.

The frozen packet is `/home/wwww/hzl/results/trace_evolution_review_round04_20261029_0703`. All **65/65 manifest-listed files passed SHA-256 verification**, including both PDFs, TeX, tables, and figures. The SHA-256 of `packet_manifest.json` itself is `87101bab5260be343cf82a5d092309b54848af766c2aaeecbdbfadb580ef302d`. This establishes internal consistency with the supplied manifest, not an independently timestamped origin for it.

I also inspected bounded canonical metadata and relevant frozen executed source from the permitted full and budget evidence roots. All **149 full-suite runtime hashes** and **144 budget-suite runtime hashes** matched. An additional **67 selected full-suite metadata hash checks** and **eight budget canonical-result hash checks** had no mismatches. The full prepared identity matches the packet: `e0f7374042eadb65fbc8146de6dfcf35a71151dc1c95c115b0d4cf9c8444edaa`. The selection seal matches across suites and selects C. The six full canonical results agree with the manuscript; checking their already-recorded pooled intersection/union arithmetic reproduces their reported mIoUs. The six prediction-seal links match the global barrier, results, and scoring seals. All six recorded four-image gates passed, with zero recorded gradient differences.

This is **not** a fresh rehash of all evidence prediction payloads, an independent replay, or proof of every historical access event. I did not run GPU work, inference, scoring, bootstrap analyses, or repairs, or scan large prediction/model payload directories. No earlier review reports, author claim ledgers, handoffs, or planning documents were consulted. A workspace memory index was queried for scope at the outset; its historical experimental findings were not used as evidence or to determine this assessment. No manuscript, figure, or frozen evidence was modified, and no agents or Git commits were used.

## Summary and Contribution

TRACE uses connected components of reset-model query predictions as regional units. For each selected region, it preserves that region's input pixels, applies a fixed contrast/channel-shift transform outside it, and measures the original query probability at the original token positions. Detached support scores weight region-averaged prompt entropy, normalized separately within source images. Only visual LayerNorm affine parameters adapt, with episodic model/optimizer reset.

The contribution is a coherent, relatively modest allocation rule, not a new dense backbone, optimizer, or identified causal estimator of correctness. The distinction between observed confidence and support under a specified context change is useful. The strongest evidence is the matched Uniform/Confidence/C comparison, not the literature-reference table or an unproven claim that high support implies safe updates.

## Strengths

1. **Direct, complete within-implementation allocation evidence.** Table 1 evaluates complete populations and keeps candidates, preprocessing, initialization, readouts, optimization settings, and order matched within each protocol. C exceeds Confidence in all six recorded runs. In particular, the Object and CD differences are substantial relative to the much smaller PC60, VOC21, and Stuff differences. These are meaningful finite-run comparisons even without significance claims.
2. **A simple method with an explicit mathematical interpretation.** Sections 3.1-3.3 and Appendix B distinguish query identities from evaluation categories, regional means from pixel-area weighting, and equal coefficient budgets from equal gradients. Rescaling and singleton limits are stated rather than hidden. Keeping the diagnostic query fixed avoids transformed-mask matching ambiguity.
3. **Method and executed implementation are consistent in the inspected path.** The single-C branch actually constructs one view, reads mean probabilities, detaches the scores, and groups the objective by source image. The appendix explicitly documents the different diagnostic/inference and adaptation readouts; this is not an undisclosed implementation mismatch.
4. **The selection and comparison interpretation are appropriately bounded.** C is globally calibration-selected before confirmation; all competing modes and negative outcomes are retained. Tables 2-3 distinguish reported and local reference operating points from matched controls, retain the negative VOC20 reference difference, and do not turn shared backbone names into matched-protocol claims.
5. **Inspectable provenance and useful qualitative honesty.** The frozen sources, manifests, results, and seal chain make the submission unusually auditable. Figure 4 includes a negative example and gives a reproducible extreme-selection rule rather than representing the cases as typical. Computation reporting distinguishes encoder inputs, calls, and measured model-side latency.

### Verified Full-Split Results

These are descriptive results, not confidence intervals or six independent population replications.

| Protocol | Images | Uniform | Confidence | TRACE(C) | C minus Confidence, pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| PC60 | 5,105 | 30.383771 | 30.337222 | 30.693152 | +0.355929 |
| COCO Object | 5,000 | 29.807068 | 30.825096 | 32.594583 | +1.769488 |
| VOC21 | 1,449 | 52.442453 | 52.325475 | 52.621491 | +0.296016 |
| VOC20 | 1,449 | 81.783666 | 82.177487 | 83.487219 | +1.309732 |
| COCO-CD | 500 | 66.216729 | 67.664308 | 74.982118 | +7.317811 |
| COCO-Stuff | 5,000 | 21.744498 | 21.793356 | 21.836221 | +0.042865 |

## Prioritized Weaknesses

### W1. Generalization and uncertainty remain limited

**Severity: High. Classification: documented exposure plus missing independent evidence, not demonstrated inference-time leakage.**

Section 4.1, Table 1, and Appendix A.7 disclose that the official evaluation splits and inherited per-dataset configurations were exposed during development. The appendix identifies full-validation-conditioned learning-rate choices and unresolved origins for some other settings. Appendix C further says historical July candidate-pool exclusions cannot be fully certified. The new seal prevents reselection during this run; it does not make that development history disappear.

The evaluation is also one backbone, one deterministic model seed, one fixed transform, and one 224 x 224 crop per resized image. The three confirmation orders reuse the same images. VOC20 and VOC21 share source images. Therefore neither six positive signs nor three orderings establish independent generalization or significance. This matters particularly for the +0.043 Stuff, +0.296 VOC21, and +0.356 PC60 differences. The very large CD gain is persuasive for its recorded three-category protocol but is not itself evidence of broad vocabulary or native-resolution robustness.

The paper largely states these limitations correctly. They still reduce the strength of the empirical contribution; disclosure is not a substitute for independent validation. I do not infer a fatal bug or test-time access to ground truth from this history.

### W2. There is no matched reset/no-adaptation reference for the current pipeline

**Severity: Medium. Classification: missing experimental evidence.**

Tables 1 and 3, Appendix E, and Appendix F do not establish the performance of the exact current diagnostic/final readout with adaptation disabled. Table 1 compares three adapting allocation policies. The historical native No Adapt row is explicitly not a certified reset readout of the new method. Thus the results answer which allocation is better here, but not how much this TTA pipeline gains over its own starting predictor on each protocol, or whether an adapting arm can still underperform that predictor.

Likewise, Tables 2-3 cannot establish competitive superiority over matched external methods: Table 2 imports reported values with different or unverified geometry, readouts, budgets, and tuning histories; CD's historical native controls retain different settings, including 0.001 learning rate and 10 updates versus C's 0.0086 and five updates. The manuscript does not claim otherwise, so I do not label this comparison misleading or invalid. It leaves the practical benefit and competitive placement less directly established than the regional-allocation effect.

### W3. Image normalization is mathematically established but its accuracy contribution is not isolated for C

**Severity: Medium. Classification: mixed evidence and a missing selected-method ablation, not a mathematical error.**

The equal coefficient-mass result in Equations 6-7 and Appendix B is correct. Table 1 holds image normalization fixed for all arms, however, so it cannot identify normalization's effect. Tables 5-6 compare image-normalized minimum against episode-normalized minimum, not image-C against episode-C. Those minimum controls have mixed outcomes: image-min is below old-min in all three PC60 confirmation orders, while it is above old-min in all three Object orders. Moreover, old-min exceeds selected C on PC60, and the equally single-probe T exceeds C on Object in every confirmation order.

These results do not contradict the stated algebra or the declared minimax selection rule. They mean the evidence supports C as one locked operating point, not C as the empirically best probe or image normalization as a general accuracy improvement. A direct C/image-versus-episode comparison is the cleanest missing ablation. Regional averaging's contribution is also not isolated from allocation in the main full-split controls.

### W4. The mechanistic explanation and distinctiveness remain less developed than the aggregate effect

**Severity: Medium. Classification: missing explanatory evidence and limited demonstrated breadth.**

The support-extreme examples in Figure 1 show that the chosen perturbation can separate initially similar confidence values. Figure 4 shows selected differences in final predictions after joint adaptation. Neither establishes how often useful separation occurs, whether the score predicts regional correctness within comparable confidence/area strata, or whether retaining the candidate-specific assignment matters beyond changing the concentration of weights. The paper appropriately does not claim such calibration or causal identification.

There is no reported current-method within-image score-permutation control, transform-strength/channel-shift ablation, or distributional account of singleton candidates and near-zero supports. The latter matters because a positive singleton score cancels, and uniformly low positive support does not suppress an image's budget. The fixed 0.35/channel-permutation choice is reproducible, but its transferability is not tested. Since the core algorithm is perturbed-confidence weighting plus an explicit normalization choice, a small number of targeted controls would add more scientific insight than more unmatched reference rows. This is a concern about demonstrated explanatory depth, not an allegation that a specific prior method already duplicates TRACE.

### W5. Portable reproduction and presentation can be improved

**Severity: Low. Classification: reproducibility/presentation gap.**

Appendix A is detailed about prompts, logits, transforms, category collapsing, and configuration. Nevertheless, the PDF omits an explicit hardware/software identification next to the timing results and a concise runnable reproduction entry point. The allowed full canonical `initial_state.json` files record RTX 5090, Python 3.10.13, and PyTorch 2.7.0+cu128; these should be surfaced in the reproducibility description rather than requiring local-path metadata discovery. A release should package the actually executed runtime/configuration and dependency information portably, not rely only on `/home/wwww/hzl/results/...` paths.

The visual material is legible in the inspected renderings, but Figure 3 largely repeats Table 1 and repeated provenance caveats consume space that could carry per-class or mechanism evidence. No conference page-limit violation is asserted here.

## Mathematical and Implementation Assessment

- **Correct within the stated finite active-set assumptions:** the normalized regional weights sum to one within an image; each active image has coefficient mass `1 / |I+|`; common positive score scaling cancels when the active set is unchanged; singleton cancellation follows. Region averaging removes an explicit area multiplier, not area dependence of features, candidate selection, or gradients.
- **The probe matches the specified operation.** In full-root `runtime/trace_ovss/method/image_balanced.py:67-120`, the C branch builds candidate-preserving views and averages the original query's probabilities at stored indices. `runtime/trace_ovss/method/trace/interventions.py:171-182` implements the full-crop mean, 0.35 contrast, channel shift, clamp, and renormalization. This is a particular perturbation response, not a proven causal confidence decomposition.
- **The objective matches the equations.** `image_balanced.py:18-50` groups by source ID and averages the per-image objectives; `runtime/trace_ovss/method/trace/objective.py:14-59` constructs fixed scalar weights and averages prompt/token entropies. The candidate query is not secretly used as a cross-entropy target. Appendix A.4's separate adaptation and inference readouts match `runtime/trace_ovss/method/adapter.py` and `runtime/trace_ovss/models/dense_clip.py:105-124`.
- **No evidenced runtime label leakage.** Full-root `runtime/tools/evolution/run_cell.py:77-103` defines an RGB-only dataset/collator. `runtime/tools/evolution_image_full/run.py:254-306` executes independent, reset arms and checks candidate identity. Annotation bytes are opened in the separate `score` path at lines 424-476, guarded by the global prediction barrier. Source inspection and the consistent seal links support the claimed separation; they do not reconstruct every historical action.
- **No demonstrated nonfinite-repair substitution in the full results.** Although inherited helpers contain `nan_to_num`, the full worker's `finite_observation` at `run.py:196-241` checks raw model outputs before downstream sanitization and checks parameters, gradients, and optimizer state around updates. The completed guarded runs are evidence against the specific accusation that nonfinite outputs were silently substituted in this full execution, not a theorem of semantic safety.
- **Computation statements are appropriately scoped.** The recorded C diagnostic input counts match Table 8. The `(S+2)P+n` formula counts inputs rather than batched calls. C costs about 42.6% more model-side time than Confidence on PC60 and 43.9% more on Stuff; the latter accompanies only +0.043 pp. The paper reports this unfavorable tradeoff rather than claiming universal acceleration. Halving diagnostic inputs relative to two probes does not halve total work or imply a full-split measured dual-probe speedup.

## Questions for the Authors

1. Can an exact reset/no-adaptation baseline be provided for the same current readout, query extensions, resize geometry, and checkpoint, without substituting the historical native row? If it is unavailable, retain that distinction explicitly.
2. What prospective evaluation could test the locked C method without recycling the disclosed tuning exposure, and what uncertainty unit would preserve complete jointly adapted episodes? Reordering the existing confirmation images alone does not answer this.
3. How much does source-image normalization change the selected C method relative to episode-normalized C, and how often does singleton cancellation make C's absolute support irrelevant?
4. What evidence would distinguish candidate-specific context information from generic weight concentration or transform-specific effects? A within-image score permutation preserving the score multiset and a targeted transform control would address different parts of this question.

## Highest-Value Feasible Fixes

1. **From existing records, without claiming new validation:** expose per-class effects, particularly the three CD classes, and the current accuracy/cost tradeoff; consolidate the result/protocol distinctions into one concise location. Add hardware, software, dependency, and reproduction-entry-point information. These are proposed author actions, not analyses executed during this review.
2. **For a subsequent statistical analysis of retained counts:** estimate conditional uncertainty by resampling complete adaptation episodes, keeping arm pairing and re-pooling class counts before computing mIoU. Do not treat pixels, candidates, orders of the same images, or VOC20/VOC21 as independent replications. Such intervals would still not remove historical tuning bias.
3. **For the next controlled experiment:** first add the exact current reset baseline; then prioritize image-C versus episode-C and a candidate-assignment-preserving-multiset permutation control over broad untargeted parameter sweeps. Keep the frozen packet unchanged and publish any new experiment as a separate versioned evidence set.
4. **For stronger generalization claims:** evaluate the already locked method under a genuinely prospective protocol and extend geometry/backbone coverage only with predeclared choices. A new clean evaluation can add evidence; it cannot retroactively make the current full splits unseen.

## Overall Judgment

**6/10, weak accept; confidence 4/5.** I favor acceptance because a simple, inspectable allocation mechanism is matched to a correct explicit objective, and complete same-pipeline comparisons show a useful effect, especially on COCO Object and CD. The global probe selection, retained negative results, execution provenance, and careful interpretation make the narrow contribution credible.

This is not a strong accept: contribution breadth is modest, independent generalization remains unestablished, the exact reset baseline is missing, and normalization/mechanism attribution is incomplete. I would not award credit for statistical significance, matched external superiority, calibrated reliability, native-resolution robustness, or universal safety. Conversely, I find no concrete fatal validity issue, no established inference-time label leakage, and no demonstrated mismatch between the selected C method and the inspected full execution. Confidence is below 5 because this was a bounded source/metadata review without a fresh payload audit or execution, and not an exhaustive literature-priority search. The rating is a judgment of the supported contribution, not an arithmetic average of checklist categories.
