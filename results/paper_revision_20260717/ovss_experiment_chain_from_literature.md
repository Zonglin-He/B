# TRACE Experiment Chain After OVSS Literature Review

Date: 2026-07-17

## Paper Story To Experimental Question

TRACE makes two deliberately limited claims: low-entropy OVSS candidates do not all deserve equal update authority, and target/context interventions provide useful information for allocating that authority. The experiment section should therefore progress from end-to-end effectiveness to mechanism, bottleneck localization, and practical boundaries. It should not read as an unordered benchmark collection.

## Recommended Main-Paper Order

| Order | Experiment | Question | Status |
|---:|---|---|---|
| 1 | Seven standard clean sets + two shortcut-oriented sets | Does TRACE improve full-validation OVSS-TTA performance? | Complete |
| 2 | Three seeds and paired bootstrap | Are the gains stable across adaptation seeds and test-image composition? | Complete |
| 3 | Context utility versus update reliability | Can context predict correctly while context-dominated updates remain risky? | Complete |
| 4 | Confidence-controlled source-risk analysis | Does the diagnostic add information beyond raw confidence/entropy in its motivating regime? | Complete; scope to four-dataset diagnostic |
| 5 | Nine-dataset thing/stuff/background diagnosis | Does the relation hold uniformly across semantic roles? | Newly completed |
| 6 | Candidate-oracle 2x2 decomposition | Is candidate geometry or update weighting the principal bottleneck? | Newly completed |
| 7 | Method-aligned leave-one-out ablation + small-object regression | Which named design choices cause the end-to-end gain, and does the fragment guard suppress real small objects? | Complete |
| 8 | Runtime/memory, perturbation robustness, same-class multi-object, second backbone | What are the practical cost and external-validity boundaries? | Complete or preexisting; place compact summary in main text and details in appendix |
| 9 | Success and failure taxonomy | What does the method fix, and where does it still fail? | Partially complete |

## What The Supplied Papers Contribute To This Design

- **MLMP:** motivates a matched OVSS-TTA protocol, component ablation, prompt-count/integration controls, multiple backbones, and explicit complexity reporting. TRACE should retain MLMP as the principal matched baseline.
- **Mask-Adapted CLIP and related mask-grounding work:** separate proposal quality from region-language classification. TRACE's candidate-oracle 2x2 is the test-time analogue, but must not call GT candidates a performance upper bound.
- **PACL:** validates the claimed local semantic property with a proxy/AUROC test before relying on downstream mIoU. TRACE follows this pattern with controlled harmful-update ranking, while the nine-dataset role analysis establishes where that proxy does not generalize.
- **CLIPSelf, NERVE, and object-context-aware OVSS work:** evaluate spatial units, thing/stuff behavior, and local semantic coherence separately. This motivates the role-specific diagnosis rather than one pooled candidate statistic.
- **FreeCP:** demonstrates that overcomplete and redundant vocabularies are an OVSS-specific source of activation confusion. This motivates a vocabulary stress test for TRACE rather than another generic corruption benchmark.
- **VocAlign:** reports best/worst classes and varies class selection, showing that vocabulary composition can alter adaptation. TRACE should report mIoU retention and harmful-update behavior as distractor labels are added.
- **Seg-TTO:** links broad domain evaluation with framework, prompt, and optimization ablations. TRACE should keep prompt robustness as a fairness check, not as a new method component.
- **SemLA:** uses multi-domain and cross-backbone evaluations while exposing the resource assumptions of adaptation. TRACE's second-backbone result should be presented as compatibility evidence, not universal superiority.
- **Self-supervised augmentation consistency:** pairs component ablation with explicit transformation consistency. TRACE's five counterfactual perturbations play the same methodological role and belong in the appendix.

## Highest-Value Missing Experiment

### Vocabulary-Composition Stress Test

This is the clearest remaining OVSS-specific test. It directly probes whether confidence-source weighting remains useful when the text label set changes the local class competition.

Run VOC20, PascalContext60, and COCO Object under four fixed vocabularies:

1. benchmark classes only;
2. benchmark classes plus synonymous/class-extension prompts;
3. benchmark classes plus semantically related or co-occurring distractors;
4. benchmark classes plus unrelated distractors, matched in count to condition 3.

Compare No Adapt, matched MLMP, and TRACE with the same prompt bank and optimizer settings within each condition. Report:

- full-validation mIoU and mIoU retention relative to benchmark-only vocabulary;
- false-positive area assigned to added labels;
- correct-to-harm rate among initially correct candidates;
- mean target/context source contrast and reliability shift;
- per-class best/worst changes for classes with synonyms or common co-occurrences.

The primary claim should be robustness to vocabulary-induced competition, not that added labels always hurt. Conditions 3 and 4 distinguish semantic competition from a pure class-count effect.

## Appendix-Only Additions

1. **Prompt-bank robustness:** one canonical template, the current shared bank, and a larger CLIP-style bank; compare TRACE-minus-MLMP under the same bank.
2. **Candidate-budget/cost Pareto:** candidate budgets 2/4/8 versus mIoU, time, forward calls, and peak memory. Keep bounded profiling numbers separate from full-validation results.
3. **Per-class headroom correlation:** correlate candidate-oracle recall improvement with practical TRACE gain. Treat this as exploratory because four datasets provide too few aggregate points; a class-level analysis is preferable.
4. **Failure taxonomy:** missed candidate, wrong text alignment, context-defined label, unstable intervention, and over-fragmentation, each with exact candidate scores and pre/post target IoU.

## Experiments Not Needed For The Current Story

- Completing all synthetic corruptions after that suite was intentionally stopped.
- A much larger domain-adaptation benchmark requiring external training data or adapter libraries.
- More unnamed component toggles that do not correspond to the Method section.
- Forcing one universal hyperparameter configuration when the paper does not claim it.

## Claim Boundary After The New Experiments

The candidate oracle substantially improves geometric purity and component recall, but does not uniformly improve entropy-TTA mIoU. Candidate construction is therefore not the sole bottleneck. The nine-dataset role analysis also shows that inverse TRACE reliability is a strong harmful-update ranking signal for background candidates, but not universally for thing and stuff candidates. The paper should consequently describe TRACE reliability as a role-conditioned update-allocation policy, and scope the stronger harm-ranking claim to the original confidence-controlled diagnostic.
