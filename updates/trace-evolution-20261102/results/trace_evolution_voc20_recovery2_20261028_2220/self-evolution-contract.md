# TRACE self-evolution contract

## Objective and protected inputs

Develop and validate a new method, revise the supplied original paper and
appendix, and obtain an independent, evidence-grounded review of at least
6/10 (weak accept). The requested 7/10 is an acceptance aspiration, not an
instruction to manufacture a score. Original Desktop TeX/PDFs, main checkout,
historical TRACE evidence and negative results are protected.

Branch: `codex/trace-self-evolution`, base
`d21bcafeac015b810cc388e17b0bac428a7eb380`.

## Frozen first method

`trace_ovss.method.two_view.TwoViewTRACE` is the explicit new entry point.
Candidates are connected predicted token-grid components selected by initial
confidence, with the inherited deterministic top-K ordering. Every candidate
receives preserved and context-perturbed views. The fixed initial model
measures T and C on the original candidate positions; rho=min(T,C). All
weights are detached. Visual LayerNorm affine parameters adapt episodically
using region-normalized spatial entropy; prediction uses the original input.

No semantic roles, suppressed view, dominance, CLS term, fragment guard,
KL temperature, sibling suppression or GT-dependent candidate selection.
Keep the old three-view entry point solely as a historical implementation
until a complete tested migration, not as the definition of the new method.

Two-view quality alone is not identified causal provenance, a calibrated
correctness probability, or proof of context invariance. In a single-candidate
normalized loss its positive scalar cancels. Claims concern relative joint
allocation, tested with fixed candidates and actual updates.

## First evidence stage

Four representative fast official full splits: VOC21 (1449), PC60 (5105),
COCO Object (5000), COCO-CD (500). The inherited dataset settings are frozen,
not retuned from these new comparisons. Compare trace, uniform, confidence,
T only, C only, arithmetic mean, and candidate-weight permutation with fixed
permutation seeds 0/1/2 at model seed 0. The first repetition stage repeats
TRACE itself at model seeds 1/2; it is not a three-seed result for the controls.
Each episode shares candidates and intervention responses, but
every policy resets the encoder and optimizer. Permutation preserves the
entire weight multiset. Report exchangeable episode coverage.

Inputs, vocabulary, episode order, source hashes and configurations are frozen
before prediction. No label pixels are opened during prediction/adaptation;
seal predictions before separate streaming offline scoring. Preserve masks,
scores and pre/post predictions for mechanism/qualitative analysis.

Historical full seed0 native baselines are valid operating-point comparisons,
not equal-tuning or equal-optimization-budget evidence. The user's latest
instruction is to cite MLMP's reported values on the datasets it reports,
without rerunning those baselines. Label published and local values and keep
backbone, classes, splits and configuration differences explicit. Existing
native results remain available; datasets without published values use local
reproduction. Same-candidate controls isolate assignment within TRACE's own
backend. Seeds 1/2 test the final TRACE configuration, not baseline retuning.
Re-evaluate COCO-Stuff before asserting distributed-class generalization.

User addition, 2026-10-28: append VOC20 without interrupting either frozen
queue. Its four cells are a four-image TRACE-only smoke gate, complete seed0,
and complete seed1/2 TRACE replays on all 1,449 evaluation images. Use the
existing VOC20 configuration without search. Both the 18-cell main suite and
the two-cell cached-control suite must finish successfully and seal before
VOC20 acquires the GPU. Only after all three suites complete and verify should
the manuscript be updated and sent for the next independent manuscript review.

Optional lazy probing and frozen-prefix caching remain separate variants.
No combining their gains without evaluating that actual combined algorithm.
Cache equivalence is checked at the same freeze scope; changing trainable
layers is an algorithm change. Diagnostic inputs, encoder calls, wall time,
optimizer steps, memory and accuracy have distinct meanings.

## Review loop

1. Freeze method, manifest and proposed claims.
2. Run tests and the bounded experiment stage, retaining all outcomes.
3. Write the paper and appendix from completed canonical evidence only.
4. Render and inspect PDFs; audit symbols, references and claim support.
5. A fresh independent reviewer reads the actual draft and artifacts under
   the fixed rubric below, without being told that a desired score is needed.
6. Log each weakness, assess its validity, and either repair the method,
   add the targeted control, narrow the claim, or document a reasoned rejection.
7. Revise and review again. A score alone never overrides unresolved validity.

Rubric: contribution, correctness, experimental support, evaluation fairness,
and reproducibility/clarity. Scores: 4 reject, 5 borderline, 6 weak accept,
7 accept, 8 strong accept. Stop only with a genuine score >=6 and no open
fatal method-identity, label-leakage, metric/protocol or unsupported-claim issue.
All review rounds, including unfavorable ones, remain archived. No reviewer
sampling until favorable, concealed test-based searches, or fabricated results.

Long jobs run sequentially under a shared GPU lock and bounded systemd memory.
No silent retries or source mutation in a frozen runtime. A failed stage
preserves its evidence, records the cause and blocks dependent claims.
