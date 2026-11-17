# TRACE Alignment Development

Actual November12 human authorizes continued method modification for empirical
improvement. This prospective training-only round follows progress305 findings;
it does not alter old releases, official MLMP, or completed experiments.

## Fixed Design

Retain native full-image query entropy plus alpha_cls=1 CLS entropy, Adam .001,
ten full steps, L14 last18 layers, B2 original episodes, seven prompts, final
official adaptive weighted readout and max-alias classification. No LR/step
grid, no native-step gating, no new corruption or benchmark.

Keep initial confidence top8 query-component candidates and the original
foreground-preserving counterfactual transformation. For each region r with
fixed semantic class y_r, compute C separately for each training prompt t with
the initial frozen model and the SAME mean-layer forward used by optimization.
Semantic logits are max over aliases followed by class softmax. This preserves
the production class argmax without introducing a sum-alias count prior.
C_tr is the mean counterfactual semantic probability of y_r inside r. It is a
heuristic, not calibrated correctness or causal context dependence. Masks and
C are fixed for the episode; no GT or outcome-dependent threshold.

Three fixed variants add coefficient1 regional bonuses averaged over all
prompts/images/grid pixels, with C=0 outside candidate support:

1. aligned_query_entropy: C_tr times native query entropy, only where the
   current detached per-prompt semantic argmax equals fixed y_r.
2. aligned_semantic_entropy: same agreement gate and C, but max-alias semantic
   entropy, avoiding extra within-class alias competition.
3. aligned_semantic_anchor: C_tr times (1-p_semantic(y_r)), without current
   agreement gating. The anchor is bounded rather than unbounded cross-entropy;
   its wrong-teacher risk remains. Full native loss remains in every step.

The first variant changes C measurement plus agreement; it is not a pure
single-factor attribution to either. Variant2 vs1 isolates the regional entropy
support; variant3 also changes target/guard, and is reported as a distinct method.
No claim of exact alignment with final adaptive layer readout, since the native
training/readout distinction remains. Initial candidates use that native readout.

## Evaluation

Use all same32 outcome-blind-selected original training sources, clean and
original15 S5,16 original B2 groups:1536 new predictions for three variants.
Reuse sealed MLMP,area_C andcoverage_native_budget parent counts. The panel
has now been used for development and is NOT independent validation.
First4 each condition: native zero-update and actual-ten-step label replay;
native-equivalent loss/parameter/logit <=1e-5/zero labels; cached candidate
readout <=1e-5/zero query labels; mean-layer probe pixel logits match training
pixel logits with CLS path on. All finite losses/gradients/parameters required.

Separate hidden-CUDA CPU scoring only after all16 prediction seals and actual
same-PID/invocation success. Same pinned indexed GT/CV2nearest224/background
and void255 ignored. Shared10000 whole original16B2 bootstrap draws,
seed2026111203, pointwise95 intervals, no multiplicity adjustment, retain
undefined, all three outcomes and all15 types. Compare each new variant with
MLMP and area_C; also compare semantic-entropy with query-entropy and anchor
with semantic-entropy. No full-validation dispatch or winner selection is
automatic. A training gain alone is not a claimed full-benchmark improvement.

Single GPU lock, GPU12GiB RAM/1GiB swap, CPU4GiB/512MiB swap, one thread,
OOMstop/Restartno, exact frozen source and actual STARTED envelopes. Failure
bytes remain; no unmodified retries. Monitors/automation remain off/paused.
New image-source preparation is unnecessary because exact sealed inputs exist.

## Retained Initial Failure

0940-v1 failed before any new variant prediction or GT read: CUDA mixed scalar
and boolean assignment to the per-prompt C map raised an internal index-size
assertion. Clean first4 native/cached/loss/parameter/logit/label checks all0.
GPU2573866/inv0dbf4aad0154438cb5e741bcd3dddd21 and pipeline2573627/
inve799ab4a251e4a0e9885e2be59b27689 exited1; original source/bytes preserved.
Repair replaces mixed boolean indexing with explicit flattened index_select,
index_copy_ and index_fill_, adds exact prompt-shape validation and a seven-
prompt fixture. No scientific objective/parameter changes. A new root/freeze
retains full predecessor journals and terminal evidence; no unchanged retry.
