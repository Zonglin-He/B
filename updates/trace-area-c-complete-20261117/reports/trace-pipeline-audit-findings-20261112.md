# TRACE Pipeline Audit Findings

## Status And Evidence

This is a component audit, not a new full-validation parameter search.
The seven prespecified controls in `trace-pipeline-audit-20261112.md` are
COMPLETE:3584 new predictions,32 unique training sources,16 original B2 groups,
clean+15S5. Three sealed parent arms reused. Root:
`[revision-results]/TRACE_pipeline_audit_train_20261112_0849_v2`.
GPU2178183/inva05f9f7d930b4f9ebc97bcebb1ec17de exited0 at09:10:28CST;
score2336457/inv4cf8d8a61fb543b4a9afd6ae94258e65 at09:10:44;
pipeline2177943/inv934c1bddb0e243499aed0d3ca3174bf0 at09:10:49.
Allthree actual Code1/Status0/success/MainPID0/exited verified;22 score-output
hashes verified; all16 point estimates and macro independently recomputed
from compact I/U; noFAILED. RESULT SHA
5d7ffda242a9ad1e6d8103d045c2f14378523b6e18ecd0a6eae0393ac5ee82ef.
Allfirst4 native loss/parameter/logit differences0; allthree parent stored
label replays0 changes. First4 diagnostics are not full-population evidence.

Full15 diagnostic report is already complete:
`[revision-results]/TRACE_pipeline_full_audit_20261112_0853_v1/RESULT.json`.
Actual CPU Exec2207575/invf1b4dea3751340b6a29156c3fa172125, Code1Status0success,
MainPID0/exited; four output byte hashes verified. RESULT SHA
daedbfa994e25cae370e0b80c0572e7a35e2fd3e481c5fd48c7c948e55fdc52c.
This reads only prior sealed counts and scalar metadata, not model/RGB/GT.

## Confirmed Cost Problem

On all1449 sources x15 original S5 types, the direct paired contrast between
coverage_native_budget and area_C is +0.0168827329 pp, pointwise95
[-0.0588276402,+0.0944018502],10000 common whole725originalB2-group draws,
seed2026111202,undefined0. This is a new posthoc contrast on historically
exposed validation, not an independent confirmation or a method-selection rule.

Mean recorded standalone diagnostics/update/readout time for B2 episodes is
1.5122077519s versus0.4097224640s, ratio3.6908099620. Startup/input decode are
excluded; timing order is not randomized. Do not call this end-to-end latency
or compare it against a separately measured nativeMLMP runtime as equal compute.

Top8 query regions cover37.7757% of the model grid on average.57.1098% of its
regions are single-grid-cell components. All-semantic coverage covers100%,
but averages59.5066 components per image per probe and66.4734% single-cell
components. The latter includes both step0/5 probes. Component counts are
not object counts, and coverage is not GT recall. Area weighting prevents
equal-C splitting from multiplying total bonus mass; it does not remove
the cost of computing separate counterfactual views for fragments.

## Confirmed Interface Limitation

`native_trace_revision.objective` hardcodes CLS coefficient1 and the adaptation
path assumes loss-prompt integration. It cannot be treated as a generic native
MLMP wrapper for alpha_cls!=1 or text-prompt integration. The completed
experiments explicitly used alpha_cls1/loss mode, so this does not invalidate
their numbers. CPU regression reproduces the nonunit-alpha mismatch.

The separate `native_pipeline_audit` implementation honors native alpha_cls
and rejects unsupported prompt integration. Original frozen source is kept
unchanged. No claim of a score improvement from this interface correction.

## Completed Single-Factor Results

All values are training-panel mIoU percent, not new full-validation results.

| Arm | Clean | S5 arithmetic macro |
|---|---:|---:|
| Native MLMP |63.9395|57.9451|
| area_C |63.7954|58.2182|
| coverage_native_budget |64.2557|57.9918|
| area_uniform_covered |64.0002|58.1538|
| area_uniform_image |63.7462|57.9235|
| area_shuffled |63.4981|58.0449|
| coverage_static |64.3239|57.7924|
| coverage_uniform_image |63.8295|57.8983|
| native_no_cls |63.5021|56.2217|
| area_no_cls |63.2786|56.6470|

The following contrasts are control minus its named intact parent.

| Control and parent | S5 delta pp | Pointwise95 paired interval |
|---|---:|---|
| Uniform covered minus area_C |-.0645|[-.3091,.7861]|
| Uniform full image minus area_C |-.2948|[-.4949,.7557]|
| Spatial shuffle minus area_C |-.1733|[-.5371,.5755]|
| Static all-regions minus refreshed all-regions |-.1994|[-.4471,.2997]|
| Uniform full image minus coverage_native_budget |-.0935|[-.7011,.4102]|
| Native-noCLS minus MLMP |-1.7234|[-3.5490,-.0519]|
| Area-noCLS minus area_C |-1.5712|[-3.2460,.2850]|

Shared10000 original whole-B2-group draws,seed2026111201; undefined0.
Pointwise intervals without multiplicity correction. The marginal native-noCLS
interval excludes0 but is not a corrected ten-contrast discovery or a proof
that CLS explains the entire historical TRACE gap. Its negative result supports
preserving native CLS in this setting. Area-noCLS also declines at its point
estimate, but its interval includes0.

All mass-matched spatial nulls have intervals crossing0 versus area_C.
Spatially aligned counterfactual C has NOT been shown better than these nulls
on this small panel. This is not proof of no effect/equivalence: intervals
remain wide. Uniform-covered preserves candidate support; uniform-image and
shuffle do not. All preserve per-image bonus mass, controlling the changed
average pixel-vs-CLS loss balance. Refresh also has uncertain benefit.
Do not select a replacement from clean or individual-corruption winners.
All ten prespecified contrasts, including area-noCLS minus native-noCLS and
both original parent comparisons, are retained in score/RESULT.json.

## Remaining Mechanism Problems

1. Added C changes spatial weighting AND average pixel-vs-CLS balance. Uniform
   covered/full-image and shuffled controls keep per-image total C mass fixed.
   A gain over nativeMLMP alone cannot attribute the gain to counterfactual C.
2. The old restoration experiment changed global pixels and CLS together.
   Native-noCLS and area-noCLS now isolate this term; no alpha sweep is performed.
3. All-coverage, semantic alias merging and step5 refresh were combined in the
   full alternative. Static-all coverage isolates the refresh contribution.
4. Reliability uses averaged text and adaptive layer weights; loss uses seven
   prompts and mean-layer features. C is class-specific at measurement but
   its entropy bonus has no class target. A different prompt/current argmax
   can therefore be sharpened. Disagreement is measured; its harm is not proved.
5. Thirty query strings map to20 semantic classes; person has7 queries and
   monitor has5. Query entropy includes within-class alias uncertainty, and
   raw per-query C is not comparable to a calibrated semantic probability.
   Shannon decomposition sums aliases only for diagnosis; the production
   max-over-alias final readout is unchanged. Altering it would be a new method.
6. Candidates and C operate on16x16 patch grid; nearest masks preserve whole
   14x14 pixel blocks, while final prediction bilinearly interpolates logits to
   224x224. Grid coverage is not accurate object-boundary coverage. Measured
   nearest-coarse/final disagreements are not GT errors.

## First4 Training Diagnostics

Four unique training sources repeat across conditions, not64 independent images.
S5 summaries average two original B2 groups across15 conditions. No GT errors
are measured, and the30 condition-episode observations are not independent.

| Diagnostic | Step0 | Step5 | Step9 |
|---|---:|---:|---:|
| Prompt/native semantic disagreement |48.34%|40.79%|35.12%|
| C-weighted prompt/native disagreement |47.87%|34.36%|26.36%|
| Within-alias share of query entropy |15.57%|15.08%|14.80%|
| CLS/pixel gradient norm ratio, mean per episode |1.505|.985|.666|
| Bonus/pixel gradient norm ratio, mean per episode |.545|.540|.488|
| Fraction of pixel/CLS gradient cosines below0 |40.0%|53.3%|53.3%|

C's adaptive-readout class can differ from the class whose entropy each prompt
minimizes. The extra loss has no fixed class target, so larger C alone does not
enforce class-consistent adaptation. This is a concrete design mismatch, not
proof that fixing it improves accuracy. Falling disagreement without GT is not
evidence of improving correctness. Some CLS/pixel gradients conflict; removing
CLS nevertheless harms the native point estimate. Gradient cosine alone does
not justify removing a loss.

Within-alias entropy means part of the pressure concerns query wording
competition rather than semantic-class uncertainty. Sum-alias entropy here is
diagnostic only; production output remains max-over-alias. Nearest-upsampled
16x16 maps and final224 bilinear-logit labels disagree on18.25% of S5 pixels
before adaptation and13.02% after in this panel. This is coarse/interpolated
geometry, not a proven scoring bug or inaccurate GT boundary fraction.

## Diagnostic Encoding Redundancy

Official adaptive readout computes all-layer features for entropy weights, then
runs the visual encoder again to form weighted features. The second pass can
potentially reuse selected raw features from the first, BEFORE ln_post. Averaging
post-LayerNorm embeddings is not equivalent. This is independent of the loss.

An opt-in `native_cached_probe` module and finite check are COMPLETE, without
changing official/old frozen code. First4trainingRGB x16conditions, B1/B2 and
actual B4 counterfactual views, before/after ten real updates, top8/all-C probes
and reference-state restoration were checked. All192 raw-logit comparisons,
128 C-map comparisons and32 paired ten-step episodes have EXACT0 differences
in checked logits/C/losses/updated parameters/final query and semantic labels.
Every diagnostic uses one rather than two visual forwards. No persistent cache,
optimizer-state mutation, leaked hooks, or weights_track growth was observed.

Root `TRACE_cached_probe_check_20261112_0912_v1`, actualGPU2344657/
inv648f58cb0c8b424e96fba853914497c3,exit0 at09:13:53CST. RESULT SHA
3c8ade2b83d015ccbee0f6990adb9e569b502f3cc6356d8b6da845d72b41475b.

| Diagnostic batch | Mean original ms | Mean cached ms | Ratio | Time reduction |
|---|---:|---:|---:|---:|
| B2 |19.3120|10.7905|1.7897x|44.13%|
| B4 actual counterfactual views |19.2166|10.5883|1.8149x|44.90%|

128 calls perpath/batch, one warmup and four repetitions perepisode, alternating
AB/BA/AB/BA, synchronized CUDA. This is readout-call timing, not total pipeline
latency or an accuracy improvement. Only the tested pinned settings/panel are
covered, not all architectures/shapes. No completed scientific run was changed
or repeated to deploy this optimization.

The original checker's COMPLETED seal preceded the final mutable queue update:
ONLY queue_state.json changed after sealing. All32 other files, including the
result and archived executed source, match the original seal. Do not call that
original seal fully valid. The append-only exact reconciliation root
`TRACE_cached_probe_reconcile_20261112_0917_v1` preserves original bytes and
verifies all33 final artifacts against actualterminal; CPU2384413/
inv8874865a898442ae85bab6571f22f5ce exited0 at09:16:50CST. Its four outputs
are hash-verified; FINAL_OUTPUTS_SEALED SHA
523e1c99856ef224d867acff1398feead59948e21ea3e67900bd7dd6faa519d5.
Future checker finalization now writes finalstatus before sealing, with a
regression test. No model rerun, prediction alteration or accuracy change.

## Excluded False Alarms And Checked Paths

- Cached RGB hashes and official source/checkpoint/settings are bound. The
  official corruption transform acts on loaded pixels before resize. The
  preserved cache follows that path; no new RGB/BGR conversion is introduced.
- CLIP normalization constants match the input loader; there is no second
  resize of cached224 RGB. Foreground candidate pixels are explicitly checked
  identical under each counterfactual view, in model input coordinates.
- Full model and Adam reset precede episodes; reference-probe swaps restore
  adapted optimized parameters, including exceptional paths. New methods
  differentiate only optimizer members. Actual first4 parity checks test loss,
  parameters, logits and stored parent labels, not scalar mIoU alone.
- One-crop aggregation is the identity. Max-alias collapse and final argmax
  retain the same class ordering. Scoring uses indexed GT, CV2nearest224,
  background/void255 ignore and paired original B2 counts. Confusion-matrix
  cross-checks match vectorized I/U; empty total union remains undefined.
- The readout diagnostic compares float32 interpolation of promoted low-grid
  logits with nativefp16 interpolation. Small nonzero logit differences here
  are not a demonstrated model mismatch; the two numeric operations differ.
  A CPU regression reproduces this distinction. Stored parent label replay is
  the actual parity check. Do not label coarse/final disagreement a scorer bug.
- The seven-control first attempt failed only in NEW entropy-diagnostic JSON:
  fp16 cannot represent epsilon1e-30, causing0log0. Changedv2 casts this
  diagnostic tofloat32, with fp16/bfloat16/float32 tests. Failed0846-v1 source,
  FAILED files and journals are preserved. This is not an explanation for
  historical TRACE performance, OOM, or a scientific outcome.

## Unresolved Boundaries

Only32 previously used training sources/16 B2 groups support new
controls; first4 sources alone support gradient/readout diagnostics. Original
full evaluation has one ordering. No new independent episode-order stability,
large development panel, clean full-split/generalization or equal-time budget
test is completed by this audit. Counterfactual C remains an uncalibrated
response to one soft transformation, not measured causal context reliance or
probability of beneficial adaptation. These limits are not repaired by more
bootstrap draws or lower loss. Positive, negative, null and failed results stay.

## Disposition

123 focused tests pass. Original automation remains PAUSED, monitors OFF and
infrastructure agents STOPPED. No worker from this audit remains running;
no upload, new paper snapshot, original baseline edit or old result overwrite.

Keep the native global+CLS objective. Do not infer C's mechanism from the full
gain, or equate 100% coarse-grid coverage with improved accuracy. The next
mechanistic tests should align the measured C class/readout with the optimized
target, separate semantic uncertainty from alias competition, and test fixed
loss-mass controls on a larger training development panel and independent
episode orderings. Do not tune these decisions on the already-exposed full15
validation outcomes. Learning rate, step count and bonus-coefficient optima
remain untested; this audit is not a parameter sweep or final method selection.
