# TRACE Development Findings, November 12

## Scope And Outcome

The actual human requested continued method modifications for higher metrics.
Two sequential, prospectively fixed training rounds completed five new variants
and 2560 predictions on the SAME32 training sources, clean+original15S5.
This panel is repeatedly exposed for development; it is not independent or
the full1449-source validation split. No new full-validation scores were read.

The area-aware semantic-candidate variant has the highest point estimates in
this panel, but its S5 advantage over current area_C is only0.0214pp and its
paired interval spanszero. No new general superiority claim or production
replacement follows. Existing full15 area_C/coverage results remain unchanged.

## All Results

| Arm | Clean mIoU | S5 arithmetic macro | S5 minus MLMP pp | S5 minus area_C pp |
|---|---:|---:|---:|---:|
| Official MLMP | 63.9395 | 57.9451 | 0 | -0.2731 |
| Current area_C | 63.7954 | 58.2182 | +0.2731 | 0 |
| Current coverage_native_budget | 64.2557 | 57.9918 | +0.0467 | -0.2264 |
| aligned_query_entropy | 63.9837 | 58.1418 | +0.1967 | -0.0764 |
| aligned_semantic_entropy | 64.3094 | 58.1772 | +0.2321 | -0.0410 |
| aligned_semantic_anchor | 64.2881 | 58.0702 | +0.1251 | -0.1480 |
| semantic_top8 | 63.7972 | 58.2342 | +0.2891 | +0.0160 |
| semantic_mass_top8 | 64.4506 | 58.2396 | +0.2945 | +0.0214 |

Complete condition tables, all13 direct contrasts and recorded costs:
`[revision-results]/TRACE_development_report_20261112_0959_v1/RESULTS.md`.
Exact values, all gates, undefined counts and verification paths are in adjacent
`RESULT.json`. Every positive/negative/null outcome is retained.

For semantic_mass_top8:

- Clean minus MLMP: +0.5111pp, pointwise95 [-0.0195,4.5622].
- Clean minus area_C: +0.6552pp, pointwise95 [0.0756,0.8287].
- S5 minus MLMP: +0.2945pp, pointwise95 [-0.3776,0.9634].
- S5 minus area_C: +0.0214pp, pointwise95 [-0.4660,0.9725].
- S5 minus semantic_top8: +0.0054pp, pointwise95 [-0.5754,0.7455].

These are10000 whole16B2-group paired draws shared across arms/types within
each round, seeds2026111203 and2026111204 respectively. All13 clean/macro
contrasts have0undefined draws. No multiplicity correction or selection-bias
adjustment; even positive pointwise intervals are not corrected discoveries.
Do not subtract marginal intervals, pool15types' pixels or call32x15 independent.

## Implemented Changes

Unchanged throughout: official MLMP source/checkpoint, L14/last18, seven prompts,
Adam .001,10 full steps, B2, native full-image query entropy+alpha_cls1 CLS,
official final adaptive-layer averaged-prompt readout, RGB/GT geometry and metric.
The official baseline is not edited or relabeled as a new implementation.

Round1 uses per-prompt initial-frozen mean-layer C with max-alias semantic
softmax, then either agreement-gated query entropy, agreement-gated semantic
entropy, or bounded semantic1-p anchor. All retain the native global objective.
Query versus semantic entropy isolates that regional term; alignment versus
area_C changes C measurement and agreement jointly. The anchor also changes
the target and removes agreement gating. None beat area_C's S5 point estimate.
Thus readout/alias alignment did not establish the dominant remaining bottleneck.

Round2 restores the exact executed area_C loss. It merges same-semantic-class
query aliases BEFORE connected components and uses maximum query probability,
without class renormalization, for semantic confidence and counterfactual C.
semantic_top8 ranks confidence; semantic_mass_top8 ranks
`confidence * sqrt(region_area / grid_pixels)`. Both keep at most8 regions/image,
initial-frozen C, the same foreground-preserving intervention and ten updates.
The direct contrast isolates ranking; versus area_C it also changes semantic
support/C aggregation. The exponent is fixed, not a searched optimum.

Across the15 trainingcorruptions, confidence-only versus area-aware ranking:

| Diagnostic | semantic_top8 | semantic_mass_top8 |
|---|---:|---:|
| Mean model-grid coverage | 47.8215% | 67.9647% |
| Single-cell region fraction | 58.8120% | 17.1885% |
| Mean candidates/image | 7.9875 | 7.9875 |
| Recorded mean B2 seconds | 0.408936 | 0.408795 |

The candidate fragmentation/coverage effect is measured, not inferred from
mIoU. It does not prove better semantic correctness: large regions can still
be wrong, and increased coverage did not yield a clear S5 accuracy gain.
C remains heuristic, not calibrated correctness/update benefit or causal effect.

Reference area_C recorded standalone cost is0.425586s/B2 after ADDING its shared
C acquisition. Alignment query/semantic/anchor costs are0.388444/0.404943/
0.407531s. These are different sequential runs, not interleaved timings or
equal-time-budget comparisons. IO/startup/first4 gates/failed attempt are not in
these per-episode numbers. Same-call feature reuse preserves the previously
validated cache math; no cached features survive parameter updates or calls.

## Execution And Verification

Alignment root: `TRACE_alignment_train_20261112_0942_v2`.
GPU2582038/inv126dde1232b9408b861833b861469f47 exited0 at09:47:08CST;
score2583191/inv9581a009ebcc4bb4a75d1a543e8c7199 at09:47:19;
pipeline2582011/inv01f2482525d340f4858b141698443cb4 at09:47:25.
1536new predictions,16child/global seals,21scoreoutputs verified.
Alignment RESULT SHA512c81b5255b492d5803e898e33d655b516993d8a26457d585c9a77e21e1b351.
Prediction global SHA4bfade31711c93265a0d377fff4248f47756fb445db0a2617dd74f07fe372f4f.
First verification CPU2583470/inv410cdb8b212c46249162319be27c60ff exited0.

Candidate root: `TRACE_candidates_train_20261112_0954_v1`.
GPU2584343/inv3895423a33cd4889863f238ca07fe53a exited0 at09:58:02CST;
score2595609/inve1ef10b49fbc49889449caa85e56eac5 at09:58:05;
pipeline2584323/inv3619c950a2da4c369ebfbd513bd5bc7d at09:58:12.
1024new predictions,16child/global seals,21scoreoutputs verified.
Candidate RESULT SHAc180bdec4160dd555ad2631e4cba0bad00d15a4db8c17aa28ae9a9c0bda1f6c5.
Prediction global SHA9bddcbc3ed48a64342587657b9d42532575468a220f07d22ec034213749a1bd1.

Final report CPU2595696/inv437a291b5ddb444da18d4c433ffcb776 exited0 at09:58:31;
all6 report hashes verified. All16condition/macro points recomputed; all6 reused
arms agree with parent results. Every condition's actualfirst4 native loss,
parameter and logit differences0; nativeNoAdapt/MLMP/area_C storedlabel replay0.
Prediction process GTfalse; separate hidden-CUDA CPU scoring followed actual
prediction exit0 and allseals. Startup envelopes capture actualPID/invocation/
cwd/argv/env/cgroup caps. GPU live state was also checked externally; fast CPU
startups were verified through their captured actual envelopes and terminal
identities, not falsely described as externally sampled while alive.

Final report RESULT SHA33e9a39cceb58946f8009eadb8f02faceaae4e1ccfc731fb1a7398e330cf36bc.
Final report COMPLETED SHAf470e394d01a2d9c270a94971240f721a0575838f178284395e5ad8b5291878e.
132focused CPU tests pass. No worker from either round remains running.

Retained failed predecessor `TRACE_alignment_train_20261112_0940_v1`:
mixed scalar/boolean CUDA assignment failed before new predictions or GT.
GPU2573866/pipeline2573627 actualexit1. Original source/bytes/fulljournals
preserved; explicit flattened indexing and7-prompt regression fixed it in a
new freeze/root. No identical-byte retry or hidden deletion of failure.

## Current Decision

Keep all new methods as opt-in development candidates. Do not replace completed
full15 results, claim a stable S5 gain, fit a rank exponent/LR on validation,
or trigger a selected winner's fullval rerun from this small panel. The remaining
empirical gap is confirmation on a larger, previously unused training panel
and multiple predeclared episode orderings before a new validation comparison.
Other November5 research families are not declared complete. Automation remains
PAUSED, monitors OFF; no new GitHub upload, paper snapshot or old-release change.
