# Baseline-first evaluation and weak-ablation diagnosis

## Decision contract

Primary acceptance is accuracy against the strongest matched external baseline
and its measured cost, not preservation of the previous TRACE score. Old TRACE
is an internal removal/staging control only. A faster approximation need not
beat old TRACE, but cannot claim a baseline advantage without matched results.
Do not enlarge ablation gaps by weakening controls, tuning harm definitions,
or amplifying attenuation after seeing test labels.

## Evidence as of 2026-10-25

The four removal panels are fixed 512-image VOC21, PC60 and COCO Object panels,
and the full 500-image COCO-CD split. The entries below are Full minus each
removal, in mIoU percentage points. They are not baseline-comparison deltas.

| Removed design | VOC21 | PC60 | COCO Object | COCO-CD |
| --- | ---: | ---: | ---: | ---: |
| Connected-region construction | +0.49 | +0.15 | -0.07 | +0.97 |
| Confidence-source tracing; use original confidence | +0.47 | +0.60 | +1.59 | +7.79 |
| Candidate reliability; use uniform weights | +0.46 | +0.59 | +3.06 | +8.75 |
| Region normalization; use weighted token mean | +1.46 | +0.77 | -0.55 | +2.50 |

Source: `/home/wwww/hzl/results/trace_removal_suite_20261025_0730/final_audit/ablation_summary.csv`.
These results do not support saying that all ablations are weak. Traced weights
matter especially for COCO-CD. Connectivity has a weaker, dataset-dependent
effect, and region normalization is not uniformly beneficial.

## Why residual dominance has little incremental effect

Let T, C and D be target, perturbed-context and residual support, respectively.
The actual code uses q=min(T,C), d=max(D-T,0), and rho=q*(1-d). A candidate with
T=0.02, C=0.03, D=0.04 gets q=0.02 and rho=0.0196: even a residual response
twice the target response only attenuates the already-small q by 2%. A strong
target-supported candidate often has D<=T and receives no attenuation at all.
This is a property of the probability-difference rule, not evidence that a
larger multiplier or a ratio would improve segmentation.

For an episode before EPS pruning, define p_i=q_i/sum(q) and mu=sum(p_i*d_i).
Then the normalized final coefficient is p'_i=p_i*(1-d_i)/(1-mu), and

`TV(p,p') = sum_i p_i*abs(d_i-mu) / (2*(1-mu))`.

A common attenuation cancels exactly. Only differences in attenuation between
candidates change allocation. EPS pruning is handled separately in the audit;
it changes activity for 8/6/20/0 candidates in the four panels, respectively.
Coefficient TV does not establish gradient, Adam-trajectory or mIoU equivalence.

| Dataset | Candidates with D>T | Their share of q mass | Mean normalized coefficient TV |
| --- | ---: | ---: | ---: |
| VOC21 | 44.98% | 9.60% | 1.44% |
| PC60 | 47.13% | 10.86% | 1.72% |
| COCO Object | 49.55% | 7.63% | 1.30% |
| COCO-CD | 23.17% | 14.39% | 2.44% |

Mass shares and TV are unweighted episode means, matching the normalization
unit. Thus frequent gate activation does not imply substantial loss influence.
The retained candidates already have little q mass, and normalization further
limits the allocation change. The audit uses all selected candidates, without
GT-dependent selection or threshold fitting.

Fresh outputs: `/home/wwww/hzl/results/trace_ablation_mechanisms_20261025/`.
The reproducible CPU audit is `tools/analyze_ablation_mechanisms.py`.

## What changes, and what does not

1. Prioritize the already implemented two-view shared-quality variant for
   baseline validation: retain candidate construction, target-preserved and
   context-perturbed probes, q=min(T,C), and normalized entropy; do not compute
   D solely for a small multiplier. This is a research candidate, not an
   automatic replacement of the production method or manuscript.
2. Retain the existing selective three-view variant as an alternate efficiency
   candidate. Its omitted-weight bound does not preserve rare-class support;
   the cache already shows loss of low-mass predicted class coverage.
3. Do not add a dominance temperature, floor, hard reject or role-specific
   correction simply to make the deletion penalty larger.
4. Keep main removals aligned with Method: connected regions, tracing replaced
   by original confidence, and reliability replaced by uniform weights. Treat
   region normalization as an allocation-design check. The tracing computation
   and application of its output are coupled stages, not independent additive
   contributions; nearly equal removal effects are not automatically a bug.
5. If larger checks keep finding connectivity neutral, retain connected
   components as a transparent candidate definition, not a separate claimed
   innovation. The class-union control changes mask extent/count as well as
   connectivity and cannot isolate instance separation alone.
6. If two-view tracing is adopted, the narrative must change from measuring
   residual-context necessity to candidate support under altered context.
   It may still test prediction reliability, but cannot claim to observe a
   target-suppressed response that is no longer computed.

## Baseline comparison boundary

The newly finished algorithm pilot has 128 images per dataset. Historical
baseline rows use different episodes, and the historical COCO Object query
list contains a duplicate background/bed alias. Neither published full scores
nor subsets of old adapted predictions are matched controls for this pilot.
Use the frozen native dataset/vocabulary and exact ordered image IDs, while
loading preserved baseline implementations. Record method-specific optimizer
settings and the baseline NaCLIP versus TRACE DenseCLIP backend distinction.
Compare No Adapt, TENT, TPT and MLMP where executable; WATT and CLIPArTT remain
excluded as requested. Missing/failed rows stay missing, not zero.

The 128-image shared-quality result changes mIoU by -0.027/-0.112/+0.134/+0.117
relative to the internal three-view control and removes exactly one-third of
diagnostic views. This motivates further validation but proves neither a
baseline advantage nor universal safety. No test-result-driven new formula or
dataset-specific switch is introduced by this diagnosis.
