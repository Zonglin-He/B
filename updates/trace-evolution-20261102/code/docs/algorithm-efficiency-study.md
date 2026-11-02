# Algorithmic efficiency: target-first source tracing

This study supersedes the unstarted parameter-only accuracy/cost search.
The user explicitly requested changes to the algorithm, not efficiency from
tuning K, batch size, learning rate or update steps. The existing VOC21 batch
search and full validation continue unchanged. The new study is research-only;
neither production TRACE nor the manuscript is replaced automatically.

## Algorithm and rationale

Production TRACE measures three supports for every selected candidate:
T from a target-preserved view, D from a target-suppressed view, and C from a
context-perturbed view. It uses q=min(T,C) and rho=q*(1-max(D-T,0)). Hence
0 <= rho <= T. All loss weights are detached.

Target-first selective tracing uses this bound to allocate diagnostic work:

1. Keep the same connected candidates and original top-K policy.
2. Measure T for every candidate. Do not use GT or unmeasured D/C to rank work.
3. Sort by decreasing measured T, with original candidate index breaking ties.
4. Measure D and C in bounded rounds, normally eight candidates/sixteen views.
5. Let A be the sum of observed rho values retained by the existing EPS filter;
   let U be the sum of T over unmeasured candidates with T>EPS.
6. Stop when U/(A+U) <= epsilon. Set unmeasured weights to zero; record their
   missing D/C/quality/dominance as null, not invented zeros.

The two fixed research variants use epsilon=0.01 and 0.05 uniformly across
datasets. This is an additional algorithmic approximation tolerance, not a
claim of a hyperparameter-free exact implementation. When A=0 and U>0, tracing
must continue. If every T<=EPS, production weights would all be pruned anyway.
The bound applies to the whole adaptation episode, matching the loss normalizer.

If unmeasured true retained mass is M, then M<=U. Relative to full tracing with
the same numerical support values, the normalized coefficient total variation
is exactly M/(A+M), bounded by U/(A+U). At identical logits, entropy is bounded
between zero and log(C), so the spatial loss difference is at most
epsilon*log(C). This does NOT bound gradient error, Adam trajectory error,
class IoU, mIoU, or error from floating-point view rebatching. Candidate overlap
does not invalidate the coefficient bound, but it complicates spatial meaning.

The mechanism is adaptive computational allocation: candidates with little
possible remaining influence need not receive two more model probes. Unlike
reducing K based on original confidence, all original candidates receive a
target-support measurement. The algorithm can skip low-mass class supports;
small global mass is not a guarantee of unimportant rare-class evidence.

## Controls

- Standard: original production TRACE, interleaved three-view execution.
- Staged full: all T first, then all D/C in the same ranking/round structure,
  without skipping. Separates batching/numerical effects from selective probes.
- Quality only: measure T/C and set rho=min(T,C), removing residual attenuation.
  It saves one-third of diagnostic views but changes the scoring rule and is
  not a certificate-preserving implementation of the same method.
- Selective 1% and 5%: the algorithm above, with the full scoring rule on every
  actually traced candidate and bounded omission elsewhere.

No variant changes K, image batch, lr, steps, prompts, layers, preprocessing,
optimizer, LayerNorm adaptation, or entropy normalization. The pre-existing
VOC21 parameter-study winner is NOT substituted into any of these cells.

## Cached evidence, not new adaptation results

The CPU audit uses the previously sealed 512-image VOC21/PC60/COCO Object
panels and complete 500-image COCO-CD split. Counts below include all episodes.
Supports are replayed without new forwards; no labels guide probe selection.

| Dataset | Diagnostic views saved, 1% | Diagnostic views saved, 5% | All forward inputs saved, 1% |
| --- | ---: | ---: | ---: |
| VOC21 | 26.74% | 37.99% | 19.23% |
| PC60 | 26.94% | 38.22% | 22.49% |
| COCO Object | 34.50% | 42.99% | 26.41% |
| COCO-CD | 4.47% | 11.95% | 2.82% |

The maximum measured normalized coefficient TV for the 1% policy is below
0.81% on these caches. Per-image class support is not preserved: the fractions
of previously active image/class pairs losing all update weight are about
13.74%, 24.27%, 26.39%, and 1.33%, respectively. These are selected predicted
class supports, NOT loss of GT object coverage or measured segmentation errors.
Their downstream effect must be measured. The two-view q-only control does
not lose class support in this cache, but shifts the relative coefficients.

## GPU protocol and decision boundary

The frozen queue first smoke-tests all five variants on four VOC21 images,
then evaluates all five on fixed 128-image panels for VOC21, PC60, COCO Object,
and COCO-CD. Variants use identical dataset configurations and image IDs.
Algorithm order rotates by dataset to reduce simple fixed-order timing bias.
Results retain both positive and negative deltas, classwise/per-image counts,
actual visual input counts and call counts, synchronized latency, and memory.
Stages/rounds can reduce hardware utilization; input reductions are not speedups.

This is a single-seed development comparison, not full-dataset or three-seed
evidence. No automatic adoption or validation-based parameter search is allowed.
A replacement is judged primarily against the strongest matched external
baseline, not against old TRACE. Report mIoU and measured cost for No Adapt,
TENT, TPT and MLMP where runnable under the audited protocol. Old TRACE and
staged-full cells remain internal mechanism/numerical controls, not an adoption
threshold. A decrease relative to old TRACE does not by itself disqualify a
variant; an accuracy gain or speedup relative to it does not establish an
advantage over baselines. Published full-split scores are references only and
cannot be subtracted from these 128-image results. Positive pilot results still
require full validation and repeated timing/seed evidence. No variant is
automatically promoted. Do not weaken ablation controls or amplify a score just
to enlarge removal deltas; simplify unsupported components instead.

Outputs: /home/wwww/hzl/results/trace_algorithm_efficiency_20261025/
The old parameter queue remains archived with automatic_resume=false.
