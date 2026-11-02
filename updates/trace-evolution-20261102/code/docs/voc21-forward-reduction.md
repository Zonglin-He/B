# VOC21: algorithmic forward reduction at fixed K and optimizer steps

## Objective and frozen reference

The October 25 request prioritizes VOC21 latency while preserving performance
above the strongest local baseline. K, optimizer steps, learning rate, episode
batch size, view batch size, prompts, selected dense layers and preprocessing
remain fixed. No production implementation, default configuration or manuscript
is promoted automatically. Existing GPU queues keep their order and execution.

The completed seed-0, 1,449-image two-view run is 52.396489 mIoU at
291.999943 ms/image. Its strongest matched-data baseline is MLMP at
50.292803 mIoU and 40.909653 ms/image. Baselines retain their method-native
settings; this is not an equal-optimizer-budget comparison. Both full-split
results and complete baseline evidence are sealed in:

- `/home/wwww/hzl/results/trace_two_view_full_20261025`
- `/home/wwww/hzl/results/trace_two_view_full_baselines_20261025`

VOC21 keeps K=33, steps=36, batch size=16, lr=0.00063 and view batch size=16.
The actual measured population has 64.7255 diagnostic image inputs and
102.7255 total visual image inputs per source image. Diagnostic inputs account
for 63.01% of these inputs, not necessarily 63.01% of latency: adaptation also
includes backward and optimizer work. Adaptation costs 290.40 ms/image; final
inference costs 1.60 ms/image. Timing excludes loading, scoring and setup.

## Two complementary algorithm changes

1. **Query fewer second views.** The already running lazy-quality study uses
   measured target support as an upper bound on min(T,C), and measures C only
   when remaining candidates could materially affect normalized coefficients.
   Its per-image/query-class variant protects low-mass groups. It does not
   lower K or steps. A coefficient bound does not bound mIoU or gradients.
2. **Reuse a genuinely frozen encoder prefix.** Restrict adaptation to the
   last 6 or 12 transformer blocks' LayerNorms plus output LayerNorm. Freeze
   input LayerNorm and earlier blocks, then cache their original-image tokens
   and selected raw layer outputs once per episode. Execute every one of the
   36 optimizer steps on the trainable suffix. Diagnostics continue to run
   the full model on the actual pixel-intervention images; no prefix cache is
   reused across intervention images. All 18 selected dense outputs remain
   in the aggregation, including the cached frozen outputs.

The second change is **not an equivalent implementation of all-LayerNorm
adaptation**. Freezing the prefix changes the optimization subspace. Caching
is exact only relative to an otherwise identical frozen-prefix algorithm,
up to numerical implementation differences that must be measured. A matching
uncached last-6-block control separates freezing from caching. An output-norm
only variant (zero trainable blocks) is a development boundary control, not a
pre-approved replacement.

Saving 18/24 block evaluations in each of 36 repeated original-image updates
would remove about 26.3% of current total forward block-image evaluations;
freezing 12/24 would remove about 17.5%. These estimates do not include backward
savings and are neither measured FLOPs nor latency predictions. Pixel-view
diagnostic forwards still execute all blocks.

## Fixed experiment plan

- After the existing lazy-quality queue finishes, full 1,449-image VOC21
  followups of BOTH lazy algorithms run first. Their tolerance remains 0.01;
  it is not chosen or changed using validation outcomes. The global and
  per-image/query-class results are both retained.
- Four 4-image VOC21 smoke cells: cached tail-6, cached tail-12, output-norm
  only, and uncached tail-6.
- Five 128-image **training/calibration** cells: the four variants plus the
  complete two-view control. Calibration and full validation image IDs must
  be disjoint. No optimizer or candidate-budget search is performed.
- Both cached tail-12 and cached tail-6 receive full 1,449-image validation,
  predeclared independently of calibration rankings. Every result is retained.

This is thirteen new sequential cells. The preceding frozen 128-image VOC21
development comparison completed while this implementation was prepared:

| Algorithm | mIoU | ms/image | Diagnostic inputs/image |
| --- | ---: | ---: | ---: |
| Complete two-view | 54.206614 | 286.856478 | 64.875000 |
| Staged complete control | 54.210655 | 290.856947 | 64.875000 |
| Global lazy C, fixed 1% | 54.262838 | 234.977711 | 51.270833 |
| Query-class protected lazy C, fixed 1% | 54.143405 | 263.928726 | 58.937500 |

The same-128 strongest baseline is MLMP, 52.332810 mIoU. These are not
full-validation scores, and the approximately 0.06-point differences among
the candidate algorithms are not evidence of a statistically reliable gain.
The global rule saves 18.09% measured time and 20.97% diagnostic inputs here,
but its cached replay removed complete low-mass query groups. Class-wise full
results remain necessary; a global coefficient bound does not protect every
semantic class.

The new queue waits for the existing lazy-quality suite's lock and verified
completion seal. No concurrent GPU workers, canceled jobs, WATT or CLIPArTT
runs are introduced. Failure preserves its artifacts and stops this new queue,
not unrelated jobs. Code, input contents, configuration, environment and
upstream evidence are frozen before execution and checked before publication.

## Acceptance and accounting

The primary reference is the strongest completed local full baseline,
50.292803 mIoU, not old TRACE. Report all signs and class deltas. A candidate
must exceed this baseline on the full split **and** reduce synchronized
latency relative to the current fixed-config two-view run to qualify as an
efficiency candidate. This is seed-0 evidence, not a stability or SOTA claim.

Count patch-embedding executions, actual per-block input counts and calls,
diagnostic views, all optimizer steps, wall time and peak memory separately.
An update using cached early features still runs the suffix and is not a
zero-cost forward. Block-image counts are not FLOPs: blocks and backwards have
different costs. Keep pre-update supports and candidate masks auditable, and
never invent values for unmeasured views.

CPU tests cover cache identity, mutation/reset invalidation, original versus
intervention separation, gradient scope, uncached equivalence and block-work
accounting. A separate review audits algorithm implementation and queue
provenance before launch. A review finding no further issue is not a proof of
accuracy preservation.

## Execution checkpoint

The new frozen queue was launched on 2026-10-25 at 11:48 China Standard Time:
`/home/wwww/hzl/results/trace_voc21_forward_reduction_20261025`.
At launch, the prior 16-cell full baseline queue and 20-cell lazy-quality queue
were both completed and sealed; neither was interrupted. The new controller
starts with global lazy C on full VOC21, followed by protected lazy C.

Integrated CPU verification before launch: 777 tests passed, with three
expected warnings from CPU-only pinned-memory fixtures. Independent review
also injected OOMs in twenty complete 36-step episodes and checked actual
forward counters against the shared validators. See `review-round-12.md`.
Full follow-up scores and actual cached-prefix GPU speedups are not yet known
at this checkpoint; do not copy the development values into a full-results row.
