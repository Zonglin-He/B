# Baseline-first pilot results, 2026-10-25

All 24 baseline/control cells completed. All 100 files in the final suite seal
were rehashed with zero mismatches. The output directory is:

`/home/wwww/hzl/results/trace_baseline_comparison_20261025_v2/`

This is a fixed 128-image, seed-0 development panel per dataset, not a full
validation or three-seed table. Images, vocabulary, preprocessing and episode
boundaries match the algorithm pilot. Baselines use their archived native
method/optimizer settings; TRACE candidates retain fixed per-dataset settings.
Thus this is not an equal-lr/equal-steps budget comparison. No parameters were
tuned using these new outcomes. WATT/CLIPArTT remain excluded as requested.

## Accuracy

| Dataset | No Adapt | TENT | TPT | MLMP | Two-view quality | Selective 1% | Selective 5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VOC21 | 47.23 | 48.06 | 49.71 | 52.33 | 54.20 | 54.21 | 54.31 |
| PC60 | 24.94 | 25.83 | 23.34 | 28.44 | 28.51 | 28.60 | 28.55 |
| COCO Object | 19.58 | 20.12 | 20.31 | 23.38 | 24.72 | 24.57 | 24.57 |
| COCO-CD | 58.41 | 70.69 | 63.74 | 69.66 | 74.18 | 73.93 | 73.77 |

All three candidates exceed the strongest of the four measured baselines on
these panels. Two-view quality has deltas +1.863, +0.073, +1.342 and +3.487,
respectively. PC60's margin is small and is not evidence of a stable advantage.
Do not call these results state of the art or substitute them for full-split
scores. Old TRACE is not used as the acceptance threshold.

## Measured cost

Milliseconds per image, synchronized adaptation plus final inference, with the
first two episodes excluded. Loading, scoring and model initialization are
excluded for every method. The GPU and implementation are local to this pilot.

| Dataset | Best measured accuracy baseline | Two-view quality | Selective 1% | Selective 5% |
| --- | ---: | ---: | ---: | ---: |
| VOC21 | MLMP: 40.23 | 289.44 | 266.50 | 230.44 |
| PC60 | MLMP: 41.03 | 118.12 | 120.12 | 107.71 |
| COCO Object | MLMP: 43.32 | 259.32 | 231.12 | 208.86 |
| COCO-CD | TENT: 37.21 | 32.40 | 38.05 | 36.48 |

Two-view quality is about 12.9% lower in latency than the best-accuracy baseline
on COCO-CD, but is 7.19x, 2.88x and 5.99x its latency on the other three panels.
This is a single profile, not a repeated-timing estimate or a hardware-independent
ratio. Selective variants remain slower than baselines on the same three panels.
Saving diagnostic views alone does not establish baseline-level efficiency.

The native versus archived multi-prompt/multi-layer no-update inference-control
mIoU differences have absolute values below 0.009 on all four datasets. This
rules out a large dataset-mIoU discrepancy in these inference controls, not
bitwise, gradient or adapted-trajectory equivalence of the two implementations.

## Decision after diagnosing weak ablations

The priority simplification candidate is the actual two-view implementation:
retain target-preserved and context-perturbed supports, use q=min(T,C), and do
not compute the target-suppressed view merely to apply the small dominance
multiplier. It has already been implemented outside production and evaluated;
this round adds matched baseline evidence, not an invented replacement score.

Its main narrative remains that original confidence is insufficient and that
candidate evidence should survive changes in surrounding context. It cannot
continue claiming to measure residual support after suppressing the candidate.
Any adoption must update that part of Method, figures and claims honestly.

The main tracing-removal results are not uniformly weak: on the larger removal
panels they cost 0.47/0.60/1.59/7.79 mIoU. The weak incremental part is dominance.
Connectedness has smaller, conditional effects and should not be made into an
independent novelty claim without stronger evidence. Region normalization has
an adverse COCO Object result that must remain visible. See the mechanism audit
for exact normalization identities and measured coefficient changes.

Production defaults and paper claims are not promoted from these 128-image
results. The next acceptance check is fixed full evaluation against matched
baselines for the simpler candidate, prioritizing PC60's narrow margin. Changes
in code or claims follow that check; no artificial attenuation amplifier or
test-selected harm threshold is introduced to enlarge ablation differences.
