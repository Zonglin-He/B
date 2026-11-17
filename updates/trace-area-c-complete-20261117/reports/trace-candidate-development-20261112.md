# Bounded Candidate Development

The actual human authorizes continued method modification for better metrics.
This follows completed alignment training: clean improved, but none of the three
new arms exceeded area_C on the training S5 macro. No new validation was read.

## Fixed Two Variants

Retain the executed area_C full-image query entropy + CLS + per-pixel C bonus,
same L14/last18/seven prompts/.001/Adam/10steps/B2/final native readout. Reuse
the validated same-call pre-ln_post diagnostic cache. No step gate or LR grid.

First collapse aliases by maximum query probability, WITHOUT renormalization,
then find four-connected semantic regions. Native class argmax is preserved.
C is initial-frozen native averaged-prompt/adaptive-readout max-alias query
probability within the foreground-preserving counterfactual region. This
merges aliases in both support and C; it is not a pure ranking-only contrast
against original query-region area_C.

- semantic_top8: select at most8 regions/image by mean semantic confidence.
- semantic_mass_top8: select at most8 by confidence * sqrt(area/grid_pixels).
  Deterministic ties: larger area, semantic class, component ID. This is a
  fixed area-aware hypothesis, not a fitted exponent or GT-informed ranking.

Both use the same loss, counterfactual and diagnostic-view cap. Their direct
comparison isolates the ordering change. A larger region can contain wrong
pixels, so area weighting is not guaranteed to help; retain all outcomes.

## Evaluation

Same32 repeatedly-used training sources, clean+original15S5,16B2 groups,
1024 new predictions. Reuse exact MLMP/area_C/coverage/alignment counts from
the preceding completed round. Current frozen methods are not selected or
retuned using the exposed full-validation data. Every condition first4:
native-equivalent actual10step losses/parameters/logits<=1e-5 and unchanged
labels; original cached area_C replay must match stored labels exactly.

All16 prediction seals and actual retained GPU success precede separate CPU
GT scoring. All eight arms reported, comparisons of each new arm with MLMP and
area_C, and direct mass-minus-confidence. 10000 shared whole16B2 draws with
seed2026111204; pointwise95/no multiplicity adjustment/undefined retained.
No automatic winner/full-validation rerun.32-source repeated development is
not independent confirmation or full-benchmark evidence. Extra compute and
failure costs remain disclosed. GPU12GiB/1GiB,CPU4GiB/512MiB,one-thread,
uniqueGPUlock/OOMstop/Restartno; automationpaused/monitorsdisabled.
