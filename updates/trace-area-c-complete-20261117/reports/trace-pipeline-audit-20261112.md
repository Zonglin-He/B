# Pipeline Single-Factor Training Audit

## Scope

The November12 human requested a component-by-component investigation after
both revised arms completed the full original15 S5 evaluation. This is method
diagnosis on the existing32-source VOC training panel, not validation tuning.
All seven controls below are fixed before any new outcome. No parameter grid,
new corruption, benchmark, monitoring agent or recurring automation is added.
Original TRACE, official MLMP, completed November11 revisions and releases are
immutable. Old negative/null trials remain reported.

## Questions And Controls

All controls retain L14, native seven prompts, native readout, Adam lr.001,
ten steps and original B2 grouping. Clean plus all original15 S5 types;32
training sources x16 conditions x7 new controls =3584 predictions. Reuse
authenticated baseline/area_C/coverage_native_budget counts, with actual
first4 training replay for all16 conditions before inference. Separate CPU
scoring follows full predictions and actual successful GPU termination.

1. `area_uniform_covered`: replace C by its per-image covered-pixel mean,
   preserving the top8 support and total bonus mass. Tests the value of C
   differences within the selected support.
2. `area_uniform_image`: uniform per-image full-grid mean C. Same total bonus
   mass, no spatial selection. Tests spatial weighting versus changing the
   pixel/CLS balance. Per-image C mass still requires probes; not a free method.
3. `area_shuffled`: one fixed PCG-independent torch shuffle seed2026111201,
   identical values and zeros per image, destroyed spatial alignment. This is
   a negative control with changed spatial smoothness, not a recommended model.
4. `coverage_static`: same all-semantic-component C bonus but no step5 refresh.
   Isolates refresh from initial all-region coverage and alias merging.
5. `coverage_uniform_image`: retain0/5 probe schedule and current C mass but
   uniformize its spatial map; compare with coverage_native_budget.
6. `native_no_cls`: remove only CLS from the native MLMP objective.
7. `area_no_cls`: remove only CLS from area_C. These distinguish the CLS term
   from the earlier combined global-pixel/CLS restoration. Neither coefficient
   is selected by validation outcome.

Prespecified primary contrasts are the above versus their named parents,
plus area_no_cls minus native_no_cls and the two existing parent contrasts.
All outcomes reported, no automatic winner.10000 common whole16B2-group draws,
seed2026111201; arithmetic15-type macro, no marginal interval subtraction,
pointwise95/no multiplicity correction, undefined/absent classes retained.
Only16 source groups and prior training exposure limit inference.

## End-To-End Checks

- Frozen RGB byte hashes, exact224 normalization, query/class mapping, full
  state and Adam reset, stored parent prediction replay, no GT in model stage.
- First4 training sources per type only: native equivalence in actual loss,
  optimized parameters and logits <=1e-5, zero changed labels; area and
  coverage parent labels also match. Not a new all32 determinism experiment.
- First4 only, steps0/5/9: separate pixel/CLS/bonus gradient norms/cosines,
  query entropy = semantic-sum entropy + within-alias entropy, prompt/mean-layer
  vs native adaptive readout semantic disagreement. Semantic summation here
  is an entropy diagnostic, NOT a changed max-over-alias final readout.
- First4 only: coarse16-grid nearest labels versus bilinear224 final labels;
  actual raw-logit interpolation parity. Disagreement is not GT error.
- C and original confidence per top8 region, singleton all-semantic components,
  area and classes. Descriptive, not calibrated probability or causal proof.
- New audit wrapper honors native alpha_cls and rejects text integration.
  Old wrapper hardcodes1: a reproducible latent portability bug, but not a
  numerical invalidation of old results whose alpha_cls was exactly1.
- CPU scoring reuses existing GT bytes/hashes, indexed labels, CV2 nearest224,
  background/void ignore255, pooled I/U and same original B2 source groups.

## Operation

Finite direct pipeline, original GPU lock, GPU12GiB/1GiB swap and CPU4GiB/
512MiB swap, one CPU thread/hiddenCUDA, OOMstop/Restartno. Failed bytes retained;
no blind retry. A code test or plan is not a scientific result. No upload,
paper rewrite, global model change or new full validation is implied.
