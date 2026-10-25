# Revised Motivation Figure QA

## Revision Scope

- M1 and M3 are kept as main-paper candidates from the original output.
- M2 is redrawn to separate context support, forced context-update harm, and damage to correct context/background regions.
- M4 is retitled and redrawn as a fragment-guard budget figure.
- M5 is simplified with stacked entropy-source fractions and a sampled reliability scatter.
- M6 is redrawn as reliable-vs-rejected attention plus trust-margin distribution; the residual unlabeled attention bucket is not plotted in the main panel.
- M7 is not used as a main-paper claim figure because the current after-update region-accuracy metric can favor hard fragment rejection.

## QA

- Source data: all panels derive from CSV/JSON files under the same motivation result directory.
- No invented values: all plotted values are recomputed from `m2_source_harm_summary.csv`, `loss_terms.csv`, `m5_entropy_source_distribution.csv`, `region_diagnostics.csv`, and `M6_attention_overlap/*`.
- Output: editable SVG and PNG are written to `figures_revised/`.
- Axis policy: all rate/proportion axes start at zero.
- Palette: restrained, colorblind-conscious palette; no rainbow/jet.
- Reviewer risk: M2 uses diagnostic region-level metrics from a bounded sample, not final benchmark mIoU. M7 remains excluded from the main figure set.
