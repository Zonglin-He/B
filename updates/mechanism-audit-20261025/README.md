# Reviewer Evidence Extension

## Status

CPU reanalysis is complete. New GPU work is queued, not completed. The watcher
waits for the existing original suite AND its existing representative ViT-B/16
suite; no live worker or frozen queue was modified. Three four-image smoke cells
precede three 64-image cells: sham on COCO Object/COCO-CD and native support
distributions on VOC20. COCO-Stuff distributions reuse its completed 64-image
role-policy cache rather than adding a duplicate GPU run.

Six CPU tests pass. Native-update equivalence, identical-operator native views,
mask area and original readout checks are required by each GPU smoke/formal cell.
GPU validity/completion must be checked after those cells actually run.

## Completed CPU Evidence

Paired bootstrap uses 10,000 shared complete-episode resamples and reaggregates
class intersection/union; it does not resample pixels or average per-image IoUs.
Canonical real paths reconcile historical symlink aliases. Source prediction
hashes and ordered image populations must match before computing an interval.

| Comparison | Images | Episodes | Delta mIoU | 95% episode-bootstrap CI |
| --- | ---: | ---: | ---: | --- |
| Full minus confidence weighting, COCO-CD | 500 | 18 | +7.8781 | [+5.4543, +10.3831] |
| Full minus confidence weighting, COCO Object | 5000 | 358 | +2.0575 | [+1.5054, +2.5905] |
| Full minus w/o Region Construction, COCO-CD | 500 | 18 | +2.7803 | [+1.5119, +4.0364] |

These are cache reanalyses, not new inference or independent seed trials.
COCO-CD groups retain aligned, counterfactual and all four original groups;
they do not establish universal worst-group dominance.

All-candidate selection audits retain both selected and excluded masks,
confidence/rho/area, offline GT purity, and actual pre/post label transitions.
Unselected candidates with rho >= 0.5 are 1.64% on PC60 and 1.48% on COCO
Object in these fixed panels. They are not absent. Confidence top-K is a budget
policy, not an oracle or a universally best selection rule. Post-update harm on
an excluded candidate reflects the complete episode, not that candidate's
isolated causal update utility. Thresholds .25/.50/.75 and population quartile
are all reported, not selected by favorable rank.

K coverage curves are offline candidate/rho-mass coverage, NOT mIoU/latency
measurements. Existing measured K/selection/accuracy-cost results remain their
own evidence. Protocol inventory copies settings and configuration sources;
historical tuning-label provenance is not invented from config files.

## Queued Sham Design

For each candidate, place an equal-area mask completely outside it. Prefer an
in-bounds shape-preserving translation; otherwise use a deterministic
noncontiguous equal-area outside mask. No GT or outcome chooses the placement.
Native mask, class/readout and original loss positions remain fixed.

Three adaptation policies use identical initial candidates and optimization:
native TRACE; sham residual only, keeping native shared quality; and all-view
sham placement using the unchanged original q/rho formula. Actual original and
sham score readouts, masks, applied pixel area and changed pixel counts are
retained. Applied area is exactly matched; unchanged flat-image pixels can make
the number of numerically changed pixels differ, so both counts are recorded.

If a candidate occupies more than the available outside area, no disjoint
equal-area sham exists. Retain its native scores in the adaptation control and
record it as ineligible for the paired sham estimand; do not drop the image or
silently substitute a zero residual. Shape-preserving and area-only controls
must be reported separately. This is a location-specificity control, not proof
of a unique causal confidence source.

## Other Recommended Items

- Connected-region fragmentation/mixing and boundary A/B/C are already finished
  under rebuttal_priority64_repaired_20261025_1940, not rerun here.
- Role-policy controls already finished; no additional role-penalty tuning.
- Existing refresh/drift and operator controls are reused; no extra frequency or
  operator sweep is added.
- Runtime/accuracy-cost and second backbone remain in their original suites.
- Figure repair is manuscript work, not represented as an experiment here.
- No paper text, submitted algorithm or existing result is overwritten.
