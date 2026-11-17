# TRACE Main Ablation: Method-Level Component Removal

## Scope

The November17 human instruction narrows the main ablation presentation to
removal of components actually proposed by the method. Parameter sensitivity,
inherited-objective checks and finer mechanism controls are not separate
proposed-method components. No new experiment or parameter selection follows
from this presentation change, and no prior outcome is discarded.

The current area_C objective retains the MLMP global pixel/CLS backbone and
adds one regional adaptation branch. That branch constructs initial regions,
measures context-conditioned C and adds the area-aware regional entropy term.
The existing full-population `no_region` control disables that complete
branch's contribution to the update by setting its loss coefficient to zero.
Every other selected operating field remains unchanged.

## Main Component Table

Full-population component ablation, three common-block episode arrangements.
Metric: mean pooled mIoU (%); delta: complete method minus removed component.

| Method | Proposed regional branch in update | Object5000 | PC605104 |
|---|---|---:|---:|
| Complete TRACE area_C | Enabled | 32.4412 | 28.8355 |
| w/o proposed regional adaptation branch | Disabled | 32.3686 | 28.8921 |
| Complete minus removed component (pp) | -- | +0.0726 | -0.0566 |
| Pointwise paired95% CI (pp) | -- | [-0.0916,+0.2442] | [-0.1228,+0.0103] |

Both intervals include zero. This is the main module-removal result; reducing
the table does not convert it into evidence of a positive component effect.
The new PC common-block mean does not replace the original shuffled main mean.

## Presentation Categories

| Existing work | Paper role |
|---|---|
| Complete versus no_region | Main proposed-component ablation |
| Fixed official MLMP versus selected TRACE | Main benchmark comparison, not a same-configuration component removal |
| Learning rate, batch size, steps, K and temperatures | Parameter sensitivity, separate from the main ablation |
| no_global/no_cls | Inherited MLMP-objective diagnostics, not TRACE's newly proposed modules |
| Uniform/shuffled mass and alternate scores/candidates | Supplementary mechanism and replacement diagnostics |

All126 diagnostic cells remain in the existing complete report and appendix;
they will not be expanded or presented as126 independent method components.
Mass-matched score controls still depend on C through total mass and cannot
be relabeled as a literal removal of C. Whole-branch removal also does not
separately identify the effects of candidate construction, C or area allocation.
No original module definition is changed to favor a result.

## Artifacts And Limits

- Numbers: `trace-paper-full-component-results-20261117.md` and the sealed
  `complete_results.json` in the independent November17 paper snapshot.
- Paper-ready table fragment:
  `paper_evolution/ablation_presentation_20261117-v1/main_ablation.tex`.
- The existing sealed17-page PDF and its delivery hashes remain unchanged.
  This update supplies a revised main-table presentation, not a newly compiled PDF.
- The accuracy wrapper still computed diagnostic C views in the disabled-loss
  arm; this table does not claim the speed or memory benefit of removing them.
- No GPU, raw GT/RGB/model/prediction payload, monitor or automation is started.
