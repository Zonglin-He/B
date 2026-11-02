# Image-budget confirmation: completed evidence and adoption decision

This is an author analysis, not a new manuscript review. The last independent
manuscript score remains round03, 5/10. The raw experiment and earlier draft
remain unchanged. The new suite finished at Beijing03:08 on2026-10-29;
loaded active/exited service, MainPID0, Resultsuccess, ExecMainStatus0. Separate
CPU verification checks the completion and all8cell/48arm identities before
the next launch; a score is not inferred from a completed job.

## Locked Decision

Calibration selected one global **C** probe, before all confirmation predictions.
The primary criterion was worst-dataset margin versus image-confidence, with
the fixed0.02pp/fewer-input/macro/tie-order rule. Calibration margins were
PC60-0.272957/Object+0.579561; this was not a claim that calibration improved
both datasets. The subsequent confirmation outcomes do not change selection.

| Dataset | Original order | Order20261030 | Order20261031 |
| --- | ---: | ---: | ---: |
| PC60: C minus image-confidence, pp | +0.176377 | +0.069026 | +0.153832 |
| Object: C minus image-confidence, pp | +2.103277 | +2.100626 | +2.924691 |

Each row reuses the same256 images across orders, not three independent
populations. These are pooled semantic mIoU differences, not per-image mean
scores or uncertainty intervals. They support proceeding to a separately
frozen full-split evaluation of C, not replacing any existing paper scores.

## What the Results Do and Do Not Establish

- All six selected-C contrasts versus image-confidence are positive. Against
  image-uniform, PC60 order20261030 instead favors Uniform by0.040789pp.
- PC60 original episode-min remains above C by0.051773/0.174038/0.137791pp.
  Removing cross-image normalizer competition is an algebraic property, not
  evidence of universal accuracy or ordering superiority.
- Object T is above C in all three orders. This does not authorize switching
  the selected global probe after looking at confirmation.
- Actual diagnostic inputs fall from11188 to5594 on PC60 and17326 to8663
  on Object, exactly50 percent. Total encoder work and elapsed time do not
  fall by50 percent because adaptation and prediction remain unchanged.
- C model-side time is approximately101.79--102.14ms/image on PC60 and
  264.23--265.27ms/image on Object. Against episode-min this is approximately
  25.1--25.3 percent and16.1--16.2 percent lower, respectively. Time excludes
  setup/decode/H2D/geometry/writes/scoring and only omits episode0; every
  image remains in accuracy and work counts.
- Historical July4096 image exposure cannot be exhaustively reconstructed.
  These cohorts are prospective only relative to the documented exclusions;
  do not call them globally unseen independent confirmation.

## Next Bounded Stage

`docs/image-full-plan.json` locks C on six complete existing fast splits with
Uniform and Confidence controls at the same inherited settings. This directly
tests the new identity and full-population mechanism/cost evidence; it does
not close the prior exposure or matched-external-baseline limitations. New
worker code must pass CPU tests, a separate code audit, a frozen-source check
and six real GT-free gates. No additional review on unchanged old evidence,
no T/min reselection, no parameter search or automatic slow-dataset expansion.
