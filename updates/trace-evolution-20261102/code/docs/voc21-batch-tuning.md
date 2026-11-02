# VOC21 class diagnosis and sequential batch-size calibration

The unchanged simplified TRACE reached 52.3504787 mIoU on all 1,449 VOC21
validation images. The earlier 52.58 result concerns a fixed 512-image subset,
not the complete validation set. Diagnosis and tuning must keep those scopes
separate.

`tools/analyze_voc21_tuning.py` first saves absolute full-validation class IoUs
and matched-panel class changes. Full and confidence-weighted candidates on
the 512-image panel are joined by episode, image, patch, class and mask hash;
the analysis does not infer that absent intervention scores were zero. Old
full-validation scalar results without per-class counts cannot support a
per-class comparison. Any old-versus-new implementation comparison identifies
the kernel and view-batching changes, rather than attributing all differences
to the reliability formula.

## Fixed search

`tools/run_voc21_batch_tuning.py` tests adaptation batch sizes **1, 2, ..., 16**
in this exact order, in separate sequential workers. Only that scalar changes.
K=33, learning rate=0.00063, steps=36, view batch size=16, the three interventions,
the shared reliability formula, connected candidates, region-normalized loss,
prompts, selected layers, and visual LayerNorm-only adaptation remain fixed.
The production package is copied and hashed before execution, and its hashes
must match the existing experiment's algorithm.

The calibration cohort is a fixed 512-image subset of the complete 1,464-image
VOC2012 segmentation **training** split, selected with seed 2027. Its image and
annotation paths must not overlap the 1,449-image validation split. Every batch
size uses the same ordered calibration images. Different batch companions
are an intentional consequence of the parameter under test, not equivalent
per-image adaptation trajectories.

Selection uses training-calibration mIoU only. Trials within 0.02 mIoU points
of the global maximum are tied; lower measured per-image latency breaks ties.
Recorded OOMs are retained as OOMs, while other unresolved failures stop the
search. A failed/interrupted attempt is never overwritten or silently selected
because it scored better.

## Scheduling and verification

The existing primary experiment queue is not interrupted. The tuner waits
first for its final-audit lock, then for its primary lock, and requires both
queues to be complete before any model execution. This order avoids a deadlock
with the audit that already waits for the primary queue. All locks are inherited
by the tuning worker so coordinator loss does not launch another active worker.

After all sixteen values have been tested, the calibration-selected batch size
is frozen and evaluated on the complete validation set once. If batch size 16
is selected, the already completed, identity-checked full result is reused
instead of rerun. The selected result is reported even if it fails to improve
the original 52.35; validation is not used to select a different trial.
No new class-specific weights, guards, role rules, view operators, or reliability
equations are introduced. Neither the manuscript nor production defaults are
automatically replaced by an unverified configuration.

Outputs include `analysis/`, the frozen runtime and protocols, `trials.csv`,
per-trial diagnostics and class counts, calibration accuracy/time curves,
`selected_config.json`, `selection.json`, `verification.json`, and
`full_class_comparison.csv`. The calibration curve is not a main-result curve,
and the single-seed verification is not a three-seed result. Earlier ablations
remain tied to their original batch-size-16 protocol.
