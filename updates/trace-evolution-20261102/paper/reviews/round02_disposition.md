# Author disposition of independent round02

The original report and its5/10 borderline score are retained verbatim.
This document is an author response, not a replacement review or an upgraded
score. Frozen reviewed packet:trace_evolution_review_round02_20261029_0015.
No fatal method-identity, inference-label, pooled-IoU or major unsupported-claim
error was identified in its bounded checks. The research goal is NOT complete.

| Finding | Assessment | Minimal action and claim boundary |
| --- | --- | --- |
| W1 parameter origin/exposure | Accept missing disclosure; do not infer inference-time leakage from it. User retains per-dataset configurations and does not request equal-budget baseline retuning. | Audit small historical calibration/config/trial records and expose unknowns. Do not describe current full splits as untouched. No new MLMP reruns. A locked common-config/regrouping check may test a different condition, but cannot erase historical data exposure. |
| W2 incremental two-view value | Accept. Existing data do not establish that min is necessary or better than every simpler probe. | Keep all contrary controls. Put candidate-specific probing/allocation, not a min optimality theorem, at the center. Audit T/C complementarity and actual normalized allocation differences on all five splits; measure genuine standalone single/dual-probe cost. Freeze any next comparison before scoring, with no aggregator selection on old test outcomes. |
| W3 current No Adapt omission | Accept and repair directly using already sealed pre_result counts. | Add the current readout's pre-adaptation row and TRACE-minus-current-pre delta separately from literature deltas on all six splits. Exporter tests require the pre result and preserve negative local deltas. Main narrative now leads with within-implementation adaptation and matched allocation. Also state224-grid scoring and class/background conventions. Do not silently replace literature No Adapt with local values. |
| W4 conditional risk association | Already bounded; do not claim causal attribution or universal C-to-W reduction. | Keep Object/Stuff tradeoffs and distinct denominators. Existing integer-count summaries remain available; the five-split descriptive analysis is not independent per-candidate inference. Reviewer-derived exploratory intervals are not imported as author experiments. |
| W5 episode/perturbation generalization | Accept bounded missing evidence, not an implementation error. | Deterministic replays remain replays. Any new subset regrouping/common-config check is explicitly conditional on reused source images, not independent population validation. No automatic slow-dataset/backbone expansion. |
| W6 standalone cost | Accept. Historical profiler and shared-arm runtime cannot answer it. | Prepare a fresh standalone worker with actual0/1/2 probe schedules and synchronized full-episode timing, keeping model setup/loading/scoring and warmup exclusions explicit. Do not infer speedups from the old shared suite. |

## In-progress bounded work

- CPU probe-complementarity audit completed on493,616 candidates/2,035
  episodes;23 tests passed. Both probes change allocation in all episodes;
  this does not prove extra mIoU benefit. Results and the audit scope are
  retained in trace_probe_complementarity_20261029_0030. Laplace is closed.
- Historical provenance audit completed:195 small sources checked; six
  inherited tuples have explicit links, but documented later LR acceptance
  used full-validation scores on VOC20/CD/Stuff. This is disclosed rather
  than described as training-only selection. See docs/parameter-provenance-audit.md
  and trace_parameter_provenance_20261029_0030/audit.json. Aristotle is closed.
- Standalone worker/queue implementation completed; independent code audit
  found noP1/P2 and main reran102 relevant tests in the CUDA environment
  with CUDA disabled. Hooke and code auditor are closed.
- Main froze136 sources and six formal manifests plus two4-image gates,
  then launched trace-standalone-followup-20261029-0105.service with12GiB
  RAM/1GiB swap, sharedGPU lock and no retry. All six128-image cells and
  30 arm-results completed, two strict native/min gates passed, and service
  terminal is loaded active/exited MainPID0 Result=success/ExecMainStatus0.
  All six prediction cells were sealed before scoring. CPU exporter validated
  numerical aggregates/counters and the root/hash/PID-bound real terminal.
- Actual outcomes are retained in generated/standalone_comparison.tex and
  standalone_cost.tex, with source/terminal evidence in standalone_provenance.json.
  Common Object gives a negative TRACE-minus-Confidence difference and both
  single views outperform min in some cells; these values are not omitted.
  The paper frames probing and aggregation separately, not a universal
  two-view/min superiority. Model-side cost is measured, not borrowed from
  old shared-arm timings. The historical profiler is no longer embedded.

These meaningful repairs and diagnostics are now complete. Rebuild and inspect
the complete paper, freeze a new packet, and only then request another fresh
independent manuscript review.
The closed round02 reviewer is not asked to change its score. Future rounds
must assess a newly frozen substantive revision, not sample reviewers on the
same evidence until one returns a favorable result.
