# Review round 8: dataset diagnosis and parameter-only efficiency search

The review covered `tools/run_accuracy_budget_search.py`, its benchmark CLI
overrides and regression tests. No production algorithm or existing frozen
experiment was edited. Two P2 issues were identified and repaired before the
new queue was registered:

- An unchecked OOM marker could exclude a trial. New data-batch OOMs now
  preserve failure/protocol hashes and stop as unresolved. Existing markers
  also stop; resource contention is not silently treated as configuration
  infeasibility.
- Upstream completion summaries did not prove that actual evidence was
  complete. The continuation now validates all sixteen calibration artifacts
  using the upstream frozen source/tools/inputs, reproduces its selected
  batch and checks the complete 1,449-image validation artifact. Both reused
  baseline and newly selected full-validation results are tied to their
  original saved result hashes.

The reviewer confirmed both findings closed. Tests cover missing trial seals,
changed results, inconsistent winners, incomplete full validation, wrong full
scores, a changed selected-full seal and unresolved OOM. Final all-CPU suite:
230 tests passed. The original VOC21 GPU queue continued throughout review.

The independent cache analyses keep cohort identities, EPS pruning, missing
measurements, class-extension mapping and negative results explicit. They
measure neither candidate GT precision nor gradient attribution. Reduced-view
calculations remain offline; production still evaluates all three views.

This closes the identified implementation issues within a bounded review.
It does not establish a global hyperparameter optimum, new full-dataset
performance, stable speedup across devices, or invariance of adaptation under
approximated weights. The new parameter search is asynchronous and must
complete before its proposed settings can be evaluated for adoption.
