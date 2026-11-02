# Review round 11: two-view full validation and lazy counterfactual probes

## Independent reviews and fixes

1. Full-panel adapter review generalized 128-image assumptions to explicit
   sealed `/full` panels. CPU simulation covers PC60's 5,105-image population,
   568 episodes and two-image final episode. Old pilot behavior stays supported.
2. Full followup audit identified parent input-content, execution-environment,
   cost-recomputation and final-artifact-validation gaps. The worker now checks
   parent content hashes and environment before GPU work; the controller derives
   step counts independently and recomputes timings and visual-input/call totals.
   Accepted artifact hashes are retained and revalidated before final sealing.
3. Post-analysis now checks vocabulary/source hashes and ownership of referenced
   results. Analysis exceptions are isolated from sealed baseline queue state.
4. Future full preparation checks the benchmark and algorithm helper hashes in
   addition to production source. The already running full snapshot was verified
   to match its pilot before these guard additions; it was not modified.
5. An independent algorithm reviewer checked the grouped/global 1% proofs,
   EPS filtering, missing measurements, actual query logic and numerical-staging
   control. A separate integration reviewer found that intact baseline and
   algorithm suites could be from different cohorts. Preparation and final
   verification now bind the baseline parent path, seal, runtime and population.
   The reviewer rechecked and closed that concrete finding.

All corrections were made in research tools and tests. Production `trace_ovss`
and the manuscript remain unchanged. The baseline verification changes are in
commit `8cc3805`; experimental queues freeze their own complete source snapshots.

## Scope boundaries

- PC60 complete two-view validation finished at 30.584325 mIoU, seed 0.
  A matched full-baseline claim remains pending the queued baseline runs.
- The lazy algorithm's cache replay proves conditional query-count and weight
  budget facts, not GPU speedups, gradient fidelity or mIoU preservation.
- Full validation overlaps the previous development panel and is single-seed.
  It is not an independent holdout or three-seed stability result.
- Baselines keep archived native optimizer settings and backends. Matching
  data/episode identities does not imply an equal optimizer or tuning budget.
- No performance-sign filtering, failed-cell overwrite or automatic production
  promotion. No WATT/CLIPArTT rerun, heavy datasets, second backbone or ACDC in
  this bounded stage.

## Verification checkpoint

The final integrated CPU suite passed **612 tests in 14.61 seconds** after
all fixes and the baseline-binding regression. Three warnings concern CPU-only
pinned memory, not experimental model failures.
Runtime source/input seals and active worker identities are checked separately
from these CPU tests. Independent review is bounded, not a guarantee that no
scientific or implementation criticism can remain.
