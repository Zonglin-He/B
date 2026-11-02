# VOC20 append suite

Authorized by the user on 2026-10-28. This suite must not interrupt the
18-cell main evolution suite or the two-cell cached-control suite. Its
controller is already running as `trace-evolution-voc20-20261028-1428.service`;
do not start duplicate workers.

The current `waiting_for_dependencies` state is expected. Both upstream
manifest identities, full completion counts, suite seals and successful
service terminal states are required before taking the shared GPU lock.
An upstream failure stops this append suite without retries or recovery.

Frozen plan: four-image TRACE-only smoke with exact reference gate, then
complete VOC20 seed0, seed1 and seed2 evaluations, 1,449 images each.
Configuration is inherited unchanged: batch1, lr0.001105, steps9, K8,
class extensions enabled, view batch16. No new baseline or hyperparameter
search. Method is TwoViewTRACE with rho=min(T,C), not historical three-view
TRACE. Fixed-order deterministic seed repeats test replay, not population
uncertainty. Predictions are sealed before separate GT scoring.

The latest user instruction requires all three suites to finish and verify
before updating the paper, appendix and figures and commissioning the next
independent manuscript review. Preserve all results and the original review.
