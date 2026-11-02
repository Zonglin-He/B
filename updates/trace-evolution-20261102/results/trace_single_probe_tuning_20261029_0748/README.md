# Per-Dataset Single-Probe TRACE Calibration

Actual service: trace-single-probe-tuning-20261029-0748.service.
Prepared SHA256: e1a1fcac0fa225cefebcc29f2f676e5366feade2a72388969b5d2d948de60426.
Code commit:71e56c6.390 joint CPU tests passed before launch.

User authorized independent B/lr/steps/K search on all eight original paper
datasets. Method stays ImageBalancedTRACE:C. See frozen_contract.md and
SUITE_PREPARED.json for exact selection ranges, tie rules and source binding.

Seven disjoint training calibrations are runnable now;Waterbirds calibration
is blocked because its existing2458 train IDs are in the original5322-image
evaluation population. Original full is unchanged. This suite must stop after
seven selections and cannot run partial full evaluation. Remaining input and
full stages will require a fresh manifest,not mutation/restart of this one.

Calibration:256 training images/dataset,seed2027,max80 unique trials/dataset.
Predictions seal before offline score. First annotation-content table is
sealed and pinned by subsequent trials. Calibration mIoU is not full mIoU.
All outcomes,including negative/null/ineligible trials,are retained.

The systemd service uses12GiBRAM/1GiBswap,Restartno,OOMPolicy stop,
RemainAfterExit yes,and a shared single-worker GPU lock. Gate/trial success
must be checked from canonical receipts,not inferred from this README.
