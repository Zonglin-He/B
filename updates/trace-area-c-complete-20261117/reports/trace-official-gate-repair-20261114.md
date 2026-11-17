# Official TRACE Gate Reference Repair

The V4 corrected-data run finished PC60 NoAdapt and MLMP predictions and CPU
scores successfully. NoAdapt is24.95 in all three trials. MLMP trials are27.95,
28.00,28.00 (official rounded mean27.9833333333, population std0.0235702260),
versus paper27.99. All selected baseline settings remain unchanged.

V4 then failed on November13 at16:33:10 CST in the area-C first-four RGB gate;
the dispatcher exited at16:33:12. No area-C prediction chunks or score exist.
The gate instantiated a new NativeTraceRevision for the second two-image batch
before resetting the native model left adapted by the preceding batch. Thus its
frozen diagnostic reference captured the preceding episode's adapted weights,
whereas LargeBatchAreaC retained the genuine initial reference. This is a gate
state-order error, not a measured scientific method failure or an OOM.

The new continuation resets before capturing each batch's reference. Actual
ten-step loss, parameter, logit differences must still be at most1e-5 and decoded
label differences exactly0. The repair does not change the adaptation method,
candidate generation, selected22 parameters, official methods, data, or metric.
It restores all RNG state after the bounded first-four check. Failed gate errors
now include the batch start and numerical differences.

V4 source snapshots, original failure bytes, unit identities and journals stay
intact. A new root and fresh units are used once. Only the two completed corrected
PC60 baselines are reused, by original absolute stage paths with true retained
prediction/score success, all output hashes, and independent source-I/U point
recomputation. They are not relabeled as new executions. Every other scientific
arm is run in the new root. Existing official data transforms and scoring are
unchanged. Monitoring stays OFF; automation stays PAUSED.

PC TRACE calibration overlaps the corrected reference validation split. Fixed
selected parameters are retained without corrected-validation retuning. These
comparisons are development evidence, not untouched held-out PC performance.
Waterbirds/COCO-CD are extensions and B16 uses the declared layer transfer.
