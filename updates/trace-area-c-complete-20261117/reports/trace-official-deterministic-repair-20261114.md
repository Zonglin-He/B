# TRACE Runtime Determinism Repair

The corrected official PC60 baselines are complete and verified: NoAdapt24.95
in three trials; MLMP27.95,28.00,28.00, mean27.9833333333,std0.0235702260.
Paper values24.95 and27.99 are not tuning targets. These successful original
stage paths and identities are reused without new inference or GT decoding.

TRACE V4 failed at its four-RGB equivalence gate. V5 corrected a real secondary
bug: a per-batch reference was captured before resetting preceding adapted
weights. This alone did not repair the first-batch trajectory discrepancy.
V6 removed unit-valued arithmetic; that also failed. Both failed source snapshots,
units, bytes and journals remain intact. No baseline or full TRACE score was
produced by either failed gate.

RGB-only stage diagnostics found equal candidates,C,initial logits, first-step
gradients and first updated parameters, followed by unequal second-step gradients
despite equal incoming parameters. The previous TRACE entry point invoked
configure_deterministic_runtime before CUDA; the new official-data entry point
only set upstream seeds and omitted strict deterministic algorithms/CUBLAS state.
With strict runtime restored, a changed four-RGB diagnostic matched every
optimized gradient and parameter at all ten steps, with exactly zero loss,
logit and decoded-label differences. The diagnostic is engineering evidence,
not a scientific score or permission to bypass the real first-four gate.

The adopted continuation restores the historical strict runtime for area-C
prediction processes only. It keeps the original LargeBatchAreaC implementation,
all22 selected fields, FP16 model dtype and optimizer unchanged. The identity
arithmetic experiment is NOT adopted. Official NoAdapt and MLMP methods,
configuration and upstream runtime are unchanged. The actual new prediction unit
must independently pass the unchanged1e-5/zero-label four-RGB comparison using
the original TRACE implementation before it writes full predictions.

All previous failures stay stopped, never retried with unchanged bytes. The new
version has a fresh root, frozen source copies and fresh units. No monitoring or
automation is activated. PC TRACE calibration overlaps corrected reference val;
future scores are development comparisons, not untouched held-out PC evidence.
