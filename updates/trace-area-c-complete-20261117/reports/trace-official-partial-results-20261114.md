# Verified Partial Official-Source Comparison

All entries below are complete three-trial scores, not gate diagnostics or
partial prediction estimates. On November14, V8 CPU prepare verified original
prediction and score retained success, all output hashes, and independently
recomputed all30 trial points from compact source I/U without decoding GT.

| L14 dataset | Images | NoAdapt | MLMP mean/std | area-C mean/std | area-C minus MLMP pp |
|---|---:|---:|---:|---:|---:|
| PC60 | 5104 | 24.95 | 27.9833 /0.0236 | 28.7133 /0.0519 | +0.7300 |
| COCO Object | 5000 | 25.03 | 30.4200 /0.0283 | 32.4400 /0.0000 | +2.0200 |
| VOC20 | 1449 | 75.91 | 83.8033 /0.0205 | 86.1200 /0.0000 | +2.3167 |

VOC21 NoAdapt is45.13; its MLMP/TRACE scores are not yet available in this
snapshot. The complete26-condition comparison, B16 and15S5 are not complete.
Std values describe three official rounded trial scores, not confidence intervals.

Evidence: `TRACE_official_baselines_20261114-0952-v8/inputs/REUSED_RESULTS_VERIFIED.json`.
Original PC60 NoAdapt/MLMP stages remain in V4; all other listed successful stages
remain in V7. They retain their original execution/runtime provenance, not new
V8 executions. Earlier successful baselines used upstream seed-only runtime;
TRACE and newly executed V8 arms use strict deterministic execution. Model,
loss and selected hyperparameters are unchanged; this runtime extension is
declared, not called an unchanged-upstream-runtime reproduction.

PC TRACE calibration overlaps corrected validation, so PC is a development
comparison rather than untouched test evidence. Object NoAdapt and MLMP exceed
paper by1.23pp and1.58pp respectively; full Object paper parity remains open.
No parameter is selected to force agreement. Historical exposure is retained.
