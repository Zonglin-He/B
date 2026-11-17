# Eight Complete L14 Clean Comparisons

All24 arms below completed full prediction and separate CPU scoring, with three
shuffled repeats per arm. Each reported value is the mean of the official
two-decimal trial scores, not a confidence interval or best trial. All new
Waterbirds/COCO-CD outputs and exact points were verified against actual retained
prediction AND score success and compact source I/U, without reopening raw GT.
Earlier completed stages have the explicit durable preboot recovery provenance
described in progress328, not manufactured current systemd success.

| Dataset | Images | NoAdapt | MLMP | TRACE area-C | TRACE minus MLMP pp |
|---|---:|---:|---:|---:|---:|
| PC60 | 5104 | 24.95 | 27.9833 | 28.7133 | +0.7300 |
| COCO Object | 5000 | 25.03 | 30.4200 | 32.4400 | +2.0200 |
| VOC20 | 1449 | 75.91 | 83.8033 | 86.1200 | +2.3167 |
| VOC21 | 1449 | 45.13 | 50.7600 | 51.1433 | +0.3833 |
| PC59 | 5104 | 28.23 | 31.9433 | 32.8767 | +0.9333 |
| COCO Stuff | 5000 | 18.34 | 21.2500 | 21.9933 | +0.7433 |
| Waterbirds segmentation | 2568 | 45.37 | 49.9033 | 54.0267 | +4.1233 |
| COCO cat/dog | 500 | 62.06 | 72.3967 | 77.3833 | +4.9867 |

Waterbirds MLMP trial std0.02055,TRACE0.04922; COCO-CD MLMP0.07717,TRACE0.05907.
No uncertainty interval is inferred by subtracting independent margins. No new
parameter search, test selection or favorable trial picking is performed.

## Provenance and Limits

Waterbirds and COCO-CD are declared extensions, not MLMP paper datasets. Their
null identity query maps were represented explicitly; classes/masks/geometry
and selected parameters did not change. Their score evidence is under
`TRACE_official_baselines_20261114-2022-v10/L14_waterbirds_seg` and `L14_coco_cd`.

PC TRACE calibration overlaps the corrected reference validation membership:
these are development comparisons, not untouched held-out PC evidence. Object
NoAdapt/MLMP remain above paper by1.23/1.58pp; full Object paper alignment is open.
Historical testing/selection exposure remains disclosed. Earlier successful
baselines used upstream seed-only runtime, while TRACE and newly executed arms
use the declared strict deterministic runtime; original runtime origins stay.
These results do not establish universal SOTA, causal mechanism or equal tuning
budget against the fixed-paper MLMP baseline.

The complete26-condition suite is not finished. B16 Object NoAdapt29.59 and
MLMP30.5267 are verified; its TRACE arm is running. Remaining B16 conditions and
the15S5 arithmetic macro must wait for complete sealed predictions and scores.
