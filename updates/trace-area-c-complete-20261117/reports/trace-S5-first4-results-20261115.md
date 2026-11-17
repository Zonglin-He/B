# First Four Complete S5 Conditions

This is a partial report, not the15-type macro. Every condition has1449 images
and all three arms completed three shuffled repeats. Main independently verified
same-current-invocation retained prediction AND score success, all output hashes,
and36 exact trial points from compact I/U without reopening raw GT.

| S5 type | NoAdapt | MLMP | TRACE area-C | TRACE minus MLMP pp |
|---|---:|---:|---:|---:|
| Gaussian noise | 62.88 | 71.1067 | 73.13 | +2.0233 |
| Shot noise | 66.25 | 74.9633 | 77.19 | +2.2267 |
| Impulse noise | 63.14 | 71.2133 | 72.83 | +1.6167 |
| Defocus blur | 72.60 | 80.4167 | 80.69 | +0.2733 |

Means use the official rounded trial scores. Three-repeat std is not a paired
confidence interval. The frozen matching clean VOC20 configuration is reused;
there is no S5-specific tuning or favorable trial selection. The complete15-type
arithmetic macro waits for the other11 conditions. Mixed B16 outcomes, historical
exposure, runtime extensions and other source-boundary caveats remain retained.
