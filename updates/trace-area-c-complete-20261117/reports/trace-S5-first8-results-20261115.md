# First Eight Complete S5 Conditions

This is a partial report, not the15-type macro. Each condition has1449 images,
three shuffled repeats and three arms. The first four conditions were verified
in progress331. In progress332, the next four conditions were independently
verified: same-current-invocation retained prediction AND score success, all
output hashes, and36 exact trial points recomputed from compact I/U without
reopening raw GT. All68 frozen local source hashes remain unchanged.

| S5 type | NoAdapt | MLMP | TRACE area-C | TRACE minus MLMP pp |
|---|---:|---:|---:|---:|
| Gaussian noise | 62.88 | 71.1067 | 73.13 | +2.0233 |
| Shot noise | 66.25 | 74.9633 | 77.19 | +2.2267 |
| Impulse noise | 63.14 | 71.2133 | 72.83 | +1.6167 |
| Defocus blur | 72.60 | 80.4167 | 80.69 | +0.2733 |
| Glass blur | 71.45 | 78.7233 | 80.18 | +1.4567 |
| Motion blur | 73.10 | 81.2733 | 82.77 | +1.4967 |
| Zoom blur | 59.02 | 69.4333 | 71.94 | +2.5067 |
| Snow | 71.50 | 79.4167 | 82.00 | +2.5833 |

Means follow the official rounded trial scores. Three-repeat std is not a paired
confidence interval. The frozen matching clean VOC20 configuration is reused
without S5-specific tuning or favorable trial selection. Seven types remain;
do not present this subset mean as the full15-type arithmetic macro. B16 negative
results, historical PC calibration exposure, Object paper discrepancy and
declared runtime provenance remain unchanged and must accompany final claims.

At06:51:48CST, frost area-C was running trial0of3,768of1449 images, GT closed.
The original single Luna/max monitor remains read-only with native alerts only.
No automation, cron, publication or manuscript was changed.
