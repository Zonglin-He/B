# TRACE shared-backbone training calibration

L14_voc20 / confirm

Parameters selected on L14 only. MLMP fixed. B16 cannot select a different configuration.

| Configuration | Batch | Mean mIoU | Delta MLMP pp | Valid |
|---|---:|---:|---:|---|
| cfg_3b143b0da8aaef9a | 2 | 82.96606658741375 | 0.31995619490024296 | True |
| cfg_ef3f61de2cf8b1e3 | 1 | 85.95587695810211 | 3.3097665655886033 | True |
| cfg_36ff8ac84572199f | 1 | 85.20854524086076 | 2.5624348483472517 | True |
| cfg_a3010ff784144cd6 | 1 | 84.57613427976192 | 1.9300238872484101 | True |

Only8groups in refinement/4in confirmation or transfer. Exploratory unadjusted intervals; orders are not independent images. L14-selected maxima are not test results.
