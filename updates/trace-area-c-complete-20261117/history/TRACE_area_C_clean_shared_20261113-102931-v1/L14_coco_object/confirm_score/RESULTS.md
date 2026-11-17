# TRACE shared-backbone training calibration

L14_coco_object / confirm

Parameters selected on L14 only. MLMP fixed. B16 cannot select a different configuration.

| Configuration | Batch | Mean mIoU | Delta MLMP pp | Valid |
|---|---:|---:|---:|---|
| cfg_3b143b0da8aaef9a | 2 | 27.716243578103256 | -0.020715373773143142 | True |
| cfg_9dd06d908181de85 | 4 | 30.084436866532595 | 2.3474779146561957 | True |
| cfg_8cd41449f75dc28a | 2 | 29.844446351038133 | 2.1074873991617338 | True |
| cfg_91cdec5165f04c35 | 1 | 30.549151889572396 | 2.812192937695997 | True |

Only8groups in refinement/4in confirmation or transfer. Exploratory unadjusted intervals; orders are not independent images. L14-selected maxima are not test results.
