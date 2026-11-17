# TRACE shared-backbone training calibration

L14_coco_cd / confirm

Parameters selected on L14 only. MLMP fixed. B16 cannot select a different configuration.

| Configuration | Batch | Mean mIoU | Delta MLMP pp | Valid |
|---|---:|---:|---:|---|
| cfg_3b143b0da8aaef9a | 2 | 71.11812101841804 | 1.4904627678582614 | True |
| cfg_c4396ee2d559724f | 8 | 74.34044039175575 | 4.712782141195973 | True |
| cfg_349646f65ec4a2fd | 8 | 74.28311535735206 | 4.655457106792284 | True |
| cfg_86a4d5fb7e8c107c | 8 | 73.64139828840723 | 4.013740037847455 | True |

Only8groups in refinement/4in confirmation or transfer. Exploratory unadjusted intervals; orders are not independent images. L14-selected maxima are not test results.
