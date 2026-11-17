# TRACE shared-backbone training calibration

L14_coco_stuff / confirm

Parameters selected on L14 only. MLMP fixed. B16 cannot select a different configuration.

| Configuration | Batch | Mean mIoU | Delta MLMP pp | Valid |
|---|---:|---:|---:|---|
| cfg_3b143b0da8aaef9a | 2 | 18.01716294675434 | -0.00902183011800517 | True |
| cfg_587a570ea35eebeb | 8 | 19.00878424819828 | 0.9825994713259334 | True |
| cfg_3672814d3e9cbc94 | 4 | 18.985713858962534 | 0.9595290820901887 | True |
| cfg_6fa51c33267a69b3 | 16 | 18.738380126925808 | 0.712195350053463 | True |

Only8groups in refinement/4in confirmation or transfer. Exploratory unadjusted intervals; orders are not independent images. L14-selected maxima are not test results.
