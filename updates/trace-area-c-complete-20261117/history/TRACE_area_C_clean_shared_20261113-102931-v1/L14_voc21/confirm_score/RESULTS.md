# TRACE shared-backbone training calibration

L14_voc21 / confirm

Parameters selected on L14 only. MLMP fixed. B16 cannot select a different configuration.

| Configuration | Batch | Mean mIoU | Delta MLMP pp | Valid |
|---|---:|---:|---:|---|
| cfg_3b143b0da8aaef9a | 2 | 48.2610597030087 | -0.07611120509111657 | True |
| cfg_e15e8475d3b418d3 | 2 | 48.85295712320525 | 0.515786215105436 | True |
| cfg_3b3842fd592eed6e | 2 | 48.767751141880204 | 0.4305802337803897 | True |
| cfg_001259de116249aa | 2 | 48.3711690887672 | 0.03399818066738902 | True |

Only8groups in refinement/4in confirmation or transfer. Exploratory unadjusted intervals; orders are not independent images. L14-selected maxima are not test results.
