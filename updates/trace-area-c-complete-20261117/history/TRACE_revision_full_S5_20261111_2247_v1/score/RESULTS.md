# Fixed Two-Arm Full S5 Revision

Historically exposed validation; both training-motivated arms reported, no winner selection.

| Condition | native_NoAdapt | MLMP | historical_TRACE_C | area_C | coverage_native_budget |
|---|---:|---:|---:|---:|---:|
| gaussian_noise | 64.3730 | 70.8749 | 68.2988 | 71.1898 | 71.2870 |
| shot_noise | 67.5815 | 75.2133 | 72.9771 | 75.4727 | 75.4459 |
| impulse_noise | 64.3957 | 71.2490 | 68.5135 | 71.4684 | 71.3891 |
| defocus_blur | 75.0193 | 80.4669 | 77.3251 | 80.6345 | 80.6553 |
| glass_blur | 73.5617 | 78.5323 | 77.2419 | 78.8958 | 78.6394 |
| motion_blur | 75.7328 | 81.3703 | 79.7006 | 81.6454 | 81.6264 |
| zoom_blur | 60.0808 | 69.3656 | 66.8585 | 69.6926 | 69.7176 |
| snow | 72.7136 | 79.4730 | 77.8452 | 79.9073 | 79.9592 |
| frost | 66.4063 | 73.3574 | 72.0065 | 73.6954 | 73.7672 |
| fog | 73.2013 | 79.8134 | 78.2396 | 80.2451 | 80.2835 |
| brightness | 77.2994 | 83.4038 | 82.1570 | 83.8603 | 83.9050 |
| contrast | 73.4776 | 79.1272 | 75.5025 | 79.3177 | 79.3680 |
| elastic_transform | 64.4092 | 73.9941 | 69.5416 | 74.2877 | 74.3600 |
| pixelate | 80.1464 | 84.8868 | 83.5860 | 85.2472 | 85.2989 |
| jpeg_compression | 75.4090 | 82.2007 | 80.0109 | 82.6196 | 82.7301 |
| Macro15 | 70.9205 | 77.5552 | 75.3203 | 77.8786 | 77.8955 |

10000 common whole-B2-episode draws; arithmetic15-type macro, not pooled pixels. All paired pointwise intervals and undefined counts in RESULT.json; no multiplicity adjustment.
