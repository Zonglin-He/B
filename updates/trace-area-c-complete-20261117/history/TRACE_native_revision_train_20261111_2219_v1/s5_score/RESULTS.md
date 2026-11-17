# TRACE Revision Training Pilot

Training-only, fixed32 sources. Not a full-benchmark superiority result.

| Condition | NoAdapt | MLMP | Region C | +Global | +Area | +Budget | +Refresh/all |
|---|---:|---:|---:|---:|---:|---:|---:|
| gaussian_noise | 49.5693 | 53.4075 | 55.2340 | 55.2460 | 54.0937 | 53.4877 | 52.9750 |
| shot_noise | 51.8540 | 57.2750 | 59.6029 | 59.3228 | 57.7627 | 57.2320 | 56.9338 |
| impulse_noise | 47.6557 | 52.3047 | 52.2740 | 53.3370 | 53.3584 | 50.4991 | 50.0112 |
| defocus_blur | 49.6988 | 53.0061 | 53.7507 | 51.3553 | 53.4450 | 50.3222 | 53.2719 |
| glass_blur | 48.2744 | 53.8742 | 51.2070 | 52.5430 | 53.9419 | 51.7199 | 52.1831 |
| motion_blur | 46.3752 | 55.7261 | 51.7518 | 53.6391 | 56.3635 | 50.7473 | 53.0633 |
| zoom_blur | 41.5970 | 61.3119 | 50.5641 | 61.5808 | 61.1891 | 52.4769 | 52.8044 |
| snow | 45.2815 | 52.6515 | 50.5142 | 51.5863 | 52.8542 | 50.8870 | 48.5731 |
| frost | 44.2890 | 54.8761 | 48.7351 | 54.8875 | 55.4630 | 50.8235 | 51.2028 |
| fog | 54.1286 | 58.0074 | 51.1987 | 54.9142 | 57.7103 | 56.2314 | 56.7442 |
| brightness | 59.9094 | 63.4162 | 66.0128 | 67.7918 | 63.7057 | 61.9315 | 61.0507 |
| contrast | 51.1763 | 59.6475 | 54.6724 | 58.8243 | 60.0399 | 54.4187 | 53.9999 |
| elastic_transform | 50.4120 | 65.4455 | 54.4461 | 62.2588 | 65.0730 | 56.9881 | 58.7275 |
| pixelate | 59.8300 | 67.6595 | 57.9429 | 65.8532 | 67.8436 | 65.0023 | 65.2693 |
| jpeg_compression | 50.7158 | 60.5680 | 53.5222 | 59.3534 | 60.4295 | 57.5548 | 56.8843 |
| Macro15 | 50.0511 | 57.9451 | 54.0952 | 57.4996 | 58.2182 | 54.6882 | 54.9130 |

Same engine/readout/base LR/step count. Diagnostics and gates change actual compute and update sizes.
Region C is a matched-engine control, NOT a rerun of the historical TRACE system.
All positive/negative/null effects and class coverage are in RESULT.json. No best-test-trial selection.
