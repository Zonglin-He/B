# Native-Engine TRACE Revision Training Results

32 VOC training sources; clean + original15 corruption types at S5. This is not independent validation.
Region C is a matched-native-engine control, not the historical TRACE implementation.

| Arm | Clean mIoU | Macro15 mIoU | Macro delta vs MLMP (95% paired CI) |
|---|---:|---:|---:|
| native_NoAdapt | 59.1928 | 50.0511 | -- |
| MLMP | 63.9395 | 57.9451 | -- |
| region_C | 62.2144 | 54.0952 | -3.8499 [-6.8388, -1.0359] |
| restore_global | 64.3486 | 57.4996 | -0.4456 [-2.6803, +1.3833] |
| area_C | 63.7954 | 58.2182 | +0.2731 [-0.6047, +0.7008] |
| absolute_budget | 61.6535 | 54.6882 | -3.2570 [-5.9414, -1.7236] |
| refresh_all | 62.3356 | 54.9130 | -3.0322 [-5.9036, -1.5878] |
| coverage_native_budget | 64.2557 | 57.9918 | +0.0467 [-0.4967, +0.7792] |
| coverage_relative_budget | 63.3016 | 56.8769 | -1.0682 [-2.8259, -0.2698] |

## Component Contrasts

| Contrast | Clean delta | Macro15 delta (95% paired CI) |
|---|---:|---:|
| restore_global - region_C | +2.1342 [-2.2893, +3.7897] | +3.4043 [+1.9604, +4.9422] |
| area_C - restore_global | -0.5531 [-2.5177, +4.8944] | +0.7187 [-1.2584, +2.7237] |
| absolute_budget - area_C | -2.1419 [-6.9263, -0.7776] | -3.5301 [-5.8023, -1.9323] |
| refresh_all - absolute_budget | +0.6821 [-3.2214, +1.2122] | +0.2248 [-0.8756, +0.7454] |
| coverage_native_budget - refresh_all | +1.9201 [+0.7379, +5.7724] | +3.0789 [+1.6573, +6.0132] |
| coverage_native_budget - area_C | +0.4603 [-3.3636, +0.6494] | -0.2264 [-0.4919, +0.5983] |
| coverage_relative_budget - coverage_native_budget | -0.9541 [-4.4333, -0.3109] | -1.1149 [-2.8448, -0.3965] |

## All Conditions

| Condition | native_NoAdapt | MLMP | region_C | restore_global | area_C | absolute_budget | refresh_all | coverage_native_budget | coverage_relative_budget |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 59.1928 | 63.9395 | 62.2144 | 64.3486 | 63.7954 | 61.6535 | 62.3356 | 64.2557 | 63.3016 |
| gaussian_noise | 49.5693 | 53.4075 | 55.2340 | 55.2460 | 54.0937 | 53.4877 | 52.9750 | 54.0709 | 53.6056 |
| shot_noise | 51.8540 | 57.2750 | 59.6029 | 59.3228 | 57.7627 | 57.2320 | 56.9338 | 57.4569 | 57.3147 |
| impulse_noise | 47.6557 | 52.3047 | 52.2740 | 53.3370 | 53.3584 | 50.4991 | 50.0112 | 53.2015 | 50.9614 |
| defocus_blur | 49.6988 | 53.0061 | 53.7507 | 51.3553 | 53.4450 | 50.3222 | 53.2719 | 53.2385 | 55.2137 |
| glass_blur | 48.2744 | 53.8742 | 51.2070 | 52.5430 | 53.9419 | 51.7199 | 52.1831 | 53.8331 | 53.2137 |
| motion_blur | 46.3752 | 55.7261 | 51.7518 | 53.6391 | 56.3635 | 50.7473 | 53.0633 | 56.3332 | 54.8518 |
| zoom_blur | 41.5970 | 61.3119 | 50.5641 | 61.5808 | 61.1891 | 52.4769 | 52.8044 | 58.0329 | 60.0994 |
| snow | 45.2815 | 52.6515 | 50.5142 | 51.5863 | 52.8542 | 50.8870 | 48.5731 | 52.9981 | 50.4338 |
| frost | 44.2890 | 54.8761 | 48.7351 | 54.8875 | 55.4630 | 50.8235 | 51.2028 | 55.4576 | 53.3310 |
| fog | 54.1286 | 58.0074 | 51.1987 | 54.9142 | 57.7103 | 56.2314 | 56.7442 | 57.6170 | 57.5250 |
| brightness | 59.9094 | 63.4162 | 66.0128 | 67.7918 | 63.7057 | 61.9315 | 61.0507 | 63.7660 | 62.2834 |
| contrast | 51.1763 | 59.6475 | 54.6724 | 58.8243 | 60.0399 | 54.4187 | 53.9999 | 60.2177 | 55.0288 |
| elastic_transform | 50.4120 | 65.4455 | 54.4461 | 62.2588 | 65.0730 | 56.9881 | 58.7275 | 65.4469 | 63.8113 |
| pixelate | 59.8300 | 67.6595 | 57.9429 | 65.8532 | 67.8436 | 65.0023 | 65.2693 | 67.8239 | 66.7244 |
| jpeg_compression | 50.7158 | 60.5680 | 53.5222 | 59.3534 | 60.4295 | 57.5548 | 56.8843 | 60.3833 | 58.7558 |

All intervals use10000 common whole-B2-episode draws. Fifteen-type macro is arithmetic, not pixel pooling.
Only16 source groups; pointwise intervals, no multiplicity correction or global never-exposed claim.
Absolute/relative C are uncalibrated reliability heuristics, not measured probabilities of beneficial updates.
Diagnostics and timings for every condition/arm, including negative and null results, are in RESULT.json.
