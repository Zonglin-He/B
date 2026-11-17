# TRACE Development: Five New Variants

Two sequential training-development rounds, same32 repeatedly exposed sources, not independent confirmation. All five new outcomes retained. No full-validation experiment or automatic winner. Separate bootstrap seeds2026111203/2026111204 within rounds,10000shared16B2 draws,pointwise/no multiplicity correction.

Recorded model+diagnostics+updates+readout only; area_C adds shared C acquisition. Different runs, not interleaved or equal-time; excludes IO/startup/gates/failed attempt.

| Arm | Clean mIoU | S5 macro | S5 vs MLMP pp | S5 vs area_C pp | Recorded B2 seconds |
|---|---:|---:|---:|---:|---:|
| MLMP | 63.9395 | 57.9451 | +0.0000 | -0.2731 | not recomputed |
| area_C | 63.7954 | 58.2182 | +0.2731 | +0.0000 | 0.425586 |
| coverage_native_budget | 64.2557 | 57.9918 | +0.0467 | -0.2264 | not recomputed |
| aligned_query_entropy | 63.9837 | 58.1418 | +0.1967 | -0.0764 | 0.388444 |
| aligned_semantic_entropy | 64.3094 | 58.1772 | +0.2321 | -0.0410 | 0.404943 |
| aligned_semantic_anchor | 64.2881 | 58.0702 | +0.1251 | -0.1480 | 0.407531 |
| semantic_top8 | 63.7972 | 58.2342 | +0.2891 | +0.0160 | 0.408936 |
| semantic_mass_top8 | 64.4506 | 58.2396 | +0.2945 | +0.0214 | 0.408795 |

## Every Condition

| Condition | MLMP | area_C | coverage_native_budget | aligned_query_entropy | aligned_semantic_entropy | aligned_semantic_anchor | semantic_top8 | semantic_mass_top8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 63.9395 | 63.7954 | 64.2557 | 63.9837 | 64.3094 | 64.2881 | 63.7972 | 64.4506 |
| gaussian_noise | 53.4075 | 54.0937 | 54.0709 | 53.3993 | 53.6878 | 53.3712 | 54.0394 | 53.6294 |
| shot_noise | 57.2750 | 57.7627 | 57.4569 | 57.5933 | 57.5109 | 57.3343 | 57.7184 | 57.5027 |
| impulse_noise | 52.3047 | 53.3584 | 53.2015 | 53.0002 | 52.7790 | 53.0411 | 53.2859 | 53.2307 |
| defocus_blur | 53.0061 | 53.4450 | 53.2385 | 53.1211 | 53.2294 | 53.0286 | 53.6896 | 53.4127 |
| glass_blur | 53.8742 | 53.9419 | 53.8331 | 53.7977 | 54.0483 | 53.9830 | 54.1437 | 53.4364 |
| motion_blur | 55.7261 | 56.3635 | 56.3332 | 56.1857 | 56.2220 | 56.1624 | 56.2938 | 56.3092 |
| zoom_blur | 61.3119 | 61.1891 | 58.0329 | 61.3153 | 61.4062 | 61.2328 | 61.2931 | 61.1339 |
| snow | 52.6515 | 52.8542 | 52.9981 | 53.0231 | 53.0837 | 53.1293 | 52.7627 | 53.1524 |
| frost | 54.8761 | 55.4630 | 55.4576 | 55.3693 | 55.1919 | 54.9640 | 55.4447 | 55.7257 |
| fog | 58.0074 | 57.7103 | 57.6170 | 57.8296 | 57.5288 | 57.7303 | 57.5667 | 57.6909 |
| brightness | 63.4162 | 63.7057 | 63.7660 | 63.7553 | 63.9807 | 63.8082 | 63.8107 | 64.0759 |
| contrast | 59.6475 | 60.0399 | 60.2177 | 59.8458 | 59.7535 | 59.8105 | 60.2842 | 60.3105 |
| elastic_transform | 65.4455 | 65.0730 | 65.4469 | 65.0662 | 65.2940 | 65.2452 | 65.0546 | 65.5992 |
| pixelate | 67.6595 | 67.8436 | 67.8239 | 68.1434 | 68.1813 | 67.8963 | 67.8249 | 67.9842 |
| jpeg_compression | 60.5680 | 60.4295 | 60.3833 | 60.6816 | 60.7606 | 60.3159 | 60.3010 | 60.4004 |

## All Direct Contrasts

| Contrast | Clean delta pp | Clean95 | S5 delta pp | S5 95 |
|---|---:|---|---:|---|
| aligned_query_entropy_minus_MLMP | 0.0442 | [-0.2650, 4.1000] | 0.1967 | [-0.3949, 0.8530] |
| aligned_query_entropy_minus_area_C | 0.1882 | [-0.3222, 0.5532] | -0.0764 | [-0.4841, 0.8514] |
| aligned_semantic_anchor_minus_MLMP | 0.3487 | [-0.0212, 4.4612] | 0.1251 | [-0.3999, 0.6537] |
| aligned_semantic_anchor_minus_aligned_semantic_entropy | -0.0213 | [-0.6175, 0.3600] | -0.1070 | [-0.5396, 0.5178] |
| aligned_semantic_anchor_minus_area_C | 0.4927 | [-0.3275, 0.9733] | -0.1480 | [-0.6430, 0.6707] |
| aligned_semantic_entropy_minus_MLMP | 0.3699 | [0.0416, 4.2362] | 0.2321 | [-0.4081, 0.5927] |
| aligned_semantic_entropy_minus_aligned_query_entropy | 0.3257 | [-0.0245, 0.5945] | 0.0354 | [-0.6900, 0.4308] |
| aligned_semantic_entropy_minus_area_C | 0.5140 | [-0.1699, 0.9746] | -0.0410 | [-0.5851, 0.6693] |
| semantic_mass_top8_minus_MLMP | 0.5111 | [-0.0195, 4.5622] | 0.2945 | [-0.3776, 0.9634] |
| semantic_mass_top8_minus_area_C | 0.6552 | [0.0756, 0.8287] | 0.0214 | [-0.4660, 0.9725] |
| semantic_mass_top8_minus_semantic_top8 | 0.6534 | [-2.0761, 0.7982] | 0.0054 | [-0.5754, 0.7455] |
| semantic_top8_minus_MLMP | -0.1423 | [-0.6473, 4.2221] | 0.2891 | [-0.4371, 0.8265] |
| semantic_top8_minus_area_C | 0.0017 | [-0.1022, 2.1258] | 0.0160 | [-0.2443, 0.5145] |

All five new variants and null/negative results retained; no baseline relabeling or old-result overwrite.
Prediction first4 and terminal evidence, counts, diagnostics and undefined draw counts remain in RESULT.json and referenced source reports.
