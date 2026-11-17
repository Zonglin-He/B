# Alignment Training Results And Verification

Training32 repeatedly used; old area_C cost adds measured shared C acquisition. Different runs, no interleaved timing or equal-compute claim. Scalar loss ratios are not gradient importance. No GT opened here.

| Arm | Clean mIoU | S5 macro mIoU | Recorded B2 seconds |
|---|---:|---:|---:|
| MLMP | 63.9395 | 57.9451 | not recomputed |
| area_C | 63.7954 | 58.2182 | 0.425586 |
| coverage_native_budget | 64.2557 | 57.9918 | not recomputed |
| aligned_query_entropy | 63.9837 | 58.1418 | 0.388444 |
| aligned_semantic_entropy | 64.3094 | 58.1772 | 0.404943 |
| aligned_semantic_anchor | 64.2881 | 58.0702 | 0.407531 |

## All Conditions

| Condition | MLMP | area_C | coverage_native_budget | aligned_query_entropy | aligned_semantic_entropy | aligned_semantic_anchor |
|---|---:|---:|---:|---:|---:|---:|
| clean | 63.9395 | 63.7954 | 64.2557 | 63.9837 | 64.3094 | 64.2881 |
| gaussian_noise | 53.4075 | 54.0937 | 54.0709 | 53.3993 | 53.6878 | 53.3712 |
| shot_noise | 57.2750 | 57.7627 | 57.4569 | 57.5933 | 57.5109 | 57.3343 |
| impulse_noise | 52.3047 | 53.3584 | 53.2015 | 53.0002 | 52.7790 | 53.0411 |
| defocus_blur | 53.0061 | 53.4450 | 53.2385 | 53.1211 | 53.2294 | 53.0286 |
| glass_blur | 53.8742 | 53.9419 | 53.8331 | 53.7977 | 54.0483 | 53.9830 |
| motion_blur | 55.7261 | 56.3635 | 56.3332 | 56.1857 | 56.2220 | 56.1624 |
| zoom_blur | 61.3119 | 61.1891 | 58.0329 | 61.3153 | 61.4062 | 61.2328 |
| snow | 52.6515 | 52.8542 | 52.9981 | 53.0231 | 53.0837 | 53.1293 |
| frost | 54.8761 | 55.4630 | 55.4576 | 55.3693 | 55.1919 | 54.9640 |
| fog | 58.0074 | 57.7103 | 57.6170 | 57.8296 | 57.5288 | 57.7303 |
| brightness | 63.4162 | 63.7057 | 63.7660 | 63.7553 | 63.9807 | 63.8082 |
| contrast | 59.6475 | 60.0399 | 60.2177 | 59.8458 | 59.7535 | 59.8105 |
| elastic_transform | 65.4455 | 65.0730 | 65.4469 | 65.0662 | 65.2940 | 65.2452 |
| pixelate | 67.6595 | 67.8436 | 67.8239 | 68.1434 | 68.1813 | 67.8963 |
| jpeg_compression | 60.5680 | 60.4295 | 60.3833 | 60.6816 | 60.7606 | 60.3159 |

## Direct Paired Contrasts

| Contrast | Clean delta | Clean 95% interval | S5 delta | S5 95% interval |
|---|---:|---|---:|---|
| aligned_query_entropy_minus_MLMP | 0.0442 | [-0.2650, 4.1000] | 0.1967 | [-0.3949, 0.8530] |
| aligned_query_entropy_minus_area_C | 0.1882 | [-0.3222, 0.5532] | -0.0764 | [-0.4841, 0.8514] |
| aligned_semantic_anchor_minus_MLMP | 0.3487 | [-0.0212, 4.4612] | 0.1251 | [-0.3999, 0.6537] |
| aligned_semantic_anchor_minus_aligned_semantic_entropy | -0.0213 | [-0.6175, 0.3600] | -0.1070 | [-0.5396, 0.5178] |
| aligned_semantic_anchor_minus_area_C | 0.4927 | [-0.3275, 0.9733] | -0.1480 | [-0.6430, 0.6707] |
| aligned_semantic_entropy_minus_MLMP | 0.3699 | [0.0416, 4.2362] | 0.2321 | [-0.4081, 0.5927] |
| aligned_semantic_entropy_minus_aligned_query_entropy | 0.3257 | [-0.0245, 0.5945] | 0.0354 | [-0.6900, 0.4308] |
| aligned_semantic_entropy_minus_area_C | 0.5140 | [-0.1699, 0.9746] | -0.0410 | [-0.5851, 0.6693] |

10000 shared whole16B2 draws, seed 2026111203; pointwise, no multiplicity adjustment or automatic winner.
All16 first4 checks are preserved in RESULT.json. All score hashes and condition/macro point estimates verified.
