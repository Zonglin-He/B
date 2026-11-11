# Original15 S5: Frozen TRACE-C vs Native MLMP

| Corruption | TRACE-C | MLMP | TRACE - MLMP (pp) | Paired95 CI |
|---|---:|---:|---:|---|
| gaussian_noise | 68.299 | 70.875 | -2.576 | [-4.243, -0.970] |
| shot_noise | 72.977 | 75.213 | -2.236 | [-3.671, -0.822] |
| impulse_noise | 68.513 | 71.249 | -2.735 | [-4.328, -1.148] |
| defocus_blur | 77.325 | 80.467 | -3.142 | [-4.693, -1.742] |
| glass_blur | 77.242 | 78.532 | -1.290 | [-2.956, 0.357] |
| motion_blur | 79.701 | 81.370 | -1.670 | [-2.722, -0.677] |
| zoom_blur | 66.859 | 69.366 | -2.507 | [-4.277, -0.769] |
| snow | 77.845 | 79.473 | -1.628 | [-2.848, -0.479] |
| frost | 72.006 | 73.357 | -1.351 | [-2.532, -0.184] |
| fog | 78.240 | 79.813 | -1.574 | [-2.706, -0.465] |
| brightness | 82.157 | 83.404 | -1.247 | [-2.365, -0.205] |
| contrast | 75.503 | 79.127 | -3.625 | [-4.867, -2.374] |
| elastic_transform | 69.542 | 73.994 | -4.452 | [-5.984, -2.955] |
| pixelate | 83.586 | 84.887 | -1.301 | [-2.386, -0.269] |
| jpeg_compression | 80.011 | 82.201 | -2.190 | [-3.318, -1.100] |

Arithmetic15-type macro: TRACE-C 75.320; MLMP 77.555; delta -2.235 pp.

10000 shared whole-original-episode draws; undefined retained; pointwise95, no multiplicity correction. Historical exposure and unequal native-system budgets/readouts remain. All three actual processes exited0;51 score output hashes were verified; no FAILED marker exists.

Macro paired95 CI: [-2.774034, -1.719761]pp;10000 common whole-episode draws across15types/arms;undefined0.

All15 TRACE-C point estimates are lower than MLMP. Fourteen individual pointwise95 intervals lie below0; glass_blur crosses0. These are not multiplicity-adjusted tests or independent replications.

The native MLMP own-NoAdapt macro is70.920504%; this is not a shared-readout adaptation-only control. TRACE-C macro75.320331%,MLMP77.555232%.

## Scope And Previous Results

All24 originally specified fixed native MLMP conditions now have scores:6L14clean,3B16clean,15L14S5. This does not complete sixTENT cells, equal-calibration-budget comparisons,R02,R04model effects or newR05 runs.

[Earlier completed results and all historical archives](../trace-results-20261111/README.md) remain unchanged. The earlier noon snapshot correctly recorded S5 as running at that time; this distinct release supplies the completed outcome.

Historical selection/development exposure, unequal native budgets/readouts, R1PARTIAL, negative/null/undefined outcomes remain. No new paper or favorable-configuration selection. No rawRGB/GT/weights/prediction arrays/per-image countstreams/credentials are exported.
