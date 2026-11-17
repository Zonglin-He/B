# Current-Method Ablations: L14_pc60

Evaluation-disjoint training diagnostic, not another full-test benchmark. All controls retained.

| Control | Mean mIoU | Arrangement std |
|---|---:|---:|
| MLMP_paper | 27.366621 | 0.049425 |
| area_default | 27.358463 | 0.052716 |
| area_selected | 27.481358 | 0.057514 |
| no_region | 27.514792 | 0.091395 |
| no_global | 24.903174 | 0.032309 |
| no_cls | 27.331583 | 0.200569 |
| no_region_no_cls | 27.442779 | 0.248531 |
| region_balanced_mass | 27.653707 | 0.037083 |
| uniform_covered_mass | 27.434106 | 0.029081 |
| uniform_image_mass | 27.537250 | 0.134483 |
| spatial_shuffle_mass | 27.609752 | 0.135034 |
| semantic_top8 | 27.481358 | 0.057514 |
| top4 | 27.471186 | 0.074613 |
| top16 | 27.334009 | 0.046074 |
| score_Uniform | 27.410496 | 0.024712 |
| score_Confidence | 27.431964 | 0.070498 |
| score_C | 27.475982 | 0.040579 |
| score_RelativeC | 27.441986 | 0.113593 |
| score_ResponseChange | 27.455593 | 0.069456 |
| score_CalibratedConfidence | 27.502857 | 0.088397 |
| score_RegionPLPD | 27.538278 | 0.061107 |

| Contrast | Difference pp | Shared whole-block pointwise95 |
|---|---:|---|
| area_default_minus_MLMP_paper | -0.008158 | [-0.0523, 0.0757] |
| area_selected_minus_MLMP_paper | +0.114737 | [-0.4047, 0.8432] |
| area_selected_minus_area_default | +0.122895 | [-0.4120, 0.8191] |
| area_selected_minus_no_region | -0.033433 | [-0.2138, 0.2262] |
| area_selected_minus_no_global | +2.578185 | [1.7579, 3.6910] |
| area_selected_minus_no_cls | +0.149776 | [-0.6880, 0.8265] |
| area_selected_minus_no_region_no_cls | +0.038579 | [-0.7561, 0.7156] |
| area_selected_minus_region_balanced_mass | -0.172349 | [-0.3276, 0.0465] |
| area_selected_minus_uniform_covered_mass | +0.047252 | [-0.1274, 0.2071] |
| area_selected_minus_uniform_image_mass | -0.055891 | [-0.3239, 0.2418] |
| area_selected_minus_spatial_shuffle_mass | -0.128394 | [-0.3024, 0.1171] |
| area_selected_minus_semantic_top8 | +0.000000 | [0.0000, 0.0000] |
| area_selected_minus_top4 | +0.010172 | [-0.1394, 0.1453] |
| area_selected_minus_top16 | +0.147349 | [-0.0240, 0.3034] |
| area_selected_minus_score_Uniform | +0.070862 | [-0.1008, 0.2143] |
| area_selected_minus_score_Confidence | +0.049395 | [-0.1391, 0.1658] |
| area_selected_minus_score_C | +0.005376 | [-0.0136, 0.0346] |
| area_selected_minus_score_RelativeC | +0.039372 | [-0.1000, 0.1395] |
| area_selected_minus_score_ResponseChange | +0.025766 | [-0.1582, 0.1773] |
| area_selected_minus_score_CalibratedConfidence | -0.021499 | [-0.3030, 0.2318] |
| area_selected_minus_score_RegionPLPD | -0.056920 | [-0.2654, 0.0909] |
| no_cls_minus_no_region_no_cls | -0.111196 | [-0.3372, 0.1787] |
| no_region_minus_no_region_no_cls | +0.072013 | [-0.7057, 0.6998] |