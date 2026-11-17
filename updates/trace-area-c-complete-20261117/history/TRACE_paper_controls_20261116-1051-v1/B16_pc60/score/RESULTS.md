# Current-Method Ablations: B16_pc60

Evaluation-disjoint training diagnostic, not another full-test benchmark. All controls retained.

| Control | Mean mIoU | Arrangement std |
|---|---:|---:|
| MLMP_paper | 27.475230 | 0.044406 |
| area_default | 27.375091 | 0.017960 |
| area_selected | 26.506104 | 0.230175 |
| no_region | 26.652990 | 0.175340 |
| no_global | 25.588934 | 0.157455 |
| no_cls | 26.151422 | 0.026595 |
| no_region_no_cls | 26.376097 | 0.014446 |
| region_balanced_mass | 26.651394 | 0.034293 |
| uniform_covered_mass | 26.742688 | 0.088130 |
| uniform_image_mass | 26.864911 | 0.081644 |
| spatial_shuffle_mass | 26.742172 | 0.126467 |
| semantic_top8 | 26.506104 | 0.230175 |
| top4 | 26.597555 | 0.160403 |
| top16 | 26.628382 | 0.069526 |
| score_Uniform | 26.715071 | 0.087161 |
| score_Confidence | 26.766691 | 0.120094 |
| score_C | 26.521520 | 0.175622 |
| score_RelativeC | 26.543117 | 0.199129 |
| score_ResponseChange | 26.700945 | 0.140230 |
| score_CalibratedConfidence | 26.668035 | 0.042195 |
| score_RegionPLPD | 26.610875 | 0.045634 |

| Contrast | Difference pp | Shared whole-block pointwise95 |
|---|---:|---|
| area_default_minus_MLMP_paper | -0.100139 | [-0.1915, -0.0030] |
| area_selected_minus_MLMP_paper | -0.969126 | [-1.9034, -0.2142] |
| area_selected_minus_area_default | -0.868987 | [-1.8372, -0.1169] |
| area_selected_minus_no_region | -0.146887 | [-0.3560, 0.0478] |
| area_selected_minus_no_global | +0.917170 | [0.2993, 1.4610] |
| area_selected_minus_no_cls | +0.354682 | [-0.6644, 0.8837] |
| area_selected_minus_no_region_no_cls | +0.130007 | [-0.8992, 0.6908] |
| area_selected_minus_region_balanced_mass | -0.145290 | [-0.3420, 0.0284] |
| area_selected_minus_uniform_covered_mass | -0.236584 | [-0.4439, -0.0475] |
| area_selected_minus_uniform_image_mass | -0.358807 | [-0.5738, -0.1094] |
| area_selected_minus_spatial_shuffle_mass | -0.236068 | [-0.4960, 0.0373] |
| area_selected_minus_semantic_top8 | +0.000000 | [0.0000, 0.0000] |
| area_selected_minus_top4 | -0.091451 | [-0.2211, 0.0455] |
| area_selected_minus_top16 | -0.122279 | [-0.3675, 0.0686] |
| area_selected_minus_score_Uniform | -0.208968 | [-0.3812, -0.0341] |
| area_selected_minus_score_Confidence | -0.260588 | [-0.4399, -0.0774] |
| area_selected_minus_score_C | -0.015416 | [-0.0787, 0.0192] |
| area_selected_minus_score_RelativeC | -0.037013 | [-0.1635, 0.0753] |
| area_selected_minus_score_ResponseChange | -0.194841 | [-0.3537, -0.0303] |
| area_selected_minus_score_CalibratedConfidence | -0.161931 | [-0.3678, 0.0619] |
| area_selected_minus_score_RegionPLPD | -0.104771 | [-0.2778, 0.0582] |
| no_cls_minus_no_region_no_cls | -0.224675 | [-0.4400, 0.0363] |
| no_region_minus_no_region_no_cls | +0.276893 | [-0.7185, 0.8055] |