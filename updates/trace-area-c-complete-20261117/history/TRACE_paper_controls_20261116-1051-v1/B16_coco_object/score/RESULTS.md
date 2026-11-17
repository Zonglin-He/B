# Current-Method Ablations: B16_coco_object

Evaluation-disjoint training diagnostic, not another full-test benchmark. All controls retained.

| Control | Mean mIoU | Arrangement std |
|---|---:|---:|
| MLMP_paper | 26.746907 | 0.059567 |
| area_default | 26.786406 | 0.052856 |
| area_selected | 26.876163 | 0.000000 |
| no_region | 26.745807 | 0.000000 |
| no_global | 24.358105 | 0.000000 |
| no_cls | 26.554645 | 0.000000 |
| no_region_no_cls | 26.503050 | 0.000000 |
| region_balanced_mass | 26.357385 | 0.000000 |
| uniform_covered_mass | 26.677313 | 0.000000 |
| uniform_image_mass | 27.060205 | 0.000000 |
| spatial_shuffle_mass | 27.095502 | 0.000000 |
| semantic_top8 | 26.812053 | 0.000000 |
| top4 | 26.897920 | 0.000000 |
| top16 | 26.965113 | 0.000000 |
| score_Uniform | 26.759582 | 0.000000 |
| score_Confidence | 26.750899 | 0.000000 |
| score_C | 26.745971 | 0.000000 |
| score_RelativeC | 26.806361 | 0.000000 |
| score_ResponseChange | 26.850275 | 0.000000 |
| score_CalibratedConfidence | 26.963677 | 0.000000 |
| score_RegionPLPD | 26.753120 | 0.000000 |

| Contrast | Difference pp | Shared whole-block pointwise95 |
|---|---:|---|
| area_default_minus_MLMP_paper | +0.039499 | [-0.0953, 0.1739] |
| area_selected_minus_MLMP_paper | +0.129256 | [-0.5579, 1.8138] |
| area_selected_minus_area_default | +0.089758 | [-0.5963, 1.7773] |
| area_selected_minus_no_region | +0.130357 | [-0.6180, 0.8587] |
| area_selected_minus_no_global | +2.518058 | [1.1483, 3.9932] |
| area_selected_minus_no_cls | +0.321518 | [-1.0552, 1.7879] |
| area_selected_minus_no_region_no_cls | +0.373114 | [-0.9790, 1.7306] |
| area_selected_minus_region_balanced_mass | +0.518778 | [-0.1993, 0.9208] |
| area_selected_minus_uniform_covered_mass | +0.198851 | [-0.2591, 0.5015] |
| area_selected_minus_uniform_image_mass | -0.184041 | [-0.9383, 0.5547] |
| area_selected_minus_spatial_shuffle_mass | -0.219338 | [-0.9541, 0.4335] |
| area_selected_minus_semantic_top8 | +0.064111 | [-0.3416, 0.4475] |
| area_selected_minus_top4 | -0.021756 | [-0.8077, 0.3160] |
| area_selected_minus_top16 | -0.088949 | [-0.5536, 0.2830] |
| area_selected_minus_score_Uniform | +0.116581 | [-0.3259, 0.3954] |
| area_selected_minus_score_Confidence | +0.125264 | [-0.5770, 0.4335] |
| area_selected_minus_score_C | +0.130193 | [0.0033, 0.1618] |
| area_selected_minus_score_RelativeC | +0.069802 | [-0.3191, 0.3819] |
| area_selected_minus_score_ResponseChange | +0.025888 | [-0.3725, 0.2646] |
| area_selected_minus_score_CalibratedConfidence | -0.087513 | [-0.6917, 0.5183] |
| area_selected_minus_score_RegionPLPD | +0.123044 | [-0.5937, 0.4245] |
| no_cls_minus_no_region_no_cls | +0.051596 | [-0.4935, 0.6439] |
| no_region_minus_no_region_no_cls | +0.242757 | [-1.1621, 1.5545] |