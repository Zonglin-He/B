# Current-Method Ablations: B16_voc20

Evaluation-disjoint training diagnostic, not another full-test benchmark. All controls retained.

| Control | Mean mIoU | Arrangement std |
|---|---:|---:|
| MLMP_paper | 85.420907 | 0.101401 |
| area_default | 85.640675 | 0.072666 |
| area_selected | 85.171920 | 0.000000 |
| no_region | 84.340774 | 0.000000 |
| no_global | 84.348583 | 0.000000 |
| no_cls | 82.893075 | 0.000000 |
| no_region_no_cls | 82.300282 | 0.000000 |
| region_balanced_mass | 85.828176 | 0.000000 |
| uniform_covered_mass | 85.156754 | 0.000000 |
| uniform_image_mass | 84.990952 | 0.000000 |
| spatial_shuffle_mass | 84.404149 | 0.000000 |
| semantic_top8 | 85.162539 | 0.000000 |
| top4 | 85.058491 | 0.000000 |
| top16 | 85.132060 | 0.000000 |

| Contrast | Difference pp | Shared whole-block pointwise95 |
|---|---:|---|
| area_default_minus_MLMP_paper | +0.219768 | [-0.0801, 0.5888] |
| area_selected_minus_MLMP_paper | -0.248987 | [-2.2202, 1.1229] |
| area_selected_minus_area_default | -0.468755 | [-2.5858, 1.0097] |
| area_selected_minus_no_region | +0.831146 | [0.0071, 2.0273] |
| area_selected_minus_no_global | +0.823337 | [-0.5141, 2.2766] |
| area_selected_minus_no_cls | +2.278846 | [0.5272, 4.0236] |
| area_selected_minus_no_region_no_cls | +2.871638 | [0.8101, 4.9929] |
| area_selected_minus_region_balanced_mass | -0.656256 | [-2.5210, 0.4114] |
| area_selected_minus_uniform_covered_mass | +0.015166 | [-0.1537, 0.1674] |
| area_selected_minus_uniform_image_mass | +0.180968 | [-0.2425, 0.6285] |
| area_selected_minus_spatial_shuffle_mass | +0.767771 | [-0.0955, 1.8815] |
| area_selected_minus_semantic_top8 | +0.009381 | [-0.1035, 0.0975] |
| area_selected_minus_top4 | +0.113429 | [-0.0942, 0.3020] |
| area_selected_minus_top16 | +0.039860 | [-0.1620, 0.2678] |
| no_cls_minus_no_region_no_cls | +0.592793 | [-0.1268, 1.4915] |
| no_region_minus_no_region_no_cls | +2.040492 | [-0.4475, 4.4243] |