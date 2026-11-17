# Current-Method Ablations: L14_waterbirds_seg

Evaluation-disjoint training diagnostic, not another full-test benchmark. All controls retained.

| Control | Mean mIoU | Arrangement std |
|---|---:|---:|
| MLMP_paper | 52.722757 | 0.156989 |
| area_default | 53.527830 | 0.067246 |
| area_selected | 56.960559 | 0.270515 |
| no_region | 56.515309 | 0.153407 |
| no_global | 54.972364 | 0.043870 |
| no_cls | 54.662732 | 0.162418 |
| no_region_no_cls | 54.360124 | 0.240785 |
| region_balanced_mass | 56.348540 | 0.419048 |
| uniform_covered_mass | 56.886607 | 0.239705 |
| uniform_image_mass | 56.954472 | 0.178139 |
| spatial_shuffle_mass | 57.042953 | 0.236105 |
| semantic_top8 | 56.960559 | 0.270515 |
| top4 | 56.891756 | 0.184002 |
| top16 | 56.853762 | 0.303741 |

| Contrast | Difference pp | Shared whole-block pointwise95 |
|---|---:|---|
| area_default_minus_MLMP_paper | +0.805073 | [0.5479, 1.0739] |
| area_selected_minus_MLMP_paper | +4.237802 | [2.4923, 6.0279] |
| area_selected_minus_area_default | +3.432729 | [1.6667, 5.2761] |
| area_selected_minus_no_region | +0.445249 | [-0.0370, 0.9314] |
| area_selected_minus_no_global | +1.988195 | [0.8895, 3.0118] |
| area_selected_minus_no_cls | +2.297826 | [0.4047, 4.3018] |
| area_selected_minus_no_region_no_cls | +2.600435 | [0.6604, 4.6019] |
| area_selected_minus_region_balanced_mass | +0.612018 | [-0.1176, 1.3176] |
| area_selected_minus_uniform_covered_mass | +0.073952 | [-0.0754, 0.2744] |
| area_selected_minus_uniform_image_mass | +0.006087 | [-0.3402, 0.3513] |
| area_selected_minus_spatial_shuffle_mass | -0.082394 | [-0.4006, 0.2182] |
| area_selected_minus_semantic_top8 | +0.000000 | [0.0000, 0.0000] |
| area_selected_minus_top4 | +0.068803 | [-0.2925, 0.4221] |
| area_selected_minus_top16 | +0.106797 | [-0.2447, 0.4928] |
| no_cls_minus_no_region_no_cls | +0.302608 | [-0.1619, 0.7866] |
| no_region_minus_no_region_no_cls | +2.155185 | [0.0963, 4.3802] |