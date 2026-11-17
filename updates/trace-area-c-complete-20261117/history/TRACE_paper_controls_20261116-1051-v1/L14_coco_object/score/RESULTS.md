# Current-Method Ablations: L14_coco_object

Evaluation-disjoint training diagnostic, not another full-test benchmark. All controls retained.

| Control | Mean mIoU | Arrangement std |
|---|---:|---:|
| MLMP_paper | 27.378898 | 0.065865 |
| area_default | 27.355931 | 0.037415 |
| area_selected | 29.769773 | 0.000000 |
| no_region | 29.875051 | 0.000000 |
| no_global | 21.508077 | 0.000000 |
| no_cls | 29.125891 | 0.000000 |
| no_region_no_cls | 29.766472 | 0.000000 |
| region_balanced_mass | 29.874514 | 0.000000 |
| uniform_covered_mass | 30.088390 | 0.000000 |
| uniform_image_mass | 30.206107 | 0.000000 |
| spatial_shuffle_mass | 30.180502 | 0.000000 |
| semantic_top8 | 29.730381 | 0.000000 |
| top4 | 29.727610 | 0.000000 |
| top16 | 29.799306 | 0.000000 |
| score_Uniform | 30.053412 | 0.000000 |
| score_Confidence | 29.496062 | 0.000000 |
| score_C | 29.767762 | 0.000000 |
| score_RelativeC | 29.770661 | 0.000000 |
| score_ResponseChange | 29.953529 | 0.000000 |
| score_CalibratedConfidence | 29.369612 | 0.000000 |
| score_RegionPLPD | 29.785309 | 0.000000 |

| Contrast | Difference pp | Shared whole-block pointwise95 |
|---|---:|---|
| area_default_minus_MLMP_paper | -0.022967 | [-0.0920, 0.0970] |
| area_selected_minus_MLMP_paper | +2.390876 | [1.7101, 4.0543] |
| area_selected_minus_area_default | +2.413843 | [1.7233, 4.0732] |
| area_selected_minus_no_region | -0.105278 | [-0.6504, 0.4376] |
| area_selected_minus_no_global | +8.261696 | [5.7559, 8.8441] |
| area_selected_minus_no_cls | +0.643882 | [-0.8495, 2.0405] |
| area_selected_minus_no_region_no_cls | +0.003302 | [-1.3439, 2.1000] |
| area_selected_minus_region_balanced_mass | -0.104741 | [-0.6242, 0.6593] |
| area_selected_minus_uniform_covered_mass | -0.318617 | [-0.7292, 0.6788] |
| area_selected_minus_uniform_image_mass | -0.436334 | [-0.9317, 0.2469] |
| area_selected_minus_spatial_shuffle_mass | -0.410728 | [-0.8533, 0.3972] |
| area_selected_minus_semantic_top8 | +0.039393 | [-0.3964, 0.3495] |
| area_selected_minus_top4 | +0.042163 | [-0.4313, 0.8155] |
| area_selected_minus_top16 | -0.029533 | [-0.6981, 0.4675] |
| area_selected_minus_score_Uniform | -0.283638 | [-0.6588, 0.7064] |
| area_selected_minus_score_Confidence | +0.273711 | [-0.0857, 0.7388] |
| area_selected_minus_score_C | +0.002012 | [-0.0047, 0.0188] |
| area_selected_minus_score_RelativeC | -0.000888 | [-0.3465, 0.2367] |
| area_selected_minus_score_ResponseChange | -0.183756 | [-0.5169, 0.4417] |
| area_selected_minus_score_CalibratedConfidence | +0.400162 | [-0.2113, 1.0227] |
| area_selected_minus_score_RegionPLPD | -0.015536 | [-0.5198, 0.9425] |
| no_cls_minus_no_region_no_cls | -0.640581 | [-0.9957, 0.4616] |
| no_region_minus_no_region_no_cls | +0.108579 | [-1.2839, 2.1487] |