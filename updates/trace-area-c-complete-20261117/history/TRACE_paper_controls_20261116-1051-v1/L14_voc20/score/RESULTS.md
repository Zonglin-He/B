# Current-Method Ablations: L14_voc20

Evaluation-disjoint training diagnostic, not another full-test benchmark. All controls retained.

| Control | Mean mIoU | Arrangement std |
|---|---:|---:|
| MLMP_paper | 85.157991 | 0.156385 |
| area_default | 85.528056 | 0.109044 |
| area_selected | 86.401209 | 0.000000 |
| no_region | 86.039895 | 0.000000 |
| no_global | 86.510922 | 0.000000 |
| no_cls | 83.773817 | 0.000000 |
| no_region_no_cls | 83.040694 | 0.000000 |
| region_balanced_mass | 86.091949 | 0.000000 |
| uniform_covered_mass | 86.395704 | 0.000000 |
| uniform_image_mass | 86.047809 | 0.000000 |
| spatial_shuffle_mass | 85.856465 | 0.000000 |
| semantic_top8 | 86.296926 | 0.000000 |
| top4 | 86.413543 | 0.000000 |
| top16 | 86.252215 | 0.000000 |

| Contrast | Difference pp | Shared whole-block pointwise95 |
|---|---:|---|
| area_default_minus_MLMP_paper | +0.370065 | [0.0962, 0.6719] |
| area_selected_minus_MLMP_paper | +1.243219 | [-1.2126, 3.5297] |
| area_selected_minus_area_default | +0.873154 | [-1.5181, 3.0765] |
| area_selected_minus_no_region | +0.361314 | [-0.1276, 0.8958] |
| area_selected_minus_no_global | -0.109713 | [-2.1377, 1.7419] |
| area_selected_minus_no_cls | +2.627393 | [1.3931, 4.3474] |
| area_selected_minus_no_region_no_cls | +3.360515 | [1.9988, 5.2940] |
| area_selected_minus_region_balanced_mass | +0.309261 | [-0.3173, 1.3202] |
| area_selected_minus_uniform_covered_mass | +0.005505 | [-0.1947, 0.1838] |
| area_selected_minus_uniform_image_mass | +0.353401 | [0.0105, 0.7873] |
| area_selected_minus_spatial_shuffle_mass | +0.544745 | [0.0256, 1.2437] |
| area_selected_minus_semantic_top8 | +0.104284 | [-0.0253, 0.2430] |
| area_selected_minus_top4 | -0.012333 | [-0.3664, 0.4416] |
| area_selected_minus_top16 | +0.148995 | [-0.0772, 0.4699] |
| no_cls_minus_no_region_no_cls | +0.733122 | [0.3700, 1.1685] |
| no_region_minus_no_region_no_cls | +2.999201 | [1.6516, 4.8176] |