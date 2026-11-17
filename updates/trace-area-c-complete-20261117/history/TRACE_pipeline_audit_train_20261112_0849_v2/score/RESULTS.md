# Pipeline Single-Factor Training Audit

Fixed32 training sources; clean + original15 S5. Not full-validation evidence.

| Arm | Clean | S5 macro |
|---|---:|---:|
| MLMP | 63.9395 | 57.9451 |
| area_C | 63.7954 | 58.2182 |
| coverage_native_budget | 64.2557 | 57.9918 |
| area_uniform_covered | 64.0002 | 58.1538 |
| area_uniform_image | 63.7462 | 57.9235 |
| area_shuffled | 63.4981 | 58.0449 |
| coverage_static | 64.3239 | 57.7924 |
| coverage_uniform_image | 63.8295 | 57.8983 |
| native_no_cls | 63.5021 | 56.2217 |
| area_no_cls | 63.2786 | 56.6470 |

| Contrast | S5 delta | Pointwise 95% paired interval |
|---|---:|---|
| area_uniform_covered_minus_area_C | -0.06447590508360672 | [-0.30912806513108965, 0.786092243698631] |
| area_uniform_image_minus_area_C | -0.2947717884447812 | [-0.4948691681016591, 0.7557409254509974] |
| area_shuffled_minus_area_C | -0.17331102924444555 | [-0.5371421334676324, 0.5755411897008971] |
| coverage_static_minus_coverage_native_budget | -0.1994001868673886 | [-0.4470843029347881, 0.29970408122245884] |
| coverage_uniform_image_minus_coverage_native_budget | -0.09349541916185682 | [-0.7010748113284447, 0.4101703056121919] |
| native_no_cls_minus_MLMP | -1.7234078230354897 | [-3.548989001969793, -0.05187100253625704] |
| area_no_cls_minus_area_C | -1.5712155666810608 | [-3.245988163063975, 0.28497915629581455] |
| area_no_cls_minus_native_no_cls | 0.42528244243630503 | [-0.30780447508167785, 1.0566082972258812] |
| area_C_minus_MLMP | 0.27309018608187613 | [-0.5902239608960705, 0.6871931650490488] |
| coverage_native_budget_minus_area_C | -0.22639589843581737 | [-0.49199365908098464, 0.5984535317843569] |

10000 shared whole-B2-group draws. No multiplicity correction or winner selection. All negative/null results and undefined counts retained.
Diagnostics cover first4 training sources per condition, not a representative population claim. No new validation predictions.
