# All Seven Current-Method Diagnostic Ablations

Actual finite queue completed on November16. These are256 evaluation-disjoint training sources per dataset family with three fixed within8-source-block arrangements, not full-test ablations. All126 task/control cells and378 exact trial points are complete;96768 method-source predictions exclude numerical-gate work. No control was selected as a replacement main method. MLMP and all22 selected TRACE fields remain fixed.

## All Core Controls

| Control | L14_coco_object | L14_pc60 | L14_voc20 | L14_waterbirds_seg | B16_coco_object | B16_pc60 | B16_voc20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLMP_paper | 27.378898 | 27.366621 | 85.157991 | 52.722757 | 26.746907 | 27.475230 | 85.420907 |
| area_default | 27.355931 | 27.358463 | 85.528056 | 53.527830 | 26.786406 | 27.375091 | 85.640675 |
| area_selected | 29.769773 | 27.481358 | 86.401209 | 56.960559 | 26.876163 | 26.506104 | 85.171920 |
| no_region | 29.875051 | 27.514792 | 86.039895 | 56.515309 | 26.745807 | 26.652990 | 84.340774 |
| no_cls | 29.125891 | 27.331583 | 83.773817 | 54.662732 | 26.554645 | 26.151422 | 82.893075 |
| no_global | 21.508077 | 24.903174 | 86.510922 | 54.972364 | 24.358105 | 25.588934 | 84.348583 |
| no_region_no_cls | 29.766472 | 27.442779 | 83.040694 | 54.360124 | 26.503050 | 26.376097 | 82.300282 |
| region_balanced_mass | 29.874514 | 27.653707 | 86.091949 | 56.348540 | 26.357385 | 26.651394 | 85.828176 |
| uniform_covered_mass | 30.088390 | 27.434106 | 86.395704 | 56.886607 | 26.677313 | 26.742688 | 85.156754 |
| uniform_image_mass | 30.206107 | 27.537250 | 86.047809 | 56.954472 | 27.060205 | 26.864911 | 84.990952 |
| spatial_shuffle_mass | 30.180502 | 27.609752 | 85.856465 | 57.042953 | 27.095502 | 26.742172 | 84.404149 |
| semantic_top8 | 29.730381 | 27.481358 | 86.296926 | 56.960559 | 26.812053 | 26.506104 | 85.162539 |
| top4 | 29.727610 | 27.471186 | 86.413543 | 56.891756 | 26.897920 | 26.597555 | 85.058491 |
| top16 | 29.799306 | 27.334009 | 86.252215 | 56.853762 | 26.965113 | 26.628382 | 85.132060 |

## Fixed Score-Source Controls

C/Confidence/RelativeC/ResponseChange/CalibratedConfidence/RegionPLPD/Uniform share candidate support and the frozen source-score protocol. RegionPLPD is a regional DeYO adaptation, not a full-DeYO reproduction. Calibration is a GT-free training-source quantile map to C, not correctness calibration.

| Control | L14 Object | L14 PC60 | B16 Object | B16 PC60 |
|---|---:|---:|---:|---:|
| score_C | 29.767762 | 27.475982 | 26.745971 | 26.521520 |
| score_Confidence | 29.496062 | 27.431964 | 26.750899 | 26.766691 |
| score_RelativeC | 29.770661 | 27.441986 | 26.806361 | 26.543117 |
| score_ResponseChange | 29.953529 | 27.455593 | 26.850275 | 26.700945 |
| score_CalibratedConfidence | 29.369612 | 27.502857 | 26.963677 | 26.668035 |
| score_RegionPLPD | 29.785309 | 27.538278 | 26.753120 | 26.610875 |
| score_Uniform | 30.053412 | 27.410496 | 26.759582 | 26.715071 |

## Regional Component Contrast

Selected area_C minus no-region, all other selected parameters unchanged. Intervals are pointwise95% from10000 identical whole8-source-block draws shared across the contrasted arms and all three arrangements, not independent-seed uncertainty or multiplicity-adjusted significance.

| Task | Delta pp | Lower95 | Upper95 | Undefined draws |
|---|---:|---:|---:|---:|
| L14_coco_object | -0.105278 | -0.650409 | +0.437585 | 0 |
| L14_pc60 | -0.033433 | -0.213806 | +0.226223 | 0 |
| L14_voc20 | +0.361314 | -0.127604 | +0.895784 | 0 |
| L14_waterbirds_seg | +0.445249 | -0.036985 | +0.931406 | 0 |
| B16_coco_object | +0.130357 | -0.618024 | +0.858698 | 0 |
| B16_pc60 | -0.146887 | -0.355984 | +0.047840 | 0 |
| B16_voc20 | +0.831146 | +0.007101 | +2.027299 | 0 |

Six of seven intervals contain zero. B16VOC20 is pointwise positive with lower bound0.007101pp, not a multiplicity-adjusted general mechanism result. L14Object/PC60 and B16PC60 no-region point estimates exceed full area_C. Uniform or spatially shuffled mass often matches or exceeds C. Thus the system-level main benchmark improvement does not identify C as its cause.

## Default-Parameter Addition

Area-default minus fixed paper MLMP is separate from the selected-system contrast.

| Task | Delta pp | Lower95 | Upper95 | Undefined draws |
|---|---:|---:|---:|---:|
| L14_coco_object | -0.022967 | -0.091997 | +0.097029 | 0 |
| L14_pc60 | -0.008158 | -0.052349 | +0.075683 | 0 |
| L14_voc20 | +0.370065 | +0.096199 | +0.671853 | 0 |
| L14_waterbirds_seg | +0.805073 | +0.547878 | +1.073891 | 0 |
| B16_coco_object | +0.039499 | -0.095321 | +0.173870 | 0 |
| B16_pc60 | -0.100139 | -0.191501 | -0.003047 | 0 |
| B16_voc20 | +0.219768 | -0.080096 | +0.588848 | 0 |

## Label-Free Transfer Diagnostics

Candidate profiles cover all256 diagnostic sources in all3 arrangements. Gradient/update profiles cover only the first fixed8-source block at initial/final steps. Means below are descriptive, not a causal model of transfer failure.

| Task | Candidate C | Covered fraction | Raw C bonus/grid | Regional/global gradient norm | Relative update L2 |
|---|---:|---:|---:|---:|---:|
| L14_coco_object | 0.309047 | 0.179779 | 0.117531 | 0.195230 | 0.040476 |
| L14_pc60 | 0.316503 | 0.267395 | 0.188604 | 0.556890 | 0.030791 |
| L14_voc20 | 0.290370 | 0.407181 | 0.271975 | 0.370983 | 0.014235 |
| L14_waterbirds_seg | 0.407624 | 0.755783 | 0.591657 | 0.747085 | 0.013121 |
| B16_coco_object | 0.395587 | 0.394830 | 0.284792 | 0.250196 | 0.034341 |
| B16_pc60 | 0.390124 | 0.490533 | 0.369853 | 0.546575 | 0.024583 |
| B16_voc20 | 0.302375 | 0.479233 | 0.323587 | 0.254294 | 0.012039 |

Object/PC60 C and coverage rise on B16, but PC60's first-block regional/global gradient norm does not rise and relative update norms are lower on B16. A blanket explanation that B16 simply receives larger updates is not established. PC60 has60 queries/classes with an identity mapping, so an alias-merging issue cannot by itself explain every transfer failure. Geometry/readout and pretrained feature differences remain hypotheses, not verified causes.

Waterbirds diagnostic groups remain134/1/1/120. The original worst-group contrast has3605 undefined draws and no95% interval. It is segmentation group mIoU, not classification worst-group accuracy. All per-arm interactions, class deltas, negative/null results and undefined values remain in each sealed task/score/RESULT.json.

## Verification And Scope

- Main cumulatively verified140 per-task prediction/score output hashes,35 controller outputs and all378 exact points from integer compact counts. The last168 B16 points were recomputed on November16 without reopening GT.
- All seven prediction AND score stages have their original same-invocation actual retained successful exits. V2 controller1801646/inv19d4e210ee3e417684a8bf04ef2a5780 is Code1/Status0/success and MainPID0.
- V2 completion SHA256:8ba17522f18de7003f0faf176780ec1bb8825f52d44b211ab37c28fc82baf45d.
- Derived transfer CPU2123546/inv60c7abe22ed54373aab93eeca67b7920 actually succeeded; all86 outputs verified. Completion SHA256:2b28f4849a3325527e47088ee47c95606c170db98259fde9e042e0a666f8deb6.
- All77 study model sources remain unchanged. Old V1 cancellation/source/log evidence, prior full26 results, B16 negatives, R03 NULLs, PC historical exposure and Object paper-value discrepancy remain.
- The subsequent R04/time/TENT and full-component queues are separate ongoing science. This completed diagnostic report is not an all-paper completion or a completed manuscript.

Study:[revision-results]/TRACE_paper_controls_20261116-1051-v1.
Transfer:[revision-results]/TRACE_paper_transfer_meta_20261116-1655-v1.
