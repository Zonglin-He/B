# R04 Current-Method Foreground Context Intervention

Actual predictions and NEW offline CPU scoring completed on November16. Main independently verified all prediction/score outputs and the same actual retained successful exits, and recomputed all16 mean foreground-accuracy points from the sealed derived accuracy array without reopening GT.

## Design

128 requested COCO cases;126 constructed, indices21/102 unavailable for no supported eligible instance. The four views are original, exact sham, hard replacement and edge replacement. Raw/model and nativefloat32 foregrounds are bitwise identical; there is no second resize. All sham predictions are exactly unchanged. Three fixed arrangements remain inside common4-source blocks;10000 shared whole-block draws use32 blocks. These are pointwise unadjusted intervals, not independent-initialization uncertainty.

Metric: selected-instance foreground pixel accuracy, averaged per object and arrangement, in percent. Not full-image/composite mIoU, classification worst-group accuracy, an independent Waterbirds test or a universal context-removal benefit. MLMP_zero_update uses the MLMP multi-prompt/readout system at zero update and is not official single-prompt NoAdapt.

## Complete Mean Table

| Method | Original | Exact sham | Hard replacement | Edge replacement |
|---|---:|---:|---:|---:|
| MLMP_zero_update | 70.622955 | 70.622955 | 56.713067 | 55.414734 |
| MLMP_paper | 71.826537 | 71.826537 | 56.884988 | 57.135531 |
| area_selected | 67.332455 | 67.332455 | 57.496331 | 57.958288 |
| no_region | 65.907531 | 65.907531 | 56.784129 | 57.486295 |

## All Paired Contrasts

| Contrast | Delta pp | Lower95 | Upper95 | Undefined |
|---|---:|---:|---:|---:|
| MLMP_paper_original_minus_replacement_edge | +14.691006 | +9.760616 | +19.872079 | 0 |
| MLMP_paper_original_minus_replacement_hard | +14.941550 | +9.005179 | +21.082570 | 0 |
| MLMP_paper_original_minus_sham | +0.000000 | +0.000000 | +0.000000 | 0 |
| MLMP_zero_update_original_minus_replacement_edge | +15.208221 | +9.917854 | +20.817104 | 0 |
| MLMP_zero_update_original_minus_replacement_hard | +13.909888 | +8.160474 | +20.205012 | 0 |
| MLMP_zero_update_original_minus_sham | +0.000000 | +0.000000 | +0.000000 | 0 |
| area_minus_MLMP_original | -4.494082 | -7.962037 | -0.996420 | 0 |
| area_minus_MLMP_replacement_edge | +0.822757 | -3.227947 | +5.092930 | 0 |
| area_minus_MLMP_replacement_hard | +0.611343 | -3.520534 | +4.713239 | 0 |
| area_minus_MLMP_sham | -4.494082 | -7.962037 | -0.996420 | 0 |
| area_minus_no_region_original | +1.424924 | +0.154566 | +2.804278 | 0 |
| area_minus_no_region_replacement_edge | +0.471993 | -0.702097 | +1.407277 | 0 |
| area_minus_no_region_replacement_hard | +0.712202 | -0.296236 | +1.799961 | 0 |
| area_minus_no_region_sham | +1.424924 | +0.154566 | +2.804278 | 0 |
| area_selected_original_minus_replacement_edge | +9.374167 | +2.943114 | +16.011370 | 0 |
| area_selected_original_minus_replacement_hard | +9.836124 | +3.557582 | +16.205136 | 0 |
| area_selected_original_minus_sham | +0.000000 | +0.000000 | +0.000000 | 0 |
| no_region_original_minus_replacement_edge | +8.421236 | +1.809246 | +15.320615 | 0 |
| no_region_original_minus_replacement_hard | +9.123402 | +2.569824 | +15.802188 | 0 |
| no_region_original_minus_sham | +0.000000 | +0.000000 | +0.000000 | 0 |

## Helpful And Harmful Context

Counts describe observed original-versus-hard-replacement outcomes, not enrollment, donor selection or tuning.

| Method | Original context beneficial | Replacement beneficial | Null |
|---|---:|---:|---:|
| MLMP_zero_update | 79 | 39 | 8 |
| MLMP_paper | 74 | 41 | 11 |
| area_selected | 61 | 51 | 14 |
| no_region | 62 | 49 | 15 |

## Interpretation

Original context is beneficial on average for every method in this constructed set. TRACE has a smaller original-to-replacement drop than MLMP, but a smaller drop from a lower initial point is not sufficient evidence of superior robustness. On original views TRACE-minus-MLMP is-4.494082pp,95[-7.962037,-0.996420]. On hard and edge replacements it is+0.611343pp and+0.822757pp, respectively, with both intervals including zero. The region component improves the original-view point relative to no-region by1.424924pp,95[0.154566,2.804278]; the replacement-view component intervals include zero. None of these findings validates calibrated update safety or universal C-specific context robustness.

## Exact Evidence

Prediction unit:2162503/invc18f2f2a7e0e454b80ed13c0eea12672, actual Code1/Status0/success/MainPID0.
Score unit:2170093/inve116237b2bda402b8b313d85d2714440, actual Code1/Status0/success/MainPID0.
Verified outputs:18.
Prediction COMPLETED SHA256:7f66e631a8135211f921fc995cf213bc3d99b45bf704291ef33b3734e65d4f5d.
Score COMPLETED SHA256:28a021cb2daf15bb7ee072a133d1548b5203170c4e135490e9a097ce8cd37ba9.
RESULT SHA256:664dc6a8c76d5a93c52041ca28fe09b92a412d245cdd9c30a8077fd5a00ac5fe.
Root:[revision-results]/TRACE_paper_finish_20261116-1637-v1/R04.
Old failure/null/exposure evidence remains. Main method, all22 parameters and official MLMP were not reselected. This report completes R04 inference/scoring only, not time/TENT/full-component or manuscript work.
