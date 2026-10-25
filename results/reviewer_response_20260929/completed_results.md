# Reviewer-response experiment results

Completed: 619 / 773; failed: 0; blocked: 0.
Active cell: `None`.

Only completed evidence is summarized. See `docs/reviewer-response-20260929.md` for all nine responses and claim restrictions. The Waterbirds historical train/test overlap and labeled calibration limitations remain unresolved by merely launching reruns.

## Budget cancellation

The user cancelled 154 unfinished WATT/CLIPArTT jobs. Completed results remain reported. These are budget/time cancellations, not execution failures; both methods have successful local runs. Completion excludes cancelled_jobs.json entries.

## Active protocol amendment

Further tuning-fairness/equal-budget runs and new ACDC reruns are cancelled. Future TRACE runs reuse its frozen dataset configurations; baselines reuse existing dataset recipes. Completed original uniform cost controls retain their protocol labels.

W9: four datasets, 256 fixed images each, six corruptions at severities 1/3/5 plus clean, seven methods, seed 0. This is a selected synthetic-corruption panel, not natural-shift or continual-adaptation evidence.

## Additional mechanism robustness

The historical threshold sweep is a reaggregation, not a fresh experiment. Quartiles and denominators are fixed across thresholds. Fresh three-seed replay uses identical masks and single-crop pre/post forwards and remains separately labeled.

| evidence | seed | threshold | low_harm_pct | high_harm_pct | reduction_pp |
| --- | --- | --- | --- | --- | --- |
| historical_posthoc | 0 | 0.001 | 39.822 | 21.929 | 17.894 |
| historical_posthoc | 0 | 0.005 | 32.738 | 17.269 | 15.469 |
| historical_posthoc | 0 | 0.010 | 27.683 | 14.365 | 13.318 |
| historical_posthoc | 0 | 0.020 | 21.990 | 10.805 | 11.185 |
| new_three_seed_replay | 0 | 0.001 | 40.004 | 21.883 | 18.121 |
| new_three_seed_replay | 0 | 0.005 | 32.811 | 17.260 | 15.551 |
| new_three_seed_replay | 0 | 0.010 | 27.595 | 14.327 | 13.268 |
| new_three_seed_replay | 0 | 0.020 | 22.156 | 10.752 | 11.404 |
| new_three_seed_replay | 1 | 0.001 | 40.004 | 21.883 | 18.121 |
| new_three_seed_replay | 1 | 0.005 | 32.811 | 17.260 | 15.551 |
| new_three_seed_replay | 1 | 0.010 | 27.595 | 14.327 | 13.268 |
| new_three_seed_replay | 1 | 0.020 | 22.156 | 10.752 | 11.404 |
| new_three_seed_replay | 2 | 0.001 | 40.004 | 21.883 | 18.121 |
| new_three_seed_replay | 2 | 0.005 | 32.811 | 17.260 | 15.551 |
| new_three_seed_replay | 2 | 0.010 | 27.595 | 14.327 | 13.268 |
| new_three_seed_replay | 2 | 0.020 | 22.156 | 10.752 | 11.404 |

Continuous probability changes are in supplement/threshold_robustness/continuous_delta.csv. Higher reliability need not imply larger mean improvement; risk stratification and mean benefit are different claims. The appended protocol is supplement_manifest.json.

## Corruption results

Means require all six operators; incomplete means are left blank.

| dataset | method | seed | severity | completed_corruptions | required_corruptions | clean_mIoU | mean_corrupted_mIoU | clean_minus_corrupted_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coco_cd | mlmp | 0 | 1 | 6 | 6 | 71.930 | 69.428 | 2.502 |
| coco_cd | mlmp | 0 | 3 | 6 | 6 | 71.930 | 65.985 | 5.945 |
| coco_cd | mlmp | 0 | 5 | 6 | 6 | 71.930 | 57.223 | 14.707 |
| coco_cd | noadapt | 0 | 1 | 6 | 6 | 60.700 | 58.682 | 2.018 |
| coco_cd | noadapt | 0 | 3 | 6 | 6 | 60.700 | 56.098 | 4.602 |
| coco_cd | noadapt | 0 | 5 | 6 | 6 | 60.700 | 51.017 | 9.683 |
| coco_cd | tent | 0 | 1 | 6 | 6 | 73.030 | 71.312 | 1.718 |
| coco_cd | tent | 0 | 3 | 6 | 6 | 73.030 | 67.567 | 5.463 |
| coco_cd | tent | 0 | 5 | 6 | 6 | 73.030 | 59.200 | 13.830 |
| coco_cd | tpt | 0 | 1 | 6 | 6 | 68.470 | 67.950 | 0.520 |
| coco_cd | tpt | 0 | 3 | 6 | 6 | 68.470 | 64.302 | 4.168 |
| coco_cd | tpt | 0 | 5 | 6 | 6 | 68.470 | 53.410 | 15.060 |
| coco_cd | trace | 0 | 1 | 6 | 6 | 75.140 | 72.855 | 2.285 |
| coco_cd | trace | 0 | 3 | 6 | 6 | 75.140 | 66.657 | 8.483 |
| coco_cd | trace | 0 | 5 | 6 | 6 | 75.140 | 50.483 | 24.657 |
| coco_object | mlmp | 0 | 1 | 6 | 6 | 28.760 | 28.542 | 0.218 |
| coco_object | mlmp | 0 | 3 | 6 | 6 | 28.760 | 26.195 | 2.565 |
| coco_object | mlmp | 0 | 5 | 6 | 6 | 28.760 | 20.947 | 7.813 |
| coco_object | noadapt | 0 | 1 | 6 | 6 | 23.140 | 22.638 | 0.502 |
| coco_object | noadapt | 0 | 3 | 6 | 6 | 23.140 | 20.630 | 2.510 |
| coco_object | noadapt | 0 | 5 | 6 | 6 | 23.140 | 16.172 | 6.968 |
| coco_object | tent | 0 | 1 | 6 | 6 | 23.800 | 23.498 | 0.302 |
| coco_object | tent | 0 | 3 | 6 | 6 | 23.800 | 20.788 | 3.012 |
| coco_object | tent | 0 | 5 | 6 | 6 | 23.800 | 16.347 | 7.453 |
| coco_object | tpt | 0 | 1 | 6 | 6 | 23.630 | 23.480 | 0.150 |
| coco_object | tpt | 0 | 3 | 6 | 6 | 23.630 | 21.058 | 2.572 |
| coco_object | tpt | 0 | 5 | 6 | 6 | 23.630 | 16.062 | 7.568 |
| coco_object | trace | 0 | 1 | 6 | 6 | 30.640 | 29.110 | 1.530 |
| coco_object | trace | 0 | 3 | 6 | 6 | 30.640 | 25.675 | 4.965 |
| coco_object | trace | 0 | 5 | 6 | 6 | 30.640 | 19.032 | 11.608 |
| pc60 | clipartt | 0 | 1 | 0 | 6 | 22.140 | -- | -- |
| pc60 | clipartt | 0 | 3 | 0 | 6 | 22.140 | -- | -- |
| pc60 | clipartt | 0 | 5 | 0 | 6 | 22.140 | -- | -- |
| pc60 | mlmp | 0 | 1 | 6 | 6 | 27.850 | 27.168 | 0.682 |
| pc60 | mlmp | 0 | 3 | 6 | 6 | 27.850 | 25.817 | 2.033 |
| pc60 | mlmp | 0 | 5 | 6 | 6 | 27.850 | 20.247 | 7.603 |
| pc60 | noadapt | 0 | 1 | 6 | 6 | 25.440 | 24.882 | 0.558 |
| pc60 | noadapt | 0 | 3 | 6 | 6 | 25.440 | 23.337 | 2.103 |
| pc60 | noadapt | 0 | 5 | 6 | 6 | 25.440 | 18.085 | 7.355 |
| pc60 | tent | 0 | 1 | 6 | 6 | 24.800 | 23.978 | 0.822 |
| pc60 | tent | 0 | 3 | 6 | 6 | 24.800 | 22.668 | 2.132 |
| pc60 | tent | 0 | 5 | 6 | 6 | 24.800 | 17.583 | 7.217 |
| pc60 | tpt | 0 | 1 | 6 | 6 | 23.110 | 22.025 | 1.085 |
| pc60 | tpt | 0 | 3 | 6 | 6 | 23.110 | 20.498 | 2.612 |
| pc60 | tpt | 0 | 5 | 6 | 6 | 23.110 | 15.713 | 7.397 |
| pc60 | trace | 0 | 1 | 6 | 6 | 28.290 | 27.247 | 1.043 |
| pc60 | trace | 0 | 3 | 6 | 6 | 28.290 | 25.192 | 3.098 |
| pc60 | trace | 0 | 5 | 6 | 6 | 28.290 | 19.032 | 9.258 |
| pc60 | watt | 0 | 1 | 0 | 6 | 14.900 | -- | -- |
| pc60 | watt | 0 | 3 | 0 | 6 | 14.900 | -- | -- |
| pc60 | watt | 0 | 5 | 0 | 6 | 14.900 | -- | -- |
| voc21 | clipartt | 0 | 1 | 6 | 6 | 37.480 | 37.805 | -0.325 |
| voc21 | clipartt | 0 | 3 | 6 | 6 | 37.480 | 36.075 | 1.405 |
| voc21 | clipartt | 0 | 5 | 6 | 6 | 37.480 | 28.967 | 8.513 |
| voc21 | mlmp | 0 | 1 | 6 | 6 | 50.310 | 50.780 | -0.470 |
| voc21 | mlmp | 0 | 3 | 6 | 6 | 50.310 | 47.978 | 2.332 |
| voc21 | mlmp | 0 | 5 | 6 | 6 | 50.310 | 41.667 | 8.643 |
| voc21 | noadapt | 0 | 1 | 6 | 6 | 44.090 | 44.270 | -0.180 |
| voc21 | noadapt | 0 | 3 | 6 | 6 | 44.090 | 42.645 | 1.445 |
| voc21 | noadapt | 0 | 5 | 6 | 6 | 44.090 | 35.373 | 8.717 |
| voc21 | tent | 0 | 1 | 6 | 6 | 44.540 | 44.733 | -0.193 |
| voc21 | tent | 0 | 3 | 6 | 6 | 44.540 | 42.632 | 1.908 |
| voc21 | tent | 0 | 5 | 6 | 6 | 44.540 | 34.853 | 9.687 |
| voc21 | tpt | 0 | 1 | 6 | 6 | 43.830 | 44.175 | -0.345 |
| voc21 | tpt | 0 | 3 | 6 | 6 | 43.830 | 41.930 | 1.900 |
| voc21 | tpt | 0 | 5 | 6 | 6 | 43.830 | 33.830 | 10.000 |
| voc21 | trace | 0 | 1 | 6 | 6 | 53.190 | 52.680 | 0.510 |
| voc21 | trace | 0 | 3 | 6 | 6 | 53.190 | 50.190 | 3.000 |
| voc21 | trace | 0 | 5 | 6 | 6 | 53.190 | 41.797 | 11.393 |
| voc21 | watt | 0 | 1 | 6 | 6 | 27.080 | 26.835 | 0.245 |
| voc21 | watt | 0 | 3 | 6 | 6 | 27.080 | 25.703 | 1.377 |
| voc21 | watt | 0 | 5 | 6 | 6 | 27.080 | 20.448 | 6.632 |

## Same-location recognition

Fixed 32-image samples per dataset; accuracy (%), default weakening. GT-component probes are only used offline, up to eight per crop. Original uses the identical unmodified pixels in a size-three view batch; Native original also reports the ordinary crop batch for transparency. These units can be correlated across overlapping crops, and are not full-dataset mIoU.

| dataset | Probes | Native original | Original | Preserved | Suppressed | Perturbed |
| --- | --- | --- | --- | --- | --- | --- |
| coco_cd | 81 | 90.123 | 90.123 | 87.654 | 69.136 | 87.654 |
| coco_object | 169 | 65.089 | 64.497 | 55.030 | 41.420 | 56.213 |
| coco_stuff | 226 | 34.513 | 34.513 | 32.301 | 15.044 | 33.186 |
| pc60 | 232 | 39.655 | 39.655 | 40.517 | 38.793 | 43.534 |
| voc21 | 130 | 69.231 | 69.231 | 62.308 | 50.769 | 66.923 |
| waterbirds_seg | 102 | 67.647 | 67.647 | 73.529 | 64.706 | 79.412 |

Identity keep/drop controls: native-batch and matched-batch paired accuracy change (pp):

| dataset | view | paired_delta_pp | batchmatched_paired_delta_pp |
| --- | --- | --- | --- |
| coco_cd | preserved | 0.000 | 0.000 |
| coco_cd | suppressed | 0.000 | 0.000 |
| coco_object | preserved | -0.592 | 0.000 |
| coco_object | suppressed | -0.592 | 0.000 |
| coco_stuff | preserved | 0.000 | 0.000 |
| coco_stuff | suppressed | 0.000 | 0.000 |
| pc60 | preserved | 0.000 | 0.000 |
| pc60 | suppressed | 0.000 | 0.000 |
| voc21 | preserved | 0.000 | 0.000 |
| voc21 | suppressed | 0.000 | 0.000 |
| waterbirds_seg | preserved | 0.000 | 0.000 |
| waterbirds_seg | suppressed | 0.000 | 0.000 |
| coco_cd | preserved | 0.000 | 0.000 |
| coco_cd | suppressed | 0.000 | 0.000 |
| coco_object | preserved | -0.592 | 0.000 |
| coco_object | suppressed | -0.592 | 0.000 |
| coco_stuff | preserved | 0.000 | 0.000 |
| coco_stuff | suppressed | 0.000 | 0.000 |
| pc60 | preserved | 0.000 | 0.000 |
| pc60 | suppressed | 0.000 | 0.000 |
| voc21 | preserved | 0.000 | 0.000 |
| voc21 | suppressed | 0.000 | 0.000 |
| waterbirds_seg | preserved | 0.000 | 0.000 |
| waterbirds_seg | suppressed | 0.000 | 0.000 |

The identity replay is recorded in `identity_audit.json`. A COCO Object background probe has an original top-two margin of 0.0000129; batch-size-dependent numerical changes flip its top-1. The original records remain unchanged. Matching the forward batch shape prevents this null effect from being interpreted as an intervention loss.

## Weakening sensitivity

Spearman correlation with default candidate reliability; same candidates and original scoring locations. Correlation is not proof of source identification.

| dataset | gaussian | inpaint | mean | noise | shuffle |
| --- | --- | --- | --- | --- | --- |
| coco_cd | 0.929 | 0.861 | 0.919 | 0.848 | 0.813 |
| coco_object | 0.833 | 0.735 | 0.765 | 0.771 | 0.749 |
| coco_stuff | 0.843 | 0.698 | 0.743 | 0.760 | 0.751 |
| pc60 | 0.820 | 0.733 | 0.771 | 0.775 | 0.720 |
| voc21 | 0.868 | 0.721 | 0.802 | 0.792 | 0.739 |
| waterbirds_seg | 0.913 | 0.777 | 0.838 | 0.840 | 0.722 |

Mask controls:

| dataset | control | n | spearman_vs_default | mean_abs_rho_change |
| --- | --- | --- | --- | --- |
| coco_cd | dilate | 128 | 0.812 | 0.141 |
| coco_cd | erode | 128 | 0.713 | 0.184 |
| coco_cd | sham_shift | 128 | 0.637 | 0.210 |
| coco_object | dilate | 1083 | 0.570 | 0.165 |
| coco_object | erode | 1083 | 0.375 | 0.128 |
| coco_object | sham_shift | 1083 | 0.355 | 0.127 |
| coco_stuff | dilate | 1056 | 0.578 | 0.166 |
| coco_stuff | erode | 1056 | 0.357 | 0.135 |
| coco_stuff | sham_shift | 1056 | 0.366 | 0.135 |
| pc60 | dilate | 704 | 0.576 | 0.192 |
| pc60 | erode | 704 | 0.348 | 0.201 |
| pc60 | sham_shift | 704 | 0.344 | 0.198 |
| voc21 | dilate | 1045 | 0.605 | 0.159 |
| voc21 | erode | 1045 | 0.434 | 0.136 |
| voc21 | sham_shift | 1045 | 0.414 | 0.144 |
| waterbirds_seg | dilate | 222 | 0.708 | 0.165 |
| waterbirds_seg | erode | 222 | 0.508 | 0.189 |
| waterbirds_seg | sham_shift | 222 | 0.457 | 0.204 |

Inpainting is classical OpenCV Telea, radius 3, not learned image generation. Full-mask regions with no known boundary pixels are undefined for this control and excluded with counts in `tables/inpaint_exclusions.csv`; their raw records are preserved.

## Conditional risk information

New post-hoc image-grouped cross-fitting on 46,746 previously measured candidates. Endpoint unchanged: GT-class probability drop >=.001. No threshold search. These are predictive associations, not unique causal identification.

| dataset | confidence_only | confidence_plus_trace | confidence_plus_covariates | covariates_plus_trace |
| --- | --- | --- | --- | --- |
| coco_cd | 0.559 | 0.824 | 0.835 | 0.837 |
| coco_object | 0.579 | 0.697 | 0.712 | 0.716 |
| pc60 | 0.651 | 0.657 | 0.687 | 0.690 |
| voc21 | 0.570 | 0.738 | 0.753 | 0.756 |

Raw scores (no supervised crossfit model):

| dataset | confidence_risk | candidate_entropy | candidate_mlmp_disagreement | trace_risk |
| --- | --- | --- | --- | --- |
| coco_cd | 0.565 | 0.602 | 0.565 | 0.746 |
| coco_object | 0.581 | 0.663 | 0.597 | 0.546 |
| pc60 | 0.653 | 0.660 | 0.461 | 0.575 |
| voc21 | 0.573 | 0.713 | 0.654 | 0.574 |

| dataset | control | n | coverage | low_harm_pct | high_harm_pct | gap_pp | ci95_low_pp | ci95_high_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| voc21 | confidence_role | 7894 | 0.493 | 27.540 | 16.443 | 11.097 | 8.827 | 13.353 |
| voc21 | confidence_role_class_area_correctness | 4166 | 0.260 | 11.618 | 9.217 | 2.400 | 0.059 | 4.709 |
| pc60 | confidence_role | 5446 | 0.485 | 43.702 | 25.670 | 18.032 | 15.035 | 21.313 |
| pc60 | confidence_role_class_area_correctness | 542 | 0.048 | 20.295 | 15.129 | 5.166 | -1.411 | 11.923 |
| coco_cd | confidence_role | 900 | 0.450 | 56.889 | 23.778 | 33.111 | 27.297 | 39.482 |
| coco_cd | confidence_role_class_area_correctness | 406 | 0.203 | 47.291 | 42.365 | 4.926 | -5.117 | 14.959 |
| coco_object | confidence_role | 8616 | 0.492 | 31.105 | 25.139 | 5.966 | 3.655 | 8.252 |
| coco_object | confidence_role_class_area_correctness | 3206 | 0.183 | 16.095 | 12.165 | 3.930 | 1.559 | 6.570 |

The strict matched analysis has low coverage on PC60 and intervals crossing zero for PC60/COCO-CD. All strata, including inconclusive ones, are retained.

## Completed segmentation evaluations

Full splits, subsets, released and uniform configurations must not be conflated. Seed-0-only cells are not three-seed means.

| cell | images | full_split | mIoU | ms_per_image | peak_allocated_mib |
| --- | --- | --- | --- | --- | --- |
| corruption/coco_cd/brightness/severity1/mlmp/seed0 | 256 | False | 71.250 | 48.161 | 2857.142 |
| corruption/coco_cd/brightness/severity1/noadapt/seed0 | 256 | False | 60.500 | 1.828 | 1847.031 |
| corruption/coco_cd/brightness/severity1/tent/seed0 | 256 | False | 72.260 | 43.396 | 2762.586 |
| corruption/coco_cd/brightness/severity1/tpt/seed0 | 256 | False | 68.040 | 80.707 | 4245.571 |
| corruption/coco_cd/brightness/severity1/trace/seed0 | 256 | False | 73.870 | 58.998 | 7940.818 |
| corruption/coco_cd/brightness/severity3/mlmp/seed0 | 256 | False | 70.690 | 48.228 | 2857.142 |
| corruption/coco_cd/brightness/severity3/noadapt/seed0 | 256 | False | 60.110 | 1.823 | 1847.031 |
| corruption/coco_cd/brightness/severity3/tent/seed0 | 256 | False | 70.950 | 43.359 | 2762.586 |
| corruption/coco_cd/brightness/severity3/tpt/seed0 | 256 | False | 67.490 | 80.969 | 4245.571 |
| corruption/coco_cd/brightness/severity3/trace/seed0 | 256 | False | 74.400 | 58.887 | 7940.813 |
| corruption/coco_cd/brightness/severity5/mlmp/seed0 | 256 | False | 68.710 | 48.311 | 2857.142 |
| corruption/coco_cd/brightness/severity5/noadapt/seed0 | 256 | False | 60.740 | 1.821 | 1847.031 |
| corruption/coco_cd/brightness/severity5/tent/seed0 | 256 | False | 70.940 | 43.336 | 2762.586 |
| corruption/coco_cd/brightness/severity5/tpt/seed0 | 256 | False | 66.370 | 80.385 | 4245.571 |
| corruption/coco_cd/brightness/severity5/trace/seed0 | 256 | False | 72.230 | 58.415 | 7940.816 |
| corruption/coco_cd/clean/severity0/mlmp/seed0 | 256 | False | 71.930 | 48.326 | 2857.142 |
| corruption/coco_cd/clean/severity0/noadapt/seed0 | 256 | False | 60.700 | 1.808 | 1847.031 |
| corruption/coco_cd/clean/severity0/tent/seed0 | 256 | False | 73.030 | 43.587 | 2762.586 |
| corruption/coco_cd/clean/severity0/tpt/seed0 | 256 | False | 68.470 | 81.129 | 4245.571 |
| corruption/coco_cd/clean/severity0/trace/seed0 | 256 | False | 75.140 | 59.295 | 7940.823 |
| corruption/coco_cd/contrast/severity1/mlmp/seed0 | 256 | False | 70.360 | 48.279 | 2857.142 |
| corruption/coco_cd/contrast/severity1/noadapt/seed0 | 256 | False | 59.580 | 1.810 | 1847.031 |
| corruption/coco_cd/contrast/severity1/tent/seed0 | 256 | False | 71.690 | 43.227 | 2762.586 |
| corruption/coco_cd/contrast/severity1/tpt/seed0 | 256 | False | 68.050 | 80.804 | 4245.571 |
| corruption/coco_cd/contrast/severity1/trace/seed0 | 256 | False | 75.150 | 58.869 | 7940.813 |
| corruption/coco_cd/contrast/severity3/mlmp/seed0 | 256 | False | 68.410 | 48.390 | 2857.142 |
| corruption/coco_cd/contrast/severity3/noadapt/seed0 | 256 | False | 58.450 | 1.813 | 1847.031 |
| corruption/coco_cd/contrast/severity3/tent/seed0 | 256 | False | 71.130 | 43.273 | 2762.586 |
| corruption/coco_cd/contrast/severity3/tpt/seed0 | 256 | False | 67.650 | 80.512 | 4245.571 |
| corruption/coco_cd/contrast/severity3/trace/seed0 | 256 | False | 73.380 | 58.677 | 7940.807 |
| corruption/coco_cd/contrast/severity5/mlmp/seed0 | 256 | False | 62.110 | 48.369 | 2857.142 |
| corruption/coco_cd/contrast/severity5/noadapt/seed0 | 256 | False | 55.160 | 1.810 | 1847.031 |
| corruption/coco_cd/contrast/severity5/tent/seed0 | 256 | False | 64.520 | 43.270 | 2762.586 |
| corruption/coco_cd/contrast/severity5/tpt/seed0 | 256 | False | 62.280 | 80.945 | 4245.571 |
| corruption/coco_cd/contrast/severity5/trace/seed0 | 256 | False | 60.510 | 57.026 | 7940.801 |
| corruption/coco_cd/defocus_blur/severity1/mlmp/seed0 | 256 | False | 69.210 | 48.327 | 2857.142 |
| corruption/coco_cd/defocus_blur/severity1/noadapt/seed0 | 256 | False | 59.140 | 1.812 | 1847.031 |
| corruption/coco_cd/defocus_blur/severity1/tent/seed0 | 256 | False | 71.120 | 43.267 | 2762.586 |
| corruption/coco_cd/defocus_blur/severity1/tpt/seed0 | 256 | False | 68.370 | 80.580 | 4245.571 |
| corruption/coco_cd/defocus_blur/severity1/trace/seed0 | 256 | False | 73.030 | 59.208 | 7940.810 |
| corruption/coco_cd/defocus_blur/severity3/mlmp/seed0 | 256 | False | 63.270 | 48.291 | 2857.142 |
| corruption/coco_cd/defocus_blur/severity3/noadapt/seed0 | 256 | False | 55.980 | 1.814 | 1847.031 |
| corruption/coco_cd/defocus_blur/severity3/tent/seed0 | 256 | False | 67.100 | 43.461 | 2762.586 |
| corruption/coco_cd/defocus_blur/severity3/tpt/seed0 | 256 | False | 62.000 | 80.643 | 4245.571 |
| corruption/coco_cd/defocus_blur/severity3/trace/seed0 | 256 | False | 62.200 | 59.555 | 7940.812 |
| corruption/coco_cd/defocus_blur/severity5/mlmp/seed0 | 256 | False | 52.510 | 48.328 | 2857.142 |
| corruption/coco_cd/defocus_blur/severity5/noadapt/seed0 | 256 | False | 50.030 | 1.812 | 1847.031 |
| corruption/coco_cd/defocus_blur/severity5/tent/seed0 | 256 | False | 54.210 | 43.288 | 2762.586 |
| corruption/coco_cd/defocus_blur/severity5/tpt/seed0 | 256 | False | 50.890 | 80.533 | 4245.571 |
| corruption/coco_cd/defocus_blur/severity5/trace/seed0 | 256 | False | 42.670 | 57.907 | 7940.812 |
| corruption/coco_cd/gaussian_noise/severity1/mlmp/seed0 | 256 | False | 67.770 | 48.389 | 2857.142 |
| corruption/coco_cd/gaussian_noise/severity1/noadapt/seed0 | 256 | False | 56.790 | 1.810 | 1847.031 |
| corruption/coco_cd/gaussian_noise/severity1/tent/seed0 | 256 | False | 70.820 | 43.469 | 2762.586 |
| corruption/coco_cd/gaussian_noise/severity1/tpt/seed0 | 256 | False | 67.730 | 80.747 | 4245.571 |
| corruption/coco_cd/gaussian_noise/severity1/trace/seed0 | 256 | False | 72.770 | 59.372 | 7940.815 |
| corruption/coco_cd/gaussian_noise/severity3/mlmp/seed0 | 256 | False | 64.080 | 48.390 | 2857.142 |
| corruption/coco_cd/gaussian_noise/severity3/noadapt/seed0 | 256 | False | 53.210 | 1.808 | 1847.031 |
| corruption/coco_cd/gaussian_noise/severity3/tent/seed0 | 256 | False | 65.100 | 43.522 | 2762.586 |
| corruption/coco_cd/gaussian_noise/severity3/tpt/seed0 | 256 | False | 60.860 | 80.729 | 4245.571 |
| corruption/coco_cd/gaussian_noise/severity3/trace/seed0 | 256 | False | 60.180 | 57.423 | 7940.805 |
| corruption/coco_cd/gaussian_noise/severity5/mlmp/seed0 | 256 | False | 51.420 | 48.398 | 2857.142 |
| corruption/coco_cd/gaussian_noise/severity5/noadapt/seed0 | 256 | False | 42.200 | 1.814 | 1847.031 |
| corruption/coco_cd/gaussian_noise/severity5/tent/seed0 | 256 | False | 51.330 | 43.529 | 2762.586 |
| corruption/coco_cd/gaussian_noise/severity5/tpt/seed0 | 256 | False | 42.290 | 81.063 | 4245.571 |
| corruption/coco_cd/gaussian_noise/severity5/trace/seed0 | 256 | False | 30.960 | 54.134 | 7940.805 |
| corruption/coco_cd/jpeg_compression/severity1/mlmp/seed0 | 256 | False | 68.250 | 48.404 | 2857.142 |
| corruption/coco_cd/jpeg_compression/severity1/noadapt/seed0 | 256 | False | 58.350 | 1.808 | 1847.031 |
| corruption/coco_cd/jpeg_compression/severity1/tent/seed0 | 256 | False | 71.460 | 43.464 | 2762.586 |
| corruption/coco_cd/jpeg_compression/severity1/tpt/seed0 | 256 | False | 67.000 | 80.947 | 4245.571 |
| corruption/coco_cd/jpeg_compression/severity1/trace/seed0 | 256 | False | 70.460 | 59.908 | 7940.810 |
| corruption/coco_cd/jpeg_compression/severity3/mlmp/seed0 | 256 | False | 65.060 | 48.384 | 2857.142 |
| corruption/coco_cd/jpeg_compression/severity3/noadapt/seed0 | 256 | False | 56.570 | 1.805 | 1847.031 |
| corruption/coco_cd/jpeg_compression/severity3/tent/seed0 | 256 | False | 67.870 | 43.468 | 2762.586 |
| corruption/coco_cd/jpeg_compression/severity3/tpt/seed0 | 256 | False | 65.270 | 80.878 | 4245.571 |
| corruption/coco_cd/jpeg_compression/severity3/trace/seed0 | 256 | False | 68.800 | 59.812 | 7940.810 |
| corruption/coco_cd/jpeg_compression/severity5/mlmp/seed0 | 256 | False | 56.030 | 48.385 | 2857.142 |
| corruption/coco_cd/jpeg_compression/severity5/noadapt/seed0 | 256 | False | 52.630 | 1.812 | 1847.031 |
| corruption/coco_cd/jpeg_compression/severity5/tent/seed0 | 256 | False | 59.820 | 43.507 | 2762.586 |
| corruption/coco_cd/jpeg_compression/severity5/tpt/seed0 | 256 | False | 53.010 | 80.930 | 4245.571 |
| corruption/coco_cd/jpeg_compression/severity5/trace/seed0 | 256 | False | 58.420 | 58.097 | 7940.802 |
| corruption/coco_cd/shot_noise/severity1/mlmp/seed0 | 256 | False | 69.730 | 48.171 | 2857.142 |
| corruption/coco_cd/shot_noise/severity1/noadapt/seed0 | 256 | False | 57.730 | 1.817 | 1847.031 |
| corruption/coco_cd/shot_noise/severity1/tent/seed0 | 256 | False | 70.520 | 43.488 | 2762.586 |
| corruption/coco_cd/shot_noise/severity1/tpt/seed0 | 256 | False | 68.510 | 80.929 | 4245.571 |
| corruption/coco_cd/shot_noise/severity1/trace/seed0 | 256 | False | 71.850 | 58.809 | 7940.813 |
| corruption/coco_cd/shot_noise/severity3/mlmp/seed0 | 256 | False | 64.400 | 48.249 | 2857.142 |
| corruption/coco_cd/shot_noise/severity3/noadapt/seed0 | 256 | False | 52.270 | 1.814 | 1847.031 |
| corruption/coco_cd/shot_noise/severity3/tent/seed0 | 256 | False | 63.250 | 43.357 | 2762.586 |
| corruption/coco_cd/shot_noise/severity3/tpt/seed0 | 256 | False | 62.540 | 80.571 | 4245.571 |
| corruption/coco_cd/shot_noise/severity3/trace/seed0 | 256 | False | 60.980 | 57.637 | 7940.807 |
| corruption/coco_cd/shot_noise/severity5/mlmp/seed0 | 256 | False | 52.560 | 48.267 | 2857.142 |
| corruption/coco_cd/shot_noise/severity5/noadapt/seed0 | 256 | False | 45.340 | 1.812 | 1847.031 |
| corruption/coco_cd/shot_noise/severity5/tent/seed0 | 256 | False | 54.380 | 43.446 | 2762.586 |
| corruption/coco_cd/shot_noise/severity5/tpt/seed0 | 256 | False | 45.620 | 80.881 | 4245.571 |
| corruption/coco_cd/shot_noise/severity5/trace/seed0 | 256 | False | 38.110 | 55.720 | 7940.809 |
| corruption/coco_object/brightness/severity1/mlmp/seed0 | 256 | False | 28.340 | 78.324 | 2410.620 |
| corruption/coco_object/brightness/severity1/noadapt/seed0 | 256 | False | 23.310 | 3.188 | 1883.514 |
| corruption/coco_object/brightness/severity1/tent/seed0 | 256 | False | 24.070 | 69.780 | 2362.018 |
| corruption/coco_object/brightness/severity1/tpt/seed0 | 256 | False | 23.240 | 323.018 | 7090.883 |
| corruption/coco_object/brightness/severity1/trace/seed0 | 256 | False | 30.770 | 440.602 | 4754.475 |
| corruption/coco_object/brightness/severity3/mlmp/seed0 | 256 | False | 27.510 | 78.233 | 2410.620 |
| corruption/coco_object/brightness/severity3/noadapt/seed0 | 256 | False | 22.710 | 3.188 | 1883.514 |
| corruption/coco_object/brightness/severity3/tent/seed0 | 256 | False | 22.890 | 69.816 | 2362.018 |
| corruption/coco_object/brightness/severity3/tpt/seed0 | 256 | False | 23.340 | 322.817 | 7090.883 |
| corruption/coco_object/brightness/severity3/trace/seed0 | 256 | False | 29.810 | 454.734 | 4754.474 |
| corruption/coco_object/brightness/severity5/mlmp/seed0 | 256 | False | 28.000 | 78.178 | 2410.620 |
| corruption/coco_object/brightness/severity5/noadapt/seed0 | 256 | False | 22.360 | 3.316 | 1883.514 |
| corruption/coco_object/brightness/severity5/tent/seed0 | 256 | False | 22.640 | 73.591 | 2362.018 |
| corruption/coco_object/brightness/severity5/tpt/seed0 | 256 | False | 22.630 | 323.642 | 7090.883 |
| corruption/coco_object/brightness/severity5/trace/seed0 | 256 | False | 28.620 | 442.632 | 4754.473 |
| corruption/coco_object/clean/severity0/mlmp/seed0 | 256 | False | 28.760 | 77.902 | 2410.620 |
| corruption/coco_object/clean/severity0/noadapt/seed0 | 256 | False | 23.140 | 3.118 | 1883.514 |
| corruption/coco_object/clean/severity0/tent/seed0 | 256 | False | 23.800 | 69.772 | 2362.018 |
| corruption/coco_object/clean/severity0/tpt/seed0 | 256 | False | 23.630 | 323.023 | 7090.883 |
| corruption/coco_object/clean/severity0/trace/seed0 | 256 | False | 30.640 | 441.363 | 4754.474 |
| corruption/coco_object/contrast/severity1/mlmp/seed0 | 256 | False | 28.360 | 78.023 | 2410.620 |
| corruption/coco_object/contrast/severity1/noadapt/seed0 | 256 | False | 22.760 | 3.139 | 1883.514 |
| corruption/coco_object/contrast/severity1/tent/seed0 | 256 | False | 23.310 | 69.749 | 2362.018 |
| corruption/coco_object/contrast/severity1/tpt/seed0 | 256 | False | 23.290 | 324.133 | 7090.883 |
| corruption/coco_object/contrast/severity1/trace/seed0 | 256 | False | 28.870 | 441.507 | 4754.473 |
| corruption/coco_object/contrast/severity3/mlmp/seed0 | 256 | False | 28.450 | 78.095 | 2410.620 |
| corruption/coco_object/contrast/severity3/noadapt/seed0 | 256 | False | 22.390 | 3.134 | 1883.514 |
| corruption/coco_object/contrast/severity3/tent/seed0 | 256 | False | 22.380 | 69.844 | 2362.018 |
| corruption/coco_object/contrast/severity3/tpt/seed0 | 256 | False | 22.450 | 322.996 | 7090.883 |
| corruption/coco_object/contrast/severity3/trace/seed0 | 256 | False | 28.490 | 443.468 | 4754.474 |
| corruption/coco_object/contrast/severity5/mlmp/seed0 | 256 | False | 25.500 | 78.180 | 2410.620 |
| corruption/coco_object/contrast/severity5/noadapt/seed0 | 256 | False | 19.530 | 3.139 | 1883.514 |
| corruption/coco_object/contrast/severity5/tent/seed0 | 256 | False | 19.320 | 69.451 | 2362.018 |
| corruption/coco_object/contrast/severity5/tpt/seed0 | 256 | False | 19.430 | 323.019 | 7090.883 |
| corruption/coco_object/contrast/severity5/trace/seed0 | 256 | False | 24.600 | 437.652 | 4754.473 |
| corruption/coco_object/defocus_blur/severity1/mlmp/seed0 | 256 | False | 28.880 | 77.728 | 2410.620 |
| corruption/coco_object/defocus_blur/severity1/noadapt/seed0 | 256 | False | 22.300 | 3.147 | 1883.514 |
| corruption/coco_object/defocus_blur/severity1/tent/seed0 | 256 | False | 22.860 | 69.579 | 2362.018 |
| corruption/coco_object/defocus_blur/severity1/tpt/seed0 | 256 | False | 23.270 | 322.922 | 7090.883 |
| corruption/coco_object/defocus_blur/severity1/trace/seed0 | 256 | False | 29.070 | 440.670 | 4754.473 |
| corruption/coco_object/defocus_blur/severity3/mlmp/seed0 | 256 | False | 23.060 | 78.112 | 2410.620 |
| corruption/coco_object/defocus_blur/severity3/noadapt/seed0 | 256 | False | 18.840 | 3.148 | 1883.514 |
| corruption/coco_object/defocus_blur/severity3/tent/seed0 | 256 | False | 18.740 | 69.386 | 2362.018 |
| corruption/coco_object/defocus_blur/severity3/tpt/seed0 | 256 | False | 18.770 | 322.908 | 7090.883 |
| corruption/coco_object/defocus_blur/severity3/trace/seed0 | 256 | False | 21.160 | 437.047 | 4754.477 |
| corruption/coco_object/defocus_blur/severity5/mlmp/seed0 | 256 | False | 14.860 | 78.112 | 2410.620 |
| corruption/coco_object/defocus_blur/severity5/noadapt/seed0 | 256 | False | 13.270 | 3.145 | 1883.514 |
| corruption/coco_object/defocus_blur/severity5/tent/seed0 | 256 | False | 13.140 | 69.747 | 2362.018 |
| corruption/coco_object/defocus_blur/severity5/tpt/seed0 | 256 | False | 12.820 | 323.170 | 7090.883 |
| corruption/coco_object/defocus_blur/severity5/trace/seed0 | 256 | False | 14.340 | 443.724 | 4754.477 |
| corruption/coco_object/gaussian_noise/severity1/mlmp/seed0 | 256 | False | 29.220 | 78.087 | 2410.620 |
| corruption/coco_object/gaussian_noise/severity1/noadapt/seed0 | 256 | False | 22.130 | 3.133 | 1883.514 |
| corruption/coco_object/gaussian_noise/severity1/tent/seed0 | 256 | False | 23.730 | 69.673 | 2362.018 |
| corruption/coco_object/gaussian_noise/severity1/tpt/seed0 | 256 | False | 24.000 | 322.981 | 7090.883 |
| corruption/coco_object/gaussian_noise/severity1/trace/seed0 | 256 | False | 28.920 | 443.570 | 4754.475 |
| corruption/coco_object/gaussian_noise/severity3/mlmp/seed0 | 256 | False | 25.250 | 78.141 | 2410.620 |
| corruption/coco_object/gaussian_noise/severity3/noadapt/seed0 | 256 | False | 19.360 | 3.135 | 1883.514 |
| corruption/coco_object/gaussian_noise/severity3/tent/seed0 | 256 | False | 20.340 | 69.405 | 2362.018 |
| corruption/coco_object/gaussian_noise/severity3/tpt/seed0 | 256 | False | 20.440 | 322.991 | 7090.883 |
| corruption/coco_object/gaussian_noise/severity3/trace/seed0 | 256 | False | 22.920 | 448.194 | 4754.474 |
| corruption/coco_object/gaussian_noise/severity5/mlmp/seed0 | 256 | False | 15.860 | 77.965 | 2410.620 |
| corruption/coco_object/gaussian_noise/severity5/noadapt/seed0 | 256 | False | 12.500 | 3.135 | 1883.514 |
| corruption/coco_object/gaussian_noise/severity5/tent/seed0 | 256 | False | 12.860 | 69.896 | 2362.018 |
| corruption/coco_object/gaussian_noise/severity5/tpt/seed0 | 256 | False | 11.870 | 322.945 | 7090.883 |
| corruption/coco_object/gaussian_noise/severity5/trace/seed0 | 256 | False | 12.790 | 442.907 | 4754.472 |
| corruption/coco_object/jpeg_compression/severity1/mlmp/seed0 | 256 | False | 26.890 | 77.935 | 2410.620 |
| corruption/coco_object/jpeg_compression/severity1/noadapt/seed0 | 256 | False | 22.780 | 3.122 | 1883.514 |
| corruption/coco_object/jpeg_compression/severity1/tent/seed0 | 256 | False | 23.880 | 69.698 | 2362.018 |
| corruption/coco_object/jpeg_compression/severity1/tpt/seed0 | 256 | False | 23.730 | 322.939 | 7090.883 |
| corruption/coco_object/jpeg_compression/severity1/trace/seed0 | 256 | False | 28.620 | 441.164 | 4754.475 |
| corruption/coco_object/jpeg_compression/severity3/mlmp/seed0 | 256 | False | 27.820 | 77.801 | 2410.620 |
| corruption/coco_object/jpeg_compression/severity3/noadapt/seed0 | 256 | False | 21.390 | 3.118 | 1883.514 |
| corruption/coco_object/jpeg_compression/severity3/tent/seed0 | 256 | False | 21.220 | 69.754 | 2362.018 |
| corruption/coco_object/jpeg_compression/severity3/tpt/seed0 | 256 | False | 21.540 | 323.135 | 7090.883 |
| corruption/coco_object/jpeg_compression/severity3/trace/seed0 | 256 | False | 26.980 | 441.213 | 4754.474 |
| corruption/coco_object/jpeg_compression/severity5/mlmp/seed0 | 256 | False | 23.430 | 77.963 | 2410.620 |
| corruption/coco_object/jpeg_compression/severity5/noadapt/seed0 | 256 | False | 17.250 | 3.129 | 1883.514 |
| corruption/coco_object/jpeg_compression/severity5/tent/seed0 | 256 | False | 17.110 | 71.437 | 2362.018 |
| corruption/coco_object/jpeg_compression/severity5/tpt/seed0 | 256 | False | 17.130 | 333.634 | 7090.883 |
| corruption/coco_object/jpeg_compression/severity5/trace/seed0 | 256 | False | 20.510 | 444.693 | 4754.473 |
| corruption/coco_object/shot_noise/severity1/mlmp/seed0 | 256 | False | 29.560 | 78.200 | 2410.620 |
| corruption/coco_object/shot_noise/severity1/noadapt/seed0 | 256 | False | 22.550 | 3.148 | 1883.514 |
| corruption/coco_object/shot_noise/severity1/tent/seed0 | 256 | False | 23.140 | 69.793 | 2362.018 |
| corruption/coco_object/shot_noise/severity1/tpt/seed0 | 256 | False | 23.350 | 323.002 | 7090.883 |
| corruption/coco_object/shot_noise/severity1/trace/seed0 | 256 | False | 28.410 | 448.048 | 4754.475 |
| corruption/coco_object/shot_noise/severity3/mlmp/seed0 | 256 | False | 25.080 | 78.205 | 2410.620 |
| corruption/coco_object/shot_noise/severity3/noadapt/seed0 | 256 | False | 19.090 | 3.153 | 1883.514 |
| corruption/coco_object/shot_noise/severity3/tent/seed0 | 256 | False | 19.160 | 69.737 | 2362.018 |
| corruption/coco_object/shot_noise/severity3/tpt/seed0 | 256 | False | 19.810 | 323.122 | 7090.883 |
| corruption/coco_object/shot_noise/severity3/trace/seed0 | 256 | False | 24.690 | 450.306 | 4754.474 |
| corruption/coco_object/shot_noise/severity5/mlmp/seed0 | 256 | False | 18.030 | 77.827 | 2410.620 |
| corruption/coco_object/shot_noise/severity5/noadapt/seed0 | 256 | False | 12.120 | 3.145 | 1883.514 |
| corruption/coco_object/shot_noise/severity5/tent/seed0 | 256 | False | 13.010 | 69.842 | 2362.018 |
| corruption/coco_object/shot_noise/severity5/tpt/seed0 | 256 | False | 12.490 | 323.043 | 7090.883 |
| corruption/coco_object/shot_noise/severity5/trace/seed0 | 256 | False | 13.330 | 447.056 | 4754.473 |
| corruption/pc60/brightness/severity1/mlmp/seed0 | 256 | False | 27.810 | 82.574 | 2389.962 |
| corruption/pc60/brightness/severity1/noadapt/seed0 | 256 | False | 25.290 | 3.167 | 1856.095 |
| corruption/pc60/brightness/severity1/tent/seed0 | 256 | False | 24.770 | 69.314 | 2340.892 |
| corruption/pc60/brightness/severity1/tpt/seed0 | 256 | False | 22.570 | 229.060 | 5404.416 |
| corruption/pc60/brightness/severity1/trace/seed0 | 256 | False | 28.120 | 241.301 | 3287.437 |
| corruption/pc60/brightness/severity3/mlmp/seed0 | 256 | False | 27.300 | 77.802 | 2389.962 |
| corruption/pc60/brightness/severity3/noadapt/seed0 | 256 | False | 25.310 | 3.170 | 1856.095 |
| corruption/pc60/brightness/severity3/tent/seed0 | 256 | False | 24.790 | 69.154 | 2340.892 |
| corruption/pc60/brightness/severity3/tpt/seed0 | 256 | False | 22.720 | 225.703 | 5404.416 |
| corruption/pc60/brightness/severity3/trace/seed0 | 256 | False | 28.180 | 226.746 | 3287.437 |
| corruption/pc60/brightness/severity5/mlmp/seed0 | 256 | False | 26.850 | 77.798 | 2389.962 |
| corruption/pc60/brightness/severity5/noadapt/seed0 | 256 | False | 24.840 | 3.166 | 1856.095 |
| corruption/pc60/brightness/severity5/tent/seed0 | 256 | False | 23.790 | 68.694 | 2340.892 |
| corruption/pc60/brightness/severity5/tpt/seed0 | 256 | False | 22.520 | 225.831 | 5404.416 |
| corruption/pc60/brightness/severity5/trace/seed0 | 256 | False | 26.530 | 223.879 | 3287.434 |
| corruption/pc60/clean/severity0/clipartt/seed0 | 256 | False | 22.140 | 1253.789 | 15102.607 |
| corruption/pc60/clean/severity0/mlmp/seed0 | 256 | False | 27.850 | 77.639 | 2389.962 |
| corruption/pc60/clean/severity0/noadapt/seed0 | 256 | False | 25.440 | 3.101 | 1856.095 |
| corruption/pc60/clean/severity0/tent/seed0 | 256 | False | 24.800 | 69.132 | 2340.892 |
| corruption/pc60/clean/severity0/tpt/seed0 | 256 | False | 23.110 | 227.565 | 5404.416 |
| corruption/pc60/clean/severity0/trace/seed0 | 256 | False | 28.290 | 234.875 | 3287.438 |
| corruption/pc60/clean/severity0/watt/seed0 | 256 | False | 14.900 | 952.297 | 2352.167 |
| corruption/pc60/contrast/severity1/mlmp/seed0 | 256 | False | 27.360 | 77.688 | 2389.962 |
| corruption/pc60/contrast/severity1/noadapt/seed0 | 256 | False | 25.080 | 3.112 | 1856.095 |
| corruption/pc60/contrast/severity1/tent/seed0 | 256 | False | 24.660 | 69.079 | 2340.892 |
| corruption/pc60/contrast/severity1/tpt/seed0 | 256 | False | 22.530 | 225.605 | 5404.416 |
| corruption/pc60/contrast/severity1/trace/seed0 | 256 | False | 28.050 | 224.112 | 3287.435 |
| corruption/pc60/contrast/severity3/mlmp/seed0 | 256 | False | 26.590 | 77.503 | 2389.962 |
| corruption/pc60/contrast/severity3/noadapt/seed0 | 256 | False | 24.470 | 3.102 | 1856.095 |
| corruption/pc60/contrast/severity3/tent/seed0 | 256 | False | 23.810 | 69.320 | 2340.892 |
| corruption/pc60/contrast/severity3/tpt/seed0 | 256 | False | 21.740 | 225.664 | 5404.416 |
| corruption/pc60/contrast/severity3/trace/seed0 | 256 | False | 26.560 | 224.434 | 3287.436 |
| corruption/pc60/contrast/severity5/mlmp/seed0 | 256 | False | 22.040 | 77.787 | 2389.962 |
| corruption/pc60/contrast/severity5/noadapt/seed0 | 256 | False | 18.990 | 3.103 | 1856.095 |
| corruption/pc60/contrast/severity5/tent/seed0 | 256 | False | 18.210 | 69.301 | 2340.892 |
| corruption/pc60/contrast/severity5/tpt/seed0 | 256 | False | 16.610 | 225.408 | 5404.416 |
| corruption/pc60/contrast/severity5/trace/seed0 | 256 | False | 21.000 | 223.222 | 3287.435 |
| corruption/pc60/defocus_blur/severity1/mlmp/seed0 | 256 | False | 26.280 | 77.690 | 2389.962 |
| corruption/pc60/defocus_blur/severity1/noadapt/seed0 | 256 | False | 24.390 | 3.112 | 1856.095 |
| corruption/pc60/defocus_blur/severity1/tent/seed0 | 256 | False | 22.460 | 69.322 | 2340.892 |
| corruption/pc60/defocus_blur/severity1/tpt/seed0 | 256 | False | 20.520 | 225.783 | 5404.416 |
| corruption/pc60/defocus_blur/severity1/trace/seed0 | 256 | False | 26.630 | 225.179 | 3287.436 |
| corruption/pc60/defocus_blur/severity3/mlmp/seed0 | 256 | False | 24.580 | 77.829 | 2389.962 |
| corruption/pc60/defocus_blur/severity3/noadapt/seed0 | 256 | False | 22.920 | 3.119 | 1856.095 |
| corruption/pc60/defocus_blur/severity3/tent/seed0 | 256 | False | 21.900 | 69.285 | 2340.892 |
| corruption/pc60/defocus_blur/severity3/tpt/seed0 | 256 | False | 20.500 | 225.693 | 5404.416 |
| corruption/pc60/defocus_blur/severity3/trace/seed0 | 256 | False | 23.490 | 225.211 | 3287.437 |
| corruption/pc60/defocus_blur/severity5/mlmp/seed0 | 256 | False | 18.490 | 77.697 | 2389.962 |
| corruption/pc60/defocus_blur/severity5/noadapt/seed0 | 256 | False | 18.130 | 3.118 | 1856.095 |
| corruption/pc60/defocus_blur/severity5/tent/seed0 | 256 | False | 16.830 | 69.150 | 2340.892 |
| corruption/pc60/defocus_blur/severity5/tpt/seed0 | 256 | False | 16.750 | 225.670 | 5404.416 |
| corruption/pc60/defocus_blur/severity5/trace/seed0 | 256 | False | 16.450 | 222.989 | 3287.438 |
| corruption/pc60/gaussian_noise/severity1/mlmp/seed0 | 256 | False | 27.600 | 82.034 | 2389.962 |
| corruption/pc60/gaussian_noise/severity1/noadapt/seed0 | 256 | False | 24.920 | 3.253 | 1856.095 |
| corruption/pc60/gaussian_noise/severity1/tent/seed0 | 256 | False | 24.040 | 73.544 | 2340.892 |
| corruption/pc60/gaussian_noise/severity1/tpt/seed0 | 256 | False | 22.190 | 241.900 | 5404.416 |
| corruption/pc60/gaussian_noise/severity1/trace/seed0 | 256 | False | 26.990 | 240.524 | 3287.436 |
| corruption/pc60/gaussian_noise/severity3/mlmp/seed0 | 256 | False | 25.530 | 82.534 | 2389.962 |
| corruption/pc60/gaussian_noise/severity3/noadapt/seed0 | 256 | False | 22.210 | 3.262 | 1856.095 |
| corruption/pc60/gaussian_noise/severity3/tent/seed0 | 256 | False | 21.510 | 73.052 | 2340.892 |
| corruption/pc60/gaussian_noise/severity3/tpt/seed0 | 256 | False | 19.270 | 242.433 | 5404.416 |
| corruption/pc60/gaussian_noise/severity3/trace/seed0 | 256 | False | 23.310 | 241.884 | 3287.436 |
| corruption/pc60/gaussian_noise/severity5/mlmp/seed0 | 256 | False | 16.030 | 77.614 | 2389.962 |
| corruption/pc60/gaussian_noise/severity5/noadapt/seed0 | 256 | False | 12.790 | 3.278 | 1856.095 |
| corruption/pc60/gaussian_noise/severity5/tent/seed0 | 256 | False | 13.010 | 69.170 | 2340.892 |
| corruption/pc60/gaussian_noise/severity5/tpt/seed0 | 256 | False | 9.960 | 225.715 | 5404.416 |
| corruption/pc60/gaussian_noise/severity5/trace/seed0 | 256 | False | 14.490 | 219.452 | 3287.434 |
| corruption/pc60/jpeg_compression/severity1/mlmp/seed0 | 256 | False | 27.010 | 77.417 | 2389.962 |
| corruption/pc60/jpeg_compression/severity1/noadapt/seed0 | 256 | False | 24.670 | 3.093 | 1856.095 |
| corruption/pc60/jpeg_compression/severity1/tent/seed0 | 256 | False | 23.600 | 68.864 | 2340.892 |
| corruption/pc60/jpeg_compression/severity1/tpt/seed0 | 256 | False | 21.940 | 225.461 | 5404.416 |
| corruption/pc60/jpeg_compression/severity1/trace/seed0 | 256 | False | 27.080 | 226.182 | 3287.436 |
| corruption/pc60/jpeg_compression/severity3/mlmp/seed0 | 256 | False | 25.870 | 77.620 | 2389.962 |
| corruption/pc60/jpeg_compression/severity3/noadapt/seed0 | 256 | False | 22.890 | 3.099 | 1856.095 |
| corruption/pc60/jpeg_compression/severity3/tent/seed0 | 256 | False | 21.920 | 69.160 | 2340.892 |
| corruption/pc60/jpeg_compression/severity3/tpt/seed0 | 256 | False | 19.060 | 225.274 | 5404.416 |
| corruption/pc60/jpeg_compression/severity3/trace/seed0 | 256 | False | 26.100 | 226.579 | 3287.435 |
| corruption/pc60/jpeg_compression/severity5/mlmp/seed0 | 256 | False | 21.000 | 77.369 | 2389.962 |
| corruption/pc60/jpeg_compression/severity5/noadapt/seed0 | 256 | False | 18.920 | 3.093 | 1856.095 |
| corruption/pc60/jpeg_compression/severity5/tent/seed0 | 256 | False | 18.760 | 69.137 | 2340.892 |
| corruption/pc60/jpeg_compression/severity5/tpt/seed0 | 256 | False | 16.280 | 225.725 | 5404.416 |
| corruption/pc60/jpeg_compression/severity5/trace/seed0 | 256 | False | 20.390 | 225.797 | 3287.436 |
| corruption/pc60/shot_noise/severity1/mlmp/seed0 | 256 | False | 26.950 | 77.748 | 2389.962 |
| corruption/pc60/shot_noise/severity1/noadapt/seed0 | 256 | False | 24.940 | 3.112 | 1856.095 |
| corruption/pc60/shot_noise/severity1/tent/seed0 | 256 | False | 24.340 | 69.319 | 2340.892 |
| corruption/pc60/shot_noise/severity1/tpt/seed0 | 256 | False | 22.400 | 225.507 | 5404.416 |
| corruption/pc60/shot_noise/severity1/trace/seed0 | 256 | False | 26.610 | 226.480 | 3287.437 |
| corruption/pc60/shot_noise/severity3/mlmp/seed0 | 256 | False | 25.030 | 77.849 | 2389.962 |
| corruption/pc60/shot_noise/severity3/noadapt/seed0 | 256 | False | 22.220 | 3.118 | 1856.095 |
| corruption/pc60/shot_noise/severity3/tent/seed0 | 256 | False | 22.080 | 69.323 | 2340.892 |
| corruption/pc60/shot_noise/severity3/tpt/seed0 | 256 | False | 19.700 | 225.716 | 5404.416 |
| corruption/pc60/shot_noise/severity3/trace/seed0 | 256 | False | 23.510 | 226.309 | 3287.435 |
| corruption/pc60/shot_noise/severity5/mlmp/seed0 | 256 | False | 17.070 | 77.698 | 2389.962 |
| corruption/pc60/shot_noise/severity5/noadapt/seed0 | 256 | False | 14.840 | 3.107 | 1856.095 |
| corruption/pc60/shot_noise/severity5/tent/seed0 | 256 | False | 14.900 | 69.315 | 2340.892 |
| corruption/pc60/shot_noise/severity5/tpt/seed0 | 256 | False | 12.160 | 225.607 | 5404.416 |
| corruption/pc60/shot_noise/severity5/trace/seed0 | 256 | False | 15.330 | 219.830 | 3287.434 |
| corruption/voc21/brightness/severity1/clipartt/seed0 | 256 | False | 37.970 | 1250.588 | 15010.891 |
| corruption/voc21/brightness/severity1/mlmp/seed0 | 256 | False | 50.550 | 77.695 | 2382.938 |
| corruption/voc21/brightness/severity1/noadapt/seed0 | 256 | False | 44.360 | 3.168 | 1851.113 |
| corruption/voc21/brightness/severity1/tent/seed0 | 256 | False | 45.230 | 69.611 | 2335.847 |
| corruption/voc21/brightness/severity1/tpt/seed0 | 256 | False | 44.120 | 219.766 | 5249.635 |
| corruption/voc21/brightness/severity1/trace/seed0 | 256 | False | 53.190 | 431.855 | 5102.815 |
| corruption/voc21/brightness/severity1/watt/seed0 | 256 | False | 27.590 | 911.100 | 2347.087 |
| corruption/voc21/brightness/severity3/clipartt/seed0 | 256 | False | 37.680 | 1252.856 | 15010.891 |
| corruption/voc21/brightness/severity3/mlmp/seed0 | 256 | False | 50.020 | 78.074 | 2382.938 |
| corruption/voc21/brightness/severity3/noadapt/seed0 | 256 | False | 43.760 | 3.155 | 1851.113 |
| corruption/voc21/brightness/severity3/tent/seed0 | 256 | False | 44.950 | 69.552 | 2335.847 |
| corruption/voc21/brightness/severity3/tpt/seed0 | 256 | False | 44.240 | 219.633 | 5249.635 |
| corruption/voc21/brightness/severity3/trace/seed0 | 256 | False | 52.480 | 435.474 | 5102.814 |
| corruption/voc21/brightness/severity3/watt/seed0 | 256 | False | 25.640 | 910.946 | 2347.087 |
| corruption/voc21/brightness/severity5/clipartt/seed0 | 256 | False | 36.890 | 1250.731 | 15010.891 |
| corruption/voc21/brightness/severity5/mlmp/seed0 | 256 | False | 51.220 | 77.281 | 2382.938 |
| corruption/voc21/brightness/severity5/noadapt/seed0 | 256 | False | 43.500 | 3.174 | 1851.113 |
| corruption/voc21/brightness/severity5/tent/seed0 | 256 | False | 44.360 | 69.105 | 2335.847 |
| corruption/voc21/brightness/severity5/tpt/seed0 | 256 | False | 42.440 | 219.656 | 5249.635 |
| corruption/voc21/brightness/severity5/trace/seed0 | 256 | False | 51.600 | 439.506 | 5102.816 |
| corruption/voc21/brightness/severity5/watt/seed0 | 256 | False | 26.150 | 911.006 | 2347.087 |
| corruption/voc21/clean/severity0/clipartt/seed0 | 256 | False | 37.480 | 1247.838 | 15010.891 |
| corruption/voc21/clean/severity0/mlmp/seed0 | 256 | False | 50.310 | 77.535 | 2382.938 |
| corruption/voc21/clean/severity0/noadapt/seed0 | 256 | False | 44.090 | 3.089 | 1851.113 |
| corruption/voc21/clean/severity0/tent/seed0 | 256 | False | 44.540 | 69.372 | 2335.847 |
| corruption/voc21/clean/severity0/tpt/seed0 | 256 | False | 43.830 | 219.832 | 5249.635 |
| corruption/voc21/clean/severity0/trace/seed0 | 256 | False | 53.190 | 433.358 | 5102.818 |
| corruption/voc21/clean/severity0/watt/seed0 | 256 | False | 27.080 | 910.559 | 2347.087 |
| corruption/voc21/contrast/severity1/clipartt/seed0 | 256 | False | 37.200 | 1275.460 | 15010.891 |
| corruption/voc21/contrast/severity1/mlmp/seed0 | 256 | False | 50.140 | 81.598 | 2382.938 |
| corruption/voc21/contrast/severity1/noadapt/seed0 | 256 | False | 43.480 | 3.117 | 1851.113 |
| corruption/voc21/contrast/severity1/tent/seed0 | 256 | False | 44.230 | 69.035 | 2335.847 |
| corruption/voc21/contrast/severity1/tpt/seed0 | 256 | False | 43.490 | 219.585 | 5249.635 |
| corruption/voc21/contrast/severity1/trace/seed0 | 256 | False | 51.970 | 435.334 | 5102.818 |
| corruption/voc21/contrast/severity1/watt/seed0 | 256 | False | 27.560 | 910.474 | 2347.087 |
| corruption/voc21/contrast/severity3/clipartt/seed0 | 256 | False | 36.840 | 1250.446 | 15010.891 |
| corruption/voc21/contrast/severity3/mlmp/seed0 | 256 | False | 49.430 | 77.584 | 2382.938 |
| corruption/voc21/contrast/severity3/noadapt/seed0 | 256 | False | 43.200 | 3.112 | 1851.113 |
| corruption/voc21/contrast/severity3/tent/seed0 | 256 | False | 43.810 | 69.039 | 2335.847 |
| corruption/voc21/contrast/severity3/tpt/seed0 | 256 | False | 43.400 | 219.591 | 5249.635 |
| corruption/voc21/contrast/severity3/trace/seed0 | 256 | False | 51.100 | 436.921 | 5102.818 |
| corruption/voc21/contrast/severity3/watt/seed0 | 256 | False | 27.590 | 910.585 | 2347.087 |
| corruption/voc21/contrast/severity5/clipartt/seed0 | 256 | False | 33.150 | 1252.721 | 15010.891 |
| corruption/voc21/contrast/severity5/mlmp/seed0 | 256 | False | 43.730 | 77.492 | 2382.938 |
| corruption/voc21/contrast/severity5/noadapt/seed0 | 256 | False | 39.840 | 3.111 | 1851.113 |
| corruption/voc21/contrast/severity5/tent/seed0 | 256 | False | 39.620 | 69.396 | 2335.847 |
| corruption/voc21/contrast/severity5/tpt/seed0 | 256 | False | 39.600 | 219.590 | 5249.635 |
| corruption/voc21/contrast/severity5/trace/seed0 | 256 | False | 47.420 | 439.182 | 5102.817 |
| corruption/voc21/contrast/severity5/watt/seed0 | 256 | False | 24.370 | 910.236 | 2347.087 |
| corruption/voc21/defocus_blur/severity1/clipartt/seed0 | 256 | False | 36.580 | 1248.279 | 15010.891 |
| corruption/voc21/defocus_blur/severity1/mlmp/seed0 | 256 | False | 51.310 | 77.653 | 2382.938 |
| corruption/voc21/defocus_blur/severity1/noadapt/seed0 | 256 | False | 43.830 | 3.189 | 1851.113 |
| corruption/voc21/defocus_blur/severity1/tent/seed0 | 256 | False | 44.480 | 69.340 | 2335.847 |
| corruption/voc21/defocus_blur/severity1/tpt/seed0 | 256 | False | 43.820 | 219.480 | 5249.635 |
| corruption/voc21/defocus_blur/severity1/trace/seed0 | 256 | False | 52.580 | 434.297 | 5102.818 |
| corruption/voc21/defocus_blur/severity1/watt/seed0 | 256 | False | 25.740 | 910.577 | 2347.087 |
| corruption/voc21/defocus_blur/severity3/clipartt/seed0 | 256 | False | 33.260 | 1253.225 | 15010.891 |
| corruption/voc21/defocus_blur/severity3/mlmp/seed0 | 256 | False | 45.620 | 77.592 | 2382.938 |
| corruption/voc21/defocus_blur/severity3/noadapt/seed0 | 256 | False | 41.420 | 3.137 | 1851.113 |
| corruption/voc21/defocus_blur/severity3/tent/seed0 | 256 | False | 41.330 | 69.355 | 2335.847 |
| corruption/voc21/defocus_blur/severity3/tpt/seed0 | 256 | False | 40.310 | 219.741 | 5249.635 |
| corruption/voc21/defocus_blur/severity3/trace/seed0 | 256 | False | 48.010 | 435.011 | 5102.819 |
| corruption/voc21/defocus_blur/severity3/watt/seed0 | 256 | False | 24.480 | 911.077 | 2347.087 |
| corruption/voc21/defocus_blur/severity5/clipartt/seed0 | 256 | False | 26.740 | 1249.974 | 15010.891 |
| corruption/voc21/defocus_blur/severity5/mlmp/seed0 | 256 | False | 38.230 | 77.731 | 2382.938 |
| corruption/voc21/defocus_blur/severity5/noadapt/seed0 | 256 | False | 33.340 | 3.116 | 1851.113 |
| corruption/voc21/defocus_blur/severity5/tent/seed0 | 256 | False | 32.490 | 68.937 | 2335.847 |
| corruption/voc21/defocus_blur/severity5/tpt/seed0 | 256 | False | 32.800 | 219.591 | 5249.635 |
| corruption/voc21/defocus_blur/severity5/trace/seed0 | 256 | False | 37.570 | 444.526 | 5102.820 |
| corruption/voc21/defocus_blur/severity5/watt/seed0 | 256 | False | 19.300 | 910.708 | 2347.087 |
| corruption/voc21/gaussian_noise/severity1/clipartt/seed0 | 256 | False | 38.040 | 1252.111 | 15010.891 |
| corruption/voc21/gaussian_noise/severity1/mlmp/seed0 | 256 | False | 51.040 | 77.693 | 2382.938 |
| corruption/voc21/gaussian_noise/severity1/noadapt/seed0 | 256 | False | 44.620 | 3.110 | 1851.113 |
| corruption/voc21/gaussian_noise/severity1/tent/seed0 | 256 | False | 44.820 | 69.119 | 2335.847 |
| corruption/voc21/gaussian_noise/severity1/tpt/seed0 | 256 | False | 44.180 | 219.672 | 5249.635 |
| corruption/voc21/gaussian_noise/severity1/trace/seed0 | 256 | False | 53.010 | 435.871 | 5102.817 |
| corruption/voc21/gaussian_noise/severity1/watt/seed0 | 256 | False | 26.660 | 910.606 | 2347.087 |
| corruption/voc21/gaussian_noise/severity3/clipartt/seed0 | 256 | False | 36.030 | 1251.570 | 15010.891 |
| corruption/voc21/gaussian_noise/severity3/mlmp/seed0 | 256 | False | 47.880 | 77.666 | 2382.938 |
| corruption/voc21/gaussian_noise/severity3/noadapt/seed0 | 256 | False | 42.360 | 3.114 | 1851.113 |
| corruption/voc21/gaussian_noise/severity3/tent/seed0 | 256 | False | 41.250 | 69.297 | 2335.847 |
| corruption/voc21/gaussian_noise/severity3/tpt/seed0 | 256 | False | 41.010 | 219.696 | 5249.635 |
| corruption/voc21/gaussian_noise/severity3/trace/seed0 | 256 | False | 48.620 | 452.122 | 5102.819 |
| corruption/voc21/gaussian_noise/severity3/watt/seed0 | 256 | False | 24.990 | 909.936 | 2347.087 |
| corruption/voc21/gaussian_noise/severity5/clipartt/seed0 | 256 | False | 23.090 | 1250.225 | 15010.891 |
| corruption/voc21/gaussian_noise/severity5/mlmp/seed0 | 256 | False | 35.500 | 77.742 | 2382.938 |
| corruption/voc21/gaussian_noise/severity5/noadapt/seed0 | 256 | False | 28.350 | 3.116 | 1851.113 |
| corruption/voc21/gaussian_noise/severity5/tent/seed0 | 256 | False | 28.320 | 69.206 | 2335.847 |
| corruption/voc21/gaussian_noise/severity5/tpt/seed0 | 256 | False | 26.060 | 219.621 | 5249.635 |
| corruption/voc21/gaussian_noise/severity5/trace/seed0 | 256 | False | 33.690 | 454.016 | 5102.813 |
| corruption/voc21/gaussian_noise/severity5/watt/seed0 | 256 | False | 14.170 | 910.817 | 2347.087 |
| corruption/voc21/jpeg_compression/severity1/clipartt/seed0 | 256 | False | 38.530 | 1251.945 | 15010.891 |
| corruption/voc21/jpeg_compression/severity1/mlmp/seed0 | 256 | False | 50.370 | 77.289 | 2382.938 |
| corruption/voc21/jpeg_compression/severity1/noadapt/seed0 | 256 | False | 44.650 | 3.090 | 1851.113 |
| corruption/voc21/jpeg_compression/severity1/tent/seed0 | 256 | False | 45.160 | 69.395 | 2335.847 |
| corruption/voc21/jpeg_compression/severity1/tpt/seed0 | 256 | False | 45.470 | 219.623 | 5249.635 |
| corruption/voc21/jpeg_compression/severity1/trace/seed0 | 256 | False | 53.120 | 433.385 | 5102.819 |
| corruption/voc21/jpeg_compression/severity1/watt/seed0 | 256 | False | 27.000 | 910.586 | 2347.087 |
| corruption/voc21/jpeg_compression/severity3/clipartt/seed0 | 256 | False | 37.070 | 1250.793 | 15010.891 |
| corruption/voc21/jpeg_compression/severity3/mlmp/seed0 | 256 | False | 49.450 | 77.516 | 2382.938 |
| corruption/voc21/jpeg_compression/severity3/noadapt/seed0 | 256 | False | 43.780 | 3.101 | 1851.113 |
| corruption/voc21/jpeg_compression/severity3/tent/seed0 | 256 | False | 43.380 | 69.308 | 2335.847 |
| corruption/voc21/jpeg_compression/severity3/tpt/seed0 | 256 | False | 42.260 | 219.522 | 5249.635 |
| corruption/voc21/jpeg_compression/severity3/trace/seed0 | 256 | False | 52.300 | 433.598 | 5102.815 |
| corruption/voc21/jpeg_compression/severity3/watt/seed0 | 256 | False | 25.570 | 909.396 | 2347.087 |
| corruption/voc21/jpeg_compression/severity5/clipartt/seed0 | 256 | False | 30.740 | 1253.944 | 15010.891 |
| corruption/voc21/jpeg_compression/severity5/mlmp/seed0 | 256 | False | 42.370 | 77.162 | 2382.938 |
| corruption/voc21/jpeg_compression/severity5/noadapt/seed0 | 256 | False | 37.210 | 3.120 | 1851.113 |
| corruption/voc21/jpeg_compression/severity5/tent/seed0 | 256 | False | 34.680 | 68.969 | 2335.847 |
| corruption/voc21/jpeg_compression/severity5/tpt/seed0 | 256 | False | 34.940 | 219.519 | 5249.635 |
| corruption/voc21/jpeg_compression/severity5/trace/seed0 | 256 | False | 43.640 | 457.598 | 5102.816 |
| corruption/voc21/jpeg_compression/severity5/watt/seed0 | 256 | False | 22.310 | 910.036 | 2347.087 |
| corruption/voc21/shot_noise/severity1/clipartt/seed0 | 256 | False | 38.510 | 1249.203 | 15010.891 |
| corruption/voc21/shot_noise/severity1/mlmp/seed0 | 256 | False | 51.270 | 77.662 | 2382.938 |
| corruption/voc21/shot_noise/severity1/noadapt/seed0 | 256 | False | 44.680 | 3.121 | 1851.113 |
| corruption/voc21/shot_noise/severity1/tent/seed0 | 256 | False | 44.480 | 69.293 | 2335.847 |
| corruption/voc21/shot_noise/severity1/tpt/seed0 | 256 | False | 43.970 | 219.426 | 5249.635 |
| corruption/voc21/shot_noise/severity1/trace/seed0 | 256 | False | 52.210 | 442.326 | 5102.820 |
| corruption/voc21/shot_noise/severity1/watt/seed0 | 256 | False | 26.460 | 909.480 | 2347.087 |
| corruption/voc21/shot_noise/severity3/clipartt/seed0 | 256 | False | 35.570 | 1252.597 | 15010.891 |
| corruption/voc21/shot_noise/severity3/mlmp/seed0 | 256 | False | 45.470 | 77.419 | 2382.938 |
| corruption/voc21/shot_noise/severity3/noadapt/seed0 | 256 | False | 41.350 | 3.121 | 1851.113 |
| corruption/voc21/shot_noise/severity3/tent/seed0 | 256 | False | 41.070 | 69.346 | 2335.847 |
| corruption/voc21/shot_noise/severity3/tpt/seed0 | 256 | False | 40.360 | 219.617 | 5249.635 |
| corruption/voc21/shot_noise/severity3/trace/seed0 | 256 | False | 48.630 | 450.908 | 5102.818 |
| corruption/voc21/shot_noise/severity3/watt/seed0 | 256 | False | 25.950 | 955.909 | 2347.087 |
| corruption/voc21/shot_noise/severity5/clipartt/seed0 | 256 | False | 23.190 | 1253.783 | 15010.891 |
| corruption/voc21/shot_noise/severity5/mlmp/seed0 | 256 | False | 38.950 | 77.603 | 2382.938 |
| corruption/voc21/shot_noise/severity5/noadapt/seed0 | 256 | False | 30.000 | 3.113 | 1851.113 |
| corruption/voc21/shot_noise/severity5/tent/seed0 | 256 | False | 29.650 | 69.373 | 2335.847 |
| corruption/voc21/shot_noise/severity5/tpt/seed0 | 256 | False | 27.140 | 219.597 | 5249.635 |
| corruption/voc21/shot_noise/severity5/trace/seed0 | 256 | False | 36.860 | 457.530 | 5102.814 |
| corruption/voc21/shot_noise/severity5/watt/seed0 | 256 | False | 16.390 | 910.136 | 2347.087 |
| corruption_smoke/noadapt | 2 | False | 12.080 | 37.857 | 1826.644 |
| corruption_smoke/trace | 2 | False | 17.860 | 605.789 | 1435.422 |
| cost/coco_object/clipartt | 128 | False | 15.850 | 1456.946 | 10307.462 |
| cost/coco_object/k1 | 128 | False | 24.960 | 137.317 | 1209.780 |
| cost/coco_object/k16 | 128 | False | 25.270 | 257.741 | 1834.700 |
| cost/coco_object/k32 | 128 | False | 25.820 | 382.491 | 1837.914 |
| cost/coco_object/k4 | 128 | False | 24.090 | 163.059 | 1392.391 |
| cost/coco_object/k8 | 128 | False | 25.070 | 194.096 | 1833.716 |
| cost/coco_object/mlmp | 128 | False | 26.060 | 128.652 | 2117.449 |
| cost/coco_object/noadapt | 128 | False | 20.600 | 3.657 | 1849.127 |
| cost/coco_object/tent | 128 | False | 20.970 | 115.986 | 2092.231 |
| cost/coco_object/tpt | 128 | False | 20.240 | 581.349 | 6732.612 |
| cost/coco_object/watt | 128 | False | 15.030 | 1758.391 | 2101.192 |
| cost/coco_stuff/clipartt | 128 | False | 11.100 | 1606.386 | 11610.685 |
| cost/coco_stuff/k1 | 128 | False | 16.840 | 148.581 | 1232.082 |
| cost/coco_stuff/k16 | 128 | False | 17.250 | 254.250 | 1868.062 |
| cost/coco_stuff/k32 | 128 | False | 17.970 | 373.556 | 1870.911 |
| cost/coco_stuff/k4 | 128 | False | 16.720 | 172.069 | 1418.467 |
| cost/coco_stuff/k8 | 128 | False | 16.990 | 192.351 | 1865.901 |
| cost/coco_stuff/mlmp | 128 | False | 17.640 | 129.433 | 2139.656 |
| cost/coco_stuff/noadapt | 128 | False | 14.990 | 3.720 | 1874.506 |
| cost/coco_stuff/tent | 128 | False | 14.020 | 116.589 | 2113.348 |
| cost/coco_stuff/tpt | 128 | False | 13.700 | 823.618 | 8648.317 |
| cost/coco_stuff/watt | 128 | False | 10.120 | 2067.625 | 2122.233 |
| cost/voc20/clipartt | 128 | False | 72.750 | 1324.801 | 8450.143 |
| cost/voc20/k1 | 128 | False | 82.280 | 149.795 | 1185.651 |
| cost/voc20/k16 | 128 | False | 82.900 | 290.083 | 1795.023 |
| cost/voc20/k32 | 128 | False | 82.850 | 438.866 | 1799.078 |
| cost/voc20/k4 | 128 | False | 84.280 | 176.125 | 1360.372 |
| cost/voc20/k8 | 128 | False | 83.600 | 212.246 | 1793.169 |
| cost/voc20/mlmp | 128 | False | 84.660 | 133.822 | 2094.502 |
| cost/voc20/noadapt | 128 | False | 73.850 | 3.772 | 1825.474 |
| cost/voc20/tent | 128 | False | 75.020 | 120.634 | 2070.069 |
| cost/voc20/tpt | 128 | False | 62.030 | 296.917 | 4047.188 |
| cost/voc20/watt | 128 | False | 77.350 | 1393.611 | 2078.456 |
| matched_released/coco_object/mlmp/seed0 | 5000 | True | 30.160 | 78.180 | 2410.620 |
| matched_released/coco_object/mlmp/seed1 | 5000 | True | 30.160 | 78.352 | 2410.620 |
| matched_released/coco_object/mlmp/seed2 | 5000 | True | 30.160 | 78.492 | 2410.620 |
| matched_released/coco_object/noadapt/seed0 | 5000 | True | 24.890 | 3.122 | 1883.514 |
| matched_released/coco_object/tent/seed0 | 5000 | True | 25.690 | 69.885 | 2362.018 |
| matched_released/coco_object/tent/seed1 | 5000 | True | 25.690 | 69.638 | 2362.018 |
| matched_released/coco_object/tent/seed2 | 5000 | True | 25.690 | 69.780 | 2362.018 |
| matched_released/coco_object/tpt/seed0 | 5000 | True | 24.250 | 323.718 | 7090.883 |
| matched_released/coco_object/tpt/seed1 | 5000 | True | 24.250 | 324.480 | 7090.883 |
| matched_released/coco_object/tpt/seed2 | 5000 | True | 24.250 | 324.531 | 7090.883 |
| matched_released/coco_object/trace/seed0 | 5000 | True | 32.000 | 444.541 | 4754.475 |
| matched_released/coco_object/trace/seed1 | 5000 | True | 32.000 | 444.373 | 4754.475 |
| matched_released/coco_object/trace/seed2 | 5000 | True | 32.000 | 444.871 | 4754.475 |
| matched_released/coco_stuff/mlmp/seed0 | 5000 | True | 21.060 | 78.387 | 2444.936 |
| matched_released/coco_stuff/mlmp/seed1 | 5000 | True | 21.060 | 78.495 | 2444.936 |
| matched_released/coco_stuff/mlmp/seed2 | 5000 | True | 21.050 | 78.573 | 2444.936 |
| matched_released/coco_stuff/noadapt/seed0 | 5000 | True | 18.240 | 3.140 | 1924.973 |
| matched_released/coco_stuff/tent/seed0 | 5000 | True | 18.470 | 69.949 | 2393.266 |
| matched_released/coco_stuff/tent/seed1 | 5000 | True | 18.470 | 70.027 | 2393.266 |
| matched_released/coco_stuff/tent/seed2 | 5000 | True | 18.470 | 70.174 | 2393.266 |
| matched_released/coco_stuff/tpt/seed0 | 5000 | True | 17.060 | 446.472 | 9009.531 |
| matched_released/coco_stuff/tpt/seed1 | 5000 | True | 17.060 | 446.653 | 9009.531 |
| matched_released/coco_stuff/tpt/seed2 | 5000 | True | 17.060 | 447.168 | 9009.531 |
| matched_released/coco_stuff/trace/seed0 | 5000 | True | 21.510 | 342.342 | 2364.033 |
| matched_released/coco_stuff/trace/seed1 | 5000 | True | 21.510 | 342.303 | 2364.033 |
| matched_released/coco_stuff/trace/seed2 | 5000 | True | 21.510 | 342.885 | 2364.033 |
| matched_released/pc59/mlmp/seed0 | 5105 | True | 32.790 | 77.554 | 2387.919 |
| matched_released/pc59/mlmp/seed1 | 5105 | True | 32.790 | 77.794 | 2387.919 |
| matched_released/pc59/mlmp/seed2 | 5105 | True | 32.790 | 77.732 | 2387.919 |
| matched_released/pc59/noadapt/seed0 | 5105 | True | 29.340 | 3.104 | 1855.043 |
| matched_released/pc59/tent/seed0 | 5105 | True | 29.720 | 69.275 | 2339.886 |
| matched_released/pc59/tent/seed1 | 5105 | True | 29.720 | 69.237 | 2339.886 |
| matched_released/pc59/tent/seed2 | 5105 | True | 29.720 | 69.208 | 2339.886 |
| matched_released/pc59/tpt/seed0 | 5105 | True | 27.050 | 224.748 | 5310.881 |
| matched_released/pc59/tpt/seed1 | 5105 | True | 27.050 | 225.461 | 5310.881 |
| matched_released/pc59/tpt/seed2 | 5105 | True | 27.050 | 225.334 | 5310.881 |
| matched_released/pc59/trace/seed0 | 5105 | True | 33.410 | 249.348 | 2764.434 |
| matched_released/pc59/trace/seed1 | 5105 | True | 33.410 | 249.332 | 2764.434 |
| matched_released/pc59/trace/seed2 | 5105 | True | 33.410 | 249.978 | 2764.434 |
| matched_released/pc60/mlmp/seed0 | 5105 | True | 29.850 | 77.709 | 2389.962 |
| matched_released/pc60/mlmp/seed1 | 5105 | True | 29.850 | 77.736 | 2389.962 |
| matched_released/pc60/mlmp/seed2 | 5105 | True | 29.850 | 77.665 | 2389.962 |
| matched_released/pc60/noadapt/seed0 | 5105 | True | 26.920 | 3.104 | 1856.095 |
| matched_released/pc60/tent/seed0 | 5105 | True | 27.130 | 69.139 | 2340.892 |
| matched_released/pc60/tent/seed1 | 5105 | True | 27.130 | 69.290 | 2340.892 |
| matched_released/pc60/tent/seed2 | 5105 | True | 27.130 | 69.188 | 2340.892 |
| matched_released/pc60/tpt/seed0 | 5105 | True | 25.030 | 226.352 | 5404.416 |
| matched_released/pc60/tpt/seed1 | 5105 | True | 25.030 | 226.281 | 5404.416 |
| matched_released/pc60/tpt/seed2 | 5105 | True | 25.030 | 226.854 | 5404.416 |
| matched_released/pc60/trace/seed0 | 5105 | True | 30.370 | 225.636 | 3287.438 |
| matched_released/pc60/trace/seed1 | 5105 | True | 30.370 | 225.799 | 3287.438 |
| matched_released/pc60/trace/seed2 | 5105 | True | 30.370 | 226.047 | 3287.438 |
| matched_released/voc20/mlmp/seed0 | 1449 | True | 84.000 | 77.453 | 2374.407 |
| matched_released/voc20/mlmp/seed1 | 1449 | True | 84.010 | 77.510 | 2374.407 |
| matched_released/voc20/mlmp/seed2 | 1449 | True | 84.010 | 77.477 | 2374.407 |
| matched_released/voc20/noadapt/seed0 | 1449 | True | 76.560 | 3.069 | 1841.700 |
| matched_released/voc20/tent/seed0 | 1449 | True | 76.940 | 68.819 | 2326.748 |
| matched_released/voc20/tent/seed1 | 1449 | True | 76.940 | 69.205 | 2326.748 |
| matched_released/voc20/tent/seed2 | 1449 | True | 76.940 | 69.209 | 2326.748 |
| matched_released/voc20/tpt/seed0 | 1449 | True | 63.880 | 172.345 | 4383.763 |
| matched_released/voc20/tpt/seed1 | 1449 | True | 63.880 | 172.279 | 4383.763 |
| matched_released/voc20/tpt/seed2 | 1449 | True | 63.880 | 172.530 | 4383.763 |
| matched_released/voc20/trace/seed0 | 1449 | True | 83.960 | 191.054 | 1793.169 |
| matched_released/voc20/trace/seed1 | 1449 | True | 83.960 | 191.030 | 1793.169 |
| matched_released/voc20/trace/seed2 | 1449 | True | 83.960 | 190.617 | 1793.169 |
| matched_released/voc21/mlmp/seed0 | 1449 | True | 50.360 | 77.469 | 2382.938 |
| matched_released/voc21/mlmp/seed1 | 1449 | True | 50.360 | 77.381 | 2382.938 |
| matched_released/voc21/mlmp/seed2 | 1449 | True | 50.370 | 79.593 | 2382.938 |
| matched_released/voc21/noadapt/seed0 | 1449 | True | 45.200 | 3.086 | 1851.113 |
| matched_released/voc21/tent/seed0 | 1449 | True | 45.400 | 71.017 | 2335.847 |
| matched_released/voc21/tent/seed1 | 1449 | True | 45.400 | 69.214 | 2335.847 |
| matched_released/voc21/tent/seed2 | 1449 | True | 45.400 | 69.466 | 2335.847 |
| matched_released/voc21/tpt/seed0 | 1449 | True | 44.310 | 219.361 | 5249.635 |
| matched_released/voc21/tpt/seed1 | 1449 | True | 44.310 | 219.438 | 5249.635 |
| matched_released/voc21/tpt/seed2 | 1449 | True | 44.310 | 219.971 | 5249.635 |
| matched_released/voc21/trace/seed0 | 1449 | True | 52.160 | 436.526 | 5102.816 |
| matched_released/voc21/trace/seed1 | 1449 | True | 52.160 | 437.317 | 5102.816 |
| matched_released/voc21/trace/seed2 | 1449 | True | 52.160 | 437.621 | 5102.816 |
| role_free/coco_cd/all_dominance | 500 | True | 75.340 | 40.923 | 7786.163 |
| role_free/coco_cd/full | 500 | True | 75.160 | 39.086 | 7786.163 |
| role_free/coco_cd/quality_only | 500 | True | 75.140 | 33.982 | 7786.163 |
| role_free/coco_object/all_dominance | 5000 | True | 31.950 | 330.930 | 4669.000 |
| role_free/coco_object/full | 5000 | True | 32.000 | 326.264 | 4669.000 |
| role_free/coco_object/quality_only | 5000 | True | 31.910 | 278.325 | 4669.000 |
| role_free/coco_stuff/all_dominance | 5000 | True | 21.480 | 252.047 | 2335.083 |
| role_free/coco_stuff/full | 5000 | True | 21.510 | 211.955 | 2335.083 |
| role_free/coco_stuff/quality_only | 5000 | True | 21.490 | 188.971 | 2335.083 |
| role_free/pc60/all_dominance | 5105 | True | 30.380 | 163.788 | 3237.935 |
| role_free/pc60/full | 5105 | True | 30.370 | 149.727 | 3237.935 |
| role_free/pc60/quality_only | 5105 | True | 30.390 | 124.733 | 3237.935 |
| role_free/voc21/all_dominance | 1449 | True | 52.120 | 369.420 | 5013.538 |
| role_free/voc21/full | 1449 | True | 52.160 | 310.227 | 5013.538 |
| role_free/voc21/quality_only | 1449 | True | 52.160 | 290.318 | 5013.538 |
| supplement/blur_kernel/coco_cd/13 | 500 | True | 75.020 | 39.129 | 7786.163 |
| supplement/blur_kernel/coco_cd/5 | 500 | True | 75.340 | 39.133 | 7786.163 |
| supplement/blur_kernel/coco_object/13 | 5000 | True | 31.920 | 302.964 | 4669.000 |
| supplement/blur_kernel/coco_object/5 | 5000 | True | 31.950 | 302.377 | 4669.000 |
| supplement/formula/coco_cd/mean | 500 | True | 74.700 | 39.150 | 7786.163 |
| supplement/formula/coco_cd/no_dominance | 500 | True | 75.150 | 39.106 | 7786.163 |
| supplement/formula/coco_cd/product | 500 | True | 75.440 | 39.124 | 7786.163 |
| supplement/formula/coco_cd/subtractive | 500 | True | 75.050 | 39.071 | 7786.163 |
| supplement/formula/coco_object/mean | 5000 | True | 31.600 | 305.108 | 4669.000 |
| supplement/formula/coco_object/no_dominance | 5000 | True | 31.940 | 302.731 | 4669.000 |
| supplement/formula/coco_object/product | 5000 | True | 31.760 | 297.366 | 4669.000 |
| supplement/formula/coco_object/subtractive | 5000 | True | 31.880 | 290.204 | 4669.000 |
| supplement/gamma/coco_cd/0.25 | 500 | True | 74.980 | 39.229 | 7786.163 |
| supplement/gamma/coco_cd/0.5 | 500 | True | 75.030 | 39.243 | 7786.163 |
| supplement/gamma/coco_cd/0.75 | 500 | True | 75.030 | 39.224 | 7786.163 |
| supplement/gamma/coco_object/0.25 | 5000 | True | 31.990 | 302.538 | 4669.000 |
| supplement/gamma/coco_object/0.5 | 5000 | True | 31.970 | 303.589 | 4669.000 |
| supplement/gamma/coco_object/0.75 | 5000 | True | 32.030 | 302.692 | 4669.000 |
| waterbirds_test_released/mlmp/seed0 | 2568 | True | 50.570 | 77.303 | 2359.356 |
| waterbirds_test_released/mlmp/seed1 | 2568 | True | 50.570 | 77.011 | 2359.356 |
| waterbirds_test_released/mlmp/seed2 | 2568 | True | 50.570 | 77.240 | 2359.356 |
| waterbirds_test_released/noadapt/seed0 | 2568 | True | 45.280 | 3.054 | 1828.115 |
| waterbirds_test_released/tent/seed0 | 2568 | True | 53.910 | 68.579 | 2312.149 |
| waterbirds_test_released/tent/seed1 | 2568 | True | 53.910 | 68.870 | 2312.149 |
| waterbirds_test_released/tent/seed2 | 2568 | True | 53.910 | 68.687 | 2312.149 |
| waterbirds_test_released/tpt/seed0 | 2568 | True | 56.700 | 136.273 | 3662.374 |
| waterbirds_test_released/tpt/seed1 | 2568 | True | 56.700 | 136.362 | 3662.374 |
| waterbirds_test_released/tpt/seed2 | 2568 | True | 56.700 | 136.870 | 3662.374 |
| waterbirds_test_released/trace/seed0 | 2568 | True | 53.430 | 83.091 | 4931.072 |
| waterbirds_test_released/trace/seed1 | 2568 | True | 53.430 | 83.125 | 4931.072 |
| waterbirds_test_released/trace/seed2 | 2568 | True | 53.430 | 83.234 | 4931.072 |
| weakening_full/coco_cd/gaussian | 500 | True | 74.980 | 39.127 | 7786.163 |
| weakening_full/coco_cd/mean | 500 | True | 74.910 | 39.036 | 7786.163 |
| weakening_full/coco_cd/noise | 500 | True | 74.740 | 39.075 | 7786.163 |
| weakening_full/coco_cd/shuffle | 500 | True | 74.840 | 39.095 | 7786.163 |
| weakening_full/coco_stuff/gaussian | 5000 | True | 21.620 | 212.534 | 2335.083 |
| weakening_full/coco_stuff/mean | 5000 | True | 20.890 | 214.475 | 2335.083 |
| weakening_full/coco_stuff/noise | 5000 | True | 20.960 | 212.802 | 2335.083 |
| weakening_full/coco_stuff/shuffle | 5000 | True | 21.390 | 212.596 | 2335.083 |
| weakening_full/pc60/gaussian | 5105 | True | 30.390 | 139.768 | 3237.935 |
| weakening_full/pc60/mean | 5105 | True | 30.100 | 139.706 | 3237.935 |
| weakening_full/pc60/noise | 5105 | True | 30.200 | 139.815 | 3237.935 |
| weakening_full/pc60/shuffle | 5105 | True | 30.360 | 139.602 | 3237.935 |
| weakening_full/voc21/gaussian | 1449 | True | 52.050 | 311.389 | 5013.538 |
| weakening_full/voc21/mean | 1449 | True | 51.790 | 311.524 | 5013.538 |
| weakening_full/voc21/noise | 1449 | True | 51.830 | 311.502 | 5013.538 |
| weakening_full/voc21/shuffle | 1449 | True | 52.110 | 310.790 | 5013.538 |

