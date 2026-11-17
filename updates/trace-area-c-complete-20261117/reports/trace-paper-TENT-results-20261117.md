# Six Full Clean Official TENT Results

All six original canonical L14 dataset tasks completed3 shuffled repeats, each with a label-isolated prediction stage and a separate NEW CPU_GT scorer. Main authenticated all1182 output hashes and actual retained same-invocation prediction AND score successes, then recomputed all18 exact trial mIoU points from integer compact I/U without reopening GT.

TENT uses the pinned upstream class, original pixel entropy objective, fixed0.001/10steps/batch2, single reference prompt and last-layer readout. No TENT hyperparameter was searched. The declared deterministic runtime, workers0 I/O extension and corrected Pascal reference are retained; this is not a claim of byte-identical upstream runtime or complete scalar paper reproduction.

## Complete Mean Table

Means below average official process_metrics-rounded trial points before averaging, consistently across all arms.

| L14 dataset | Images | NoAdapt | TENT | MLMP | Adopted area_C |
|---|---:|---:|---:|---:|---:|
| coco_object | 5000 | 25.030000 | 26.053333 | 30.420000 | 32.440000 |
| pc60 | 5104 | 24.950000 | 25.280000 | 27.983333 | 28.713333 |
| voc20 | 1449 | 75.910000 | 76.996667 | 83.803333 | 86.120000 |
| voc21 | 1449 | 45.130000 | 45.643333 | 50.760000 | 51.143333 |
| pc59 | 5104 | 28.230000 | 28.723333 | 31.943333 | 32.876667 |
| coco_stuff | 5000 | 18.340000 | 18.756667 | 21.250000 | 21.993333 |

## Exact TENT Trial Values

| Dataset | Order1 | Order2 | Order3 | Exact mean | Exact population std |
|---|---:|---:|---:|---:|---:|
| coco_object | 26.047727 | 26.081548 | 26.030524 | 26.053266 | 0.021196 |
| pc60 | 25.289694 | 25.259831 | 25.291140 | 25.280221 | 0.014430 |
| voc20 | 76.958381 | 77.009990 | 77.022384 | 76.996918 | 0.027716 |
| voc21 | 45.652240 | 45.712616 | 45.565952 | 45.643603 | 0.060186 |
| pc59 | 28.728158 | 28.700696 | 28.737414 | 28.722089 | 0.015592 |
| coco_stuff | 18.753765 | 18.768832 | 18.746881 | 18.756493 | 0.009167 |

Repeat standard deviation is descriptive order dispersion, not a confidence interval or independent-seed uncertainty. TENT is below both MLMP and adopted area_C on these six means. TENT was not run on the extra Waterbirds/COCO-CD extensions, B16 or15S5 conditions under this six-cell authorization; missing entries are not zeros.

Official NoAdapt and TENT use single-prompt/final-layer readout; MLMP and area_C retain native MLMP multi-prompt/multi-layer final readout. Consequently TENT-to-MLMP is a complete-system contrast, not an isolated CLS-loss or region-weight comparison. The dedicated same-readout ablations address different questions.

## Seals

| Dataset | Prediction COMPLETED SHA256 | Score COMPLETED SHA256 |
|---|---|---|
| coco_object | c70a5cce19a4cd954f7c4e097328a0158d8be5383a587ef768707a0f7bce1230 | c9b0778c225cb29b37de5a08650d5def74b405d482cdc0f37b31da9d238a205d |
| pc60 | eb4dd927c9a962434599e62b253d67cf73249fc30528f263b38a4e73e3d4806e | 50806ffa6393b588ba99fd95c61c45ad3ba534189960aea01881f5728e508421 |
| voc20 | c010204a162c699d9b9cfd0e93253721e107b1d372f0807185a2c4861f67773e | 5c7ac29624e507ca199bae6e7f78b0b9992d250d06bf7a1469c3103d68c9ff38 |
| voc21 | ed3f75d1edb6aa9b9064394fc9bbb20023d71245ea1c7223f61b94689a1720f5 | c2a6631ce31672eccef98ac5954211bc8b4f5e9525f8507dd0aaccec885d3dbc |
| pc59 | a307038e8e3337ddb2cbc798fe1bf6585db639ab2847d8fd0a486e31aa4960ea | a9c2610a3af858bb3405798ce8ccf55c565c5ad6fcd2a857792202c620b781e9 |
| coco_stuff | a77a8217afc4418e7b2e9de55d0be5dccda5a1507ade1e0a002d6f10843073bb | 493251211579a268fd3d8f0b6fea952ec9e4917634192bc5f1109dd34d61292f |

Root:[revision-results]/TRACE_paper_finish_recovery_20261116-2200-v2.
Machine-readable main verification:TENT_main_verified_20261117.json.
All96 frozen source files, original method parameters, MLMP settings, negative/null history and prior PC exposure remain. The later full component controls and new manuscript are not declared complete by this report.
