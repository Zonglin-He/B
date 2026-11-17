# Full Official-Source Comparison After Reboot Recovery

Completed 2026-11-15 at 10:51:12 CST. The main agent independently verified the final artifacts at 2026-11-15T02:55:20.773532+00:00.
This report covers the finite 26-condition comparison only, not all November 5 research or manuscript delivery.

## Result

All 11 clean and 15 L14 VOC20-C severity-5 conditions completed all 3 arms and 3 shuffled repeats. Each dataset's 22 clean-selected TRACE fields were copied unchanged to its matching B16/S5 conditions. MLMP paper hyperparameters were not tuned.

| Condition | Images | NoAdapt | MLMP | TRACE area-C | TRACE minus MLMP pp |
|---|---:|---:|---:|---:|---:|
| L14 PC60 | 5104 | 24.9500 | 27.9833 | 28.7133 | +0.7300 |
| L14 COCO Object | 5000 | 25.0300 | 30.4200 | 32.4400 | +2.0200 |
| L14 VOC20 | 1449 | 75.9100 | 83.8033 | 86.1200 | +2.3167 |
| L14 VOC21 | 1449 | 45.1300 | 50.7600 | 51.1433 | +0.3833 |
| L14 PC59 | 5104 | 28.2300 | 31.9433 | 32.8767 | +0.9333 |
| L14 COCO Stuff | 5000 | 18.3400 | 21.2500 | 21.9933 | +0.7433 |
| L14 Waterbirds segmentation | 2568 | 45.3700 | 49.9033 | 54.0267 | +4.1233 |
| L14 COCO-CD | 500 | 62.0600 | 72.3967 | 77.3833 | +4.9867 |
| B16 COCO Object | 5000 | 29.5900 | 30.5267 | 29.7900 | -0.7367 |
| B16 PC60 | 5104 | 28.9000 | 29.6133 | 29.2000 | -0.4133 |
| B16 VOC20 | 1449 | 77.5900 | 84.1233 | 84.3800 | +0.2567 |

## Complete S5 Results

All 15 types have 1449 images each. Macro is the arithmetic mean across all 15 type-level means, not an image-pooled metric or a selected subset.

| Condition | Images | NoAdapt | MLMP | TRACE area-C | TRACE minus MLMP pp |
|---|---:|---:|---:|---:|---:|
| gaussian_noise | 1449 | 62.8800 | 71.1067 | 73.1300 | +2.0233 |
| shot_noise | 1449 | 66.2500 | 74.9633 | 77.1900 | +2.2267 |
| impulse_noise | 1449 | 63.1400 | 71.2133 | 72.8300 | +1.6167 |
| defocus_blur | 1449 | 72.6000 | 80.4167 | 80.6900 | +0.2733 |
| glass_blur | 1449 | 71.4500 | 78.7233 | 80.1800 | +1.4567 |
| motion_blur | 1449 | 73.1000 | 81.2733 | 82.7700 | +1.4967 |
| zoom_blur | 1449 | 59.0200 | 69.4333 | 71.9400 | +2.5067 |
| snow | 1449 | 71.5000 | 79.4167 | 82.0000 | +2.5833 |
| frost | 1449 | 65.3700 | 73.3000 | 76.9500 | +3.6500 |
| fog | 1449 | 70.6800 | 79.9067 | 82.8700 | +2.9633 |
| brightness | 1449 | 74.9500 | 83.4300 | 85.8700 | +2.4400 |
| contrast | 1449 | 71.5100 | 79.2133 | 81.1300 | +1.9167 |
| elastic_transform | 1449 | 62.8700 | 74.0067 | 77.1700 | +3.1633 |
| pixelate | 1449 | 77.2900 | 84.9667 | 86.4900 | +1.5233 |
| jpeg_compression | 1449 | 72.5900 | 82.1967 | 83.9900 | +1.7933 |
| 15-type arithmetic macro | 21735 condition-images | 69.0133 | 77.5711 | 79.6800 | +2.1089 |

TRACE exceeds MLMP on all 15 S5 means. B16 Object (-0.7367 pp) and PC60 (-0.4133 pp) regressions remain; B16 VOC20 improves +0.2567 pp. These outcomes do not establish universal backbone transfer.

## Repeat Dispersion

Means above follow official process_metrics rounding within each trial before averaging. Exact unrounded trial points are retained in sealed RESULT.json files. The following population standard deviations describe three shuffled repeats, not three independent seeds and not confidence intervals. No paired interval was inferred by subtracting marginal intervals.

| Condition | NoAdapt std | MLMP std | TRACE area-C std |
|---|---:|---:|---:|
| L14 PC60 | 0.000000 | 0.023570 | 0.051854 |
| L14 COCO Object | 0.000000 | 0.028284 | 0.000000 |
| L14 VOC20 | 0.000000 | 0.020548 | 0.000000 |
| L14 VOC21 | 0.000000 | 0.008165 | 0.041096 |
| L14 PC59 | 0.000000 | 0.032998 | 0.065997 |
| L14 COCO Stuff | 0.000000 | 0.008165 | 0.036818 |
| L14 Waterbirds segmentation | 0.000000 | 0.020548 | 0.049216 |
| L14 COCO-CD | 0.000000 | 0.077172 | 0.059067 |
| B16 COCO Object | 0.000000 | 0.009428 | 0.000000 |
| B16 PC60 | 0.000000 | 0.009428 | 0.037417 |
| B16 VOC20 | 0.000000 | 0.085765 | 0.000000 |
| gaussian_noise | 0.000000 | 0.091043 | 0.000000 |
| shot_noise | 0.000000 | 0.128149 | 0.000000 |
| impulse_noise | 0.000000 | 0.116142 | 0.000000 |
| defocus_blur | 0.000000 | 0.098093 | 0.000000 |
| glass_blur | 0.000000 | 0.137679 | 0.000000 |
| motion_blur | 0.000000 | 0.075865 | 0.000000 |
| zoom_blur | 0.000000 | 0.193620 | 0.000000 |
| snow | 0.000000 | 0.070396 | 0.000000 |
| frost | 0.000000 | 0.035590 | 0.000000 |
| fog | 0.000000 | 0.033993 | 0.000000 |
| brightness | 0.000000 | 0.035590 | 0.000000 |
| contrast | 0.000000 | 0.082597 | 0.000000 |
| elastic_transform | 0.000000 | 0.080139 | 0.000000 |
| pixelate | 0.000000 | 0.051854 | 0.000000 |
| jpeg_compression | 0.000000 | 0.041096 | 0.000000 |

## Recovery And Verification

- The shutdown changed the boot. Original successful transient units were not recreated or called current retained successes.
- Eighteen preboot scoring arms were reused through original PID/invocation, fsynced actual-success snapshots, persistent same-invocation journals and original output hashes.
- Sixty postboot scoring arms have same-current-invocation retained successful prediction AND score units. No successful condition was rerun for recovery.
- All 78 result/origin/prediction-success bindings and root/summary seals were independently reconciled. The prior 171 exact trial points were already verified in progress329-332; the final 21 arms/63 points were newly hash-verified and recomputed from compact I/U without reopening raw GT.
- Dispatcher PID 238209/invocation 6132fdd6ef95401cb572a01c1d65d379 and summary PID 1581979/invocation 71763411617a4e4493df11ae52d6c479 actually ended Code 1/Status 0/Result success, active/exited, MainPID 0.
- Root has 522 sealed outputs and no active FAILED.json/CANCELLED.json flags. Historical failure evidence remains preserved under distinct audit filenames.
- All 68 local frozen source hashes remain unchanged. Native kernel/NVML and actual model libcuda were verified as 580.178.04 after reboot; no old 580.173.02 override was used.
- Waterbirds/COCO-CD null identity query maps were narrowly represented as explicit identity arrays. Failed V8/V9 bytes/logs/source were preserved; V10 was a new frozen version, not an unchanged-byte retry.
- The single Luna/max read-only monitor returned native COMPLETE_CANDIDATE and ended. No further polling, ancestor-message bypass, new monitor, cron or automation was created. Original trace automation remains PAUSED.

## Claim Boundaries

- Official MLMP source commit 2a887de018b3f2c3b132f69fe806bab12a2d511f, native losses, prompts and readout are retained; lr 0.001, 10 steps, batch 2, alpha_cls 1 and 7 prompts are fixed.
- Newly executed stages use explicitly declared strict deterministic runtime. Reused earlier baseline stages retain their original upstream-seeds-only runtime. This is not an unchanged-upstream-runtime claim.
- B16 uses declared 75-percent-last-layer architecture transfer; Waterbirds/COCO-CD are manifest extensions, not original paper datasets.
- The corrected Pascal Context reference population has 5104 images. Earlier TRACE calibration overlaps it: PC59 screen/refine/confirm 33/62/59 and PC60 34/61/61. Do not call these untouched held-out TRACE tests.
- Object paper-value discrepancy remains OPEN: L14 NoAdapt 25.03 versus 23.80 and MLMP 30.42 versus 28.84. No tuning was used to force paper-value agreement.
- Historical exposure, negative/null/partial results and failure/cancellation evidence are retained. No universal SOTA, causal, paired-significance or untouched-development claim follows from this table.
- This completion does not finish all R00-R05 research or paper/publication work. No Windows manuscript was delivered/read, and no manuscript/upload was changed here.

## Artifacts

- Suite: `[revision-results]/TRACE_official_baselines_20261114-2022-v10`
- Main verification: `[revision-results]/TRACE_official_baselines_20261114-2022-v10_MAIN_VERIFIED_20261115.json`
- PLAN SHA256: `52ca4c37298e2aa12f3df71fa7b777d73568c09fed87112e328fc5a5f5873a89`
- Root COMPLETED SHA256: `013c8e3a25dcef607f37affcb524c79ffa202a8d1caa460c46c26b6ac4647ff0`
- Summary COMPLETED SHA256: `afafb8a7fdbdae0ec53fc93464a78db0f87b836e9c08017d607d3f268791dfe1`
- Summary RESULT SHA256: `88cb8a34bb21189808598b4a723300e6ced497a04f267f746e3cbe4ee65d3f46`
- Summary RESULTS.md SHA256: `6365f6abccb26ced35b60018afb590a2018e0656ea67faef842f8b1421e41ae8`
