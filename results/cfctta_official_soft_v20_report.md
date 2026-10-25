# CFC-TTA Formal Implementation Report

Date: 2026-06-17

## Protocol

Our run:
- Method: CFC-TTA formal `soft` reliability implementation
- OVSS backbone: NACLIP ViT-L/14
- Dataset: Pascal VOC 20 validation
- Input: 224 x 224, patch stride 112
- Adaptation: episodic TTA, visual LayerNorm only, text frozen
- Steps/trials: 1 step, 1 trial
- CFC config: `alpha_cls=0.0`, `cfc_weight_mode=soft`, `cfc_reliability_threshold=0.5`, `lambda_cf=0.05`, `lambda_amb=0.02`

Baseline source:
- MLMP official paper/repository table for NACLIP on V20/V20-C:
  - https://arxiv.org/html/2505.21844v1
  - https://github.com/dosowiechi/MLMP

Important caveat:
- The quoted baselines are official reported numbers. Our CFC run here is a local 1-step/1-trial run. The result is enough to diagnose current behavior, but it is not a fully matched 10-step/3-trial reproduction.

## Clean V20

| Method | mIoU |
|---|---:|
| NoAdapt. | 75.91 |
| TENT | 77.00 |
| TPT | 75.93 |
| WATT | 57.73 |
| CLIPArTT | 72.77 |
| MLMP | 83.76 |
| CFC-TTA formal, ours | 60.95 |

Our local result path:
- `/home/wwww/hzl/results/cfctta_official_soft_v20_clean_s1/voc20/cfctta/results.txt`

## V20-C Selected Corruptions

| Corruption | NoAdapt. | TENT | TPT | WATT | CLIPArTT | MLMP | CFC-TTA formal, ours |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gaussian Noise | 62.89 | 63.02 | 62.98 | 36.44 | 53.36 | 71.13 | 45.59 |
| Shot Noise | 66.26 | 65.88 | 66.33 | 40.95 | 58.15 | 75.02 | 47.90 |
| Impulse Noise | 63.16 | 64.17 | 63.12 | 34.90 | 54.83 | 71.34 | 44.27 |
| Defocus Blur | 72.59 | 72.06 | 72.55 | 52.43 | 65.39 | 80.36 | 64.92 |
| Motion Blur | 73.10 | 73.50 | 73.16 | 53.35 | 67.48 | 81.41 | 63.14 |
| Fog | 70.69 | 70.54 | 70.70 | 52.96 | 67.85 | 79.81 | 55.69 |
| Contrast | 71.51 | 70.51 | 71.49 | 50.74 | 66.01 | 79.06 | 64.62 |
| Pixelate | 77.28 | 76.95 | 77.31 | 59.76 | 73.14 | 84.97 | 64.23 |
| JPEG Compression | 72.59 | 71.84 | 72.56 | 53.44 | 68.21 | 82.06 | 53.22 |
| Selected Avg | 70.01 | 69.83 | 70.02 | 48.33 | 63.82 | 78.35 | 55.95 |

Our local result paths:
- `/home/wwww/hzl/results/cfctta_official_soft_v20_c_s1/voc20/cfctta/00_gaussian_noise/results.txt`
- `/home/wwww/hzl/results/cfctta_official_soft_v20_c_s1/voc20/cfctta/01_shot_noise/results.txt`
- `/home/wwww/hzl/results/cfctta_official_soft_v20_c_s1/voc20/cfctta/02_impulse_noise/results.txt`
- `/home/wwww/hzl/results/cfctta_official_soft_v20_c_s1/voc20/cfctta/03_defocus_blur/results.txt`
- `/home/wwww/hzl/results/cfctta_official_soft_v20_selected_s1/voc20/cfctta/results.txt`

## Full V20-C Attempt

I started the full 15-corruption run. It completed Gaussian Noise, Shot Noise, Impulse Noise, and Defocus Blur. The run was stopped at Glass Blur because the on-the-fly glass-blur corruption saturated CPU workers for several minutes while GPU utilization stayed near idle. The completed outputs above were preserved.

## Interpretation

Current conclusion:
- The formal CFC-TTA implementation runs end-to-end.
- The current default reliability threshold and weighting are too strict or miscalibrated for the standard VOC20/VOC20-C benchmark.
- On clean V20, CFC-TTA formal is 60.95 mIoU, below NoAdapt 75.91 and MLMP 83.76.
- On the 9 selected V20-C corruptions, CFC-TTA formal averages 55.95 mIoU, below NoAdapt 70.01 and MLMP 78.35.

This does not invalidate the motivation; it says the current formal method should be evaluated next on high-CFC/context-conflict subsets and then calibrated so standard clean performance is not destroyed.
