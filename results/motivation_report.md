# CFC Motivation Experiment Log

Date: 2026-06-17

## Code and Environment

- Baseline source: `external/MLMP`, cloned from `https://github.com/dosowiechi/MLMP`.
- Dataset: PASCAL VOC 2012 validation split, downloaded to `data/VOCdevkit/VOC2012`.
- Runtime env: conda env `mlmp`, Python 3.10, PyTorch 2.7.0 + CUDA 12.8, `mmcv-lite==2.1.0`, `mmsegmentation==1.2.2`.
- Backbone used for fast motivation runs: `NACLIP ViT-B/16`.
- Adaptation baselines: TENT and MLMP, both called through the official MLMP implementation.

## Scripts

- `experiments/motivation_mlmp_voc.py`
  - Natural VOC probing.
  - Supports GT regions or predicted connected components.
  - Supports VOC21-style vocabulary with background.
  - Computes object-only, background-only, counterfactual views.
  - Computes TENT/MLMP harmful adaptation statistics.

- `experiments/synthetic_cfc_voc.py`
  - Synthetic context-conflict copy-paste experiment.
  - Pastes VOC objects onto backgrounds from images that do not contain the same class.
  - Computes predicted-class background leakage and TENT/MLMP harmful adaptation.

## Main Runs

### Natural VOC, GT Regions

Output: `results/motivation_mlmp_voc21_vitb16_r80`

- Regions: 80
- Error rate: 27.5%
- Mean entropy: 0.215
- Mean background leakage: 0.360
- TENT harmful adaptation rate: 63.0%
- MLMP harmful adaptation rate: 70.4%
- TENT mean entropy delta: -0.070
- MLMP mean entropy delta: -0.056

Interpretation: strong evidence that confidence-driven TTA often reduces entropy without improving region quality. Natural GT-region leakage is common but not yet predictive of errors.

### Natural VOC, Predicted Regions, Hole BLS

Output: `results/motivation_mlmp_voc21_pred_hole_vitb16_r80`

- Predicted connected components: 80
- Error rate: 92.5%
- Mean entropy: 0.507
- Mean background leakage: 0.196
- Low-entropy/high-leakage error rate: 69.2%
- TENT harmful adaptation rate: 45.5%
- MLMP harmful adaptation rate: 59.1%

Interpretation: predicted regions reveal many high-confidence wrong components. Some qualitative cases are CFC-like, especially grass/sheep-style cases, but leakage is not yet a clean scalar predictor by itself.

### Synthetic Context Conflict

Output: `results/synthetic_cfc_voc_predscore_hole_vitb16_s80`

- Synthetic samples: 80
- Error rate: 35.0%
- Mean entropy: 0.347
- Mean predicted-class background leakage: 0.566
- TENT harmful adaptation rate: 56.3%
- MLMP harmful adaptation rate: 52.5%

Interpretation: copy-paste context conflict makes errors and harmful adaptation visible. Current random background pairing is too noisy to prove that high leakage is worse than low leakage; the next version should use curated class-context pairs.

## Current Takeaway

The strongest result so far is TTA amplification:

- TENT and MLMP reliably reduce entropy.
- A large fraction of regions have non-improving or worse region quality after adaptation.
- This supports the need for a "safe TTA" filter before entropy minimization.

The CFC-specific leakage metric is qualitatively promising but not yet quantitatively sharp with random VOC sampling. The next experiment should curate context pairs such as `cow/grass`, `boat/water`, `airplane/sky`, `train/track`, and then swap objects into conflicting backgrounds such as beach, road, indoor, or sky-like scenes.
