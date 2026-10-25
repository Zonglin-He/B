# Per-Dataset Table 4(c) Baseline Expansion

## Protocol

- Evidence type: newly run matched-candidate analysis plus reused matched runs for NoAdapt/TENT/MLMP/CFC-TTA.
- Datasets are reported separately; no cross-dataset pooling is used.
- Candidate masks and CFC reliability are frozen before adaptation.
- Baseline protocol: lr=0.001 and 10 adaptation steps; TPT uses 4 context tokens, WATT uses L=2/M=5, and CLIPArTT uses K=3.
- Quartiles are assigned within each dataset and semantic role, then shared across methods.
- Correct-to-harm is conditioned on each method's own initially correct candidate prediction.
- AUC is undefined when a method creates no harmful transitions, as for No Adapt.

## Compact Results

### VOC20

| Method | mIoU | Q1 C->H | Q2 | Q3 | Q4 | CFC-risk AUC | Entropy AUC | Flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No Adapt | 75.79 | 0.0% | 0.0% | 0.0% | 0.0% | -- | -- | 0.0% |
| TENT | 70.89 | 35.7% | 32.5% | 20.0% | 21.9% | 0.585 | 0.687 | 40.8% |
| TPT | 60.15 | 44.1% | 50.9% | 45.0% | 51.9% | 0.487 | 0.645 | 57.0% |
| WATT | 69.88 | 48.3% | 27.4% | 30.4% | 25.1% | 0.589 | 0.613 | 45.0% |
| CLIPArTT | 68.89 | 36.4% | 28.4% | 23.9% | 27.8% | 0.539 | 0.691 | 41.4% |
| MLMP | 81.96 | 24.3% | 14.2% | 9.4% | 5.9% | 0.662 | 0.709 | 27.7% |
| CFC-TTA | 83.49 | 29.7% | 14.8% | 8.3% | 6.9% | 0.685 | 0.704 | 29.7% |

### PC60

| Method | mIoU | Q1 C->H | Q2 | Q3 | Q4 | CFC-risk AUC | Entropy AUC | Flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No Adapt | 27.47 | 0.0% | 0.0% | 0.0% | 0.0% | -- | -- | 0.0% |
| TENT | 25.68 | 19.9% | 24.9% | 17.5% | 16.0% | 0.481 | 0.671 | 52.3% |
| TPT | 24.46 | 51.4% | 51.8% | 40.3% | 33.6% | 0.542 | 0.667 | 75.2% |
| WATT | 16.20 | 61.6% | 78.3% | 68.3% | 71.7% | 0.388 | 0.633 | 85.4% |
| CLIPArTT | 23.03 | 42.1% | 46.2% | 34.7% | 39.3% | 0.420 | 0.715 | 62.1% |
| MLMP | 28.40 | 12.0% | 18.1% | 14.2% | 8.7% | 0.495 | 0.667 | 34.0% |
| CFC-TTA | 28.83 | 15.8% | 21.7% | 10.1% | 3.6% | 0.655 | 0.679 | 44.5% |

### COCO-CD

| Method | mIoU | Q1 C->H | Q2 | Q3 | Q4 | CFC-risk AUC | Entropy AUC | Flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No Adapt | 62.76 | 0.0% | 0.0% | 0.0% | 0.0% | -- | -- | 0.0% |
| TENT | 71.76 | 0.0% | 0.0% | 0.0% | 0.0% | -- | -- | 30.6% |
| TPT | 69.20 | 10.3% | 8.6% | 10.0% | 7.1% | 0.922 | 0.192 | 49.6% |
| WATT | 39.31 | 56.7% | 53.3% | 38.3% | 35.1% | 0.518 | 0.688 | 61.2% |
| CLIPArTT | 55.86 | 19.0% | 22.4% | 15.0% | 12.5% | 0.434 | 0.858 | 24.6% |
| MLMP | 71.17 | 0.0% | 0.0% | 0.0% | 0.0% | -- | -- | 10.6% |
| CFC-TTA | 73.04 | 3.4% | 1.7% | 1.7% | 0.0% | 0.934 | 0.517 | 22.1% |

### Waterbirds-Seg

| Method | mIoU | Q1 C->H | Q2 | Q3 | Q4 | CFC-risk AUC | Entropy AUC | Flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No Adapt | 39.96 | 0.0% | 0.0% | 0.0% | 0.0% | -- | -- | 0.0% |
| TENT | 45.36 | 8.4% | 2.9% | 0.0% | 1.2% | 0.703 | 0.783 | 71.2% |
| TPT | 44.67 | 10.8% | 15.7% | 6.0% | 8.3% | 0.812 | 0.635 | 79.6% |
| WATT | 22.99 | 70.7% | 66.0% | 80.2% | 62.5% | 0.506 | 0.555 | 90.6% |
| CLIPArTT | 25.95 | 75.9% | 63.7% | 55.3% | 47.6% | 0.542 | 0.782 | 74.3% |
| MLMP | 42.59 | 27.7% | 14.7% | 9.5% | 4.8% | 0.863 | 0.804 | 51.4% |
| CFC-TTA | 49.26 | 9.6% | 10.8% | 2.4% | 3.6% | 0.884 | 0.770 | 60.7% |

## Alignment Audit

| Dataset | Diagnostic candidates | Shared valid candidates | Shared fraction | Max reliability delta |
|---|---:|---:|---:|---:|
| VOC20 | 1328 | 841 | 63.3% | 0 |
| PC60 | 1920 | 1920 | 100.0% | 0 |
| COCO-CD | 240 | 240 | 100.0% | 0 |
| Waterbirds-Seg | 480 | 480 | 100.0% | 0 |

## Artifacts

- Detailed quartiles: `/home/wwww/hzl/results/paper_revision_20260718/table4c_per_dataset_7baselines/per_dataset_method_quartiles.csv`
- Method summaries and AUC: `/home/wwww/hzl/results/paper_revision_20260718/table4c_per_dataset_7baselines/per_dataset_method_summary.csv`
- Baseline C->H/Flip CSV: `/home/wwww/hzl/results/paper_revision_20260718/table4c_per_dataset_7baselines/per_dataset_baseline_ch_flip.csv`
- Baseline C->H/Flip LaTeX: `/home/wwww/hzl/results/paper_revision_20260718/table4c_per_dataset_7baselines/table4c_per_dataset_ch_flip.tex`
- Reliability-quartile CSV: `/home/wwww/hzl/results/paper_revision_20260718/table4c_per_dataset_7baselines/per_dataset_baseline_reliability_quartiles.csv`
- Reliability-quartile LaTeX: `/home/wwww/hzl/results/paper_revision_20260718/table4c_per_dataset_7baselines/table4c_reliability_quartiles_all_datasets.tex`
- LaTeX table: `/home/wwww/hzl/results/paper_revision_20260718/table4c_per_dataset_7baselines/table4c_per_dataset_baselines.tex`
- Per-dataset LaTeX tables: `/home/wwww/hzl/results/paper_revision_20260718/table4c_per_dataset_7baselines/table4c_<dataset>.tex`
- Detailed per-dataset tables: `/home/wwww/hzl/results/paper_revision_20260718/table4c_per_dataset_7baselines/table4c_<dataset>_detailed.tex`
- Provenance: `/home/wwww/hzl/results/paper_revision_20260718/table4c_per_dataset_7baselines/provenance.json`
