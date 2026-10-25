# Per-Dataset Table 4(c) Baseline Expansion

## Protocol

- Evidence type: newly run matched-candidate analysis plus reused matched runs for NoAdapt/TENT/MLMP/CFC-TTA.
- Datasets are reported separately; no cross-dataset pooling is used.
- Candidate masks and CFC reliability are frozen before adaptation.
- Quartiles are assigned within each dataset and semantic role, then shared across methods.
- Correct-to-harm is conditioned on each method's own initially correct candidate prediction.
- AUC is undefined when a method creates no harmful transitions, as for No Adapt.

## Compact Results

### VOC20

| Method | mIoU | Q1 C->H | Q2 | Q3 | Q4 | CFC-risk AUC | Entropy AUC | Flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TPT | 100.00 | 0.0% | --% | --% | --% | -- | -- | 0.1% |
| WATT | 100.00 | 0.0% | --% | --% | --% | -- | -- | 0.1% |
| CLIPArTT | 100.00 | 0.0% | --% | --% | --% | -- | -- | 0.1% |

## Alignment Audit

| Dataset | Diagnostic candidates | Shared valid candidates | Shared fraction | Max reliability delta |
|---|---:|---:|---:|---:|
| VOC20 | 7 | 1 | 14.3% | 0 |

## Artifacts

- Detailed quartiles: `/home/wwww/hzl/results/paper_revision_20260718/table4c_builder_smoke/per_dataset_method_quartiles.csv`
- Method summaries and AUC: `/home/wwww/hzl/results/paper_revision_20260718/table4c_builder_smoke/per_dataset_method_summary.csv`
- LaTeX table: `/home/wwww/hzl/results/paper_revision_20260718/table4c_builder_smoke/table4c_per_dataset_baselines.tex`
- Provenance: `/home/wwww/hzl/results/paper_revision_20260718/table4c_builder_smoke/provenance.json`
