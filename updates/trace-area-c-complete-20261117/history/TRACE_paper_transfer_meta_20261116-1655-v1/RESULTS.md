# Candidate and Update Diagnostics

Descriptive, label-free. Gradients cover only the first fixed8-source block.
Region areas use the actual L14/B16 feature grid denominator.

| Task | Queries/classes | Selected C | Coverage | Raw C bonus/grid |
|---|---:|---:|---:|---:|
| L14_coco_object | 113/81 | 0.309047 | 0.179779 | 0.117531 |
| L14_pc60 | 60/60 | 0.316503 | 0.267395 | 0.188604 |
| L14_voc20 | 30/20 | 0.290370 | 0.407181 | 0.271975 |
| L14_waterbirds_seg | 3/3 | 0.407624 | 0.755783 | 0.591657 |
| B16_coco_object | 113/81 | 0.395587 | 0.394830 | 0.284792 |
| B16_pc60 | 60/60 | 0.390124 | 0.490533 | 0.369853 |
| B16_voc20 | 30/20 | 0.302375 | 0.479233 | 0.323587 |

Profiles do not prove backbone-transfer causality or beneficial-update prediction.
Undefined ratios/cosines remain explicit. No parameter selection is performed.