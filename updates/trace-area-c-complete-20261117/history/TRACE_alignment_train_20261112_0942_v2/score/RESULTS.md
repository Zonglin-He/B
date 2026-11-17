# TRACE Alignment Development

Same32 training sources, not full validation or independent confirmation.

| Arm | Clean | S5 macro |
|---|---:|---:|
| MLMP | 63.9395 | 57.9451 |
| area_C | 63.7954 | 58.2182 |
| coverage_native_budget | 64.2557 | 57.9918 |
| aligned_query_entropy | 63.9837 | 58.1418 |
| aligned_semantic_entropy | 64.3094 | 58.1772 |
| aligned_semantic_anchor | 64.2881 | 58.0702 |

| Contrast | S5 delta pp | Pointwise95 paired interval |
|---|---:|---|
| aligned_query_entropy_minus_MLMP | 0.1967 | [-0.3949, 0.8530] |
| aligned_query_entropy_minus_area_C | -0.0764 | [-0.4841, 0.8514] |
| aligned_semantic_entropy_minus_MLMP | 0.2321 | [-0.4081, 0.5927] |
| aligned_semantic_entropy_minus_area_C | -0.0410 | [-0.5851, 0.6693] |
| aligned_semantic_anchor_minus_MLMP | 0.1251 | [-0.3999, 0.6537] |
| aligned_semantic_anchor_minus_area_C | -0.1480 | [-0.6430, 0.6707] |
| aligned_semantic_entropy_minus_aligned_query_entropy | 0.0354 | [-0.6900, 0.4308] |
| aligned_semantic_anchor_minus_aligned_semantic_entropy | -0.1070 | [-0.5396, 0.5178] |

10000 shared whole16B2-group draws; undefined retained; no multiplicity correction, validation tuning or automatic winner.
