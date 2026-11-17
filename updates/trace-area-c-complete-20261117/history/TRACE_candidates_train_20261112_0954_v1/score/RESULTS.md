# Bounded Semantic Candidates: Training Results

Repeated small training development; all arms retained, not independent confirmation or equal-compute evidence; pointwise/no multiplicity adjustment.

| Arm | Clean | S5 macro |
|---|---:|---:|
| MLMP | 63.9395 | 57.9451 |
| area_C | 63.7954 | 58.2182 |
| coverage_native_budget | 64.2557 | 57.9918 |
| aligned_query_entropy | 63.9837 | 58.1418 |
| aligned_semantic_entropy | 64.3094 | 58.1772 |
| aligned_semantic_anchor | 64.2881 | 58.0702 |
| semantic_top8 | 63.7972 | 58.2342 |
| semantic_mass_top8 | 64.4506 | 58.2396 |

| Contrast | S5 delta pp | Pointwise95 paired interval |
|---|---:|---|
| semantic_top8_minus_MLMP | 0.2891 | [-0.4371, 0.8265] |
| semantic_top8_minus_area_C | 0.0160 | [-0.2443, 0.5145] |
| semantic_mass_top8_minus_MLMP | 0.2945 | [-0.3776, 0.9634] |
| semantic_mass_top8_minus_area_C | 0.0214 | [-0.4660, 0.9725] |
| semantic_mass_top8_minus_semantic_top8 | 0.0054 | [-0.5754, 0.7455] |
