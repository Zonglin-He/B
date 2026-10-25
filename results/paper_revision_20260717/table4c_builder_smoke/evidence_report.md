# Expanded Table 4(c) Evidence Report

Evidence type: `newly_run_analysis_of_preexisting_run`.
Input: `/home/wwww/hzl/results/requested_experiments_20260703_1020/07_qualitative_current_method/region_diagnostics.csv`.
Output: `results/paper_revision_20260717/table4c_builder_smoke`.

## Protocol

- Uses 1366 initially correct regions from the `full_soft` diagnostic variant.
- CFC risk is `1 - reliability`; entropy and inverse confidence are simple baselines.
- Confidence-band AUC asks whether source-aware reliability adds information among similarly confident candidates.
- Reliability quartiles are assigned separately inside each dataset/semantic-role stratum before pooling.
- Confidence-threshold CIs use 100 image/patch-cluster bootstrap replicates (seed 20260717).

## Main Findings

- At confidence >= 0.90 (515 regions, 22 harms), CFC risk AUC is 0.745, versus entropy 0.489 and inverse confidence 0.578.
- The image-clustered 95% CI for CFC AUC at confidence >= 0.90 is [0.601, 0.852].
- After dataset/role control, correct-to-harm changes from 15.7% in the lowest reliability quartile to 3.8% in the highest; flip changes from 18.6% to 10.0%.

## Claim Boundary

- The pooled fixed-bin trend is descriptive because high reliability is enriched with background regions.
- The confidence-controlled AUC is the primary mechanism evidence: it tests information beyond raw confidence/entropy.
- This diagnostic run is not a new benchmark evaluation and must not be mixed with three-seed full-clean means.
- AUC establishes ranking association, not causal identification of the evidence source.

## Confidence Bands

| Confidence | n | harms | CFC AUC | Entropy AUC | Inv-conf AUC |
|---|---:|---:|---:|---:|---:|
| [0.00,0.70) | 84 | 32 | 0.523 | 0.480 | 0.507 |
| [0.70,0.80) | 223 | 40 | 0.701 | 0.521 | 0.604 |
| [0.80,0.90) | 544 | 45 | 0.699 | 0.393 | 0.490 |
| [0.90,1.00] | 515 | 22 | 0.745 | 0.489 | 0.578 |
