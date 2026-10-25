# Expanded Table 4(c) Evidence Report

Evidence type: `newly_run_20260717`.
Input: `results/paper_revision_20260717/role_diagnosis_9datasets_240/region_diagnostics.csv`.
Output: `results/paper_revision_20260717/table4c_9dataset_role_expansion`.

## Protocol

- Uses 75762 initially correct regions from the `full_soft` diagnostic variant.
- CFC risk is `1 - reliability`; entropy and inverse confidence are simple baselines.
- Confidence-band AUC asks whether source-aware reliability adds information among similarly confident candidates.
- Reliability quartiles are assigned separately inside each dataset/semantic-role stratum before pooling.
- Confidence-threshold CIs use 0 image/patch-cluster bootstrap replicates (seed 20260717).

## Main Findings

- At confidence >= 0.90 (19537 regions, 534 harms), CFC risk AUC is 0.504, versus entropy 0.617 and inverse confidence 0.693.
- The image-clustered 95% CI for CFC AUC at confidence >= 0.90 is [--, --].
- After dataset/role control, correct-to-harm changes from 16.6% in the lowest reliability quartile to 10.3% in the highest; flip changes from 18.1% to 17.6%.

## Claim Boundary

- The pooled fixed-bin trend is descriptive because high reliability is enriched with background regions.
- The confidence-controlled AUC is the primary mechanism evidence: it tests information beyond raw confidence/entropy.
- This diagnostic run is not a new benchmark evaluation and must not be mixed with three-seed full-clean means.
- AUC establishes ranking association, not causal identification of the evidence source.

## Confidence Bands

| Confidence | n | harms | CFC AUC | Entropy AUC | Inv-conf AUC |
|---|---:|---:|---:|---:|---:|
| [0.00,0.70) | 25532 | 8029 | 0.493 | 0.479 | 0.558 |
| [0.70,0.80) | 14847 | 2742 | 0.501 | 0.457 | 0.555 |
| [0.80,0.90) | 15846 | 1577 | 0.509 | 0.448 | 0.554 |
| [0.90,1.00] | 19537 | 534 | 0.504 | 0.617 | 0.693 |
