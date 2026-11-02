# Probe Complementarity Audit

Descriptive mechanism characterization of existing development-exposed frozen arms, not a new independent validation. No causality, independent significance, necessity, or speed claim. Candidates within episodes are not independent replicates. All five datasets and both comparison directions are retained; no outcome selection.

## Scope And Definitions

All five A1338 full-split seed-0 mechanism cells; frozen candidates and arms. No GT or saved score streams used. Optional GT association is not performed.
Candidate comparisons: T<C iff T-C < -1e-6; C<T iff T-C > 1e-6; otherwise tie. Large gap: |T-C| >= 0.05 (no threshold tuning).
Active means weight > 1e-8. For each episode and each arm independently, zero inactive weights, divide by the sum of active weights, then compute TV = 0.5 * sum(abs(p_min-p_arm)) over aligned candidate indices. Empty distributions have undefined TV (null), never silently zero. Means/medians are episode-weighted, not candidate-weighted. Primary TV rows retain every episode with both distributions defined; the >=2 min-active subset is separate.

## Candidate Limiting Counts

Fractions below pool candidates within each dataset for description only. The JSON also retains the min-active-only denominator and directional large gaps.

| Dataset | Episodes | Candidates | T<C | C<T | Ties | Gap >=.05 | >=2 min-active episodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| coco_cd | 18 | 1998 | 1056 (52.853%) | 942 (47.147%) | 0 (0.000%) | 1221 (61.111%) | 18 |
| pc60 | 568 | 111514 | 64686 (58.007%) | 45959 (41.214%) | 869 (0.779%) | 58083 (52.086%) | 568 |
| voc21 | 91 | 46892 | 27389 (58.409%) | 19167 (40.875%) | 336 (0.717%) | 24870 (53.037%) | 91 |
| coco_object | 358 | 168669 | 103572 (61.405%) | 61173 (36.268%) | 3924 (2.326%) | 79334 (47.035%) | 358 |
| coco_stuff | 1000 | 164543 | 100612 (61.146%) | 60377 (36.694%) | 3554 (2.160%) | 74015 (44.982%) | 1000 |

## Normalized Allocation TV

Both directions retained. The last column gives mean/median/fraction>.05 for the >=2 min-active subset (its eligible n is in summary.json).

| Dataset | Comparison | Eligible n | Mean | Median | TV>.05 n (%) | >=2 subset: mean / median / fraction |
|---|---|---:|---:|---:|---:|---|
| coco_cd | min_vs_T_only | 18 | 0.128669 | 0.123917 | 18 (100.000%) | 0.128669 / 0.123917 / 1.000000 |
| coco_cd | min_vs_C_only | 18 | 0.095255 | 0.096594 | 18 (100.000%) | 0.095255 / 0.096594 / 1.000000 |
| pc60 | min_vs_T_only | 568 | 0.208134 | 0.206510 | 568 (100.000%) | 0.208134 / 0.206510 / 1.000000 |
| pc60 | min_vs_C_only | 568 | 0.308590 | 0.306680 | 568 (100.000%) | 0.308590 / 0.306680 / 1.000000 |
| voc21 | min_vs_T_only | 91 | 0.227840 | 0.223858 | 91 (100.000%) | 0.227840 / 0.223858 / 1.000000 |
| voc21 | min_vs_C_only | 91 | 0.325763 | 0.323552 | 91 (100.000%) | 0.325763 / 0.323552 / 1.000000 |
| coco_object | min_vs_T_only | 358 | 0.201875 | 0.200317 | 358 (100.000%) | 0.201875 / 0.200317 / 1.000000 |
| coco_object | min_vs_C_only | 358 | 0.343052 | 0.341948 | 358 (100.000%) | 0.343052 / 0.341948 / 1.000000 |
| coco_stuff | min_vs_T_only | 1000 | 0.210885 | 0.207587 | 1000 (100.000%) | 0.210885 / 0.207587 / 1.000000 |
| coco_stuff | min_vs_C_only | 1000 | 0.359163 | 0.355178 | 1000 (100.000%) | 0.359163 / 0.355178 / 1.000000 |

## Rank And Multiset Are Not Allocation

Same rank means the exact weak ordering of normalized shares, including exact ties, is unchanged. It does not mean the relative magnitudes are unchanged. Same multiset compares sorted normalized shares at absolute tolerance 1e-6 (rtol=0); same assignment compares shares at their original candidate indices with that tolerance. A permutation can preserve the multiset while changing assignments. These diagnostics compare min against each single-view arm, not a new permuted-arm outcome evaluation.

| Dataset | Comparison | Same rank | Same rank and TV>.05 | Same multiset | Same assignment | Multiset same, assignment different |
|---|---|---:|---:|---:|---:|---:|
| coco_cd | min_vs_T_only | 0 | 0 | 0 | 0 | 0 |
| coco_cd | min_vs_C_only | 0 | 0 | 0 | 0 | 0 |
| pc60 | min_vs_T_only | 0 | 0 | 0 | 0 | 0 |
| pc60 | min_vs_C_only | 0 | 0 | 0 | 0 | 0 |
| voc21 | min_vs_T_only | 0 | 0 | 0 | 0 | 0 |
| voc21 | min_vs_C_only | 0 | 0 | 0 | 0 | 0 |
| coco_object | min_vs_T_only | 0 | 0 | 0 | 0 | 0 |
| coco_object | min_vs_C_only | 0 | 0 | 0 | 0 | 0 |
| coco_stuff | min_vs_T_only | 0 | 0 | 0 | 0 | 0 |
| coco_stuff | min_vs_C_only | 0 | 0 | 0 | 0 | 0 |

## Empty And Limiter Nulls

- coco_cd: active episode bins (zero/one/>=2), min/T/C = {"C_only": {"at_least_two": 18, "one": 0, "zero": 0}, "T_only": {"at_least_two": 18, "one": 0, "zero": 0}, "min": {"at_least_two": 18, "one": 0, "zero": 0}}; both limiting directions present: 18 all-candidate episodes, 18 min-active episodes.
- pc60: active episode bins (zero/one/>=2), min/T/C = {"C_only": {"at_least_two": 568, "one": 0, "zero": 0}, "T_only": {"at_least_two": 568, "one": 0, "zero": 0}, "min": {"at_least_two": 568, "one": 0, "zero": 0}}; both limiting directions present: 568 all-candidate episodes, 568 min-active episodes.
- voc21: active episode bins (zero/one/>=2), min/T/C = {"C_only": {"at_least_two": 91, "one": 0, "zero": 0}, "T_only": {"at_least_two": 91, "one": 0, "zero": 0}, "min": {"at_least_two": 91, "one": 0, "zero": 0}}; both limiting directions present: 91 all-candidate episodes, 91 min-active episodes.
- coco_object: active episode bins (zero/one/>=2), min/T/C = {"C_only": {"at_least_two": 358, "one": 0, "zero": 0}, "T_only": {"at_least_two": 358, "one": 0, "zero": 0}, "min": {"at_least_two": 358, "one": 0, "zero": 0}}; both limiting directions present: 358 all-candidate episodes, 358 min-active episodes.
- coco_stuff: active episode bins (zero/one/>=2), min/T/C = {"C_only": {"at_least_two": 1000, "one": 0, "zero": 0}, "T_only": {"at_least_two": 1000, "one": 0, "zero": 0}, "min": {"at_least_two": 1000, "one": 0, "zero": 0}}; both limiting directions present: 1000 all-candidate episodes, 1000 min-active episodes.

## Provenance And Verification

Rehashed suite-linked manifest, queue state, five cell prediction/scoring seals and results, prediction protocols/panels/state/checkpoint identity metadata, and all 84 frozen-runtime source files. Checked complete episode NPZ population and NPZ/receipt coverage in linked sealed cache inventories. Episode receipt hashes are checked during streaming. No NPZ or prediction payload rehash; no prediction pixels, checkpoint payload, score-count streams, input images or annotations opened. Existing inventory hashes bind cache provenance, not a fresh cache-byte integrity proof.

The source inventory lists expected episode NPZ/receipt SHA-256 hashes from linked prediction seals. Consumed T/C/q/rho bytes are additionally fingerprinted in summary.json; that new fingerprint is not compared with an old independent fingerprint. Masks and identity arrays are not decompressed. No all-payload integrity claim is made.

Source: `/home/wwww/hzl/results/trace_self_evolution_20261028_1338`.
Audit code SHA-256: `f87d3806d501f07853419868fe2e66dfe07cd4af835860de9cec0ef7a8fad9c0`.
Test source SHA-256: `b70d8cd412b49a889ca4818e80d79516745c5115ad3f6e932ec704d600bada2a`.
`source_inventory.json` binds source seals and verified frozen-runtime source hashes. `episodes.jsonl` retains one descriptive record per episode. `AUDIT_SEALED.json` hashes all audit outputs and binds the two new source files.

No datasets, parameters, metrics or arms were selected using these results. No model import, inference, GPU work, annotation access or manuscript edit occurred.
