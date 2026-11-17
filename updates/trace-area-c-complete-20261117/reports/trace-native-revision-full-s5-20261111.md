# Fixed Two-Arm Full S5 Revision

## Prospective Decision

Fixed on November11 after the first32-source training pilot, before the two
coverage-followup S5 scores. Evaluate BOTH `area_C` and
`coverage_native_budget` on all original1449 VOC20 validation sources and all
original15 severity5 corruption types. No winning validation arm is selected
for a revised superiority claim. No parameter changes or automatic retry.

The training pilot found that restoring global pixel/CLS optimization closes
much of the regional-loss deficit; area weighting is slightly positive versus
MLMP on the training S5 macro, with a zero-crossing interval. Raw-C whole-update
attenuation is negative. The two full arms preserve native optimization budget
and distinguish fixed top8 from all-region/semantic-merge/refresh coverage.
The second contrast is a combined coverage block, not an isolated K effect.

## Exact Evaluation

- Unchanged official MLMP source and settings, L14,7 prompts, native readout,
  last18 visual layers, Adam lr.001,10 steps, B2 and final single-image episode.
- Global pixel+CLS objective, bounded per-pixel C bonus coefficient1.
- `area_C`: confidence top8 query regions, fixed for the episode.
- `coverage_native_budget`: all semantic connected regions, refresh at0/5,
  original frozen model for C, native full update budget. This does NOT claim
  that raw C can decide whether the native MLMP update should occur.
- Original sealed21735 RGB condition-images reused, no new corruption or
  regeneration.43470 new predictions. No GT in prediction.
- Reuse actual completed MLMP/NoAdapt and historical TRACE compact counts from
  `R01_MLMP_S5_direct_20261110_0915`, with parent true-success and all-child/global
  metadata checks. Actual first4 baseline-label replay and native-equivalent
  loss/parameter/logit parity <=1e-5/changedlabels0 for every corruption.
- Separate CPU scoring only after complete predictions and actual GPU exit0.
  Indexed GT, CV2 nearest224, reducebackground/ignore255, original ordered B2
  groups and GT hashes identical to baseline.10000 common whole-group draws,
  seed20261111415, arithmetic15-type macro, pointwise95/no multiplicity adjustment.
- Both new arms compared against native MLMP and historical TRACE, every type
  and macro, including negative/null/undefined results. No outcome-driven stop.

## Boundaries

These validation data have already been used in historical studies. This is
an exploratory full re-evaluation, NOT independent unseen-test confirmation.
Both new arms share native MLMP settings/readout but cost more diagnostic
forwards; no equal-wallclock, equal-calibration-budget or SOTA claim.
Historical TRACE has different native budget/readout and remains separately
labeled. No old artifact, paper or release is overwritten.

Finite pipeline only; original automation remains PAUSED, monitors OFF. One
existing GPU lock,12GiB RAM/1GiB swap, OOMstop/Restartno; offline CPU4GiB,
one thread/hiddenCUDA. No external upload in this round.
