# M02 Spatial Preservation Results

M02 is complete: all8 full prediction cells were sealed before any GT scoring,
then a fresh independently admitted CPU scoring suite ran successfully. The
score service retained active/exited, MainPID0, ExecMainPID2135111, Code1,
Status0, Resultsuccess, same invocation35a355f3fe2b4e09ba80112a7bc950bb.
It ran2026-11-03 02:10:16 through02:11:48 Beijing,88.829483s controller time,
with4GiB RAM/512MiB swap/one CPU/hidden CUDA; peak memory917716992bytes.

Suite: M02_offline_score_20261103_0156; prepared
92f758aa4dd172b8487b034bd6217aae57d4e0e536b882f179b4ff38bb6032e5.
Global analysis SHA
ebc89d85ad0d72880f662c076583bd782bde94b703462fd6f110cce901b112a0.
Main verified285runtime files, parent and own metadata/admission/report/start,
all4 output hashes and6224939 output bytes after real service success. No
prediction/RGB/GT payload was rescanned during terminal verification.
External receipt: M02_offline_score_terminal_001.json in the study root.

## Full-Population Scores

All numbers are local native pooled class-I/U mIoU, not mean image IoU.
Native preprocessing/readout uses PIL bilinear RGB and historical PIL NEAREST
GT at224x224, NACLIP ViT-L/14 last18/24, fixed existing clean configuration.
These are not G's OpenCV-resized prediction/annotation results.

| Dataset | Sources / Episodes | Native C | Translated Preserve | Global Perturb | Confidence |
| --- | --- | --- | --- | --- | --- |
| Object | 5000 / 385 | 32.671159 | 31.842732 | 31.643692 | 29.361107 |
| PC60 | 5105 / 393 | 30.782843 | 30.677572 | 30.665917 | 30.573158 |

## Prospective Paired Contrasts

| Dataset | Native Minus Translated | Native Minus Global | Native Minus Confidence |
| --- | --- | --- | --- |
| Object | +0.828427 [0.433495,1.235680] | +1.027467 [0.605023,1.446426] | +3.310052 [2.646047,3.881292] |
| PC60 | +0.105271 [-0.028811,0.234378] | +0.116925 [-0.014564,0.244739] | +0.209685 [0.064268,0.351183] |

Units are mIoU percentage points; brackets are95% pointwise paired percentile
intervals. Each dataset uses10000 whole original-episode PCG64 draws shared by
all4 modes, including the last partial episode. No undefined draws occurred.
The prospective M02 analysis protocol SHA is
84df1e6183124027de039ef342bed1eb1d1238b8809dc9fc109c67d00ea9a21c.
The distinct M02 namespace gives different Monte Carlo endpoints from existing
E08 intervals; neither resampling run changes point outcomes or selects bins.

## Preservation And Admission Coverage

All modes used the same original candidate identity and original readout/loss.
Translated geometry was recomputed from each original16x16 mask and the frozen
deterministic translation policy during scoring. Translation retains equal
preserved area, but it does not hold pixel content/location or transformed
context identical; global perturbation changes the preservation condition.

| Dataset | Original Candidates | Native Active | Translated Active | Global Active | Confidence Active | Shiftable / Fallback |
| --- | --- | --- | --- | --- | --- | --- |
| Object | 173503 | 171161 | 172885 | 172898 | 173503 | 173043 / 460 |
| PC60 | 101505 | 101002 | 101179 | 101179 | 101505 | 100891 / 614 |

Summed original-candidate preserved-overlap fractions were1053.338434 on
Object and1718.552313 onPC60 across all translated candidate views, including
unshiftable fallbacks. These are coverage summaries, not new outcome estimates.
Natural EPS admission differs across controls; these contrasts include that
mechanism and are not a common-active candidate-only causal effect.

## Scientific Boundary

Object supports an advantage of original-candidate preservation over both
predeclared spatial controls in this fixed setting. PC60 point estimates have
the same direction, but both spatial-control intervals cross zero; the data do
not establish a spatial-specific advantage there. Native C exceeds Confidence
on both datasets under the fixed protocol. The result does not prove universal
candidate-preservation benefit, independent C risk information or isolated
candidate causality. M01's adjusted-risk sparse/null finding is unchanged.

Historical full/test exposure and prior clean-configuration selection remain.
Intervals are conditional on this fixed setup, not independent calibration,
multi-seed generalization or multiplicity-corrected significance. All negative,
null and fallback/admission coverage is retained. No score is used to change
production C, tune a spatial operator, select a new default or rerun outcomes.
M03/M04/V01 and the remaining fixed package are not completed by this result.
