# M04 Ranking And Distribution Results

M04 is complete for the fixed 16-cell, 80,840-evaluation prediction and offline
scoring package. Predictions were sealed before new independently admitted GT
scoring. The replacement CPU scorer succeeded; the failed 0604 scorer remains
unchanged and is not counted as a scientific result. No prediction was rerun.

## Completion Evidence

- Prediction parent: M04_factorial_20261102_2243, prepared
  02d9e169ae5bf5f7fd723f896ae5cb208c2278a1c85d7a2bd8f04b83d907f201.
- Parent prediction global:
  eb201ac715e7a06d23c2297dcb7332c60be813136a853157ba477951ed5320a3.
- Scoring suite: M04_offline_score_20261103_0646, prepared
  46f5c1b644cbc7a0361b08b3e5ef1bec0dcad77d96b64324b6960022f31b3ee9.
- Independent new admission:
  5f65b15bb0bc310bb37b3b83d296dac9d0912e05bfeb9350ba96a72580c04e48.
- Independent report:
  da78fc903041c227736a892e353c444d01df85260c067e95412c7586a7d20674.
- Scoring global:
  b17a1d44381276205c0b80fd75c86ac96c22e95980e8be95f010fde6ff981b57.
- Prospective analysis:
  cf08822ef2e406b4508f2ac8f1b2c3093edd92248fc322aa161107c2aae63867.

The manually dispatched trace-factorial-score-20261103-0646.service ran
2026-11-03 06:57:18 through 07:01:43 Beijing, controller time 258.446644s.
The retained service is loaded/active/exited, MainPID 0, ExecMainPID 2204185,
Code 1, Status 0, Result success, same invocation
f65ee607060b43e08aefdfbe7544155b. Dispatch and immediate live verification ran
in one tool orchestration: actual argv/exe/cwd, whitelisted environment,
invocation and effective 4GiB RAM/512MiB swap/one CPU/Nice10/idleIO cgroup
were checked. Logged cv2/torch threads were 1/1/1 and CUDA uninitialized.
Peak memory reached the 4GiB cap, swap peak was zero; actual exit was successful.

Main terminal verification revalidated all 312 runtime members, 68 native
bindings, parent/source/input/protocol/audit/report/STARTED/gates/raw cost/reset
metadata, the complete parent stat graph and all four new output hashes.
Outputs total 8,255,051 bytes. This terminal check did not rescan actual GT,
RGB or prediction arrays. External receipts are
M04_CPU_score_admission_validation_002.json,
M04_offline_score_execution_started_002.json and
M04_offline_score_terminal_001.json in the study root.

The 0604 attempt failed before annotation decoding because its CPU validator
used direct division instead of the pinned CUDA scalar reciprocal-multiply
operation. Only new unfrozen validation arithmetic/protocol binding was
repaired, independently reviewed and freshly frozen. Exact coefficient
equality remains; no tolerance, algorithm/config/selection change or score
rerun seeking a positive outcome was introduced.

## Fixed Factorial Design

D denotes the multiset of score magnitudes used as the distribution and O the
candidate ordering to which it is assigned. k denotes original confidence,
c the single-C diagnostic. The four arms are Dk_Ok (Kk), Dc_Oc (Cc),
Dc_Ok (C distribution with confidence ordering) and Dk_Oc (confidence
distribution with C ordering). All arms measure C on the same original
confidence-TopK pool, with the original objective, RNG and episodic reset.

Natural retains each distribution's native EPS admission; common_active
conditions every arm on their shared active candidates. The latter changes
the comparator and is not an unchanged production Confidence run. The schema
name native_C_minus_Confidence in common_active means its conditioned Cc-Kk
contrast, not a claim that both arms are the unmodified production methods.

The architecture/readout is NACLIP ViT-L/14 last18/24 under the existing clean
configuration. RGB uses native PIL bilinear, GT historical PIL NEAREST at
224x224; these are not the G package's OpenCV geometry. Scores below are native
pooled per-class I/U mIoU, not mean image IoU.

## Full-Population Scores

| Dataset | Study | Sources / Episodes | Dk_Ok | Dc_Oc | Dc_Ok | Dk_Oc |
| --- | --- | --- | --- | --- | --- | --- |
| Object | natural | 5000 / 385 | 29.361107 | 32.671159 | 30.963480 | 28.635600 |
| PC60 | natural | 5105 / 393 | 30.573158 | 30.782843 | 30.141726 | 30.675108 |
| Object | common_active | 5000 / 385 | 29.369258 | 32.671159 | 30.911142 | 28.743283 |
| PC60 | common_active | 5105 / 393 | 30.584040 | 30.782843 | 30.151959 | 30.669075 |

## All Prospective Contrasts

Units are mIoU percentage points. Brackets are 95% pointwise paired percentile
intervals from 10,000 whole original-episode PCG64 draws, shared across all
eight arms/studies within each dataset, including the partial final episode.
All contrasts and arm intervals had zero undefined draws. No draws were
dropped and no multiplicity correction was applied.

| Fixed Contrast | Object Natural | PC60 Natural | Object Common Active | PC60 Common Active |
| --- | --- | --- | --- | --- |
| Cc - Kk | +3.310052 [2.655806, 3.875565] | +0.209685 [0.065338, 0.348429] | +3.301901 [2.656346, 3.854385] | +0.198803 [0.057014, 0.335040] |
| C order - confidence order at C distribution | +1.707679 [1.223534, 2.151879] | +0.641117 [0.482589, 0.791911] | +1.760017 [1.268344, 2.200737] | +0.630884 [0.470505, 0.783031] |
| C order - confidence order at confidence distribution | -0.725507 [-1.117743, -0.307830] | +0.101950 [0.056772, 0.149078] | -0.625975 [-0.951717, -0.282566] | +0.085035 [0.039391, 0.130311] |
| C distribution - confidence distribution at C order | +4.035559 [3.284321, 4.658609] | +0.107735 [-0.025543, 0.239708] | +3.927875 [3.162963, 4.552427] | +0.113768 [-0.018522, 0.241992] |
| C distribution - confidence distribution at confidence order | +1.602373 [0.985038, 2.179105] | -0.431432 [-0.533588, -0.329592] | +1.541884 [0.930724, 2.113008] | -0.432081 [-0.533168, -0.330614] |
| Distribution-order interaction | +2.433186 [1.844696, 2.956681] | +0.539167 [0.392245, 0.682790] | +2.385992 [1.820066, 2.865251] | +0.545849 [0.394964, 0.689430] |

The natural Cc/Kk point scores equal the independently executed native
C/Confidence M02 anchors. Their intervals are not byte-identical because M04
uses its own prospectively fixed bootstrap namespace. Neither comparison is
an independent unseen-data replication or a license to select endpoints.

## Coverage And Cost

Object has 173,503 original candidates; C-based natural distributions activate
171,161, confidence distributions 173,503. All common-active arms retain and
activate 171,161. PC60 has 101,505 original candidates; the corresponding
numbers are 101,002 and 101,505, with all common-active arms at 101,002.
Every arm retains all 5,000 / 5,105 active sources. Natural admission
differences are retained rather than silently conditioned away.

All raw model seconds below include original episode 0; auxiliary/controller
and IO costs are distinct. Dk_Ok has Confidence update mathematics but still
pays for C diagnostics, so these rows are not the zero-diagnostic cost baseline
and do not answer V01's equal-time question.

| Arm | Object Model Seconds | PC60 Model Seconds |
| --- | --- | --- |
| natural.Dk_Ok | 1655.057604 | 460.199140 |
| natural.Dc_Oc | 1647.215982 | 460.054234 |
| natural.Dc_Ok | 1648.037722 | 459.330947 |
| natural.Dk_Oc | 1655.227645 | 459.811684 |
| common_active.Dk_Ok | 1646.732984 | 459.734051 |
| common_active.Dc_Oc | 1646.638095 | 459.603649 |
| common_active.Dc_Ok | 1646.286341 | 459.137472 |
| common_active.Dk_Oc | 1647.738956 | 459.553722 |

Every arm has 11,550 Object optimizer steps and 2,751 PC60 steps; C diagnostic
forward inputs are 173,503 / 101,505 per arm. Actual work and both raw and
warmup-adjusted costs are preserved in the sealed DATASET_RESULTS.json.
Near-equal measured times here do not establish equal-budget value or justify
reusing these timing anchors in V01.

## Scientific Boundary

Ranking and score distribution interact; neither is a universally separable
benefit. Object's C ordering helps at the C distribution but hurts at the
confidence distribution, including after common-active conditioning. PC60's
C ordering helps at both distributions, but replacing confidence magnitudes
by C magnitudes hurts under confidence ordering; under C ordering the
distribution contrast has an interval crossing zero. Interaction contrasts
are positive in both datasets/studies under the fixed pointwise protocol.

These are controlled full-pipeline contrasts under a fixed, historically
exposed configuration. They do not establish independent candidate risk,
isolated candidate causality, universal usefulness of C magnitudes or ordering,
unseen-data calibration, multi-seed robustness or compute-allocation value.
M01's adjusted sparse/null risk result, M02's PC60 spatial null, G's negative
SCLIP transfer and V01's unfinished equal-time study remain unchanged.
No result changes production C, chooses a new default or triggers retuning.

Claim-evidence check: interaction is supported by all four declared interaction
intervals; universal ranking benefit is contradicted by Object's confidence
distribution contrast; universal magnitude benefit is contradicted by PC60's
confidence-order contrast and is not established by its C-order null. The
remaining package is not complete, and this report is not a final paper.
