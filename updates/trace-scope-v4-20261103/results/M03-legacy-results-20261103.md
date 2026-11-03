# M03 Single-C Versus Pinned Legacy Diagnostic Results

M03 is complete for its fixed8-cell/40,420-evaluation prediction and separately
admitted offline scoring package. Its primary contrast is mixed: single-C is
lower than the pinned legacy diagnostic on Object and higher on PC60 under
the same image objective. This does not establish a universal accuracy
advantage for the simplification. Negative and null results are retained.

## Completion Evidence

- Prediction suite M03_legacy_20261103_0017, prepared
  514b4b16531005b0ceea5c4fb9884387a637986f407e83bad0f9209708f7580b.
- Prediction global
  13daf2fb890492e7c905f1411d4eef806744f93325390a649cf6cc4feab0dd8c.
- New scoring suite M03_offline_score_20261103_0856, prepared
  1c2f7280a4ee0a2eda531745018cae56fa5ee8cabd94a0ffd9466db88a67226b.
- Independent exact CPU_GT AND M03_GT admission
  d2f2606250593138bae7c2f5584bd2bb0e7d5d6ac26dba3af90af607a73d09ac.
- Independent report
  955637742c540133efd26e0da8fa891642788190680219e79902a707ea7aff4f.
- Analysis global
  0ada9651e815e216394ce19aa6f0922ac997e27d651bb3a43f1894e4fe0c67e4.
- Prospective analysis
  d302adb2178d79188e2eced086ffc24965d1e2f0738467435f6950870dab4fc3.

The manually dispatched trace-legacy-score-20261103-0856.service ran
2026-11-03 09:05:22 through09:06:54 Beijing. Retained state is
loaded/active/exited, MainPID0, ExecMainPID2237275, Code1/Status0/Resultsuccess,
same invocation e1a5aadc86cb4268852178937a1884e3. Controller elapsed84.314824s.
Dispatch and immediate live verification were in one tool orchestration:
actual argv/exe/cwd/whitelisted environment/invocation and effective4GiB RAM,
512MiB swap, CPU100percent, Nice10, idleIO, OOMstop, Restartno, RemainAfterExit,
KillModecontrolgroup were verified. Logged cv2/torch threads were1/1/1 and CUDA
uninitialized. Peak RAM2,591,879,168B, swap peak0; actual exit was successful.

Main terminal verification revalidated325 frozen runtime members,68 native
bindings, exact source/input/protocol/parent startup/gates/audit/global/two
children, raw cost/reset and complete43,538-entry parent payload stat graph.
All four output hashes and6,228,658 bytes were checked. No terminal GT/RGB/
prediction-array payload rescan occurred. Authenticated48,504B archive content
was read only through the protected five-component loader; no model was loaded.
Receipts are M03_prediction_terminal_001.json,
M03_CPU_score_admission_validation_001.json,
M03_offline_score_execution_started_001.json and
M03_offline_score_terminal_001.json in the study root.

## Fixed Design And Estimand

C/legacy diagnostic is crossed with image/episode objective. Every arm uses
the same original confidenceTopK candidate pool, current engine/Adam/readout,
source order, config, RNG and exact episodic reset. Native NACLIP ViT-L/14
last18/24 and original PIL bilinear RGB/PIL indexed NEAREST224 GT are used,
not the G package's OpenCV geometry. Scores are native pooled class I/U mIoU,
not average per-image IoU.

The pinned archive is
d88873fa15ada2751e07d7129cb30e96c94f832fbf4c053d63b10384f6b1dc46.
Only constants, role taxonomy, reliability, indexing and interventions are
loaded. The legacy role/view/score logic is genuine authenticated component
code. Old engine, optimizer, OOM retry and project restoration are not used.
The episode objective is pinned to
d1dce2a27de03a371b8d00716609c49676b6a444c6d0f4e39b3c09963c8af589.
This is a diagnostic-component comparison within the current pipeline, not
a reproduction of the entire historical method.

## All Four Scores

| Dataset | Sources / Original Episodes | C.image | C.episode | legacy.image | legacy.episode |
| --- | --- | --- | --- | --- | --- |
| Object | 5000 / 385 | 32.671159 | 32.623965 | 33.051460 | 33.007537 |
| PC60 | 5105 / 393 | 30.782843 | 30.851537 | 30.641330 | 30.714876 |

All four per-arm intervals and full canonical records remain in sealed
DATASET_RESULTS.json; no winner is selected for production.

## All Five Prospective Contrasts

Units are mIoU percentage points, C minus legacy or image minus episode as
stated. Intervals are95percent pointwise linear percentiles of10,000 shared
whole original-source-episode PCG64 draws, root20261102, fixed M03 namespace.
Partial final episodes and episode0 are retained. All8 arm and10 contrast
intervals have zero undefined draws; no draw removal or multiplicity
adjustment was applied.

| Fixed Contrast | Object Delta [95% CI] | PC60 Delta [95% CI] |
| --- | --- | --- |
| C.image - legacy.image (primary) | -0.380302 [-0.742158, -0.040039] | +0.141513 [0.031509, 0.250399] |
| C.episode - legacy.episode | -0.383572 [-0.744255, -0.026695] | +0.136661 [0.030262, 0.241688] |
| C.image - C.episode | +0.047194 [-0.226658, 0.277763] | -0.068694 [-0.134034, -0.005848] |
| legacy.image - legacy.episode | +0.043923 [-0.255293, 0.299258] | -0.073546 [-0.138389, -0.005904] |
| (C.image-C.episode) - (legacy.image-legacy.episode) | +0.003270 [-0.326725, 0.332871] | +0.004852 [-0.078441, 0.083852] |

The primary Object interval is negative and the PC60 interval positive.
Both episode-objective diagnostic contrasts have the same dataset-dependent
signs. On PC60, image-minus-episode is negative for both diagnostics; both
Object objective intervals cross zero. Interaction intervals cross zero on
both datasets; this is not proof of equivalence or absence of interaction.

C.image point scores match M02's native C anchors. Their intervals differ
because M03 uses its separately prospective bootstrap namespace. This is not
an independent unseen-data replication, endpoint selection, or reason to
replace another study's sealed interval.

## Coverage And Descriptive Cost

Each original pool is shared across all four arms; EPS-active counts may
differ by diagnostic and are not silently conditioned away. All modes retain
all sources. The same per-diagnostic initial candidate identities and cost/
reset records were authenticated before GT.

| Dataset | Original Candidates | C Active | Legacy Active | C Diagnostic Inputs | Legacy Diagnostic Inputs |
| --- | --- | --- | --- | --- | --- |
| Object | 173503 | 171161 | 165204 | 173503 | 440935 |
| PC60 | 101505 | 101002 | 100299 | 101505 | 242915 |

Raw model seconds below include original episode0; full raw rows, encoder
work and warmup-adjusted descriptive throughput remain separately sealed.
These are observed costs, not a controlled equal-time accuracy comparison,
timing uncertainty estimate or V01 result.

| Arm | Object Model Seconds | PC60 Model Seconds | Object / PC60 Optimizer Steps |
| --- | --- | --- | --- |
| C.image | 1636.948665 | 451.523086 | 11550 / 2751 |
| C.episode | 1630.053885 | 450.158662 | 11550 / 2751 |
| legacy.image | 2053.977867 | 679.829984 | 11550 / 2751 |
| legacy.episode | 2048.044615 | 678.530229 | 11550 / 2751 |

Single-C used fewer diagnostic views and lower observed raw model time in
both datasets, with the documented opposite-signed accuracy comparison.
This is a bounded accuracy/diagnostic-work tradeoff, not universal dominance,
independent risk calibration, candidate causality or an equal-budget claim.
V01's fixed accuracy-time protocol remains separate and unfinished.

## Claim Boundaries

Settings retain historical WB development overlap and test exposure. The
intervals condition on this fixed exposed configuration; they are not
untouched generalization, independent hyperparameter selection or familywise
significance guarantees. Raw role-specific views and active-population
differences are part of the diagnostic replacement, not isolated causal
effects of a single component. No config or production C change, extra seed,
post-result selection, rebinning, retry for a positive outcome or inference
rerun followed these results. The negative Object finding and every null
secondary interval remain part of the scientific record. The overall
necessary TRACE evidence package is still ACTIVE/incomplete.
