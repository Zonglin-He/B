# Current TRACE: Adversarial Evidence Audit

This is a live research audit, not a completed manuscript, acceptance forecast,
or a replacement for the protected November3 release. It records the actual
November16 findings before the remaining frozen experiments finish. Numerical
claims below refer to the exact sealed reports named in the evidence column.

| Priority | Reviewer objection | Actual evidence | Required treatment |
|---|---|---|---|
| Critical | Full-system improvements do not identify the contribution of C. | Four L14 evaluation-disjoint training diagnostics: selected-minus-no-region intervals all include zero. Object/PC60 no-region point estimates exceed full area_C; uniform or shuffled mass sometimes also exceeds full. See trace-paper-L14-ablation-results-20261116.md. | No claim that C explains the system gain, is necessary in every dataset, or beats simpler weighting universally. Report every control. Full Object/PC60 fixed component controls are frozen but not yet results. |
| High | Dataset-specific tuning can explain part of the main contrast. | TRACE uses22 selected fields; MLMP remains official0.001/10/B2/seven prompts. The diagnostic default-area comparison isolates adding the region term more closely than the selected-system comparison. | Separate matched-default addition, selected-system performance and selected-minus-no-region ablation. Do not call these the same estimand. |
| High | Corrected PC validation is not untouched by historical selection. | Calibration overlaps corrected reference: PC59 screen/refine/confirm33/62/59; PC6034/61/61. | Disclose exact exposure and the corrected5104 population. New diagnostics exclude all actual official evaluation IDs and physical aliases. Do not claim an untouched TRACE PC test. |
| High | General backbone robustness is contradicted. | Full fixed-transfer B16 Object-0.7367pp, PC60-0.4133pp, VOC20+0.2567pp versus MLMP on official-rounded means. | Report stable negative transfer, not random run instability. Candidate/gradient metadata are descriptive only; do not infer a proven causal source. No B16 test retuning. |
| High | Official-source execution does not guarantee paper-value reproduction. | Full L14 Object NoAdapt25.03 versus paper23.80; MLMP30.42 versus28.84. Source/metric/geometry are pinned but discrepancy remains open. | State the remaining mismatch and declared runtime extensions. Never tune a baseline to hit a desired paper number or call scalar agreement a complete equivalence proof. |
| High | The reliability interpretation is stronger than the evidence. | Historical R03 primary hard-harm OOF increments: all24 pointwise intervals cross zero; secondary nonlinear quality-adjusted Object losses worsen. | Call C context-conditioned self-support/allocation, not calibrated correctness, independent risk estimation, safe updates or a causal guarantee. Preserve the whole R03 table and cohort history. |
| High | Efficiency cannot use diagnostic shared-view timings. | Ablation wrapper amortizes probes and adds a measurement forward. Current equal-time pilot/formal stages are separately frozen and not yet complete. | Standalone timing comes only from the synchronized E2E time study with its specified boundary. Preserve NULL pilot selections and formal budget failures. No interpolation or accuracy-selected steps. |
| High | Standard image bootstrap breaks shared-update dependence. | Old full TRACE/MLMP crossed arrangements form a single dependence component. | No image-iid CI or marginal-CI subtraction. New controls use common8-source blocks and10000 shared whole-block draws, explicitly pointwise and conditional on fixed runs. Three orders are not three independent initialization seeds. |
| Medium | Rare Waterbirds groups make a diagnostic worst-group CI undefined. | Diagnostic group counts134/1/1/120;3605 of10000 worst-group draws undefined. | Retain undefined draws and no CI. Full strict-test group segmentation results are separate from diagnostic results and from classification worst-group accuracy. |
| Medium | Background replacement can change the foreground through transforms. |126 historical constructions have raw/model proofs; remaining inference uses exact224 RGB without another resize and verifies native normalized foreground. | Require actual normalized foreground/sham checks and identical-sham predictions. Report selected-instance foreground accuracy, unavailable21/102, helpful/harmful/null contexts. Not full-composite mIoU or an independent Waterbirds test. |
| Medium | NoAdapt-to-MLMP includes a readout/prompt change. | Official NoAdapt uses a single reference prompt/final layer; MLMP uses its native multiple prompts/layers. | Treat it as a complete-system contrast, not a pure gradient-update effect. Keep MLMP zero-update readout control separately named in the context study. |

## Contribution

The current evidence supports reporting an evaluated area-aware contextual
allocation system with mixed transfer and bounded diagnostic findings. It does
not yet support the stronger scientific claim that counterfactual C is the
identified cause of the observed full-system gains. More completed experiments
can test this claim; completion alone cannot make the claim true.

## Reproducible Method

The adopted area_C objective retains global pixel entropy and CLS entropy,
plus a regional C bonus normalized over the full image grid. It is not the
older region-only image-normalized loss. C is not an absolute update-admission
gate: the main method still executes its fixed selected steps. Top-K candidates
are fixed at the initial parameters. Readout aliases and raw-query candidates
must be distinguished. Layer selection describes feature/readout aggregation,
not freezing every parameter outside those layers; actual optimized parameters
are visual LayerNorm parameters.

## Completion Requirements

1. Authenticate all seven diagnostic prediction/score exits and all126 cells.
2. Finish the already frozen R04, time-only pilot/selection, formal full-time
   comparison and six official TENT tasks, preserving failures and NULLs.
3. Finish the already frozen full Object/PC60 C-versus-no-region controls.
4. Authenticate the derived candidate/gradient/update metadata and its limited
   first8-source-block gradient coverage without causal extrapolation.
5. Assemble all exact/official-rounded tables with their distinct metric scopes,
   uncertainty eligibility, parameter provenance and retained historical results.
6. Only then create an independent new paper snapshot. Compile, render every
   page, inspect fonts/references/overfulls, and perform a fresh five-dimension
   self-review. No old paper, release, reviews, result bytes or main checkout
   may be overwritten.

The current queues being started satisfies none of the pending result claims
by itself. No all-paper completion is asserted in this audit.
