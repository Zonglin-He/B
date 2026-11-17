# TRACE Paper Completion Protocol

## Authorization And Fixed Boundaries

The actual human requested self-review and completion of all experiments needed
for a paper, explicitly including ablations. This is a new prospective study,
not permission to overwrite previous results or to optimize against exposed
test scores. The adopted method remains area_C. Dataset-specific L14 clean
parameters stay identical on the matching B16 and corruption conditions.
Official MLMP retains paper parameters and unchanged source. No new benchmark,
corruption type, recurring automation, monitor identity or baseline is added.

## Self-Review And Required Evidence

| Item | Reviewer question | Existing evidence | Required disposition |
|---|---|---|---|
| P00 | Does the method improve full benchmarks? | Verified 26-condition/78-arm comparison | Reuse sealed results; retain B16 negative outcomes |
| P01 | Is parameter selection separated from evaluation? | Corrected PC population overlaps old calibration | Audit exact IDs and source identities; exclude official validation from every new diagnostic panel; retain historical exposure |
| P02 | Is the gain C or a changed optimizer/CLS/batch? | Full TRACE differs from fixed MLMP in multiple fields | Matched-MLMP-parameter and selected-TRACE-parameter ablations |
| P03 | Does spatial C beat simpler weighting? | Old 32-source controls remain null/uncertain | Larger fixed-panel mass-matched constant, covered-constant and shuffled controls; confidence, relative-C, positive response and Region-PLPD controls |
| P04 | Do global, CLS and area terms matter? | Early cumulative pilots and old no-CLS controls | Remove one component at a time under the current implementation; retain interactions and negative results |
| P05 | Why does fixed B16 transfer weaken? | Object/PC60 stable negative differences | Same-source, same-configuration L14/B16 controls; C/support/gradient/update diagnostics; no B16 test tuning |
| P06 | Is C an independent update-risk predictor? | R03 historical grouped OOF primary NULL | Preserve all null/negative findings; no calibrated safety or causality claim |
| P07 | Does the context intervention preserve foreground? | 126 constructed cases pass raw/model integrity; no scientific inference yet | Original/sham/replacement/edge/context-benefit controls and label-isolated inference, then offline scoring |
| P08 | Is uncertainty correctly paired? | New benchmark has different shuffled update batches | Audit dependence components; shared whole-group draws only; report unidentifiable intervals rather than invent image-iid CIs |
| P09 | Is the method useful at equal time? | Old matching studies NULL | Time-only fine grid and at least three fixed episode arrangements; no interpolation or accuracy-selected match |
| P10 | What is the actual cost? | Old readout-only and sequential timings | End-to-end boundary stated, diagnostic/update/readout costs, synchronized latency and peak CUDA allocation; unchanged MLMP |
| P11 | Are remaining authorized baselines covered? | Full official MLMP complete; six clean TENT lower priority | Fixed official TENT without parameter search |
| P12 | Are claims reproducible and supported? | November3 paper is a protected completed release | New independent paper snapshot only after results; full tables, failure cases, nulls, references, compile/render/fonts/overfull review |

## Prespecified Ablation Families

All components will be tested with the same dataset population, checkpoint,
readout, preprocessing, label isolation and predeclared episode arrangements.
Controls do not become alternative test-selected main methods.

1. Fixed MLMP and its matching area-C addition, with unchanged MLMP settings.
2. Selected area-C and the same configuration without its regional bonus.
3. Removal of global pixel entropy, CLS entropy, and regional bonus separately.
4. Area-aware pixel weighting versus normalized per-region weighting.
5. Counterfactual C versus confidence, relative C, positive response,
   train-fold-calibrated confidence and a declared regional PLPD adaptation.
6. Constant-covered, constant-full-image and spatially shuffled bonus maps,
   preserving each image's total bonus mass.
7. Raw-query versus semantic-alias candidate construction and fixed top-K
   coverage controls, without silently changing the official final readout.

## First CPU Audit

Only sealed JSON metadata and compact SOURCE_COUNTS.npz are scientific inputs.
No raw RGB, GT, checkpoint, model execution or prediction-array payload is
required. The original suite's all-child/global success evidence and current
retained terminal identity are prerequisites, including explicit historical
preboot recovery entries. The analysis authenticates source memberships and
exact trial points, evaluates split overlap and dependence-component structure,
and records which paired intervals can actually be identified.

Whole components contain every source sharing an actual gradient update across
the contrasted arms and included arrangements. Zero-update NoAdapt batches do
not create adaptation edges. A conservative bootstrap eligibility rule requires
at least 30 components and no component exceeding one quarter of sources.
Eligible analyses use 10000 shared component draws and recompute class omission
within each draw. A single giant component is a design limitation, not a reason
to subtract marginal intervals, silently resample individual images, or drop
undefined draws. Trial dispersion is not a confidence interval.

Resource policy remains one shared GPU lock, GPU12GiB RAM/1GiB swap and
CPU4GiB RAM/1GiB swap, one CPU thread, hidden CUDA for CPU stages, OOMstop,
Restart=no. Failed bytes are retained; changed-code repairs require a new freeze
and a single controlled execution. CODE tests alone grant no model or GT result.
