# Dataset response and forward budget study

Status update: the user subsequently requested algorithmic efficiency instead
of parameter tuning. The parameter-only continuation below was superseded
before any GPU cell started. See `algorithm-efficiency-study.md` for the active
study. The completed CPU audits and the original VOC21 batch search remain valid.

This continuation keeps the released algorithm and the currently running
VOC21 batch-size experiment unchanged. The studies are on the isolated
`codex/trace-simplification` branch. Artifacts are under
`/home/wwww/hzl/results/trace_efficiency_diagnosis_20261025`.

## What the completed removal experiments establish

Use the audited Full COCO-CD control (74.9568), not the older external control.
Full minus confidence-only weighting is +0.4662 on VOC21, +0.5992 on PC60,
+1.5919 on COCO Object, and +7.7911 on COCO-CD. The first three are matched
512-image panels; COCO-CD is its complete 500-image split. These are not
Full-minus-MLMP differences or three-seed estimates.

Removing region normalization changes mIoU by -1.4628, -0.7729, +0.5548,
and -2.5033 respectively. The opposite sign on COCO Object is evidence
against a universal benefit claim for this allocation rule. It is not proof
that a different objective will generalize. The removed-construction control
also changes mask count, extent and coverage, so its effect cannot be
attributed to connectedness alone.

The VOC21 matched-panel diagnosis finds sofa (-3.703), pottedplant (-2.989),
and background (-1.330) harmed relative to confidence weighting, while chair
improves (+4.928). Plant's relative loss coefficient increases even as its
IoU falls. Thus "the losing classes just need more weight" is not supported.
These are allocation coefficients, not observed gradient contributions.

The active batch search already varies training-calibration mIoU substantially
without changing the formula. All sixteen integer values must finish before
selection. Batch size changes which images share an episodic update and
normalizer; it is not solely a GPU utilization setting. This search estimates
its conditional effect with the old K/lr/steps fixed, not an independent
universal optimal batch size.

The per-image cached prefix audit is an additional warning against assuming
that lower K only removes useless regions. At K=16, the mean retained episode
coefficient mass is 65.47% for VOC21, 80.48% for PC60 and 60.70% for COCO
Object; corresponding diagnostic-view reductions are 50.86%, 27.04%, 52.68%.
At K=8 the retained VOC21 mass is only 35.04%. Ranking uses original confidence,
not the subsequently measured reliability. These are cached allocation/count
calculations, not predictions of new mIoU or GPU runtime. Cross-dataset
absolute reliability medians also have different vocabulary/softmax class
counts and do not measure GT correctness on a common calibrated scale.

## Measuring compute correctly

For P input crops, N selected candidates and S adaptation steps, the current
nonempty episode encodes `3*N + (S+2)*P` visual inputs: three diagnostic views,
one original source pass, S training passes and final inference. Backpropagation
adds cost beyond this forward-input count. Batching changes model-call count
and kernel utilization; it does not remove the underlying inputs.

Across the completed panels, diagnostic views per image are about 97.47
(VOC21), 65.67 (PC60), 101.18 (COCO Object), and 11.99 (COCO-CD). These counts
explain why a fixed optimization cannot be assumed to save the same fraction
of time on every dataset. Recorded timing is synchronized adaptation plus
final inference on this host, excluding loading and scoring. Original and
refreshed evidence have different environment-recording completeness; use the
new paired study for quantitative speedup claims.

`tools/analyze_forward_budget.py` audits cached supports and the inequality
`q*T <= rho <= q`. Replacing residual attenuation with q changes an individual
raw weight by at most `q*(1-T)`. That bound is on a scalar weight, not on
normalized coefficient distance, gradients, Adam trajectories or mIoU.
The script explicitly checks normalized coefficients, EPS-pruned candidates,
empty objectives and a bounded entropy difference at the SAME logits.

Exact objective-preserving residual skips save only 0.395%, 0.384%, 1.567%,
and 0% of diagnostic inputs in these four panels. A fixed bound threshold
0.001 would save 8.666%, 9.048%, 13.514%, and 0.234%, but also activates
8/6/20/0 previously EPS-pruned candidates. This approximate strategy is
offline analysis only and is not installed in production. Separating two
view stages also changes batch packing, so fewer inputs do not guarantee
proportional latency reduction or numerical identity.

## Parameter-only continuation

`tools/run_accuracy_budget_search.py` is queued after all sixteen VOC21 batch
trials and their selected full validation. It inherits the batch selected by
TRAINING calibration, never selects a batch based on that full validation.
The source package, search rules and input identities are frozen before GPU
execution. It acquires the batch-tuner lock before the audit/primary locks
and passes locks into workers. Existing queues are not interrupted.

Two datasets are included: VOC21 and PC60. For each, freeze 256 official
training images (selection seed 2027) and 512 validation images (selection
seed 20261024). Image and annotation paths must be disjoint. The validation
panel is a development panel already used by previous ablations, not a new
untouched test cohort. Parameters are selected only on the training panel.

Coordinate order:

1. K: 4, 8, 16, 32 and the inherited K, with duplicates removed.
2. Steps: 4, 8, 16, 24 and the inherited step count.
3. Learning rate: 0.5, 1.0, 1.5 times the inherited value.

Each stage fixes every other parameter. This is one bounded coordinate pass,
not an exhaustive joint optimum. Batch size remains fixed throughout it.
After the stages, choose from ALL visited configurations by global maximum
calibration mIoU. Prefer lower measured time only within 0.02 points of that
maximum. Keep every trial and the full accuracy/time Pareto set. Run the
starting and selected configurations on the same 512 validation images with
the same new worker and environment. A validation decrease is retained and
does not trigger selection of a different trial. Do not automatically change
production defaults or manuscript tables.

COCO Object retains its completed CPU diagnosis, but its new parameter search
is deferred: the converted object-mask training directory is absent. Raw
COCO training images and instance JSON exist, but need an audited conversion
matching the validation semantics before use. The initial failed preparation
is preserved under `parameter_search/`; the executable two-dataset study is
`parameter_search_final/`. The unexecuted v2/v3 preparations are retained after review
required stricter upstream artifact validation. Validation labels are not
substituted for training. Any data-batch OOM stops as unresolved rather than
silently excluding a setting; view batching retains the existing bounded retry.
COCO-CD already uses only K=4 and five steps, so the expensive configurations
take priority. Slow datasets, a second backbone and ACDC are not added.

The study changes no reliability formula, role mapping, candidate ranking,
intervention operator, prompt/layer protocol, entropy normalization, or
optimizer type. Three views remain one confidence-source tracing component.
