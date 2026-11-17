# TRACE Area-C: Add Adaptation Batch8 And16

## Authorization And Fixed Baseline

The latest actual human requests batch8and16 in TRACE tuning, explicitly keeping
MLMP at its paper configuration. TRACE's adaptation batch candidates become
1,2,4,8,16. MLMP remains lr.001/10steps/B2/alpha_cls1/seven prompts, with existing
declared B16 last9-layer transfer. No MLMP hyperparameter is searched or changed.
Diagnostic-view forward batch remains4 and is not the adaptation batch size.

## Append-Only Continuation

Original root TRACE_area_C_sensitivity_20261112-115044-v1 and its24pinned source
files remain unchanged. Its dispatcher was intentionally stopped after identity
verification while B16Object prediction was running; the independent child is
allowed to finish and its same-invocation success is required before continuation.
Do not call the stopped dispatcher a successful completed experiment.

The new finite queue reuses all sealed completed original predictions and scores,
including fixed MLMP and default TRACE. It executes missing original screens only
where no prediction directory exists; no failed/partial directory is retried.
Old compact I/U are merged into larger common bootstrap groups, retaining exact
pooled points. Original intervals are not copied into the expanded comparison.

Keep all78original configs. Add18: default-other-parameter batch8and16 single-factor
trials, plus16fixed-seed Sobol joint configs with batch restricted to8or16 and all
other original ranges unchanged. Thus96unique TRACE screen settings, with full
curves across five batch values and32total joint settings. Each new config uses
the same64training sources andtwo composition orders. Added predictions59904
across26conditions; combined old+new screen324480, excluding first4checks.

Scope remains8L14clean including Waterbirds,3B16clean,andone joint15VOC20S5family.
After all12extended screens, top6sensitive factors guide bounded joint refinement
on128training sources(max36configs), then best3plusdefault on the other128training
sources/three compositions(max4configs). Expanded batch levels are available to
both refinement and confirmation. Keep one chosen config across all15corruptions.
MLMP is rerun only at its unchanged settings for new refinement/confirmation
panels, not tuned. No heldout scores select parameters. Prior training exposure,
all losses/nulls/failures/undefined draws and unequal tuning costs remain explicit.

## Batch Execution And Shared Groups

Batch8/16 means actual8/16-image model forwards and optimizer updates, not gradient
accumulation or a relabeled microbatch. The new LargeBatchAreaC subclass changes
only batch validation/storage; the original objective/probe/optimizer/reset and
native final readout remain inherited. The old configure's batch field controls
no tensor operation; its other validation/reset behavior is reused without edits.

Screen/refinement uses nested16-source groups built from four old four-source
groups. All1/2/4/8/16-image adaptation batches stay inside those groups. Existing
small-batch episode membership and within-batch ordering are unchanged; only the
order of independent reset episodes can differ. B8 gets different pairings of
four-source subgroups. B16's two coarse orders have identical membership and are
not presented as two independent compositions or extra independent examples.

Confirmation uses32-source groups andthree predetermined index bit rotations.
This produces three different membership partitions for each ofB2,B4,B8,B16.
B1 compositions necessarily remain singleton; repeating orders is not an
independent-source replicate. Bootstrap always resamples whole common groups,
using10000shared draws across every arm/order/corruption, never per-image draws
that would split a large-batch dependency.

There are only4coarse groups,8refinement groups,and4confirmation groups. These
are weak uncertainty estimates: pointwise/unadjusted exploratory intervals only,
not stable significance or generalization evidence. Increasing the batch cannot
create more independent units. Selected maxima remain subject to selection bias.
No R05 independent-episode study is declared complete by this calibration queue.

## Execution Controls

New process-local worker bindings reuse the frozen prediction engine explicitly:
frozen-source check, class, phase design, trial list, panel lookup and episode
ordering. They do not mutate any running old process, baseline source, original
PLAN or original output. Each new root freezes all29dependency files and records
these bindings in PLAN. Separate new scoring/merging code handles the larger
groups and records import provenance; model code and score code are not mixed.

Default first4actualRGB must match historical area_C andofficialMLMP loss,
parameters andlogits<=1e-5 withzero changed labels before new predictions. Every
real batch checks finite losses/gradients/parameters. Numerical failures retain
partials and invalid settings; baseline/default/OOM/infrastructure failure stops
the queue. No automatic retry, smaller-batch substitution or resource-cap change.

One shared GPU lock;12GiB RAM/1GiBswap; CPU4GiB/1GiBswap/one thread/hiddenCUDA;
OOMstop/Restartno/10GiBdiskreserve. Separate CPU GT scoring only after sealed
predictions andactual same-invocationexit0. Completed old GT scores can be reused
without reopening raw GT. All actual process/cwd/argv/env/caps are recorded.
Original automation stays PAUSED, monitors OFF; no new agents/cron, uploads,
paper updates or formal test runs. Full scientific completion is not claimed.
