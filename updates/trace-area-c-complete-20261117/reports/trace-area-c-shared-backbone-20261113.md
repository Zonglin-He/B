# TRACE Shared Hyperparameters Across Backbones

Actual human instruction: the second backbone must use the first backbone's
parameters. This replaces independent B16 parameter selection, not the area_C
objective, inference geometry, data split, MLMP configuration, or resource caps.

## Selection Rule

Select one complete 22-field TRACE configuration per L14 calibration family.
For the existing B16 COCO Object, PC60, and VOC20 tasks, copy the corresponding
clean L14 configuration exactly. Copy learning rate, optimizer parameters,
steps, batch size, loss weights, temperatures, prompts, candidate and probe
parameters, and numeric layer_fraction without backbone-specific scaling.
The fraction maps to each architecture's actual depth; checkpoint and feature
dimensions necessarily differ. MLMP stays at .001 learning rate, 10 steps,
batch2, alpha_cls1 and seven prompts with its existing native backbone setup.

Clean VOC20 and the joint VOC20-C S5 family remain separate calibration
families. All15 S5 types share one configuration; no per-corruption selection.
This change does not add new B16 datasets or corruption experiments.

## Finite Continuation

1. Preserve the intentionally stopped old dispatcher and partial independent
   B16 PC60 refinement, including exact invocation logs. Neither is completed.
2. Reuse the sealed training inputs, all eight completed L14 clean refinements,
   and completed L14 joint15S5 screen. Verify retained actual successful parents.
3. Run the remaining joint15S5 refinement with the unchanged bounded top6-factor
   design, 128 training sources per type and two episode orders.
4. Run nine L14 confirmations, best3 plus default, 128 confirmation training
   sources and three predeclared episode compositions.
5. Export the L14-selected configurations and their exact result hashes before
   B16 transfer. Evaluate each fixed B16 configuration, default TRACE reference,
   fixed MLMP and NoAdapt on the confirmation training panel. Never select
   another configuration based on B16 results, even if the transfer fails.
6. Summarize every L14-selected configuration and its B16 transfer result.

The old independent B16 coarse screens and Object refinement are historical
only. Their selection outcomes are excluded from the new policy. Their prior
exposure remains disclosed; B16 cannot be described as never inspected or
untouched. No test data are used for parameter selection and no new full-test
evaluation is implicitly authorized by this continuation.

## Execution And Interpretation

Only new driver/tests/protocol files are added. All29 prior frozen source files,
plans, and results remain unchanged. Existing GPU prediction engine, realfirst4
parity checks, label guard, lock, model reset, and scoring primitives are reused.
GPU12GiB/1GiB swap; CPU4GiB/1GiB swap/one thread/hiddenCUDA; Restart=no,
OOMPolicy=stop. A failed stage is retained without automatic retries.
Original automation is PAUSED and monitors are OFF. This is a finite sequential
experimental dependency queue, not a recurring monitoring task.

CPU scoring requires sealed predictions and the same actual successful process
termination, exact source IDs, episode composition, step counts, and hashes.
Paired I/U intervals use shared10000 whole-group draws: 16-source groups for
refinement, 32-source groups for confirmation/transfer. Only8/4 groups are
available; intervals are exploratory, pointwise and unadjusted. Reordered
episodes are not independent image samples. Preserve failures, undefined draws,
negative/null outcomes and prior training exposure. L14 selection maxima and
training-panel B16 transfer results are not new full-validation performance.
