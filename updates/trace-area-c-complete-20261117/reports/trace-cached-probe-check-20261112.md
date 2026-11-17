# Native Diagnostic Readout Cache Check

This is an engineering equivalence check, not a new accuracy experiment.
Use the first four already-fixed training RGB sources in each clean/original
15 S5 condition. No new selection, validation image, GT access, or hyperparameter
search. Wait for the active seven-control pipeline's actual successful exit;
verify its parent prediction/score seals and use the one shared GPU lock.

Keep the official MLMP source and all previous scientific outputs unchanged.
The optional cache stores raw selected-layer reduced features BEFORE ln_post
only within one raw_logits call. It uses the official all-layer pass to compute
adaptive entropy weights, then performs the same pre-ln_post weighted reduction
and projection. Never average already-normalized/projected features. No cache
survives an optimizer update or reference-parameter swap. Evaluation mode only.

For each original B2 episode, compare B1/B2 real images and B4 actual
counterfactual views, before and after ten real area_C updates. Check top8 and
all-semantic-region C maps, including frozen-reference parameter restoration.
Run complete old/cached area_C ten-step episodes with independent resets and
compare losses, updated parameters and final unchanged-native readout labels.
Thresholds: all numeric differences <=1e-5; zero changed semantic labels.
Check exactly two versus one visual forwards per diagnostic call, hook cleanup,
no optimized-parameter/optimizer-state mutation and no weights_track growth.

Time uncached/cached B2 and B4 calls after one warmup each, four repetitions
in fixed AB/BA/AB/BA order with CUDA synchronization. Report all samples and
aggregate by batch size. This is diagnostic-call timing, not total pipeline
speedup or equal-compute evidence. No formal speed significance test.

GPU12GiB RAM/1GiB swap, single thread libraries, OOMstop/Restartno,
full actual STARTED execution envelope and immutable code snapshot. Reject and
retain a failed attempt rather than relaxing parity. No automatic retry.
Successful outputs require actual same-PID/invocation exit0 plus hash seals.
