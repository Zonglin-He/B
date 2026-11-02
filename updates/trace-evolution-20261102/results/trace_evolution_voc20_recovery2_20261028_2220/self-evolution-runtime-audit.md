# Runtime repeatability amendment

The first frozen suite `trace_self_evolution_20261028_1328` stopped on its
first four-image COCO-CD smoke cell. It did not reach full evaluation or
ground-truth scoring. Its failing gate and artifacts are retained unchanged.
Candidates and all diagnostic supports agreed exactly; post-update loss
differed by 0.021861 and 6,322 of 200,704 predicted labels differed.

## GT-free isolation

`trace_evolution_repeatability_20261028_1332` replays the same four unlabeled
inputs under old/old, new/new and old/new implementation pairs. All resets
and RNG states are controlled; gradient and final trainable-weight hashes
are recorded. No label image is opened and no accuracy is selected.

With the default nondeterministic runtime, old/old differs at 11,024 labels
and new/new at 4,583 labels, despite equal diagnostic supports. Thus the
first gate cannot be interpreted as evidence of a two-view formula change.

Enabling deterministic PyTorch algorithms, deterministic cuDNN and the
documented cuBLAS workspace while retaining default attention dispatch makes
all three replay pairs agree exactly: supports, per-step losses, gradient
hashes, trained parameters and pixel labels. A separate forced-math attention
control also agrees exactly within that runtime but changes initial floating-
point responses; it is not needed and is not the chosen implementation.

## New-suite rule

Use deterministic default attention dispatch with
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`,
`cudnn.deterministic=True`, and `cudnn.benchmark=False`. Keep the strict
equivalence thresholds and every algorithm/configuration setting unchanged.
Make a new source snapshot and manifest, never edit or relabel the failed
suite. Record the actual flags and environment in each prediction protocol.

Historical nondeterministic full scores remain historical operating points;
new deterministic results must be recorded independently. With fixed order
and no stochastic adaptation component, seed repetitions test replay rather
than independent data-order or population robustness. Do not present their
standard deviation as uncertainty over arbitrary episodes.
