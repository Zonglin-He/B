# TRACE simplification and measured implementation tradeoffs

Date: 2026-10-25. Branch: `codex/trace-simplification`. Package: 0.3.0.

## Scope and recoverable starting point

The worktree is `/home/wwww/.codex/worktrees/trace-simplification/hzl`.
Commit `d5b21e7` snapshots the relevant working TRACE implementation before this
change, including the pre-existing runtime optimizations. The original
`/home/wwww/hzl` checkout and its manuscript remain on their previous state.

Experimental artifacts are under
`/home/wwww/hzl/results/trace_simplification_20261024/`. They include immutable
reference/candidate packages, commands, source hashes, input IDs, candidate
responses, per-image intersection/union, predictions' checksums, runtime and
memory records. Datasets and results are excluded from Git.

## Method

The algorithm now has one rule for every candidate. The target-preserved,
target-suppressed, and context-perturbed views give mean predicted-class
supports T, D, and C at the original candidate locations:

```
q = min(T, C)
d = max(D - T, 0)
rho = q * (1 - d)
```

Connected components and the per-image top-K budget are unchanged. Candidate
views are evaluated in bounded batches, reliability is detached, and one
region-normalized entropy objective updates the executed visual LayerNorm
affine parameters. Source weights and Adam state are restored before each
test batch. Adapted patch logits are reconstructed into pixel predictions.

The production method no longer accepts `class_roles`, needs no thing/stuff
lexicon, and has no role-dependent intervention count or reliability branch.
Stuff/background remain valid segmentation classes. All three views form one
confidence-source tracing module. No fragment/area guard, CLS objective,
source floor, temperature, sibling suppression, region merging or extra loss
is introduced.

For valid probabilities, q <= T and q <= C, 0 <= d <= 1-T, and therefore
q*T <= rho <= q. Increasing D with T and C fixed cannot increase rho. These
are properties of the scoring function; they do not prove GT correctness,
an optimal aggregation rule, causal identification or a harmless update.
Because entropy is normalized by the total reliability, rho allocates relative
update coefficients. Multiplying every weight by the same positive constant
does not change the objective, and a singleton positive weight cancels.

## Implementation changes

- Removed semantic-role metadata and policies from the production package.
- Reused the original candidate confidence instead of recomputing its mean.
  Independently constructed probes without a confidence still compute it.
- Stored only indices, batch index and rho in production loss terms; CPU
  masks remain available in candidate diagnostics for visualization.
- Generated RGB masks within each view batch, so mask allocation participates
  in the existing OOM batch-halving retry. All candidate masks are no longer
  materialized on GPU at once.
- Initialized weak and counterfactual caches independently so a failed second
  allocation cannot leave a poisoned cache entry. Logit conversion/sanitizing
  allocations also participate in the forward-batch OOM retry.
- Disabled default layer-weight history recording and removed the unreachable
  prompt-integration branch. Multi-prompt entropy itself remains.
- Kept text embedding axes intact for a singleton class or prompt.
- Excluded unexecuted blocks and the unused reduced last-selected LN2 from
  optimizer membership, adapted-parameter names and reset snapshots.
- Replaced ordinary visual attention with PyTorch scaled-dot-product
  attention, explicit scale, zero dropout, and no causal mask. NaCLIP
  neighborhood and CSA attention retain their own formulas. Unused KK/Nonly
  experimental attention modes were removed.
- Used one diagnostic view-batch default, 16, across all dataset settings.
  Existing K, data batch, learning rate and step counts remain inherited.

The method hyperparameter is the candidate budget K. Learning rate, steps,
and data batch are optimization settings. View batch controls execution and
memory; its FP16 rounding effects were measured rather than assumed absent.
The fixed RGB operators remain grayscale/9x9 average blur and channel shift
with contrast factor 0.35.

## Accuracy and cost

No optimization setting was retuned for the comparison. View batches
8/16/32/64 were profiled on the same 40 training images for each of PC60 and
COCO-Stuff. Selection minimized mean relative synchronized runtime, choosing
the smaller batch within 3% of the minimum. This selected 16 without using
evaluation scores. Manual-kernel batch-8 and batch-16 results were both kept.

The fused kernel passed a predeclared training-panel gate: worst dataset
delta >= -0.25 mIoU and mean delta >= -0.10. The training changes were -0.063
on PC60 and +0.265 on COCO-Stuff. This is an engineering acceptance check on
small panels, not a statistical guarantee.

All numbers below are seed 0. COCO-CD is its complete 500-image balanced test
split. Other rows are fixed 512-image validation panels, selected by image ID
without labels; they are not full-dataset or three-seed main-table scores.

| Dataset | Images | Original role-aware | Uniform, manual kernel | Final, fused kernel |
|---|---:|---:|---:|---:|
| COCO-CD | 500 full | 75.16 | 75.18 | 75.04 |
| PC60 | 512 | 29.93 | 29.86 | 29.98 |
| COCO-Stuff | 512 | 21.14 | 21.13 | 21.37 |
| VOC21 | 512 | 52.32 | 52.35 | 52.61 |

Uniform/manual and final/fused both use view batch 16. The original role-aware
arm uses its inherited batch 8. Additional same-batch uniform controls and
all calibration points remain in the raw tables.

| Dataset | Original ms/image | Final ms/image | Change | Original peak MiB | Final peak MiB |
|---|---:|---:|---:|---:|---:|
| COCO-CD | 39.98 | 36.03 | -9.89% | 7786 | 6820 |
| PC60 | 142.89 | 143.26 | +0.26% | 3238 | 2928 |
| COCO-Stuff | 217.49 | 218.72 | +0.57% | 2335 | 2162 |
| VOC21 | 321.65 | 317.65 | -1.24% | 5014 | 4463 |

The fused kernel saves 5.98%--8.23% runtime relative to the uniform/manual arm.
Relative to the original role-aware method, the extra suppressed views largely
offset that saving on PC60 and COCO-Stuff. Memory falls by 7.41%--12.40% across
these four comparisons. The four-panel average mIoU change is +0.111 versus the
original, with an explicit -0.129 COCO-CD regression. This motivated adoption
as a measured simplification/cost tradeoff, not an all-dataset improvement.

Timing is GPU-synchronized adaptation plus final inference, excluding loading,
metric computation and model setup, with initial warm-up episodes excluded.
Hardware was RTX 5090, PyTorch 2.7.0+cu128. These are single-host measurements,
not device-independent speed guarantees or production latency estimates.

The final mask-streaming edit occurred after the four-panel measurements.
Its fixed 56-image GPU comparison against the measured fused snapshot is
recorded separately: all 224 candidate support/quality/dominance/reliability
rows matched exactly, but final prediction hashes did not. The measured
mIoU difference was -0.012, so no bitwise adaptation-trajectory preservation
is asserted. The final source was then rerun on all 500 COCO-CD images and
again obtained 75.04 mIoU, 36.21 ms/image and 6820 MiB peak allocation.
After the exception-path cache fix, another final-source full run obtained
75.05 mIoU, 36.20 ms/image and the same 6820 MiB peak allocation. The small
repeat-run variation is retained in the artifacts rather than hidden by
replacing earlier measurements.
Version metadata, the shared default (the experiments
explicitly passed 16), and strict discovery checks for missing data are the
other subsequent package changes. Full paired reruns remain necessary before
promoting final-source scores to a new benchmark main table.

## Numerical verification

The same-rule batch-8 reference and simplified implementation matched exactly
on 56 COCO-CD image prediction hashes and 224 candidate diagnostic rows.
This isolates the earlier engineering cleanup from the intentional role-policy
change. It is a bounded control.

CPU FP64/FP32 tests compare fused attention forward values and input/projection
gradients against the original manual formula. GPU FP16 is not numerically
identical: a fixed-weight, two-image COCO-Stuff probe measured maximum logit
error 0.40625, mean error 0.00987, token argmax agreement 99.02%, and relative
gradient L2 error 38.73%. Consequently no bitwise equality or unchanged-update
trajectory claim is made for the fused backend. The segmentation panel above
tests the resulting behavior directly.

## Data and vocabulary checks

The COCO Object background extensions previously included the literal query
`bed`, which also belongs to the bed class. The duplicate can tie the same
query response to background and prevent the bed class from winning collapse.
It is removed from background, and the vocabulary loader now rejects exact
queries assigned to two classes. A regression test verifies that bed can win.
No legacy COCO Object score is evidence for this corrected protocol; every
compared method must use the same repaired vocabulary in a future table.

Missing image/mask pairs and duplicate split IDs now raise errors. Fixed
experimental panels additionally require complete declared pair sets, their
official evaluation pool sizes, and the requested 40/56/512 records. Resume
checks bind data root, ordered image/label IDs, vocabulary, seed, configuration,
pool count, file size/mtime metadata and source identities before accepting
cached cells. Runtime resources, including tokenizer gzip, are fingerprinted.

The completed legacy manual panel used its original coordinator. Its exact
runner/coordinator are archived under `tools_snapshot_v2`. Input consistency
and completeness were checked again after completion; the derived audit is
explicitly retrospective and is not described as a pre-run content seal.

## Review and remaining evidence

Independent reviews identified and corrected optimizer membership, changed-root
resume reuse, incomplete fixed-panel acceptance, and partial image-cache state
after an OOM. Reports preserve the
findings, reproduction scope and fixes rather than deleting negative evidence.
The final review report is stored alongside the earlier rounds.

[Round 5](review-round-5.md) reports no unresolved actionable P1/P2 issue in
the established review scope. This closes the implementation findings; it
does not establish that no further scientific or deployment criticism exists.

The tests cover the full engine, A/B/A reset, all-invalid inputs, singleton
text axes, cached and manually constructed supports, real estimator terms,
entropy gradients, parameter membership and selected-layer execution,
streamed views and OOM retries, fixed-input resume and completeness failures,
query collisions, aggregation and package configurations.

At final verification 74 tests passed. The 0.3.0 wheel built successfully;
the extracted wheel passed configuration and packaged-resource import checks
without installing over the original working environment.

The main narrative remains connected candidates -> candidate-specific source
tracing -> reliability-weighted region entropy. Manuscript role-dependent
branches, role-map requirements and role-specific guarantees must be removed
when migrating to this implementation. Residual attenuation is an additional
soft comparison, not an independently demonstrated universal performance gain.

Nine-dataset full multi-seed evaluation, new-method harmful-update diagnostics,
and matched baselines for the repaired COCO Object vocabulary remain outside
this bounded implementation panel. Historical results must retain their old
method/protocol labels.

## Commands

```bash
PYTHONPATH=. python -m pytest tests -q

python tools/run_simplification_suite.py \
  --output /path/to/fresh/manual_panel \
  --reference-root /path/to/reference_snapshot \
  --candidate-root /path/to/uniform_manual_snapshot \
  --data-root /path/to/data

python tools/run_fused_panel.py \
  --output /path/to/fresh/fused_panel \
  --reference-panel /path/to/completed/manual_panel \
  --candidate-root /path/to/fused_snapshot \
  --data-root /path/to/data
```

Snapshots are intentionally external to production imports. The method package
has no dependency on an external benchmark repository. Use fresh output
directories after source/protocol changes; completed evidence is not silently
overwritten.
