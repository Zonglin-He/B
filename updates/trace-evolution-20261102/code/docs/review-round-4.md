# Independent Review Round 4

## Findings

### [P2] Do not leave an incomplete image cache after a recoverable OOM

Location: `trace_ovss/method/trace/interventions.py:54-60`, interacting with the
OOM retry at `:127-140`.

Trigger: the first `make_trace_weak_image(image)` succeeds, but the subsequent
`make_trace_counterfactual_background(image)` raises a CUDA OOM. The weak image
has already been inserted into `weak_images`; the corresponding counterfactual
entry has not been inserted. The view-batch wrapper catches the OOM and halves
the batch. On retry, `if batch_idx not in weak_images` is false, so construction
of the missing counterfactual is skipped. The lookup at line 60 then raises
`KeyError`, which is not handled by the OOM backoff.

Effect: an allocation failure that could be retried at a smaller view batch
becomes a fatal unrelated exception. The new interpolation-allocation retry
test passes, but does not exercise an OOM after the first cache insertion.

Independent CPU reproduction: used the actual intervention mixin with eight
nonempty candidates, configured view batch 16, and a counterfactual factory
that raises `torch.cuda.OutOfMemoryError` only on its first call. CUDA cache
clearing was mocked to a no-op; no GPU operation was performed.

```text
after_first_counterfactual_OOM=KeyError 0
mask_candidate_batch_sizes_before_abort=[6, 3]
counterfactual_factory_calls=1
```

The two mask counts show that the smaller retry was reached, but the factory
was never called again. Successful recovery should call it a second time and
return all eight candidate records without omissions or reordering.

Correction: initialize the two caches independently, or publish their entries
only after both image variants have been constructed successfully. Add a test
that injects a first-call counterfactual OOM and checks recovery against a
non-failing reference, including supports and candidate order.

This is **pre-existing**, not introduced by the final mask-streaming edit.
The same partial-cache sequence exists in baseline `d5b21e7`, and the failure
was also reproduced with the immutable pre-mask SDPA intervention module.
It remains actionable in the reviewed final implementation because the public
execution path promises to retry view construction after OOM. No production
fix was made during this review.

No P1 findings. No additional actionable P1/P2 was identified in the reviewed
resume/completeness fixes, fused coordinator, analyzer, final-report generator,
attention changes, or mask ordering/index lookup.

## Mask-Streaming Verification

- Reviewed per-batch candidate-ID deduplication, local-mask lookup, view order,
  split-candidate boundaries, and placement of interpolation inside the OOM
  handler. No new indexing or candidate-order defect was found.
- Four focused CPU tests passed: the new mask-allocation OOM case, streamed
  views versus materialized reference, view-forward OOM ordering, and batched
  versus serial view evaluation. The new mask test covers the intended
  16 -> 8 -> 4 view-batch retry.
- An additional independent CPU comparison with the measured pre-mask module
  matched all scalar supports and reliabilities exactly for seven candidates
  across two images, a 3x4 token grid, 9x11 input images, and view batches
  1/2/4/8/16/17.

The separately executed GPU control completed during review. Its artifacts
were inspected read-only under
`/home/wwww/hzl/results/trace_simplification_20261024/final_mask_control/`.
Protocol checksums, recorded source fingerprints, image/label order, seeds,
vocabulary, and configuration matched their intended reference/current arms;
both used 56 images and view batch 16.

| Check | Observed Result |
| --- | --- |
| Candidate diagnostic rows | All 224 identical |
| Exact prediction hashes | 0 of 56 identical |
| Per-image intersection/union records | All 56 differ |
| Measured snapshot mIoU | 76.23006330638816 |
| Current worktree mIoU | 76.21803792486158 |
| Current minus snapshot | -0.01202538152658 mIoU points |

This control **does not certify exact prediction equality**. Identical
candidate diagnostics and a small aggregate metric difference do not establish
an identical adaptation trajectory or full-panel metric equivalence. The cause
of the prediction differences was not isolated; no GPU rerun was started to
attribute them to allocator effects, kernel nondeterminism, or the source edit.
This is an explicit verification limit, not a second diagnosed correctness bug.
It is unrelated to the injected-OOM failure above: both GPU control cells
completed normally.

## Previous Fixes

- Reviewed production rejection of missing image/mask pairs and duplicate
  split IDs. The suite helper independently rechecks the declared population,
  including archived loaders that silently filtered missing pairs.
- Required calibration/control/panel sizes and official evaluation populations
  are enforced before input sealing. The analyzer now rejects shortened full
  splits and regenerates actual ordered input identities. The earlier
  changed-root and incomplete-cohort findings are addressed.
- Fourteen focused CPU pytest cases passed across `tests/test_suite_inputs.py`
  and the production missing-mask/shared-default tests. Combined with the four
  mask/view tests above, this round ran 18 passing pytest cases, not the full
  test suite. The additional OOM fault injection is reported separately as
  the unresolved finding.
- Independent disposable-fixture checks verified that an archived permissive
  loader cannot hide a missing declared mask; duplicate split IDs are rejected
  by both the archived-loader audit and production loader. Runtime fingerprint
  checks included a gzip resource and a Markdown resource, detected a binary
  modification, and ignored `__pycache__` bytecode.
- The single inherited view-batch default is 16 and the package version is
  0.3.0. Inspected configuration diffs retain K, learning rate, step count,
  image/data batch, resize, patch, and class-extension settings.

## Measurement And Provenance

- Regenerated current strict input manifests without image decoding or model
  loading. Complete discovery passed for PC60 train/evaluation pools
  4998/5105, COCO-Stuff train/evaluation pools 118287/5000, COCO-CD 500, and
  VOC21 1449. Requested 40/56/512-image cohorts had exactly those sizes.
- Independently audited all 28 completed cells in `panel_v2` and `fused_panel`:
  ordered image/label IDs, base configurations, vocabulary and seeds matched
  regenerated inputs; result/protocol checksums and recorded package hashes
  matched; per-image intersection/union sums reproduced stored metrics; and
  warmup-excluded `cost.csv` rows reproduced synchronized ms/image. The fused
  manifest's selected-file size/mtime metadata also matched the current audit.
- The stored fused-snapshot evaluation matrix is 75.04/29.98/21.37/52.61 for
  COCO-CD/PC60/COCO-Stuff/VOC21. The reported original-role-aware comparators
  are 75.16/29.93/21.14/52.32. Report generation preserves the negative COCO-CD
  delta and distinguishes original-role-aware from uniform-manual timing.
  No field calculation or row-selection defect was found in the inspected
  fused coordinator, analyzer, or final-report generator.
- Verified the stored GPU numerical probe's input-protocol and fused-attention
  source hashes. Its relative gradient L2 error is 0.3872670531, not evidence of
  FP16 equivalence. The measured SDPA adoption is an explicit accuracy/cost
  tradeoff, not a lossless or bitwise-preserving optimization.
- Current runtime files differ from the four-panel measured SDPA snapshot only
  in `__init__.py`, `configs/trace.py`, `data/datasets.py`, and
  `method/trace/interventions.py`: version, default batch, strict discovery,
  and mask streaming. The measured/current attention source is unchanged.
  The four-panel numbers belong to the immutable measured snapshot; the
  post-measurement mask control does not transfer them as exact current-source
  scores.
- The original manual runner hash matches its archived copy in
  `tools_snapshot_v2`, and the archived coordinator is present. Legacy cell
  fingerprints have their original narrower resource coverage. Supplemental
  current resource audits do not retroactively become pre-run seals. The
  tokenizer gzip is identical across baseline, uniform-manual, measured SDPA,
  and current packages. Existing retrospective-audit labeling was preserved.

## Review Limits

Reviewed branch `codex/trace-simplification` against baseline `d5b21e7` and the
round-3 state, including the final user-requested mask edit. Also inspected
`docs/simplification.md` as it was being written; it separates the measured
snapshot from subsequent source changes and does not claim general SDPA
equivalence. The unresolved OOM issue above must not be conflated with the
ordinary completed controls or their accuracy results.

No GPU work, process control, production/tool edits, or writes to real
experiment artifacts were performed. Analyzer/report writers were not invoked
against the real result directories. CPU test/reproduction fixtures were
disposable, with CUDA hidden and bytecode/cache writes disabled. The final
full pytest suite and additional GPU runs were not executed by this reviewer.
`git diff --check d5b21e7 -- trace_ovss tests README.md pyproject.toml` passed.
The only persistent edit is this report.
