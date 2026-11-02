# Review 12: VOC21 algorithmic forward reduction

Scope: research-only cached-prefix variants, their isolated queue, live
parameter inventory and measured execution costs. Production code, dataset
defaults, manuscript and already frozen experiment runtimes are unchanged.

## Independent reviews

- Cache implementation: no confirmed blocker in original-image identity,
  mutation/reset invalidation, raw selected-output caching, diagnostic bypass
  or trainable suffix isolation. CPU tests compare initial logits, gradients,
  optimizer trajectories and final predictions against independently frozen
  uncached models, including K=33 and all 36 steps.
- Queue implementation: no confirmed blocker in 13-cell order, real reference
  artifact compatibility, source/environment/input sealing, train/validation
  separation or dependency-lock ordering. Four actual full baseline records
  and the full two-view record pass verification. Both lazy full outcomes and
  both tail-depth full outcomes are retained; no score-based promotion.
- Counter review found three actionable issues. These were corrected before
  launch, not dismissed as irrelevant because ordinary smoke runs avoid OOM.

## Corrections

1. **Recovered OOM work.** Forward hooks also count partial failed attempts.
   `ExecutionAudit` records their stem/block subset separately without erasing
   it. Successful-work identities use total minus failed; timing and reported
   total work include retries. Pixel-view construction failures that enter no
   encoder have no encoder input count but remain in elapsed time. Hooks
   measure module entries, not FLOPs or guaranteed completed block operations.
2. **Call-count integrity.** Each successful prefix block must have the same
   call count as the successful stem. Every suffix block must execute all
   successful diagnostic batches, proposal, configured optimizer steps and
   final inference. Zero-call vectors, missing tail calls and inconsistent
   prefix calls are rejected, as are inconsistent input-vector sums.
3. **Actual optimization scope.** The benchmark reads live model
   `requires_grad` tensors and optimizer parameter identities, checks exact
   correspondence to the declaration, and records per-parameter sizes. Both
   before and after a cell, frozen-prefix parameters must be excluded. Offline
   validation checks the names and nonzero size inventory, not just a declared
   list copied from the method.

Regressions include real tiny-model OOM injection, zero-call records, an
omitted tail call, all parameters frozen, an extra trainable prefix parameter,
an optimizer missing output LayerNorm and mismatched parameter totals.

## Remaining empirical boundary

Freezing changes the adaptation algorithm. Cached/uncached equivalence proves
neither GPU latency nor unchanged accuracy relative to all-LayerNorm TRACE.
Lazy-view coefficient bounds do not bound gradient trajectories or rare-class
accuracy. Both changes require the newly queued full VOC21 results. Current
128-image lazy improvements remain development evidence, not full results or
three-seed evidence. No production/paper promotion is authorized by these tests.
