# Review round 9: algorithmic forward reduction

The independent review checked the target-first normalized-weight certificate,
missing measurements, source/optimizer invariance outside the experimental
algorithm, real forward counters, and the frozen fixed-config queue. Two P2
findings were repaired before preparing the GPU snapshot:

- Query loops retained previous-batch logits and their final slice across the
  next model call. Both references are now released; three weak-reference
  regressions verify the tensor lifetime at the next forward.
- OOM halving was reset at every query round. The effective cap now stays
  monotonic within an episode while the configured batch size is restored.
  A synthetic capacity-four run verifies attempts 12, 8, 4 followed only by
  batches at most four, with reported effective cap four and config still 16.

Both findings were independently confirmed closed. The parent also corrected
the queue audit to accept zero steps only for a diagnostically empty objective;
nonempty objectives must run the unchanged configured number of steps.

Verification: 353 all-CPU tests passed. The two pytest XML-format warnings are
about record_property with xunit2, not algorithm execution. This includes 76
queue tests, 26 algorithm tests, and exact selector replay against 288 cached
episode-policy cases. Production trace_ovss sources were not edited.

This bounded review does not prove GPU speedup, accuracy preservation, or
precision-invariant adaptation. Cached view-count savings are not timings.
GPU experiments retain negative deltas and do not automatically adopt a method.
