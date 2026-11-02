# Independent Review Round 5

## Conclusion

**No unresolved actionable P1/P2 findings within the established review scope.**
The round-4 partial-cache OOM finding is closed. The additional logit-conversion
OOM protection is correct in the reviewed code. No new feature or optional
enhancement is being treated as a delivery blocker.

## Closure

- **Round 4, partial image cache:** closed at
  `trace_ovss/method/trace/interventions.py:54-61`. Weak and counterfactual caches
  are checked independently. A successful weak-image allocation no longer
  suppresses reconstruction of a missing counterfactual after OOM.
- **Logit allocation/retry boundary:** verified at
  `trace_ovss/method/trace/interventions.py:122-154`. Float conversion and
  `nan_to_num` are inside the protected operation; both temporary variables
  have sentinels and are deleted on failure. Failed attempts do not yield
  results or advance the key cursor. Batch-size-one OOM propagates unchanged.
- The no-OOM path retains the same view construction, conversion order,
  sanitization arguments, detached scoring, and numerical operations. This
  inspection does not establish exact GPU adaptation-trajectory equality.
- The earlier changed-root resume and incomplete-panel findings remain closed
  as audited in round 4. This focused round did not repeat that full data,
  control, and measurement-provenance audit.

## Verification

Six focused CPU pytest cases passed:

- `test_counterfactual_oom_does_not_poison_the_image_cache`
- `test_logit_conversion_oom_retries_without_skipping_views`
- The existing mask-allocation OOM, view-forward OOM ordering, streamed versus
  materialized view, and batched versus serial view tests.

Additional independent CPU fault injections passed:

- Failed counterfactual construction for the second image after the first
  image was cached. Both images recovered; factory calls were `{0: 1, 1: 2}`.
  All eight candidate markers and every scalar support, quality, dominance,
  and reliability matched the non-failing reference exactly.
- Failed the FP16-to-FP32 cast itself above two views. Conversion attempts were
  `[5, 2, 2, 2, 1]`; all seven keys and output values were preserved in order.
- Injected terminal batch-size-one OOM separately during view construction,
  model forward, and sanitization. Each propagated the original OOM rather
  than raising a secondary exception or looping indefinitely.

`git diff --check d5b21e7 -- trace_ovss tests README.md pyproject.toml` passed.

## Limits

No GPU work, production/tool edits, process control, or writes to experiment
artifacts occurred. CPU fixtures were disposable, CUDA was hidden, and
bytecode/pytest-cache writes were disabled. Only this report was added.

The parent's full 74-test run, final post-fix full COCO-CD rerun, and rebuilt
wheel were not independently executed or certified in this review. They remain
delivery verification steps, not additional diagnosed P1/P2 code findings.
Pre-fix measurements and the round-4 GPU control retain their recorded scope;
this closure adds no exact GPU trajectory or general metric-equivalence claim.
