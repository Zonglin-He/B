# Independent Review Round 3

## Findings

### [P2] Reject incomplete fixed panels before freezing or reporting them

Primary locations: `tools/suite_inputs.py:50-55` and
`tools/analyze_simplification.py:88-94`.
Related acceptance paths: `tools/run_simplification_suite.py:87-88` and
`tools/benchmark_simplification.py:77-82,103`.

Trigger: start a fresh suite on an incomplete dataset, such as a COCO-CD split
with a listed image whose mask is missing, or an evaluation pool smaller than
the promised 512-image panel. Dataset discovery silently drops missing pairs
(`trace_ovss/data/datasets.py:246-256`). The input sealer then treats that
filtered population as authoritative, and `min(limit, len(data))` silently
reduces fixed-size panels. The coordinator accepts the reduced count for the
same reason. A stable input seal proves consistency of the surviving inputs,
not completeness of the requested cohort.

Effect: the benchmark can label an incomplete declared split `full_split=True`
because its denominator is the already-filtered dataset length. The analyzer
accepts shortened cohorts and unconditionally emits the scope "One full
COCO-CD split and fixed 512-image subsets elsewhere, seed 0". This can report
an incomplete/subsampled evaluation as the promised fixed protocol, affecting
the interpretation of accuracy and runtime comparisons.

Independent CPU reproduction, using temporary files only:

1. Created a COCO-CD split listing `a`, `b`, and `c`, with three image files but
   masks only for `a` and `b`. Called the actual `expected_inputs()` helper.
2. It sealed two available images without error. The benchmark's actual
   `len(indices) == total_images` full-split expression consequently returns
   `True`. Asking the helper for 512 images also sealed only two.
3. Repeated the incomplete-pool setup for the four evaluation datasets and
   constructed mutually matched, protocol-checksummed synthetic result files,
   with two records per cell and equal engineering-control diagnostics.
4. Called the actual `analyze()` on that disposable panel. It succeeded and
   wrote decision evidence claiming the full-split/512-image scope, while all
   four comparison rows contained `images=2`.

Observed output:

```text
declared_COCO_CD_split_entries=3
discovered_pairs=2
benchmark_full_split_expression=True
analyzer_accepted_images_by_dataset={'coco_cd': 2, 'pc60': 2, 'coco_stuff': 2, 'voc21': 2}
decision_scope=One full COCO-CD split and fixed 512-image subsets elsewhere, seed 0
```

Required correction: at the fixed-suite boundary, reject a pool smaller than
the requested 40/56/512-image cohort rather than silently shrinking it. For a
declared full split, validate all required split-file image/mask pairs before
using the discovered length as the full-split denominator. Make the analyzer
validate the planned cohort sizes and full-split coverage before emitting
decision evidence; derive its scope from validated inputs. The generic
benchmark's `--max-images` may remain a cap, but the fixed suite must enforce
its stronger promise. Explicitly smaller experiments need a different plan
and accurate scope, not the existing fixed-panel label.

Regression acceptance: missing a listed COCO-CD mask fails before benchmark
launch; a 511-image pool cannot satisfy a 512-image fixed panel; the analyzer
rejects or accurately relabels shortened cohorts. Complete datasets and
unchanged-input resumes continue to pass.

This is distinct from the round-2 changed-root resume finding. No evidence was
found that this fresh-run incompleteness trigger affected the existing manual
panel: its observed evaluation counts were 500 for full COCO-CD, 56 for its
engineering control, and 512 for each other panel.

No P1 findings. No additional actionable P1/P2 issue was identified in the
reviewed attention implementation, selected-layer freezing, or changed-input
resume protection.

## Resume Fix

- The new coordinator compares the resolved data root before input preparation
  or benchmark launch. The schema-2 manifest includes ordered panel IDs, seed,
  vocabulary, base configuration, available count, size/mtime metadata, and
  source/tool identities. Cached-cell checks compare IDs and configuration;
  input regeneration at completion detects persistent metadata/input changes.
- A targeted temporary-data check verified correct PC60 train/evaluation
  discovery, restoration of the global dataset specification, acceptance of
  unchanged inputs and a view-batch override, and rejection of a same-root
  image-file size/mtime change without overwriting the original manifest.
- Inspected `tests/test_suite_inputs.py`, including the real `main()`
  changed-root early-rejection test. The original changed-root resume hole is
  addressed in the reviewed new coordinator; this does not retroactively give
  the already-running legacy coordinator schema-2 guarantees.

## Attention Checks

- Reviewed the token/batch/head reshape and inverse mapping, explicit scale,
  zero dropout and noncausal setting, projection use, and gradient flow. The
  remaining NaCLIP and CSA calculations retain their manual formulas.
- Independently compared the actual vanilla attention method with a separate
  head-axis `einsum` reference using a random upstream gradient, not just the
  existing test's squared-output loss. Checked input and all input/output
  projection weight/bias gradients with non-contiguous inputs, token counts
  1/7/17, batch sizes 1/2/3, head counts 2/3/4/12, and widths 32/48/64/768.

| CPU Dtype | Cases | Max Forward Absolute Error | Max Gradient Absolute Error |
| --- | --- | --- | --- |
| FP64 | 4 | 2.78e-16 | 3.55e-15 |
| FP32 | 4 | 1.49e-7 | 1.91e-6 |

- Inspected `tests/test_fused_attention.py` and the selected-layer tests in
  `tests/test_simplified_trace.py`. The engine freezes blocks after the actual
  last selected layer and the reduced last-selected LN2, matching the encoder's
  execution path. No actionable layout, gradient, or parameter-freezing defect
  was found in these checks.
- These are CPU mathematical/gradient checks only. They establish neither GPU
  FP16 kernel equivalence nor measured segmentation accuracy, runtime, memory,
  or adaptation-trajectory equivalence for SDPA. Those measurements are pending
  the separate immutable SDPA benchmark.

## Evidence And Limits

- Reviewed the updated files listed above on `codex/trace-simplification`
  against baseline `d5b21e7` and the round-2 state. No prior independent
  round-1 report was read, and no production or tool file was edited.
- Read existing manual-panel artifacts without running the analyzer against
  them. The archived benchmark runner matched the original `panel_v2` runner
  hash; the archived coordinator was present. All 22 completed cell protocols
  matched their original reference/candidate source manifests and protocol
  checksums. Same-dataset/split/count groups had identical ordered image and
  label IDs. This is a provenance/record consistency check, not a new
  annotation-content audit or an SDPA result.
- Confirmed that the current SDPA `dense_clip.py` differs from the immutable
  candidate that produced the manual panel. No manual-panel accuracy was
  attributed to SDPA. The separately evolving SDPA benchmark coordinator was
  outside this round's reviewed runner scope.
- Did not rerun the parent's full or expanded pytest suites. All independent
  executions were small CPU reproductions with `CUDA_VISIBLE_DEVICES=` and
  bytecode writes disabled; generated fixtures and analyzer outputs were in
  disposable temporary directories. No GPU use, process control, or writes to
  the real experiment artifacts occurred. `git diff --check d5b21e7 --
  trace_ovss tests README.md` passed. The only persistent edit is this report.
