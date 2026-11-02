# Independent Review Round 2

## Findings

### [P2] Freeze dataset identity before accepting cached cells

Location: `tools/run_simplification_suite.py:26-39` (suite manifest) and
`tools/run_simplification_suite.py:65-76` (cached-cell validation).

Trigger: resume a partially completed suite with the same output directory and
package snapshots but a different `--data-root`. The manifest freezes the
package roots, source hashes, runner hash, and schedule, but omits the data
root. Cached cells are checked against their own saved image counts and source
hashes, not against the currently requested dataset root or expected image and
label IDs. `saved["config"]["data_dir"]` is never compared with the current
request. The same gap also matters if the discoverable sample list changes
under an unchanged data root.

Effect: completed cells from data source A are reused while pending cells run
against data source B, and the queue can still report `completed`. Equal image
counts do not make those cells comparable: the selected samples, labels, or
episode grouping can differ. This invalidates the intended fixed-input
accuracy/cost comparison without causing a resume error.

Independent CPU reproduction: called the actual suite `main()` with
`subprocess.run` mocked to produce valid, self-consistent cell protocols and
results in a temporary directory. No benchmark subprocess or GPU work ran.

1. Ran the fixed schedule against temporary `data_A`, producing 18 mock cells.
2. Removed only the temporary final VOC21 `unified8/result.json` to represent an
   unfinished cell.
3. Invoked the same suite/output/package arguments with `--data-root data_B`.
4. Resume reused all completed A cells, launched only the unfinished cell on B,
   and wrote `queue_state.json` with `status="completed"`.

Observed output:

```text
initial_cells=18
resume_launched_only=[('unified8', '<temporary>/data_B')]
accepted_data_roots=['<temporary>/data_A', '<temporary>/data_B']
final_queue_status=completed
```

Required correction: include the resolved data root in the frozen suite
protocol and validate each cached cell's resolved data directory against the
current request. Seal or regenerate the expected ordered image/label IDs for
each split and panel, including sampling seeds and episode configuration;
reject cached cells that do not match. Require identical ordered IDs across
comparison arms for the same panel. A protocol checksum proves that a saved
protocol is unchanged, not that it belongs to the current request.

Regression acceptance: an unchanged-input resume reuses completed cells;
changing the data root or selected sample manifest fails before launching a
pending cell or producing a combined summary. No production fix was made in
this review, and the running suite was not altered.

No P1 findings. No additional actionable P1/P2 issue was identified in the
reviewed method implementation. The resume failure was reproduced with mock
fixtures, not observed as contamination of the running `panel_v2` evidence.

## Verification Scope

- Reviewed branch `codex/trace-simplification`, including uncommitted and new
  implementation/test/tool files, against baseline
  `d5b21e7bba83aae3f1fac60c64daf15a43c83fdf`. The previous reviewer report was
  not read.
- Inspected public API/call sites, uniform three-view ordering, finite/range
  validation and logit sanitization, detached scoring, confidence reuse,
  connected-component/top-K selection, patch/image indices and masks,
  region-normalized per-prompt entropy, visual LayerNorm parameter collection,
  reduced final LN2 exclusion, singleton class/prompt dimensions, Adam state
  reset, and opt-in layer-weight recording. Inspected the associated tests;
  did not independently rerun the parent's test suites.
- Inspected synchronized timing, training-split calibration, runtime-only batch
  selection, fixed evaluation panels, same-batch engineering controls,
  extension collapsing, vocabulary collisions, source/protocol hashes,
  completion markers, and resume behavior in both new runner files.
- Read existing evidence only under
  `/home/wwww/hzl/results/trace_simplification_20261024/panel_v2`. Both package
  source manifests and the benchmark runner hash matched the suite manifest.
  All 43 tracked baseline package files matched `d5b21e7`; candidate source
  hashes matched the reviewed worktree at inspection.
- Independently compared the completed COCO-CD `reference_uniform8` and
  `unified8_equivalence` cells: all 56 per-image records, including prediction
  hashes and intersection/union counts, were identical. All 224 candidate
  diagnostic rows were identical. Ordered image/label IDs, vocabulary hash,
  configurations, and seeds matched. This validates the recorded bounded
  batch-8 control, not every dataset or view-batch size.
- Verified that the 40 calibration images for each of PC60 and COCO-Stuff came
  from their respective training splits, and that all four batch-size options
  used the same ordered images. Selection code uses synchronized runtime only.
- At the final evidence read, 15 completed cells passed protocol-checksum and
  image-count checks. Their data directories were under the requested data
  root, and same-split/dataset/count groups had identical ordered image and
  label IDs. The full suite was still in progress; no final-suite accuracy or
  cost acceptance conclusion is asserted here. Historical COCO Object results
  were not treated as evidence for the corrected vocabulary.
- Ran only small CPU checks with `CUDA_VISIBLE_DEVICES=` and bytecode writes
  disabled. The resume reproduction used disposable temporary fixtures. No
  GPU work, process termination, production-file edit, or experiment-artifact
  write was performed. `git diff --check d5b21e7 -- trace_ovss tests README.md`
  passed. The only persistent review edit is this report.
