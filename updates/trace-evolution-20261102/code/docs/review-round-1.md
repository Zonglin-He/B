# TRACE Simplification: Independent Review Round 1

## Result and Scope

One actionable P2 issue was found. No P1 issue was identified in this static
review. This is not a numerical-equivalence, benchmark, or safety validation.

- Worktree: `/home/wwww/.codex/worktrees/trace-simplification/hzl`.
- Branch: `codex/trace-simplification`.
- Baseline and current HEAD: `d5b21e7bba83aae3f1fac60c64daf15a43c83fdf`.
- Reviewed the current uncommitted diff against that baseline, including public
  exports, CLI construction, dense-model inference, adaptation, candidate
  construction, three interventions, indexed loss, reset, and both test files.
- No tests, model forwards, GPU commands, or experiments were run. Production
  files and tests were not modified. The parent owns test execution and the
  experiment runner.

## Actionable Finding

### [P2] Exclude the frozen final reduced LayerNorm from optimizer collection

**Location:** `trace_ovss/method/trace/engine.py:82-87`, with the cause in
`trace_ovss/method/adapter.py:69-74` and optimizer construction at
`trace_ovss/method/trace/engine.py:90-95`.

**Trigger:** Construct `TRACE` with the default reduced NaCLIP model, or an
injected `DenseCLIP` whose `architecture` is `"reduced"`.

**Defect:** The new code sets the last block's `ln_2.requires_grad_(False)`, then
calls `collect_ln_params()`. That collector appends every LayerNorm `weight` and
`bias` without checking `param.requires_grad`. Both final `ln_2` parameters
therefore still enter Adam's parameter group and `adapted_parameter_names`.
Freezing a parameter does not remove it from the optimizer.

**Effect:** The specified optimizer exclusion is not implemented. Optimizer
membership and the reported adaptation-parameter list include two frozen,
unused parameters that the trainable-only reset snapshot excludes. This is an
optimizer-selection/audit contract defect, not evidence of changed predictions:
with their usual `grad=None`, Adam skips these parameters and does not create
moment tensors for them.

**Required correction:** Collect only LayerNorm affine parameters with
`requires_grad=True`, or filter the parameter/name pairs before constructing
Adam. Keep the names aligned with actual optimizer membership.

**Missing regression test:** Instantiate a tiny real `TRACE` engine for reduced
and vanilla `DenseCLIP` models. Assert that optimizer parameter identities equal
the intended trainable visual LayerNorm identities, that the reduced final
`ln_2.{weight,bias}` are absent from the optimizer and adapted names, and that the
vanilla final `ln_2` remains included. The existing
`test_only_unused_final_reduced_mlp_is_skipped`
(`tests/test_runtime_optimizations.py:247-260`) checks branch execution and
gradients on `DenseCLIP` directly; it never constructs `TRACE` or its optimizer.

## Intended Changes, Not Additional Defects

- Removing semantic-role metadata, the `class_roles` constructor argument, role
  counters, `TRACE_ROLES`, and `trace_view_types` is consistent with the stated
  API change. The current `trace_ovss/method/__init__.py` exports `TRACE_VIEWS`;
  the previously reported leftover export is fixed and is not unresolved.
- Every candidate schedules target, residual, and counterfactual views. View
  construction and key-based support lookup follow those meanings, including
  when a batch starts in the middle of a candidate's three views.
- `reliability.py:37-39` implements `q=min(T,C)`, `d=max(D-T,0)`, and
  `rho=q*(1-d)` without semantic roles. Invalid probability rejection is explicit
  in the updated tests, rather than an accidental role-compatibility break.
- Connected-component construction and top-K ordering are unchanged apart from
  removing role fields. Estimator-produced terms now contain indices instead of
  duplicate GPU masks; the objective consumes those indices and retains its
  region-normalized, prompt-averaged entropy calculation.
- The simplified adaptation forward is the old, always-selected `"loss"`
  branch. Keeping the text tensor unsqueezed preserves the class axis for a
  singleton vocabulary. Evaluation still uses the averaged text embedding and
  adaptive layer aggregation; disabling weight recording changes diagnostics,
  not the aggregation expression.
- Episodic reset still restores the saved trainable tensors, reloads initial
  Adam state, and clears optimizer gradients. Static inspection found no new
  reset-path defect, subject to the optimizer membership issue above.

## Focused Verification Gaps

These are missing checks, not additional established defects. Test execution
and any additions remain with the parent.

1. Neither current test file constructs the full `TRACE` engine. Add a tiny CPU
   engine check covering initialization, adaptation, evaluation, and an A/B/A
   episode sequence, plus an all-invalid-mask episode that must reset without
   updating. The existing reset test uses a generic sequential network and
   does not exercise candidate estimation or engine orchestration.
2. Exercise actual estimator-produced, index-only region terms against an
   independently constructed boolean-mask reference, with unequal region areas,
   unequal reliabilities, multiple patches, and multiple prompts. Current
   objective tests use hand-built terms, generally retaining masks even when
   indices are supplied.
3. Cover a singleton class and singleton prompt through the engine after the
   `.squeeze()` removal. Also verify cached `original_support` against the base
   probability mean for collector-produced candidates and verify the fallback
   for candidates without `confidence`. Existing view comparisons do not assert
   `original_support`; synthetic candidates in `test_batched_views_match_serial_view_evaluation`
   use arbitrary confidence values.

## Evidence and Design Boundary

The uniform residual penalty intentionally changes weights for candidates that
previously used the stuff/background rule. This is an algorithm change, not
just runtime cleanup; unchanged configurations or passing unit tests would not
establish accuracy preservation or equivalence to `d5b21e7`.

`README.md:21-31` still describes role-specific scoring. Update that method
description before publishing the new implementation, and do not apply the
runtime-only preservation language at `README.md:109-113` to this algorithm
change without separate evidence. This is a documentation/evidence boundary,
not an additional runtime correctness finding.

## Reviewed Source Identity

Git blob hashes recorded for the finding's source files, to disambiguate
concurrent parent edits:

- `trace_ovss/method/adapter.py`: `f18332a67624876044645b894882af617db0155c`.
- `trace_ovss/method/trace/engine.py`: `e2fc5bd8ca7077ae742bbe0ca3599e9c38f54b66`.
