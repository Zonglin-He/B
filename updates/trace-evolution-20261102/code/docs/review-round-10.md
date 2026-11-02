# Round 10: baseline-first comparison and ablation mechanism audit

The user changed acceptance from preserving old TRACE accuracy to comparison
against baselines, and requested diagnosis before changing weak components.
Production `trace_ovss/` and the manuscript remain unchanged in this round.

## Diagnosis

The CPU audit reproduces 144 sealed episodes and separates dominance activation,
quality-weighted attenuation and normalized allocation change. The pre-EPS
identity `TV = E_q[abs(d-E_q[d])] / (2*(1-E_q[d]))` is tested independently from
production EPS pruning. It explains cancellation of a common attenuation, not
equivalence of model gradients or optimizer trajectories. Dataset-specific
numbers and claim limits are in `docs/ablation-mechanism-diagnosis.md`.

## Independent review

A read-only reviewer checked the formula, all 144 cached episodes, removal
deltas and new comparison coordinator. Two concrete issues were found:

1. Per-cell source checks did not exclude baseline changes between cells.
2. The coordinator trusted worker warmup/episode totals without enforcing the
   frozen episode boundaries.

Both were fixed. The coordinator now freezes the native/baseline/adapter sources
and the union of archived control files before execution, rechecks before each
cell and at completion, and compares accepted cell fingerprints to that seal.
Cost validation enforces the exact image order, batch boundaries, contiguous
episodes, update counts and two excluded warmup episodes. Regression tests cover
between-cell drift, one-episode/warmup-zero substitution, changed counts and
wrong adaptation steps. The reviewer rechecked these fixes and found no remaining
concrete issue within that scope. This is not a claim of exhaustive correctness.

## Baseline adapter

The baseline worker reads the exact sealed 128-image panels and native vocabulary,
including the repaired COCO Object query list. It reuses archived OVSS baseline
implementations at MLMP commit `ba9f57bf47846ec727c616c5163233a98ca0bf94`;
it does not vendor their source into the production package. Baselines retain
their archived method settings. The only explicitly replaced evaluation setting
is the episode batching needed to match the new panel.

No Adapt, TENT, TPT and MLMP are primary comparisons. Native and archived
multi-prompt/multi-layer no-update controls distinguish inference-path changes
from adaptation. The single-prompt No Adapt baseline is not itself a pure
backend-equivalence control. WATT and CLIPArTT are excluded as requested.

The first coordinator attempt failed before any GPU cell because it tried to
hash `code/__pycache__` as a file. Its output directory and explicit preflight
failure record remain intact. The regular-file filter was repaired before
executing the fresh `_v2` suite; this was not a score-selected retry.

## Verification

- Full worktree CPU tests: **398 passed**.
- Focused baseline adapter tests: **31 passed**.
- Real-data CPU preflight: six checks spanning four datasets and the two
  no-update inference controls, with CUDA initialization forbidden.
- No whitespace errors from `git diff --check`.
- GPU outputs belong in
  `/home/wwww/hzl/results/trace_baseline_comparison_20261025_v2/`.
  Consult its queue state and suite seal for completion, not this source note.

No baseline winner, production promotion, or universal mIoU/speedup claim is
encoded in the coordinator. Negative comparisons and failed cells are retained.
