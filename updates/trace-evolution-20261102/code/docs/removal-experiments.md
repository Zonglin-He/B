# TRACE removal experiments

This suite evaluates the simplified implementation on branch
`codex/trace-simplification`. Its reliability is shared by all classes:
`q = min(target_support, counterfactual_support)`,
`dominance = max(residual_support - target_support, 0)`, and
`rho = q * (1 - dominance)`. It does not evaluate the retired role-aware
formula, fragment guard, CLS gate, or sibling suppression.

## Scope

- Complete evaluation splits, seed 0: VOC20 (1,449), VOC21 (1,449),
  PC60 (5,105), and COCO-CD (500).
- Method-aligned removal controls: VOC21, PC60, and COCO Object on fixed
  512-image subsets; COCO-CD on its complete 500-image evaluation split.
- Five 56-image COCO-CD runs are smoke tests only, not headline evidence.
- PC59, COCO Object full, COCO-Stuff full, Cityscapes, Waterbirds, the second
  backbone, and ACDC are deferred to keep this continuation bounded.

The suite fixes the inherited per-dataset K, learning rate, steps, data batch
size, prompts, selected layers, and view batch size. There is no tuning using
the new evaluation outcomes. All variants use the same image order and batch
boundaries within each dataset panel. Subset ablations cannot be substituted
for complete-split main-table results. Single-seed results are not three-seed
means.

## Removal controls

| Display label | Change relative to Full TRACE |
| --- | --- |
| Full TRACE | Unmodified production implementation. |
| w/o Connected-Region Construction | Replace connected components with one class-union mask per input patch, retaining confidence ranking and the per-image top-K budget. |
| w/o Confidence-Source Tracing | Keep connected candidates but use their original confidence as loss weights; no intervention views are forwarded. |
| w/o Reliability Weighting | Retain connected candidates and all diagnostic forwards, but give every selected region weight one. |
| w/o Region Normalization | Retain tracing and weights but average entropy by weighted token count instead of weighted region count. |

The first control removes connectedness, not every notion of a candidate. It
can change the number and spatial extent of selected masks, which is part of
the tested construction policy, not an isolated connectivity effect. Class
unions use predicted text-query indices before canonical synonym collapse.
Candidate counts and area distributions accompany this comparison.
The third control intentionally retains the
cost of tracing to isolate application of its scores; it is not an efficient
deployment of uniform weighting. Three views are one tracing component, not
three separately ablated modules. Region normalization is an additional
within-module design check, not a newly introduced method component.
Reliability weights change relative allocation between candidates: their
common scale cancels in the normalized loss. With only one positive-weight
candidate, Full and uniform weighting have the same mathematical objective.

## Provenance

The suite freezes production files and experimental scripts into its own
`runtime` directory before GPU work. `suite_protocol.json` records image and
label paths, file sizes and modification times, vocabulary hashes, complete
population sizes, configurations, seeds, source hashes, and removal definitions.
The worker hashes the implementation before and after each cell, and saves
dataset intersection/union, per-image intersection/union and prediction hashes,
candidate diagnostics, timing scope, and peak CUDA allocation. An intervention
support that is not measured is missing, never a fabricated zero.

Resume revalidates source, vocabulary, configurations, ordered input IDs,
population size, result/protocol hashes, and metric aggregation. Failed attempts
remain in separate directories. A previously completed Full COCO-CD result may
be reused only if these identities match, and its original path remains visible
in the summary. Reuse is not described as a newly run experiment.

The first running suite snapshot predates the stricter per-cell file/environment
identity checks. Its external legacy Full COCO-CD entry is provisional.
`tools/finalize_removal_suite.py` runs after that queue without interrupting its
workers and replaces the external entry with one fresh complete Full COCO-CD
run. Other new cells are checked against the suite's original pre-run file
identity seal. The audited tables reside under `final_audit/`, and retain a flag
that extended driver/precision identity was not recorded for those earlier
profiles. Their timings remain exploratory, not a fully matched cost claim.
The new coordinator passes its exclusive queue lock into workers and adopts
the earliest completed attempt on resume, preventing duplicate active workers
or selection of a better-scoring retry. Environment changes and unsealed
external cached input identities are rejected.

COCO Object uses the corrected vocabulary in which `bed` belongs only to its
foreground class, not simultaneously to background. Its archived pre-fix scores
cannot serve as matched baselines for this suite. The local converted data
directory is an explicit CLI argument, not an upstream-project dependency in
the production package.

## Outputs and execution

- `queue_state.json`: active cell or terminal status.
- `summary.csv`: every completed cell, including its evidence scope and origin.
- `ablation_summary.csv`: removal results and deltas from the Full run on the
  identical image panel, retaining positive, neutral, and negative results.
- `<stage>/<dataset>/<variant>/selected_result.json`: the immutable selected
  result path and its SHA-256, not a metric-selected trial.
- `attempt_*/`: protocol, result, progress, cost, per-image counts, diagnostic
  CSV, and stdout for each actual execution.

Run `tools/run_removal_suite.py --help` for required local paths. `--prepare-only`
freezes inputs and code without using the GPU. Execution is sequential on one
GPU. Profiling is implementation-, batching-, and device-specific; the results
are not hardware-independent slowdown factors. All removals live in
`tools/trace_ablation_variants.py`, outside the production API.
