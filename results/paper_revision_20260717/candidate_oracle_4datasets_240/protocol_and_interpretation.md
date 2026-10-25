# Candidate-oracle bottleneck decomposition

## Protocol

- Evidence type: `newly_run`.
- Split: deterministic prefix of each clean validation loader (`shuffle=False`).
- Maximum original images per dataset: `240` (converted to whole loader batches).
- Four cells: predicted/oracle candidates x uniform/TRACE reliability.
- Oracle candidates use GT connected components and real-class labels, but choose the best matching class extension from the frozen pre-adaptation prediction.
- Oracle candidates keep the configured image/per-class region budgets; the prediction confidence threshold is intentionally bypassed.
- CLS entropy is disabled in all four cells to isolate the local spatial path.
- All other optimizer and method settings are held fixed within each dataset.

## Interpretation

- `candidate_oracle_gain_*` estimates headroom attributable to candidate construction under a fixed reliability rule.
- `trace_gain_*` estimates the gain from TRACE reliability under a fixed candidate source.
- Oracle rows provide best-case candidate geometry under the retained region budget, but they are not performance upper bounds: entropy adaptation can interact non-monotonically with the changed regions. They must not be compared as deployable methods.

## Completed datasets

- voc20: candidate gain (TRACE) -0.67; reliability gain (predicted candidates) +0.87.
- pc60: candidate gain (TRACE) -0.41; reliability gain (predicted candidates) +0.41.
- coco_object: candidate gain (TRACE) +0.73; reliability gain (predicted candidates) -1.19.
- coco_stuff: candidate gain (TRACE) -0.06; reliability gain (predicted candidates) +0.12.
