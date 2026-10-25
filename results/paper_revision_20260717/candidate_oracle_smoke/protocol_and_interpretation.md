# Candidate-oracle bottleneck decomposition

## Protocol

- Evidence type: `newly_run`.
- Split: deterministic prefix of each clean validation loader (`shuffle=False`).
- Maximum original images per dataset: `1` (converted to whole loader batches).
- Four cells: predicted/oracle candidates x uniform/TRACE reliability.
- Oracle candidates use GT connected components and real-class labels, but choose the best matching class extension from the frozen pre-adaptation prediction.
- Oracle candidates keep the configured image/per-class region budgets; the prediction confidence threshold is intentionally bypassed.
- CLS entropy is disabled in all four cells to isolate the local spatial path.
- All other optimizer and method settings are held fixed within each dataset.

## Interpretation

- `candidate_oracle_gain_*` estimates headroom attributable to candidate construction under a fixed reliability rule.
- `trace_gain_*` estimates the gain from TRACE reliability under a fixed candidate source.
- Oracle rows are diagnostic upper bounds and must not be compared as deployable methods.

## Completed datasets

- voc20: candidate gain (TRACE) +19.53; reliability gain (predicted candidates) -23.13.
