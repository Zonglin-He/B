# Predeclared two-dataset diagnostic-cache followup

Scope is fixed to complete seed-0 PC60 (5105 images) and COCO Object (5000).
These datasets were chosen before inspecting their parent full-control results.
No dataset selection, tuning, extra T/C forwards, automatic retry or resume.
The old `tools/evolution` files and the parent runtime/results are read-only.

## Controls

- `product`: coefficient `T*C`, on the parent `q>EPS` active candidates. There
  is no second EPS cutoff on the smaller product; normalization includes all
  admitted positive products. This avoids confounding the aggregation rule
  with a new candidate-removal threshold.
- `pixel_normalized`: coefficient `q*token_area` applied to mean region entropy,
  exactly implementing `sum(q*sum(token_entropy))/sum(q*area)` for the parent
  `q>EPS` active set. Area cannot reactivate a parent-inactive candidate.

The bounded diagnostic `q=min(T,C)` is unchanged and stored separately from
`effective_weights`, which are not claimed to be bounded reliabilities.
The production objective's internal term key `rho` is an implementation detail;
new control archives do not label area-weighted coefficients as rho.

## Execution and safety boundaries

Both the parent cell's prediction seal and the ENTIRE parent suite's
`SUITE_SEALED.json` must exist and verify before GPU execution. The runner fails
immediately if they are absent or invalid; it does not wait or retry. It takes
the existing `reviewer_followup_gpu.lock` nonblocking for prediction. Phase
scheduling belongs to the parent controller/user, not this runner.

`--package-root` must have the exact parent package AND unchanged old
`tools/evolution` tree, plus the parent's transitive helper hashes. Using the
1338 frozen runtime as package-root is simplest; the new script stays outside
that runtime and never writes into it. New replay sources have their own hashes.
No full data directory scan beyond the native strict split/path enumeration is
used, and annotation contents are never opened during prediction.

The runner checks parent suite/manifest/prediction seals; official counts;
image/annotation order and metadata; image content hashes; vocabulary hash;
configuration; checkpoint file SHA; complete initial model state and optimizer
state; every cache fingerprint, mask, ownership and top-K budget. It verifies
the parent NPZ/JSON hashes again at each read.

Before any control prediction, the FIRST COMPLETE parent episode runs cached
TRACE with original `q=min(T,C)` and no diagnostic forwards. Initial trainable
and Adam hashes must match the parent's episode receipt, and every adapted
label must exactly match `parent/predictions/trace/INDEX.npy`. A failed gate
is preserved in `replay_gate.json` and stops the run. There is no tolerance
override, GT use or metric-based choice. Model/optimizer and RNG are restored
after the gate. No whole-dataset cached-TRACE equivalence claim is implied.

`predict` produces all pre/control labels, raw caches/links, source bindings
and a prediction seal before returning. `score` verifies all bindings and
that seal before streaming GT. `run` executes predict then score. Existing
outputs are rejected. The result format and `SCORED_SEALED.json` follow the
existing runner.

## Commands (not executed during development)

Run these only AFTER the 1338 parent suite is completely sealed and its GPU
lock is free. Each output path must be new. Execution is sequential; a failure
must be inspected, not retried automatically.

```bash
python -B tools/evolution_replay/run_cell.py \
  --parent-cell /home/wwww/hzl/results/trace_self_evolution_20261028_1338/cells/mechanism/pc60 \
  --package-root /home/wwww/hzl/results/trace_self_evolution_20261028_1338/runtime \
  --output /home/wwww/hzl/results/trace_replay_product_pixel_NEW/pc60 \
  --phase run

python -B tools/evolution_replay/run_cell.py \
  --parent-cell /home/wwww/hzl/results/trace_self_evolution_20261028_1338/cells/mechanism/coco_object \
  --package-root /home/wwww/hzl/results/trace_self_evolution_20261028_1338/runtime \
  --output /home/wwww/hzl/results/trace_replay_product_pixel_NEW/coco_object \
  --phase run
```

There are no data/seed/config overrides: all are bound to the parent.
Separate `--phase predict` and `--phase score` use the same three path arguments.

## Artifacts and interpretation

- `prediction_protocol.json`: parent chain, source hashes, inputs, inherited
  config, two predeclared controls and their active-set definitions.
- `replay_gate.json`: first-complete-episode exact-label outcome and initial
  weight/optimizer fingerprints, sealed with all predictions.
- `episodes/EPISODE.npz`: token masks, ownership, T/C/q, token area, parent
  active mask and policy-by-candidate `effective_weights`.
- `episodes/EPISODE.parent.npz`: byte-for-byte original parent cache. Its old
  `rho` field denotes parent control data, not the new effective weights.
- `episodes/EPISODE.cachelink.json`: parent NPZ/JSON hashes and fingerprint.
- Existing per-image label payloads, small progress/results, cost records,
  prediction/score seals and per-episode streaming intersection/union.

Cache-read time is not diagnostic-forward time, and measured replay runtime
is not the standalone cost of these methods. These are joint allocation
controls, not singleton-normalization causal evidence. Repeated seeds measure
replay stability, not independent-sample uncertainty.

## CPU verification

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -B -m pytest \
  tools/evolution_replay/test_run_cell.py -q -p no:cacheprovider
```

Existing numpy, torch, Pillow and pytest suffice. Tests use real tiny CPU
encoders, sealed synthetic parent artifacts and poisoned GT during prediction;
they do not load public model weights or run CUDA. Real GPU exact replay and
the full two-dataset followup still require separate validation.
