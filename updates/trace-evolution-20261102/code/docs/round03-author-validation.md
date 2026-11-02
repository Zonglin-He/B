# Round03 author validation

This is a build/evidence checklist, not an independent review or score.
The reviewed packet is immutable:
`/home/wwww/hzl/results/trace_evolution_review_round03_20261029_0145`.

## Build identity

- Packet manifest SHA256:
  `77b3884bd3aabb63364e680f63bd37ca20c1986697a8d4427ab1043ce0ed608a`.
- Combined main PDF SHA256:
  `49514322814af97c344cd81785d38f77d73127e78e53d04fcfa8df6e4e18bec7`.
- Standalone supplement PDF SHA256:
  `b74cef7baf0f22b3bbb66910c57be34b18961c17812c0d5b886a5b12446f42b1`.
- Build succeeded without overfull boxes or unresolved citations/references.
- All20 combined-main and9 supplement pages were rendered and inspected.
  New timing/count tables share one float; no clipping or overlapping text was
  observed. This is a working draft, not page-limit certification.
- Both PDF font inventories contain embedded Type1 fonts and no Type3/CID.
- `git diff --check` passed.

## CPU regression

Actual environment Python, CUDA disabled, bytecode/cache writes disabled,
one-thread BLAS/OpenMP:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wwww/miniconda3/envs/mlmp/bin/python -B -m pytest -q \
  -p no:cacheprovider paper_evolution/tools/test_export_evolution.py \
  paper_evolution/tools/test_export_standalone.py \
  tests/test_evolution_completion.py tests/test_standalone_probes.py \
  tests/test_standalone_queue.py tests/test_probe_complementarity.py
```

Result:202 passed in9.97s. Exporter-only unittest also passed13 cases.
CPU tests are not independent scientific runs or GPU benchmarks.

## Claim alignment

1. Contribution: context-probed relative allocation and region averaging,
   not an optimality/causal-source-identification theorem.
2. Correctness: crop/query/token/pixel spaces, separate readouts and fixed
   episode diagnostics are explicit. Positive scalar cancellation for a
   single active candidate is disclosed.
3. Evidence: six full split scores are distinct from128-image common-config
   and regrouping diagnostics. All single-view winners and negative min
   differences are retained. Current reset-model readout is shown separately.
4. Comparison: literature MLMP numbers have source attribution; local controls
   isolate within-backend allocation. Neither comparison certifies equal
   historical tuning. Parameter exposure is disclosed in the appendix.
5. Cost: zero/one/two-view schedules were actually executed independently;
   model-side timings exclude shared setup/data preparation/scoring, exclude
   only episode0 from timing and keep all images in accuracy/work counts.
6. Interpretation: transition denominators are explicit; joint-update
   associations are not isolated candidate causality. Replays are not
   independent uncertainty. No bootstrap/significance result was invented.

The fresh independent reviewer is responsible for its own assessment.
Author checks do not settle contribution strength or a conference decision.
