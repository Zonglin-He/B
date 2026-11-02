# Image-Budget Completion Audit

Date: 2026-10-29. Authorized CPU-only research sidecar, not a manuscript review.
Worktree: `/home/wwww/.codex/worktrees/trace-simplification/hzl`.
Suite: `/home/wwww/hzl/results/trace_image_budget_20261029_0233`.

## Outcome

The bounded importer passed and exported **8 canonical cells / 48 CSV arm rows**:
two calibration128 cells and six confirmation256 cells, with every one of the
six arms and all three confirmation orders retained. Global **C** was already
selected and sealed using calibration only. This import audits its recorded
statistics and links; it does not invoke selection or change the chosen mode.

Prepared SHA-256:
`579372bf5a4208ad9e1f465698b0d82193ecd8d7c70162b3b1b3c67c84f056c5`.
Whole-suite seal SHA-256:
`cef8346382ade4c4fadf2fc48e1ca097ff52c6522766d5341e65f021aa40928a`.
Selection SHA-256:
`90fa3a20097698e9b83cafd416a4b05fe7f033dcd8aadb027c970a9788ac11e4`.

Real service `trace-image-budget-20261029-0233.service` was queried twice per
successful import, before result admission and immediately before publication:

| Field | Verified value |
| --- | --- |
| LoadState | loaded |
| ActiveState / SubState | active / exited |
| MainPID | 0 |
| ExecMainPID | 3677967, equal to EXECUTION_STARTED.json pid |
| Result | success |
| ExecMainCode / ExecMainStatus | 1 / 0 |
| ExecStart | tools.evolution_confirmation.queue run, exact suite root and prepared hash |

No commit, reviewer, model initialization, GPU job, scoring, inference, mode
search, manuscript edit, or write to the frozen suite/runtime was performed.
Unrelated concurrent worktree changes were not edited.

## Verification

The implementation adapts the admission, finite-number, bounded-stream,
pooled-count, service, drift-check and staged-publication patterns from
`paper_evolution/tools/export_standalone.py`, without importing a frozen worker.
Only Python standard-library modules are used. `CUDA_VISIBLE_DEVICES=''` is
mandatory, although the importer does not load any GPU/model library.

1. Exact prepared-hash admission; fixed eight-cell ordering and paths; fixed
   six-arm scope; exact frozen plan hash; inherited configuration and checkpoint
   metadata links; parent protocol/seal/checkpoint links.
2. All **144** declared runtime files rehashed, plus bounded source-only inventory
   comparison that rejects extra files and symlinks. The frozen queue/worker
   sources are read as evidence, never executed.
3. Explicit prepared and whole-suite artifact hashes; exact canonical seal
   coverage; both stage-wide prediction barriers and their full cell populations;
   scoring-start copies of the correct stage barrier; each prediction-to-score
   seal chain; gate attempt records, equivalence tolerances and actual probe
   counters; deterministic runtime and shared exclusive GPU-lock identity.
4. Exact frozen cohort ID fingerprints, calibration/confirmation disjointness,
   and each manifest's image/label metadata rows; preserved unverified-history
   flags. Label hashes are checked as recorded metadata, not by opening labels.
   Order identity is bound to the pinned manifests, not regenerated using a new
   random generator implementation.
5. **18** execution-child command/receipt/log links, successful exit codes,
   stage ordering and CPU-only scoring flags. Completed queue counters must
   equal two gates, eight predictions and eight scores, with no error/active cell.
6. Every episode's integer class intersections/unions summed and compared with
   its canonical result. Pooled mIoU is recomputed as the mean of class ratios
   over positive-union classes, not an episode mean; zero-union classes are
   omitted. All accuracy counts include episode 0.
7. Every cost row checked for unique episode/arm identity, source-image count,
   alternating arm execution order, shared candidate digest, 0/1/2-view inputs,
   diagnostic call batching, evaluation/adaptation schedule and encoder sums.
   Every recorded counter total and timing denominator is independently
   recomputed, then linked to predicted and scored canonical records.
8. Exactly **48** unique CSV rows, with all scores, signed margins, timing
   fields, diagnostic costs, selected-mode flags and provenance hashes linked
   back to canonical results. No row or adverse sign is discarded.
9. Every admitted evidence file rehashed again before output replacement, and
   service success rechecked. Output contents are staged; replacement failures
   roll back previous outputs. This is not a multi-file crash-atomic transaction.

The successful run admitted **307 explicit evidence files / 66,315,785 bytes**.
Each ordinary file is capped at 16 MiB; the explicitly named 38,801,018-byte
`frozen_cohorts.json` has a 64 MiB exception. Aggregate evidence is capped at
128 MiB; inventory and stream row/line limits also apply. Evidence bytes are
cached during validation and rehashed once at the final drift boundary. There
are no recursive result-root or prediction-directory scans.

## Payload Boundary

This is **not a new current-prediction-array rehash certificate**. No prediction
NPY array, per-episode candidate receipt, RGB content, annotation content or
checkpoint weight was read. Prediction artifact *names and recorded digests*
are checked for exact coverage, but their bytes are not reopened.

The pinned frozen `tools/evolution_confirmation/queue.py` final loop, before
writing `SUITE_SEALED.json`, calls `validate_prediction` and `validate_score` for
every cell. The former reaches `run.py:verify_prediction_seal`, which checks
every linked prediction/episode artifact hash; the latter checks scoring
artifacts and invokes prediction verification again. The real root/hash/PID-bound
successful service and linked whole-suite seal support reliance on that
**completed queue's full-payload verification**, rather than repeating its
payload scans here. Original checkpoint/input verification and temporal GT
isolation also rely on that pinned execution, not on metadata alone proving
absence of out-of-band reads. Later prediction-array mutation would not be
detected by this importer if its metadata remained unchanged.

## Descriptive Results

Values below are pooled mIoU, displayed to six decimals. `Old min` means
`episode_min`; the other columns are `image_uniform`, `image_confidence`,
`image_T`, `image_C`, and `image_min`. Full unrounded values and pooled class
counts are retained in generated provenance.

### Calibration Only

These 128-image cohorts were used for selection and are not confirmation results.

| Dataset | Old min | Uniform | Confidence | T | C | Image min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PC60 | 24.846289 | 25.250079 | 25.212533 | 24.693866 | 24.939576 | 24.903935 |
| Object | 25.799302 | 25.620627 | 25.639792 | 26.711807 | 26.219353 | 25.574809 |

C minus confidence is **-0.272957 pp** on calibration PC60 and **+0.579561 pp**
on calibration Object. The recorded worst-dataset criterion selected C despite
its negative PC60 calibration margin. The sealed rule and all three candidates'
statistics remain in provenance; no confirmation information is fed into a new
mode choice.

### Confirmation Only

Each dataset uses the same 256 identities across three dependent episode orders.

| Dataset / order | Old min | Uniform | Confidence | T | C | Image min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PC60 / sorted | 26.569958 | 26.352055 | 26.341809 | 26.287044 | 26.518186 | 26.458232 |
| PC60 / 20261030 | 26.629852 | 26.496603 | 26.386787 | 26.412501 | 26.455814 | 26.456690 |
| PC60 / 20261031 | 26.681803 | 26.358987 | 26.390180 | 26.301490 | 26.544012 | 26.356117 |
| Object / sorted | 29.955178 | 26.573758 | 27.221342 | 31.100526 | 29.324619 | 29.975437 |
| Object / 20261030 | 30.351900 | 26.833645 | 28.066020 | 30.881844 | 30.166647 | 30.837342 |
| Object / 20261031 | 29.796538 | 26.328251 | 27.041885 | 30.440226 | 29.966576 | 29.832325 |

All signed confirmation contrasts versus image confidence, in percentage points:

| Dataset / order | Old min | Uniform | Confidence | T | C | Image min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PC60 / sorted | +0.228150 | +0.010246 | 0.000000 | -0.054764 | +0.176377 | +0.116423 |
| PC60 / 20261030 | +0.243065 | +0.109815 | 0.000000 | +0.025714 | +0.069026 | +0.069902 |
| PC60 / 20261031 | +0.291623 | -0.031193 | 0.000000 | -0.088690 | +0.153832 | -0.034063 |
| Object / sorted | +2.733836 | -0.647584 | 0.000000 | +3.879184 | +2.103277 | +2.754095 |
| Object / 20261030 | +2.285879 | -1.232375 | 0.000000 | +2.815823 | +2.100626 | +2.771322 |
| Object / 20261031 | +2.754653 | -0.713634 | 0.000000 | +3.398341 | +2.924691 | +2.790440 |

Selected C beats confidence on both confirmation datasets in all three orders,
but this does **not** establish best-arm performance. On Object, T exceeds C by
**+1.775907, +0.715197, +0.473650 pp** in the declared order sequence. Old min
exceeds C on PC60 by **+0.051773, +0.174039, +0.137792 pp**. On Object, C minus
old min is **-0.630559, -0.185253, +0.170038 pp**, which has mixed signs. Image
min and uniform also have mixed PC60 signs versus confidence. All remain visible.

## Cost And Timing

Accuracy and counter denominators are **128 calibration / 256 confirmation**
source images. Only episode 0 is excluded from timing. Inherited batches are
PC60 **9** and Object **14**, producing calibration timing denominators
**119 / 114** and confirmation denominators **247 / 242** respectively. The
partial last episode is included; neither timing nor counters divide by patches
or candidates instead of source images.

All-arm synchronized model-side milliseconds per measured source image:

| Dataset / stage / order | Time N | Old min | Uniform | Confidence | T | C | Image min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PC60 / calibration / sorted | 119 | 142.189 | 69.796 | 69.663 | 106.605 | 106.839 | 142.796 |
| Object / calibration / sorted | 114 | 327.658 | 216.959 | 216.465 | 270.054 | 271.848 | 326.164 |
| PC60 / confirmation / sorted | 247 | 136.390 | 66.483 | 66.448 | 101.829 | 102.030 | 136.874 |
| PC60 / confirmation / 20261030 | 247 | 136.450 | 66.446 | 66.487 | 101.851 | 102.141 | 136.939 |
| PC60 / confirmation / 20261031 | 247 | 136.319 | 66.588 | 66.424 | 101.655 | 101.789 | 136.771 |
| Object / confirmation / sorted | 242 | 314.542 | 209.470 | 209.574 | 261.083 | 264.232 | 316.154 |
| Object / confirmation / 20261030 | 242 | 314.703 | 209.650 | 209.782 | 261.031 | 264.567 | 315.874 |
| Object / confirmation / 20261031 | 242 | 315.985 | 210.886 | 210.439 | 261.734 | 265.267 | 317.099 |

Actual diagnostic encoder inputs / total encoder inputs per all source images:

| Dataset / stage | Old min | Uniform | Confidence | T | C | Image min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PC60 / calibration | 43.688 / 56.688 | 0 / 13 | 0 / 13 | 21.844 / 34.844 | 21.844 / 34.844 | 43.688 / 56.688 |
| Object / calibration | 67.609 / 98.609 | 0 / 31 | 0 / 31 | 33.805 / 64.805 | 33.805 / 64.805 | 67.609 / 98.609 |
| PC60 / confirmation, each order | 43.703 / 56.703 | 0 / 13 | 0 / 13 | 21.852 / 34.852 | 21.852 / 34.852 | 43.703 / 56.703 |
| Object / confirmation, each order | 67.680 / 98.680 | 0 / 31 | 0 / 31 | 33.840 / 64.840 | 33.840 / 64.840 | 67.680 / 98.680 |

Timing includes reset, initial readout, candidates, requested diagnostics,
adaptation, final inference/reconstruction and CPU prediction transfer. It
excludes model setup/loading, RGB decode, H2D/geometry preparation, artifact
writes, post-run provenance checks and offline scoring. These are not
end-to-end or full-split throughput claims. Exact per-arm call/input totals and
timing denominators are preserved separately for every cell in provenance.

## Tests And Outputs

Executed successfully from this worktree:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s paper_evolution/tools -p test_export_image_budget.py -v
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python \
  paper_evolution/tools/export_image_budget.py --check-only
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python \
  paper_evolution/tools/export_image_budget.py
```

**28 tests passed**. Synthetic fixtures have no prediction/episode arrays or real
labels. Failure coverage includes CPU guard, prepared/source/inventory drift,
wrong service root/hash/PID or terminal status, absent whole seal, coverage gaps,
missing/duplicate/nonfinite/wrong-sign CSV values and hash links, stage barriers,
selection links/statistics, scored metadata hashes, missing score episodes,
pooled/rounded metrics, counter/warmup/candidate errors, timing denominators,
gate counters, deterministic runtime, JSON duplicate keys, size/path/payload
guards, final evidence/service drift, check-only behavior and publication rollback.

Only these generated files were exported, after checks passed:

| Output | Bytes | SHA-256 |
| --- | ---: | --- |
| `paper_evolution/generated/image_budget_provenance.json` | 286378 | `12ee8f49ff61b7999c601cabdb288e144aef60f0962c906be9a1e2f7e87840cb` |
| `paper_evolution/generated/image_budget_comparison.tex` | 2788 | `21be7887f49a713582cd861191a34576bfbf902d46f5b87f0286e245f6027031` |
| `paper_evolution/generated/image_budget_cost.tex` | 3243 | `5a9ef9d384a14e689e2fab60fff63050ce6d211f1de73b3dbd658cf11ce5feac` |

Importer SHA-256:
`8ede2bbcc91664690d2c49b9ba2d21450be1a0db13cf8b2c488b52dcbac4a16c`.
Focused test SHA-256:
`71378e0728119cdc8b68f893f3ad960b2145f5f957461b7242e0be093c9e0804`.
The JSON stores source fingerprints, original selection/barriers/service fields,
per-class pooled counts, all six-arm results/costs, signed per-order margins,
descriptive ranges and additional C-versus-old-min / T-versus-C contrasts.
The TeX files are generated sidecars only; no manuscript inclusion was added.
They were not compiled with the manuscript.

## Claim Limits

July history is **uncertified**, not reconstructed: the frozen cohort audit
retains `historical_pool_audit.status=unresolved`,
`candidate_pool_ids_reconstructed=false`, and
`supports_identifiable_384_image_unused_cohort=false`. The prior confirmation
audit documents the archived loader/train-split mismatch and missing exact
candidate-pool mapping; no old payload/archive rescan was performed here.
`global_never_seen_certified`, `historical_index_exhaustive` and
`strong_independent_confirmation_supported` all remain false.

The supported result is descriptive improvement over image confidence for the
already selected C on these prospectively partitioned, partial-known-history
training-source confirmation cohorts. It is not independently unseen evidence,
an official full-split result, an external-method comparison, best-arm
superiority, a confidence interval, significance, or a generalization guarantee.
The three orders reuse the same images and are not independent replications.
No bootstrap or inferential CI was added, and these sidecars do not replace the
current full-split manuscript evidence or authorize any further experiment.
