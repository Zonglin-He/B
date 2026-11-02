# Image-Full Completion Import Audit

## Role And Scope

CPU metadata importer author, not a manuscript reviewer. This audit concerns the
completed fixed-C full-split suite and publication of its verified tables. It
does not assess manuscript quality, acceptance, adoption, or statistical
significance. No manuscript, historical PDF, frozen runtime, result, service, or
archive was modified. No Git commit was made.

Owned outputs:

- `paper_evolution/tools/export_image_full.py`
- `paper_evolution/tools/test_export_image_full.py`
- `docs/image-full-completion-audit.md`
- `paper_evolution/generated/image_full_comparison.tex`
- `paper_evolution/generated/image_full_cost.tex`
- `paper_evolution/generated/image_full_reference.tex`
- `paper_evolution/generated/image_full_provenance.json`

## Completion Evidence

Suite: `/home/wwww/hzl/results/trace_image_full_20261029_0333`.

- Prepared SHA-256:
  `e0f7374042eadb65fbc8146de6dfcf35a71151dc1c95c115b0d4cf9c8444edaa`.
- Whole-suite seal SHA-256:
  `4c2efc97904573e64f586d59f26ce5d94e58a29ea1ca93fb58c71fb5c1ed1fb2`.
- Actual retained service: `trace-image-full-20261029-0333.service`.
- Observed `LoadState=loaded`, `ActiveState=active`, `SubState=exited`,
  `MainPID=0`, `ExecMainPID=3694645`, `ExecMainCode=1`,
  `ExecMainStatus=0`, `Result=success`.
- Started `2026-10-29 03:33:23 CST`; exited `2026-10-29 06:29:28 CST`.
- Exact service command binds the frozen queue module, Python executable, suite
  root, and prepared hash. `EXECUTION_STARTED.json` binds the same execution PID.
- The importer independently checks the real service twice, including immediately
  before publication. It does not treat the suite's
  `service_success_verified=false` field as an actual failure or as a success
  certificate: the queue cannot certify its own eventual service exit.
- Exactly six prepared canonical cells, six passed GT-free gates, six sealed
  prediction cells, six sealed score cells, and 18 CSV rows were validated.
  Queue completion counters agree. Six prepare receipts and 18 ordered
  gate/predict/score receipts have exact root/hash/dataset commands, stage-appropriate
  CPU flags, zero exits, and matching log and receipt hashes.

## Independent Checks

1. Rehash all 149 frozen runtime files and reject missing, added, aliased, or
   changed runtime source paths. Independently compare the complete inherited
   61-file dependency subset with the pinned selected-C parent's source inventory.
2. Rehash the frozen plan and prepared artifacts. Verify the pinned selection
   prepared hash and selection hash, the completed selection-parent seal, the
   exact two calibration score seals and calibration barrier, and the recorded
   calibration-only candidate statistics and locked tie rule. This audits the
   pre-existing C choice; it never chooses a mode using full-split results.
3. Check exact six-dataset admission and all three arms. Compare each full manifest
   against its pinned inherited parent protocol and parent seal receipt: full
   input records, original sorted official order, unique image/label paths,
   vocabulary, query mapping, zero-label convention, checkpoint identity, and
   verbatim resolved configuration. No current configuration resolver is used.
4. Verify all six four-image GT-free gates, attempt markers, prepared/source/input
   links, actual batching, candidate equivalence, reset equivalence, zero changed
   label pixels, and support/coefficient/loss/gradient tolerances. Check actual
   diagnostic schedules for all three arms and deterministic runtime/exclusive
   lock linkage to prediction protocols.
5. Check the exact expected prediction/episode filename inventory in each
   prediction seal without opening its NPY or candidate payloads. Independently
   rehash the small protocol, initial-state, prediction-result, and cost metadata.
   All six prediction protocols record all six gate hashes and the same initial
   model-state digest.
6. Verify the exact global six-cell prediction barrier and all score-start links,
   prediction and score seals, canonical result identity, and label-hash *metadata*
   population. The score stream includes every episode, including episode 0.
7. Recompute all class intersections/unions from all 3,484 scored episode records,
   then recompute every pooled mIoU, omitting only zero-union classes. Do not
   average episode mIoUs. All 18 canonical arm metrics agree.
8. Recompute all ten actual forward/encoder counters from all 10,452 cost rows.
   Check alternating arm order, shared candidate identity/count, inactive-candidate
   step behavior, exact 0/0/1 probe schedule, batched diagnostic calls, evaluation
   and adaptation accounting, and all aggregate costs. Only episode 0 is excluded
   from timing; no prediction, accuracy, or work observation is excluded.
9. Compare every field in every one of the exact 18 CSV rows with the independently
   verified canonical metrics, signed differences, costs, timing denominators,
   manifests, selection, and seals. CSV is not accepted as primary evidence.
10. Rehash 326 explicitly read metadata/source/reference files before publication,
    recheck runtime inventory and real terminal service, and stage all four
    outputs before replacement. Validation failures leave outputs untouched;
    replacement failures restore prior outputs. Exit codes are 0 for verified
    import/check, 3 for pending, and 2 for invalid evidence.

Full populations are 5,105 / 5,000 / 1,449 / 1,449 / 500 / 5,000 in dataset order
PC60 / Object / VOC21 / VOC20 / COCO-CD / Stuff. These are 18,503 source-image
observations per arm and 55,509 prediction arrays across the three arms, not
independent replicates.

## Results And Cost

All 18 unrounded metrics, both control contrasts, pooled class counts, all actual
counter totals, configs, checkpoint identity, timing totals, and source/seal
hashes are preserved in `image_full_provenance.json`. TeX displays two decimals.

| Dataset | Uniform | Confidence | TRACE(C) | C - Confidence (pp) |
| --- | ---: | ---: | ---: | ---: |
| PC60 | 30.38 | 30.34 | 30.69 | +0.36 |
| Object | 29.81 | 30.83 | 32.59 | +1.77 |
| VOC21 | 52.44 | 52.33 | 52.62 | +0.30 |
| VOC20 | 81.78 | 82.18 | 83.49 | +1.31 |
| COCO-CD | 66.22 | 67.66 | 74.98 | +7.32 |
| Stuff | 21.74 | 21.79 | 21.84 | +0.04 |

TRACE(C) unrounded values are 30.693151576843825, 32.594583407028814,
52.621491003257155, 83.48721918638837, 74.98211836400083, and
21.836221453760693. C exceeds Confidence on all six datasets; Stuff's difference
is only +0.042865028753904966 pp, not significance evidence.

| Dataset | Timing N | Uniform ms/image | Confidence ms/image | C ms/image | C diagnostic calls / inputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| PC60 | 5096 | 83.65 | 83.73 | 119.41 | 7317 / 111514 |
| Object | 4986 | 237.34 | 237.41 | 292.00 | 10662 / 168669 |
| VOC21 | 1433 | 278.48 | 278.24 | 330.59 | 2956 / 46892 |
| VOC20 | 1448 | 279.88 | 279.94 | 295.56 | 1449 / 11583 |
| COCO-CD | 472 | 23.38 | 23.38 | 29.86 | 125 / 1998 |
| Stuff | 4995 | 129.44 | 129.35 | 186.14 | 10964 / 164543 |

Both controls have zero diagnostic calls and inputs. The C probe counts above
include warmup and equal actual candidate totals; timing N excludes only the
first episode. No old-cohort ratio substitutes for these full-population counts.
The 0/0/1 schedule does not imply C is faster than either zero-probe control.

Timing is synchronized model-side wallclock including episodic reset, initial
readout, candidate construction/selection, requested diagnostic views, LayerNorm
optimization, final inference and reconstruction/CPU prediction transfer, and
accounting hooks. It excludes model setup/loading, RGB decode, H2D/geometry
preparation, post-run provenance checks, prediction/receipt disk writes, and
offline scoring. It is not end-to-end throughput.

## Reference Boundaries

`reported_baselines.json` supplies all six reported methods across VOC20, VOC21,
PC60, Object, and Stuff. The importer calls the existing
`export_evolution.verify_reported` helper for a fresh `pdftotext -layout`
extraction, source PDF hash check, and exact main Table 5 (Original NACLIP)
mean/standard-deviation crosscheck. Every historical paper row is R and cites
`noori2025mlmp`; the new C row is L. No Adapt has no reported SD, and the single
deterministic new C run has no fabricated SD.

C minus strongest reported R, in the five-column reference order, is:

- VOC20: **-0.2727808136116323 pp**, displayed **-0.27** against MLMP 83.76.
- VOC21: +1.841491003257154 pp.
- PC60: +2.703151576843826 pp.
- Object: +3.7545834070288144 pp.
- Stuff: +0.5862214537606931 pp.

These are descriptive reference operating-point differences, not matched
checkpoint, loader, query, episode, optimization-budget, or tuning comparisons.
The negative VOC20 result is retained in the table and unrounded provenance.

The separate COCO-CD local-reference table independently rehashes the existing
native result/protocol pairs against their historical seal and recomputes their
pooled scores. Values: No Adapt 61.14434758192533, TENT 72.20029853468192,
TPT 64.27195385834406, MLMP 71.00717681177846. TENT is best; new C minus native
TENT is +2.7818198293189056 pp. Full recorded image order and checkpoint hashes
match, but native method backends, prompts and optimization budgets retain their
own protocols. These four references are not the new matched controls or reruns.
Historical native No Adapt is explicitly not certified as the new method's reset
readout. No own-method No Adapt is reused, and no old adapted method result is
relabeled as TRACE(C).

## Inputs And Labels

- `image_full_comparison.tex`: `tab:image-full-comparison` (single column).
- `image_full_cost.tex`: `tab:image-full-cost`, `tab:image-full-work` (wide).
- `image_full_reference.tex`: `tab:image-full-reference`,
  `tab:image-full-cd-reference` (wide).

Reader labels are Uniform, Confidence and TRACE(C), not internal runtime arm
identifiers. Object/Stuff abbreviations are defined in the comparison caption.
The standalone two-column article harness compiled all three inputs successfully
to a temporary three-page PDF, with no overfull hbox. The actual comparison
tabular measured 227.94174 pt against a 229.5 pt column. This verifies the table
inputs, not the concurrently edited manuscript or its final float placement.

## Tests And Reproduction

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest paper_evolution.tools.test_export_image_full
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest paper_evolution.tools.test_export_image_budget paper_evolution.tools.test_export_standalone
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python paper_evolution/tools/export_image_full.py --check-only
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python paper_evolution/tools/export_image_full.py
```

34 focused synthetic tests passed, including broken population/order/config,
149-source binding, 61-source continuity, pinned selection, barriers, gate
numeric/probe failures, actual cost/candidate/order failures, timing denominator,
pooled counts/metric/episode population, malformed/duplicate/missing CSV rows,
child stages/exits, terminal service failures/running state, pending/failed suite,
late source drift, publication rollback, payload-read and CPU guards, duplicate
JSON/nonfinite values, fresh-PDF disagreement, native nonpooled references, and
reference failure before publication. The optional installed-pdflatex test also
passed. The 41 existing image-budget/standalone exporter tests passed unchanged.

The actual final import exited 0. Measured with `/usr/bin/time -v`: 0.76 seconds
elapsed, peak RSS 136,500 KiB, no swaps. Metadata files are bounded at 16 MiB each,
with the reused evidence reader's 128 MiB raw-byte budget and 1,024-file limit;
runtime path discovery is bounded at 2,048 entries. The reference PDF is capped
at 32 MiB. Array/candidate directories are never traversed by the importer.

## Limits

No NPY prediction arrays, per-episode candidate receipts, RGB content, annotation
content, or checkpoint weights were reopened. No worker/model library was
executed or imported, no GPU work was started, and no scoring was performed.
Prediction/candidate payload integrity, original checkpoint/input checks and
temporal GT isolation explicitly rely on the pinned frozen queue's final
`validate_prediction` / `validate_score` full-payload checks before the suite
seal, plus the independently observed root/hash/PID-bound successful service.
This is **not a fresh current-payload rehash certificate**.

The complete full splits and inherited per-dataset settings have development
exposure. Completion is verified; untouched-holdout status, equal tuning,
statistical significance, independent-unseen status, exhaustive July history,
and external-method superiority under a matched protocol are not certified.
