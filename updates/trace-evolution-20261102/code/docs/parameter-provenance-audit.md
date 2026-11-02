# Historical Parameter-Provenance Audit

Audit date: 2026-10-29. Worktree: `/home/wwww/.codex/worktrees/trace-simplification/hzl`.
Scope: VOC20, VOC21, PC60, COCO Object, COCO-Stuff, COCO-CD.
Machine-readable evidence: [audit.json](/home/wwww/hzl/results/trace_parameter_provenance_20261029_0030/audit.json).

## Conclusion

The current four-parameter tuples are inherited development choices, not six
fresh calibrations for the final two-view method. Explicit frozen records link
them to July role-aware TRACE searches. Those records support training-subset
selection for the documented search stages; they do **not** certify a globally
unexposed validation history. Earlier full evaluations exist, the frozen
wrapper's no-selection flag is written unconditionally by its copier, and later
full-validation comparisons explicitly conditioned learning-rate retention or
replacement on VOC20, COCO-Stuff and COCO-CD.

A1338 and Ccli2312 did not newly search these four parameters. That final-suite
boundary does not erase earlier development exposure. Per-dataset settings are
permitted; this disclosure neither requires equal search budgets nor authorizes
baseline reruns, GPU work, retries or tuning.

## Appendix-Ready Summary

Tuple order is **episode batch / learning rate / optimizer steps / K**.
`D` = `dominance_tuning`; `F` = `dominance_fine_retune`, under historical root
`H=/home/wwww/hzl/results/trace_v2_simplification_20260721_225456`.
Rows count surviving ledger entries, **not** unique configurations, independent
replicates, all attempts, or globally exhaustive tuning budgets.

| Dataset | Current tuple | Historical selected tuple | Explicit selection record under H | Recorded linked search rows | Disclosure / uncertainty |
| --- | --- | --- | --- | --- | --- |
| VOC20 | 1 / 0.001105 / 9 / 8 | Same | `F/voc20/best_refined_config.json` | 20 + 22 + 46 = 88 | Fast-128 screen/refine precede 512 retune. Later full-validation comparison retained LR over two alternatives. Earlier coarse ledger is separate, not a proven lineage link. |
| VOC21 | 16 / 0.00063 / 36 / 33 | Same | `D/voc21/best_config.json` | 35 | Explicit early whole-config freeze. K/batch searched; LR stage incomplete; no step search. Original source selecting 36 steps is unknown. Selected-record score conflicts with its CSV/result. |
| PC60 | 9 / 0.00225 / 11 / 22 | Same, apart from float serialization | `D/pc60/best_config.json` | 59 | Seven documented coordinate stages. Earlier pre-v2 provenance/exposure is not certified. |
| COCO Object | 14 / 0.003 / 29 / 34 | Same | `D/coco_object/best_config.json` | 60 | Seven documented coordinate stages. Later fullscores are not its selection record. |
| COCO-Stuff | 5 / 0.0034 / 12 / 33 | 5 / 0.003375 / 12 / 33 | `F/coco_stuff/best_refined_config.json` | 59 + 61 = 120 | Initial full eval before fine retune; aggressive LR search. Current LR was later accepted using full-validation mIoU, not training-only calibration. |
| COCO-CD | 28 / 0.0086 / 5 / 4 | 28 / 0.008625 / 5 / 4 | `F/coco_cd/best_refined_config.json` | 59 + 55 = 114 | Earlier full evaluations; batch extension reached 32. Current LR was later accepted using full-validation mIoU. |

Historical JSON literally stores PC60 LR `0.0022500000000000003`,
COCO-Stuff `0.0033750000000000004`, and COCO-CD `0.008624999999999999`.
The JSON audit retains those raw numbers. PC60's representation difference is
not treated as a substantive new choice; the two COCO LR replacements are.

This table describes whole-config inheritance into the current suites.
Historical per-coordinate selections are separately recorded in the JSON;
it must not be rewritten as four independent new selections per current dataset.
Execution view batch is a separate control: historically 24 for VOC20 and 8
for the other five, currently 16. No identity of the entire old/new method,
execution configuration, or scores is asserted.

## Calibration Identity

Exact manifest paths are `H/calibration_splits/<key>.json` for each key below.
All declare `split=train`, seed 2027, requested/selected size 512, and
`class-aware greedy over deterministic candidate pool`. The JSON audit copies
the **ordered indices and image IDs**, records path prefixes, and hashes the
exact source files. Each has 512 unique indices and sample paths.

| Key | Manifest filename | Selected / declared dataset size | Candidate pool | Declared covered classes |
| --- | --- | --- | --- | --- |
| voc20 | `calibration_splits/voc20.json` | 512 / 1,464 | 1,464 | 20 / 20 |
| voc21 | `calibration_splits/voc21.json` | 512 / 1,464 | 1,464 | 21 / 21 |
| pc60 | `calibration_splits/pc60.json` | 512 / 4,998 | 4,096 | 60 / 60 |
| coco_object | `calibration_splits/coco_object.json` | 512 / 118,287 | 4,096 | 81 / 81 |
| coco_stuff | `calibration_splits/coco_stuff.json` | 512 / 118,287 | 4,096 | 171 / 171 |
| coco_cd | `calibration_splits/coco_cd.json` | 512 / 1,668 | 1,668 | 3 / 3 |

VOC20 additionally has `H/calibration_splits_fast128/voc20.json`: 128 unique
training indices, seed 2027, 20/20 declared classes, drawn from the same
1,464-image pool. Its indices are a subset of the 512 manifest's indices.
The 128 screen/refine sources explicitly name this manifest; the final
dominance retune explicitly names the 512 manifest.

Coverage and parent dataset sizes above are manifest declarations, not a fresh
mask-level audit. The builder source uses training labels for class-aware
subset construction. No labels, images, masks or predictions were reopened;
physical sample identity, label correctness and global train/eval disjointness
were not independently recertified.

For all 29 primary stage-winner records, the surviving command/config files
specify `--data_split train`, the correct dataset-specific 512 manifest,
seed 2027, one evaluation trial, and no max-batch truncation. Those checks
support the recorded selection scope; they are not a replay of every trial.

## Search Records

Primary selection metrics are `training-split calibration mIoU` for D,
and `class-covered training-subset mIoU` for F and fast screen/refine.
The coordinate runner's tie rule is: within 0.02 mIoU, prefer lower measured
runtime. Thus a stage winner need not have the largest rounded mIoU.
The CSVs, exact stage value lists, statuses and token counts are in the JSON.

The following are **observed min/max trial values** in the explicitly linked
ledgers. They are not continuous grids, Cartesian products, predeclared global
limits, or the complete historical budget.

| Dataset | Batch | LR | Steps | K | Stage row counts |
| --- | --- | --- | --- | --- | --- |
| VOC20 | 1..24 | 0.0001..0.01 | 3..17 | 5..11 | Fast screen 8/6/6; fast refine 7/8/7; final refine 7/24/8/7 |
| VOC21 | 1..16 | 0.00003..0.01 | Not searched; 36 retained | 1..35 | K coarse 7; K fine 7; batch 16; partial LR 5 |
| PC60 | 1..16 | 0.00001..0.01 | 1..48 | 1..32 | 7/7/16/7/5/10/7 |
| COCO Object | 1..16 | 0.00001..0.01 | 1..48 | 1..35 | 7/7/16/7/5/11/7 |
| COCO-Stuff | 1..24 | 0.00001..0.05 | 1..48 | 1..35 | Initial 59; refine K 7, batch 24, aggressive LR 15+8, steps 7 |
| COCO-CD | 1..32 | 0.00001..0.02 | 1..48 | 1..32 | Initial 59; refine K 7, batch 24, extension 9, LR 8, steps 7 |

PC60/Object's seven stages are K coarse/fine, batch, LR coarse/fine,
steps coarse/fine. COCO-CD's extension includes the incumbent 24 and
25..32, hence nine rows rather than eight new batch values.

Explicit VOC20 links are final F -> `fast_coordinate_refine/voc20/
best_refined_config.json` -> `fast_coordinate_screen/voc20/
best_screen_config.json`. The fast screen has no K-search stage and carries
K=8. Its ultimate fixed-K selection source is unknown within these links.
The separate `tuning/voc20/trials.csv` has five rows (K values
32, 2, 8, 14, 24). A sixth K=16 scalar result exists without an exit-code or
duration file and is absent from that ledger. It is not counted as a verified
completed trial or silently added to the final lineage.

COCO-Stuff and COCO-CD F records explicitly name their corresponding
D `best_config.json` as starts. Their historical selected tuples changed
from (5, 0.00225, 13, 32) to (5, 0.003375, 12, 33), and from
(16, 0.0075, 6, 6) to (28, 0.008625, 5, 4), respectively.

VOC21's record states an early freeze at the user's request after 35 completed
trials. LR=0.00063 was already present during K/batch stages; the five recorded
LR trials do not establish a finished independent LR optimization, and no
step stage exists. Its `interim_best` and frozen embedded payload report
52.11 mIoU, while the referenced batch-16 CSV row and `results.txt` report
52.10. The earlier queue log also reports 52.11. Both versions are preserved;
the cause is unknown. Other 28 primary stage-winner scores agree with their
referenced results. These scores are training-subset selection evidence, not
current-method full-evaluation performance.

## Freeze and Exposure

The post-fast queue log and shell source record this order:

1. Fine retune VOC20, COCO-CD, COCO-Stuff.
2. Copy selected configs into `frozen_configs_final`.
3. Generate `final_experiment_suite`.
4. Run the three-seed full main table and later analyses.

All six frozen embedded source payloads exactly match their current linked
selection files; the copier preserves source config fields and inserts method
defaults. File mtimes place the six frozen files between
**2026-07-23 03:33:12.953955 and 03:33:13.061107 UTC**.
This is a filesystem/queue-order boundary, not a timestamped cryptographic
pre-exposure seal. Exact original freeze time remains unknown.
`freeze_config.py` unconditionally writes
`heldout_validation_used_for_selection=false`; its presence cannot prove
that no earlier full scores were seen or influenced development.

Known earlier full-evaluation exposures below have surviving scalar results
and commands using `--data_split eval` without indices/max-batch restriction.
The audit did not reopen evaluation payloads to recertify population counts.

| Dataset | Earlier full-run cell under H | Tuple at that run | Reported mIoU |
| --- | --- | --- | --- |
| VOC20 | `dominance_full_voc20/voc20/seed_0` | 1 / 0.0013 / 10 / 8 | 83.94 |
| VOC20 | `dominance_lr15_full_voc20/voc20/seed_0` | 1 / 0.0015 / 10 / 8 | 83.84 |
| VOC21 | `dominance_single_seed_full/voc21/seed_0` | 16 / 0.00063 / 36 / 33 | 52.16 |
| PC60 | `dominance_single_seed_full/pc60/seed_0` | 9 / 0.00225 / 11 / 22 | 30.37 |
| COCO Object | `dominance_single_seed_full/coco_object/seed_0` | 14 / 0.003 / 29 / 34 | 32.00 |
| COCO-Stuff | `dominance_single_seed_full/coco_stuff/seed_0` | 5 / 0.00225 / 13 / 32 | 21.46 |
| COCO-CD | `other_datasets_initial_full_dominance/coco_cd/seed_0` | 4 / 0.003 / 5 / 1 | 52.88 |
| COCO-CD | `dominance_single_seed_full/coco_cd/seed_0` | 16 / 0.0075 / 6 / 6 | 73.38 |

The fast-queue log explicitly interleaves tune/full runs; its COCO-Stuff full
result predates the later F retune. File mtimes place all eight above before
the final frozen registry. Existence establishes exposure, not the causal use
of each score in a later choice. That causal use remains unknown except for
the documented rounding acceptance below.

`final_experiment_suite/command_manifest.json` points at the frozen registry;
its six copied four-parameter configs match that registry.
`08_three_seed_main/mean_std_summary.csv` is **later fullscores**, not the
original selection source. Its reported means are VOC20 83.96, VOC21 52.16,
PC60 30.37, Object 32.00, Stuff 21.50, CD 75.13 (three seeds declared each).
No current-method superiority or independent parameter validation follows.

## Later LR Acceptance

[rounding_validation.md](/home/wwww/hzl/results/trace_hparam_cleanup_20260825/rounding_validation.md)
explicitly says full-validation mIoU must not decrease. Its eight direct
`summary.json` pointers were parsed and hashed, not inferred from equal
config values.

| Dataset | Values compared and reported full mIoU | Decision |
| --- | --- | --- |
| VOC20 | 0.001105: 83.96; 0.00110: 83.94; 0.00111: 83.94 | Retain 0.001105 |
| COCO-Stuff | 0.003375: 21.50; 0.0034: 21.51 | Replace with 0.0034 |
| COCO-CD | 0.008625: 75.13; 0.0085: 75.08; 0.0086: 75.16; 0.009: 75.00 | Replace with 0.0086 |

VOC20/CD values have direct cleanup scalar summaries, all with seed 0.
Stuff's accepted value has a direct cleanup summary; its old 21.50 is
documented in the note and historical three-seed summary, with no separate
paired old-value cleanup-summary pointer in that note. Do not invent one.
These are three/two/four documented comparison values, not an exhaustive count
of all cleanup attempts. This was validation-conditioned acceptance, not merely
notation cleanup or training-only selection.

Git config blobs at `f16de98`, `3c619a4`, and `08e002a^` preserve the
original rates, subsequent replacements, and pre-simplification config.
Resolved commits and SHA-256 hashes are in the JSON. The latter two also
explicitly disclose the full-validation acceptance condition in comments.

## Current Final Suites

A1338 = `/home/wwww/hzl/results/trace_self_evolution_20261028_1338`.
Ccli2312 = `/home/wwww/hzl/results/trace_evolution_voc20_cli_recovery_20261028_2312`.
Their small runtime config snapshots match all six current four-parameter
tuples. Frozen controller sources set `parameter_search=False` and obtain
configs through `suite_inputs.expected_inputs` / `best_config`.

Small suite seals, manifest seals, queue states and summary CSVs were checked:
18/18 and 4/4 cells are recorded complete, and summary/queue hashes match
their suite seals. A1338 summaries declare full populations CD 500,
PC60 5,105, VOC21 1,449, Object 5,000, Stuff 5,000; Ccli2312 declares
VOC20 1,449. These are fixed-config evaluations, not original selection records.
The large manifest payloads and prediction seals' target payloads were not
opened or rehashed; this is not a complete final-suite recertification.

## Evidence Scope and Unknowns

The JSON `source_files` registry identifies **195 exact files, 1,207,479 bytes**,
including the 389,475-byte compressed root source snapshot. Each textual
JSON/CSV/source/log read was capped at 128 KiB; each parsed ledger at 1,000
rows. It contains all seven manifests' ordered indices/image IDs, the exact
per-stage tried values/counts, 29 primary stage-winner checks, eight earlier
full-eval command/result pairs, rounding pointers, and separately hashed Git
blobs. Hashes establish the bytes inspected now, not historical immutability.

The root `_snapshot/root_method_and_experiments.tar.gz` inventory has 177
members but lacks `cfc_tta/configs/trace_v2.py` and the v2 tuning runner.
The live legacy runner imports that now-missing canonical config module.
Legacy scripts were hashed as present now, not certified as the exact July
execution bytes; saved commands/configs/ledgers are the run-specific evidence.
The older CFC `best_clean.py` snapshot/patch is context, **not an explicit
lineage bridge**. Matching numbers there do not prove how TRACE inherited
them. The four inspected snapshot items covered by `SHA256SUMS` all match.

Unresolved fields remain null/unknown: the original selection source for
VOC21's 36 steps and other missing canonical starting choices; the fixed-K
origin before VOC20 fast screen; the VOC21 score discrepancy's cause; global
attempt/retry budgets; exact cryptographically sealed historical freeze time;
complete pre-v2 exposure history; and causal influence of earlier full scores
on later search design.

Excluded: prior reviews, model/image/label/prediction payloads, candidate
diagnostic tables, large suite manifests, the paper archive, unrelated result
roots and global exposure searches. Only this new document and the new JSON
audit were written. No configuration, manuscript, experiment, existing result
or baseline was changed, and no GPU work, retry or tuning was launched.

**Disclosure boundary:** inherited per-dataset settings with documented
training-subset search stages, known earlier full exposures, and explicit
validation-conditioned later LR decisions. No global exhaustive no-exposure
certification and no claim of equal tuning budgets.
