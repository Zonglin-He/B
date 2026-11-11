# TRACE Results Through November11,2026

This is a summary-only append-only publication, not a live dashboard or a new paper. Prior releases and all historical outcomes are unchanged.

## Nine Completed TRACE-C vs Native MLMP Comparisons

| Backbone | Dataset | TRACE-C | MLMP | TRACE - MLMP (pp) |
|---|---|---:|---:|---:|
| L14 | coco_object | 32.671 | 30.728 | +1.943 |
| L14 | pc60 | 30.783 | 29.851 | +0.931 |
| L14 | voc21 | 53.381 | 50.365 | +3.016 |
| L14 | coco_stuff | 21.912 | 21.053 | +0.858 |
| L14 | pc59 | 33.587 | 32.788 | +0.799 |
| L14 | voc20 | 83.614 | 84.011 | -0.397 |
| B16 | coco_object | 29.952 | 30.746 | -0.794 |
| B16 | pc60 | 30.867 | 31.385 | -0.517 |
| B16 | voc20 | 83.648 | 84.329 | -0.681 |

Five TRACE-C point estimates are higher and four lower. Different native-system budgets/readouts remain; no direct paired significance or equally-tuned/SOTA claim. L14 uses native PIL-nearest224; B16 uses frozen G palette/CV2-nearest224. The earlier B16 PIL comparison is explicitly superseded.

## All Completed New Analyses

- [R00 all26 contrasts and15-type macro](R00_TRACE_minus_NoAdapt.csv), [full summary](R00_paired_summary.json). TRACE-minus-own-NoAdapt corruption macro: +4.399671pp,95%CI[3.718050,5.071765]; NOT a comparison to MLMP.
- [R03 all16 OOF models](R03_all16_OOF_models.json), [all48 metric rows](R03_all16_OOF_metrics.csv). All families/endpoints/quality controls retained; no best-model selection or causal claim.
- [Original TRACE eight-clean all-arm scores](TRACE_L14_8clean_all_arms.csv), [G all19-condition all-arm scores](TRACE_G_all19_conditions_all_arms.csv).
- [Nine native MLMP results and own-NoAdapt intervals](R01_native_MLMP_9conditions.json), [direct point comparison CSV](TRACE_vs_MLMP_9clean.csv).
- [R04 construction status](R04_construction_only.json):126/128 constructed/normalized foreground gates; no accuracy result.
- [Family completion ledger](FAMILY_STATUS.json), [failures and limitations](FAILURES_AND_LIMITATIONS.md), [source hashes](SOURCE_HASHES.json).

## Running Or Unfinished

The original15S5 MLMP job is running; no S5 MLMP score is published in this snapshot. [Timestamped status](S5_STATUS_NOT_SCORED.json). Separate scoring and direct TRACE-C comparison will run after sealed prediction completion and actual successful GPUexit. R02, new R04 model experiments, new R05 and six TENT cells are not complete.

## Earlier Complete And Historical Results

- [trace-scope-v4-20261103](https://github.com/Zonglin-He/B/releases/tag/trace-scope-v4-20261103)
- [trace-evolution-20261102](https://github.com/Zonglin-He/B/releases/tag/trace-evolution-20261102)
- [sham-20261025](https://github.com/Zonglin-He/B/releases/tag/sham-20261025)
- [results-20261025](https://github.com/Zonglin-He/B/releases/tag/results-20261025)
- [rebuttal64-20261025](https://github.com/Zonglin-He/B/releases/tag/rebuttal64-20261025)

The [archive index](HISTORICAL_ARCHIVE_INDEX.json) enumerates 11812 existing result/update/index files without overwriting them. The November3 release retains M01-M04, B16/SCLIP/corruption transfer, G03 severity curves, V01, groups/cases, reviewed manuscript and R1 PARTIAL. Older archives retain32 tuned full-split outcomes,455 calibration trials, allocation controls, reviewer/sham controls, smoke/failed/cancelled records.

## Exclusions

No raw RGB, GT/masks, weights, prediction arrays, per-image/candidate streams, credentials, dependency trees or TRACE-origin code are newly uploaded. No old paper/review/release is overwritten. Numeric bootstrap draws are omitted from this compact summary; estimator,seeds,replicate counts,undefined counts,point estimates and intervals remain.
