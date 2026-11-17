# MLMP Paper Alignment Audit

Audit date: 2026-11-13, approximately 15:00 CST. This is a diagnostic report,
not a new model evaluation, parameter search, or correction of sealed results.

## Scope and Evidence

Current full-evaluation root:
`[revision-results]/TRACE_area_C_full_20261113-1100-v1`.
Official source: `upstream/MLMP_source_v1`, pinned upstream commit
`2a887de018b3f2c3b132f69fe806bab12a2d511f`.

Paper: [NeurIPS 2025 MLMP, Table 5 and Appendix B/D](https://proceedings.neurips.cc/paper_files/paper/2025/file/6c5b82193c5d8e6aa5806239676ddc97-Paper-Conference.pdf).
All following clean comparisons use NACLIP with ViT-L/14. Paper values are
three-repeat means; this local full evaluation uses one canonical episode order.
Waterbirds segmentation and COCO-CD have no corresponding result in that table.

The CPU-only unit `trace-mlmp-paper-audit-20261113-1500` finished successfully.
It checked all 40 frozen source hashes, verified retained actual successful score
exits and five output hashes for each of the eight completed L14 conditions, and
recomputed all 24 arm mIoU points exactly from compact GROUP_COUNTS I/U. No raw
GT was reopened, no model was built, and no GPU control was run by this audit.

## Actual Clean Results

| Dataset | Images | Paper MLMP | Current MLMP | Difference (pp) | Current area_C |
| --- | ---: | ---: | ---: | ---: | ---: |
| VOC20 | 1449 | 83.76 | 83.6624038499 | -0.0975961501 | 86.1213273375 |
| VOC21 | 1449 | 50.78 | 50.6490107381 | -0.1309892619 | 50.9848424075 |
| PC59 | 5105 | 31.95 | 32.7201133233 | +0.7701133233 | 33.6183983734 |
| PC60 | 5105 | 27.99 | 29.7994074740 | +1.8094074740 | 30.6588345279 |
| COCO Object | 5000 | 28.84 | 30.9344903948 | +2.0944903948 | 33.0278304955 |
| COCO Stuff | 5000 | 21.25 | 21.2496303098 | -0.0003696902 | 22.0255699136 |
| Waterbirds test | 2568 | n/a | 50.3257399050 | n/a | 54.9894198143 |
| COCO-CD test | 500 | n/a | 72.2805476530 | n/a | 77.7033091402 |

Canonical local evidence is `<root>/<task>/score/RESULT.json`, with tasks
`L14_voc20`, `L14_voc21`, `L14_pc59`, `L14_pc60`, `L14_coco_object`,
`L14_coco_stuff`, `L14_waterbirds_seg`, and `L14_coco_cd` respectively.
Current scores are not uniformly higher than the paper.

## Confirmed Object Vocabulary Difference

AST extraction of the official `get_cls_idx` and comparison against the actual
frozen full INPUTS vocabulary found 113 official queries versus 112 local queries.
The sole removed `(query, class)` pair is `bed` mapped to background class 0.
The foreground `bed` entry is retained locally. Evidence:

- Official `upstream/MLMP_source_v1/utils/class_extensions/coco_object.txt`,
  line 1 includes background `bed`; line 61 is the foreground class.
- Packaged `trace_ovss/resources/class_names/cls_coco_object.txt`, line 1 omits
  background `bed`; line 61 retains the foreground class.
- The frozen full INPUTS contains the packaged 112-query vocabulary.

This changes the input to the official entropy objective and text-query
competition. It is a confirmed protocol mismatch, not a proven quantitative
explanation for the entire +2.09449 pp difference. The effect requires an isolated
113-versus-112-query evaluation with other factors held fixed. No such control
has been run here. Both local MLMP and area_C use the same 112-query vocabulary,
so their result is a shared-local-protocol comparison, not strict paper protocol.
First-four parity under a common supplied vocabulary cannot detect this mismatch.

## PC59 and PC60 Are Not Explained Yet

PC59 has exact ordered query/class-pair equality. PC60 differs only by trailing
space in official `wood ` versus local `wood`; the official tokenizer produced
identical token IDs for all seven official prompt templates, so this whitespace
does not explain the PC60 gap.

Official `prepare_data` shuffles samples, while this evaluation freezes one
canonical ordering. With two-image episodic adaptation, changing batch membership
can change results. This is a protocol difference, not a demonstrated explanation
for +0.77011/+1.80941 pp. Paper standard deviations are 0.02/0.03 pp, so the gaps
must not be dismissed as ordinary reported variability. Exact annotation
conversion/provenance, official split identity, and episode-composition effects
remain to be isolated; this report does not claim those checks are complete.

## NoAdapt Naming and Other Checks

Local `native_NoAdapt` is MLMP-zero-update: it retains MLMP multi-layer fusion
and averaged seven-prompt evaluation. The official `bash/v20/no_adapt.sh` uses
the default `tent` method without adaptation, whose readout is the final-layer
reference-prompt baseline. These NoAdapt columns are not interchangeable.

The local MLMP optimization settings remain fixed: Adam, lr 0.001, 10 steps,
batch size 2, alpha_cls 1, last 18 L14 layers, seven official prompts, and reset
of model/optimizer per episode. TRACE-selected settings were not applied to MLMP.
However, fixed optimization settings do not imply complete paper reproduction.
Prior shorthand statements saying only 'MLMP paper fixed' were too broad.

Both the paper/official evaluation and this current full clean run resize to
224x224. Code inspection agrees on CV2 bilinear RGB resize, nearest-neighbor GT
resize, ignore 255, and dataset-specific reduce-zero-label rules. Both use pooled
per-class intersection/union followed by a class mean, not a mean of image mIoUs.
This does not certify that every underlying annotation file is identical.
The earlier PIL-bilinear clean runs are a separate geometry and must not be
silently substituted for this current CV2-bilinear comparison.

## Execution and Reporting Boundary

The finite full queue continues unchanged: eight L14 conditions have completed;
B16 COCO Object is active. No frozen source/config was edited, no completed run
was retried, no monitoring agent/automation was enabled, and no result uploaded.
Do not label the Object baseline a strict reproduction. Do not adjust parameters
to force agreement with published numbers. Any official-vocabulary control must
be a separate preserved version, not an overwrite or hot change of this queue.
Do not claim all PC gaps are explained or that current full results establish
paper-protocol superiority over MLMP. Historical development exposure remains.
