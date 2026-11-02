# TRACE Evolution Snapshot, 2026-11-02

This is an explicitly authorized publication of existing experiment evidence,
current source code and the existing manuscript. It does not start experiments,
repair the baseline worker, create new scientific results or claim completion.
Historical files below retain their original wording and dates; the status in
this document supersedes historical statements that publication or a new
baseline attempt had not yet been authorized.

## Contents

Download the [code and paper bundle](https://github.com/Zonglin-He/B/releases/download/trace-evolution-20261102/trace-evolution-code-paper-20261102.tar.gz)
or browse the files below. The [release](https://github.com/Zonglin-He/B/releases/tag/trace-evolution-20261102)
also provides standalone main/supplement PDFs and all raw-result archive parts.

- [Current implementation, controllers and tests](code/)
- [Current TeX sources and generated figures/tables](paper/)
- [Main paper including appendix, PDF](paper/build/main.pdf)
- [Standalone supplementary PDF](paper/build/supplement.pdf)
- [Exact frozen round05 manuscript and its original manifest](paper-round05/)
- [Source-file provenance and checksums](source_manifest.json)
- [Current 32-outcome tuned full-split table](tables/tuned_full_32.csv)
- [Ten completed allocation-control outcomes](tables/allocation_full_10.csv)
- [All 455 calibration configurations, separately sourced](tables/calibration_trials_455.csv)
- [Captured baseline service status and human decision](execution_status_at_capture.json)
- [Browsable result records and seals](results/)
- [Complete result-path to archived-object manifest](index/files.csv.gz)
- [Export coverage](snapshot.json) and [archive verification](verification.json)

Captured coverage: 31 suites, 689,669 result files and 585,799 unique objects;
71,594,242,619 logical bytes compressed into 50 parts totaling 3,415,198,137
bytes. The separate code/paper snapshot has 466 checksum-listed files.

The release contains raw predictions, per-image counts, JSONL episode/cost
records, logs and other included diagnostics. Files are content-addressed and
deduplicated without losing their original relative paths. This export covers
TRACE experiment/review directories dated October 28-30, including failed,
unexecuted, historical, negative and mixed outcomes. Dates identify capture
scope, not a common algorithm or a guarantee that every directory completed.
No outcome is selected for publication because it favors TRACE.

## Current Scientific Status

The current ImageBalancedTRACE experiment is
`trace_single_probe_waterbirds_full_20261029_0817`: eight full protocols,
32 outcomes and 455 calibration configurations including its imported
`trace_single_probe_tuning_20261029_0748` parent. Waterbirds has 256 calibration
examples overlapping the unchanged 5,322-image full population, so it is
development evaluation, not untouched test evidence. Other historical exposure
and asymmetric control selection remain disclosed. One seed is not an
uncertainty estimate. Uniform/Confidence replacements use TRACE-selected
settings, not independently optimized competitor settings.

The allocation-control suite `trace_allocation_controls_20261029_2055` completed
ten outcomes: five fixed-config arms on COCO Object (5,000 images) and PC60
(5,105). Its complete source-bound certification is recorded in the baseline
suite's `DEPENDENCY_COMPLETED.json`, SHA256
`23dbd7be5a46b205194f68a71d332c7592f60cbec874d87e592d12edb6c09496`.
The evidence is mixed: image-token scores 33.607080 vs production image-region
32.671159 on Object, while episode-region scores 30.851537 vs image-region
30.782843 on PC60. These are ablations; production global C is not replaced by
the full-result winner. They do not resolve selection or generalization debt.

The first local 18-cell baseline queue
`trace_local_baselines_20261030_0008` failed during an unrelated CFCTTA eager
package import before the MLMP constructor. It has 0/18 passed gates, 0/18
formal predictions and 0/18 new scores, not measured numerical inequivalence
or OOM. Preserve the failed runtime record. On November 2 the human explicitly
approved a minimal import-isolation repair, real CPU regression, independent
audit, fresh freeze and one controlled new queue. That approved repair/queue
has **not been executed in this publication task**. See the captured status.

The manuscript is the existing round05 eight-dataset draft, not a final paper
integrating the mechanism and future baseline evidence. Its actual independent
review was 5/10, confidence 4/5; raw reviews and dispositions remain in
`paper/reviews`. No earlier score carries forward. The frozen packet manifest
SHA256 is `386ae89c92a21329dcef08d21262071dac27e71788da07879170885943c8fb23`.

One pending correction is stated explicitly: primary MLMP Table 5 reports
PC59 Original MLMP 31.95 (+/-0.02); the old paper reference ledger omitted it.
The only absent MLMP protocols in the agreed scope are COCO-CD and Waterbirds,
not PC59. Existing frozen PDFs are not silently rewritten to conceal this.
Published MLMP references and local rerun results have separate provenance.

## Download, Verify And Restore

From the repository root, using Python 3.10+ and GitHub CLI:

```bash
gh release download trace-evolution-20261102 --repo Zonglin-He/B \
  --pattern 'results-objects-*.tar.gz' \
  --dir updates/trace-evolution-20261102/assets
python scripts/export_results.py verify \
  --dest updates/trace-evolution-20261102 --full
python scripts/export_results.py restore \
  --dest updates/trace-evolution-20261102 \
  --output ../trace-evolution-restored \
  --suite trace_single_probe_waterbirds_full_20261029_0817
```

The archive verification checks publication integrity, not a repeated scientific
queue certification. The restore operation refuses existing destination files.
Absolute original paths in seals/configs are provenance, not portable paths.
Use `code/pyproject.toml` to install the published implementation. Archived
controllers require the original datasets/checkpoints and pinned external
baseline dependencies; this is not a bundled one-command dataset installation.

## Publication Boundaries

The source snapshot includes dirty and untracked non-ignored research files
as present at capture, based on commit `aba513c`, without publishing the source
repository's Git history or altering either research checkout. Source hashes
are listed separately from experiment result hashes. Existing manuscript PDFs
and all 62 frozen round05 packet files were checked at capture. No new review
or revised-paper compilation is claimed by this upload.

Model/optimizer checkpoints, datasets/original images, GT annotation directories,
credentials, symlinks, locks and bytecode are excluded. Frozen result runtime
and copied-parent dependency directories are not exported recursively; the
current implementation is supplied separately in `code/`. Every exclusion is
recorded in the result manifest. Included figure assets remain part of the
authorized manuscript, not a redistribution of complete datasets.

Old review/delivery suites and older configurations remain historical evidence.
Some generic legacy run-index fields do not parse the new `arm_results` schema;
use the canonical 32- and 10-row tables above for the current comparisons.
Scientific stopping conditions have not been met; no acceptance is promised.
