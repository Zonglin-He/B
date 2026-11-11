# B: TRACE Experiment Results

## November11 Results Snapshot

[All completed results and current status](updates/trace-results-20261111/README.md):9 native MLMP comparisons, corrected B16 geometry, R00 paired intervals, all16R03 models and historical archive index. Original15S5 is still running/unscored. Earlier releases and limitations remain unchanged.

Public evidence archive for TRACE research. The November 2 update also includes
an explicitly authorized implementation and manuscript snapshot. It preserves historical results, the submitted method's
reviewer evidence, and newer algorithm variants without treating them as one
interchangeable method.

## Start Here

- [SCOPE_v4 fixed studies and reviewed manuscript, 2026-11-03](updates/trace-scope-v4-20261103/README.md)
- [SCOPE_v4 main PDF](updates/trace-scope-v4-20261103/paper/main.pdf) and [supplement](updates/trace-scope-v4-20261103/paper/supplement.pdf)
- [Original full review](updates/trace-scope-v4-20261103/reviews/manuscript_review_draft_v001.md) and [same-reviewer revision verification](updates/trace-scope-v4-20261103/reviews/manuscript_review_draft_v002_closure.md)
- [New experiment results, code and paper, 2026-11-02](updates/trace-evolution-20261102/README.md)
- [Historical November 2 paper PDF](updates/trace-evolution-20261102/paper/build/main.pdf) and [supplement](updates/trace-evolution-20261102/paper/build/supplement.pdf)
- [November 2 implementation and tests](updates/trace-evolution-20261102/code/)
- [Completed sham intervention controls, 2026-10-25](updates/sham-20261025/README.md)
- [Completed 64-image rebuttal update, 2026-10-25](updates/rebuttal64-20261025/README.md)
- [Completed context/top-K/episode-bootstrap cache analysis](updates/mechanism-audit-20261025/README.md)
- [Latest completed results](LATEST_RESULTS.md)
- [Experiment suites](index/suites.csv)
- [Individual result records](index/runs.csv)
- [Snapshot coverage and queue states](snapshot.json)
- [Full artifact path and SHA-256 manifest](index/files.csv.gz)
- [Archive checksums and object counts](index/archives.json)
- [Integrity verification](verification.json)
- [Archive policy and interpretation](ARCHIVE_POLICY.md)

The November 3 follow-up is a bounded summary/source/manuscript delivery, not a
repeat of the historical large archive. Its independent revision verification
marks R2-R4 corrected and R1 partial: local COCO-CD/Waterbirds mask and subset
construction remains unrecovered. Historical exposure, mixed/negative/null
results, unmatched equal-time comparisons and this reproducibility limitation
remain explicit. It is not an acceptance rating or a turnkey dataset release.

Browsable tables, small result records, protocols, and configuration files are in
`results/<original-suite-name>/`. Larger diagnostics, per-image counts,
prediction arrays, logs, and figures are preserved in content-addressed archives
attached to the [results release](https://github.com/Zonglin-He/B/releases/tag/results-20261025).
The manifest maps every archived object back to its original result path.
Duplicate file contents are stored once; original paths remain individually indexed.
The run index is derived from captured archive bytes, including nested job/outcome
schemas. Generic historical optimization scores are not silently labeled mIoU.

## Download and Verify

Use Python 3.10 or newer and the GitHub CLI authenticated to an account with access:

```bash
git clone https://github.com/Zonglin-He/B.git
cd B
gh release download results-20261025 --repo Zonglin-He/B \
  --pattern 'results-objects-*.tar.gz' --dir assets
python scripts/export_results.py verify --dest . --full
```

Restore an individual suite into a **new** directory:

```bash
python scripts/export_results.py restore --dest . \
  --output ../B-restored-results \
  --suite rebuttal_original_20261025_122642
```

Omit `--suite` to restore all included artifacts. Restore refuses existing target
files rather than overwriting experiments. Archives are checked before use, and
each restored object's SHA-256 is checked independently.

## Evidence Boundaries

1. Full-validation results and fixed diagnostic/calibration subsets are different
   populations. Preserve image counts, splits, configurations, and seeds.
2. The original submitted TRACE and later two-view, selective-diagnostic,
   lazy-quality, and restricted-adaptation variants are different algorithms.
3. Historical CFC fragment guards, CLS branches, and early role formulas are not
   evidence for the final submitted algorithm.
4. Published baseline constants and local baseline reruns have distinct provenance.
5. The internal 5,322-image Waterbirds population and the strict 2,568-image test
   population must not be combined.
6. Smoke, partial, failed, cancelled, and planned records remain labeled as such.
   A queued job is not a completed experimental result.
7. This is a read-only export of a live filesystem, not an atomic system snapshot.
   Queue states describe their capture time, not the current remote machine.

No dataset collection, pretrained model weights, external MLMP implementation,
environment installation, GPU driver package, or GitHub credential is included.
Excluded paths and reasons are recorded in the artifact manifest. Existing source
hashes remain provenance references; they do not imply that every source/runtime
dependency is bundled here.
