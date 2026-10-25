# Completed Sham Intervention Controls

New, immutable evidence update for the archived submitted TRACE. Both datasets
use a fixed 64-image diagnostic subset, seed 0, plus separate four-image smoke
checks. These are not full-validation results or three-seed estimates.

## Results and Evidence

- [Unrounded mIoU and actual correct-to-wrong transitions](results/rebuttal_sham_priority_20261025_2023/tables/sham_adaptation.csv)
- [Paired intervention-location effects and eligibility](results/rebuttal_sham_priority_20261025_2023/tables/sham_location_summary.csv)
- [Completion marker and result checksums](results/rebuttal_sham_priority_20261025_2023/SHAM_RESULTS_READY.json)
- [Capture scope and verified per-cell evidence](results/rebuttal_sham_priority_20261025_2023/capture_provenance.json)
- [Per-policy outcome index](index/runs.csv)
- [Every artifact path and SHA-256](index/files.csv.gz)
- [Archive checksums](index/archives.json)
- [Full object-hash and credential-scan verification](verification.json)

| Dataset | Native TRACE | Sham residual only | All-view sham |
|---|---:|---:|---:|
| COCO-CD | 74.4558 | 74.0739 | 72.2335 |
| COCO Object | 19.5348 | 19.9531 | 19.4728 |

Values are dataset-aggregated mIoU on the recorded 64-image subsets. The linked
CSV also reports actual whole-image top-1 correct-to-wrong transitions, whose
denominator is initially correct valid pixels; this is not a probability-drop
threshold. All three policies and both datasets are included without outcome
selection. In particular, the COCO Object residual-only sham exceeds native
TRACE by 0.4184 mIoU on this subset.

## What Is Held Fixed

Native TRACE, residual-only sham, and all-view sham use identical selected
candidates, readout locations, loss masks, and optimization settings. Sham
changes the intervention location while preserving the operator and applied
mask area. Residual-only sham retains native target and counterfactual quality;
all-view sham uses the sham location for all required intervention responses.
Applied pixel counts are paired exactly. Numerically changed pixel counts are
also recorded separately and are not claimed to be equal in flat image regions.

The paired residual-response difference is positive for 95.8974% of eligible
COCO-CD thing candidates (195) and 98.1682% of eligible COCO Object thing
candidates (1,201). This indicates response sensitivity to intervention location;
it does not establish unique causal provenance or a uniform downstream gain.
Most sham masks are shape-preserving translations; the explicit fallback is an
area-matched noncontiguous mask. Ineligible candidates retain native scores in
adaptation and are excluded only from the paired sham estimand. Eligibility,
placement type, and native/sham masks are preserved for inspection.

## Download Complete Prediction and Candidate Artifacts

Large prediction arrays and candidate/mask diagnostics are stored in the
[sham release](https://github.com/Zonglin-He/B/releases/tag/sham-20261025), not Git
blobs. From the repository root:

```bash
gh release download sham-20261025 --repo Zonglin-He/B \
  --pattern 'results-objects-*.tar.gz' --dir updates/sham-20261025/assets
python scripts/export_results.py verify --dest updates/sham-20261025 --full
python scripts/export_results.py restore --dest updates/sham-20261025 \
  --output ../B-restored-sham
```

The immutable original manifest also lists two unrelated VOC20 signal-collection
jobs; those jobs are not part of this completed sham update. Its scheduling
README is a historical execution plan, not a statement that all original or
backbone work has completed. No active or cancelled full COCO Object job is
included here. This upload does not modify or dispatch the GPU queue.

No dataset images, GT annotations, pretrained weights, external implementation,
runtime packages, or credentials are bundled. Original source hashes and config
paths are provenance references, not included dependencies. Public repository
visibility is unchanged under the user's explicit upload authorization.
