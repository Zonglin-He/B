# Completed 64-Image Rebuttal Update

This is a separate, immutable addition to the earlier full results snapshot.
It contains six completed 64-image experiment cells, four separate four-image
smoke cells, and offline geometry/instance audits. It does not contain unfinished
original-queue cells or claim the waiting second backbone has completed.

## Results

- [Full interpretation and controlled comparisons](results/rebuttal_priority64_repaired_20261025_1940/RESULTS.md)
- [All unrounded experimental outcomes](results/rebuttal_priority64_repaired_20261025_1940/tables/summary.csv)
- [Offline pixel-space instance geometry](results/rebuttal_priority64_repaired_20261025_1940/offline_pixel_geometry/result.json)
- [Schema-aware individual outcome index](index/runs.csv)
- [Full path/content-hash manifest](index/files.csv.gz)
- [Archive checksums](index/archives.json)
- [Integrity and credential-scan verification](verification.json)

All outcomes are retained. In particular, finer internal allocation improves
COCO Object on this fixed subset but not PC60, and role policy differences do
not establish a large, consistent gain. C-to-W here means actual top-1 correct
to wrong transitions, not probability decrease. Read each denominator/protocol.

## Download the Complete Prediction and Diagnostic Evidence

Large arrays, per-image and per-candidate evidence are release assets rather
than Git blobs. From the repository root:

```bash
gh release download rebuttal64-20261025 --repo Zonglin-He/B \
  --pattern 'results-objects-*.tar.gz' --dir updates/rebuttal64-20261025/assets
python scripts/export_results.py verify --dest updates/rebuttal64-20261025 --full
python scripts/export_results.py restore --dest updates/rebuttal64-20261025 \
  --output ../B-restored-rebuttal64
```

No external MLMP implementation, dataset images/annotations, model weights,
GPU driver package or credentials are bundled. Frozen source hashes and
configuration paths are retained as provenance, not as bundled dependencies.
The public upload was explicitly authorized; repository visibility is unchanged.
