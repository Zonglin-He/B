# Archive Policy

## Source and Non-Interference

The source is `/home/wwww/hzl/results`. The exporter reads artifacts without
editing source results, resetting Git state, stopping queues, changing adaptation
code, or starting GPU work. Archive compression runs with reduced CPU and I/O
priority. Capture start and finish times are recorded in `snapshot.json`.

## Coverage

All discovered experiment suites are indexed. Included artifact types are result
JSON/CSV/TSV, documentation, configuration/protocol records, logs, result tables,
checksums, prediction NPZ/NPY outputs, generated PNG/JPEG/SVG/PDF figures, and
SQLite trial databases. Small, useful records are also copied into Git for browsing.
The complete included set is available from the release, not just the browsable
subset. No result is selected for upload based on whether it favors TRACE.

The following are intentionally not uploaded:

- Model/optimizer/checkpoint tensors such as `.pt`.
- Frozen source/runtime directories, vendored implementation, source archives,
  dependencies, dataset/input asset directories, and system driver packages.
- Shell/Python experiment implementation files; this archive is for result evidence.
- Source-bearing `.diff` and `.patch` files.
- Locks, PIDs, bytecode, installers, and unknown file formats.
- Symlinks, which are indexed but never followed outside the result tree.
- Files matching credential/private-key patterns.
- Files that change while hashing and cannot be captured consistently. A change
  during archive insertion aborts export instead of publishing inconsistent bytes.

Each exclusion has an explicit reason in `index/files.csv.gz`. An excluded
directory is represented by its directory path; its contents are not recursively
inventoried. Sensitive content is not printed in an exclusion report.

## Integrity and Reconstructability

Every included path has a content SHA-256, byte count, archive name, and member
name. In this snapshot, per-path mtime is retained from source stats predating
capture where available, and left blank otherwise; shared-object timestamps are
not presented as per-path observation times. Archive members use `objects/<sha256>` and are deduplicated by
content, including repeated seed outputs. `index/archives.json` records archive
hashes and unique-object counts. Individual archives are partitioned to remain
below GitHub's release asset limit; large files do not enter Git history.

`verify --full` checks archive hashes, every decompressed object's hash, all
manifest references, and browsable-copy hashes. `restore` reconstructs original
relative paths without unsafe path traversal or overwriting existing outputs.

## Interpretation

The run index exposes recorded metadata; it does not harmonize metric units,
resolve differences in datasets/splits, or rank all historical methods together.
Read each experiment's protocol and configuration before comparing numbers.
An absent status or population field means `not declared`, not `completed` or
`full validation`. Results in a still-running suite can be completed individual
cells, but the suite as a whole must not be called complete.
Nested `job`/`outcomes` records are expanded by the schema-aware finalizer. A
generic legacy `score` is kept separately from explicitly named mIoU values.
Original absolute paths are provenance references, not portable runtime paths.

Snapshot capture is not atomic. Each included artifact is checked for stability
while hashing and packing; files can have different observation times. A newer
experiment finishing after capture requires a new explicit snapshot/update.
The finalizer reads metrics from captured archive bytes, not a second read of the
live source, and includes top-level result reports as well as suite directories.
