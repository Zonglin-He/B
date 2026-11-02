# Single-Probe Tuning Input Audit

Audit date: 2026-10-29. Worktree:
`/home/wwww/.codex/worktrees/trace-simplification/hzl`.
This is a new input-preparation stage, not a change to the reviewed packet,
not a model evaluation, and not a claim that new tuning has succeeded.

## API and Ownership

`tools.evolution_tuning.inputs.prepare_inputs(package_root, data_root, output,
calibration_size=256, seed=2027)` returns a JSON-serializable bundle:

```python
{
    "schema_version": 1,
    "algorithm": "ImageBalancedTRACE:C",
    "datasets": {
        "voc20": {"config": ..., "calibration": ..., "evaluation": ...},
        # All eight keys, including the blocked dataset, are always retained.
        "waterbirds_seg": {
            "config": ...,
            "calibration": None,
            "evaluation": ...,  # Complete original 5322-image population.
            "blocked_reason": ...,
        },
    },
    "source_hashes": ...,  # Absolute metadata/source paths, SHA256 and bytes.
    "source_sha256": ...,  # Package-relative metadata/source hashes.
    "annotation_pixels_read": False,
    "offline_construction_performed": False,
}
```

The controller creates the suite directory and persists `inputs.json`.
The input function accepts that directory after `queue_state.json` is created,
or an uncreated output path with an existing parent. It refuses an existing
`inputs.json` or a target inside official `data_root`. It does not itself create
any directory or file. `output` is reserved for caller-authorized fresh offline
construction artifacts if an exact builder is recovered later.

Panels include `inputs`, `ordered_ids` in the same order, `classes`, `queries`,
`query_to_class`, `reduce_zero_label`, vocabulary hash, image count, full-split
indicator, source hashes, and an input identity digest. Each explicit input has
`image`, `label`, `official_id`, RGB-byte `image_sha256`, and worker-compatible
`label_metadata=[size,mtime_ns]`. Additional `label_metadata_details` preserves
device, inode, byte size, modification time and `contents_read=False`.
The controller/worker own runtime source freezing, model execution, prediction
seals, scoring barriers and final stage status; none is performed here.

## Exact Populations

All roots below are relative to `/home/wwww/hzl/data`. Image/mask membership is
required to match the complete declared loader population. No missing pair is
silently dropped. Sorting matches the original loader's declared image paths,
including split-file datasets; symlink destinations do not regroup episodes.

| Dataset key | Root | Original full population | Training source | Calibration |
| --- | --- | --- | --- | --- |
| `voc20` | `VOCdevkit/VOC2012` | 1449, `ImageSets/Segmentation/val.txt` | 1464, same directory's `train.txt` | 256 |
| `voc21` | `VOCdevkit/VOC2012` | 1449, `ImageSets/Segmentation/val.txt` | 1464, same directory's `train.txt` | 256 |
| `pc59` | `VOCdevkit/VOC2012` | 5105, `ImageSets/SegmentationContext/val.txt` | 4998, same directory's `train.txt` | 256 |
| `pc60` | `VOCdevkit/VOC2012` | 5105, `ImageSets/SegmentationContext/val.txt` | 4998, same directory's `train.txt` | 256 |
| `coco_object` | `coco_mlmp` | 5000, `images/val2017` | 118287, `images/train2017` | 256 |
| `coco_stuff` | `coco_stuff` | 5000, `images/val2017` | 118287, `images/train2017` | 256 |
| `coco_cd` | `coco_cd_ovss` | 500, `splits/test_balanced.txt` | 1668, `splits/train_corr0.95.txt` | 256 |
| `waterbirds_seg` | `waterbirds_seg/correlation_0.95` | 5322, every original matched pair | 2458, `splits/train.txt`, entirely overlapping full | Blocked |

VOC uses `JPEGImages/*.jpg` and `SegmentationClass/*.png`; Context uses the
same RGB root and `SegmentationClassContext/*.png`. Object uses
`annotations/{val2017,train2017}/*_instanceTrainIds.png`; Stuff uses
`annotations/{val2017,train2017}/*_labelTrainIds.png`. COCO-CD uses
`images/<split-relative-stem>.jpg` and respectively
`masks/test_balanced/<stem>.png` or `masks/train_corr0.95/<stem>.png`.
Waterbirds retains `images/<CUB-relative-stem>.jpg` and `masks/<stem>.png`.
The original full loader is not changed to the 2568-image test-only protocol.

The six-dataset Object config's `data_subdir="coco_stuff"` is legacy metadata;
its explicit inherited `data_dir` and this preparation both use `coco_mlmp`.
The resolved worker config preserves the inherited schema and records both
the inherited root and the root rebased under the explicit `data_root` argument.

## Inherited Starts

For six datasets, read the exact JSON at
`/home/wwww/hzl/results/trace_image_full_20261029_0333/configs/<key>.json`.
For PC59 and Waterbirds, read
`/home/wwww/hzl/results/trace_v2_simplification_20260721_225456/`
`frozen_configs_final/<key>/best_config.json` and carry only its four optimizer
parameters, not its old role-aware objective or reliability controls.

| Key | Batch | LR | Steps | K |
| --- | --- | --- | --- | --- |
| `voc20` | 1 | 0.001105 | 9 | 8 |
| `voc21` | 16 | 0.00063 | 36 | 33 |
| `pc59` | 7 | 0.0015 | 11 | 24 |
| `pc60` | 9 | 0.00225 | 11 | 22 |
| `coco_object` | 14 | 0.003 | 29 | 34 |
| `coco_stuff` | 5 | 0.0034 | 12 | 33 |
| `coco_cd` | 28 | 0.0086 | 5 | 4 |
| `waterbirds_seg` | 16 | 0.003 | 4 | 7 |

Every config has 224-square resize and patch geometry, stride 112, view batch
16, and original query-extension flags. Extensions are enabled for VOC,
Context, Object and Stuff, disabled for COCO-CD and Waterbirds. VOC20 and PC59
retain `reduce_zero_label=True`. Vocabularies and background handling mirror
`trace_ovss/data/vocabulary.py` without importing the tensor-backed package.
Historical view batch 8 for PC59/Waterbirds is recorded as provenance, not
inherited into the new fixed execution protocol.

## Selection and Isolation

The prospective panel uses seed 2027, without mask inspection or class-coverage
selection. Rank each training source by ascending SHA256 of this UTF-8 string:

```text
trace-single-probe-tuning-v1\n{seed}\n{dataset}\n{official_id}\n
```

Use the first 256, with official ID as a collision tie-break, then sort those
selected rows by original image path to freeze episode order. This is a new
explicit deterministic sampler, not a replay of July's class-aware greedy
selection or an assertion of identical historical selected IDs.

Source IDs are `voc:<YYYY_NNNNNN>`, `coco:<12-digit-id>` independent of split,
and `cub:<species-directory/image-stem>` independent of composite path.
Before hashing any RGB, check RGB/label physical aliases over all enumerated
training and evaluation populations. Duplicate identities/paths/inodes within
a population fail. For the seven native training splits, any own-evaluation
ID, canonical-path or inode overlap is fatal; it is not repaired by filtering
the evaluation set or silently discarding overlapping training examples.
The audit retains ordered source-ID hashes for the complete training and full
populations, not just the selected panel.

Only RGB contents and bounded textual source/split/identity metadata are read.
Segmentation labels are only stat-ed, never decoded or hashed. RGB stat changes
while hashing fail; metadata/source hashes are rechecked before returning.
No dataset registry globals are modified. The function imports only standard
library code and the CPU-only confirmation cohort helpers. It does not import
torch, instantiate a loader, open checkpoints, or perform model inference.

Own-target train/eval disjointness does not establish cross-dataset disjointness
or erase historical development exposure. In particular VOC/Context and the
COCO-derived tasks can share sources across tasks. No globally untouched,
never-seen, class-coverage or all-population RGB-content deduplication claim is
made. The historical parameter audit and reviewed packet are unchanged.

## Waterbirds Block

Live identity/split metadata, without reading segmentation masks, establishes:

- Original full contains 5322 distinct CUB-relative source IDs.
- `train.txt` contains 2458 distinct IDs, all within full; metadata `split=0`
  independently matches exactly those IDs.
- The existing historical 512-example calibration manifest is also entirely
  within the original full population: 512/512 overlap.
- CUB has 5994 training IDs; 3240 are outside the original 5322-image full.
  The sorted remaining-ID list has SHA256
  `957fa5b9f8a59cfccaa0efad2f03f4f6b0479b9f909f3280ee60295cd7d4eed9`.
  These are candidate source identities, not 3240 validated Waterbirds inputs.
- Existing full metadata names 5322 distinct Places background identities.
  This audit does not claim any new Places-background population is prepared.

Bounded source inspection uses the two source tarballs in historical `_snapshot`,
each under 1 MiB, verified against its existing `SHA256SUMS`; no extraction,
prediction-history search, or large paper archive scan is performed:

| Source archive | SHA256 |
| --- | --- |
| `root_method_and_experiments.tar.gz` | `2b80f347545fa0e943188ed4589c7edb970030b5f93ddab5103102185d5986b7` |
| `mlmp_trace_sources.tar.gz` | `7ab2d93b9e065eed58b009c34a6dfa53a3e067821dec6ac10f258d1cfb4aca33` |

Their inventories and hashes of the inspected prepare-data/dataset-loader
members are retained in the machine-readable Waterbirds audit. The root
archive's `experiments/prepare_mlmp_data.py` prepares Pascal Context and COCO
Object, not CUB/Places composites. Neither source archive includes
`experiments/build_trace_v2_calibration_splits.py` or an identifiable original
Waterbirds compositor. The bounded current `experiments`/`tools` source search
also did not locate that compositor; absence outside these sources is not
certified.

The existing local `experiments/build_trace_v2_calibration_splits.py` has SHA256
`98c79da331520cefaf346982fe081c40ec947f233e7698bdf069dd0cf345548b`.
It generates `train.txt` from existing Waterbirds metadata and selects existing
training examples using their masks. It does not construct new composites.
It was read as source, never imported or executed; its directory-changing,
split-writing and label-reading behavior is not part of ordinary preparation.

Therefore the returned Waterbirds entry is explicitly `calibration=None` plus
`blocked_reason`, with its 5322-image full panel and inherited config intact.
The seven other datasets can progress; this is not an eight-dataset completion.

The feasible construction route requires recovery and freezing of the exact
original compositor: bird-class mapping, Places class/background mapping,
foreground placement/crop/resize, segmentation threshold, correlation and RNG
protocol. Only then can a separately invoked **offline calibration-data build**
create 256 training composites from remaining CUB training IDs and Places
background IDs outside original full, under a fresh caller-supplied output.
It must record builder/dependency hashes, source image and source-mask hashes,
source identities, transformation settings, RNG seeds and generated file hashes.
Reading source masks at that explicit construction stage is not inference or
ordinary annotation-blind discovery. Never write into the official evaluation
root, silently reuse overlapping `train.txt`, shrink full, or infer missing
dataset semantics from desired labels. No such construction was performed.

## Verification

Fixture tests: `python -m pytest -q tests/test_evolution_tuning_inputs.py`.
All 18 CPU fixture tests pass after controller-interface alignment; the combined
run with `tests/test_confirmation_cohorts.py` and the CPU-only
`tests/test_evolution_tuning_queue.py` passes 100 tests. Coverage
includes all eight keys, complete full order/counts, source/config/vocabulary
hashes, fixed geometry/extensions, source-overlap failure, missing-mask failure,
duplicate splits, image/label hardlink aliases, annotation-read guards, stat
drift, deterministic panels, inherited-config drift, metadata-source drift during
RGB hashing, archive checksums, source
root selection, controller-owned output, and a subprocess import test excluding
torch/model packages. Fixtures deliberately use non-image label sentinel bytes.

A real CPU-only `prepare_inputs` pass on `/home/wwww/hzl/data`, seed 2027 and
256-image panels, completed in 22.69 seconds: all eight full counts above matched,
all seven native calibration panels contained exactly 256 disjoint source IDs,
and Waterbirds returned blocked with full intact. It hashed every returned RGB,
recorded label stat metadata and 39 textual/source file identities. An additional
`Path.open` guard rejected any attempted `.png` read; none occurred. The pass
created no output path or file. A second real pass on the final implementation,
including complete-population identity digests and end-of-preparation source
hash rechecks, completed in 21.79 seconds with the same counts, no torch import,
no annotation PNG read and no output path created.
This checks preparation, not model/gate execution,
prediction seals, calibration quality, scores or research success.

## Explicit Waterbirds Development-Overlap Opt-In

Amendment date: 2026-10-29. The user explicitly authorized using the existing
Waterbirds training population for tuning and describing the overlapping
original full as **development evaluation**, not an independent test. This
appendix changes only that opt-in path; the preceding default-blocked behavior,
historical evidence and earlier verification receipts remain intact. The
separate official-compositor design does not authorize generation in this path.

The input API now accepts:

```python
prepare_inputs(
    package_root, data_root, output,
    calibration_size=256,
    seed=2027,
    allow_waterbirds_development_overlap=True,
)
```

`allow_waterbirds_development_overlap` defaults to `False` and requires exact
Python `bool` type. Integers, floats, strings, `None`, lists and dictionaries
are rejected before filesystem discovery; truthiness is not consent. Explicit
`False` and an omitted flag return the same blocked Waterbirds behavior.
The opt-in path is bounded to `calibration_size=256` and `seed=2027`; conflicting
values fail rather than silently changing the protocol. Other datasets retain
their existing configurable behavior when this opt-in is disabled.

When enabled, Waterbirds samples only the existing 2458 `train.txt` identities
using the unchanged SHA256 ranking already documented above, including dataset
key `waterbirds_seg`, seed 2027 and official CUB-relative identity. Take the first
256, then freeze order by original source image path. No source images, masks,
new pairings, or synthetic examples are constructed. Calibration is no longer
`None`; full retains all 5322 original input rows and exactly their original
order, vocabulary, label reduction and geometry. There is no evaluation
population filtering, train/evaluation relabeling of files, or data mutation.

The Waterbirds-only branch does not call the disjoint selector that necessarily
rejects this known overlap. It retains discovery and global RGB/label alias
guards, rejects duplicate source IDs, canonical paths and image inodes, and
requires training source/path/inode and label stat metadata to match the same
official IDs in full. It explicitly verifies that the 256 unique selected IDs
belong to both existing train and original full. The other seven datasets still
use the original disjoint selector, ranking and selection metadata; the opt-in
does not permit any overlap or alter their panels.

All four Waterbirds metadata locations carry the exposure boundary:
`calibration`, `evaluation`, `input_audit`, and `selection`. Specifically:

| Field | Opt-in value |
| --- | --- |
| `protocol` / `evaluation_scope` | `development_overlap` |
| `development_overlap` | `True` |
| `allow_waterbirds_development_overlap` | `True` |
| `cal_full_source_id_overlap` | `256` |
| `train_full_source_id_overlap` / `declared_train_count` | `2458` |
| `original_full_count` | `5322` |
| `independent_test_claim` | `False` |
| `global_never_seen_certified` | `False` |

The claim boundary states that selected training sources overlap full and that
the full result is development evaluation with no independent-test or globally
never-seen claim. Waterbirds selection's `disjointness_scope` explicitly records
overlap instead of retaining the default disjointness description. Opt-in audit
reason/feasible-choice text describes the authorized reuse, not a still-blocked
construction. The bundle records the exact bool and retains
`annotation_pixels_read=False` and `offline_construction_performed=False`.
RGB hashes, source hashes, input digests and label stat receipts remain required.
No mask bytes are read by either path, and this flag never authorizes scoring
before prediction sealing.

Verification for this amendment used CPU fixtures only, including an exact
5322-full/2458-train synthetic Waterbirds population and native training fixtures
large enough for all seven 256-image panels. Tests assert the exact hash-selected
IDs, repeatability, original full membership/order, identical other-seven entries
with the flag toggled, metadata at all four exposure locations, strict bool and
fixed size/seed admission, default blocking, no label reads, and rejection of
identity/path/inode/label aliases even when overlap is permitted. A separate test
confirms other-dataset overlap remains fatal under opt-in. The existing CPU-only
import check continues to exclude torch/model package imports.

No real input preparation, source generation, queue/controller/worker edit,
runtime replacement, model/GPU execution, scoring, or Git commit was performed
for this amendment. Main owns the fresh continuation, its success/config-seal
dependencies and all-eight selection/full-stage barriers; this input flag alone
does not launch it or establish that any tuning or evaluation has completed.

Amendment test results: `tests/test_evolution_tuning_inputs.py` passes 42 tests;
the combined run with `tests/test_confirmation_cohorts.py` passes 107 tests.
`git diff --check` reports no whitespace errors in the three owned files.
