# Waterbirds Source-Disjoint Calibration Design

Date: 2026-10-29. This is a new bounded official-source audit, following the
discovery of `acharaakshit/label-flips`; it supersedes the earlier *bounded
source-not-found* explanation, not the earlier overlap measurements. It does
not modify that audit, inputs.py, the controller, any dataset, or the reviewed
packet. No renderer was executed and no images or masks were generated.

## Decision

**The official compositor is now identified, its 0.95 pairing manifest matches
the local original manifest byte-for-byte, and sufficient local source files
exist for 256 distinct CUB-training foregrounds and 256 distinct Places-training
backgrounds, both source-disjoint from the original full 5322.** Missing
compositor/source availability is no longer the blocker.

There are two essential qualifications:

- The remaining CUB-training population has **3238 landbirds and only two
  waterbirds**. A 128/128 class-balanced, distinct-source training calibration
  is impossible under the requested original-full exclusion. A 254/2 panel is
  feasible but has only two independent waterbird foreground sources.
- The official CLI renders a fixed bundled pairing table; it cannot request
  new foreground/background identities. Independent calibration requires an
  explicitly frozen **new pairing manifest** and a thin invocation of the
  unchanged official per-example renderer. This is a new training-source
  calibration extension, not an existing official split or an unchanged
  invocation of the published CLI.

The reproducible 254/2 design below is not an authorization to generate it.
No existing input API or queue status was changed. Main's seven-dataset work
remains calibration-only; **no full scoring before all eight selections**.

## Official Source Pin

Repository: [acharaakshit/label-flips](https://github.com/acharaakshit/label-flips).
Resolved once through the GitHub commit API, then all source/manifest downloads
used the immutable commit:

`f736a34cb02a4268d1284e9387e9e570afbc4d69`

Commit date: `2026-03-16T05:26:18Z`; message: `add dataset sources`.
Commit tree: `13dca2e08eeb4a8f969ad940d2f3abefe671ff59`.
The recursive tree response was not truncated. Every downloaded source and CSV
was checked against both its exact tree byte size and Git blob SHA1; SHA256
values are listed below. No clone, archive of the repository, LFS object,
checkpoint, model, dataset download, or package installation was performed.

All ten downloaded files, including the two API JSON receipts, reside only at:

`/home/wwww/hzl/results/waterbirds_calibration_source_audit_20261029_0800`

Total downloaded payload: **1,023,663 bytes**. Largest individual file: 496,562
bytes. Source paths below are repository-relative; local downloads are flat
and the package initializer is named `segshort_init.py`.

| Official source | Bytes | SHA256 |
| --- | ---: | --- |
| `README.md` | 4564 | `df0ddce6120170389b52264032b829e0f9edc936c056e0e08de35a745e840fb3` |
| `scripts/generate_datasets.sh` | 248 | `96e86f312852df4a27d0c90c022bdaffa82ec1490298ba2e38b81d595e496aca` |
| `src/segshort/dataset_generation/generate_waterbirds.py` | 6612 | `caf89c73a575e3ec98891b9d5bd63d5adc0d04817cd2ca81114792c5d844a2ca` |
| `src/segshort/__init__.py` | 221 | `770e69ab4a603a011c19b27ddd252401534bdc83b722289bc58f66ddf861278e` |
| `requirements.txt` | 243 | `4b0dd10c937560f31577281064cb08a8b75044fcc83ab45e6b1f66172533187d` |
| `setup.py` | 1978 | `a9a7957538412f738dd881a88202c3cfc525895448f73da14b6ae38b0028ae7f` |
| `annotations/waterbirds_seg/metadata_corr0.95.csv` | 496293 | `307165f782c614a8f30397990eed9a160dc93b381ca1f98cdaac3792362915e0` |
| `annotations/waterbirds_seg/metadata_corr0.5.csv` | 496562 | `760aa93693667d7bed817acc150b11202671cae9e74b9f34ea12902bc7a983e1` |

The renderer's Git blob SHA1 is
`dc49e769d51f5ee5fd93f4f35911160215fcf198`; the 0.95 CSV's is
`5f6401c9e219f78ac925e189f18b10383b538d3e`.
The retained `commit.json` and `tree.json` bind the source paths and blobs to
the commit. `tree.json` SHA256 is
`3fab2e0eac09df081bba5e6d60b9660b63a8ee0886951a682cc03c45a0b9ec8b`.

## Renderer and Match

The [pinned CLI script](https://github.com/acharaakshit/label-flips/blob/f736a34cb02a4268d1284e9387e9e570afbc4d69/scripts/generate_datasets.sh)
dispatches Waterbirds to the dataset-generation module. Its
[pinned implementation](https://github.com/acharaakshit/label-flips/blob/f736a34cb02a4268d1284e9387e9e570afbc4d69/src/segshort/dataset_generation/generate_waterbirds.py)
contains the actual compositor, not a model-based replacement:

- `render_example` reads the original CUB RGB and its corresponding CUB
  segmentation, and resolves the metadata's Places background.
- The background is center-cropped to foreground aspect ratio and resized
  with Pillow Lanczos, including the source's recursive upsize rule and
  integer offsets.
- Composition uses the segmentation's RGB values divided by 255 as alpha;
  foreground and background contributions are rounded separately to uint8
  before addition. It is not binary-mask RGB compositing.
- Output labels use grayscale segmentation strictly greater than 127:
  background 0, landbird 1, waterbird 2. Output dimensions are original CUB
  dimensions; TRACE's 224-square preprocessing remains downstream.
- RGB uses the input `.jpg` suffix and Pillow's default save settings; masks
  use grayscale PNG. The renderer itself draws no random numbers. The CLI
  copies the selected bundled CSV and renders its rows; it provides neither
  new-pair sampling nor a generation-seed option.

The official 0.95 CSV and existing
`data/waterbirds_seg/correlation_0.95/images/metadata.csv` both hash to
`307165f782c614a8f30397990eed9a160dc93b381ca1f98cdaac3792362915e0`.
That proves exact metadata bytes, row order, source pairing, class assignment
and split declarations match. Both have 5322 rows, 2661 per bird class, and
2458/296/2568 rows declared train/val/test. Every `group_id` equals
`2*y + place` in both official manifests.

This identifies a provenance-justified official rendering implementation for
the matching dataset. **It does not prove existing JPEG/PNG pixel or byte
identity with a replay**: no rendering or evaluation-label reading occurred,
and the historical Pillow/codec versions are not certified. The repository
specifies dependency lower bounds, not an exact historical lockfile.

## Local Source Availability

Raw root: `/home/wwww/hzl/data/waterbirds_sources`.
`data/label_flips_raw` remains empty and was not used or populated.

### CUB Foregrounds

Read only `CUB_200_2011/images.txt`, `train_test_split.txt`, official metadata
and filesystem stat information. The official 0.95 manifest covers all 200
species; each species has exactly one bird-class assignment. The 0.5 manifest
has no conflicting species assignments. No image-class classifier or mask
inspection was needed to obtain the bird classes.

| Quantity | Verified count |
| --- | ---: |
| CUB image identities in `images.txt` | 11788 |
| CUB training identities, flag 1 | 5994 |
| CUB training identities outside original full | 3240 |
| Remaining landbird training identities | 3238 |
| Remaining waterbird training identities | 2 |
| Remaining identities with existing, nonempty source RGB | 3240 |
| Remaining identities with existing, nonempty source segmentation | 3240 |
| Physical RGB inode overlap with original full's raw CUB sources | 0 |

The two remaining waterbird IDs are:

- `1318`: `024.Red_faced_Cormorant/Red_Faced_Cormorant_0046_23446.jpg`
- `3578`: `062.Herring_Gull/Herring_Gull_0143_46461.jpg`

Both have CUB training flag 1. In the *other* official 0.5 manifest, ID 1318
is assigned to validation and ID 3578 to training. Thus the proposed eligibility
is explicitly **raw CUB training, outside the original 0.95 full**, not
"training in every historical protocol" or globally untouched data.

`CUB_200_2011/segmentations` is an existing symlink to `../segmentations`;
it resolves all required source masks without any filesystem change. Mask
contents, dimensions and decode validity were not opened or certified here.
Identity metadata hashes:

- `images.txt`: `e1cfc5ed1f4bd1b838040784b6f75c44f7a25d9b31707a3adab49ec694618c1f`
- `train_test_split.txt`: `8bbbfcc09b4d3c9c6e3e40c9874bd863eb6658f4f456e472df1b33d3403d2600`

### Places Backgrounds

The exact matching manifest uses two land and two water categories. The
official path resolver supports the existing local category-directory layout:

| Official category | Local training directory | Local files | Used by original full | Remaining nonempty files |
| --- | --- | ---: | ---: | ---: |
| `/b/bamboo_forest/` | `places365_standard/train/bamboo_forest` | 5000 | 1360 | 3640 |
| `/f/forest/broadleaf/` | `places365_standard/train/forest-broadleaf` | 5000 | 1301 | 3699 |
| `/l/lake/natural/` | `places365_standard/train/lake-natural` | 5000 | 1342 | 3658 |
| `/o/ocean/` | `places365_standard/train/ocean` | 5000 | 1319 | 3681 |
| Total | Four bounded directories only | 20000 | 5322 | 14678 |

Land pool: 7339 unused files. Water pool: 7339 unused files. All original
5322 background identities resolve within these local training directories;
the remaining pools have zero physical inode overlap with them. Identity
normalization must preserve category plus basename, not basename alone, and
must equate official easyformat paths with local flattened-category paths.

The large global Places train/val text lists have 1,803,460 and 36,500 lines,
respectively. No global image inventory or full archive scan was needed:
availability checks were limited to the four relevant directories and named
CUB sources. No RGB or source-mask bytes were read during availability checks.
This is an identity/path/inode audit, not an all-content-duplicate audit.

### Existing Alternatives

The official 0.5 CSV has 5324 rows. Of its training rows, 858 foregrounds are
outside the original 0.95 full, but **zero training rows have both foreground
and background outside that full**. Across the entire 0.5 table, only two
background identities are new. Neither unchanged official CLI invocation
therefore supplies the requested 256 fully source-disjoint calibration pairs.

## Reproducible Construction Plan

The following is a proposed, explicit new pairing policy to freeze *before*
generation or any trial score. It preserves the official class definitions,
background categories and renderer; it does not pretend the original
publication supplied this sampling algorithm or this class mixture.

1. **Freeze source and runtime.** Use the exact commit, renderer, official CSVs
   and source metadata hashes above. Use the existing CPU-capable interpreter
   `/home/wwww/miniconda3/envs/mlmp/bin/python`, not an installation of the
   repository's broad model dependencies. Metadata-only inspection finds
   Python 3.10.13, NumPy 1.26.4, Pillow 12.2.0, pandas 2.3.3 and tqdm 4.68.2.
   Record the actual selected runtime and Pillow JPEG/zlib codec versions in
   the future construction receipt. These are available current versions,
   not certified original rendering versions.

2. **Freeze candidate identities.** Build CUB eligibility from the native
   training flag and subtract every CUB ID in original full, independent of
   its local filename or background. Build land/water Places eligibility from
   the four native training directories and subtract every normalized
   original-full background ID. Reject canonical-path/inode aliases and empty
   files. Recheck all counts and species-class consistency before selection.

3. **Select 254 landbirds and both remaining waterbirds.** For deterministic
   rankings define `R(domain, id)` as ascending SHA256 of the UTF-8 string
   `trace-waterbirds-calibration-pairs-v1\n2027\n{domain}\n{id}\n`, then ID as
   tie-break. Use canonical `cub:<species-directory/stem>` and
   `places:<category>/<basename>` identities. In each bird class select the
   required number by `R(foreground, id)`. This metadata-stratified choice is
   necessary to retain any waterbird examples: applying the existing
   unstratified seed-2027 tuning sampler to all 3240 remaining CUB identities
   yields 256 landbirds and zero waterbirds. Do not silently call that panel
   class-covered or representative of the original balanced training set.

4. **Freeze correlation and unique pairings.** For each bird class with `n`
   chosen foregrounds, use `floor(0.95*n + 0.5)` aligned backgrounds; choose
   which foregrounds are aligned by `R(context-assignment, id)`. This gives
   the exact group counts below. Select the required number of distinct
   unused backgrounds from each context pool by `R(background, id)`; pair
   ascending `R(pair-order, foreground-id)` with ascending selected-background
   rank within each context. No reuse, post-render replacement, or image/mask
   quality-based resampling. Emit a fresh CSV with CUB `img_id`, relative RGB
   filename, official species-derived `y`, calibration `split=0`, context
   `place`, `group_id=2*y+place`, and an official-style Places path. It must be
   identified as newly constructed calibration metadata, not a copied
   published training split. Freeze final CSV order by ascending relative CUB
   RGB filename. A decode/shape error stops the build.

   | Bird class | Land background | Water background | Total |
   | --- | ---: | ---: | ---: |
   | landbird, `y=0` | 241 | 13 | 254 |
   | waterbird, `y=1` | 0 | 2 | 2 |
   | Total | 241 | 15 | 256 |

   Aligned fraction is `243/256 = 0.94921875`; exact 0.95 is not an integer
   count at 256 examples. Waterbird context coverage is necessarily absent
   for land backgrounds in this rounded design; it is not a four-group
   balanced calibration. This limitation is explicit, not repaired by
   duplicating the two waterbirds or using evaluation foregrounds.

5. **Render only after separate authorization.** Import the unchanged pinned
   downloaded generation module directly, avoiding installation of unrelated
   training dependencies. Invoke its per-example renderer with the frozen
   rows, existing raw roots, and image/mask directories under a *new*,
   caller-specified calibration output. Do not invoke its default fixed-CSV
   CLI or edit module globals, official files, raw sources, symlinks, or
   `correlation_0.95`. Only this explicit offline construction stage may read
   selected raw CUB segmentations and create training masks. It is not model
   inference and does not open current full-evaluation masks.

6. **Seal construction before admission.** Record exact row order; seed and
   ranking/group rules; renderer, dependency and codec versions; hashes of
   raw selected RGBs, source masks and Places backgrounds; canonical IDs,
   paths and inode/stat information; and hashes of every constructed RGB,
   label, CSV and receipt. Verify exactly 256 unique foreground IDs and 256
   unique background IDs, both disjoint from all original-full source IDs;
   recheck physical aliases, dimensions and class mapping. New training-mask
   validation is a construction check, not full-evaluation scoring. Ordinary
   subsequent discovery returns to RGB hashes plus stat-only label metadata.
   Main must separately implement/admit these artifacts; this audit makes
   no inputs.py or controller change.

### Metadata-Only Plan Check

The proposed rules were recomputed in memory, without calling or importing the
renderer and without writing a proposed CSV. Assertions verified 256 distinct
foreground IDs and 256 distinct background IDs, both disjoint from original
full, and 256 distinct physical RGB inodes on each side. The resulting group
counts are exactly 241/13/0/2 for group IDs 0/1/2/3.

For a later independent implementation, the in-memory proposal has the
following reproducibility checks. SHA256 uses UTF-8 canonical JSON with sorted
keys, separators `(',', ':')`, no NaN, and integer numeric row fields; source
ID lists follow the final relative-filename row order:

- Ordered foreground ID list:
  `1fc246a8207880eb01802830e7febdd23d55de7f0e4d3eb85f41d2d9004b05e9`
- Ordered background ID list:
  `0d3881c0e3b84224c6126081193f673cc28626211d23b81eeafd400cd6b320b1`
- Full pairing-row list, with the seven CSV field names listed in step 4:
  `1bcae0e8da26ac5fb60591aa86cf68ef41def50a731d79273595aa08738d2873`

These are hashes of a **proposal**, not a construction seal, an on-disk manifest,
validated image payload, generated calibration panel, or successful experiment.

## Statistical and Execution Boundary

There is no missing-data or renderer-discovery barrier to the proposed 254/2
offline build. There **is** a genuine capacity barrier if class balance or a
substantial independent waterbird validation population is required: only two
eligible waterbird training foregrounds remain after excluding original full.
Multiple backgrounds, crops or repetitions cannot create new independent
foreground sources and must not be counted as additional eligible IDs.

A 254/2 panel has a markedly different class mixture from the original 1229/1229
training population. Pooled three-class mIoU on it can be highly unstable for
waterbirds, and hyperparameter selection can overfit those two foregrounds.
Neither this protocol nor its later performance should be described as a
balanced official training calibration, independent confirmation of a global
claim, or evidence of a never-seen development history. No full result may
choose this pairing policy or fix the class-coverage problem retrospectively.

Current stage status remains: **official source and raw availability audited;
new pairing design specified; generation not executed; Waterbirds calibration
not yet prepared or selected**. Seven calibration-only queues may continue.
The all-eight-selection barrier remains in force before any full scoring.
