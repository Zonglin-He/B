# Prospective Confirmation Cohort Audit

Date: 2026-10-29. Scope: CPU-only metadata preparation for a newly authorized
TRACE method repair. This is not an experiment result or manuscript score.
No research result root or real-data cohort JSON was created during preparation. The reusable API
returns a bundle in memory; its CLI writes only to a fresh, explicitly named
JSON file in an already existing directory.

## Predeclared Selection

- Datasets: `pc60` and `coco_object`, both training-source populations.
- Seed: **20261029**. Calibration/development: **128** images per dataset.
  Confirmation: **256** other images per dataset.
- Rank each eligible official ID by the hexadecimal SHA-256 of the ASCII text
  `trace-confirmation-v1\n20261029\n{dataset}\n{official_id}\n`.
  Sort ascending by digest, with the official ID as tie break. Take ranks
  0..127 for calibration and 128..383 for confirmation, retaining rank order.
- Official IDs are `voc:YYYY_NNNNNN` and `coco:NNNNNNNNNNNN`. COCO identity
  deliberately does not contain train/validation or Object/Stuff/CD namespaces.
- Exclude the union of all seven indexed July calibration selections, both
  hash-verified October 26 calibration panels, and all enumerated evaluation
  IDs in the two source-image families. Include
  COCO-CD's validation list as an additional conservative exclusion.
- No class-coverage optimization, label-derived selection, GT-based filtering,
  image-content ranking, parameter search, model imports, or GPU execution.
  Read selected RGB bytes only to obtain SHA-256 hashes. Labels are only
  enumerated and `stat`-checked; do not read or hash label contents.

For this protocol, confirmation is a prospectively partitioned training-source
cohort, **not certified historically independent confirmation**. It is not the
official evaluation set. Calibration labels may be used for
development only after that stage's predictions are sealed. Confirmation
predictions must also be sealed before confirmation labels are opened; loss
selection and stopping rules belong to the separate worker/protocol.

## Actual Sources

`tools/suite_inputs.py` defines calibration as the training counterpart of the
repository's evaluation spec. Its `expected_inputs` constructs a torch dataset,
so this adapter reads `DATASET_SPECS`, `BEST_CONFIGS`, and `EVALUATION_COUNTS`
as AST literals without executing their modules, then applies the same training
split transformation. Every source file is hashed. No actual dataset object is
needed, and no inference hyperparameters are selected here.

The checked dry run explicitly used these directories:

| Dataset | Source directory | Training identity / masks | Observed train / own eval |
| --- | --- | --- | --- |
| PC60 | `/home/wwww/hzl/data/VOCdevkit/VOC2012` | `ImageSets/SegmentationContext/train.txt`; `JPEGImages/{id}.jpg`; `SegmentationClassContext/{id}.png` | 4,998 / 5,105 |
| COCO Object | `/home/wwww/hzl/data/coco_mlmp` | `images/train2017/{id}.jpg`; `annotations/train2017/{id}_instanceTrainIds.png` | 118,287 / 5,000 |

The COCO override matches the small sealed A1338 controller's explicit
`data_dirs` setting. Repository defaults instead name `coco_stuff`; do not
silently treat the two annotation roots as equivalent. Their RGB directories
resolve to the same `/home/wwww/hzl/data/coco/{train2017,val2017}` sources.
Label-content equivalence between annotation roots has **not** been checked.
Use `--data-dir coco_object=/home/wwww/hzl/data/coco_mlmp` for the counts and
selected identity fingerprints reported here. Explicit source overrides are
recorded in `audit.data_dirs`, and source-ID manifests bind actual label paths.

Additional enumerations: VOC training 1,464 and validation 1,449; Context
validation 5,105; COCO validation 5,000; COCO-CD `test_balanced` 500 and
`val_balanced` 200. VOC20/VOC21 and PC59/PC60 share their respective official
split identities; Object and Stuff share COCO val2017 identities. Other dataset
families are outside this source-identity scope, not silently certified by a
global content-duplicate search.

## Counts

These are filesystem/identity counts from the bounded in-memory dry run, not
mask-level class coverage, measured performance, or estimated counts. History
exclusions include the seven explicitly mapped July selections and both
recovered October 26 panels. The July label-inspected candidate pools remain
unresolved, so these are **partial-history eligible** counts, not globally
untouched counts. This follow-up authorizes only the two exact referenced
input metadata files up to 16 MiB each; no prediction payloads were read.

| Quantity | PC60 | COCO Object |
| --- | ---: | ---: |
| Complete training image/mask pairs | 4,998 | 118,287 |
| Own official evaluation population | 5,105 | 5,000 |
| Own train/eval identity or physical-path overlap | 0 | 0 |
| Training images in any enumerated evaluation list | 383 | 670 |
| Training images in indexed historical selections | 652 | 1,745 |
| History/evaluation exclusion intersection | 31 | 8 |
| Exclusion union | 1,004 | 2,407 |
| Eligible after exclusions | **3,994** | **115,880** |
| Calibration | 128 | 128 |
| Confirmation | 256 | 256 |
| Remaining unselected eligible images | 3,610 | 115,496 |

The 383 PC60 exclusions are VOC official-validation images. COCO-CD includes
COCO train2017 images: its test/validation exclusion union intersects the COCO
training source in 670 images. Excluding only a target's own evaluation split
would miss these cross-dataset evaluation images. History counts union VOC
selections into PC60 and Object/Stuff/CD selections into COCO Object; they are
not just the target's own historical 512-image manifest.
The follow-up adds 256 unique recent COCO identities, of which four were
already historical exclusions and two were already evaluation exclusions.
Thus 250 additional images are excluded. None intersects either previous
calibration128 or confirmation256 array, so **all four selected-ID
fingerprints below remain unchanged**, despite the smaller eligible population.

Ordered official-ID array fingerprints, using SHA-256 of JSON encoded with
`sort_keys=True, separators=(",", ":"), allow_nan=False` and UTF-8:

| Dataset / cohort | SHA-256 | First ranked ID |
| --- | --- | --- |
| PC60 calibration128 | `e87f7ea00581403a1f2ecfacd50f1402fdbf932fec620f6a2f876bed6d879f19` | `voc:2008_000648` |
| PC60 confirmation256 | `6e6a61edc7f6031d1b4d4bc3ea086b8a67874cf5caddb0609128a22fa97019b9` | `voc:2009_000290` |
| Object calibration128 | `fdff87dfadd4dd9543f919a8fcf838e6046b96ac2101e6852dd8773fbdb1ce67` | `coco:000000366257` |
| Object confirmation256 | `cdc524a427d45f55bc01bbde12141bf88bdd76baa5fcc9c9e46732fcda9dcc6b` | `coco:000000040891` |

Before running a worker, verify these fingerprints against
`datasets[dataset].selected_ids_sha256`. A changed source population or new
historical exclusion may legitimately require a new cohort declaration; do not
silently present changed IDs as this declaration.

The dry run's full source-ID manifest digests cover the ordered arrays of
`{official_id,image,label}` triples, not annotation contents:

| Source-ID manifest | Canonical-JSON SHA-256 |
| --- | --- |
| `pc60/train` | `729b8d1fef0d11c6b06ddc08839971ebf478cfb4084ad5a221ed225f6ddd8788` |
| `coco_object/train` | `296c8147d6ad8328c1e6259a1842d3d2d0bfda3cb93a717649fee594fc40fc64` |
| `voc21/train` | `def75b80004ebb015298da00635e4ea59295b1aa22c255c27096886064d5cfac` |
| `pc60/evaluation` | `914565f7f21a241ad253eb237b3843995d4a85a5a2c9bc7c4fcc0f8222ca4229` |
| `coco_object/evaluation` | `c803905a2dbc7c8bce585775c8f66c040e2b69001cac43589528ebc2f606932b` |
| `voc21/evaluation` | `00bebfc519cdf43fba55d3d50ad4c5148d858b918a1bbaf362dd90a25d13d855` |
| `coco_cd/evaluation` | `dadfb779beebe5ea5d90576395e6a339132f063d7035f05bca314130e5fe86df` |
| `coco_cd_validation/evaluation` | `dcd8c365cf09b0d1f6d52d0dd26e6f8f4b65ee6d64a0f95f354c54182804ad69` |

## Historical Evidence

The starting historical index is
`/home/wwww/hzl/results/trace_parameter_provenance_20261029_0030/audit.json`
(335,451 bytes; SHA-256
`d5a6617ac78ad09992ae74df164afc2bba9f8f53979c26218b025dea7385e510`).
`docs/parameter-provenance-audit.md` supplies the scope and caveats.
The seven small manifests explicitly referenced by that index are read and
checked against its recorded source hashes and ordered IDs/indices. There is
no recursive result-root scan, large result payload read, model load, download,
or reopening of old predictions or annotations.

Historical root:
`/home/wwww/hzl/results/trace_v2_simplification_20260721_225456`.

| Relative manifest | Selected rows | Exact-byte SHA-256 |
| --- | ---: | --- |
| `calibration_splits/pc60.json` | 512 | `d05decc312ce65bf32536a33b61f1b5455e6deb5a331286d56b2a20312008585` |
| `calibration_splits/coco_object.json` | 512 | `79b51604245b16fb5493536feea02bc0e201d1e6f092ec36c12f44c23d820542` |
| `calibration_splits/coco_stuff.json` | 512 | `ce7f6f6d2389b303117b1cf490d15dbd4c8e4fd7d01c017fdf3a98a20ddf1777` |
| `calibration_splits/coco_cd.json` | 512 | `b49607826c23a48416e51e4140fe39a3f5bf2fc7c9099d9bfab106efca21d1af` |
| `calibration_splits/voc20.json` | 512 | `58942a414232187ec49ee67417816a132a3b572c92d92a10e171b44fd7212ac3` |
| `calibration_splits/voc21.json` | 512 | `b31c03ec85246c54d27eab4fe3ff2538cb52dc8fbc00739ae07cd65dbc559679` |
| `calibration_splits_fast128/voc20.json` | 128 | `cf3951b56532137f941a31d3ed62a0ba8264bc04be1c915fe8131f1e98d87cf5` |

All declare training-source selection. Original path strings, historical
indices, mapped official IDs and source hashes are preserved in the bundle.
Indices alone are never projected onto today's filesystem ordering. Missing
or unmappable explicit IDs, a hash mismatch, disagreement with the audit,
or an index-only manifest stops preparation before a bundle is written.
`--history-manifest` can add further explicitly named small manifests with a
supported `dataset_key`, `split=train`, and explicit `sample_paths` or
`image_ids`; all supplied exclusions must map, not just those hitting a target.

Coverage is **partial known history**, not a complete never-seen certificate:

- The July subset builder inspected label contents in larger candidate pools
  (PC60/Object/Stuff declare 4,096 candidates), whereas the index retains the
  selected 512 IDs. Exact candidate-pool mappings are not established here.
  Excluding selected subsets does not prove absence of earlier label inspection
  or of unindexed image-level calibration elsewhere.
- Pre-v2/global exposure, other result roots, interrupted/unindexed attempts,
  and missing historical canonical-loader mappings remain unknown.
- Small A1338/Ccli2312 controller sources use evaluation panels, not new
  calibration panels. Their evaluation images are excluded by source identity;
  their enormous manifests/payloads were not reopened or recertified.
- Source paths, official IDs, symlinks and physical file identities are checked.
  No all-population RGB-content duplicate audit is claimed. Selected RGB hashes
  identify bytes now, not immutable historical content.

Therefore `audit.global_never_seen_certified=false` and
`audit.historical_index_exhaustive=false`, and
`audit.strong_independent_confirmation_supported=false`. The supported claim is only
disjointness from the enumerated official evaluation images and the explicitly
indexed selected historical calibration images, under this prospective
protocol. Unknown exposure must not be reworded as proven absence.

### Candidate-Pool Reconstruction Check

The July `SHA256SUMS` points to the **17,388-byte**
`_snapshot/mlmp_trace_sources.tar.gz`, which contains the archived dataset
definitions and loader. No archive was extracted to disk or imported.
The exact-byte evidence is:

| Source | SHA-256 |
| --- | --- |
| Live July builder, matched to starting audit | `98c79da331520cefaf346982fe081c40ec947f233e7698bdf069dd0cf345548b` |
| `mlmp_trace_sources.tar.gz` | `7ab2d93b9e065eed58b009c34a6dfa53a3e067821dec6ac10f258d1cfb4aca33` |
| Archived `utils/segmentation_datasets.py` | `b3aae1bb18c846a74f828edde1d8599a87ce0d5552b60019d35839fbd1e09af3` |
| Archived `utils/simple_seg_dataset.py` | `9b562606fd21fbc33ce58cbdad39dc215200b098b8795c06af7b6d8f856f0bc8` |

The archived base loader sorts `data_list` by image path. That alone is
insufficient. Its `prepare_data` has **no `data_split` parameter**, while the
inspected builder passes `data_split="train"`. Archived definitions still name
validation paths, and the root source archive lacks the canonical
`cfc_tta/configs/trace_v2.py` imported by the builder. Thus the archived loader
does not establish the exact training branch that generated the saved
manifests. The builder saves selected indices/paths but **not candidate-pool
indices/paths**. No exact, archived full training-ID list was established by
these bounded pointers. Today's sorted list plus a seed must not be promoted
to a reconstructed historical pool.

**PC60 unused capacity is not certified.** Declared population 4,998 minus the
label-inspected pool of 4,096 leaves at most 902 images outside that pool.
The current source lists place 400 PC60 training images in VOC training and
383 in VOC validation. Conservatively accounting for the entire VOC training
source, VOC evaluation, and PC60's known selected512 gives a fixed exclusion
union of 1,219 IDs. Conditional on the declared July pool containing those
selected512, the combinatorial unused-capacity bounds are:

```text
lower = max(0, 4998 - 4096 - (1219 - 512)) = 195
upper = min(4998 - 4096, 4998 - 1219)      = 902
```

These bounds ignore unindexed exposure and do not identify any pool-unexposed
IDs. They are not bounds on global never-seen images. Since the lower bound is
below 384 and the actual pool identities remain unresolved, the bounded audit
does **not** establish a selectable 128+256 truly unused PC60 cohort. The
partial-history cohort remains available with its explicit weaker scope; the
sizes and seed have not been changed to manufacture an independence claim.

### Recent Named Calibration Roots

The first bounded pass read only the named small `README.md`, `manifest.json`
and `queue_state.json` records below. The follow-up explicitly authorized the
two referenced input JSONs as metadata under a **16 MiB / 16,777,216-byte
per-file cap**. Both are now read and verified against their exact
parent-declared hashes before parsing. There is no recursive scan, no
trial/result/request/prediction payload read and no annotation-byte read.

| Root suffix | Record | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `rebuttal_backbone_object_refine_recovery_20261026_1130` | `README.md` | 1,235 | `cc095bee89fd016ae603c68a9e38d5b50bee70f13a9ce1320a2fb2e8aadc60a7` |
| same | `manifest.json` | 10,813 | `26241f10a3ab165f36931b7466a6d2cd5d8011e1ad6c53d12a6be779c885ca6a` |
| same | `queue_state.json` | 63,824 | `3e5c460b4cc6a58e3fdab8c7e1a2ee6f1de421b19d64cfbdce04a965f4ae350f` |
| `rebuttal_backbone_stuff_refine_20261026_1307` | `README.md` | 1,300 | `63b6933a17c1703626e3dbc71196d11095239dce065f8c539ca852439970283d` |
| same | `manifest.json` | 12,754 | `1c21f2e1bc0e4f4939ad36011cfee45bd8af91193a064dd11136acc9b6e5925c` |
| same | `queue_state.json` | 56,694 | `ba6c7cacbc9c97be5ff9e5ceead097787f02f94d8d8076a49240bb85a236b209` |

Both parents declare 256 training calibration images and seed 2027. Their
`training_inputs={}` / `training_inputs_source=null` fields do not contain the
IDs, but the referenced input JSONs do:

| Referenced input | Verified bytes | Parent-declared and verified SHA-256 |
| --- | ---: | --- |
| Object root: `inputs/coco_object.json` | 9,075,459 | `9566fe4036bb12f364bbd6269a9ac52e5b825064985ba23db14dec1cb678dcbb` |
| Stuff root: `inputs/coco_stuff.json` | 9,065,061 | `964f86ddfa0e5917ba619287ce29b13e703b6134c3a64323dec99b1c84854493` |

Each contains an explicit ordered `train_ids` array of 118,287 images,
`calibration_indices` of length 256, `calibration` of length 256 and
`evaluation` of length 5,000. The parser verifies the dataset/parent seed,
declared counts, official train2017/val2017 specs, uniqueness, full training
and evaluation identity-set matches to the enumerated source, and own
train/evaluation disjointness. Each explicit calibration `image_id`, `stem`
and label filename must agree with **that archived array's**
`train_ids[calibration_indices[i]]`. The current filesystem order is never
used to reconstruct a historical selection. Unknown/mismatched IDs, incomplete
arrays, invalid indices, source mismatches, absent hashes, hash mismatches and
oversized inputs all fail before cohort output.

The Object and Stuff panels have **identical ordered 256 image IDs and
archived indices**, despite different annotation suffixes. Their union has
256 identities, not 512. Identity equality is not a label-content equivalence
claim. Both panels are automatically normalized into `history_manifests`
and unioned into exclusions; no caller-supplied extraction is needed for these
two known roots. Generic additional small manifests remain supported through
`extra_history` / `--history-manifest`.

Shared canonical-JSON fingerprints:

| Array / mapping | SHA-256 |
| --- | --- |
| Ordered calibration official IDs | `2a979fcdf07e85c5fd35818e361916265b8ef07e54d608e3de6aec158e4f309b` |
| Archived calibration indices | `6800fec59709f0ea5c8730197f35a71e76d67c5ed24bdf151c9b3dadb5dd6b91` |
| Ordered `{archived_train_index,official_id}` mapping | `8b8d840661184356124b8e8ada9d96e617fa677a3f817e0910c72932ab25748b` |
| Archived full training path array | `2bffed0093f2cfb01c1ae3c241fe2fb16462214792229652ac8c85d02f99d994` |
| Archived full training official-ID array | `99cd330579becb5dbfc34d9366ff228dcbc1beda9cecca93657d2916895fb910` |
| Archived full evaluation official-ID array | `d7a6cf336dba697de84529c51615efc32a6b5bfb586c439c315441a176222a88` |

Exact intersections of the 256-image recent union with the previous protocol:

| Intersection | PC60 | COCO Object |
| --- | ---: | ---: |
| Target training source | 0 | 256 |
| Previous mapped historical exclusions | 0 | 4 |
| Enumerated evaluation exclusions | 0 | 2 |
| Previous exclusion union | 0 | 6 |
| Newly excluded | 0 | 250 |
| Previous calibration128 | 0 | 0 |
| Previous confirmation256 | 0 | 0 |

Previous historical intersections are `coco:000000069747`,
`coco:000000102632`, `coco:000000354537`, `coco:000000477807`.
Evaluation intersections are `coco:000000237944` and `coco:000000483867`.
Their exact source attribution is:

| Existing exclusion source | Recent IDs already excluded there |
| --- | --- |
| July `calibration_splits/coco_object.json` | `coco:000000102632` |
| July `calibration_splits/coco_stuff.json` | `coco:000000069747` |
| July `calibration_splits/coco_cd.json` | `coco:000000354537`, `coco:000000477807` |
| COCO-CD `test_balanced` | `coco:000000237944`, `coco:000000483867` |
| COCO-CD `val_balanced` | None |

These sets are disjoint. The exact new-exclusion set is the recent union
minus these six IDs; exact arrays and old-cohort intersections are retained
under `audit.recent_history_intersections`, including the per-source
attribution arrays.

The exact shared mapping is listed as `archived_train_index:COCO_ID`; prepend
`coco:` to obtain each official ID. For example, index 29 maps to image
`/home/wwww/hzl/data/coco/train2017/000000000144.jpg`. Its archived Object label
is `/home/wwww/hzl/data/coco_mlmp/annotations/train2017/000000000144_instanceTrainIds.png`;
the Stuff label uses `coco_stuff/annotations/train2017` and
`_labelTrainIds.png`. All original image/label paths and archived positions
are retained in each normalized history's `mappings` array.

<!-- recent-index-identity-map -->
```text
29:000000000144 327:000000001579 970:000000005083 1526:000000007819 2823:000000014203 2855:000000014352 3666:000000018267 4199:000000020779
4204:000000020795 4334:000000021417 4357:000000021534 4444:000000022002 5916:000000028870 6302:000000030699 6460:000000031561 6532:000000031913
7596:000000037009 7624:000000037125 8299:000000040583 9009:000000044045 9058:000000044286 9260:000000045188 9461:000000046149 10237:000000050061
10694:000000052312 10808:000000052886 10969:000000053670 12218:000000059798 13793:000000067558 14106:000000069048 14241:000000069747 14555:000000071347
14574:000000071466 14663:000000071914 16298:000000080091 16314:000000080165 18836:000000092646 19632:000000096473 19733:000000096975 20426:000000100303
20599:000000101172 20836:000000102349 20891:000000102632 21793:000000106823 22892:000000112289 24168:000000118679 24242:000000118964 24265:000000119061
24317:000000119320 24642:000000120994 25458:000000125106 25829:000000126826 25830:000000126832 26153:000000128309 26700:000000130974 27055:000000132746
27186:000000133412 27197:000000133476 27441:000000134643 29221:000000143275 29428:000000144250 29871:000000146469 30511:000000149467 30768:000000150924
31238:000000153205 32070:000000157397 32547:000000159771 32609:000000160102 32880:000000161304 34054:000000167359 34426:000000168993 34556:000000169638
35204:000000173161 35413:000000174211 36336:000000178801 37613:000000185168 38538:000000189645 38732:000000190566 39047:000000192095 40135:000000197585
40409:000000198993 40512:000000199555 40816:000000201123 41627:000000205230 42525:000000209468 43808:000000215781 45020:000000221883 46020:000000226713
46694:000000230094 46776:000000230507 46795:000000230598 48273:000000237798 48307:000000237944 48708:000000240022 49065:000000241844 49151:000000242248
49490:000000243980 49523:000000244157 49541:000000244240 49665:000000244862 50094:000000246928 50120:000000247050 50935:000000251211 51157:000000252340
51375:000000253432 51444:000000253730 52229:000000257443 53483:000000263664 55356:000000273123 55463:000000273710 55674:000000274758 55920:000000275997
56441:000000278569 57013:000000281227 57175:000000281977 57480:000000283441 57510:000000283600 57747:000000284684 58061:000000286149 58578:000000288712
59182:000000291841 59321:000000292505 59598:000000293793 59738:000000294437 59840:000000294942 60443:000000297802 61241:000000301764 61424:000000302643
61525:000000303227 62117:000000306140 62494:000000307990 63224:000000311448 63569:000000313220 64159:000000315960 64825:000000319266 65492:000000322567
65595:000000323093 66665:000000328181 66759:000000328663 67175:000000330650 67185:000000330694 67545:000000332463 67559:000000332544 68078:000000334850
68373:000000336166 68792:000000338104 68956:000000338991 69012:000000339290 70674:000000347620 71370:000000350987 71604:000000352182 71902:000000353624
72090:000000354537 72731:000000357641 72839:000000358103 72920:000000358489 73128:000000359451 73522:000000361359 73582:000000361603 74169:000000364455
76175:000000374302 76303:000000374994 76445:000000375759 76897:000000378029 76902:000000378049 76949:000000378302 76995:000000378548 78045:000000383652
78118:000000384015 78156:000000384173 79320:000000389811 79387:000000390232 79798:000000392222 79854:000000392585 81100:000000398722 81215:000000399274
81419:000000400235 82330:000000404428 82427:000000404886 82785:000000406591 82945:000000407355 84132:000000413248 84142:000000413295 84149:000000413334
84586:000000415659 84727:000000416338 85193:000000418711 85302:000000419245 86096:000000423107 86437:000000424823 86880:000000426996 86964:000000427399
87146:000000428301 87593:000000430652 88829:000000436649 88836:000000436685 90409:000000444496 90448:000000444703 90718:000000446113 91625:000000450724
91746:000000451305 92154:000000453334 93950:000000461998 94418:000000464443 95373:000000469092 96362:000000473919 96545:000000474868 96559:000000474970
96648:000000475475 97134:000000477807 97470:000000479469 97851:000000481239 98287:000000483382 98375:000000483867 98500:000000484414 100065:000000492151
100087:000000492246 100143:000000492472 101137:000000497261 101627:000000499784 103230:000000507871 103932:000000511406 104146:000000512495 106390:000000523571
107436:000000528944 107778:000000530683 108545:000000534292 108629:000000534722 108972:000000536403 109191:000000537520 109360:000000538372 109580:000000539476
110168:000000542338 110280:000000542910 110471:000000543754 110608:000000544472 110708:000000545033 111084:000000546966 111208:000000547594 111314:000000548183
112256:000000552832 112546:000000554176 112976:000000556306 113335:000000558015 114483:000000563444 114552:000000563771 114723:000000564570 114899:000000565365
115096:000000566357 116629:000000573981 116802:000000574760 117035:000000575834 117275:000000576941 117683:000000578813 117774:000000579232 118018:000000580586
```
<!-- /recent-index-identity-map -->

The bundle's `audit.recent_history_audit` now marks both inputs hash-verified
and mapped. This resolves the two specifically named recent gaps, not all
historical exposure. The July 4,096-image label-inspected pools remain
unreconstructed; all partial-history and negative certification flags remain.

## Bundle Contract

`prepare_cohorts(package_root, data_root, *, audit_path, data_dirs,
extra_history, require_independent=False)` returns the bundle without any output writes. `data_dirs` is a
dictionary of explicit dataset roots; omitted roots follow repository configs.

```text
datasets[dataset].calibration: 128 rows
datasets[dataset].confirmation: 256 rows
row = {
  official_id: string,
  image: canonical absolute RGB path,
  label: canonical absolute annotation path,
  image_sha256: exact RGB-file SHA-256,
  label_metadata: {size_bytes, mtime_ns, device, inode, contents_read: false}
}
```

`datasets[dataset]` additionally has exact `counts`, rank-ordered `eligible_ids`,
and `selected_ids_sha256`. Top-level `exclusions` records official-evaluation,
known-history and union IDs per target. `source_hashes` records exact bytes and
SHA-256 for the bounded input records and implementation/specification sources.
`source_id_manifests` contains every discovered source ID/image/label triple,
its population count and canonical-JSON digest. `history_manifests` retains
all historical path/ID mappings. `audit` records directory binding, scope,
unknowns and explicit negative certification flags. A manifest hash is not a
prediction seal. Metadata is not a label-content hash.

The CLI requires a fresh JSON path, not a result directory. Example syntax,
not executed during this task:

```bash
python -B tools/evolution_confirmation/cohorts.py \
  --data-root /home/wwww/hzl/data \
  --data-dir coco_object=/home/wwww/hzl/data/coco_mlmp \
  --output /existing/explicit/directory/fresh-cohorts.json
```

An existing file, existing directory, dangling output symlink or absent parent
is rejected. No resume/overwrite mode exists. Full source enumeration and all
mapping checks finish before output creation. A write interrupted after
exclusive creation may leave an incomplete file; treat it as invalid, do not
resume it. Verify JSON completeness and record the final bundle byte hash
before prediction work.

`--require-independent` (API: `require_independent=True`) explicitly fails
closed before dataset discovery/output creation because the historical pools
remain unresolved. The default emits only the
partial-history prospective protocol with all negative certification flags.

## Validation and Boundaries

Synthetic filesystem tests cover deterministic ranking, default sizes,
cross-evaluation/history unions, own train/eval overlap, repeated basename IDs,
symlink/hardlink aliases, stale-root official-ID mapping, unknown or ambiguous
identities, incomplete populations, bounded reads, source-hash disagreement,
index-only manifests, fresh-output behavior and the worker row schema. A label
read sentinel forbids opening annotation files. A fresh subprocess blocks
torch/torchvision/CUDA/numpy/PIL/TRACE imports while importing the module,
discovering paths, hashing selected RGB bytes and reading repository specs.

Validation command:

```bash
python -B -m pytest -q -p no:cacheprovider tests/test_confirmation_cohorts.py tests/test_suite_inputs.py
```

Current result: **77 passed**. Follow-up tests additionally cover
hash-bound 16 MiB metadata parsing, archived-index/explicit-row consistency,
source population/evaluation identities, duplicate IDs, wrong stems/mask
names, label-read sentinels and exact exclusion/old-cohort intersections.

Real-data verification was also run entirely in memory with blocked
model/image-library imports and a `.png` open sentinel. The two authorized
metadata files are now read under the 16 MiB bound, with parent hashes checked:
**768 selected RGB hashes**, zero label opens and zero output writes; measured
elapsed time **15.0 seconds** on this host, with **33 exact source/member
records hashed**. It reproduced all four selected-ID fingerprints above. All eight source
populations had complete nonempty file pairs and the expected counts. This is
metadata/preparation validation, not validation of annotation pixel semantics.

Only the new cohort module, its test file and this document are owned by this
task. No current paper, existing worker, manifest, result, baseline or sealed
artifact was modified. No GPU job, download, deletion or GitHub action was
performed. Cohort availability is established; scientific success is not.
