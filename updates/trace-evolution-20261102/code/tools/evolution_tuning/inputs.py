"""CPU-only, annotation-blind input preparation for the eight-dataset search.

``prepare_inputs`` returns a bundle with all eight entries under ``datasets``.
The controller owns persisting inputs.json in its fresh output directory.
This module never constructs training images, reads segmentation labels,
imports torch, or executes a model. Waterbirds defaults to blocked calibration;
explicit development-overlap opt-in reuses existing training images only.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import replace
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile

from tools.evolution_confirmation import cohorts


DATASETS = ("voc20", "voc21", "pc59", "pc60", "coco_object", "coco_stuff",
            "coco_cd", "waterbirds_seg")
EVALUATION_COUNTS = dict(voc20=1449, voc21=1449, pc59=5105, pc60=5105,
                         coco_object=5000, coco_stuff=5000, coco_cd=500,
                         waterbirds_seg=5322)
TRAIN_COUNTS = dict(voc20=1464, voc21=1464, pc59=4998, pc60=4998,
                    coco_object=118287, coco_stuff=118287, coco_cd=1668,
                    waterbirds_seg=2458)
INHERITED_CONFIG_ROOT = Path("/home/wwww/hzl/results/trace_image_full_20261029_0333/configs")
HISTORICAL_ROOT = Path("/home/wwww/hzl/results/trace_v2_simplification_20260721_225456")
METADATA_LIMIT = 4 * 1024 * 1024
InputError = cohorts.CohortError


def _read(path, sources, *, limit=METADATA_LIMIT):
    path = Path(path).resolve(strict=True)
    expected = sources.get(str(path), {}).get("sha256")
    if path.suffix != ".csv":
        return cohorts.read_metadata(path, sources, limit=limit, expected=expected)
    # Only the identity/split/background metadata CSV is an allowed addition
    # to the shared helper's textual metadata whitelist.
    if path.name != "metadata.csv" or path.stat().st_size > limit:
        raise InputError(f"Unsupported or oversized identity CSV: {path}")
    with path.open("rb") as handle:
        content = handle.read(limit + 1)
    digest = hashlib.sha256(content).hexdigest()
    if len(content) > limit or (expected is not None and digest != expected):
        raise InputError(f"Identity CSV exceeds bound or changed: {path}")
    sources[str(path)] = dict(sha256=digest, bytes=len(content))
    return content.decode("utf-8")


def _json(path, sources):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise InputError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(_read(path, sources), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(InputError(value)))


def _literal_sources(package_root, sources):
    trees = {name: ast.parse(_read(package_root / relative, sources)) for name, relative in (
        ("config", "trace_ovss/configs/trace.py"),
        ("dataset", "trace_ovss/data/datasets.py"),
        ("vocabulary", "trace_ovss/data/vocabulary.py"))}
    defaults = {}
    declaration = cohorts._assignment(trees["config"], "BEST_CONFIGS")
    for key, value in zip(declaration.keys, declaration.values):
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name) or value.func.id != "TraceBenchmarkConfig":
            raise InputError("Unsupported configuration declaration")
        defaults[ast.literal_eval(key)] = {kw.arg: ast.literal_eval(kw.value) for kw in value.keywords}
    specs = {}
    declaration = cohorts._assignment(trees["dataset"], "DATASET_SPECS")
    for key, value in zip(declaration.keys, declaration.values):
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name) or value.func.id != "DatasetSpec":
            raise InputError("Unsupported dataset declaration")
        if value.keywords:
            raise InputError("Audit dataset keyword declarations before using them")
        args = [ast.literal_eval(arg) for arg in value.args]
        if not 4 <= len(args) <= 6:
            raise InputError("Unexpected dataset specification arity")
        specs[ast.literal_eval(key)] = args + ([None, False] if len(args) == 4 else [False] * (6 - len(args)))
    vocabulary = ast.literal_eval(cohorts._assignment(trees["vocabulary"], "VOCABULARY_FILES"))
    background_node = cohorts._assignment(trees["vocabulary"], "BACKGROUND_FIRST")
    if not isinstance(background_node, ast.Call) or not isinstance(background_node.func, ast.Name) or background_node.func.id != "frozenset":
        raise InputError("Unsupported background vocabulary declaration")
    backgrounds = ast.literal_eval(background_node.args[0])
    return defaults, specs, vocabulary, backgrounds


def _config(key, defaults, data_root, sources):
    config = json.loads(json.dumps(defaults[key]))
    config["trace_view_batch_size"] = 16
    if key in {"pc59", "waterbirds_seg"}:
        path = HISTORICAL_ROOT / "frozen_configs_final" / key / "best_config.json"
        record = _json(path, sources)
        inherited = record["config"]
        # Carry only inherited optimizer parameters, never the historical method.
        for name in ("batch_size", "lr", "steps", "trace_max_regions"):
            config[name] = inherited[name]
        if inherited["dataset"] != config["dataset"] or inherited["class_extensions"] != config["class_extensions"]:
            raise InputError(f"Historical vocabulary/dataset drift: {key}")
        if inherited["resize"] != [224, 224] or inherited["patch"] != [224, 224] or inherited["stride"] != 112:
            raise InputError(f"Historical geometry drift: {key}")
        provenance = dict(path=str(path.resolve()), kind="historical_best_optimizer_tuple",
                          historical_view_batch_size=inherited["trace_view_batch_size"],
                          historical_no_selection_flag_is_not_global_exposure_certificate=True)
    else:
        path = INHERITED_CONFIG_ROOT / f"{key}.json"
        inherited = _json(path, sources)
        if set(inherited) != set(config) | {"data_dir"}:
            raise InputError(f"Inherited resolved config schema differs: {key}")
        config.update(inherited)
        provenance = dict(path=str(path.resolve()), kind="selected_C_six_dataset_inherited_config")
    # data_subdir in the six-dataset Object config is stale; the explicit root
    # is coco_mlmp, not coco_stuff. Rebase that same source under data_root.
    subdir = "coco_mlmp" if key == "coco_object" else defaults[key]["data_subdir"]
    config["data_dir"] = str((data_root / subdir).resolve(strict=True))
    if config["dataset"] != defaults[key]["dataset"] or type(config["class_extensions"]) is not bool or config["class_extensions"] != defaults[key]["class_extensions"]:
        raise InputError(f"Inherited dataset/extension drift: {key}")
    if any(config[name] != [224, 224] for name in ("resize", "patch_size")) or config["patch_stride"] != 112 or config["trace_view_batch_size"] != 16:
        raise InputError(f"Fixed geometry/views changed: {key}")
    for name in ("batch_size", "steps", "trace_max_regions", "patch_stride", "trace_view_batch_size"):
        if type(config[name]) is not int or config[name] < 1:
            raise InputError(f"Invalid inherited parameter: {key}/{name}")
    if not isinstance(config["lr"], (int, float)) or isinstance(config["lr"], bool) or not math.isfinite(config["lr"]) or config["lr"] <= 0:
        raise InputError(f"Invalid inherited learning rate: {key}")
    provenance["sha256"] = sources[str(path.resolve())]["sha256"]
    provenance["inherited_data_root"] = inherited["data_dir"]
    provenance["resolved_data_root"] = config["data_dir"]
    return config, provenance


def _waterbird_identity(stem):
    cohorts._safe_stem(stem)
    if len(Path(stem).parts) != 2:
        raise InputError(f"Unknown CUB relative identity: {stem}")
    return f"cub:{stem}"


def discover(layout, sources, *, expected_count):
    """Mirror the complete original loader order; labels are stat-only."""
    if layout.dataset != "waterbirds_seg":
        split_digest = None
        if layout.split_file:
            split = (layout.root / layout.split_file).resolve(strict=True)
            _read(split, sources)
            split_digest = sources[str(split)]["sha256"]
        rows = cohorts.discover_population(layout, sources, expected_count=expected_count)
        if split_digest is not None and sources[str(split)]["sha256"] != split_digest:
            raise InputError(f"Split changed during discovery: {split}")
    else:
        # The shared discovery helper intentionally supports only VOC/COCO IDs.
        image_root, mask_root = layout.root / layout.image_dir, layout.root / layout.mask_dir
        if not image_root.is_dir() or not mask_root.is_dir():
            raise InputError(f"Missing Waterbirds image/mask directories: {layout.root}")
        if layout.split_file:
            stems = [line.strip() for line in _read(layout.root / layout.split_file, sources).splitlines() if line.strip()]
        else:
            stems = [p.relative_to(image_root).as_posix()[:-4] for p in sorted(image_root.rglob("*.jpg"))]
        if len(stems) != expected_count or len(stems) != len(set(stems)):
            raise InputError(f"Incomplete or duplicated Waterbirds population: expected {expected_count}, found {len(stems)}")
        rows = []
        for stem in stems:
            identity = _waterbird_identity(stem)
            image, label = image_root / f"{stem}.jpg", mask_root / f"{stem}.png"
            if not image.is_file() or not label.is_file():
                raise InputError(f"Missing Waterbirds pair: {image}, {label}")
            canonical, canonical_label = image.resolve(strict=True), label.resolve(strict=True)
            if canonical.suffix != ".jpg" or canonical_label.suffix != ".png":
                raise InputError("Waterbirds image/label alias changes file type")
            if canonical.relative_to(image_root.resolve()).with_suffix("").as_posix() != stem:
                raise InputError("Waterbirds alias changes CUB identity")
            im, lab = canonical.stat(), canonical_label.stat()
            if not im.st_size or not lab.st_size:
                raise InputError("Empty Waterbirds pair")
            rows.append(dict(official_id=identity, image=str(canonical), label=str(canonical_label),
                             source_image=str(image.absolute()), source_label=str(label.absolute()),
                             image_metadata=dict(size_bytes=im.st_size, mtime_ns=im.st_mtime_ns,
                                                 device=im.st_dev, inode=im.st_ino),
                             label_metadata=dict(size_bytes=lab.st_size, mtime_ns=lab.st_mtime_ns,
                                                 device=lab.st_dev, inode=lab.st_ino, contents_read=False)))
    # Sort by declared source path, as SegmentationPatchDataset does, not by
    # resolved symlink destination or selection order in the split file.
    rows.sort(key=lambda row: row["source_image"])
    cohorts.identity_index({"population": rows})
    if len({r["image"] for r in rows}) != len(rows) or len({(r["image_metadata"]["device"], r["image_metadata"]["inode"]) for r in rows}) != len(rows):
        raise InputError("Duplicate physical image in population")
    return rows


def select_calibration(dataset, training, evaluation, *, size=256, seed=2027):
    """Own-evaluation source/path/inode overlap fails closed, never filters full."""
    if type(size) is not int or size < 1 or type(seed) is not int or seed < 0:
        raise InputError("Calibration size must be positive and seed nonnegative integers")
    cohorts.identity_index({"training": training, "evaluation": evaluation})
    full_tokens = {token for row in evaluation for token in cohorts._tokens(row)}
    overlap = [row["official_id"] for row in training if cohorts._tokens(row) & full_tokens]
    if overlap:
        raise InputError(f"{dataset}: own train/eval overlap ({len(overlap)} source IDs); full population must not be reduced")
    if len(training) < size:
        raise InputError(f"{dataset}: requires {size} training images, found {len(training)}")
    return _ranked_calibration(dataset, training, size=size, seed=seed)


def _ranked_calibration(dataset, training, *, size, seed):
    def rank(row):
        value = f"trace-single-probe-tuning-v1\n{seed}\n{dataset}\n{row['official_id']}\n"
        return hashlib.sha256(value.encode("utf-8")).hexdigest(), row["official_id"]

    selected = sorted(training, key=rank)[:size]
    return sorted(selected, key=lambda row: row["source_image"])


def _select_waterbirds_development_overlap(training, evaluation, *, size, seed):
    """Permit declared overlap, never aliases or different physical inputs."""
    cohorts.identity_index({"training": training, "evaluation": evaluation})
    for rows in (training, evaluation):
        for field in ("id", "path", "inode"):
            tokens = [next(token for token in cohorts._tokens(row) if token[0] == field) for row in rows]
            if len(tokens) != len(set(tokens)):
                raise InputError("Waterbirds development overlap has duplicate source/path/inode aliases")
    full = {row["official_id"]: row for row in evaluation}
    for row in training:
        reference = full.get(row["official_id"])
        if reference is None or cohorts._tokens(row) != cohorts._tokens(reference):
            raise InputError("Waterbirds development training source/path/inode must match original full")
        if row["label"] != reference["label"] or row["label_metadata"] != reference["label_metadata"]:
            raise InputError("Waterbirds development training label metadata must match original full")
    if len(training) < size:
        raise InputError(f"Waterbirds development calibration requires {size} training images, found {len(training)}")
    selected = _ranked_calibration("waterbirds_seg", training, size=size, seed=seed)
    selected_ids = {row["official_id"] for row in selected}
    if len(selected_ids) != size or not selected_ids <= {row["official_id"] for row in training} & full.keys():
        raise InputError("Waterbirds development calibration must be entirely in train and original full")
    return selected


def _vocabulary(package_root, config, filename, backgrounds, sources):
    path = package_root / "trace_ovss/resources/class_names" / filename
    lines = [line.strip() for line in _read(path, sources).splitlines() if line.strip()]
    groups = [[word.strip() for word in line.split(",")] for line in lines]
    if not groups or any(not word for group in groups for word in group):
        raise InputError("Empty vocabulary/class query")
    classes = [group[0] for group in groups]
    if config["dataset"] in backgrounds:
        classes[0] = "background"
    owners = {}
    for index, group in enumerate(groups):
        for word in group:
            if word.casefold() in owners and owners[word.casefold()] != index:
                raise InputError(f"Ambiguous text query: {word}")
            owners[word.casefold()] = index
    if config["class_extensions"]:
        queries = [word for group in groups for word in group]
        mapping = [index for index, group in enumerate(groups) for _ in group]
    else:
        queries, mapping = classes, None
    return dict(classes=classes, queries=queries, query_to_class=mapping,
                vocabulary_sha256=cohorts.value_hash([queries, mapping]))


def _freeze_rows(rows, hash_cache):
    frozen = []
    for row in rows:
        key = (row["image"], *(row["image_metadata"][name] for name in ("size_bytes", "mtime_ns", "device", "inode")))
        if key not in hash_cache:
            hash_cache[key] = cohorts._selected_row(row)["image_sha256"]
        else:
            im, lab = Path(row["image"]).stat(), Path(row["label"]).stat()
            for stat, metadata in ((im, row["image_metadata"]), (lab, row["label_metadata"])):
                if (stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino) != (metadata["size_bytes"], metadata["mtime_ns"], metadata["device"], metadata["inode"]):
                    raise InputError("Input metadata changed while preparing panels")
        metadata = row["label_metadata"]
        frozen.append({name: row[name] for name in ("official_id", "image", "label")} |
                      dict(image_sha256=hash_cache[key],
                           label_metadata=[metadata["size_bytes"], metadata["mtime_ns"]],
                           label_metadata_details=dict(metadata)))
    return frozen


def _archive_audit(sources):
    """Read only the two bounded source archives, never histories/predictions."""
    snapshot = HISTORICAL_ROOT / "_snapshot"
    checksums = {line.split(None, 1)[1].strip(): line.split(None, 1)[0]
                 for line in _read(snapshot / "SHA256SUMS", sources).splitlines() if line.strip()}
    audit = {}
    for name in ("root_method_and_experiments.tar.gz", "mlmp_trace_sources.tar.gz"):
        path = snapshot / name
        size = path.stat().st_size
        if size > 1024 * 1024:
            raise InputError(f"Archived source exceeds bounded read: {path}")
        with path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != size or digest != checksums.get(str(path)):
            raise InputError(f"Archived source checksum mismatch: {path}")
        sources[str(path.resolve())] = dict(sha256=digest, bytes=size)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > 256:
                raise InputError("Unexpected archived source inventory size")
            names = [member.name for member in members if member.isfile()]
            inspected = {}
            for member in members:
                if member.name not in {"experiments/prepare_mlmp_data.py",
                                       "experiments/build_trace_v2_calibration_splits.py",
                                       "external/MLMP/utils/segmentation_datasets.py",
                                       "external/MLMP/utils/simple_seg_dataset.py"}:
                    continue
                if not member.isfile() or member.size > 131072:
                    raise InputError("Oversized archived builder/loader member")
                content = archive.extractfile(member).read(131073)
                if len(content) != member.size:
                    raise InputError("Incomplete archived source member")
                inspected[member.name] = hashlib.sha256(content).hexdigest()
            audit[name] = dict(path=str(path), sha256=digest, members=names,
                               inspected_member_sha256=inspected,
                               calibration_builder_present="experiments/build_trace_v2_calibration_splits.py" in names)
    return audit


def _waterbirds_audit(data_root, layout, training, evaluation, sources, archives):
    rows = list(csv.DictReader(io.StringIO(_read(layout.root / "images/metadata.csv", sources))))
    names = [str(Path(row["img_filename"]).with_suffix("")) for row in rows]
    if len(names) != len(set(names)) or {_waterbird_identity(name) for name in names} != {r["official_id"] for r in evaluation}:
        raise InputError("Waterbirds metadata does not equal original full loader")
    declared_train = {_waterbird_identity(str(Path(row["img_filename"]).with_suffix(""))) for row in rows if row["split"] == "0"}
    if declared_train != {r["official_id"] for r in training}:
        raise InputError("Waterbirds train.txt disagrees with metadata split=0")
    cub = data_root / "waterbirds_sources/CUB_200_2011"
    remaining = None
    if (cub / "images.txt").is_file() and (cub / "train_test_split.txt").is_file():
        image_names = dict(line.split(maxsplit=1) for line in _read(cub / "images.txt", sources).splitlines() if line.strip())
        split_rows = [line.split() for line in _read(cub / "train_test_split.txt", sources).splitlines() if line.strip()]
        if len(image_names) != len(split_rows) or len({row[0] for row in split_rows}) != len(split_rows):
            raise InputError("Invalid CUB source identity metadata")
        cub_train = {_waterbird_identity(str(Path(image_names[index]).with_suffix(""))) for index, flag in split_rows if flag == "1"}
        remaining = sorted(cub_train - {r["official_id"] for r in evaluation})
    historical = _json(HISTORICAL_ROOT / "calibration_splits/waterbirds_seg.json", sources)
    historical_ids = {_waterbird_identity(Path(value).relative_to(layout.root / "images").with_suffix("").as_posix()) for value in historical["sample_paths"]}
    historical_overlap = historical_ids & {r["official_id"] for r in evaluation}
    builder = HISTORICAL_ROOT.parents[1] / "experiments/build_trace_v2_calibration_splits.py"
    builder_available = builder.is_file()
    if builder_available:
        _read(builder, sources, limit=131072)
    return dict(original_full_count=len(evaluation), declared_train_count=len(training),
                train_full_source_id_overlap=len(declared_train & {r["official_id"] for r in evaluation}),
                historical_selected_count=len(historical_ids), historical_selected_full_overlap=len(historical_overlap),
                historical_manifest=str(HISTORICAL_ROOT / "calibration_splits/waterbirds_seg.json"),
                remaining_cub_train_count=None if remaining is None else len(remaining),
                remaining_cub_train_ids_sha256=None if remaining is None else cohorts.value_hash(remaining),
                existing_places_backgrounds=len({row["place_filename"] for row in rows}),
                archived_source_audit=archives, calibration_subset_builder=str(builder) if builder_available else None,
                construction_performed=False, segmentation_label_contents_read=False,
                reason="All existing Waterbirds training IDs overlap original full. Remaining CUB training IDs are not materialized Waterbirds examples. No exact provenance-justified original CUB/Places compositing builder is available in the bounded archived source inventory; the existing calibration builder only selects existing training examples and reads their masks.",
                feasible_choice="Recover the original compositor and its class/background mapping, crop/resize, mask threshold and RNG protocol, then explicitly construct a fresh training-only calibration dataset using CUB AND Places source IDs outside original full, recording source hashes and construction provenance. Until then block Waterbirds training, retain all 5322 full examples, and proceed with the other seven datasets. Do not shrink full or claim never-seen data.")


def prepare_inputs(package_root, data_root, output, calibration_size=256, seed=2027,
                   allow_waterbirds_development_overlap=False):
    """Return ``{datasets: {dataset: {config, calibration, evaluation}}, ...}``.

    ``output`` is the caller's fresh suite root (possibly already containing
    queue_state.json); inputs.json must not exist. The controller writes that
    file. This function performs no writes. Waterbirds defaults to
    calibration=None and blocked_reason, with its full panel complete. A strict
    bool opt-in admits only its existing 256-image seed-2027 training panel as
    development-overlap calibration, never an independent test. No construction
    is performed. An input/provenance failure raises InputError.
    """
    if type(allow_waterbirds_development_overlap) is not bool:
        raise InputError("allow_waterbirds_development_overlap must be a strict bool")
    if type(calibration_size) is not int or calibration_size < 1 or type(seed) is not int or seed < 0:
        raise InputError("Calibration size must be positive and seed nonnegative integers")
    if allow_waterbirds_development_overlap and (calibration_size != 256 or seed != 2027):
        raise InputError("Waterbirds development overlap requires calibration_size=256 and seed=2027")
    package_root, data_root = Path(package_root).expanduser().resolve(strict=True), Path(data_root).expanduser().resolve(strict=True)
    output = Path(output).expanduser().absolute()
    if output.is_symlink() or (output.exists() and not output.is_dir()) or (output / "inputs.json").exists():
        raise FileExistsError(f"Fresh input output directory required: {output}")
    if not output.parent.is_dir():
        raise InputError("Input output parent must already exist")
    if output.resolve().is_relative_to(data_root):
        raise InputError("Input outputs must be outside official data_root")
    sources = {}
    defaults, specs, files, backgrounds = _literal_sources(package_root, sources)
    for relative in ("tools/evolution_tuning/inputs.py", "tools/evolution_confirmation/cohorts.py",
                     "tools/evolution_confirmation/run.py", "docs/single-probe-tuning-contract.md",
                     "trace_ovss/resources/prompts.txt"):
        _read(package_root / relative, sources)
    archives = _archive_audit(sources)
    datasets, populations, cache = {}, {}, {}
    for key in DATASETS:
        config, provenance = _config(key, defaults, data_root, sources)
        args = specs[config["dataset"]]
        family = "cub" if key == "waterbirds_seg" else "voc" if key.startswith(("voc", "pc")) else "coco"
        layout = cohorts.Layout(key, family, Path(config["data_dir"]), *args[:5])
        full = discover(layout, sources, expected_count=EVALUATION_COUNTS[key])
        if key == "coco_cd":
            train_layout = replace(layout, mask_dir="masks/train_corr0.95", split_file="splits/train_corr0.95.txt")
        elif key == "waterbirds_seg":
            train_layout = replace(layout, split_file="splits/train.txt")
        else:
            train_layout = cohorts.training_layout(layout)
        train = discover(train_layout, sources, expected_count=TRAIN_COUNTS[key])
        populations[f"{key}/full"], populations[f"{key}/train"] = full, train
        # Guard image/label physical aliases across all discovered populations
        # before opening any RGB for hashing.
        vocab = _vocabulary(package_root, config, files[config["dataset"]], backgrounds, sources)
        selected, blocked = [], None
        development = None
        if key == "waterbirds_seg":
            audit = _waterbirds_audit(data_root, layout, train, full, sources, archives)
            if allow_waterbirds_development_overlap:
                selected = _select_waterbirds_development_overlap(train, full, size=calibration_size, seed=seed)
                development = dict(
                    protocol="development_overlap", development_overlap=True,
                    allow_waterbirds_development_overlap=True,
                    cal_full_source_id_overlap=len(selected),
                    train_full_source_id_overlap=len(train),
                    declared_train_count=len(train), original_full_count=len(full),
                    independent_test_claim=False, global_never_seen_certified=False,
                    evaluation_scope="development_overlap",
                    claim_boundary="Development evaluation: selected training sources overlap original full; no independent-test or globally never-seen claim.")
                audit.update(development)
                audit["reason"] = "Explicit opt-in reuses existing Waterbirds training images for development-overlap calibration; original full remains development evaluation."
                audit["feasible_choice"] = "Use only the existing training panel and unmodified original full under the declared development-overlap protocol; no source generation or independent-test claim."
            else:
                blocked = audit["reason"]
        else:
            selected = select_calibration(key, train, full, size=calibration_size, seed=seed)
            audit = dict(train_population=len(train), own_evaluation_population=len(full),
                         own_train_eval_source_id_overlap=0, own_train_eval_path_inode_overlap=0)
        audit["training_ordered_source_ids_sha256"] = cohorts.value_hash([row["official_id"] for row in train])
        audit["evaluation_ordered_source_ids_sha256"] = cohorts.value_hash([row["official_id"] for row in full])
        datasets[key] = dict(config=config, config_provenance=provenance,
                             calibration=dict(**vocab, reduce_zero_label=bool(args[5]), inputs=selected,
                                              available=blocked is None, block_reason=blocked, full_split=False,
                                              selection_seed=seed, requested_images=calibration_size,
                                              available_images=len(train), stage="calibration"),
                             evaluation=dict(**vocab, reduce_zero_label=bool(args[5]), inputs=full,
                                             available=True, full_split=True, stage="full"),
                             input_audit=audit)
        if development is not None:
            for name in ("calibration", "evaluation"):
                datasets[key][name].update(development)
    cohorts.identity_index(populations)
    for value in datasets.values():
        for name in ("calibration", "evaluation"):
            panel = value[name]
            if name == "calibration" and not panel["available"]:
                value["blocked_reason"] = panel["block_reason"]
                value["calibration"] = None
                continue
            panel["inputs"] = _freeze_rows(panel["inputs"], cache)
            panel["ordered_ids"] = [row["official_id"] for row in panel["inputs"]]
            panel["images"] = len(panel["inputs"])
            panel["source_hashes"] = dict(sources)
            panel["source_sha256"] = {str(Path(path).relative_to(package_root)): meta["sha256"]
                                      for path, meta in sources.items() if Path(path).is_relative_to(package_root)}
            panel["input_identity_sha256"] = cohorts.value_hash(panel["inputs"])
        value["selection"] = dict(seed=seed, size=calibration_size, annotation_pixels_read=False,
                                  ranking="SHA256 UTF-8 trace-single-probe-tuning-v1\\n{seed}\\n{dataset}\\n{official_id}\\n ascending; ID tie break; selected panel sorted by original source path",
                                  historical_development_exposure_retained=True, global_never_seen_certified=False,
                                  disjointness_scope="Own evaluation source IDs, canonical paths and inodes only; not cross-dataset or global historical independence")
        if value["input_audit"].get("development_overlap") is True:
            value["selection"].update({name: value["calibration"][name] for name in (
                "protocol", "development_overlap", "allow_waterbirds_development_overlap",
                "cal_full_source_id_overlap", "train_full_source_id_overlap",
                "declared_train_count", "original_full_count", "independent_test_claim",
                "global_never_seen_certified", "evaluation_scope", "claim_boundary")})
            value["selection"]["disjointness_scope"] = "Explicit Waterbirds development overlap: all calibration sources are in train and original full; no disjoint or independent-test claim."
    for path, metadata in sources.items():
        with Path(path).open("rb") as stream:
            content = stream.read(METADATA_LIMIT + 1)
        if len(content) != metadata["bytes"] or hashlib.sha256(content).hexdigest() != metadata["sha256"]:
            raise InputError(f"Metadata/source changed during preparation: {path}")
    return dict(schema_version=1, algorithm="ImageBalancedTRACE:C", datasets=datasets,
                source_hashes=sources,
                source_sha256={str(Path(path).relative_to(package_root)): meta["sha256"]
                               for path, meta in sources.items() if Path(path).is_relative_to(package_root)},
                output=str(output), annotation_pixels_read=False, offline_construction_performed=False,
                allow_waterbirds_development_overlap=allow_waterbirds_development_overlap,
                historical_development_exposure_retained=True, global_never_seen_certified=False)
