"""CPU-only, annotation-blind preparation of prospective TRACE cohorts.

No model/dataset imports: suite_inputs.expected_inputs constructs a torch
dataset, so we read its repository specs as literals and mirror its train
split transformation. Only RGB files and bounded textual metadata are read.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, replace
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile


SEED = 20261029
AUDIT_PATH = Path("/home/wwww/hzl/results/trace_parameter_provenance_20261029_0030/audit.json")
FAMILIES = {"pc60": "voc", "pc59": "voc", "voc20": "voc", "voc21": "voc",
            "coco_object": "coco", "coco_stuff": "coco", "coco_cd": "coco"}
HISTORY_KEYS = {"voc20", "voc21", "pc60", "coco_object", "coco_stuff", "coco_cd", "voc20_fast128"}
TRAIN_COUNTS = {"pc60": 4998, "coco_object": 118287, "voc21": 1464}
SMALL_FILE_LIMIT = 131072
RECENT_INPUT_LIMIT = 16 * 1024 * 1024
RECENT_HISTORY_ROOTS = (
    Path("/home/wwww/hzl/results/rebuttal_backbone_object_refine_recovery_20261026_1130"),
    Path("/home/wwww/hzl/results/rebuttal_backbone_stuff_refine_20261026_1307"),
)


class CohortError(RuntimeError):
    """The requested identity or provenance claim cannot be supported."""


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def value_hash(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read_metadata(path, source_hashes, *, limit=SMALL_FILE_LIMIT, expected=None):
    path = Path(path).expanduser().resolve(strict=True)
    if path.suffix not in {".json", ".txt", ".md", ".py"} and path.name != "SHA256SUMS":
        raise CohortError(f"Not a supported textual metadata source: {path}")
    if path.stat().st_size > limit:
        raise CohortError(f"Metadata file exceeds {limit}-byte bound: {path}")
    with path.open("rb") as handle:
        content = handle.read(limit + 1)
    if len(content) > limit:
        raise CohortError(f"Metadata grew beyond read bound: {path}")
    digest = hashlib.sha256(content).hexdigest()
    if expected is not None and digest != expected:
        raise CohortError(f"Source hash changed: {path}")
    source_hashes[str(path)] = {"sha256": digest, "bytes": len(content)}
    return content.decode("utf-8")


@dataclass(frozen=True)
class Layout:
    dataset: str
    family: str
    root: Path
    image_dir: str
    mask_dir: str
    image_suffix: str
    mask_suffix: str
    split_file: str | None = None


def _assignment(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node.value
    raise CohortError(f"Missing repository metadata declaration: {name}")


def repository_layouts(package_root, data_root, source_hashes, data_dirs=None):
    """Read actual naming/configuration without executing package imports."""
    root, data_root = Path(package_root), Path(data_root).expanduser().resolve(strict=True)
    specifications = ast.parse(read_metadata(root / "trace_ovss/data/datasets.py", source_hashes))
    configurations = ast.parse(read_metadata(root / "trace_ovss/configs/trace.py", source_hashes))
    inputs = ast.parse(read_metadata(root / "tools/suite_inputs.py", source_hashes))
    counts = ast.literal_eval(_assignment(inputs, "EVALUATION_COUNTS"))
    specs, configs = {}, {}
    for key, call in zip(_assignment(specifications, "DATASET_SPECS").keys,
                         _assignment(specifications, "DATASET_SPECS").values):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "DatasetSpec":
            raise CohortError("Unsupported dataset specification; audit the metadata adapter")
        specs[ast.literal_eval(key)] = [ast.literal_eval(arg) for arg in call.args]
    declaration = _assignment(configurations, "BEST_CONFIGS")
    for key, call in zip(declaration.keys, declaration.values):
        kwargs = {k.arg: k.value for k in call.keywords}
        configs[ast.literal_eval(key)] = (ast.literal_eval(kwargs["dataset"]), ast.literal_eval(kwargs["data_subdir"]))
    data_dirs = data_dirs or {}
    if set(data_dirs) - {"pc60", "coco_object", "voc21", "coco_cd"}:
        raise CohortError("Unknown dataset-directory override")
    layouts = {}
    for key in ("pc60", "coco_object", "voc21", "coco_cd"):
        name, subdir = configs[key]
        args = specs[name]
        layouts[key] = Layout(key, FAMILIES[key], Path(data_dirs.get(key, data_root / subdir)).expanduser().resolve(strict=True),
                              *args[:4], args[4] if len(args) >= 5 and isinstance(args[4], str) else None)
    return layouts, counts


def training_layout(layout):
    """The calibration branch of tools/suite_inputs.py, without dataset IO."""
    if layout.dataset.startswith(("voc", "pc")) and layout.split_file:
        return replace(layout, split_file=layout.split_file.replace("val.txt", "train.txt"))
    if layout.dataset in {"coco_object", "coco_stuff"}:
        return replace(layout, image_dir=layout.image_dir.replace("val2017", "train2017"),
                       mask_dir=layout.mask_dir.replace("val2017", "train2017"))
    raise CohortError(f"Unaudited calibration split: {layout.dataset}")


def official_identity(value, family):
    """Benchmark-wide identity, deliberately independent of split and path."""
    name = str(value)
    if name.startswith(family + ":"):
        name = name[len(family) + 1:]
    name = Path(name).name
    if name.endswith(".jpg"):
        name = name[:-4]
    pattern = r"[0-9]{4}_[0-9]{6}" if family == "voc" else r"[0-9]{12}" if family == "coco" else None
    if pattern is None or re.fullmatch(pattern, name) is None:
        raise CohortError(f"Unknown {family} official identity: {value}")
    return f"{family}:{name}"


def _safe_stem(stem):
    path = Path(stem)
    if not stem or path.is_absolute() or ".." in path.parts or str(path) != stem:
        raise CohortError(f"Unsafe or noncanonical split ID: {stem!r}")


def discover_population(layout, source_hashes, *, expected_count=None):
    """Discover every declared pair; stat labels, never read their contents."""
    image_root, mask_root = layout.root / layout.image_dir, layout.root / layout.mask_dir
    if not image_root.is_dir() or not mask_root.is_dir():
        raise CohortError(f"Missing image/mask directory: {image_root}, {mask_root}")
    if layout.split_file:
        text = read_metadata(layout.root / layout.split_file, source_hashes)
        stems = [line.strip() for line in text.splitlines() if line.strip()]
        if len(stems) != len(set(stems)):
            raise CohortError("Repeated split IDs")
        for stem in stems:
            _safe_stem(stem)
        images = [image_root / f"{stem}{layout.image_suffix}" for stem in stems]
    else:
        images = sorted(image_root.rglob(f"*{layout.image_suffix}"))
    if expected_count is not None and len(images) != expected_count:
        raise CohortError(f"Incomplete {layout.dataset} population: expected {expected_count}, found {len(images)}")
    if not images:
        raise CohortError(f"Empty {layout.dataset} population")
    rows, ids, paths, physical = [], set(), set(), set()
    for image in images:
        stem = image.relative_to(image_root).as_posix()[:-len(layout.image_suffix)]
        label = mask_root / f"{stem}{layout.mask_suffix}"
        if not image.is_file() or not label.is_file():
            raise CohortError(f"Incomplete declared image/mask pair: {image}, {label}")
        canonical, canonical_label = image.resolve(strict=True), label.resolve(strict=True)
        if canonical.suffix != layout.image_suffix or canonical_label.suffix != ".png":
            raise CohortError(f"Image/label alias changes file type: {image}, {label}")
        identity = official_identity(stem, layout.family)
        # An alternate filename cannot disguise a known different image ID.
        try:
            resolved_identity = official_identity(canonical.name, layout.family)
        except CohortError:
            resolved_identity = identity
        if resolved_identity != identity:
            raise CohortError(f"Image alias changes official identity: {image}")
        image_stat, label_stat = canonical.stat(), canonical_label.stat()
        if image_stat.st_size == 0 or label_stat.st_size == 0:
            raise CohortError(f"Empty declared image/mask file: {image}, {label}")
        inode = (image_stat.st_dev, image_stat.st_ino)
        if identity in ids or str(canonical) in paths or inode in physical:
            raise CohortError(f"Repeated official identity/path alias: {image}")
        ids.add(identity)
        paths.add(str(canonical))
        physical.add(inode)
        rows.append(dict(official_id=identity, image=str(canonical), label=str(canonical_label),
                         source_image=str(image.absolute()), source_label=str(label.absolute()),
                         image_metadata=dict(size_bytes=image_stat.st_size, mtime_ns=image_stat.st_mtime_ns,
                                             device=image_stat.st_dev, inode=image_stat.st_ino),
                         label_metadata=dict(size_bytes=label_stat.st_size, mtime_ns=label_stat.st_mtime_ns,
                                             device=label_stat.st_dev, inode=label_stat.st_ino, contents_read=False)))
    return rows


def _tokens(row):
    meta = row["image_metadata"]
    return {("id", row["official_id"]), ("path", row["image"]), ("inode", meta["device"], meta["inode"])}


def identity_index(populations):
    index = {}
    label_files = {(row["label_metadata"]["device"], row["label_metadata"]["inode"])
                   for rows in populations.values() for row in rows}
    for rows in populations.values():
        for row in rows:
            meta = row["image_metadata"]
            if (meta["device"], meta["inode"]) in label_files:
                raise CohortError("RGB/label physical file alias; refusing possible annotation read")
            for token in _tokens(row):
                index.setdefault(token, {})[row["official_id"]] = row
    if any(len(matches) != 1 for matches in index.values()):
        raise CohortError("Conflicting official identities for the same canonical/physical image")
    return index


def resolve_exclusion(value, family, index):
    """Require a historical ID/path to map to an enumerated official source."""
    if not isinstance(value, str) or not value:
        raise CohortError(f"Unknown exclusion identity: {value!r}")
    matches = {}
    path = Path(value)
    if path.is_absolute():
        matches.update(index.get(("path", str(path.resolve())), {}))
        if path.is_file():
            stat = path.stat()
            matches.update(index.get(("inode", stat.st_dev, stat.st_ino), {}))
    try:
        identity = official_identity(value, family)
    except CohortError:
        if not matches:
            raise
    else:
        matches.update(index.get(("id", identity), {}))
        if matches and identity not in matches:
            raise CohortError(f"Exclusion path/official ID conflict: {value}")
    if len(matches) != 1:
        raise CohortError(f"Unresolved or ambiguous exclusion identity: {value}")
    if not next(iter(matches)).startswith(family + ":"):
        raise CohortError(f"Exclusion source-family conflict: {value}")
    return next(iter(matches.values()))


def load_history_manifest(path, populations, source_hashes, *, expected=None, audited=None):
    record = json.loads(read_metadata(path, source_hashes, expected=expected))
    key = record.get("dataset_key")
    if key not in FAMILIES or record.get("split") != "train":
        raise CohortError(f"Unknown history dataset/split: {path}")
    values = record.get("sample_paths", record.get("image_ids", record.get("ordered_image_ids")))
    if not isinstance(values, list) or not values:
        raise CohortError(f"History requires explicit IDs/paths, not index-only selection: {path}")
    if record.get("selected_size", len(values)) != len(values):
        raise CohortError(f"Incomplete history manifest: {path}")
    if "indices" in record and (len(record["indices"]) != len(values) or
                                len(set(record["indices"])) != len(values)):
        raise CohortError(f"Invalid historical index list: {path}")
    index = identity_index(populations)
    rows = [resolve_exclusion(value, FAMILIES[key], index) for value in values]
    ids = [row["official_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise CohortError(f"Repeated historical identity/path alias: {path}")
    if audited is not None:
        audited_ids = [official_identity(value, FAMILIES[key]) for value in audited["ordered_image_ids"]]
        if ids != audited_ids or record.get("indices") != audited.get("indices"):
            raise CohortError(f"Historical source disagrees with audit: {path}")
    return dict(path=str(Path(path).resolve()), dataset=key, declared_split="train",
                declared_population=record.get("dataset_size"), source_values=values,
                historical_indices=record.get("indices"), official_ids=ids, rows=rows,
                identity_mapping="canonical path OR enumerated official identity; never numeric index")


def inspect_pool_history(audit, source_hashes):
    """Inspect named small archived sources, not today's loader ordering."""
    registry = {entry["path"]: entry["sha256"] for entry in audit["source_files"]}
    builders = [path for path in registry if Path(path).name == "build_trace_v2_calibration_splits.py"]
    if len(builders) != 1:
        raise CohortError("Missing or ambiguous bounded historical builder pointer")
    builder = read_metadata(builders[0], source_hashes, expected=registry[builders[0]])
    snapshot = Path(audit["historical_root"]) / "_snapshot"
    sums = snapshot / "SHA256SUMS"
    if str(sums) not in registry:
        raise CohortError("Historical source checksums are not indexed")
    checksums = {line.split(None, 1)[1].strip(): line.split(None, 1)[0]
                 for line in read_metadata(sums, source_hashes, expected=registry[str(sums)]).splitlines() if line.strip()}
    archive = snapshot / "mlmp_trace_sources.tar.gz"
    if str(archive) not in checksums or archive.stat().st_size > 524288:
        raise CohortError("Missing or oversized bounded loader source archive")
    with archive.open("rb") as handle:
        raw = handle.read(524289)
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) > 524288 or digest != checksums[str(archive)]:
        raise CohortError("Archived loader source hash mismatch")
    source_hashes[str(archive)] = dict(sha256=digest, bytes=len(raw))
    archived = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as handle:
        members = handle.getmembers()
        if len(members) > 256:
            raise CohortError("Unexpected archived source inventory size")
        for name in ("external/MLMP/utils/segmentation_datasets.py", "external/MLMP/utils/simple_seg_dataset.py"):
            member = handle.getmember(name)
            if not member.isfile() or member.size > SMALL_FILE_LIMIT:
                raise CohortError("Unsupported archived loader member")
            content = handle.extractfile(member).read(SMALL_FILE_LIMIT + 1)
            if len(content) != member.size:
                raise CohortError("Incomplete archived loader member")
            archived[name] = content.decode("utf-8")
            source_hashes[f"{archive}::{name}"] = dict(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content),
                                                     container_sha256=digest)
    tree = ast.parse(archived["external/MLMP/utils/segmentation_datasets.py"])
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_data"]
    if len(functions) != 1:
        raise CohortError("Archived loader signature is unknown")
    arguments = [arg.arg for arg in functions[0].args.args + functions[0].args.kwonlyargs]
    calls = [node for node in ast.walk(ast.parse(builder)) if isinstance(node, ast.Call) and
             isinstance(node.func, ast.Name) and node.func.id == "prepare_data"]
    builder_requests_train = any(any(k.arg == "data_split" and isinstance(k.value, ast.Constant) and
                                    k.value.value == "train" for k in node.keywords) for node in calls)
    return dict(status="unresolved", candidate_pool_ids_reconstructed=False,
                builder_path=builders[0], builder_requests_data_split_train=builder_requests_train,
                archived_prepare_data_arguments=arguments, archived_loader_has_data_split="data_split" in arguments,
                source_archive=str(archive),
                reason="Archived loader lacks the builder's train-split API; canonical TRACE-v2 configuration and exact historical full train-ID mapping are not established. Do not reconstruct using today's index alone.",
                selected_manifest_ids_only=True, strong_independent_confirmation_supported=False)


def inspect_recent_history(roots, source_hashes):
    """Read named parent records; defer input mapping until populations exist."""
    records = []
    for root in roots:
        root = Path(root)
        read_metadata(root / "README.md", source_hashes)
        manifest = json.loads(read_metadata(root / "manifest.json", source_hashes))
        queue = json.loads(read_metadata(root / "queue_state.json", source_hashes))
        calibration = manifest.get("calibration", {})
        if calibration.get("split") != "train" or not isinstance(calibration.get("count"), int):
            raise CohortError(f"Unknown recent calibration declaration: {root}")
        inputs = []
        for dataset, item in manifest.get("inputs", {}).items():
            path = Path(item["path"])
            inputs.append(dict(dataset=dataset, path=str(path), bytes=path.stat().st_size,
                               declared_sha256=item.get("sha256"), contents_opened=False, hash_verified=False))
        records.append(dict(root=str(root), manifest_path=str(root / "manifest.json"),
                            declared_calibration_count=calibration["count"],
                            declared_calibration_split=calibration["split"],
                            declared_selection_seed=calibration.get("selection_seed"),
                            queue_status=queue.get("status"),
                            training_inputs=manifest.get("training_inputs"),
                            training_inputs_source=manifest.get("training_inputs_source"),
                            referenced_inputs=inputs, explicit_training_ids_mapped=False,
                            status="pending hash-verified bounded input mapping"))
    return records


def load_recent_history_manifest(path, populations, source_hashes, *, dataset, expected,
                                 expected_count, selection_seed):
    """Map explicit calibration rows against their archived full ID array.

    The 16 MiB exception is only for caller-named, parent-hash-bound metadata.
    Historical mask/RGB digests are strings, not requests to open those files.
    """
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise CohortError("Recent input requires an exact parent-declared SHA-256")
    record = json.loads(read_metadata(path, source_hashes, limit=RECENT_INPUT_LIMIT, expected=expected))
    if dataset not in {"coco_object", "coco_stuff"} or record.get("dataset") != dataset:
        raise CohortError(f"Unknown recent input dataset: {path}")
    if record.get("selection_seed") != selection_seed or selection_seed is None:
        raise CohortError(f"Recent calibration seed disagrees with parent: {path}")
    if record.get("train_spec", {}).get("image_dir") != "images/train2017" or \
            record.get("evaluation_spec", {}).get("image_dir") != "images/val2017":
        raise CohortError(f"Unknown recent official train/eval source specification: {path}")
    train_values, evaluation = record.get("train_ids"), record.get("evaluation")
    calibration, indices = record.get("calibration"), record.get("calibration_indices")
    if not isinstance(train_values, list) or not train_values or record.get("train_count") != len(train_values):
        raise CohortError(f"Incomplete archived training IDs: {path}")
    if not isinstance(evaluation, list) or not evaluation or record.get("evaluation_count") != len(evaluation):
        raise CohortError(f"Incomplete archived evaluation IDs: {path}")
    if not isinstance(calibration, list) or len(calibration) != expected_count or \
            record.get("calibration_size", expected_count) != expected_count:
        raise CohortError(f"Incomplete recent calibration panel: {path}")
    if not isinstance(indices, list) or len(indices) != expected_count or \
            any(type(index) is not int or not 0 <= index < len(train_values) for index in indices) or \
            len(set(indices)) != expected_count:
        raise CohortError(f"Invalid recent calibration index list: {path}")
    family = FAMILIES[dataset]
    if any(not isinstance(value, str) for value in train_values):
        raise CohortError(f"Unknown archived training identity: {path}")
    train_ids = [official_identity(value, family) for value in train_values]
    current_train = populations.get("coco_object/train")
    current_eval = populations.get("coco_object/evaluation")
    if current_train is None or current_eval is None:
        raise CohortError("Missing source-bound COCO train/evaluation populations")
    if len(set(train_ids)) != len(train_ids) or set(train_ids) != {row["official_id"] for row in current_train}:
        raise CohortError(f"Archived training population does not match enumerated official source: {path}")
    if any(not isinstance(row, dict) or not isinstance(row.get("image_id"), str) for row in evaluation):
        raise CohortError(f"Unknown archived evaluation identity: {path}")
    evaluation_ids = [official_identity(row["image_id"], family) for row in evaluation]
    if len(set(evaluation_ids)) != len(evaluation_ids) or \
            set(evaluation_ids) != {row["official_id"] for row in current_eval} or set(evaluation_ids) & set(train_ids):
        raise CohortError(f"Archived official train/eval mapping is incomplete or overlaps: {path}")
    index = identity_index(populations)
    rows, mappings = [], []
    mask_suffix = record.get("train_spec", {}).get("mask_suffix")
    expected_suffix = "_instanceTrainIds.png" if dataset == "coco_object" else "_labelTrainIds.png"
    if mask_suffix != expected_suffix:
        raise CohortError(f"Unknown recent mask naming specification: {path}")
    for position, entry in zip(indices, calibration):
        if not isinstance(entry, dict) or not isinstance(entry.get("image_id"), str) or \
                not isinstance(entry.get("stem"), str) or not isinstance(entry.get("mask_path"), str):
            raise CohortError(f"Unknown explicit recent calibration row: {path}")
        row = resolve_exclusion(entry["image_id"], family, index)
        identity = row["official_id"]
        mask_name = Path(entry["mask_path"]).name
        if not mask_name.endswith(mask_suffix) or \
                official_identity(mask_name[:-len(mask_suffix)], family) != identity or \
                official_identity(entry["stem"], family) != identity or train_ids[position] != identity:
            raise CohortError(f"Explicit recent row disagrees with archived index/stem/mask identity: {path}")
        rows.append(row)
        mappings.append(dict(official_id=identity, source_image=entry["image_id"],
                             source_label=entry["mask_path"], archived_train_index=position,
                             archived_train_id=train_values[position], canonical_image=row["image"]))
    ids = [row["official_id"] for row in rows]
    if len(set(ids)) != expected_count:
        raise CohortError(f"Repeated recent calibration identity: {path}")
    return dict(path=str(Path(path).resolve()), dataset=dataset, declared_split="train",
                declared_population=len(train_ids), source_values=[entry["image_id"] for entry in calibration],
                historical_indices=indices, official_ids=ids, rows=rows, mappings=mappings,
                official_ids_sha256=value_hash(ids), archived_training_ids_sha256=value_hash(train_values),
                archived_calibration_indices_sha256=value_hash(indices),
                archived_index_identity_map_sha256=value_hash([
                    dict(archived_train_index=position, official_id=identity)
                    for position, identity in zip(indices, ids)]),
                archived_training_official_ids_sha256=value_hash(train_ids),
                archived_evaluation_official_ids_sha256=value_hash(evaluation_ids),
                source_population_match=True, own_train_eval_disjoint=True,
                identity_mapping="explicit calibration rows cross-checked with archived train_ids[calibration_indices]; no current-index inference")


def recent_intersections(dataset, train, evaluation, previous_histories, recent_histories):
    """Exact exclusion/old-cohort intersections before adding recent panels."""
    previous = select_cohorts(dataset, train, evaluation, previous_histories)
    train_ids = {row["official_id"] for row in train}
    recent_ids = {row["official_id"] for history in recent_histories for row in history["rows"]} & train_ids
    excluded = set(previous["exclusions"]["union_ids"])
    return dict(recent_ids_in_training=sorted(recent_ids),
                previous_source_intersections=[dict(path=history.get("path"), official_ids=sorted(
                    recent_ids & {row["official_id"] for row in history["rows"]}))
                    for history in previous_histories
                    if recent_ids & {row["official_id"] for row in history["rows"]}],
                evaluation_source_intersections=[dict(dataset=key, official_ids=sorted(
                    recent_ids & {row["official_id"] for row in rows}))
                    for key, rows in evaluation.items() if recent_ids & {row["official_id"] for row in rows}],
                previous_known_history_intersection_ids=sorted(recent_ids & set(previous["exclusions"]["known_history_ids"])),
                official_evaluation_intersection_ids=sorted(recent_ids & set(previous["exclusions"]["official_evaluation_ids"])),
                previous_exclusion_union_intersection_ids=sorted(recent_ids & excluded),
                newly_excluded_ids=sorted(recent_ids - excluded),
                previous_calibration_intersection_ids=sorted(recent_ids & {row["official_id"] for row in previous["calibration"]}),
                previous_confirmation_intersection_ids=sorted(recent_ids & {row["official_id"] for row in previous["confirmation"]}),
                previous_selected_ids_sha256={name: value_hash([row["official_id"] for row in previous[name]])
                                             for name in ("calibration", "confirmation")})


def _overlap(left, right):
    tokens = set().union(*(_tokens(row) for row in right)) if right else set()
    return [row for row in left if _tokens(row) & tokens]


def select_cohorts(dataset, train, evaluation, histories, *, calibration_size=128, confirmation_size=256):
    """Fixed hash ranking; own split overlap is fatal, cross-eval rows excluded."""
    if calibration_size <= 0 or confirmation_size <= 0:
        raise CohortError("Cohort sizes must be positive")
    own = evaluation.get(dataset)
    if own is None or not own:
        raise CohortError(f"Missing official evaluation population for {dataset}")
    for rows in [train, *evaluation.values()]:
        if len(rows) != len({row["official_id"] for row in rows}):
            raise CohortError("Repeated official identity within a population")
    identity_index({"train": train, **evaluation})
    if _overlap(train, own):
        raise CohortError(f"Official train/eval overlap for {dataset}")
    all_eval = [row for rows in evaluation.values() for row in rows]
    known = [row for history in histories for row in history["rows"]]
    eval_rows, history_rows = _overlap(train, all_eval), _overlap(train, known)
    excluded = {row["official_id"] for row in eval_rows + history_rows}
    eligible = [row for row in train if row["official_id"] not in excluded]
    if len(eligible) < calibration_size + confirmation_size:
        raise CohortError(f"Insufficient eligible {dataset} images: {len(eligible)}")
    ranked = sorted(eligible, key=lambda row: (rank_digest(dataset, row["official_id"]), row["official_id"]))
    calibration = ranked[:calibration_size]
    confirmation = ranked[calibration_size:calibration_size + confirmation_size]
    return dict(calibration=calibration, confirmation=confirmation,
                eligible_ids=[row["official_id"] for row in ranked],
                exclusions=dict(official_evaluation_ids=sorted(row["official_id"] for row in eval_rows),
                                known_history_ids=sorted(row["official_id"] for row in history_rows),
                                union_ids=sorted(excluded)),
                counts=dict(train=len(train), own_evaluation=len(own), eligible=len(eligible),
                            excluded_union=len(excluded), official_evaluation=len(eval_rows),
                            known_history=len(history_rows), overlap_history_evaluation=len(eval_rows) + len(history_rows) - len(excluded),
                            calibration=len(calibration), confirmation=len(confirmation),
                            unused_eligible=len(eligible) - calibration_size - confirmation_size))


def rank_digest(dataset, official_id):
    return hashlib.sha256(f"trace-confirmation-v1\n{SEED}\n{dataset}\n{official_id}\n".encode("ascii")).hexdigest()


def _selected_row(row):
    label_stat, label_meta = Path(row["label"]).stat(), row["label_metadata"]
    if (label_stat.st_dev, label_stat.st_ino, label_stat.st_size, label_stat.st_mtime_ns) != (
            label_meta["device"], label_meta["inode"], label_meta["size_bytes"], label_meta["mtime_ns"]):
        raise CohortError(f"Selected label metadata changed after discovery: {row['label']}")
    before = Path(row["image"]).stat()
    meta = row["image_metadata"]
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            meta["device"], meta["inode"], meta["size_bytes"], meta["mtime_ns"]):
        raise CohortError(f"Selected RGB changed after discovery: {row['image']}")
    digest = hashlib.sha256()
    with Path(row["image"]).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = Path(row["image"]).stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise CohortError(f"Selected RGB changed while hashing: {row['image']}")
    return {key: row[key] for key in ("official_id", "image", "label", "label_metadata")} | {"image_sha256": digest.hexdigest()}


def prepare_cohorts(package_root, data_root, *, audit_path=AUDIT_PATH, data_dirs=None, extra_history=(),
                    require_independent=False):
    """Return a fully auditable bundle in memory; never create an output root.

    data_dirs is an explicit {dataset: directory} override, e.g. the sealed
    A controller's coco_object=/home/wwww/hzl/data/coco_mlmp. No sealed payload
    is inferred or recursively searched. extra_history names small ID manifests.
    """
    package_root = Path(package_root).resolve(strict=True)
    sources = {}
    audit = json.loads(read_metadata(audit_path, sources, limit=524288))
    read_metadata(package_root / "docs/parameter-provenance-audit.md", sources)
    read_metadata(package_root / "docs/confirmation-cohort-audit.md", sources)
    read_metadata(Path(__file__), sources)
    pool_history = inspect_pool_history(audit, sources)
    recent_history = inspect_recent_history(RECENT_HISTORY_ROOTS, sources)
    if require_independent:
        raise CohortError("Independent-confirmation claim unsupported: historical candidate-pool mappings remain unresolved")
    layouts, eval_counts = repository_layouts(package_root, data_root, sources, data_dirs)
    train, evaluation = {}, {}
    for key in ("pc60", "coco_object", "voc21"):
        train[key] = discover_population(training_layout(layouts[key]), sources, expected_count=TRAIN_COUNTS[key])
        evaluation[key] = discover_population(layouts[key], sources, expected_count=eval_counts[key])
    evaluation["coco_cd"] = discover_population(layouts["coco_cd"], sources, expected_count=eval_counts["coco_cd"])
    cd_val = replace(layouts["coco_cd"], split_file="splits/val_balanced.txt", mask_dir="masks/val_balanced")
    evaluation["coco_cd_validation"] = discover_population(cd_val, sources, expected_count=200)
    populations = {**{f"{k}/train": v for k, v in train.items()},
                   **{f"{k}/evaluation": v for k, v in evaluation.items()}}
    manifests = audit.get("calibration_manifests", {})
    if set(manifests) != HISTORY_KEYS:
        raise CohortError("Bounded historical calibration index changed; review coverage explicitly")
    pc_ids = {row["official_id"] for row in train["pc60"]}
    pc_selected = {official_identity(value, "voc") for value in manifests["pc60"]["ordered_image_ids"]}
    pc_pool_size = manifests["pc60"]["pool_size"]
    conservative_fixed = pc_ids & (pc_selected | {row["official_id"] for row in train["voc21"]} |
                                   {row["official_id"] for rows in evaluation.values() for row in rows})
    outside_pool = len(pc_ids) - pc_pool_size
    pool_history["pc60_unused_capacity"] = dict(
        declared_population=len(pc_ids), declared_label_inspected_pool=pc_pool_size,
        outside_this_pool_before_other_exclusions=outside_pool,
        conservative_fixed_exclusion_union=len(conservative_fixed),
        conditional_unused_lower_bound=max(0, outside_pool - len(conservative_fixed - pc_selected)),
        conditional_unused_upper_bound=min(outside_pool, len(pc_ids - conservative_fixed)),
        identified_pool_unexposed_ids=0, actual_globally_unused_count=None,
        supports_identifiable_384_image_unused_cohort=False,
        condition="Combinatorial bounds assume declared July pool contains its selected512 and VOC training source was fully label-inspected. They ignore unindexed exposure, do not identify any unused IDs, and are not global never-seen bounds.")
    registry = {entry["path"]: entry["sha256"] for entry in audit["source_files"]}
    histories = []
    for key in sorted(manifests):
        entry = manifests[key]
        if entry["path"] not in registry:
            raise CohortError(f"Unhashed historical source: {entry['path']}")
        histories.append(load_history_manifest(entry["path"], populations, sources,
                                               expected=registry[entry["path"]], audited=entry))
    for path in extra_history:
        histories.append(load_history_manifest(path, populations, sources))
    previous_histories = list(histories)
    recent_histories = []
    for recent in recent_history:
        if not recent["referenced_inputs"]:
            raise CohortError(f"Missing parent-bound recent input manifest: {recent['root']}")
        for reference in recent["referenced_inputs"]:
            history = load_recent_history_manifest(reference["path"], populations, sources,
                dataset=reference["dataset"], expected=reference["declared_sha256"],
                expected_count=recent["declared_calibration_count"], selection_seed=recent["declared_selection_seed"])
            history["parent_manifest"] = recent["manifest_path"]
            recent_histories.append(history)
            reference.update(contents_opened=True, hash_verified=True, mapped_count=len(history["official_ids"]),
                             official_ids_sha256=history["official_ids_sha256"],
                             source_population_match=True, own_train_eval_disjoint=True)
        recent.update(explicit_training_ids_mapped=True, status="verified explicit calibration IDs included in exclusions")
    histories.extend(recent_histories)
    intersections = {key: recent_intersections(key, train[key], evaluation, previous_histories, recent_histories)
                     for key in ("pc60", "coco_object")}
    cross_source = []
    for position, left in enumerate(recent_histories):
        for right in recent_histories[position + 1:]:
            overlap = sorted(set(left["official_ids"]) & set(right["official_ids"]))
            cross_source.append(dict(left=left["path"], right=right["path"], official_ids=overlap,
                                     count=len(overlap), sha256=value_hash(overlap),
                                     same_ordered_image_ids=left["official_ids"] == right["official_ids"],
                                     same_archived_indices=left["historical_indices"] == right["historical_indices"]))
    datasets, exclusions = {}, {}
    for key in ("pc60", "coco_object"):
        selected = select_cohorts(key, train[key], evaluation, histories)
        exclusions[key] = selected.pop("exclusions")
        for name in ("calibration", "confirmation"):
            selected[name] = [_selected_row(row) for row in selected[name]]
        selected["selected_ids_sha256"] = {name: value_hash([row["official_id"] for row in selected[name]])
                                           for name in ("calibration", "confirmation")}
        datasets[key] = selected
    history_metadata = [{k: v for k, v in history.items() if k != "rows"} for history in histories]
    source_manifests = {}
    for key, rows in populations.items():
        samples = [{field: row[field] for field in ("official_id", "image", "label")} for row in rows]
        source_manifests[key] = dict(count=len(samples), sha256=value_hash(samples), samples=samples)
    return dict(schema_version=1, kind="prospective_known_history_training_source_cohorts",
                selection=dict(seed=SEED, calibration_size=128, confirmation_size=256,
                               ranking="SHA256 ASCII trace-confirmation-v1\\n{seed}\\n{dataset}\\n{official_id}\\n; ascending hex, ID tie break",
                               class_coverage_selection=False, annotation_pixels_read=False),
                datasets=datasets, exclusions=exclusions, source_hashes=sources,
                source_id_manifests=source_manifests,
                history_manifests=history_metadata,
                audit=dict(scope="prospective protocol; disjoint from enumerated official evaluation and recorded known-history exclusions only",
                           global_never_seen_certified=False, historical_index_exhaustive=False,
                           strong_independent_confirmation_supported=False,
                           historical_pool_audit=pool_history, recent_history_audit=recent_history,
                           recent_history_intersections=intersections, recent_cross_source_intersections=cross_source,
                           historical_index=str(Path(audit_path).resolve()),
                           unknowns=audit.get("unknowns", {}),
                           limitations=["No global pre-v2/unindexed exposure certification.",
                                        "Historical candidate pools were label-inspected but exact pool IDs are not indexed here; selected-subset exclusions are not all historical label exposure.",
                                        "Official identity/path/inode audit, not an all-population RGB-content duplicate audit.",
                                        "Historical indices are retained as audit evidence, never remapped by present ordering.",
                                        "No label bytes read or hashed; downstream labels remain closed until each stage's prediction seal.",
                                        "Confirmation is training-source holdout for this prospective method repair, not official evaluation or a manuscript score."],
                           train_eval_disjointness="Own target train/eval must be disjoint; cross-dataset evaluation overlap is excluded.",
                           official_evaluation_scope=["PC59/PC60 Context val", "VOC20/VOC21 Segmentation val",
                                                      "COCO Object/Stuff val2017", "COCO-CD test_balanced and conservative val_balanced"],
                           data_dirs={key: str(layout.root) for key, layout in layouts.items()},
                           source_binding="Repository specs/configs and explicit directory overrides; no annotation-content equivalence claim between roots.",
                           no_result_root_created=True, predictions_sealed=False))


def write_bundle(bundle, output):
    """Create one explicitly named fresh JSON, never overwrite/resume."""
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Use a fresh explicit JSON output: {output}")
    if output.suffix != ".json" or not output.parent.is_dir():
        raise CohortError("Output must be a .json file in an existing directory")
    with output.open("x", encoding="utf-8") as handle:
        json.dump(bundle, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh .json bundle; parent must already exist")
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--data-dir", action="append", default=[], metavar="DATASET=PATH")
    parser.add_argument("--history-manifest", type=Path, action="append", default=[])
    parser.add_argument("--require-independent", action="store_true", help="Fail closed if bounded records cannot support independent-confirmation claims")
    args = parser.parse_args(argv)
    args.output = args.output.expanduser().absolute()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Use a fresh explicit JSON output: {args.output}")
    if args.output.suffix != ".json" or not args.output.expanduser().parent.is_dir():
        parser.error("--output must name a fresh .json file in an existing directory")
    directories = {}
    for item in args.data_dir:
        if "=" not in item:
            parser.error("--data-dir requires DATASET=PATH")
        key, value = item.split("=", 1)
        if key in directories or not value:
            parser.error("Duplicate or empty --data-dir")
        directories[key] = value
    bundle = prepare_cohorts(args.package_root, args.data_root, audit_path=args.audit,
                             data_dirs=directories, extra_history=args.history_manifest,
                             require_independent=args.require_independent)
    write_bundle(bundle, args.output)
    print(json.dumps({"output": str(args.output), "counts": {key: value["counts"] for key, value in bundle["datasets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
