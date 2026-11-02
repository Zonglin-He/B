"""Fixture-only CPU tests: labels are sentinels, never decoded or opened."""

import builtins
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from tools.evolution_tuning import inputs


ROOT = Path(__file__).resolve().parents[1]


def put(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode() if isinstance(content, str) else content)
    return path


def pair(root, image_dir, label_dir, stem, suffix=".png"):
    put(root / image_dir / f"{stem}.jpg", f"RGB fixture: {image_dir}/{stem}")
    put(root / label_dir / f"{stem}{suffix}", b"UNREAD LABEL SENTINEL")


@pytest.fixture
def suite(tmp_path, monkeypatch, request):
    development = getattr(request, "param", None) == "waterbirds_development"
    native_train_count = 260 if development else 3
    waterbirds_train_count = 2458 if development else 2
    waterbirds_full_count = 5322 if development else 4
    data = tmp_path / "data"
    history = tmp_path / "results/history"
    inherited = tmp_path / "results/inherited"
    monkeypatch.setattr(inputs, "INHERITED_CONFIG_ROOT", inherited)
    monkeypatch.setattr(inputs, "HISTORICAL_ROOT", history)
    monkeypatch.setattr(inputs, "EVALUATION_COUNTS", {key: waterbirds_full_count if key == "waterbirds_seg" else 2 for key in inputs.DATASETS})
    monkeypatch.setattr(inputs, "TRAIN_COUNTS", {key: waterbirds_train_count if key == "waterbirds_seg" else native_train_count for key in inputs.DATASETS})
    defaults, specs, _, _ = inputs._literal_sources(ROOT, {})
    for key, default in defaults.items():
        if key not in inputs.DATASETS:
            continue
        root = data / ("coco_mlmp" if key == "coco_object" else default["data_subdir"])
        cfg = dict(default, data_dir=str(root), trace_view_batch_size=16)
        if key in {"pc59", "waterbirds_seg"}:
            historical = dict(cfg, patch=cfg.pop("patch_size"), stride=cfg.pop("patch_stride"))
            historical["trace_view_batch_size"] = 8
            put(history / "frozen_configs_final" / key / "best_config.json",
                json.dumps(dict(config=historical)))
        else:
            put(inherited / f"{key}.json", json.dumps(cfg))
        imdir, labdir, _, suffix, splitfile, _ = specs[default["dataset"]]
        if key.startswith(("voc", "pc")):
            train = [f"2007_{index:06d}" for index in (range(native_train_count, 0, -1) if development else (3, 1, 2))]
            full = [f"2008_{index:06d}" for index in (2, 1)]
            put(root / splitfile, "\n".join(full) + "\n")
            put(root / splitfile.replace("val.txt", "train.txt"), "\n".join(train) + "\n")
            for stem in train + full:
                pair(root, imdir, labdir, stem, suffix)
        elif key in {"coco_object", "coco_stuff"}:
            for partition, ids in (("val2017", (1002, 1001)), ("train2017", range(native_train_count, 0, -1))):
                for index in ids:
                    pair(root, imdir.replace("val2017", partition), labdir.replace("val2017", partition), f"{index:012d}", suffix)
        elif key == "coco_cd":
            for part, ids, imgpart in (("test_balanced", (1002, 1001), "val2017"),
                                       ("train_corr0.95", range(native_train_count, 0, -1), "train2017")):
                stems = [f"{imgpart}/{index:012d}" for index in ids]
                put(root / f"splits/{part}.txt", "\n".join(stems) + "\n")
                for stem in stems:
                    pair(root, "images", f"masks/{part}", stem)
        else:
            stems = [f"001.Fixture/Bird_{index:04d}" for index in range(waterbirds_full_count, 0, -1)]
            put(root / "splits/train.txt", "\n".join(stems[:waterbirds_train_count]) + "\n")
            for stem in stems:
                pair(root, "images", "masks", stem)
            metadata = "img_id,img_filename,split,place_filename\n" + "".join(
                f"{index},{stem}.jpg,{0 if index < waterbirds_train_count else 2},/f/forest/{index}.jpg\n"
                for index, stem in enumerate(stems))
            put(root / "images/metadata.csv", metadata)
            put(history / "calibration_splits/waterbirds_seg.json", json.dumps(dict(
                sample_paths=[str(root / "images" / f"{stem}.jpg") for stem in stems[:min(512, waterbirds_train_count)]])))
            cub = data / "waterbirds_sources/CUB_200_2011"
            cubstems = stems + [f"001.Fixture/Bird_{waterbirds_full_count + index:04d}" for index in (1, 2)]
            put(cub / "images.txt", "".join(f"{i} {stem}.jpg\n" for i, stem in enumerate(cubstems)))
            put(cub / "train_test_split.txt", "".join(f"{i} 1\n" for i in range(len(cubstems))))
    checksums = []
    for archive_name, member_name in (
        ("root_method_and_experiments.tar.gz", "experiments/prepare_mlmp_data.py"),
        ("mlmp_trace_sources.tar.gz", "external/MLMP/utils/simple_seg_dataset.py")):
        path = history / "_snapshot" / archive_name
        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "w:gz") as archive:
            content = b"# archived CPU source fixture\n"
            info = tarfile.TarInfo(member_name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n")
    put(history / "_snapshot/SHA256SUMS", "".join(checksums))
    return data, tmp_path / "output"


def test_all_eight_complete_cpu_panels_and_blocked_waterbirds(suite):
    data, output = suite
    bundle = inputs.prepare_inputs(ROOT, data, output, calibration_size=2)
    result = bundle["datasets"]
    assert set(result) == set(inputs.DATASETS)
    assert bundle == json.loads(json.dumps(bundle))
    assert not output.exists()
    for key, entry in result.items():
        full, calibration, cfg = entry["evaluation"], entry["calibration"], entry["config"]
        assert cfg["resize"] == cfg["patch_size"] == [224, 224]
        assert cfg["trace_view_batch_size"] == 16
        assert cfg["class_extensions"] is (key not in {"coco_cd", "waterbirds_seg"})
        assert full["full_split"] is True and full["available"] is True
        assert len(full["inputs"]) == (4 if key == "waterbirds_seg" else 2)
        assert full["ordered_ids"] == [row["official_id"] for row in full["inputs"]]
        assert full["source_sha256"]["tools/evolution_tuning/inputs.py"]
        assert full["source_hashes"]
        assert not entry["selection"]["global_never_seen_certified"]
        assert entry["selection"]["seed"] == 2027
        if key != "waterbirds_seg":
            assert calibration["available"] and len(calibration["inputs"]) == 2
            assert not set(full["ordered_ids"]) & set(calibration["ordered_ids"])
        for panel in (full, calibration):
            if panel is None:
                continue
            for row in panel["inputs"]:
                assert row["image_sha256"] == hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest()
                assert row["label_metadata"] == [Path(row["label"]).stat().st_size, Path(row["label"]).stat().st_mtime_ns]
                assert row["label_metadata_details"]["contents_read"] is False
    wb = result["waterbirds_seg"]
    assert wb["calibration"] is None
    assert "overlap" in wb["blocked_reason"]
    assert wb["input_audit"]["train_full_source_id_overlap"] == 2
    assert wb["input_audit"]["remaining_cub_train_count"] == 2
    assert wb["input_audit"]["construction_performed"] is False
    assert "coco_mlmp" in result["coco_object"]["config"]["data_dir"]
    assert "masks/train_corr0.95" in result["coco_cd"]["calibration"]["inputs"][0]["label"]
    assert result["pc59"]["evaluation"]["reduce_zero_label"] is True
    assert result["pc60"]["evaluation"]["reduce_zero_label"] is False
    assert result["voc21"]["evaluation"]["classes"][0] == "background"
    assert wb["evaluation"]["query_to_class"] is None


@pytest.mark.parametrize("suite,opt_in", [(None, False), ("waterbirds_development", True)], indirect=["suite"])
def test_discovery_and_preparation_never_open_labels(suite, monkeypatch, opt_in):
    data, output = suite
    original_path_open, original_builtin_open = Path.open, builtins.open
    reads = []

    def check(path, mode):
        if str(path).endswith(".png") and "r" in mode:
            raise AssertionError(f"Annotation was opened: {path}")
        if "r" in mode:
            reads.append(str(path))

    def path_open(path, mode="r", *args, **kwargs):
        check(path, mode)
        return original_path_open(path, mode, *args, **kwargs)

    def builtin_open(path, mode="r", *args, **kwargs):
        check(path, mode)
        return original_builtin_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(builtins, "open", builtin_open)
    inputs.prepare_inputs(ROOT, data, output, calibration_size=256 if opt_in else 2,
                          allow_waterbirds_development_overlap=opt_in)
    assert any(path.endswith(".jpg") for path in reads)
    assert not any(path.endswith(".png") for path in reads)


def test_repeated_seed_is_stable_and_controller_owns_output(suite):
    data, output = suite
    result = inputs.prepare_inputs(ROOT, data, output, calibration_size=2)
    output.mkdir()
    put(output / "queue_state.json", '{"phase":"preparing"}')
    assert inputs.prepare_inputs(ROOT, data, output, calibration_size=2) == result
    put(output / "inputs.json", json.dumps(result))
    with pytest.raises(FileExistsError):
        inputs.prepare_inputs(ROOT, data, output, calibration_size=2)
    other = inputs.prepare_inputs(ROOT, data, output.with_name("other"), calibration_size=2)
    assert result["datasets"] == other["datasets"]


def test_population_must_not_shrink_when_mask_missing(suite):
    data, output = suite
    (data / "coco_mlmp/annotations/val2017/000000001001_instanceTrainIds.png").unlink()
    with pytest.raises(inputs.InputError, match="Incomplete declared"):
        inputs.prepare_inputs(ROOT, data, output, calibration_size=2)
    assert not output.exists()


def test_duplicate_split_and_incorrect_cardinality_fail(suite):
    data, _ = suite
    layout = inputs.cohorts.Layout("voc21", "voc", data / "VOCdevkit/VOC2012",
                                   "JPEGImages", "SegmentationClass", ".jpg", ".png",
                                   "ImageSets/Segmentation/val.txt")
    put(layout.root / layout.split_file, "2008_000001\n2008_000001\n")
    with pytest.raises(inputs.InputError, match="Repeated split"):
        inputs.discover(layout, {}, expected_count=2)
    put(layout.root / layout.split_file, "2008_000001\n")
    with pytest.raises(inputs.InputError, match="expected 2"):
        inputs.discover(layout, {}, expected_count=2)


@pytest.mark.parametrize("opt_in", [False, True])
def test_training_source_overlap_is_fatal_not_silently_filtered(suite, opt_in):
    data, output = suite
    voc = data / "VOCdevkit/VOC2012"
    put(voc / "ImageSets/Segmentation/train.txt", "2008_000001\n2007_000001\n2007_000002\n")
    with pytest.raises(inputs.InputError, match="own train/eval overlap"):
        inputs.prepare_inputs(ROOT, data, output, calibration_size=256 if opt_in else 2,
                              allow_waterbirds_development_overlap=opt_in)
    assert not output.exists()


def test_wrong_coco_object_root_is_not_used(suite):
    data, output = suite
    # A plausible Stuff root cannot stand in for the Object annotations.
    for path in (data / "coco_mlmp/annotations/val2017").glob("*.png"):
        path.unlink()
    with pytest.raises(inputs.InputError, match="Incomplete declared"):
        inputs.prepare_inputs(ROOT, data, output, calibration_size=2)


def test_sorting_and_selection_do_not_depend_on_split_file_order(suite):
    data, output = suite
    first = inputs.prepare_inputs(ROOT, data, output, calibration_size=2)["datasets"]
    path = data / "VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"
    put(path, "2008_000001\n2008_000002\n")
    second = inputs.prepare_inputs(ROOT, data, output.with_name("second"), calibration_size=2)["datasets"]
    assert first["voc21"]["evaluation"]["inputs"] == second["voc21"]["evaluation"]["inputs"]
    assert first["voc21"]["calibration"]["ordered_ids"] == second["voc21"]["calibration"]["ordered_ids"]


def test_stat_alias_is_rejected_before_rgb_hash_read(suite):
    data, output = suite
    image = data / "coco_mlmp/images/train2017/000000000001.jpg"
    label = data / "coco_mlmp/annotations/train2017/000000000001_instanceTrainIds.png"
    image.unlink()
    os.link(label, image)
    with pytest.raises(inputs.InputError, match="RGB/label physical file alias"):
        inputs.prepare_inputs(ROOT, data, output, calibration_size=2)


def test_labels_and_official_data_are_not_written(suite, monkeypatch):
    data, output = suite
    original = Path.open

    def guarded(path, mode="r", *args, **kwargs):
        if Path(path).is_relative_to(data) and any(letter in mode for letter in "wax+"):
            raise AssertionError(f"Official data write: {path}")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    inputs.prepare_inputs(ROOT, data, output, calibration_size=2)
    assert not output.exists()
    with pytest.raises(inputs.InputError, match="outside official"):
        inputs.prepare_inputs(ROOT, data, data / "forbidden", calibration_size=2)


def test_metadata_drift_is_detected_before_freeze(suite):
    data, _ = suite
    layout = inputs.cohorts.Layout("coco_object", "coco", data / "coco_mlmp",
                                   "images/val2017", "annotations/val2017", ".jpg", "_instanceTrainIds.png")
    rows = inputs.discover(layout, {}, expected_count=2)
    put(Path(rows[0]["label"]), "Changed label metadata, never read")
    with pytest.raises(inputs.InputError, match="metadata changed"):
        inputs._freeze_rows(rows, {})


def test_small_pool_and_invalid_parameters_fail(suite):
    data, output = suite
    with pytest.raises(inputs.InputError, match="requires 4 training"):
        inputs.prepare_inputs(ROOT, data, output, calibration_size=4)
    for size, seed in ((0, 2027), (True, 2027), (2, -1), (2, True)):
        with pytest.raises(inputs.InputError, match="integers"):
            inputs.prepare_inputs(ROOT, data, output, calibration_size=size, seed=seed)


def test_inherited_config_drift_fails(suite):
    data, output = suite
    path = inputs.INHERITED_CONFIG_ROOT / "voc20.json"
    config = json.loads(path.read_text())
    config["trace_view_batch_size"] = 8
    put(path, json.dumps(config))
    with pytest.raises(inputs.InputError, match="Fixed geometry/views"):
        inputs.prepare_inputs(ROOT, data, output, calibration_size=2)


def test_archive_digest_must_match(suite):
    data, output = suite
    archive = inputs.HISTORICAL_ROOT / "_snapshot/mlmp_trace_sources.tar.gz"
    put(archive, b"Not the frozen archived source")
    with pytest.raises(inputs.InputError, match="checksum mismatch"):
        inputs.prepare_inputs(ROOT, data, output, calibration_size=2)


def test_cpu_import_does_not_load_tensor_or_model_packages():
    script = "from tools.evolution_tuning import inputs; import sys; assert not any(k == 'torch' or k.startswith(('torch.', 'trace_ovss.')) for k in sys.modules)"
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_default_256_sampler_matches_exact_seed_ranking_and_ignores_paths():
    def row(index, directory):
        return dict(official_id=f"coco:{index:012d}", image=f"/{directory}/{index:012d}.jpg",
                    label=f"/{directory}/{index:012d}.png", source_image=f"/{directory}/{index:012d}.jpg",
                    image_metadata=dict(size_bytes=7, mtime_ns=1, device=1, inode=index),
                    label_metadata=dict(size_bytes=8, mtime_ns=1, device=2, inode=index))

    training = [row(index, "train") for index in range(1, 301)]
    evaluation = [row(1001, "evaluation")]
    selected = inputs.select_calibration("coco_object", training, evaluation)
    expected = sorted(training, key=lambda r: (hashlib.sha256(
        f"trace-single-probe-tuning-v1\n2027\ncoco_object\n{r['official_id']}\n".encode()).hexdigest(), r["official_id"]))[:256]
    assert len(selected) == 256
    assert {r["official_id"] for r in selected} == {r["official_id"] for r in expected}
    reverse = inputs.select_calibration("coco_object", list(reversed(training)), evaluation)
    relocated = inputs.select_calibration("coco_object", [row(index, "relocated") for index in range(1, 301)], evaluation)
    assert [r["official_id"] for r in reverse] == [r["official_id"] for r in selected]
    assert [r["official_id"] for r in relocated] == [r["official_id"] for r in selected]
    with pytest.raises(inputs.InputError, match="own train/eval overlap"):
        inputs.select_calibration("coco_object", training, [row(1, "physically_distinct_copy")])


def test_complete_population_identity_digests_are_retained(suite):
    data, output = suite
    bundle = inputs.prepare_inputs(ROOT, data, output, calibration_size=2)
    for entry in bundle["datasets"].values():
        audit = entry["input_audit"]
        assert len(audit["training_ordered_source_ids_sha256"]) == 64
        assert audit["evaluation_ordered_source_ids_sha256"] == inputs.cohorts.value_hash(entry["evaluation"]["ordered_ids"])


def test_source_metadata_change_during_hashing_fails_closed(suite, monkeypatch):
    data, output = suite
    original = inputs._freeze_rows
    changed = False

    def mutate_after_first_panel(rows, cache):
        nonlocal changed
        result = original(rows, cache)
        if not changed:
            path = inputs.INHERITED_CONFIG_ROOT / "voc20.json"
            put(path, path.read_text() + "\n")
            changed = True
        return result

    monkeypatch.setattr(inputs, "_freeze_rows", mutate_after_first_panel)
    with pytest.raises(inputs.InputError, match="Metadata/source changed"):
        inputs.prepare_inputs(ROOT, data, output, calibration_size=2)
    assert not output.exists()


def test_waterbirds_default_equals_explicit_false_and_remains_blocked(suite):
    data, output = suite
    default = inputs.prepare_inputs(ROOT, data, output, calibration_size=2)
    explicit = inputs.prepare_inputs(ROOT, data, output, calibration_size=2,
                                     allow_waterbirds_development_overlap=False)
    assert default == explicit
    assert default["allow_waterbirds_development_overlap"] is False
    waterbirds = default["datasets"]["waterbirds_seg"]
    assert waterbirds["calibration"] is None and waterbirds["blocked_reason"]
    assert "development_overlap" not in waterbirds["selection"]
    assert waterbirds["evaluation"]["images"] == 4


@pytest.mark.parametrize("value", [None, 0, 1, 0.0, 1.0, "true", "false", [], {}])
def test_waterbirds_overlap_flag_requires_strict_bool_before_discovery(tmp_path, value):
    with pytest.raises(inputs.InputError, match="strict bool"):
        inputs.prepare_inputs(tmp_path / "missing_package", tmp_path / "missing_data", tmp_path / "out",
                              allow_waterbirds_development_overlap=value)


@pytest.mark.parametrize("size,seed", [(2, 2027), (255, 2027), (257, 2027), (256, 0), (256, 2028)])
def test_waterbirds_opt_in_keeps_bounded_256_seed2027_protocol(tmp_path, size, seed):
    with pytest.raises(inputs.InputError, match="calibration_size=256 and seed=2027"):
        inputs.prepare_inputs(tmp_path / "missing_package", tmp_path / "missing_data", tmp_path / "out",
                              calibration_size=size, seed=seed, allow_waterbirds_development_overlap=True)


@pytest.mark.parametrize("suite", ["waterbirds_development"], indirect=True)
def test_waterbirds_opt_in_256_train_sources_full5322_seed_stable_other_seven_unchanged(suite, monkeypatch):
    data, output = suite
    disjoint_selector = inputs.select_calibration

    def guarded_selector(dataset, *args, **kwargs):
        assert dataset != "waterbirds_seg", "Development overlap must not call the disjoint selector"
        return disjoint_selector(dataset, *args, **kwargs)

    monkeypatch.setattr(inputs, "select_calibration", guarded_selector)
    blocked = inputs.prepare_inputs(ROOT, data, output)
    enabled = inputs.prepare_inputs(ROOT, data, output, allow_waterbirds_development_overlap=True)
    repeated = inputs.prepare_inputs(ROOT, data, output, allow_waterbirds_development_overlap=True)
    assert enabled == repeated
    assert enabled["allow_waterbirds_development_overlap"] is True
    assert enabled["annotation_pixels_read"] is False
    assert enabled["offline_construction_performed"] is False
    assert enabled["global_never_seen_certified"] is False
    assert not output.exists()
    for dataset in inputs.DATASETS:
        if dataset != "waterbirds_seg":
            assert enabled["datasets"][dataset] == blocked["datasets"][dataset]

    waterbirds = enabled["datasets"]["waterbirds_seg"]
    calibration, evaluation = waterbirds["calibration"], waterbirds["evaluation"]
    assert "blocked_reason" not in waterbirds
    assert calibration["available"] is True and calibration["block_reason"] is None
    assert calibration["full_split"] is False and evaluation["full_split"] is True
    assert calibration["images"] == len(calibration["ordered_ids"]) == 256
    assert len(set(calibration["ordered_ids"])) == 256
    assert evaluation["images"] == len(evaluation["ordered_ids"]) == 5322
    assert len(set(evaluation["ordered_ids"])) == 5322
    assert evaluation["inputs"] == blocked["datasets"]["waterbirds_seg"]["evaluation"]["inputs"]
    assert evaluation["ordered_ids"] == blocked["datasets"]["waterbirds_seg"]["evaluation"]["ordered_ids"]
    train_path = data / "waterbirds_seg/correlation_0.95/splits/train.txt"
    train_ids = {f"cub:{stem}" for stem in train_path.read_text().splitlines()}
    assert len(train_ids) == 2458
    expected = sorted(train_ids, key=lambda identity: (hashlib.sha256(
        f"trace-single-probe-tuning-v1\n2027\nwaterbirds_seg\n{identity}\n".encode()).hexdigest(), identity))[:256]
    assert set(calibration["ordered_ids"]) == set(expected)
    assert set(calibration["ordered_ids"]) <= train_ids & set(evaluation["ordered_ids"])
    assert calibration["ordered_ids"] == sorted(calibration["ordered_ids"])
    for metadata in (calibration, evaluation, waterbirds["input_audit"], waterbirds["selection"]):
        assert metadata["protocol"] == metadata["evaluation_scope"] == "development_overlap"
        assert metadata["development_overlap"] is True
        assert metadata["allow_waterbirds_development_overlap"] is True
        assert metadata["cal_full_source_id_overlap"] == 256
        assert metadata["train_full_source_id_overlap"] == metadata["declared_train_count"] == 2458
        assert metadata["original_full_count"] == 5322
        assert metadata["independent_test_claim"] is False
        assert metadata["global_never_seen_certified"] is False
        assert "Development evaluation" in metadata["claim_boundary"]
    assert "development overlap" in waterbirds["selection"]["disjointness_scope"]
    assert waterbirds["input_audit"]["construction_performed"] is False
    assert waterbirds["input_audit"]["segmentation_label_contents_read"] is False


@pytest.mark.parametrize("corruption", ["duplicate_id", "path", "inode", "label", "label_alias", "outside_full"])
def test_waterbirds_development_overlap_does_not_bypass_identity_guards(suite, corruption):
    data, _ = suite
    root = data / "waterbirds_seg/correlation_0.95"
    layout = inputs.cohorts.Layout("waterbirds_seg", "cub", root, "images", "masks", ".jpg", ".png")
    evaluation = inputs.discover(layout, {}, expected_count=4)
    training = copy.deepcopy(evaluation[-2:])
    if corruption == "duplicate_id":
        training[1] = copy.deepcopy(training[0])
    elif corruption == "path":
        training[0]["image"] = "/another/source.jpg"
    elif corruption == "inode":
        training[0]["image_metadata"]["inode"] += 1000000
    elif corruption == "label":
        training[0]["label"] = "/another/label.png"
    elif corruption == "label_alias":
        for field in ("device", "inode"):
            training[0]["image_metadata"][field] = training[0]["label_metadata"][field]
    else:
        training[0]["official_id"] = "cub:001.Fixture/Outside_full"
    with pytest.raises(inputs.InputError):
        inputs._select_waterbirds_development_overlap(training, evaluation, size=2, seed=2027)
