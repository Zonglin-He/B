"""Synthetic, CPU-only identity/provenance tests; no annotation reads."""

import builtins
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "tools/evolution_confirmation/cohorts.py"
spec = importlib.util.spec_from_file_location("confirmation_cohort_tests", SOURCE)
cohorts = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cohorts
spec.loader.exec_module(cohorts)


def population(tmp_path, names, *, family="coco", dataset="coco_object", directory="source"):
    root = tmp_path / directory
    image_root, label_root = root / "images", root / "labels"
    image_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        image, label = image_root / f"{name}.jpg", label_root / f"{name}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"RGB {name}".encode())
        label.write_bytes(b"LABEL READ MUST NOT HAPPEN")
    layout = cohorts.Layout(dataset, family, root, "images", "labels", ".jpg", ".png")
    return layout, cohorts.discover_population(layout, {})


def names(start, count):
    return [f"{n:012d}" for n in range(start, start + count)]


def choose(train, evaluation, histories=(), **kwargs):
    return cohorts.select_cohorts("coco_object", train, {"coco_object": evaluation}, list(histories),
                                  calibration_size=2, confirmation_size=3, **kwargs)


def ids(rows):
    return [row["official_id"] for row in rows]


def test_fixed_hash_ranking_is_order_and_path_independent(tmp_path):
    _, train = population(tmp_path, names(1, 12))
    _, evaluation = population(tmp_path, names(101, 2), directory="eval")
    selected = choose(train, evaluation)
    reversed_selection = choose(list(reversed(train)), list(reversed(evaluation)))
    expected = sorted(train, key=lambda row: hashlib.sha256(
        f"trace-confirmation-v1\n20261029\ncoco_object\n{row['official_id']}\n".encode()).hexdigest())
    assert ids(selected["calibration"]) == ids(expected[:2])
    assert ids(selected["confirmation"]) == ids(expected[2:5])
    assert selected == reversed_selection
    _, relocated = population(tmp_path, names(1, 12), directory="relocated")
    again = choose(relocated, evaluation)
    assert ids(again["calibration"]) == ids(selected["calibration"])
    assert ids(again["confirmation"]) == ids(selected["confirmation"])
    assert set(ids(selected["calibration"])).isdisjoint(ids(selected["confirmation"]))


def test_default_sizes_are_128_and_256(tmp_path):
    _, train = population(tmp_path, names(1, 400))
    _, evaluation = population(tmp_path, names(1001, 2), directory="eval")
    selected = cohorts.select_cohorts("coco_object", train, {"coco_object": evaluation}, [])
    assert len(selected["calibration"]) == 128
    assert len(selected["confirmation"]) == 256
    assert selected["counts"]["unused_eligible"] == 16


def test_cross_dataset_eval_and_history_union_deduplicates(tmp_path):
    _, train = population(tmp_path, names(1, 12))
    _, evaluation = population(tmp_path, names(101, 2), directory="eval")
    _, cross_eval = population(tmp_path, names(1, 2), directory="cd_eval")
    histories = [{"rows": train[1:4]}, {"rows": train[2:4]}]
    selection = cohorts.select_cohorts("coco_object", train,
        {"coco_object": evaluation, "coco_cd": cross_eval}, histories,
        calibration_size=2, confirmation_size=3)
    assert selection["counts"]["official_evaluation"] == 2
    assert selection["counts"]["known_history"] == 3
    assert selection["counts"]["overlap_history_evaluation"] == 1
    assert selection["counts"]["excluded_union"] == 4
    assert selection["counts"]["eligible"] == 8
    assert not (set(ids(selection["calibration"] + selection["confirmation"])) & set(ids(train[:4])))


def test_own_train_eval_overlap_is_fatal_even_at_different_paths(tmp_path):
    _, train = population(tmp_path, names(1, 12))
    _, evaluation = population(tmp_path, names(1, 2), directory="eval_copy")
    with pytest.raises(cohorts.CohortError, match="train/eval overlap"):
        choose(train, evaluation)


def test_repeated_basename_in_nested_directories_is_not_two_ids(tmp_path):
    with pytest.raises(cohorts.CohortError, match="Repeated official identity"):
        population(tmp_path, ["a/000000000001", "b/000000000001"])


@pytest.mark.parametrize("alias_type", ["symlink", "hardlink"])
def test_discovery_rejects_physical_aliases(tmp_path, alias_type):
    layout, _ = population(tmp_path, names(1, 2))
    destination = layout.root / "images/000000000002.jpg"
    destination.unlink()
    original = layout.root / "images/000000000001.jpg"
    if alias_type == "symlink":
        destination.symlink_to(original)
    else:
        os.link(original, destination)
    with pytest.raises(cohorts.CohortError, match="alias"):
        cohorts.discover_population(layout, {})


def test_exclusion_path_alias_maps_canonically_and_stale_root_maps_by_id(tmp_path):
    _, train = population(tmp_path, names(1, 5))
    index = cohorts.identity_index({"train": train})
    alias = tmp_path / "unrelated-name.jpg"
    alias.symlink_to(train[0]["image"])
    assert cohorts.resolve_exclusion(str(alias), "coco", index)["official_id"] == train[0]["official_id"]
    assert cohorts.resolve_exclusion("/old/mount/000000000001.jpg", "coco", index)["official_id"] == train[0]["official_id"]
    assert cohorts.resolve_exclusion("coco:000000000001", "coco", index)["official_id"] == train[0]["official_id"]


def test_exclusion_alias_cannot_claim_a_different_official_id(tmp_path):
    _, train = population(tmp_path, names(1, 5))
    alias = tmp_path / "000000000002.jpg"
    alias.symlink_to(train[0]["image"])
    with pytest.raises(cohorts.CohortError, match="ambiguous|conflict"):
        cohorts.resolve_exclusion(str(alias), "coco", cohorts.identity_index({"train": train}))


def test_canonical_path_cannot_override_declared_source_family(tmp_path):
    _, train = population(tmp_path, names(1, 5))
    with pytest.raises(cohorts.CohortError, match="source-family"):
        cohorts.resolve_exclusion(train[0]["image"], "voc", cohorts.identity_index({"train": train}))


@pytest.mark.parametrize("identity", ["garbage", "coco:999999999999", "1", "voc:2008_000001"])
def test_unknown_or_unenumerated_identity_fails_closed(tmp_path, identity):
    _, train = population(tmp_path, names(1, 5))
    with pytest.raises(cohorts.CohortError):
        cohorts.resolve_exclusion(identity, "coco", cohorts.identity_index({"train": train}))


def test_source_bound_identity_conflict_fails_closed(tmp_path):
    _, train = population(tmp_path, names(1, 5))
    conflicting = copy.deepcopy(train[0])
    conflicting["official_id"] = "coco:999999999999"
    with pytest.raises(cohorts.CohortError, match="Conflicting"):
        cohorts.identity_index({"a": train, "b": [conflicting]})


def history_file(tmp_path, **updates):
    value = dict(dataset_key="coco_object", split="train", selected_size=2,
                 indices=[1, 2], sample_paths=["/old/000000000001.jpg", "/old/000000000002.jpg"])
    value.update(updates)
    path = tmp_path / "history.json"
    path.write_text(json.dumps(value))
    return path


def test_history_preserves_explicit_ids_and_hashes(tmp_path):
    _, train = population(tmp_path, names(1, 5))
    path = history_file(tmp_path)
    sources = {}
    record = cohorts.load_history_manifest(path, {"train": train}, sources)
    assert record["official_ids"] == ids(train[:2])
    assert sources[str(path)]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["source_values"] == json.loads(path.read_text())["sample_paths"]
    with pytest.raises(cohorts.CohortError, match="hash changed"):
        cohorts.load_history_manifest(path, {"train": train}, {}, expected="0" * 64)


@pytest.mark.parametrize("updates", [dict(sample_paths=None), dict(sample_paths=[]),
    dict(sample_paths=["000000000001", "999999999999"]), dict(selected_size=3),
    dict(indices=[1, 1]), dict(sample_paths=["000000000001", "000000000001"]),
    dict(dataset_key="unknown"), dict(split="evaluation")])
def test_incomplete_or_unknown_history_fails_closed(tmp_path, updates):
    _, train = population(tmp_path, names(1, 5))
    with pytest.raises(cohorts.CohortError):
        cohorts.load_history_manifest(history_file(tmp_path, **updates), {"train": train}, {})


def test_audit_disagreement_and_index_only_history_fail(tmp_path):
    _, train = population(tmp_path, names(1, 5))
    path = history_file(tmp_path)
    audited = dict(ordered_image_ids=["000000000002", "000000000001"], indices=[1, 2])
    with pytest.raises(cohorts.CohortError, match="disagrees"):
        cohorts.load_history_manifest(path, {"train": train}, {}, audited=audited)
    record = json.loads(path.read_text())
    del record["sample_paths"]
    path.write_text(json.dumps(record))
    with pytest.raises(cohorts.CohortError, match="index-only"):
        cohorts.load_history_manifest(path, {"train": train}, {})


def test_split_manifest_hash_count_duplicate_and_missing_pair(tmp_path):
    layout, _ = population(tmp_path, names(1, 5))
    split = layout.root / "train.txt"
    split.write_text("\n".join(names(1, 5)) + "\n")
    layout = cohorts.replace(layout, split_file="train.txt")
    sources = {}
    assert len(cohorts.discover_population(layout, sources, expected_count=5)) == 5
    assert str(split) in sources
    with pytest.raises(cohorts.CohortError, match="expected 6"):
        cohorts.discover_population(layout, {}, expected_count=6)
    split.write_text("000000000001\n000000000001\n")
    with pytest.raises(cohorts.CohortError, match="Repeated split"):
        cohorts.discover_population(layout, {})
    split.write_text("../000000000001\n")
    with pytest.raises(cohorts.CohortError, match="Unsafe"):
        cohorts.discover_population(layout, {})
    split.write_text("000000000999\n")
    with pytest.raises(cohorts.CohortError, match="Incomplete declared"):
        cohorts.discover_population(layout, {})


def test_label_read_sentinel_and_rgb_hash_contract(tmp_path, monkeypatch):
    layout, original = population(tmp_path, names(1, 8))
    _, evaluation = population(tmp_path, names(101, 2), directory="eval")
    real_builtin, real_io = builtins.open, io.open

    def guard(real):
        def checked(file, *args, **kwargs):
            if isinstance(file, (str, os.PathLike)) and str(file).endswith(".png"):
                pytest.fail("Label content was opened before prediction seal")
            return real(file, *args, **kwargs)
        return checked

    monkeypatch.setattr(builtins, "open", guard(real_builtin))
    monkeypatch.setattr(io, "open", guard(real_io))
    rows = cohorts.discover_population(layout, {})
    selection = choose(rows, evaluation)
    output = cohorts._selected_row(selection["calibration"][0])
    assert set(output) == {"official_id", "image", "label", "image_sha256", "label_metadata"}
    assert output["image_sha256"] == hashlib.sha256(Path(output["image"]).read_bytes()).hexdigest()
    assert output["label_metadata"]["contents_read"] is False
    assert rows == original


def test_changed_rgb_during_preparation_fails(tmp_path):
    _, rows = population(tmp_path, names(1, 2))
    Path(rows[0]["image"]).write_bytes(b"replacement")
    with pytest.raises(cohorts.CohortError, match="RGB changed"):
        cohorts._selected_row(rows[0])


def test_label_metadata_change_and_empty_pair_fail_without_label_read(tmp_path):
    layout, rows = population(tmp_path, names(1, 2))
    Path(rows[0]["label"]).write_bytes(b"different metadata length")
    with pytest.raises(cohorts.CohortError, match="label metadata changed"):
        cohorts._selected_row(rows[0])
    Path(rows[0]["label"]).write_bytes(b"")
    with pytest.raises(cohorts.CohortError, match="Empty declared"):
        cohorts.discover_population(layout, {})


def test_duplicate_rows_are_rejected_by_reusable_selector(tmp_path):
    _, train = population(tmp_path, names(1, 6))
    _, evaluation = population(tmp_path, names(101, 2), directory="eval")
    with pytest.raises(cohorts.CohortError, match="Repeated official identity"):
        choose(train + train[:1], evaluation)


def test_missing_eval_and_insufficient_population_fail(tmp_path):
    _, train = population(tmp_path, names(1, 4))
    _, evaluation = population(tmp_path, names(101, 2), directory="eval")
    with pytest.raises(cohorts.CohortError, match="Missing official evaluation"):
        cohorts.select_cohorts("coco_object", train, {}, [])
    with pytest.raises(cohorts.CohortError, match="Insufficient"):
        choose(train, evaluation)


def test_repository_metadata_adapter_uses_actual_specs_without_imports(tmp_path):
    sources = {}
    layouts, counts = cohorts.repository_layouts(SOURCE.parents[2], tmp_path, sources,
        {key: tmp_path for key in ("pc60", "coco_object", "voc21", "coco_cd")})
    pc = cohorts.training_layout(layouts["pc60"])
    coco = cohorts.training_layout(layouts["coco_object"])
    assert pc.split_file == "ImageSets/SegmentationContext/train.txt"
    assert pc.mask_dir == "SegmentationClassContext"
    assert coco.image_dir == "images/train2017"
    assert coco.mask_suffix == "_instanceTrainIds.png"
    assert counts["pc60"] == 5105 and counts["coco_object"] == 5000
    assert len(sources) == 3


def test_metadata_discovery_has_no_torch_cuda_numpy_or_pil_imports(tmp_path):
    layout, _ = population(tmp_path, names(1, 2))
    script = r'''
import importlib.abc, importlib.util, pathlib, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'numpy', 'PIL', 'trace_ovss'} or 'cuda' in fullname.lower():
            raise AssertionError('Forbidden metadata import: ' + fullname)
sys.meta_path.insert(0, Block())
spec = importlib.util.spec_from_file_location('cohorts', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
root = pathlib.Path(sys.argv[2])
layout = module.Layout('coco_object', 'coco', root, 'images', 'labels', '.jpg', '.png')
rows = module.discover_population(layout, {})
assert len(rows) == 2
module._selected_row(rows[0])
module.repository_layouts(pathlib.Path(sys.argv[1]).parents[2], root, {},
    {key: root for key in ('pc60', 'coco_object', 'voc21', 'coco_cd')})
'''
    result = subprocess.run([sys.executable, "-B", "-c", script, str(SOURCE), str(layout.root)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_bounded_metadata_read_and_changed_source(tmp_path):
    path = tmp_path / "oversize.json"
    path.write_bytes(b"x" * 16)
    with pytest.raises(cohorts.CohortError, match="exceeds"):
        cohorts.read_metadata(path, {}, limit=15)


def test_metadata_and_rgb_paths_cannot_alias_label_files(tmp_path):
    label = tmp_path / "label.png"
    label.write_bytes(b"not to be read")
    alias = tmp_path / "history.json"
    alias.symlink_to(label)
    with pytest.raises(cohorts.CohortError, match="textual metadata"):
        cohorts.read_metadata(alias, {})
    layout, rows = population(tmp_path, names(1, 2))
    image = Path(rows[0]["image"])
    image.unlink()
    image.symlink_to(rows[0]["label"])
    with pytest.raises(cohorts.CohortError, match="file type"):
        cohorts.discover_population(layout, {})


def test_hardlinked_image_and_label_is_fatal(tmp_path):
    layout, rows = population(tmp_path, names(1, 2))
    image = Path(rows[0]["image"])
    image.unlink()
    os.link(rows[0]["label"], image)
    rows = cohorts.discover_population(layout, {})
    with pytest.raises(cohorts.CohortError, match="RGB/label physical"):
        cohorts.identity_index({"train": rows})


def test_fresh_json_bundle_never_overwrites_or_creates_parent(tmp_path):
    path = tmp_path / "cohorts.json"
    bundle = {"datasets": {"pc60": {"calibration": [], "confirmation": []}}}
    cohorts.write_bundle(bundle, path)
    assert json.loads(path.read_text()) == bundle
    with pytest.raises(FileExistsError):
        cohorts.write_bundle({}, path)
    missing = tmp_path / "not-created/cohorts.json"
    with pytest.raises(cohorts.CohortError, match="existing directory"):
        cohorts.write_bundle(bundle, missing)
    assert not missing.parent.exists()
    alias = tmp_path / "dangling.json"
    alias.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        cohorts.write_bundle(bundle, alias)


def test_cli_requires_fresh_output_before_preparing(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Preparation reached before checking fresh explicit output")
    monkeypatch.setattr(cohorts, "prepare_cohorts", forbidden)
    with pytest.raises(SystemExit):
        cohorts.main(["--data-root", str(tmp_path)])
    path = tmp_path / "existing.json"
    path.write_text("original")
    with pytest.raises(FileExistsError):
        cohorts.main(["--output", str(path), "--data-root", str(tmp_path)])
    assert path.read_text() == "original"
    with pytest.raises(SystemExit):
        cohorts.main(["--output", str(tmp_path / "missing/out.json"), "--data-root", str(tmp_path)])
    assert not (tmp_path / "missing").exists()


def test_cli_exposes_exact_worker_bundle_and_explicit_source_override(tmp_path, monkeypatch):
    bundle = dict(datasets={"pc60": dict(calibration=[], confirmation=[], counts={})},
                  exclusions={}, source_hashes={}, audit={})
    calls = []
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return bundle
    monkeypatch.setattr(cohorts, "prepare_cohorts", prepare)
    path = tmp_path / "fresh.json"
    cohorts.main(["--output", str(path), "--data-root", str(tmp_path),
                  "--data-dir", "coco_object=/sealed/source"])
    assert json.loads(path.read_text()) == bundle
    assert calls[0][1]["data_dirs"] == {"coco_object": "/sealed/source"}


def test_pool_inspection_uses_hashed_archived_loader_not_current_order(tmp_path):
    snapshot = tmp_path / "_snapshot"
    snapshot.mkdir()
    archive = snapshot / "mlmp_trace_sources.tar.gz"
    contents = {
        "external/MLMP/utils/segmentation_datasets.py": b"def prepare_data(dataset, shuffle=True):\n    pass\n",
        "external/MLMP/utils/simple_seg_dataset.py": b"# archived sorted source; no historical train ID list\n",
    }
    with cohorts.tarfile.open(archive, "w:gz") as handle:
        for name, content in contents.items():
            member = cohorts.tarfile.TarInfo(name)
            member.size = len(content)
            handle.addfile(member, io.BytesIO(content))
    sums = snapshot / "SHA256SUMS"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    sums.write_text(f"{digest}  {archive}\n")
    builder = tmp_path / "build_trace_v2_calibration_splits.py"
    builder.write_text("prepare_data('pc60', data_split='train')\n")
    audit = dict(historical_root=str(tmp_path), source_files=[dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                                                           for path in (sums, builder)])
    hashes = {}
    result = cohorts.inspect_pool_history(audit, hashes)
    assert result["builder_requests_data_split_train"] is True
    assert result["archived_loader_has_data_split"] is False
    assert result["candidate_pool_ids_reconstructed"] is False
    assert result["strong_independent_confirmation_supported"] is False
    assert str(archive) in hashes
    assert all(f"{archive}::{name}" in hashes for name in contents)


def test_recent_history_does_not_open_large_referenced_inputs(tmp_path, monkeypatch):
    root = tmp_path / "recent"
    root.mkdir()
    input_path = root / "large-input.json"
    input_path.write_bytes(b"x" * (cohorts.SMALL_FILE_LIMIT + 1))
    (root / "README.md").write_text("256-image train calibration")
    (root / "manifest.json").write_text(json.dumps(dict(calibration=dict(count=256, split="train"),
        training_inputs={}, training_inputs_source=None,
        inputs={"coco_object": dict(path=str(input_path), sha256="declared-not-verified")})))
    (root / "queue_state.json").write_text(json.dumps(dict(status="completed")))
    real = io.open
    def guarded(file, *args, **kwargs):
        if str(file) == str(input_path):
            pytest.fail("Large input payload opened outside bounded history scope")
        return real(file, *args, **kwargs)
    monkeypatch.setattr(io, "open", guarded)
    hashes = {}
    records = cohorts.inspect_recent_history([root], hashes)
    assert records[0]["explicit_training_ids_mapped"] is False
    assert records[0]["declared_calibration_count"] == 256
    assert records[0]["referenced_inputs"][0]["contents_opened"] is False
    assert str(input_path) not in hashes


def recent_input_fixture(tmp_path, *, dataset="coco_object", padding=0, train_count=6):
    _, train = population(tmp_path, names(1, train_count))
    _, evaluation = population(tmp_path, names(1001, 2), directory="eval")
    suffix = "_instanceTrainIds.png" if dataset == "coco_object" else "_labelTrainIds.png"
    indices = [4, 1]
    calibration = [dict(image_id=train[index]["image"], stem=Path(train[index]["image"]).stem,
                        image_sha256="historical-rgb-digest-not-rehashed",
                        mask_path=str(tmp_path / "historical-labels" / (Path(train[index]["image"]).stem + suffix)),
                        mask_sha256="historical-label-digest-not-rehashed") for index in indices]
    record = dict(dataset=dataset, train_count=len(train), evaluation_count=len(evaluation),
                  train_spec=dict(image_dir="images/train2017", mask_suffix=suffix),
                  evaluation_spec=dict(image_dir="images/val2017"),
                  train_ids=[row["image"] for row in train], calibration_indices=indices,
                  selection_seed=2027, calibration=calibration,
                  evaluation=[dict(image_id=row["image"]) for row in evaluation], padding="x" * padding)
    path = tmp_path / "recent-input.json"
    path.write_text(json.dumps(record))
    populations = {"coco_object/train": train, "coco_object/evaluation": evaluation}
    return path, record, populations


def load_recent(path, populations, hashes=None, *, dataset="coco_object", expected=None):
    digest = expected or hashlib.sha256(path.read_bytes()).hexdigest()
    return cohorts.load_recent_history_manifest(path, populations, {} if hashes is None else hashes,
        dataset=dataset, expected=digest, expected_count=2, selection_seed=2027)


@pytest.mark.parametrize("dataset", ["coco_object", "coco_stuff"])
def test_recent_explicit_rows_use_archived_positions_not_current_order(tmp_path, monkeypatch, dataset):
    path, record, populations = recent_input_fixture(tmp_path, dataset=dataset,
                                                   padding=cohorts.SMALL_FILE_LIMIT + 1)
    original = list(populations["coco_object/train"])
    populations["coco_object/train"].reverse()
    real = io.open
    def sentinel(file, *args, **kwargs):
        if str(file).endswith(".png"):
            pytest.fail("Historical digest metadata must not cause a label open")
        return real(file, *args, **kwargs)
    monkeypatch.setattr(io, "open", sentinel)
    hashes = {}
    result = load_recent(path, populations, hashes, dataset=dataset)
    assert result["official_ids"] == [original[4]["official_id"], original[1]["official_id"]]
    assert result["historical_indices"] == [4, 1]
    assert result["mappings"][0]["archived_train_id"] == record["train_ids"][4]
    assert result["mappings"][0]["source_label"] == record["calibration"][0]["mask_path"]
    assert result["source_population_match"] is True and result["own_train_eval_disjoint"] is True
    assert result["official_ids_sha256"] == cohorts.value_hash(result["official_ids"])
    assert hashes[str(path)]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("damage", ["wrong_dataset", "wrong_seed", "wrong_spec", "missing_training_ids",
    "unknown_training_id", "duplicate_training_id", "short_calibration", "duplicate_calibration",
    "archived_index_disagrees", "index_out_of_range", "boolean_index", "duplicate_indices",
    "unknown_explicit_id", "wrong_stem", "wrong_mask_id", "evaluation_overlap", "short_evaluation"])
def test_recent_unknown_or_inconsistent_metadata_fails_closed(tmp_path, damage):
    path, record, populations = recent_input_fixture(tmp_path)
    if damage == "wrong_dataset":
        record["dataset"] = "coco_stuff"
    elif damage == "wrong_seed":
        record["selection_seed"] += 1
    elif damage == "wrong_spec":
        record["train_spec"]["image_dir"] = "images/unknown"
    elif damage == "missing_training_ids":
        del record["train_ids"]
    elif damage == "unknown_training_id":
        record["train_ids"][0] = "/old/train2017/999999999999.jpg"
    elif damage == "duplicate_training_id":
        record["train_ids"][0] = record["train_ids"][1]
    elif damage == "short_calibration":
        record["calibration"].pop()
    elif damage == "duplicate_calibration":
        record["calibration"][1] = copy.deepcopy(record["calibration"][0])
    elif damage == "archived_index_disagrees":
        record["calibration_indices"][0] = 0
    elif damage == "index_out_of_range":
        record["calibration_indices"][0] = 6
    elif damage == "boolean_index":
        record["calibration_indices"][0] = True
    elif damage == "duplicate_indices":
        record["calibration_indices"][0] = record["calibration_indices"][1]
    elif damage == "unknown_explicit_id":
        record["calibration"][0]["image_id"] = "/old/999999999999.jpg"
    elif damage == "wrong_stem":
        record["calibration"][0]["stem"] = "000000000006"
    elif damage == "wrong_mask_id":
        record["calibration"][0]["mask_path"] = "/old/000000000006_instanceTrainIds.png"
    elif damage == "evaluation_overlap":
        record["evaluation"][0]["image_id"] = record["train_ids"][0]
    elif damage == "short_evaluation":
        record["evaluation"].pop()
    path.write_text(json.dumps(record))
    with pytest.raises(cohorts.CohortError):
        load_recent(path, populations)


def test_recent_parent_hash_and_16mib_bound_are_mandatory(tmp_path, monkeypatch):
    path, record, populations = recent_input_fixture(tmp_path)
    with pytest.raises(cohorts.CohortError, match="parent-declared"):
        load_recent(path, populations, expected="not-a-sha256")
    with pytest.raises(cohorts.CohortError, match="hash changed"):
        load_recent(path, populations, expected="0" * 64)
    with path.open("wb") as handle:
        handle.truncate(cohorts.RECENT_INPUT_LIMIT + 1)
    real = io.open
    def sentinel(file, *args, **kwargs):
        if str(file) == str(path):
            pytest.fail("Oversized metadata must be rejected before opening")
        return real(file, *args, **kwargs)
    monkeypatch.setattr(io, "open", sentinel)
    with pytest.raises(cohorts.CohortError, match="exceeds"):
        load_recent(path, populations, expected="0" * 64)


def test_recent_intersections_report_old_cohorts_and_deduplicated_exclusions(tmp_path):
    _, train = population(tmp_path, names(1, 400))
    _, own_eval = population(tmp_path, names(1001, 2), directory="eval")
    evaluation = {"coco_object": own_eval, "coco_cd": [train[1]]}
    previous = [{"rows": [train[0]]}]
    old = cohorts.select_cohorts("coco_object", train, evaluation, previous)
    new_a, new_b = old["calibration"][10], old["confirmation"][20]
    recent = [{"rows": [train[0], train[1], new_a, new_b]}, {"rows": [new_a, new_b]}]
    result = cohorts.recent_intersections("coco_object", train, evaluation, previous, recent)
    assert result["previous_known_history_intersection_ids"] == ids([train[0]])
    assert result["official_evaluation_intersection_ids"] == ids([train[1]])
    assert result["newly_excluded_ids"] == sorted(ids([new_a, new_b]))
    assert result["previous_calibration_intersection_ids"] == ids([new_a])
    assert result["previous_confirmation_intersection_ids"] == ids([new_b])
    assert result["previous_source_intersections"][0]["official_ids"] == ids([train[0]])
    assert result["evaluation_source_intersections"] == [dict(dataset="coco_cd", official_ids=ids([train[1]]))]
    current = cohorts.select_cohorts("coco_object", train, evaluation, previous + recent)
    assert current["counts"]["excluded_union"] == old["counts"]["excluded_union"] + 2
    assert set(ids(current["calibration"] + current["confirmation"])).isdisjoint(ids([new_a, new_b]))


def test_prepare_automatically_unions_both_parent_bound_recent_panels(tmp_path, monkeypatch):
    path, record, populations = recent_input_fixture(tmp_path, train_count=400)
    stuff_path = tmp_path / "recent-stuff.json"
    stuff = copy.deepcopy(record)
    stuff["dataset"] = "coco_stuff"
    stuff["train_spec"]["mask_suffix"] = "_labelTrainIds.png"
    for row in stuff["calibration"]:
        row["mask_path"] = row["mask_path"].replace("_instanceTrainIds", "_labelTrainIds")
    stuff_path.write_text(json.dumps(stuff))
    _, pc_train = population(tmp_path, [f"2008_{i:06d}" for i in range(400)], family="voc",
                             dataset="pc60", directory="pc_source")
    _, pc_eval = population(tmp_path, ["2009_000001", "2009_000002"], family="voc",
                            dataset="pc60", directory="pc_eval")
    train = {"pc60": pc_train, "voc21": pc_train[:2], "coco_object": populations["coco_object/train"]}
    evaluation = {"pc60": pc_eval, "voc21": pc_eval, "coco_object": populations["coco_object/evaluation"],
                  "coco_cd": populations["coco_object/evaluation"]}
    layouts = {key: cohorts.Layout(key, cohorts.FAMILIES[key], tmp_path, "images/val2017", "labels", ".jpg", ".png",
                                   "ImageSets/Segmentation/val.txt" if key.startswith(("pc", "voc")) else None)
               for key in ("pc60", "voc21", "coco_object", "coco_cd")}
    audit_path = tmp_path / "synthetic-audit.json"
    manifests = {key: dict(path=str(tmp_path / f"{key}.json")) for key in cohorts.HISTORY_KEYS}
    manifests["pc60"].update(ordered_image_ids=[pc_train[0]["official_id"]], pool_size=300)
    audit = dict(calibration_manifests=manifests,
                 source_files=[dict(path=value["path"], sha256="0" * 64) for value in manifests.values()])
    audit_path.write_text(json.dumps(audit))
    recent = []
    for dataset, input_path in (("coco_object", path), ("coco_stuff", stuff_path)):
        recent.append(dict(root=str(tmp_path / dataset), manifest_path=str(tmp_path / dataset / "manifest.json"),
                           declared_calibration_count=2, declared_selection_seed=2027,
                           referenced_inputs=[dict(dataset=dataset, path=str(input_path),
                               declared_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest())]))
    monkeypatch.setattr(cohorts, "inspect_pool_history", lambda *args: {"status": "unresolved"})
    monkeypatch.setattr(cohorts, "inspect_recent_history", lambda *args: recent)
    monkeypatch.setattr(cohorts, "repository_layouts", lambda *args: (layouts, {key: len(rows) for key, rows in evaluation.items()}))
    monkeypatch.setattr(cohorts, "TRAIN_COUNTS", {key: len(rows) for key, rows in train.items()})
    def discover(layout, *args, **kwargs):
        is_train = "train2017" in layout.image_dir or bool(layout.split_file and layout.split_file.endswith("train.txt"))
        return train[layout.dataset] if is_train else evaluation[layout.dataset]
    monkeypatch.setattr(cohorts, "discover_population", discover)
    def july(path, *args, **kwargs):
        key = Path(path).stem
        rows = populations["coco_object/train"][:1] if key.startswith("coco") else pc_train[:1]
        return dict(path=str(path), rows=rows)
    monkeypatch.setattr(cohorts, "load_history_manifest", july)
    real = io.open
    def sentinel(file, *args, **kwargs):
        if str(file).endswith(".png"):
            pytest.fail("Cohort preparation opened label bytes")
        return real(file, *args, **kwargs)
    monkeypatch.setattr(io, "open", sentinel)
    bundle = cohorts.prepare_cohorts(SOURCE.parents[2], tmp_path, audit_path=audit_path)
    assert bundle["datasets"]["coco_object"]["counts"]["known_history"] == 3
    assert bundle["datasets"]["coco_object"]["counts"]["eligible"] == 397
    assert bundle["audit"]["recent_cross_source_intersections"][0]["count"] == 2
    assert bundle["audit"]["recent_cross_source_intersections"][0]["same_archived_indices"] is True
    assert all(row["explicit_training_ids_mapped"] for row in bundle["audit"]["recent_history_audit"])
    recent_ids = {row["official_id"] for row in populations["coco_object/train"] if row["image"] in
                  {entry["image_id"] for entry in record["calibration"]}}
    selected = bundle["datasets"]["coco_object"]
    assert recent_ids <= set(bundle["exclusions"]["coco_object"]["known_history_ids"])
    assert recent_ids.isdisjoint(ids(selected["calibration"] + selected["confirmation"]))
    assert bundle["audit"]["strong_independent_confirmation_supported"] is False
    assert bundle["selection"]["seed"] == 20261029
    assert len(selected["calibration"]) == 128 and len(selected["confirmation"]) == 256


def test_documented_recent_index_identity_map_matches_fingerprints():
    text = (SOURCE.parents[2] / "docs/confirmation-cohort-audit.md").read_text()
    section = text.split("<!-- recent-index-identity-map -->", 1)[1].split("<!-- /recent-index-identity-map -->", 1)[0]
    tokens = section.split("```text\n", 1)[1].split("```", 1)[0].split()
    pairs = [dict(archived_train_index=int(token.split(":")[0]), official_id="coco:" + token.split(":")[1]) for token in tokens]
    assert len(pairs) == 256 and len({row["official_id"] for row in pairs}) == 256
    assert cohorts.value_hash(pairs) == "8b8d840661184356124b8e8ada9d96e617fa677a3f817e0910c72932ab25748b"
    assert cohorts.value_hash([row["official_id"] for row in pairs]) == "2a979fcdf07e85c5fd35818e361916265b8ef07e54d608e3de6aec158e4f309b"


def test_strong_independence_request_fails_before_population_discovery(tmp_path, monkeypatch):
    monkeypatch.setattr(cohorts, "read_metadata", lambda *args, **kwargs: "{}")
    monkeypatch.setattr(cohorts, "inspect_pool_history", lambda *args: {"status": "unresolved"})
    monkeypatch.setattr(cohorts, "inspect_recent_history", lambda *args: [{"explicit_training_ids_mapped": False}])
    def forbidden(*args, **kwargs):
        pytest.fail("Strong unsupported claim reached population discovery")
    monkeypatch.setattr(cohorts, "repository_layouts", forbidden)
    with pytest.raises(cohorts.CohortError, match="Independent-confirmation claim unsupported"):
        cohorts.prepare_cohorts(tmp_path, tmp_path, require_independent=True)
