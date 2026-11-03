"""CPU-only checks for matched VOC21 analysis and dataset-level class deltas."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_voc21_tuning import (Cell, Evidence, aggregate_counts, analyze, class_deltas,
    compare_weights, digest, load_cell, load_vocabulary, normalized_weights, read_candidates,
    spearman, total_variation, verify_frozen_inputs, verify_match, vocabulary_hash)


def identity():
    return dict(dataset="voc21", split="evaluation", image_ids=["a", "b"],
        label_ids=["a.png", "b.png"], vocabulary_sha256="vocab", seed=0,
        selection_seed=20261024, available_images=1449,
        config=dict(dataset="PascalVOC21Dataset", batch_size=16,
                    trace_view_batch_size=16, data_dir="/original"))


@pytest.mark.parametrize("field,value", [
    ("dataset", "voc20"), ("split", "calibration"), ("image_ids", ["b", "a"]),
    ("label_ids", ["wrong.png", "b.png"]), ("vocabulary_sha256", "different"),
    ("seed", 1), ("selection_seed", 2), ("available_images", 1448),
])
def test_identity_mismatches_are_rejected(field, value):
    first, other = identity(), identity()
    other[field] = value
    with pytest.raises(ValueError, match=field):
        verify_match(first, other)


@pytest.mark.parametrize("field,value", [("batch_size", 1), ("data_dir", "/changed"),
                                         ("trace_view_batch_size", 8)])
def test_configuration_mismatches_are_rejected(field, value):
    first, other = identity(), identity()
    other["config"][field] = value
    with pytest.raises(ValueError, match="configuration"):
        verify_match(first, other)
    if field == "trace_view_batch_size":
        verify_match(first, other, allow_view_batch_difference=True)
    else:
        with pytest.raises(ValueError, match="configuration"):
            verify_match(first, other, allow_view_batch_difference=True)


def metric_cell(records):
    intersection, union, iou, miou = aggregate_counts(records, classes=2)
    return Cell(Path("unused"), {}, dict(images=len(records)), intersection, union, iou, dict(miou=miou))


def test_class_deltas_use_ratio_of_summed_counts_and_preserve_row_order():
    full = metric_cell([dict(intersection=[1, 0], union=[1, 0]),
                        dict(intersection=[1, 4], union=[9, 8])])
    raw = metric_cell([dict(intersection=[0, 1], union=[2, 2]),
                       dict(intersection=[1, 1], union=[3, 2])])
    rows = class_deltas(full, raw, ["background", "cat"], "no_source_tracing")
    assert [row["class_name"] for row in rows] == ["background", "cat"]
    assert rows[0]["full_iou_pct"] == 20
    assert rows[0]["control_iou_pct"] == 20
    assert rows[0]["delta_full_minus_control_pp"] == 0
    assert rows[1]["full_iou_pct"] == 50
    assert rows[1]["control_iou_pct"] == 50
    assert rows[0]["full_iou_pct"] != pytest.approx(100 * (1 + 1 / 9) / 2)
    assert rows[0]["full_union"] == 10
    assert rows[0]["images"] == 2 and rows[0]["full_split"] is False


def test_class_delta_direction_and_absent_classes_are_explicit():
    full = metric_cell([dict(intersection=[1, 0], union=[4, 0])])
    raw = metric_cell([dict(intersection=[1, 0], union=[2, 0])])
    rows = class_deltas(full, raw, ["background", "cat"], "control")
    assert rows[0]["delta_full_minus_control_pp"] == -25
    assert rows[1]["full_iou_pct"] is None
    assert rows[1]["delta_full_minus_control_pp"] is None


@pytest.mark.parametrize("record", [
    dict(intersection=[2, 0], union=[1, 0]),
    dict(intersection=[0.5, 0], union=[1, 0]),
    dict(intersection=[-1, 0], union=[1, 0]),
    dict(intersection=[float("nan"), 0], union=[1, 0]),
    dict(intersection=[0], union=[1, 0]),
])
def test_invalid_count_axes_and_values_are_rejected(record):
    with pytest.raises(ValueError):
        aggregate_counts([record], classes=2)


def test_vocabulary_is_resource_order_with_background_and_query_mapping(tmp_path):
    path = tmp_path / "trace_ovss/resources/class_names/cls_voc21.txt"
    path.parent.mkdir(parents=True)
    path.write_text("sky, wall\n" + "\n".join(f"class{index}" for index in range(1, 21)))
    vocabulary = load_vocabulary(tmp_path, Evidence())
    assert vocabulary["classes"] == ["background"] + [f"class{index}" for index in range(1, 21)]
    assert vocabulary["query_to_class"][:3] == [0, 0, 1]
    assert vocabulary["queries"][:2] == ["sky", "wall"]
    expected = hashlib.sha256(json.dumps([vocabulary["queries"], vocabulary["query_to_class"]]).encode()).hexdigest()
    assert vocabulary_hash(vocabulary, True) == expected


def completed_cell(tmp_path):
    package = tmp_path / "runtime/trace_ovss"
    package.mkdir(parents=True)
    source = package / "__init__.py"
    source.write_text("# frozen\n")
    directory = tmp_path / "cell"
    directory.mkdir()
    protocol = dict(**identity(), images=2, full_split=False, variant="full",
                    package_root=str(package.parent), source_hashes={"__init__.py": digest(source)})
    records = [dict(image_id="a", intersection=[1, 1], union=[2, 2]),
               dict(image_id="b", intersection=[2, 1], union=[2, 2])]
    path = directory / "protocol.json"
    path.write_text(json.dumps(protocol))
    result = dict(status="completed", dataset="voc21", split="evaluation", images=2, full_split=False,
        protocol_sha256=digest(path), intersection=[3, 2], union=[4, 4],
        miou_unrounded=62.5, metrics=dict(mIoU=62.5))
    (directory / "result.json").write_text(json.dumps(result))
    (directory / "per_image_counts.json").write_text(json.dumps(records))
    return directory, source, protocol, result, records


def test_load_cell_verifies_counts_protocol_and_recorded_source(tmp_path):
    directory, source, _, _, _ = completed_cell(tmp_path)
    cell = load_cell(directory, 2, False, "full", Evidence(), classes=2)
    assert cell.audit["miou"] == 62.5
    source.write_text("# changed\n")
    with pytest.raises(ValueError, match="Source hash"):
        load_cell(directory, 2, False, "full", Evidence(), classes=2)


@pytest.mark.parametrize("kind", ["protocol", "order", "counts", "miou", "scope", "size"])
def test_load_cell_rejects_inconsistent_artifacts(tmp_path, kind):
    directory, _, protocol, result, records = completed_cell(tmp_path)
    if kind == "protocol":
        protocol["seed"] = 8
        (directory / "protocol.json").write_text(json.dumps(protocol))
    elif kind == "order":
        (directory / "per_image_counts.json").write_text(json.dumps(records[::-1]))
    elif kind == "counts":
        result["intersection"][0] = 4
    elif kind == "miou":
        result["miou_unrounded"] += 1
    elif kind == "scope":
        result["full_split"] = True
    elif kind == "size":
        result["images"] = 512
    (directory / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        load_cell(directory, 2, False, "full", Evidence(), classes=2)


def test_evidence_changes_and_metadata_mismatches_are_rejected(tmp_path):
    path = tmp_path / "evidence"
    path.write_text("frozen")
    evidence = Evidence()
    evidence.read(path)
    with pytest.raises(ValueError, match="metadata"):
        evidence.check_metadata(path, [0, 0])
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed during"):
        evidence.recheck()


def test_pre_run_companion_is_used_without_inventing_cell_metadata(tmp_path):
    directory, _, _, _, _ = completed_cell(tmp_path)
    cell = load_cell(directory, 2, False, "full", Evidence(), classes=2)
    expected = identity()
    paths = []
    for name in ("a", "b", "a.png", "b.png"):
        path = tmp_path / name
        path.write_text(name)
        paths.append(str(path))
    expected.update(image_ids=paths[:2], label_ids=paths[2:])
    cell.protocol.update(image_ids=paths[:2], label_ids=paths[2:])
    expected["file_metadata"] = [[Path(path).stat().st_size, Path(path).stat().st_mtime_ns]
                                 for pair in zip(paths[:2], paths[2:]) for path in pair]
    manifest = tmp_path / "suite_protocol.json"
    manifest.write_text("frozen")
    import os
    stamp = (directory / "protocol.json").stat().st_mtime_ns - 1
    os.utime(manifest, ns=(stamp, stamp))
    verify_frozen_inputs(cell, expected, manifest, Evidence())
    assert "original worker did not embed" in cell.audit["input_metadata_basis"]
    expected["file_metadata"][0][0] += 1
    with pytest.raises(ValueError, match="metadata mismatch"):
        verify_frozen_inputs(cell, expected, manifest, Evidence())


def test_spearman_ties_constants_and_order():
    assert spearman([1, 2, 2], [1, 2, 3]) == pytest.approx(3 ** 0.5 / 2)
    assert spearman([3, 2, 1], [1, 2, 3]) == -1
    assert spearman([1, 1], [1, 2]) is None
    assert spearman([1], [2]) is None


def candidate_fixture():
    values = [(0, "a", 0, 0.2, 0.8), (0, "b", 1, 0.8, 0.2), (1, "c", 1, 0.5, 0.5)]
    full, raw = {}, {}
    for episode, image, cls, rho, confidence in values:
        key = (episode, image, 0, cls, "a" * 64)
        full[key] = dict(episode=episode, evaluation_class_idx=cls, area=2,
                         reliability=rho, quality=rho, dominance=0.0)
        raw[key] = dict(full[key], reliability=confidence)
    return dict(full=full, no_source_tracing=raw)


def test_weight_tv_is_episode_normalized_and_class_mass_uses_absent_zeroes():
    summary, episodes, classes = compare_weights(candidate_fixture(), dict(classes=["background", "cat"]))
    assert episodes[0]["normalized_weight_tv"] == pytest.approx(0.6)
    assert episodes[1]["normalized_weight_tv"] == 0
    assert summary["normalized_weight_tv"]["mean"] == pytest.approx(0.3)
    assert classes[0]["delta_full_minus_raw_loss_weight"] == pytest.approx(-0.3)
    assert classes[1]["delta_full_minus_raw_loss_weight"] == pytest.approx(0.3)
    assert sum(row["full_mean_episode_loss_weight"] for row in classes) == pytest.approx(1)
    assert summary["actual_per_class_objective_loss_available"] is False


def test_common_weight_scale_cancels_and_zero_mass_is_undefined_not_zero_tv():
    assert total_variation(normalized_weights([0.2, 0.8]), normalized_weights([2, 8])) == 0
    assert normalized_weights([1e-9, 1]) == [0, 1]
    assert total_variation(normalized_weights([0, 0]), normalized_weights([1, 1])) is None


def test_candidate_matching_rejects_mask_identity_and_area_differences():
    candidates = candidate_fixture()
    candidates["no_source_tracing"].pop(next(iter(candidates["no_source_tracing"])))
    with pytest.raises(ValueError, match="Candidate identity"):
        compare_weights(candidates, dict(classes=["background", "cat"]))
    candidates = candidate_fixture()
    next(iter(candidates["no_source_tracing"].values()))["area"] = 3
    with pytest.raises(ValueError, match="Candidate area"):
        compare_weights(candidates, dict(classes=["background", "cat"]))


def test_analysis_never_overwrites_existing_output(tmp_path):
    output = tmp_path / "analysis"
    output.mkdir()
    marker = output / "summary.json"
    marker.write_text("immutable")
    with pytest.raises(ValueError, match="fresh analysis"):
        analyze(tmp_path / "suite", tmp_path / "reference", tmp_path / "package", output)
    assert marker.read_text() == "immutable"


def test_cli_help_needs_no_ml_or_gpu_imports():
    script = Path(__file__).resolve().parents[1] / "tools/analyze_voc21_tuning.py"
    result = subprocess.run([sys.executable, "-S", str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--reference-panel" in result.stdout


def test_candidate_query_aliases_map_to_background_and_duplicates_fail(tmp_path):
    directory = tmp_path / "cell"
    directory.mkdir()
    protocol = dict(**identity(), images=2, variant="full")
    protocol["config"].update(batch_size=2, class_extensions=True)
    row = dict(episode=0, image_id="a", patch_in_batch=0, class_idx=1, area=1,
        mask_sha256="a" * 64, target_support=0.5, residual_support=0.25,
        counterfactual_support=0.5, quality=0.5, dominance=0, reliability=0.5)

    def save(rows):
        with (directory / "candidate_diagnostics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)

    save([row])
    (directory / "cost.csv").write_text("episode,images,candidates,diagnostic_views\n0,2,1,3\n")
    cell = Cell(directory, protocol, {}, [], [], [], {})
    vocabulary = dict(classes=["background", "cat"], query_to_class=[0, 0, 1])
    candidates = read_candidates(cell, Evidence(), vocabulary)
    assert next(iter(candidates.values()))["evaluation_class_idx"] == 0
    save([row, row])
    with pytest.raises(ValueError, match="Duplicate candidate"):
        read_candidates(cell, Evidence(), vocabulary)
