"""Cached evidence must belong to the currently frozen input panel."""

import copy
import importlib.util
from pathlib import Path

import pytest


source = Path(__file__).resolve().parents[1] / "tools/suite_inputs.py"
spec = importlib.util.spec_from_file_location("suite_input_tests", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def panel():
    return dict(dataset="pc60", split="evaluation", seed=0, selection_seed=20261024, images=2,
                image_ids=["/data/A/a.jpg", "/data/A/b.jpg"],
                label_ids=["/data/A/a.png", "/data/A/b.png"],
                vocabulary_sha256="frozen-vocabulary", available_images=10,
                config=dict(data_dir="/data/A", batch_size=2, lr=.001,
                            steps=3, trace_max_regions=4, trace_view_batch_size=8))


def test_unchanged_resume_accepts_only_view_batch_override():
    expected = panel()
    saved = copy.deepcopy(expected)
    saved["config"]["trace_view_batch_size"] = 16
    module.validate_cell_inputs(saved, expected)


@pytest.mark.parametrize("change", ["root", "order", "labels", "vocabulary", "batch", "seed", "pool"])
def test_cached_cell_cannot_be_reused_for_a_different_panel(change):
    expected, saved = panel(), panel()
    if change == "root":
        expected["config"]["data_dir"] = "/data/B"
    elif change == "order":
        expected["image_ids"].reverse()
    elif change == "labels":
        expected["label_ids"][0] = "/data/B/a.png"
    elif change == "vocabulary":
        expected["vocabulary_sha256"] = "another-vocabulary"
    elif change == "batch":
        expected["config"]["batch_size"] = 1
    elif change == "seed":
        expected["selection_seed"] += 1
    else:
        expected["available_images"] += 1
    with pytest.raises(RuntimeError):
        module.validate_cell_inputs(saved, expected)


def test_manifest_changes_fail_without_overwriting_original(tmp_path):
    path = tmp_path / "suite.json"
    original = dict(data_root="/data/A", input_panels=panel())
    module.freeze_protocol(path, original)
    saved = path.read_bytes()
    module.freeze_protocol(path, original)
    for changed in (dict(data_root="/data/B", input_panels=panel()),
                    dict(data_root="/data/A", input_panels={**panel(), "image_ids": ["another image"]})):
        with pytest.raises(RuntimeError):
            module.freeze_protocol(path, changed)
        assert path.read_bytes() == saved


def test_fixed_population_must_be_complete_and_large_enough():
    with pytest.raises(RuntimeError, match="requires 512"):
        module.validate_population("pc60", "evaluation", 512, 511)
    with pytest.raises(RuntimeError, match="Incomplete coco_cd"):
        module.validate_population("coco_cd", "evaluation", None, 499)
    module.validate_population("coco_cd", "evaluation", None, 500)
    module.validate_population("pc60", "evaluation", 512, 5105)


def test_suite_rejects_changed_root_before_launch_or_input_preparation(tmp_path, monkeypatch):
    import json
    import sys

    tool_root = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tool_root))
    import run_simplification_suite as suite

    output = tmp_path / "panel"
    output.mkdir()
    original = json.dumps({"data_root": str(tmp_path / "A")})
    (output / "suite_protocol.json").write_text(original)
    monkeypatch.setattr(sys, "argv", ["suite", "--output", str(output),
        "--reference-root", str(tmp_path / "reference"), "--candidate-root", str(tmp_path / "candidate"),
        "--data-root", str(tmp_path / "B")])

    def unexpected(*args, **kwargs):
        pytest.fail("Changed-root resume must fail before any preparation or benchmark launch")

    monkeypatch.setattr(suite, "expected_inputs", unexpected)
    monkeypatch.setattr(suite.subprocess, "run", unexpected)
    with pytest.raises(RuntimeError, match="data root changed"):
        suite.main()
    assert (output / "suite_protocol.json").read_text() == original


def test_analyzer_rejects_shortened_full_split(tmp_path, monkeypatch):
    import hashlib
    import json

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    import analyze_simplification as analyzer

    directory = tmp_path / "evaluation/coco_cd/reference"
    directory.mkdir(parents=True)
    protocol = dict(available_images=2, image_ids=["a", "b"])
    path = directory / "protocol.json"
    path.write_text(json.dumps(protocol))
    (directory / "result.json").write_text(json.dumps(dict(status="completed", images=2,
        full_split=True, protocol_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    (directory / "per_image_counts.json").write_text(json.dumps([dict(image_id="a"), dict(image_id="b")]))
    with pytest.raises(RuntimeError, match="Incomplete coco_cd"):
        analyzer.load_cell(tmp_path, "coco_cd", "reference")
