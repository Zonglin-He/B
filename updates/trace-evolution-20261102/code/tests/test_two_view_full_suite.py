"""Full validation preserves the pilot algorithm, configuration and population."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_two_view_full_suite as suite
from run_algorithm_efficiency_suite import worker_command


def test_plan_is_full_pc60_first_and_quality_only():
    plan = suite.experiment_plan()
    assert [c["dataset"] for c in plan] == ["pc60", "voc21", "coco_cd", "coco_object"]
    assert [c["limit"] for c in plan] == [5105, 1449, 500, 5000]
    assert all(c["algorithm"] == "quality_only" and c["variant"] == "full" for c in plan)
    assert all(c["panel"] == c["dataset"] + "/full" for c in plan)


def test_commands_cannot_change_accuracy_parameters(tmp_path):
    manifest = dict(data_root="/data", data_dirs={"coco_object": "/coco_mlmp"})
    forbidden = {"--steps", "--lr", "--max-regions", "--batch-size", "--view-batch-size"}
    for cell in suite.experiment_plan():
        cmd = worker_command(tmp_path, manifest, cell, tmp_path / cell["dataset"])
        assert not forbidden.intersection(cmd)
        assert cmd[cmd.index("--max-images") + 1] == str(cell["limit"])
        assert cmd[cmd.index("--algorithm") + 1] == "quality_only"


def test_discovery_requests_entire_split_without_outcome_selection(monkeypatch, tmp_path):
    seen = {}
    def discover(*args, **kwargs):
        seen.update(plan=args[2], **kwargs)
        return {"ok": True}
    monkeypatch.setattr(suite, "expected_inputs", discover)
    assert suite.discover(tmp_path, dict(data_root="/data", data_dirs={}, cells=suite.experiment_plan()))
    assert set(seen["plan"].values()) == {(d, "evaluation", None) for d in suite.DATASETS}


def test_input_hashes_detect_content_not_just_names(tmp_path):
    image, label = tmp_path / "image", tmp_path / "label"
    image.write_bytes(b"img")
    label.write_bytes(b"gt")
    panels = {"pc60/full": dict(image_ids=[str(image)], label_ids=[str(label)])}
    before = suite.input_hashes(panels)
    image.write_bytes(b"new")
    after = suite.input_hashes(panels)
    assert before[str(image)] != after[str(image)]
    assert before[str(label)] == after[str(label)]


def test_snapshot_includes_full_controller_and_transitive_workers():
    files = suite.tool_dependencies(Path(suite.__file__).parent, (suite.CONTROLLER,))
    assert {suite.CONTROLLER, "benchmark_simplification.py", "trace_algorithm_variants.py",
            "suite_inputs.py", "run_algorithm_efficiency_suite.py"}.issubset(files)


def test_pilot_inheritance_checks_algorithm_outside_production(tmp_path):
    names = ("trace_algorithm_variants.py", "trace_ablation_variants.py", "benchmark_simplification.py")
    for name in names:
        (tmp_path / name).write_text("original")
    parent = dict(tool_hashes={n: suite.digest(tmp_path / n) for n in names})
    suite.verify_inherited_algorithm(tmp_path, parent)
    (tmp_path / names[0]).write_text("changed")
    with pytest.raises(RuntimeError, match="implementation changed"):
        suite.verify_inherited_algorithm(tmp_path, parent)
