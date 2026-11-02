import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_simplification import with_budget
from run_accuracy_budget_search import (config_key, configured, pareto_flags,
    select_accuracy_first, values_for, worker_command, audit_dependency, run_cell)
import run_accuracy_budget_search as search


def row(score, time, **kwargs):
    return dict(status="completed", mIoU=score, ms_per_image=time,
                trace_max_regions=33, steps=36, lr=.00063, **kwargs)


def test_overrides_preserve_algorithm_and_other_parameters():
    config = dict(trace_max_regions=33, steps=36, lr=.00063, batch_size=16,
                  trace_view_batch_size=16, class_extensions=True)
    saved = copy.deepcopy(config)
    assert with_budget(config, max_regions=8, steps=16, lr=.001) == {
        **saved, "trace_max_regions": 8, "steps": 16, "lr": .001}
    assert config == saved
    for values in ({"max_regions": 0}, {"steps": 1.5}, {"steps": True},
                   {"lr": float("nan")}, {"lr": -1}, {"lr": float("inf")}):
        with pytest.raises(ValueError):
            with_budget(config, **values)


def test_accuracy_precedes_speed_and_ties_are_global():
    rows = [row(60, 100), row(59.99, 80), row(59.97, 10)]
    rows[2]["validation_mIoU"] = 100
    assert select_accuracy_first(rows) is rows[1]
    with pytest.raises(RuntimeError):
        select_accuracy_first([dict(status="failed")])
    with pytest.raises(RuntimeError):
        select_accuracy_first([row(float("nan"), 10)])
    assert select_accuracy_first([dict(status="oom"), rows[0]]) is rows[0]


def test_coordinate_ranges_include_initial_and_are_unique():
    initial = dict(trace_max_regions=33, steps=36, lr=.00063)
    assert values_for("trace_max_regions", initial) == [4, 8, 16, 32, 33]
    assert values_for("steps", initial) == [4, 8, 16, 24, 36]
    assert values_for("lr", initial) == pytest.approx([.000315, .00063, .000945])
    initial["trace_max_regions"] = 16
    assert values_for("trace_max_regions", initial) == [4, 8, 16, 32]


def test_pareto_does_not_drop_accuracy_loss_as_better():
    assert pareto_flags([row(60, 100), row(59, 90), row(59, 110), row(60, 100)]) == [True, True, False, True]


def test_cache_key_binds_full_config_and_worker_passes_overrides():
    config = dict(trace_max_regions=33, steps=36, lr=.00063, batch_size=16,
                  trace_view_batch_size=16, data_dir="/data/VOCdevkit/VOC2012")
    expected = dict(dataset="voc21", split="calibration", selection_seed=2027,
                    image_ids=["a", "b"], config=config)
    replacement = {**config, "steps": 8}
    panel = configured(expected, replacement)
    assert expected["config"] == config
    assert config_key(config) != config_key(replacement)
    assert config_key(config) == config_key(dict(reversed(list(config.items()))))
    command = worker_command(Path("/runtime"), panel, Path("/out/attempt_001"))
    for option, value in (("--steps", "8"), ("--batch-size", "16"), ("--max-images", "2"),
                          ("--max-regions", "33"), ("--split", "calibration")):
        assert command[command.index(option)+1] == value


def dependency_fixture(tmp_path, monkeypatch):
    dependency, parent = tmp_path / "dependency", tmp_path / "parent"
    dependency.mkdir()
    parent.mkdir()
    base = dict(config=dict(batch_size=16), image_ids=["a"], label_ids=["b"])
    upstream = dict(calibration=base, evaluation=base, source_hashes={"a": "b"},
                    environment={}, tool_hashes={"benchmark_simplification.py": "worker", "trace_ablation_variants.py": "variants"})
    (dependency / "suite_protocol.json").write_text(json.dumps(upstream))
    results = {}
    rows = []
    for batch in range(1, 17):
        directory = dependency / "calibration" / f"batch_{batch:02d}" / "attempt_001"
        directory.mkdir(parents=True)
        result = dict(miou_unrounded=50 + batch / 10, metrics={"mAcc": 80},
                      ms_per_image=200/batch, peak_allocated_mib=100)
        results[str(directory)] = result
        for name in ("result.json", "protocol.json", "per_image_counts.json"):
            (directory / name).write_text("{}")
        (directory.parent / "selected_result.json").write_text(json.dumps(dict(
            result_path=str(directory / "result.json"), result_sha256=search.digest(directory / "result.json"))))
        rows.append(search.calibration_row(batch, directory, result))
    full = dependency / "full"
    full.mkdir()
    for name in ("result.json", "protocol.json", "per_image_counts.json"):
        (full / name).write_text("{}")
    results[str(full)] = dict(images=1449, full_split=True, miou_unrounded=52.5)
    (dependency / "selection.json").write_text(json.dumps(dict(selected_trial=search.choose_batch(rows), validation_used_for_selection=False)))
    (dependency / "verification.json").write_text(json.dumps(dict(batch_size=16,
        result_path=str(full / "result.json"), reused_existing_full=False, mIoU=52.5)))
    (dependency / "selected_full_validation").mkdir()
    (dependency / "selected_full_validation/selected_result.json").write_text(json.dumps(dict(
        result_path=str(full / "result.json"), result_sha256=search.digest(full / "result.json"))))
    calls = []
    def validator(directory, expected, *args, **kwargs):
        calls.append((str(directory), expected["config"]["batch_size"]))
        return results[str(directory)]
    monkeypatch.setattr(search, "validate_artifact", validator)
    return dependency, parent, results, calls


def test_upstream_all_trials_and_full_artifact_are_validated(tmp_path, monkeypatch):
    dependency, parent, _, calls = dependency_fixture(tmp_path, monkeypatch)
    audited = audit_dependency(dependency, parent, {})
    assert audited["batch_size"] == 16
    assert audited["all_sixteen_calibration_trials_validated"]
    assert len(calls) == 17
    assert [b for _, b in calls[:16]] == list(range(1, 17))
    assert len(audited["evidence_sha256"]) == 51


@pytest.mark.parametrize("fault", ["missing_trial", "changed_result", "wrong_winner", "incomplete_full", "wrong_full_score", "full_seal"])
def test_dependency_summaries_cannot_replace_actual_evidence(tmp_path, monkeypatch, fault):
    dependency, parent, results, _ = dependency_fixture(tmp_path, monkeypatch)
    if fault == "missing_trial":
        (dependency / "calibration/batch_04/selected_result.json").unlink()
    elif fault == "changed_result":
        (dependency / "calibration/batch_04/attempt_001/result.json").write_text("changed")
    elif fault == "wrong_winner":
        path = dependency / "selection.json"
        data = json.loads(path.read_text())
        data["selected_trial"]["batch_size"] = 1
        path.write_text(json.dumps(data))
    elif fault == "incomplete_full":
        results[str(dependency / "full")]["images"] = 512
    elif fault == "full_seal":
        path = dependency / "selected_full_validation/selected_result.json"
        data = json.loads(path.read_text())
        data["result_sha256"] = "different"
        path.write_text(json.dumps(data))
    else:
        results[str(dependency / "full")]["miou_unrounded"] = 80
    with pytest.raises(RuntimeError):
        audit_dependency(dependency, parent, {})


def test_unresolved_oom_never_silently_excludes_a_trial(tmp_path, monkeypatch):
    expected = dict(split="calibration", dataset="voc21", config={"steps": 4})
    cell = tmp_path / "calibration/voc21" / config_key(expected["config"])
    cell.mkdir(parents=True)
    (cell / "oom.json").write_text("not even a valid OOM record")
    def forbidden(*args, **kwargs):
        raise AssertionError("Must not start a worker or select around unresolved OOM")
    monkeypatch.setattr(search.subprocess, "run", forbidden)
    with pytest.raises(RuntimeError, match="Unresolved OOM"):
        run_cell(tmp_path, tmp_path, expected, {}, [], "steps")
