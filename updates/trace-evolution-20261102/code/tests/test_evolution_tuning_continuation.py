"""CPU-only continuation, dependency and development-overlap reporting tests."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.evolution_tuning import continuation as module
from tools.evolution_tuning.common import DATASETS, ORDER, FULL_ARMS, canonical_hash, sha256, write_json


def status(**changes):
    return dict(LoadState="loaded", ActiveState="active", SubState="exited", MainPID="0",
                Result="success", ExecMainStatus="0", **changes)


def test_success_requires_retained_process_terminal_and_partial_selection_stage():
    assert module.parent_ready(status(), dict(phase="awaiting_all_eight_calibrations"))
    running = status()
    running.update(SubState="running", MainPID="123")
    assert not module.parent_ready(running, dict(phase="predict"))


@pytest.mark.parametrize("changes,phase", [
    ({"LoadState": "not-found"}, "awaiting_all_eight_calibrations"),
    ({"Result": "oom-kill"}, "awaiting_all_eight_calibrations"),
    ({"ExecMainStatus": "1"}, "awaiting_all_eight_calibrations"),
    ({"MainPID": "12"}, "awaiting_all_eight_calibrations"),
    ({}, "completed"), ({"ActiveState": "failed"}, "failed"),
    ({"SubState": "running", "MainPID": "0"}, "predict"),
])
def test_missing_service_bad_exit_wrong_phase_and_oom_fail_closed(changes, phase):
    actual = status()
    actual.update(changes)
    with pytest.raises(RuntimeError):
        module.parent_ready(actual, dict(phase=phase))


def test_parent_and_continuation_require_exact_method_and_optimizer_starts():
    parent = dict(runtime_hashes={"trace_ovss/method/image_balanced.py": "a",
        "tools/evolution_tuning/worker.py": "b", "tools/evolution_tuning/inputs.py": "old"},
        checkpoint={"sha256": "weights"}, initial_configs={"pc60": {"lr": .001}})
    current = copy.deepcopy(parent)
    current["runtime_hashes"].update({"tools/evolution_tuning/inputs.py": "new",
                                      "tools/evolution_tuning/continuation.py": "controller"})
    module.unchanged_method(parent, current)
    current["runtime_hashes"]["tools/evolution_tuning/worker.py"] = "changed"
    with pytest.raises(RuntimeError, match="method/worker"):
        module.unchanged_method(parent, current)
    current = copy.deepcopy(parent)
    current["initial_configs"]["pc60"]["lr"] = .01
    with pytest.raises(RuntimeError, match="settings"):
        module.unchanged_method(parent, current)


def test_panel_identity_ignores_only_rebased_provenance_not_population():
    original = {k: [] for k in module.PANEL_KEYS}
    changed = dict(original, source_hashes={"new": "source"})
    assert module.panel_identity(original) == module.panel_identity(changed)
    changed["ordered_ids"] = ["different"]
    assert module.panel_identity(original) != module.panel_identity(changed)


def test_relative_parent_and_python_are_resolved_before_preparation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("parent", "source", "data"):
        (tmp_path / name).mkdir()
    (tmp_path / "python").write_text("executable identity fixture")
    args = module.normalize_paths(SimpleNamespace(parent_suite=Path("parent"), python=Path("python"),
        source_root=Path("source"), data_root=Path("data"), output=Path("future-suite")))
    assert args.parent_suite == tmp_path / "parent" and args.python == tmp_path / "python"
    assert args.source_root == tmp_path / "source" and args.output == tmp_path / "future-suite"
    monkeypatch.chdir(args.source_root)
    assert args.parent_suite.exists() and args.python.exists()


def full_fixture(tmp_path):
    entries = {d: dict(evaluation=dict(inputs=[d])) for d in DATASETS}
    selected = {d: dict(config=dict(dataset=d)) for d in DATASETS}
    for dataset in DATASETS:
        directory = tmp_path / "selections" / dataset
        directory.mkdir(parents=True)
        write_json(directory / "SELECTED_CONFIG_SEALED.json", selected[dataset])
    return entries, selected


def test_all_eight_configs_required_before_any_full(tmp_path):
    entries, selected = full_fixture(tmp_path)
    selected.pop("waterbirds_seg")
    with pytest.raises(RuntimeError, match="All eight"):
        module.full_stage(tmp_path, {}, entries, selected)
    assert not (tmp_path / "ALL_CONFIGS_SELECTED.json").exists()


def test_all_eight_predictions_before_score_and_waterbirds_development_rows(tmp_path, monkeypatch):
    entries, selected = full_fixture(tmp_path)
    calls = []
    monkeypatch.setattr(module.queue, "make_manifest", lambda *a, **k: Path(str(a[-1]) + ".manifest.json"))
    def child(suite, prepared, phase, manifest, output, **kwargs):
        calls.append((phase, output.name))
        return 0, "synthetic-child-receipt"
    monkeypatch.setattr(module.queue, "child", child)
    def boundary(suite, stage, outputs, path):
        assert calls == [("predict", d) for d in ORDER]
        assert len(outputs) == 8 and stage == "full"
        calls.append(("barrier", "all8"))
    monkeypatch.setattr(module.queue, "barrier", boundary)
    monkeypatch.setattr(module.queue, "validate_result", lambda output, dataset, panel, arms, **k: dict(
        images=5322 if dataset == "waterbirds_seg" else 2,
        arm_results={a: dict(miou_unrounded=50.) for a in FULL_ARMS},
        timings={a: dict(seconds_per_image=.1) for a in FULL_ARMS}))
    cells, rows = module.full_stage(tmp_path, {}, entries, selected)
    assert len(cells) == 8 and len(rows) == 32
    assert calls == [("predict", d) for d in ORDER] + [("barrier", "all8")] + [("score", d) for d in ORDER]
    birds = [r for r in rows if r["dataset"] == "waterbirds_seg"]
    assert len(birds) == 4
    assert all(r["evaluation_scope"] == "development_overlap" and r["calibration_full_source_id_overlap"] == 256
               and r["images"] == 5322 and not r["independent_test_claim"] for r in birds)
    assert all(r["calibration_full_source_id_overlap"] == 0 for r in rows if r["dataset"] != "waterbirds_seg")


def test_unsafe_systemctl_service_not_dispatched():
    with pytest.raises(RuntimeError, match="Unsafe"):
        module.service_status("--service.service")


def parent_fixture(tmp_path, monkeypatch):
    parent, suite = tmp_path / "parent", tmp_path / "continuation"
    parent.mkdir()
    suite.mkdir()
    entries = {d: dict(calibration={"inputs": [d]}, config=dict(dataset=d, lr=.001,
               batch_size=1, steps=9, trace_max_regions=8)) for d in DATASETS}
    initial = {d: e["config"] for d, e in entries.items()}
    upstream = dict(runtime_hashes={"trace_ovss/method/image_balanced.py": "math"}, checkpoint={}, initial_configs=initial)
    prepared = dict(upstream, _verified_sha256="b" * 64, continuation=dict(parent_suite=str(parent),
        parent_prepared_sha256="a" * 64, parent_service="fixture.service"))
    write_json(parent / "inputs.json", dict(datasets=entries))
    seals, gates, results, rows = {}, {}, {}, []
    for dataset in module.PARENT_DATASETS:
        gate = parent / "gates" / dataset / "GATE_PASSED.json"
        gate.parent.mkdir(parents=True)
        write_json(gate, dict(passed=True))
        gates[dataset] = sha256(gate)
        cfg = entries[dataset]["config"]
        key = module.queue.config_key(cfg)
        output = parent / "trials" / dataset / f"000_{key}"
        output.mkdir(parents=True)
        manifest = output.parent / (output.name + ".manifest.json")
        write_json(manifest, dict(config=cfg, dataset=dataset, stage="calibration", suite_root=str(parent),
            suite_prepared_sha256="a" * 64, source_sha256=upstream["runtime_hashes"], panel=entries[dataset]["calibration"]))
        write_json(output / "PREDICTIONS_SEALED.json", {})
        write_json(output / "SCORED_SEALED.json", {})
        result = dict(config=cfg, manifest_sha256=sha256(manifest), prediction_seal_sha256=sha256(output / "PREDICTIONS_SEALED.json"),
            arm_results=dict(image_C=dict(miou_unrounded=50.)), timings=dict(image_C=dict(seconds_per_image=.1)))
        results[str(output)] = result
        write_json(output / "result.json", result)
        row = dict(dataset=dataset, trial=0, status="completed", config=cfg, mIoU=50., ms_per_image=100.,
            result_path=str(output / "result.json"), scored_seal_sha256=sha256(output / "SCORED_SEALED.json"))
        rows.append(row)
        selection = parent / "selections" / dataset / "SELECTED_CONFIG_SEALED.json"
        selection.parent.mkdir(parents=True)
        write_json(selection, dict(dataset=dataset, config=cfg, selected_trial=row, unique_trials=1,
            all_trials=[dict(key=key, status="completed", result_path=row["result_path"])], calibration_only=True,
            full_evaluation_used_for_selection=False, calibration_panel_sha256=canonical_hash(entries[dataset]["calibration"]),
            suite_prepared_sha256="a" * 64))
        seals[dataset] = sha256(selection)
    module.queue.save_csv(parent / "trials.csv", rows)
    write_json(parent / "ALL_GATES_PASSED.json", dict(complete=True, gates=gates))
    write_json(parent / "CALIBRATION_SELECTIONS_SEALED.json", dict(complete=False, full_evaluation_ready=False,
        suite_prepared_sha256="a" * 64, blocked=dict(waterbirds_seg="pending choice"), selections=seals))
    monkeypatch.setattr(module.queue, "verify", lambda *a, **k: upstream)
    monkeypatch.setattr(module.queue, "validate_result", lambda output, *a, **k: results[str(output)])
    monkeypatch.setattr(module.queue, "tune", lambda *a, **k: pytest.fail("Seven parent searches must not repeat"))
    return parent, suite, prepared, entries, rows


def test_parent_import_reuses_exact_seven_winners_and_seals_origin(tmp_path, monkeypatch):
    parent, suite, prepared, entries, _ = parent_fixture(tmp_path, monkeypatch)
    selected = module.read_parent_selections(suite, prepared, entries, status())
    assert set(selected) == set(module.PARENT_DATASETS)
    assert all(s["selection_reused_without_search"] and s["config"] == entries[d]["config"]
               and s["origin"]["source_suite"] == str(parent) for d, s in selected.items())
    assert (suite / "PARENT_IMPORT_SEALED.json").exists()
    assert not (suite / "selections/waterbirds_seg").exists()


def test_parent_import_rejects_csv_score_drift(tmp_path, monkeypatch):
    parent, suite, prepared, entries, rows = parent_fixture(tmp_path, monkeypatch)
    rows[0]["mIoU"] += 1
    module.queue.save_csv(parent / "trials.csv", rows)
    with pytest.raises(RuntimeError, match="CSV metric"):
        module.read_parent_selections(suite, prepared, entries, status())
    assert not (suite / "PARENT_IMPORT_SEALED.json").exists()


def test_parent_import_rejects_missing_or_changed_selection(tmp_path, monkeypatch):
    parent, suite, prepared, entries, _ = parent_fixture(tmp_path, monkeypatch)
    selection = parent / "selections/voc20/SELECTED_CONFIG_SEALED.json"
    selection.write_text("{}")
    with pytest.raises(RuntimeError, match="selected config drift"):
        module.read_parent_selections(suite, prepared, entries, status())
