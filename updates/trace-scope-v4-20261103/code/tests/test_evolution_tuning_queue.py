"""CPU-only search, selection and fresh-evidence boundary tests."""

import copy
import json
from types import SimpleNamespace

import pytest

from tools.evolution_tuning import common, queue


def config():
    return dict(dataset="PascalVOC20Dataset", data_dir="/data/voc", data_subdir="voc",
                resize=[224, 224], patch_size=[224, 224], patch_stride=112,
                batch_size=1, steps=9, lr=.001105, trace_max_regions=8,
                trace_view_batch_size=16, class_extensions=True)


def row(score, time, **changes):
    return dict(status="completed", mIoU=score, ms_per_image=time, config={**config(), **changes})


def test_all_eight_original_protocols_not_automatic_other_backbone_expansion():
    assert len(common.DATASETS) == len(set(common.ORDER)) == 8
    assert "pc59" in common.DATASETS and "waterbirds_seg" in common.DATASETS
    assert set(common.ORDER) == set(common.DATASETS)
    assert "cityscapes" not in common.DATASETS and "acdc" not in common.DATASETS
    assert common.ALGORITHM == "ImageBalancedTRACE:C"
    assert common.FULL_ARMS == ["no_adapt", "image_uniform", "image_confidence", "image_C"]


def test_batches_individual_and_original_large_batch_preserved():
    assert common.coarse_values("batch_size", config()) == list(range(1, 17))
    assert common.coarse_values("batch_size", {**config(), "batch_size": 28}) == list(range(1, 17)) + [28]
    assert common.fine_values("batch_size", {**config(), "batch_size": 31}) == list(range(28, 33))


@pytest.mark.parametrize("coordinate", common.COORDINATES)
def test_original_values_and_fine_centers_never_dropped(coordinate):
    assert config()[coordinate] in common.coarse_values(coordinate, config())
    assert config()[coordinate] in common.fine_values(coordinate, config())


def test_trial_bound_for_complete_coordinate_search():
    # Even without deduplication, the fixed schedule fits within80 trials.
    inherited = {**config(), "batch_size": 28, "steps": 29, "trace_max_regions": 34}
    upper = 1 + sum(len(common.coarse_values(k, inherited)) + len(common.fine_values(k, inherited))
                    for k in common.COORDINATES)
    assert upper <= common.MAX_TRIALS


def test_accuracy_first_and_single_global_tolerance():
    slow_best = row(70, 100)
    too_low = row(69.979, 1)
    tied_fast = row(69.981, 50)
    assert common.choose([slow_best, too_low, tied_fast]) is tied_fast
    assert common.choose([slow_best, too_low]) is slow_best
    assert common.choose([row(69.97, 10), row(69.95, 5), slow_best]) is slow_best


def test_invalid_or_nonfinite_evidence_not_silently_selected():
    assert common.choose([row(50, 2), dict(status="ineligible", mIoU=None, ms_per_image=None)])['mIoU'] == 50
    with pytest.raises(RuntimeError, match="No valid"):
        common.choose([dict(status="ineligible")])
    with pytest.raises(RuntimeError, match="Nonfinite"):
        common.choose([row(float("nan"), 1)])
    with pytest.raises(RuntimeError, match="Nonfinite"):
        common.choose([row(50, 0)])


def test_exact_tie_order_reproducible():
    a, b = row(50, 1, steps=5), row(50, 1, steps=8)
    assert common.choose([b, a]) is a
    assert common.choose([a, b]) is a


def test_pooled_metric_is_not_average_of_image_metrics():
    assert common.pooled_miou([1, 9], [2, 10]) == pytest.approx(70)
    with pytest.raises(RuntimeError, match="Invalid pooled"):
        common.pooled_miou([3], [2])
    with pytest.raises(RuntimeError, match="zero union"):
        common.pooled_miou([0], [0])


def test_duplicate_json_and_nan_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"a": 1,"a": 2}')
    with pytest.raises(RuntimeError, match="Duplicate JSON"):
        common.read_json(path)
    path.write_text('{"a": NaN}')
    with pytest.raises(ValueError):
        common.read_json(path)


def test_hash_inventory_rejects_escape_and_drift(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("a")
    digest = common.sha256(path)
    common.verify_hashes(tmp_path, {"a.txt": digest})
    path.write_text("b")
    with pytest.raises(RuntimeError, match="drift"):
        common.verify_hashes(tmp_path, {"a.txt": digest})
    with pytest.raises(RuntimeError, match="drift"):
        common.verify_hashes(tmp_path, {"../elsewhere": digest})


def test_manifest_only_four_search_fields_may_change(tmp_path):
    initial = config()
    prepared = dict(initial_configs={"voc20": initial}, checkpoint={},
                    runtime_hashes={}, _verified_sha256="a" * 64)
    output = tmp_path / "trial"
    path = queue.make_manifest(tmp_path, prepared, "voc20", "calibration",
                               {**initial, "steps": 12}, {}, ["image_C"], output)
    actual = common.read_json(path)
    assert actual["suite_prepared_sha256"] == "a" * 64
    assert actual["config"]["steps"] == 12
    with pytest.raises(RuntimeError, match="Non-search"):
        queue.make_manifest(tmp_path, prepared, "voc20", "calibration",
                            {**initial, "class_extensions": False}, {}, ["image_C"], tmp_path / "other")
    with pytest.raises(RuntimeError, match="bounds"):
        queue.make_manifest(tmp_path, prepared, "voc20", "calibration",
                            {**initial, "steps": 65}, {}, ["image_C"], tmp_path / "large")


def test_config_key_binds_entire_resolved_config():
    assert common.config_key(config()) == common.config_key(copy.deepcopy(config()))
    assert common.config_key(config()) != common.config_key({**config(), "lr": .002})


def test_barrier_requires_actual_seal_files(tmp_path):
    output = tmp_path / "trial"
    output.mkdir()
    (output / "prediction.npy").write_bytes(b"example")
    common.write_json(output / "PREDICTIONS_SEALED.json", dict(complete=True, gt_contents_opened=False,
        artifact_sha256={"prediction.npy": common.sha256(output / "prediction.npy")}))
    common.write_json(tmp_path / "SUITE_PREPARED.json", {})
    queue.barrier(tmp_path, "calibration", [output], tmp_path / "barrier.json")
    bound = common.read_json(tmp_path / "barrier.json")
    assert bound["prediction_seal_sha256"] == {str(output): common.sha256(output / "PREDICTIONS_SEALED.json")}
    (output / "prediction.npy").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="drift"):
        queue.prediction_seal(output)


def test_source_prepare_digest_pinned_in_child(tmp_path, monkeypatch):
    captured = []
    def fail_verify(suite, expected, **kwargs):
        captured.append(expected)
        raise RuntimeError("stop before process")
    monkeypatch.setattr(queue, "verify", fail_verify)
    with pytest.raises(RuntimeError, match="stop before process"):
        queue.child(tmp_path, {"_verified_sha256": "ORIGINAL"}, "predict", tmp_path / "manifest", tmp_path / "out")
    assert captured == ["ORIGINAL"]


def test_extra_runtime_module_or_shadow_package_rejected(tmp_path):
    runtime = tmp_path / "runtime"
    for tree in ("tools", "tests", "trace_ovss"):
        (runtime / tree).mkdir(parents=True)
    source = runtime / "tools/worker.py"
    source.write_text("# frozen\n")
    common.write_json(tmp_path / "inputs.json", {})
    (tmp_path / "frozen_contract.md").write_text("contract")
    prepared = dict(algorithm=common.ALGORITHM, datasets=list(common.DATASETS), order=list(common.ORDER),
        full_arms=common.FULL_ARMS, max_unique_trials_per_dataset=80, coordinates=list(common.COORDINATES),
        inputs_sha256=common.sha256(tmp_path / "inputs.json"),
        contract_sha256=common.sha256(tmp_path / "frozen_contract.md"),
        runtime_hashes={"tools/worker.py": common.sha256(source)})
    common.write_json(tmp_path / "SUITE_PREPARED.json", prepared)
    expected = common.sha256(tmp_path / "SUITE_PREPARED.json")
    queue.verify(tmp_path, expected)
    shadow = runtime / "tools/worker/__init__.py"
    shadow.parent.mkdir()
    shadow.write_text("# not admitted\n")
    with pytest.raises(RuntimeError, match="inventory"):
        queue.verify(tmp_path, expected)


@pytest.mark.parametrize("name", ["sitecustomize.py", "__pycache__/worker.cpython-311.pyc", "tools/worker.pyc"])
def test_unrecorded_startup_and_cached_bytecode_rejected(tmp_path, name):
    runtime = tmp_path / "runtime"
    for tree in ("tools", "tests", "trace_ovss"):
        (runtime / tree).mkdir(parents=True)
    source = runtime / "tools/worker.py"
    source.write_text("# frozen\n")
    common.write_json(tmp_path / "inputs.json", {})
    (tmp_path / "frozen_contract.md").write_text("contract")
    prepared = dict(algorithm=common.ALGORITHM, datasets=list(common.DATASETS), order=list(common.ORDER),
        full_arms=common.FULL_ARMS, max_unique_trials_per_dataset=80, coordinates=list(common.COORDINATES),
        inputs_sha256=common.sha256(tmp_path / "inputs.json"),
        contract_sha256=common.sha256(tmp_path / "frozen_contract.md"),
        runtime_hashes={"tools/worker.py": common.sha256(source)})
    common.write_json(tmp_path / "SUITE_PREPARED.json", prepared)
    expected = common.sha256(tmp_path / "SUITE_PREPARED.json")
    queue.verify(tmp_path, expected)
    extra = runtime / name
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"unregistered startup or cache")
    with pytest.raises(RuntimeError, match="inventory|Unrecorded"):
        queue.verify(tmp_path, expected)


def numerical_receipt(tmp_path, *, kind="nonfinite", **changes):
    manifest = dict(config=config(), checkpoint={"sha256": "a" * 64}, source_sha256={"worker.py": "b" * 64}, dataset="voc20")
    path = tmp_path / "trial.manifest.json"
    common.write_json(path, manifest)
    output = tmp_path / "trial"
    output.mkdir()
    failure = dict(status="failed", phase="predict", fatal=False, ineligible_trial=True,
        retry_allowed=False, failure_kind=kind, manifest_sha256=common.sha256(path),
        config=manifest["config"], checkpoint=manifest["checkpoint"], source_sha256=manifest["source_sha256"],
        dataset="voc20", output=str(output), stage="calibration", seed=0, calibration=True, full_split=False,
        error="RuntimeError: non-finite adaptation gradient", **changes)
    common.write_json(output / "PREDICTION_FAILED.json", failure)
    return output, path


def test_structured_numerical_failure_not_free_text(tmp_path):
    output, manifest = numerical_receipt(tmp_path)
    assert queue.ineligible_failure(output, manifest)["failure_kind"] == "nonfinite"
    failure = common.read_json(output / "PREDICTION_FAILED.json")
    failure.update(fatal=True, ineligible_trial=False, failure_kind="fatal_drift",
                   error="Source drift: /CUDA out of memory/file.json")
    common.write_json(output / "PREDICTION_FAILED.json", failure, replace=True)
    with pytest.raises(RuntimeError, match="Fatal/malformed"):
        queue.ineligible_failure(output, manifest)


def test_numerical_failure_must_bind_same_manifest_config_and_output(tmp_path):
    output, manifest = numerical_receipt(tmp_path, kind="oom")
    failure = common.read_json(output / "PREDICTION_FAILED.json")
    failure["config"]["steps"] = 12
    common.write_json(output / "PREDICTION_FAILED.json", failure, replace=True)
    with pytest.raises(RuntimeError, match="identity/provenance"):
        queue.ineligible_failure(output, manifest)


def test_inherited_oom_preserved_but_small_batches_tested(tmp_path, monkeypatch):
    seen = []
    def fake_trial(suite, prepared, dataset, entry, cfg, stage, rows, cache):
        key = common.config_key(cfg)
        if key in cache:
            return cache[key]
        seen.append(copy.deepcopy(cfg))
        value = dict(dataset=dataset, status="ineligible" if cfg["batch_size"] == 28 else "completed",
                     config=copy.deepcopy(cfg), mIoU=50 if cfg["batch_size"] == 1 else 49,
                     ms_per_image=1, result_path="fixture", stage=stage)
        cache[key] = value
        rows.append(value)
        return value
    monkeypatch.setattr(queue, "trial", fake_trial)
    common.write_json(tmp_path / "SUITE_PREPARED.json", {})
    selected = queue.tune(tmp_path, {}, "voc20", {"config": {**config(), "batch_size": 28}, "calibration": {}}, [])
    assert seen[0]["batch_size"] == 28 and seen[1]["batch_size"] == 1
    assert selected["config"]["batch_size"] == 1
    assert any(r["status"] == "ineligible" for r in selected["all_trials"])
    assert selected["full_evaluation_used_for_selection"] is False


def test_blocked_eighth_dataset_holds_full_prediction_and_scoring(tmp_path, monkeypatch):
    runnable = [d for d in common.ORDER if d != "waterbirds_seg"]
    prepared = dict(runnable=runnable, blocked={"waterbirds_seg": "needs input decision"})
    panel = dict(inputs=[{}, {}, {}, {}], ordered_ids=list(range(4)), full_split=False)
    common.write_json(tmp_path / "inputs.json", dict(datasets={d: dict(config=config(), calibration=panel) for d in runnable}))
    monkeypatch.setattr(queue, "verify", lambda *args, **kwargs: prepared)
    calls = []
    def fake_manifest(suite, prep, dataset, stage, cfg, panel, arms, output, **kwargs):
        return output.with_suffix(".manifest.json")
    def fake_child(suite, prep, phase, manifest, output, **kwargs):
        calls.append(phase)
        assert phase == "gate"
        output.mkdir(parents=True)
        common.write_json(output / "GATE_PASSED.json", dict(passed=True))
        return 0, "fixture"
    def fake_tune(suite, prep, dataset, entry, rows):
        target = suite / "selections" / dataset
        target.mkdir(parents=True)
        common.write_json(target / "SELECTED_CONFIG_SEALED.json", dict(config=config()))
        return dict(config=config())
    monkeypatch.setattr(queue, "make_manifest", fake_manifest)
    monkeypatch.setattr(queue, "child", fake_child)
    monkeypatch.setattr(queue, "tune", fake_tune)
    queue.run(SimpleNamespace(suite=tmp_path, suite_sha256="a" * 64))
    assert calls == ["gate"] * 7
    assert common.read_json(tmp_path / "queue_state.json")["phase"] == "awaiting_all_eight_calibrations"
    assert not (tmp_path / "ALL_CONFIGS_SELECTED.json").exists()
    assert not (tmp_path / "ALL_FULL_PREDICTIONS_SEALED.json").exists()
    assert not (tmp_path / "full_metrics.csv").exists()


def test_input_provenance_rebound_to_frozen_bytes(tmp_path):
    source, suite = tmp_path / "checkout", tmp_path / "suite"
    source.mkdir()
    path = suite / "runtime/tools/source.py"
    path.parent.mkdir(parents=True)
    path.write_text("# copied\n")
    origin = str(source / "tools/source.py")
    datasets = dict(voc20=dict(calibration=dict(source_hashes={origin: {"sha256": common.sha256(path)}}), evaluation=None))
    mapping = queue.rebase_panel_sources(datasets, source, suite, {"tools/source.py": common.sha256(path)})
    assert mapping[origin] == str(path)
    assert list(datasets["voc20"]["calibration"]["source_hashes"]) == [str(path)]
