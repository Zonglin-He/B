"""CPU mocks only: never prepare real outputs, load real weights, or use CUDA."""

import copy
import fcntl
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.evolution_followup import queue
from tools.evolution_followup import smoke
from tools.evolution_followup import standalone as worker
from test_simplified_trace import tiny_trace


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("No CUDA in queue tests"))
    yield
    torch.set_num_threads(before)


def plan():
    return queue.read_json(queue.ROOT / "docs/trace-standalone-followup-plan.json")


def parent_fixture(root, dataset):
    from trace_ovss.configs import best_config
    config = best_config(dataset, "/synthetic/data")
    config = json.loads(json.dumps(dict(config, data_dir=str(config["data_dir"]))))
    config["lr"] = .009  # Distinguishes the sealed parent from live defaults.
    checkpoint = dict(path="/synthetic/existing-official.pt", sha256="a" * 64, bytes=123, mtime_ns=5,
                      official_url="https://example.invalid/official", backbone="ViT-L/14")
    inputs = [dict(image=f"/synthetic/{dataset}/images/{i:06d}.jpg",
        label=f"/synthetic/{dataset}/labels/{i:06d}.png", image_sha256=f"{i:064x}", label_metadata=[20, 5])
        for i in range(queue.POPULATIONS[dataset])]
    protocol = dict(dataset=dataset, seed=0, full_split=True, images=len(inputs), inputs=inputs,
        config=config, checkpoint=checkpoint, classes=["a", "b"], queries=["a", "b"],
        query_to_class=[0, 1], reduce_zero_label=False)
    cell = root / "cells/mechanism" / dataset
    cell.mkdir(parents=True)
    queue.write_json(cell / "prediction_protocol.json", protocol)
    queue.write_json(cell / "PREDICTIONS_SEALED.json", dict(complete=True, gt_contents_opened=False,
        images=len(inputs), artifact_sha256={"prediction_protocol.json": queue.sha256(cell / "prediction_protocol.json")}))
    return protocol


def mock_manifest(suite, dataset, cfg_path, count, order_seed, arms, parent):
    indices = worker.selected_indices(queue.POPULATIONS[dataset], 20261029, count, order_seed)
    inputs = [dict(parent["inputs"][i], dataset_index=i, official_id=f"{i:06d}.jpg") for i in indices]
    sources = worker.checked_sources()
    manifest = dict(format_version=1, worker="standalone_probes", dataset=dataset,
        config=queue.read_json(cfg_path), config_path=str(cfg_path), config_sha256=queue.sha256(cfg_path),
        seed=0, selection_seed=20261029, episode_order_seed=order_seed, max_images=count,
        arms=arms, arm_order="alternating", source_sha256={str(suite / "runtime" / Path(k).relative_to(queue.ROOT)): v
                                                         for k, v in sources.items()},
        checkpoint=parent["checkpoint"], timing_scope=worker.TIMING_SCOPE, exposure_note=queue.EXPOSURE,
        claim_boundary="Synthetic CPU fixture, no scientific evidence",
        panel=dict(inputs=inputs, ordered_ids=[r["official_id"] for r in inputs],
            selected_dataset_indices=sorted(indices), available_images=queue.POPULATIONS[dataset], full_split=False,
            classes=parent["classes"], queries=parent["queries"], query_to_class=parent["query_to_class"],
            reduce_zero_label=False, vocabulary_sha256=worker.canonical_hash([parent["queries"], parent["query_to_class"]])))
    return manifest


@pytest.fixture
def prepared_suite(tmp_path, monkeypatch):
    parent_root = tmp_path / "parents-source"
    parents = {d: parent_fixture(parent_root, d) for d in queue.DATASETS}
    events = []
    def fake_prepare(suite, hashes, python, arguments, label, *, cpu=False):
        assert cpu is True and arguments[:2] == ["tools.evolution_followup.standalone", "prepare"]
        assert queue.source_inventory(suite / "runtime") == hashes
        def value(flag):
            return arguments[arguments.index(flag) + 1]
        dataset = value("--dataset")
        start = arguments.index("--arms") + 1
        arms = arguments[start:arguments.index("--arm-order")]
        order = int(value("--episode-order-seed")) if "--episode-order-seed" in arguments else None
        manifest = mock_manifest(suite, dataset, Path(value("--config")), int(value("--max-images")), order, arms, parents[dataset])
        queue.write_json(Path(value("--manifest")), manifest)
        log = f"logs/{label}.log"
        (suite / log).write_bytes(b"CPU mock, not a real prepare\n")
        events.append(arguments)
        return dict(log=log, log_sha256=queue.sha256(suite / log), cpu_only=True)
    monkeypatch.setattr(queue, "child", fake_prepare)
    suite = tmp_path / "suite"
    queue.prepare(SimpleNamespace(source_root=queue.ROOT, plan=queue.ROOT / "docs/trace-standalone-followup-plan.json",
        parent_root=parent_root, python=Path(__import__("sys").executable), output=suite))
    prepared = queue.read_json(suite / "SUITE_PREPARED.json")
    return suite, prepared, events


def test_plan_and_parent_provenance_no_default_substitution(tmp_path):
    queue.validate_plan(plan())
    for dataset in queue.DATASETS:
        parent = parent_fixture(tmp_path, dataset)
        actual, binding = queue.read_parent(tmp_path, dataset)
        assert actual == parent and binding["protocol_sha256"] == queue.sha256(Path(binding["protocol_path"]))
        assert queue.derive_config(parent, "inherited_sorted")["lr"] == .009
        assert queue.derive_config(parent, "inherited_regrouped") == parent["config"]
        common = queue.derive_config(parent, "common_sorted")
        assert {k: v for k, v in common.items() if k not in queue.OVERRIDES} == {
            k: v for k, v in parent["config"].items() if k not in queue.OVERRIDES}
        assert all(common[k] == v for k, v in queue.OVERRIDES.items())
        path = Path(binding["protocol_path"])
        altered = copy.deepcopy(parent)
        altered["config"]["lr"] = .5
        path.write_text(json.dumps(altered))
        with pytest.raises(RuntimeError, match="PREDICTIONS_SEALED"):
            queue.read_parent(tmp_path, dataset)


@pytest.mark.parametrize("field,value", [("subset_selection_seed", 0), ("model_seed", 1),
    ("images_per_dataset", 129), ("arms", ["trace"]), ("parameter_search", True)])
def test_plan_rejects_new_search_scope(field, value):
    changed = plan()
    changed[field] = value
    with pytest.raises(RuntimeError):
        queue.validate_plan(changed)


def test_six_configs_eight_cpu_manifests_and_only_regrouped_order_changes(prepared_suite):
    suite, prepared, events = prepared_suite
    assert len(prepared["cells"]) == 6 and len(prepared["smokes"]) == 2 and len(events) == 8
    assert len(list((suite / "configs").glob("*.json"))) == 6
    manifests = queue.validate_panels(suite, prepared)
    for dataset in queue.DATASETS:
        a, b, c = [manifests[f"{dataset}/{condition}"] for condition in queue.CONDITIONS]
        assert a["panel"] == b["panel"]
        assert a["panel"]["selected_dataset_indices"] == c["panel"]["selected_dataset_indices"]
        assert a["panel"]["ordered_ids"] != c["panel"]["ordered_ids"]
        assert a["config"]["lr"] == .009 and b["config"]["lr"] == .001
        assert a["arms"] == queue.ARMS
    queue.verify_prepared(suite, queue.sha256(suite / "SUITE_PREPARED.json"))
    with pytest.raises(RuntimeError, match="frozen snapshot"):
        queue.verify_prepared(suite, queue.sha256(suite / "SUITE_PREPARED.json"), frozen=True)


def test_snapshot_is_small_cache_free_and_tamper_detected(prepared_suite):
    suite, prepared, _ = prepared_suite
    assert not any("__pycache__" in key or key.endswith(".pyc") for key in prepared["runtime_hashes"])
    target = suite / "runtime/tools/evolution_followup/smoke.py"
    target.write_text(target.read_text() + "\n# Synthetic tamper\n")
    with pytest.raises(RuntimeError, match="runtime"):
        queue.verify_prepared(suite, queue.sha256(suite / "SUITE_PREPARED.json"))


def test_prepared_suite_has_no_live_parent_lifecycle_dependency(prepared_suite):
    suite, prepared, _ = prepared_suite
    shutil.rmtree(suite.parent / "parents-source")
    queue.verify_prepared(suite, queue.sha256(suite / "SUITE_PREPARED.json"))


def test_source_snapshot_rejects_symlinks(tmp_path):
    for tree in queue.TREES:
        (tmp_path / tree).mkdir()
    (tmp_path / "tools/escape.py").symlink_to("/dev/null")
    with pytest.raises(RuntimeError, match="symlink"):
        queue.source_inventory(tmp_path)


def test_child_explicit_cpu_environment_log_bound_and_source_gate(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for tree in queue.TREES:
        (runtime / tree).mkdir()
    (tmp_path / "logs").mkdir()
    hashes = queue.source_inventory(runtime)
    calls = []
    class Process:
        def __init__(self, argv, **kwargs):
            calls.append((argv, kwargs))
            self.stdout = io.BytesIO(b"z" * (queue.LOG_BYTES + 12345))
        def wait(self, **kwargs):
            return 0
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "main-gpu")
    monkeypatch.setattr(queue.subprocess, "Popen", Process)
    receipt = queue.child(tmp_path, hashes, Path("/fake/python"), ["module", "prepare"], "cpu", cpu=True)
    assert receipt["retained_bytes"] == (tmp_path / "logs/cpu.log").stat().st_size == queue.LOG_BYTES
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert calls[0][1]["cwd"] == runtime and calls[0][0][:3] == ["/fake/python", "-B", "-m"]
    assert "LD_LIBRARY_PATH" not in calls[0][1]["env"] or calls[0][1]["env"]["LD_LIBRARY_PATH"] == __import__("os").environ["LD_LIBRARY_PATH"]
    queue.child(tmp_path, hashes, Path("/fake/python"), ["module", "predict"], "gpu")
    assert calls[1][1]["env"]["CUDA_VISIBLE_DEVICES"] == "main-gpu"
    (runtime / "tools/changed.py").write_text("# changed\n")
    with pytest.raises(RuntimeError, match="before child"):
        queue.child(tmp_path, hashes, Path("/fake/python"), ["module"], "rejected")
    assert len(calls) == 2


def mock_execution(prepared_suite, monkeypatch, fail_predict=None):
    suite, prepared, _ = prepared_suite
    verify = queue.verify_prepared
    monkeypatch.setattr(queue, "verify_prepared", lambda suite, expected, **kwargs: verify(suite, expected))
    events = []
    def launch(suite, hashes, python, arguments, label, *, cpu=False):
        kind = "smoke" if arguments[0].endswith("smoke") else arguments[1]
        events.append((kind, label))
        if kind == "predict" and len([e for e in events if e[0] == "predict"]) == fail_predict:
            raise RuntimeError("Synthetic child failed")
        if kind == "smoke":
            dataset = arguments[arguments.index("--dataset") + 1]
            queue.write_json(suite / f"smoke/{dataset}.json", {})
        else:
            output = Path(arguments[arguments.index("--output") + 1])
            if kind == "predict":
                output.mkdir(parents=True)
                queue.write_json(output / "PREDICTIONS_SEALED.json", {})
            else:
                assert (suite / "ALL_PREDICTIONS_SEALED.json").exists()
                assert sum(e[0] == "predict" for e in events) == 6
                queue.write_json(output / "result.json", {})
                queue.write_json(output / "SCORED_SEALED.json", {})
        path = suite / f"logs/{label}.log"
        path.write_text("Mock child\n")
        return dict(log=str(path.relative_to(suite)), log_sha256=queue.sha256(path))
    monkeypatch.setattr(queue, "child", launch)
    monkeypatch.setattr(queue, "validate_smoke", lambda s, e: queue.sha256(s / e["output"]))
    monkeypatch.setattr(queue, "validate_prediction", lambda s, e: queue.sha256(s / e["output"] / "PREDICTIONS_SEALED.json"))
    monkeypatch.setattr(queue, "validate_score", lambda s, e: dict(
        arm_results={a: dict(miou_unrounded=10.) for a in queue.ARMS},
        timings={a: dict(measured_images=100, seconds=10., seconds_per_image=.1, images_per_second=10.) for a in queue.ARMS}))
    monkeypatch.setattr(worker, "gpu_lock", lambda: pytest.fail("Controller must not take shared GPU lock"))
    return suite, prepared, events


def test_queue_two_gates_then_all_predict_then_all_score_and_30_rows(prepared_suite, monkeypatch):
    suite, prepared, events = mock_execution(prepared_suite, monkeypatch)
    args = SimpleNamespace(suite=suite, suite_sha256=queue.sha256(suite / "SUITE_PREPARED.json"))
    queue.run(args)
    assert [e[0] for e in events] == ["smoke"] * 2 + ["predict"] * 6 + ["score"] * 6
    assert len((suite / "rows.csv").read_text().splitlines()) == 31
    seal = queue.read_json(suite / "SUITE_SEALED.json")
    assert seal["complete"] is True and seal["formal_cells"] == 6 and seal["arm_results"] == 30
    queue.check_hashes(suite, seal["artifact_sha256"])
    assert queue.read_json(suite / "queue_state.json")["phase"] == "completed"
    with pytest.raises(RuntimeError, match="retry"):
        queue.run(args)
    assert len(events) == 14


def test_failed_prediction_stops_without_scoring_or_resume(prepared_suite, monkeypatch):
    suite, prepared, events = mock_execution(prepared_suite, monkeypatch, fail_predict=3)
    args = SimpleNamespace(suite=suite, suite_sha256=queue.sha256(suite / "SUITE_PREPARED.json"))
    with pytest.raises(RuntimeError, match="Synthetic child"):
        queue.run(args)
    assert [e[0] for e in events] == ["smoke"] * 2 + ["predict"] * 3
    assert not (suite / "ALL_PREDICTIONS_SEALED.json").exists() and not (suite / "SUITE_SEALED.json").exists()
    assert queue.read_json(suite / "queue_state.json")["sealed_predictions"] == 2
    with pytest.raises(RuntimeError, match="retry"):
        queue.run(args)


def test_failed_smoke_stops_before_formal_prediction(prepared_suite, monkeypatch):
    suite, prepared, events = mock_execution(prepared_suite, monkeypatch)
    monkeypatch.setattr(queue, "validate_smoke", lambda *args: (_ for _ in ()).throw(RuntimeError("Failed gate")))
    with pytest.raises(RuntimeError, match="Failed gate"):
        queue.run(SimpleNamespace(suite=suite, suite_sha256=queue.sha256(suite / "SUITE_PREPARED.json")))
    assert [e[0] for e in events] == ["smoke"] and not (suite / "cells").exists()


def strict_runtime():
    return dict(cublas_workspace_config=":4096:8", deterministic_algorithms=True,
        deterministic_warn_only=False, deterministic_debug_mode=2, cudnn_deterministic=True,
        cudnn_benchmark=False, sdpa_enabled=dict(flash=True, math=True, mem_efficient=True, cudnn=True))


def smoke_report(suite, entry):
    manifest = queue.read_json(suite / entry["manifest"])
    return dict(passed=True, gt_contents_opened=False, images=4, dataset=entry["dataset"],
        manifest_sha256=entry["manifest_sha256"], suite_prepared_sha256=queue.sha256(suite / "SUITE_PREPARED.json"),
        candidate_masks_equal=True, same_steps=True, max_support_abs_difference=0., max_loss_abs_difference=0.,
        changed_label_pixels=0, atol=1e-5, finite=True, count_checks_passed=True,
        source_sha256=manifest["source_sha256"], checkpoint=manifest["checkpoint"],
        ordered_ids=manifest["panel"]["ordered_ids"],
        input_image_sha256={r["image"]: r["image_sha256"] for r in manifest["panel"]["inputs"]},
        prediction_shapes=[[224, 224]] * 4, total_label_pixels=4 * 224 * 224, candidates=4,
        runs={name: dict(steps=manifest["config"]["steps"]) for name in ("native", "standalone_min")},
        deterministic_runtime=strict_runtime(), gpu_lock=dict(path=str(worker.GPU_LOCK), exclusive=True))


def test_strict_smoke_report_bindings_reject_boolean_only_success(prepared_suite):
    suite, prepared, _ = prepared_suite
    entry = prepared["smokes"][0]
    (suite / "smoke").mkdir()
    path = suite / entry["output"]
    report = smoke_report(suite, entry)
    queue.write_json(path, report)
    assert queue.validate_smoke(suite, entry) == queue.sha256(path)
    for key, value in (("changed_label_pixels", 1), ("source_sha256", {}), ("gt_contents_opened", True),
                       ("images", 128), ("prediction_shapes", [[32, 32]] * 4), ("manifest_sha256", "0" * 64)):
        changed = dict(report, **{key: value})
        path.write_text(json.dumps(changed))
        with pytest.raises(RuntimeError):
            queue.validate_smoke(suite, entry)


def test_prediction_score_scope_timing_and_seal_bindings(prepared_suite, monkeypatch):
    suite, prepared, _ = prepared_suite
    smoke_entry, entry = prepared["smokes"][0], prepared["cells"][0]
    (suite / "smoke").mkdir()
    queue.write_json(suite / smoke_entry["output"], smoke_report(suite, smoke_entry))
    manifest = queue.read_json(suite / entry["manifest"])
    output = suite / entry["output"]
    output.mkdir(parents=True)
    queue.write_json(output / "PREDICTIONS_SEALED.json", dict(synthetic=True))
    pred_hash = queue.sha256(output / "PREDICTIONS_SEALED.json")
    protocol = dict(manifest=manifest, manifest_sha256=entry["manifest_sha256"],
        manifest_path=str(suite / entry["manifest"]), deterministic_runtime=strict_runtime(),
        gpu_lock=dict(path=str(worker.GPU_LOCK), exclusive=True))
    monkeypatch.setattr(worker, "verify_prediction_seal", lambda *args: protocol)
    batch = manifest["config"]["batch_size"]
    costs = [dict(arm=arm, episode=start // batch, images=min(batch, 128 - start), seconds=1.)
             for start in range(0, 128, batch) for arm in queue.ARMS]
    timings = worker.timing_summary(costs, queue.ARMS)
    (output / "cost.jsonl").write_text("".join(json.dumps(r) + "\n" for r in costs))
    queue.write_json(output / "prediction_result.json", dict(status="predicted", images=128, full_split=False,
        timing_scope=worker.TIMING_SCOPE, accuracy_includes_warmup=True, timings=timings))
    assert queue.validate_prediction(suite, entry) == pred_hash
    queue.write_json(output / "SCORING_STARTED.json", dict(prediction_seal_sha256=pred_hash))
    queue.write_json(output / "score_inputs.json", {})
    (output / "scores_by_episode.jsonl").write_text("{}\n")
    result = dict(status="completed", dataset=entry["dataset"], seed=0, images=128, full_split=False,
        prediction_seal_sha256=pred_hash, label_contents_first_opened_after_all_requested_arms_sealed=True,
        timing_scope=worker.TIMING_SCOPE, exposure_note=queue.EXPOSURE, claim_boundary=manifest["claim_boundary"],
        timings=timings, arm_results={a: dict(miou_unrounded=10.) for a in queue.ARMS})
    queue.write_json(output / "result.json", result)
    sealed = dict(prediction_seal_sha256=pred_hash, artifact_sha256={p: queue.sha256(output / p) for p in
        ("SCORING_STARTED.json", "score_inputs.json", "scores_by_episode.jsonl", "result.json")})
    queue.write_json(output / "SCORED_SEALED.json", sealed)
    assert queue.validate_score(suite, entry) == result
    protocol["deterministic_runtime"] = dict(strict_runtime(), sdpa_enabled={"math": True})
    with pytest.raises(RuntimeError, match="dispatch"):
        queue.validate_prediction(suite, entry)
    protocol["deterministic_runtime"] = strict_runtime()
    changed = dict(result, full_split=True)
    (output / "result.json").write_text(json.dumps(changed))
    sealed["artifact_sha256"]["result.json"] = queue.sha256(output / "result.json")
    (output / "SCORED_SEALED.json").write_text(json.dumps(sealed))
    with pytest.raises(RuntimeError, match="scope incomplete"):
        queue.validate_score(suite, entry)


def tiny_standalone():
    base = tiny_trace()
    return worker.StandaloneTRACE(classes=base.classes, lr=base.lr, steps=base.steps,
        trace_max_regions=base.trace_max_regions, trace_view_batch_size=base.trace_view_batch_size,
        vision_outputs=base.vision_outputs, model=base.model, tokenize_fn=base.tokenize,
        prompts=base.prompt_templates, device="cpu")


def smoke_inputs():
    images = torch.randn(4, 3, 32, 32)
    meta = dict(patch_grid_shape=[(1, 1)] * 4, img_shape=[(32, 32)] * 4)
    cfg = dict(patch_size=[32, 32], patch_stride=16, steps=2, trace_view_batch_size=3)
    return images, torch.ones(4, 1, 32, 32), torch.arange(4), meta, cfg, [0, 1]


def test_actual_cpu_smoke_complete_same_object_reset_rng_and_count_gate():
    method = tiny_standalone()
    before = worker.shared.restored_state(method)
    rng = worker.shared.rng_state(method.device)
    report = smoke.compare_episode(method, *smoke_inputs())
    assert report["passed"] is True and report["changed_label_pixels"] == 0
    assert report["max_loss_abs_difference"] == report["max_support_abs_difference"] == 0
    assert report["count_checks_passed"] is True and report["prediction_shapes"] == [[32, 32]] * 4
    assert type(method) is worker.StandaloneTRACE and worker.shared.restored_state(method) == before
    # compare_episode restores the caller RNG, including the input-generation state at entry.
    worker.shared.restore_rng(rng, method.device)


def test_smoke_rejects_changed_labels_and_restores_class_on_error(monkeypatch):
    method = tiny_standalone()
    original = smoke.final_predictions
    def changed(method, *args):
        result = original(method, *args)
        if type(method) is worker.StandaloneTRACE:
            result[0][0, 0] = 1 - result[0][0, 0]
        return result
    monkeypatch.setattr(smoke, "final_predictions", changed)
    report = smoke.compare_episode(method, *smoke_inputs())
    assert report["passed"] is False and report["changed_label_pixels"] == 1
    def failed(*args):
        raise RuntimeError("Synthetic finite gate failure")
    monkeypatch.setattr(smoke, "final_predictions", failed)
    with pytest.raises(RuntimeError):
        smoke.compare_episode(method, *smoke_inputs())
    assert type(method) is worker.StandaloneTRACE and not hasattr(method, "_trace_execution_audit")


@pytest.fixture
def predict_args(tmp_path, monkeypatch):
    parent = parent_fixture(tmp_path / "parent", "pc60")
    cfg = tmp_path / "config.json"
    queue.write_json(cfg, parent["config"])
    manifest = mock_manifest(tmp_path, "pc60", cfg, 4, None, ["min"], parent)
    manifest["source_sha256"] = worker.checked_sources()
    path = tmp_path / "manifest.json"
    queue.write_json(path, manifest)
    monkeypatch.setattr(worker, "build_panel", lambda *args: (None, [], copy.deepcopy(manifest["panel"])))
    monkeypatch.setattr(worker.shared, "checkpoint_identity", lambda *args: copy.deepcopy(manifest["checkpoint"]))
    monkeypatch.setattr(worker.shared, "configure_deterministic_runtime", lambda: {})
    return SimpleNamespace(manifest=path, manifest_sha256=queue.sha256(path), output=tmp_path / "predictions"), manifest


@pytest.mark.parametrize("failure", ["source", "config", "checkpoint", "panel", "digest", "lock"])
def test_actual_predict_rejects_before_model_constructor(predict_args, monkeypatch, tmp_path, failure):
    args, manifest = predict_args
    constructor = []
    monkeypatch.setattr(worker, "StandaloneTRACE", lambda **kwargs: constructor.append(kwargs))
    if failure == "source":
        monkeypatch.setattr(worker, "checked_sources", lambda: {})
    elif failure == "config":
        Path(manifest["config_path"]).write_text("{}")
    elif failure == "checkpoint":
        monkeypatch.setattr(worker.shared, "checkpoint_identity", lambda *args: {})
    elif failure == "panel":
        monkeypatch.setattr(worker, "build_panel", lambda *args: (None, [], {}))
    elif failure == "digest":
        args.manifest_sha256 = "0" * 64
    else:
        lock = tmp_path / "gpu.lock"
        monkeypatch.setattr(worker, "GPU_LOCK", lock)
        with lock.open("a") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(BlockingIOError):
                worker.predict(args)
        assert constructor == []
        return
    with pytest.raises(RuntimeError):
        worker.predict(args)
    assert constructor == []


def test_actual_predict_releases_shared_lock_after_constructor_error(predict_args, monkeypatch, tmp_path):
    args, manifest = predict_args
    lock = tmp_path / "gpu.lock"
    monkeypatch.setattr(worker, "GPU_LOCK", lock)
    def failed(**kwargs):
        raise RuntimeError("Synthetic model-constructor failure before CUDA")
    monkeypatch.setattr(worker, "StandaloneTRACE", failed)
    with pytest.raises(RuntimeError, match="constructor failure"):
        worker.predict(args)
    with lock.open("a") as next_owner:
        fcntl.flock(next_owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(next_owner, fcntl.LOCK_UN)
    assert not torch.cuda.is_initialized()


def test_smoke_owns_and_releases_global_lock_on_model_error(prepared_suite, monkeypatch, tmp_path):
    suite, prepared, _ = prepared_suite
    entry = prepared["smokes"][0]
    manifest = queue.read_json(suite / entry["manifest"])
    (suite / "smoke").mkdir()
    lock = tmp_path / "smoke-gpu.lock"
    monkeypatch.setattr(queue, "verify_prepared", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(worker, "checked_sources", lambda: manifest["source_sha256"])
    monkeypatch.setattr(worker, "build_panel", lambda *args: (None, [], manifest["panel"]))
    monkeypatch.setattr(worker.shared, "checkpoint_identity", lambda *args: manifest["checkpoint"])
    monkeypatch.setattr(worker.shared, "configure_deterministic_runtime", strict_runtime)
    monkeypatch.setattr(worker, "GPU_LOCK", lock)
    def failed(**kwargs):
        with lock.open("a") as observer:
            with pytest.raises(BlockingIOError):
                fcntl.flock(observer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise RuntimeError("Synthetic gate constructor error, no CUDA")
    monkeypatch.setattr(worker, "StandaloneTRACE", failed)
    with pytest.raises(RuntimeError, match="gate constructor error"):
        smoke.run(SimpleNamespace(suite=suite, suite_sha256=queue.sha256(suite / "SUITE_PREPARED.json"), dataset=entry["dataset"]))
    assert queue.read_json(suite / entry["output"])["passed"] is False
    with lock.open("a") as next_owner:
        fcntl.flock(next_owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(next_owner, fcntl.LOCK_UN)
    assert not torch.cuda.is_initialized()
