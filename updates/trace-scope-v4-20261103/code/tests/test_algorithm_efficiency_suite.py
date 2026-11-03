"""CPU-only checks for the fixed-config algorithm queue and its evidence seals."""

import copy
from contextlib import ExitStack
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_algorithm_efficiency_suite as suite


def lazy_row(image, cls, target, cf):
    return dict(image_id=image, class_idx=str(cls), target_support=str(target),
        counterfactual_support="" if cf is None else str(cf), residual_support="", dominance="",
        quality="" if cf is None else str(min(target, cf)),
        reliability="0.0" if cf is None else str(min(target, cf)), omitted=str(cf is None))


def test_lazy_validator_catches_unprotected_small_class():
    candidates = [lazy_row("a", 0, .9, .9), lazy_row("a", 1, .005, None)]
    cost = dict(omitted_mass_bound=.005 / .905, omitted_upper_mass=.005)
    suite.validate_lazy_candidates(candidates, cost, "quality_lazy_01", 1e-8)
    with pytest.raises(RuntimeError, match="Per-image/query-class"):
        suite.validate_lazy_candidates(candidates, cost, "quality_lazy_class_01", 1e-8)
    candidates[1]["class_idx"] = "0"
    suite.validate_lazy_candidates(candidates, cost, "quality_lazy_class_01", 1e-8)


def test_lazy_validator_catches_invented_support_and_false_bound():
    candidates = [lazy_row("a", 0, .9, .8), lazy_row("a", 0, .001, None)]
    cost = dict(omitted_mass_bound=.001 / .801, omitted_upper_mass=.001)
    suite.validate_lazy_candidates(candidates, cost, "quality_lazy_01", 1e-8)
    with pytest.raises(RuntimeError, match="certificate"):
        suite.validate_lazy_candidates(candidates, dict(cost, omitted_mass_bound=0), "quality_lazy_01", 1e-8)
    candidates[0]["residual_support"] = "0"
    with pytest.raises(RuntimeError, match="invented residual"):
        suite.validate_lazy_candidates(candidates, cost, "quality_lazy_01", 1e-8)


def test_staged_quality_requires_every_cf_and_no_missing_scores():
    cost = dict(omitted_mass_bound=0, omitted_upper_mass=0)
    suite.validate_lazy_candidates([lazy_row("a", 0, .9, .8)], cost, "quality_staged", 1e-8)
    with pytest.raises(RuntimeError, match="Missing probe"):
        suite.validate_lazy_candidates([lazy_row("a", 0, 0, None)], cost, "quality_staged", 1e-8)


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    monkeypatch.setattr(suite, "execution_environment", lambda: {"device": "cpu-test"})
    import torch
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("CPU queue tests must not initialize CUDA"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cell_fixture(directory, algorithm="standard", *, candidates=2, rho=.5, episodes=3):
    directory.mkdir(parents=True, exist_ok=True)
    images = [f"image-{i}" for i in range(episodes)]
    expected = dict(dataset="voc21", split="evaluation", seed=0, selection_seed=suite.SELECTION_SEED,
        image_ids=images, label_ids=[f"{image}.png" for image in images],
        file_metadata=[[1, 2]] * (2 * episodes), vocabulary_sha256="vocabulary", available_images=1449,
        config=dict(batch_size=1, steps=5, lr=.001, trace_max_regions=33, trace_view_batch_size=16))
    manifest = dict(source_hashes={"source.py": "original"}, environment={"device": "cpu-test"},
        tool_hashes={"benchmark_simplification.py": "benchmark", "trace_ablation_variants.py": "ablation",
                     "trace_algorithm_variants.py": "algorithms"}, objective_epsilon=1e-8,
        input_panels={"voc21/4": expected}, data_root="/data", data_dirs={})
    protocol = dict(**expected, algorithm=algorithm, variant="full", images=episodes, full_split=False,
        source_hashes=manifest["source_hashes"], experiment_hashes=suite.experiment_hashes(manifest, algorithm),
        environment=manifest["environment"], configuration_overrides={}, force_uniform_policy=False,
        timing_scope=suite.TIMING_SCOPE)
    write_json(directory / "protocol.json", protocol)
    counts = [dict(image_id=image, prediction_sha256="a" * 64, intersection=[1, 1], union=[2, 2])
              for image in images]
    write_json(directory / "per_image_counts.json", counts)
    costs, diagnostic_rows = [], []
    for episode, image in enumerate(images):
        traced = candidates if algorithm != "quality_only" else 0
        views = candidates * (2 if algorithm == "quality_only" else 3)
        costs.append(dict(episode=episode, images=1, adaptation_seconds=episode + 1., inference_seconds=.5,
            total_seconds=episode + 1.5, peak_allocated_mib=100 + episode,
            update_steps=5 if candidates and rho > 1e-8 else 0, candidates=candidates,
            diagnostic_views=views, fully_traced_candidates=traced, effective_view_batch_size=16,
            calls=10, inputs=views + 10, omitted_mass_bound=0., omitted_upper_mass=0.))
        diagnostic_rows.extend(dict(episode=episode, image_id=image, reliability=rho,
                                    fully_traced=algorithm != "quality_only") for _ in range(candidates))
    write_csv(directory / "cost.csv", costs)
    if diagnostic_rows:
        write_csv(directory / "candidate_diagnostics.csv", diagnostic_rows)
    warmup = min(2, episodes - 1)
    measured = costs[warmup:]
    result = dict(status="completed", algorithm=algorithm, variant="full", dataset="voc21",
        split="evaluation", images=episodes, full_split=False, intersection=[episodes, episodes],
        union=[2 * episodes, 2 * episodes], miou_unrounded=50., metrics={"mIoU": 50., "mAcc": 75.},
        ms_per_image=1000 * sum(row["total_seconds"] for row in measured) / len(measured),
        peak_allocated_mib=max(row["peak_allocated_mib"] for row in measured),
        diagnostic_views_per_image=measured[0]["diagnostic_views"], warmup_episodes=warmup,
        protocol_sha256=suite.digest(directory / "protocol.json"))
    write_json(directory / "result.json", result)
    return expected, manifest, protocol, result, costs


def reseal_protocol(directory, protocol):
    write_json(directory / "protocol.json", protocol)
    result = json.loads((directory / "result.json").read_text())
    result["protocol_sha256"] = suite.digest(directory / "protocol.json")
    write_json(directory / "result.json", result)


def test_fixed_plan_has_all_smoke_first_and_twenty_rotated_evaluation_cells():
    cells = suite.experiment_plan()
    assert len(cells) == 25 and len({c["key"] for c in cells}) == 25
    assert [c["algorithm"] for c in cells[:5]] == list(suite.ALGORITHMS)
    assert all(c["stage"] == "smoke" and c["dataset"] == "voc21" and c["limit"] == 4 for c in cells[:5])
    for index, dataset in enumerate(suite.DATASETS):
        group = [c for c in cells[5:] if c["dataset"] == dataset]
        assert [c["algorithm"] for c in group] == list(suite.ALGORITHMS[index:] + suite.ALGORITHMS[:index])
        assert all(c["stage"] == "evaluation" and c["limit"] == 128 for c in group)
    assert all(c["variant"] == "full" for c in cells)


def test_worker_commands_never_override_optimizer_k_batch_steps_or_lr(tmp_path):
    manifest = dict(data_root="/dataset-root", data_dirs={"coco_object": "/converted-coco"})
    forbidden = {"--batch-size", "--view-batch-size", "--max-regions", "--steps", "--lr", "--force-uniform-policy"}
    for cell in suite.experiment_plan():
        command = suite.worker_command(tmp_path / "runtime", manifest, cell, tmp_path / cell["key"])
        assert not forbidden.intersection(command)
        assert command[command.index("--algorithm") + 1] == cell["algorithm"]
        assert command[command.index("--variant") + 1] == "full"
        assert command[command.index("--selection-seed") + 1] == "20261024"
        assert command[command.index("--split") + 1] == "evaluation"
        assert command[command.index("--max-images") + 1] == str(cell["limit"])
        assert ("--data-dir" in command) == (cell["dataset"] == "coco_object")


def test_input_discovery_uses_identical_seed_and_no_training_selection(tmp_path, monkeypatch):
    captured = {}
    def discover(root, data_root, plan, **kwargs):
        captured.update(root=root, data_root=data_root, plan=plan, **kwargs)
        return {"panels": "fixed"}
    monkeypatch.setattr(suite, "expected_inputs", discover)
    manifest = dict(data_root="/data", data_dirs={"coco_object": "/objects"}, cells=suite.experiment_plan())
    assert suite.discover_panels(tmp_path, manifest) == {"panels": "fixed"}
    assert len(captured["plan"]) == 5
    assert captured["selection_seed"] == 20261024
    assert set(captured["plan"].values()) == {("voc21", "evaluation", 4), *[(d, "evaluation", 128) for d in suite.DATASETS]}


def test_transitive_freeze_includes_function_imports_relative_helpers_and_packages(tmp_path):
    (tmp_path / "entry.py").write_text("import helper\nfrom package import child\n")
    (tmp_path / "helper.py").write_text("def f():\n    import leaf\n")
    (tmp_path / "leaf.py").write_text("import sys\n")
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "child.py").write_text("from . import sibling\n")
    (package / "sibling.py").write_text("from leaf import value\n")
    (tmp_path / "unrelated.py").write_text("raise RuntimeError('do not import')\n")
    assert set(suite.tool_dependencies(tmp_path, ("entry.py",))) == {
        "entry.py", "helper.py", "leaf.py", "package/__init__.py", "package/child.py", "package/sibling.py"}


def test_real_freeze_contains_every_helper_and_algorithm_without_importing_model():
    tools = Path(suite.__file__).parent
    names = suite.tool_dependencies(tools)
    assert {suite.CONTROLLER, "trace_algorithm_variants.py", "benchmark_simplification.py",
            "trace_ablation_variants.py", "suite_inputs.py", "run_removal_suite.py",
            "finalize_removal_suite.py"}.issubset(names)
    assert "run_voc21_batch_tuning.py" not in names
    assert "run_accuracy_budget_search.py" not in names


def test_snapshot_freezes_source_and_transitive_helpers_and_rejects_drift(tmp_path):
    package, tools, output = tmp_path / "package", tmp_path / "tools", tmp_path / "out"
    (package / "trace_ovss").mkdir(parents=True)
    (package / "trace_ovss/__init__.py").write_text("source = 1\n")
    tools.mkdir()
    (tools / "entry.py").write_text("import helper\n")
    (tools / "helper.py").write_text("value = 1\n")
    request = dict(source_hashes=suite.source_hashes(package / "trace_ovss"),
        tool_hashes={name: suite.digest(tools / name) for name in suite.tool_dependencies(tools, ("entry.py",))})
    runtime = suite.snapshot_runtime(output, package, tools, request)
    assert (runtime / "tools/helper.py").exists()
    (tools / "helper.py").write_text("value = 2\n")
    suite.verify_snapshot(runtime, request)
    (runtime / "tools/helper.py").write_text("value = 2\n")
    with pytest.raises(RuntimeError, match="transitive helpers"):
        suite.verify_snapshot(runtime, request)


def test_incomplete_snapshot_is_not_removed_or_retried(tmp_path):
    incomplete = tmp_path / "runtime.preparing"
    incomplete.mkdir()
    marker = incomplete / "partial"
    marker.write_text("preserve")
    with pytest.raises(RuntimeError, match="Incomplete snapshot"):
        suite.snapshot_runtime(tmp_path, tmp_path, tmp_path, {})
    assert marker.read_text() == "preserve"


@pytest.mark.parametrize("algorithm", suite.ALGORITHMS)
def test_artifacts_require_explicit_algorithm_and_algorithm_specific_hashes(tmp_path, algorithm):
    expected, manifest, protocol, result, costs = cell_fixture(tmp_path, algorithm)
    validated, summary, raw = suite.validate_algorithm_artifact(tmp_path, expected, manifest, algorithm)
    assert validated == result and len(raw) == 3
    assert summary["measured_images"] == 1
    assert summary["all_visual_inputs"] == sum(row["inputs"] for row in costs)
    assert summary["all_visual_calls"] == 30
    assert summary["visual_inputs_per_image"] == costs[-1]["inputs"]
    assert ("trace_algorithm_variants.py" in protocol["experiment_hashes"]) == (algorithm != "standard")
    protocol["algorithm"] = "wrong"
    reseal_protocol(tmp_path, protocol)
    with pytest.raises(RuntimeError, match="Algorithm identity"):
        suite.validate_algorithm_artifact(tmp_path, expected, manifest, algorithm)


@pytest.mark.parametrize("change,match", [
    ("hash", "implementation"), ("source", "Algorithm identity"), ("environment", "environment"),
    ("config", "configuration"), ("override", "override parameters"), ("cohort", "frozen inputs"),
    ("metadata", "input-file identity"), ("timing", "timing scope"), ("variant", "Algorithm identity")])
def test_protocol_rejects_identity_cohort_environment_and_configuration_drift(tmp_path, change, match):
    expected, manifest, protocol, _, _ = cell_fixture(tmp_path)
    if change == "hash":
        protocol["experiment_hashes"]["benchmark_simplification.py"] = "changed"
    elif change == "source":
        protocol["source_hashes"] = {}
    elif change == "environment":
        protocol["environment"] = {}
    elif change == "config":
        protocol["config"] = {**protocol["config"], "steps": 3}
    elif change == "override":
        protocol["configuration_overrides"] = {"batch_size": 1}
    elif change == "cohort":
        protocol["image_ids"] = list(reversed(protocol["image_ids"]))
    elif change == "metadata":
        protocol["file_metadata"] = []
    elif change == "timing":
        protocol["timing_scope"] = "unsynchronized"
    else:
        protocol["variant"] = "no_source_tracing"
    reseal_protocol(tmp_path, protocol)
    with pytest.raises(RuntimeError, match=match):
        suite.validate_algorithm_artifact(tmp_path, expected, manifest, "standard")


@pytest.mark.parametrize("key,value", [("ms_per_image", 1), ("peak_allocated_mib", 10000),
    ("diagnostic_views_per_image", 0), ("warmup_episodes", 0), ("miou_unrounded", 99)])
def test_metrics_and_cost_summaries_are_recomputed(tmp_path, key, value):
    expected, manifest, _, result, _ = cell_fixture(tmp_path)
    result[key] = value
    write_json(tmp_path / "result.json", result)
    with pytest.raises(RuntimeError):
        suite.validate_algorithm_artifact(tmp_path, expected, manifest, "standard")


@pytest.mark.parametrize("algorithm", suite.ALGORITHMS)
@pytest.mark.parametrize("candidates,rho", [(0, 0), (2, 0), (2, 1e-8)])
def test_zero_steps_only_when_candidate_audit_proves_empty_objective(tmp_path, algorithm, candidates, rho):
    expected, manifest, _, _, _ = cell_fixture(tmp_path, algorithm, candidates=candidates, rho=rho)
    suite.validate_algorithm_artifact(tmp_path, expected, manifest, algorithm)
    if not candidates:
        assert not (tmp_path / "candidate_diagnostics.csv").exists()


@pytest.mark.parametrize("steps", [0, 1, 4, 6])
def test_nonempty_objective_cannot_silently_reduce_update_steps(tmp_path, steps):
    expected, manifest, _, _, costs = cell_fixture(tmp_path)
    costs[0]["update_steps"] = steps
    write_csv(tmp_path / "cost.csv", costs)
    with pytest.raises(RuntimeError, match="update steps"):
        suite.validate_algorithm_artifact(tmp_path, expected, manifest, "standard")


def test_missing_candidate_audit_cannot_justify_zero_steps(tmp_path):
    expected, manifest, _, _, _ = cell_fixture(tmp_path, rho=0)
    (tmp_path / "candidate_diagnostics.csv").unlink()
    with pytest.raises(RuntimeError, match="complete episode"):
        suite.validate_algorithm_artifact(tmp_path, expected, manifest, "standard")


@pytest.mark.parametrize("algorithm", suite.ALGORITHMS)
def test_algorithm_view_relationships_cannot_be_inferred_or_falsified(tmp_path, algorithm):
    expected, manifest, _, _, costs = cell_fixture(tmp_path, algorithm)
    costs[0]["diagnostic_views"] += 1
    write_csv(tmp_path / "cost.csv", costs)
    with pytest.raises(RuntimeError, match="diagnostic views"):
        suite.validate_algorithm_artifact(tmp_path, expected, manifest, algorithm)


@pytest.mark.parametrize("algorithm,bound", [("selective_01", .01), ("selective_05", .05)])
def test_selective_actual_bounds_are_checked_per_episode(tmp_path, algorithm, bound):
    expected, manifest, _, _, costs = cell_fixture(tmp_path, algorithm)
    costs[0]["omitted_mass_bound"] = bound + .001
    write_csv(tmp_path / "cost.csv", costs)
    with pytest.raises(RuntimeError, match="tolerance"):
        suite.validate_algorithm_artifact(tmp_path, expected, manifest, algorithm)


@pytest.mark.parametrize("field,value", [("inputs", "nan"), ("calls", -1), ("episode", 7),
    ("images", 2), ("total_seconds", 0), ("fully_traced_candidates", 100), ("omitted_upper_mass", -1)])
def test_invalid_costs_are_not_accepted(tmp_path, field, value):
    expected, manifest, _, _, costs = cell_fixture(tmp_path)
    costs[0][field] = value
    write_csv(tmp_path / "cost.csv", costs)
    with pytest.raises(RuntimeError):
        suite.validate_algorithm_artifact(tmp_path, expected, manifest, "standard")


def test_queue_waits_for_batch_then_audit_then_parent_locks_and_checks_states(tmp_path, monkeypatch):
    dependency, parent, output = tmp_path / "batch", tmp_path / "parent", tmp_path / "output"
    output.mkdir()
    manifest = dict(batch_tuning=str(dependency), parent_suite=str(parent))
    for root in (dependency, parent, parent / "final_audit"):
        write_json(root / "queue_state.json", {"status": "completed"})
    for key, root in (("dependency_protocol_sha256", dependency), ("parent_protocol_sha256", parent)):
        write_json(root / "suite_protocol.json", {"source_hashes": {}})
        manifest[key] = suite.digest(root / "suite_protocol.json")
    order, sleeps = [], []
    real_lock = suite.acquire_lock
    def lock(stack, path, **kwargs):
        order.append(path)
        return real_lock(stack, path, **kwargs)
    calls = iter(([123], [], [456], []))
    monkeypatch.setattr(suite, "acquire_lock", lock)
    monkeypatch.setattr(suite, "active_workers", lambda root: next(calls))
    monkeypatch.setattr(suite.time, "sleep", sleeps.append)
    with ExitStack() as stack:
        locks = suite.wait_for_dependencies(output, manifest, stack)
        assert len(locks) == 3
    assert order == [dependency / "queue.lock", parent / "final_audit/audit.lock", parent / "queue.lock"]
    assert sleeps == [5, 5]
    check = json.loads((output / "dependency_checks.json").read_text())
    assert check["no_active_benchmark_workers"] and check["inherited_config_or_outcomes"] is False


def test_failed_batch_dependency_is_checked_after_its_lock_and_never_runs_workers(tmp_path, monkeypatch):
    dependency = tmp_path / "batch"
    write_json(dependency / "queue_state.json", {"status": "failed"})
    monkeypatch.setattr(suite, "active_workers", lambda root: pytest.fail("must reject the failed dependency first"))
    with ExitStack() as stack, pytest.raises(RuntimeError, match="Dependency did not complete"):
        suite.wait_for_dependencies(tmp_path, dict(batch_tuning=str(dependency), parent_suite=str(tmp_path / "parent")), stack)


def test_cell_passes_all_locks_to_worker_and_keeps_original_config(tmp_path, monkeypatch):
    cell = suite.experiment_plan()[0]
    expected, manifest, _, _, _ = cell_fixture(tmp_path / "fixture")
    captured = {}
    def worker(command, **kwargs):
        captured.update(command=command, **kwargs)
        directory = Path(command[command.index("--output") + 1])
        cell_fixture(directory)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(suite.subprocess, "run", worker)
    original = copy.deepcopy(manifest)
    with ExitStack() as stack:
        locks = tuple(suite.acquire_lock(stack, tmp_path / f"{i}.lock") for i in range(4))
        row, episodes = suite.run_cell(tmp_path, tmp_path / "runtime", manifest, cell, locks)
        assert captured["pass_fds"] == tuple(lock.fileno() for lock in locks)
    assert manifest == original and row["config_fixed"] and len(episodes) == 3
    assert (tmp_path / cell["key"] / "attempt_001/checks.json").exists()


def test_failure_is_preserved_and_no_automatic_retry_or_later_attempt(tmp_path, monkeypatch):
    cell = suite.experiment_plan()[0]
    _, manifest, _, _, _ = cell_fixture(tmp_path / "fixture")
    calls = []
    def worker(command, **kwargs):
        calls.append(command)
        directory = Path(command[command.index("--output") + 1])
        write_json(directory / "failure.json", {"error": "OutOfMemoryError"})
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(suite.subprocess, "run", worker)
    with pytest.raises(RuntimeError, match="evidence retained"):
        suite.run_cell(tmp_path, tmp_path, manifest, cell, ())
    with pytest.raises(RuntimeError, match="no automatic retry"):
        suite.run_cell(tmp_path, tmp_path, manifest, cell, ())
    assert len(calls) == 1
    assert (tmp_path / cell["key"] / "attempt_001/failure.json").exists()
    assert not (tmp_path / cell["key"] / "attempt_002").exists()


def test_accepted_artifact_cost_and_diagnostic_seals_reject_later_changes(tmp_path):
    cell = suite.experiment_plan()[0]
    directory = tmp_path / cell["key"] / "attempt_001"
    _, manifest, _, _, _ = cell_fixture(directory)
    suite.run_cell(tmp_path, tmp_path, manifest, cell, ())
    suite.run_cell(tmp_path, tmp_path, manifest, cell, (), allow_execution=False)
    with (directory / "candidate_diagnostics.csv").open("a") as handle:
        handle.write("extra evidence\n")
    with pytest.raises(RuntimeError, match="seal changed"):
        suite.run_cell(tmp_path, tmp_path, manifest, cell, ())


def test_final_validation_cannot_launch_a_missing_cell(tmp_path, monkeypatch):
    cell = suite.experiment_plan()[0]
    _, manifest, _, _, _ = cell_fixture(tmp_path / "fixture")
    monkeypatch.setattr(suite.subprocess, "run", lambda *a, **k: pytest.fail("No retry during final validation"))
    with pytest.raises(RuntimeError, match="no automatic retry"):
        suite.run_cell(tmp_path, tmp_path, manifest, cell, (), allow_execution=False)


def test_reports_retain_negative_and_positive_deltas_without_promotion(tmp_path):
    rows = [dict(stage=stage, dataset=dataset, algorithm=algorithm, mIoU=score)
            for stage in ("smoke", "evaluation") for dataset in suite.DATASETS
            for algorithm, score in zip(suite.ALGORITHMS, (50, 51, 49, 52, 50))]
    suite.save_tables(tmp_path, rows, [dict(calls=7, inputs=99, fully_traced_candidates=2)])
    with (tmp_path / "algorithm_summary.csv").open() as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 20
    quality = [r for r in summary if r["algorithm"] == "quality_only"]
    assert all(float(r["delta_mIoU_vs_standard"]) == -1 and float(r["delta_mIoU_vs_staged_full"]) == -2 for r in quality)
    assert any(float(r["delta_mIoU_vs_standard"]) > 0 for r in summary)
    assert not (tmp_path / "selection.json").exists()
    assert not (tmp_path / "selected_config.json").exists()


def test_detached_launch_uses_frozen_controller_and_inherited_lock(tmp_path, monkeypatch):
    args = SimpleNamespace(**{name: tmp_path / name for name in suite.PATH_ARGUMENTS}, detach=True)
    args.output.mkdir()
    captured = {}
    def launch(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(pid=12345)
    monkeypatch.setattr(suite.subprocess, "Popen", launch)
    with ExitStack() as stack:
        lock = suite.acquire_lock(stack, args.output / "queue.lock")
        suite.launch_controller(args, lock)
        assert captured["pass_fds"] == (lock.fileno(),)
    assert captured["command"][1] == str(args.output / "runtime/tools" / suite.CONTROLLER)
    assert captured["cwd"] == str(args.output / "runtime")
    assert captured["start_new_session"] and "--execute-prepared" in captured["command"]
    assert "--detach" not in captured["command"]
    assert json.loads((args.output / "launch.json").read_text())["pid"] == 12345


def test_inherited_lock_survives_parent_handle_close_without_gpu(tmp_path):
    lock = (tmp_path / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
                             stdin=subprocess.PIPE, pass_fds=(lock.fileno(),))
    lock.close()
    try:
        with (tmp_path / "queue.lock").open("a") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        child.communicate(b"x", timeout=10)
    with (tmp_path / "queue.lock").open("a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_prepare_only_never_launches_waits_or_touches_dependencies(tmp_path, monkeypatch):
    arguments = []
    for name in suite.PATH_ARGUMENTS:
        arguments += ["--" + name.replace("_", "-"), str(tmp_path / name)]
    prepared = []
    monkeypatch.setattr(suite, "prepare_suite", lambda args: prepared.append(args))
    monkeypatch.setattr(suite, "launch_controller", lambda *a: pytest.fail("prepare-only must not launch"))
    monkeypatch.setattr(suite, "wait_for_dependencies", lambda *a: pytest.fail("prepare-only must not wait"))
    suite.main([*arguments, "--prepare-only"])
    assert len(prepared) == 1
    state = json.loads((tmp_path / "output/queue_state.json").read_text())
    assert state == dict(status="prepared", smoke_cells=5, evaluation_cells=20, cells_total=25)
    assert not (tmp_path / "batch_tuning").exists()
    assert not (tmp_path / "parent_suite").exists()


def test_failed_queue_is_not_reprepared_or_restarted(tmp_path, monkeypatch):
    write_json(tmp_path / "output/queue_state.json", dict(status="failed", reason="keep evidence"))
    arguments = [value for name in suite.PATH_ARGUMENTS
                 for value in ("--" + name.replace("_", "-"), str(tmp_path / name))]
    monkeypatch.setattr(suite, "prepare_suite", lambda args: pytest.fail("do not retry"))
    with pytest.raises(RuntimeError, match="no automatic retry"):
        suite.main(arguments)
    assert json.loads((tmp_path / "output/queue_state.json").read_text())["reason"] == "keep evidence"


def test_internal_execution_requires_frozen_path_before_any_gpu_work(tmp_path):
    arguments = [value for name in suite.PATH_ARGUMENTS
                 for value in ("--" + name.replace("_", "-"), str(tmp_path / name))]
    with pytest.raises(RuntimeError, match="frozen controller"):
        suite.main([*arguments, "--execute-prepared"])


def test_completed_suite_cannot_replace_missing_or_changed_sealed_files(tmp_path):
    write_json(tmp_path / "summary.json", {"score": 50})
    write_json(tmp_path / "SUITE_SEALED.json", dict(artifact_sha256={
        "summary.json": suite.digest(tmp_path / "summary.json")}))
    assert suite.verify_suite_seal(tmp_path)
    write_json(tmp_path / "summary.json", {"score": 99})
    with pytest.raises(RuntimeError, match="Completed suite evidence changed"):
        suite.verify_suite_seal(tmp_path)
    (tmp_path / "summary.json").unlink()
    with pytest.raises(RuntimeError, match="Completed suite evidence changed"):
        suite.verify_suite_seal(tmp_path)


def test_whole_cpu_queue_runs_all_negative_rows_and_final_revalidation_without_selection(tmp_path, monkeypatch):
    manifest = dict(cells=suite.experiment_plan())
    calls = []
    monkeypatch.setattr(suite, "wait_for_dependencies", lambda *args: ())
    monkeypatch.setattr(suite, "verify_snapshot", lambda *args: None)
    monkeypatch.setattr(suite, "verify_execution_inputs", lambda *args: None)
    for name in ("suite_protocol.json", "requested_protocol.json", "fixed_configs.json", "dependency_checks.json"):
        write_json(tmp_path / name, {})
    def run(output, runtime, protocol, cell, locks, *, allow_execution=True):
        calls.append((cell["key"], allow_execution))
        write_json(output / cell["key"] / "selected_result.json", {"validated": True})
        row = dict(stage=cell["stage"], dataset=cell["dataset"], algorithm=cell["algorithm"],
                   mIoU=50 if cell["algorithm"] in {"standard", "staged_full"} else 1)
        return row, [dict(algorithm=cell["algorithm"], inputs=99, calls=3)]
    monkeypatch.setattr(suite, "run_cell", run)
    with ExitStack() as stack:
        lock = suite.acquire_lock(stack, tmp_path / "queue.lock")
        suite.execute_suite(tmp_path, manifest, lock, stack)
    assert calls[:25] == [(c["key"], True) for c in manifest["cells"]]
    assert calls[25:] == [(c["key"], False) for c in manifest["cells"]]
    assert json.loads((tmp_path / "queue_state.json").read_text())["completed"] == 25
    assert suite.verify_suite_seal(tmp_path)
    assert not (tmp_path / "selection.json").exists()
    calls.clear()
    with ExitStack() as stack:
        lock = suite.acquire_lock(stack, tmp_path / "queue.lock")
        suite.execute_suite(tmp_path, manifest, lock, stack)
    assert len(calls) == 50 and all(not execution for _, execution in calls)


def test_foreground_launcher_propagates_controller_failure(tmp_path, monkeypatch):
    args = SimpleNamespace(**{name: tmp_path / name for name in suite.PATH_ARGUMENTS}, detach=False)
    args.output.mkdir()
    captured = {}
    def run(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(returncode=9)
    monkeypatch.setattr(suite.subprocess, "run", run)
    with ExitStack() as stack:
        lock = suite.acquire_lock(stack, args.output / "queue.lock")
        with pytest.raises(RuntimeError, match="Frozen controller failed"):
            suite.launch_controller(args, lock)
        assert captured["pass_fds"] == (lock.fileno(),)


def test_script_help_works_from_unrelated_cwd_without_pythonpath_or_gpu(tmp_path):
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run([sys.executable, str(Path(suite.__file__).resolve()), "--help"],
                            cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    for flag in ("--batch-tuning", "--parent-suite", "--coco-object-dir", "--prepare-only", "--detach"):
        assert flag in result.stdout
