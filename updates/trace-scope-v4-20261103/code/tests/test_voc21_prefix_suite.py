"""CPU-only contracts for the bounded, detached VOC21 forward-reduction queue."""

from contextlib import ExitStack
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_voc21_prefix_suite as suite
from run_algorithm_efficiency_suite import experiment_hashes, worker_command


def panels():
    config = dict(batch_size=16, steps=36, trace_max_regions=33, lr=.00063,
                  trace_view_batch_size=16, resize=[224, 224], class_extensions=True)

    def panel(split, count, available):
        return dict(dataset="voc21", split=split, seed=0, selection_seed=20261024,
                    image_ids=[f"/data/{split}_{i}.jpg" for i in range(count)],
                    label_ids=[f"/data/{split}_{i}.png" for i in range(count)],
                    file_metadata=[[1, 2] for _ in range(2 * count)], config=dict(config),
                    vocabulary_sha256="a" * 64, available_images=available)

    return {suite.SMOKE_PANEL: panel("calibration", 4, 1464),
            suite.CALIBRATION_PANEL: panel("calibration", 128, 1464),
            suite.FULL_PANEL: panel("evaluation", 1449, 1449)}


def seal(root, names):
    suite.write_json(root / "SUITE_SEALED.json", dict(
        artifact_sha256={name: suite.digest(root / name) for name in names}))


def parameter_record(algorithm):
    names = suite.parameter_names(algorithm)
    return dict(algorithm=algorithm, encoder_blocks=24, names=list(reversed(names)),
                actual_names=list(names), optimizer_names=list(names),
                parameter_numel={name: 1024 for name in names}, trainable_parameters=1024 * len(names))


def test_plan_is_exactly_thirteen_fixed_cells_with_lazy_followups_first():
    cells = suite.plan()
    assert len(cells) == len({c["key"] for c in cells}) == suite.TOTAL_CELLS == 13
    assert [c["algorithm"] for c in cells[:2]] == ["quality_lazy_01", "quality_lazy_class_01"]
    assert all(c["stage"] == "full_validation" and c["limit"] is None for c in cells[:2])
    assert [c["algorithm"] for c in cells[2:6]] == list(suite.ALGORITHMS)
    assert all(c["stage"] == "smoke" and c["limit"] == 4 for c in cells[2:6])
    assert [c["algorithm"] for c in cells[6:11]] == [*suite.ALGORITHMS, "quality_only"]
    assert all(c["stage"] == "calibration" and c["limit"] == 128 for c in cells[6:11])
    assert [c["algorithm"] for c in cells[-2:]] == ["quality_tail12", "quality_tail6"]
    assert all(c["dataset"] == "voc21" and c["variant"] == "full" for c in cells)
    assert all(c["split"] == "calibration" for c in cells[2:11])
    assert all(c["split"] == "evaluation" and c["limit"] is None for c in cells if c["stage"] == "full_validation")


def test_commands_preserve_split_and_full_population_without_tuning(tmp_path):
    forbidden = {"--steps", "--lr", "--max-regions", "--batch-size", "--view-batch-size",
                 "--force-uniform-policy", "--resize", "--prompts"}
    manifest = dict(data_root="/data", data_dirs={"voc21": "/voc"})
    for cell in suite.plan():
        command = worker_command(tmp_path, manifest, cell, tmp_path / cell["key"])
        assert command[command.index("--split") + 1] == cell["split"]
        assert command[command.index("--algorithm") + 1] == cell["algorithm"]
        assert command[command.index("--data-dir") + 1] == "/voc"
        assert command[command.index("--seed") + 1] == "0"
        assert command[command.index("--selection-seed") + 1] == "20261024"
        assert not forbidden.intersection(command)
        if cell["limit"] is None:
            assert "--max-images" not in command
        else:
            assert command[command.index("--max-images") + 1] == str(cell["limit"])


def test_discovery_uses_fresh_runtime_and_split_specific_panels(monkeypatch, tmp_path):
    seen = {}

    def expected(root, data, plan, **kwargs):
        seen.update(root=root, data=data, plan=plan, **kwargs)
        return panels()

    monkeypatch.setattr(suite, "expected_inputs", expected)
    assert suite.discover(tmp_path, dict(cells=suite.plan(), data_root="/data", data_dirs={})) == panels()
    assert seen["root"] == tmp_path
    assert seen["plan"] == {suite.SMOKE_PANEL: ("voc21", "calibration", 4),
                            suite.CALIBRATION_PANEL: ("voc21", "calibration", 128),
                            suite.FULL_PANEL: ("voc21", "evaluation", None)}


@pytest.mark.parametrize("change", ["full_order", "full_metadata", "overlap", "alias_overlap",
                                   "config", "vocabulary", "duplicate", "smoke_not_nested", "split"])
def test_panel_validation_rejects_drift_and_leakage(change):
    data = panels()
    parent = dict(input_panels=copy.deepcopy(data))
    suite.validate_panels(data, parent)
    if change == "full_order":
        data[suite.FULL_PANEL]["image_ids"].reverse()
    elif change == "full_metadata":
        data[suite.FULL_PANEL]["file_metadata"][0][0] += 1
    elif change in {"overlap", "alias_overlap"}:
        path = data[suite.FULL_PANEL]["image_ids"][0]
        data[suite.CALIBRATION_PANEL]["image_ids"][0] = path if change == "overlap" else path.replace("/data/", "/alias/")
    elif change == "config":
        data[suite.CALIBRATION_PANEL]["config"]["steps"] = 1
    elif change == "vocabulary":
        data[suite.CALIBRATION_PANEL]["vocabulary_sha256"] = "b" * 64
    elif change == "duplicate":
        data[suite.CALIBRATION_PANEL]["image_ids"][1] = data[suite.CALIBRATION_PANEL]["image_ids"][0]
    elif change == "smoke_not_nested":
        data[suite.SMOKE_PANEL]["image_ids"][0] = "/data/unselected.jpg"
    else:
        data[suite.CALIBRATION_PANEL]["split"] = "evaluation"
    with pytest.raises(RuntimeError):
        suite.validate_panels(data, parent)


@pytest.mark.parametrize("key,value", [("automatic_promotion", True), ("parameter_search", True),
                                     ("automatic_retry", True), ("validation_selection", True),
                                     ("full_algorithms", ["quality_tail6"]),
                                     ("full_runs_independent_of_calibration_scores", False)])
def test_frozen_policy_rejects_selection_retry_and_omitted_full_run(key, value):
    policy = suite.study_policy()
    suite.validate_policy(policy)
    policy[key] = value
    with pytest.raises(RuntimeError, match="policy changed"):
        suite.validate_policy(policy)


def test_exact_optimizer_scope_includes_tail_zero_and_all_lazy_parameters(tmp_path):
    policy = suite.study_policy()
    assert suite.parameter_names("quality_tail0") == ["visual.ln_post.bias", "visual.ln_post.weight"]
    assert suite.parameter_names("quality_tail6") == suite.parameter_names("quality_tail6_uncached")
    assert suite.parameter_names("quality_lazy_01") == suite.parameter_names("quality_only")
    assert suite.parameter_names("quality_lazy_class_01") == suite.parameter_names("quality_only")
    assert len(suite.parameter_names("quality_tail6")) == 24
    assert len(suite.parameter_names("quality_tail12")) == 48
    assert len(suite.parameter_names("quality_only")) == 98
    for algorithm in suite.PARAMETER_ALGORITHMS:
        record = parameter_record(algorithm)
        suite.write_json(tmp_path / "adaptation_parameters.json", record)
        suite.validate_parameter_metadata(tmp_path, policy, algorithm)
        record["names"].append("visual.ln_pre.weight")
        suite.write_json(tmp_path / "adaptation_parameters.json", record)
        with pytest.raises(RuntimeError, match="Optimizer parameter names"):
            suite.validate_parameter_metadata(tmp_path, policy, algorithm)


@pytest.mark.parametrize("algorithm", suite.PARAMETER_ALGORITHMS)
@pytest.mark.parametrize("field", ["actual_names", "optimizer_names", "parameter_numel",
                                   "parameter_numel_value", "trainable_parameters"])
def test_parameter_metadata_rejects_live_optimizer_and_size_drift(tmp_path, algorithm, field):
    record = parameter_record(algorithm)
    name = record["names"][0]
    if field in {"actual_names", "optimizer_names"}:
        record[field][-1] = record[field][0]
    elif field == "parameter_numel":
        record[field].pop(name)
    elif field == "parameter_numel_value":
        record["parameter_numel"][name] = 0
    else:
        record[field] += 1
    suite.write_json(tmp_path / "adaptation_parameters.json", record)
    with pytest.raises(RuntimeError):
        suite.validate_parameter_metadata(tmp_path, suite.study_policy(), algorithm)


def test_snapshot_dependency_closure_covers_prefix_lazy_and_validators():
    root = Path(suite.__file__).parent
    names = suite.tool_dependencies(root, (suite.CONTROLLER, "trace_cached_prefix.py"))
    assert {suite.CONTROLLER, "trace_cached_prefix.py", "trace_lazy_quality.py",
            "trace_algorithm_variants.py", "trace_ablation_variants.py", "benchmark_simplification.py",
            "run_algorithm_efficiency_suite.py", "suite_inputs.py", "run_matched_baseline_suite.py"}.issubset(names)
    assert all(Path(name).suffix == ".py" for name in names)
    manifest = dict(tool_hashes={name: name for name in names})
    for algorithm in suite.ALGORITHMS:
        assert "trace_cached_prefix.py" in experiment_hashes(manifest, algorithm)
    for algorithm in suite.LAZY_ALGORITHMS:
        assert "trace_lazy_quality.py" in experiment_hashes(manifest, algorithm)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    roots = {key: tmp_path / key for key in suite.PATHS}
    for root in roots.values():
        root.mkdir()
    package = roots["package_root"]
    (package / "trace_ovss").mkdir()
    (package / "trace_ovss/__init__.py").write_text("# frozen\n")
    for name in ("build", "assets", "__pycache__"):
        (package / name).mkdir()
        (package / name / "unused.py").write_text("# exclude\n")
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / suite.CONTROLLER).write_text("import helper\n")
    (tools / "helper.py").write_text("# transitive helper\n")
    (tools / "trace_cached_prefix.py").write_text("# cached prefix\n")
    lazy_tools = ("trace_lazy_quality.py", "trace_algorithm_variants.py", "trace_ablation_variants.py")
    for name in lazy_tools:
        (tools / name).write_text("# unchanged lazy implementation\n")
    monkeypatch.setattr(suite, "__file__", str(tools / suite.CONTROLLER))
    source = suite.source_hashes(package / "trace_ovss")
    data = panels()
    all_panels = {**copy.deepcopy(data), **{f"{d}/full": dict(image_ids=[f"/{d}.jpg"])
                                            for d in ("pc60", "coco_cd", "coco_object")}}
    parent = dict(source_hashes=source, environment={"python": "test"}, input_panels=all_panels,
                  data_root="/data", data_dirs={}, objective_epsilon=1e-8)
    parent_root, baseline_root = roots["parent_suite"], roots["baseline_suite"]
    hashes = {p: "d" * 64 for panel in data.values() for p in (*panel["image_ids"], *panel["label_ids"])}
    for name, value in (("suite_protocol.json", parent), ("fixed_configs.json", {}),
                        ("input_hashes.json", hashes), ("queue_state.json", dict(status="completed"))):
        suite.write_json(parent_root / name, value)
    seal(parent_root, ("suite_protocol.json", "fixed_configs.json", "input_hashes.json", "queue_state.json"))
    checkpoint = dict(path="/checkpoint.pt", bytes=10, mtime_ns=1, sha256="c" * 64)
    controls = {str(parent_root / name): suite.digest(parent_root / name)
                for name in ("SUITE_SEALED.json", "suite_protocol.json", "input_hashes.json", "fixed_configs.json")}
    controls[checkpoint["path"]] = checkpoint["sha256"]
    record = dict(suite_root=str(parent_root), source_fingerprint=dict(native=source, controls=controls))
    suite.write_json(baseline_root / "manifest.json", record)
    suite.write_json(baseline_root / "code_seal.json", {})
    suite.write_json(baseline_root / "queue_state.json", dict(status="completed"))
    baseline_rows = [dict(dataset=d, method=m, images=len(all_panels[d + "/full"]["image_ids"]),
                          status="completed") for d in ("pc60", "voc21", "coco_cd", "coco_object") for m in suite.BASELINES]
    suite.save_csv(baseline_root / "baselines.csv", baseline_rows)
    seal(baseline_root, ("manifest.json", "code_seal.json", "queue_state.json", "baselines.csv"))
    pending = dict(source_hashes=source, environment=parent["environment"], wait_for=str(baseline_root),
                   dependency_code_sha256=suite.digest(baseline_root / "code_seal.json"),
                   tool_hashes={name: suite.digest(tools / name) for name in lazy_tools},
                   cells=[dict(stage="evaluation", dataset="voc21", limit=128, algorithm=a) for a in suite.LAZY_ALGORITHMS],
                   input_panels={"voc21/128": dict(config=data[suite.FULL_PANEL]["config"])})
    suite.write_json(roots["wait_for"] / "suite_protocol.json", pending)
    for key in ("parent_suite", "baseline_suite", "wait_for"):
        (roots[key] / "queue.lock").touch()
    monkeypatch.setattr(suite, "execution_environment", lambda: parent["environment"])
    monkeypatch.setattr(suite, "checkpoint_identity", lambda: dict(checkpoint))
    monkeypatch.setattr(suite, "discover", lambda *args: copy.deepcopy(data))
    monkeypatch.setattr(suite, "input_hashes", lambda *args: dict(hashes))
    args = SimpleNamespace(**roots, detach=True, prepare_only=False, lock_fd=None)
    manifest = suite.prepare(args)
    return args, manifest, data


def test_prepare_freezes_pending_protocol_without_requiring_completion_and_excludes_assets(prepared):
    args, manifest, _ = prepared
    assert not (args.wait_for / "SUITE_SEALED.json").exists()
    assert manifest["dependency_protocol_sha256"] == suite.digest(args.wait_for / "suite_protocol.json")
    assert manifest["baseline_suite_seal_sha256"] == suite.digest(args.baseline_suite / "SUITE_SEALED.json")
    assert manifest["checkpoint"]["sha256"] == "c" * 64
    assert set(manifest["tool_hashes"]) == {suite.CONTROLLER, "helper.py", "trace_cached_prefix.py"}
    assert {p.name for p in (args.output / "runtime").iterdir()} == {"tools", "trace_ovss"}
    assert suite.read_json(args.output / "queue_state.json")["total"] == 13
    suite.verify_inputs(args.output, manifest)


@pytest.mark.parametrize("change", ["pending", "source", "helper", "checkpoint", "environment", "input_bytes", "input_metadata", "own_seal"])
def test_frozen_inputs_reject_post_prepare_drift(prepared, monkeypatch, change):
    args, manifest, data = prepared
    if change == "pending":
        suite.write_json(args.wait_for / "suite_protocol.json", dict(changed=True))
    elif change in {"source", "helper"}:
        name = "trace_ovss/__init__.py" if change == "source" else "tools/helper.py"
        (args.output / "runtime" / name).write_text("# changed\n")
    elif change == "checkpoint":
        monkeypatch.setattr(suite, "checkpoint_identity", lambda: dict(manifest["checkpoint"], sha256="e" * 64))
    elif change == "environment":
        monkeypatch.setattr(suite, "execution_environment", lambda: dict(changed=True))
    elif change == "input_bytes":
        monkeypatch.setattr(suite, "input_hashes", lambda *args: {})
    elif change == "input_metadata":
        data[suite.CALIBRATION_PANEL]["file_metadata"][0][0] += 1
    else:
        suite.write_json(args.output / "fixed_configs.json", dict(changed=True))
    with pytest.raises(RuntimeError):
        suite.verify_inputs(args.output, manifest)


def complete_dependency(args):
    suite.write_json(args.wait_for / "queue_state.json", dict(status="completed"))
    seal(args.wait_for, ("suite_protocol.json", "queue_state.json"))


def test_wait_locks_downstream_first_and_rechecks_new_completion_seal(prepared, monkeypatch):
    args, manifest, _ = prepared
    complete_dependency(args)
    calls = []
    monkeypatch.setattr(suite.fcntl, "flock", lambda lock, mode: calls.append((Path(lock.name), mode)))
    with ExitStack() as stack:
        locks = suite.wait_for_dependencies(args.output, manifest, stack)
        assert len(locks) == 3
        assert [p for p, _ in calls] == [root / "queue.lock" for root in (args.wait_for, args.baseline_suite, args.parent_suite)]
        assert all(mode == fcntl.LOCK_EX for _, mode in calls)
        assert all(lock.mode == "r" for lock in locks)
        suite.verify_completion_dependencies(args.output, manifest)
        seal_record = suite.read_json(args.wait_for / "SUITE_SEALED.json")
        suite.write_json(args.wait_for / "SUITE_SEALED.json", dict(seal_record, changed=True))
        with pytest.raises(RuntimeError, match="completion seal changed"):
            suite.verify_completion_dependencies(args.output, manifest)
    assert all(lock.closed for lock in locks)


def test_lock_release_without_successful_seal_does_not_launch_worker(prepared, monkeypatch):
    args, manifest, _ = prepared
    suite.write_json(args.wait_for / "queue_state.json", dict(status="failed"))
    seal(args.wait_for, ("suite_protocol.json", "queue_state.json"))
    monkeypatch.setattr(suite, "ensure_idle", lambda: pytest.fail("must not query/start GPU before dependency validation"))
    with ExitStack() as stack:
        own = suite.acquire_lock(stack, args.output / "queue.lock")
        with pytest.raises(ValueError, match="must finish"):
            suite.execute(args.output, manifest, own, stack)


@pytest.mark.parametrize("benchmark,gpu", [(True, ""), (False, "123"), (False, "")])
def test_idle_guard_detects_cpu_startup_and_gpu_workers(tmp_path, monkeypatch, benchmark, gpu):
    process = tmp_path / "123"
    process.mkdir()
    (process / "cmdline").write_bytes(b"python\0/path/benchmark_matched_baselines.py\0--output\0/cell\0" if benchmark else b"python\0other.py\0")
    monkeypatch.setattr(suite.subprocess, "check_output", lambda *a, **kw: gpu)
    if benchmark or gpu:
        with pytest.raises(RuntimeError, match="Existing workers"):
            suite.ensure_idle(tmp_path)
    else:
        suite.ensure_idle(tmp_path)


def test_detached_launch_runs_frozen_controller_with_inherited_lock(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    args = suite.parse_args(["--output", str(output), "--package-root", str(tmp_path / "package"), "--detach"])
    seen = {}

    def launch(command, **kwargs):
        seen.update(command=command, **kwargs)
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(suite.subprocess, "Popen", launch)
    with (output / "queue.lock").open("a") as lock:
        suite.launch_controller(args, lock)
        assert seen["pass_fds"] == (lock.fileno(),)
        assert seen["command"][1] == str(output / "runtime/tools" / suite.CONTROLLER)
        assert seen["command"][seen["command"].index("--lock-fd") + 1] == str(lock.fileno())
    assert seen["start_new_session"] is True
    assert seen["stdin"] == suite.subprocess.DEVNULL
    assert seen["cwd"] == output / "runtime"
    assert suite.read_json(output / "launch.json")["pid"] == 123
    assert args.wait_for == suite.DEFAULT_PATHS["wait_for"]


def test_existing_queue_cannot_be_retried_or_modified(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    suite.write_json(output / "queue_state.json", dict(status="failed", message="retained"))
    before = (output / "queue_state.json").read_bytes()
    monkeypatch.setattr(suite, "prepare", lambda *a: pytest.fail("no retry"))
    with pytest.raises(FileExistsError, match="fresh study directory"):
        suite.main(["--output", str(output), "--package-root", str(tmp_path / "package")])
    assert (output / "queue_state.json").read_bytes() == before


def references():
    return [dict(dataset="voc21", method=m, images=1449, mIoU=80., ms_per_image=200.,
                 peak_allocated_mib=1024., result_path=f"/references/{m}/result.json")
            for m in (*suite.BASELINES, "quality_only")]


def test_all_outcomes_and_optional_cost_columns_are_kept_without_calibration_comparisons(tmp_path):
    rows = [dict(stage=c["stage"], dataset="voc21", algorithm=c["algorithm"],
                 images=1449 if c["limit"] is None else c["limit"], full_split=c["limit"] is None,
                 mIoU=70., ms_per_image=100., peak_allocated_mib=512., result_path=f"/{c['key']}/result.json",
                 block_inputs_per_image=123., block_calls_per_image=45.) for c in suite.plan()]
    suite.publish(tmp_path, rows, [], references())
    assert len(suite.read_csv(tmp_path / "summary.csv")) == 13
    assert len(suite.read_csv(tmp_path / "calibration_summary.csv")) == 5
    assert len(suite.read_csv(tmp_path / "algorithm_summary.csv")) == 4
    comparisons = suite.read_csv(tmp_path / "comparison.csv")
    assert len(comparisons) == 20
    assert {r["algorithm"] for r in comparisons} == set(suite.FULL_ALGORITHMS)
    assert {r["reference"] for r in comparisons} == {*suite.BASELINES, "quality_only"}
    assert all(float(r["delta_mIoU"]) == -10 for r in comparisons)
    assert "block_inputs_per_image" in suite.read_csv(tmp_path / "algorithm_summary.csv")[0]


def test_execution_runs_every_cell_in_order_with_all_locks_and_revalidates(prepared, monkeypatch):
    args, manifest, _ = prepared
    complete_dependency(args)
    calls, lock_ids = [], []
    monkeypatch.setattr(suite, "ensure_idle", lambda: None)
    refs = references()
    for reference in refs:
        path = args.output / (reference["method"] + ".json")
        suite.write_json(path, dict(intersection=[1] * 21, union=[2] * 21))
        reference["result_path"] = str(path)
    monkeypatch.setattr(suite, "reference_rows", lambda m: refs)

    def run(output, runtime, frozen, cell, locks, *, allow_execution=True):
        calls.append((cell["key"], allow_execution))
        lock_ids.append(tuple(lock.fileno() for lock in locks))
        assert len(locks) == 4 and all(not lock.closed for lock in locks)
        directory = output / cell["key"] / "attempt_001"
        directory.mkdir(parents=True, exist_ok=True)
        suite.write_json(directory / "adaptation_parameters.json", parameter_record(cell["algorithm"]))
        row = dict(stage=cell["stage"], dataset="voc21", algorithm=cell["algorithm"],
                   images=cell["limit"] or 1449, full_split=cell["limit"] is None,
                   mIoU=1., ms_per_image=100., peak_allocated_mib=512., result_path=str(directory / "result.json"))
        suite.write_json(directory / "result.json", dict(row, intersection=[0] * 21, union=[1] * 21))
        return row, [dict(images=row["images"])]

    monkeypatch.setattr(suite, "run_cell", run)
    with ExitStack() as stack:
        own = suite.acquire_lock(stack, args.output / "queue.lock")
        suite.execute(args.output, manifest, own, stack)
    assert calls == [(c["key"], True) for c in suite.plan()] + [(c["key"], False) for c in suite.plan()]
    assert len(set(lock_ids)) == 1
    frozen = suite.read_json(args.output / "SUITE_SEALED.json")
    assert frozen["lazy_full_validation_cells"] == frozen["prefix_full_validation_cells"] == 2
    assert frozen["automatic_promotion"] is False
    assert sum(name.endswith("adaptation_parameters.json") for name in frozen["artifact_sha256"]) == 13
    assert len(suite.read_csv(args.output / "full_class_comparison.csv")) == 84
    suite.verify_suite(args.output)


def test_class_report_keeps_zero_union_blank_and_all_classes(tmp_path):
    refs = references()
    for index, reference in enumerate(refs):
        path = tmp_path / (reference["method"] + ".json")
        suite.write_json(path, dict(intersection=[0, 0, 1] + [2] * 18, union=[0, 1, 1] + [2] * 18))
        reference.update(result_path=str(path), mIoU=80 + index)
    path = tmp_path / "trace.json"
    suite.write_json(path, dict(intersection=[0, 0, 0] + [1] * 18, union=[1, 0, 1] + [2] * 18))
    rows = [dict(stage="full_validation", algorithm="quality_lazy_class_01", result_path=str(path)),
            dict(stage="calibration", algorithm="quality_tail6", result_path="not_read")]
    suite.publish_class_report(tmp_path, rows, refs)
    records = suite.read_csv(tmp_path / "full_class_comparison.csv")
    assert len(records) == 21
    assert [int(row["class_index"]) for row in records] == list(range(21))
    assert {row["baseline"] for row in records} == {"mlmp"}
    assert records[0]["baselineIoU"] == records[0]["delta"] == ""
    assert records[0]["TRACEIoU"] == "0.0"
    assert records[1]["TRACEIoU"] == records[1]["delta"] == ""
    assert records[1]["baselineIoU"] == "0.0"
    assert float(records[2]["delta"]) == -100


@pytest.mark.parametrize("alteration", ["missing", "duplicate", "failed", "wrong_count", "wrong_parent"])
def test_baseline_binding_requires_complete_full_suite(prepared, alteration):
    args, _, _ = prepared
    parent = suite.read_json(args.parent_suite / "suite_protocol.json")
    rows = suite.read_csv(args.baseline_suite / "baselines.csv")
    if alteration == "missing":
        rows.pop()
    elif alteration == "duplicate":
        rows[-1] = rows[0]
    elif alteration == "failed":
        rows[0]["status"] = "failed"
    elif alteration == "wrong_count":
        rows[0]["images"] = 128
    else:
        record = suite.read_json(args.baseline_suite / "manifest.json")
        record["suite_root"] = str(args.wait_for)
        suite.write_json(args.baseline_suite / "manifest.json", record)
    suite.save_csv(args.baseline_suite / "baselines.csv", rows)
    with pytest.raises(RuntimeError):
        suite.verify_baseline_binding(args.parent_suite, args.baseline_suite, parent)


def test_inherited_worker_failure_retains_progress_and_no_retry(prepared, monkeypatch):
    args, _, _ = prepared
    monkeypatch.setattr(suite, "__file__", str(args.output / "runtime/tools" / suite.CONTROLLER))

    def fail(output, *unused):
        suite.write_json(output / "queue_state.json", dict(status="running", completed=2, current="smoke"))
        raise RuntimeError("worker failed; evidence retained")

    monkeypatch.setattr(suite, "execute", fail)
    with ExitStack() as stack:
        lock = suite.acquire_lock(stack, args.output / "queue.lock")
        argv = [part for name in suite.PATHS for part in ("--" + name.replace("_", "-"), str(getattr(args, name)))]
        with pytest.raises(RuntimeError, match="worker failed"):
            suite.main([*argv, "--lock-fd", str(os.dup(lock.fileno()))])
        state = suite.read_json(args.output / "queue_state.json")
        assert state["status"] == "failed" and state["completed"] == 2
        assert state["automatic_retry"] is False
        before = (args.output / "queue_state.json").read_bytes()
        with pytest.raises(RuntimeError, match="no automatic retry"):
            suite.main([*argv, "--lock-fd", str(os.dup(lock.fileno()))])
        assert (args.output / "queue_state.json").read_bytes() == before
