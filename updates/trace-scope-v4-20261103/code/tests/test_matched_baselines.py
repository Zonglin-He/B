"""CPU-only sealed-input, source-freezing and baseline-contract checks."""

import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/benchmark_matched_baselines.py"
spec = importlib.util.spec_from_file_location("matched_baselines", PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("No CUDA initialization in CPU tests"))
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("No CUDA initialization in CPU tests"))


def make_panel(tmp_path, count=128, dataset="voc21", batch_size=16, *, write_files=True):
    samples = []
    for i in range(count):
        image, label = tmp_path / f"{i:05}.jpg", tmp_path / f"{i:05}.png"
        if write_files:
            image.write_bytes(b"image")
            label.write_bytes(b"label")
        samples.append((image, label))
    config = dict(dataset="PascalVOC21Dataset", resize=[224, 224], patch_size=[224, 224],
                  patch_stride=112, class_extensions=True, data_dir=str(tmp_path), batch_size=batch_size,
                  lr=.00063, steps=36, trace_max_regions=33, trace_view_batch_size=16)
    classes, mapping = ("background", "object"), (0, 1)
    frozen = dict(dataset=dataset, split="evaluation", seed=0, selection_seed=20261024,
                  image_ids=[str(x) for x, _ in samples], label_ids=[str(x) for _, x in samples],
                  available_images=count, config=config,
                  vocabulary_sha256=hashlib.sha256(json.dumps([classes, mapping]).encode()).hexdigest(),
                  file_metadata=([[p.stat().st_size, p.stat().st_mtime_ns] for pair in samples for p in pair]
                                 if write_files else [[5, 0] for _ in range(2 * count)]))
    return frozen, samples, config, classes, mapping


@pytest.fixture
def panel(tmp_path):
    return make_panel(tmp_path)


def test_sealed_pairs_are_used_exactly(panel):
    frozen, samples, config, classes, mapping = panel
    assert runner.select_inputs(frozen, samples, config, classes, mapping) == list(range(128))
    assert len(runner.input_snapshot(frozen)["sha256"]) == 256
    runner.check_batch(frozen, 0, frozen["image_ids"][:16], frozen["label_ids"][:16])


@pytest.mark.parametrize("change,match", [
    ("missing", "Missing sealed"), ("duplicate", "Duplicate"), ("label", "label ID/order"),
    ("reverse", "image order"), ("count", "image/label IDs"), ("population", "population"),
    ("vocab", "Vocabulary"), ("mapping", "Vocabulary"), ("batch", "configuration"),
    ("lr", "configuration"), ("resize", "configuration"), ("extensions", "configuration")])
def test_reject_input_vocab_order_and_configuration_drift(panel, change, match):
    frozen, samples, config, classes, mapping = panel
    config = copy.deepcopy(config)
    if change == "missing":
        frozen["image_ids"][0] += ".missing"
    elif change == "duplicate":
        frozen["image_ids"][0] = frozen["image_ids"][1]
    elif change == "label":
        frozen["label_ids"][0:2] = reversed(frozen["label_ids"][0:2])
    elif change == "reverse":
        frozen["image_ids"].reverse()
        frozen["label_ids"].reverse()
    elif change == "count":
        frozen["image_ids"].pop()
    elif change == "population":
        frozen["available_images"] += 1
    elif change == "vocab":
        classes = ("object", "background")
    elif change == "mapping":
        mapping = (1, 0)
    else:
        key = {"batch": "batch_size", "extensions": "class_extensions"}.get(change, change)
        config[key] = False if change == "extensions" else [112, 112] if change == "resize" else 1
    with pytest.raises(RuntimeError, match=match):
        runner.select_inputs(frozen, samples, config, classes, mapping)


def test_batch_order_labels_and_boundaries_are_enforced(panel):
    frozen = panel[0]
    for images, labels in [(frozen["image_ids"][:8], frozen["label_ids"][:8]),
                           (frozen["image_ids"][16:32], frozen["label_ids"][:16]),
                           (frozen["image_ids"][:16], list(reversed(frozen["label_ids"][:16])))]:
        with pytest.raises(RuntimeError, match="episode boundary"):
            runner.check_batch(frozen, 0, images, labels)


def test_metadata_and_same_metadata_content_changes(panel):
    frozen, samples, *_ = panel
    before = runner.input_snapshot(frozen)
    image = samples[0][0]
    stamp = image.stat()
    image.write_bytes(b"other")
    os.utime(image, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert runner.input_snapshot(frozen) != before
    image.write_bytes(b"longer image")
    with pytest.raises(RuntimeError, match="size/mtime"):
        runner.input_snapshot(frozen)


def test_source_freeze_tracks_only_runner_but_all_native_and_baseline_resources(tmp_path):
    package, baseline = tmp_path / "native", tmp_path / "baseline"
    (package / "trace_ovss").mkdir(parents=True)
    baseline.mkdir()
    experiment, controls = tmp_path / "runner.py", tmp_path / "suite.json"
    files = [package / "trace_ovss/a.py", baseline / "method.py", baseline / "prompts.yaml", experiment, controls]
    for path in files:
        path.write_text("original")
    frozen = runner.source_snapshot(package, baseline, [controls], experiment)
    (tmp_path / "unrelated_helper.py").write_text("not an imported dependency")
    (baseline / "__pycache__").mkdir()
    (baseline / "__pycache__/method.pyc").write_bytes(b"cache")
    runner.check_snapshot(frozen, runner.source_snapshot(package, baseline, [controls], experiment))
    for path in files:
        path.write_text("changed")
        with pytest.raises(RuntimeError, match="hashes changed"):
            runner.check_snapshot(frozen, runner.source_snapshot(package, baseline, [controls], experiment))
        path.write_text("original")
    extra = baseline / "new_dependency.py"
    extra.write_text("extra")
    with pytest.raises(RuntimeError, match="hashes changed"):
        runner.check_snapshot(frozen, runner.source_snapshot(package, baseline, [controls], experiment))


def archive_fixture(tmp_path, config, dataset="voc21"):
    root = tmp_path / "archive/baseline_source"
    root.mkdir(parents=True)
    (root / "method.py").write_text("snapshot")
    source = root / "recipe.sh"
    source.write_text('OUT_VISION="' + " ".join(map(str, runner.VISION_OUTPUTS)) + '"\n')
    record = dict(dataset=dataset, method="mlmp", config={**config, "lr": .001, "steps": 10, "batch_size": 2,
                  "alpha_cls": 1.0}, source=str(source), source_sha256=runner.digest(source))
    config_path = root.parent / "released_configs" / f"{dataset}_mlmp.json"
    config_path.parent.mkdir()
    runner.write_json(config_path, record)
    manifest = root.parent / "manifest_v2.json"
    runner.write_json(manifest, dict(jobs=[dict(dataset=dataset, method="mlmp", settings=dict(config_file=str(config_path)))]))
    runner.write_json(root.parent / "config_seals.json", dict(configs={str(config_path): runner.digest(config_path)},
                                                            manifest_sha256=runner.digest(manifest)))
    runner.write_json(root.parent / "source_provenance.json", dict(legacy_commit=runner.LEGACY_COMMIT,
                                                                 baseline_sources={"method.py": runner.digest(root / "method.py")}))
    return root, config_path, manifest


@pytest.mark.parametrize("method", ["mlmp", "legacy_trace_noadapt"])
def test_archived_manifest_binds_frozen_optimization_not_panel_batch(panel, tmp_path, method):
    config = panel[2]
    root, _, _ = archive_fixture(tmp_path, config)
    record, controls = runner.archived_settings(root, "voc21", method, config)
    assert record["config"]["batch_size"] == 2 and config["batch_size"] == 16
    assert len(controls) == 5


@pytest.mark.parametrize("change,match", [("config", "config seal"), ("manifest", "sealed baseline manifest"),
                                         ("source", "baseline source"), ("recipe", "recipe source")])
def test_archived_settings_reject_changed_evidence(panel, tmp_path, change, match):
    root, config, manifest = archive_fixture(tmp_path, panel[2])
    path = {"config": config, "manifest": manifest, "source": root / "method.py", "recipe": root / "recipe.sh"}[change]
    path.write_text(path.read_text() + " ")
    with pytest.raises(RuntimeError, match=match):
        runner.archived_settings(root, "voc21", "mlmp", panel[2])


def test_noadapt_controls_keep_distinct_models_and_prompts(panel, tmp_path):
    regular = runner.method_settings("noadapt", panel[2], tmp_path)
    native = runner.method_settings("native_noadapt", panel[2], tmp_path)
    legacy = runner.method_settings("legacy_trace_noadapt", panel[2], tmp_path)
    assert all(s["update_steps"] == 0 and not s["backend_equivalence_claim"] for s in [regular, native, legacy])
    assert "vision_outputs" not in regular["constructor"]
    assert legacy["constructor"]["vision_outputs"] == native["constructor"]["vision_outputs"]
    assert legacy["constructor"]["alpha_cls"] == 1.0
    assert legacy["constructor"]["prompt_integration"] == "loss"
    assert native["constructor"]["lr"] == panel[2]["lr"]
    assert runner.method_settings("tpt", panel[2], tmp_path)["constructor"]["n_ctx"] == 4


def test_loader_bypasses_adapt_init_and_common(tmp_path, monkeypatch):
    (tmp_path / "adapt").mkdir()
    (tmp_path / "adapt/__init__.py").write_text('raise AssertionError("do not import adapt init")')
    (tmp_path / "common.py").write_text('raise AssertionError("do not import common")')
    (tmp_path / "adapt/mlmp.py").write_text("class MLMP: pass\n")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    cls = runner.load_baseline(tmp_path, "legacy_trace_noadapt")
    assert cls.__name__ == "MLMP"


def test_wrong_checkout_module_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ovss", SimpleNamespace(__file__="/wrong/ovss/__init__.py"))
    with pytest.raises(RuntimeError, match="wrong checkout"):
        runner.load_baseline(tmp_path, "tent")


def test_warmup_two_and_actual_forward_counts():
    rows = [dict(images=n, total_seconds=2, adaptation_seconds=1, inference_seconds=1,
                 inputs=3*n, calls=5, peak_allocated_mib=20) for n in [28, 28, 28, 28, 16]]
    result = runner.summarize_cost(rows)
    assert result["warmup_episodes"] == 2 and result["measured_images"] == 72
    assert result["visual_inputs_per_image"] == 3
    assert result["visual_calls_per_image"] == 15/72
    assert result["ms_per_image"] == 6000/72


def test_existing_output_never_overwritten_and_help_has_no_gpu(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    result_path = output / "result.json"
    result_path.write_text("keep")
    arguments = ["--method", "tent", "--dataset", "voc21", "--output", str(output)]
    for flag in ["--suite-root", "--baseline-root", "--package-root"]:
        arguments.extend([flag, str(tmp_path)])
    with pytest.raises(FileExistsError):
        runner.main(arguments)
    assert result_path.read_text() == "keep"
    help_result = subprocess.run([sys.executable, "-B", str(PATH), "--help"], cwd=tmp_path,
                                 capture_output=True, text=True, timeout=30)
    assert help_result.returncode == 0
    assert "legacy_trace_noadapt" in help_result.stdout
    assert "--panel-key" in help_result.stdout
    assert "dataset/full" in help_result.stdout
    assert "--lr" not in help_result.stdout


def test_preflight_failure_is_recorded_without_result(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "preflight", lambda args: runner.require(False, "bad seal"))
    arguments = ["--method", "tent", "--dataset", "voc21", "--output", str(tmp_path / "failed")]
    for flag in ["--suite-root", "--baseline-root", "--package-root"]:
        arguments.extend([flag, str(tmp_path)])
    with pytest.raises(RuntimeError, match="Cell failed"):
        runner.main(arguments)
    output = tmp_path / "failed"
    assert runner.read_json(output / "failure.json")["message"] == "bad seal"
    assert (output / "stdout.log").exists() and not (output / "result.json").exists()


@pytest.mark.parametrize("drift", [None, "source", "input"])
def test_final_checks_gate_success_and_artifact_hash_exports(tmp_path, monkeypatch, drift):
    prepared = dict(sources={"source": "before"}, controls=[], panel={}, inputs={"input": "before"})
    monkeypatch.setattr(runner, "preflight", lambda args: prepared)
    monkeypatch.setattr(runner, "source_snapshot", lambda *a: {"source": "after" if drift == "source" else "before"})
    monkeypatch.setattr(runner, "input_snapshot", lambda *a: {"input": "after" if drift == "input" else "before"})

    def fake_execute(args, p, predictions):
        runner.write_json(args.output / "protocol.json", {"cpu_test": True})
        (args.output / "cost.csv").write_text("episode,inputs,calls\n0,128,1\n")
        predictions.extend([dict(image_id=str(i)) for i in range(128)])
        return dict(status="completed", images=128, miou_unrounded=25.123456789,
                    ms_per_image=1, visual_inputs_per_image=1, visual_calls_per_image=1/128)

    monkeypatch.setattr(runner, "execute", fake_execute)
    output = tmp_path / "cell"
    arguments = ["--method", "tent", "--dataset", "voc21", "--output", str(output)]
    for flag in ["--suite-root", "--baseline-root", "--package-root"]:
        arguments.extend([flag, str(tmp_path)])
    if drift:
        with pytest.raises(RuntimeError, match="Cell failed"):
            runner.main(arguments)
        assert not (output / "result.json").exists()
        assert "integrity_error" in runner.read_json(output / "failure.json")
    else:
        runner.main(arguments)
        result = runner.read_json(output / "result.json")
        for name, suffix in [("protocol", ".json"), ("per_image_counts", ".json"), ("cost", ".csv")]:
            assert result[name + "_sha256"] == runner.digest(output / (name + suffix))
        assert result["adapter_sha256"] == runner.digest(PATH)
        assert result["images"] == 128 and result["miou_unrounded"] == 25.123456789


@pytest.mark.parametrize("dataset", runner.POPULATIONS)
def test_full_panels_cover_native_population_and_exact_tail(tmp_path, dataset):
    count = runner.POPULATIONS[dataset]
    frozen, samples, config, classes, mapping = make_panel(tmp_path, count, dataset, 9, write_files=False)
    suite = dict(selection_seed=20261024, input_panels={dataset + "/full": frozen})
    key, selected = runner.select_panel(suite, dataset, dataset + "/full")
    assert key == dataset + "/full" and selected is frozen
    assert runner.select_inputs(selected, samples, config, classes, mapping) == list(range(count))
    sizes = []
    for offset in range(0, count, config["batch_size"]):
        images, labels = frozen["image_ids"][offset:offset+9], frozen["label_ids"][offset:offset+9]
        runner.check_batch(frozen, offset, images, labels)
        sizes.append(len(images))
    assert sum(sizes) == count and sizes[-1] == ((count - 1) % 9) + 1
    if dataset == "pc60":
        assert len(sizes) == 568 and sizes[-1] == 2


def test_default_pilot_does_not_fall_back_to_full(panel):
    frozen = panel[0]
    frozen["available_images"] = runner.POPULATIONS["voc21"]
    suite = dict(selection_seed=20261024, input_panels={"voc21/128": frozen, "voc21/full": {}})
    for key in (None, "voc21/128"):
        assert runner.select_panel(suite, "voc21", key) == ("voc21/128", frozen)
    del suite["input_panels"]["voc21/128"]
    with pytest.raises(RuntimeError, match="Missing explicitly sealed panel"):
        runner.select_panel(suite, "voc21")


@pytest.mark.parametrize("key", ["pc60/full", "voc21/4", "full", "", "voc21/129"])
def test_panel_keys_are_explicit_and_dataset_bound(key):
    with pytest.raises(RuntimeError, match="Panel key must match"):
        runner.select_panel({"input_panels": {}}, "voc21", key)


@pytest.mark.parametrize("mode,count", [("full", 128), ("full", 1448), ("128", 129), ("128", 1449)])
def test_panel_names_do_not_allow_count_substitution(tmp_path, mode, count):
    frozen = make_panel(tmp_path, count, write_files=False)[0]
    frozen["available_images"] = runner.POPULATIONS["voc21"]
    key = "voc21/" + mode
    with pytest.raises(RuntimeError, match="Expected exactly"):
        runner.select_panel(dict(selection_seed=20261024, input_panels={key: frozen}), "voc21", key)


@pytest.mark.parametrize("change,match", [
    ("empty", "image/label"), ("short_labels", "image/label"), ("metadata", "metadata"),
    ("duplicate_image", "Duplicate"), ("duplicate_label", "Duplicate"),
    ("population", "population"), ("split", "identity"), ("dataset", "identity"),
    ("seed", "identity"), ("selection_seed", "identity"),
    ("zero_batch", "batch size"), ("bool_batch", "batch size"), ("float_batch", "batch size"),
    ("warmup_only", "two episodes")])
def test_full_panel_rejects_malformed_identity_and_episodes(tmp_path, change, match):
    frozen = make_panel(tmp_path, 500, "coco_cd", write_files=False)[0]
    if change == "empty":
        frozen["image_ids"] = frozen["label_ids"] = []
    elif change == "short_labels":
        frozen["label_ids"].pop()
    elif change == "metadata":
        frozen["file_metadata"].pop()
    elif change.startswith("duplicate"):
        key = "image_ids" if change == "duplicate_image" else "label_ids"
        frozen[key][1] = frozen[key][0]
    elif change.endswith("batch") or change == "warmup_only":
        frozen["config"]["batch_size"] = dict(zero_batch=0, bool_batch=True, float_batch=16.0, warmup_only=250)[change]
    else:
        key = "available_images" if change == "population" else change
        frozen[key] = "other" if change in {"split", "dataset"} else frozen[key] + 1
    with pytest.raises(RuntimeError, match=match):
        runner.select_panel(dict(selection_seed=20261024, input_panels={"coco_cd/full": frozen}),
                            "coco_cd", "coco_cd/full")


@pytest.mark.parametrize("offset", [-16, 1, 17, 128, 144, True, 0.0])
def test_batches_cannot_start_mid_episode_or_outside_panel(panel, offset):
    with pytest.raises(RuntimeError, match="episode boundary"):
        runner.check_batch(panel[0], offset, [], [])


@pytest.mark.parametrize("mode", ["128", "full"])
def test_full_order_and_vocabulary_guards_match_pilot(tmp_path, mode):
    count = 128 if mode == "128" else 500
    frozen, samples, config, classes, mapping = make_panel(tmp_path, count, "coco_cd", write_files=False)
    for change, match in [("reverse", "image order"), ("label", "label ID/order"),
                          ("missing", "Missing sealed"), ("vocab", "Vocabulary"),
                          ("config", "configuration"), ("native_duplicate", "Duplicate native")]:
        altered, actual, settings, vocabulary = copy.deepcopy(frozen), list(samples), dict(config), classes
        if change == "reverse":
            altered["image_ids"].reverse()
            altered["label_ids"].reverse()
        elif change == "label":
            altered["label_ids"][-2:] = reversed(altered["label_ids"][-2:])
        elif change == "missing":
            altered["image_ids"][-1] += ".missing"
        elif change == "vocab":
            vocabulary = tuple(reversed(classes))
        elif change == "config":
            settings["batch_size"] += 1
        else:
            actual[-1] = actual[0]
        with pytest.raises(RuntimeError, match=match):
            runner.select_inputs(altered, actual, settings, vocabulary, mapping)


@pytest.fixture
def full_suite(tmp_path, monkeypatch):
    frozen, samples, config, classes, mapping = make_panel(tmp_path, 500, "coco_cd", 28)
    native_config = copy.deepcopy(config)
    suite_root = tmp_path / "suite"
    runtime = suite_root / "runtime"
    (runtime / "trace_ovss").mkdir(parents=True)
    (runtime / "trace_ovss/__init__.py").write_text("# native snapshot\n")
    (runtime / "tools").mkdir()
    (runtime / "tools/worker.py").write_text("# frozen tool\n")
    baseline, _, _ = archive_fixture(tmp_path, config, "coco_cd")
    (baseline / "prompts.yaml").write_text("prompt_templates: ['a photo of a {}']\n")
    checkpoint = tmp_path / ".cache/clip/ViT-L-14.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"cpu-test checkpoint")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(runner, "CHECKPOINT_SHA256", runner.digest(checkpoint))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    for name in list(sys.modules):
        if name.split(".")[0] == "trace_ovss":
            monkeypatch.delitem(sys.modules, name)
    modules = {
        "trace_ovss": {}, "trace_ovss.data": {},
        "trace_ovss.configs": dict(best_config=lambda *a: copy.deepcopy(native_config), VISION_OUTPUTS=runner.VISION_OUTPUTS),
        "trace_ovss.data.datasets": dict(prepare_data=lambda *a, **kw: (
            SimpleNamespace(dataset=SimpleNamespace(samples=samples)),
            SimpleNamespace(classes=classes, extensions=classes, extension_to_class=mapping))),
        "trace_ovss.data.vocabulary": dict(load_prompt_templates=lambda: ["a photo of a {}"]),
    }
    for name, values in modules.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__file__=str(runtime / name.replace(".", "/")), **values))
    suite = dict(config_fixed=True, seed=0, selection_seed=20261024, data_root=str(tmp_path),
                 source_hashes=runner.tree_hashes(runtime / "trace_ovss"),
                 tool_hashes=runner.tree_hashes(runtime / "tools"), input_panels={"coco_cd/full": frozen},
                 environment=dict(python="test-python", torch="test-torch", cuda="test-cuda", gpu="test-gpu",
                                  gpu_inventory=["test-gpu, test-uuid, test-driver"], cuda_visible_devices=None,
                                  float32_matmul_precision="highest", matmul_tf32=False, cudnn_tf32=False))
    environment = copy.deepcopy(suite["environment"])
    monkeypatch.setattr(runner, "execution_environment", lambda root: copy.deepcopy(environment))
    runner.write_json(suite_root / "suite_protocol.json", suite)
    runner.write_json(suite_root / "fixed_configs.json", {"coco_cd": config})
    runner.write_json(suite_root / "input_hashes.json", {
        path: runner.digest(path) for pair in zip(frozen["image_ids"], frozen["label_ids"]) for path in pair})
    seal = dict(artifact_sha256={name: runner.digest(suite_root / name)
                                for name in ("suite_protocol.json", "fixed_configs.json", "input_hashes.json")})
    runner.write_json(suite_root / "SUITE_SEALED.json", seal)
    args = SimpleNamespace(suite_root=suite_root, package_root=runtime, baseline_root=baseline,
                           dataset="coco_cd", method="mlmp", panel_key="coco_cd/full", workers=0)
    return args, suite, seal


def test_full_preflight_and_frozen_control_coverage(full_suite):
    args, suite, _ = full_suite
    prepared = runner.preflight(args)
    assert prepared["panel_key"] == "coco_cd/full" and prepared["full_split"] is True
    assert prepared["indices"] == list(range(500))
    assert prepared["panel"] == suite["input_panels"]["coco_cd/full"]
    assert len(prepared["inputs"]["sha256"]) == 1000
    assert str(args.suite_root / "fixed_configs.json") in prepared["sources"]["controls"]
    assert str(args.suite_root / "input_hashes.json") in prepared["sources"]["controls"]
    assert str(args.package_root / "tools/worker.py") in prepared["sources"]["controls"]


@pytest.mark.parametrize("change,match", [
    ("seal", "protocol seal"), ("fixed", "configurations seal"), ("missing_fixed_seal", "configurations seal"),
    ("fixed_binding", "Fixed configuration differs"), ("native", "Native source differs"),
    ("tool", "runtime tools"), ("new_tool", "runtime tools"), ("missing_tool", "runtime tools"),
    ("missing_tool_hashes", "runtime tools"), ("wrong_runtime", "sealed suite runtime"),
    ("checkpoint", "Checkpoint SHA256"), ("image", "size/mtime"),
    ("config", "Native configuration"), ("vocab", "Vocabulary"),
    ("prompts", "multi-prompt"), ("baseline", "baseline source"),
    ("provenance", "baseline commit"), ("suite_seed", "configuration/seed")])
def test_full_preflight_rejects_drift_before_execute(full_suite, change, match):
    args, suite, seal = full_suite
    panel = suite["input_panels"]["coco_cd/full"]
    if change in {"seal", "suite_seed", "config", "vocab", "missing_tool_hashes"}:
        if change == "suite_seed":
            suite["seed"] = 1
        elif change == "config":
            panel["config"]["trace_max_regions"] += 1
        elif change == "vocab":
            panel["vocabulary_sha256"] = "changed"
        elif change == "missing_tool_hashes":
            del suite["tool_hashes"]
        (args.suite_root / "suite_protocol.json").write_text(json.dumps(suite) + " ")
        if change != "seal":
            seal["artifact_sha256"]["suite_protocol.json"] = runner.digest(args.suite_root / "suite_protocol.json")
        if change == "config":
            (args.suite_root / "fixed_configs.json").write_text(json.dumps({args.dataset: panel["config"]}))
            seal["artifact_sha256"]["fixed_configs.json"] = runner.digest(args.suite_root / "fixed_configs.json")
    elif change == "missing_fixed_seal":
        del seal["artifact_sha256"]["fixed_configs.json"]
    elif change == "fixed_binding":
        (args.suite_root / "fixed_configs.json").write_text(json.dumps({args.dataset: {}}))
        seal["artifact_sha256"]["fixed_configs.json"] = runner.digest(args.suite_root / "fixed_configs.json")
    elif change == "wrong_runtime":
        args.package_root = args.suite_root
    elif change == "missing_tool":
        (args.package_root / "tools/worker.py").unlink()
    elif change == "provenance":
        path = args.baseline_root.parent / "source_provenance.json"
        value = runner.read_json(path)
        value["legacy_commit"] = "different"
        path.write_text(json.dumps(value))
    else:
        paths = dict(fixed=args.suite_root / "fixed_configs.json", native=args.package_root / "trace_ovss/__init__.py",
                     tool=args.package_root / "tools/worker.py", new_tool=args.package_root / "tools/new.py",
                     checkpoint=Path.home() / ".cache/clip/ViT-L-14.pt", image=Path(panel["image_ids"][-1]),
                     prompts=args.baseline_root / "prompts.yaml", baseline=args.baseline_root / "method.py")
        paths[change].write_text("prompt_templates: ['different']" if change == "prompts" else "changed")
    (args.suite_root / "SUITE_SEALED.json").write_text(json.dumps(seal))
    with pytest.raises(RuntimeError, match=match):
        runner.preflight(args)


def test_pilot_preflight_preserves_original_control_fingerprint(full_suite):
    args, suite, seal = full_suite
    args.panel_key = None
    full = suite["input_panels"].pop("coco_cd/full")
    pilot = copy.deepcopy(full)
    for name, count in (("image_ids", 128), ("label_ids", 128), ("file_metadata", 256)):
        pilot[name] = pilot[name][:count]
    suite["input_panels"]["coco_cd/128"] = pilot
    (args.suite_root / "suite_protocol.json").write_text(json.dumps(suite))
    seal["artifact_sha256"]["suite_protocol.json"] = runner.digest(args.suite_root / "suite_protocol.json")
    (args.suite_root / "SUITE_SEALED.json").write_text(json.dumps(seal))
    prepared = runner.preflight(args)
    assert prepared["indices"] == list(range(128)) and not prepared["full_split"]
    _, controls = runner.archived_settings(args.baseline_root, args.dataset, args.method, pilot["config"])
    controls += [args.suite_root / "suite_protocol.json", args.suite_root / "SUITE_SEALED.json", prepared["checkpoint"]]
    assert prepared["sources"] == runner.source_snapshot(args.package_root, args.baseline_root, controls)


@pytest.mark.parametrize("drift", [None, "fixed", "tool", "new_tool", "input", "input_hashes", "environment"])
def test_full_final_checks_and_cli_propagation(full_suite, tmp_path, monkeypatch, drift):
    args, suite, _ = full_suite
    output = tmp_path / "cell"

    def execute(a, prepared, predictions):
        assert a.panel_key == prepared["panel_key"] == "coco_cd/full"
        assert prepared["indices"] == list(range(500))
        runner.write_json(a.output / "protocol.json", dict(images=500, full_split=True))
        (a.output / "cost.csv").write_text("episode,images\n0,28\n")
        predictions.extend(dict(image_id=i, label_id=l) for i, l in
                           zip(prepared["panel"]["image_ids"], prepared["panel"]["label_ids"]))
        if drift == "environment":
            monkeypatch.setattr(runner, "execution_environment", lambda root: {"gpu": "different"})
        elif drift:
            path = dict(fixed=a.suite_root / "fixed_configs.json", tool=a.package_root / "tools/worker.py",
                        new_tool=a.package_root / "tools/new.py", input=Path(prepared["panel"]["image_ids"][-1]),
                        input_hashes=a.suite_root / "input_hashes.json")[drift]
            path.write_text("changed after preflight")
        return dict(status="completed", images=500, full_split=True, panel_key=prepared["panel_key"])

    monkeypatch.setattr(runner, "execute", execute)
    arguments = ["--dataset", args.dataset, "--method", args.method, "--panel-key", args.panel_key,
                 "--output", str(output)]
    for name in ("suite_root", "baseline_root", "package_root"):
        arguments += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    if drift:
        with pytest.raises(RuntimeError, match="Cell failed"):
            runner.main(arguments)
        assert not (output / "result.json").exists()
        assert "integrity_error" in runner.read_json(output / "failure.json")
    else:
        runner.main(arguments)
        result = runner.read_json(output / "result.json")
        assert result["images"] == 500 and result["full_split"] and result["panel_key"] == "coco_cd/full"


def test_invalid_cli_panel_fails_before_output_or_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "preflight", lambda *a: pytest.fail("Invalid panel reached preflight"))
    output = tmp_path / "cell"
    args = ["--dataset", "pc60", "--method", "tent", "--panel-key", "voc21/full", "--output", str(output)]
    for name in ("suite-root", "baseline-root", "package-root"):
        args += ["--" + name, str(tmp_path)]
    with pytest.raises(SystemExit):
        runner.main(args)
    assert not output.exists()


@pytest.mark.parametrize("mode,count", [("128", 128), ("full", 5105)])
@pytest.mark.parametrize("environment_drift", [False, True])
def test_cpu_execute_exports_panel_counts_tail_and_warmup(tmp_path, monkeypatch, capsys, mode, count, environment_drift):
    import torch
    from trace_ovss import cli

    frozen, samples, config, classes, mapping = make_panel(tmp_path, count, "pc60", 9, write_files=False)
    frozen["available_images"] = runner.POPULATIONS["pc60"]
    config.update(patch_size=[1, 1], patch_stride=1)
    data = [dict(img_patches=torch.zeros(1, 3, 1, 1), gt=torch.zeros(1, 1, 1, dtype=torch.long),
                 meta=dict(img_path=str(image), seg_map_path=str(label), patch_grid_shape=(1, 1), img_shape=(1, 1)))
            for image, label in samples]

    class CPUModel:
        def __init__(self, classes, device, lr, steps, **kwargs):
            assert device == "cuda"
            self.model = torch.nn.Module()
            self.model.weight = torch.nn.Parameter(torch.zeros(1))
            self.model.visual = torch.nn.Module()
            self.model.visual.conv1 = torch.nn.Identity()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.steps, self.prompt = steps, "a photo of a {}"

        def evaluate(self, images):
            self.model.visual.conv1(images)
            return torch.zeros(len(images), 2, 1, 1)

    monkeypatch.setattr(runner, "load_baseline", lambda *a: CPUModel)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kw: self)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU test double")
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(cli, "set_global_seed", lambda seed: None)
    args = SimpleNamespace(method="noadapt", dataset="pc60", package_root=PATH.parents[1],
                           baseline_root=tmp_path, suite_root=tmp_path, output=tmp_path, workers=0)
    runner.write_json(tmp_path / "suite_protocol.json", {"cpu_test": True})
    environment = dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
                       matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                       cudnn_tf32=torch.backends.cudnn.allow_tf32)
    monkeypatch.setattr(runner, "execution_environment", lambda root: (
        {"gpu": "different"} if environment_drift else copy.deepcopy(environment)))
    prepared = dict(config=config, panel=frozen, panel_key="pc60/" + mode, full_split=mode == "full",
                    suite=dict(environment=environment), prompts=["a photo of a {}"], archived={}, sources={}, inputs={},
                    classes=classes, mapping=mapping, vocab=SimpleNamespace(classes=classes), dataset=data,
                    indices=list(range(count)))
    predictions = []
    if environment_drift and mode == "full":
        with pytest.raises(RuntimeError, match="Execution environment"):
            runner.execute(args, prepared, predictions)
        assert not predictions and not (tmp_path / "protocol.json").exists()
        return
    result = runner.execute(args, prepared, predictions)
    protocol = runner.read_json(tmp_path / "protocol.json")
    for value in (protocol, result):
        assert value["images"] == count and value["full_split"] == (mode == "full")
        assert value["panel_key"] == "pc60/" + mode and value["warmup_episodes"] == 2
    with (tmp_path / "cost.csv").open() as handle:
        costs = list(csv.DictReader(handle))
    assert len(costs) == (count + 8) // 9 and int(costs[-1]["images"]) == 2
    assert [r["image_id"] for r in predictions] == frozen["image_ids"]
    assert [r["label_id"] for r in predictions] == frozen["label_ids"]
    assert result["measured_images"] == count - 18
    assert result["visual_inputs_per_image"] == 1 and result["all_visual_inputs"] == count
    assert result["visual_calls_per_image"] == (len(costs) - 2) / (count - 18)
    assert result["ms_per_image"] == pytest.approx(1000 * sum(float(r["total_seconds"]) for r in costs[2:]) / (count - 18))
    assert result["all_visual_calls"] == len(costs)
    assert f"{count}/{count}" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["image_ids", "label_ids"])
def test_full_preflight_rejects_changed_bytes_with_preserved_metadata(full_suite, kind):
    args, suite, _ = full_suite
    path = Path(suite["input_panels"]["coco_cd/full"][kind][-1])
    before = path.stat()
    path.write_bytes(b"x" * before.st_size)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(RuntimeError, match="Input content differs from parent"):
        runner.preflight(args)


@pytest.mark.parametrize("change", ["missing_seal", "missing_file", "changed", "missing_entry", "wrong_hash", "not_map"])
def test_full_preflight_requires_sealed_complete_parent_hashes(full_suite, change):
    args, suite, seal = full_suite
    path = args.suite_root / "input_hashes.json"
    if change == "missing_seal":
        del seal["artifact_sha256"]["input_hashes.json"]
    elif change == "missing_file":
        path.unlink()
    else:
        hashes = runner.read_json(path)
        key = suite["input_panels"]["coco_cd/full"]["label_ids"][-1]
        if change == "missing_entry":
            del hashes[key]
        elif change == "not_map":
            hashes = []
        else:
            hashes[key] = "0" * 64
        path.write_text(json.dumps(hashes))
        if change != "changed":
            seal["artifact_sha256"]["input_hashes.json"] = runner.digest(path)
    (args.suite_root / "SUITE_SEALED.json").write_text(json.dumps(seal))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        runner.preflight(args)


@pytest.mark.parametrize("field", ["python", "torch", "cuda", "gpu", "gpu_inventory", "cuda_visible_devices",
                                   "float32_matmul_precision", "matmul_tf32", "cudnn_tf32", "missing"])
def test_full_preflight_rejects_environment_drift_before_execute(full_suite, monkeypatch, field):
    args, suite, _ = full_suite
    actual = copy.deepcopy(suite["environment"])
    if field == "missing":
        del actual["gpu_inventory"]
    else:
        actual[field] = not actual[field] if isinstance(actual[field], bool) else "different"
    monkeypatch.setattr(runner, "execution_environment", lambda root: actual)
    with pytest.raises(RuntimeError, match="Execution environment"):
        runner.preflight(args)


def test_environment_reader_uses_requested_frozen_native_helper(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "benchmark_simplification.py").write_text(
        "def execution_environment():\n    return {'python': 'frozen-reader'}\n")
    before = sys.path[:]
    assert runner.execution_environment(tmp_path) == {"python": "frozen-reader"}
    assert sys.path == before
