"""CPU-only contract, real tiny-model accounting, barriers and scoring tests."""

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import shutil

import numpy as np
from PIL import Image
import pytest
import torch

from tools.evolution_tuning import worker
from test_simplified_trace import tiny_trace


def config(dataset="pc60", batch=2, tiny=False):
    size = 32 if tiny else 224
    return dict(dataset=worker.DATASET_CLASSES[dataset], data_dir="/synthetic", data_subdir="synthetic",
        resize=[size, size], patch_size=[size, size], patch_stride=16 if tiny else 112,
        batch_size=batch, lr=.001, steps=2, trace_max_regions=4,
        trace_view_batch_size=3 if tiny else 16, class_extensions=dataset not in ("coco_cd", "waterbirds_seg"))


def meta(n):
    return dict(img_path=[str(i) for i in range(n)], seg_map_path=[f"never-open-{i}" for i in range(n)],
        ori_shape=[(32, 32)] * n, img_shape=[(32, 32)] * n, patch_shape=[(32, 32)] * n, patch_grid_shape=[(1, 1)] * n)


@pytest.fixture
def method():
    torch.set_num_threads(1)
    base = tiny_trace()
    return worker.ExperimentTRACE(classes=base.classes, lr=base.lr, steps=base.steps,
        trace_max_regions=base.trace_max_regions, trace_view_batch_size=base.trace_view_batch_size,
        vision_outputs=base.vision_outputs, prompts=base.prompt_templates, model=base.model,
        tokenize_fn=base.tokenize, device="cpu", view_mode="C")


def reference(method):
    method.reset()
    return worker.GateReference(classes=method.classes, lr=method.lr, steps=method.steps,
        trace_max_regions=method.trace_max_regions, trace_view_batch_size=method.trace_view_batch_size,
        vision_outputs=method.vision_outputs, prompts=method.prompt_templates, model=method.model,
        tokenize_fn=method.tokenize, device="cpu", view_mode="C")


def manifest(tmp_path, dataset="pc60", stage="full", n=2):
    rows = []
    for i in range(n):
        image, label = tmp_path / f"image{i}.png", tmp_path / f"label{i}.png"
        if not image.exists():
            Image.fromarray(np.full((2, 2, 3), i % 255, np.uint8)).save(image)
            Image.fromarray(np.zeros((2, 2), np.uint8)).save(label)
        stat = label.stat()
        rows.append(dict(image=str(image), label=str(label), image_sha256=worker.sha256(image),
            label_metadata=[stat.st_size, stat.st_mtime_ns], official_id=f"official-{i}"))
    return dict(schema_version=1, algorithm="ImageBalancedTRACE:C", suite_root=str(tmp_path),
        runtime_root=str(tmp_path / "runtime"), dataset=dataset, stage=stage, seed=0,
        config=config(dataset), checkpoint=dict(path=str(tmp_path / "checkpoint.pt"), sha256="a" * 64),
        panel=dict(inputs=rows, ordered_ids=[r["official_id"] for r in rows], classes=["a", "b"],
            queries=["a", "b"], query_to_class=[0, 1] if config(dataset)["class_extensions"] else None,
            reduce_zero_label=False, full_split=stage == "full"),
        arms=worker.ARMS.copy() if stage == "full" else ["image_C"], source_sha256={"worker.py": "b" * 64})


def cost_row(man, arm, episode=0, position=0):
    cfg = man["config"]
    images = min(cfg["batch_size"], len(man["panel"]["inputs"]) - episode * cfg["batch_size"])
    steps = 0 if arm == "no_adapt" else cfg["steps"]
    evals = 1 if arm == "no_adapt" else 2
    views = 1 if arm == "image_C" else 0
    counts = {k: 0 for k in worker.COUNTERS}
    counts.update(eval_forward_calls=evals, eval_forward_inputs=evals * images,
        adapt_forward_calls=steps, adapt_forward_inputs=steps * images,
        diagnostic_forward_calls=views, diagnostic_forward_inputs=views,
        diagnostic_encoder_calls=views, diagnostic_encoder_inputs=views,
        encoder_calls=evals + steps + views, encoder_inputs=(evals + steps) * images + views)
    return dict(arm=arm, images=images, patches=images, candidate_count=0 if arm == "no_adapt" else 1,
        active_candidates=0 if arm == "no_adapt" else 1, steps=steps, seconds=.1, episode=episode,
        order_position=position, warmup=episode == 0, candidate_sha256="c" * 64, **worker.scope(man), **counts)


def sealed_output(tmp_path, dataset="pc60", stage="full", n=2):
    man = manifest(tmp_path, dataset, stage, n)
    output = tmp_path / f"output-{dataset}-{stage}"
    output.mkdir()
    man["output"] = str(output)
    path = tmp_path / f"manifest-{dataset}-{stage}.json"
    worker.write_json(path, man)
    manifest_sha = worker.sha256(path)
    worker.write_json(output / "prediction_protocol.json", dict(manifest=man, manifest_path=str(path),
        manifest_sha256=manifest_sha, output=str(output), gt_contents_opened=False))
    worker.write_json(output / "initial_state.json", {})
    rows = []
    cfg = man["config"]
    for episode, start in enumerate(range(0, n, cfg["batch_size"])):
        for position, arm in enumerate(worker.confirmation.prior.arm_order(man["arms"], episode, "alternating")):
            row = cost_row(man, arm, episode, position)
            rows.append(row)
            episode_path = output / f"episodes/{episode:06d}.{arm}.json"
            episode_path.parent.mkdir(exist_ok=True)
            worker.write_json(episode_path, dict(row, candidates=[]))
            for i in range(start, min(start + cfg["batch_size"], n)):
                pred = output / f"predictions/{arm}/{i:06d}.npy"
                pred.parent.mkdir(parents=True, exist_ok=True)
                with pred.open("xb") as stream:
                    np.save(stream, np.zeros((224, 224), np.uint16), allow_pickle=False)
    with (output / "cost.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    costs = {a: {k: sum(r[k] for r in rows if r["arm"] == a) for k in worker.COUNTERS} for a in man["arms"]}
    worker.write_json(output / "prediction_result.json", dict(status="predicted", images=n, arms=man["arms"],
        config=cfg, costs=costs, timings=worker.confirmation.prior.timing_summary(rows, man["arms"]),
        timing_scope=worker.TIMING_SCOPE, **worker.scope(man)))
    worker.write_json(output / "PREDICTIONS_SEALED.json", dict(complete=True, gt_contents_opened=False,
        output=str(output), manifest_sha256=manifest_sha, images=n, arms=man["arms"], **worker.scope(man),
        artifact_sha256={name: worker.sha256(output / name) for name in sorted(worker.artifact_names(man))}))
    return man, SimpleNamespace(manifest=path, manifest_sha256=manifest_sha, output=output)


def barrier(tmp_path, args_list, stage="full"):
    path = tmp_path / "barrier.json"
    worker.write_json(path, dict(complete=True, stage=stage, suite_prepared_sha256="d" * 64,
        prediction_seal_sha256={str(a.output): worker.sha256(a.output / "PREDICTIONS_SEALED.json") for a in args_list}))
    return path


def test_shared_math_and_helpers_are_reused():
    assert worker.ExperimentTRACE is worker.confirmation.ExperimentTRACE
    assert worker.make_data is worker.confirmation.make_data
    assert worker.finite_observation is worker.full.finite_observation
    assert worker.candidate_receipts is worker.full.candidate_receipts


@pytest.mark.parametrize("dataset", worker.DATASETS)
def test_all_eight_explicit_manifests_accepted(tmp_path, dataset):
    worker.validate_manifest(manifest(tmp_path, dataset))


@pytest.mark.parametrize("field,value", [("arms", ["image_C"]), ("seed", True), ("dataset", "cityscapes"),
    ("runtime_root", "/tmp/elsewhere"), ("source_sha256", {"../escape.py": "a" * 64})])
def test_manifest_rejects_wrong_scope(tmp_path, field, value):
    man = manifest(tmp_path)
    man[field] = value
    with pytest.raises(RuntimeError):
        worker.validate_manifest(man)


def test_calibration_exact_panel_and_arms(tmp_path):
    man = manifest(tmp_path, stage="calibration", n=256)
    worker.validate_manifest(man)
    man["arms"] = worker.ARMS.copy()
    with pytest.raises(RuntimeError, match="arms"):
        worker.validate_manifest(man)


@pytest.mark.parametrize("arm", worker.ARMS)
@pytest.mark.parametrize("batch", [1, 3])
def test_real_cpu_policy_schedule_reset_and_no_grad_diagnostics(method, arm, batch):
    images = torch.randn(batch, 3, 32, 32)
    method.reset()
    initial = worker.shared.restored_state(method)
    seen = []

    def observe(module, inputs):
        if getattr(method, "_diagnostic", False):
            seen.append(torch.is_grad_enabled())

    handle = method.model.register_forward_pre_hook(observe)
    try:
        original = method.adapt
        with mock.patch.object(method, "adapt", side_effect=AssertionError("NoAdapt called adapt")) if arm == "no_adapt" else mock.patch.object(method, "adapt", wraps=original):
            losses, predictions, stats, row = worker.run_policy(method, arm, images, None, torch.arange(batch),
                meta(batch), config(batch=batch, tiny=True), None, initial)
        worker.validate_cost(row, arm, config(batch=batch, tiny=True))
        assert len(predictions) == batch
        assert not any(seen)
        if arm == "no_adapt":
            assert not losses and not stats and not seen
            assert worker.shared.restored_state(method) == initial
        else:
            receipts = worker.candidate_receipts(stats, arm, None)
            assert receipts and len(losses) == 2
            assert row["diagnostic_encoder_inputs"] == (len(stats) if arm == "image_C" else 0)
    finally:
        handle.remove()
        method.reset()
    assert worker.shared.restored_state(method) == initial


def test_gate_direct_c_includes_actual_parameters_and_all_gradients(method):
    report = worker.compare_gate_episode(method, reference(method), torch.randn(4, 3, 32, 32), None,
        torch.arange(4), meta(4), config(batch=4, tiny=True), None)
    assert report["candidates"] > 0 and report["changed_label_pixels"] == 0
    assert report["max_parameter_abs_difference"] == report["max_gradient_abs_difference"] == 0
    assert all(r["diagnostic_encoder_inputs"] == report["candidates"] for r in report["actual_arm_counts"].values())
    assert all(not p._backward_hooks for p in method.model.parameters())


def test_gate_rejects_parameter_only_difference(method):
    direct = reference(method)
    original = worker.run_policy

    def changed(current, *args):
        outcome = original(current, *args)
        if current is direct:
            with torch.no_grad():
                next(p for p in current.model.parameters() if p.requires_grad).add_(.01)
        return outcome

    with mock.patch.object(worker, "run_policy", side_effect=changed), pytest.raises(RuntimeError, match="equivalence"):
        worker.compare_gate_episode(method, direct, torch.randn(1, 3, 32, 32), None,
            torch.arange(1), meta(1), config(batch=1, tiny=True), None)


def test_nonfinite_guard_cleans_hooks_and_failure_kinds(method):
    before = len(method.model._forward_hooks)
    parameter = next(p for p in method.model.parameters() if p.requires_grad)
    handle = parameter.register_hook(lambda gradient: gradient * float("nan"))
    try:
        with pytest.raises(RuntimeError, match="[Nn]onfinite") as exc:
            worker.run_policy(method, "image_C", torch.randn(1, 3, 32, 32), None, torch.arange(1), meta(1),
                config(tiny=True), None, worker.shared.restored_state(method))
        receipt = worker.failure_receipt(exc.value, "predict", "a" * 64)
        assert receipt["failure_kind"] == "nonfinite" and receipt["ineligible_trial"]
        assert receipt["context"]["actual_partial_counters"]["adapt_forward_calls"] == 1
    finally:
        handle.remove()
    assert len(method.model._forward_hooks) == before
    assert worker.failure_receipt(torch.cuda.OutOfMemoryError("test only"), "predict", "a" * 64)["failure_kind"] == "oom"
    assert worker.failure_receipt(RuntimeError("source drift"), "predict", "a" * 64)["fatal"]


def test_persistent_gate_attempt_marker_has_no_retry(tmp_path):
    output = tmp_path / "gate"
    marker = worker.confirmation.claim_gate_attempt(output, "a" * 64)
    assert marker.exists() and not output.exists()
    with pytest.raises(RuntimeError, match="attempted"):
        worker.confirmation.claim_gate_attempt(output, "a" * 64)


def test_input_hash_and_label_stat_drift_without_annotation_reads(tmp_path):
    man = manifest(tmp_path)
    with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("annotation read")):
        worker.confirmation.verify_label_metadata(man["panel"]["inputs"][0])
    man["panel"]["inputs"][0]["image_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="RGB"):
        worker.verify_inputs(man)


def test_image_and_provenance_aliases_of_labels_rejected_before_hashing(tmp_path):
    man = manifest(tmp_path)
    label = man["panel"]["inputs"][0]["label"]
    man["panel"]["source_hashes"] = {label: {"sha256": "a" * 64}}
    with mock.patch.object(worker, "sha256", side_effect=AssertionError("label bytes opened")):
        with pytest.raises(RuntimeError, match="Label isolation"):
            worker.verify_inputs(man)
    man["panel"].pop("source_hashes")
    image = Path(man["panel"]["inputs"][0]["image"])
    image.unlink()
    image.hardlink_to(label)
    with mock.patch.object(worker, "sha256", side_effect=AssertionError("label bytes opened")):
        with pytest.raises(RuntimeError, match="Label isolation"):
            worker.verify_inputs(man)


def test_source_drift_and_runtime_root_fail_closed(tmp_path):
    man = manifest(tmp_path)
    with pytest.raises(RuntimeError, match="frozen runtime"):
        worker.verify_sources(man)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "worker.py").write_text("not the expected source")
    with mock.patch.object(worker, "ROOT", runtime), pytest.raises(RuntimeError, match="source drift"):
        worker.verify_sources(man)


def test_exact_calibration_barrier_and_output_binding(tmp_path):
    man, args = sealed_output(tmp_path, stage="calibration", n=256)
    path = barrier(tmp_path, [args], "calibration")
    assert worker.verify_barrier(path, man, args.output, args.manifest_sha256) == worker.sha256(path)
    changed = worker.read_json(path)
    changed["prediction_seal_sha256"][str(tmp_path / "extra")] = "a" * 64
    worker.write_json(path, changed, replace=True)
    with pytest.raises(RuntimeError, match="exactly this trial"):
        worker.verify_barrier(path, man, args.output, args.manifest_sha256)
    with pytest.raises(RuntimeError, match="CLI/sealed"):
        worker.verify_prediction_seal(args.output, man, "0" * 64)


def test_full_barrier_requires_eight_distinct_dataset_outputs(tmp_path):
    pairs = [sealed_output(tmp_path, d) for d in worker.DATASETS]
    man, args = pairs[0]
    path = barrier(tmp_path, [a for _, a in pairs])
    assert worker.verify_barrier(path, man, args.output, args.manifest_sha256)
    changed = worker.read_json(path)
    changed["prediction_seal_sha256"].pop(str(pairs[-1][1].output))
    worker.write_json(path, changed, replace=True)
    with pytest.raises(RuntimeError, match="exact prepared runnable"):
        worker.verify_barrier(path, man, args.output, args.manifest_sha256)


def test_barrier_boolean_missing_own_link_and_stage_are_not_sufficient(tmp_path):
    man, args = sealed_output(tmp_path)
    path = tmp_path / "barrier.json"
    for value in (dict(complete=True, stage="full", prediction_seal_sha256={}),
                  dict(complete=True, stage="calibration", prediction_seal_sha256={str(args.output): "a" * 64})):
        worker.write_json(path, value, replace=path.exists())
        with pytest.raises(RuntimeError):
            worker.verify_barrier(path, man, args.output, args.manifest_sha256)


def test_payload_tamper_rejected_before_any_label_read(tmp_path):
    man, args = sealed_output(tmp_path, stage="calibration", n=256)
    args.barrier = barrier(tmp_path, [args], "calibration")
    with (args.output / "predictions/image_C/000000.npy").open("ab") as stream:
        stream.write(b"tamper")
    with mock.patch.object(worker, "admission", return_value=(man, args.output)), mock.patch.object(worker, "verify_inputs"), \
            mock.patch.object(Path, "read_bytes", side_effect=AssertionError("label opened")):
        with pytest.raises(RuntimeError, match="payload drift"):
            worker.score(args)
    assert not (args.output / "SCORING_STARTED.json").exists()


def test_bounded_streaming_score_pooled_counts_hash_once_and_no_retry(tmp_path):
    man, args = sealed_output(tmp_path, stage="calibration", n=256)
    args.barrier = barrier(tmp_path, [args], "calibration")
    original_hash, hashed_predictions = worker.sha256, []

    def observed_hash(path):
        if str(path).endswith(".npy"):
            hashed_predictions.append(str(path))
        return original_hash(path)

    with mock.patch.object(worker, "admission", return_value=(man, args.output)), mock.patch.object(worker, "verify_inputs"), \
            mock.patch.object(worker, "sha256", side_effect=observed_hash):
        result = worker.score(args)
        assert result["arm_results"]["image_C"]["miou_unrounded"] == 100.
        assert result["arm_results"]["image_C"]["intersection"] == [256 * 224 * 224, 0]
        assert result["calibration"] and not result["full_split"]
        assert len(hashed_predictions) == len(set(hashed_predictions)) == 256
        assert (args.output / "SCORED_SEALED.json").exists()
        with pytest.raises(RuntimeError, match="overwrite/resume"):
            worker.score(args)


def test_later_calibration_trial_rejects_annotation_content_drift(tmp_path):
    man, args = sealed_output(tmp_path, stage="calibration", n=256)
    args.barrier = barrier(tmp_path, [args], "calibration")
    with mock.patch.object(worker, "admission", return_value=(man, args.output)), mock.patch.object(worker, "verify_inputs"):
        first = worker.score(args)
    labels = tmp_path / "calibration_labels/pc60.json"
    assert first["calibration_label_seal_sha256"] == worker.sha256(labels)
    later = copy.deepcopy(man)
    later["config"]["lr"] *= 2
    later.update(output=str(tmp_path / "second-trial"), calibration_label_seal_path=str(labels),
                 calibration_label_seal_sha256=worker.sha256(labels))
    destination = Path(later["output"])
    shutil.copytree(args.output, destination, ignore=shutil.ignore_patterns(*worker.SCORE_FILES))
    path = tmp_path / "second.manifest.json"
    worker.write_json(path, later)
    digest = worker.sha256(path)
    worker.write_json(destination / "prediction_protocol.json", dict(manifest=later, manifest_path=str(path),
        manifest_sha256=digest, output=str(destination), gt_contents_opened=False), replace=True)
    predicted = worker.read_json(destination / "prediction_result.json")
    predicted["config"] = later["config"]
    worker.write_json(destination / "prediction_result.json", predicted, replace=True)
    worker.write_json(destination / "PREDICTIONS_SEALED.json", dict(complete=True, gt_contents_opened=False,
        output=str(destination), manifest_sha256=digest, images=256, arms=later["arms"], **worker.scope(later),
        artifact_sha256={name: worker.sha256(destination / name) for name in sorted(worker.artifact_names(later))}), replace=True)
    next_args = SimpleNamespace(manifest=path, manifest_sha256=digest, output=destination)
    boundary = tmp_path / "second-boundary"
    boundary.mkdir()
    next_args.barrier = barrier(boundary, [next_args], "calibration")
    changed = io.BytesIO()
    Image.fromarray(np.ones((2, 2), np.uint8)).save(changed, format="PNG")
    original_read = Path.read_bytes
    annotation = Path(later["panel"]["inputs"][0]["label"])
    def substituted_read(path):
        return changed.getvalue() if path == annotation else original_read(path)
    with mock.patch.object(worker, "admission", return_value=(later, destination)), mock.patch.object(worker, "verify_inputs"), \
            mock.patch.object(Path, "read_bytes", substituted_read), pytest.raises(RuntimeError, match="label content changed"):
        worker.score(next_args)
    assert not (destination / "SCORED_SEALED.json").exists()


def test_cli_requires_barrier_only_for_score():
    base = ["--manifest", "/tmp/manifest.json", "--manifest-sha256", "a" * 64, "--output", "/tmp/output"]
    with pytest.raises(SystemExit):
        worker.main(["--phase", "score", *base])
    with pytest.raises(SystemExit):
        worker.main(["--phase", "predict", *base, "--barrier", "/tmp/barrier.json"])


def attach_prepared_scope(tmp_path, pairs):
    """Make the controller's dynamic manifest and frozen-prepared binding."""
    prepared_path = tmp_path / "SUITE_PREPARED.json"
    runnable = [m["dataset"] for m, _ in pairs]
    entries = {m["dataset"]: dict(config=m["config"], evaluation=m["panel"], calibration=m["panel"]) for m, _ in pairs}
    worker.write_json(tmp_path / "inputs.json", dict(datasets=entries))
    (tmp_path / "frozen_contract.md").write_text("frozen contract")
    prepared = dict(runnable=runnable, blocked={d: "explicit block" for d in worker.DATASETS if d not in runnable},
        runtime_hashes=pairs[0][0]["source_sha256"], inputs_sha256=worker.sha256(tmp_path / "inputs.json"),
        contract_sha256=worker.sha256(tmp_path / "frozen_contract.md"),
        initial_configs={m["dataset"]: m["config"] for m, _ in pairs}, checkpoint=pairs[0][0]["checkpoint"])
    worker.write_json(prepared_path, prepared)
    for man, args in pairs:
        man["suite_prepared_sha256"] = worker.sha256(prepared_path)
        if man["stage"] == "full":
            selection = tmp_path / "selections" / man["dataset"] / "SELECTED_CONFIG_SEALED.json"
            selection.parent.mkdir(parents=True)
            worker.write_json(selection, dict(config=man["config"], calibration_only=True, full_evaluation_used_for_selection=False))
            man.update(selected_config_path=str(selection), selected_config_sha256=worker.sha256(selection))
    if pairs[0][0]["stage"] == "full":
        selections = {m["dataset"]: m["selected_config_sha256"] for m, _ in pairs}
        worker.write_json(tmp_path / "ALL_CONFIGS_SELECTED.json", dict(complete=True, selections=selections))
    for man, args in pairs:
        if man["stage"] == "full":
            man["all_selections_sha256"] = worker.sha256(tmp_path / "ALL_CONFIGS_SELECTED.json")
            old_output = args.output
            args.output = tmp_path / "full" / man["dataset"]
            args.output.parent.mkdir(exist_ok=True)
            old_output.rename(args.output)
            man["output"] = str(args.output)
        worker.write_json(args.manifest, man, replace=True)
        args.manifest_sha256 = worker.sha256(args.manifest)
        protocol = worker.read_json(args.output / "prediction_protocol.json")
        protocol.update(manifest=man, output=str(args.output), manifest_sha256=args.manifest_sha256)
        worker.write_json(args.output / "prediction_protocol.json", protocol, replace=True)
        seal = worker.read_json(args.output / "PREDICTIONS_SEALED.json")
        seal.update(output=str(args.output), manifest_sha256=args.manifest_sha256,
            suite_prepared_sha256=man["suite_prepared_sha256"])
        seal["artifact_sha256"]["prediction_protocol.json"] = worker.sha256(args.output / "prediction_protocol.json")
        worker.write_json(args.output / "PREDICTIONS_SEALED.json", seal, replace=True)
    return prepared


def test_prepared_runnable_scope_blocks_full_until_waterbirds_selected(tmp_path):
    pairs = [sealed_output(tmp_path, d) for d in worker.DATASETS if d != "waterbirds_seg"]
    attach_prepared_scope(tmp_path, pairs)
    man, args = pairs[0]
    path = barrier(tmp_path, [a for _, a in pairs])
    value = worker.read_json(path)
    value["suite_prepared_sha256"] = man["suite_prepared_sha256"]
    worker.write_json(path, value, replace=True)
    with pytest.raises(RuntimeError, match="all eight"):
        worker.verify_barrier(path, man, args.output, args.manifest_sha256)


def test_prepared_config_and_selection_binding_fail_closed(tmp_path):
    pairs = [sealed_output(tmp_path, d) for d in worker.DATASETS]
    attach_prepared_scope(tmp_path, pairs)
    man, args = pairs[0]
    worker.prepared_scope(man)
    man["config"]["lr"] *= 2
    with pytest.raises(RuntimeError, match="calibration-selected"):
        worker.prepared_scope(man)


def test_failure_kind_does_not_use_incidental_nonfinite_filename():
    assert worker.failure_receipt(RuntimeError("Source drift: /nonfinite/value.py"), "predict", "a" * 64)["fatal"]
    assert worker.failure_receipt(RuntimeError("non-finite adaptation gradient"), "predict", "a" * 64)["failure_kind"] == "nonfinite"


def test_gate_wrapper_writes_controller_receipt_and_cannot_retry(tmp_path, method):
    man = manifest(tmp_path, stage="gate", n=4)
    man["config"] = config(tiny=True)
    output = tmp_path / "gate-output"
    args = SimpleNamespace(manifest=tmp_path / "gate-manifest.json", manifest_sha256="a" * 64, output=output)
    image_meta = meta(4)
    data = [dict(img_patches=torch.randn(1, 3, 32, 32), meta={k: v[i] for k, v in image_meta.items()}) for i in range(4)]
    with mock.patch.object(worker, "admission", return_value=(man, output)), \
            mock.patch.object(worker, "verify_inputs", return_value=data), \
            mock.patch.object(worker, "build_method", return_value=method), \
            mock.patch.object(worker.shared, "configure_deterministic_runtime", return_value={}), \
            mock.patch.object(worker.shared, "deterministic_runtime_state", return_value={}), \
            mock.patch.object(worker.confirmation.prior, "gpu_lock") as lock:
        lock.return_value.__enter__.return_value = {"cpu_test": True}
        report = worker.gate(args)
        assert report["passed"] and sum(report["batch_images"]) == 4
        assert worker.read_json(output / "GATE_PASSED.json")["passed"] is True
        assert (tmp_path / "gate-output.STARTED.json").exists()
        with pytest.raises(RuntimeError, match="already attempted"):
            worker.gate(args)


def test_predict_wrapper_is_sealed_and_exact_output_bound_on_cpu(tmp_path, method):
    man = manifest(tmp_path, n=3)
    man["config"] = config(tiny=True)
    output = tmp_path / "predict-output"
    args = SimpleNamespace(manifest=tmp_path / "predict-manifest.json", manifest_sha256="a" * 64, output=output)
    image_meta = meta(3)
    data = [dict(img_patches=torch.randn(1, 3, 32, 32), meta={k: v[i] for k, v in image_meta.items()}) for i in range(3)]
    with mock.patch.object(worker, "admission", return_value=(man, output)), \
            mock.patch.object(worker, "verify_inputs", return_value=data), \
            mock.patch.object(worker, "build_method", return_value=method), \
            mock.patch.object(worker.shared, "configure_deterministic_runtime", return_value={}), \
            mock.patch.object(worker.shared, "deterministic_runtime_state", return_value={}), \
            mock.patch.object(worker.confirmation.prior, "gpu_lock") as lock:
        lock.return_value.__enter__.return_value = {"cpu_test": True}
        worker.predict(args)
        with pytest.raises(FileExistsError):
            worker.predict(args)
    sealed = worker.read_json(output / "PREDICTIONS_SEALED.json")
    assert sealed["complete"] and not sealed["gt_contents_opened"]
    assert set(sealed["artifact_sha256"]) == worker.artifact_names(man)
    assert worker.read_json(output / "prediction_protocol.json")["output"] == str(output)
    for arm in worker.ARMS:
        receipt = worker.read_json(output / f"episodes/000000.{arm}.json")
        assert receipt["full_split"] and not receipt["calibration"]
        assert bool(receipt["candidates"]) is (arm != "no_adapt")
    worker.validate_cost_stream(output, man)


def test_full_streaming_four_policies_matches_controller_result_schema(tmp_path):
    from tools.evolution_tuning import queue

    pairs = [sealed_output(tmp_path, dataset, n=3) for dataset in worker.DATASETS]
    attach_prepared_scope(tmp_path, pairs)
    man, args = next(pair for pair in pairs if pair[0]["dataset"] == "pc60")
    seal = worker.read_json(args.output / "PREDICTIONS_SEALED.json")
    for arm in ("image_confidence", "image_C"):
        for index in range(3):
            pred = np.ones((224, 224), np.uint16)
            if arm == "image_C":
                pred[:, :112] = 0
            path = args.output / f"predictions/{arm}/{index:06d}.npy"
            with path.open("wb") as stream:
                np.save(stream, pred, allow_pickle=False)
            seal["artifact_sha256"][str(path.relative_to(args.output))] = worker.sha256(path)
    worker.write_json(args.output / "PREDICTIONS_SEALED.json", seal, replace=True)
    args.barrier = barrier(tmp_path, [item[1] for item in pairs])
    bound = worker.read_json(args.barrier)
    bound["suite_prepared_sha256"] = man["suite_prepared_sha256"]
    worker.write_json(args.barrier, bound, replace=True)
    with mock.patch.object(worker, "admission", return_value=(man, args.output)), mock.patch.object(worker, "verify_inputs"):
        result = worker.score(args)
    checked = queue.validate_result(args.output, man["dataset"], man["panel"], worker.ARMS, full=True)
    assert checked == result
    assert {a: r["miou_unrounded"] for a, r in result["arm_results"].items()} == dict(
        no_adapt=100., image_uniform=100., image_confidence=0., image_C=25.)
    assert all(t["seconds_per_image"] == .1 for t in result["timings"].values())
    assert result["costs"]["no_adapt"]["adapt_forward_calls"] == 0


def test_predict_oom_retains_partial_counter_receipt_without_sealing(tmp_path, method):
    man = manifest(tmp_path)
    man["config"] = config(tiny=True)
    output = tmp_path / "failed-output"
    args = SimpleNamespace(manifest=tmp_path / "manifest.json", manifest_sha256="a" * 64, output=output)
    image_meta = meta(2)
    data = [dict(img_patches=torch.randn(1, 3, 32, 32), meta={k: v[i] for k, v in image_meta.items()}) for i in range(2)]
    with mock.patch.object(worker, "admission", return_value=(man, output)), \
            mock.patch.object(worker, "verify_inputs", return_value=data), \
            mock.patch.object(worker, "build_method", return_value=method), \
            mock.patch.object(method, "adapt", side_effect=torch.cuda.OutOfMemoryError("simulated, CPU only")), \
            mock.patch.object(worker.shared, "configure_deterministic_runtime", return_value={}), \
            mock.patch.object(worker.confirmation.prior, "gpu_lock") as lock:
        lock.return_value.__enter__.return_value = {"cpu_test": True}
        with pytest.raises(torch.cuda.OutOfMemoryError):
            worker.predict(args)
    receipt = worker.read_json(output / "PREDICTION_FAILED.json")
    assert receipt["failure_kind"] == "oom" and "OutOfMemoryError" in receipt["error"]
    assert receipt["context"]["episode"] == 0
    assert receipt["context"]["arm"] == "image_uniform"
    assert "actual_partial_counters" in receipt["context"]
    assert not receipt["retry_allowed"] and not (output / "PREDICTIONS_SEALED.json").exists()
