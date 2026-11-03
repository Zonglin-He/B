"""CPU mathematical identity, actual work, evidence and GT-barrier regressions."""

import copy
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
import pytest
import torch

from tools.evolution_confirmation import run as old
from tools.evolution_image_full import run as worker
from test_simplified_trace import tiny_trace


@pytest.fixture
def method():
    torch.set_num_threads(1)
    base = tiny_trace()
    return worker.ExperimentTRACE(classes=base.classes, lr=base.lr, steps=base.steps,
        trace_max_regions=base.trace_max_regions, trace_view_batch_size=base.trace_view_batch_size,
        vision_outputs=base.vision_outputs, prompts=base.prompt_templates, model=base.model,
        tokenize_fn=base.tokenize, device="cpu", view_mode="C")


def config(batch=2):
    return dict(dataset="PascalContext60Dataset", data_dir="/synthetic", data_subdir="synthetic",
        resize=[32, 32], patch_size=[32, 32], patch_stride=16, batch_size=batch, lr=.001,
        steps=2, trace_max_regions=4, trace_view_batch_size=3, class_extensions=False)


def batch_meta(n):
    return dict(img_path=[str(i) for i in range(n)], seg_map_path=[f"not-opened-{i}" for i in range(n)],
        ori_shape=[(32, 32)] * n, img_shape=[(32, 32)] * n, patch_shape=[(32, 32)] * n,
        patch_grid_shape=[(1, 1)] * n)


def direct_reference(method):
    method.reset()
    return worker.GateReference(classes=method.classes, lr=method.lr, steps=method.steps,
        trace_max_regions=method.trace_max_regions, trace_view_batch_size=method.trace_view_batch_size,
        vision_outputs=method.vision_outputs, prompts=method.prompt_templates, model=method.model,
        tokenize_fn=method.tokenize, device=method.device, view_mode="C")


def full_manifest(dataset="pc60"):
    cfg = config()
    cfg.update(resize=[224, 224], patch_size=[224, 224])
    rows = [dict(image=f"/synthetic/{i:06d}.jpg", label=f"/synthetic/{i:06d}.png",
                 image_sha256="a" * 64, label_metadata=[1, 1]) for i in range(worker.POPULATIONS[dataset])]
    return dict(format_version=1, worker="image_full", dataset=dataset, stage="full", seed=0,
        order_seed=None, max_images=len(rows), arms=worker.ARMS.copy(), arm_order="alternating",
        config=cfg, timing_scope=worker.TIMING_SCOPE, plan_sha256=worker.queue.PLAN_SHA256,
        selection_sha256=worker.queue.SELECTION_SHA256, preupdate_prediction_status=worker.PREUPDATE_STATUS,
        panel=dict(inputs=rows, ordered_ids=[r["image"] for r in rows], full_split=True,
                   classes=["a", "b"], queries=["a", "b"], query_to_class=None, reduce_zero_label=False))


def test_exact_proven_mathematical_class_is_used():
    assert worker.ExperimentTRACE is old.ExperimentTRACE
    assert worker.make_data is old.make_data
    assert worker.objective_coefficients is old.objective_coefficients


@pytest.mark.parametrize("batch", [9, 14, 16, 1, 28, 5])
@pytest.mark.parametrize("arm", worker.ARMS)
def test_actual_configured_batch_all_zero_single_view_counts(method, batch, arm):
    cfg = config(batch)
    method.arm = arm
    images = torch.randn(batch, 3, 32, 32)
    with method.count_work() as counts:
        losses = method.adapt(images, image_ids=torch.arange(batch))
        method.evaluate(images)
    stats = method.last_candidate_stats
    row = dict(arm=arm, images=batch, patches=batch, seconds=.01, steps=len(losses),
               candidate_count=len(stats), active_candidates=sum(r["reliability"] > worker.EPS for r in stats), **counts)
    worker.validate_cost(row, arm, cfg)
    assert counts["diagnostic_encoder_inputs"] == (len(stats) if arm == "image_C" else 0)
    assert stats and len(losses) == cfg["steps"]
    assert all(r["target_support"] is None for r in stats)


@pytest.mark.parametrize("batch", [1, 4])
def test_direct_c_gate_compares_masks_coefficients_support_losses_gradients_and_labels(method, batch):
    reference = direct_reference(method)
    images = torch.randn(batch, 3, 32, 32)
    report = worker.compare_gate_episode(method, reference, images, None, torch.arange(batch), batch_meta(batch), config(batch), None)
    assert report["candidate_masks_equal"] and report["reset_equal"]
    assert report["changed_label_pixels"] == 0
    for key in ("max_support_abs_difference", "max_coefficient_abs_difference", "max_loss_abs_difference", "max_gradient_abs_difference"):
        assert report[key] == 0.
    assert set(report["actual_arm_counts"]) == set(worker.ARMS)
    assert all(not p._backward_hooks for p in method.model.parameters())
    worker.shared.restored_state(method)


def test_gate_rejects_nonfinite_gradient_without_modifying_hooks(method):
    reference = direct_reference(method)
    parameter = next(p for p in method.model.parameters() if p.requires_grad)
    handle = parameter.register_hook(lambda gradient: gradient * float("nan"))
    try:
        with pytest.raises(RuntimeError, match="[Nn]onfinite|schedule"):
            worker.compare_gate_episode(method, reference, torch.randn(1, 3, 32, 32), None,
                                        torch.tensor([0]), batch_meta(1), config(1), None)
    finally:
        handle.remove()


def test_packed_region_roundtrip_native_stats_null_t_and_category(method):
    method.arm = "image_C"
    method.adapt(torch.randn(3, 3, 32, 32), image_ids=torch.tensor([0, 0, 2]))
    stats = method.last_candidate_stats
    receipts = worker.candidate_receipts(stats, "image_C", [1, 0])
    assert receipts
    for original, row in zip(stats, receipts, strict=True):
        packed = row["region"]
        mask = np.unpackbits(np.frombuffer(bytes.fromhex(packed["data"]), dtype=np.uint8), bitorder="little")
        mask = mask[:np.prod(packed["shape"])].reshape(packed["shape"]).astype(bool)
        assert np.array_equal(mask, original["region"])
        assert packed["true_pixels"] == int(mask.sum())
        assert row["target_support"] is None
        assert row["counterfactual_support"] == original["counterfactual_support"]
        assert row["preupdate_candidate_category_idx"] == [1, 0][original["class_idx"]]
        assert row["batch_idx"] == original["batch_idx"] and row["image_idx"] == original["image_idx"]
    assert sum(r["objective_coefficient"] for r in receipts) == pytest.approx(1.)
    json.dumps(receipts, allow_nan=False)


def test_region_receipts_reject_nonboolean_or_nonfinite_stats():
    row = dict(region=np.zeros((3, 3)), image_idx=0, class_idx=0, reliability=.5)
    with pytest.raises(RuntimeError, match="boolean"):
        worker.candidate_receipts([row], "image_C", None)
    row.update(region=np.ones((3, 3), dtype=bool), confidence=float("nan"))
    with pytest.raises(ValueError):
        worker.candidate_receipts([row], "image_C", None)


@pytest.mark.parametrize("dataset", worker.DATASETS)
def test_new_full_scope_accepts_exact_population_not_old_cohort(dataset):
    manifest = full_manifest(dataset)
    worker.validate_manifest(manifest)
    manifest["max_images"] = 128
    with pytest.raises(RuntimeError):
        worker.validate_manifest(manifest)


@pytest.mark.parametrize("change", ["arms", "stage", "order", "seed", "selection", "timing", "subset"])
def test_full_manifest_scope_tampering_fails_closed(change):
    manifest = full_manifest()
    if change == "arms":
        manifest["arms"] = list(old.ARMS)
    elif change == "stage":
        manifest["stage"] = "confirmation"
    elif change == "order":
        manifest["panel"]["inputs"].reverse()
        manifest["panel"]["ordered_ids"].reverse()
    elif change == "seed":
        manifest["seed"] = True
    elif change == "selection":
        manifest["selection_sha256"] = "0" * 64
    elif change == "timing":
        manifest["timing_scope"] = "omit warmup accuracy"
    else:
        manifest["panel"]["full_split"] = False
    with pytest.raises(RuntimeError):
        worker.validate_manifest(manifest)


def test_config_does_not_depend_on_todays_best_configs():
    cfg = full_manifest()["config"]
    cfg["lr"], cfg["batch_size"] = .12345, 7
    with mock.patch.dict(old.BEST_CONFIGS, {}, clear=True):
        worker.validate_config(cfg)


def test_image_only_loader_stat_allowed_never_reads_gt(tmp_path, monkeypatch):
    image, label = tmp_path / "rgb.png", tmp_path / "gt.png"
    Image.fromarray(np.full((32, 32, 3), 120, np.uint8)).save(image)
    label.write_bytes(b"GT MUST NOT BE OPENED")
    rows = [dict(image=str(image), label=str(label), image_sha256=worker.sha256(image),
                 label_metadata=[label.stat().st_size, label.stat().st_mtime_ns])]
    original = Path.open

    def guard(path, *args, **kwargs):
        assert path != label
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    worker.confirmation.verify_label_metadata(rows[0])
    cfg = dict(config(), data_dir=str(tmp_path))
    assert set(worker.make_data(cfg, rows)[0]) == {"img_patches", "meta"}


def test_execute_streams_all_predictions_receipts_and_costs_including_episode_zero(method, tmp_path):
    cfg, meta = config(2), batch_meta(5)
    data = [dict(img_patches=torch.randn(1, 3, 32, 32), meta={k: v[i] for k, v in meta.items()}) for i in range(5)]
    manifest = dict(config=cfg, max_images=5, panel=dict(ordered_ids=list(range(5)), query_to_class=None, classes=["a", "b"]))
    timings, costs = worker.execute(method, data, manifest, tmp_path)
    worker.write_json(tmp_path / "prediction_result.json", dict(status="predicted", images=5, full_split=True,
        timings=timings, costs=costs, timing_scope=worker.TIMING_SCOPE))
    worker.validate_cost_stream(tmp_path, manifest)
    for arm in worker.ARMS:
        assert len(list((tmp_path / "predictions" / arm).glob("*.npy"))) == 5
        assert timings[arm]["measured_images"] == 3
        assert timings[arm]["accuracy_includes_warmup"] is True
        receipt = worker.read_json(tmp_path / f"episodes/000000.{arm}.json")
        assert receipt["candidates"] and "region" in receipt["candidates"][0]
        assert receipt["candidates"][0]["target_support"] is None
        assert receipt["preupdate_prediction_status"] == worker.PREUPDATE_STATUS
    assert len(list(worker.bounded_json_lines(tmp_path / "cost.jsonl", 9))) == 9
    assert not any(p.grad is not None and torch.count_nonzero(p.grad) for p in method.model.parameters())


@pytest.mark.parametrize("mutation", ["negative", "nan", "floatcounter", "badviews", "badcalls", "boolcounter", "steps"])
def test_actual_cost_validation_rejects_invalid_and_inferred_counts(method, mutation):
    method.arm = "image_C"
    with method.count_work() as counts:
        losses = method.adapt(torch.randn(2, 3, 32, 32), image_ids=torch.tensor([0, 1]))
        method.evaluate(torch.randn(2, 3, 32, 32))
    stats = method.last_candidate_stats
    row = dict(arm="image_C", images=2, patches=2, seconds=.01, steps=len(losses), candidate_count=len(stats),
               active_candidates=sum(r["reliability"] > worker.EPS for r in stats), **counts)
    if mutation == "negative":
        row["encoder_inputs"] = -1
    elif mutation == "nan":
        row["seconds"] = float("nan")
    elif mutation == "floatcounter":
        row["encoder_calls"] = float(row["encoder_calls"])
    elif mutation == "boolcounter":
        row["diagnostic_encoder_calls"] = True
    elif mutation == "badviews":
        row["diagnostic_encoder_inputs"] += 1
    elif mutation == "badcalls":
        row["diagnostic_forward_calls"] += 1
    else:
        row["steps"] -= 1
    with pytest.raises(RuntimeError):
        worker.validate_cost(row, "image_C", config())


def test_jsonl_reader_bounds_lines_and_record_count(tmp_path):
    path = tmp_path / "stream.jsonl"
    path.write_text('{"a":1}\n' * 2)
    with pytest.raises(RuntimeError, match="Unbounded"):
        list(worker.bounded_json_lines(path, 1))
    path.write_text(" " * (64 * 1024 + 1) + "\n")
    with pytest.raises(RuntimeError, match="Unbounded"):
        list(worker.bounded_json_lines(path, 10))


@pytest.mark.parametrize("phase", ["initial readout", "diagnostic", "adaptation", "final evaluation"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_raw_nonfinite_aborts_before_sanitization_or_argmax_and_retains_failure(method, tmp_path, phase, nonfinite):
    def inject(model, args, kwargs, output):
        actual = ("adaptation" if kwargs.get("vision_out_type") == "mean" else
                  "diagnostic" if getattr(method, "_diagnostic", False) else
                  "final evaluation" if kwargs.get("interpolate") else "initial readout")
        if actual == phase:
            logits = output[0].clone()
            logits[(0,) * logits.ndim] = nonfinite
            return logits, *output[1:]

    handle = method.model.register_forward_hook(inject, with_kwargs=True)
    meta = batch_meta(2)
    data = [dict(img_patches=torch.randn(1, 3, 32, 32), meta={k: v[i] for k, v in meta.items()}) for i in range(2)]
    manifest = dict(config=config(), max_images=2, panel=dict(ordered_ids=[0, 1], query_to_class=None, classes=["a", "b"]))
    try:
        with pytest.raises(RuntimeError, match="Nonfinite raw " + phase):
            worker.execute(method, data, manifest, tmp_path)
    finally:
        handle.remove()
    assert not (tmp_path / "PREDICTIONS_SEALED.json").exists()
    assert not (tmp_path / "SCORED_SEALED.json").exists()
    failure = tmp_path / "PREDICTION_FAILED.json"
    assert worker.read_json(failure)["retry_allowed"] is False
    digest = worker.sha256(failure)
    with pytest.raises(FileExistsError):
        worker.execute(method, data, manifest, tmp_path)
    assert worker.sha256(failure) == digest
    assert not method.model._forward_hooks


@pytest.mark.parametrize("role", ["worker", "reference"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_finite_guard_checks_gradients_before_first_optimizer_update(method, role, nonfinite):
    current = method if role == "worker" else direct_reference(method)
    if role == "worker":
        current.arm = "image_C"
    before = worker.prior.state_digest(current.model_state)
    parameter = next(p for p in current.model.parameters() if p.requires_grad)
    handle = parameter.register_hook(lambda g: g * nonfinite)
    try:
        with worker.finite_observation(current):
            with pytest.raises(RuntimeError, match="Nonfinite pre-update"):
                current.adapt(torch.randn(2, 3, 32, 32), image_ids=torch.tensor([0, 1]))
    finally:
        handle.remove()
    assert not current.optimizer.state
    assert worker.prior.state_digest({n: p for n, p in current.model.named_parameters() if p.requires_grad}) == before
    assert not current.model._forward_hooks
    assert not current.optimizer._optimizer_step_pre_hooks
    assert not current.optimizer._optimizer_step_post_hooks


@pytest.mark.parametrize("when", ["pre", "post"])
def test_optimizer_parameter_guard_before_and_after_updates(method, when):
    method.arm = "image_C"
    parameter = next(p for p in method.model.parameters() if p.requires_grad)

    def corrupt(optimizer, args, kwargs):
        with torch.no_grad():
            parameter.fill_(float("inf"))

    install = method.optimizer.register_step_pre_hook if when == "pre" else method.optimizer.register_step_post_hook
    handle = install(corrupt)
    try:
        with worker.finite_observation(method):
            with pytest.raises(RuntimeError, match="Nonfinite " + when + "-update"):
                method.adapt(torch.randn(2, 3, 32, 32), image_ids=torch.tensor([0, 1]))
    finally:
        handle.remove()


@pytest.mark.parametrize("phase", ["initial readout", "diagnostic", "adaptation", "final evaluation"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_raw_guard_also_applies_to_direct_gate_reference(method, phase, nonfinite):
    reference = direct_reference(method)

    def inject(model, args, kwargs, output):
        actual = ("adaptation" if kwargs.get("vision_out_type") == "mean" else
                  "diagnostic" if getattr(reference, "_diagnostic", False) else
                  "final evaluation" if kwargs.get("interpolate") else "initial readout")
        if actual == phase:
            logits = output[0].clone()
            logits[(0,) * logits.ndim] = nonfinite
            return logits, *output[1:]

    handle = reference.model.register_forward_hook(inject, with_kwargs=True)
    try:
        with pytest.raises(RuntimeError, match="Nonfinite raw " + phase):
            worker.compare_gate_episode(method, reference, torch.randn(1, 3, 32, 32), None,
                                        torch.tensor([0]), batch_meta(1), config(1), None)
    finally:
        handle.remove()


def test_finite_flags_use_one_host_item_for_all_parameters(monkeypatch):
    original = torch.Tensor.item
    calls = []

    def item(tensor):
        calls.append(tensor.shape)
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "item", item)
    worker.require_finite([torch.ones(3, 4) for _ in range(100)], "100 parameters")
    assert calls == [torch.Size([])]


@pytest.mark.parametrize("requested_device", ["cpu", "cpu:0"])
@pytest.mark.parametrize("state_key", ["exp_avg", "exp_avg_sq", "step"])
def test_all_optimizer_state_checked_for_indexed_and_unindexed_requested_device(method, requested_device, state_key):
    method.device = torch.device(requested_device)
    method.arm = "image_C"

    def corrupt_state(optimizer, args, kwargs):
        first = next(iter(optimizer.state.values()))
        first[state_key].fill_(float("inf"))

    handle = method.optimizer.register_step_post_hook(corrupt_state)
    try:
        with worker.finite_observation(method):
            with pytest.raises(RuntimeError, match="Nonfinite post-update parameters/optimizer"):
                method.adapt(torch.randn(2, 3, 32, 32), image_ids=torch.tensor([0, 1]))
    finally:
        handle.remove()


def test_observation_guards_leave_finite_trajectory_and_actual_work_identical(method):
    method.arm = "image_C"
    images = torch.randn(4, 3, 32, 32)
    rng = worker.shared.rng_state(method.device)
    outcomes = []
    for guarded in (False, True):
        worker.shared.restore_rng(rng, method.device)
        with worker.finite_observation(method) if guarded else nullcontext():
            with method.count_work() as counts:
                losses = method.adapt(images, image_ids=torch.arange(4))
                predictions = method.evaluate(images).detach().clone()
        outcomes.append((losses, predictions, counts, worker.prior.state_digest(
            {n: p for n, p in method.model.named_parameters() if p.requires_grad})))
    a, b = outcomes
    assert a[0] == b[0] and torch.equal(a[1], b[1]) and a[2:] == b[2:]


def test_score_never_opens_gt_without_exact_six_prediction_barrier(tmp_path, monkeypatch):
    entry = dict(dataset="pc60", output="cell")
    manifest = dict(suite_root=str(tmp_path))
    monkeypatch.setattr(worker, "admission", lambda args: (tmp_path, entry, manifest))
    monkeypatch.setattr(worker.queue, "verify_prepared", lambda *a, **k: dict(cells=worker.queue.cells()))
    with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("GT opened")):
        with pytest.raises(FileNotFoundError):
            worker.score(SimpleNamespace(suite_sha256="pinned"))
    assert not (tmp_path / "cell/SCORING_STARTED.json").exists()


def test_gate_marker_persists_failure_and_prevents_retry(tmp_path):
    output = tmp_path / "gate.json"
    claim = worker.confirmation.claim_gate_attempt(output, "a" * 64)
    assert claim.exists() and not output.exists()
    with pytest.raises(RuntimeError, match="no retry"):
        worker.confirmation.claim_gate_attempt(output, "a" * 64)


def test_streaming_scores_are_pooled_not_mean_of_episodes(tmp_path):
    manifest = dict(config=config(1), max_images=2, panel=dict(classes=["a", "b"], inputs=[dict(label="/a"), dict(label="/b")]))
    counts = [dict(intersection=[1, 0], union=[1, 1]), dict(intersection=[0, 9], union=[9, 9])]
    path = tmp_path / "scores_by_episode.jsonl"
    path.write_text("".join(json.dumps(dict(episode=i, images=1, arms={a: c for a in worker.ARMS})) + "\n" for i, c in enumerate(counts)))
    pooled = dict(intersection=[1, 9], union=[10, 10], miou_unrounded=50.)
    result = dict(arm_results={a: copy.deepcopy(pooled) for a in worker.ARMS})
    worker.write_json(tmp_path / "score_inputs.json", dict(label_sha256={"/a": "a" * 64, "/b": "b" * 64}))
    worker.validate_score_stream(tmp_path, manifest, result)
    result["arm_results"]["image_C"]["miou_unrounded"] = 47.5
    with pytest.raises(RuntimeError, match="Nonpooled|nonpooled"):
        worker.validate_score_stream(tmp_path, manifest, result)
