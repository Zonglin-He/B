"""CPU-only image-budget experiment admission, accounting and loss tests."""

import json
from pathlib import Path
from unittest import mock

import pytest
import torch
import numpy as np
from PIL import Image

from tools.evolution_confirmation import run as worker
from trace_ovss.method.trace.objective import TraceObjectiveMixin
from test_simplified_trace import tiny_trace


@pytest.fixture
def method():
    base = tiny_trace()
    return worker.ExperimentTRACE(classes=base.classes, lr=base.lr, steps=base.steps,
        trace_max_regions=base.trace_max_regions, trace_view_batch_size=base.trace_view_batch_size,
        vision_outputs=base.vision_outputs, prompts=base.prompt_templates, model=base.model,
        tokenize_fn=base.tokenize, device="cpu", view_mode="min")


@pytest.mark.parametrize("arm", worker.ARMS)
def test_actual_view_counts_and_receipt_budget(method, arm):
    torch.set_num_threads(1)
    images = torch.randn(3, 3, 32, 32)
    method.arm = arm
    with method.count_work() as counts:
        losses = method.adapt(images, image_ids=torch.tensor([7, 7, 33]))
        method.evaluate(images)
    rows = method.last_candidate_stats
    assert rows and len(losses) == method.steps
    views = worker.expected_views(arm, len(rows))
    assert counts["diagnostic_encoder_inputs"] == counts["diagnostic_forward_inputs"] == views
    assert counts["encoder_inputs"] == (method.steps + 2) * 3 + views
    assert counts["diagnostic_encoder_calls"] == counts["diagnostic_forward_calls"]
    coefficients = worker.objective_coefficients(rows, arm)
    assert sum(coefficients) == pytest.approx(1.)
    if arm != "episode_min":
        sources = {r["image_idx"] for r in rows if r["reliability"] > worker.EPS}
        for source in sources:
            assert sum(w for r, w in zip(rows, coefficients) if r["image_idx"] == source) == pytest.approx(1 / len(sources))
    if arm == "image_T":
        assert all(r["counterfactual_support"] is None for r in rows)
    if arm == "image_C":
        assert all(r["target_support"] is None for r in rows)
    method.reset()
    worker.shared.restored_state(method)


def test_episode_control_uses_original_loss_not_balanced_loss(method):
    entropy = torch.tensor([[[1., 3.]], [[10., 20.]]], requires_grad=True)
    terms = [dict(batch_idx=0, image_idx=7, rho=.1, indices=torch.tensor([0])),
             dict(batch_idx=1, image_idx=9, rho=.9, indices=torch.tensor([1]))]
    method.arm = "episode_min"
    actual = method.trace_region_normalized_entropy(entropy, terms,
        prepared=method.prepare_trace_objective(entropy, terms))
    expected = TraceObjectiveMixin.trace_region_normalized_entropy(entropy, terms)
    assert torch.equal(actual, expected)
    method.arm = "image_min"
    balanced = method.trace_region_normalized_entropy(entropy, terms,
        prepared=method.prepare_trace_objective(entropy, terms))
    assert balanced.item() == pytest.approx(10.5)
    assert not torch.isclose(balanced, actual)


def test_objective_receipts_include_inactive_sources_only_when_active():
    rows = [dict(image_idx=1, reliability=.1), dict(image_idx=1, reliability=.2),
            dict(image_idx=2, reliability=.3), dict(image_idx=3, reliability=0.)]
    assert worker.objective_coefficients(rows, "image_T") == pytest.approx([1/6, 1/3, .5, 0.])
    assert worker.objective_coefficients(rows, "episode_min") == pytest.approx([1/6, 1/3, .5, 0.])
    rows[2]["reliability"] = .9
    assert worker.objective_coefficients(rows, "image_T") == pytest.approx([1/6, 1/3, .5, 0.])
    assert worker.objective_coefficients(rows, "episode_min") != pytest.approx([1/6, 1/3, .5, 0.])
    assert worker.objective_coefficients([dict(image_idx=0, reliability=0.)], "image_min") == [0.]


def test_confirmation_predict_needs_already_frozen_global_selection(tmp_path):
    manifest = dict(stage="confirmation", suite_root=str(tmp_path), cohort_sha256="cohort")
    with pytest.raises(FileNotFoundError):
        worker.selection_before_confirmation(manifest)
    path = tmp_path / "SELECTION_SEALED.json"
    path.write_text(json.dumps(dict(complete=True, selected_mode="T", cohort_sha256="different")))
    with pytest.raises(RuntimeError, match="selection"):
        worker.selection_before_confirmation(manifest)
    path.write_text(json.dumps(dict(complete=True, selected_mode="T", cohort_sha256="cohort")))
    with mock.patch.object(worker, "verify_selection_record") as verify:
        result = worker.selection_before_confirmation(manifest)
        verify.assert_called_once_with(tmp_path, worker.read_json(path))
    assert result["selected_mode"] == "T" and result["sha256"] == worker.sha256(path)
    assert worker.selection_before_confirmation(dict(stage="calibration")) is None


@pytest.mark.parametrize("count", [0, 1, 5, 17])
def test_view_schedule(count):
    assert worker.expected_views("episode_min", count) == worker.expected_views("image_min", count) == 2 * count
    assert worker.expected_views("image_T", count) == worker.expected_views("image_C", count) == count
    assert worker.expected_views("image_uniform", count) == worker.expected_views("image_confidence", count) == 0


def test_score_refuses_unbarriered_labels_before_opening_contents(tmp_path):
    output = tmp_path / "cells" / "calibration" / "pc60"
    output.mkdir(parents=True)
    manifest = dict(stage="calibration", suite_root=str(tmp_path))
    with mock.patch.object(worker, "verify_prediction_seal", return_value=dict(manifest=manifest)), \
         mock.patch.object(worker, "checked_sources", return_value={}):
        manifest["source_sha256"] = {}
        with pytest.raises(FileNotFoundError):
            worker.score(output)
    assert not (output / "SCORING_STARTED.json").exists()


def test_order_permutations_change_order_not_membership():
    rows = [dict(official_id=str(i)) for i in range(20)]
    first, second = (worker.reordered(rows, s) for s in (20261030, 20261031))
    assert first != rows and second != rows and first != second
    assert sorted(r["official_id"] for r in first) == sorted(r["official_id"] for r in rows)
    with pytest.raises(RuntimeError):
        worker.reordered(rows, 99)


def test_explicit_image_loader_never_opens_label_pixels(tmp_path, monkeypatch):
    image, label = tmp_path / "image.png", tmp_path / "mask.png"
    Image.fromarray(np.full((32, 32, 3), 120, np.uint8)).save(image)
    label.write_bytes(b"not an image: opening this annotation must fail")
    rows = [dict(official_id="x", image=str(image), label=str(label),
                 image_sha256=worker.sha256(image), label_metadata=[label.stat().st_size, label.stat().st_mtime_ns])]
    original = Path.open

    def guard(path, *args, **kwargs):
        assert path.resolve() != label.resolve(), "Annotation bytes opened during prediction"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    config = dict(dataset="PascalContext60Dataset", data_dir=str(tmp_path), resize=[32, 32],
                  patch_size=[32, 32], patch_stride=16)
    item = worker.make_data(config, rows)[0]
    assert set(item) == {"img_patches", "meta"}
    assert item["img_patches"].shape == (1, 3, 32, 32)
    assert item["meta"]["seg_map_path"] == str(label)


def test_selection_record_recomputes_choice_and_rejects_rule_or_mode_tampering(tmp_path):
    from tools.evolution_confirmation import queue
    rule = dict(tie_band_pp=.02)
    worker.write_json(tmp_path / "frozen_plan.json", dict(selection=rule))
    worker.write_json(tmp_path / "SUITE_PREPARED.json", dict(complete=True))
    worker.write_json(tmp_path / "ALL_CALIBRATION_PREDICTIONS_SEALED.json", dict(complete=True))
    results, hashes = {}, {}
    for dataset in worker.DATASETS:
        cell = tmp_path / "cells" / dataset
        cell.mkdir(parents=True)
        metrics = {a: dict(intersection=[20], union=[100], miou_unrounded=20.) for a in worker.ARMS}
        metrics["image_T"] = dict(intersection=[30], union=[100], miou_unrounded=30.)
        metrics["image_C"] = dict(intersection=[40], union=[100], miou_unrounded=40.)
        metrics["image_min"] = dict(intersection=[50], union=[100], miou_unrounded=50.)
        result = dict(dataset=dataset, stage="calibration", images=128, order_seed=None,
            arm_results=metrics, costs={a: dict(diagnostic_encoder_inputs=1) for a in worker.ARMS})
        results[dataset] = result
        worker.write_json(cell / "result.json", result)
        worker.write_json(cell / "SCORED_SEALED.json", dict(artifact_sha256={"result.json": worker.sha256(cell / "result.json")}))
        hashes[str(cell.relative_to(tmp_path))] = worker.sha256(cell / "SCORED_SEALED.json")
    mode, candidates = queue.select_mode(results, rule)
    selection = dict(rule=rule, selected_mode=mode, candidates=candidates, calibration_score_sha256=hashes,
        suite_prepared_sha256=worker.sha256(tmp_path / "SUITE_PREPARED.json"),
        calibration_barrier_sha256=worker.sha256(tmp_path / "ALL_CALIBRATION_PREDICTIONS_SEALED.json"))
    worker.verify_selection_record(tmp_path, selection)
    with pytest.raises(RuntimeError, match="locked calibration"):
        worker.verify_selection_record(tmp_path, dict(selection, selected_mode="T"))
    with pytest.raises(RuntimeError, match="rule differs"):
        worker.verify_selection_record(tmp_path, dict(selection, rule=dict(tie_band_pp=.1)))
    with pytest.raises(RuntimeError, match="suite binding"):
        worker.verify_selection_record(tmp_path, dict(selection, suite_prepared_sha256="0"*64))


@pytest.mark.parametrize("stage", ["calibration", "confirmation"])
def test_stage_barrier_exact_inventory_and_alias_rejection(tmp_path, monkeypatch, stage):
    from tools.evolution_confirmation.queue import cells
    entries = [e for e in cells() if e["stage"] == stage]
    rows = {}
    for entry in entries:
        rows[str(tmp_path / entry["manifest"])] = dict(stage=stage, dataset=entry["dataset"],
            order_seed=entry["order_seed"], max_images=entry["images"], cohort_sha256="cohort")
    barrier = dict(complete=True, stage=stage, prediction_seal_sha256={e["output"]:"seal" for e in entries})
    monkeypatch.setattr(worker, "sha256", lambda p: "seal")
    monkeypatch.setattr(worker, "read_json", lambda p: rows.get(str(p), barrier))
    by_output = {str(tmp_path / e["output"]):e for e in entries}

    def protocol(path):
        e = by_output[str(path)]
        return dict(manifest=rows[str(tmp_path / e["manifest"])], manifest_sha256="seal",
                    manifest_path=str(tmp_path / e["manifest"]))

    monkeypatch.setattr(worker, "verify_prediction_seal", protocol)
    manifest = dict(stage=stage, suite_root=str(tmp_path), cohort_sha256="cohort")
    assert worker.stage_barrier(tmp_path / entries[0]["output"], manifest) == barrier
    original = dict(barrier["prediction_seal_sha256"])
    barrier["prediction_seal_sha256"] = {"./"*i + entries[0]["output"]:"seal" for i in range(len(entries))}
    with pytest.raises(RuntimeError, match="exact dataset/order"):
        worker.stage_barrier(tmp_path / entries[0]["output"], manifest)
    barrier["prediction_seal_sha256"] = original
    last = entries[-1]
    rows[str(tmp_path / last["manifest"])]["dataset"] = "wrong-dataset"
    with pytest.raises(RuntimeError, match="dataset/order differs"):
        worker.stage_barrier(tmp_path / entries[0]["output"], manifest)


def test_gate_attempt_claim_survives_interruption(tmp_path):
    output = tmp_path / "gate.json"
    marker = worker.claim_gate_attempt(output, "manifest")
    assert marker.is_file() and not output.exists()
    assert worker.read_json(marker) == dict(status="started", manifest_sha256="manifest")
    with pytest.raises(RuntimeError, match="no retry"):
        worker.claim_gate_attempt(output, "manifest")


@pytest.mark.parametrize("failure", ["view", "forward"])
def test_native_gate_reference_never_retries_oom(method, monkeypatch, failure):
    method.__class__ = worker.NativeGateReference
    attempts = []

    def build(keys):
        attempts.append("view")
        if failure == "view":
            raise torch.cuda.OutOfMemoryError("synthetic view construction")
        return torch.ones(len(keys), 3, 32, 32)

    def forward(*args, **kwargs):
        attempts.append("forward")
        raise torch.cuda.OutOfMemoryError("synthetic encoder")

    monkeypatch.setattr(method, "forward_eval_logits", forward)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        list(method._forward_trace_view_batches([(0,"target"), (0,"counterfactual")], build))
    assert attempts == (["view"] if failure == "view" else ["view", "forward"])
