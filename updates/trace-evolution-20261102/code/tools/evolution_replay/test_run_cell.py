"""CPU-only cache identity, objective algebra, replay gate and seal tests."""

import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from tools.evolution import run_cell as shared
from tools.evolution.policies import candidate_digest, state_digest
from tools.evolution.test_run_cell import data, fixed_method, cpu_threads
from tools.evolution_replay import run_cell as replay
from trace_ovss.method.trace.constants import EPS
from trace_ovss.method.two_view import TwoViewTRACE
from test_simplified_trace import tiny_trace


@pytest.fixture
def parent(tmp_path, data):
    suite = tmp_path / "suite"
    cell = suite / "cells/mechanism/pc60"
    cell.mkdir(parents=True)
    root = suite / "runtime"
    files = {}
    for relative in ("trace_ovss/a.py", "tools/evolution/a.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen test source\n")
        files[relative] = replay.digest(path)
    config = dict(batch_size=2, patch_size=(32, 32), patch_stride=16, trace_max_regions=4)
    inputs = [dict(image=str(image), label=str(label), image_sha256=shared.sha256(image),
                   label_metadata=[label.stat().st_size, label.stat().st_mtime_ns]) for image, label in data.samples]
    protocol = dict(dataset="pc60", seed=0, full_split=True, classes=["a", "b"], inputs=inputs,
        config=config, variants=[("trace", "trace", None)], reduce_zero_label=False,
        timing_scope="test", data_root=str(data.data_dir), package_root=str(root),
        source_sha256={str(root / name): value for name, value in files.items()})
    shared.write_json(cell / "prediction_protocol.json", protocol)
    method = fixed_method()
    shared.write_json(cell / "initial_state.json", dict(model_sha256=state_digest(method.model.state_dict()),
        optimizer_sha256=state_digest(method.optimizer_state)))
    shared.write_json(cell / "checkpoint_identity.json", {})
    shared.execute_predictions(method, data, [0, 1], config, None, [("trace", "trace", None)], cell)
    shared.write_json(cell / "prediction_result.json", dict(status="predicted"))
    shared.seal_predictions(cell, protocol)
    manifest = dict(runtime_hashes=files)
    shared.write_json(suite / "manifest.json", manifest)
    shared.write_json(suite / "MANIFEST_SEALED.json", dict(manifest_sha256=replay.digest(suite / "manifest.json")))
    shared.write_json(suite / "SUITE_SEALED.json", dict(artifact_sha256={
        "manifest.json": replay.digest(suite / "manifest.json"),
        str((cell / "PREDICTIONS_SEALED.json").relative_to(suite)): replay.digest(cell / "PREDICTIONS_SEALED.json")}))
    return cell, protocol, data, root


def method_for(parent):
    cell, protocol, _, _ = parent
    method = tiny_trace()
    method.__class__ = replay.make_replay_class(TwoViewTRACE, cell, protocol["config"])
    return method


def batch_for(data):
    from trace_ovss.cli import build_patch_image_ids
    batch = shared.image_collate([data[i] for i in range(len(data))])
    return batch, build_patch_image_ids(batch["meta"], "cpu")


def test_parent_chain_and_complete_source_hashes(parent, monkeypatch):
    cell, protocol, _, root = parent
    monkeypatch.setitem(replay.DATASETS, "pc60", 2)
    actual, binding, sources = replay.verify_parent(cell, root, shared)
    assert actual["seed"] == 0 and binding["parent_prediction_seal_sha256"] == replay.digest(cell / "PREDICTIONS_SEALED.json")
    assert len(sources) == 2
    (root / "trace_ovss/a.py").write_text("# changed\n")
    with pytest.raises(RuntimeError, match="source differs"):
        replay.verify_parent(cell, root, shared)


def test_parent_must_be_finished_before_followup(tmp_path):
    cell = tmp_path / "suite/cells/mechanism/pc60"
    cell.mkdir(parents=True)
    shared.write_json(tmp_path / "suite/MANIFEST_SEALED.json", {})
    with pytest.raises(FileNotFoundError):
        replay.verify_parent(cell, tmp_path, shared)


def test_reconstructs_exact_parent_fingerprint(parent):
    cell, protocol, _, _ = parent
    stats, receipt = replay.load_cache(cell, 0, protocol["config"], candidate_digest)
    assert candidate_digest(stats) == receipt["candidate_sha256"]
    with np.load(cell / "episodes/000000.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal([row["quality"] for row in stats], archive["q"])
        np.testing.assert_array_equal([row["image_idx"] for row in stats], archive["image_index"])


def test_cache_corruption_is_rejected(parent):
    cell, protocol, _, _ = parent
    with (cell / "episodes/000000.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(RuntimeError, match="cache changed"):
        replay.load_cache(cell, 0, protocol["config"], candidate_digest)


def test_fingerprint_is_checked_even_if_cache_has_a_matching_file_hash(parent):
    cell, protocol, _, _ = parent
    path = cell / "episodes/000000.npz"
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    arrays["confidence"][0] = .123456
    np.savez_compressed(path, **arrays)
    seal = shared.read_json(cell / "PREDICTIONS_SEALED.json")
    seal["artifact_sha256"]["episodes/000000.npz"] = replay.digest(path)
    shared.write_json(cell / "PREDICTIONS_SEALED.json", seal)
    with pytest.raises(RuntimeError, match="fingerprint differs"):
        replay.load_cache(cell, 0, protocol["config"], candidate_digest)


def test_cached_trace_gate_exact_and_zero_diagnostic_forwards(parent, tmp_path, monkeypatch):
    cell, protocol, data, _ = parent
    method = method_for(parent)
    monkeypatch.setattr(method, "evaluate_trace_views", lambda *a, **k: pytest.fail("No T/C forwards allowed"))
    output = tmp_path / "gate"
    output.mkdir()
    replay.exact_gate(method, data, protocol["config"], None, cell, shared, output)
    report = shared.read_json(output / "replay_gate.json")
    assert report["passed"] and report["changed_label_pixels"] == 0
    assert method.replay_episode == 0


def test_exact_label_gate_failure_is_preserved_without_control_predictions(parent, tmp_path, monkeypatch):
    cell, protocol, data, _ = parent
    original = shared.label_predictions

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        result[0][0, 0] = 1 - result[0][0, 0]
        return result

    monkeypatch.setattr(shared, "label_predictions", changed)
    with pytest.raises(RuntimeError, match="exact-label gate failed"):
        replay.exact_gate(method_for(parent), data, protocol["config"], None, cell, shared, tmp_path)
    assert shared.read_json(tmp_path / "replay_gate.json")["passed"] is False
    assert not (tmp_path / "predictions").exists()


def test_initial_hash_mismatch_stops_gate(parent, tmp_path):
    cell, protocol, data, _ = parent
    receipt = shared.read_json(cell / "episodes/000000.json")
    receipt["reset_assertions"]["trace"]["optimizer_sha256"] = "wrong"
    shared.write_json(cell / "episodes/000000.json", receipt)
    seal = shared.read_json(cell / "PREDICTIONS_SEALED.json")
    seal["artifact_sha256"]["episodes/000000.json"] = replay.digest(cell / "episodes/000000.json")
    shared.write_json(cell / "PREDICTIONS_SEALED.json", seal)
    with pytest.raises(RuntimeError, match="initial model/optimizer"):
        replay.exact_gate(method_for(parent), data, protocol["config"], None, cell, shared, tmp_path)


def test_pixel_objective_matches_direct_weighted_pixel_entropy_and_gradients(parent):
    method = method_for(parent)
    masks = [np.array([[1, 0], [0, 0]], dtype=bool), np.array([[0, 1], [1, 1]], dtype=bool)]
    method.fixed_stats = [dict(batch_idx=0, image_idx=0, class_idx=i, confidence=.8,
        target_support=q, counterfactual_support=q, quality=q, region=mask)
        for i, (q, mask) in enumerate(zip((.8, .6), masks))]
    method.fixed_digest = candidate_digest(method.fixed_stats)
    method.fixed_image = torch.zeros(1)
    method.fixed_indices = [torch.from_numpy(mask.flatten().nonzero()[0]) for mask in masks]
    method.select_policy("pixel_normalized")
    np.testing.assert_allclose(method.fixed_weights, [.8, 1.8])
    terms = method.estimate_trace_reliability(method.fixed_image)["region_terms"]
    entropy = torch.tensor([[[1., 2.], [3., 4.]]], requires_grad=True)
    actual = method.trace_region_normalized_entropy(entropy, terms,
        prepared=method.prepare_trace_objective(entropy, terms))
    expected = (.8 * entropy[0, 0, 0] + .6 * entropy[0].flatten()[1:].sum()) / (.8 + .6 * 3)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, entropy, retain_graph=True)[0], torch.autograd.grad(expected, entropy)[0])


def test_parent_active_condition_cannot_be_reactivated_by_area(parent):
    method = method_for(parent)
    method.fixed_stats = [dict(batch_idx=0, image_idx=0, class_idx=0, confidence=.9,
        target_support=EPS / 2, counterfactual_support=.9, quality=EPS / 2, region=np.ones((4, 4), dtype=bool))]
    method.fixed_digest = candidate_digest(method.fixed_stats)
    method.fixed_image = torch.zeros(1)
    method.fixed_indices = [torch.arange(16)]
    method.select_policy("pixel_normalized")
    assert method.fixed_weights.tolist() == [0.0]
    assert method.estimate_trace_reliability(method.fixed_image)["region_terms"] == []


def test_product_does_not_silently_add_another_eps_threshold(parent):
    method = method_for(parent)
    q = EPS * 2
    method.fixed_stats = [dict(batch_idx=0, image_idx=0, class_idx=0, confidence=.9,
        target_support=q, counterfactual_support=q, quality=q, region=np.ones((1, 1), dtype=bool))]
    method.fixed_digest = candidate_digest(method.fixed_stats)
    method.fixed_image = torch.zeros(1)
    method.fixed_indices = [torch.tensor([0])]
    method.select_policy("product")
    assert 0 < method.fixed_weights[0] < EPS
    terms = method.estimate_trace_reliability(method.fixed_image)["region_terms"]
    entropy = torch.tensor([[[3.]]], requires_grad=True)
    actual = method.trace_region_normalized_entropy(entropy, terms, prepared=method.prepare_trace_objective(entropy, terms))
    torch.testing.assert_close(actual, torch.tensor(3.))


def test_controls_predict_seal_then_score_without_gt_in_updates(parent, tmp_path, monkeypatch):
    cell, protocol, data, _ = parent
    method = method_for(parent)
    batch, owners = batch_for(data)
    method.diagnose(batch["img_patches"], image_ids=owners)
    method.select_policy("product")
    t = np.array([row["target_support"] for row in method.fixed_stats])
    c = np.array([row["counterfactual_support"] for row in method.fixed_stats])
    np.testing.assert_array_equal(method.fixed_weights, np.where(np.minimum(t, c) > EPS, t * c, 0))
    method.replay_episode = 0
    output = tmp_path / "controls"
    output.mkdir()
    own_protocol = {**protocol, "variants": replay.VARIANTS}
    shared.write_json(output / "prediction_protocol.json", own_protocol)
    shared.write_json(output / "initial_state.json", {})
    shared.write_json(output / "checkpoint_identity.json", {})
    originals = {label: label.read_bytes() for _, label in data.samples}
    for label in originals:
        label.write_bytes(b"GT unavailable during predict")
    monkeypatch.setattr(method, "evaluate_trace_views", lambda *a, **k: pytest.fail("diagnostics must be cached"))
    shared.execute_predictions(method, data, [0, 1], protocol["config"], None, replay.VARIANTS, output)
    replay.archive_links(cell, output, own_protocol, shared)
    with np.load(output / "episodes/000000.npz", allow_pickle=False) as archive:
        assert "rho" not in archive.files and "effective_weights" in archive.files and "q" in archive.files
        assert archive["effective_weights"].shape[0] == 2
    for label, content in originals.items():
        label.write_bytes(content)
    for row in own_protocol["inputs"]:
        path = Path(row["label"])
        row["label_metadata"] = [path.stat().st_size, path.stat().st_mtime_ns]
    shared.write_json(output / "prediction_protocol.json", own_protocol)
    shared.write_json(output / "prediction_result.json", {})
    shared.seal_predictions(output, own_protocol)
    result = shared.score(output)
    assert {row["policy"] for row in result["policy_results"]} == {"product", "pixel_normalized"}
    assert (output / "SCORED_SEALED.json").is_file()
