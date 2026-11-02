"""CPU-only tests: real tiny encoder, no weights download and no CUDA work."""

import copy
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from tools.evolution.policies import (POLICIES, candidate_digest, expand_policies,
                                     make_fixed_candidate_class, policy_weights)
from tools.evolution.run_cell import (bind_expected_panel, execute_predictions, image_collate,
    checkpoint_identity, image_only_dataset_class, read_json, restored_state, score, seal_predictions, sha256,
    streaming_counts, verify_prediction_seal, write_json)
from trace_ovss.data.datasets import SegmentationPatchDataset
from trace_ovss.method.two_view import TwoViewTRACE
from test_simplified_trace import tiny_trace


def test_every_requested_controller_dataset_is_accepted_by_worker_cli():
    from tools.evolution.run_cell import parser
    from tools.self_evolution_controller import plan
    cells = plan() + plan(["voc20"], trace_only=True)
    for dataset in {cell["dataset"] for cell in cells}:
        args = parser().parse_args(["--package-root", "/tmp/source", "--data-root", "/tmp/data",
            "--output", "/tmp/output", "--dataset", dataset, "--policies", "trace"])
        assert args.dataset == dataset and args.policies == ["trace"]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "data"
    for name in ("JPEGImages", "SegmentationClass", "ImageSets/Segmentation"):
        (root / name).mkdir(parents=True)
    (root / "ImageSets/Segmentation/val.txt").write_text("b\na\n")
    rng = np.random.default_rng(9)
    for name, shape in (("a", (40, 64)), ("b", (64, 40))):
        Image.fromarray(rng.integers(0, 256, (*shape, 3), dtype=np.uint8)).save(root / "JPEGImages" / f"{name}.jpg")
        mask = rng.integers(0, 2, shape, dtype=np.uint8)
        mask[0] = 255
        Image.fromarray(mask).save(root / "SegmentationClass" / f"{name}.png")
    return image_only_dataset_class(SegmentationPatchDataset)("PascalVOC21Dataset", root,
        resize=(48, 32), patch_size=(32, 32), patch_stride=16, include_auxiliary=False)


@pytest.fixture
def stats():
    return [dict(batch_idx=i % 2, image_idx=i % 2, class_idx=i % 2,
                 confidence=.2 + .1 * i, target_support=.9 - .1 * i,
                 counterfactual_support=.1 + .15 * i, region=np.array([[True, False], [False, True]]))
            for i in range(5)]


@pytest.mark.parametrize("policy", POLICIES)
def test_weights_and_fixed_masks(stats, policy):
    before = candidate_digest(stats)
    weights, donor = policy_weights(stats, policy, permutation_seed=2, episode=4)
    t = np.array([row["target_support"] for row in stats])
    c = np.array([row["counterfactual_support"] for row in stats])
    q = np.minimum(t, c)
    expected = {"trace": q, "uniform": np.ones(5), "confidence": np.array([row["confidence"] for row in stats]),
                "target": t, "counterfactual": c, "mean": (t + c) / 2, "permuted": q[donor]}[policy]
    np.testing.assert_array_equal(weights, expected)
    assert candidate_digest(stats) == before
    if policy == "permuted":
        np.testing.assert_array_equal(np.sort(weights), np.sort(q))
        np.testing.assert_array_equal(policy_weights(stats, policy, permutation_seed=2, episode=4)[0], weights)


def test_expansion_and_invalid_values(stats):
    variants = expand_policies(list(POLICIES), [0, 1, 2])
    assert len(variants) == 9 and variants[-1] == ("permuted_2", "permuted", 2)
    for policies, seeds in ((["trace", "trace"], [0]), (["role"], [0]), (["trace"], [-1]), (["trace"], [])):
        with pytest.raises(ValueError):
            expand_policies(policies, seeds)
    stats[0]["target_support"] = float("nan")
    with pytest.raises(ValueError):
        policy_weights(stats, "trace")
    assert policy_weights([], "permuted")[0].size == 0


def test_image_only_matches_native_pixels_patches_and_metadata(data, monkeypatch):
    native = SegmentationPatchDataset("PascalVOC21Dataset", data.data_dir,
        resize=data.resize, patch_size=data.patch_size, patch_stride=data.patch_stride, include_auxiliary=False)
    expected = [native[i] for i in range(len(native))]
    original = Image.open

    def no_gt(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and "SegmentationClass" in str(path):
            raise AssertionError("Image-only loader opened GT")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", no_gt)
    for i in range(len(data)):
        actual = data[i]
        assert "gt" not in actual
        assert torch.equal(actual["img_patches"], expected[i]["img_patches"])
        assert actual["meta"] == expected[i]["meta"]
    with pytest.raises(ValueError):
        image_collate([expected[0]])


def test_input_content_change_is_detected(data):
    data.expected_image_hashes = {str(data.samples[0][0]): "0" * 64}
    with pytest.raises(RuntimeError, match="Image bytes"):
        data[0]


def fixed_method():
    method = tiny_trace()
    method.__class__ = make_fixed_candidate_class(TwoViewTRACE)
    return method


def test_cached_trace_exact_reference_and_independent_reset(monkeypatch):
    from tools.trace_algorithm_variants import make_algorithm_class
    from trace_ovss.method.trace import TRACE
    method = tiny_trace()
    images = torch.randn(2, 3, 32, 32)
    method.__class__ = make_algorithm_class(TRACE, "quality_only")
    expected_loss = method.adapt(images)
    expected_output = method.evaluate(images).clone()
    method.__class__ = make_fixed_candidate_class(TwoViewTRACE)
    method.diagnose(images)
    fingerprint = method.fixed_digest
    # Any fresh diagnostic forward inside adapt must fail this test.
    monkeypatch.setattr(method, "evaluate_trace_views", lambda *a, **k: pytest.fail("diagnostics repeated"))
    outputs = []
    for policy in ("trace", "uniform", "confidence", "trace"):
        method.reset()
        restored_state(method)
        method.select_policy(policy)
        losses = method.adapt(images)
        output = method.evaluate(images).clone()
        if policy == "trace":
            assert losses == expected_loss
            assert torch.equal(output, expected_output)
        outputs.append(output)
        assert method.fixed_digest == fingerprint
    assert torch.equal(outputs[0], outputs[-1])
    with pytest.raises(RuntimeError, match="another episode"):
        method.adapt(images.clone())
    method.fixed_stats[0]["region"][0, 0] ^= True
    with pytest.raises(RuntimeError, match="changed"):
        method.select_policy("trace")


def test_actual_optimizer_reset_assertion_detects_corruption():
    method = fixed_method()
    restored_state(method)
    method.optimizer.param_groups[0]["lr"] *= 2
    with pytest.raises(RuntimeError, match="restore"):
        restored_state(method)
    method.reset()
    restored_state(method)
    method.optimizer.param_groups[0]["params"].pop()
    with pytest.raises(RuntimeError, match="membership"):
        restored_state(method)


def test_official_count_gate_and_parent_binding(tmp_path):
    from tools.suite_inputs import validate_population
    with pytest.raises(RuntimeError, match="Incomplete"):
        validate_population("voc21", "evaluation", 4, 1448)
    validate_population("voc21", "evaluation", 4, 1449)
    panel = dict(dataset="voc21", split="evaluation", seed=0, selection_seed=20261024,
                 image_ids=["a"], label_ids=["b"], file_metadata=[[1, 2]], vocabulary_sha256="v",
                 config={"steps": 36}, available_images=1449, image_sha256={"a": "known-parent-hash"})
    with pytest.raises(RuntimeError, match="Full runs"):
        bind_expected_panel(tmp_path, panel, max_images=None)
    assert bind_expected_panel(tmp_path, panel, max_images=4) is None
    write_json(tmp_path / "expected_panel.json", {**panel, "seed": 9})
    with pytest.raises(RuntimeError, match="seed"):
        bind_expected_panel(tmp_path, panel, max_images=4)
    write_json(tmp_path / "expected_panel.json", panel)
    assert bind_expected_panel(tmp_path, panel, max_images=4) == sha256(tmp_path / "expected_panel.json")
    for key, value in (("config", {"steps": 35}), ("available_images", 1448), ("image_ids", ["c"]),
                       ("vocabulary_sha256", "changed"), ("file_metadata", [[2, 2]]), ("selection_seed", 4),
                       ("image_sha256", {"a": "changed-worker-hash"}), ("seed", 1)):
        changed = {**panel, key: value}
        with pytest.raises(RuntimeError, match=key):
            bind_expected_panel(tmp_path, changed, max_images=4)


@pytest.fixture
def sealed_cell(tmp_path, data):
    output = tmp_path / "cell"
    output.mkdir()
    variants = expand_policies(list(POLICIES), [0, 1, 2])
    config = dict(batch_size=2, patch_size=(32, 32), patch_stride=16, trace_max_regions=4)
    inputs = [dict(image=str(image), label=str(label), image_sha256=sha256(image),
                   label_metadata=[label.stat().st_size, label.stat().st_mtime_ns]) for image, label in data.samples]
    protocol = dict(dataset="voc21", seed=0, full_split=False, classes=["a", "b"], inputs=inputs,
                    config=config, variants=variants, reduce_zero_label=False, timing_scope="shared diagnostic only",
                    data_root=str(data.data_dir))
    write_json(output / "prediction_protocol.json", protocol)
    write_json(output / "initial_state.json", {})
    write_json(output / "checkpoint_identity.json", {})
    # Invalid GT bytes must not affect any predictions or reference checks.
    originals = {Path(row["label"]): Path(row["label"]).read_bytes() for row in inputs}
    for label in originals:
        label.write_bytes(b"NOT AN IMAGE")
    execute_predictions(fixed_method(), data, [0, 1], config, None, variants, output, verify=True)
    for label, content in originals.items():
        label.write_bytes(content)
    # Preserve scoring metadata only after the deliberately poisoned-loader test.
    for row in inputs:
        path = Path(row["label"])
        row["label_metadata"] = [path.stat().st_size, path.stat().st_mtime_ns]
    write_json(output / "prediction_protocol.json", protocol)
    write_json(output / "prediction_result.json", dict(status="predicted"))
    seal_predictions(output, protocol)
    return output


def test_cpu_real_engine_full_predict_seal_score(sealed_cell):
    output = sealed_cell
    protocol = verify_prediction_seal(output)
    candidates = np.load(output / "episodes/000000.npz", allow_pickle=False)
    assert candidates["rho"].shape[0] == 9
    for index in (6, 7, 8):
        np.testing.assert_array_equal(np.sort(candidates["rho"][index]), np.sort(candidates["q"]))
    assert set(candidates["image_index"]).issubset({0, 1})
    receipt = read_json(output / "episodes/000000.json")
    assert len(receipt["reset_assertions"]) == 9
    assert read_json(output / "episodes/000000.reference.json")["passed"]
    result = score(output, dataset="voc21", seed=0)
    assert result["images"] == 2 and not result["full_split"]
    assert len(result["policy_results"]) == 9
    assert all(isinstance(row["miou_unrounded"], float) for row in result["policy_results"])
    assert (output / "SCORED_SEALED.json").is_file()
    assert len(read_json(output / "score_inputs.json")["label_sha256"]) == 2
    assert len((output / "cost.jsonl").read_text().splitlines()) == 9
    with pytest.raises(FileExistsError):
        score(output)


@pytest.mark.parametrize("relative", ["predictions/trace/000000.npy", "episodes/000000.npz", "prediction_protocol.json", "initial_state.json"])
def test_corrupt_payload_fails_before_gt(sealed_cell, relative, monkeypatch):
    output = sealed_cell
    with (output / relative).open("ab") as stream:
        stream.write(b"corrupt")
    monkeypatch.setattr(Image, "open", lambda *a, **k: pytest.fail("GT opened before seal validation"))
    with pytest.raises(RuntimeError, match="seal mismatch"):
        score(output)


def test_unsealed_score_cannot_open_gt(tmp_path, monkeypatch):
    monkeypatch.setattr(Image, "open", lambda *a, **k: pytest.fail("GT opened without seal"))
    with pytest.raises(FileNotFoundError):
        score(tmp_path)


def test_streaming_iou_matches_hand_counts_and_rejects_bad_labels():
    gt = np.array([[0, 0], [1, 255]])
    prediction = np.array([[0, 1], [1, 0]])
    intersection, union = streaming_counts(prediction, gt, 2)
    np.testing.assert_array_equal(intersection, [1, 1])
    np.testing.assert_array_equal(union, [2, 2])
    with pytest.raises(ValueError):
        streaming_counts(np.array([[2, 1], [1, 0]]), gt, 2)
    with pytest.raises(ValueError):
        streaming_counts(prediction, gt[:1], 2)


def test_incomplete_predictions_cannot_be_sealed(tmp_path):
    (tmp_path / "predictions/trace").mkdir(parents=True)
    np.save(tmp_path / "predictions/trace/000000.npy", np.zeros((2, 2), dtype=np.uint16))
    protocol = dict(inputs=[{}], variants=[("trace", "trace", None)], config=dict(batch_size=1))
    with pytest.raises(RuntimeError, match="incomplete predictions"):
        seal_predictions(tmp_path, protocol)


def test_score_request_must_match_sealed_identity(sealed_cell):
    for kwargs in (dict(dataset="pc60"), dict(seed=1), dict(data_root="/different/root")):
        with pytest.raises(RuntimeError, match="Score request"):
            score(sealed_cell, **kwargs)
    assert not (sealed_cell / "result.json").exists()


def test_all_invalid_episode_has_no_candidates_or_optimizer_updates():
    method = fixed_method()
    images = torch.randn(2, 3, 32, 32)
    method.diagnose(images, valid_mask=torch.zeros(2, 1, 32, 32))
    assert method.fixed_stats == []
    for name in POLICIES:
        method.select_policy(name)
        assert method.adapt(images) == []
        restored_state(method)


def test_new_episode_replaces_cache_and_old_tensor_is_rejected():
    method = fixed_method()
    first, second = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
    method.diagnose(first)
    method.select_policy("trace")
    expected = method.adapt(first)
    prediction = method.evaluate(first).clone()
    method.diagnose(second)
    method.select_policy("mean")
    method.adapt(second)
    method.diagnose(first)
    method.select_policy("trace")
    assert method.adapt(first) == expected
    assert torch.equal(method.evaluate(first), prediction)


def test_parent_panel_hash_is_checked_by_prediction_seal(sealed_cell):
    output = sealed_cell
    protocol = read_json(output / "prediction_protocol.json")
    write_json(output / "expected_panel.json", {"identity": 1})
    protocol["expected_panel_sha256"] = sha256(output / "expected_panel.json")
    write_json(output / "prediction_protocol.json", protocol)
    seal = read_json(output / "PREDICTIONS_SEALED.json")
    seal["artifact_sha256"]["prediction_protocol.json"] = sha256(output / "prediction_protocol.json")
    # Simulate a changed parent panel even if omitted from a malformed seal.
    write_json(output / "expected_panel.json", {"identity": 2})
    write_json(output / "PREDICTIONS_SEALED.json", seal)
    with pytest.raises(RuntimeError, match="Parent panel"):
        verify_prediction_seal(output)


def test_checkpoint_is_streamed_once_and_must_match_official_identity(tmp_path, monkeypatch):
    from trace_ovss.models._vendor_clip import clip
    import tools.evolution.run_cell as worker
    path = tmp_path / "ViT-L-14.pt"
    path.write_bytes(b"test checkpoint bytes, not actual model weights")
    expected = sha256(path)
    monkeypatch.setitem(clip._MODELS, "ViT-L/14", f"https://example.invalid/{expected}/ViT-L-14.pt")
    calls = []
    original = worker.sha256

    def count(file):
        calls.append(file)
        return original(file)

    monkeypatch.setattr(worker, "sha256", count)
    identity = checkpoint_identity(path)
    assert identity["sha256"] == expected and identity["bytes"] == path.stat().st_size
    assert len(calls) == 1
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Checkpoint differs"):
        checkpoint_identity(path)


def test_dense_loader_forwards_local_checkpoint_path(monkeypatch, tmp_path):
    import trace_ovss.models.dense_clip as module
    calls = []
    sentinel = object()

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return sentinel, None

    class FakeDense:
        def __init__(self, backbone, **kwargs):
            assert backbone is sentinel

        def to(self, device):
            assert str(device) == "cpu"
            return self

    monkeypatch.setattr(module, "load_clip", load)
    monkeypatch.setattr(module, "DenseCLIP", FakeDense)
    path = str(tmp_path / "ViT-L-14.pt")
    module.load_dense_ovss(backbone=path, device="cpu")
    assert calls[0][0] == path
