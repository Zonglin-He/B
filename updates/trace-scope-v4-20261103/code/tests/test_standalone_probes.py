"""CPU-only tests of the new standalone protocol, not scientific GPU evidence."""

import copy
import fcntl
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from tools.evolution_followup import standalone as worker
from trace_ovss.configs import best_config
from trace_ovss.method.trace import TRACE
from trace_ovss.method.two_view import TwoViewTRACE
from test_simplified_trace import tiny_trace


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def method():
    base = tiny_trace()
    return worker.StandaloneTRACE(classes=base.classes, lr=base.lr, steps=base.steps,
        trace_max_regions=base.trace_max_regions, trace_view_batch_size=base.trace_view_batch_size,
        vision_outputs=base.vision_outputs, model=base.model, tokenize_fn=base.tokenize,
        prompts=base.prompt_templates, device="cpu")


def config(tmp_path):
    value = best_config("pc60", tmp_path)
    return json.loads(json.dumps(dict(value, data_dir=str(value["data_dir"]),
        resize=[32, 32], patch_size=[32, 32], patch_stride=16, batch_size=2,
        lr=.001, steps=2, trace_max_regions=4, trace_view_batch_size=3)))


def manifest_fixture(tmp_path, count=3, episode_order_seed=None):
    cfg = config(tmp_path)
    indices = worker.selected_indices(5105, 23, count, episode_order_seed)
    inputs = []
    for offset, index in enumerate(indices):
        image, label = tmp_path / f"rgb{index}.png", tmp_path / f"gt{index}.png"
        Image.fromarray(np.full((32, 32, 3), 80 + offset * 30, np.uint8)).save(image)
        Image.fromarray(np.full((32, 32), offset % 2, np.uint8)).save(label)
        inputs.append(dict(dataset_index=index, official_id=image.name, image=str(image), label=str(label),
            image_sha256=worker.shared.sha256(image), label_metadata=[label.stat().st_size, label.stat().st_mtime_ns]))
    panel = dict(inputs=inputs, available_images=5105, full_split=False,
        ordered_ids=[r["official_id"] for r in inputs], selected_dataset_indices=sorted(indices),
        classes=["object", "surface"], queries=["object", "surface"], query_to_class=[0, 1],
        reduce_zero_label=False, vocabulary_sha256=worker.canonical_hash([["object", "surface"], [0, 1]]))
    manifest = dict(format_version=1, worker="standalone_probes", dataset="pc60", config=cfg,
        seed=0, selection_seed=23, episode_order_seed=episode_order_seed, max_images=count,
        arms=list(worker.ARMS), arm_order="alternating", panel=panel, timing_scope=worker.TIMING_SCOPE,
        exposure_note="Synthetic CPU fixture only", claim_boundary="Not scientific evidence",
        source_sha256=worker.checked_sources())
    return manifest


@pytest.mark.parametrize("arm,multiplier", [("min", 2), ("T", 1), ("C", 1), ("Uniform", 0), ("Confidence", 0)])
def test_view_schedule(arm, multiplier):
    schedule = worker.view_schedule(arm, 5)
    assert len(schedule) == 5 * multiplier
    assert worker.view_schedule(arm, 0) == []
    assert len(set(schedule)) == len(schedule)


@pytest.mark.parametrize("arm", worker.ARMS)
def test_actual_one_two_zero_view_forwards_and_missing_support(method, arm):
    images = torch.randn(2, 3, 32, 32)
    method.arm = arm
    with method.count_work() as counts:
        losses = method.adapt(images)
        method.evaluate(images)
    rows = method.last_candidate_stats
    assert rows and len(losses) == method.steps
    views = len(worker.view_schedule(arm, len(rows)))
    assert counts["diagnostic_forward_inputs"] == counts["diagnostic_encoder_inputs"] == views
    assert counts["diagnostic_forward_calls"] == (views + 2) // 3
    assert counts["diagnostic_encoder_calls"] == counts["diagnostic_forward_calls"]
    assert counts["eval_forward_calls"] == 2
    assert counts["eval_forward_inputs"] == 4
    assert counts["adapt_forward_calls"] == method.steps
    assert counts["adapt_forward_inputs"] == 2 * method.steps
    assert counts["encoder_calls"] == 2 + method.steps + (views + 2) // 3
    assert counts["encoder_inputs"] == 4 + 2 * method.steps + views
    assert method.last_reliability_summary["diagnostic_views"] == views
    for row in rows:
        assert (row["target_support"] is None) == (arm not in ("min", "T"))
        assert (row["counterfactual_support"] is None) == (arm not in ("min", "C"))
        assert row["residual_support"] is row["dominance"] is None
        assert row["reliability"] == worker.allocation(arm, row["target_support"],
            row["counterfactual_support"], row["confidence"])
    weights = worker.normalized_weights([r["reliability"] for r in rows])
    assert sum(weights) == pytest.approx(1)
    assert not hasattr(method, "work")


def test_candidates_identical_across_all_arms_and_single_view_support_matches_min(method):
    images = torch.randn(2, 3, 32, 32)
    reference = None
    for arm in worker.ARMS:
        method.arm = arm
        method.adapt(images)
        rows = method.last_candidate_stats
        identity = worker.candidate_identity(rows)
        if reference is None:
            reference = copy.deepcopy(rows)
        assert identity == worker.candidate_identity(reference)
        if arm in ("T", "C"):
            key = "target_support" if arm == "T" else "counterfactual_support"
            np.testing.assert_allclose([r[key] for r in rows], [r[key] for r in reference], atol=1e-6, rtol=0)
    changed = copy.deepcopy(reference)
    changed[0]["region"] = ~changed[0]["region"]
    assert worker.candidate_identity(changed) != worker.candidate_identity(reference)


def test_single_view_pixels_exactly_equal_the_corresponding_native_views(method):
    class Capture(worker.StandaloneTRACE):
        def forward_eval_logits(self, images, *args, **kwargs):
            if getattr(self, "_diagnostic", False):
                self.captured.extend(images.detach().clone().unbind())
            return super().forward_eval_logits(images, *args, **kwargs)
    method.__class__ = Capture
    images = torch.randn(2, 3, 32, 32)
    reference = {}
    for arm in ("min", "T", "C", "Uniform", "Confidence"):
        method.arm, method.captured = arm, []
        method.adapt(images)
        keys = worker.view_schedule(arm, len(method.last_candidate_stats))
        assert len(keys) == len(method.captured)
        if arm == "min":
            reference = dict(zip(keys, method.captured))
        else:
            assert all(torch.equal(pixels, reference[key]) for key, pixels in zip(keys, method.captured))


@pytest.mark.parametrize("arm", ["T", "C", "Uniform", "Confidence"])
def test_unused_view_helpers_are_not_called(method, arm):
    class CheckHelpers(worker.StandaloneTRACE):
        def make_trace_weak_image(self, image):
            assert self.arm == "T"
            return super().make_trace_weak_image(image)
        def make_trace_counterfactual_background(self, image):
            assert self.arm == "C"
            return super().make_trace_counterfactual_background(image)
    method.__class__ = CheckHelpers
    method.arm = arm
    method.adapt(torch.randn(1, 3, 32, 32))


def test_min_exactly_matches_native_two_view_and_inherits_scientific_logic(method):
    reference = TwoViewTRACE(classes=method.classes, lr=method.lr, steps=method.steps,
        trace_max_regions=method.trace_max_regions, trace_view_batch_size=method.trace_view_batch_size,
        vision_outputs=method.vision_outputs, model=copy.deepcopy(method.model),
        tokenize_fn=method.tokenize, prompts=method.prompt_templates, device="cpu")
    images = torch.randn(2, 3, 32, 32)
    assert method.adapt(images) == reference.adapt(images)
    assert torch.equal(method.evaluate(images), reference.evaluate(images))
    for name in ("adapt", "collect_trace_candidates", "trace_region_normalized_entropy",
                 "make_trace_weak_image", "make_trace_counterfactual_background"):
        assert getattr(worker.StandaloneTRACE, name) is getattr(TRACE, name)
    entropy = torch.rand(2, 2, 2, 2, requires_grad=True)
    terms = method.last_region_terms
    expected = sum(t["rho"] * entropy[:, t["batch_idx"]].flatten(1).index_select(1, t["indices"]).mean()
                   for t in terms) / sum(t["rho"] for t in terms)
    torch.testing.assert_close(method.trace_region_normalized_entropy(entropy, terms), expected)


@pytest.mark.parametrize("arm", worker.ARMS)
def test_empty_candidates_skip_diagnostics_and_adaptation(method, arm):
    method.arm = arm
    images = torch.randn(2, 3, 32, 32)
    with method.count_work() as counts:
        assert method.adapt(images, valid_mask=torch.zeros(2, 1, 32, 32)) == []
        method.evaluate(images)
    assert method.last_candidate_stats == []
    assert counts["diagnostic_forward_inputs"] == counts["adapt_forward_calls"] == 0
    assert counts["eval_forward_calls"] == counts["encoder_calls"] == 2


def test_missing_and_inactive_allocation_not_fabricated():
    assert worker.allocation("T", .3, None, .9) == .3
    assert worker.allocation("C", None, 0, .9) == 0
    assert worker.allocation("Uniform", None, None, .9) == 1
    assert worker.allocation("Confidence", None, None, .9) == .9
    for arm in ("min", "T", "C"):
        with pytest.raises(RuntimeError, match="missing"):
            worker.allocation(arm, None, None, .9)
    assert worker.normalized_weights([]) is None
    assert worker.normalized_weights([0, worker.EPS]) is None
    assert worker.normalized_weights([0, .2, .8]) == [0, .2, .8]
    with pytest.raises(RuntimeError, match="missing"):
        worker.normalized_weights([None])


def test_zero_support_is_inactive_without_a_floor(method):
    rows = [dict(batch_idx=0, image_idx=0, class_idx=0, confidence=0., area=1,
                 component_idx=1, region=np.array([[True, False], [False, False]]))]
    class ZeroConfidence(worker.StandaloneTRACE):
        def collect_trace_candidates(self, *args, **kwargs):
            return rows
    method.__class__ = ZeroConfidence
    method.arm = "Confidence"
    with method.count_work() as counts:
        assert method.adapt(torch.randn(1, 3, 32, 32)) == []
    assert method.last_candidate_stats[0]["reliability"] == 0
    assert counts["diagnostic_forward_inputs"] == counts["adapt_forward_calls"] == 0


def test_oom_is_not_retried_or_rebatched():
    class OOMMethod(worker.StandaloneTRACE):
        trace_view_batch_size = 3
        attempts = 0
        def forward_eval_logits(self, *args, **kwargs):
            self.attempts += 1
            raise torch.cuda.OutOfMemoryError("synthetic CPU test")
    method = object.__new__(OOMMethod)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        list(method._forward_trace_view_batches([(0, "target"), (1, "target")], lambda keys: None))
    assert method.attempts == 1 and method._diagnostic is False


def test_protocol_validation_and_common_transfer_config(tmp_path):
    manifest = manifest_fixture(tmp_path)
    worker.validate_manifest(manifest)
    common = dict(manifest["config"], batch_size=2, lr=.001, steps=10,
                  trace_max_regions=16, trace_view_batch_size=16)
    worker.validate_request("pc60", common, 0, 23, 128, list(worker.ARMS), "fixed")
    for key, value in (("lr", 0), ("steps", True), ("batch_size", -1),
                       ("class_extensions", False), ("new_floor", .01), ("dataset", "COCOObjectDataset")):
        changed = dict(common, **{key: value})
        with pytest.raises(RuntimeError):
            worker.validate_config(changed, "pc60")
    for key, value in (("max_images", 129), ("seed", -1), ("arms", ["T", "T"]),
                       ("arm_order", "search"), ("episode_order_seed", -1)):
        changed = copy.deepcopy(manifest)
        changed[key] = value
        with pytest.raises(RuntimeError):
            worker.validate_manifest(changed)
    changed = copy.deepcopy(manifest)
    changed["panel"]["full_split"] = True
    with pytest.raises(RuntimeError):
        worker.validate_manifest(changed)


def test_episode_order_seed_changes_only_order_not_subset(tmp_path):
    sorted_ids = worker.selected_indices(5105, 23, 128)
    ordered_ids = worker.selected_indices(5105, 23, 128, 77)
    assert sorted_ids == sorted(ordered_ids) and sorted_ids != ordered_ids
    assert ordered_ids == worker.selected_indices(5105, 23, 128, 77)
    assert len(ordered_ids) == len(set(ordered_ids)) == 128
    manifest = manifest_fixture(tmp_path, episode_order_seed=77)
    worker.validate_manifest(manifest)
    manifest["panel"]["inputs"].reverse()
    with pytest.raises(RuntimeError):
        worker.validate_manifest(manifest)


def test_timing_denominators_exclude_first_complete_episode_only():
    rows = [dict(episode=e, arm=a, images=n, seconds=s)
            for e, n, s in ((0, 9, 99), (1, 9, 18), (2, 2, 6)) for a in worker.ARMS]
    summary = worker.timing_summary(rows, worker.ARMS)
    for value in summary.values():
        assert value["measured_episodes"] == 2 and value["measured_images"] == 11
        assert value["seconds"] == 24 and value["seconds_per_image"] == 24 / 11
        assert value["images_per_second"] == 11 / 24
        assert value["accuracy_includes_warmup"] is True
    assert worker.timing_summary(rows[:5], worker.ARMS)["T"]["seconds_per_image"] is None
    assert worker.arm_order(worker.ARMS, 0, "alternating") == list(worker.ARMS)
    assert worker.arm_order(worker.ARMS, 1, "alternating") == list(reversed(worker.ARMS))


def test_rgb_only_loader_never_reads_labels_and_collate_rejects_admission(tmp_path, monkeypatch):
    manifest = manifest_fixture(tmp_path, count=1)
    row = manifest["panel"]["inputs"][0]
    data = object.__new__(worker.shared.image_only_dataset_class(worker.SegmentationPatchDataset))
    data.samples = [(Path(row["image"]), Path(row["label"]))]
    data.resize, data.patch_size, data.patch_stride = (32, 32), (32, 32), 16
    data.expected_image_hashes = {row["image"]: row["image_sha256"]}
    original = Path.read_bytes
    def guarded(path):
        assert str(path) != row["label"], "Prediction attempted label read"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    item = data[0]
    assert set(item) == {"img_patches", "meta"}
    worker.shared.image_collate([item])
    with pytest.raises(ValueError, match="rejects"):
        worker.shared.image_collate([dict(item, gt=torch.zeros(1))])
    data.expected_image_hashes[row["image"]] = "0" * 64
    with pytest.raises(RuntimeError, match="Image bytes"):
        data[0]


def test_sources_and_explicit_hash_gate(tmp_path):
    sources = worker.checked_sources()
    assert str(Path(worker.__file__).resolve()) in sources
    assert not any("probe_audit.py" in p for p in sources)
    assert all(Path(p).is_relative_to(worker.ROOT) for p in sources)
    manifest = manifest_fixture(tmp_path)
    path = tmp_path / "manifest.json"
    worker.write_json(path, manifest)
    expected = worker.shared.sha256(path)
    assert worker.verify_manifest_file(path, expected) == manifest
    with pytest.raises(RuntimeError, match="hash mismatch"):
        worker.verify_manifest_file(path, "0" * 64)
    with pytest.raises(FileExistsError):
        worker.write_json(path, {})
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"seed":0,"seed":1}')
    with pytest.raises(RuntimeError, match="Duplicate"):
        worker.read_json(duplicate)


def test_gpu_lock_is_exclusive_nonblocking_and_released_without_cuda(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.lock"
    monkeypatch.setattr(worker, "GPU_LOCK", path)
    with path.open("a") as parent:
        fcntl.flock(parent, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            with worker.gpu_lock():
                pytest.fail("Contended global lock admitted a second owner")
        fcntl.flock(parent, fcntl.LOCK_UN)
    with worker.gpu_lock() as receipt:
        assert receipt["exclusive"] is True and receipt["inode"] == path.stat().st_ino
    with path.open("a") as parent:
        fcntl.flock(parent, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(parent, fcntl.LOCK_UN)


@pytest.fixture
def predicted(tmp_path, method, monkeypatch):
    manifest = manifest_fixture(tmp_path)
    data = object.__new__(worker.shared.image_only_dataset_class(worker.SegmentationPatchDataset))
    data.samples = [(Path(r["image"]), Path(r["label"])) for r in manifest["panel"]["inputs"]]
    data.resize, data.patch_size, data.patch_stride = (32, 32), (32, 32), 16
    data.expected_image_hashes = {r["image"]: r["image_sha256"] for r in manifest["panel"]["inputs"]}
    output = tmp_path / "output"
    output.mkdir()
    protocol = dict(manifest=manifest, manifest_sha256="synthetic CPU fixture",
        selection_seed=manifest["selection_seed"], episode_order_seed=manifest["episode_order_seed"],
        ordered_ids=manifest["panel"]["ordered_ids"])
    worker.write_json(output / "prediction_protocol.json", protocol)
    worker.write_json(output / "initial_state.json", {})
    labels = {r["label"] for r in manifest["panel"]["inputs"]}
    original = Path.read_bytes
    def guarded(path):
        assert str(path) not in labels, "Prediction read label"
        return original(path)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", guarded)
        timings = worker.execute_predictions(method, data, list(range(3)), manifest, output)
        worker.write_json(output / "prediction_result.json", dict(timings=timings))
        worker.seal_predictions(output, protocol)
    return output, manifest


def test_cpu_end_to_end_streaming_seal_score_all_arms_and_warmup(predicted):
    output, manifest = predicted
    records = [json.loads(line) for line in (output / "cost.jsonl").read_text().splitlines()]
    assert len(records) == 10
    for episode in (0, 1):
        rows = [r for r in records if r["episode"] == episode]
        assert len({r["candidate_sha256"] for r in rows}) == 1
        assert len({r["episode_sha256"] for r in rows}) == 1
        assert [r["arm"] for r in rows] == worker.arm_order(worker.ARMS, episode, "alternating")
        assert all(r["total_eval_forward_calls"] + r["adapt_forward_calls"] == r["total_forward_calls"] for r in rows)
        assert all(r["total_eval_forward_inputs"] + r["adapt_forward_inputs"] == r["total_forward_inputs"] for r in rows)
    assert worker.verify_prediction_seal(output)["manifest"] == manifest
    result = worker.score(output)
    assert result["images"] == 3 and result["full_split"] is False
    assert set(result["arm_results"]) == set(worker.ARMS)
    assert len((output / "scores_by_episode.jsonl").read_text().splitlines()) == 2
    for value in result["timings"].values():
        assert value["measured_images"] == 1
    with pytest.raises(RuntimeError, match="retry"):
        worker.score(output)


def test_incomplete_all_arm_seal_blocks_scoring_before_labels(predicted, monkeypatch):
    output, manifest = predicted
    (output / "predictions" / "C" / "000002.npy").unlink()
    original = Path.read_bytes
    labels = {r["label"] for r in manifest["panel"]["inputs"]}
    def guarded(path):
        assert str(path) not in labels
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        worker.score(output)
    assert not (output / "SCORING_STARTED.json").exists()


def test_parser_requires_explicit_protocol_not_implicit_run():
    with pytest.raises(SystemExit):
        worker.parser().parse_args(["prepare", "--dataset", "pc60"])
    args = worker.parser().parse_args(["prepare", "--dataset", "pc60", "--config", "cfg.json",
        "--seed", "0", "--selection-seed", "23", "--max-images", "128", "--arms", *worker.ARMS,
        "--arm-order", "alternating", "--checkpoint", "existing.pt", "--manifest", "manifest.json",
        "--exposure-note", "Previously exposed evaluation images"])
    assert args.episode_order_seed is None
    assert args.seed == 0 and args.selection_seed == 23 and args.max_images == 128
