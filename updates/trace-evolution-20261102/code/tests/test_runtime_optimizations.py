"""Algorithm-preserving runtime changes: values, gradients and reset semantics."""

import copy

import numpy as np
import pytest
import torch
from torch import nn

from trace_ovss.method.adapter import ModelAdapterMixin
from trace_ovss.method.trace import TraceInterventionMixin, TraceObjectiveMixin
from trace_ovss.models.dense_clip import DenseCLIP
from trace_ovss.models._vendor_clip.model import CLIP
from trace_ovss.cli import build_patch_valid_mask, collapse_class_extensions
from trace_ovss.data.datasets import SegmentationPatchDataset, trace_collate
from trace_ovss.evaluation import aggregate_patch_logits


def reference_loss(entropy, terms):
    numerator = entropy.sum() * 0.0
    denominator = entropy.new_tensor(0.0)
    for term in terms:
        mask = term["mask"]
        if term["rho"] <= 1e-8 or not mask.any():
            continue
        selected = (entropy[:, term["batch_idx"]][:, mask] if entropy.ndim == 4
                    else entropy[term["batch_idx"]][mask])
        rho = entropy.new_tensor(term["rho"])
        numerator = numerator + rho * selected.mean()
        denominator = denominator + rho
    return numerator if denominator.item() <= 1e-8 else numerator / denominator


@pytest.mark.parametrize("prompts", [False, True])
@pytest.mark.parametrize("cached_indices", [False, True])
def test_indexed_objective_preserves_values_and_gradients(prompts, cached_indices):
    torch.manual_seed(7)
    entropy = torch.rand((3, 2, 4, 5) if prompts else (2, 4, 5), requires_grad=True)
    terms = []
    for batch, rho in [(0, 0.3), (1, 0.7), (0, 0.0)]:
        mask = torch.rand(4, 5) > 0.4
        term = {"batch_idx": batch, "rho": rho, "mask": mask}
        if cached_indices:
            term["indices"] = mask.flatten().nonzero().flatten()
        terms.append(term)
    terms.append({"batch_idx": 0, "rho": 0.5, "mask": torch.zeros(4, 5, dtype=torch.bool)})
    expected = reference_loss(entropy, terms)
    actual = TraceObjectiveMixin.trace_region_normalized_entropy(entropy, terms)
    expected_grad = torch.autograd.grad(expected, entropy)[0]
    actual_grad = torch.autograd.grad(actual, entropy)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_layernorm_snapshot_restores_same_adam_trajectory_as_full_reset():
    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 3))
    ModelAdapterMixin.set_ln_grads(model)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
    reference_optimizer = torch.optim.Adam([p for p in reference.parameters() if p.requires_grad], lr=0.01)
    snapshot, optimizer_snapshot = ModelAdapterMixin.copy_model_and_optimizer(model, optimizer)
    full_snapshot = copy.deepcopy(reference.state_dict())
    assert set(snapshot) == {n for n, p in model.named_parameters() if p.requires_grad}
    assert sum(t.numel() for t in snapshot.values()) < sum(t.numel() for t in full_snapshot.values())
    for _ in range(3):
        ModelAdapterMixin.load_model_and_optimizer(model, optimizer, snapshot, optimizer_snapshot)
        reference.load_state_dict(full_snapshot)
        reference_optimizer.load_state_dict(optimizer_snapshot)
        for _ in range(2):
            images = torch.randn(2, 4)
            for net, opt in [(model, optimizer), (reference, reference_optimizer)]:
                net(images).square().mean().backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_tiny_reliability_preserves_near_zero_denominator_fallback(dtype):
    entropy = torch.full((1, 2, 2), 0.5, dtype=dtype, requires_grad=True)
    terms = [{"batch_idx": 0, "mask": torch.ones(2, 2, dtype=torch.bool), "rho": 1.00000001e-8}]
    expected = reference_loss(entropy, terms)
    actual = TraceObjectiveMixin.trace_region_normalized_entropy(entropy, terms)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, entropy)[0],
                               torch.autograd.grad(expected, entropy)[0], rtol=0, atol=0)


def test_prepared_objective_is_reusable_without_retaining_backward_graph():
    mask = torch.tensor([[True, False], [True, True]])
    terms = [{"batch_idx": 0, "mask": mask, "rho": 0.3}]
    prepared = TraceObjectiveMixin.prepare_trace_objective(torch.zeros(1, 2, 2), terms)
    assert not prepared[1].requires_grad
    for _ in range(3):
        entropy = torch.rand(1, 2, 2, requires_grad=True)
        actual = TraceObjectiveMixin.trace_region_normalized_entropy(entropy, terms, prepared=prepared)
        expected = reference_loss(entropy, terms)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(actual, entropy)[0],
                                   torch.autograd.grad(expected, entropy)[0], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_class_extension_max_reduction_matches_groupwise_max(dtype):
    torch.manual_seed(3)
    mapping = [0, 2, 1, 0, 2, 1]
    probabilities = torch.rand(6, 7, 9, dtype=dtype)
    probabilities[0, 0, 0] = float("nan")
    probabilities[:, 1, 1] = 0.0
    expected = torch.stack([
        probabilities[torch.tensor(mapping) == c].max(dim=0).values for c in range(3)
    ])
    actual = collapse_class_extensions(probabilities, mapping)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("shapes", [[(4, 4), (4, 4)], [(5, 6), (1, 1)], []])
def test_bulk_valid_masks_match_per_patch_construction(shapes):
    grids = [(2, 2)] * len(shapes)
    expected = []
    for (h, w), (gh, gw) in zip(shapes, grids):
        for row in range(gh):
            for col in range(gw):
                mask = torch.zeros(1, 3, 3, dtype=torch.bool)
                mask[:, :max(0, min(3, h - row)), :max(0, min(3, w - col))] = True
                expected.append(mask)
    actual = build_patch_valid_mask({"img_shape": shapes, "patch_grid_shape": grids}, (3, 3), 1, "cpu")
    if expected:
        assert torch.equal(actual, torch.stack(expected))
    else:
        assert actual is None


def test_single_patch_reconstruction_preserves_values_and_independent_output():
    patches = torch.randn(2, 3, 4, 4)
    outputs = aggregate_patch_logits(patches, [(1, 1), (1, 1)], [(4, 4), (3, 2)], (4, 4), 2)
    assert torch.equal(outputs[0], patches[0])
    assert torch.equal(outputs[1], patches[1, :, :3, :2])
    original = patches.clone()
    outputs[0].zero_()
    assert torch.equal(patches, original)


def test_lean_evaluation_loader_preserves_images_labels_and_metadata(tmp_path):
    from PIL import Image

    (tmp_path / "images").mkdir()
    (tmp_path / "masks/test_balanced").mkdir(parents=True)
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits/test_balanced.txt").write_text("test\n")
    Image.fromarray(np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)).save(tmp_path / "images/test.jpg")
    Image.fromarray(np.zeros((10, 12), dtype=np.uint8)).save(tmp_path / "masks/test_balanced/test.png")
    kwargs = dict(dataset_name="COCOCDDataset", data_dir=tmp_path, resize=(8, 8), patch_size=(4, 4), patch_stride=2)
    original = trace_collate([SegmentationPatchDataset(**kwargs)[0]])
    lean = trace_collate([SegmentationPatchDataset(**kwargs, include_auxiliary=False)[0]])
    assert set(original) - set(lean) == {"img", "gt_patches"}
    assert original["meta"] == lean["meta"]
    assert torch.equal(original["img_patches"], lean["img_patches"])
    assert torch.equal(original["gt"][0], lean["gt"][0])


class _ViewHarness(TraceInterventionMixin):
    def __init__(self, limit=3):
        self.trace_view_batch_size = limit
        self._mean = torch.zeros(1, 3, 1, 1)
        self._std = torch.ones(1, 3, 1, 1)
        self.batch_sizes = []

    def forward_eval_logits(self, images, interpolate, save_weights=False):
        self.batch_sizes.append(len(images))
        return images[:, :2]


@pytest.mark.parametrize("shape", [(1, 3, 8, 8), (2, 3, 7, 11)])
def test_single_channel_blur_matches_repeated_channel_reference(shape):
    from trace_ovss.method.trace.constants import BLUR_KERNEL

    harness = _ViewHarness()
    harness._mean = torch.tensor([12., 24., 36.]).view(1, 3, 1, 1)
    harness._std = torch.tensor([2., 4., 6.]).view(1, 3, 1, 1)
    image = torch.randn(shape)
    pixels = image * harness._std + harness._mean
    gray = (0.2989 * pixels[:, 0:1] + 0.5870 * pixels[:, 1:2]
            + 0.1140 * pixels[:, 2:3]).repeat(1, 3, 1, 1)
    weak = torch.nn.functional.avg_pool2d(
        gray, BLUR_KERNEL, stride=1, padding=BLUR_KERNEL // 2,
    )
    expected = (weak - harness._mean) / harness._std
    assert torch.equal(harness.make_trace_weak_image(image), expected)


def test_packed_region_indices_preserve_row_major_order_and_empty_regions():
    from trace_ovss.method.trace.indexing import pack_region_indices

    masks = [np.array([[False, True], [True, False]]), np.zeros((2, 2), dtype=bool)]
    indices = pack_region_indices(masks, "cpu")
    assert [index.tolist() for index in indices] == [[1, 2], []]
    assert pack_region_indices([], "cpu") == ()


def test_streamed_views_match_materialized_reference_and_keep_order():
    torch.manual_seed(9)
    images = torch.randn(2, 3, 8, 8)
    original = images[:, :2].softmax(1)
    candidates = []
    expected = []
    harness = _ViewHarness()
    for i in range(4):
        mask = torch.rand(8, 8) > 0.65
        image = images[i % 2:i % 2 + 1]
        weak = harness.make_trace_weak_image(image)
        shifted = harness.make_trace_counterfactual_background(image)
        views = [image * mask + weak * (~mask),
                 image * mask + shifted * (~mask),
                 image * (~mask) + weak * mask]
        values = [float(view[0, :2].softmax(0)[i % 2][mask].mean()) for view in views]
        expected.append(values)
        candidates.append({"batch_idx": i % 2, "class_idx": i % 2, "region": mask.numpy()})
    actual = harness.evaluate_trace_views(images, original, candidates)
    assert harness.batch_sizes == [3, 3, 3, 3]
    for result, values in zip(actual, expected):
        assert result["target_support"] == values[0]
        assert result["counterfactual_support"] == values[1]
        assert result["residual_support"] == values[2]


def test_streamed_views_retry_oom_without_losing_or_reordering_views():
    harness = _ViewHarness(limit=5)
    keys = [(i, "target") for i in range(7)]
    builds = []

    def build(batch):
        builds.append(len(batch))
        if len(batch) > 2:
            raise torch.cuda.OutOfMemoryError("simulated view allocation failure")
        return torch.tensor([i for i, _ in batch]).view(-1, 1, 1, 1).expand(-1, 3, 2, 2).float()

    results = list(harness._forward_trace_view_batches(keys, build))
    assert builds == [5, 2, 2, 2, 1]
    assert [key for batch, _ in results for key in batch] == keys
    assert torch.cat([logits[:, 0, 0, 0] for _, logits in results]).tolist() == list(range(7))
    assert harness.last_trace_view_batch_size == 2


def test_input_mask_allocation_is_inside_the_view_batch_oom_retry(monkeypatch):
    import trace_ovss.method.trace.interventions as module

    harness = _ViewHarness(limit=16)
    images = torch.randn(1, 3, 8, 8)
    original = images[:, :2].softmax(1)
    candidates = [{"batch_idx": 0, "class_idx": 0, "region": np.ones((8, 8), bool)} for _ in range(8)]
    reference_interpolate = module.F.interpolate
    mask_batches = []

    def bounded_interpolate(inputs, *args, **kwargs):
        mask_batches.append(len(inputs))
        if len(inputs) > 2:
            raise torch.cuda.OutOfMemoryError("simulated input-mask allocation OOM")
        return reference_interpolate(inputs, *args, **kwargs)

    monkeypatch.setattr(module.F, "interpolate", bounded_interpolate)
    stats = harness.evaluate_trace_views(images, original, candidates)
    assert len(stats) == len(candidates)
    assert mask_batches[:3] == [6, 3, 2]
    assert harness.last_trace_view_batch_size == 4
    assert all(row["residual_support"] is not None for row in stats)


def test_counterfactual_oom_does_not_poison_the_image_cache(monkeypatch):
    torch.manual_seed(12)
    images = torch.randn(1, 3, 8, 8)
    original = images[:, :2].softmax(1)
    candidates = [{"batch_idx": 0, "class_idx": index % 2, "marker": index,
                   "region": (torch.rand(8, 8) > .5).numpy()} for index in range(8)]
    reference = _ViewHarness(limit=16).evaluate_trace_views(images, original, candidates)
    harness = _ViewHarness(limit=16)
    make_counterfactual = harness.make_trace_counterfactual_background
    calls = []

    def fail_once(image):
        calls.append(True)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("counterfactual cache allocation")
        return make_counterfactual(image)

    monkeypatch.setattr(harness, "make_trace_counterfactual_background", fail_once)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    actual = harness.evaluate_trace_views(images, original, candidates)
    assert len(calls) == 2
    assert harness.last_trace_view_batch_size == 8
    assert [row["marker"] for row in actual] == list(range(8))
    for a, b in zip(actual, reference):
        for field in ("target_support", "residual_support", "counterfactual_support", "quality", "reliability"):
            assert a[field] == b[field]


def test_logit_conversion_oom_retries_without_skipping_views(monkeypatch):
    harness = _ViewHarness(limit=5)
    keys = [(i, "target") for i in range(7)]
    reference = torch.nan_to_num

    def bounded_conversion(value, *args, **kwargs):
        if len(value) > 2:
            raise torch.cuda.OutOfMemoryError("logit sanitization allocation")
        return reference(value, *args, **kwargs)

    monkeypatch.setattr(torch, "nan_to_num", bounded_conversion)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def build(batch):
        return torch.tensor([index for index, _ in batch]).float().view(-1, 1, 1, 1).expand(-1, 3, 2, 2)

    result = list(harness._forward_trace_view_batches(keys, build))
    assert [key for batch, _ in result for key in batch] == keys
    assert harness.batch_sizes == [5, 2, 2, 2, 1]
    assert harness.last_trace_view_batch_size == 2


@pytest.mark.parametrize("architecture,expected_calls", [("reduced", 0), ("vanilla", 1)])
def test_only_unused_final_reduced_mlp_is_skipped(architecture, expected_calls):
    backbone = CLIP(embed_dim=16, image_resolution=32, vision_layers=2,
                    vision_width=64, vision_patch_size=16, context_length=8,
                    vocab_size=32, transformer_width=64, transformer_heads=1,
                    transformer_layers=1)
    model = DenseCLIP(backbone, architecture=architecture)
    calls = []
    hook = backbone.visual.transformer.resblocks[-1].mlp.register_forward_hook(
        lambda *unused: calls.append(True))
    model.encode_dense_layers(torch.randn(2, 3, 32, 32), (-1, -2)).sum().backward()
    hook.remove()
    assert len(calls) == expected_calls
    assert (backbone.visual.transformer.resblocks[-1].ln_2.weight.grad is not None) == bool(expected_calls)
