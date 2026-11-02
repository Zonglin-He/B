from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from trace_ovss.configs import BEST_CONFIGS, best_config
from trace_ovss.data.datasets import SegmentationPatchDataset
from trace_ovss.data.vocabulary import load_vocabulary
from trace_ovss.evaluation import SegmentationMetrics, aggregate_patch_logits
from trace_ovss.method.adapter import ModelAdapterMixin
from trace_ovss.method.trace import (
    TraceCandidateMixin,
    TraceEstimatorMixin,
    TraceInterventionMixin,
    TraceObjectiveMixin,
    trace_reliability,
    TRACE_VIEWS,
)
from trace_ovss.models.dense_clip import DenseCLIP
from trace_ovss.models._vendor_clip.model import CLIP as VendorCLIP


def test_reliability_uses_minimum_quality_and_residual_dominance():
    score = trace_reliability(0.8, 0.6, 0.25)
    assert score.quality == pytest.approx(0.6)
    assert score.dominance == pytest.approx(0.0)
    assert score.reliability == pytest.approx(0.6)

    dominated = trace_reliability(0.8, 0.6, 0.9)
    assert dominated.dominance == pytest.approx(0.1)
    assert dominated.reliability == pytest.approx(0.54)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_supports_cannot_produce_trusted_candidates(invalid):
    for scores in ((invalid, .6, .9), (.8, invalid, .9), (.8, .6, invalid)):
        with pytest.raises(ValueError, match="finite probabilities"):
            trace_reliability(*scores)


def test_reliability_is_bounded_and_monotone_after_dominance_starts():
    values = [
        trace_reliability(0.5, 0.7, residual).reliability
        for residual in (0.0, 0.5, 0.7, 1.0)
    ]
    assert all(0.0 <= value <= 1.0 for value in values)
    assert values == sorted(values, reverse=True)


def test_every_candidate_uses_the_same_three_views():
    assert TRACE_VIEWS == (
        "target",
        "residual",
        "counterfactual",
    )


class _CandidateHarness(TraceCandidateMixin):
    classes = ("object", "surface")
    trace_max_regions = 2
    record_candidate_proposals = True
    last_candidate_proposals = []


def test_candidate_selection_is_topk_deterministic_and_valid_masked():
    harness = _CandidateHarness()
    predictions = torch.tensor(
        [[[0, 0, 1, 1, 1], [0, 1, 1, 0, 0], [1, 1, 0, 0, 1]]]
    )
    probabilities = torch.full((1, 2, 3, 5), 0.1)
    probabilities[0, 0][predictions[0] == 0] = 0.8
    probabilities[0, 1][predictions[0] == 1] = 0.9
    valid = torch.ones((1, 3, 5), dtype=torch.bool)
    valid[:, :, -1] = False

    first = harness.collect_trace_candidates(
        probabilities,
        predictions,
        valid,
    )
    second = harness.collect_trace_candidates(
        probabilities,
        predictions,
        valid,
    )
    assert [(item["class_idx"], item["area"]) for item in first] == [
        (1, 6),
        (0, 3),
    ]
    assert [item["component_idx"] for item in first] == [
        item["component_idx"] for item in second
    ]
    assert all(not item["region"][:, -1].any() for item in first)


def test_candidate_budget_is_shared_across_patches_of_one_image():
    harness = _CandidateHarness()
    harness.trace_max_regions = 1
    predictions = torch.tensor([[[0, 0], [0, 0]], [[1, 1], [1, 1]]])
    probabilities = torch.full((2, 2, 2, 2), 0.1)
    probabilities[0, 0] = 0.8
    probabilities[1, 1] = 0.9

    selected = harness.collect_trace_candidates(
        probabilities,
        predictions,
        image_ids=torch.tensor([0, 0]),
    )
    assert len(selected) == 1
    assert selected[0]["batch_idx"] == 1


def test_region_normalization_does_not_scale_with_candidate_area():
    entropy = torch.full((1, 3, 4), 0.7)
    small = torch.zeros((3, 4), dtype=torch.bool)
    small[0, 0] = True
    large = torch.ones((3, 4), dtype=torch.bool)

    small_loss = TraceObjectiveMixin.trace_region_normalized_entropy(
        entropy,
        [{"batch_idx": 0, "mask": small, "rho": 0.4}],
    )
    large_loss = TraceObjectiveMixin.trace_region_normalized_entropy(
        entropy,
        [{"batch_idx": 0, "mask": large, "rho": 0.4}],
    )
    assert small_loss.item() == pytest.approx(0.7)
    assert large_loss.item() == pytest.approx(0.7)


class _ViewHarness(TraceInterventionMixin):
    def __init__(self, batch_size):
        self.trace_view_batch_size = batch_size
        self._mean = torch.zeros((1, 3, 1, 1))
        self._std = torch.ones((1, 3, 1, 1))

    @staticmethod
    def forward_eval_logits(x, interpolate, save_weights=False):
        return torch.stack((x[:, 0], x[:, 1]), dim=1)


def test_batched_views_match_serial_view_evaluation():
    images = (
        torch.arange(3 * 6 * 6, dtype=torch.float32).reshape(1, 3, 6, 6)
        / 100
    )
    thing = np.zeros((6, 6), dtype=bool)
    thing[:3, :3] = True
    stuff = np.zeros((6, 6), dtype=bool)
    stuff[3:, 3:] = True
    candidates = [
        {
            "batch_idx": 0,
            "image_idx": 0,
            "class_idx": 0,
            "class_name": "object",
            "region": thing,
            "area": 9,
            "confidence": 0.8,
            "component_idx": 1,
        },
        {
            "batch_idx": 0,
            "image_idx": 0,
            "class_idx": 1,
            "class_name": "surface",
            "region": stuff,
            "area": 9,
            "confidence": 0.7,
            "component_idx": 1,
        },
    ]
    original = _ViewHarness.forward_eval_logits(
        images,
        False,
    ).softmax(dim=1)
    serial = _ViewHarness(1).evaluate_trace_views(
        images,
        original,
        candidates,
    )
    batched = _ViewHarness(8).evaluate_trace_views(
        images,
        original,
        candidates,
    )

    for serial_item, batched_item in zip(serial, batched):
        for key in (
            "target_support",
            "residual_support",
            "counterfactual_support",
            "quality",
            "dominance",
            "reliability",
        ):
            assert serial_item[key] == pytest.approx(
                batched_item[key],
                abs=1e-7,
            )


def test_valid_mask_is_resized_to_the_dense_token_grid():
    valid = torch.tensor([[[[1, 1], [1, 0]]]], dtype=torch.bool)
    resized = TraceEstimatorMixin.prepare_trace_valid_mask(
        valid,
        batch=1,
        height=4,
        width=4,
        device=torch.device("cpu"),
    )
    assert resized.shape == (1, 4, 4)
    assert resized[0, -1, -1].item() is False


def test_only_visual_layernorm_affine_parameters_are_trainable():
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.ReLU())
    ModelAdapterMixin.set_ln_grads(model)
    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {"1.weight", "1.bias"}


def test_released_configs_have_no_machine_specific_paths(tmp_path):
    assert set(BEST_CONFIGS) == {
        "voc20",
        "voc21",
        "pc59",
        "pc60",
        "coco_object",
        "coco_stuff",
        "cityscapes",
        "coco_cd",
        "waterbirds_seg",
    }
    config = best_config("coco_cd", tmp_path)
    assert config["data_dir"] == (tmp_path / "coco_cd_ovss").resolve()
    assert config["trace_max_regions"] == 4
    assert config["batch_size"] == 28
    assert config["lr"] == pytest.approx(8.6e-3)
    for released in BEST_CONFIGS.values():
        assert not Path(released.data_subdir).is_absolute()


def test_split_file_samples_are_sorted_before_batching(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks" / "test_balanced"
    split_dir = tmp_path / "splits"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    for stem in ("b", "a"):
        (image_dir / f"{stem}.jpg").touch()
        (mask_dir / f"{stem}.png").touch()
    (split_dir / "test_balanced.txt").write_text(
        "b\na\n",
        encoding="utf-8",
    )

    dataset = SegmentationPatchDataset(
        "COCOCDDataset",
        tmp_path,
        resize=(224, 224),
        patch_size=(224, 224),
        patch_stride=112,
    )
    assert [image.stem for image, _ in dataset.samples] == ["a", "b"]


def test_packaged_vocabulary_matches_all_benchmark_label_counts():
    expected = {
        "PascalVOC20Dataset": 20,
        "PascalVOC21Dataset": 21,
        "PascalContext59Dataset": 59,
        "PascalContext60Dataset": 60,
        "COCOObjectDataset": 81,
        "COCOStuffDataset": 171,
        "CityscapesDataset": 19,
        "COCOCDDataset": 3,
        "WaterbirdsSegDataset": 3,
    }
    for dataset_name, count in expected.items():
        vocabulary = load_vocabulary(dataset_name)
        assert len(vocabulary.classes) == count
        assert len(vocabulary.extensions) == len(
            vocabulary.extension_to_class
        )
        assert max(vocabulary.extension_to_class) == count - 1


def test_patch_aggregation_averages_overlap_without_changing_extent():
    patches = torch.tensor(
        [
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[3.0, 3.0], [3.0, 3.0]]],
        ]
    )
    reconstructed = aggregate_patch_logits(
        patches,
        patch_grid_shapes=[(1, 2)],
        image_shapes=[(2, 3)],
        patch_size=(2, 2),
        patch_stride=1,
    )[0]
    assert reconstructed.shape == (1, 2, 3)
    assert torch.equal(
        reconstructed,
        torch.tensor([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]),
    )


def test_segmentation_metrics_use_dataset_level_intersection_and_union():
    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(
        torch.tensor([[0, 1], [1, 1]]),
        torch.tensor([[0, 0], [1, 255]]),
    )
    values = metrics.compute()
    assert values["mIoU"] == pytest.approx(50.0)
    assert values["mAcc"] == pytest.approx(75.0)


def test_dense_visual_layer_indices_are_canonicalized():
    assert DenseCLIP._resolve_layers((-1, -2, 3), 12) == (3, 10, 11)
    with pytest.raises(ValueError):
        DenseCLIP._resolve_layers((-13,), 12)


class _RecordingLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def forward(self, features):
        self.shapes.append(tuple(features.shape))
        return features


class _ProjectionVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_post = _RecordingLayerNorm()
        self.proj = nn.Parameter(torch.eye(4))


class _ProjectionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _ProjectionVisual()


def test_dense_visual_layers_are_projected_independently():
    model = DenseCLIP(_ProjectionBackbone())
    features = torch.randn(3, 2, 5, 4)
    projected = model._project_visual(features)

    assert projected.shape == features.shape
    assert torch.equal(projected, features)
    assert model.visual.ln_post.shapes == [(2, 5, 4)] * 3


def test_standalone_dense_model_supports_mean_and_adaptive_layer_fusion():
    backbone = VendorCLIP(
        embed_dim=16,
        image_resolution=32,
        vision_layers=2,
        vision_width=64,
        vision_patch_size=16,
        context_length=8,
        vocab_size=32,
        transformer_width=64,
        transformer_heads=1,
        transformer_layers=1,
    )
    model = DenseCLIP(
        backbone,
        architecture="reduced",
        attention_strategy="naclip",
    )
    image = torch.randn(2, 3, 32, 32)
    text = torch.nn.functional.normalize(torch.randn(1, 3, 16), dim=-1)
    for fusion in ("mean", "adaptive_weighted_mean"):
        logits, features, _ = model(
            image,
            text,
            True,
            vision_outputs=(-1, -2),
            interpolate=False,
            vision_out_type=fusion,
        )
        assert logits.shape == (1, 2, 3, 2, 2)
        assert features.shape == (2, 5, 16)
        assert torch.isfinite(logits).all()
