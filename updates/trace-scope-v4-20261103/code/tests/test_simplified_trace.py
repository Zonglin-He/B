"""Engine and public-interface regression tests for uniform source tracing."""

import numpy as np
import pytest
import torch

from trace_ovss.method.trace import TRACE, TraceObjectiveMixin
from trace_ovss.models._vendor_clip.model import CLIP
from trace_ovss.models.dense_clip import DenseCLIP
from trace_ovss.data.vocabulary import load_vocabulary
from trace_ovss.cli import collapse_class_extensions
from trace_ovss.data.datasets import SegmentationPatchDataset
from trace_ovss.configs import BEST_CONFIGS


def tiny_trace(architecture="reduced", classes=("an object", "a surface"), vision_outputs=(-1, -2)):
    torch.manual_seed(73)
    backbone = CLIP(embed_dim=16, image_resolution=32, vision_layers=2,
                    vision_width=64, vision_patch_size=16, context_length=8,
                    vocab_size=32, transformer_width=64, transformer_heads=1,
                    transformer_layers=1)

    def tokenize(texts):
        tokens = torch.zeros(len(texts), 8, dtype=torch.long)
        tokens[:, 0] = 30
        tokens[:, 1] = torch.tensor([1 + sum(text.encode()) % 28 for text in texts])
        tokens[:, 2] = 31
        return tokens

    return TRACE(classes=classes, lr=.001, steps=2, trace_max_regions=4,
                 trace_view_batch_size=3, vision_outputs=vision_outputs,
                 prompts=("a photo of {}",), tokenize_fn=tokenize,
                 model=DenseCLIP(backbone, architecture=architecture), device="cpu")


@pytest.mark.parametrize("architecture", ["reduced", "vanilla"])
def test_optimizer_contains_exactly_the_enabled_visual_affine_parameters(architecture):
    method = tiny_trace(architecture)
    wanted = {id(p) for p in method.model.visual.parameters() if p.requires_grad}
    actual = {id(p) for group in method.optimizer.param_groups for p in group["params"]}
    assert actual == wanted
    assert len(actual) == len(method.adapted_parameter_names)
    assert all("visual." in name for name in method.adapted_parameter_names)
    final_ln2 = method.model.visual.transformer.resblocks[-1].ln_2
    assert (id(final_ln2.weight) in actual) == (architecture == "vanilla")
    assert (id(final_ln2.bias) in actual) == (architecture == "vanilla")


def test_engine_restores_identical_episode_trajectory_and_skips_invalid_inputs():
    method = tiny_trace()
    torch.manual_seed(6)
    first, second = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
    initial = {name: p.detach().clone() for name, p in method.model.named_parameters()}

    def episode(images):
        losses = method.adapt(images)
        prediction = method.evaluate(images)
        parameters = {name: p.detach().clone() for name, p in method.model.named_parameters()}
        return losses, prediction, parameters

    a, output_a, params_a = episode(first)
    episode(second)
    repeated, output_repeated, params_repeated = episode(first)
    assert a == repeated and len(a) == method.steps
    assert torch.equal(output_a, output_repeated)
    assert all(torch.equal(params_a[name], params_repeated[name]) for name in params_a)
    changed = {name for name in initial if not torch.equal(initial[name], params_a[name])}
    assert changed and changed <= {name for name, p in method.model.named_parameters() if p.requires_grad}
    assert method.model.weights_track == []
    assert method.last_reliability_summary["diagnostic_views"] == 3 * len(method.last_candidate_stats)
    assert all("mask" not in term and "indices" in term for term in method.last_region_terms)

    assert method.adapt(first, valid_mask=torch.zeros(2, 1, 32, 32, dtype=torch.bool)) == []
    assert method.last_region_terms == []
    assert not method.optimizer.state
    assert all(torch.equal(initial[name], p) for name, p in method.model.named_parameters())


def test_singleton_class_and_prompt_axes_survive_text_embedding_cache():
    method = tiny_trace(classes=("a previously unseen class",))
    images = torch.randn(1, 3, 32, 32)
    assert method.text_x.shape == (2, 1, 16)
    assert method.forward_adapt_logits(images).shape == (1, 1, 1, 2, 2)
    assert method.forward_eval_logits(images, interpolate=False).shape == (1, 1, 2, 2)
    assert len(method.adapt(images)) == 2
    assert method.evaluate(images).shape == (1, 1, 32, 32)


@pytest.mark.parametrize("architecture", ["reduced", "vanilla"])
def test_layer_selection_excludes_unexecuted_blocks_from_adaptation(architecture):
    method = tiny_trace(architecture=architecture, vision_outputs=(-2,))
    method.forward_adapt_logits(torch.randn(2, 3, 32, 32)).square().mean().backward()
    assert all(parameter.grad is not None for parameter in method.model.parameters() if parameter.requires_grad)
    assert all(not p.requires_grad for p in method.model.visual.transformer.resblocks[-1].parameters())


def test_collector_support_cache_and_index_only_estimator_match_mask_reference():
    method = tiny_trace()
    torch.manual_seed(21)
    images = torch.randn(2, 3, 32, 32)
    terms = method.estimate_trace_reliability(images)["region_terms"]
    assert terms
    logits = method.forward_eval_logits(images, interpolate=False)
    probabilities = logits.float().softmax(1)
    probabilities = probabilities / probabilities.sum(1, keepdim=True).clamp_min(1e-8)
    for stat in method.last_candidate_stats:
        selected = probabilities[stat["batch_idx"], stat["class_idx"]][torch.from_numpy(stat["region"])]
        assert stat["original_support"] == stat["confidence"] == float(selected.mean())
    entropy = torch.rand(3, 2, 2, 2, requires_grad=True)
    numerator = entropy.sum() * 0
    denominator = entropy.new_tensor(0)
    for term in terms:
        mask = torch.zeros(2 * 2, dtype=torch.bool)
        mask[term["indices"]] = True
        numerator = numerator + term["rho"] * entropy[:, term["batch_idx"]][:, mask.reshape(2, 2)].mean()
        denominator = denominator + entropy.new_tensor(term["rho"])
    expected = numerator / denominator
    actual = TraceObjectiveMixin.trace_region_normalized_entropy(entropy, terms)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, entropy)[0],
                               torch.autograd.grad(expected, entropy)[0], rtol=0, atol=0)


def test_manual_probe_support_falls_back_to_the_original_probabilities():
    method = tiny_trace()
    images = torch.randn(1, 3, 32, 32)
    original = method.forward_eval_logits(images, interpolate=False).softmax(1)
    region = np.array([[True, False], [False, True]])
    probe = dict(batch_idx=0, class_idx=0, region=region)
    result = method.evaluate_trace_views(images, original, [probe])[0]
    assert result["original_support"] == float(original[0, 0][torch.from_numpy(region)].mean())
    assert all(result[key] is not None for key in ("target_support", "residual_support", "counterfactual_support"))


def test_coco_bed_query_cannot_be_collapsed_into_background():
    vocabulary = load_vocabulary("COCOObjectDataset")
    bed_class = vocabulary.classes.index("bed")
    bed_queries = [i for i, text in enumerate(vocabulary.extensions) if text == "bed"]
    assert len(bed_queries) == 1
    assert vocabulary.extension_to_class[bed_queries[0]] == bed_class
    logits = torch.full((len(vocabulary.extensions), 1, 1), -10.0)
    logits[bed_queries] = 10.0
    collapsed = collapse_class_extensions(logits.softmax(0), vocabulary.extension_to_class)
    assert collapsed.argmax(0).item() == bed_class


def test_missing_declared_mask_fails_before_evaluation(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks/test_balanced").mkdir(parents=True)
    (tmp_path / "splits").mkdir()
    for name in ("a", "b", "c"):
        (tmp_path / f"images/{name}.jpg").touch()
    for name in ("a", "b"):
        (tmp_path / f"masks/test_balanced/{name}.png").touch()
    (tmp_path / "splits/test_balanced.txt").write_text("a\nb\nc\n")
    with pytest.raises(FileNotFoundError, match="Incomplete COCOCDDataset"):
        SegmentationPatchDataset("COCOCDDataset", tmp_path, resize=(224, 224),
                                 patch_size=(224, 224), patch_stride=112)


def test_benchmark_settings_share_one_view_batch_default():
    assert {config.trace_view_batch_size for config in BEST_CONFIGS.values()} == {16}
