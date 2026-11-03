"""CPU equivalence, isolation and actual-work checks for frozen-prefix TRACE."""

import copy
import gc
import importlib
from pathlib import Path
import weakref

import numpy as np
import pytest
import torch

from tools.trace_cached_prefix import (
    CACHED_PREFIX_DEFINITIONS,
    CACHED_PREFIX_SETTINGS,
    CachedPrefixDenseCLIP,
    make_cached_prefix_class,
)
from trace_ovss.method.trace import TRACE
from trace_ovss.models._vendor_clip.model import CLIP
from trace_ovss.models.dense_clip import DenseCLIP


def tiny_dense(*, layers=8, architecture="reduced", attention_strategy="naclip"):
    torch.manual_seed(73)
    backbone = CLIP(embed_dim=16, image_resolution=32, vision_layers=layers,
                    vision_width=64, vision_patch_size=16, context_length=8,
                    vocab_size=32, transformer_width=64, transformer_heads=1,
                    transformer_layers=1)
    return DenseCLIP(backbone, architecture=architecture, attention_strategy=attention_strategy)


def tiny_method(algorithm="quality_tail6", *, layers=8, architecture="reduced",
                steps=3, vision_outputs=(-1, -2, -4, -7)):
    def tokenize(texts):
        tokens = torch.zeros(len(texts), 8, dtype=torch.long)
        tokens[:, 0] = 30
        tokens[:, 1] = torch.tensor([1 + sum(text.encode()) % 28 for text in texts])
        tokens[:, 2] = 31
        return tokens

    method_class = make_cached_prefix_class(TRACE, algorithm)
    return method_class(classes=("an object", "a surface"), lr=.001, steps=steps,
                        trace_max_regions=33, trace_view_batch_size=16,
                        vision_outputs=vision_outputs,
                        prompts=("a photo of {}", "a cropped {}"), tokenize_fn=tokenize,
                        model=tiny_dense(layers=layers, architecture=architecture), device="cpu")


class WorkCounter:
    """Observe actual inputs, independent of cache accounting or forward labels."""

    def __init__(self, model):
        self.stems = []
        self.blocks = [[] for _ in model.visual.transformer.resblocks]
        self.handles = [model.visual.conv1.register_forward_pre_hook(
            lambda module, args: self.stems.append(len(args[0])),
        )]
        for index, block in enumerate(model.visual.transformer.resblocks):
            self.handles.append(block.ln_1.register_forward_pre_hook(
                lambda module, args, i=index: self.blocks[i].append(args[0].shape[1]),
            ))

    def close(self):
        for handle in self.handles:
            handle.remove()


def cache_references(model):
    cached = model._prefix_cache
    assert cached is not None
    tensors = cached.raw_outputs + ((cached.tokens,) if cached.tokens is not None else ())
    assert all(not t.requires_grad and t.grad_fn is None for t in tensors)
    return [weakref.ref(t) for t in tensors]


def test_registry_and_script_import(monkeypatch):
    assert set(CACHED_PREFIX_SETTINGS) == set(CACHED_PREFIX_DEFINITIONS) == {
        "quality_tail6", "quality_tail12", "quality_tail0", "quality_tail6_uncached",
    }
    assert CACHED_PREFIX_SETTINGS["quality_tail6_uncached"] == dict(tail_depth=6, cache_enabled=False)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    module = importlib.import_module("trace_cached_prefix")
    assert module.CACHED_PREFIX_SETTINGS == CACHED_PREFIX_SETTINGS
    assert module.CACHED_PREFIX_DEFINITIONS == CACHED_PREFIX_DEFINITIONS
    with pytest.raises(ValueError, match="Unknown"):
        make_cached_prefix_class(TRACE, "quality_unknown")


@pytest.mark.parametrize("tail_depth", [0, 1, 6])
@pytest.mark.parametrize("architecture,strategy", [
    ("reduced", "naclip"), ("reduced", "vanilla"), ("vanilla", "csa"),
])
@pytest.mark.parametrize("shape", [(32, 32), (48, 32)])
def test_initial_raw_projected_and_aggregated_outputs_match_denseclip(tail_depth, architecture, strategy, shape):
    source = tiny_dense(architecture=architecture, attention_strategy=strategy)
    outputs = (-1, -2, -4, -7)
    model = CachedPrefixDenseCLIP(source, vision_outputs=outputs, tail_depth=tail_depth)
    images = torch.randn(2, 3, *shape)
    text = torch.nn.functional.normalize(torch.randn(2, 3, 16), dim=-1)
    model.bind_original(images)
    with torch.no_grad():
        for project in (False, True):
            expected = source.encode_dense_layers(images, outputs, project=project)
            actual = model.encode_dense_layers(images, outputs, project=project)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for aggregation in ("mean", "adaptive_weighted_mean"):
            kwargs = dict(vision_outputs=outputs, interpolate=True, vision_out_type=aggregation)
            expected = source(images, text, True, **kwargs)
            actual = model(images, text, True, **kwargs)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert model.cache_stats["cache_builds"] == 1
    assert model.cache_stats["cache_hits"] == 3
    cache_references(model)


@pytest.mark.parametrize("algorithm", tuple(CACHED_PREFIX_SETTINGS))
def test_all_18_outputs_protocol_and_optimizer_snapshots(algorithm):
    outputs = tuple(range(-1, -19, -1))
    method = tiny_method(algorithm, layers=24, steps=36, vision_outputs=outputs)
    assert method.vision_outputs == outputs
    assert (method.steps, method.trace_max_regions, method.trace_view_batch_size, method.lr) == (36, 33, 16, .001)
    assert method.prompt_templates == ("a photo of {}", "a cropped {}")
    assert method.text_x.shape == (3, 2, 16)
    assert method.cached_prefix_settings == CACHED_PREFIX_SETTINGS[algorithm]
    assert method.cached_prefix_algorithm == algorithm
    assert method.model.selected_layers == tuple(range(6, 24))
    assert method.model.prefix_depth == 24 - CACHED_PREFIX_SETTINGS[algorithm]["tail_depth"]

    enabled = {name: p for name, p in method.model.named_parameters() if p.requires_grad}
    assert set(enabled) == set(method.model_state)
    assert {id(p) for p in enabled.values()} == {
        id(p) for group in method.optimizer.param_groups for p in group["params"]
    }
    assert {name.removeprefix("backbone.") for name in enabled} == set(method.adapted_parameter_names)
    assert not method.optimizer_state["state"]
    assert len(method.optimizer_state["param_groups"][0]["params"]) == len(enabled)
    assert method.optimizer.param_groups[0]["lr"] == method.lr
    assert method.optimizer.param_groups[0]["betas"] == (.9, .999)
    assert method.optimizer.param_groups[0]["weight_decay"] == 0
    assert not any(p.requires_grad for p in method.model.visual.ln_pre.parameters())
    assert not any(p.requires_grad for p in method.model.visual.transformer.resblocks[-1].ln_2.parameters())
    for index, block in enumerate(method.model.visual.transformer.resblocks):
        for name, parameter in block.named_parameters():
            expected = index >= method.model.prefix_depth and (
                name.startswith("ln_1.") or (index < 23 and name.startswith("ln_2."))
            )
            assert parameter.requires_grad == expected
    images = torch.randn(1, 3, 32, 32)
    method.model.bind_original(images)
    raw = method.model.encode_dense_layers(images, outputs, project=False)
    assert raw.shape == (18, 1, 5, 64)
    if method.model.cache_enabled:
        cached = method.model._prefix_cache
        assert len(cached.raw_outputs) == 18 - method.model.tail_depth
        assert all(t.shape == (1, 5, 64) for t in cached.raw_outputs)
    if algorithm == "quality_tail0":
        assert set(method.adapted_parameter_names) == {"visual.ln_post.weight", "visual.ln_post.bias"}


@pytest.mark.parametrize("architecture", ["reduced", "vanilla"])
@pytest.mark.parametrize("tail_depth", [0, 1, 6])
def test_gradients_and_adam_trajectory_match_same_frozen_uncached_model(architecture, tail_depth):
    source = tiny_dense(architecture=architecture)
    uncached = copy.deepcopy(source)
    outputs = (-1, -2, -4, -7)
    model = CachedPrefixDenseCLIP(source, vision_outputs=outputs, tail_depth=tail_depth)
    reference = CachedPrefixDenseCLIP(uncached, vision_outputs=outputs,
                                     tail_depth=tail_depth, cache_enabled=False)
    images = torch.randn(2, 3, 32, 32)
    text = torch.nn.functional.normalize(torch.randn(2, 3, 16), dim=-1)
    model.bind_original(images)
    optimizers = [torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=.001)
                  for m in (model, reference)]
    kwargs = dict(vision_outputs=outputs, vision_out_type="adaptive_weighted_mean")
    with torch.no_grad():
        model(images, text, True, **kwargs)
    cached_refs = cache_references(model)
    for _ in range(3):
        for m, optimizer in zip((model, reference), optimizers):
            optimizer.zero_grad(set_to_none=True)
            logits = m(images, text, True, **kwargs)[0]
            logits.square().mean().backward()
        for (name, p), (other_name, q) in zip(model.named_parameters(), reference.named_parameters()):
            assert name == other_name
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all()
                assert ".ln_" in name or ".ln_post." in name
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
            else:
                assert p.grad is q.grad is None
        for optimizer in optimizers:
            optimizer.step()
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert model.cache_stats["cache_builds"] == 1
    assert all(ref() is not None for ref in cached_refs)
    assert reference.cache_stats == dict(cache_builds=0, cache_hits=0, cache_bytes=0)


def test_complete_k33_steps36_episode_matches_uncached_and_counts_all_phases():
    method = tiny_method(steps=36)
    reference = tiny_method("quality_tail6_uncached", steps=36)
    images = torch.randn(16, 3, 32, 32)
    hooks, full_hooks = WorkCounter(method.model), WorkCounter(reference.model)
    losses = method.adapt(images)
    expected = reference.adapt(images)
    assert len(losses) == len(expected) == 36
    assert losses == expected
    assert method.adapted_parameter_names == reference.adapted_parameter_names
    assert method.last_reliability_summary == reference.last_reliability_summary
    stats = method.last_candidate_stats
    assert stats
    diagnostic_images = method.last_reliability_summary["diagnostic_views"]
    assert diagnostic_images == 2 * len(stats)
    for row, other in zip(stats, reference.last_candidate_stats):
        np.testing.assert_array_equal(row["region"], other["region"])
        assert row["residual_support"] is row["dominance"] is None
        for key in ("target_support", "counterfactual_support", "quality", "reliability"):
            assert row[key] == other[key]
    cached_refs = cache_references(method.model)
    actual_prediction, expected_prediction = method.evaluate(images), reference.evaluate(images)
    hooks.close()
    full_hooks.close()
    assert torch.equal(actual_prediction, expected_prediction)
    for p, q in zip(method.model.parameters(), reference.model.parameters()):
        assert torch.equal(p, q)
    assert sum(hooks.stems) == len(images) + diagnostic_images
    assert sum(full_hooks.stems) == len(images) * 38 + diagnostic_images
    assert [sum(values) for values in hooks.blocks] == (
        [len(images) + diagnostic_images] * 2
        + [len(images) * 38 + diagnostic_images] * 6
    )
    assert [sum(values) for values in full_hooks.blocks] == [len(images) * 38 + diagnostic_images] * 8
    assert method.model.cache_stats == dict(cache_builds=1, cache_hits=37, cache_bytes=0)
    assert method.model._original_ref is None
    assert all(ref() is None for ref in cached_refs)


@pytest.mark.parametrize("algorithm", ["quality_tail0", "quality_tail12"])
def test_output_ln_only_and_depth12_complete_episodes_reuse_only_frozen_work(algorithm):
    method = tiny_method(algorithm, layers=24, vision_outputs=tuple(range(-1, -19, -1)))
    reference = tiny_method(algorithm, layers=24, vision_outputs=tuple(range(-1, -19, -1)))
    reference.model.cache_enabled = False
    images = torch.randn(2, 3, 32, 32)
    hooks = WorkCounter(method.model)
    losses, expected = method.adapt(images), reference.adapt(images)
    assert losses == expected and len(losses) == 3
    diagnostic_images = method.last_reliability_summary["diagnostic_views"]
    assert torch.equal(method.evaluate(images), reference.evaluate(images))
    hooks.close()
    prefix = method.model.prefix_depth
    assert [sum(values) for values in hooks.blocks] == (
        [2 + diagnostic_images] * prefix + [10 + diagnostic_images] * (24 - prefix)
    )
    assert method.model.cache_stats == dict(cache_builds=1, cache_hits=4, cache_bytes=0)
    for p, q in zip(method.model.parameters(), reference.model.parameters()):
        assert torch.equal(p, q)


def test_first_training_forward_counts_prefix_build_and_tail_is_never_stale():
    method = tiny_method()
    images = torch.randn(2, 3, 32, 32)
    hooks = WorkCounter(method.model)
    first = method.forward_adapt_logits(images)
    first.square().mean().backward()
    method.optimizer.step()
    method.optimizer.zero_grad(set_to_none=True)
    second = method.forward_adapt_logits(images)
    hooks.close()
    assert [sum(values) for values in hooks.blocks] == [2] * 2 + [4] * 6
    assert hooks.stems == [2]
    assert not torch.equal(first, second)
    with method.model.cache_disabled():
        expected = method.forward_adapt_logits(images)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)
    second.square().mean().backward()
    assert method.model.cache_stats["cache_builds"] == 1


def test_diagnostic_views_force_full_encoder_even_for_identical_original_tensor(monkeypatch):
    method = tiny_method()
    images = torch.randn(2, 3, 32, 32)
    method.estimate_trace_reliability(images)
    cached = method.model._prefix_cache
    counters = dict(method.model.cache_stats)
    hooks = WorkCounter(method.model)

    def batches(keys, builder):
        assert method.model._cache_disabled_depth == 1
        for start in range(0, len(keys), 2):
            chunk = keys[start:start + 2]
            # Deliberately return the exact original object. The scope, not
            # pixel equality or a fortuitous tensor identity, enforces bypass.
            yield chunk, method.forward_eval_logits(images, interpolate=False)[:len(chunk)]

    monkeypatch.setattr(method, "_forward_trace_view_batches", batches)
    method.evaluate_trace_views(images, torch.ones(2, 2, 2, 2) / 2, method.last_candidate_stats)
    hooks.close()
    assert hooks.stems and all(values == hooks.stems for values in hooks.blocks)
    assert method.model._prefix_cache is cached
    assert method.model.cache_stats == counters
    assert method.model._cache_disabled_depth == 0


def test_different_tensor_views_mutations_and_frozen_weights_cannot_reuse_cache():
    method = tiny_method()
    images = torch.randn(2, 3, 32, 32)
    method.forward_adapt_logits(images)
    cached = method.model._prefix_cache
    hooks = WorkCounter(method.model)
    for other in (images.clone(), images[:], images[:1], images.flip(-1)):
        method.forward_eval_logits(other, interpolate=False)
    assert all(values == hooks.stems for values in hooks.blocks)
    assert method.model._prefix_cache is cached
    images.add_(.1)
    method.forward_adapt_logits(images)
    assert method.model.cache_stats["cache_builds"] == 2
    with torch.no_grad():
        method.model.visual.ln_pre.bias.add_(.1)
    actual = method.forward_adapt_logits(images)
    assert method.model.cache_stats["cache_builds"] == 3
    with method.model.cache_disabled():
        expected = method.forward_adapt_logits(images)
    assert torch.equal(actual, expected)
    replacement = images.clone()
    method.forward_adapt_logits(replacement)
    assert method.model.cache_stats["cache_builds"] == 4
    assert method.model._original_ref() is replacement
    hooks.close()


def test_reset_and_new_episode_release_cache_restore_parameters_and_trajectory():
    method = tiny_method()
    images = torch.randn(2, 3, 32, 32)
    initial = {name: p.detach().clone() for name, p in method.model.named_parameters()}
    losses = method.adapt(images)
    refs = cache_references(method.model)
    assert method.optimizer.state
    method.reset()
    assert method.model.cache_stats == dict(cache_builds=0, cache_hits=0, cache_bytes=0)
    assert all(ref() is None for ref in refs)
    assert not method.optimizer.state
    assert all(torch.equal(initial[name], p) for name, p in method.model.named_parameters())
    assert method.adapt(images) == losses
    refs = cache_references(method.model)
    method.adapt(torch.randn_like(images))
    assert all(ref() is None for ref in refs)
    assert method.model._prefix_cache is None  # The temporary original was released.
    assert method.adapt(images) == losses
    refs = cache_references(method.model)
    assert method.adapt(images, valid_mask=torch.zeros(2, 1, 32, 32)) == []
    assert method.model.cache_stats["cache_bytes"] == 0
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("phase", ["estimate_trace_reliability", "adapt", "evaluate"])
def test_exception_paths_release_cache_and_restore_diagnostic_scope(monkeypatch, phase):
    method = tiny_method()
    images = torch.randn(2, 3, 32, 32)
    method.forward_adapt_logits(images)
    refs = cache_references(method.model)

    def fail(*args, **kwargs):
        raise RuntimeError("injected forward failure")

    monkeypatch.setattr(method, "forward_eval_logits", fail)
    with pytest.raises(RuntimeError, match="injected"):
        getattr(method, phase)(images)
    assert method.model.cache_stats["cache_bytes"] == 0
    assert method.model._original_ref is None
    assert all(ref() is None for ref in refs)
    with pytest.raises(RuntimeError, match="injected"):
        with method.model.cache_disabled():
            fail()
    assert method.model._cache_disabled_depth == 0


def test_clear_cache_and_original_garbage_collection_release_feature_storage():
    method = tiny_method()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        method.forward_adapt_logits(images)
    refs = cache_references(method.model)
    method.model.clear_cache()
    assert all(ref() is None for ref in refs)
    with torch.no_grad():
        method.forward_adapt_logits(images)
    refs = cache_references(method.model)
    image_ref = weakref.ref(images)
    del images
    gc.collect()
    assert image_ref() is None
    assert all(ref() is None for ref in refs)
    assert method.model.cache_stats["cache_bytes"] == 0


def test_inference_proposal_then_autograd_and_inference_tensor_fallback():
    method = tiny_method()
    images = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        method.estimate_trace_reliability(images)
    method.forward_adapt_logits(images).square().mean().backward()
    assert method.model.cache_stats["cache_builds"] == 1
    method.reset()
    with torch.inference_mode():
        inference_images = torch.randn_like(images)
        method.estimate_trace_reliability(inference_images)
    assert method.model.cache_stats["cache_builds"] == 0


def test_autocast_policy_changes_invalidate_prefix():
    method = tiny_method()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        method.forward_adapt_logits(images)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = method.forward_adapt_logits(images)
            with method.model.cache_disabled():
                expected = method.forward_adapt_logits(images)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        method.forward_adapt_logits(images)
    assert method.model.cache_stats["cache_builds"] == 3


@pytest.mark.parametrize("depth", [-1, True, 1.5, 8, 9])
def test_invalid_or_full_tail_depth_rejected(depth):
    with pytest.raises(ValueError, match="tail_depth"):
        CachedPrefixDenseCLIP(tiny_dense(), vision_outputs=(-1,), tail_depth=depth)


def test_earlier_last_selected_layer_sets_depth_and_excludes_unexecuted_blocks():
    model = CachedPrefixDenseCLIP(tiny_dense(), vision_outputs=(-2, -4), tail_depth=2)
    assert model.executed_depth == 7 and model.prefix_depth == 5
    assert not any(p.requires_grad for p in model.visual.transformer.resblocks[-1].parameters())
    assert not any(p.requires_grad for p in model.visual.transformer.resblocks[-2].ln_2.parameters())
    with pytest.raises(ValueError, match="executed"):
        CachedPrefixDenseCLIP(tiny_dense(), vision_outputs=(-2,), tail_depth=7)
    with pytest.raises(ValueError, match="Selected visual layers"):
        model.encode_dense_layers(torch.randn(1, 3, 32, 32), (-1,))


def test_wrong_encoder_image_gradients_and_unfrozen_prefix_rejected():
    with pytest.raises(TypeError, match="DenseCLIP"):
        CachedPrefixDenseCLIP(torch.nn.Linear(2, 2), vision_outputs=(-1,), tail_depth=0)
    method = tiny_method()
    with pytest.raises(ValueError, match="image gradients"):
        method.forward_adapt_logits(torch.randn(1, 3, 32, 32, requires_grad=True))
    images = torch.randn(1, 3, 32, 32)
    method.model.visual.ln_pre.requires_grad_(True)
    with pytest.raises(RuntimeError, match="must stay frozen"):
        method.forward_adapt_logits(images)
